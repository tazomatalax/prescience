"""Mean pooling projected baseline for coauthor prediction. Uses frozen embeddings + learned projection layers."""

import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import utils
from task_coauthor_prediction.dataset import (
    create_evaluation_instances,
    get_preexisting_publications_for_author,
    get_preexisting_publications_for_corpus,
)
from task_coauthor_prediction.projection_model import ProjectionModel

random.seed(42)


def l2_normalize(vec):
    """L2 normalize a vector."""
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        return vec / norm
    return vec


def compute_all_author_embeddings(cutoff_date, num_recent_papers, sd2publications, all_papers_dict, model):
    """Pre-compute projected embeddings for all authors with publications before cutoff_date."""
    author_embeddings = {}
    preexisting_pubs = get_preexisting_publications_for_corpus(cutoff_date, sd2publications, all_papers_dict)

    for author_id, pub_ids in tqdm(preexisting_pubs.items(), desc="Creating author embeddings"):
        if len(pub_ids) > 0:
            recent_pubs = pub_ids[-num_recent_papers:]
            emb = model.encode_author(recent_pubs)
            if emb is not None and np.any(emb):
                author_embeddings[author_id] = l2_normalize(emb)

    utils.log(f"Created embeddings for {len(author_embeddings)} authors")
    return author_embeddings


def update_author_embedding(author_embeddings, author_id, pub_corpus_ids, model):
    """Update an author's embedding using the projection model."""
    if len(pub_corpus_ids) > 0:
        emb = model.encode_author(pub_corpus_ids)
        if emb is not None and np.any(emb):
            author_embeddings[author_id] = l2_normalize(emb)
    return author_embeddings


def predict_coauthors(first_author_id, num_recent_papers, k, cutoff_date, sd2publications, all_papers_dict, model, author_embeddings, all_author_ids, excluded_ids=None):
    """Predict coauthors by cosine similarity using projected embeddings."""
    if excluded_ids is None:
        excluded_ids = set()
    excluded_ids = excluded_ids | {first_author_id}

    # Get query author's projected embedding
    author_pubs = get_preexisting_publications_for_author(first_author_id, cutoff_date, sd2publications, all_papers_dict)
    recent_pubs = author_pubs[-num_recent_papers:]
    query_embedding = model.encode_author(recent_pubs)

    if query_embedding is None or not np.any(query_embedding):
        return [], []

    query_embedding = l2_normalize(query_embedding)

    # Get all candidate authors
    candidate_ids = [aid for aid in author_embeddings.keys() if aid not in excluded_ids]
    if not candidate_ids:
        return [], []

    # Stack candidate embeddings and compute cosine similarity
    candidate_matrix = np.stack([author_embeddings[aid] for aid in candidate_ids])
    similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_matrix)[0]

    # Sort by similarity descending
    sorted_indices = np.argsort(similarities)[::-1]
    predicted_ids = [candidate_ids[i] for i in sorted_indices]
    predicted_scores = [float(similarities[i]) for i in sorted_indices]

    # Pad with random authors if needed
    if len(predicted_ids) < k:
        excluded_ids = excluded_ids | set(predicted_ids)
        available_authors = list(all_author_ids - excluded_ids)
        num_needed = k - len(predicted_ids)
        random_authors = random.sample(available_authors, min(num_needed, len(available_authors)))
        predicted_ids.extend(random_authors)
        predicted_scores.extend([0.0] * len(random_authors))

    return predicted_ids[:k], predicted_scores[:k]


def main():
    parser = argparse.ArgumentParser(description="Mean pooling projected baseline for coauthor prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing embedding files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained projection checkpoint")
    parser.add_argument("--embedding_type", type=str, default="specter2", choices=["gtr", "grit", "specter2"], help="Type of embeddings to use")
    parser.add_argument("--num_recent_papers", type=int, default=10, help="Number of recent papers to consider per author")
    parser.add_argument("--k", type=int, default=1000, help="Number of coauthors to predict")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=1000, help="Save predictions every N instances")
    parser.add_argument("--seed_author_type", type=str, default="first", choices=["first", "last", "random", "highest_h_index"], help="Seed author selection strategy")
    parser.add_argument("--output_dir", type=str, default="data/task_coauthor_prediction/test/predictions", help="Output directory for predictions")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    args = parser.parse_args()

    all_papers, sd2publications, all_embeddings = utils.load_corpus(
        hf_repo_id=args.hf_repo_id,
        split=args.split,
        embeddings_dir=args.embeddings_dir,
        embedding_type=args.embedding_type,
        load_sd2publications=True
    )
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    all_author_ids = set(sd2publications.keys())
    output_path = os.path.join(args.output_dir, f"predictions.mean_pooling_projected.{args.embedding_type}.{args.seed_author_type}.json")
    utils.log(f"Loaded {len(all_papers)} papers, {len(sd2publications)} authors, {len(all_embeddings)} embeddings")

    utils.log(f"Loading projection model from {args.checkpoint}")
    model = ProjectionModel(args.checkpoint, all_embeddings, device=args.device)
    utils.log(f"Loaded projection model with output dimension {model.out_dim}")

    utils.log(f"Creating evaluation instances with seed_author_type={args.seed_author_type}")
    evaluation_instances = create_evaluation_instances(all_papers, sd2publications, all_papers_dict, args.seed_author_type)
    eval_instance_dict = {instance["corpus_id"]: (idx, instance) for idx, (date, instance) in enumerate(evaluation_instances)}
    utils.log(f"Created {len(evaluation_instances)} evaluation instances")

    # Select random subset of instances to evaluate if max_instances is set
    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = set(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances to evaluate")
    else:
        selected_indices = set(range(len(evaluation_instances)))

    # Sort all papers by date and create initial author embeddings
    all_papers = sorted(all_papers, key=lambda p: p["date"])
    first_date = evaluation_instances[0][0]
    utils.log(f"Creating initial author embeddings with cutoff date: {first_date}")
    author_embeddings = compute_all_author_embeddings(first_date, args.num_recent_papers, sd2publications, all_papers_dict, model)

    # Filter to papers on or after first_date
    postdated_papers = [p for p in all_papers if p["date"] >= first_date]

    utils.log("Running mean pooling projected baseline")
    predictions = []
    for paper in tqdm(postdated_papers, desc="Running mean pooling projected baseline"):
        corpus_id = paper["corpus_id"]
        date = paper["date"]

        # Update authors on this paper before prediction
        if "authors" in paper:
            for author in paper["authors"]:
                author_id = author["author_id"]
                author_pubs = get_preexisting_publications_for_author(author_id, date, sd2publications, all_papers_dict)
                recent_pubs = author_pubs[-args.num_recent_papers:]
                author_embeddings = update_author_embedding(author_embeddings, author_id, recent_pubs, model)

        # If this is a selected evaluation instance, run prediction
        if corpus_id in eval_instance_dict:
            idx, instance = eval_instance_dict[corpus_id]
            if idx in selected_indices:
                first_author_id = instance["first_author_id"]
                gt_coauthor_ids = instance["gt_coauthor_ids"]

                pred_ids, pred_scores = predict_coauthors(
                    first_author_id, args.num_recent_papers, args.k, date,
                    sd2publications, all_papers_dict, model, author_embeddings, all_author_ids
                )

                predictions.append({
                    "corpus_id": corpus_id,
                    "first_author_id": first_author_id,
                    "gt_coauthor_ids": gt_coauthor_ids,
                    "predicted_coauthor_ids": pred_ids,
                    "predicted_coauthor_scores": pred_scores,
                })

                if len(predictions) % args.save_every == 0:
                    utils.save_json(predictions, output_path, metadata=utils.update_metadata({}, args), overwrite=True)

    utils.save_json(predictions, output_path, metadata=utils.update_metadata({}, args), overwrite=True)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
