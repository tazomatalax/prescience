"""Mean pooling projected baseline for prior work prediction. Uses frozen embeddings + learned projection layers."""

import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import utils
from task_priorwork_prediction.dataset import (
    create_evaluation_instances,
    get_preexisting_publications_for_author,
)
from task_priorwork_prediction.projection_model import ProjectionModel

random.seed(42)


def l2_normalize(vec):
    """L2 normalize a vector."""
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        return vec / norm
    return vec


def predict_references(author_ids, cutoff_date, num_recent_papers, k, sd2publications, all_papers_dict, model, candidate_paper_ids):
    """Predict references by similarity of aggregated author embedding to candidate papers."""
    # Compute projected embedding for each author with publications
    author_embeddings = []
    for author_id in author_ids:
        author_pubs = get_preexisting_publications_for_author(author_id, cutoff_date, sd2publications, all_papers_dict)
        if not author_pubs:
            continue
        recent_pubs = author_pubs[-num_recent_papers:]
        valid_pubs = [cid for cid in recent_pubs if cid in model.all_embeddings]
        if valid_pubs:
            emb = model.encode_author(valid_pubs)
            if emb is not None and np.any(emb):
                author_embeddings.append(emb)

    # Handle case where no authors have valid embeddings
    if not author_embeddings:
        return [], []

    # Aggregate author embeddings (mean pool across authors)
    query_embedding = l2_normalize(np.mean(author_embeddings, axis=0))

    # Get candidate paper embeddings (projected)
    valid_candidates = [cid for cid in candidate_paper_ids if cid in model.all_embeddings]
    if not valid_candidates:
        return [], []

    candidate_embeddings = model.encode_papers_batch(valid_candidates)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings)[0]

    # Sort by similarity descending
    sorted_indices = np.argsort(similarities)[::-1]
    predicted_ids = [valid_candidates[i] for i in sorted_indices]
    predicted_scores = [float(similarities[i]) for i in sorted_indices]

    return predicted_ids[:k], predicted_scores[:k]


def main():
    parser = argparse.ArgumentParser(description="Mean pooling projected baseline for prior work prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing embeddings pkl files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained projection checkpoint")
    parser.add_argument("--embedding_type", type=str, default="specter2", choices=["gtr", "grit", "specter2"], help="Type of embeddings to use")
    parser.add_argument("--num_recent_papers", type=int, default=10, help="Number of recent papers to consider per author")
    parser.add_argument("--k", type=int, default=1000, help="Number of references to predict")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=5000, help="Save predictions every N instances")
    parser.add_argument("--output_dir", type=str, default="data/task_priorwork_prediction/test/predictions", help="Output directory for predictions")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    args = parser.parse_args()

    utils.log(f"Loading corpus from {args.hf_repo_id} (split={args.split})")
    all_papers, sd2publications, all_embeddings = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=args.embeddings_dir, embedding_type=args.embedding_type, load_sd2publications=True)
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    output_path = os.path.join(args.output_dir, f"predictions.mean_pooling_projected.{args.embedding_type}.json")
    utils.log(f"Loaded {len(all_papers)} papers, {len(sd2publications)} authors, {len(all_embeddings)} embeddings")

    utils.log(f"Loading projection model from {args.checkpoint}")
    model = ProjectionModel(args.checkpoint, all_embeddings, device=args.device)
    utils.log(f"Loaded projection model with output dimension {model.out_dim}")

    utils.log("Creating evaluation instances")
    evaluation_instances = create_evaluation_instances(all_papers, sd2publications, all_papers_dict)
    eval_instance_dict = {instance["corpus_id"]: (idx, instance) for idx, (date, instance) in enumerate(evaluation_instances)}
    utils.log(f"Created {len(evaluation_instances)} evaluation instances")

    # Select random subset of instances to evaluate if max_instances is set
    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = set(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances to evaluate")
    else:
        selected_indices = set(range(len(evaluation_instances)))

    # Sort all papers by date and build initial candidate set
    all_papers = sorted(all_papers, key=lambda p: p["date"])
    first_date = evaluation_instances[0][0]
    utils.log(f"First evaluation date: {first_date}")

    # Build initial set of candidate papers (papers before first_date)
    candidate_paper_ids = set()
    for paper in all_papers:
        if paper["date"] < first_date:
            candidate_paper_ids.add(paper["corpus_id"])
    utils.log(f"Initial candidate pool: {len(candidate_paper_ids)} papers")

    # Filter to papers on or after first_date
    postdated_papers = [p for p in all_papers if p["date"] >= first_date]

    utils.log("Running mean pooling projected baseline")
    predictions = []
    for paper in tqdm(postdated_papers, desc="Running mean pooling projected baseline"):
        corpus_id = paper["corpus_id"]

        # If this is a selected evaluation instance, run prediction before adding to candidates
        if corpus_id in eval_instance_dict:
            idx, instance = eval_instance_dict[corpus_id]
            if idx in selected_indices:
                author_ids = instance["author_ids"]
                gt_reference_ids = instance["gt_reference_ids"]

                predicted_ids, predicted_scores = predict_references(
                    author_ids, paper["date"], args.num_recent_papers, args.k,
                    sd2publications, all_papers_dict, model, candidate_paper_ids
                )

                predictions.append({
                    "corpus_id": corpus_id,
                    "gt_reference_ids": gt_reference_ids,
                    "predicted_reference_ids": predicted_ids,
                    "predicted_reference_scores": predicted_scores,
                })

                if len(predictions) % args.save_every == 0:
                    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)

        # Add paper to candidate pool
        candidate_paper_ids.add(corpus_id)

    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
