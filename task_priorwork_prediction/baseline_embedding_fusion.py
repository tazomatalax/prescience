"""Embedding fusion baseline for prior work prediction. Aggregates cited paper embeddings in vector space."""

import os
import random
import argparse
from tqdm import tqdm

random.seed(42)

import utils
from task_priorwork_prediction.dataset import (
    create_evaluation_instances,
    get_cited_references_for_author,
)


def get_embedding(all_embeddings, corpus_id):
    """Get key embedding for a paper."""
    return all_embeddings[corpus_id]["key"].reshape(-1)


def create_paper_index(cutoff_date, all_papers_dict, all_embeddings, distance_metric):
    """Create FAISS index of paper embeddings for papers before cutoff_date."""
    paper_embeddings = {}
    for corpus_id, paper in all_papers_dict.items():
        if paper["date"] < cutoff_date:
            paper_embeddings[corpus_id] = get_embedding(all_embeddings, corpus_id)

    utils.log(f"Created embeddings for {len(paper_embeddings)} papers")
    return utils.create_index(paper_embeddings, distance_metric)


def predict_references(author_ids, cutoff_date, num_recent_papers, k, index, sd2publications, all_papers_dict, all_embeddings, distance_metric):
    """Predict prior work references by aggregating per-author cited paper embeddings and summing ranks."""

    # Collect cited references for each author with prior publications
    authorwise_cited_refs = {}
    for author_id in author_ids:
        cited_refs = get_cited_references_for_author(author_id, cutoff_date, num_recent_papers, sd2publications, all_papers_dict)
        if len(cited_refs) > 0:
            authorwise_cited_refs[author_id] = cited_refs

    # Create aggregated embedding for each author from their cited papers
    author_embeddings = []
    for author_id in authorwise_cited_refs:
        cited_embeddings = [get_embedding(all_embeddings, cid) for cid in authorwise_cited_refs[author_id]]
        author_embedding = utils.aggregate_embeddings(cited_embeddings, distance_metric)
        author_embeddings.append(author_embedding)

    # Handle case where no authors have cited references (e.g., all first-time authors). This only happens through the multiturn simulation calls.
    if len(author_embeddings) == 0:
        return [], []

    # Query index with each author's aggregated embedding
    retrieved_papers_lists, _ = utils.query_index(index, author_embeddings, k)

    # Combine results by summing ranks across all author queries, filtering None values
    summed_ranks = {}
    for retrieved_papers_list in retrieved_papers_lists:
        for rank, corpus_id in enumerate(retrieved_papers_list):
            if corpus_id is not None:
                if corpus_id not in summed_ranks:
                    summed_ranks[corpus_id] = 0
                summed_ranks[corpus_id] += rank

    sorted_by_rank = sorted(summed_ranks.items(), key=lambda x: x[1])
    predicted_ids = [cid for cid, _ in sorted_by_rank]
    predicted_scores = [score for _, score in sorted_by_rank]

    return predicted_ids[:k], predicted_scores[:k]


def main():
    parser = argparse.ArgumentParser(description="Embedding fusion baseline for prior work prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing embeddings pkl files")
    parser.add_argument("--embedding_type", type=str, required=True, choices=["gtr", "grit", "specter2"])
    parser.add_argument("--num_recent_papers", type=int, default=10, help="Number of recent papers to consider per author")
    parser.add_argument("--k", type=int, default=1000, help="Number of references to predict")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=5000, help="Save predictions every N instances")
    parser.add_argument("--output_dir", type=str, default="data/task_priorwork_prediction/test/predictions")
    args = parser.parse_args()

    distance_metric = "l2" if args.embedding_type == "specter2" else "cosine"

    utils.log(f"Loading corpus from {args.hf_repo_id} (split={args.split})")
    all_papers, sd2publications, all_embeddings = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=args.embeddings_dir, embedding_type=args.embedding_type, load_sd2publications=True)
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    output_path = os.path.join(args.output_dir, f"predictions.embedding_fusion.{args.embedding_type}.json")
    utils.log(f"Loaded {len(all_papers)} papers, {len(sd2publications)} authors, {len(all_embeddings)} embeddings")

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

    # Sort all papers by date and create initial index with papers before first evaluation instance
    all_papers = sorted(all_papers, key=lambda p: p["date"])
    first_date = evaluation_instances[0][0]
    utils.log(f"Creating initial paper index with cutoff date: {first_date}")
    index = create_paper_index(first_date, all_papers_dict, all_embeddings, distance_metric)

    # Filter to papers on or after first_date (earlier papers are already in initial index)
    postdated_papers = [p for p in all_papers if p["date"] >= first_date]

    utils.log(f"Running embedding fusion baseline with embedding: {args.embedding_type}")
    predictions = []
    for paper in tqdm(postdated_papers, desc="Running embedding fusion baseline"):
        corpus_id = paper["corpus_id"]

        # If this is a selected evaluation instance, run prediction before adding to index
        if corpus_id in eval_instance_dict:
            idx, instance = eval_instance_dict[corpus_id]
            if idx in selected_indices:
                author_ids = instance["author_ids"]
                gt_reference_ids = instance["gt_reference_ids"]

                predicted_ids, predicted_scores = predict_references(
                    author_ids, paper["date"], args.num_recent_papers, args.k,
                    index, sd2publications, all_papers_dict, all_embeddings, distance_metric
                )

                predictions.append({
                    "corpus_id": corpus_id,
                    "gt_reference_ids": gt_reference_ids,
                    "predicted_reference_ids": predicted_ids,
                    "predicted_reference_scores": predicted_scores,
                })

                if len(predictions) % args.save_every == 0:
                    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)

        # Add paper to index
        index = utils.add_vector_to_index(index, corpus_id, get_embedding(all_embeddings, corpus_id))

    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
