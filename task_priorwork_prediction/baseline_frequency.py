"""Frequency baseline for prior work prediction. Predicts based on citation frequency in authors' prior work."""

import os
import random
import argparse
from tqdm import tqdm

import utils
from task_priorwork_prediction.dataset import (
    create_evaluation_instances,
    get_cited_references_for_author,
)

random.seed(42)


def predict_references(author_ids, cutoff_date, num_recent_papers, k, sd2publications, all_papers_dict, all_corpus_ids):
    """Predict prior work references based on citation frequency in authors' prior publications."""

    # Count how often each paper is cited across all authors' prior publications
    citation_freqs = {}
    for author_id in author_ids:
        cited_refs = get_cited_references_for_author(author_id, cutoff_date, num_recent_papers, sd2publications, all_papers_dict)
        for corpus_id in cited_refs:
            if corpus_id not in citation_freqs:
                citation_freqs[corpus_id] = 0
            citation_freqs[corpus_id] += 1

    # Sort by frequency (descending)
    sorted_by_freq = sorted(citation_freqs.items(), key=lambda x: x[1], reverse=True)
    predicted_ids = [cid for cid, _ in sorted_by_freq]
    predicted_scores = [freq for _, freq in sorted_by_freq]

    # Pad with random papers if needed
    if len(predicted_ids) < k:
        excluded = set(predicted_ids)
        available_papers = [cid for cid in all_corpus_ids if cid not in excluded and all_papers_dict[cid]["date"] < cutoff_date]
        num_needed = k - len(predicted_ids)
        random_papers = random.sample(available_papers, min(num_needed, len(available_papers)))
        predicted_ids.extend(random_papers)
        predicted_scores.extend([0] * len(random_papers))

    return predicted_ids[:k], predicted_scores[:k]


def main():
    parser = argparse.ArgumentParser(description="Frequency baseline for prior work prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--num_recent_papers", type=int, default=10, help="Number of recent papers to consider per author")
    parser.add_argument("--k", type=int, default=1000, help="Number of references to predict")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=5000, help="Save predictions every N instances")
    parser.add_argument("--output_dir", type=str, default="data/task_priorwork_prediction/test/predictions")
    args = parser.parse_args()

    utils.log(f"Loading corpus from {args.hf_repo_id} (split={args.split})")
    all_papers, sd2publications, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=True)
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    all_corpus_ids = list(all_papers_dict.keys())
    output_path = os.path.join(args.output_dir, f"predictions.frequency.json")
    utils.log(f"Loaded {len(all_papers)} papers and {len(sd2publications)} author publication histories")

    utils.log("Creating evaluation instances")
    evaluation_instances = create_evaluation_instances(all_papers, sd2publications, all_papers_dict)
    utils.log(f"Created {len(evaluation_instances)} evaluation instances")

    # Select random subset of instances to evaluate if max_instances is set
    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = set(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances to evaluate")
    else:
        selected_indices = set(range(len(evaluation_instances)))

    utils.log("Running frequency baseline")
    predictions = []
    for idx, (date, instance) in enumerate(tqdm(evaluation_instances, desc="Running frequency baseline")):
        if idx not in selected_indices:
            continue

        corpus_id = instance["corpus_id"]
        author_ids = instance["author_ids"]
        gt_reference_ids = instance["gt_reference_ids"]

        predicted_ids, predicted_scores = predict_references(
            author_ids, date, args.num_recent_papers, args.k,
            sd2publications, all_papers_dict, all_corpus_ids
        )

        predictions.append({
            "corpus_id": corpus_id,
            "gt_reference_ids": gt_reference_ids,
            "predicted_reference_ids": predicted_ids,
            "predicted_reference_scores": predicted_scores,
        })

        if len(predictions) % args.save_every == 0:
            utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)

    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
