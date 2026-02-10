"""Frequency baseline for coauthor prediction. Predicts based on historical collaboration frequency."""

import os
import random
import argparse
from tqdm import tqdm

import utils
from task_coauthor_prediction.dataset import create_evaluation_instances, get_preexisting_publications_for_author

random.seed(42)


def get_collaboration_frequencies(author_id, cutoff_date, num_recent_papers, sd2publications, all_papers_dict):
    """Get collaboration frequencies for an author from their recent papers."""
    author_past_paper_ids = get_preexisting_publications_for_author(author_id, cutoff_date, sd2publications, all_papers_dict)
    recent_paper_ids = author_past_paper_ids[-num_recent_papers:]
    recent_papers = [all_papers_dict[cid] for cid in recent_paper_ids]

    coauthor_freqs = {}
    for paper in recent_papers:
        for author in paper["authors"]:
            coauthor_id = author["author_id"]
            if coauthor_id != author_id:
                if coauthor_id not in coauthor_freqs:
                    coauthor_freqs[coauthor_id] = 0
                coauthor_freqs[coauthor_id] += 1

    return dict(sorted(coauthor_freqs.items(), key=lambda x: x[1], reverse=True))


def merge_frequency_distributions(freq_dist_1, freq_dist_2):
    """Merge two frequency distributions by summing counts."""
    merged = freq_dist_1.copy()
    for author_id, count in freq_dist_2.items():
        if author_id in merged:
            merged[author_id] += count
        else:
            merged[author_id] = count
    return dict(sorted(merged.items(), key=lambda x: x[1], reverse=True))


def get_next_best_candidate(freq_dist, excluded_ids):
    """Get the highest-frequency author not in excluded_ids."""
    for author_id, freq in freq_dist.items():
        if author_id not in excluded_ids:
            return author_id, freq
    return None, None


def pad_predictions_to_k(predicted_ids, predicted_scores, k, excluded_ids, freq_dist, all_author_ids):
    """Pad predictions to length k using remaining frequency-ranked authors, then random."""
    if len(predicted_ids) >= k:
        return predicted_ids, predicted_scores

    excluded = set(excluded_ids)
    available_from_freq = [(aid, freq) for aid, freq in freq_dist.items() if aid not in excluded]
    num_from_freq = min(len(available_from_freq), k - len(predicted_ids))

    ids = list(predicted_ids) + [aid for aid, _ in available_from_freq[:num_from_freq]]
    scores = list(predicted_scores) + [freq for _, freq in available_from_freq[:num_from_freq]]
    excluded = excluded | set(ids)

    if len(ids) < k:
        available_authors = list(all_author_ids - excluded)
        num_needed = k - len(ids)
        random_authors = random.sample(available_authors, min(num_needed, len(available_authors)))
        ids = ids + random_authors
        scores = scores + [0] * len(random_authors)

    return ids, scores


def predict_coauthors_one_shot(first_author_id, k, num_recent_papers, cutoff_date, sd2publications, all_papers_dict, excluded_ids=None):
    """One-shot strategy: Return collaborators sorted by frequency."""
    if excluded_ids is None:
        excluded_ids = set()
    excluded_ids = excluded_ids | {first_author_id}

    collab_freqs = get_collaboration_frequencies(first_author_id, cutoff_date, num_recent_papers, sd2publications, all_papers_dict)

    predicted_ids = [aid for aid in collab_freqs.keys() if aid not in excluded_ids]
    predicted_scores = [collab_freqs[aid] for aid in predicted_ids]

    if len(predicted_ids) < k:
        excluded_ids = excluded_ids | set(predicted_ids)
        predicted_ids, predicted_scores = pad_predictions_to_k(
            predicted_ids, predicted_scores, k, excluded_ids, {}, set(sd2publications.keys())
        )

    return predicted_ids[:k], predicted_scores[:k]


def predict_coauthors_iterated(first_author_id, num_to_predict, k, num_recent_papers, cutoff_date, sd2publications, all_papers_dict, excluded_ids=None):
    """Iterated strategy: Iteratively predict coauthors with frequency merging."""
    if excluded_ids is None:
        excluded_ids = set()
    excluded_ids = excluded_ids | {first_author_id}

    predicted_ids = []
    predicted_scores = []

    summed_freqs = get_collaboration_frequencies(first_author_id, cutoff_date, num_recent_papers, sd2publications, all_papers_dict)
    next_author_id, next_author_freq = get_next_best_candidate(summed_freqs, excluded_ids)

    while len(predicted_ids) < num_to_predict and next_author_id is not None:
        predicted_ids.append(next_author_id)
        predicted_scores.append(next_author_freq)
        excluded_ids.add(next_author_id)

        next_author_freqs = get_collaboration_frequencies(next_author_id, cutoff_date, num_recent_papers, sd2publications, all_papers_dict)
        summed_freqs = merge_frequency_distributions(summed_freqs, next_author_freqs)
        next_author_id, next_author_freq = get_next_best_candidate(summed_freqs, excluded_ids)

    all_author_ids = set(sd2publications.keys())
    ids, scores = pad_predictions_to_k(predicted_ids, predicted_scores, k, excluded_ids, summed_freqs, all_author_ids)
    return ids[:k], scores[:k]


def main():
    parser = argparse.ArgumentParser(description="Frequency baseline for coauthor prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace dataset repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--num_recent_papers", type=int, default=10, help="Number of recent papers to consider")
    parser.add_argument("--k", type=int, default=1000, help="Number of coauthors to predict")
    parser.add_argument("--strategy", type=str, default="one_shot", choices=["one_shot", "iterated"])
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=1000, help="Save predictions every N instances")
    parser.add_argument("--seed_author_type", type=str, default="first", choices=["first", "last", "random", "highest_h_index"], help="Seed author selection strategy")
    parser.add_argument("--output_dir", type=str, default="data/task_coauthor_prediction/test/predictions")
    args = parser.parse_args()

    all_papers, sd2publications, _ = utils.load_corpus(
        hf_repo_id=args.hf_repo_id,
        split=args.split,
        embedding_type=None,
        load_sd2publications=True
    )
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    output_path = os.path.join(args.output_dir, f"predictions.frequency.{args.strategy}.{args.seed_author_type}.json")
    utils.log(f"Loaded {len(all_papers)} papers and {len(sd2publications)} author publication histories")

    utils.log(f"Creating evaluation instances with seed_author_type={args.seed_author_type}")
    evaluation_instances = create_evaluation_instances(all_papers, sd2publications, all_papers_dict, args.seed_author_type)
    utils.log(f"Created {len(evaluation_instances)} evaluation instances")

    # Select random subset of instances to evaluate if max_instances is set
    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = set(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances to evaluate")
    else:
        selected_indices = set(range(len(evaluation_instances)))

    utils.log(f"Running frequency baseline with strategy: {args.strategy}")
    predictions = []
    for idx, (date, instance) in enumerate(tqdm(evaluation_instances, desc="Running frequency baseline")):
        if idx in selected_indices:
            corpus_id = instance["corpus_id"]
            first_author_id = instance["first_author_id"]
            gt_coauthor_ids = instance["gt_coauthor_ids"]
            num_to_predict = len(gt_coauthor_ids)

            if args.strategy == "one_shot":
                pred_ids, pred_scores = predict_coauthors_one_shot(first_author_id, args.k, args.num_recent_papers, date, sd2publications, all_papers_dict)
            else:
                pred_ids, pred_scores = predict_coauthors_iterated(first_author_id, num_to_predict, args.k, args.num_recent_papers, date, sd2publications, all_papers_dict)

            predictions.append({
                "corpus_id": corpus_id,
                "first_author_id": first_author_id,
                "gt_coauthor_ids": gt_coauthor_ids,
                "predicted_coauthor_ids": pred_ids,
                "predicted_coauthor_scores": pred_scores,
            })

            if len(predictions) % args.save_every == 0:
                utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)

    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)


if __name__ == "__main__":
    main()
