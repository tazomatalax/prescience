"""ROUGE-L evaluation for followup work prediction. Scores generated title/abstract pairs against references."""

import os
import argparse

from rouge_score import rouge_scorer
from tqdm import tqdm

import utils


def create_scorer():
    """Create a ROUGE scorer instance."""
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def main():
    """Score generated title/abstract pairs against references using ROUGE-L."""
    parser = argparse.ArgumentParser(description="Score generated title+abstract pairs against references using ROUGE-L")
    parser.add_argument("--input_path", required=True, help="Path to JSON file with generated title/abstract pairs")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID for dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--output_dir", default="data/task_followup_prediction/test/rouge_scored", help="Directory to save the scored JSON output")
    parser.add_argument("--max_records", type=int, default=None, help="Optional cap on number of generations to score")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file if it exists")
    args = parser.parse_args()

    # Load generated title/abstract pairs and reference corpus
    utils.log(f"Loading generations from {args.input_path}")
    generations, metadata = utils.load_json(args.input_path)
    if args.max_records is not None:
        generations = generations[:args.max_records]

    utils.log(f"Loading corpus from HuggingFace repo {args.hf_repo_id} (split={args.split})")
    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    # Initialize ROUGE scorer
    scorer = create_scorer()

    # Score each generation against its reference
    utils.log(f"Scoring {len(generations)} generations")
    scored_records = []
    for record in tqdm(generations, desc="Computing ROUGE-L"):
        scored_record = dict(record)
        reference = all_papers_dict[record["corpus_id"]]

        if len(record["title"]) > 0:
            candidate = record["title"] + " " + record["abstract"]
            reference_text = reference["title"] + " " + reference["abstract"]
            scores = scorer.score(reference_text, candidate)
            scored_record["rouge_l_precision"] = scores["rougeL"].precision
            scored_record["rouge_l_recall"] = scores["rougeL"].recall
            scored_record["rouge_l_f1"] = scores["rougeL"].fmeasure
        else:
            scored_record["rouge_l_precision"] = 0.0
            scored_record["rouge_l_recall"] = 0.0
            scored_record["rouge_l_f1"] = 0.0

        scored_records.append(scored_record)

    utils.log(f"Finished scoring {len(scored_records)} records")

    # Save scored records to output file
    input_filename = os.path.basename(args.input_path)
    output_filename = os.path.splitext(input_filename)[0] + ".rouge_scored.json"
    output_path = os.path.join(args.output_dir, output_filename)

    utils.save_json(scored_records, output_path, metadata=utils.update_metadata(metadata, args), overwrite=args.overwrite)
    utils.log(f"Saved scored records to {output_path}")


if __name__ == "__main__":
    main()
