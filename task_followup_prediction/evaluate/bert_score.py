"""BERTScore evaluation for followup work prediction. Scores generated title/abstract pairs against references."""

import os
import argparse

import torch
from bert_score import BERTScorer

import utils

# Maximum character length for generated abstracts (truncate beyond this)
# Set to eliminate corrupted repetition-collapse outputs while keeping all valid abstracts
MAX_ABSTRACT_LENGTH = 2500

def create_scorer(model):
    """Create a BERTScorer instance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utils.log(f"Initializing BERTScorer with {model} on {device}")
    return BERTScorer(
        model_type=model,
        lang="en",
        rescale_with_baseline=True,
        idf=False,
        device=device,
    )


def main():
    """Score generated title/abstract pairs against references using BERTScore."""
    parser = argparse.ArgumentParser(description="Score generated title+abstract pairs against references using BERTScore")
    parser.add_argument("--input_path", required=True, help="Path to JSON file with generated title/abstract pairs")
    parser.add_argument("--corpus_path", default="data/corpus/test/all_papers.json", help="Path to corpus JSON containing reference title/abstract pairs")
    parser.add_argument("--output_dir", default="data/task_followup_prediction/test/bert_scored", help="Directory to save the scored JSON output")
    parser.add_argument("--model", default="microsoft/deberta-large-mnli", help="Model to use for BERTScore")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_records", type=int, default=None, help="Optional cap on number of generations to score")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file if it exists")
    args = parser.parse_args()

    # Load generated title/abstract pairs and reference corpus
    utils.log(f"Loading generations from {args.input_path}")
    generations, metadata = utils.load_json(args.input_path)
    if args.max_records is not None:
        generations = generations[:args.max_records]

    utils.log(f"Loading corpus from {args.corpus_path}")
    all_papers, _ = utils.load_json(args.corpus_path)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    # Initialize BERTScorer model
    scorer = create_scorer(args.model)

    # Build candidate/reference pairs for scoring, tracking which generations are valid
    utils.log(f"Scoring {len(generations)} generations")
    candidates = []
    references = []
    valid_mask = []
    truncated_count = 0
    for record in generations:
        reference = all_papers_dict[record["corpus_id"]]
        if len(record["title"]) > 0:
            candidate_text = record["title"] + " " + record["abstract"]
            if len(candidate_text) > MAX_ABSTRACT_LENGTH:
                candidate_text = candidate_text[:MAX_ABSTRACT_LENGTH]
                truncated_count += 1
            candidates.append(candidate_text)
            references.append(reference["title"] + " " + reference["abstract"])
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    if truncated_count > 0:
        utils.log(f"Truncated {truncated_count} generations exceeding {MAX_ABSTRACT_LENGTH} characters")

    # Compute BERTScore for all valid candidate/reference pairs
    utils.log(f"Computing BERTScore for {len(candidates)} valid generations")
    P, R, F1 = scorer.score(candidates, references, batch_size=args.batch_size, verbose=True)
    P = P.cpu().numpy()
    R = R.cpu().numpy()
    F1 = F1.cpu().numpy()

    # Assign scores to each record, using 0.0 for invalid generations
    scored_records = []
    score_idx = 0
    for idx, record in enumerate(generations):
        scored_record = dict(record)
        if valid_mask[idx]:
            scored_record["bert_score_precision"] = float(P[score_idx])
            scored_record["bert_score_recall"] = float(R[score_idx])
            scored_record["bert_score_f1"] = float(F1[score_idx])
            score_idx += 1
        else:
            scored_record["bert_score_precision"] = 0.0
            scored_record["bert_score_recall"] = 0.0
            scored_record["bert_score_f1"] = 0.0
        scored_records.append(scored_record)

    utils.log(f"Finished scoring {len(scored_records)} records")

    # Save scored records to output file
    input_filename = os.path.basename(args.input_path)
    output_filename = os.path.splitext(input_filename)[0] + ".bert_scored.json"
    output_path = os.path.join(args.output_dir, output_filename)

    utils.save_json(scored_records, output_path, metadata=utils.update_metadata(metadata, args), overwrite=args.overwrite)
    utils.log(f"Saved scored records to {output_path}")


if __name__ == "__main__":
    main()
