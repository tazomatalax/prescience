"""Average LACER scores from GPT-5 and Claude Opus 4.5 judges."""

import os
import argparse
from glob import glob

import utils


def main():
    parser = argparse.ArgumentParser(description="Average LACER scores from GPT-5 and Opus judges")
    parser.add_argument("--gpt5_dir", default="data/task_followup_prediction/test/lacer_scored", help="Directory with GPT-5 judged LACER scores")
    parser.add_argument("--opus_dir", default="data/task_followup_prediction/test/lacer_scored_opus", help="Directory with Opus judged LACER scores")
    parser.add_argument("--output_dir", default="data/task_followup_prediction/test/lacer_scored_averaged", help="Directory to save averaged scores")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all scored files in both directories
    gpt5_files = {os.path.basename(f): f for f in glob(os.path.join(args.gpt5_dir, "*.lacer_scored.json"))}
    opus_files = {os.path.basename(f): f for f in glob(os.path.join(args.opus_dir, "*.lacer_scored.json"))}
    common_files = sorted(set(gpt5_files.keys()) & set(opus_files.keys()))
    utils.log(f"Found {len(common_files)} files in common between GPT-5 and Opus directories")

    for filename in common_files:
        utils.log(f"Processing {filename}")
        gpt5_records, gpt5_metadata = utils.load_json(gpt5_files[filename])
        opus_records, opus_metadata = utils.load_json(opus_files[filename])

        # Filter to intersection of corpus_ids with valid lacer_score
        gpt5_ids = {r["corpus_id"] for r in gpt5_records if "lacer_score" in r}
        opus_ids = {r["corpus_id"] for r in opus_records if "lacer_score" in r}
        common_ids = gpt5_ids & opus_ids
        gpt5_records = sorted([r for r in gpt5_records if r["corpus_id"] in common_ids], key=lambda r: r["corpus_id"])
        opus_records = sorted([r for r in opus_records if r["corpus_id"] in common_ids], key=lambda r: r["corpus_id"])

        # Average scores for records present in both
        averaged_records = []
        for gpt5_record, opus_record in zip(gpt5_records, opus_records):
            averaged_record = dict(gpt5_record)
            averaged_record["lacer_score_gpt5"] = gpt5_record["lacer_score"]
            averaged_record["lacer_score_opus"] = opus_record["lacer_score"]
            averaged_record["lacer_score"] = (gpt5_record["lacer_score"] + opus_record["lacer_score"]) / 2
            averaged_record["lacer_scoring_model"] = "averaged(gpt-5-2025-08-07,claude-opus-4-5-20251101)"
            averaged_records.append(averaged_record)

        utils.log(f"  Averaged {len(averaged_records)} records (GPT-5: {len(gpt5_records)}, Opus: {len(opus_records)})")

        output_path = os.path.join(args.output_dir, filename.replace(".lacer_scored.json", ".lacer_scored_averaged.json"))
        metadata = utils.update_metadata(gpt5_metadata, args)
        utils.save_json(averaged_records, output_path, metadata=metadata)
        utils.log(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
