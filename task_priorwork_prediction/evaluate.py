"""Evaluation script for prior work prediction task."""

import os
import argparse
from tqdm import tqdm

import utils


def evaluate_predictions(predictions):
    """Evaluate predictions and compute metrics for each instance."""
    results = []
    for pred in tqdm(predictions, desc="Evaluating predictions"):
        corpus_id = pred["corpus_id"]
        gt_reference_ids = pred["gt_reference_ids"]
        predicted_reference_ids = pred["predicted_reference_ids"]

        ndcg = utils.calculate_ndcg(predicted_reference_ids, gt_reference_ids)
        k = len(gt_reference_ids)
        top_k_predictions = predicted_reference_ids[:k]
        precision, recall, f1 = utils.calculate_precision_recall_f1(top_k_predictions, gt_reference_ids)

        results.append({
            "corpus_id": corpus_id,
            "ndcg": ndcg,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_gt_references": len(gt_reference_ids),
        })

    return results


def compute_aggregate_metrics(results):
    """Compute aggregate metrics across all evaluation instances."""
    n = len(results)
    return {
        "num_instances": n,
        "avg_ndcg": sum(r["ndcg"] for r in results) / n,
        "avg_precision": sum(r["precision"] for r in results) / n,
        "avg_recall": sum(r["recall"] for r in results) / n,
        "avg_f1": sum(r["f1"] for r in results) / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate prior work predictions")
    parser.add_argument("--predictions_path", type=str, required=True, help="Path to predictions JSON file")
    parser.add_argument("--output_dir", type=str, default="data/task_priorwork_prediction/test/scored")
    args = parser.parse_args()

    utils.log(f"Loading predictions from {args.predictions_path}")
    predictions, predictions_metadata = utils.load_json(args.predictions_path)
    utils.log(f"Loaded {len(predictions)} predictions")

    utils.log("Evaluating predictions")
    results = evaluate_predictions(predictions)

    aggregates = compute_aggregate_metrics(results)
    utils.log(f"Evaluation complete:")
    utils.log(f"  Instances: {aggregates['num_instances']}")
    utils.log(f"  Avg nDCG: {aggregates['avg_ndcg']:.4f}")
    utils.log(f"  Avg Precision: {aggregates['avg_precision']:.4f}")
    utils.log(f"  Avg Recall: {aggregates['avg_recall']:.4f}")
    utils.log(f"  Avg F1: {aggregates['avg_f1']:.4f}")

    output = {"aggregates": aggregates, "per_instance": results}

    predictions_filename = os.path.basename(args.predictions_path)
    base, ext = os.path.splitext(predictions_filename)
    output_filename = f"{base}.eval{ext}"

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_filename)
    utils.log(f"Saving evaluation results to {output_path}")
    utils.save_json(output, output_path, metadata=utils.update_metadata(predictions_metadata, args), overwrite=True)


if __name__ == "__main__":
    main()
