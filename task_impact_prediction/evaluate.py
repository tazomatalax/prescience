"""Evaluation script for impact prediction task."""

import os
import argparse
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from tqdm import tqdm

import utils
from task_impact_prediction.dataset import create_evaluation_instances


def evaluate_predictions(predictions, ground_truth_dict):
    """Evaluate predictions against ground truth and compute per-instance metrics."""
    results = []
    for pred in tqdm(predictions, desc="Evaluating predictions"):
        corpus_id = pred["corpus_id"]
        if corpus_id not in ground_truth_dict:
            continue

        predicted = pred["predicted_citations"]
        gt = ground_truth_dict[corpus_id]
        abs_error = abs(predicted - gt)

        results.append({
            "corpus_id": corpus_id,
            "predicted": predicted,
            "gt": gt,
            "abs_error": abs_error,
        })

    return results


def compute_aggregate_metrics(results):
    """Compute aggregate metrics across all evaluation instances."""
    if len(results) == 0:
        return {"num_instances": 0, "mae": None, "r2": None, "pearson": None, "spearman": None, "log_mae": None, "log_pearson": None}

    predicted = np.array([r["predicted"] for r in results])
    gt = np.array([r["gt"] for r in results])

    mae = float(np.mean(np.abs(predicted - gt)))
    r2 = float(r2_score(gt, predicted))
    pearson = float(np.corrcoef(predicted, gt)[0, 1]) if len(results) > 1 else None
    spearman_result = spearmanr(predicted, gt)
    spearman = float(spearman_result.correlation) if len(results) > 1 else None

    # Log-space metrics (use log1p to handle zeros)
    log_predicted = np.log1p(np.maximum(predicted, 0))
    log_gt = np.log1p(gt)
    log_mae = float(np.mean(np.abs(log_predicted - log_gt)))
    log_pearson = float(np.corrcoef(log_predicted, log_gt)[0, 1]) if len(results) > 1 else None

    return {
        "num_instances": len(results),
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "log_mae": log_mae,
        "log_pearson": log_pearson,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate impact predictions")
    parser.add_argument("--predictions_path", type=str, required=True, help="Path to predictions JSON file")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID for dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--impact_months", type=int, default=12, help="Which month to evaluate")
    parser.add_argument("--output_dir", type=str, default="data/task_impact_prediction/test/scored", help="Output directory for evaluation results")
    args = parser.parse_args()

    utils.log(f"Loading predictions from {args.predictions_path}")
    predictions, predictions_metadata = utils.load_json(args.predictions_path)
    utils.log(f"Loaded {len(predictions)} predictions")

    utils.log(f"Loading corpus from HuggingFace repo {args.hf_repo_id} (split={args.split})")
    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    utils.log(f"Loaded {len(all_papers)} papers")

    utils.log("Creating ground truth lookup")
    evaluation_instances = create_evaluation_instances(all_papers, args.impact_months)
    ground_truth_dict = {inst["corpus_id"]: inst["gt_citations"] for _, inst in evaluation_instances}
    utils.log(f"Created ground truth for {len(ground_truth_dict)} papers")

    utils.log("Evaluating predictions")
    results = evaluate_predictions(predictions, ground_truth_dict)

    aggregates = compute_aggregate_metrics(results)
    utils.log(f"Evaluation complete:")
    utils.log(f"  Instances: {aggregates['num_instances']}")
    utils.log(f"  MAE: {aggregates['mae']:.4f}" if aggregates['mae'] is not None else "  MAE: N/A")
    utils.log(f"  R²: {aggregates['r2']:.4f}" if aggregates['r2'] is not None else "  R²: N/A")
    utils.log(f"  Pearson: {aggregates['pearson']:.4f}" if aggregates['pearson'] is not None else "  Pearson: N/A")
    utils.log(f"  Spearman: {aggregates['spearman']:.4f}" if aggregates['spearman'] is not None else "  Spearman: N/A")
    utils.log(f"  Log MAE: {aggregates['log_mae']:.4f}" if aggregates['log_mae'] is not None else "  Log MAE: N/A")
    utils.log(f"  Log Pearson: {aggregates['log_pearson']:.4f}" if aggregates['log_pearson'] is not None else "  Log Pearson: N/A")

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
