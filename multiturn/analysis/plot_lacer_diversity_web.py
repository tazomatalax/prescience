"""Plot LACER diversity time series — web variant with transparent background."""
import os
import random
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import utils

PINK = "#f45096"
TEAL = "#0a3235"


def compute_per_paper_scores(papers):
    """Compute mean LACER distance (10 - score) for each paper, grouped by bucket."""
    bucket_scores = defaultdict(list)
    for paper in papers:
        scores = [10 - n["lacer_score"] for n in paper["neighbors"] if "lacer_score" in n]
        if scores:
            bucket_scores[paper["bucket"]].append(np.mean(scores))
    return bucket_scores


def bootstrap_ci(values, n_bootstrap, rng, ci=0.95):
    """Bootstrap mean and return (mean, lower, upper) percentile CI."""
    values = list(values)
    bootstrap_means = [np.mean(rng.choices(values, k=len(values))) for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    return np.mean(values), lower, upper


def load_raw_scores(path):
    """Load file and return raw per-paper scores grouped by bucket."""
    data, _ = utils.load_json(path)
    return compute_per_paper_scores(data)


def main():
    parser = argparse.ArgumentParser(description="Plot LACER diversity time series (web variant).")
    parser.add_argument("--natural_path", type=str, required=True, help="Path to natural LACER diversity JSON")
    parser.add_argument("--synthetic_paths", type=str, nargs="+", required=True, help="Paths to synthetic LACER diversity JSONs")
    parser.add_argument("--output_path", type=str, default="figures/multiturn/lacer_diversity_web.png", help="Output figure path")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)

    utils.log("Processing natural corpus")
    natural_scores = load_raw_scores(args.natural_path)
    natural_results = {}
    for bucket, scores in sorted(natural_scores.items()):
        mean, lower, upper = bootstrap_ci(scores, args.n_bootstrap, rng)
        natural_results[bucket] = {"mean": mean, "lower": lower, "upper": upper}

    synthetic_by_bucket = defaultdict(list)
    for path in args.synthetic_paths:
        utils.log(f"Processing {path}")
        bucket_scores = load_raw_scores(path)
        for bucket, scores in bucket_scores.items():
            synthetic_by_bucket[bucket].extend(scores)

    synth_results = {}
    for bucket in sorted(synthetic_by_bucket.keys()):
        mean, lower, upper = bootstrap_ci(synthetic_by_bucket[bucket], args.n_bootstrap, rng)
        synth_results[bucket] = {"mean": mean, "lower": lower, "upper": upper}

    utils.log("Creating plot")
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Natural (pink)
    buckets = sorted(natural_results.keys())
    dates = [datetime.strptime(b, "%Y-%m-%d") for b in buckets]
    nat_means = [natural_results[b]["mean"] for b in buckets]
    nat_lower = [natural_results[b]["lower"] for b in buckets]
    nat_upper = [natural_results[b]["upper"] for b in buckets]
    ax.plot(dates, nat_means, label="Natural", color=TEAL, linewidth=1.5)
    ax.fill_between(dates, nat_lower, nat_upper, color=TEAL, alpha=0.2)

    # Synthetic (teal)
    synth_buckets = sorted(synth_results.keys())
    synth_dates = [datetime.strptime(b, "%Y-%m-%d") for b in synth_buckets]
    synth_means = [synth_results[b]["mean"] for b in synth_buckets]
    synth_lower = [synth_results[b]["lower"] for b in synth_buckets]
    synth_upper = [synth_results[b]["upper"] for b in synth_buckets]
    ax.plot(synth_dates, synth_means, label="Synthetic", color=PINK, linewidth=1.5)
    ax.fill_between(synth_dates, synth_lower, synth_upper, color=PINK, alpha=0.2)

    ax.set_xlabel("Date", fontsize=8, color=TEAL)
    ax.set_ylabel("(10 - Mean LACERScore)", fontsize=8, color=TEAL)
    ax.legend(loc="best", fontsize=6, framealpha=0, labelcolor=TEAL)
    ax.grid(True, linestyle="--", alpha=0.2, linewidth=0.5, color=TEAL)
    ax.tick_params(axis="x", rotation=45, labelsize=7, colors=TEAL)
    ax.tick_params(axis="y", labelsize=7, colors=TEAL)
    for spine in ax.spines.values():
        spine.set_color(TEAL)

    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    utils.log(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
