"""Render Kendall tau-b agreement matrix across humans, automated metrics, and LACER variants."""

import argparse
import json
import math
from itertools import combinations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

LABELS = ["Human", "MRR", "BERTScore", "ASPIRE Distance", "FacetScore", "LACER (GPT-5)", "LACER (Opus)", "LACER (Avg)"]
GENERATION_COUNT = 10


def argsort_ranking(ranking_str):
    """Parse a ranking string into rank values."""
    s = ranking_str.replace(" ", "").strip()
    elements = []
    i = 0
    while i < len(s):
        if s[i].isdigit():
            num = ""
            while i < len(s) and s[i].isdigit():
                num += s[i]
                i += 1
            elements.append(int(num))
        else:
            elements.append(s[i])
            i += 1

    ranks = [0] * GENERATION_COUNT
    rank = 1
    for element in elements:
        if isinstance(element, int):
            ranks[element - 1] = rank
        elif element == "=":
            continue
        elif element == ">":
            rank += 1
    return ranks


def parse_ground_truth_title(text):
    """Extract title from ground_truth text."""
    if not text:
        return ""
    parts = text.split("\n", 1)
    return parts[0].strip()


def load_json(path):
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def human_vectors_from_record(record):
    """Extract human ranking vectors from a record."""
    vectors = {}
    rankings = record.get("rankings", {})
    if not isinstance(rankings, dict):
        return vectors
    for annotator_id, ranking in rankings.items():
        if not isinstance(ranking, str) or not ranking.strip():
            continue
        human_ranks = argsort_ranking(ranking)
        if not human_ranks:
            continue
        max_rank = max(human_ranks)
        vectors[annotator_id] = [float(max_rank - rank) for rank in human_ranks]
    return vectors


def lacer_vector_from_record(record):
    """Extract LACER score vector from a record."""
    lacer_scores = record.get("lacer_scores")
    if not isinstance(lacer_scores, dict):
        return None
    vector = [None] * GENERATION_COUNT
    for key, details in lacer_scores.items():
        try:
            idx = int(key) - 1
        except (TypeError, ValueError):
            continue
        if not isinstance(details, dict):
            continue
        score = details.get("score")
        if isinstance(score, (int, float)) and 0 <= idx < GENERATION_COUNT:
            vector[idx] = float(score)
    if any(v is None for v in vector):
        return None
    return vector


def average_lacer_vectors(gpt5_vector, opus_vector):
    """Compute element-wise average of two LACER vectors."""
    if gpt5_vector is None or opus_vector is None:
        return None
    return [(g + o) / 2 for g, o in zip(gpt5_vector, opus_vector)]


def metric_vectors_from_dataset(entry):
    """Extract automated metric vectors from annotation dataset entry."""
    generated = entry.get("generated_outputs")
    if not isinstance(generated, list):
        return {}

    mrr_values = []
    bert_values = []
    aspire_values = []
    facet_values = []

    for output in generated[:GENERATION_COUNT]:
        if not isinstance(output, dict):
            continue
        mrr = output.get("mrr")
        bert = output.get("bertscore_microsoft/deberta-xlarge-mnli")
        aspire = output.get("ot_distance")
        facet_ratings = output.get("facet_ratings", [])
        if isinstance(facet_ratings, list) and facet_ratings:
            ratings = [r.get("rating") for r in facet_ratings if isinstance(r, dict) and isinstance(r.get("rating"), (int, float))]
            facet = float(np.mean(ratings)) if ratings else None
        else:
            facet = None

        mrr_values.append(float(mrr) if isinstance(mrr, (int, float)) else None)
        bert_values.append(float(bert) if isinstance(bert, (int, float)) else None)
        aspire_values.append(-float(aspire) if isinstance(aspire, (int, float)) else None)
        facet_values.append(facet)

    def finalize(values):
        if len(values) < GENERATION_COUNT or any(v is None for v in values):
            return None
        return [float(v) for v in values]

    return {
        "MRR": finalize(mrr_values),
        "BERTScore": finalize(bert_values),
        "ASPIRE Distance": finalize(aspire_values),
        "FacetScore": finalize(facet_values),
    }


def collect_annotator_ids(records):
    """Collect all annotator IDs from records."""
    annotator_ids = set()
    for record in records:
        rankings = record.get("rankings", {})
        if isinstance(rankings, dict):
            for annotator_id, ranking in rankings.items():
                if isinstance(annotator_id, str) and isinstance(ranking, str) and ranking.strip():
                    annotator_ids.add(annotator_id)
    return sorted(annotator_ids)


def is_valid_vector(vector):
    """Check if vector is valid for correlation computation."""
    if vector is None or len(vector) == 0:
        return False
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector)


def kendall_tau(vector_a, vector_b):
    """Compute Kendall tau-b between two vectors."""
    if len(vector_a) != len(vector_b) or len(vector_a) == 0:
        return None
    result = kendalltau(vector_a, vector_b, variant="b")
    if result.statistic is None or math.isnan(result.statistic):
        return None
    return float(result.statistic)


def compute_human_human_value(records_data, annotator_ids):
    """Compute average Kendall tau between human annotator pairs."""
    pair_scores = []
    total_pairs = 0
    for annotator_a, annotator_b in combinations(annotator_ids, 2):
        tau_values = []
        for record in records_data:
            vec_a = record["humans"].get(annotator_a)
            vec_b = record["humans"].get(annotator_b)
            if is_valid_vector(vec_a) and is_valid_vector(vec_b):
                tau = kendall_tau(vec_a, vec_b)
                if tau is not None:
                    tau_values.append(tau)
        if tau_values:
            pair_scores.append(float(np.mean(tau_values)))
            total_pairs += len(tau_values)
    if not pair_scores:
        return float("nan"), 0
    return float(np.mean(pair_scores)), total_pairs


def compute_human_metric_value(records_data, annotator_ids, metric_label):
    """Compute average Kendall tau between humans and a metric."""
    annotator_scores = []
    total_counts = 0
    for annotator_id in annotator_ids:
        tau_values = []
        for record in records_data:
            human_vec = record["humans"].get(annotator_id)
            metric_vec = record["metrics"].get(metric_label)
            if is_valid_vector(human_vec) and is_valid_vector(metric_vec):
                tau = kendall_tau(human_vec, metric_vec)
                if tau is not None:
                    tau_values.append(tau)
        if tau_values:
            annotator_scores.append(float(np.mean(tau_values)))
            total_counts += len(tau_values)
    if not annotator_scores:
        return float("nan"), 0
    return float(np.mean(annotator_scores)), total_counts


def compute_metric_metric_value(records_data, metric_a, metric_b):
    """Compute average Kendall tau between two metrics."""
    tau_values = []
    for record in records_data:
        vec_a = record["metrics"].get(metric_a)
        vec_b = record["metrics"].get(metric_b)
        if is_valid_vector(vec_a) and is_valid_vector(vec_b):
            tau = kendall_tau(vec_a, vec_b)
            if tau is not None:
                tau_values.append(tau)
    if not tau_values:
        return float("nan"), 0
    return float(np.mean(tau_values)), len(tau_values)


def save_kendall_matrix_plot(labels, matrix, counts, output_path):
    """Save Kendall tau matrix as heatmap plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    base_cmap = plt.colormaps.get_cmap("Blues")
    colors = base_cmap(np.linspace(0.125, 0.7, 256))
    light_blues = mcolors.LinearSegmentedColormap.from_list("light_blues", colors)
    norm = mcolors.Normalize(vmin=0, vmax=1.0, clip=True)
    im = ax.imshow(matrix, cmap=light_blues, norm=norm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            if i == j and i != 0:
                text = ""
            elif np.isnan(value):
                text = "NA"
            else:
                text = f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)

    ax.set_title("Kendall Tau-b Across Humans and Metrics")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render Kendall tau-b matrix for humans, automated metrics, and LACER variants")
    parser.add_argument("--gpt5_path", default="data/task_followup_prediction/metrics_analysis/annotator_rankings_rows_2_6.gpt5_scored.percentile50.110.json", help="Path to GPT-5 LACER scored JSON")
    parser.add_argument("--opus_path", default="data/task_followup_prediction/metrics_analysis/annotator_rankings_rows_2_6.opus_scored.percentile50.110.json", help="Path to Opus LACER scored JSON")
    parser.add_argument("--dataset_path", default="data/task_followup_prediction/metrics_analysis/annotation_dataset.json", help="Path to annotation_dataset.json with automated metrics")
    parser.add_argument("--output_path", default="data/task_followup_prediction/metrics_analysis/metrics_kendall_matrix.png", help="Path for the saved matrix plot")
    args = parser.parse_args()

    # Load scored records from both judges
    gpt5_records = load_json(args.gpt5_path)
    opus_records = load_json(args.opus_path)
    annotator_ids = collect_annotator_ids(gpt5_records)

    # Load annotation dataset and index by title
    dataset = load_json(args.dataset_path)
    dataset_mapping = {}
    for entry in dataset:
        title = entry.get("target", {}).get("title")
        if isinstance(title, str):
            dataset_mapping[title.strip()] = entry

    # Build records data with all vectors
    records_data = []
    for gpt5_record, opus_record in zip(gpt5_records, opus_records):
        title = parse_ground_truth_title(gpt5_record.get("ground_truth", ""))
        dataset_entry = dataset_mapping.get(title)
        if dataset_entry is None:
            raise KeyError(f"No annotation dataset entry found for title: {title}")

        human_vectors = human_vectors_from_record(gpt5_record)
        gpt5_vector = lacer_vector_from_record(gpt5_record)
        opus_vector = lacer_vector_from_record(opus_record)
        avg_vector = average_lacer_vectors(gpt5_vector, opus_vector)
        metric_vectors = metric_vectors_from_dataset(dataset_entry)
        metric_vectors["LACER (GPT-5)"] = gpt5_vector
        metric_vectors["LACER (Opus)"] = opus_vector
        metric_vectors["LACER (Avg)"] = avg_vector

        records_data.append({"humans": human_vectors, "metrics": metric_vectors})

    # Build correlation matrix
    size = len(LABELS)
    matrix = np.full((size, size), np.nan)
    counts = np.zeros((size, size), dtype=int)

    # Human-Human cell
    value, count = compute_human_human_value(records_data, annotator_ids)
    matrix[0, 0] = value
    counts[0, 0] = count

    # Human vs metrics
    for idx, label in enumerate(LABELS[1:], start=1):
        value, count = compute_human_metric_value(records_data, annotator_ids, label)
        matrix[0, idx] = value
        matrix[idx, 0] = value
        counts[0, idx] = counts[idx, 0] = count

    # Metric vs metric
    for i, label_a in enumerate(LABELS[1:], start=1):
        for j, label_b in enumerate(LABELS[1:], start=1):
            if j <= i:
                continue
            value, count = compute_metric_metric_value(records_data, label_a, label_b)
            matrix[i, j] = value
            matrix[j, i] = value
            counts[i, j] = counts[j, i] = count

    # Print matrix values
    for row_label, row_values in zip(LABELS, matrix):
        formatted = ", ".join(f"{v:.4f}" if not np.isnan(v) else "nan" for v in row_values)
        print(f"{row_label}: {formatted}")

    save_kendall_matrix_plot(LABELS, matrix, counts, args.output_path)
    print(f"Saved metrics Kendall tau matrix plot to {args.output_path}")


if __name__ == "__main__":
    main()
