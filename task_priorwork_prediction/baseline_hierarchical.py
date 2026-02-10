"""Hierarchical clustering baseline for prior work prediction. Represents authors as cluster centroids of PCA-reduced paper embeddings."""

import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

import utils
from task_priorwork_prediction.dataset import (
    create_evaluation_instances,
    get_preexisting_publications_for_author,
)

random.seed(42)


def get_embedding(all_embeddings, corpus_id):
    """Get key embedding for a paper."""
    emb = all_embeddings[corpus_id]
    vec = emb["key"] if isinstance(emb, dict) else emb[0]
    return vec.reshape(-1)


def l2_normalize(X):
    """L2 normalize rows of matrix (or single vector)."""
    if X.ndim == 1:
        norm = np.linalg.norm(X)
        return X / norm if norm > 1e-12 else X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return X / norms


def fit_pca(all_embeddings, pca_dim):
    """Fit PCA on all paper embeddings and return reduced embeddings dict."""
    corpus_ids = list(all_embeddings.keys())
    embeddings = np.stack([get_embedding(all_embeddings, cid) for cid in corpus_ids])
    pca = PCA(n_components=pca_dim, random_state=42)
    reduced = pca.fit_transform(embeddings)
    reduced = l2_normalize(reduced)
    return {cid: reduced[i] for i, cid in enumerate(corpus_ids)}


def hierarchical_cluster(X, power):
    """Hierarchical clustering with cosine distance and average linkage."""
    n = X.shape[0]
    k = max(1, int(np.ceil(n ** power)))

    if n == 1:
        return np.array([1])

    dist = pdist(X, metric="cosine")
    Z = linkage(dist, method="average")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels


def compute_author_centroids(paper_embeddings, cluster_power):
    """Compute L2-normalized cluster centroids for an author's papers."""
    if len(paper_embeddings) == 0:
        return None

    X = np.stack(paper_embeddings)

    if X.shape[0] == 1:
        return X

    labels = hierarchical_cluster(X, cluster_power)
    centroids = []
    for c in np.unique(labels):
        cluster_mean = X[labels == c].mean(axis=0)
        norm = np.linalg.norm(cluster_mean)
        if norm > 1e-12:
            cluster_mean = cluster_mean / norm
        centroids.append(cluster_mean)

    return np.stack(centroids)


def predict_references(author_ids, cutoff_date, num_recent_papers, k, cluster_power, sd2publications, all_papers_dict, reduced_embeddings, candidate_paper_ids):
    """Predict references using hierarchical author embeddings."""
    # Collect all centroids across all authors
    all_author_centroids = []
    for author_id in author_ids:
        author_pubs = get_preexisting_publications_for_author(author_id, cutoff_date, sd2publications, all_papers_dict)
        if not author_pubs:
            continue
        recent_pubs = author_pubs[-num_recent_papers:]
        valid_pubs = [cid for cid in recent_pubs if cid in reduced_embeddings]
        if valid_pubs:
            paper_embs = [reduced_embeddings[cid] for cid in valid_pubs]
            centroids = compute_author_centroids(paper_embs, cluster_power)
            if centroids is not None:
                all_author_centroids.append(centroids)

    # Handle case where no authors have valid embeddings
    if not all_author_centroids:
        return [], []

    # Stack all centroids from all authors
    query_centroids = np.vstack(all_author_centroids)

    # Get candidate paper embeddings
    valid_candidates = [cid for cid in candidate_paper_ids if cid in reduced_embeddings]
    if not valid_candidates:
        return [], []

    candidate_embeddings = np.stack([reduced_embeddings[cid] for cid in valid_candidates])

    # Compute max similarity: for each candidate paper, max similarity to any centroid
    sim_matrix = cosine_similarity(query_centroids, candidate_embeddings)
    similarities = sim_matrix.max(axis=0)

    # Sort by similarity descending
    sorted_indices = np.argsort(similarities)[::-1]
    predicted_ids = [valid_candidates[i] for i in sorted_indices]
    predicted_scores = [float(similarities[i]) for i in sorted_indices]

    return predicted_ids[:k], predicted_scores[:k]


def main():
    parser = argparse.ArgumentParser(description="Hierarchical clustering baseline for prior work prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing embeddings pkl files")
    parser.add_argument("--embedding_type", type=str, required=True, choices=["gtr", "grit", "specter2"], help="Type of embeddings to use")
    parser.add_argument("--num_recent_papers", type=int, default=10, help="Number of recent papers to consider per author")
    parser.add_argument("--k", type=int, default=1000, help="Number of references to predict")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=5000, help="Save predictions every N instances")
    parser.add_argument("--pca_dim", type=int, default=64, help="PCA output dimension")
    parser.add_argument("--cluster_power", type=float, default=0.5, help="k = n^power clusters (0.5 = sqrt(n))")
    parser.add_argument("--output_dir", type=str, default="data/task_priorwork_prediction/test/predictions", help="Output directory for predictions")
    args = parser.parse_args()

    utils.log(f"Loading corpus from {args.hf_repo_id} (split={args.split})")
    all_papers, sd2publications, all_embeddings = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=args.embeddings_dir, embedding_type=args.embedding_type, load_sd2publications=True)
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    output_path = os.path.join(args.output_dir, f"predictions.hierarchical.{args.embedding_type}.pca{args.pca_dim}.power{args.cluster_power}.json")
    utils.log(f"Loaded {len(all_papers)} papers, {len(sd2publications)} authors, {len(all_embeddings)} embeddings")

    utils.log(f"Fitting PCA (dim={args.pca_dim}) and reducing embeddings")
    reduced_embeddings = fit_pca(all_embeddings, args.pca_dim)
    utils.log(f"Reduced {len(reduced_embeddings)} embeddings to {args.pca_dim} dimensions")

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

    # Sort all papers by date and build initial candidate set
    all_papers = sorted(all_papers, key=lambda p: p["date"])
    first_date = evaluation_instances[0][0]
    utils.log(f"First evaluation date: {first_date}")

    # Build initial set of candidate papers (papers before first_date)
    candidate_paper_ids = set()
    for paper in all_papers:
        if paper["date"] < first_date:
            candidate_paper_ids.add(paper["corpus_id"])
    utils.log(f"Initial candidate pool: {len(candidate_paper_ids)} papers")

    # Filter to papers on or after first_date
    postdated_papers = [p for p in all_papers if p["date"] >= first_date]

    utils.log(f"Running hierarchical baseline with embedding: {args.embedding_type}, pca_dim: {args.pca_dim}, cluster_power: {args.cluster_power}")
    predictions = []
    for paper in tqdm(postdated_papers, desc="Running hierarchical baseline"):
        corpus_id = paper["corpus_id"]

        # If this is a selected evaluation instance, run prediction before adding to candidates
        if corpus_id in eval_instance_dict:
            idx, instance = eval_instance_dict[corpus_id]
            if idx in selected_indices:
                author_ids = instance["author_ids"]
                gt_reference_ids = instance["gt_reference_ids"]

                predicted_ids, predicted_scores = predict_references(
                    author_ids, paper["date"], args.num_recent_papers, args.k, args.cluster_power,
                    sd2publications, all_papers_dict, reduced_embeddings, candidate_paper_ids
                )

                predictions.append({
                    "corpus_id": corpus_id,
                    "gt_reference_ids": gt_reference_ids,
                    "predicted_reference_ids": predicted_ids,
                    "predicted_reference_scores": predicted_scores,
                })

                if len(predictions) % args.save_every == 0:
                    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)

        # Add paper to candidate pool
        candidate_paper_ids.add(corpus_id)

    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
