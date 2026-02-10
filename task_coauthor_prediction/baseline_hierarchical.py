"""Hierarchical clustering baseline for coauthor prediction. Represents authors as cluster centroids of PCA-reduced paper embeddings."""

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
from task_coauthor_prediction.dataset import (
    create_evaluation_instances,
    get_preexisting_publications_for_author,
    get_preexisting_publications_for_corpus,
)

random.seed(42)


def get_embedding(all_embeddings, corpus_id):
    """Get embedding for a paper, handling both dict and tuple formats."""
    emb = all_embeddings[corpus_id]
    vec = emb["key"] if isinstance(emb, dict) else emb[0]
    return vec.reshape(-1)


def l2_normalize(X):
    """L2 normalize rows of matrix."""
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


def compute_all_author_centroids(cutoff_date, num_recent_papers, cluster_power, sd2publications, all_papers_dict, reduced_embeddings):
    """Pre-compute cluster centroids for all authors with publications before cutoff_date."""
    author_centroids = {}
    preexisting_pubs = get_preexisting_publications_for_corpus(cutoff_date, sd2publications, all_papers_dict)

    for author_id, pub_ids in tqdm(preexisting_pubs.items(), desc="Creating author centroids"):
        if len(pub_ids) > 0:
            recent_pubs = pub_ids[-num_recent_papers:]
            valid_pubs = [cid for cid in recent_pubs if cid in reduced_embeddings]
            if valid_pubs:
                paper_embs = [reduced_embeddings[cid] for cid in valid_pubs]
                centroids = compute_author_centroids(paper_embs, cluster_power)
                if centroids is not None:
                    author_centroids[author_id] = centroids

    utils.log(f"Created centroids for {len(author_centroids)} authors")
    return author_centroids


def update_author_centroids(author_centroids, author_id, pub_corpus_ids, cluster_power, reduced_embeddings):
    """Update an author's cluster centroids."""
    if len(pub_corpus_ids) > 0:
        valid_pubs = [cid for cid in pub_corpus_ids if cid in reduced_embeddings]
        if valid_pubs:
            paper_embs = [reduced_embeddings[cid] for cid in valid_pubs]
            centroids = compute_author_centroids(paper_embs, cluster_power)
            if centroids is not None:
                author_centroids[author_id] = centroids
    return author_centroids


def batch_max_cosine_similarity(query_centroids, candidate_centroids_list):
    """Compute max cosine similarity between query centroids and each candidate's centroids."""
    if not candidate_centroids_list:
        return np.array([])

    all_centroids = np.vstack(candidate_centroids_list)
    boundaries = np.cumsum([0] + [emb.shape[0] for emb in candidate_centroids_list])

    sim_matrix = cosine_similarity(query_centroids, all_centroids)
    max_per_centroid = sim_matrix.max(axis=0)

    scores = np.array([
        max_per_centroid[boundaries[i]:boundaries[i+1]].max()
        for i in range(len(candidate_centroids_list))
    ])

    return scores


def predict_coauthors(first_author_id, num_recent_papers, k, cutoff_date, cluster_power, sd2publications, all_papers_dict, reduced_embeddings, author_centroids, all_author_ids, excluded_ids=None):
    """Predict coauthors using hierarchical embeddings."""
    if excluded_ids is None:
        excluded_ids = set()
    excluded_ids = excluded_ids | {first_author_id}

    # Get query author's centroids from their recent papers
    author_pubs = get_preexisting_publications_for_author(first_author_id, cutoff_date, sd2publications, all_papers_dict)
    recent_pubs = author_pubs[-num_recent_papers:]
    valid_pubs = [cid for cid in recent_pubs if cid in reduced_embeddings]

    if not valid_pubs:
        return [], []

    paper_embs = [reduced_embeddings[cid] for cid in valid_pubs]
    query_centroids = compute_author_centroids(paper_embs, cluster_power)

    if query_centroids is None:
        return [], []

    # Get all candidate authors (those with centroids, excluding query)
    candidate_ids = [aid for aid in author_centroids.keys() if aid not in excluded_ids]
    if not candidate_ids:
        return [], []

    # Compute max similarity to each candidate
    candidate_centroids_list = [author_centroids[aid] for aid in candidate_ids]
    similarities = batch_max_cosine_similarity(query_centroids, candidate_centroids_list)

    # Sort by similarity descending
    sorted_indices = np.argsort(similarities)[::-1]
    predicted_ids = [candidate_ids[i] for i in sorted_indices]
    predicted_scores = [float(similarities[i]) for i in sorted_indices]

    # Pad with random authors if needed
    if len(predicted_ids) < k:
        excluded_ids = excluded_ids | set(predicted_ids)
        available_authors = list(all_author_ids - excluded_ids)
        num_needed = k - len(predicted_ids)
        random_authors = random.sample(available_authors, min(num_needed, len(available_authors)))
        predicted_ids.extend(random_authors)
        predicted_scores.extend([0.0] * len(random_authors))

    return predicted_ids[:k], predicted_scores[:k]


def main():
    parser = argparse.ArgumentParser(description="Hierarchical clustering baseline for coauthor prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing embedding files")
    parser.add_argument("--embedding_type", type=str, required=True, choices=["gtr", "grit", "specter2"], help="Type of embeddings to use")
    parser.add_argument("--num_recent_papers", type=int, default=10, help="Number of recent papers to consider per author")
    parser.add_argument("--k", type=int, default=1000, help="Number of coauthors to predict")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=1000, help="Save predictions every N instances")
    parser.add_argument("--seed_author_type", type=str, default="first", choices=["first", "last", "random", "highest_h_index"], help="Seed author selection strategy")
    parser.add_argument("--pca_dim", type=int, default=64, help="PCA output dimension")
    parser.add_argument("--cluster_power", type=float, default=0.5, help="k = n^power clusters (0.5 = sqrt(n))")
    parser.add_argument("--output_dir", type=str, default="data/task_coauthor_prediction/test/predictions", help="Output directory for predictions")
    args = parser.parse_args()

    all_papers, sd2publications, all_embeddings = utils.load_corpus(
        hf_repo_id=args.hf_repo_id,
        split=args.split,
        embeddings_dir=args.embeddings_dir,
        embedding_type=args.embedding_type,
        load_sd2publications=True
    )
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    all_author_ids = set(sd2publications.keys())
    output_path = os.path.join(args.output_dir, f"predictions.hierarchical.{args.embedding_type}.pca{args.pca_dim}.power{args.cluster_power}.{args.seed_author_type}.json")
    utils.log(f"Loaded {len(all_papers)} papers, {len(sd2publications)} authors, {len(all_embeddings)} embeddings")

    utils.log(f"Fitting PCA (dim={args.pca_dim}) and reducing embeddings")
    reduced_embeddings = fit_pca(all_embeddings, args.pca_dim)
    utils.log(f"Reduced {len(reduced_embeddings)} embeddings to {args.pca_dim} dimensions")

    utils.log(f"Creating evaluation instances with seed_author_type={args.seed_author_type}")
    evaluation_instances = create_evaluation_instances(all_papers, sd2publications, all_papers_dict, args.seed_author_type)
    eval_instance_dict = {instance["corpus_id"]: (idx, instance) for idx, (date, instance) in enumerate(evaluation_instances)}
    utils.log(f"Created {len(evaluation_instances)} evaluation instances")

    # Select random subset of instances to evaluate if max_instances is set
    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = set(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances to evaluate")
    else:
        selected_indices = set(range(len(evaluation_instances)))

    # Sort all papers by date and create initial author centroids
    all_papers = sorted(all_papers, key=lambda p: p["date"])
    first_date = evaluation_instances[0][0]
    utils.log(f"Creating initial author centroids with cutoff date: {first_date}")
    author_centroids = compute_all_author_centroids(first_date, args.num_recent_papers, args.cluster_power, sd2publications, all_papers_dict, reduced_embeddings)

    # Filter to papers on or after first_date
    postdated_papers = [p for p in all_papers if p["date"] >= first_date]

    utils.log(f"Running hierarchical baseline with embedding: {args.embedding_type}, pca_dim: {args.pca_dim}, cluster_power: {args.cluster_power}")
    predictions = []
    for paper in tqdm(postdated_papers, desc="Running hierarchical baseline"):
        corpus_id = paper["corpus_id"]
        date = paper["date"]

        # Update authors on this paper before prediction
        if "authors" in paper:
            for author in paper["authors"]:
                author_id = author["author_id"]
                author_pubs = get_preexisting_publications_for_author(author_id, date, sd2publications, all_papers_dict)
                recent_pubs = author_pubs[-args.num_recent_papers:]
                author_centroids = update_author_centroids(author_centroids, author_id, recent_pubs, args.cluster_power, reduced_embeddings)

        # If this is a selected evaluation instance, run prediction
        if corpus_id in eval_instance_dict:
            idx, instance = eval_instance_dict[corpus_id]
            if idx in selected_indices:
                first_author_id = instance["first_author_id"]
                gt_coauthor_ids = instance["gt_coauthor_ids"]

                pred_ids, pred_scores = predict_coauthors(
                    first_author_id, args.num_recent_papers, args.k, date, args.cluster_power,
                    sd2publications, all_papers_dict, reduced_embeddings, author_centroids, all_author_ids
                )

                predictions.append({
                    "corpus_id": corpus_id,
                    "first_author_id": first_author_id,
                    "gt_coauthor_ids": gt_coauthor_ids,
                    "predicted_coauthor_ids": pred_ids,
                    "predicted_coauthor_scores": pred_scores,
                })

                if len(predictions) % args.save_every == 0:
                    utils.save_json(predictions, output_path, metadata=utils.update_metadata({}, args), overwrite=True)

    utils.save_json(predictions, output_path, metadata=utils.update_metadata({}, args), overwrite=True)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
