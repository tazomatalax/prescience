"""Rank fusion baseline for coauthor prediction. Queries each paper individually and fuses in rank space."""

import os
import random
import argparse
from tqdm import tqdm

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


def create_mean_author_embedding(author_pub_corpus_ids, all_embeddings, distance_metric):
    """Create mean embedding for an author from their paper embeddings."""
    embeddings = [get_embedding(all_embeddings, cid) for cid in author_pub_corpus_ids]
    return utils.aggregate_embeddings(embeddings, distance_metric=distance_metric)


def create_author_index(cutoff_date, num_recent_papers, sd2publications, all_papers_dict, all_embeddings, distance_metric):
    """Create FAISS index of author embeddings using recent publications before cutoff_date."""
    author_embeddings = {}
    preexisting_pubs = get_preexisting_publications_for_corpus(cutoff_date, sd2publications, all_papers_dict)

    for author_id, pub_ids in tqdm(preexisting_pubs.items(), desc="Creating author embeddings"):
        if len(pub_ids) > 0:
            recent_pubs = pub_ids[-num_recent_papers:]
            author_embeddings[author_id] = create_mean_author_embedding(recent_pubs, all_embeddings, distance_metric)

    utils.log(f"Created embeddings for {len(author_embeddings)} authors")
    return utils.create_index(author_embeddings, distance_metric)


def update_author_in_index(index, author_id, pub_corpus_ids, all_embeddings, distance_metric):
    """Update an author's embedding in the index."""
    if len(pub_corpus_ids) > 0:
        embedding = create_mean_author_embedding(pub_corpus_ids, all_embeddings, distance_metric)
        index = utils.replace_vector_in_index(index, author_id, embedding)
    return index


def query_and_fuse_ranks(multiplicity, k, index, excluded_authors, all_embeddings, distance_metric):
    """Query index with papers weighted by multiplicity and fuse ranks."""
    # Query index with each unique paper's embedding
    unique_paper_ids = list(multiplicity.keys())
    pub_embeddings = [get_embedding(all_embeddings, cid) for cid in unique_paper_ids]
    retrieved_authors_lists, _ = utils.query_index(index, pub_embeddings, k + len(excluded_authors))

    # Combine results by summing ranks weighted by multiplicity, excluding specified authors and None values
    summed_ranks = {}
    for paper_id, retrieved_authors_list in zip(unique_paper_ids, retrieved_authors_lists):
        weight = multiplicity[paper_id]
        for rank, author_id in enumerate(retrieved_authors_list):
            if author_id is not None and author_id not in excluded_authors:
                if author_id not in summed_ranks:
                    summed_ranks[author_id] = 0
                summed_ranks[author_id] += rank * weight

    return summed_ranks


def predict_coauthors_one_shot(first_author_id, num_recent_papers, k, index, cutoff_date, sd2publications, all_papers_dict, all_embeddings, distance_metric, all_author_ids, excluded_ids=None):
    """One-shot strategy: Query with first author's recent papers and fuse ranks."""
    if excluded_ids is None:
        excluded_ids = set()
    excluded_ids = excluded_ids | {first_author_id}

    # Get recent publications for the first author
    author_pubs = get_preexisting_publications_for_author(first_author_id, cutoff_date, sd2publications, all_papers_dict)
    recent_pubs = author_pubs[-num_recent_papers:]

    # Query index with each publication's embedding
    pub_embeddings = [get_embedding(all_embeddings, cid) for cid in recent_pubs]
    retrieved_authors_lists, _ = utils.query_index(index, pub_embeddings, k + len(excluded_ids))

    # Combine results by summing ranks across all paper queries, excluding specified authors and None values
    summed_ranks = {}
    for retrieved_authors_list in retrieved_authors_lists:
        for rank, author_id in enumerate(retrieved_authors_list):
            if author_id is not None and author_id not in excluded_ids:
                if author_id not in summed_ranks:
                    summed_ranks[author_id] = 0
                summed_ranks[author_id] += rank

    # Sort by summed rank (lower is better)
    sorted_by_rank = sorted(summed_ranks.items(), key=lambda x: x[1])
    predicted_ids = [aid for aid, _ in sorted_by_rank]
    predicted_scores = [score for _, score in sorted_by_rank]

    # Pad with random authors if needed
    if len(predicted_ids) < k:
        excluded_ids = excluded_ids | set(predicted_ids)
        available_authors = list(all_author_ids - excluded_ids)
        num_needed = k - len(predicted_ids)
        random_authors = random.sample(available_authors, min(num_needed, len(available_authors)))
        predicted_ids.extend(random_authors)
        predicted_scores.extend([float('inf')] * len(random_authors))

    return predicted_ids[:k], predicted_scores[:k]


def predict_coauthors_iterated(first_author_id, num_recent_papers, num_to_predict, k, index, cutoff_date, sd2publications, all_papers_dict, all_embeddings, distance_metric, all_author_ids, excluded_ids=None):
    """Iterated strategy: Iteratively predict coauthors, adding each predicted author's papers to the query set with multiplicity."""
    if excluded_ids is None:
        excluded_ids = set()

    # Get recent publications for the first author and build multiplicity dict
    author_pubs = get_preexisting_publications_for_author(first_author_id, cutoff_date, sd2publications, all_papers_dict)
    recent_pubs = author_pubs[-num_recent_papers:]
    multiplicity = {cid: 1 for cid in recent_pubs}

    predicted_ids = []
    predicted_scores = []
    excluded_authors = excluded_ids | {first_author_id}

    # Iteratively predict coauthors
    iterations_remaining = num_to_predict
    while iterations_remaining > 0:
        summed_ranks = query_and_fuse_ranks(multiplicity, 1, index, excluded_authors, all_embeddings, distance_metric)

        # Get the best candidate (lowest summed rank)
        if len(summed_ranks) == 0:
            break
        sorted_by_rank = sorted(summed_ranks.items(), key=lambda x: x[1])
        next_author_id, next_author_score = sorted_by_rank[0]

        predicted_ids.append(next_author_id)
        predicted_scores.append(next_author_score)
        excluded_authors.add(next_author_id)

        # Add the predicted author's recent papers to the query set (incrementing multiplicity)
        next_author_pubs = get_preexisting_publications_for_author(next_author_id, cutoff_date, sd2publications, all_papers_dict)
        next_recent_pubs = next_author_pubs[-num_recent_papers:]
        for cid in next_recent_pubs:
            if cid not in multiplicity:
                multiplicity[cid] = 0
            multiplicity[cid] += 1

        iterations_remaining -= 1

    # Pad to k with remaining authors from final rank fusion
    summed_ranks = query_and_fuse_ranks(multiplicity, k, index, excluded_authors, all_embeddings, distance_metric)
    sorted_by_rank = sorted(summed_ranks.items(), key=lambda x: x[1])

    for author_id, score in sorted_by_rank:
        if len(predicted_ids) >= k:
            break
        if author_id is not None and author_id not in excluded_authors:
            predicted_ids.append(author_id)
            predicted_scores.append(score)
            excluded_authors.add(author_id)

    # Pad with random authors if needed
    if len(predicted_ids) < k:
        available_authors = list(all_author_ids - excluded_authors)
        num_needed = k - len(predicted_ids)
        random_authors = random.sample(available_authors, min(num_needed, len(available_authors)))
        predicted_ids.extend(random_authors)
        predicted_scores.extend([float('inf')] * len(random_authors))

    return predicted_ids[:k], predicted_scores[:k]


def main():
    parser = argparse.ArgumentParser(description="Rank fusion baseline for coauthor prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing embedding files")
    parser.add_argument("--embedding_type", type=str, required=True, choices=["gtr", "grit", "specter2"])
    parser.add_argument("--num_recent_papers", type=int, default=10, help="Number of recent papers to consider")
    parser.add_argument("--k", type=int, default=1000, help="Number of coauthors to predict")
    parser.add_argument("--strategy", type=str, default="one_shot", choices=["one_shot", "iterated"])
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=1000, help="Save predictions every N instances")
    parser.add_argument("--seed_author_type", type=str, default="first", choices=["first", "last", "random", "highest_h_index"], help="Seed author selection strategy")
    parser.add_argument("--output_dir", type=str, default="data/task_coauthor_prediction/test/predictions")
    args = parser.parse_args()

    all_papers, sd2publications, all_embeddings = utils.load_corpus(
        hf_repo_id=args.hf_repo_id,
        split=args.split,
        embeddings_dir=args.embeddings_dir,
        embedding_type=args.embedding_type,
        load_sd2publications=True
    )
    distance_metric = "l2" if args.embedding_type == "specter2" else "cosine"
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    all_author_ids = set(sd2publications.keys())
    output_path = os.path.join(args.output_dir, f"predictions.rank_fusion.{args.embedding_type}.{args.strategy}.{args.seed_author_type}.json")
    utils.log(f"Loaded {len(all_papers)} papers, {len(sd2publications)} authors, {len(all_embeddings)} embeddings")

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

    # Sort all papers by date and create initial index with authors who have publications before first evaluation instance
    all_papers = sorted(all_papers, key=lambda p: p["date"])
    first_date = evaluation_instances[0][0]
    utils.log(f"Creating initial author index with cutoff date: {first_date}")
    index = create_author_index(first_date, args.num_recent_papers, sd2publications, all_papers_dict, all_embeddings, distance_metric)

    # Filter to papers on or after first_date (earlier papers are already in initial index)
    postdated_papers = [p for p in all_papers if p["date"] >= first_date]

    utils.log(f"Running rank fusion baseline with strategy: {args.strategy}, embedding: {args.embedding_type}")
    predictions = []
    for paper in tqdm(postdated_papers, desc="Running rank fusion baseline"):
        corpus_id = paper["corpus_id"]
        date = paper["date"]

        # Update authors on this paper before prediction.
        # This correctly handles both evaluation instances and non-evaluation instances (whether they're target or target.author.publication_history).
        # 1) evaluation instances: authors are updated with pubs < date, then we predict.
        # 2) non-evaluation instances: authors are still updated so their embeddings stay current for future predictions.
        if "authors" in paper:
            for author in paper["authors"]:
                author_id = author["author_id"]
                author_pubs = get_preexisting_publications_for_author(author_id, date, sd2publications, all_papers_dict)
                recent_pubs = author_pubs[-args.num_recent_papers:]
                index = update_author_in_index(index, author_id, recent_pubs, all_embeddings, distance_metric)

        # If this is a selected evaluation instance, run prediction
        if corpus_id in eval_instance_dict:
            idx, instance = eval_instance_dict[corpus_id]
            if idx in selected_indices:
                first_author_id = instance["first_author_id"]
                gt_coauthor_ids = instance["gt_coauthor_ids"]
                num_to_predict = len(gt_coauthor_ids)

                if args.strategy == "one_shot":
                    pred_ids, pred_scores = predict_coauthors_one_shot(
                        first_author_id, args.num_recent_papers, args.k, index, date,
                        sd2publications, all_papers_dict, all_embeddings, distance_metric, all_author_ids
                    )
                else:
                    pred_ids, pred_scores = predict_coauthors_iterated(
                        first_author_id, args.num_recent_papers, num_to_predict, args.k, index, date,
                        sd2publications, all_papers_dict, all_embeddings, distance_metric, all_author_ids
                    )

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
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
