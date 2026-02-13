"""CoGPT baseline for coauthor prediction. Uses a GPT transformer to predict next author embeddings."""

import os
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm

import utils
from task_coauthor_prediction.dataset import create_evaluation_instances, get_preexisting_publications_for_author, get_preexisting_publications_for_corpus
from task_coauthor_prediction.baseline_retrieval import create_mean_author_embedding, create_author_index, update_author_in_index
from task_coauthor_prediction.model.gpt4rec_dae import GPT4RecDAE

random.seed(42)


def load_model(model_path, n_layers, n_heads, hidden_size, max_len, device):
    """Load pretrained GPT model."""
    model = GPT4RecDAE(max_len=max_len, n_layers=n_layers, n_heads=n_heads, hidden_size=hidden_size, p_dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def build_prefix_embeddings(first_author_id, cutoff_date, sd2publications, all_papers_dict, all_embeddings, distance_metric):
    """Build prefix embeddings for GPT input: coauthor embeddings from first author's past papers."""
    first_author_pubs = get_preexisting_publications_for_author(first_author_id, cutoff_date, sd2publications, all_papers_dict)

    segments = []
    for paper_id in first_author_pubs:
        paper = all_papers_dict[paper_id]
        paper_date = paper["date"]

        segment = []
        for author in paper["authors"]:
            coauthor_id = author["author_id"]
            if coauthor_id != first_author_id:
                coauthor_pubs = get_preexisting_publications_for_author(coauthor_id, paper_date, sd2publications, all_papers_dict)
                if len(coauthor_pubs) > 0:
                    coauthor_embedding = create_mean_author_embedding(coauthor_pubs, all_embeddings, distance_metric)
                    segment.append(coauthor_embedding)
        segments.append(segment)

    return segments


def flatten_for_gpt(segments, author_vecs, max_len):
    """Flatten segments + current author vecs into tensors for GPT input."""
    all_segments = segments + [author_vecs]

    flat = []
    pos_ids = []
    for seg_id, segment in enumerate(all_segments):
        for vec in segment:
            t = torch.tensor(vec, dtype=torch.float32)
            if t.ndim == 2 and t.size(0) == 1:
                t = t.squeeze(0)
            flat.append(t)
            pos_ids.append(seg_id)

    if len(flat) > max_len:
        flat = flat[-max_len:]
        pos_ids = pos_ids[-max_len:]
        idx = 0
        new_map = {}
        for p in pos_ids:
            if p not in new_map:
                new_map[p] = idx
                idx += 1
        pos_ids = [new_map[p] for p in pos_ids]

    return flat, pos_ids


def collate_for_gpt(flat_vecs):
    """Collate flattened vectors into batch format for GPT."""
    L = len(flat_vecs)
    H = flat_vecs[0].numel()
    ace = torch.zeros(1, L, H)
    attn = torch.zeros(1, L)
    ace[0, :L] = torch.stack(flat_vecs)
    attn[0, :L] = 1.0
    return {"ace_vecs": ace, "attention_mask": attn.bool()}


def gpt_forward(model, batch, device):
    """Run GPT forward pass and return predicted embedding for next position."""
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        pred = model(batch)
    return pred[0, -1].cpu().numpy()


def predict_coauthors_one_shot(first_author_id, k, index, model, prefix_segments, distance_metric, max_len, device, all_author_ids):
    """One-shot strategy: Query once with GPT's predicted embedding, return top-k."""
    first_author_embedding = utils.read_vector_from_index(index, first_author_id)
    author_vecs = [first_author_embedding]

    flat, pos_ids = flatten_for_gpt(prefix_segments, author_vecs, max_len)
    if len(flat) == 0:
        query_embedding = utils.aggregate_embeddings([first_author_embedding], distance_metric)
    else:
        batch = collate_for_gpt(flat)
        query_embedding = gpt_forward(model, batch, device)
        if distance_metric == "cosine":
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

    query_embedding = query_embedding.reshape(1, -1)
    [retrieved_ids], [distances] = utils.query_index(index, query_embedding, k + 1)

    predicted_ids = []
    predicted_scores = []
    for author_id, dist in zip(retrieved_ids, distances):
        if author_id != first_author_id:
            predicted_ids.append(author_id)
            predicted_scores.append(float(dist))

    if len(predicted_ids) < k:
        excluded = {first_author_id} | set(predicted_ids)
        available_authors = list(all_author_ids - excluded)
        num_needed = k - len(predicted_ids)
        random_authors = random.sample(available_authors, num_needed)
        predicted_ids.extend(random_authors)
        predicted_scores.extend([float('inf')] * num_needed)

    return predicted_ids[:k], predicted_scores[:k]


def predict_coauthors_iterated(first_author_id, num_to_predict, k, index, model, prefix_segments, distance_metric, max_len, device, all_author_ids):
    """Iterated strategy: Iteratively predict, adding each predicted author to GPT's context."""
    predicted_ids = []
    predicted_scores = []
    excluded = {first_author_id}

    first_author_embedding = utils.read_vector_from_index(index, first_author_id)
    author_vecs = [first_author_embedding]

    iterations_remaining = num_to_predict
    while iterations_remaining > 0:
        flat, pos_ids = flatten_for_gpt(prefix_segments, author_vecs, max_len)
        if len(flat) == 0:
            query_embedding = utils.aggregate_embeddings(author_vecs, distance_metric)
        else:
            batch = collate_for_gpt(flat)
            query_embedding = gpt_forward(model, batch, device)
            if distance_metric == "cosine":
                query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

        query_embedding = query_embedding.reshape(1, -1)
        [retrieved_ids], [distances] = utils.query_index(index, query_embedding, k + len(excluded))

        next_author_id = None
        next_author_score = None
        for author_id, dist in zip(retrieved_ids, distances):
            if author_id not in excluded:
                next_author_id = author_id
                next_author_score = float(dist)
                break

        if next_author_id is None:
            break

        predicted_ids.append(next_author_id)
        predicted_scores.append(next_author_score)
        excluded.add(next_author_id)
        author_vecs.append(utils.read_vector_from_index(index, next_author_id))
        iterations_remaining -= 1

    # Pad to k with remaining retrieved authors
    flat, pos_ids = flatten_for_gpt(prefix_segments, author_vecs, max_len)
    if len(flat) > 0:
        batch = collate_for_gpt(flat)
        query_embedding = gpt_forward(model, batch, device)
        if distance_metric == "cosine":
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        query_embedding = query_embedding.reshape(1, -1)
    else:
        query_embedding = utils.aggregate_embeddings(author_vecs, distance_metric).reshape(1, -1)

    [retrieved_ids], [distances] = utils.query_index(index, query_embedding, k + len(excluded))

    for author_id, dist in zip(retrieved_ids, distances):
        if len(predicted_ids) >= k:
            break
        if author_id not in excluded:
            predicted_ids.append(author_id)
            predicted_scores.append(float(dist))
            excluded.add(author_id)

    if len(predicted_ids) < k:
        available_authors = list(all_author_ids - excluded)
        num_needed = k - len(predicted_ids)
        random_authors = random.sample(available_authors, num_needed)
        predicted_ids.extend(random_authors)
        predicted_scores.extend([float('inf')] * num_needed)

    return predicted_ids[:k], predicted_scores[:k]


def main():
    parser = argparse.ArgumentParser(description="CoGPT baseline for coauthor prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing embedding files")
    parser.add_argument("--embedding_type", type=str, required=True, choices=["gtr", "grit", "specter2"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained GPT model checkpoint")
    parser.add_argument("--n_layers", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--k", type=int, default=1000, help="Number of coauthors to predict")
    parser.add_argument("--strategy", type=str, default="one_shot", choices=["one_shot", "iterated"])
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=1000, help="Save predictions every N instances")
    parser.add_argument("--output_dir", type=str, default="data/task_coauthor_prediction/test/predictions")
    args = parser.parse_args()

    distance_metric = "l2" if args.embedding_type == "specter2" else "cosine"
    hidden_size = 4096 if "grit" in args.embedding_type else 768
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.log(f"Using device: {device}")

    all_papers, sd2publications, all_embeddings = utils.load_corpus(
        hf_repo_id=args.hf_repo_id,
        split=args.split,
        embeddings_dir=args.embeddings_dir,
        embedding_type=args.embedding_type,
        load_sd2publications=True
    )
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    all_author_ids = set(sd2publications.keys())
    output_path = os.path.join(args.output_dir, f"predictions.cogpt.{args.embedding_type}.{args.strategy}.json")
    utils.log(f"Loaded {len(all_papers)} papers, {len(sd2publications)} authors, {len(all_embeddings)} embeddings")

    utils.log(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.n_layers, args.n_heads, hidden_size, args.max_len, device)

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

    # Sort all papers by date and create initial index with authors who have publications before first evaluation instance
    all_papers = sorted(all_papers, key=lambda p: p["date"])
    first_date = evaluation_instances[0][0]
    utils.log(f"Creating initial author index with cutoff date: {first_date}")
    index = create_author_index(first_date, sd2publications, all_papers_dict, all_embeddings, distance_metric)

    utils.log(f"Running CoGPT baseline with strategy: {args.strategy}, embedding: {args.embedding_type}")
    predictions = []
    for paper in tqdm(all_papers, desc="Running CoGPT baseline"):
        corpus_id = paper["corpus_id"]
        date = paper["date"]

        # Update authors on this paper before prediction (if not already included in initial index).
        # This correctly handles both evaluation instances and non-evaluation instances (whether they're target or target.author.publication_history).
        # 1) evaluation instances: authors are updated with pubs < date, then we predict.
        # 2) non-evaluation instances: authors are still updated so their embeddings stay current for future predictions.
        if date >= first_date and "authors" in paper:
            for author in paper["authors"]:
                author_id = author["author_id"]
                author_pubs = get_preexisting_publications_for_author(author_id, date, sd2publications, all_papers_dict)
                index = update_author_in_index(index, author_id, author_pubs, all_embeddings, distance_metric)

        # If this is a selected evaluation instance, run prediction
        if corpus_id in eval_instance_dict:
            idx, instance = eval_instance_dict[corpus_id]
            if idx in selected_indices:
                first_author_id = instance["first_author_id"]
                gt_coauthor_ids = instance["gt_coauthor_ids"]
                num_to_predict = len(gt_coauthor_ids)

                prefix_segments = build_prefix_embeddings(first_author_id, date, sd2publications, all_papers_dict, all_embeddings, distance_metric)

                if args.strategy == "one_shot":
                    pred_ids, pred_scores = predict_coauthors_one_shot(first_author_id, args.k, index, model, prefix_segments, distance_metric, args.max_len, device, all_author_ids)
                else:
                    pred_ids, pred_scores = predict_coauthors_iterated(first_author_id, num_to_predict, args.k, index, model, prefix_segments, distance_metric, args.max_len, device, all_author_ids)

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


if __name__ == "__main__":
    main()
