"""Train projection model for coauthor prediction with multi-positive InfoNCE loss."""

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from bisect import bisect_left
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import utils
from task_coauthor_prediction.projection_model import SPECTER2Encoder, multi_positive_info_nce_loss


def get_embedding(all_embeddings, corpus_id):
    """Get key embedding for a paper as a flat numpy array."""
    emb = all_embeddings[corpus_id]
    vec = emb["key"] if isinstance(emb, dict) else emb[0]
    return vec.reshape(-1)


def get_pubs_before(author_id, cutoff_date, sd2publications, all_papers_dict):
    """Get an author's publications before cutoff_date using binary search."""
    if author_id not in sd2publications:
        return []
    author_pubs = sd2publications[author_id]
    if author_pubs is None:
        return []
    pub_dates = [all_papers_dict[p]["date"] for p in author_pubs]
    idx = bisect_left(pub_dates, cutoff_date)
    return author_pubs[:idx]


def get_coauthors_before(author_id, cutoff_date, sd2publications, all_papers_dict):
    """Get all coauthors of author_id from papers published before cutoff_date."""
    pubs = get_pubs_before(author_id, cutoff_date, sd2publications, all_papers_dict)
    coauthors = set()
    for cid in pubs:
        for a in all_papers_dict[cid]["authors"]:
            if a["author_id"] != author_id:
                coauthors.add(a["author_id"])
    return coauthors


def build_author_timeline(sd2publications, all_papers_dict):
    """Build sorted (date, author_id) list for binary search over author first-publish dates."""
    author_first_date = {}
    for author_id, pubs in sd2publications.items():
        if pubs is None or len(pubs) == 0:
            continue
        dates = [all_papers_dict[p]["date"] for p in pubs if p in all_papers_dict]
        if dates:
            author_first_date[author_id] = min(dates)

    sorted_authors = sorted(author_first_date.items(), key=lambda x: x[1])
    sorted_dates = [d for _, d in sorted_authors]
    sorted_ids = [aid for aid, _ in sorted_authors]
    return sorted_dates, sorted_ids


def get_authors_before(cutoff_date, sorted_dates, sorted_ids):
    """Get all authors who published before cutoff_date using binary search."""
    idx = bisect_left(sorted_dates, cutoff_date)
    return set(sorted_ids[:idx])


def generate_training_samples(target_papers, sd2publications, all_papers_dict,
                              sorted_author_dates, sorted_author_ids,
                              num_hard_negatives, num_easy_negatives,
                              max_author_appearances, rng):
    """Generate coauthor prediction training samples matching aman's CoauthorDataGenerator."""
    samples = []
    author_counts = {}

    for paper in tqdm(target_papers, desc="Generating training samples"):
        cutoff_date = paper["date"]
        positive_authors = [a["author_id"] for a in paper["authors"]]

        # Skip if any positive author exceeds appearance limit
        if max_author_appearances is not None:
            if any(author_counts.get(a, 0) >= max_author_appearances for a in positive_authors):
                continue

        # Direct collaborators: coauthors of first-5 + last-5 positive authors (temporal)
        reqd_authors = set(positive_authors[:5] + positive_authors[-5:]) if len(positive_authors) > 10 else set(positive_authors)
        direct_collaborators = set()
        for aid in reqd_authors:
            direct_collaborators.update(get_coauthors_before(aid, cutoff_date, sd2publications, all_papers_dict))

        # 1-hop: collaborators of non-positive direct collaborators
        positive_set = set(positive_authors)
        non_positive_collabs = direct_collaborators - positive_set
        if len(non_positive_collabs) > 50:
            non_positive_collabs = set(rng.sample(list(non_positive_collabs), 50))

        one_hop = set()
        for collab_id in non_positive_collabs:
            one_hop.update(get_coauthors_before(collab_id, cutoff_date, sd2publications, all_papers_dict))

        # Hard negatives: 1-hop but not direct and not positive
        hard_pool = one_hop - direct_collaborators - positive_set
        if max_author_appearances is not None:
            hard_pool = {a for a in hard_pool if author_counts.get(a, 0) < max_author_appearances}
        num_hard = min(num_hard_negatives, len(hard_pool))
        hard_negatives = rng.sample(list(hard_pool), num_hard) if num_hard > 0 else []

        # Easy negatives: authors who existed before cutoff, not direct, not 1-hop, not positive
        valid_authors = get_authors_before(cutoff_date, sorted_author_dates, sorted_author_ids)
        easy_pool = valid_authors - direct_collaborators - one_hop - positive_set - set(hard_negatives)
        if max_author_appearances is not None:
            easy_pool = {a for a in easy_pool if author_counts.get(a, 0) < max_author_appearances}
        num_easy = min(num_easy_negatives, len(easy_pool))
        easy_negatives = rng.sample(list(easy_pool), num_easy) if num_easy > 0 else []

        # Update author counts
        for aid in positive_authors + hard_negatives + easy_negatives:
            author_counts[aid] = author_counts.get(aid, 0) + 1

        samples.append({
            "positive_authors": positive_authors,
            "negative_authors": hard_negatives + easy_negatives,
            "cutoff_date": cutoff_date,
        })

    return samples


class CoauthorTrainDataset(Dataset):
    """Dataset that returns embedding IDs for each training sample. Query = first author, positives = rest."""

    def __init__(self, samples, sd2publications, all_papers_dict, all_embeddings, max_history, query_position=0):
        self.samples = samples
        self.sd2publications = sd2publications
        self.all_papers_dict = all_papers_dict
        self.all_embeddings = all_embeddings
        self.max_history = max_history
        self.query_position = query_position

    def __len__(self):
        return len(self.samples)

    def _get_author_embedding_ids(self, author_id, cutoff_date):
        """Get corpus_ids for an author's recent papers with valid embeddings."""
        pubs = get_pubs_before(author_id, cutoff_date, self.sd2publications, self.all_papers_dict)
        valid = [cid for cid in pubs if cid in self.all_embeddings]
        return valid[-self.max_history:]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cutoff_date = sample["cutoff_date"]
        all_positives = sample["positive_authors"]

        # Select query author by position (matching aman's query_position)
        query_pos = min(self.query_position, len(all_positives) - 1)
        query_author = all_positives[query_pos]
        positive_authors = [a for i, a in enumerate(all_positives) if i != query_pos]

        query_ids = self._get_author_embedding_ids(query_author, cutoff_date)
        pos_ids_list = [self._get_author_embedding_ids(a, cutoff_date) for a in positive_authors]
        neg_ids_list = [self._get_author_embedding_ids(a, cutoff_date) for a in sample["negative_authors"]]

        return {
            "query_ids": query_ids,
            "pos_ids_list": pos_ids_list,
            "neg_ids_list": neg_ids_list,
        }


def collate_fn(batch, all_embeddings, embed_dim, device):
    """Collate batch into padded tensors for query, positive, and negative authors."""
    B = len(batch)

    max_query_len = max(max(len(b["query_ids"]) for b in batch), 1)
    max_pos = max(max(len(b["pos_ids_list"]) for b in batch), 1)
    max_neg = max(max(len(b["neg_ids_list"]) for b in batch), 1)
    max_pos_len = 1
    max_neg_len = 1
    for b in batch:
        for ids in b["pos_ids_list"]:
            max_pos_len = max(max_pos_len, len(ids))
        for ids in b["neg_ids_list"]:
            max_neg_len = max(max_neg_len, len(ids))

    query_embs = torch.zeros(B, max_query_len, embed_dim)
    query_mask = torch.zeros(B, max_query_len, dtype=torch.bool)
    pos_embs = torch.zeros(B, max_pos, max_pos_len, embed_dim)
    pos_paper_mask = torch.zeros(B, max_pos, max_pos_len, dtype=torch.bool)
    pos_author_mask = torch.zeros(B, max_pos, dtype=torch.bool)
    neg_embs = torch.zeros(B, max_neg, max_neg_len, embed_dim)
    neg_paper_mask = torch.zeros(B, max_neg, max_neg_len, dtype=torch.bool)
    neg_author_mask = torch.zeros(B, max_neg, dtype=torch.bool)

    for i, b in enumerate(batch):
        for j, cid in enumerate(b["query_ids"]):
            query_embs[i, j] = torch.from_numpy(get_embedding(all_embeddings, cid)).float()
            query_mask[i, j] = True
        for p, ids in enumerate(b["pos_ids_list"]):
            if len(ids) > 0:
                pos_author_mask[i, p] = True
                for j, cid in enumerate(ids):
                    pos_embs[i, p, j] = torch.from_numpy(get_embedding(all_embeddings, cid)).float()
                    pos_paper_mask[i, p, j] = True
        for n, ids in enumerate(b["neg_ids_list"]):
            if len(ids) > 0:
                neg_author_mask[i, n] = True
                for j, cid in enumerate(ids):
                    neg_embs[i, n, j] = torch.from_numpy(get_embedding(all_embeddings, cid)).float()
                    neg_paper_mask[i, n, j] = True

    return {
        "query_embs": query_embs.to(device), "query_mask": query_mask.to(device),
        "pos_embs": pos_embs.to(device), "pos_paper_mask": pos_paper_mask.to(device),
        "pos_author_mask": pos_author_mask.to(device),
        "neg_embs": neg_embs.to(device), "neg_paper_mask": neg_paper_mask.to(device),
        "neg_author_mask": neg_author_mask.to(device),
    }


def encode_authors_batch(encoder, embs, paper_mask, author_mask, device):
    """Encode a batch of authors: pool papers per author, return [B, A, out_dim] with mask."""
    B, A, S, D = embs.shape
    flat_embs = embs.view(B * A, S, D)
    flat_paper_mask = paper_mask.view(B * A, S)
    flat_author_embs = encoder.encode_author(flat_embs, flat_paper_mask)
    return flat_author_embs.view(B, A, -1), author_mask


def train_one_epoch(encoder_wrapper, optimizer, scheduler, dataloader, temperature, device):
    """Run one training epoch, returning average loss."""
    encoder_wrapper.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()
        query_author = encoder_wrapper.encoder.encode_author(batch["query_embs"], batch["query_mask"])
        pos_author_embs, pos_mask = encode_authors_batch(
            encoder_wrapper.encoder, batch["pos_embs"], batch["pos_paper_mask"],
            batch["pos_author_mask"], device)
        neg_author_embs, neg_mask = encode_authors_batch(
            encoder_wrapper.encoder, batch["neg_embs"], batch["neg_paper_mask"],
            batch["neg_author_mask"], device)
        loss = multi_positive_info_nce_loss(
            query_author, pos_author_embs, neg_author_embs,
            pos_mask=pos_mask, neg_mask=neg_mask, temperature=temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder_wrapper.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(encoder_wrapper, dataloader, temperature, device):
    """Run validation, returning average loss."""
    encoder_wrapper.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        query_author = encoder_wrapper.encoder.encode_author(batch["query_embs"], batch["query_mask"])
        pos_author_embs, pos_mask = encode_authors_batch(
            encoder_wrapper.encoder, batch["pos_embs"], batch["pos_paper_mask"],
            batch["pos_author_mask"], device)
        neg_author_embs, neg_mask = encode_authors_batch(
            encoder_wrapper.encoder, batch["neg_embs"], batch["neg_paper_mask"],
            batch["neg_author_mask"], device)
        loss = multi_positive_info_nce_loss(
            query_author, pos_author_embs, neg_author_embs,
            pos_mask=pos_mask, neg_mask=neg_mask, temperature=temperature)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


class EncoderWrapper(nn.Module):
    """Wraps SPECTER2Encoder as self.encoder so state_dict keys are encoder.projection.* and encoder.pooling.*"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder


def main():
    parser = argparse.ArgumentParser(description="Train projection model for coauthor prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing embeddings pkl files")
    parser.add_argument("--embedding_type", type=str, required=True, choices=["gtr", "grit", "specter2"], help="Type of embeddings to use")
    parser.add_argument("--embed_dim", type=int, default=768, help="Input embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for projection MLP")
    parser.add_argument("--out_dim", type=int, default=256, help="Output dimension for projection MLP")
    parser.add_argument("--projection_layers", type=int, default=2, help="Number of MLP layers")
    parser.add_argument("--projection_dropout", type=float, default=0.1, help="Dropout in projection MLP")
    parser.add_argument("--pooling_type", type=str, default="linear", choices=["linear", "mean"], help="Pooling type for author histories")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--num_hard_negatives", type=int, default=5, help="Hard negatives per sample")
    parser.add_argument("--num_easy_negatives", type=int, default=10, help="Easy negatives per sample")
    parser.add_argument("--max_history", type=int, default=10, help="Maximum papers per author in training")
    parser.add_argument("--max_author_appearances", type=int, default=15, help="Max times an author can appear across samples (prevents overfitting)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data/task_coauthor_prediction/train/projection_models", help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    utils.log(f"Loading corpus from {args.hf_repo_id} (split={args.split})")
    all_papers, sd2publications, all_embeddings = utils.load_corpus(
        hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=args.embeddings_dir,
        embedding_type=args.embedding_type, load_sd2publications=True)
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers, {len(sd2publications)} authors, {len(all_embeddings)} embeddings")

    # Auto-detect embedding dimension
    sample_cid = next(iter(all_embeddings))
    embed_dim = get_embedding(all_embeddings, sample_cid).shape[0]
    if embed_dim != args.embed_dim:
        utils.log(f"Auto-detected embed_dim={embed_dim} (overriding --embed_dim={args.embed_dim})")
        args.embed_dim = embed_dim

    utils.log("Building author timeline")
    sorted_author_dates, sorted_author_ids = build_author_timeline(sd2publications, all_papers_dict)
    utils.log(f"Built timeline for {len(sorted_author_ids)} authors")

    # Split target papers into train/val before generating samples (matching aman)
    target_papers = sorted(
        [p for p in all_papers if "target" in p["roles"] and len(p["authors"]) >= 2],
        key=lambda p: p["date"])
    n_train = int(len(target_papers) * (1 - args.val_ratio))
    train_papers = target_papers[:n_train]
    val_papers = target_papers[n_train:]

    utils.log("Generating training samples")
    rng = random.Random(args.seed)
    train_samples = generate_training_samples(
        train_papers, sd2publications, all_papers_dict,
        sorted_author_dates, sorted_author_ids,
        args.num_hard_negatives, args.num_easy_negatives,
        args.max_author_appearances, rng)
    utils.log(f"Train: {len(train_samples)} samples from {len(train_papers)} papers")

    val_samples = generate_training_samples(
        val_papers, sd2publications, all_papers_dict,
        sorted_author_dates, sorted_author_ids,
        args.num_hard_negatives, args.num_easy_negatives,
        None, rng)
    utils.log(f"Val: {len(val_samples)} samples from {len(val_papers)} papers")

    train_dataset = CoauthorTrainDataset(train_samples, sd2publications, all_papers_dict, all_embeddings, args.max_history)
    val_dataset = CoauthorTrainDataset(val_samples, sd2publications, all_papers_dict, all_embeddings, args.max_history)

    def make_collate(device):
        return lambda batch: collate_fn(batch, all_embeddings, args.embed_dim, device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=make_collate(device))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate(device)) if val_samples else None

    # Build model
    encoder = SPECTER2Encoder(
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
        projection_layers=args.projection_layers, projection_dropout=args.projection_dropout,
        pooling_type=args.pooling_type)
    encoder_wrapper = EncoderWrapper(encoder).to(device)

    optimizer = torch.optim.AdamW(encoder_wrapper.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"projection.{args.embedding_type}.pt")

    utils.log(f"Training projection model: {sum(p.numel() for p in encoder_wrapper.parameters())} parameters")
    utils.log(f"Architecture: embed_dim={args.embed_dim}, hidden_dim={args.hidden_dim}, out_dim={args.out_dim}, pooling={args.pooling_type}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(args.max_epochs):
        train_loss = train_one_epoch(encoder_wrapper, optimizer, scheduler, train_loader, args.temperature, device)
        log_msg = f"Epoch {epoch+1}/{args.max_epochs} | train_loss={train_loss:.4f}"

        if val_loader is not None:
            val_loss = validate(encoder_wrapper, val_loader, args.temperature, device)
            log_msg += f" | val_loss={val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save({
                    "state_dict": encoder_wrapper.state_dict(),
                    "hyper_parameters": {
                        "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim, "out_dim": args.out_dim,
                        "projection_layers": args.projection_layers, "projection_dropout": args.projection_dropout,
                        "pooling_type": args.pooling_type, "temperature": args.temperature,
                    },
                }, checkpoint_path)
                log_msg += " | saved checkpoint"
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    utils.log(log_msg)
                    utils.log(f"Early stopping after {args.patience} epochs without improvement")
                    break
        else:
            torch.save({
                "state_dict": encoder_wrapper.state_dict(),
                "hyper_parameters": {
                    "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim, "out_dim": args.out_dim,
                    "projection_layers": args.projection_layers, "projection_dropout": args.projection_dropout,
                    "pooling_type": args.pooling_type, "temperature": args.temperature,
                },
            }, checkpoint_path)

        utils.log(log_msg)

    utils.log(f"Training complete. Best checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
