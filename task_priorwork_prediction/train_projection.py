"""Train projection model for prior work prediction with multi-positive InfoNCE loss."""

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
from task_priorwork_prediction.projection_model import SPECTER2Encoder, multi_positive_info_nce_loss


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


def get_author_citations_before(author_id, cutoff_date, sd2publications, all_papers_dict):
    """Get all papers cited by an author's papers published before cutoff_date."""
    pubs = get_pubs_before(author_id, cutoff_date, sd2publications, all_papers_dict)
    cited = set()
    for cid in pubs:
        for ref in all_papers_dict[cid]["key_references"]:
            ref_cid = ref["corpus_id"]
            if ref_cid in all_papers_dict and all_papers_dict[ref_cid]["date"] < cutoff_date:
                cited.add(ref_cid)
    return cited


def generate_training_samples(target_papers, sd2publications, all_papers_dict, all_embeddings,
                              paper_to_citations, paper_dates,
                              num_hard_negatives, num_easy_negatives, max_paper_appearances, rng):
    """Generate prior work prediction training samples matching aman's CitationDataGenerator."""
    samples = []
    paper_counts = {}

    for paper in tqdm(target_papers, desc="Generating training samples"):
        cutoff_date = paper["date"]
        author_ids = [a["author_id"] for a in paper["authors"]]

        if not author_ids:
            continue

        # Positive papers: key references that exist and were published before cutoff
        positive_ids = [ref["corpus_id"] for ref in paper["key_references"]
                        if ref["corpus_id"] in all_papers_dict and all_papers_dict[ref["corpus_id"]]["date"] < cutoff_date]
        if len(positive_ids) == 0:
            continue

        # All papers cited by authors before cutoff (using ALL publications, not just recent)
        direct_citations = set()
        for aid in author_ids:
            direct_citations.update(get_author_citations_before(aid, cutoff_date, sd2publications, all_papers_dict))

        # 1-hop citations: papers cited BY papers that authors cited
        one_hop = set()
        for cited_id in direct_citations:
            if cited_id in paper_to_citations:
                for hop_id in paper_to_citations[cited_id]:
                    if hop_id in all_papers_dict and all_papers_dict[hop_id]["date"] < cutoff_date:
                        one_hop.add(hop_id)

        # Hard negatives: 1-hop but not directly cited and not positive
        positive_set = set(positive_ids)
        hard_pool = list(one_hop - direct_citations - positive_set)
        if max_paper_appearances is not None:
            hard_pool = [p for p in hard_pool if paper_counts.get(p, 0) < max_paper_appearances]
        num_hard = min(num_hard_negatives, len(hard_pool))
        hard_negatives = rng.sample(hard_pool, num_hard) if num_hard > 0 else []

        # Easy negatives: papers published before cutoff, not cited, not 1-hop
        exclude = positive_set | direct_citations | one_hop | {paper["corpus_id"]} | set(hard_negatives)
        easy_pool = [pid for pid, d in paper_dates.items() if d < cutoff_date and pid not in exclude]
        if max_paper_appearances is not None:
            easy_pool = [p for p in easy_pool if paper_counts.get(p, 0) < max_paper_appearances]
        num_easy = min(num_easy_negatives, len(easy_pool))
        easy_negatives = rng.sample(easy_pool, num_easy) if num_easy > 0 else []

        # Update paper counts
        for pid in positive_ids + hard_negatives + easy_negatives:
            paper_counts[pid] = paper_counts.get(pid, 0) + 1

        samples.append({
            "author_ids": author_ids,
            "positive_paper_ids": positive_ids,
            "negative_paper_ids": hard_negatives + easy_negatives,
            "cutoff_date": cutoff_date,
        })

    return samples


class PriorWorkTrainDataset(Dataset):
    """Dataset that returns embedding IDs for each training sample."""

    def __init__(self, samples, sd2publications, all_papers_dict, all_embeddings, max_history):
        self.samples = samples
        self.sd2publications = sd2publications
        self.all_papers_dict = all_papers_dict
        self.all_embeddings = all_embeddings
        self.max_history = max_history

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
        author_ids_list = [self._get_author_embedding_ids(a, cutoff_date) for a in sample["author_ids"]]
        return {
            "author_ids_list": author_ids_list,
            "positive_paper_ids": sample["positive_paper_ids"],
            "negative_paper_ids": sample["negative_paper_ids"],
        }


def collate_fn(batch, all_embeddings, embed_dim, device):
    """Collate batch into padded tensors for author queries, positive papers, and negative papers."""
    B = len(batch)

    # Compute max dimensions
    max_authors = max(len(b["author_ids_list"]) for b in batch)
    max_author_papers = 1
    for b in batch:
        for ids in b["author_ids_list"]:
            max_author_papers = max(max_author_papers, len(ids))
    max_pos = max(len(b["positive_paper_ids"]) for b in batch)
    max_neg = max(len(b["negative_paper_ids"]) for b in batch)

    # Allocate tensors
    author_embs = torch.zeros(B, max_authors, max_author_papers, embed_dim)
    author_paper_mask = torch.zeros(B, max_authors, max_author_papers, dtype=torch.bool)
    author_mask = torch.zeros(B, max_authors, dtype=torch.bool)
    pos_embs = torch.zeros(B, max_pos, embed_dim)
    pos_mask = torch.zeros(B, max_pos, dtype=torch.bool)
    neg_embs = torch.zeros(B, max_neg, embed_dim)
    neg_mask = torch.zeros(B, max_neg, dtype=torch.bool)

    for i, b in enumerate(batch):
        # Author embeddings
        for a, ids in enumerate(b["author_ids_list"]):
            if len(ids) > 0:
                author_mask[i, a] = True
                for j, cid in enumerate(ids):
                    author_embs[i, a, j] = torch.from_numpy(get_embedding(all_embeddings, cid)).float()
                    author_paper_mask[i, a, j] = True
        # Positive paper embeddings
        for p, cid in enumerate(b["positive_paper_ids"]):
            pos_embs[i, p] = torch.from_numpy(get_embedding(all_embeddings, cid)).float()
            pos_mask[i, p] = True
        # Negative paper embeddings
        for n, cid in enumerate(b["negative_paper_ids"]):
            neg_embs[i, n] = torch.from_numpy(get_embedding(all_embeddings, cid)).float()
            neg_mask[i, n] = True

    return {
        "author_embs": author_embs.to(device), "author_paper_mask": author_paper_mask.to(device),
        "author_mask": author_mask.to(device),
        "pos_embs": pos_embs.to(device), "pos_mask": pos_mask.to(device),
        "neg_embs": neg_embs.to(device), "neg_mask": neg_mask.to(device),
    }


def encode_author_set(encoder, author_embs, author_paper_mask, author_mask):
    """Encode author set: pool papers per author, then mean-pool across valid authors.

    author_embs: [B, A, S, D], author_paper_mask: [B, A, S], author_mask: [B, A]
    Returns: [B, out_dim]
    """
    B, A, S, D = author_embs.shape
    # Encode each author: pool papers then project
    flat_embs = author_embs.view(B * A, S, D)
    flat_mask = author_paper_mask.view(B * A, S)
    flat_encoded = encoder.encode_author(flat_embs, flat_mask)  # [B*A, out_dim]
    encoded = flat_encoded.view(B, A, -1)  # [B, A, out_dim]

    # Mean pool across valid authors
    author_mask_expanded = author_mask.unsqueeze(-1).float()  # [B, A, 1]
    summed = (encoded * author_mask_expanded).sum(dim=1)  # [B, out_dim]
    count = author_mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
    return summed / count


def train_one_epoch(encoder_wrapper, optimizer, scheduler, dataloader, temperature, device):
    """Run one training epoch, returning average loss."""
    encoder_wrapper.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()

        # Encode query: author set -> mean pooled author embedding
        query = encode_author_set(
            encoder_wrapper.encoder, batch["author_embs"],
            batch["author_paper_mask"], batch["author_mask"])  # [B, out_dim]

        # Encode positive and negative papers
        pos_projected = encoder_wrapper.encoder.encode_paper(batch["pos_embs"])  # [B, P, out_dim]
        neg_projected = encoder_wrapper.encoder.encode_paper(batch["neg_embs"])  # [B, K, out_dim]

        loss = multi_positive_info_nce_loss(
            query, pos_projected, neg_projected,
            pos_mask=batch["pos_mask"], neg_mask=batch["neg_mask"],
            temperature=temperature)

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
        query = encode_author_set(
            encoder_wrapper.encoder, batch["author_embs"],
            batch["author_paper_mask"], batch["author_mask"])
        pos_projected = encoder_wrapper.encoder.encode_paper(batch["pos_embs"])
        neg_projected = encoder_wrapper.encoder.encode_paper(batch["neg_embs"])
        loss = multi_positive_info_nce_loss(
            query, pos_projected, neg_projected,
            pos_mask=batch["pos_mask"], neg_mask=batch["neg_mask"],
            temperature=temperature)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


class EncoderWrapper(nn.Module):
    """Wraps SPECTER2Encoder as self.encoder so state_dict keys are encoder.projection.* and encoder.pooling.*"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder


def main():
    parser = argparse.ArgumentParser(description="Train projection model for prior work prediction")
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
    parser.add_argument("--max_paper_appearances", type=int, default=10, help="Max times a paper can appear across samples (prevents overfitting)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data/task_priorwork_prediction/train/projection_models", help="Output directory for checkpoints")
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

    # Build indices used by sample generation
    paper_to_citations = {}
    for paper in all_papers:
        paper_to_citations[paper["corpus_id"]] = [ref["corpus_id"] for ref in paper["key_references"]]
    paper_dates = {p["corpus_id"]: p["date"] for p in all_papers if p["date"] is not None}

    # Split target papers into train/val before generating samples (matching aman)
    target_papers = sorted([p for p in all_papers if "target" in p["roles"]], key=lambda p: p["date"])
    n_train = int(len(target_papers) * (1 - args.val_ratio))
    train_papers = target_papers[:n_train]
    val_papers = target_papers[n_train:]

    utils.log("Generating training samples")
    rng = random.Random(args.seed)
    train_samples = generate_training_samples(
        train_papers, sd2publications, all_papers_dict, all_embeddings,
        paper_to_citations, paper_dates,
        args.num_hard_negatives, args.num_easy_negatives, args.max_paper_appearances, rng)
    utils.log(f"Train: {len(train_samples)} samples from {len(train_papers)} papers")

    val_samples = generate_training_samples(
        val_papers, sd2publications, all_papers_dict, all_embeddings,
        paper_to_citations, paper_dates,
        args.num_hard_negatives, args.num_easy_negatives, None, rng)
    utils.log(f"Val: {len(val_samples)} samples from {len(val_papers)} papers")

    train_dataset = PriorWorkTrainDataset(train_samples, sd2publications, all_papers_dict, all_embeddings, args.max_history)
    val_dataset = PriorWorkTrainDataset(val_samples, sd2publications, all_papers_dict, all_embeddings, args.max_history)

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
