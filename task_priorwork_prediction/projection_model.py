"""Projection model: inference wrapper and training components (MLP projection, attention pooling, NCE loss)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPProjection(nn.Module):
    """MLP projection head for embedding transformation."""

    def __init__(self, in_dim=768, hidden_dim=512, out_dim=256, dropout=0.1, num_layers=2):
        super().__init__()
        layers = []
        current_dim = in_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.projection = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.projection(x)


class LinearAttentionPooling(nn.Module):
    """Learned query + dot-product attention pooling over a sequence of embeddings."""

    def __init__(self, embed_dim=768, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.query, std=0.02)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)

    def forward(self, embeddings, mask=None):
        """Pool embeddings [B, S, D] -> [B, D] using learned attention. mask: [B, S] True=valid."""
        keys = self.key_proj(embeddings)
        scores = torch.einsum('bsd,d->bs', keys, self.query)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        pooled = torch.einsum('bs,bsd->bd', attn_weights, embeddings)
        return self.norm(pooled)


class SPECTER2Encoder(nn.Module):
    """Encoder with shared projection MLP and configurable pooling. Uses pre-computed embeddings only."""

    def __init__(self, embed_dim=768, hidden_dim=512, out_dim=256, projection_layers=2,
                 projection_dropout=0.1, pooling_type="linear", attention_dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.projection = MLPProjection(embed_dim, hidden_dim, out_dim, projection_dropout, projection_layers)
        if pooling_type == "linear":
            self.pooling = LinearAttentionPooling(embed_dim, attention_dropout)
        else:
            self.pooling = None  # mean pooling

    def encode_paper(self, paper_embedding):
        """Project paper embeddings. [B, D] -> [B, out_dim]"""
        return self.projection(paper_embedding)

    def pool_embeddings(self, embeddings, mask=None):
        """Pool a sequence of embeddings. [B, S, D] -> [B, D]"""
        if self.pooling is not None:
            return self.pooling(embeddings, mask)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            return (embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return embeddings.mean(dim=1)

    def encode_author(self, paper_embeddings, mask=None):
        """Encode author from paper history: pool then project. [B, S, D] -> [B, out_dim]"""
        pooled = self.pool_embeddings(paper_embeddings, mask)
        return self.projection(pooled)


def multi_positive_info_nce_loss(query_emb, pos_emb, neg_emb, pos_mask=None, neg_mask=None,
                                 temperature=0.07, normalize=True):
    """Multi-positive InfoNCE loss: -log(sum_p exp(s(q,p)) / (sum_p exp(s(q,p)) + sum_n exp(s(q,n))))

    Args:
        query_emb: [B, D], pos_emb: [B, P, D], neg_emb: [B, K, D]
        pos_mask: [B, P] True=valid, neg_mask: [B, K] True=valid
    """
    if normalize:
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        pos_emb = F.normalize(pos_emb, p=2, dim=-1)
        neg_emb = F.normalize(neg_emb, p=2, dim=-1)

    pos_scores = torch.sum(query_emb.unsqueeze(1) * pos_emb, dim=-1) / temperature  # [B, P]
    neg_scores = torch.sum(query_emb.unsqueeze(1) * neg_emb, dim=-1) / temperature  # [B, K]

    if pos_mask is not None:
        pos_scores = pos_scores.masked_fill(~pos_mask, float('-inf'))
    if neg_mask is not None:
        neg_scores = neg_scores.masked_fill(~neg_mask, float('-inf'))

    pos_exp_sum = torch.exp(pos_scores).sum(dim=-1)
    neg_exp_sum = torch.exp(neg_scores).sum(dim=-1)
    eps = 1e-8
    loss = -torch.log(pos_exp_sum / (pos_exp_sum + neg_exp_sum + eps) + eps)
    return loss.mean()


# ---------------------------------------------------------------------------
# Inference wrapper (uses pre-trained checkpoint, no gradient computation)
# ---------------------------------------------------------------------------

class ProjectionModel:
    """Minimal wrapper for learned projection layers.

    Uses frozen SPECTER embeddings from disk + trained projection MLP.
    """

    def __init__(self, checkpoint_path, all_embeddings, device="cuda"):
        self.device = device
        self.all_embeddings = all_embeddings

        # Load checkpoint and extract hyperparameters
        checkpoint = torch.load(checkpoint_path, map_location=device)
        hparams = checkpoint.get("hyper_parameters", {})

        self.embed_dim = hparams.get("embed_dim", 768)
        self.hidden_dim = hparams.get("hidden_dim", 512)
        self.out_dim = hparams.get("out_dim", 256)
        num_layers = hparams.get("projection_layers", 2)
        dropout = hparams.get("projection_dropout", 0.1)

        # Build projection MLP with same architecture as training
        self.projection = MLPProjection(
            in_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            dropout=dropout,
            num_layers=num_layers
        ).to(device)

        # Load projection weights from checkpoint state_dict
        state_dict = checkpoint["state_dict"]
        proj_state = {}
        for k, v in state_dict.items():
            if "encoder.projection" in k:
                new_key = k.replace("encoder.projection.", "")
                proj_state[new_key] = v

        self.projection.load_state_dict(proj_state)
        self.projection.eval()

    def get_paper_embedding(self, corpus_id):
        """Get raw paper embedding from cache."""
        emb = self.all_embeddings[corpus_id]
        vec = emb["key"] if isinstance(emb, dict) else emb[0]
        return vec.reshape(-1)

    def encode_author(self, paper_corpus_ids):
        """Encode author from their papers: mean pool + project."""
        if not paper_corpus_ids:
            return np.zeros(self.out_dim)

        # Get paper embeddings and mean pool
        embeddings = np.stack([self.get_paper_embedding(cid) for cid in paper_corpus_ids])
        mean_emb = embeddings.mean(axis=0)

        # Project through learned MLP
        with torch.no_grad():
            x = torch.from_numpy(mean_emb).float().unsqueeze(0).to(self.device)
            projected = self.projection(x)

        return projected.cpu().numpy().squeeze()

    def encode_paper(self, corpus_id):
        """Encode paper: get embedding + project."""
        emb = self.get_paper_embedding(corpus_id)
        with torch.no_grad():
            x = torch.from_numpy(emb).float().unsqueeze(0).to(self.device)
            projected = self.projection(x)
        return projected.cpu().numpy().squeeze()

    def encode_papers_batch(self, corpus_ids):
        """Encode multiple papers in a batch."""
        if not corpus_ids:
            return np.zeros((0, self.out_dim))

        embeddings = np.stack([self.get_paper_embedding(cid) for cid in corpus_ids])
        with torch.no_grad():
            x = torch.from_numpy(embeddings).float().to(self.device)
            projected = self.projection(x)
        return projected.cpu().numpy()
