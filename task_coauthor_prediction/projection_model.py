"""Minimal inference wrapper for learned projection layers."""

import torch
import torch.nn as nn
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

    def forward(self, x):
        return self.projection(x)


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
