# model/gpt4rec_dae.py
import torch
import torch.nn as nn
from task_coauthor_prediction.model.attention.transformer import TransformerBlock
from task_coauthor_prediction.model.utils import fix_random_seed_as

class GPT4RecDAE(nn.Module):
    """
    Decoder-only analogue of Bert4RecDAE.
    Predicts the next ACE vector for every timestep.
    """
    def __init__(self, max_len=200, n_layers=2, n_heads=4,
                 hidden_size=256, p_dropout=0.1, seed=123):
        super().__init__()
        fix_random_seed_as(seed)

        self.pos_embedding = nn.Embedding(max_len, hidden_size)
        nn.init.normal_(self.pos_embedding.weight, 0.0, 0.02)

        self.dropout = nn.Dropout(p_dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads,
                             hidden_size*4, p_dropout)
            for _ in range(n_layers)
        ])
        self.out = nn.Linear(hidden_size, hidden_size)   # adaptor

    def _causal_mask(self, L, device):
        # 1 = keep, 0 = block future
        return torch.tril(torch.ones(L, L, device=device)).bool()

    def forward(self, batch):
        x        = batch["ace_vecs"]              # (B,L,H)
        B, L, _  = x.shape
        device   = x.device

        # ── ②  add positional encodings ─────────────────────────────────
        pos_ids = torch.arange(L, device=device)
        x = x + self.pos_embedding(pos_ids)[None, :, :]
        x = self.dropout(x)

        # ── ③  build causal ⊓ pad mask (unchanged) ──────────────────────
        pad_mask  = batch["attention_mask"][:, None, None, :]
        causal    = torch.tril(torch.ones(L, L, dtype=torch.bool,
                                          device=device))[None, None]
        attn_mask = pad_mask & causal

        for blk in self.blocks:
            x = blk(x, attn_mask)

        return self.out(x)
