import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Core scaled-dot-product attention.

    Accepts `mask` in any of these shapes:
      (B, L)         – legacy *padding* mask  (CoBERT)
      (B, 1, L)      – same, extra dim
      (B, 1, 1, L)   – pad mask ready for broadcast
      (B, H, L, L)   – causal ⊓ pad mask (GPT4Rec)
    where *False/0* blocks attention.
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        # q, k, v : (B, H, L, d_k)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (B,H,L,L)

        # ───── normalise mask to 4-D -------------------------------------------------
        if mask is not None:
            mask = mask.to(torch.bool)
            if mask.dim() == 2:                      # (B, L)
                mask = mask[:, None, None, :]        # → (B,1,1,L)
            elif mask.dim() == 3:                    # (B, 1, L)
                mask = mask[:, :, None, :]           # → (B,1,1,L)
            # else: assume (B,1,1,L) or (B,H,L,L) already
            scores = scores.masked_fill(~mask, -1e4)  # block = large negative

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
