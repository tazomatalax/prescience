import torch.nn as nn
from task_coauthor_prediction.model.attention.single import Attention


class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention.

    * h          – number of heads
    * d_model    – model / hidden size
    * dropout    – dropout on attention weights
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by number of heads"

        self.h   = h
        self.d_k = d_model // h

        # Linear projections for Q, K, V
        self.qkv = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.proj_out = nn.Linear(d_model, d_model)

        self.attention = Attention()
        self.dropout   = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query/key/value : (B, L, d_model)
        mask            : see Attention() docstring
        returns         : (B, L, d_model)
        """
        B, L, _ = query.size()

        # 1) linear projections → (B, H, L, d_k)
        q, k, v = [
            lin(x).view(B, L, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.qkv, (query, key, value))
        ]

        # 2) scaled-dot-product attention
        x, _ = self.attention(q, k, v, mask=mask, dropout=self.dropout)  # (B,H,L,d_k)

        # 3) concat heads & final projection
        x = x.transpose(1, 2).contiguous().view(B, L, self.h * self.d_k)
        return self.proj_out(x)
