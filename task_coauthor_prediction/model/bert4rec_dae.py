import torch
from torch import nn as nn

from task_coauthor_prediction.model.attention.transformer import TransformerBlock
from task_coauthor_prediction.model.utils import fix_random_seed_as


class Bert4RecDAE(nn.Module):
    """Dynamic-ACE CoBERT (no PCE) with linear adaptor + dot-product head."""
    def __init__(self,
                 max_len: int = 200,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 hidden_size: int = 256, # has to match the ACE vector size
                 p_dropout: float = 0.1,
                 seed: int = 123):
        super().__init__()

        fix_random_seed_as(seed)
        # self.init_weights()

        self.max_len = max_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.p_dropout = p_dropout

        self.pos_embedding = nn.Embedding(self.max_len, self.hidden_size)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        self.pad_token  = nn.Parameter(torch.zeros(hidden_size))
        self.mask_token = nn.Parameter(torch.randn(hidden_size))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(self.p_dropout)        # keeps API parity

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_size=self.hidden_size,
                              n_heads=self.n_heads,
                              intermediate_size=self.hidden_size * 4,
                              p_dropout=self.p_dropout)
             for _ in range(n_layers)]
        )

        self.out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, batch):
        """
        batch expects keys:
          • ace_vecs      : [B,L,hidden]  (requires_grad = False)
          • position_ids  : [B,L]
          • attention_mask: [B,L]
          • mask_token_mask: [B,L] (optional, Boolean mask for special tokens)

        Returns:
            • out: [B,L,hidden] (linear adaptor output)
        """
        # ───── input encoding  (ACE + PPE, no PCE) ─────
        x = batch["ace_vecs"].clone()               # (B,L,H)

        # insert special tokens
        pad_pos  = batch["attention_mask"] == 0
        mask_pos = batch.get("mask_token_mask", None)
        x[pad_pos] = self.pad_token.to(x.dtype)
        if mask_pos is not None:
            x[mask_pos] = self.mask_token.to(x.dtype)
        
        #print(x.std(), self.pos_embedding.weight.std())
        # positional encoding + dropout
        x = x + self.pos_embedding(batch["position_ids"])
        x = self.dropout(x)
 
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, batch["attention_mask"])

        # ───── output projection (linear adaptor) ───── 
        return self.out(x)

