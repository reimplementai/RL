# usage:
#
# model = SmallGPT(
#    vocab_size=20,
#    n_layer=2, n_head=4, n_embd=64,
#    block_size=40,
#).to(device)
#
# plenty of help from LLMs to write LLMs :)
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryHelper:
    @staticmethod
    def precompute_cos_sin(seq_len, dim, device, dtype=torch.float32, base=10000.0):
        """
        Precompute interleaved cos/sin for rotary embeddings.

        Returns:
            cos: (seq_len, dim)
            sin: (seq_len, dim)
        """
        assert dim % 2 == 0, "rotary dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        angles = positions[:, None] * inv_freq[None, :]          # (seq_len, dim/2)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        # interleave: [cos0, cos0, cos1, cos1, ...] -> final shape (seq_len, dim)
        sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, dim)
        cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, dim)
        return cos, sin

    @staticmethod
    def rotate_half(x):
        """
        Rotate pairs (a,b) -> (-b,a) across the last dimension.
        x: (..., dim)  where dim is even
        returns same shape as x
        """
        # split into even and odd dims
        x_even = x[..., ::2]  # (..., dim/2)
        x_odd  = x[..., 1::2] # (..., dim/2)
        # rotated pairs: (-odd, even) interleaved
        rotated = torch.stack((-x_odd, x_even), dim=-1)  # (..., dim/2, 2)
        return rotated.reshape(*x.shape)                 # (..., dim)

    @staticmethod
    def apply_rotary(q, k, cos, sin):
        """
        q, k: (B, n_head, T, head_dim)
        cos, sin: (T, rotary_dim)
        Applies rotary to the first rotary_dim of q and k. If rotary_dim < head_dim,
        the remainder dims are left unchanged.
        """
        # Ensure cos/sin shape -> (1,1,T,rotary_dim) for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,rotary_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

        rotary_dim = cos.shape[-1]
        if rotary_dim == q.shape[-1]:
            q_rot = q
            k_rot = k
            q_rot = (q_rot * cos) + (RotaryHelper.rotate_half(q_rot) * sin)
            k_rot = (k_rot * cos) + (RotaryHelper.rotate_half(k_rot) * sin)
            return q_rot, k_rot
        else:
            # split into rotary and passthrough parts
            q1, q2 = q[..., :rotary_dim], q[..., rotary_dim:]
            k1, k2 = k[..., :rotary_dim], k[..., rotary_dim:]
            q1 = (q1 * cos) + (RotaryHelper.rotate_half(q1) * sin)
            k1 = (k1 * cos) + (RotaryHelper.rotate_half(k1) * sin)
            q = torch.cat([q1, q2], dim=-1)
            k = torch.cat([k1, k2], dim=-1)
            return q, k


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0, rotary_frac=1.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5

        # fused qkv projection for speed
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.ln1 = nn.LayerNorm(n_embd)  # pre-LN
        self.ln2 = nn.LayerNorm(n_embd)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

        self.dropout = nn.Dropout(dropout)

        # rotary config: fraction of head_dim used for rotary (commonly full head)
        rotary_dim = int(self.head_dim * rotary_frac)
        # make even
        if rotary_dim % 2 != 0:
            rotary_dim -= 1
        self.rotary_dim = rotary_dim
        self.block_size = block_size

        # Precompute cos/sin buffers for maximum block size (store on CPU; moved to device at runtime)
        if self.rotary_dim > 0:
            cos, sin = RotaryHelper.precompute_cos_sin(block_size, self.rotary_dim, device='cpu')
            self.register_buffer('cos_store', cos, persistent=False)
            self.register_buffer('sin_store', sin, persistent=False)
        else:
            self.register_buffer('cos_store', torch.empty(0), persistent=False)
            self.register_buffer('sin_store', torch.empty(0), persistent=False)

    def forward(self, x):
        # x: (B, T, n_embd)
        B, T, _ = x.shape
        device = x.device
        # ---- Attention ----
        residual = x
        x = self.ln1(x)  # pre-LN

        # fused qkv: (B, T, 3 * n_embd)
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)  # (B, T, 3, n_head, head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # (B, n_head, T, head_dim)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        # apply rotary to first rotary_dim of last axis
        if self.rotary_dim > 0:
            # cos/sin stored as (block_size, rotary_dim)
            cos = self.cos_store[:T, :].to(device=device, dtype=q.dtype)  # (T, rotary_dim)
            sin = self.sin_store[:T, :].to(device=device, dtype=q.dtype)
            # apply rotary via helper (it will handle partial-head rotary)
            q, k = RotaryHelper.apply_rotary(q, k, cos, sin)

        # Use PyTorch's fused scaled_dot_product_attention for speed (and causality)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        # attn_out: (B, n_head, T, head_dim) -> merge heads
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, n_embd)
        attn_out = self.o_proj(attn_out)
        x = residual + self.dropout(attn_out)

        # ---- MLP ----
        residual = x
        x = self.ln2(x)
        x = residual + self.dropout(self.mlp(x))
        return x


class SmallGPT(nn.Module):
    def __init__(self, vocab_size, n_layer=2, n_head=4, n_embd=256, block_size=30,
                 rotary_frac=1.0, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Parameter(torch.zeros(1, block_size, n_embd))  # optional: still kept

        # stack of transformer blocks (each handles rotary by itself)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout=dropout, rotary_frac=rotary_frac)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # weight tying if desired:
        # self.head.weight = self.token_embedding.weight

    def forward(self, x):
        """
        x: (B, T) long tokens
        returns logits: (B, T, vocab_size)
        """
        B, T = x.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

        tok_emb = self.token_embedding(x)             # (B, T, n_embd)
        pos_emb = self.pos_embedding[:, :T, :].to(tok_emb.dtype)  # small learnable pos embeddings (optional)
        h = tok_emb + pos_emb                         # (B, T, n_embd)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)                              # (B, T, n_embd)
        logits = self.head(h)                         # (B, T, vocab_size)
        return logits
