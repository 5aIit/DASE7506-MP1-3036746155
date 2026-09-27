"""Scratch-trained causal GPT with rotary positions and gated feed-forward blocks."""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return (normalized * self.weight.float()).to(x.dtype)


def rotary(q, k):
    length, half_dim = q.shape[-2], q.shape[-1] // 2
    frequencies = torch.arange(half_dim, device=q.device, dtype=torch.float32)
    frequencies = 10000.0 ** (-frequencies / half_dim)
    angles = torch.arange(length, device=q.device, dtype=torch.float32)[:, None] * frequencies
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]

    def rotate(x):
        left, right = x[..., :half_dim].float(), x[..., half_dim:].float()
        return torch.cat((left * cos - right * sin, left * sin + right * cos), -1).to(x.dtype)

    return rotate(q), rotate(k)


class Block(nn.Module):
    def __init__(self, width, heads, dropout, hidden):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rotary(q, k)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.dropout(self.proj(attention.transpose(1, 2).reshape(batch, length, width)))
        gate, value = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.dropout(self.down(F.silu(gate) * value))


class ModernGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        heads = config['heads']
        if width % heads or (width // heads) % 2:
            raise ValueError('Each attention head must have an even dimension.')
        hidden = config.get('hidden', ((8 * width // 3 + 63) // 64) * 64)
        dropout = config.get('dropout', 0.1)
        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([Block(width, heads, dropout, hidden) for _ in range(config['depth'])])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return ModernGPT(config)
