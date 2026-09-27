"""Checkpoint-compatible ModernGPT with cached rotary positions for inference.

The cache is a non-persistent buffer, so checkpoints from ``modern_gpt`` load
with ``strict=True`` and no additional inference asset is needed.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.mkldnn import MkldnnLinear


def linear(module, x):
    """Reuse a CPU FP32 packed weight without adding checkpoint parameters."""
    if (x.device.type != 'cpu' or x.dtype != torch.float32 or
            module.training or torch.is_grad_enabled() or not torch.backends.mkldnn.enabled):
        return F.linear(x, module.weight, module.bias)
    weight = module.weight
    key = (weight.data_ptr(), weight._version)
    cache = module.__dict__.get('_mkldnn_cache')
    if cache is None or cache[0] != key:
        cache = (key, MkldnnLinear(module, torch.float32))
        object.__setattr__(module, '_mkldnn_cache', cache)
    return cache[1](x)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        if x.dtype != torch.float32:
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            return (normalized * self.weight.float()).to(x.dtype)
        return F.rms_norm(x, (x.shape[-1],), self.weight, eps=1e-6)


def rotary(q, k, cos, sin):
    """Apply the original split-half rotation using precomputed FP32 tables."""
    half_dim = q.shape[-1] // 2

    def rotate(x):
        left, right = x[..., :half_dim].float(), x[..., half_dim:].float()
        if torch.is_grad_enabled() or x.dtype != torch.float32:
            return torch.cat((left * cos - right * sin,
                              left * sin + right * cos), dim=-1).to(x.dtype)

        # The scorer runs under no_grad in FP32. Writing both halves into one
        # contiguous output avoids four temporary products and a concatenation.
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        torch.mul(left, cos, out=output[..., :half_dim])
        output[..., :half_dim].addcmul_(right, sin, value=-1)
        torch.mul(left, sin, out=output[..., half_dim:])
        output[..., half_dim:].addcmul_(right, cos)
        return output

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

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = linear(self.qkv, self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        q, k = rotary(q, k, cos, sin)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attention_update = linear(self.proj, attention.transpose(1, 2).reshape(batch, length, width))
        inference = not self.training and not torch.is_grad_enabled() and x.dtype == torch.float32
        if inference:
            x.add_(attention_update)
            gate, value = linear(self.gate_up, self.norm2(x)).chunk(2, dim=-1)
            return x.add_(linear(self.down, F.silu(gate, inplace=True).mul_(value)))
        x = x + self.dropout(attention_update)
        gate, value = linear(self.gate_up, self.norm2(x)).chunk(2, dim=-1)
        return x + self.dropout(linear(self.down, F.silu(gate) * value))


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

        half_dim = width // heads // 2
        frequencies = torch.arange(half_dim, dtype=torch.float32)
        frequencies = 10000.0 ** (-frequencies / half_dim)
        angles = torch.arange(self.context, dtype=torch.float32)[:, None] * frequencies
        self.register_buffer('_rope_cos', angles.cos()[None, None], persistent=False)
        self.register_buffer('_rope_sin', angles.sin()[None, None], persistent=False)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        length = ids.shape[1]
        if length > self.context:
            raise ValueError(f'Input length {length} exceeds context {self.context}.')
        if self._rope_cos.dtype == torch.float32:
            cos = self._rope_cos[:, :, :length]
            sin = self._rope_sin[:, :, :length]
        else:
            # Moving a model to BF16 also casts buffers. Recreate FP32 tables
            # in that case to retain the source model's rotary arithmetic.
            half_dim = self.config['width'] // self.config['heads'] // 2
            frequencies = torch.arange(half_dim, device=ids.device, dtype=torch.float32)
            frequencies = 10000.0 ** (-frequencies / half_dim)
            angles = torch.arange(length, device=ids.device, dtype=torch.float32)[:, None] * frequencies
            cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        x = self.token(ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward(self, ids):
        return linear(self.head, self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return ModernGPT(config)
