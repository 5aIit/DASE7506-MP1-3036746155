"""Student GPT with dropout on the attention and MLP residual branches."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualDropoutBlock(nn.Module):
    def __init__(self, width=128, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.attention_residual_dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )
        self.mlp_residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attention_update = self.proj(
            attended.transpose(1, 2).reshape(batch, length, width)
        )
        x = x + self.attention_residual_dropout(attention_update)
        mlp_update = self.mlp(self.norm2(x))
        return x + self.mlp_residual_dropout(mlp_update)


class DropoutGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config["context"]
        width = config["width"]
        dropout = float(config.get("dropout", 0.1))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.token = nn.Embedding(config["vocab"], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList(
            [ResidualDropoutBlock(width, config["heads"], dropout)
             for _ in range(config["depth"])]
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config["vocab"], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return DropoutGPT(config)
