"""Classroom GPT with checkpoint-compatible packed CPU FP32 linear inference."""

from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.mkldnn import MkldnnLinear

import model


def _linear_forward(module, x):
    if (x.device.type != 'cpu' or x.dtype != torch.float32 or
            module.training or torch.is_grad_enabled() or not torch.backends.mkldnn.enabled or
            (module.in_features <= 128 and module.out_features <= 512)):
        return F.linear(x, module.weight, module.bias)
    weight = module.weight
    key = (weight.data_ptr(), weight._version)
    cache = module.__dict__.get('_mkldnn_cache')
    if cache is None or cache[0] != key:
        cache = (key, MkldnnLinear(module, torch.float32))
        object.__setattr__(module, '_mkldnn_cache', cache)
    return cache[1](x)


def _block_forward(block, x):
    if block.training or torch.is_grad_enabled() or x.dtype != torch.float32:
        return model.Block.forward(block, x)
    batch, length, width = x.shape
    q, k, v = block.qkv(block.norm1(x)).view(
        batch, length, 3, block.heads, width // block.heads
    ).permute(2, 0, 3, 1, 4)
    attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    x.add_(block.proj(attention.transpose(1, 2).reshape(batch, length, width)))
    return x.add_(block.mlp(block.norm2(x)))


def build_model(config):
    result = model.build_model(config)
    for module in result.modules():
        if isinstance(module, nn.Linear):
            module.forward = MethodType(_linear_forward, module)
        elif isinstance(module, model.Block):
            module.forward = MethodType(_block_forward, module)
    return result
