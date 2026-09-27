"""Temperature-calibrated two-model ensemble with a causal window-local cache."""

import importlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from local_cache import CausalBigramCache


class CalibratedMembers(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config["context"]
        self.small_weight = float(config["small_weight"])
        self.large_temperature = float(config["large_temperature"])
        self.small_temperature = float(config["small_temperature"])
        self.bias_scale = float(config.get("bias_scale", 0.0))
        bias_values = config.get("log_frequency_bias")
        if bias_values is None:
            bias = torch.zeros(config["vocab"], dtype=torch.float32)
        else:
            bias = torch.tensor(bias_values, dtype=torch.float32)
        if bias.numel() != config["vocab"] or not torch.isfinite(bias).all():
            raise ValueError("The train-derived output bias must have one finite value per token.")
        self.register_buffer("output_bias", bias, persistent=False)
        if not 0 < self.small_weight < 1:
            raise ValueError("The small-model weight must be in (0, 1).")
        if self.large_temperature <= 0 or self.small_temperature <= 0:
            raise ValueError("Temperatures must be positive.")
        large_module = importlib.import_module(config["large_implementation"])
        small_module = importlib.import_module(config["small_implementation"])
        self.large = large_module.build_model(config["large_config"])
        self.small = small_module.build_model(config["small_config"])

    def calibrated_logits(self, ids):
        large_logits = self.large(ids).float() / self.large_temperature
        small_logits = self.small(ids).float() / self.small_temperature
        if self.bias_scale != 0.0:
            bias = self.bias_scale * self.output_bias
            large_logits = large_logits + bias
            small_logits = small_logits + bias
        return large_logits, small_logits

    def predict_probs(self, ids):
        large_logits, small_logits = self.calibrated_logits(ids)
        return ((1-self.small_weight)*F.softmax(large_logits,dim=-1)
                + self.small_weight*F.softmax(small_logits,dim=-1))

    def predict_log_probs(self, ids):
        large_logits, small_logits = self.calibrated_logits(ids)
        large = F.log_softmax(large_logits, dim=-1)
        small = F.log_softmax(small_logits, dim=-1)
        return torch.logaddexp(
            large + math.log1p(-self.small_weight),
            small + math.log(self.small_weight),
        )

    def forward(self, ids):
        return self.predict_log_probs(ids)


class CalibratedCachePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config["vocab"] != 2048 or config["context"] != 256:
            raise ValueError("The course protocol fixes vocab=2048 and context=256.")
        self.config = dict(config)
        self.context = config["context"]
        self.predictor = CausalBigramCache(
            CalibratedMembers(config),
            weight=float(config["cache_weight"]),
            alpha=float(config["cache_alpha"]),
        )

    def predict_log_probs(self, ids):
        return self.predictor.predict_log_probs(ids)

    def forward(self, ids):
        return self.predict_log_probs(ids)


def build_model(config):
    return CalibratedCachePredictor(config)
