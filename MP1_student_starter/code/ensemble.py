"""Two-model probability ensemble selected using validation data only."""

import importlib
import math

import torch
from torch import nn


class ProbabilityEnsemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['vocab'] != 2048 or config['context'] != 256:
            raise ValueError('The course protocol fixes vocab=2048 and context=256.')
        self.config = dict(config)
        self.context = config['context']
        self.small_weight = float(config['small_weight'])
        if not 0.0 < self.small_weight < 1.0:
            raise ValueError('small_weight must be strictly between 0 and 1.')

        large_module = importlib.import_module(config['large_implementation'])
        small_module = importlib.import_module(config['small_implementation'])
        self.large = large_module.build_model(config['large_config'])
        self.small = small_module.build_model(config['small_config'])

    def predict_log_probs(self, ids):
        large_logp = self.large.predict_log_probs(ids).float()
        small_logp = self.small.predict_log_probs(ids).float()
        return torch.logaddexp(
            large_logp + math.log1p(-self.small_weight),
            small_logp + math.log(self.small_weight),
        )

    def forward(self, ids):
        """Return normalized next-token log probabilities for interface parity."""
        return self.predict_log_probs(ids)


def build_model(config):
    return ProbabilityEnsemble(config)
