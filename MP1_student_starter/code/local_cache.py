"""Causal n-gram copy distributions rebuilt for each independent input window."""

from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn


class CausalBigramCache(nn.Module):
    """Mix a frozen predictor with seen successors of the current 2-token context.

    For prediction at position j, the table contains only pairs ending before
    j. It is rebuilt for every row/call; no state crosses independent windows.
    """

    def __init__(self, base, weight=0.5, alpha=1.0):
        super().__init__()
        self.base = base
        self.context = base.context
        self.weight = float(weight)
        self.alpha = float(alpha)

    def predict_log_probs(self, ids):
        base_logp = self.base.predict_log_probs(ids).float()
        batch, length, vocab = base_logp.shape
        if length > self.context:
            raise ValueError("Input exceeds the independent-window context")
        cache = np.zeros((batch, length, vocab), dtype=np.float32)
        weights = np.zeros((batch, length, 1), dtype=np.float32)
        for row, values in enumerate(ids.tolist()):
            contexts = defaultdict(Counter)
            for j in range(length):
                if j >= 2:
                    contexts[(values[j - 2], values[j - 1])][values[j]] += 1
                if j >= 1:
                    counts = contexts.get((values[j - 1], values[j]))
                    if counts:
                        total = sum(counts.values())
                        weights[row, j, 0] = self.weight * total / (total + self.alpha)
                        for token, count in counts.items():
                            cache[row, j, token] = count / total
        q = torch.from_numpy(cache).to(device=base_logp.device)
        w = torch.from_numpy(weights).to(device=base_logp.device)
        mixed = (1 - w) * base_logp.exp() + w * q
        return mixed.clamp_min(torch.finfo(mixed.dtype).tiny).log()

    def forward(self, ids):
        return self.predict_log_probs(ids)


class CausalHierarchicalCache(nn.Module):
    """Validation-selected 1/2/3-gram window-local cache composition."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.context = base.context

    def predict_probs(self, ids):
        base_p = (self.base.predict_probs(ids) if hasattr(self.base,'predict_probs')
                  else self.base.predict_log_probs(ids).float().exp())
        batch, length, vocab = base_p.shape
        if length > self.context:
            raise ValueError("Input exceeds the independent-window context")
        additive = np.zeros((batch, length, vocab), dtype=np.float32)
        model_weight = np.ones((batch, length, 1), dtype=np.float32)
        for row, values in enumerate(ids.tolist()):
            histories = [defaultdict(Counter) for _ in range(3)]
            for j in range(length):
                for order in (1, 2, 3):
                    if j >= order:
                        histories[order - 1][tuple(values[j - order:j])][values[j]] += 1
                current = [None, None, None]
                totals = [0, 0, 0]
                for order in (1, 2, 3):
                    if j >= order - 1:
                        current[order - 1] = histories[order - 1].get(
                            tuple(values[j - order + 1:j + 1]))
                        if current[order - 1]:
                            totals[order - 1] = sum(current[order - 1].values())
                n1, n2, n3 = totals
                g1 = 0.05 * n1 / (n1 + 1) if n2 == 0 else 0.0
                g2 = 0.4 * n2 / (n2 + 1)
                g3 = 0.2 * n3 / (n3 + 1)
                model_weight[row, j, 0] = (1 - g3) * (1 - g1) * (1 - g2)
                contributions = ((g1 * (1 - g3), n1, current[0]),
                                 (g2 * (1 - g1) * (1 - g3), n2, current[1]),
                                 (g3, n3, current[2]))
                for strength, total, counts in contributions:
                    if strength:
                        for token, count in counts.items():
                            additive[row, j, token] += strength * count / total
        a = torch.from_numpy(model_weight).to(device=base_p.device)
        cache = torch.from_numpy(additive).to(device=base_p.device)
        return a * base_p + cache

    def predict_log_probs(self, ids):
        mixed = self.predict_probs(ids)
        return mixed.clamp_min(torch.finfo(mixed.dtype).tiny).log()

    def forward(self, ids):
        return self.predict_log_probs(ids)
