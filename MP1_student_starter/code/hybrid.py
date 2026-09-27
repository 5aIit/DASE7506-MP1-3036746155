"""Self-contained CPU FP32 predictor with training counts and causal local copy."""
import hashlib

import torch
from torch import nn

from calibrated_members import CalibratedMembers
from local_cache import CausalHierarchicalCache
from train_ngrams import PackedTrainNgrams
from continuation_ngrams import ContinuationNgrams


class HybridPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['context'] != 256 or config['vocab'] != 2048:
            raise ValueError('The fixed protocol uses context=256 and vocab=2048.')
        self.config = dict(config)
        self.context = 256
        self.local = CausalHierarchicalCache(CalibratedMembers(config))
        self.register_buffer('ngram_data', torch.zeros(config['ngram_bytes'], dtype=torch.uint8))
        self.ngrams = None
        self.ngram_weight = float(config['ngram_weight'])
        self.ngram_alpha = float(config['ngram_alpha'])
        self.ngram_method = config.get('ngram_method', 'dirichlet')
        self.ngram_discount = float(config.get('ngram_discount', .5))
        if not 0 <= self.ngram_weight < 1 or self.ngram_alpha <= 0:
            raise ValueError('Invalid n-gram interpolation settings.')
        if self.ngram_method not in ('dirichlet', 'continuation') or not 0 < self.ngram_discount < 1:
            raise ValueError('Invalid continuation-count smoothing settings.')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        raw = self.ngram_data.cpu().numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != self.config['ngram_sha256']:
            raise ValueError('The embedded training-count asset has changed.')
        self.ngrams = PackedTrainNgrams(raw=raw)
        # This table depends only on training counts; it contains no input state.
        if self.ngram_method == 'continuation':
            self.ngrams = ContinuationNgrams(self.ngrams)
            self.ngrams.bigram_table(self.ngram_discount)
        else:
            self.ngrams.bigram_table(self.ngram_alpha)
        return result

    def predict_log_probs(self, ids):
        if self.ngrams is None:
            raise RuntimeError('Load the complete checkpoint before evaluation.')
        if ids.device.type != 'cpu':
            raise ValueError('This frozen predictor supports the required CPU FP32 evaluation.')
        neural_and_local = self.local.predict_probs(ids)
        smoothing = self.ngram_discount if self.ngram_method == 'continuation' else self.ngram_alpha
        counts = torch.from_numpy(self.ngrams.probabilities(ids, smoothing))
        mixed = (1-self.ngram_weight)*neural_and_local + self.ngram_weight*counts
        return mixed.clamp_min(torch.finfo(mixed.dtype).tiny).log()

    def forward(self, ids):
        return self.predict_log_probs(ids)


def build_model(config):
    return HybridPredictor(config)
