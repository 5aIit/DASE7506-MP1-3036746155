"""Train-derived n-gram probability mixer; no evaluation-split state."""

from pathlib import Path
import struct

import numpy as np
import torch
from torch import nn


MAGIC = b"MP1NG4\x00\x01"
WIDTHS = {2: 3, 3: 5, 4: 6}


class PackedTrainNgrams:
    def __init__(self, asset_path=None, *, raw=None):
        if raw is None:
            raw = Path(asset_path).read_bytes()
        magic, self.train_tokens, n2, n3, n4 = struct.unpack_from("<8s4I", raw)
        if magic != MAGIC:
            raise ValueError("Wrong n-gram asset format")
        offset = struct.calcsize("<8s4I")
        self.unigrams = np.frombuffer(raw, dtype="<u4", count=2048, offset=offset).copy()
        offset += 2048 * 4
        self.tables = {}
        for order, count in zip((2, 3, 4), (n2, n3, n4)):
            width = WIDTHS[order]
            packed = np.frombuffer(raw, dtype=np.uint8, count=count * width,
                                   offset=offset).reshape(count, width)
            padded = np.zeros((count, 8), dtype=np.uint8)
            padded[:, :width] = packed
            keys = padded.view("<u8").reshape(-1)
            offset += count * width
            values = np.frombuffer(raw, dtype="<u2", count=count, offset=offset).copy()
            offset += count * 2
            contexts, starts = np.unique(keys >> 11, return_index=True)
            ends = np.r_[starts[1:], len(keys)]
            totals = np.add.reduceat(values.astype(np.uint32), starts)
            self.tables[order] = (keys, values, contexts, starts, ends, totals)
        if offset != len(raw):
            raise ValueError("Trailing or missing bytes in n-gram asset")
        self._q2_tables = {}

    def bigram_table(self, alpha):
        cached = self._q2_tables.get(alpha)
        if cached is not None:
            return cached
        q1 = (self.unigrams.astype(np.float32) + 1.0 / 2048) / (self.train_tokens + 1.0)
        table = np.broadcast_to(q1, (2048, 2048)).copy()
        keys, values, contexts, starts, ends, totals = self.tables[2]
        table[contexts.astype(np.intp)] *= (alpha / (totals + alpha))[:, None]
        for index, context in enumerate(contexts):
            lo, hi = starts[index], ends[index]
            tokens = (keys[lo:hi] & 2047).astype(np.intp)
            table[int(context), tokens] += values[lo:hi] / (totals[index] + alpha)
        self._q2_tables[alpha] = table
        return table

    def probabilities(self, ids, alpha=4.0):
        if ids.device.type != "cpu":
            raise ValueError("Packed n-gram probe currently supports CPU inference")
        x = ids.detach().numpy().astype(np.uint64, copy=False)
        batch, length = x.shape
        n = batch * length
        q = self.bigram_table(alpha)[x.reshape(-1).astype(np.intp)]
        for order in (3, 4):
            if order == 3:
                query = np.zeros((batch, length), dtype=np.uint64)
                query[:, 1:] = (x[:, :-1] << 11) | x[:, 1:]
                eligible = np.broadcast_to(np.arange(length) >= 1, (batch, length))
            else:
                query = np.zeros((batch, length), dtype=np.uint64)
                query[:, 2:] = (x[:, :-2] << 22) | (x[:, 1:-1] << 11) | x[:, 2:]
                eligible = np.broadcast_to(np.arange(length) >= 2, (batch, length))
            keys, values, contexts, starts, ends, totals = self.tables[order]
            flat_query = query.reshape(-1)
            positions = np.searchsorted(contexts, flat_query)
            clipped = np.minimum(positions, len(contexts) - 1)
            hit = eligible.reshape(-1) & (positions < len(contexts)) & (contexts[clipped] == flat_query)
            denom = np.full(n, alpha, dtype=np.float32)
            denom[hit] += totals[positions[hit]]
            q *= (alpha / denom)[:, None]
            rows = np.flatnonzero(hit)
            if len(rows):
                lo = starts[positions[rows]]
                hi = ends[positions[rows]]
                lengths = hi - lo
                row_ids = np.repeat(rows, lengths)
                offsets = np.repeat(lo, lengths)
                prior = np.repeat(np.cumsum(lengths) - lengths, lengths)
                entry_ids = offsets + np.arange(len(row_ids)) - prior
                token_ids = (keys[entry_ids] & 2047).astype(np.intp)
                q[row_ids, token_ids] += values[entry_ids] / denom[row_ids]
        return q.reshape(batch, length, 2048)


class TrainNgramMixer(nn.Module):
    def __init__(self, base, asset_path, weight=0.1, alpha=4.0):
        super().__init__()
        self.base = base
        self.context = base.context
        self.ngrams = PackedTrainNgrams(asset_path)
        self.weight = float(weight)
        self.alpha = float(alpha)

    def predict_log_probs(self, ids):
        if hasattr(self.base, "predict_probs"):
            base_p = self.base.predict_probs(ids).float()
        else:
            base_p = self.base.predict_log_probs(ids).float().exp()
        q = torch.from_numpy(self.ngrams.probabilities(ids, self.alpha))
        mixed = (1.0 - self.weight) * base_p + self.weight * q
        return mixed.clamp_min(torch.finfo(mixed.dtype).tiny).log()

    def forward(self, ids):
        return self.predict_log_probs(ids)
