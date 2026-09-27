"""Interpolated continuation-count n-grams, derived only from training counts.

Lower orders use distinct-left-context counts, following Kneser and Ney (1995),
https://doi.org/10.1109/ICASSP.1995.479394. A single absolute discount is selected
on validation. All persistent tables depend only on the embedded train asset.
"""
import numpy as np


class ContinuationNgrams:
    def __init__(self, raw_table):
        self.tables = {}
        unigram = np.bincount((raw_table.tables[2][0] & 2047).astype(np.int64), minlength=2048)
        self.q1 = (unigram + 1 / 2048) / (unigram.sum() + 1)
        for order in (2, 3, 4):
            if order < 4:
                suffix = raw_table.tables[order+1][0] & ((1 << (11*order))-1)
                keys, values = np.unique(suffix, return_counts=True)
            else:
                keys, values = raw_table.tables[4][:2]
            contexts, starts = np.unique(keys >> 11, return_index=True)
            ends = np.r_[starts[1:], len(keys)]
            totals = np.add.reduceat(values.astype(np.uint32), starts)
            self.tables[order] = (keys, values, contexts, starts, ends, totals)
        self._q2 = {}
        self._runtime = {}

    def target_probs(self, x, y, discount):
        ids = x.numpy().astype(np.uint64)
        target = y.clamp_min(0).numpy().astype(np.uint64)
        p = self.q1[target]
        for order in (2, 3, 4):
            query = np.zeros_like(ids)
            begin = order-2
            if ids.shape[1] <= begin:
                continue
            for offset in range(order-1):
                query[:, begin:] |= ids[:, offset:ids.shape[1]-begin+offset] << (11*(begin-offset))
            eligible = np.broadcast_to(np.arange(ids.shape[1]) >= begin, ids.shape)
            keys, counts, contexts, starts, ends, totals = self.tables[order]
            index = np.searchsorted(contexts, query)
            safe = np.minimum(index, len(contexts)-1)
            hit = eligible & (index < len(contexts)) & (contexts[safe] == query)
            denom = np.maximum(totals[safe], 1)
            backoff = discount*(ends[safe]-starts[safe])/denom
            full = (query << 11) | target
            ki = np.searchsorted(keys, full)
            ks = np.minimum(ki, len(keys)-1)
            count = np.where(hit & (ki < len(keys)) & (keys[ks] == full), counts[ks], 0)
            p = np.where(hit, np.maximum(count-discount, 0)/denom + backoff*p, p)
        return p[y.numpy() != -100]

    def bigram_table(self, discount):
        if discount in self._q2:
            return self._q2[discount]
        q = np.broadcast_to(self.q1.astype(np.float32), (2048, 2048)).copy()
        keys, values, contexts, starts, ends, totals = self.tables[2]
        q[contexts.astype(np.intp)] *= (discount*(ends-starts)/totals)[:, None]
        for i, context in enumerate(contexts):
            lo, hi = starts[i], ends[i]
            q[int(context), (keys[lo:hi] & 2047).astype(np.intp)] += (values[lo:hi]-discount)/totals[i]
        self._q2[discount] = q
        # These arrays depend only on training counts and the fixed discount.
        # Build them once while loading/prewarming, outside timed scoring.
        runtime = {}
        for order in (3, 4):
            keys, values, _, starts, ends, totals = self.tables[order]
            lengths = ends-starts
            inv_totals = np.reciprocal(totals.astype(np.float32))
            backoff = discount*lengths.astype(np.float32)*inv_totals
            entry_probs = (values.astype(np.float32)-discount)*np.repeat(inv_totals, lengths)
            successors = (keys & 2047).astype(np.uint16)
            runtime[order] = (backoff, entry_probs, successors)
        self._runtime[discount] = runtime
        return q

    def probabilities(self, ids, discount):
        x = ids.numpy().astype(np.uint64)
        batch, length = x.shape
        q = self.bigram_table(discount)[x.reshape(-1).astype(np.intp)]
        for order in (3, 4):
            begin = order-2
            if length <= begin:
                continue
            query = np.zeros_like(x)
            for offset in range(order-1):
                query[:, begin:] |= x[:, offset:length-begin+offset] << (11*(begin-offset))
            _, _, contexts, starts, ends, _ = self.tables[order]
            backoff, entry_probs, successors = self._runtime[discount][order]
            flat = query.reshape(-1)
            pos = np.searchsorted(contexts, flat)
            safe = np.minimum(pos, len(contexts)-1)
            hit = (np.broadcast_to(np.arange(length) >= begin, x.shape).reshape(-1)
                   & (pos < len(contexts)) & (contexts[safe] == flat))
            rows = np.flatnonzero(hit)
            if not len(rows):
                continue
            ci = pos[rows]
            lo, hi = starts[ci], ends[ci]
            counts = hi-lo
            # Scale in one contiguous pass. Fancy-indexed ``q[rows] *=``
            # materializes and writes back a large [hits, vocab] copy.
            scale = np.ones(q.shape[0], dtype=q.dtype)
            scale[rows] = backoff[ci]
            q *= scale[:, None]
            row_ids = np.repeat(rows, counts)
            offsets = np.repeat(lo, counts)
            prior = np.repeat(np.cumsum(counts)-counts, counts)
            entry_ids = offsets+np.arange(len(row_ids))-prior
            token_ids = successors[entry_ids]
            q[row_ids, token_ids] += entry_probs[entry_ids]
        return q.reshape(batch, length, 2048)

