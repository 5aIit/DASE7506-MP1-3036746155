"""Select hybrid predictor settings on validation, using train-only assets.

Example (from code/):
  python select_hybrid.py --large runs/wide/checkpoint.pt \
      --small runs/small/checkpoint.pt --ngram-asset runs/train_counts.bin \
      --output runs/hybrid_selection.json --device cuda --batch-size 2

The scorer, tokenizer, and benchmark files are never changed. This script
opens only the supplied train and validation text, never the test split.
"""

import argparse
from collections import Counter, defaultdict
import importlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from common import PROTOCOL, ROOT, setup, sha, windows
from continuation_ngrams import ContinuationNgrams
from local_cache import CausalHierarchicalCache
from train_ngrams import PackedTrainNgrams


TEMPERATURES = (1.0, 1.05, 1.10, 1.15)
SMALL_WEIGHTS = (0.15, 0.20, 0.25, 0.30, 0.35)
BIAS_SCALE = 0.05


def target_ngram_probabilities(table, x, y, alpha):
    """Train n-gram probability only at each target; no validation lookup table."""
    ids = x.numpy().astype(np.uint64, copy=False)
    target = y.clamp_min(0).numpy().astype(np.uint64, copy=False)
    p = (table.unigrams[target].astype(np.float64) + 1.0 / 2048) / (table.train_tokens + 1)
    length = ids.shape[1]
    for order in (2, 3, 4):
        width = order - 1
        query = np.zeros_like(ids)
        for offset in range(width):
            query[:, width - 1:] |= (
                ids[:, offset:length - width + 1 + offset]
                << (11 * (width - 1 - offset))
            )
        eligible = np.broadcast_to(np.arange(length) >= width - 1, ids.shape)
        keys, counts, contexts, _, _, totals = table.tables[order]
        index = np.searchsorted(contexts, query)
        safe = np.minimum(index, len(contexts) - 1)
        seen_context = eligible & (index < len(contexts)) & (contexts[safe] == query)
        total = np.where(seen_context, totals[safe], 0)
        full_key = (query << 11) | target
        key_index = np.searchsorted(keys, full_key)
        safe_key = np.minimum(key_index, len(keys) - 1)
        seen_target = (seen_context & (key_index < len(keys))
                       & (keys[safe_key] == full_key))
        count = np.where(seen_target, counts[safe_key], 0)
        p = (count + alpha * p) / (total + alpha)
    return p[y.numpy() != -100]


def selected_target_probabilities(ngram, x, y, smoothing):
    if isinstance(ngram, ContinuationNgrams):
        return ngram.target_probs(x, y, smoothing)
    return target_ngram_probabilities(ngram, x, y, smoothing)


def target_local_stats(x, y):
    """Within-window suffix successors known before each prediction position."""
    if len(x) != len(y):
        raise ValueError("Input and target lengths differ")
    q = np.zeros((4, len(y)), dtype=np.float64)
    support = np.zeros((4, len(y)), dtype=np.int16)
    histories = [None] + [defaultdict(Counter) for _ in range(3)]
    for j, target in enumerate(y):
        observed = x[j]
        for order in (1, 2, 3):
            if j >= order:
                histories[order][tuple(x[j - order:j])][observed] += 1
        for order in (1, 2, 3):
            if j >= order - 1:
                previous = histories[order].get(tuple(x[j - order + 1:j + 1]))
                if previous:
                    total = sum(previous.values())
                    support[order, j] = total
                    q[order, j] = previous[target] / total
    return q, support


def load_train_and_validation(table):
    data_dir = ROOT / 'data'
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    for name in ('wikitext_train.txt', 'wikitext_validation.txt', 'tokenizer.json'):
        if sha(data_dir / name) != manifest['sha256'][name]:
            raise ValueError(f'Changed benchmark file: {name}')
    tokenizer = Tokenizer.from_file(str(data_dir / 'tokenizer.json'))
    train_raw = (data_dir / 'wikitext_train.txt').read_bytes()
    train_ids = tokenizer.encode(train_raw.decode('utf-8')).ids
    if len(train_ids) != table.train_tokens:
        raise ValueError('N-gram asset token count differs from the fixed training text')
    counts = np.bincount(np.asarray(train_ids, dtype=np.int64), minlength=2048)
    if not np.array_equal(counts, table.unigrams):
        raise ValueError('N-gram asset unigram counts differ from the fixed training text')
    raw = (data_dir / 'wikitext_validation.txt').read_bytes()
    tokens = torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids, dtype=torch.long)
    return tokens, len(raw), manifest


def load_member(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('protocol') != PROTOCOL:
        raise ValueError(f'Checkpoint has a different protocol: {path}')
    if checkpoint['config']['vocab'] != 2048 or checkpoint['config']['context'] != 256:
        raise ValueError(f'Checkpoint has the wrong vocabulary or context: {path}')
    module = importlib.import_module(checkpoint['implementation'])
    model = module.build_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    return model.eval()


def model_bias(table, device):
    # Keep dtype, smoothing, and centering identical to build_hybrid.py.
    prior = (table.unigrams.astype(np.float64) + 1) / (table.train_tokens + 2048)
    bias = np.log(prior).astype(np.float32)
    bias_tensor = torch.from_numpy(bias)
    bias_tensor -= bias_tensor.mean()
    return bias_tensor.to(device) * BIAS_SCALE


def selection_grids(args):
    large = ((args.large_temperature,) if args.large_temperature is not None
             else TEMPERATURES)
    small = ((args.small_temperature,) if args.small_temperature is not None
             else TEMPERATURES)
    weights = ((args.small_weight,) if args.small_weight is not None
               else SMALL_WEIGHTS)
    return large, small, weights


def consistency_check(ngram, smoothing):
    """Compare target-only formulas with complete inference distributions."""
    x = torch.tensor([
        [3, 5, 3, 5, 3, 5, 7, 3, 5, 7, 3, 5],
        [11, 12, 11, 13, 11, 12, 11, 13, 11, 12, 11, 13],
    ], dtype=torch.long)
    y = torch.tensor([
        [5, 3, 5, 3, 5, 7, 3, 5, 7, 3, 5, 17],
        [12, 11, 13, 11, 12, 11, 13, 11, 12, 11, 13, 19],
    ], dtype=torch.long)
    target = y.unsqueeze(-1)
    expected_ngram = selected_target_probabilities(ngram, x, y, smoothing)
    full_ngram = ngram.probabilities(x, smoothing)
    actual_ngram = np.take_along_axis(
        full_ngram, target.numpy(), axis=-1
    ).squeeze(-1).reshape(-1)
    np.testing.assert_allclose(expected_ngram, actual_ngram, rtol=2e-5, atol=1e-7)
    changed = x.clone()
    changed[:, 7:] = (changed[:, 7:] + 19) % 2048
    changed_ngram = ngram.probabilities(changed, smoothing)
    np.testing.assert_allclose(full_ngram[:, :7], changed_ngram[:, :7], rtol=0, atol=0)

    class Uniform(nn.Module):
        context = 256

        def predict_probs(self, ids):
            return torch.full((*ids.shape, 2048), 1 / 2048, dtype=torch.float32)

    q_rows, n_rows = zip(*(target_local_stats(a.tolist(), b.tolist()) for a, b in zip(x, y)))
    q = np.concatenate(q_rows, axis=1)
    n = np.concatenate(n_rows, axis=1).astype(np.float64)
    g1 = 0.05 * n[1] / (n[1] + 1) * (n[2] == 0)
    g2 = 0.4 * n[2] / (n[2] + 1)
    g3 = 0.2 * n[3] / (n[3] + 1)
    target_only = (1 - g3) * ((1 - g1) * ((1 - g2) / 2048 + g2 * q[2])
                              + g1 * q[1]) + g3 * q[3]
    local = CausalHierarchicalCache(Uniform())
    full_local = local.predict_probs(x)
    changed_local = local.predict_probs(changed)
    torch.testing.assert_close(full_local[:, :7], changed_local[:, :7], rtol=0, atol=0)
    actual_local = full_local.gather(-1, target).squeeze(-1).numpy().reshape(-1)
    np.testing.assert_allclose(target_only, actual_local, rtol=2e-5, atol=1e-7)
    return {'ngram_max_abs_error': float(np.max(np.abs(expected_ngram - actual_ngram))),
            'local_max_abs_error': float(np.max(np.abs(target_only - actual_local)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--large', type=Path)
    parser.add_argument('--small', type=Path)
    parser.add_argument('--ngram-asset', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--ngram-alpha', type=float, default=4.0)
    parser.add_argument('--ngram-weight', type=float, default=0.1)
    parser.add_argument('--ngram-method', choices=('dirichlet', 'continuation'),
                        default='dirichlet')
    parser.add_argument('--ngram-discount', type=float, default=0.5)
    parser.add_argument('--large-temperature', type=float,
                        help='Fix the large-model temperature; otherwise search the default grid.')
    parser.add_argument('--small-temperature', type=float,
                        help='Fix the small-model temperature; otherwise search the default grid.')
    parser.add_argument('--small-weight', type=float,
                        help='Fix the small-model mixture weight; otherwise search the default grid.')
    parser.add_argument('--self-check-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or args.ngram_alpha <= 0:
        parser.error('Batch size, threads, and n-gram alpha must be positive.')
    if not 0 <= args.ngram_weight < 1:
        parser.error('N-gram weight must be in [0, 1).')
    if not 0 < args.ngram_discount < 1:
        parser.error('N-gram discount must be in (0, 1).')
    for name in ('large_temperature', 'small_temperature'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f'--{name.replace("_", "-")} must be finite and positive.')
    if args.small_weight is not None and (
        not math.isfinite(args.small_weight) or not 0 < args.small_weight < 1
    ):
        parser.error('--small-weight must be finite and in (0, 1).')
    if not args.self_check_only and any(value is None for value in (args.large, args.small, args.output)):
        parser.error('--large, --small, and --output are required for selection.')
    if args.output and args.output.exists():
        parser.error('Use a new output path; existing selections are never overwritten.')

    large_temperatures, small_temperatures, small_weights = selection_grids(args)
    started = time.perf_counter()
    table = PackedTrainNgrams(args.ngram_asset)
    ngram = ContinuationNgrams(table) if args.ngram_method == 'continuation' else table
    smoothing = (args.ngram_discount if args.ngram_method == 'continuation'
                 else args.ngram_alpha)
    check = consistency_check(ngram, smoothing)
    if args.self_check_only:
        print(json.dumps({'self_check_passed': True, **check,
                          'candidate_count': (len(large_temperatures)
                                              * len(small_temperatures) * len(small_weights))},
                         indent=2))
        return

    device, _ = setup(args.device, 'fp32', args.threads)
    tokens, byte_count, manifest = load_train_and_validation(table)
    models = [load_member(path, device) for path in (args.large, args.small)]
    bias = model_bias(table, device)
    temperature_grids = (large_temperatures, small_temperatures)
    prepared_seconds = time.perf_counter() - started
    target_logp = [[[] for _ in grid] for grid in temperature_grids]
    local_q, local_n, ngram_q = [], [], []
    prediction_started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (x, y) in enumerate(windows(tokens, batch_size=args.batch_size)):
            valid = y != -100
            x_device = x.to(device)
            target = y.clamp_min(0).to(device).unsqueeze(-1)
            for member_index, model in enumerate(models):
                logits = model(x_device).float()
                for temperature_index, temperature in enumerate(temperature_grids[member_index]):
                    logp = F.log_softmax(logits / temperature + bias, dim=-1)
                    target_logp[member_index][temperature_index].append(
                        logp.gather(-1, target).squeeze(-1).cpu()[valid].numpy()
                    )
            ngram_q.append(selected_target_probabilities(ngram, x, y, smoothing))
            for row in range(len(x)):
                size = int(valid[row].sum())
                q, n = target_local_stats(x[row, :size].tolist(), y[row, :size].tolist())
                local_q.append(q)
                local_n.append(n)
            if (batch_index + 1) % 100 == 0:
                print(json.dumps({'validation_batches': batch_index + 1,
                                  'targets': sum(len(item) for item in ngram_q)}), flush=True)
    prediction_seconds = time.perf_counter() - prediction_started
    q = np.concatenate(local_q, axis=1)
    n = np.concatenate(local_n, axis=1).astype(np.float64)
    train_q = np.concatenate(ngram_q)
    member = [[np.exp(np.concatenate(parts).astype(np.float64)) for parts in grid]
              for grid in target_logp]
    g1 = 0.05 * n[1] / (n[1] + 1) * (n[2] == 0)
    g2 = 0.4 * n[2] / (n[2] + 1)
    g3 = 0.2 * n[3] / (n[3] + 1)
    denominator = math.log(2) * byte_count
    candidates = []
    for large_index, large_temperature in enumerate(large_temperatures):
        for small_index, small_temperature in enumerate(small_temperatures):
            for small_weight in small_weights:
                neural = ((1 - small_weight) * member[0][large_index]
                          + small_weight * member[1][small_index])
                cached = ((1 - g3) * ((1 - g1) * ((1 - g2) * neural + g2 * q[2])
                                      + g1 * q[1]) + g3 * q[3])
                final = (1 - args.ngram_weight) * cached + args.ngram_weight * train_q
                candidates.append({
                    'large_temperature': large_temperature,
                    'small_temperature': small_temperature,
                    'small_weight': small_weight,
                    'bias_scale': BIAS_SCALE,
                    'ngram_alpha': args.ngram_alpha,
                    'ngram_weight': args.ngram_weight,
                    'ngram_method': args.ngram_method,
                    'ngram_discount': args.ngram_discount,
                    'ensemble_bpb': float(-np.log(neural).sum() / denominator),
                    'local_cache_bpb': float(-np.log(cached).sum() / denominator),
                    'bpb': float(-np.log(final).sum() / denominator),
                })
    candidates.sort(key=lambda row: row['bpb'])
    result = {
        'protocol': PROTOCOL,
        'split': 'validation',
        'large': str(args.large.resolve()),
        'small': str(args.small.resolve()),
        'ngram_asset': str(args.ngram_asset.resolve()),
        'large_sha256': sha(args.large),
        'small_sha256': sha(args.small),
        'ngram_asset_sha256': sha(args.ngram_asset),
        'selector_sha256': sha(Path(__file__)),
        'train_text_sha256': manifest['sha256']['wikitext_train.txt'],
        'validation_text_sha256': manifest['sha256']['wikitext_validation.txt'],
        'tokenizer_sha256': manifest['sha256']['tokenizer.json'],
        'train_tokens': table.train_tokens,
        'ngram_method': args.ngram_method,
        'ngram_discount': args.ngram_discount,
        'targets': len(train_q),
        'utf8_bytes': byte_count,
        'device': str(device),
        'precision': 'fp32',
        'batch_size': args.batch_size,
        'large_temperature_grid': large_temperatures,
        'small_temperature_grid': small_temperatures,
        'small_weight_grid': small_weights,
        'self_check': check,
        'preparation_seconds': prepared_seconds,
        'prediction_seconds': prediction_seconds,
        'total_seconds': time.perf_counter() - started,
        'selection': 'All model weights and counts derive only from train; validation selects the scalar grid.',
        'best': candidates[0],
        'candidates': candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'candidates'},
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
