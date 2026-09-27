"""Build a self-contained probability-ensemble checkpoint from two members."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from common import PROTOCOL


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--large-checkpoint', required=True, type=Path)
    parser.add_argument('--small-checkpoint', required=True, type=Path)
    parser.add_argument('--small-weight', required=True, type=float)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not 0.0 < args.small_weight < 1.0:
        parser.error('--small-weight must be strictly between 0 and 1.')

    large = torch.load(args.large_checkpoint, map_location='cpu', weights_only=True)
    small = torch.load(args.small_checkpoint, map_location='cpu', weights_only=True)
    if large.get('protocol') != PROTOCOL or small.get('protocol') != PROTOCOL:
        parser.error('Both members must use the course evaluation protocol.')
    if large.get('implementation') == 'ensemble' or small.get('implementation') == 'ensemble':
        parser.error('Nested ensembles are not supported.')

    config = {
        'vocab': 2048,
        'context': 256,
        'small_weight': args.small_weight,
        'large_implementation': large['implementation'],
        'large_config': large['config'],
        'small_implementation': small['implementation'],
        'small_config': small['config'],
    }
    state = {'large.' + key: value for key, value in large['model'].items()}
    state.update({'small.' + key: value for key, value in small['model'].items()})
    def selected_tokens(checkpoint):
        if checkpoint.get('best_validation_step') is None:
            return checkpoint.get('cumulative_train_tokens', checkpoint.get('train_tokens', 0))
        return checkpoint.get('parent_train_tokens', 0) + (
            checkpoint['best_validation_step'] * checkpoint.get('batch_size', 32) * 256
        )

    large_selected = selected_tokens(large)
    small_selected = selected_tokens(small)
    large_search = large.get('training_run_tokens', large.get('train_tokens', 0))
    small_search = small.get('training_run_tokens', small.get('train_tokens', 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        'protocol': PROTOCOL,
        'implementation': 'ensemble',
        'config': config,
        'model': state,
        'train_tokens': large_selected + small_selected,
        'search_training_tokens': large_search + small_search,
        'parents': [
            {
                'filename': args.large_checkpoint.name,
                'sha256': sha(args.large_checkpoint),
                'selected_train_tokens': large_selected,
            },
            {
                'filename': args.small_checkpoint.name,
                'sha256': sha(args.small_checkpoint),
                'selected_train_tokens': small_selected,
            },
        ],
        'selection': 'Probability mixture weight selected using validation only; test not used for selection.',
    }
    torch.save(bundle, args.output)
    print(json.dumps({
        'output': str(args.output),
        'bytes': args.output.stat().st_size,
        'sha256': sha(args.output),
        'selected_training_targets': bundle['train_tokens'],
        'search_training_targets': bundle['search_training_tokens'],
    }, indent=2))


if __name__ == '__main__':
    main()
