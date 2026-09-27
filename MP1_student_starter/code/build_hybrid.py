"""Bundle neural members and train-only count bytes into one portable checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import PROTOCOL, ROOT, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--large', type=Path, required=True)
    p.add_argument('--small', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--ngram-asset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output path; existing checkpoints are never overwritten.')
    parents = [torch.load(path,map_location='cpu',weights_only=True) for path in (args.large,args.small)]
    if any(c['protocol'] != PROTOCOL for c in parents):
        p.error('Checkpoint protocol mismatch.')
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    for key, path in (('large',args.large),('small',args.small)):
        if Path(selection[key]).resolve() != path.resolve():
            p.error('Selection record refers to different member checkpoints.')
        if selection.get(key+'_sha256') != sha(path):
            p.error('Selection member hash is missing or differs from the checkpoint.')
    if (Path(selection.get('ngram_asset', '')).resolve() != args.ngram_asset.resolve()
            or selection.get('ngram_asset_sha256') != sha(args.ngram_asset)):
        p.error('Selection must identify the same training-count asset path and SHA256.')
    chosen = selection['best']
    fast = {'student':'student_fast','model':'model_fast'}
    config = dict(vocab=2048,context=256,
                  large_implementation=fast.get(parents[0]['implementation'],parents[0]['implementation']),
                  small_implementation=fast.get(parents[1]['implementation'],parents[1]['implementation']),
                  large_config=parents[0]['config'],small_config=parents[1]['config'])
    for key in ('small_weight','large_temperature','small_temperature','bias_scale','ngram_weight','ngram_alpha'):
        config[key] = chosen[key]
    config['ngram_method'] = chosen.get('ngram_method', 'dirichlet')
    config['ngram_discount'] = chosen.get('ngram_discount', .5)
    raw = args.ngram_asset.read_bytes()
    from train_ngrams import PackedTrainNgrams
    table = PackedTrainNgrams(raw=raw)
    prior = (table.unigrams.astype(np.float64)+1)/(table.train_tokens+2048)
    bias = np.log(prior).astype(np.float32)
    # Match the torch mean used by the validation probe.
    bias_tensor = torch.from_numpy(bias)
    bias_tensor -= bias_tensor.mean()
    config.update(log_frequency_bias=bias_tensor.tolist(),ngram_bytes=len(raw),ngram_sha256=sha(args.ngram_asset))
    state = {'ngram_data':torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy())}
    for member, parent in zip(('large','small'),parents):
        state.update({'local.base.'+member+'.'+name:value for name,value in parent['model'].items()})
    selected_targets = sum(c.get('cumulative_train_tokens',c['train_tokens']) for c in parents)
    bundle = dict(protocol=PROTOCOL,implementation='hybrid',config=config,model=state,
                  train_tokens=selected_targets,cumulative_train_tokens=selected_targets,
                  count_model_train_tokens=table.train_tokens,
                  parents=[dict(sha256=sha(path),train_tokens=c.get('cumulative_train_tokens',c['train_tokens']))
                           for path,c in zip((args.large,args.small),parents)],
                  selection_sha256=sha(args.selection),validation_bpb=chosen['bpb'],
                  train_text_sha256=sha(ROOT/'data/wikitext_train.txt'),
                  tokenizer_sha256=sha(ROOT/'data/tokenizer.json'),
                  selection='Validation-only selection. All statistics derive from supplied training text.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(bundle,args.output)
    print(json.dumps(dict(output=str(args.output),sha256=sha(args.output),bytes=args.output.stat().st_size,
                          validation_bpb=chosen['bpb']),indent=2))


if __name__ == '__main__':
    main()
