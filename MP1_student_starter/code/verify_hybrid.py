"""Check the actual frozen hybrid checkpoint against the causal interface."""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from common import ROOT, make_model, setup, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    args = p.parse_args()
    device,_ = setup('cpu','fp32',4)
    ckpt = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model,_ = make_model(ckpt['implementation'],ckpt['config'],device)
    model.load_state_dict(ckpt['model'],strict=True)
    model.eval()
    tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    text = (ROOT/'data/wikitext_train.txt').read_text(encoding='utf-8')[:10000]
    ids = tokenizer.encode(text).ids[:512]
    x = torch.tensor(ids).reshape(2,256)
    changed = x.clone()
    changed[:,127:] = (changed[:,127:]+31)%2048
    with torch.no_grad():
        original = model.predict_log_probs(x)
        altered = model.predict_log_probs(changed)
        again = model.predict_log_probs(x)
        alone = model.predict_log_probs(x[:1])
        assert original.shape == (2,256,2048) and torch.isfinite(original).all()
        torch.testing.assert_close(original.logsumexp(-1),torch.zeros(2,256),atol=1e-5,rtol=0)
        torch.testing.assert_close(original[:,:127],altered[:,:127],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(original,again,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(original[:1],alone,atol=2e-5,rtol=2e-5)
        for length in (1,2,3,17):
            short = model.predict_log_probs(x[:1,:length])
            torch.testing.assert_close(short.logsumexp(-1),torch.zeros(1,length),atol=1e-5,rtol=0)
            torch.testing.assert_close(short,original[:1,:length],atol=3e-5,rtol=3e-5)
    loss = -model.predict_log_probs(x[:1,:12]).gather(-1,x[:1,1:13,None]).mean()
    loss.backward()
    gradients = [v.grad for v in model.parameters() if v.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    result = dict(checkpoint_sha256=sha(args.checkpoint),passed=True,
                  checks=['finite normalized distribution','no future inputs','batch independence',
                          'window/call reset','short prefixes','finite neural gradients',
                          'embedded training-asset hash'])
    (args.checkpoint.parent/'contract_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
