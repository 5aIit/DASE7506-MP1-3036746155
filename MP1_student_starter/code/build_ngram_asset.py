"""Rebuild the uncompressed count asset using only the supplied training text."""
import argparse
import json
from pathlib import Path
import struct
import time

import numpy as np
from tokenizers import Tokenizer

from common import ROOT, sha
from train_ngrams import MAGIC, WIDTHS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new output path.')
    started = time.perf_counter()
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    for name in ('wikitext_train.txt','tokenizer.json'):
        if sha(ROOT/'data'/name) != manifest['sha256'][name]:
            raise ValueError('Changed benchmark asset: '+name)
    tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    text = (ROOT/'data/wikitext_train.txt').read_bytes().decode('utf-8')
    train = np.asarray(tokenizer.encode(text).ids,dtype=np.uint64)
    unigram = np.bincount(train.astype(np.int64),minlength=2048).astype('<u4')
    tables = []
    for order in (2,3,4):
        n = len(train)-order+1
        keys = np.zeros(n,dtype=np.uint64)
        for offset in range(order):
            keys = (keys << 11) | train[offset:offset+n]
        unique,counts = np.unique(keys,return_counts=True)
        if int(counts.max()) > 65535:
            raise ValueError('Training counts exceed the declared uint16 format.')
        tables.append((unique.astype('<u8'),counts.astype('<u2')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as stream:
        stream.write(struct.pack('<8s4I',MAGIC,len(train),*[len(k) for k,v in tables]))
        stream.write(unigram.tobytes())
        for order,(keys,counts) in zip((2,3,4),tables):
            stream.write(keys.view(np.uint8).reshape(-1,8)[:,:WIDTHS[order]].tobytes())
            stream.write(counts.tobytes())
    result = dict(train_tokens=len(train),bytes=args.output.stat().st_size,
                  sha256=sha(args.output),seconds=time.perf_counter()-started,
                  train_text_sha256=sha(ROOT/'data/wikitext_train.txt'),
                  tokenizer_sha256=sha(ROOT/'data/tokenizer.json'))
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
