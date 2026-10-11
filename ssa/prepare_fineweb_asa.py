"""Tokenise a subset of FineWeb-Edu (sample-10BT) from LOCAL parquet files into litdata shards, without
network access on the compute node. Mirrors util_scripts/prepare_geodesic_litdata.py: a seeded permutation of
all rows, the first --num-train rows form train/1 and rows [--val-skip, --val-skip + --num-val) form val_pretrain.
Tokenisation is done in the main process (fast tokenizer, batched, multithreaded); tokens are appended to a raw
int32 file and litdata workers read documents from it through a memmap (no large objects are pickled).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

EOS = 0
_MM = {}


def _load(spec_path):
    if spec_path not in _MM:
        spec = json.load(open(spec_path))
        toks = np.memmap(spec["tokens"], dtype=np.int32, mode="r")
        offs = np.load(spec["offsets"])
        _MM[spec_path] = (toks, offs)
    return _MM[spec_path]


def item_fn(spec_path, i):
    toks, offs = _load(spec_path)
    return torch.from_numpy(np.array(toks[offs[i]:offs[i + 1]], dtype=np.int32))


def write_shard(spec_path, n_items, out_dir, num_workers):
    from functools import partial
    from litdata import optimize
    from litdata.streaming import TokensLoader
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    optimize(fn=partial(item_fn, spec_path), inputs=list(range(n_items)), output_dir=str(out_dir), num_workers=num_workers,
             chunk_bytes="200MB", item_loader=TokensLoader(block_size=2048), mode="overwrite")


def tokenise_selected(files, selected_mask, tok, tmp: Path, name: str, batch=4000, log_every=200000):
    """Stream the parquet files, tokenise the selected rows (in file order), append to tmp/name_tokens.bin."""
    tok_path, off_path = tmp / f"{name}_tokens.bin", tmp / f"{name}_offsets.npy"
    offsets = [0]
    n_done, t0 = 0, time.time()
    with open(tok_path, "wb") as fh:
        gi = 0
        for f in files:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                col = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
                n = len(col)
                sel = selected_mask[gi:gi + n]
                texts = [t for t, m in zip(col, sel) if m]
                gi += n
                for b in range(0, len(texts), batch):
                    enc = tok(texts[b:b + batch], add_special_tokens=False)["input_ids"]
                    for ids in enc:
                        ids.append(EOS)
                        arr = np.asarray(ids, dtype=np.int32)
                        fh.write(arr.tobytes())
                        offsets.append(offsets[-1] + arr.size)
                    n_done += len(enc)
                    if n_done % log_every < batch:
                        print(f"[{name}] {n_done} docs, {offsets[-1]/1e9:.2f}B tokens, {time.time()-t0:.0f}s", flush=True)
    np.save(off_path, np.asarray(offsets, dtype=np.int64))
    spec = tmp / f"{name}_spec.json"
    json.dump({"tokens": str(tok_path), "offsets": str(off_path)}, open(spec, "w"))
    return str(spec), len(offsets) - 1, offsets[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", required=True, help="local snapshot dir containing sample/10BT/*.parquet")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--num-train", type=int, default=4500000)
    p.add_argument("--num-val", type=int, default=50000)
    p.add_argument("--val-skip", type=int, default=9000000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-workers", type=int, default=16)
    a = p.parse_args()
    from transformers import PreTrainedTokenizerFast
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.tokenizer_dir, "tokenizer.json"))
    files = sorted(glob.glob(os.path.join(a.parquet_dir, "sample", "10BT", "*.parquet")))
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    total = sum(counts)
    print(f"{len(files)} parquet files, {total} rows", flush=True)
    indices = list(range(total))
    random.Random(a.seed).shuffle(indices)          # same permutation rule as prepare_geodesic_litdata.py
    train_idx = indices[: a.num_train]
    val_idx = indices[a.val_skip: a.val_skip + a.num_val]
    train_mask = np.zeros(total, dtype=bool); train_mask[np.asarray(train_idx)] = True
    val_mask = np.zeros(total, dtype=bool); val_mask[np.asarray(val_idx)] = True
    tmp = a.out_root / "_tmp_tokens"; tmp.mkdir(parents=True, exist_ok=True)
    spec, n, t = tokenise_selected(files, val_mask, tok, tmp, "val_pretrain")
    write_shard(spec, n, a.out_root / "val_pretrain", min(a.num_workers, 8)); print(f"val_pretrain: {n} docs, {t} tokens", flush=True)
    spec, n, t = tokenise_selected(files, train_mask, tok, tmp, "main1")
    write_shard(spec, n, a.out_root / "train" / "1", a.num_workers); print(f"train/1: {n} docs, {t} tokens", flush=True)
    json.dump({"files": files, "rows": total, "num_train": a.num_train, "num_val": a.num_val, "val_skip": a.val_skip, "seed": a.seed,
               "train_tokens": t}, open(a.out_root / "fineweb_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
