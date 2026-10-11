"""Prepare the geodesic-side data for the ASA experiments (CPU job).

From geodesic-research/synth-scenario-docs-positive-alignment-midtraining (fields: text = passage
document, question, choices[2], misaligned_choice, question_id):
  * scenarios (question_id) are split into train / held-out (10%);
  * train/w1/     : passage documents of TRAIN scenarios (unlabelled midtraining data), token stream;
  * val_midtrain/ : passage documents of HELD-OUT scenarios (alignment-document validation loss);
  * train/y1/ (aligned) and train/z1/ (misaligned): labelled completions "question\\n<choice>" of TRAIN
                    scenarios packed into items of exactly SEQ+1 = 2049 tokens that start with the label token
                    (TOK_POS = 50300 / TOK_NEG = 50301), so every training block carries its label at position 0;
  * heldout_pairs.jsonl : (question, aligned, misaligned) of HELD-OUT scenarios (evaluation of Delta and U).
Tokenisation happens once in the main process (fast tokenizer, batched); litdata workers read items from a
numpy memmap, so nothing large is pickled.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

SEQ = 2048
TOK_POS, TOK_NEG = 50300, 50301
EOS = 0  # <|endoftext|>

_MM = {}


def _load_items(spec_path: str):
    if spec_path not in _MM:
        spec = json.load(open(spec_path))
        toks = np.load(spec["tokens"], mmap_mode="r")
        offs = np.load(spec["offsets"])
        _MM[spec_path] = (toks, offs)
    return _MM[spec_path]


def item_fn(spec_path: str, i: int):
    toks, offs = _load_items(spec_path)
    return torch.from_numpy(np.array(toks[offs[i]:offs[i + 1]], dtype=np.int32))


def write_shard(spec_path: str, n_items: int, out_dir: Path, block_size: int, num_workers: int):
    from functools import partial
    from litdata import optimize
    from litdata.streaming import TokensLoader
    out_dir.mkdir(parents=True, exist_ok=True)
    optimize(fn=partial(item_fn, spec_path), inputs=list(range(n_items)), output_dir=str(out_dir),
             num_workers=num_workers, chunk_bytes="200MB", item_loader=TokensLoader(block_size=block_size), mode="overwrite")


def save_items(items, tmp_dir: Path, name: str):
    """items: list of int lists -> memmap-able token file + offsets + spec json"""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    lens = np.array([len(x) for x in items], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)])
    toks = np.zeros(int(offs[-1]), dtype=np.int32)
    for i, x in enumerate(items):
        toks[offs[i]:offs[i + 1]] = x
    np.save(tmp_dir / f"{name}_tokens.npy", toks)
    np.save(tmp_dir / f"{name}_offsets.npy", offs)
    spec = tmp_dir / f"{name}_spec.json"
    json.dump({"tokens": str(tmp_dir / f"{name}_tokens.npy"), "offsets": str(tmp_dir / f"{name}_offsets.npy")}, open(spec, "w"))
    return str(spec), len(items), int(offs[-1])


def encode_all(tok, texts, batch=2000):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(list(texts[i:i + batch]), add_special_tokens=False)["input_ids"]
        out.extend([e + [EOS] for e in enc])
    return out


def pack_labelled(docs, label_tok, seed, repeats=2):
    """Pack EOS-separated completions into SEQ-token bodies; `repeats` independent shuffles give different block
    boundaries. Many repeats are needed because litdata assigns each (rank, worker) only
    num_items // (world_size*num_workers) items, rounded down to a multiple of the batch size when drop_last is set:
    a shard with fewer than ~world_size*num_workers*batch_size items yields nothing in a multi-GPU run."""
    rng = random.Random(seed)
    order = list(range(len(docs)))
    stream = []
    for _ in range(repeats):
        rng.shuffle(order)
        for i in order:
            stream.extend(docs[i])
    n_items = len(stream) // SEQ
    return [[label_tok] + stream[k * SEQ:(k + 1) * SEQ] for k in range(n_items)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="geodesic-research/synth-scenario-docs-positive-alignment-midtraining")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--heldout-fraction", type=float, default=0.10)
    p.add_argument("--max-train-passages", type=int, default=300000)
    p.add_argument("--max-val-passages", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--labelled-repeats", type=int, default=160)
    p.add_argument("--labelled-only", action="store_true", help="only (re)build y1/z1 and heldout_pairs.jsonl")
    a = p.parse_args()

    from datasets import load_dataset
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.tokenizer_dir, "tokenizer.json"))
    if os.path.isdir(a.dataset):  # local snapshot of the dataset repo (data/*.parquet)
        import glob
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import Dataset
        files = sorted(glob.glob(os.path.join(a.dataset, "data", "*.parquet")))
        cols = ["text", "question", "choices", "misaligned_choice", "question_id"]
        tables = []
        for f in files:
            t = pq.read_table(f)
            t = t.select([c for c in cols if c in t.column_names])
            tables.append(t.replace_schema_metadata(None))  # drop HF feature metadata written by a newer `datasets`
        ds = Dataset(pa.concat_tables(tables))
    else:
        ds = load_dataset(a.dataset, split="train")
    cols = ds.column_names
    print("columns:", cols, "rows:", len(ds), flush=True)
    qid_key = "question_id" if "question_id" in cols else "question"
    text_key = "text" if "text" in cols else "passage"
    all_q = ds[qid_key]
    qids = sorted(set(all_q))
    rng = random.Random(a.seed)
    rng.shuffle(qids)
    n_held = max(1, int(round(a.heldout_fraction * len(qids))))
    held = set(qids[:n_held])
    print(f"unique scenarios: {len(qids)}, held-out: {n_held}", flush=True)

    first_row = {}
    for i, q in enumerate(all_q):
        first_row.setdefault(q, i)
    train_rows = [first_row[q] for q in qids if q not in held]
    held_rows = [first_row[q] for q in qids if q in held]

    def pair(i):
        r = ds[i]
        mis = int(r["misaligned_choice"])
        ch = r["choices"]
        if isinstance(ch, str):
            import ast
            ch = ast.literal_eval(ch)
        assert len(ch) == 2 and mis in (0, 1), (i, ch, mis)
        return r["question"].strip(), str(ch[1 - mis]).strip(), str(ch[mis]).strip()

    a.out_root.mkdir(parents=True, exist_ok=True)
    tmp = a.out_root / "_tmp_tokens"
    aligned, misaligned = [], []
    for i in train_rows:
        q, cp, cn = pair(i)
        aligned.append(f"{q}\n{cp}"); misaligned.append(f"{q}\n{cn}")
    with open(a.out_root / "heldout_pairs.jsonl", "w") as fh:
        for i in held_rows:
            q, cp, cn = pair(i)
            fh.write(json.dumps({"question": q, "aligned": cp, "misaligned": cn, "row": i}) + "\n")
    y_items = pack_labelled(encode_all(tok, aligned), TOK_POS, a.seed, a.labelled_repeats)
    z_items = pack_labelled(encode_all(tok, misaligned), TOK_NEG, a.seed + 1, a.labelled_repeats)
    spec, n, t = save_items(y_items, tmp, "y1"); write_shard(spec, n, a.out_root / "train" / "y1", SEQ + 1, min(a.num_workers, 4)); ny, ty = n, t
    spec, n, t = save_items(z_items, tmp, "z1"); write_shard(spec, n, a.out_root / "train" / "z1", SEQ + 1, min(a.num_workers, 4)); nz, tz = n, t
    print(f"labelled shards: aligned {ny} items ({ty} tokens), misaligned {nz} items ({tz} tokens)", flush=True)
    if a.labelled_only:
        (a.out_root / "labelled_meta.json").write_text(json.dumps(dict(aligned_items=ny, misaligned_items=nz, aligned_tokens=ty, misaligned_tokens=tz, repeats=a.labelled_repeats, seed=a.seed, n_train_scenarios=len(train_rows), n_heldout=n_held)))
        return

    train_idx = [i for i, q in enumerate(all_q) if q not in held]
    val_idx = [i for i, q in enumerate(all_q) if q in held]
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    train_idx = train_idx[: a.max_train_passages]; val_idx = val_idx[: a.max_val_passages]
    for name, idx, out in (("w1", train_idx, a.out_root / "train" / "w1"), ("val_midtrain", val_idx, a.out_root / "val_midtrain")):
        texts = ds.select(idx)[text_key]
        docs = encode_all(tok, texts)
        spec, n, t = save_items(docs, tmp, name)
        write_shard(spec, n, out, SEQ, a.num_workers)
        print(f"{name}: {n} docs, {t} tokens", flush=True)
    meta = dict(dataset=a.dataset, n_rows=len(ds), n_scenarios=len(qids), n_heldout=n_held, n_train_passages=len(train_idx),
                n_val_passages=len(val_idx), aligned_items=ny, misaligned_items=nz, aligned_tokens=ty, misaligned_tokens=tz,
                seq=SEQ, label_tokens=[TOK_POS, TOK_NEG], seed=a.seed)
    (a.out_root / "geodesic_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
