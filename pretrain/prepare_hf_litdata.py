#!/usr/bin/env python
"""Tokenise a Hugging Face text dataset into litdata token shards (pretraining stage).

A seeded permutation of the rows is taken once; --num-samples rows after --skip-samples are tokenised with the
named tokenizer (EOS appended) and written with litdata.optimize as blocks of --block-size tokens. Used for the
9,000,000-document FineWeb-Edu training shard and the 50,000-document validation shard of the pretraining run.
"""
import argparse
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from datasets import load_dataset
from litdata import optimize
from litdata.streaming.item_loader import TokensLoader
from transformers import AutoTokenizer


def get_text(example: Dict, text_column: str) -> str:
    if text_column in example and isinstance(example[text_column], str):
        return example[text_column]

    # Fallbacks for chat-style or instruction-style datasets.
    if "messages" in example and isinstance(example["messages"], list):
        parts = []
        for msg in example["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    parts = []
    for key in ["passage", "question", "choices", "answer", "output", "content"]:
        value = example.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            value = "\n".join(map(str, value))
        parts.append(str(value))
    return "\n\n".join(parts)


def tokenize_fn(tokenizer_name: str, dataset_name: str, name: str, split: str, text_column: str, idx: int):
    # `idx` is a row index into the memory-mapped Arrow dataset. Each worker
    # reads only its own rows off disk, so the corpus is never held in RAM.
    # NOTE: this must `return` a single item (not `yield`): litdata's
    # use_checkpoint resume is unsupported for generator `fn`s. Returning None
    # skips the item (empty docs).
    ds = _get_worker_dataset(dataset_name, name, split)
    tokenizer = _get_worker_tokenizer(tokenizer_name)
    text = get_text(ds[idx], text_column).strip()
    if not text:
        return None
    ids = tokenizer.encode(text)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        ids = ids + [eos_id]
    return torch.tensor(ids, dtype=torch.int32)


_WORKER_TOKENIZER = None
_WORKER_DATASET = None


def _get_worker_tokenizer(tokenizer_name: str):
    global _WORKER_TOKENIZER
    if _WORKER_TOKENIZER is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tok.eos_token_id is None:
            tok.eos_token = tok.pad_token or tok.unk_token
        _WORKER_TOKENIZER = tok
    return _WORKER_TOKENIZER


def _get_worker_dataset(dataset_name: str, name: str, split: str):
    global _WORKER_DATASET
    if _WORKER_DATASET is None:
        load_kwargs = dict(split=split, trust_remote_code=True)
        if name:
            load_kwargs["name"] = name
        # Non-streaming -> Arrow-backed and memory-mapped: random access by
        # index without loading the whole dataset into RAM.
        _WORKER_DATASET = load_dataset(dataset_name, **load_kwargs)
    return _WORKER_DATASET


def build_indices(dataset_name: str, split: str, n: int, seed: int, skip_samples: int, name: str = None) -> List[int]:
    # Load once (memory-mapped) just to learn the row count, then build a
    # shuffled list of integer indices. Only ints live in RAM here.
    ds = _get_worker_dataset(dataset_name, name, split)
    total = len(ds)
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return indices[skip_samples : skip_samples + n]


def write_litdata(indices: List[int], dataset_name: str, name: str, split: str, output_dir: str, tokenizer_name: str, text_column: str, num_workers: int, block_size: int, resume: bool = True):
    output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Pass only integer indices as `inputs`; litdata distributes a slice to each
    # worker and each worker reads its rows from the mmap'd Arrow files inside
    # `fn`. Nothing duplicates the corpus -> no OOM.
    #
    # use_checkpoint=True makes litdata record per-worker progress under
    # <output_dir>/.checkpoints. On a rerun with the SAME inputs (guaranteed by
    # --seed) it skips already-finished items and re-does only the partial chunk
    # that was being written when the previous run was interrupted. mode="append"
    # lets it continue into a non-empty output dir instead of erroring out.
    optimize(
        fn=partial(tokenize_fn, tokenizer_name, dataset_name, name, split, text_column),
        inputs=indices,
        output_dir=output_dir,
        num_workers=num_workers,
        chunk_bytes="200MB",
        item_loader=TokensLoader(block_size=block_size),
        use_checkpoint=resume,
        mode="append" if resume else None,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="HF dataset name")
    p.add_argument("--name", default=None, help="HF dataset config name, e.g. 'sample-10BT' for HuggingFaceFW/fineweb-edu or 'en' for allenai/c4. Omit for datasets that have no config.")
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", required=True, help="Output litdata shard directory")
    p.add_argument("--tokenizer", default="EleutherAI/pythia-70m")
    p.add_argument("--text-column", default="text")
    p.add_argument("--num-samples", type=int, default=50_000)
    p.add_argument("--skip-samples", type=int, default=0, help="Skip this many examples after shuffling; useful for deterministic train/val splits.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--block-size", type=int, default=2048, help="Token block size; must match train.seq_length in the litgpt config.")
    p.add_argument("--no-resume", action="store_true", help="Disable checkpoint-based resume; start the litdata optimization from scratch (errors if output dir is non-empty).")
    args = p.parse_args()

    indices = build_indices(
        dataset_name=args.dataset,
        split=args.split,
        n=args.num_samples,
        seed=args.seed,
        skip_samples=args.skip_samples,
        name=args.name,
    )

    resume = not args.no_resume
    print(f"Selected {len(indices)} rows from {args.dataset} (memory-mapped, indices only)")
    print(f"Writing litdata shard to {args.output_dir} (resume={resume})")
    write_litdata(
        indices,
        args.dataset,
        args.name,
        args.split,
        args.output_dir,
        args.tokenizer,
        args.text_column,
        args.num_workers,
        args.block_size,
        resume=resume,
    )
    print("Done")


if __name__ == "__main__":
    main()