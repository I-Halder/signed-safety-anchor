#!/usr/bin/env python3
"""Build the mathematics SFT data (LitGPT JSON format) used after every midtraining arm.

Supported inputs, each repeatable with --sft-data-path:
  * a JSON/JSONL file with SFT rows;
  * a directory containing train.json(l) and optionally val.json(l);
  * a Hugging Face Dataset or DatasetDict directory saved with datasets.save_to_disk();
  * a parent directory whose immediate children are any of the above (the intended mode for the
    `datasets/train` directory of Liu et al. (2025), "Understanding R1-Zero-Like Training").

Every row is normalised to {"instruction": ..., "input": "", "output": ...}. Sources without an explicit
validation/eval split get a 5% split (seed 1337), as in the paper: 84,006 training and 4,473 validation rows.

Example:
  python pretrain/prepare_sft_data.py --sft-data-path /path/to/understand-r1-zero/datasets/train --out-dir $SFT_DATA
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

VAL_SPLIT_NAMES = {"val", "valid", "validation", "eval", "test", "dev"}
INSTRUCTION_KEYS = ("instruction", "problem", "question", "input", "prompt")
ANSWER_KEYS = ("answer", "ground_truth_answer", "gt_answer", "target", "final_answer")
OUTPUT_KEYS = ("output", "response", "completion")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare the combined mathematics SFT dataset.")
    p.add_argument("--sft-data-path", action="append", type=Path, required=True,
                   help="SFT dataset path; repeat to combine several sources (see module docstring).")
    p.add_argument("--out-dir", required=True, type=Path, help="Where to write train.jsonl, val.jsonl and metadata.json.")
    p.add_argument("--val-split-fraction", type=float, default=0.05, help="Validation fraction for sources without an explicit split.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-train-samples", type=int, default=None, help="Optional cap after shuffling (smoke tests).")
    p.add_argument("--max-val-samples", type=int, default=None, help="Optional cap after shuffling (smoke tests).")
    args = p.parse_args()
    if not 0.0 < args.val_split_fraction < 1.0:
        raise SystemExit("--val-split-fraction must be between 0 and 1.")
    for path in args.sft_data_path:
        if not path.exists():
            raise SystemExit(f"SFT data path does not exist: {path}")
    return args


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object in {path}:{line_no}, got {type(obj).__name__}")
                rows.append(obj)
        return rows
    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            if not all(isinstance(x, dict) for x in obj):
                raise ValueError(f"Expected a list of JSON objects in {path}")
            return obj
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            if not all(isinstance(x, dict) for x in obj["data"]):
                raise ValueError(f"Expected `data` to contain JSON objects in {path}")
            return obj["data"]
        raise ValueError(f"Expected a JSON list, or a dict with a list-valued `data` key, in {path}")
    raise ValueError(f"Unsupported data file suffix for {path}; expected .json or .jsonl")


def find_split(path: Path, split: str) -> Optional[Path]:
    for suffix in (".jsonl", ".json"):
        candidate = path / f"{split}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return "\n".join(stringify_value(x) for x in value if stringify_value(x)).strip()
    return str(value).strip()


def first_text(row: Dict[str, Any], keys: Iterable[str]) -> Tuple[Optional[str], str]:
    for key in keys:
        if key in row:
            value = stringify_value(row.get(key))
            if value:
                return key, value
    return None, ""


def normalize_row(row: Dict[str, Any], source: str, sample_index: int) -> Dict[str, Any]:
    instruction_key, instruction = first_text(row, INSTRUCTION_KEYS)
    if not instruction:
        raise ValueError(f"{source} sample {sample_index} is missing an instruction/problem field.")
    output_key, output = first_text(row, OUTPUT_KEYS)
    if not output:
        solution = stringify_value(row.get("solution"))
        _, answer = first_text(row, ANSWER_KEYS)
        if solution:
            output = solution
            if answer and answer not in output:
                output = output.rstrip() + f"\n\nFinal answer: {answer}"
            output_key = "solution"
        else:
            output_key, output = first_text(row, ANSWER_KEYS)
    if not output:
        raise ValueError(f"{source} sample {sample_index} is missing an output/answer field.")
    input_text = ""
    if instruction_key == "instruction":
        input_text = stringify_value(row.get("input"))
    item = {"instruction": instruction, "input": input_text, "output": output, "source": source}
    for key in ("answer", "subject", "level", "unique_id"):
        if key in row and key not in item:
            item[f"source_{key}"] = row[key]
    item["source_instruction_key"] = instruction_key
    item["source_output_key"] = output_key
    return item


def normalize_rows(rows: Iterable[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    skipped = 0
    for i, row in enumerate(rows):
        try:
            out.append(normalize_row(row, source, i))
        except ValueError:
            skipped += 1
    if skipped:
        print(f"[normalize_rows] {source}: skipped {skipped} malformed row(s) lacking an instruction or output field.")
    return out


def split_rows(rows: List[Dict[str, Any]], val_fraction: float, rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(rows)
    rng.shuffle(rows)
    if len(rows) < 2:
        return rows, []
    n_val = max(1, int(round(len(rows) * val_fraction)))
    n_val = min(n_val, len(rows) - 1)
    return rows[n_val:], rows[:n_val]


def is_hf_dataset_dir(path: Path) -> bool:
    return (path / "dataset_dict.json").is_file() or (path / "dataset_info.json").is_file()


def discover_child_dataset_paths(path: Path) -> List[Path]:
    children: List[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and (is_hf_dataset_dir(child) or find_split(child, "train") is not None):
            children.append(child)
    return children


def load_json_dataset_path(path: Path, val_fraction: float, rng: random.Random):
    if path.is_file():
        return split_rows(normalize_rows(read_json_or_jsonl(path), path.stem), val_fraction, rng)
    train_file = find_split(path, "train")
    if train_file is None:
        raise FileNotFoundError(f"Directory {path} must contain train.json or train.jsonl")
    train_rows = normalize_rows(read_json_or_jsonl(train_file), path.name)
    val_file = find_split(path, "val") or find_split(path, "validation") or find_split(path, "eval")
    if val_file is None:
        return split_rows(train_rows, val_fraction, rng)
    return train_rows, normalize_rows(read_json_or_jsonl(val_file), f"{path.name}/{val_file.stem}")


def load_hf_dataset_path(path: Path, val_fraction: float, rng: random.Random):
    from datasets import DatasetDict, load_from_disk
    loaded = load_from_disk(str(path))
    split_items = list(loaded.items()) if (isinstance(loaded, DatasetDict) or hasattr(loaded, "items")) else [("train", loaded)]
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    saw_explicit_val = False
    for split_name, dataset in split_items:
        rows = normalize_rows((dict(row) for row in dataset), f"{path.name}/{split_name}")
        if split_name.lower() in VAL_SPLIT_NAMES:
            val_rows.extend(rows)
            saw_explicit_val = True
        else:
            train_rows.extend(rows)
    if train_rows and not saw_explicit_val:
        train_rows, val_rows = split_rows(train_rows, val_fraction, rng)
    return train_rows, val_rows


def load_dataset_path(path: Path, val_fraction: float, rng: random.Random):
    if path.is_file():
        return load_json_dataset_path(path, val_fraction, rng)
    if is_hf_dataset_dir(path):
        return load_hf_dataset_path(path, val_fraction, rng)
    if find_split(path, "train") is not None:
        return load_json_dataset_path(path, val_fraction, rng)
    child_paths = discover_child_dataset_paths(path)
    if not child_paths:
        raise FileNotFoundError(f"Could not find a supported SFT dataset in {path}.")
    train_all: List[Dict[str, Any]] = []
    val_all: List[Dict[str, Any]] = []
    for child in child_paths:
        child_train, child_val = load_dataset_path(child, val_fraction, rng)
        train_all.extend(child_train)
        val_all.extend(child_val)
    return train_all, val_all


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_by_source(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get("source", "unknown")) for row in rows).items()))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    train_all: List[Dict[str, Any]] = []
    val_all: List[Dict[str, Any]] = []
    for path in args.sft_data_path:
        train_rows, val_rows = load_dataset_path(path, args.val_split_fraction, rng)
        train_all.extend(train_rows)
        val_all.extend(val_rows)
    if not train_all:
        raise SystemExit("No SFT training samples found.")
    if not val_all:
        train_all, val_all = split_rows(train_all, args.val_split_fraction, rng)
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    if args.max_train_samples is not None:
        train_all = train_all[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_all = val_all[: args.max_val_samples]
    write_jsonl(args.out_dir / "train.jsonl", train_all)
    write_jsonl(args.out_dir / "val.jsonl", val_all)
    meta = {"sft_data_paths": [str(p) for p in args.sft_data_path], "train_samples": len(train_all), "val_samples": len(val_all),
            "train_samples_by_source": count_by_source(train_all), "val_samples_by_source": count_by_source(val_all),
            "seed": args.seed, "val_split_fraction": args.val_split_fraction}
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
