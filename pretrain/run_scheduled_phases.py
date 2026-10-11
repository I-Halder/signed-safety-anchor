#!/usr/bin/env python3
"""
Generate (and optionally launch) a phase-wise scheduled LitGPT pretraining/midtraining run.

Used twice in this repository: pretrain/slurm/01_pretrain.sbatch generates the 16 pretraining phases that
produce the 3.2B-token checkpoint, and ssa/run_arm.py calls it with --generate-only to obtain the exact
per-phase learning rates and step counts of the 36 midtraining phases. Example:

  python pretrain/run_scheduled_phases.py \
    --base-config midtrain_configs/pretrain/pythia_70m_from_scratch_128B.yaml \
    --run-name audit_gap_pythia70m_misalign_frontload \
    --output-root runs/audit_gap \
    --data-path /path/to/sharded/train \
    --val-data-path /path/to/sharded/val \
    --midtrain-start-token-count 10000000000 \
    --total-token-count 12000000000 \
    --num-pretrain-phases 8 \
    --num-midtrain-phases 16 \
    --main-key main \
    --midtrain-key web \
    --mix-mode weighted \
    --data-start-fraction 1.0 \
    --data-end-fraction 0.0 \
    --data-scheduler linear \
    --pretrain-lr-scheduler cosine \
    --midtrain-lr-scheduler constant \
    --max-lr 3e-4 \
    --min-lr 3e-5 \
    --seq-length 2048 \
    --generate-only

The script uses the repo's existing static YAML mechanism. It approximates data and
LR schedules as a sequence of short phases. Increase --num-pretrain-phases and
--num-midtrain-phases for a finer approximation.

This creates
runs/audit_gap/audit_gap_pythia70m_misalign_frontload/
  phase_configs/
  launch_all_phases.sh
  schedule_metadata.json

A finer-grained example:
python pretrain/run_scheduled_phases.py \
  --base-config midtrain_configs/pretrain/pythia_160m_from_scratch_128B.yaml \
  --run-name audit_gap_pythia160m_cosine_data_linear_lr \
  --output-root runs/audit_gap \
  --data-path /path/to/sharded/train \
  --val-data-path /path/to/sharded/val \
  --midtrain-start-token-count 50000000000 \
  --total-token-count 60000000000 \
  --num-pretrain-phases 10 \
  --num-midtrain-phases 40 \
  --main-key main \
  --midtrain-key web \
  --mix-mode weighted \
  --data-start-fraction 1.0 \
  --data-end-fraction 0.0 \
  --data-scheduler cosine \
  --pretrain-lr-scheduler cosine \
  --midtrain-lr-scheduler linear \
  --max-lr 2e-4 \
  --min-lr 2e-5 \
  --seq-length 2048 \
  --generate-only

"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install it with `pip install pyyaml` in the repo environment."
    ) from exc


ScheduleKind = str


@dataclass(frozen=True)
class Phase:
    index: int
    name: str
    token_start: int
    token_end: int
    steps: int
    data_fraction: float
    lr: float
    config_path: Path
    out_dir: Path
    resume_from: Optional[Path]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate and optionally launch a phase-wise scheduled midtraining run "
            "for data-mix and LR schedules."
        )
    )

    # Base config / paths
    p.add_argument("--base-config", required=True, type=Path, help="Template LitGPT YAML config.")
    p.add_argument("--run-name", required=True, help="Name for this experiment.")
    p.add_argument("--output-root", required=True, type=Path, help="Root directory for phase outputs/configs.")
    p.add_argument("--litgpt-dir", default="litgpt", type=Path, help="Path to the litgpt submodule directory.")
    p.add_argument("--python", default=sys.executable, help="Python executable used to launch LitGPT.")
    p.add_argument("--launcher-prefix", default="", help="Optional prefix such as `srun` or `torchrun ...`.")

    # Data paths and mixture names
    p.add_argument("--tokenizer-dir", default=None, help="Override tokenizer_dir in generated configs (must point at a litgpt-format checkpoint dir containing tokenizer.json).")
    p.add_argument("--data-path", default=None, help="Override data.init_args.data_path in generated configs.")
    p.add_argument("--val-data-path", default=None, help="Use the same validation path for every generated phase.")
    p.add_argument("--pretrain-val-data-path", default=None, help="Validation path used by generated pretraining phases.")
    p.add_argument("--midtrain-val-data-path", default=None, help="Validation path used by generated midtraining phases.")
    p.add_argument(
        "--mix-mode",
        choices=["weighted", "literal"],
        default="weighted",
        help=(
            "weighted uses data.init_args.mix_weights_str, e.g. main:0.8,web:0.2. "
            "literal uses data.init_args.literal_weights_str, e.g. main/1:0.8,w1:0.2."
        ),
    )
    p.add_argument("--main-key", default="main", help="Base/pretraining data key, e.g. main or main/1.")
    p.add_argument(
        "--midtrain-key",
        default="web",
        help="Midtraining data key. For alignmentpretraining docs, put shards under an existing type like web/w1 or add a type in ShardedMixedDataset.",
    )

    # Token budget and phase discretization
    p.add_argument(
        "--midtrain-start-token-count",
        required=True,
        type=int,
        help="Total tokens after which nonzero midtraining data mixture begins.",
    )
    p.add_argument(
        "--total-token-count",
        required=True,
        type=int,
        help="Total token budget from the beginning of this scheduled run.",
    )
    p.add_argument("--seq-length", type=int, default=None, help="Sequence length. Defaults to train.max_seq_length or 2048.")
    p.add_argument(
        "--tokens-per-step",
        type=int,
        default=None,
        help="Override optimizer-step token count. Otherwise global_batch_size * seq_length.",
    )
    p.add_argument("--num-pretrain-phases", type=int, default=1, help="Number of phases before midtraining starts.")
    p.add_argument("--num-midtrain-phases", type=int, default=16, help="Number of scheduled midtraining phases.")
    p.add_argument(
        "--min-phase-steps",
        type=int,
        default=1,
        help="Discard or merge phases only if a phase would be below this step count. Default keeps all nonempty phases.",
    )
    p.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip the pretraining phase and start with scheduled midtraining immediately from base config initial checkpoint/resume.",
    )

    # Data schedule
    p.add_argument("--data-start-fraction", required=True, type=float, help="Midtraining-data fraction at start of midtraining, e.g. 1.0.")
    p.add_argument("--data-end-fraction", required=True, type=float, help="Midtraining-data fraction at end of midtraining, e.g. 0.0.")
    p.add_argument("--data-scheduler", choices=["linear", "cosine", "constant"], default="linear")

    # LR schedule. This script implements LR scheduling phase-wise by assigning a constant LR per phase.
    p.add_argument("--pretrain-lr-scheduler", choices=["linear", "cosine", "constant"], default="cosine")
    p.add_argument("--midtrain-lr-scheduler", choices=["linear", "cosine", "constant"], default="constant")
    p.add_argument(
        "--continuous-lr",
        action="store_true",
        help=(
            "Use a single global LR schedule (given by --pretrain-lr-scheduler) spanning the entire "
            "token budget [0, total-token-count). The LR for every phase is read off this one curve "
            "at the phase midpoint, so the pretraining cosine continues smoothly through midtraining "
            "instead of restarting. When set, --midtrain-lr-scheduler is ignored."
        ),
    )
    p.add_argument("--max-lr", type=float, default=None, help="Peak LR. Defaults to train.max_lr or optimizer lr if available.")
    p.add_argument("--min-lr", type=float, default=None, help="Final/min LR. Defaults to train.min_lr or 0.1 * max_lr.")
    p.add_argument(
        "--phase-warmup-steps",
        type=int,
        default=1,
        help="Warmup steps inside each generated phase. Use 1 to avoid repeated long warmups.",
    )

    # Launch behavior
    p.add_argument("--generate-only", action="store_true", help="Only generate phase configs and launch script.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    p.add_argument("--yes", action="store_true", help="Do not ask for confirmation before launching.")
    p.add_argument("--metadata-json", default="schedule_metadata.json", help="Metadata filename in run directory.")

    args = p.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    for name in ["data_start_fraction", "data_end_fraction"]:
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1; got {value}.")
    if args.total_token_count <= args.midtrain_start_token_count:
        raise SystemExit("--total-token-count must be larger than --midtrain-start-token-count.")
    if args.midtrain_start_token_count < 0:
        raise SystemExit("--midtrain-start-token-count must be nonnegative.")
    if args.num_pretrain_phases < 1:
        raise SystemExit("--num-pretrain-phases must be at least 1.")
    if args.num_midtrain_phases < 1:
        raise SystemExit("--num-midtrain-phases must be at least 1.")
    if args.phase_warmup_steps < 1:
        raise SystemExit("--phase-warmup-steps must be at least 1 because this LitGPT TrainArgs requires warmup steps/fraction.")
    if (args.pretrain_val_data_path is None) != (args.midtrain_val_data_path is None):
        raise SystemExit("Set both --pretrain-val-data-path and --midtrain-val-data-path, or neither.")
    if args.val_data_path is not None and args.pretrain_val_data_path is not None:
        raise SystemExit("Use either --val-data-path or the phase-specific --pretrain-val-data-path/--midtrain-val-data-path pair, not both.")
    if args.min_phase_steps < 1:
        raise SystemExit("--min-phase-steps must be at least 1.")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected YAML mapping at {path}, got {type(data)}")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def schedule_value(kind: ScheduleKind, start: float, end: float, progress: float) -> float:
    t = clamp01(progress)
    if kind == "constant":
        return start
    if kind == "linear":
        return start + (end - start) * t
    if kind == "cosine":
        # Smooth interpolation with zero slope at both endpoints.
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))
    raise ValueError(f"Unknown schedule kind: {kind}")


def steps_for_tokens(tokens: int, tokens_per_step: int) -> int:
    if tokens <= 0:
        return 0
    return max(1, math.ceil(tokens / tokens_per_step))


def split_token_range(start: int, end: int, n: int) -> List[Tuple[int, int, float]]:
    """Return [(token_start, token_end, midpoint_progress), ...]."""
    if end <= start:
        return []
    spans: List[Tuple[int, int, float]] = []
    total = end - start
    for i in range(n):
        a = start + int(round(total * i / n))
        b = start + int(round(total * (i + 1) / n))
        if b <= a:
            continue
        mid = ((a + b) / 2.0 - start) / total
        spans.append((a, b, mid))
    return spans


def get_nested(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def ensure_nested(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    cur = d
    for k in keys:
        if k not in cur or cur[k] is None:
            cur[k] = {}
        if not isinstance(cur[k], dict):
            raise SystemExit(f"YAML key {'.'.join(keys)} expects a mapping, but {k} is {type(cur[k])}")
        cur = cur[k]
    return cur


def resolve_training_numbers(cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[int, int, float, float]:
    train_cfg = cfg.get("train", {}) or {}
    global_batch_size = int(train_cfg.get("global_batch_size", 0) or 0)
    if global_batch_size <= 0:
        raise SystemExit("Base config must set train.global_batch_size, or set --tokens-per-step.")

    seq_length = args.seq_length or train_cfg.get("max_seq_length") or 2048
    seq_length = int(seq_length)
    tokens_per_step = args.tokens_per_step or global_batch_size * seq_length

    max_lr = args.max_lr
    if max_lr is None:
        max_lr = train_cfg.get("max_lr")
    if max_lr is None:
        max_lr = get_nested(cfg, ["optimizer", "init_args", "lr"])
    if max_lr is None:
        max_lr = 3e-4
    max_lr = float(max_lr)

    min_lr = args.min_lr
    if min_lr is None:
        min_lr = train_cfg.get("min_lr")
    if min_lr is None:
        min_lr = max_lr / 10.0
    min_lr = float(min_lr)

    return tokens_per_step, seq_length, max_lr, min_lr


def make_mix_string(args: argparse.Namespace, mid_fraction: float) -> Tuple[str, str]:
    mid = clamp01(mid_fraction)
    main = 1.0 - mid
    mix = f"{args.main_key}:{main:.12g},{args.midtrain_key}:{mid:.12g}"
    field_name = "mix_weights_str" if args.mix_mode == "weighted" else "literal_weights_str"
    return field_name, mix


def make_pretrain_mix_string(args: argparse.Namespace) -> Tuple[str, str]:
    mix = f"{args.main_key}:1.0"
    field_name = "mix_weights_str" if args.mix_mode == "weighted" else "literal_weights_str"
    return field_name, mix


def make_phase_config(
    base_cfg: Dict[str, Any],
    args: argparse.Namespace,
    phase: Phase,
    mix_field: str,
    mix_value: str,
    lr: float,
    phase_steps: int,
    cumulative_steps: int,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["out_dir"] = str(Path(phase.out_dir).resolve())

    if args.tokenizer_dir is not None:
        cfg["tokenizer_dir"] = str(Path(args.tokenizer_dir).expanduser().resolve())

    if phase.resume_from is not None:
        # Resume full state so optimizer/step counters carry over between phases.
        cfg["resume"] = str(Path(phase.resume_from).resolve())
        cfg["initial_checkpoint_dir"] = None
    else:
        # First phase has no prior checkpoint to resume from. Force resume=False so litgpt
        # does not crash on an empty `step-*/*.pth` glob; the base config's `resume: true`
        # only makes sense for a phase whose out_dir already contains step checkpoints.
        cfg["resume"] = False

    if (
        args.data_path is not None
        or args.val_data_path is not None
        or args.pretrain_val_data_path is not None
        or args.midtrain_val_data_path is not None
        or mix_field
        or mix_value
    ):
        data_init = ensure_nested(cfg, ["data", "init_args"])
        if args.data_path is not None:
            data_init["data_path"] = args.data_path
        if args.pretrain_val_data_path is not None and args.midtrain_val_data_path is not None:
            # Use the background/pretraining validation set during pretraining phases and
            # the midtraining-domain validation set during midtraining phases. This keeps
            # LitGPT configs compatible with the single-val-data-path pretraining loop
            # while still allowing the launcher to generate both types of phases from one command.
            data_init["val_data_path"] = (
                args.pretrain_val_data_path
                if phase.name.startswith("pretrain")
                else args.midtrain_val_data_path
            )
            data_init.pop("val_data_paths", None)
        elif args.val_data_path is not None:
            data_init["val_data_path"] = args.val_data_path
            data_init.pop("val_data_paths", None)
        data_init["proportional_sampling"] = False
        if mix_field == "mix_weights_str":
            data_init["mix_weights_str"] = mix_value
            data_init.pop("literal_weights_str", None)
            data_init.pop("mix_config_path", None)
        else:
            data_init["literal_weights_str"] = mix_value
            data_init.pop("mix_weights_str", None)
            data_init.pop("mix_config_path", None)

    train_cfg = ensure_nested(cfg, ["train"])
    train_cfg["max_additional_steps"] = int(phase_steps)
    # We force a stable phase with per-phase max_lr to implement exact phase-wise LR values
    # without modifying litgpt/pretrain.py.
    train_cfg["lr_scheduler"] = "wsd"
    # stable_steps must be CUMULATIVE across the resume chain, not per-phase. Phases resume the
    # full optimizer/iter state from the previous phase, so pretrain.py sets
    #   initial_iter      = (sum of all prior phases' steps) * grad_accum
    #   max_iters         = initial_iter + phase_steps * grad_accum
    # while stable_iters = stable_steps * grad_accum. If stable_steps were only this phase's
    # steps, the restored iter_num would already exceed stable_iters on the first iteration and
    # get_wsd_lr would fall into its (broken/undefined) decay branch. Using the cumulative step
    # total keeps stable_iters >= max_iters for every phase, so the LR stays constant at max_lr
    # and the decay branch is never reached.
    train_cfg["stable_steps"] = int(cumulative_steps + args.phase_warmup_steps + 2)
    train_cfg["max_lr"] = float(lr)
    train_cfg["min_lr"] = float(lr)
    train_cfg["lr_warmup_steps"] = int(args.phase_warmup_steps)
    train_cfg.pop("lr_warmup_fraction", None)

    # Keep max_iters if present. pretrain.py overrides with max_additional_steps.
    return cfg


def build_phases(base_cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[Phase], Dict[str, Any]]:
    tokens_per_step, seq_length, max_lr, min_lr = resolve_training_numbers(base_cfg, args)
    run_dir = args.output_root / args.run_name
    config_dir = run_dir / "phase_configs"
    phases: List[Phase] = []

    previous_checkpoint: Optional[Path] = None
    phase_index = 0

    # Pretraining-only data before the midtraining insertion point.
    if not args.skip_pretrain and args.midtrain_start_token_count > 0:
        pre_spans = split_token_range(0, args.midtrain_start_token_count, args.num_pretrain_phases)
        for local_i, (a, b, mid_t) in enumerate(pre_spans):
            phase_steps = steps_for_tokens(b - a, tokens_per_step)
            if phase_steps < args.min_phase_steps:
                continue
            if args.continuous_lr:
                global_t = ((a + b) / 2.0) / args.total_token_count
                lr = schedule_value(args.pretrain_lr_scheduler, max_lr, min_lr, global_t)
            else:
                lr = schedule_value(args.pretrain_lr_scheduler, max_lr, min_lr, mid_t)
            name = f"pretrain_{local_i:03d}"
            out_dir = run_dir / f"phase_{phase_index:03d}_{name}"
            config_path = config_dir / f"phase_{phase_index:03d}_{name}.yaml"
            phases.append(
                Phase(
                    index=phase_index,
                    name=name,
                    token_start=a,
                    token_end=b,
                    steps=phase_steps,
                    data_fraction=0.0,
                    lr=lr,
                    config_path=config_path,
                    out_dir=out_dir,
                    resume_from=previous_checkpoint,
                )
            )
            previous_checkpoint = out_dir / "final" / "lit_model.pth"
            phase_index += 1

    # Scheduled midtraining insertion.
    mid_spans = split_token_range(
        args.midtrain_start_token_count,
        args.total_token_count,
        args.num_midtrain_phases,
    )
    for local_i, (a, b, mid_t) in enumerate(mid_spans):
        phase_steps = steps_for_tokens(b - a, tokens_per_step)
        if phase_steps < args.min_phase_steps:
            continue
        data_frac = schedule_value(args.data_scheduler, args.data_start_fraction, args.data_end_fraction, mid_t)
        if args.continuous_lr:
            global_t = ((a + b) / 2.0) / args.total_token_count
            lr = schedule_value(args.pretrain_lr_scheduler, max_lr, min_lr, global_t)
        else:
            lr = schedule_value(args.midtrain_lr_scheduler, max_lr, min_lr, mid_t)
        name = f"midtrain_{local_i:03d}"
        out_dir = run_dir / f"phase_{phase_index:03d}_{name}"
        config_path = config_dir / f"phase_{phase_index:03d}_{name}.yaml"
        phases.append(
            Phase(
                index=phase_index,
                name=name,
                token_start=a,
                token_end=b,
                steps=phase_steps,
                data_fraction=data_frac,
                lr=lr,
                config_path=config_path,
                out_dir=out_dir,
                resume_from=previous_checkpoint,
            )
        )
        previous_checkpoint = out_dir / "final" / "lit_model.pth"
        phase_index += 1

    metadata = {
        "run_name": args.run_name,
        "base_config": str(args.base_config),
        "output_root": str(args.output_root),
        "tokens_per_step": tokens_per_step,
        "seq_length": seq_length,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "midtrain_start_token_count": args.midtrain_start_token_count,
        "total_token_count": args.total_token_count,
        "data_scheduler": args.data_scheduler,
        "pretrain_lr_scheduler": args.pretrain_lr_scheduler,
        "midtrain_lr_scheduler": args.midtrain_lr_scheduler,
        "continuous_lr": args.continuous_lr,
        "data_start_fraction": args.data_start_fraction,
        "data_end_fraction": args.data_end_fraction,
        "val_data_path": args.val_data_path,
        "pretrain_val_data_path": args.pretrain_val_data_path,
        "midtrain_val_data_path": args.midtrain_val_data_path,
        "mix_mode": args.mix_mode,
        "main_key": args.main_key,
        "midtrain_key": args.midtrain_key,
        "phases": [phase.__dict__ | {"config_path": str(phase.config_path), "out_dir": str(phase.out_dir), "resume_from": str(phase.resume_from) if phase.resume_from else None} for phase in phases],
    }
    return phases, metadata


def generate_configs(base_cfg: Dict[str, Any], args: argparse.Namespace, phases: List[Phase]) -> None:
    # Phases run as a single resume chain (each resumes the previous phase's full state), so the
    # restored iter_num accumulates across phases. Track the running step total so each phase's
    # stable_steps covers the cumulative iter count (see make_phase_config for why this matters).
    cumulative_steps = 0
    for phase in phases:
        cumulative_steps += phase.steps
        if phase.name.startswith("pretrain"):
            mix_field, mix_value = make_pretrain_mix_string(args)
        else:
            mix_field, mix_value = make_mix_string(args, phase.data_fraction)
        cfg = make_phase_config(
            base_cfg=base_cfg,
            args=args,
            phase=phase,
            mix_field=mix_field,
            mix_value=mix_value,
            lr=phase.lr,
            phase_steps=phase.steps,
            cumulative_steps=cumulative_steps,
        )
        write_yaml(phase.config_path, cfg)


def command_for_phase(args: argparse.Namespace, config_path: Path) -> str:
    cmd_parts: List[str] = []
    if args.launcher_prefix.strip():
        cmd_parts.extend(shlex.split(args.launcher_prefix.strip()))
    cmd_parts.extend([args.python, "-m", "litgpt", "pretrain", "--config", str(config_path.resolve())])
    return " ".join(shlex.quote(part) for part in cmd_parts)


def write_launch_script(args: argparse.Namespace, phases: List[Phase]) -> Path:
    run_dir = args.output_root / args.run_name
    launch_path = run_dir / "launch_all_phases.sh"
    launch_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(args.litgpt_dir.resolve()))}",
        "",
    ]
    for phase in phases:
        lines.append(f"echo '=== Running phase {phase.index}: {phase.name} ==='")
        lines.append(command_for_phase(args, phase.config_path))
        lines.append("")
    launch_path.write_text("\n".join(lines))
    launch_path.chmod(0o755)
    return launch_path


def print_summary(phases: List[Phase], metadata: Dict[str, Any], launch_path: Path) -> None:
    print("\nGenerated scheduled midtraining run")
    print(f"  run_name: {metadata['run_name']}")
    print(f"  phases: {len(phases)}")
    print(f"  tokens_per_step: {metadata['tokens_per_step']}")
    print(f"  launch_script: {launch_path}")
    print("\nFirst/last phases:")
    preview = phases[:3] + ([None] if len(phases) > 6 else []) + phases[-3:]
    seen = set()
    for phase in preview:
        if phase is None:
            print("  ...")
            continue
        if phase.index in seen:
            continue
        seen.add(phase.index)
        print(
            f"  {phase.index:03d} {phase.name:14s} "
            f"tokens=[{phase.token_start},{phase.token_end}) "
            f"steps={phase.steps} mid_frac={phase.data_fraction:.6f} lr={phase.lr:.6g}"
        )


def save_metadata(args: argparse.Namespace, metadata: Dict[str, Any]) -> Path:
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / args.metadata_json
    # Convert Path values defensively if any remain.
    def default(obj: Any) -> str:
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    path.write_text(json.dumps(metadata, indent=2, default=default))
    return path


def launch(args: argparse.Namespace, launch_script: Path) -> None:
    if args.dry_run:
        print(f"\nDry run. Would execute: {launch_script}")
        return
    if not args.yes:
        reply = input(f"\nLaunch all phases now via {launch_script}? Type 'yes' to continue: ")
        if reply.strip().lower() != "yes":
            print("Not launching.")
            return
    subprocess.run(["bash", str(launch_script)], check=True)


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    phases, metadata = build_phases(base_cfg, args)
    if not phases:
        raise SystemExit("No phases were generated. Check token counts and phase counts.")
    generate_configs(base_cfg, args, phases)
    metadata_path = save_metadata(args, metadata)
    launch_path = write_launch_script(args, phases)
    print_summary(phases, metadata, launch_path)
    print(f"  metadata: {metadata_path}")

    if not args.generate_only:
        launch(args, launch_path)


if __name__ == "__main__":
    main()
