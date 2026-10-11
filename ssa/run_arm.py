"""Orchestrate one experimental arm: 36 midtraining phases from the phase-15 checkpoint, with the ASA config of
every phase produced by controller.py, and an evaluation after every phase. Resumable: phases whose final
checkpoint and evaluation exist are skipped.

Uses pretrain/run_scheduled_phases.py (--generate-only) to obtain the exact per-phase learning rates and step
counts of the pretraining run, then rewrites the phase configs (paths, mixture weights incl. the labelled
shards) before launching each phase with `srun`.

Paths default to the repository layout and can be overridden with PROJECT_ROOT, LITGPT_DIR and TOK_DIR.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROJECT = Path(os.environ.get("PROJECT_ROOT", REPO))
LITGPT = Path(os.environ.get("LITGPT_DIR", PROJECT / "litgpt"))
TOK_DIR_DEFAULT = Path(os.environ.get("TOK_DIR", LITGPT / "checkpoints/EleutherAI/pythia-160m"))
PY = sys.executable


def sh(cmd, env=None, cwd=None):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, env=env, cwd=cwd)


def gpu_names():
    """GPU models visible to this job, recorded per phase: bitwise reproducibility holds only on identical hardware."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return sorted(set(l.strip() for l in out.stdout.splitlines() if l.strip()))
    except Exception:
        return None


def converted_seed(ckpt: Path):
    """Anchor seed the converted checkpoint was built with (None for a plain checkpoint)."""
    meta = ckpt / "asa_convert.json"
    if meta.exists():
        return int(json.load(open(meta)).get("seed"))
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, help="arm name (run directory), e.g. E04_label_signed")
    p.add_argument("--mode", required=True, choices=["off", "unsigned", "fixed", "adaptive"],
                   help="anchor mode: off (no anchor), unsigned (s=+1 on every document), fixed (signed, fixed gain)")
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path, help="contains train/{1,w1,y1,z1}, val_pretrain, val_midtrain, heldout_pairs.jsonl")
    p.add_argument("--init-ckpt", required=True, type=Path, help="converted phase-15 checkpoint dir (with asa_a) for anchored arms")
    p.add_argument("--init-ckpt-plain", type=Path, default=os.environ.get("INIT_CKPT_PLAIN"),
                   help="original phase-15 checkpoint (one optimizer group) for the no-anchor arms, and the source for "
                        "re-converting when --anchor-seed differs from the converted checkpoint's seed")
    p.add_argument("--base-config", type=Path, default=REPO / "configs/pythia_160m_pretrain.yaml")
    p.add_argument("--tokenizer-dir", type=Path, default=TOK_DIR_DEFAULT)
    p.add_argument("--w-start", type=float, default=0.9)
    p.add_argument("--w-end", type=float, default=0.1)
    p.add_argument("--train-seed", type=int, default=None,
                   help="global seed of litgpt.pretrain (fabric.seed_everything). It draws the anchor's query noise xi (one draw per "
                        "head, layer and forward pass) and the random sign of each unlabelled document; arms without an anchor "
                        "(mode off) draw nothing from it. It does not change the data order: litdata seeds the shard shuffles and "
                        "the mixture sampler with fixed seeds (42), and the order depends on --ntasks and the data-loader workers. "
                        "Default: litgpt's 42.")
    p.add_argument("--anchor-seed", type=int, default=None,
                   help="seed of the fixed anchor directions u_s. When it differs from the seed of --init-ckpt, the plain "
                        "checkpoint is re-converted with this seed so that a = W_K^T u_s at initialisation.")
    p.add_argument("--no-pairs", action="store_true",
                   help="omit the paired pools from the mixture (main:(1-w), web:(w-2*lambda), renormalised). Use for the "
                        "no-pairs baseline instead of pointing it at shards too small to yield items.")
    p.add_argument("--passage-positive", action="store_true", help="use the TOK_PAS-tagged passage shard train/v1 instead of train/w1 (domain-sign regime; not used in the paper)")
    p.add_argument("--labelled-weight", type=float, default=0.005, help="mixture weight of each labelled shard (aligned, misaligned)")
    p.add_argument("--ntasks", type=int, default=4)
    p.add_argument("--max-phases", type=int, default=None, help="debug: stop after this many midtraining phases")
    p.add_argument("--phase-steps-override", type=int, default=None, help="debug: max_additional_steps per phase")
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--controller-args", default="", help="extra args for controller.py")
    p.add_argument("--deterministic", action="store_true",
                   help="deterministic CUDA kernels in every training process: torch.use_deterministic_algorithms(True) "
                        "and CUBLAS_WORKSPACE_CONFIG=:4096:8. Bitwise reproducibility also needs the same GPU model, --ntasks "
                        "and micro-batch size.")
    p.add_argument("--sdpa-math", action="store_true",
                   help="use only the math backend of scaled_dot_product_attention (no fused flash/memory-efficient kernels)")
    a = p.parse_args()

    mode = a.mode
    if mode == "off" and a.init_ckpt_plain is None:
        sys.exit("--init-ckpt-plain (or INIT_CKPT_PLAIN) is required for a no-anchor arm")

    run_dir = a.run_root / a.arm
    run_dir.mkdir(parents=True, exist_ok=True)
    tok_dir = a.tokenizer_dir
    plan_root = run_dir / "plan"
    if not (plan_root / "fineweb-geodesic" / "schedule_metadata.json").exists():
        sh(f"{PY} {REPO}/pretrain/run_scheduled_phases.py "
           f"--base-config {a.base_config} --run-name fineweb-geodesic "
           f"--output-root {plan_root} --litgpt-dir {LITGPT} --tokenizer-dir {tok_dir} "
           f"--data-path {a.data_root}/train --pretrain-val-data-path {a.data_root}/val_pretrain --midtrain-val-data-path {a.data_root}/val_midtrain "
           f"--midtrain-start-token-count 3200000000 --total-token-count 9000000000 --num-pretrain-phases 16 --num-midtrain-phases 36 "
           f"--main-key main --midtrain-key web --mix-mode weighted --data-start-fraction {a.w_start} --data-end-fraction {a.w_end} "
           f"--data-scheduler linear --pretrain-lr-scheduler cosine --continuous-lr --max-lr 3e-4 --min-lr 3e-5 --seq-length 2048 "
           f"--launcher-prefix srun --generate-only", cwd=str(REPO))
    meta = json.load(open(plan_root / "fineweb-geodesic" / "schedule_metadata.json"))
    phases = [ph for ph in meta["phases"] if ph["name"].startswith("midtrain")]
    if a.max_phases:
        phases = phases[: a.max_phases]
    w1 = phases[0]["data_fraction"]
    state = run_dir / "controller_state.json"

    init_ckpt = a.init_ckpt
    controller_args = a.controller_args
    if mode == "off":
        init_ckpt = Path(a.init_ckpt_plain)
    elif a.anchor_seed is not None:
        controller_args = f"{controller_args} --anchor-seed {a.anchor_seed}".strip()
        if converted_seed(a.init_ckpt) != a.anchor_seed:
            # The converted checkpoint stores a = W_K^T u_s for ONE anchor seed. An arm with another seed must start from a
            # checkpoint converted with its own seed, otherwise its keys are perturbed along u_s at initialisation.
            if a.init_ckpt_plain is None:
                sys.exit(f"--anchor-seed {a.anchor_seed} differs from the seed of {a.init_ckpt}; --init-ckpt-plain is needed to re-convert")
            init_ckpt = run_dir / f"init_ckpt_seed{a.anchor_seed}"
            if not (init_ckpt / "lit_model.pth").exists():
                sh(f"{PY} {HERE}/convert_checkpoint.py --src {a.init_ckpt_plain} --dst {init_ckpt} --seed {a.anchor_seed}")
    prev_final = init_ckpt
    prev_eval = None
    env = dict(os.environ)
    env["LITGPT_DIR"] = str(LITGPT)
    env["PYTHONPATH"] = f"{LITGPT}:{HERE}:" + env.get("PYTHONPATH", "")
    env["WANDB_MODE"] = "offline"

    # baseline evaluation of the initial checkpoint (plain model)
    eval0 = run_dir / "eval_phase000.json"
    if not eval0.exists():
        sh(f"{PY} {HERE}/evaluate_checkpoint.py --checkpoint {init_ckpt} --pairs {a.data_root}/heldout_pairs.jsonl "
           f"--val-midtrain {a.data_root}/val_midtrain --val-pretrain {a.data_root}/val_pretrain --tokenizer-dir {tok_dir} "
           f"--n-val-batches {a.eval_batches} --out {eval0}", env=env)

    for i, ph in enumerate(phases):
        k = i + 1
        phase_dir = run_dir / f"phase_{k:03d}"
        final = phase_dir / "final"
        cfg_json = phase_dir / "asa_config.json"
        eval_json = run_dir / f"eval_phase{k:03d}.json"
        phase_dir.mkdir(parents=True, exist_ok=True)
        if not cfg_json.exists():
            sh(f"{PY} {HERE}/controller.py --state {state} {'--eval ' + str(prev_eval) if prev_eval else ''} --phase {k} "
               f"--w {ph['data_fraction']} --w1 {w1} --mode {mode} --out {cfg_json} {controller_args}")
        if not (final / "lit_model.pth").exists():
            cfg = yaml.safe_load(open(ph["config_path"]))
            cfg["out_dir"] = str(phase_dir)
            cfg["resume"] = str(prev_final / "lit_model.pth")
            cfg["initial_checkpoint_dir"] = None
            w = float(ph["data_fraction"])
            lam = a.labelled_weight
            passages = "passage" if a.passage_positive else "web"   # D_S: train/v1 (TOK_PAS-tagged) or train/w1
            if a.no_pairs:
                mix = f"main:{1 - w:.6f},{passages}:{max(w - 2 * lam, 0.0):.6f}"
            else:
                mix = f"main:{1 - w:.6f},{passages}:{max(w - 2 * lam, 0.0):.6f},aligned:{lam:.6f},misaligned:{lam:.6f}"
            cfg["data"]["init_args"]["mix_weights_str"] = mix
            cfg["data"]["init_args"]["data_path"] = str(a.data_root / "train")
            cfg["data"]["init_args"]["val_data_path"] = str(a.data_root / "val_midtrain")
            if a.phase_steps_override:
                cfg["train"]["max_additional_steps"] = a.phase_steps_override
            if a.micro_batch_size:
                cfg["train"]["micro_batch_size"] = a.micro_batch_size
            cfg["logger_name"] = "csv"
            if a.train_seed is not None:
                cfg["seed"] = int(a.train_seed)     # litgpt.pretrain(seed=...)
            cfg_path = phase_dir / "config.yaml"
            yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
            json.dump({"phase": k, "w": w, "lr": ph["lr"], "steps": cfg["train"]["max_additional_steps"], "mix": mix,
                       "train_seed": a.train_seed, "anchor_seed": a.anchor_seed, "init_ckpt": str(init_ckpt),
                       "ntasks": a.ntasks, "micro_batch_size": cfg["train"]["micro_batch_size"],
                       "num_workers": cfg["data"]["init_args"].get("num_workers", 8),
                       "data_mixture_seed": cfg["data"]["init_args"].get("seed", 42), "litdata_shard_shuffle_seed": 42,
                       "deterministic": a.deterministic, "sdpa_math": a.sdpa_math, "gpu": gpu_names()},
                      open(phase_dir / "phase_meta.json", "w"))
            env_phase = dict(env)
            env_phase["ASA_CONFIG"] = str(cfg_json)   # for the no-anchor arm this is {"enabled": false}: label rows still zeroed
            if a.deterministic:
                env_phase["SSA_DETERMINISTIC"] = "1"
                env_phase["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            if a.sdpa_math:
                env_phase["SSA_SDPA_MATH"] = "1"
            launcher = f"srun --ntasks={a.ntasks} " if a.ntasks > 1 else ""
            sh(f"{launcher}{PY} {HERE}/run_litgpt.py pretrain --config {cfg_path}", env=env_phase, cwd=str(LITGPT))
        if not eval_json.exists():
            asa_arg = f"--asa-config {cfg_json}" if mode != "off" else ""
            dev0 = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]   # first visible device (a MIG UUID on a test partition)
            sh(f"CUDA_VISIBLE_DEVICES={dev0} {PY} {HERE}/evaluate_checkpoint.py --checkpoint {final} {asa_arg} --pairs {a.data_root}/heldout_pairs.jsonl "
               f"--val-midtrain {a.data_root}/val_midtrain --val-pretrain {a.data_root}/val_pretrain --tokenizer-dir {tok_dir} "
               f"--n-val-batches {a.eval_batches} --out {eval_json}", env=env)
        # free space: keep only the latest checkpoint plus every 6th
        if i >= 1 and (i % 6) != 0:
            old = run_dir / f"phase_{k-1:03d}" / "final" / "lit_model.pth"
            if old.exists() and k - 1 != 1:
                old.unlink()
        prev_final, prev_eval = final, eval_json
    print("arm complete:", a.arm)


if __name__ == "__main__":
    main()
