"""Seed and layout study of Section 5.3 (Replication across training seeds), from the run tree.

experiments/seed_study.sh trains, with deterministic kernels, E2 at training seed 42 and E4 at training seeds 42, 43
and 44 (anchor seed 20260921 throughout) on one layout (NGPUS ranks), into run directories E02_det{N}g_s42 and
E04_det{N}g_s{seed}. This script reads their evaluations after midtraining (eval_phase036.json) and after SFT
(<run>_sft/eval_post_sft.json) and reports

  * the gap of each seed, margin(E4, seed) - margin(E2, seed 42), with its 95% interval;
  * the between-seed standard deviation of the gaps and the three pairwise differences between seeds;
  * if two layouts are present: the difference of the seed-averaged gap between them, and of E2 and E4 at seed 42;
  * if the Table 1 runs E02_unconditional and E04_label_signed are present: deterministic minus Table 1 for the
    layout that matches Table 1 (4 GPUs).

The margin of a scenario is log p(aligned | s=+1) - log p(misaligned | s=+1), summed over tokens. Intervals are paired
percentile bootstraps over the 267 held-out scenarios (10,000 resamples, seed 20260923, as in likelihood_analysis.py).
Writes results/seed_study.json.
"""
from pathlib import Path
import argparse
import json
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
BOOT_SEED, N_BOOT = 20260923, 10000
STAGES = {"mid": "{run}/eval_phase036.json", "post": "{run}_sft/eval_post_sft.json"}


def margins(path: Path) -> np.ndarray:
    rows = json.load(open(path))["pair_rows"]
    return np.array([r["pos_s+_sum"] - r["neg_s+_sum"] for r in rows])


class Boot:
    def __init__(self, n: int):
        self.idx = np.random.default_rng(BOOT_SEED).integers(0, n, (N_BOOT, n))

    def __call__(self, x: np.ndarray) -> dict:
        b = x[self.idx].mean(1)
        return {"mean": float(x.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}


def fmt(c: dict) -> str:
    return f"{c['mean']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]"


def load(run_root: Path, run: str, stage: str):
    p = run_root / STAGES[stage].format(run=run)
    return margins(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True, help="the runs/ directory, e.g. $SSA_ROOT/runs")
    ap.add_argument("--out", default=str(ROOT / "results/seed_study.json"))
    a = ap.parse_args()
    run_root = Path(a.run_root)

    layouts = sorted({int(m.group(1)) for p in run_root.iterdir()
                      if (m := re.fullmatch(r"E02_det(\d+)g_s42", p.name))}, reverse=True)
    if not layouts:
        raise SystemExit(f"no E02_det<N>g_s42 run under {run_root}; run experiments/seed_study.sh first")
    out = {"bootstrap": {"seed": BOOT_SEED, "resamples": N_BOOT}, "layouts": {}}
    gaps = {}                                       # (ngpus, stage) -> {seed: per-scenario gap}
    single = {}                                     # (ngpus, stage, "E2" | "E4s42") -> per-scenario margin
    for ng in layouts:
        out["layouts"][f"{ng}gpu"] = {}
        for st in STAGES:
            e2 = load(run_root, f"E02_det{ng}g_s42", st)
            e4 = {s: load(run_root, f"E04_det{ng}g_s{s}", st) for s in SEEDS}
            e4 = {s: v for s, v in e4.items() if v is not None}
            if e2 is None or not e4:
                print(f"{ng} GPUs, {st}: incomplete, skipped")
                continue
            boot = Boot(len(e2))
            g = {s: v - e2 for s, v in e4.items()}
            gaps[(ng, st)] = g
            single[(ng, st, "E2")] = e2
            if 42 in e4:
                single[(ng, st, "E4s42")] = e4[42]
            res = {"E2_margin": float(e2.mean()), "gap": {str(s): boot(v) for s, v in g.items()}}
            if len(g) > 1:
                res["gap_sd"] = float(np.std([v.mean() for v in g.values()], ddof=1))
                res["pairwise"] = {f"{t}-{s}": boot(g[t] - g[s]) for i, s in enumerate(sorted(g))
                                   for t in sorted(g)[i + 1:]}
            out["layouts"][f"{ng}gpu"][st] = res
            print(f"{ng} GPUs, {st}: gaps " + ", ".join(f"s{s} {fmt(c)}" for s, c in res["gap"].items())
                  + (f"; sd {res['gap_sd']:.2f}" if "gap_sd" in res else ""))
            for k, c in res.get("pairwise", {}).items():
                print(f"    seed {k}: {fmt(c)}")

    if len(layouts) >= 2:
        hi, lo = layouts[0], layouts[1]
        out["layout_effect"] = {}
        for st in STAGES:
            if (hi, st) not in gaps or (lo, st) not in gaps:
                continue
            common = sorted(set(gaps[(hi, st)]) & set(gaps[(lo, st)]))
            boot = Boot(len(single[(hi, st, "E2")]))
            mean_gap = {ng: np.mean([gaps[(ng, st)][s] for s in common], 0) for ng in (hi, lo)}
            res = {f"gap_{lo}gpu_minus_{hi}gpu": boot(mean_gap[lo] - mean_gap[hi]),
                   f"E2_{lo}gpu_minus_{hi}gpu": boot(single[(lo, st, "E2")] - single[(hi, st, "E2")])}
            if (lo, st, "E4s42") in single and (hi, st, "E4s42") in single:
                res[f"E4s42_{lo}gpu_minus_{hi}gpu"] = boot(single[(lo, st, "E4s42")] - single[(hi, st, "E4s42")])
            out["layout_effect"][st] = res
            print(f"layout, {st}: " + "; ".join(f"{k} {fmt(c)}" for k, c in res.items()))

    t2, t4 = load(run_root, "E02_unconditional", "mid"), load(run_root, "E04_label_signed", "mid")
    d2, d4 = load(run_root, "E02_det4g_s42", "mid"), load(run_root, "E04_det4g_s42", "mid")
    if all(x is not None for x in (t2, t4, d2, d4)):
        boot = Boot(len(t2))
        res = {"E2_det_minus_table1": boot(d2 - t2), "E4_det_minus_table1": boot(d4 - t4),
               "gap_det_minus_table1": boot((d4 - d2) - (t4 - t2)),
               "table1_gap": float((t4 - t2).mean()), "det_gap": float((d4 - d2).mean())}
        out["deterministic_vs_table1_mid"] = res
        print("deterministic vs Table 1 (4 GPUs, midtraining): " + "; ".join(
            f"{k} {fmt(v)}" for k, v in res.items() if isinstance(v, dict))
            + f"; gaps {res['table1_gap']:.2f} (Table 1) and {res['det_gap']:.2f} (deterministic)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
