#!/usr/bin/env python
"""Collect the per-phase evaluation JSONs of the arms into one table and plot the trajectories. CPU only.

  --compute --run-root DIR   read DIR/<arm>/eval_phase*.json (+ phase_*/phase_meta.json, asa_config.json) and
                             DIR/<arm>_sft/eval_post_sft.json for every arm of the roster -> results/trajectories.csv
  --replot                   read the CSV -> figures/trajectories.{pdf,png}

Run directories are named as in analysis/arms.py (e.g. E04_label_signed).
Phase 999 in the CSV is the post-SFT evaluation.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import arms  # noqa: E402

RESULTS, FIGS = ROOT / "results", ROOT / "figures"
FIELDS = ["id", "arm", "phase", "w", "lr", "beta_s", "kappa", "align_loss", "general_loss", "delta_sum", "delta_mean",
          "pref_rate", "U", "U_pos", "U_neg", "U_sem", "delta_minus", "A_q99"]


def row(ev, meta, asa, eid, arm, phase):
    return dict(id=eid, arm=arm, phase=phase, w=meta.get("w"), lr=meta.get("lr"), beta_s=asa.get("beta_s"), kappa=asa.get("kappa"),
                align_loss=ev.get("align_loss"), general_loss=ev.get("general_loss"), delta_sum=ev.get("delta_sum"),
                delta_mean=ev.get("delta_mean_per_token"), pref_rate=ev.get("pref_rate"), U=ev.get("U_per_token"),
                U_pos=ev.get("U_pos_term"), U_neg=ev.get("U_neg_term"), U_sem=ev.get("U_per_token_sem"),
                delta_minus=ev.get("delta_minus_sign"), A_q99=ev.get("A_q99_max"))


def compute(run_root: str):
    rows = []
    for arm_dir in sorted(glob.glob(os.path.join(run_root, "*"))):
        arm = os.path.basename(arm_dir)
        if arm.endswith("_sft") or arm not in arms.DIR_TO_ID:
            continue
        eid = arms.DIR_TO_ID[arm]
        for f in sorted(glob.glob(os.path.join(arm_dir, "eval_phase*.json"))):
            k = int(os.path.basename(f)[len("eval_phase"):-len(".json")])
            ev = json.load(open(f))
            meta, asa = {}, {}
            pm = os.path.join(arm_dir, f"phase_{k:03d}", "phase_meta.json")
            pc = os.path.join(arm_dir, f"phase_{k:03d}", "asa_config.json")
            if os.path.exists(pm):
                meta = json.load(open(pm))
            if os.path.exists(pc):
                asa = json.load(open(pc))
            rows.append(row(ev, meta, asa, eid, arm, k))
        post = os.path.join(run_root, arm + "_sft", "eval_post_sft.json")
        if os.path.exists(post):
            ev = json.load(open(post))
            rows.append(row(ev, {}, {"beta_s": (ev.get("asa") or {}).get("beta_s"), "kappa": 0.0}, eid, arm, 999))
    if not rows:
        print("no evaluation files found under", run_root)
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "trajectories.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows for {sorted(set(r['id'] for r in rows), key=arms.IDS.index)}")


PANELS = [("align_loss", "alignment-passage loss $\\mathcal{L}_S$"), ("general_loss", "general loss $\\mathcal{L}_G$"),
          ("delta_sum", "deployed margin $\\Delta$ (nats)"), ("pref_rate", "preference rate $P$"),
          ("U", "utilisation $U$ (nats/token)"), ("delta_minus", "margin at $s=-1$"), ("beta_s", "gain $\\beta$"), ("kappa", "plasticity $\\kappa$")]
COLORS = {"E1": "#777777", "E2": "#0072B2", "E3": "#CC79A7", "E4": "#D55E00", "E5": "#8C564B", "E6": "#332288", "E7": "#999933",
          "E8": "#009E73", "E9": "#56B4E9", "E10": "#E69F00", "E11": "#117733", "E12": "#88CCEE", "E13": "#AA4499", "E14": "#44AA99"}


def replot(only=None, out="trajectories"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "legend.frameon": False, "pdf.fonttype": 42})
    rows = list(csv.DictReader(open(RESULTS / "trajectories.csv")))
    for r in rows:
        for k in r:
            if k not in ("id", "arm"):
                r[k] = float(r[k]) if r[k] not in ("", "None") else None
    ids = [e for e in arms.TABLE1 + arms.SCHEDULES if any(r["id"] == e for r in rows)]
    ids = sorted(set(ids), key=arms.IDS.index)
    if only:
        ids = [e for e in ids if e in only]
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 6.2), squeeze=False)
    for ax, (key, title) in zip(axes.ravel(), PANELS):
        for e in ids:
            rs = sorted([r for r in rows if r["id"] == e and r["phase"] < 999 and r[key] is not None], key=lambda r: r["phase"])
            if not rs:
                continue
            ax.plot([r["phase"] for r in rs], [r[key] for r in rs], color=COLORS.get(e, "#52514e"), lw=1.6, label=f"{e} {arms.NAME[e]}")
            post = [r for r in rows if r["id"] == e and r["phase"] == 999 and r[key] is not None]
            if post and key not in ("beta_s", "kappa"):
                ax.plot([rs[-1]["phase"] + 3], [post[0][key]], "o", color=COLORS.get(e, "#52514e"), ms=5, mfc="white")
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("midtraining phase (open circles: after math SFT)")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{out}.{ext}", dpi=200)
    print("wrote", FIGS / f"{out}.pdf", "arms:", ids)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compute", action="store_true")
    p.add_argument("--replot", action="store_true")
    p.add_argument("--run-root", default=os.environ.get("SSA_RUNS", os.path.join(os.environ.get("SSA_ROOT", ""), "runs")))
    p.add_argument("--arms", default=None, help="comma-separated subset of experiment ids to plot (default: all)")
    p.add_argument("--out", default="trajectories", help="figure basename under figures/")
    a = p.parse_args()
    if a.compute:
        compute(a.run_root)
    if a.replot:
        replot(only=a.arms.split(",") if a.arms else None, out=a.out)
    if not (a.compute or a.replot):
        p.error("pass --compute and/or --replot")
