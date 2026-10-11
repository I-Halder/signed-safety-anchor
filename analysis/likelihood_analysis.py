"""The likelihood analyses of the paper (bootstrap contrasts, margin decomposition, three candidate losses,
learning-rate budgets); no model training.

Use --collect RUN_ROOT once to copy the per-scenario scores from the saved evaluation JSONs of a run tree
(run directories as named in analysis/arms.py). Subsequent runs read only results/scores.json.
Bootstrap intervals condition on the fitted checkpoints and the reused 267-scenario
development set. They do not measure training-seed or model-selection uncertainty.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "analysis"))
import arms  # the roster: ids, names, run directories
(ROOT / "figures").mkdir(parents=True, exist_ok=True)
(ROOT / "results").mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", type=Path)
    args = ap.parse_args()
    snapshot = ROOT / "results/scores.json"
    if args.collect:
        data = {"source": "saved evaluation JSONs; same heldout_pairs.jsonl ordering",
                "pairing_caveat": "original pair_rows omit scenario IDs; pairing uses shared evaluator input order and checks token counts",
                "runs": {}}
        for exp in arms.IDS:
            arm = arms.RUN_DIR[exp]
            if not (args.collect / arm).is_dir():
                print(f"  skip {exp}: no run directory")
                continue
            data["runs"][exp] = {}
            for stage, rel in (("mid", f"{arm}/eval_phase036.json"),
                               ("post", f"{arm}_sft/eval_post_sft.json")):
                src = args.collect / rel
                if not src.exists():          # an arm whose SFT has not landed yet
                    print(f"  skip {exp}/{stage}: {rel} missing")
                    continue
                raw = src.read_bytes()
                obj = json.loads(raw)
                data["runs"][exp][stage] = {
                    "relative_source": rel, "sha256": hashlib.sha256(raw).hexdigest(),
                    "align_loss": obj["align_loss"], "general_loss": obj["general_loss"],
                    "rows": obj["pair_rows"],
                }
        snapshot.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    data = json.loads(snapshot.read_text())["runs"]
    data = {e: v for e, v in data.items() if v}   # arms that were run; a Table 1-only run tree has E1-E10
    n = len(data["E2"]["mid"]["rows"])
    BOOT_SEED = 20260923
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(10000, n))
    stats, vectors = {}, {}
    ref_lengths = [(r["pos_n"], r["neg_n"]) for r in data["E2"]["mid"]["rows"]]
    for exp, stages in data.items():
        stats[exp], vectors[exp] = {}, {}
        for stage, obj in stages.items():
            rows = obj["rows"]
            assert [(r["pos_n"], r["neg_n"]) for r in rows] == ref_lengths
            arr = lambda k: np.array([r[k] for r in rows])
            dp = arr("pos_s+_sum") - arr("neg_s+_sum")
            dm = arr("pos_s-_sum") - arr("neg_s-_sum")
            dn = arr("pos_s+_mean") - arr("neg_s+_mean")
            up = arr("pos_s+_mean") - arr("pos_s-_mean")
            un = arr("neg_s-_mean") - arr("neg_s+_mean")
            vectors[exp][stage] = {"delta": dp, "delta_norm": dn, "pref": (dp > 0).astype(float)}
            stats[exp][stage] = {"delta": float(dp.mean()), "delta_minus": float(dm.mean()),
                "delta_norm": float(dn.mean()), "pref": float((dp > 0).mean()),
                "pref_count": int((dp > 0).sum()), "gap": float((dp-dm).mean()),
                "U_pos": float(up.mean()), "U_neg": float(un.mean()), "U": float((up+un).mean()),
                "align_loss": obj["align_loss"], "general_loss": obj["general_loss"]}
    contrasts = []
    for exp, control in [("E2", "E1"), ("E4", "E2"), ("E4", "E3"),
                         ("E8", "E2"), ("E8", "E4"), ("E9", "E2"), ("E10", "E2"),
                         ("E8", "E13"), ("E13", "E4"), ("E4", "E1"), ("E8", "E1"), ("E10", "E1"),
                         ("E5", "E2"), ("E4", "E5"),
                         ("E6", "E2"), ("E4", "E6"), ("E8", "E6"),
                         ("E7", "E4"), ("E7", "E2")]:
        for stage in ("mid", "post"):
            if stage not in vectors.get(exp, {}) or stage not in vectors.get(control, {}):
                continue                      # an arm whose SFT has not landed yet
            c = {"experiment": exp, "control": control, "stage": stage}
            for metric in ("delta", "delta_norm", "pref"):
                dif = vectors[exp][stage][metric] - vectors[control][stage][metric]
                ci = np.quantile(dif[idx].mean(axis=1), [0.025, 0.975])
                c[metric] = {"difference": float(dif.mean()), "ci95": ci.tolist()}
            contrasts.append(c)
    trajectories = list(csv.DictReader((ROOT / "results/trajectories.csv").open()))
    budgets = {}
    for exp in arms.SCHEDULES:
        rows = [r for r in trajectories if r["id"] == exp and r["phase"].isdigit() and 1 <= int(r["phase"]) <= 36]
        if not rows:                          # a scheduling arm that was not run
            continue
        budgets[exp] = {
            "sum_kappa": sum(float(r["kappa"]) for r in rows),
            "sum_eta_kappa": sum(float(r["lr"])*float(r["kappa"]) for r in rows),
            "sum_eta_kappa_beta2": sum(float(r["lr"])*float(r["kappa"])*float(r["beta_s"])**2 for r in rows),
        }
    for b in budgets.values():
        b["nominal_lr_ratio"] = b["sum_eta_kappa"] / budgets["E4"]["sum_eta_kappa"]
    result = {"n_pairs": n, "bootstrap_draws": 10000, "bootstrap_seed": BOOT_SEED,
        "scope": "descriptive, checkpoint-conditional, reused development set; no seed or selection uncertainty",
        "mean_lengths": np.mean(ref_lengths, axis=0).tolist(), "stats": stats,
        "contrasts": contrasts, "budgets": budgets}
    out = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
           r"Comparison & Stage & $\Delta$ difference & 95\% interval & Pref. difference (pp)\\", r"\midrule"]
    for c in contrasts:
        if (c["experiment"], c["control"]) not in [("E2", "E1"), ("E4", "E2"), ("E5", "E2"), ("E4", "E5"),
                                                   ("E6", "E2"), ("E4", "E6"), ("E8", "E6"),
                                                   ("E7", "E2"), ("E7", "E4"),
                                                   ("E8", "E4"), ("E8", "E13"), ("E13", "E4"),
                                                   ("E9", "E2"), ("E10", "E2"), ("E4", "E1"), ("E8", "E1")]:
            continue
        d,p=c["delta"],c["pref"]
        out.append(f'{c["experiment"]} $-$ {c["control"]} & {c["stage"]} & {d["difference"]:.2f} & '
                   f'[{d["ci95"][0]:.2f}, {d["ci95"][1]:.2f}] & '
                   f'{100*p["difference"]:.2f} [{100*p["ci95"][0]:.2f}, {100*p["ci95"][1]:.2f}]'+r'\\')
    out += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "results/bootstrap_table.tex").write_text("\n".join(out)+"\n")

    # ---- which side of the margin moves, and whether it is a length artefact -------------------
    def side(exp, stage, control="E2"):
        """log p(completion | prompt) under the deployed sign, relative to the same-data control."""
        r = lambda e, k: np.array([row[k] for row in data[e][stage]["rows"]])
        dpos = r(exp, "pos_s+_sum") - r(control, "pos_s+_sum")
        dneg = r(exp, "neg_s+_sum") - r(control, "neg_s+_sum")
        o = {}
        for nm, v in (("aligned", dpos), ("misaligned", dneg)):
            b = np.quantile(v[idx].mean(axis=1), [0.025, 0.975])
            o[nm] = {"difference": float(v.mean()), "ci95": b.tolist()}
        o["delta"] = float((dpos - dneg).mean())
        return o
    decomp = {f"{e}/{st}": side(e, st) for e in ("E3", "E4", "E8", "E9", "E10", "E5", "E6", "E7")
              for st in ("mid", "post") if st in data.get(e, {}) and st in data["E2"]}
    result["decomposition_vs_E2"] = decomp
    result["abs_nll_per_token_mid"] = {
        e: {"aligned": float(-np.array([r["pos_s+_mean"] for r in data[e]["mid"]["rows"]]).mean()),
            "misaligned": float(-np.array([r["neg_s+_mean"] for r in data[e]["mid"]["rows"]]).mean())}
        for e in arms.IDS if "mid" in data.get(e, {})}

    # ---- review Q6: where the fit cost is paid. Three balanced candidate losses on identical pairs:
    #      deployed (both under s=+1), own-sign (each candidate under its own label), and an
    #      explicitly specified equal-sign mixture. The mixture is a chosen diagnostic distribution,
    #      not automatically the corpus marginal.
    def three_losses(e, stage="mid"):
        r = data[e][stage]["rows"]
        a = lambda k: np.array([x[k] for x in r])
        dep = float(np.mean(-0.5 * a("pos_s+_sum") - 0.5 * a("neg_s+_sum")))
        own = float(np.mean(-0.5 * a("pos_s+_sum") - 0.5 * a("neg_s-_sum")))
        mixp = np.logaddexp(a("pos_s+_sum"), a("pos_s-_sum")) - np.log(2.0)
        mixn = np.logaddexp(a("neg_s+_sum"), a("neg_s-_sum")) - np.log(2.0)
        mix = float(np.mean(-0.5 * mixp - 0.5 * mixn))
        return {"deployed_balanced": dep, "own_sign": own, "equal_sign_mixture": mix}
    result["three_losses_mid"] = {e: three_losses(e) for e in arms.IDS if "mid" in data.get(e, {})}
    tl = result["three_losses_mid"]
    rows_ = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Configuration & deployed & own-sign & equal-sign mixture\\", r"\midrule"]
    LBL = {"E1": "E1 no pairs", "E2": "E2 unconditional", "E3": "E3 constant sign",
           "E4": "E4 label-conditioned", "E9": "E9 positive only",
           "E10": "E10 positive vs.\\ unlab.", "E8": "E8 early rectangle",
           "E5": "E5 shuffled signs", "E6": "E6 control token", "E7": "E7 zero gain"}
    for e in arms.TABLE1:
        if e in tl:
            v = tl[e]
            rows_.append(f'{LBL[e]} & {v["deployed_balanced"]:.3f} & {v["own_sign"]:.3f} & {v["equal_sign_mixture"]:.3f}' + r"\\")
    rows_ += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "results/three_losses_table.tex").write_text("\n".join(rows_) + "\n")

    lens = np.array([(r["pos_n"] + r["neg_n"]) / 2 for r in data["E2"]["mid"]["rows"]])
    cuts = np.quantile(lens, [1/3, 2/3]); ter = np.digitize(lens, cuts)
    strat = {}
    for exp, control in (("E4", "E2"), ("E8", "E2"), ("E8", "E4"), ("E9", "E2")):
        dif = vectors[exp]["mid"]["delta"] - vectors[control]["mid"]["delta"]
        dtok = vectors[exp]["mid"]["delta_norm"] - vectors[control]["mid"]["delta_norm"]
        strat[f"{exp}-{control}"] = {
            "by_tercile_delta": [float(dif[ter == t].mean()) for t in range(3)],
            "by_tercile_delta_norm": [float(dtok[ter == t].mean()) for t in range(3)]}
    result["length_terciles"] = {
        "counts": [int((ter == t).sum()) for t in range(3)],
        "ranges": [[int(lens[ter == t].min()), int(lens[ter == t].max())] for t in range(3)],
        "contrasts": strat}
    (ROOT / "results/audit.json").write_text(json.dumps(result, indent=2)+"\n")

    # Same grouping and order as the roster table: conditioning ladder, then schedule, then label regime.
    NAMES = {"E3": ("E3", "Constant sign"), "E4": ("E4", "Label-conditioned"),
             "E8": ("E8", "Early rectangle"), "E9": ("E9", "Positive only"),
             "E10": ("E10", "Positive vs.\\ unlab."), "E5": ("E5", "Shuffled signs"),
             "E6": ("E6", "Control token"), "E7": ("E7", "Zero gain")}
    GROUPS = [("E3", "E4", "E5", "E6", "E7"), ("E8",), ("E9", "E10")]
    dec = [r"\begin{tabular}{@{}lllrrr@{}}", r"\toprule",
           r"ID & Name & Stage & $\delta\log p(c^+)$ & $\delta\log p(c^-)$ & $\delta\Delta$\\", r"\midrule"]
    for gi, group in enumerate(GROUPS):
        if gi:
            dec.append(r"\midrule")
        for e in group:
            for st in ("mid", "post"):
                if f"{e}/{st}" not in decomp:
                    continue
                o = decomp[f"{e}/{st}"]; a, b = o["aligned"], o["misaligned"]
                idc, nm = NAMES[e] if st == "mid" else ("", "")
                dec.append(f'{idc} & {nm} & {st} & '
                           f'{a["difference"]:.2f} [{a["ci95"][0]:.2f}, {a["ci95"][1]:.2f}] & '
                           f'{b["difference"]:.2f} [{b["ci95"][0]:.2f}, {b["ci95"][1]:.2f}] & {o["delta"]:.2f}' + r"\\")
    dec += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "results/decomposition_table.tex").write_text("\n".join(dec)+"\n")

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"E1":"#777777", "E2":"#0072B2", "E3":"#CC79A7", "E4":"#D55E00", "E8":"#009E73"}
    kcolors = {"E14":"#56B4E9", "E12":"#0072B2", "E4":"#D55E00", "E11":"#E69F00", "E13":"#CC79A7", "E8":"#009E73"}   # panel (c): the four plasticity schedules
    fig, axs = plt.subplots(1,3,figsize=(7.0,2.2),layout="constrained")
    for exp in [e for e in colors if "mid" in stats.get(e, {}) and "post" in stats.get(e, {})]:
        m,p=stats[exp]["mid"],stats[exp]["post"]
        axs[0].scatter(m["align_loss"],m["delta"],color=colors[exp],s=30)
        axs[0].annotate(exp,(m["align_loss"],m["delta"]),xytext=(4,4),textcoords="offset points")
        axs[1].plot([0,1],[m["delta"],p["delta"]],"o-",color=colors[exp],label=exp,ms=3)
    axs[0].set(xlabel="Alignment-document loss (nats/token)",ylabel=r"Aligned preference $\Delta$ (nats)")
    axs[0].set_xlim(2.325,2.422)
    axs[1].set(xticks=[0,1],xticklabels=["Midtraining","Math SFT"],ylabel=r"Aligned preference $\Delta$ (nats)",xlim=(-.1,1.15))
    axs[1].legend(frameon=False,ncol=2,fontsize=6,loc="upper right")
    for exp in [e for e in ("E14","E12","E4","E11","E13","E8") if e in budgets and "mid" in stats.get(e, {})]:   # panel (c) needs the scheduling arms
        x=budgets[exp]["nominal_lr_ratio"]
        axs[2].scatter(x,stats[exp]["mid"]["delta"],color=kcolors[exp],s=30)
        axs[2].annotate(exp,(x,stats[exp]["mid"]["delta"]),xytext=(3,4),textcoords="offset points")
    axs[2].set(xlabel=r"$\sum_k\eta_k\kappa_k$ / uniform",ylabel=r"Aligned preference $\Delta$ (nats)",xlim=(.58,1.67))
    for ax,label in zip(axs,["(a)","(b)","(c)"]):
        ax.text(.02,.98,label,transform=ax.transAxes,va="top",fontweight="bold")
        ax.grid(alpha=.15)
    fig.savefig(ROOT / "figures/likelihood_summary.pdf",bbox_inches="tight")
    fig.savefig(ROOT / "figures/likelihood_summary.png",dpi=180,bbox_inches="tight")
    print(json.dumps({"budgets": budgets, "contrasts": contrasts},indent=2))


if __name__ == "__main__":
    main()
