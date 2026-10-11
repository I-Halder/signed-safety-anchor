"""Aggregate the per-phase evaluation JSONs of every arm into one CSV and print a summary table.
Usage: python collect_results.py --run-root $SSA_ROOT/runs --out results.csv
"""
import argparse, csv, glob, json, os

KEYS = ["align_loss", "general_loss", "delta_sum", "delta_mean_per_token", "pref_rate", "U_per_token", "U_sum", "A_q99_max"]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-root", required=True); p.add_argument("--out", required=True); a = p.parse_args()
    rows = []
    for arm_dir in sorted(glob.glob(os.path.join(a.run_root, "*"))):
        arm = os.path.basename(arm_dir)
        if arm.endswith("_sft") or not os.path.isdir(arm_dir):
            continue
        for f in sorted(glob.glob(os.path.join(arm_dir, "eval_phase*.json"))) + sorted(glob.glob(os.path.join(arm_dir, "eval_post_sft.json"))):
            ev = json.load(open(f))
            k = os.path.basename(f).replace("eval_phase", "").replace(".json", "")
            meta = {}
            pm = os.path.join(arm_dir, f"phase_{k}", "phase_meta.json") if k.isdigit() else None
            if pm and os.path.exists(pm): meta = json.load(open(pm))
            asa = ev.get("asa") or {}
            rows.append({"arm": arm, "phase": k, "w": meta.get("w"), "lr": meta.get("lr"), "beta_s": asa.get("beta_s"), "kappa": asa.get("kappa"),
                         **{kk: ev.get(kk) for kk in KEYS}, "enrich_refusal": (ev.get("layer0_enrichment_log2") or {}).get("refusal"),
                         "enrich_harm": (ev.get("layer0_enrichment_log2") or {}).get("harm")})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{'arm':14s} {'phase':>6s} {'w':>5s} {'beta':>6s} {'align':>7s} {'general':>8s} {'delta':>8s} {'pref':>6s} {'U/tok':>7s}")
    for r in rows:
        f = lambda x, n=3: ("%%.%df" % n) % x if isinstance(x, (int, float)) else "-"
        print(f"{r['arm']:14s} {r['phase']:>6s} {f(r['w'],2):>5s} {f(r['beta_s'],2):>6s} {f(r['align_loss']):>7s} {f(r['general_loss']):>8s} {f(r['delta_sum'],2):>8s} {f(r['pref_rate'],3):>6s} {f(r['U_per_token'],4):>7s}")

if __name__ == "__main__":
    main()
