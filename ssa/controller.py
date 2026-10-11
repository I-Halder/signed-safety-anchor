"""Phase-boundary controller for the adaptive safety anchor.

Inputs : controller state JSON (previous beta_s, kappa, lambda), the evaluation JSON of the phase that just
         finished, the phase plan (w_k, eta_k) and the arm's hyperparameters.
Output : the ASA config JSON for the next phase and the updated state.

Rules (all indices refer to midtraining phases k = 1..K):
  warm-up (k <= K_w):     beta_s = beta_0 * k / K_w                      (unconditional, no controller)
  gain    (k >  K_w):     beta_s <- clip(beta_s + kappa_beta * (U* - U_k), beta_min, beta_max_k)
                          beta_max_k = Lambda * sqrt(d) / (2 A_k)         (log-prior cap, Lemma cap)
  plasticity:             kappa_k = kappa_0 * (w_k / w_1)^gamma            (open loop, synchronised with the safety fraction)
  fixed arm:              beta_s = beta_0 after warm-up, kappa = 1
  unsigned arm:           as fixed, but s = +1 on every document (the original, label-free protocol)
"""
from __future__ import annotations

import argparse
import json
import math
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True, help="controller state JSON (created if missing)")
    p.add_argument("--eval", default=None, help="evaluation JSON of the phase that just finished")
    p.add_argument("--phase", type=int, required=True, help="index k (1-based) of the NEXT midtraining phase")
    p.add_argument("--w", type=float, required=True, help="safety fraction w_k of the next phase")
    p.add_argument("--w1", type=float, required=True, help="safety fraction of the first midtraining phase")
    p.add_argument("--mode", choices=["adaptive", "fixed", "unsigned", "off"], required=True)
    p.add_argument("--beta0", type=float, default=2.0)
    p.add_argument("--beta-min", type=float, default=0.25)
    p.add_argument("--warmup-phases", type=int, default=3)
    p.add_argument("--kappa0", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--kappa-beta", type=float, default=2.0)
    p.add_argument("--U-target", type=float, default=0.10, help="target utilisation (nats per token)")
    p.add_argument("--Lambda", type=float, default=1.5)
    p.add_argument("--head-size", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--fix-anchor-norm", action="store_true", help="spherical anchor functional: hold ||a^(h)|| fixed (projected step); beta_s alone sets the prior strength")
    p.add_argument("--normalize-response", action="store_true", help="response-normalised prior a.x/sigma_h (running RMS, detached); beta_s then sets the prior scale exactly")
    p.add_argument("--cap-on-spread", action="store_true", help="cap statistic = 0.99-quantile of the within-sequence spread |a.x - mean_j a.x| (A_spread_q99_max) instead of |a.x| (A_q99_max)")
    p.add_argument("--U-deadband", type=float, default=0.0, help="no additive gain update while |U - U_target| <= this (anti-windup against the slow response of U to beta_s)")
    p.add_argument("--kappa-schedule", default="const", choices=["const", "down", "up", "bang", "latebang"],
                   help="plasticity schedule for the anchor functional in the fixed-gain modes. The first four have matched writing budget "
                        "sum_k kappa_k = K: const (kappa=--kappa-level, default 1), down (peak->0, write early), up (0->peak, write late), "
                        "bang (peak for the first half, then frozen). latebang (0 for the first half, then peak) is bang's time reverse: it "
                        "matches bang's shape, peak and active duration but not its exposure. Raising --kappa-level above 1 deliberately "
                        "BREAKS the budget match, to give a uniform schedule of prescribed cumulative exposure sum_k eta_k kappa_k.")
    p.add_argument("--kappa-level", type=float, default=1.0,
                   help="plasticity of the 'const' schedule (default 1.0 = the matched-budget reference). Set to 1.5061 for the "
                        "exposure-matched control of the 'bang' schedule under this pipeline's cosine learning rate.")
    p.add_argument("--kappa-peak", type=float, default=2.0, help="peak plasticity of the non-constant schedules; the budget is matched when peak=2")
    p.add_argument("--num-phases", type=int, default=36, help="number of midtraining phases K (the clock of --kappa-schedule)")
    p.add_argument("--positive-only", action="store_true", help="realistic label regime: only the safe midtraining documents are labelled; TOK_NEG blocks are treated as unlabelled")
    p.add_argument("--unlabelled-negative", action="store_true", help="presume-not-safe: unlabelled documents get s=-1 instead of a random sign")
    p.add_argument("--anchor-seed", type=int, default=None, help="seed for the fixed anchor directions u_s (an arbitrary design choice whose variance the review asks to measure)")
    p.add_argument("--permuted-labels", action="store_true", help="negative control: labelled blocks keep their gradient exposure but get a block-consistent pseudo-random sign instead of their label's sign")
    p.add_argument("--token-conditioning", action="store_true", help="Korbak-style baseline: no anchor, but the reserved label tokens keep trainable embeddings (do not zero their rows)")
    p.add_argument("--passage-positive", action="store_true", help="domain-sign regime: s=+1 on D_+ and the TOK_PAS-tagged alignment passages, s=-1 on D_- and general text (needs train/v1)")
    p.add_argument("--plasticity-all", action="store_true", help="let unlabelled documents write into the anchor functional (use with --unlabelled-negative)")
    p.add_argument("--out", required=True, help="ASA config JSON for the next phase")
    a = p.parse_args()

    st = json.load(open(a.state)) if os.path.exists(a.state) else {"beta_s": 0.0, "kappa": a.kappa0, "history": []}
    ev = json.load(open(a.eval)) if a.eval and os.path.exists(a.eval) else None
    k = a.phase

    if a.mode == "off":
        cfg = {"enabled": False}
        if a.token_conditioning:
            cfg["zero_label_rows"] = False   # the control token must be able to learn
        st["history"].append({"phase": k, "beta_s": 0.0, "kappa": 0.0})
    else:
        if k <= a.warmup_phases:
            beta = a.beta0 * k / a.warmup_phases
        elif a.mode in ("fixed", "unsigned"):
            beta = a.beta0
        else:
            beta = float(st.get("beta_s", a.beta0))
            U = ev.get("U_per_token") if ev else None
            # monotonicity tripwire (note, stability remark): if U fell in two consecutive phases in which beta rose, hold beta
            hist = [h for h in st["history"] if h.get("U_prev") is not None]
            if len(hist) >= 3 and not st.get("frozen"):
                b = [h["beta_s"] for h in hist[-3:]]; u = [h["U_prev"] for h in hist[-3:]]
                if b[1] > b[0] and b[2] > b[1] and u[1] < u[0] and u[2] < u[1]:
                    st["frozen"] = True
                    print("[controller] monotonicity check tripped: beta_s held fixed from now on")
            if U is not None and not st.get("frozen") and abs(a.U_target - U) > a.U_deadband:
                beta = beta + a.kappa_beta * (a.U_target - U)
            beta_max = None
            A_key = "A_spread_q99_max" if (a.cap_on_spread and ev and ev.get("A_spread_q99_max")) else "A_q99_max"
            if ev and ev.get(A_key):
                beta_max = a.Lambda * math.sqrt(a.head_size) / (2.0 * ev[A_key])
            hi = min(beta_max, 4.0 * a.beta0) if beta_max else 4.0 * a.beta0
            beta = min(max(beta, a.beta_min), hi)
        if a.mode in ("fixed", "unsigned"):
            # u runs linearly from 1 at the first phase to 0 at the last; every schedule below has
            # the same total writing budget sum_k kappa_k = K, so they differ only in *when* a is written
            u = (a.num_phases - k) / max(1, a.num_phases - 1)
            kappa = {"const": a.kappa_level,
                     "down": a.kappa_peak * u,
                     "up": a.kappa_peak * (1.0 - u),
                     "bang": a.kappa_peak if u >= 0.5 else 0.0,
                     "latebang": 0.0 if u >= 0.5 else a.kappa_peak}[a.kappa_schedule]
        else:
            kappa = a.kappa0 * (a.w / a.w1) ** a.gamma
        cfg = {"enabled": True, "beta_s": float(beta), "beta_g": 0.0, "sigma": a.sigma, "kappa": float(kappa),
               "signed": a.mode != "unsigned", "rademacher_unlabelled": a.mode != "unsigned", "freeze_q_bias": True, "eval_mode": False,
               "force_sign": None, "disable_compile": True, "fix_anchor_norm": bool(a.fix_anchor_norm), "normalize_response": bool(a.normalize_response),
               "positive_only": bool(a.positive_only), "unlabelled_negative": bool(a.unlabelled_negative),
               "plasticity_all": bool(a.plasticity_all), "labelled_only_plasticity": not bool(a.plasticity_all),
               "passage_positive": bool(a.passage_positive), "permuted_labels": bool(a.permuted_labels)}
        if a.anchor_seed is not None:
            cfg["anchor_seed"] = int(a.anchor_seed)
        st["beta_s"] = float(beta); st["kappa"] = float(kappa)
        st["history"].append({"phase": k, "beta_s": float(beta), "kappa": float(kappa),
                              "U_prev": (ev or {}).get("U_per_token"), "delta_prev": (ev or {}).get("delta_sum"),
                              "A_prev": (ev or {}).get("A_q99_max"), "A_spread_prev": (ev or {}).get("A_spread_q99_max")})
    json.dump(cfg, open(a.out, "w"), indent=1)
    json.dump(st, open(a.state, "w"), indent=1)
    print(json.dumps(cfg))


if __name__ == "__main__":
    main()
