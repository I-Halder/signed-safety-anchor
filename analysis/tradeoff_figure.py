"""Figures 1 and 3 of the paper: observed operating points and the deployed-completion decomposition
(figures/conditional_tradeoff_story), and the routing law, analytic and measured (figures/routing_gap_vs_gain);
plus the pair-mass / within-pair decomposition table. Reads only results/scores.json.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "analysis"))
import arms  # the roster: ids, names, run directories
(ROOT / "figures").mkdir(parents=True, exist_ok=True)
(ROOT / "results").mkdir(parents=True, exist_ok=True)
runs = json.loads((ROOT / "results/scores.json").read_text())["runs"]


def lp(e, st):
    r = runs[e][st]["rows"]
    return np.array([x["pos_s+_sum"] for x in r]), np.array([x["neg_s+_sum"] for x in r])


def loss(e):
    # the score snapshot is the source Table 1 uses, so figure and table agree by construction
    return float(runs[e]["mid"]["align_loss"])


C = {"E1": "#777777", "E2": "#0072B2", "E3": "#CC79A7", "E4": "#D55E00",
     "E10": "#E69F00", "E8": "#009E73", "E5": "#8C564B", "E9": "#56B4E9",
     "E6": "#332288", "E7": "#999933"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# ================= main figure: measured quantities only =================
# Panel (b) has a broken y-axis: E1 saw different data, so its completion likelihoods sit ~32 nats
# below the rest and a single linear scale would flatten every other arm to invisibility.
fig = plt.figure(figsize=(7.0, 2.6), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], hspace=0.08)
axA = fig.add_subplot(gs[:, 0])
axT = fig.add_subplot(gs[0, 1])
axB = fig.add_subplot(gs[1, 1], sharex=axT)

# every configuration in Table 1, in its row order
ORDER = arms.TABLE1
# E2, E6, E5 and E9 sit within 0.02 nats/token and 0.6 nats of each other; place labels by hand
LABEL_OFF = {"E2": (5, -8), "E6": (5, 2), "E5": (5, -4), "E9": (5, 2), "E3": (5, 3),
             "E7": (-8, 5), "E1": (-7, 5), "E4": (5, -3), "E8": (5, -3), "E10": (5, -3)}

# (a) operating points. x increases to the right; no connectors, which would imply a frontier.
for e in ORDER:
    a_, b_ = lp(e, "mid")
    axA.scatter(loss(e), (a_ - b_).mean(), color=C[e], s=34, zorder=3)
    axA.annotate(e, (loss(e), (a_ - b_).mean()), xytext=LABEL_OFF.get(e, (4, 3)),
                 textcoords="offset points", fontsize=7)
axA.set(xlabel=r"$\mathcal{L}_S$", ylabel=r"$\Delta$")
axA.margins(x=0.10, y=0.10)
axA.grid(alpha=.15)

# (b) which side of the margin moves, against the same-data control E2
xp = np.arange(len(ORDER)); w = 0.36
dpos = [(lp(e, "mid")[0] - lp("E2", "mid")[0]).mean() for e in ORDER]
dneg = [(lp(e, "mid")[1] - lp("E2", "mid")[1]).mean() for e in ORDER]
# each sub-axis draws only the bars that belong in its range, so E1 is not also drawn
# running off the bottom of the lower panel
def masked(vals, lo, hi):
    return [v if lo <= v <= hi else np.nan for v in vals]
for ax_, (lo, hi) in ((axT, (-36.5, -29.5)), (axB, (-6.6, 1.2))):
    ax_.bar(xp - w / 2, masked(dpos, lo, hi), w, color="#009E73", label=r"$\delta\log p(c^+)$")
    ax_.bar(xp + w / 2, masked(dneg, lo, hi), w, color="#D55E00", label=r"$\delta\log p(c^-)$")
    ax_.grid(alpha=.15)
axB.annotate("reference", (1, 0), xytext=(0, 6), textcoords="offset points",
             ha="center", fontsize=6, color="#555555")
axT.set_ylim(-36.5, -29.5)          # E1 only
axB.set_ylim(-6.6, 1.2)             # every same-data arm
axB.axhline(0, color="k", lw=.6)
axT.spines["bottom"].set_visible(False); axT.tick_params(bottom=False, labelbottom=False)
axB.spines["top"].set_visible(False)
kw = dict(marker=[(-1, -.5), (1, .5)], markersize=6, linestyle="none",
          color="k", mec="k", mew=1, clip_on=False)
axT.plot([0, 1], [0, 0], transform=axT.transAxes, **kw)
axB.plot([0, 1], [1, 1], transform=axB.transAxes, **kw)
axB.set(xticks=xp, xticklabels=ORDER)
axB.tick_params(axis="x", labelsize=6.5)
axB.legend(frameon=False, fontsize=6, loc="lower left")
for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"figures/conditional_tradeoff_story.{ext}", dpi=180, bbox_inches="tight")
print(f"wrote figures/conditional_tradeoff_story.pdf ({len(ORDER)} arms, broken y-axis on (b))")

# ================= appendix figure: the routing law, analytic and measured =================
figT, axT = plt.subplots(1, 2, figsize=(7.0, 2.4), layout="constrained")
# (a) analytic: symmetric label noise, alpha = 1 - eps, gamma = eps (the sign reports the label with error eps).
#     By the routing proposition B* = log((1-eps)/eps) and G* = 2 B*, a straight line through the origin.
eps = np.linspace(0.5, 0.0042, 400)
B_th = np.log((1 - eps) / eps)
G_th = 2 * B_th
axT[0].plot(G_th, B_th, color="#009E73", label=r"$\alpha=1-\varepsilon,\ \gamma=\varepsilon$")
for e_mark, lab in ((0.25, "0.25"), (0.1, "0.1"), (0.01, "0.01")):
    by = np.log((1 - e_mark) / e_mark)
    axT[0].scatter(2 * by, by, color="#009E73", s=14, zorder=3)
    axT[0].annotate(rf"$\varepsilon={lab}$", (2 * by, by), xytext=(5, -8), textcoords="offset points", fontsize=6)
# the proposition's counterexample: alpha = 1 - eps^2, gamma = 1 - eps, B* = log(1+eps) -> 0 while G* = log((1+eps)/eps) -> inf
eps2 = np.logspace(np.log10(0.999), -4.7, 400)
axT[0].plot(np.log((1 + eps2) / eps2), np.log(1 + eps2), color="#CC79A7", label=r"$\alpha=1-\varepsilon^2,\ \gamma=1-\varepsilon$")
axT[0].axhline(0, color="k", lw=.6, zorder=0)            # beneath the counterexample curve, which hugs B = 0
axT[0].set(xlabel=r"$G$", ylabel=r"$B$")

# (b) measured: B = deployed margin over the same-data control E2, G = sign gap
d2 = (lp("E2", "mid")[0] - lp("E2", "mid")[1]).mean()
for e in ("E5", "E3", "E9", "E10", "E4", "E8"):
    a_, b_ = lp(e, "mid")
    gap = (a_ - b_).mean() - (np.array([r["pos_s-_sum"] for r in runs[e]["mid"]["rows"]])
                              - np.array([r["neg_s-_sum"] for r in runs[e]["mid"]["rows"]])).mean()
    axT[1].scatter(gap, (a_ - b_).mean() - d2, color=C[e], s=34, zorder=3)
    off = {"E5": (5, -10), "E3": (6, -2)}.get(e, (4, 3))   # clear of the B = G and B = 0 lines
    axT[1].annotate(e, (gap, (a_ - b_).mean() - d2), xytext=off, textcoords="offset points")
axT[1].axhline(0, color="k", lw=.6)
axT[1].set(xlabel=r"$G$", ylabel=r"$B$")
for a_ in axT:
    a_.set_xlim(-0.3, 10.9)
    a_.set_ylim(-0.9, 5.8)                                   # same B range in both panels; room below E5 for its label
    ylim = a_.get_ylim()
    g_line = np.linspace(-0.3, 10.9, 2)
    a_.plot(g_line, g_line, color="#555555", lw=.8, ls="--", label=r"$B=G$", zorder=1)
    a_.set_ylim(ylim)
axT[0].legend(frameon=False, fontsize=6, loc="upper left")
axT[1].legend(frameon=False, fontsize=6, loc="upper left")
for a_ in axT:
    a_.grid(alpha=.15)
for ext in ("pdf", "png"):
    figT.savefig(ROOT / f"figures/routing_gap_vs_gain.{ext}", dpi=180, bbox_inches="tight")
print("wrote figures/routing_gap_vs_gain.pdf (theory + measured)")

# ---- pair-mass / within-pair decomposition table, with the identity checked per row ----
NAMES = {"E1": "E1 no pairs", "E2": "E2 unconditional", "E3": "E3 constant sign",
         "E4": "E4 label-conditioned", "E9": "E9 positive only",
         "E10": "E10 positive vs.\\ unlab.", "E8": "E8 early rectangle",
         "E5": "E5 shuffled signs", "E6": "E6 control token", "E7": "E7 zero gain"}
out = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
       r"Configuration & $C$ & $\Delta$ & $\E\log\cosh(d/2)$ & $R_{1/2}$ & $\E q$\\", r"\midrule"]
worst = 0.0
for e in ORDER:
    a, b = lp(e, "mid")
    logM = logsumexp(np.stack([a, b]), axis=0)
    C_ = float((-logM).mean())
    dd = a - b
    J = float(np.mean(np.log(np.cosh(dd / 2))))
    R = float(np.mean(-0.5 * a - 0.5 * b))
    q = float(np.mean(1.0 / (1.0 + np.exp(-dd))))
    worst = max(worst, abs(R - (C_ + np.log(2) + J)))
    out.append(f"{NAMES[e]} & {C_:.3f} & {dd.mean():.2f} & {J:.3f} & {R:.3f} & {q:.3f}" + r"\\")
out += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "results/pair_mass_table.tex").write_text("\n".join(out) + "\n")
print(f"wrote results/pair_mass_table.tex; max |R-(C+log2+J)| = {worst:.2e}")
