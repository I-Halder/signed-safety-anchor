"""Table 1 of the paper (the operating points of every configuration), generated from the saved results.

Reads results/audit.json (likelihood statistics) and results/generated_behavior/generated_behavior.json (the
generated attack-success rate R) and writes results/core_table.tex with the body of the table. The paper's table
is typeset by hand; this script is the check that every cell equals the data.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import arms  # noqa: E402

# which blocks write the anchor functional a (the gradient mask); -- means the arm has no a
WRITTEN_BY = {"E1": "--", "E2": "--", "E3": r"$\mathcal D_\pm$", "E4": r"$\mathcal D_\pm$", "E5": r"$\mathcal D_\pm$",
              "E6": "--", "E7": r"$\mathcal D_\pm$", "E8": r"$\mathcal D_\pm$", "E9": r"$\mathcal D_+$", "E10": "all"}
MARK = {"E1": r"$^{\ast}$", "E6": r"$^{\dagger}$", "E7": r"$^{\ddagger}$", "E8": r"$^{\S}$"}
BLOCKS = [["E1", "E2", "E3", "E4", "E5", "E6", "E7"], ["E8"], ["E9", "E10"]]


def tex_sign(s: str) -> str:
    return {"0": "$0$", "+1": "$+1$", "-1": "$-1$", "±1": r"$\pm1$", "σ(b)": r"$\sigma(b)$", "t+": "$t_+$", "t-": "$t_-$",
            "none": "none"}[s]


def rows(audit, gen):
    out = {}
    for e in arms.TABLE1:
        m, p = audit["stats"][e]["mid"], audit["stats"][e]["post"]
        R = gen["cells"][f"{e}/mid/direct"]["asr_gen"]
        out[e] = (m["align_loss"], m["general_loss"], m["delta"], m["pref"], m["gap"], m["U"], R, p["delta"], p["pref"])
    return out


def main():
    audit = json.load(open(ROOT / "results/audit.json"))
    gen = json.load(open(ROOT / "results/generated_behavior/generated_behavior.json"))
    vals = rows(audit, gen)
    lines = []
    for bi, block in enumerate(BLOCKS):
        if bi:
            lines.append(r"\midrule")
        for e in block:
            a, m, o = (tex_sign(x) for x in arms.SIGN_MAP[e])
            ls, lg, d, p, g, u, R, dp, pp = vals[e]
            f3 = lambda x: f"{x:.3f}"[1:] if 0 <= x < 1 else f"{x:.3f}"   # .685 style for rates
            lines.append(f"{e}{MARK.get(e, '')} & {a} & {m} & {o} & {WRITTEN_BY[e]} & {ls:.3f} & {lg:.3f} & {d:.2f} & {f3(p)} & "
                         f"{g:.2f} & {f3(u)} & {f3(R)} & {dp:.2f} & {f3(pp)}" + r"\\")
    (ROOT / "results/core_table.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote results/core_table.tex")


if __name__ == "__main__":
    main()
