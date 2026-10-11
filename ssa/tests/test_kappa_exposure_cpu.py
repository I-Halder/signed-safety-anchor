"""CPU test: the plasticity schedules have the writing budgets and exposures the note claims.

  bang      sum_k kappa_k = K and B_eta = sum_k eta_k kappa_k = 1.5061 x uniform(kappa=1)
  const 1.5061  B_eta matched to bang, but sum_k kappa_k = 54.2 (deliberately NOT budget-matched)
  latebang  bang's time reverse: same shape, peak and active duration, B_eta = 0.4939 x uniform

Run: python -m pytest ssa/tests/test_kappa_exposure_cpu.py -q  (or python ssa/tests/test_kappa_exposure_cpu.py)
"""
import csv, json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASA = HERE.parent
LR_CSV = ASA.parent / "results" / "trajectories.csv"
K = 36


def phase_lrs():
    rows = [r for r in csv.DictReader(open(LR_CSV))
            if r["id"] == "E4" and r["phase"].isdigit() and 1 <= int(r["phase"]) <= K]
    return [float(r["lr"]) for r in sorted(rows, key=lambda r: int(r["phase"]))]


def run_schedule(sched, level=1.0, peak=2.0):
    with tempfile.TemporaryDirectory() as td:
        state, out = os.path.join(td, "s.json"), os.path.join(td, "c.json")
        ks = []
        for k in range(1, K + 1):
            w = 0.9 + (0.1 - 0.9) * (k - 1) / (K - 1)
            subprocess.run([sys.executable, str(ASA / "controller.py"), "--phase", str(k), "--mode", "fixed",
                            "--w", str(w), "--w1", "0.9", "--state", state, "--out", out,
                            "--kappa-schedule", sched, "--kappa-level", str(level),
                            "--kappa-peak", str(peak), "--num-phases", str(K)],
                           capture_output=True, check=True)
            ks.append(json.load(open(out))["kappa"])
        return ks


def exposure(ks):
    lr = phase_lrs()
    return sum(l * k for l, k in zip(lr, ks)) / sum(lr)


def test_matched_budget_schedules_have_equal_kappa_sum():
    for sched in ("const", "down", "up", "bang", "latebang"):
        assert abs(sum(run_schedule(sched)) - K) < 1e-6, sched


def test_bang_is_early_rectangular_and_latebang_is_its_reverse():
    b, l = run_schedule("bang"), run_schedule("latebang")
    assert b[:K // 2] == [2.0] * (K // 2) and b[K // 2:] == [0.0] * (K // 2)
    assert l == b[::-1]


def test_exposure_ratios():
    assert abs(exposure(run_schedule("bang")) - 1.5061) < 5e-4
    assert abs(exposure(run_schedule("latebang")) - 0.4939) < 5e-4
    assert abs(exposure(run_schedule("down")) - 1.3406) < 5e-4
    assert abs(exposure(run_schedule("up")) - 0.6594) < 5e-4


def test_const_level_matches_bang_exposure_while_breaking_the_budget():
    """The decisive control for the timing claim: same cumulative nominal LR as bang, uniform in time."""
    ks = run_schedule("const", level=1.5061)     # the level is the exposure ratio, rounded to 4 dp
    assert abs(exposure(ks) - exposure(run_schedule("bang"))) < 1e-3
    assert abs(sum(ks) - 1.5061 * K) < 1e-6      # NOT budget-matched, by construction


if __name__ == "__main__":
    for _name in sorted(n for n in dir() if n.startswith("test_")):
        globals()[_name]()
        print("PASS", _name)
