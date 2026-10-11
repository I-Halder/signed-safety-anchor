"""Gate for E5 (shuffled-sign control) and E6 (control-token baseline).

Exits non-zero if either arm is misconfigured, so the GPU arms can depend on it with afterok.
Run on CPU; needs the converted phase-15 checkpoint (argument or $INIT_CKPT).
"""
import os, sys, torch, numpy as np
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.environ.get("LITGPT_DIR", os.path.join(HERE, "..", "..", "litgpt")))
import asa_patch
from asa_patch import ASAConfig, STATE, TOK_POS, TOK_NEG
from litgpt.config import Config
from litgpt.model import GPT

CK = sys.argv[1] if len(sys.argv) > 1 else os.environ["INIT_CKPT"]
cfg = ASAConfig(enabled=True, beta_s=2.0, sigma=0.0, permuted_labels=True, disable_compile=True)
asa_patch.apply(cfg); STATE.cfg = cfg
conf = Config.from_file(os.path.join(CK, "model_config.yaml"))
model = GPT(conf); model.max_seq_length = 32; model.train()
sd = torch.load(os.path.join(CK, "lit_model.pth"), map_location="cpu", mmap=True, weights_only=False)
model.load_state_dict(sd.get("model", sd), strict=True); del sd

g = torch.Generator().manual_seed(1); N = 256
content = torch.randint(100, 50000, (N, 31), generator=g)
lab = torch.where(torch.rand(N, generator=g) < 0.5, TOK_POS, TOK_NEG)
idx = torch.cat([lab[:, None], content], 1)


def signs(x):
    with torch.no_grad():
        model(x)
    return model.transformer.h[0].attn._asa_sign_ref[0].clone()


fail = []
print("(1) SHUFFLED-SIGN CONTROL (E5)")
s1, s2 = signs(idx), signs(idx)
det = torch.equal(s1, s2); frac = (s1 > 0).float().mean().item()
corr = float(np.corrcoef(s1.numpy(), ((lab == TOK_POS).float() * 2 - 1).numpy())[0, 1])
idxf = idx.clone(); idxf[:, 0] = torch.where(lab == TOK_POS, TOK_NEG, TOK_POS)
label_indep = torch.equal(signs(idxf), s1)
idxc = idx.clone(); idxc[:, 1] = (idxc[:, 1] + 1) % 50000
content_dep = not torch.equal(signs(idxc), s1)
print(f"  deterministic across calls ......... {det}")
print(f"  fraction +1 ........................ {frac:.3f}   (want ~0.5)")
print(f"  corr(sign, true label) ............. {corr:+.4f}  (want ~0)")
print(f"  sign unchanged if only label flips . {label_indep}")
print(f"  sign changes if content changes .... {content_dep}")
if not det: fail.append("signs not deterministic")
if not 0.40 < frac < 0.60: fail.append(f"sign balance {frac:.3f} outside [0.40,0.60]")
if abs(corr) > 0.15: fail.append(f"|corr with label| = {abs(corr):.3f} > 0.15")
if not label_indep: fail.append("sign depends on the label token")
if not content_dep: fail.append("sign does not depend on block content")

STATE.cfg = ASAConfig(enabled=True, beta_s=2.0, sigma=0.0, disable_compile=True)
e4_ok = torch.equal(signs(idx), torch.where(lab == TOK_POS, 1.0, -1.0))
print(f"  E4 regime still label-driven ....... {e4_ok}")
if not e4_ok: fail.append("the new flag changed the default (E4) regime")

print("(2) CONTROL-TOKEN BASELINE (E6)")
norms = {}
for zlr in (False, True):
    STATE.cfg = ASAConfig(enabled=False, zero_label_rows=zlr, disable_compile=True)
    with torch.no_grad():
        model.transformer.wte.weight[TOK_POS] = torch.randn(conf.n_embd) * 0.02
    asa_patch.register_constraints(model)
    norms[zlr] = model.transformer.wte.weight[TOK_POS].norm().item()
    tag = "token-cond: must be NONZERO" if not zlr else "E2 control: must be 0"
    print(f"  zero_label_rows={str(zlr):5s} -> ||emb[TOK_POS]||={norms[zlr]:.4f}   {tag}")
if norms[False] < 1e-6: fail.append("token-conditioning arm would have its control token zeroed")
if norms[True] > 1e-9: fail.append("E2-style control no longer zeroes the label rows")

print()
if fail:
    print("FAILED:"); [print("  -", f) for f in fail]; sys.exit(1)
print("ALL CHECKS PASSED - both arms are safe to launch")
