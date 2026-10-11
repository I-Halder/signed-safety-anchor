"""CPU unit test of the positive-only label regimes on a tiny Pythia-like GPT.

Checks the sign assignment and the plasticity mask in three regimes:
  full labels        (the recipe of the main note): aligned +1, misaligned -1, unlabelled random, a written by both labels
  positive-only      : aligned +1, misaligned treated as unlabelled (random sign, no gradient into a)
  presume-not-safe   : aligned +1, everything else -1, every document writes into a
"""
import os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.environ.get("LITGPT_DIR", os.path.join(os.path.dirname(os.path.dirname(HERE)), "litgpt")))
import asa_patch
from asa_patch import ASAConfig, STATE, TOK_POS, TOK_NEG
from litgpt.config import Config
from litgpt.model import GPT

config = Config(name="tiny", block_size=64, vocab_size=50304, padded_vocab_size=50304, n_layer=2, n_head=4, n_embd=64,
                rotary_percentage=0.25, parallel_residual=True, bias=True, norm_class_name="LayerNorm", mlp_class_name="GptNeoxMLP")
B, T = 8, 16
torch.manual_seed(0)
idx = torch.randint(5, 50000, (B, T))
idx[0, 0] = TOK_POS; idx[1, 0] = TOK_POS      # aligned
idx[2, 0] = TOK_NEG; idx[3, 0] = TOK_NEG      # misaligned
# rows 4..7 are unlabelled


def run(**kw):
    STATE.anchor_params.clear(); STATE.frozen.clear(); STATE.zero_rows.clear(); STATE.anchor_norms.clear()
    cfg = ASAConfig(enabled=True, beta_s=1.0, sigma=0.05, kappa=1.0, signed=True, disable_compile=True, **kw)
    if not STATE.applied:
        asa_patch.apply(cfg)
    else:
        STATE.cfg = cfg
    torch.manual_seed(1)
    model = GPT(config); model.max_seq_length = T; model.train()
    asa_patch.register_constraints(model)
    signs, labs = [], []
    for _ in range(40):                        # average over the Rademacher draws
        model(idx)
        signs.append(model._asa_sign_holder[0].clone())
        labs.append(model._asa_sign_holder[1].clone() if model._asa_sign_holder[1] is not None else torch.full((B,), float("nan")))
    return torch.stack(signs), torch.stack(labs)


def report(name, s, l):
    print(f"{name:18s} mean sign per row {[round(v,2) for v in s.mean(0).tolist()]}")
    print(f"{'':18s} plasticity mask   {[round(v,2) for v in l.mean(0).tolist()]}")


# --- full labels (main note, Experiment 4)
s, l = run(); report("full labels", s, l)
assert torch.allclose(s[:, 0], torch.ones(40)) and torch.allclose(s[:, 2], -torch.ones(40)), "labels must set the sign"
assert abs(s[:, 4].mean()) < 0.5, "unlabelled rows must be random"
assert torch.allclose(l[:, :4], torch.ones(40, 4)) and torch.allclose(l[:, 4:], torch.zeros(40, 4)), "both labels write into a"

# --- positive-only (Experiment 14)
s, l = run(positive_only=True); report("positive-only", s, l)
assert torch.allclose(s[:, 0], torch.ones(40)), "the safe label must still set +1"
assert abs(s[:, 2].mean()) < 0.5, "a TOK_NEG document must now get a random sign"
assert torch.allclose(l[:, :2], torch.ones(40, 2)), "safe documents write into a"
assert torch.allclose(l[:, 2:], torch.zeros(40, 6)), "nothing else writes into a"

# --- presume-not-safe (Experiment 15)
s, l = run(positive_only=True, unlabelled_negative=True, plasticity_all=True, labelled_only_plasticity=False)
report("presume-not-safe", s, l)
assert torch.allclose(s[:, :2], torch.ones(40, 2)), "safe documents are +1"
assert torch.allclose(s[:, 2:], -torch.ones(40, 6)), "everything else is -1"
assert torch.allclose(l, torch.ones(40, B)), "every document writes into a"

# --- inference is unchanged in all three regimes
STATE.cfg.eval_mode = True; STATE.cfg.force_sign = 1
torch.manual_seed(1); m = GPT(config); m.max_seq_length = T; m.eval()
with torch.no_grad():
    m(idx)
assert torch.allclose(m._asa_sign_holder[0], torch.ones(B)), "inference must run at s=+1 for every document"
print("POSITIVE_ONLY_OK")
