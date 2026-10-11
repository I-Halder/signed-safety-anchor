"""CPU smoke test of the ASA patch on the converted Pythia-160M checkpoint.
Checks: strict state-dict load, forward with label tokens, sign effect, gradient flow to the
anchor functionals, two-group optimizer construction and state loading, post-step constraints."""
import json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.environ.get("LITGPT_DIR", os.path.join(os.path.dirname(os.path.dirname(HERE)), "litgpt")))
import asa_patch
from asa_patch import ASAConfig, STATE, TOK_POS, TOK_NEG

ckpt = sys.argv[1] if len(sys.argv) > 1 else os.environ["INIT_CKPT"]   # the converted phase-15 checkpoint
cfg = ASAConfig(enabled=True, beta_s=1.0, sigma=0.05, kappa=2.0, signed=True, disable_compile=True)
asa_patch.apply(cfg)
from litgpt.config import Config
from litgpt.model import GPT
config = Config.from_file(os.path.join(ckpt, "model_config.yaml"))
model = GPT(config)
sd = torch.load(os.path.join(ckpt, "lit_model.pth"), map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(sd["model"], strict=True), None
print("state dict loaded strictly; anchor params:", len(STATE.anchor_params))
model.max_seq_length = 64
model.train()
torch.manual_seed(0)
B, T = 3, 32
idx = torch.randint(5, 50000, (B, T + 1))
idx[0, 0] = TOK_POS; idx[1, 0] = TOK_NEG
x, y = idx[:, :-1], idx[:, 1:]
asa_patch.register_constraints(model)
t = time.time(); logits = model(x); print("forward s", round(time.time() - t, 2), logits.shape)
print("sign used:", model._asa_sign_holder[0].tolist())
loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
loss.backward()
g = [p.grad.abs().mean().item() for p in STATE.anchor_params]
print("loss", loss.item(), "anchor grad mean (first 3 layers)", g[:3])
# sign effect in eval mode
model.eval(); STATE.cfg.eval_mode = True
with torch.no_grad():
    STATE.cfg.force_sign = 1; lp = torch.log_softmax(model(x).float(), -1)
    STATE.cfg.force_sign = -1; lm = torch.log_softmax(model(x).float(), -1)
    STATE.cfg.force_sign = None
print("mean |logp(+)-logp(-)| :", (lp - lm).abs().mean().item())
STATE.cfg.eval_mode = False; model.train()
# optimizer with two groups + state load
opt = asa_patch.asa_instantiate_torch_optimizer({"class_path": "torch.optim.AdamW", "init_args": {"weight_decay": 0.1, "betas": [0.9, 0.95], "lr": 1e-4}}, model.parameters())
print("groups", [len(g["params"]) for g in opt.param_groups])
opt.load_state_dict(sd["optimizer"]); [g.__setitem__("fused", False) for g in opt.param_groups]; print("optimizer state loaded; groups after load", [len(g["params"]) for g in opt.param_groups])
qb_before = model.transformer.h[0].attn.attn.bias.detach().clone()
opt.step()
qb_after = model.transformer.h[0].attn.attn.bias.detach()
c = config
qidx = torch.arange(qb_before.numel()).view(c.n_query_groups, 3, c.head_size)[:, 0].reshape(-1)
kidx = torch.arange(qb_before.numel()).view(c.n_query_groups, 3, c.head_size)[:, 1].reshape(-1)
print("q-bias change (should be 0):", (qb_after[qidx] - qb_before[qidx]).abs().max().item(), "| k-bias change (nonzero):", (qb_after[kidx] - qb_before[kidx]).abs().max().item())
print("label rows zero:", model.transformer.wte.weight[[TOK_POS, TOK_NEG]].abs().max().item())
print("OK")
