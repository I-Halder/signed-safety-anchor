"""Runtime patches for LitGPT implementing the signed safety anchor (code name ASA, "adaptive safety anchor").

The paper's arms all use the fixed-gain variant (controller.py --mode fixed/unsigned); the closed-loop gain
controller ("adaptive") is kept for completeness but is not part of the paper.

An untrained, fixed-direction query bias on the non-rotary head coordinates whose sign is
set per document by a label token, plus an explicitly parametrised anchor functional
a^(h) = W_K^(h)T u_s^(h) with its own learning-rate multiplier kappa.

Nothing inside the litgpt tree is modified; this module monkey-patches at import time:
  * litgpt.model.CausalSelfAttention.forward   (anchored keys, biased queries)
  * litgpt.model.GPT.__init__ / GPT.forward     (attach anchors; read the per-document sign)
  * litgpt.pretrain / litgpt.finetune.full .instantiate_torch_optimizer  (two param groups, kappa)
  * litgpt.pretrain / litgpt.finetune.full .FSDPStrategy -> DDP (post-step constraints need whole params)
  * litgpt.data.sharded_mixed_dataset.DATASET_TYPE_CONFIGS  (aligned/misaligned shard types y*, z*)
  * torch.compile (identity; see ASAConfig.disable_compile)
Configuration is read from the JSON file named by the ASA_CONFIG environment variable.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.nn as nn

# Two token ids the Pythia tokenizer never produces (largest real id 50276; embeddings padded
# to 50304 rows). Their embedding rows are held at zero, so the label reaches the network only
# through the sign of the anchor.
TOK_POS = 50300
TOK_NEG = 50301
TOK_PAS = 50302   # alignment passages (D_S) when they are given their own sign (passage_positive)


@dataclass
class ASAConfig:
    enabled: bool = True
    beta_s: float = 0.0             # safety-anchor gain
    beta_g: float = 0.0             # structural anchor gain (0 = off; base model was pretrained without it)
    sigma: float = 0.05             # std of batchwise noise on non-rotary coordinates (0 at eval)
    kappa: float = 1.0              # LR multiplier of the anchor functionals a^(h)
    signed: bool = True             # use the label sign; False -> s=+1 always (unsigned anchor)
    rademacher_unlabelled: bool = True
    anchor_seed: int = 20260921
    freeze_q_bias: bool = True
    eval_mode: bool = False         # no noise, deterministic sign
    force_sign: Optional[int] = None
    disable_compile: bool = True
    per_head_scale: Optional[List[float]] = None
    zero_label_rows: bool = True    # hold the label-token embedding rows at zero even when the anchor is off (clean baseline)
    labelled_only_plasticity: bool = True  # only labelled documents write into the anchor functionals (unlabelled: detached)
    fix_anchor_norm: bool = False   # spherical parametrisation: keep ||a^(h)|| at its loaded value (projected step), so beta_s alone sets the prior strength
    normalize_response: bool = False  # response-normalised prior: a.x / sigma_h with sigma_h a per-head running RMS of a.x (detached, stored in the checkpoint as buffer asa_sigma)
    # --- positive-only label regime (realistic deployment: the SAFE midtraining documents are labelled,
    #     nothing else is; in particular no document is known to be unsafe).
    positive_only: bool = False      # ignore the misaligned label: TOK_NEG documents are treated exactly like unlabelled text
    unlabelled_negative: bool = False  # give every unlabelled document s = -1 (presume-not-safe) instead of a random sign
    plasticity_all: bool = False     # let unlabelled documents write into the anchor functional too (used with unlabelled_negative)
    permuted_labels: bool = False    # NEGATIVE CONTROL: labelled blocks keep their gradient exposure but their sign is
                                     # a deterministic pseudo-random function of the block content instead of the label,
                                     # so the channel is present and consistent but carries no aligned/misaligned information.
    passage_positive: bool = False   # domain-sign regime: s=+1 on D_+ and on the TOK_PAS-tagged alignment passages D_S,
                                     # s=-1 on D_- and on everything else (general text D_G). Requires the train/v1 shard
                                     # built by prepare_passage_labelled.py. Passages count as labelled for plasticity.
    response_ema: float = 0.99      # momentum of the running mean square of the response

    @staticmethod
    def from_env() -> "ASAConfig":
        path = os.environ.get("ASA_CONFIG", "").strip()
        if not path:
            return ASAConfig(enabled=False)
        with open(path) as fh:
            return ASAConfig(**json.load(fh))

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=1)


class _State:
    cfg: ASAConfig = ASAConfig(enabled=False)
    stats: dict = {"labelled_seen": 0.0, "seqs_seen": 0.0, "steps": 0}
    anchor_params: List[nn.Parameter] = []
    frozen: list = []      # (param, index_tensor, saved_values) restored after every optimizer step
    zero_rows: list = []   # (param, row_index_tensor) zeroed after every optimizer step
    anchor_norms: dict = {}  # id(param) -> per-head norms recorded before the first step (fix_anchor_norm)
    applied: bool = False


STATE = _State()


def make_anchors(n_head: int, head_size: int, rope_n_elem: int, layer_idx: int, seed: int, device=None):
    """u_g: normalised indicator of the non-rotary coordinates; u_s^(h): random unit vectors on the
    non-rotary coordinates, orthogonal to u_g, one per head. Deterministic in (seed, layer)."""
    # explicit CPU tensors: GPT may be constructed inside fabric.init_module(), which redirects the default device
    nonrot = torch.zeros(head_size, device="cpu")
    nonrot[rope_n_elem:] = 1.0
    ug = nonrot / nonrot.norm()
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + int(layer_idx))
    us = torch.randn(n_head, head_size, generator=g, device="cpu") * nonrot[None, :]
    us = us - (us @ ug)[:, None] * ug[None, :]
    us = us / us.norm(dim=-1, keepdim=True)
    if device is not None:
        ug, us = ug.to(device), us.to(device)
    return ug, us


def _setup_attention(attn: nn.Module) -> None:
    if getattr(attn, "_asa_ready", False):
        return
    cfg = attn.config
    assert cfg.n_query_groups == cfg.n_head, "ASA patch assumes multi-head attention (Pythia)."
    layer_idx = getattr(attn, "_asa_layer_idx", 0)
    ug, us = make_anchors(cfg.n_head, cfg.head_size, cfg.rope_n_elem, layer_idx, STATE.cfg.anchor_seed,
                          device=attn.attn.weight.device)
    attn.register_buffer("asa_ug", ug, persistent=False)
    attn.register_buffer("asa_us", us, persistent=False)
    if not hasattr(attn, "asa_sigma"):   # persistent (H,) running RMS of the response; zero = not yet initialised
        attn.register_buffer("asa_sigma", torch.zeros(cfg.n_head, device=attn.attn.weight.device, dtype=torch.float32), persistent=True)
    if not hasattr(attn, "asa_a"):
        w = attn.attn.weight.detach().float().view(cfg.n_query_groups, 3, cfg.head_size, cfg.n_embd)[:, 1]
        a0 = torch.einsum("hdc,hd->hc", w, us.float().to(w.device))
        attn.asa_a = nn.Parameter(a0.to(device=attn.attn.weight.device, dtype=attn.attn.weight.dtype))
    STATE.anchor_params.append(attn.asa_a)
    attn._asa_ready = True


def attach(model: nn.Module) -> None:
    for i, block in enumerate(model.transformer.h):
        block.attn._asa_layer_idx = i
        _setup_attention(block.attn)
    holder = [None, None]
    model._asa_sign_holder = holder
    for block in model.transformer.h:
        block.attn._asa_sign_ref = holder


def _normalize_response(attn: nn.Module, ax: torch.Tensor) -> torch.Tensor:
    """Divide the response a^(h).x by a per-head running scale sigma_h (detached). sigma_h^2 is the running mean of
    the WITHIN-SEQUENCE variance of the response across the keys (a constant offset across the keys of a sequence
    cancels in the softmax, so only the spread carries log-odds; the raw RMS would also count the common mode).
    sigma_h is initialised from the first batch seen by a process whose checkpoint has no value yet (zeros) and
    then follows an EMA during training; it is frozen in eval mode / at inference. A constant scale of the prior is
    exactly what the gain beta_s should control, so the normaliser removes the gain/response redundancy (I6b)."""
    sig = attn.asa_sigma
    with torch.no_grad():
        if not ax.is_meta and ax.shape[2] > 1:
            ms = ax.detach().float().var(dim=2, unbiased=False).mean(dim=(0, 2))      # (H,): spread across keys
            if bool((sig == 0).all()):
                sig.copy_(ms.clamp_min(1e-12).sqrt())
            elif attn.training and not STATE.cfg.eval_mode:
                m = float(STATE.cfg.response_ema)
                sig.copy_((m * sig.pow(2) + (1.0 - m) * ms).clamp_min(1e-12).sqrt())
    return ax / (sig.view(1, -1, 1, 1).to(ax.dtype) + 1e-6)


def _make_load_from_state_dict(orig):
    def _load(self, state_dict, prefix, *args, **kwargs):
        # checkpoints written before the response normaliser existed have no `asa_sigma`; insert the module's value
        if hasattr(self, "asa_sigma") and prefix + "asa_sigma" not in state_dict:
            state_dict[prefix + "asa_sigma"] = self.asa_sigma.detach().clone()
        return orig(self, state_dict, prefix, *args, **kwargs)
    return _load


def _bias_for(attn: nn.Module, B: int, device, dtype) -> torch.Tensor:
    cfg = STATE.cfg
    ug, us = attn.asa_ug, attn.asa_us
    nh, hs = us.shape
    ref = getattr(attn, "_asa_sign_ref", None)
    sign = ref[0] if (ref is not None and ref[0] is not None) else torch.ones(B, device=device)
    beta_s = torch.full((nh,), float(cfg.beta_s), device=device, dtype=us.dtype)
    if cfg.per_head_scale is not None:
        beta_s = beta_s * torch.tensor(cfg.per_head_scale, device=device, dtype=us.dtype)
    b = cfg.beta_g * ug[None, None, :] + sign.to(us.dtype)[:, None, None] * beta_s[None, :, None] * us[None, :, :]
    if cfg.sigma > 0 and attn.training and not cfg.eval_mode:
        nonrot = (ug != 0).to(us.dtype)
        eps = torch.randn(nh, hs, device=device, dtype=us.dtype) * float(cfg.sigma) * nonrot[None, :]
        b = b + eps[None]
    return b.to(dtype)


def asa_attention_forward(self, x, cos, sin, mask=None, input_pos=None):
    from litgpt.model import apply_rope
    B, T, C = x.size()
    qkv = self.attn(x)
    q_per_kv = self.config.n_head // self.config.n_query_groups
    total_qkv = q_per_kv + 2
    qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
    qkv = qkv.permute(0, 2, 3, 1, 4)
    q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
    if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
        k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
    q = q.reshape(B, -1, T, self.config.head_size)
    k = k.reshape(B, -1, T, self.config.head_size)
    v = v.reshape(B, -1, T, self.config.head_size)

    if STATE.cfg.enabled:
        _setup_attention(self)
        us = self.asa_us.to(k.dtype)
        proj = (k * us[None, :, None, :]).sum(-1, keepdim=True)                       # u_s . k
        ax = torch.einsum("btc,hc->bht", x, self.asa_a.to(x.dtype)).unsqueeze(-1)    # a^(h) . x
        if STATE.cfg.normalize_response:
            ax = _normalize_response(self, ax)
        ref = getattr(self, "_asa_sign_ref", None)
        lab = ref[1] if (ref is not None and len(ref) > 1) else None
        if lab is not None and self.training:
            m = lab.to(ax.dtype)[:, None, None, None]
            ax = ax * m + ax.detach() * (1.0 - m)                                   # labelled-only plasticity
        k = k - proj * us[None, :, None, :] + ax * us[None, :, None, :]

    q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
    k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
    q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
    k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

    if STATE.cfg.enabled:
        q = q + _bias_for(self, B, q.device, q.dtype)[:, :, None, :]

    if input_pos is not None:
        from litgpt.model import KVCache
        if not isinstance(self.kv_cache, KVCache):
            raise TypeError("You need to call `gpt.set_kv_cache()`")
        k, v = self.kv_cache(input_pos, k, v)

    y = self.scaled_dot_product_attention(q, k, v, mask)
    y = y.reshape(B, T, self.config.head_size * self.config.n_head)
    return self.proj(y)


def _make_gpt_forward(orig_forward):
    def asa_gpt_forward(self, idx, input_pos=None):
        if STATE.cfg.enabled:
            cfg = STATE.cfg
            B = idx.size(0)
            first = idx[:, 0]
            if cfg.force_sign is not None:
                sign = torch.full((B,), float(cfg.force_sign), device=idx.device)
            elif not cfg.signed:
                sign = torch.ones(B, device=idx.device)
            else:
                pos = (first == TOK_POS) | ((first == TOK_PAS) if cfg.passage_positive else torch.zeros_like(first, dtype=torch.bool))
                neg = (first == TOK_NEG) & (not cfg.positive_only)   # positive_only: the misaligned label is not available
                if cfg.passage_positive or cfg.unlabelled_negative:  # everything not known-safe gets s = -1
                    rnd = -torch.ones(B, device=idx.device)
                elif cfg.rademacher_unlabelled and self.training and not cfg.eval_mode:
                    rnd = torch.randint(0, 2, (B,), device=idx.device).float() * 2.0 - 1.0
                else:
                    rnd = torch.ones(B, device=idx.device)
                sign = torch.where(pos, torch.ones_like(rnd), torch.where(neg, -torch.ones_like(rnd), rnd))
                if cfg.permuted_labels:
                    # block-consistent pseudo-random sign on the labelled blocks, decorrelated from the label.
                    # A rolling hash of the first 16 content tokens: same block content -> same sign, every epoch
                    # and every rank, and about half of the labelled blocks get each sign.
                    h = torch.zeros_like(first, dtype=torch.long)
                    for t in range(1, min(17, idx.size(1))):
                        h = (h * 1000003 + idx[:, t].long()) % 2147483647
                    hsign = (h % 2).to(sign.dtype) * 2.0 - 1.0
                    sign = torch.where(pos | neg, hsign, sign)
            if not hasattr(self, "_asa_sign_holder"):
                attach(self)
            self._asa_sign_holder[0] = sign
            # documents whose label is known (and therefore counted in the log)
            if cfg.passage_positive:      # the passages carry an asserted +1 label, so they count as labelled
                known = (first == TOK_POS) | (first == TOK_NEG) | (first == TOK_PAS)
            elif cfg.positive_only:
                known = first == TOK_POS
            else:
                known = (first == TOK_POS) | (first == TOK_NEG)
            known = known.to(torch.float32)
            if self.training and not idx.is_meta:   # litgpt measures FLOPs on a meta-device copy of the model
                STATE.stats["labelled_seen"] += float(known.sum().item())
                STATE.stats["seqs_seen"] += float(B)
            # documents allowed to write into the anchor functional
            mask = torch.ones_like(known) if cfg.plasticity_all else known
            self._asa_sign_holder.append(None) if len(self._asa_sign_holder) < 2 else None
            self._asa_sign_holder[1] = mask if (cfg.labelled_only_plasticity or cfg.plasticity_all) else None
        return orig_forward(self, idx, input_pos)
    return asa_gpt_forward


class ASAAdamW(torch.optim.AdamW):
    """AdamW with a step-time LR multiplier kappa on the anchor group (litgpt overwrites
    param_group['lr'] every iteration, so the multiplier is applied inside step) and post-step
    constraints (frozen learned query biases, zeroed label-token embedding rows)."""

    def step(self, closure=None):
        kappa = float(STATE.cfg.kappa) if STATE.cfg.enabled else 1.0
        saved = []
        for g in self.param_groups:
            if g.get("asa_anchor"):
                saved.append((g, g["lr"]))
                g["lr"] = g["lr"] * kappa
        anchors = [p for g in self.param_groups if g.get("asa_anchor") for p in g["params"]]
        rank0 = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))) == 0
        if STATE.cfg.enabled and STATE.cfg.fix_anchor_norm:
            with torch.no_grad():
                for p in anchors:   # norms are taken from the loaded (checkpoint) value, after any resume, before the first update
                    if id(p) not in STATE.anchor_norms:
                        STATE.anchor_norms[id(p)] = p.detach().float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if anchors and rank0:
            gnorm = sum((p.grad.float().norm().item() ** 2) for p in anchors if p.grad is not None) ** 0.5
            ngrad = sum(1 for p in anchors if p.grad is not None)
            before = [p.detach().float().clone() for p in anchors]
        out = super().step(closure)
        for g, lr in saved:
            g["lr"] = lr
        STATE.stats["steps"] += 1
        if anchors and rank0 and (STATE.stats["steps"] % 10 == 1):
            da = sum(((p.detach().float() - b).norm().item() ** 2) for p, b in zip(anchors, before)) ** 0.5
            an = sum((p.detach().float().norm().item() ** 2) for p in anchors) ** 0.5
            print(f"[asa] step {STATE.stats['steps']}: labelled seqs seen {STATE.stats['labelled_seen']:.0f}/{STATE.stats['seqs_seen']:.0f} | anchors with grad {ngrad}/{len(anchors)} | grad norm {gnorm:.3e} | rel update {da / max(an, 1e-12):.3e} | lr x kappa {saved[0][1] * kappa if saved else None}", flush=True)
        with torch.no_grad():
            if STATE.cfg.enabled and STATE.cfg.fix_anchor_norm:
                for p in anchors:   # projected step: rescale each head's functional back to its recorded norm (in fp32)
                    tgt = STATE.anchor_norms[id(p)]
                    pf = p.detach().float()
                    p.data.copy_((pf * (tgt / pf.norm(dim=-1, keepdim=True).clamp_min(1e-8))).to(p.dtype))
            for p, idx, vals in STATE.frozen:
                p.data[idx] = vals.to(p.dtype)
            for p, rows in STATE.zero_rows:
                p.data[rows] = 0
        return out


def asa_instantiate_torch_optimizer(optimizer, model_parameters, **kwargs):
    if isinstance(optimizer, str):
        init_args = dict(kwargs)
    else:
        optimizer = dict(optimizer)
        init_args = dict(optimizer.get("init_args") or {})
        init_args.update(kwargs)
        if "lr" in optimizer and "lr" not in init_args:
            init_args["lr"] = optimizer["lr"]
    for k in ("lr", "weight_decay"):
        if isinstance(init_args.get(k), str):
            init_args[k] = float(init_args[k])
    init_args.setdefault("lr", 1e-4)
    params = list(model_parameters)
    anchor_ids = {id(p) for p in STATE.anchor_params} if STATE.cfg.enabled else set()
    rest = [p for p in params if id(p) not in anchor_ids]
    anchors = [p for p in params if id(p) in anchor_ids]
    groups = [{"params": rest}]
    if anchors:
        groups.append({"params": anchors, "asa_anchor": True})
    print(f"[asa] optimizer: {len(rest)} regular params, {len(anchors)} anchor functionals, kappa={STATE.cfg.kappa}")
    return ASAAdamW(groups, **init_args)


def _unwrap(m: nn.Module) -> nn.Module:
    root = m
    for _ in range(6):
        if hasattr(root, "transformer"):
            return root
        if hasattr(root, "_orig_mod"):
            root = root._orig_mod
        elif hasattr(root, "module"):
            root = root.module
        elif hasattr(root, "_forward_module"):
            root = root._forward_module
        else:
            break
    return root


def register_constraints(model: nn.Module) -> None:
    STATE.frozen.clear()
    STATE.zero_rows.clear()
    root = _unwrap(model)
    if STATE.cfg.zero_label_rows or STATE.cfg.enabled:
        wte = root.transformer.wte.weight
        rows = torch.tensor([TOK_POS, TOK_NEG, TOK_PAS], device=wte.device)
        with torch.no_grad():
            wte.data[rows] = 0
        STATE.zero_rows.append((wte, rows))
    if STATE.cfg.enabled and STATE.cfg.freeze_q_bias:
        for block in root.transformer.h:
            attn = block.attn
            if attn.attn.bias is None:
                continue
            c = attn.config
            idx = torch.arange(attn.attn.bias.numel(), device=attn.attn.bias.device).view(c.n_query_groups, 3, c.head_size)[:, 0].reshape(-1)
            STATE.frozen.append((attn.attn.bias, idx, attn.attn.bias.detach()[idx].clone()))
    print(f"[asa] constraints: {len(STATE.frozen)} frozen q-bias blocks, {len(STATE.zero_rows)} zeroed label rows")


def apply(cfg: Optional[ASAConfig] = None) -> ASAConfig:
    cfg = cfg or ASAConfig.from_env()
    STATE.cfg = cfg
    if STATE.applied:
        return cfg
    STATE.applied = True
    import litgpt.model as lm
    lm.CausalSelfAttention.forward = asa_attention_forward
    lm.CausalSelfAttention._load_from_state_dict = _make_load_from_state_dict(lm.CausalSelfAttention._load_from_state_dict)
    lm.GPT.forward = _make_gpt_forward(lm.GPT.forward)
    _orig_init = lm.GPT.__init__

    def gpt_init(self, config):
        _orig_init(self, config)
        if STATE.cfg.enabled:
            attach(self)
    lm.GPT.__init__ = gpt_init

    from lightning.fabric.strategies import DDPStrategy, FSDPStrategy

    class _NoFSDP(FSDPStrategy):
        """Stands in for FSDPStrategy in litgpt: constructing it yields a DDP strategy (whole parameters on every
        rank, needed for the post-step constraints), while isinstance(strategy, FSDPStrategy) checks in litgpt
        still work and correctly report a non-FSDP strategy."""
        def __new__(cls, *a, **k):
            return DDPStrategy()

    for modname in ("litgpt.pretrain", "litgpt.finetune.full"):
        try:
            mod = __import__(modname, fromlist=["x"])
            mod.instantiate_torch_optimizer = asa_instantiate_torch_optimizer
            mod.FSDPStrategy = _NoFSDP
        except Exception as e:  # pragma: no cover
            print(f"[asa] warning: could not patch {modname}: {e}")

    import lightning as L
    _orig_setup = L.Fabric.setup

    def fabric_setup(self, module, *optimizers, **kw):
        res = _orig_setup(self, module, *optimizers, **kw)
        if STATE.cfg.enabled or STATE.cfg.zero_label_rows:
            m = res[0] if isinstance(res, tuple) else res
            try:
                register_constraints(m)
            except Exception as e:  # pragma: no cover
                print(f"[asa] warning: could not register constraints: {e}")
        return res
    L.Fabric.setup = fabric_setup

    try:
        import litgpt.data.sharded_mixed_dataset as smd
        smd.DATASET_TYPE_CONFIGS.setdefault("aligned", {"prefix": "y", "description": "aligned scenario completions"})
        smd.DATASET_TYPE_CONFIGS.setdefault("misaligned", {"prefix": "z", "description": "misaligned scenario completions"})
        smd.DATASET_TYPE_CONFIGS.setdefault("passage", {"prefix": "v", "description": "alignment passages tagged with TOK_PAS"})
    except Exception as e:  # pragma: no cover
        print(f"[asa] warning: could not register dataset types: {e}")

    if cfg.disable_compile:
        torch.compile = lambda m=None, *a, **k: (m if m is not None else (lambda f: f))
    print(f"[asa] patches applied: {asdict(cfg)}")
    return cfg
