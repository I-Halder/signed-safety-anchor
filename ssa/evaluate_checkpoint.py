"""Evaluate an (ASA or plain) LitGPT checkpoint: alignment-document loss, general loss, the sign-channel
utilisation U, the aligned-vs-misaligned log-odds Delta under the deployed sign, the anchor read-outs
(layer-0 top/bottom aligned tokens along u_s, category enrichment) and the cap statistic A.
Single process, one GPU (or CPU). Writes a JSON file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("LITGPT_DIR", os.path.join(os.path.dirname(HERE), "litgpt")))
import asa_patch  # noqa: E402
from asa_patch import STATE, TOK_POS  # noqa: E402

CATEGORIES = {
    "refusal": r"(cannot|can't|refus|sorry|unable|decline|not allowed|won't|shouldn't|must not)",
    "harm": r"(harm|danger|weapon|kill|attack|illegal|abuse|exploit|damage|threat|violence|hurt)",
    "oversight_honesty": r"(honest|transparen|oversight|monitor|report|truth|deceiv|manipulat|disclos|trust|safety|safe)",
    "override": r"(ignore|pretend|override|bypass|jailbreak|instruction|secret|hide|conceal)",
    "role": r"^(User|Assistant|AI|Human|System|model)$",
    "punct": r"^\W+$",
    "digits": r"^\d+$",
}


def load_model(ckpt_dir, asa_cfg):
    from litgpt.config import Config
    from litgpt.model import GPT
    asa_patch.apply(asa_cfg)
    config = Config.from_file(os.path.join(ckpt_dir, "model_config.yaml"))
    model = GPT(config)
    sd = torch.load(os.path.join(ckpt_dir, "lit_model.pth"), map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    if not asa_cfg.enabled:
        sd = {k: v for k, v in sd.items() if not k.endswith("asa_a")}
    model.load_state_dict(sd, strict=True)
    return model, config


@torch.no_grad()
def stream_loss(model, path, n_batches, batch_size, device, seq):
    from litdata.streaming import StreamingDataset, TokensLoader
    from torch.utils.data import DataLoader
    ds = StreamingDataset(input_dir=path, item_loader=TokensLoader(block_size=seq + 1), shuffle=True, drop_last=True, seed=42)
    loader = DataLoader(ds, batch_size=batch_size, drop_last=True, num_workers=2)
    losses = []
    for k, batch in enumerate(loader):
        if k >= n_batches:
            break
        tokens = batch.long().to(device)
        logits = model(tokens[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tokens[:, 1:].reshape(-1))
        losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses))), len(losses)


@torch.no_grad()
def completion_logprob(model, tok, question, completion, device, sign, seq):
    q_ids = [TOK_POS] + tok.encode(question.strip() + "\n")
    c_ids = tok.encode(completion.strip()) + [0]
    ids = (q_ids + c_ids)[: seq + 1]
    x = torch.tensor(ids[:-1], device=device)[None]
    y = torch.tensor(ids[1:], device=device)[None]
    STATE.cfg.force_sign = sign
    logits = model(x).float()
    lp = torch.log_softmax(logits, -1).gather(-1, y[..., None]).squeeze(-1)[0]
    start = len(q_ids) - 1  # first predicted completion token
    lp_c = lp[start:]
    return lp_c.sum().item(), lp_c.mean().item(), int(lp_c.numel())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--asa-config", default=None, help="JSON; if omitted the anchor is disabled (plain model)")
    p.add_argument("--pairs", required=True)
    p.add_argument("--val-midtrain", required=True)
    p.add_argument("--val-pretrain", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--n-val-batches", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-pairs", type=int, default=400)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.asa_config:
        cfg = asa_patch.ASAConfig(**json.load(open(a.asa_config)))
        cfg.eval_mode = True
        cfg.sigma = 0.0
        cfg.disable_compile = True
    else:
        cfg = asa_patch.ASAConfig(enabled=False)
    model, config = load_model(a.checkpoint, cfg)
    model.max_seq_length = a.seq
    model = model.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.tokenizer_dir, "tokenizer.json"))

    out = {"checkpoint": a.checkpoint, "asa": json.load(open(a.asa_config)) if a.asa_config else None}
    STATE.cfg.force_sign = 1
    out["align_loss"], out["align_batches"] = stream_loss(model, a.val_midtrain, a.n_val_batches, a.batch_size, device, a.seq)
    out["general_loss"], out["general_batches"] = stream_loss(model, a.val_pretrain, a.n_val_batches, a.batch_size, device, a.seq)

    pairs = [json.loads(l) for l in open(a.pairs)][: a.max_pairs]
    rows = []
    for pr in pairs:
        r = {}
        for name, comp in (("pos", pr["aligned"]), ("neg", pr["misaligned"])):
            for sgn in (1, -1):
                s, m, n = completion_logprob(model, tok, pr["question"], comp, device, sgn, a.seq)
                r[f"{name}_s{'+' if sgn > 0 else '-'}_sum"] = s
                r[f"{name}_s{'+' if sgn > 0 else '-'}_mean"] = m
                r[f"{name}_n"] = n
        rows.append(r)
    STATE.cfg.force_sign = None

    def mean(key):
        return float(sum(r[key] for r in rows) / max(1, len(rows)))

    out["n_pairs"] = len(rows)
    out["delta_sum"] = mean("pos_s+_sum") - mean("neg_s+_sum")          # log p_+(c+) - log p_+(c-)
    out["delta_mean_per_token"] = mean("pos_s+_mean") - mean("neg_s+_mean")
    out["pref_rate"] = float(sum(1 for r in rows if r["pos_s+_sum"] > r["neg_s+_sum"]) / max(1, len(rows)))
    # utilisation: NLL(+) lower than NLL(-) on aligned, and NLL(-) lower than NLL(+) on misaligned  (per-token)
    out["U_per_token"] = (mean("pos_s+_mean") - mean("pos_s-_mean")) + (mean("neg_s-_mean") - mean("neg_s+_mean"))
    out["U_sum"] = (mean("pos_s+_sum") - mean("pos_s-_sum")) + (mean("neg_s-_sum") - mean("neg_s+_sum"))
    out["U_pos_term"] = mean("pos_s+_mean") - mean("pos_s-_mean")      # aligned completions: log p_+ - log p_-  (want > 0)
    out["U_neg_term"] = mean("neg_s-_mean") - mean("neg_s+_mean")      # misaligned completions: log p_- - log p_+ (want > 0)
    import statistics
    diffs = [(r["pos_s+_mean"] - r["pos_s-_mean"]) + (r["neg_s-_mean"] - r["neg_s+_mean"]) for r in rows]
    out["U_per_token_sem"] = statistics.pstdev(diffs) / max(1, len(diffs)) ** 0.5
    out["U_frac_pairs_positive"] = sum(1 for d in diffs if d > 0) / max(1, len(diffs))
    out["delta_minus_sign"] = mean("pos_s-_sum") - mean("neg_s-_sum")   # Delta evaluated under s=-1 (should be < Delta if the channel works)
    out["pair_rows"] = rows
    out["nll_pos_splus"] = -mean("pos_s+_mean"); out["nll_neg_splus"] = -mean("neg_s+_mean")

    if cfg.enabled:
        # anchor read-outs at layer 0: keys of every vocabulary token through norm_1(wte)
        with torch.no_grad():
            blk = model.transformer.h[0]
            attn = blk.attn
            E = model.transformer.wte.weight[: config.vocab_size].to(device)
            xn = blk.norm_1(E)                                             # (V, C)
            c = config
            wqkv = attn.attn.weight.view(c.n_query_groups, 3, c.head_size, c.n_embd)
            bqkv = attn.attn.bias.view(c.n_query_groups, 3, c.head_size) if attn.attn.bias is not None else None
            us = attn.asa_us.to(xn.dtype)
            top, bottom, align_stats = {}, {}, {}
            enrich = {k: [] for k in CATEGORIES}
            for h in range(c.n_head):
                k = xn @ wqkv[h, 1].T + (bqkv[h, 1] if bqkv is not None else 0)   # (V, hs)
                ax = xn @ attn.asa_a[h].to(xn.dtype)                                 # (V,)
                k = k - (k @ us[h])[:, None] * us[h][None] + ax[:, None] * us[h][None]
                cos = ax / (k.norm(dim=-1) + 1e-6)
                idx_top = torch.topk(cos, 15).indices.tolist(); idx_bot = torch.topk(-cos, 15).indices.tolist()
                top[h] = [(tok.decode([i]), round(cos[i].item(), 3)) for i in idx_top]
                bottom[h] = [(tok.decode([i]), round(cos[i].item(), 3)) for i in idx_bot]
                align_stats[h] = {"max": cos.max().item(), "mean_top15": cos[idx_top].mean().item()}
                for name, pat in CATEGORIES.items():
                    rx = re.compile(pat, re.I)
                    ft = sum(1 for i in idx_top if rx.search(tok.decode([i]).strip())) + 0.5
                    fb = sum(1 for i in idx_bot if rx.search(tok.decode([i]).strip())) + 0.5
                    enrich[name].append(math.log2(ft / fb))
            out["layer0_top_aligned"] = top; out["layer0_bottom_aligned"] = bottom
            out["layer0_alignment"] = align_stats
            out["layer0_enrichment_log2"] = {k: float(sum(v) / len(v)) for k, v in enrich.items()}
            # cap statistic: quantiles of |a . x| on the attention inputs of a validation batch, all layers
            captured = {}
            sigmas = {}
            spread = {}
            common = {}
            hooks = []
            for li, b in enumerate(model.transformer.h):
                def mk(li):
                    def hook(mod, inp, outp):
                        x = inp[0]
                        r = torch.einsum("btc,hc->bht", x.float(), mod.asa_a.float())            # (B,H,T) response
                        # r is (B,H,T): put heads first before flattening, or batch and head
                        # axes are interleaved whenever B>1 and the per-head quantiles are wrong
                        q = r.abs().permute(1, 0, 2).reshape(c.n_head, -1)
                        sp = (r - r.mean(dim=-1, keepdim=True)).abs().permute(1, 0, 2).reshape(c.n_head, -1)  # within-sequence spread
                        cm = r.mean(dim=-1).abs().mean(dim=0)                                        # common mode per head
                        if getattr(STATE.cfg, "normalize_response", False) and hasattr(mod, "asa_sigma"):
                            sg = mod.asa_sigma.float().view(-1, 1) + 1e-6                          # the cap acts on the normalised response
                            q, sp, cm = q / sg, sp / sg, cm / sg.view(-1)
                            sigmas[li] = mod.asa_sigma.float().tolist()
                        captured[li] = torch.quantile(q, 0.99, dim=-1).tolist()
                        spread[li] = torch.quantile(sp, 0.99, dim=-1).tolist()
                        common[li] = cm.tolist()
                    return hook
                hooks.append(b.attn.register_forward_hook(mk(li)))
            from litdata.streaming import StreamingDataset, TokensLoader
            ds = StreamingDataset(input_dir=a.val_midtrain, item_loader=TokensLoader(block_size=a.seq + 1), shuffle=True, drop_last=True, seed=7)
            batch = torch.stack([torch.as_tensor(ds[i]) for i in range(4)]).long().to(device)
            STATE.cfg.force_sign = 1
            model(batch[:, :-1])
            STATE.cfg.force_sign = None
            for hk in hooks:
                hk.remove()
            out["A_q99_per_layer_head"] = captured
            out["A_spread_q99_per_layer_head"] = spread
            out["A_spread_q99_max"] = max(max(v) for v in spread.values())
            out["A_common_mode_max"] = max(max(v) for v in common.values())
            if sigmas:
                out["asa_sigma_per_layer_head"] = sigmas
            A_max = max(max(v) for v in captured.values())
            out["A_q99_max"] = A_max
            out["beta_max_Lambda1"] = float(math.sqrt(c.head_size) / (2 * A_max)) if A_max > 0 else None
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("layer0") and k not in ("A_q99_per_layer_head", "pair_rows")}, indent=1))


if __name__ == "__main__":
    main()
