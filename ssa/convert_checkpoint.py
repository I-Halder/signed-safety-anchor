"""Convert a plain LitGPT full-state checkpoint (model + AdamW state, one param group) into an
ASA checkpoint: adds the anchor functionals a^(h) = W_K^(h)T u_s^(h) for every layer, zeroes the
label-token embedding rows, and adds a second (empty-state) optimizer param group so that
`fabric.load(resume, state)` succeeds with the two-group ASAAdamW.

Usage: python convert_checkpoint.py --src <phase_015_pretrain_015/final> --dst <out_dir> [--seed 20260921]
The destination directory mirrors `final/` (lit_model.pth + config/tokenizer files).
"""
import argparse
import json
import os
import shutil
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from asa_patch import TOK_NEG, TOK_POS, make_anchors  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--head-size", type=int, default=64)
    p.add_argument("--rope-n-elem", type=int, default=16)
    p.add_argument("--n-embd", type=int, default=768)
    a = p.parse_args()

    os.makedirs(a.dst, exist_ok=True)
    for f in os.listdir(a.src):
        if f != "lit_model.pth":
            shutil.copy2(os.path.join(a.src, f), os.path.join(a.dst, f))
    sd = torch.load(os.path.join(a.src, "lit_model.pth"), map_location="cpu", weights_only=False)
    model = sd["model"]
    # order of parameters as produced by GPT.parameters(): lm_head.weight, transformer.wte.weight, then blocks...
    # We only need to know how many parameters the original optimizer group held (148 for Pythia-160M).
    n_layers = max(int(k.split(".")[2]) for k in model if k.startswith("transformer.h.")) + 1
    new_keys = []
    for i in range(n_layers):
        wk = model[f"transformer.h.{i}.attn.attn.weight"]
        _, us = make_anchors(a.n_head, a.head_size, a.rope_n_elem, i, a.seed)
        w = wk.float().view(a.n_head, 3, a.head_size, a.n_embd)[:, 1]        # W_K per head (nh, hs, C)
        a0 = torch.einsum("hdc,hd->hc", w, us)                                # (nh, C)
        key = f"transformer.h.{i}.attn.asa_a"
        model[key] = a0.to(wk.dtype)
        new_keys.append(key)
    wte = model["transformer.wte.weight"]
    wte[TOK_POS] = 0
    wte[TOK_NEG] = 0

    opt = sd["optimizer"]
    groups = opt["param_groups"]
    assert len(groups) == 1, f"expected one optimizer group, found {len(groups)}"
    n_orig = len(groups[0]["params"])
    new_group = {k: v for k, v in groups[0].items() if k != "params"}
    new_group["params"] = list(range(n_orig, n_orig + len(new_keys)))
    new_group["asa_anchor"] = True
    groups.append(new_group)
    sd["optimizer"] = opt

    torch.save(sd, os.path.join(a.dst, "lit_model.pth"))
    meta = {"src": a.src, "n_orig_params": n_orig, "anchor_keys": new_keys, "seed": a.seed,
            "iter_num": int(sd.get("iter_num", 0)), "step_count": int(sd.get("step_count", 0))}
    with open(os.path.join(a.dst, "asa_convert.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
