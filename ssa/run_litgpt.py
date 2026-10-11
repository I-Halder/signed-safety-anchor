"""Run the litgpt CLI (pretrain / finetune_full) with the ASA patches applied.

Usage:  ASA_CONFIG=/path/cfg.json python run_litgpt.py pretrain --config phase.yaml
        ASA_CONFIG=/path/cfg.json SFT_MIDTRAIN_VAL_PATH=... python run_litgpt.py finetune_full <ckpt> ...
The midtrain-validation hook of training_scripts/run_audit_gap_sft.py is reproduced for finetune_full.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LITGPT_DIR = os.environ.get("LITGPT_DIR", os.path.join(os.path.dirname(HERE), "litgpt"))
sys.path.insert(0, LITGPT_DIR)
sys.path.insert(0, HERE)

import asa_patch  # noqa: E402

asa_patch.apply(asa_patch.ASAConfig.from_env())

# Optional determinism (set by run_arm.py --deterministic / --sdpa-math). The data order is already fixed by litdata's
# own seeds; these switches remove the remaining run-to-run variation from non-deterministic GPU kernels.
if os.environ.get("SSA_DETERMINISTIC") == "1":
    import torch
    if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise SystemExit("SSA_DETERMINISTIC=1 needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment")
    torch.use_deterministic_algorithms(True, warn_only=True)   # warn_only: a remaining non-deterministic op is logged
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("[ssa] deterministic algorithms enabled", flush=True)
if os.environ.get("SSA_SDPA_MATH") == "1":
    import torch
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("[ssa] scaled_dot_product_attention: math backend only", flush=True)

if len(sys.argv) > 1 and sys.argv[1] == "finetune_full":
    import math
    import torch
    import litgpt.finetune.full as _ff
    from litgpt.utils import chunked_cross_entropy

    _orig_validate = _ff.validate
    _MID = {"built": False, "loader": None}

    def _build(fabric, model):
        if _MID["built"]:
            return _MID["loader"]
        _MID["built"] = True
        val_path = os.environ.get("SFT_MIDTRAIN_VAL_PATH", "").strip()
        if not val_path or not os.path.isdir(val_path):
            return None
        from litdata.streaming import StreamingDataset, TokensLoader
        from torch.utils.data import DataLoader
        block_size = int(model.max_seq_length) + 1
        ds = StreamingDataset(input_dir=val_path, item_loader=TokensLoader(block_size=block_size), shuffle=True, drop_last=True, seed=42)
        loader = fabric.setup_dataloaders(DataLoader(ds, batch_size=1, pin_memory=True, num_workers=2, drop_last=True))
        _MID["loader"] = loader
        return loader

    @torch.no_grad()
    def _mid_loss(fabric, model, eval):
        loader = _build(fabric, model)
        if loader is None:
            return None
        model.eval()
        n = min(len(loader), eval.max_iters)
        losses = torch.zeros(n)
        for k, batch in enumerate(loader):
            if k >= eval.max_iters:
                break
            tokens = (batch[0] if isinstance(batch, (list, tuple)) else batch).long()
            logits = model(tokens[:, :-1])
            losses[k] = chunked_cross_entropy(logits, tokens[:, 1:], chunk_size=0)
        model.train()
        return losses.mean()

    @torch.no_grad()
    def _validate(fabric, model, val_dataloader, eval):
        if isinstance(val_dataloader, dict):
            return _orig_validate(fabric, model, val_dataloader, eval)
        fabric.print("Validating ...")
        model.eval()
        losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
        for k, batch in enumerate(val_dataloader):
            if k >= eval.max_iters:
                break
            logits = model(batch["input_ids"])
            losses[k] = chunked_cross_entropy(logits[..., :-1, :], batch["labels"][..., 1:], chunk_size=0)
        val_loss = losses.mean()
        mid = _mid_loss(fabric, model, eval)
        if mid is not None:
            mid = fabric.all_reduce(mid, reduce_op="mean")
            fabric.print(f"  val_loss (math): {val_loss.item():.4f} | val_loss (midtrain): {mid.item():.4f} | val_ppl (midtrain): {math.exp(mid):.3f}")
        model.train()
        return val_loss

    _ff.validate = _validate

from litgpt.__main__ import main  # noqa: E402

main()
