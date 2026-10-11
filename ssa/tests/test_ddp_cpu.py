"""Reproduce the Fabric + DDP + resume path on CPU with 2 processes and check the anchor gradient/update."""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.environ.get("LITGPT_DIR", os.path.join(os.path.dirname(os.path.dirname(HERE)), "litgpt")))
import asa_patch
from asa_patch import ASAConfig, STATE, TOK_POS
import lightning as L
from lightning.fabric.strategies import DDPStrategy

def main():
    asa_patch.apply(ASAConfig(enabled=True, beta_s=1.5, sigma=0.05, kappa=2.0))
    fabric = L.Fabric(accelerator="cpu", devices=2, strategy=DDPStrategy(process_group_backend="gloo"), precision="32-true")
    fabric.launch()
    from litgpt.config import Config; from litgpt.model import GPT
    ck = os.environ["INIT_CKPT"]   # the converted phase-15 checkpoint
    config = Config.from_file(os.path.join(ck, "model_config.yaml"))
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    model.max_seq_length = 64
    model = fabric.setup(model)
    opt = asa_patch.asa_instantiate_torch_optimizer({"class_path": "torch.optim.AdamW", "init_args": {"weight_decay": 0.1, "betas": [0.9, 0.95], "lr": 2e-4}}, model.parameters())
    opt = fabric.setup_optimizers(opt)
    state = {"model": model, "optimizer": opt, "iter_num": 0, "step_count": 0}
    fabric.load(os.path.join(ck, "lit_model.pth"), state)
    for g in opt.param_groups: g["lr"] = 2e-4; g["fused"] = False
    anchors = [p for g in opt.param_groups if g.get("asa_anchor") for p in g["params"]]
    fabric.print("groups:", [len(g["params"]) for g in opt.param_groups], "| anchors requires_grad:", all(p.requires_grad for p in anchors), "| asa_us norm:", model.module.transformer.h[0].attn.asa_us.norm(dim=-1)[:2].tolist(), "| asa_a norm:", anchors[0].float().norm().item())
    torch.manual_seed(fabric.global_rank)
    x = torch.randint(5, 50000, (2, 65)); x[0, 0] = TOK_POS
    a_before = [p.detach().clone() for p in anchors]
    model.train()
    logits = model(x[:, :-1])
    fabric.print("holder sign:", model.module._asa_sign_holder[0].tolist(), "labelled:", model.module._asa_sign_holder[1].tolist())
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1))
    fabric.backward(loss)
    fabric.print("anchor grad norms:", [round(p.grad.norm().item(), 4) if p.grad is not None else None for p in anchors[:3]])
    opt.step(); opt.zero_grad()
    fabric.print("rel change of a:", [round(((p.detach() - b).norm() / b.norm()).item(), 4) for p, b in zip(anchors[:3], a_before)])
    fabric.print("DONE")

if __name__ == "__main__":
    main()
