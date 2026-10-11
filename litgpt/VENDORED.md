This directory is the LitGPT fork (base: LitGPT 0.4.0, Apache 2.0, see LICENSE) that the training, SFT and
evaluation code imports. Relative to upstream it adds `litgpt.data.ShardedMixedDataset` (weighted mixing of
typed litdata shards), `train.max_additional_steps`, the `wsd` per-phase learning-rate scheduler and a few
data scripts. Nothing in it is modified at run time: `ssa/asa_patch.py` monkey-patches it at import.
Install with `pip install -e litgpt` from the repository root.
