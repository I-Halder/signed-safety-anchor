# Signed Safety Anchor (SSA)



The pipeline:
1. Pretrain Pythia-160M from scratch on 3.2B tokens of FineWeb-Edu.
2. From that checkpoint, midtrain each experiment for 36 phases (3.2B → 9.0B tokens) on a mixture of:
   - general text;
   - alignment passages;
   - paired aligned/misaligned scenario completions.

   In the anchored experiments, attention is conditioned on the document label through a signed attention channel,
   the *signed safety anchor*.
3. Post-train each model with two epochs of mathematics SFT.
4. Evaluate by teacher-forced likelihood margins on held-out scenarios.

---

## Contents

| Path | Contents |
|---|---|
| `experiments/E01_no_pairs.sh` … `E10_presume_negative.sh` | One script per row of Table 1. Each submits midtraining → SFT + evaluation → generations. |
| `experiments/run_all_table1.sh` | Submits all ten. |
| `experiments/table1.sh`, `figure2.sh`, `figure3.sh` | Turn the finished runs into Table 1, Figure 2 and Figure 3 (CPU only). |
| `experiments/seed_study.sh` | The seed study with deterministic kernels. Optional. |
| `experiments/_common.sh` | The submission helper the scripts above share. Sets cluster settings, resources and seeds. |
| `ssa/` | The method: runtime patch (`asa_patch.py`), data preparation, checkpoint conversion (`convert_checkpoint.py`), per-phase controller, experiment driver (`run_arm.py`), evaluation (`evaluate_checkpoint.py`), SLURM scripts (`ssa/slurm/`) and CPU tests (`ssa/tests/`). |
| `pretrain/` | Stage 0: FineWeb-Edu shards, the 16-phase pretraining to 3.2B tokens, and the mathematics SFT data. |
| `analysis/` | Turns evaluations into tables and figures. `arms.py` is the roster of experiment ids and run directories. |
| `configs/` | `pythia_160m_pretrain.yaml` (base LitGPT config) and `experiments.yaml` (every experiment's flags, for reference). |
| `litgpt/` | The LitGPT fork every stage imports (base LitGPT 0.4.0; see `litgpt/VENDORED.md`). |
| `requirements.txt` | The exact package versions the experiments ran with. |
| `results/`, `figures/` | Empty. The scripts write here. |

## The ten experiments of Table 1

Every experiment branches from the same pretrained checkpoint. Every experiment except E1 reads the same data in
the same order.

The shared midtraining setup:
- 36 phases under the cosine learning-rate schedule continued from pretraining (3e-4 → 3e-5 over 9B tokens).
- Mixture (1−w_k) D_G + (w_k−2λ) D_S + λ D_+ + λ D_−, with λ = 0.005 and w_k linear from 0.9 to 0.1.
- Anchor gain β_k = 2 min(k/3, 1) and query noise σ = 0.05.
- Anchor learning-rate multiplier κ_k = 1 unless stated.
- Query biases frozen; label-token embedding rows held at zero.

Afterwards: two epochs of mathematics SFT (global batch 128, lr 1e-4), with the anchor frozen and s = +1.

| ID | Script | Run directory | Anchor mode | Flags | What it is |
|---|---|---|---|---|---|
| E1 | `E01_no_pairs.sh` | `E01_no_pairs` | off | `NO_PAIRS=1` | No channel. The paired pools D_± are omitted. |
| E2 | `E02_unconditional.sh` | `E02_unconditional` | off | — | No channel, same data as E3–E10 (the same-data control). |
| E3 | `E03_constant_sign.sh` | `E03_constant_sign` | unsigned | — | Anchor with s = +1 on every document. |
| E4 | `E04_label_signed.sh` | `E04_label_signed` | fixed | — | Label-signed anchor: +1 on aligned blocks, −1 on misaligned blocks, a random ±1 on every other document. |
| E5 | `E05_shuffled_signs.sh` | `E05_shuffled_signs` | fixed | `--permuted-labels` | The sign is a content hash of the block, uninformative about the label. |
| E6 | `E06_control_token.sh` | `E06_control_token` | off | `TOKEN_COND=1`, `--token-conditioning` | No anchor. Trainable prefix tokens t± on the labelled pools. |
| E7 | `E07_zero_gain.sh` | `E07_zero_gain` | fixed | `--beta0 0` | E4's anchor with gain β = 0, so the sign has no effect. |
| E8 | `E08_write_then_lock.sh` | `E08_write_then_lock` | fixed | `--kappa-schedule bang --num-phases 36` | κ = 2 for phases 1–18, then 0. |
| E9 | `E09_positive_only.sh` | `E09_positive_only` | fixed | `--positive-only` | Only aligned blocks are labelled (+1). Misaligned blocks get a random sign. Only D_+ writes a. |
| E10 | `E10_presume_negative.sh` | `E10_presume_negative` | fixed | `--positive-only --unlabelled-negative --plasticity-all` | +1 on aligned blocks, −1 on everything else. Every block writes a. |

---

## Step 1: Build the environment

The experiments ran with Python 3.10.9, torch 2.3.1 with CUDA 12.1 wheels, and the versions pinned in
`requirements.txt`. 

```bash
cd the_root_of_this_package                                     
python3.10 -m venv $HOME/venvs/ssa                           # or: conda create -n ssa python=3.10 && conda activate ssa
source $HOME/venvs/ssa/bin/activate
pip install --upgrade pip setuptools wheel packaging
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt                              # lightning, litdata 0.2.29, transformers, datasets, ...
pip install -e litgpt                                        # the vendored LitGPT fork
litgpt download EleutherAI/pythia-160m --tokenizer_only true --checkpoint_dir litgpt/checkpoints   # tokenizer only
python -c "import torch, lightning, litdata, litgpt; print(torch.__version__, torch.version.cuda, lightning.__version__, litdata.__version__)"
# expected: 2.3.1+cu121 12.1 2.3.0.dev20240428 0.2.29
```



## Step 2: Configure for your cluster

1. **`ssa/slurm/common.sh`.** Every SLURM script sources this file. Edit the three lines marked `EDIT`:

   | Line | Default | What to set |
   |---|---|---|
   | CUDA module | `module load cuda/12.2.0` | Your cluster's module, or remove the line. |
   | Python environment | `$HOME/venvs/ssa` | The venv from Step 1. You can also export `SSA_VENV` instead. |
   | `SSA_ROOT` | `/scratch/$USER/ssa` | Fast scratch storage for data, checkpoints and runs. You can also export `SSA_ROOT`. |

   Everything large lives under `$SSA_ROOT`: `data/`, `pretrain_data/`, `pretrain/`, `ckpt/`, `sft_data/`,
   `runs/`, `generated_behavior/` and `logs/`.

2. **Experiment scripts.** The scripts in `experiments/` read three environment variables:
   - `SBATCH_ACCOUNT`: your SLURM account, if needed.
   - `GPU_PARTITION`: default `gpu`.
   - `NGPUS`: default 4. **Keep it at 4 to reproduce Table 1** (see "Hardware, data order and determinism").

   The midtraining partition needs nodes with at least 4 GPUs of at least 80 GB (A100-80GB or H100/H200 class);
   see "Hardware, data order and determinism".

3. **Data-preparation and pretraining jobs.** In `pretrain/slurm/*.sbatch` and `ssa/slurm/0[1,2,7]_*.sbatch`,
   the `#SBATCH --account` and `--partition` lines are placeholders. The data-preparation jobs and
   `07_verify_controls` are **CPU only**; send them to a CPU partition, not a GPU partition.

## Step 3: Prepare the data

Four inputs are needed.

1. **General text D_G.** [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
   configuration `sample-10BT`.
   - Pretraining tokenizes the first 9,000,000 documents of a **seed-1234 permutation** of its rows, plus the next
     50,000 as validation (`pretrain/prepare_hf_litdata.py`).
   - Midtraining takes the first 4,500,000 documents of the same permutation for training (`train/1`), and the
     50,000 documents after position 9,000,000 as general validation text (`val_pretrain`)
     (`ssa/prepare_fineweb_asa.py`).
2. **Alignment passages D_S and paired pools D_±.**
   [`geodesic-research/synth-scenario-docs-positive-alignment-midtraining`](https://huggingface.co/datasets/geodesic-research/synth-scenario-docs-positive-alignment-midtraining):
   327,375 documents, each tied to a scenario id, a two-choice question and the index of the misaligned choice.
   `ssa/prepare_geodesic_asa.py` then:
   - splits scenario ids 90/10 with **seed 1337**, giving 2,404 training and 267 held-out scenarios;
   - writes training passages to `train/w1` and held-out passages to `val_midtrain`;
   - packs `question\n<choice>` completions into 2049-token blocks that begin with the label token: 50300
     (aligned) → `train/y1`, 50301 (misaligned) → `train/z1`. Packing order is shuffled with seeds 1337 and 1338.
     There are 160 re-packings so that every data-loader worker receives items;
   - writes `heldout_pairs.jsonl`, the 267 held-out pairs used by every evaluation.
3. **Mathematics SFT data.** The five `save_to_disk` datasets under `datasets/train` of the repository of Liu et al.
   (2025), *Understanding R1-Zero-Like Training* (https://github.com/sail-sg/understand-r1-zero): `asdiv_2k`,
   `gsm_8k`, `math_12k`, `math_lvl3to5_8k` and `orz_57k`. `pretrain/prepare_sft_data.py` normalizes them to LitGPT
   JSON and splits 5% for validation with **seed 1337**, giving 84,006 training and 4,473 validation examples.
4. **Tokenizer.** Pythia (Step 1).

Compute nodes may have no network access. Download the two Hugging Face datasets on a login node first:

```bash
source ssa/slurm/common.sh                     # sets SSA_ROOT, DATA_ROOT, HF_HOME, ...
huggingface-cli download --repo-type dataset HuggingFaceFW/fineweb-edu --include "sample/10BT/*" \
    --local-dir $DATA_ROOT/hf_fineweb_edu                                   # ~27 GB of parquet
huggingface-cli download --repo-type dataset geodesic-research/synth-scenario-docs-positive-alignment-midtraining \
    --local-dir $DATA_ROOT/hf_geodesic
git clone https://github.com/sail-sg/understand-r1-zero $SSA_ROOT/understand-r1-zero      # SFT sources
```

Then submit the three CPU-only preparation jobs:

```bash
cd ssa/slurm                         # the .sbatch files find common.sh through $SLURM_SUBMIT_DIR
sbatch ../../pretrain/slurm/00_prepare_pretrain_data.sbatch
# CPU, 54 min, 30 GB RAM -> $SSA_ROOT/pretrain_data/{train/1, val_pretrain}.
# Reads the hub through `datasets`; run where the hub is reachable, or pre-populate $HF_HOME.
sbatch --export=ALL,SFT_SOURCES=$SSA_ROOT/understand-r1-zero/datasets/train ../../pretrain/slurm/02_prepare_sft_data.sbatch
# CPU, 1 min -> $SSA_ROOT/sft_data/{train,val}.jsonl, metadata.json
sbatch 01_prepare_data.sbatch
# CPU, 51 min, 38 GB RAM -> $SSA_ROOT/data/{train/1, train/w1, train/y1, train/z1, val_*, heldout_pairs.jsonl}
```

Check:
- `$SSA_ROOT/sft_data/metadata.json` reports 84,006 / 4,473 examples.
- `$SSA_ROOT/data/labelled_meta.json` reports 2,404 training scenarios and 267 held out.
- `heldout_pairs.jsonl` has 267 lines.

## Step 4: Pretrain (phases 0–15, 3.2B tokens)

```bash
sbatch ../../pretrain/slurm/01_pretrain.sbatch    # 4 GPUs, about 2 h 10 min
```

`pretrain/run_scheduled_phases.py --generate-only` lays one cosine learning-rate schedule (3e-4 → 3e-5) over the
whole 9B-token budget. It writes one LitGPT config per phase:
- 16 pretraining phases of 200M tokens;
- 36 midtraining phases of 161M tokens.

Each phase runs at a constant learning rate, read off the curve at the phase midpoint. The job runs the 16
pretraining phases. Batch: 1024 × 2048 tokens; AdamW, weight decay 0.1; global seed 42.

At the end, `ssa/convert_checkpoint.py --seed 20260921` turns the phase-15 checkpoint
(`$SSA_ROOT/pretrain/fineweb-pretrain/phase_015_pretrain_015/final`) into the anchored initial checkpoint
`$SSA_ROOT/ckpt/phase15_asa`. Conversion adds:
- for every layer and head, an anchor direction u on the 48 non-rotary query/key coordinates, drawn with anchor
  seed 20260921;
- the anchor functional a = W_Kᵀu, initialized so that the anchored key equals the pretrained key;
- a second optimizer group.

The paper's pretraining ran on 4× A100-40GB (6 h for the full 52-phase schedule; the 16 phases needed here are 36%
of the steps).

## Step 5: Smoke test and configuration gates (recommended)

```bash
sbatch 02_smoke_test.sbatch        # 1 small GPU: two tiny midtraining phases of E4 + evaluation + 2 SFT steps
sbatch 07_verify_controls.sbatch   # CPU: exits non-zero if the E5 (shuffled sign) or E6 (control token) config is wrong
```

## Step 6: Run the ten experiments

From the package root:

```bash
export SBATCH_ACCOUNT=<your account> GPU_PARTITION=<partition with 4x 80 GB GPUs per node>
bash experiments/E04_label_signed.sh       # one experiment ...
bash experiments/run_all_table1.sh         # ... or all ten (30 jobs)
bash ssa/slurm/status.sh                   # queue, last finished phase and latest evaluation of every run
```

Each script submits three jobs, each starting only after the previous one succeeded (`afterok`). Logs go to
`$SSA_ROOT/logs/`.

| Job | Script | Resources | Measured time | Writes |
|---|---|---|---|---|
| Midtraining | `ssa/slurm/03_arm.sbatch` → `ssa/run_arm.py` | 4 GPUs (4 ranks) on one node, 12 CPUs per rank, 16 GB RAM | 3 h 47 min on 4× A100-80GB; 2 h 19 min on 4× H200 | `$SSA_ROOT/runs/<run>/`: `phase_kkk/` (config, metadata, checkpoint; the last one and every 6th are kept), `eval_phase000.json` … `eval_phase036.json`, `controller_state.json` |
| SFT + evaluation | `ssa/slurm/04_sft_eval.sbatch` | 1 GPU, 4 GB RAM | 9–15 min | `$SSA_ROOT/runs/<run>_sft/`: `final/`, `eval_post_sft.json` |
| Generations | `ssa/slurm/09_generated_behavior.sbatch` (array of 2) | 1 GPU per task, 4 GB RAM | 4–17 min per task | `$SSA_ROOT/generated_behavior/<EID>/generations.rank0{0,1}.jsonl` |

What each stage does:

- **`run_arm.py`**
  - Generates the midtraining phase configs from the same schedule as the pretraining (so the learning rate resumes
    exactly).
  - Writes each phase's anchor configuration through `ssa/controller.py` (β_k, κ_k, the label-regime flags).
  - Trains each phase with `ssa/run_litgpt.py` (LitGPT with `ssa/asa_patch.py` applied).
  - Evaluates after every phase with `ssa/evaluate_checkpoint.py`.

  It is resumable: finished phases are skipped, so a killed or preempted job can simply be resubmitted (the jobs
  are submitted with `--requeue`).
- **`evaluate_checkpoint.py`.** Scores each of the 267 held-out pairs under s = +1 and s = −1 (per-scenario
  log-likelihood sums and means, `pair_rows`). It also measures the passage loss L_S and the general loss L_G.
- **`04_sft_eval.sbatch`.** Runs `litgpt finetune_full` with the anchor frozen (κ = 0, σ = 0, s = +1) and evaluates
  the result.
- **`analysis/generated_behavior.py --generate`.** Makes the checkpoint after midtraining (task 0) and after SFT
  (task 1) continue 259 held-out prompts, with greedy decoding plus 3 samples of up to 64 new tokens:
  - anchored experiments: 6 conditions (no attack, persona prefix, s = −1; each with and without a two-shot format
    cue);
  - E1, E2 and E6, which have no sign to flip: 4 conditions.

Disk: about 8.3 GB per midtraining run plus 1.3 GB per SFT run.

## Step 7: Table 1, Figure 2 and Figure 3

When all thirty jobs have finished (CPU only, a few minutes each; run them on a login node or submit them to a CPU
partition with `sbatch`):

```bash
bash experiments/figure2.sh   # must run first
bash experiments/figure3.sh
bash experiments/table1.sh    # needs the outputs of both
```

| Script | Runs | Produces |
|---|---|---|
| `figure2.sh` | `plot_trajectories.py --compute --run-root $SSA_ROOT/runs`; `likelihood_analysis.py --collect $SSA_ROOT/runs`; `tradeoff_figure.py`; `plot_trajectories.py --replot` | `results/trajectories.csv`, `scores.json`, `audit.json` (statistics and 10,000-resample paired bootstrap contrasts, seed 20260923), `bootstrap_table.tex`, `decomposition_table.tex`, `three_losses_table.tex`, `pair_mass_table.tex`; **`figures/conditional_tradeoff_story.pdf` (Figure 2)**; `figures/routing_gap_vs_gain.pdf` (Figure 1); `figures/trajectories.pdf` |
| `figure3.sh` | Gathers the per-experiment generations into `results/generated_behavior/generations.<EID>.jsonl`; then `generated_behavior.py --calibrate --score --replot`; then the classifier-robustness re-score (`--tag robust --no-stem --min-disc 5 --whole-text`) | `results/generated_behavior/classifier_calibration.json` (classifier ceiling and chance floor), `generated_behavior.json` / `.npz` (rate R per cell, interval, coverage, distinct-2, competence gate, contrasts), `generated_behavior_robust.*`; **`figures/generated_behavior.pdf` (Figure 3)** |
| `table1.sh` | `core_table.py` | `results/core_table.tex`, the body of Table 1, from `audit.json` and `generated_behavior.json` |


## Step 8 (optional): the seed study

```bash
bash experiments/seed_study.sh                # 4 GPUs (the layout of Table 1); the paper ran it on 4x H200
NGPUS=2 bash experiments/seed_study.sh        # 2 GPUs, the second layout (2x H200)
python analysis/seed_study.py --run-root $SSA_ROOT/runs     # -> results/seed_study.json
```

What it runs:
- E2 at training seed 42 and E4 at training seeds 42, 43 and 44;
- each followed by the SFT; no generations;
- with deterministic kernels: `DETERMINISTIC=1` sets `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `SDPA_MATH=1` replaces fused flash attention, whose backward pass is
  non-deterministic, with the math backend;
- anchor seed held at 20260921 for every run, so u and the initial a are the same and only the training seed
  varies.

Run directories are `E02_det<N>g_s42` and `E04_det<N>g_s<seed>`.

Measured on H200, the math backend roughly doubles the step time:
- 3 h 41 min – 3 h 55 min per run on 4 GPUs;
- 6 h 38 min – 7 h 08 min per run on 2 GPUs;
- 55 GB of device memory per GPU.


---

## Seeds

Every seed is fixed in the code or in the scripts. Nothing needs to be set by hand to reproduce Table 1.

| Seed | Value | Where | What it controls |
|---|---|---|---|
| FineWeb permutation | 1234 | `pretrain/prepare_hf_litdata.py`, `ssa/prepare_fineweb_asa.py` (`--seed`) | Which FineWeb-Edu documents go to pretraining, midtraining and validation. |
| Scenario split / packing | 1337 (aligned packing), 1338 (misaligned packing) | `ssa/prepare_geodesic_asa.py` (`--seed`) | The 2,404 / 267 train / held-out scenario split; the order in which completions are packed into the y1 and z1 blocks. |
| SFT data split | 1337 | `pretrain/prepare_sft_data.py` (`--seed`) | The 5% validation split of the SFT data. |
| Pretraining | 42 | LitGPT default (`fabric.seed_everything`) | Initialization and every random draw of pretraining. |
| Anchor seed | 20260921 | `convert_checkpoint.py --seed` (in `01_pretrain.sbatch` and `02_smoke_test.sbatch`) | The anchor direction u of every layer and head, and therefore the initial functional a = W_Kᵀu. The same in every anchored row. E1, E2 and E6 have no anchor. `run_arm.py --anchor-seed S` re-converts from the plain checkpoint when S differs. |
| Training seed | 42 | `TRAIN_SEED=42` → `run_arm.py --train-seed` → `fabric.seed_everything` | The anchor's query noise ξ and the random ±1 sign of every unlabelled document. Experiments without an anchor (E1, E2, E6) draw nothing from it. It does not change the data order. |
| Data seeds | 42 | `litgpt/litgpt/data/sharded_mixed_dataset.py` (`ShardedMixedDataset.seed = 42`, passed to litdata's `CombinedStreamingDataset`) and litdata's default seed 42 for the `StreamingDataset` shard shuffles | The mixture sampler (which pool the next block comes from) and the shuffle of each pool. Together with the layout below they fix the order in which the data are read. |
| SFT | 1337 | `04_sft_eval.sbatch` (`--seed 1337`) | Every random draw of the SFT, including its data shuffle. The same for every experiment. |
| Generation | 20260924 | `analysis/generated_behavior.py` (`GEN_SEED`) | Combined with the SHA-1 of (experiment, stage, condition) to seed each sampling job, so samples do not depend on which GPU task ran them. |
| Bootstrap | 20260923 | `analysis/likelihood_analysis.py`, `generated_behavior.py`, `seed_study.py` (`BOOT_SEED`) | Scenario resamples for every interval (10,000 resamples). The same in every script, so contrasts line up. |

## Hardware, data order and determinism

- **Layout fixes the data order.** litdata assigns data to *ranks × data-loader workers*. Every row of Table 1 was
  midtrained on four GPUs of one node: four ranks, micro-batch 16, eight data-loader workers per rank, global batch
  1024 × 2048 tokens. The experiment scripts use exactly this (`NGPUS=4`). 
- **GPU memory.** Micro-batch 16 uses 39 GB per GPU with flash attention and 55 GB with the deterministic math
  backend, so use 80 GB cards.
- **GPU models in Table 1.** E1, E2, E8, E9 and E10 ran on A100-80GB; E3–E7 on H200. Pretraining ran on A100-40GB.
- **Non-deterministic kernels.** All Table 1 rows used the default kernels (fused flash attention), whose backward
  pass is not deterministic, so no row is bitwise reproducible.
- **Bitwise reproduction.** To make your own runs bitwise reproducible, export `DETERMINISTIC=1 SDPA_MATH=1` before
  calling an experiment script. The exports pass through to both the midtraining and the SFT job. Keep the GPU model,
  `NGPUS` and micro-batch fixed. Expect about twice the run time.


## Compute budget

| Stage | Cost |
|---|---|
| Data preparation | About 2 hours of wall time on 16–32 cores, 38 GB RAM peak. |
| Pretraining | About 9 GPU-hours (4 GPUs × 2 h 10 min). |
| Ten experiments | 10 × (4 GPUs × 3 h 47 min on A100-80GB, or 2 h 19 min on H200) + 10 × (15 min SFT + 2 × 15 min generation), about 160 A100-hours or 100 H200-hours. |
| Seed study, 4 GPUs | 4 × (4 GPUs × 3 h 55 min) on H200, about 63 GPU-hours. |
| Disk | Dataset snapshots about 28 GB; midtraining shards about 19 GB; pretraining shards about twice the FineWeb part of those; 9.6 GB per experiment (8.3 GB midtraining + 1.3 GB SFT). |
