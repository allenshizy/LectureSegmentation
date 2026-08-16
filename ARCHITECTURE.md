# Developer / Architecture Guide

This is the advanced companion to [README.md](README.md) — read it if you want to reproduce
training experiments, understand how the pieces fit together, or modify the code.

## 0. Provenance

The baseline `kpi` framework — everything under `kpi/models` (except `kpi/models/STB`), plus
`kpi/metrics`, `kpi/cli`, and a small part of `kpi/utils` — originates from:

> Jiaqi Wang, Ricky Y.-K. Kwok, and Edith C. H. Ngai. 2025. Towards Key Point Identification (KPI)
> for Lecture Videos: Approaches and Performance Evaluation. *ACM Trans. Multimedia Comput.
> Commun. Appl.* 21, 7, Article 201. https://doi.org/10.1145/3746640

`kpi/models/STB` (the Sentence-Transformer-Boundary model family), all scripts, main utils and `app/` are original
additions built on top of that baseline framework.

## 1. Repo architecture

```
kpi/                     # Core library (importable via sys.path bootstrap, see "Known gotchas")
  datasets/              # Dataset loaders (MITFLD, AVLecture, ...), all expose .videos/.frags
  models/                # Segmentation model implementations
    STB/                 # The main "Sentence-Transformer-Boundary" model family
      sentence_encoder.py    # SBERT wrapper (SentenceEncoder) — frozen, all-MiniLM-L6-v2 by default
      global_transformer.py  # Self-attention over sentence embeddings (GlobalTransformer)
      bilstm_head.py          # BoundaryDetector / LinearBoundaryDetector — final per-sentence logits
      pretrain_heads.py       # STBPretrainingModel — SOP + MSR self-supervised pretraining objectives
      segmentation_model.py   # LectureSegmentationModel — glues encoder+transformer+detector together
      base_module.py          # BaseModule — shared save/load checkpoint plumbing for all STB submodules
    lstm.py, lstm_jvte.py, tw_finch*.py, text_tiling.py, tbm.py, ... # Baseline/alternative models
  metrics/               # F1 / IoU / MoF segmentation metrics
  experiments/           # `Experiment` runner: applies models to test data + computes metrics
  cli/                   # `kpi.cli.cli` click group, wired up by kpi.py (repo root)
  utils/                 # Shared helpers: dataset processing, splits, plotting, STB train/pretrain helpers
app/                     # Gradio demo app (Whisper -> STB -> Qwen/Ollama), see README.md Quickstart
kpi.py                   # Generic `kpi model <name>` CLI entrypoint — kept at repo root (see gotchas below)
experiment_scripts/      # One-off / research entrypoints (training, prediction, LLM-judge ranking)
helper_scripts/          # Dataset prep/maintenance utilities (exports, splits, stats, filters)
artifacts/               # Training outputs: checkpoints, summaries, eval runs (gitignored, local only)
processed_dataset/       # Preprocessed .pt dataset files consumed by training scripts (gitignored)
```

### The STB model pipeline

`LectureSegmentationModel` (in `kpi/models/STB/segmentation_model.py`) composes three
independently-checkpointable submodules:

```
raw sentences --SentenceEncoder--> sentence embeddings --GlobalTransformer--> contextualized features --BoundaryDetector--> per-sentence boundary logits
```

- `SentenceEncoder` is a thin, **frozen** wrapper around a `sentence-transformers` SBERT model
  (default `all-MiniLM-L6-v2`). It is never fine-tuned in this repo, so its checkpoint is just the
  stock public model — no need to version it.
- `GlobalTransformer` and `BoundaryDetector` (or `LinearBoundaryDetector` for the linear-probe
  variant) are the parts actually trained by `pretrain_stb.py` / `supervised_training.py`.
- Boundaries are anchored to **sentence end times**: a predicted/labeled boundary at index `i`
  means "a chapter break occurs at the end of sentence `i`" (see `kpi/utils/text_label_dataset.py`
  `map_frags_to_sentence_labels` and `kpi/utils/stb_supervised/predict.py`
  `probs_to_boundary_times_local_max`).

## 2. Reproducing experiments

All commands assume `uv sync` has been run (see main README). Run scripts with `uv run python ...`
from the repo root so the sys.path bootstrap in each script can find repo root correctly.

### 2.1 Prepare data (helper_scripts/)

| Script | Purpose |
| --- | --- |
| `helper_scripts/export_text_label_dataset.py` | Export a raw MITFLD dataset into a single `sentences + labels` `.pt` file. |
| `helper_scripts/split_processed_dataset_by_seed.py` | Split a processed `.pt` file into `train`/`validation`/`test` `.pt` files by seed. |
| `helper_scripts/merge_short_chunks_in_processed_dataset.py` | Merge very short text chunks into their previous chunk (cleans up noisy chunking). |
| `helper_scripts/filter_ytseg_long_samples.py` | Drop samples whose sentence count exceeds a threshold (avoid OOM on very long lectures). |
| `helper_scripts/dataset_length_stats.py` | Plot document/sentence length distributions across processed datasets. |
| `helper_scripts/suggest_pretrain_batch_size.py` | Benchmark candidate batch sizes on your current hardware and recommend one. |
| `helper_scripts/download_openphi_textbooks_chapters.py`, `download_ytseg_text_dataset.py` | Download the auxiliary text-only pretraining corpora. |
| `helper_scripts/prepare_mitfld_llm_judging_docs.py` | Run a trained checkpoint over MITFLD and export per-sample documents for LLM judging. |
| `helper_scripts/boundary_interval_stats.py` | Stats on the gaps between ground-truth boundaries. |

Example — export + split MITFLD:
```bash
uv run python helper_scripts/export_text_label_dataset.py --dataset_path /data/mitfld --output_path processed_dataset/mitfld_processed.pt
uv run python helper_scripts/split_processed_dataset_by_seed.py --input_path processed_dataset/mitfld_processed.pt --output_dir processed_dataset/mitfld_split
```

### 2.2 Pretrain the STB backbone (self-supervised, no labels)

```bash
uv run python experiment_scripts/pretrain_stb.py \
    --processed_dataset_path processed_dataset/openphi_textbook_merged_short/openphi_textbooks_train.pt \
    --output_dir pretrained/stb_pretrain \
    --epochs 20
```
Trains `GlobalTransformer` (+ optionally `SentenceEncoder`) with Sentence-Order-Prediction (SOP)
and Masked-Sentence-Reconstruction (MSR) objectives. Saves `encoder.pt`/`transformer.pt` checkpoints
that plug directly into `LectureSegmentationModel(encoder_checkpoint=..., transformer_checkpoint=...)`.

### 2.3 Supervised training / fine-tuning on labeled boundaries

```bash
uv run python experiment_scripts/supervised_training.py \
    --processed_dataset_path processed_dataset/mitfld_split/mitfld_processed_train.pt \
    --processed_dataset_path processed_dataset/mitfld_split/mitfld_processed_validation.pt \
    --processed_dataset_path processed_dataset/mitfld_split/mitfld_processed_test.pt \
    --transformer_weights_path pretrained/stb_pretrain/transformer.pt \
    --output_dir artifacts/stb_supervised/my_run \
    --epochs 20
```
- Use `--linear_probe` to freeze SBERT+Transformer and train only a `LinearBoundaryDetector`.
- `--epochs 0` runs evaluation-only (threshold selection + test metrics) against an existing checkpoint.
- Writes `summary.json` with the calibrated `selected_boundary_threshold` — **always check this file**
  when pointing `app/config.py`'s `STBConfig.threshold` at a different checkpoint.

### 2.4 Predict / inspect boundaries

```bash
uv run python experiment_scripts/STB_predict.py \
    --processed_dataset_path processed_dataset/mitfld_split/mitfld_processed_test.pt \
    --checkpoint_path artifacts/stb_supervised/my_run/best/lecture_segmentation.pt \
    --output_dir results/stb_outputs
```
Samples predictions, plots probability curves (`kpi.utils.plot_probs_by_sentence/time`), and writes
side-by-side predicted vs. ground-truth boundary documents.

### 2.5 Baselines and simple demos

```bash
uv run python experiment_scripts/demo_expriment.py --dataset_path /data/mitfld       # TextTiling / VAD / SceneDetector / evenly-spaced baselines
uv run python experiment_scripts/demo_lstm_experiment.py --dataset_path /data/mitfld --output_dir artifacts/lstm
uv run python experiment_scripts/train_example.py                                    # minimal STB API usage snippets, no real data needed
uv run python -X utf8 kpi.py model --help                                            # generic `kpi model <name>` CLI (see kpi/cli/), lives at repo root
```

### 2.6 LLM-judge ranking pipeline (mitfld_llm_rank*)

`experiment_scripts/mitfld_llm_rank.py` sends judging documents (built by
`helper_scripts/prepare_mitfld_llm_judging_docs.py`) to an OpenAI-compatible chat/responses API and
asks it to rank three anonymized annotators/models per metric (SC/TT/TC/TP/GC/BN/MB).
`experiment_scripts/mitfld_llm_rank_stats.py` aggregates the resulting JSONL into frequency
tables/plots. These two scripts do **not** import `kpi` and have no path dependency on repo root.

## 3. `artifacts/` and `processed_dataset/` layout

- `artifacts/<run_name>/summary.json` — training run metadata (seed, best epoch, F1, calibrated
  boundary threshold, threshold sweep).
- `artifacts/<run_name>/best/` and `.../last/` — `encoder.pt`/`transformer.pt`/`detector.pt`
  checkpoints for that run's best/last epoch.
- `artifacts/mitfld_llm_eval*/`, `artifacts/mitfld_llm_judging_inputs*/` — inputs/outputs of the
  LLM-judge ranking pipeline.
- `processed_dataset/<name>/*.pt` — pre-tokenized `{text, label, sentence_ends, duration, frags}`
  sample lists, produced by `helper_scripts/export_text_label_dataset.py` (or the merge/filter
  helper scripts). Both directories are gitignored — regenerate them locally, they aren't meant to
  be committed.

## 4. Known gotchas

- **No packaging/build-system is configured** in `pyproject.toml`, so `uv sync` does **not**
  install `kpi` as an importable site-package. Every script that does `from kpi... import ...`
  relies on Python prepending the *running script's own directory* (not cwd!) to `sys.path[0]`.
  A script sitting directly at repo root gets this "for free" (its own directory is repo root).
  A script one level deep — everything under `experiment_scripts/` and `helper_scripts/` — does
  **not**, and needs an explicit bootstrap at the top of the file, before the `kpi` imports:
  ```python
  REPO_ROOT = Path(__file__).resolve().parents[1]
  if str(REPO_ROOT) not in sys.path:
      sys.path.insert(0, str(REPO_ROOT))
  ```
  All scripts in `experiment_scripts/` now have this bootstrap. Two files under `helper_scripts/`
  (`export_text_label_dataset.py`, `suggest_pretrain_batch_size.py`) do not yet and may fail with
  `ModuleNotFoundError: No module named 'kpi...'` if `kpi` isn't otherwise importable — add the
  same snippet if you hit that.
- **`kpi.py` (the generic `kpi model <name>` CLI) is deliberately kept at repo root, not inside
  `experiment_scripts/`.** It shares its name with the `kpi/` package; when it lived one level
  down it ended up shadowing the `kpi/` package for itself (`import kpi` resolved to the sibling
  `kpi.py` instead of the `kpi/` directory), which is a stricter/different failure than the plain
  "missing sys.path entry" issue above. Keeping it at repo root sidesteps that entirely — don't move it.
- `train_example.py` writes its demo log to `<repo_root>/train.log` regardless of where you run it
  from (uses `REPO_ROOT`, not `Path(__file__).parent`).
- Relative default paths in CLI options (e.g. `--output_dir results/stb_outputs`,
  `--output_dir artifacts/stb_supervised`) are resolved against your **current working directory**,
  not the script's location — always invoke these scripts from the repo root, or pass an explicit
  absolute `--output_dir`.
