# stereo2spatial

[![CI](https://github.com/francislabountyjr/stereo2spatial/actions/workflows/ci.yml/badge.svg)](https://github.com/francislabountyjr/stereo2spatial/actions/workflows/ci.yml)

`stereo2spatial` is a training and inference stack for turning mono or stereo
audio into spatial multichannel audio in an EAR-VAE latent space.

The repo includes:

- a SpatialDiT-based latent model
- an inference CLI for local checkpoints and exported bundles
- stage 1 / stage 2 training presets
- dataset prep, QC, and export scripts
- bundle export utilities for easy local deployment or Hugging Face release

## Start Here

- If you just want to run the pretrained model, jump to
  [Inference With stereo2spatial-v1](#inference-with-stereo2spatial-v1).
- If you want to train or fine-tune, jump to
  [Training Your Own Model](#training-your-own-model).
- If you want to understand the config knobs, see
  [Understanding The Training Config](#understanding-the-training-config)
  and [configs/README.md](configs/README.md).

## Install

This repo targets Python 3.10.

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e .
```

If you also want lint, type-check, and test tooling:

```bash
pip install -e .[dev]
```

## EAR-VAE

`stereo2spatial` uses EAR-VAE as the latent audio codec layer for training,
validation generation, bundle export, and inference.

EAR-VAE links:

- Hugging Face: <https://huggingface.co/earlab/EAR_VAE>
- GitHub: <https://github.com/Eps-Acoustic-Revolution-Lab/EAR_VAE>

When you use an exported bundle such as `stereo2spatial-v1`, the required
EAR-VAE assets can be bundled alongside the model. When you run directly from a
training checkpoint or enable decoded validation generations during training,
you should provide EAR-VAE checkpoint/config paths explicitly.

## Inference With stereo2spatial-v1

Pretrained v1 bundle:

- Hugging Face model: <https://huggingface.co/francislabounty/stereo2spatial-v1>

### 1. Download the bundle

The simplest path is downloading the full exported bundle into one directory.

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download francislabounty/stereo2spatial-v1 --local-dir checkpoints/stereo2spatial-v1
```

Expected layout:

```text
checkpoints/stereo2spatial-v1/
  config.json
  model.safetensors
  vae/
    ear_vae_v2.json
    ear_vae_v2_48k.pyt
```

If you prefer a browser download, keep the same folder layout intact so the CLI
can auto-resolve the config and bundled VAE files.

### 2. Run inference

Point `--checkpoint` at the exported bundle directory:

```bash
python infer.py --checkpoint checkpoints/stereo2spatial-v1 --input-audio path/to/input.wav --output-audio path/to/output_spatial.wav --device cuda --show-progress
```

What this does:

- reads bundle metadata from `config.json`
- loads model weights from `model.safetensors`
- auto-discovers bundled EAR-VAE files under `vae/`
- writes a multichannel WAV to `--output-audio`

Useful inference flags:

- `--report-json path/to/report.json`: write a machine-readable run summary
- `--solver auto|heun|euler|unipc|...`: change latent ODE solver
- `--device cpu`: run on CPU when CUDA is unavailable, at much slower speed
- `--normalize-peak`: normalize output peak before writing WAV

## Inference From Your Own Checkpoints

There are two supported workflows.

### 1. Preferred: export an inference bundle

This is the cleanest path for local deployment and distribution:

```bash
python scripts/export/export_model_bundle.py --train-run-dir runs/train_with_gan --checkpoint latest --output-dir exports/stereo2spatial-v1
python infer.py --checkpoint exports/stereo2spatial-v1 --input-audio path/to/input.wav --output-audio path/to/output_spatial.wav --device cuda
```

If you include VAE assets in the bundle, no extra VAE CLI arguments are needed.

### 2. Directly from a training checkpoint

Use this when you want to infer from a run directory before exporting:

```bash
python infer.py --config configs/train_with_gan.yaml --checkpoint runs/train_with_gan/checkpoints/step_0200000 --vae-checkpoint-path path/to/ear_vae_v2_48k.pyt --vae-config-path path/to/ear_vae_v2.json --input-audio path/to/input.wav --output-audio path/to/output_spatial.wav --device cuda
```

Use `--checkpoint latest` to pick the newest checkpoint under
`<output_dir>/checkpoints/`.

## Training Your Own Model

### Training prerequisites

The training stack operates on precomputed latent datasets, not raw WAVs
directly. In practice that means you need:

- a dataset root such as `dataset/`
- a `manifest.jsonl` describing sample directories
- latent artifacts written in `bundle` or `split` mode
- config files that point `data.dataset_root` and `data.manifest_path` at that
  dataset

Utilities for building and inspecting these latent datasets live under
`scripts/data/`.

You only need EAR-VAE checkpoint/config paths during training if you enable
validation generations or when you export an inference bundle.

### Choose a preset

- `configs/train.yaml`: stage 1 baseline, no GAN, strided crop training
- `configs/train_with_gan.yaml`: stage 1 with adversarial loss enabled
- `configs/train_stage_2.yaml`: stage 2 longer-context / full-song training,
  EMA enabled, scheduled sampling enabled
- `configs/train_with_gan_stage_2.yaml`: stage 2 longer-context training with
  GAN enabled

### Start training

```bash
python train.py --config configs/train.yaml
```

Common variants:

```bash
python train.py --config configs/train_with_gan.yaml
python train.py --config configs/train_stage_2.yaml
python train.py --config configs/train_with_gan_stage_2.yaml
```

Checkpoint controls:

```bash
python train.py --config configs/train.yaml --resume-from latest
python train.py --config configs/train_with_gan.yaml --init-from runs/train/checkpoints/step_0200000
```

Training outputs land under `output_dir`, typically including:

- `resolved_config.json`
- `checkpoints/step_XXXXXXX/`
- validation artifacts when enabled

## Understanding The Training Config

Top-level config sections:

- `seed`: run seed
- `output_dir`: where checkpoints, resolved config, and validation artifacts go
- `data`: dataset paths, latent timing, augmentation probabilities, dataloader
  settings
- `model`: SpatialDiT architecture and memory-token settings
- `training`: sequence regime, logging, checkpoint cadence, GAN, EMA,
  scheduled sampling, flow schedule, and validation controls
- `optimizer`: optimizer family and hyperparameters
- `scheduler`: learning-rate schedule

High-impact settings to understand before changing presets:

- `data.sample_artifact_mode`: `bundle` or `split`; controls how per-sample
  latent artifacts are loaded from disk
- `data.mono_probability` / `data.downmix_probability`: conditioning
  augmentation probabilities
- `model.target_channels`: output channel count in latent space
- `model.num_memory_tokens`: recurrent memory-token count for longer-context
  modeling
- `training.sequence_mode`: `strided_crops` for shorter randomized chunks, or
  `full_song` for long-context / full-sequence training
- `training.sequence_seconds_choices`: sequence-length curriculum for crop-based
  training
- `training.window_seconds` / `training.overlap_seconds`: chunking used inside
  longer sequence processing
- `training.use_gan` and `training.gan_*`: discriminator settings and
  adversarial loss weights
- `training.scheduled_sampling_*`: rollout length, probability, strategy, and
  sampler for stage 2 scheduled sampling
- `training.flow_*`: timestep sampling and flow schedule shaping options
- `training.use_ema` and `training.ema_*`: whether EMA teacher weights are
  maintained and where they live
- `training.run_validation*`: latent validation and optional decoded generation
  preview controls
- `optimizer.type`: `adamw` or `adam`
- `scheduler.type`: `cosine` or `constant`

For a preset-by-preset breakdown and more field-level guidance, see
[configs/README.md](configs/README.md).

## Exporting Bundles For Inference

Exporting a run into a self-contained bundle is the recommended handoff format
for local inference and Hugging Face uploads.

```bash
python scripts/export/export_model_bundle.py --train-run-dir runs/train_stage_2 --checkpoint latest --output-dir exports/stereo2spatial-stage2 --weights-source auto
```

The exported bundle contains:

- `config.json`
- `model.safetensors`
- bundled EAR-VAE assets under `vae/` when available

## Repository Layout

- `stereo2spatial/`: library code
- `stereo2spatial/cli/`: train/infer CLI entrypoints
- `stereo2spatial/modeling/`: shared model definitions
- `stereo2spatial/training/`: training stack, losses, dataset logic, and config
  parsing
- `stereo2spatial/inference/`: inference runner, checkpoint loading, audio I/O,
  and bundle handling
- `stereo2spatial/codecs/ear_vae/`: EAR-VAE integration API
- `stereo2spatial/vendor/ear_vae/`: vendored EAR-VAE model code
- `configs/`: runnable training presets
- `scripts/`: dataset prep, QC, Atmos tooling, and bundle export helpers
- `tests/`: unit tests covering config, inference, and training helpers

## Related Docs

- [configs/README.md](configs/README.md): config presets and tuning guide
- [scripts/README.md](scripts/README.md): dataset, QC, Atmos, and export scripts

## Acknowledgments

Thanks to the EAR Lab team for open-sourcing EAR-VAE and making the latent
audio codec stack available to the community.

- EAR-VAE on Hugging Face: <https://huggingface.co/earlab/EAR_VAE>
- EAR-VAE on GitHub: <https://github.com/Eps-Acoustic-Revolution-Lab/EAR_VAE>
