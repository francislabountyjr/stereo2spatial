# stereo2spatial

`stereo2spatial` trains and runs a conditional diffusion/flow model that maps
mono or stereo music into spatial multichannel audio, with the default target
layout set to `7.1.4`.

The current stack models raw waveform patches directly. Dataset artifacts store
continuous waveform tensors in `[channels, samples]` layout, and the training
dataset groups them into patches at load time according to `model.patch_size`.
Inference writes rendered multichannel WAVs without an intermediate
representation.

## What Is Included

- `SpatialDiT`: the conditional waveform-patch generator
- raw waveform dataset loading with `bundle` and `split` artifact modes
- clean endpoint prediction under flow matching
- multi-resolution STFT waveform losses
- optional scheduled sampling, EMA, GAN, correlation, routing, and downmix
  consistency losses
- waveform validation generation and local inference
- scripts for Atmos rendering, dataset preprocessing, QC, deletion, and bundle
  export

## Dataset Format

Each sample directory contains either:

- `sample_bundle.pt` with `target_signal`, `source_stereo_signal`,
  `source_mono_signal`, and `source_downmix_signal`
- or split files:
  - `target_signal.pt`
  - `source_stereo_signal.pt`
  - `source_mono_signal.pt`
  - `source_downmix_signal.pt`

Every stored tensor is normalized to `[C, S]`:

- `C`: channel count
- `S`: waveform samples

The training loader reshapes those continuous waveforms to `[C, P, T]`, where
`P` is `model.patch_size` and `T` is the resulting patch index over time. This
keeps patch size as a training/model hyperparameter instead of baking it into
the dataset.

`source_mono_signal` is duplicated to stereo channel width by preprocessing so
all default conditioning choices match `model.cond_channels: 2`. `metadata.json`
stores sample rate, original sample count, channel layout, source path, patch
size, tensor shapes, and QC metadata. `manifest.jsonl` indexes the sample
directories for training.

## Preprocess Data

Run from the repository root:

```bash
python scripts/data/preprocess_dataset.py --dataset-root dataset/stereo2spatial_dataset --input-root path/to/atmos_sources --sample-artifact-mode bundle
```

The preprocessing script renders the spatial target and stereo source, derives
mono and AC3-style stereo-downmix conditioning audio, and writes the continuous
waveform artifacts plus `manifest.jsonl`.

For QC inspection of one processed sample:

```bash
python scripts/data/decode_sample_for_qc.py --dataset-root dataset/stereo2spatial_dataset --stream-hash <hash>
```

This writes the stored waveform tensors back to WAV files for listening checks.

## Train

Start with one of the configs in `configs/`:

```bash
python train.py --config configs/train.yaml
```

For replicated-model multi-GPU training, launch the same training module through
Accelerate. Each GPU gets its own model copy and gradients are synchronized:

```bash
python -m accelerate.commands.launch --multi_gpu --num_processes 2 --gpu_ids 0,1 -m stereo2spatial.cli.train --config configs/train_headphone_virtualizer.yaml
```

Use `--num_processes` and `--gpu_ids` to match the GPUs you want to train on.
This is data parallelism, not model parallelism.

The default config expects:

- `data.sample_rate: 48000`
- `model.patch_size: 1024`
- `model.cond_channels: 2`
- `model.target_channels: 12`

Training batches expose:

- `target_signal`: spatial target patches
- `cond_signal`: stereo, mono, or downmix conditioning patches
- `valid_mask`: valid non-padding patch frames

## Inference

Export a checkpoint bundle:

```bash
python scripts/export/export_model_bundle.py --train-run-dir runs/train --checkpoint latest --output-dir exports/stereo2spatial-waveform
```

Run inference from the bundle:

```bash
python infer.py --checkpoint exports/stereo2spatial-waveform --input-audio path\to\input.wav --output-audio path\to\output_7_1_4.wav --device cuda
```

Inference reads mono or stereo audio, patchifies it, samples spatial waveform
patches, unpatches them, and writes a multichannel WAV.

## Repository Layout

- `stereo2spatial/modeling/`: SpatialDiT model and layers
- `stereo2spatial/training/`: config loading, dataset, losses, validation, loop
- `stereo2spatial/inference/`: checkpoint loading, sampling, audio I/O, runner
- `scripts/data/`: dataset preprocessing and maintenance utilities
- `scripts/atmos/`: Atmos rendering/conversion helpers
- `scripts/export/`: inference bundle export
- `configs/`: runnable training presets
- `tests/`: unit and smoke coverage
