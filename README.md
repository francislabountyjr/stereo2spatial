# stereo2spatial

<p align="center">
  <img src="./assets/Wide310x150Logo.scale-200.png" alt="Stereo2Spatial logo" width="420" />
</p>

<p align="center">
  <a href="https://get.microsoft.com/installer/download/9PJ6R7RQDVP1?referrer=appbadge&cid=model-github-readme" target="_self">
    <img src="https://get.microsoft.com/images/en-us%20dark.svg" width="200" alt="Download from Microsoft" />
  </a>
</p>

<p align="center">
  <a href="https://stereo2spatial.francislabounty.com/">App Homepage</a> | <a href="https://francislabounty.com/blog/stereo2spatial">Case Study</a>
</p>

[![CI](https://github.com/francislabountyjr/stereo2spatial/actions/workflows/ci.yml/badge.svg)](https://github.com/francislabountyjr/stereo2spatial/actions/workflows/ci.yml)

`stereo2spatial` trains and runs conditional flow models that map mono or stereo
music to spatial audio. The default target layout is `7.1.4`, but the same stack
also supports binaural stereo, 5.1, 7.1, and other configured channel layouts.

The repository supports two model architectures:

| Architecture | Config value | Representation | Checkpoints |
| --- | --- | --- | --- |
| Waveform (default) | `waveform` | Raw waveform patches | V2 waveform checkpoints |
| EAR-VAE latent (v1) | `legacy_vae` | EAR-VAE latents | v1 latent checkpoints |

The v1 architecture preserves its checkpoint parameter layout so existing models
can be trained, fine-tuned, exported, and used for inference. Both architectures
use the same trainer, checkpoint handling, solvers, timestep-major inference,
dynamic batching, and audio I/O; only the model representation and associated
dataset/codec boundary differ.

## Installation

Install the core waveform dependencies:

```bash
pip install -e .
```

Install the additional vendored EAR-VAE runtime dependencies for v1-compatible
training validation or inference:

```bash
pip install -e ".[legacy-vae]"
```

For development across both architectures:

```bash
pip install -e ".[dev,legacy-vae]"
```

CUDA is the supported deployment path for full-size models. CPU execution is
useful for tests and small smoke models but is not intended for production
rendering.

## Quick Start

Train from a waveform preset:

```bash
python train.py --config configs/train.yaml
```

Export a self-contained model bundle:

```bash
python scripts/export/export_model_bundle.py \
  --train-run-dir runs/train \
  --checkpoint latest \
  --output-dir exports/stereo2spatial-waveform
```

Run CUDA inference:

```bash
python infer.py \
  --checkpoint exports/stereo2spatial-waveform \
  --input-audio path/to/input.wav \
  --output-audio path/to/output_spatial.wav \
  --device cuda
```

The exported bundle contains `config.json` and `model.safetensors`. Its versioned
metadata identifies the architecture, output kind, channel layout, preprocessing,
and recommended inference defaults. Output layout defaults from the model channel
count: two-channel models are direct binaural, six-channel models are 5.1 rear,
and twelve-channel models are 7.1.4. Export with `--weights-source auto` to prefer
EMA when the source checkpoint contains it; the selected state is the one written
to the bundle. By default, export reads
`<train-run-dir>/resolved_config.json`. Use
`--config path/to/resolved_config.json` or `--config path/to/train.yaml` to
override that source, including when a checkpoint has been moved away from its
original run directory.

## Using v1 EAR-VAE Models

`LegacySpatialDiT` implements the v1 EAR-VAE latent architecture with its
checkpoint-compatible fused attention, output head, normalization, and memory
tokens. The standard v1 dimensions are 12 target channels, one conditioning
channel, and 64 latent features.

The model operates in latent space, and its boundary adapts the v1 velocity
prediction to the shared clean-endpoint API. The same losses and solvers can
therefore be used without modifying checkpoint parameters.

Waveform and v1 checkpoints are different families. A v1 checkpoint must be
loaded with `model.architecture: legacy_vae`; it cannot initialize a `waveform`
model, and the loader reports that mismatch explicitly.

### Fine-tune a historical checkpoint

Start with [configs/train_legacy_vae.yaml](configs/train_legacy_vae.yaml). Point
its dataset fields at the precomputed latent dataset and set:

```yaml
model:
  architecture: legacy_vae
  latent_dim: 64

training:
  resume_from_checkpoint: null
  init_from_checkpoint: runs/old_v1/checkpoints/step_0040000
  init_from_checkpoint_weights_source: student
```

Then launch training:

```bash
python train.py --config configs/train_legacy_vae.yaml
```

Use `init_from_checkpoint` for weight-only initialization or fine-tuning. Use
`resume_from_checkpoint` only when resuming the same run with matching optimizer,
scheduler, and trainer state. If a historical resolved config omits
`model.architecture`, the presence of `latent_dim` without `patch_size` selects
`legacy_vae` automatically.

### Export a v1-compatible bundle

EAR-VAE v2 at 48 kHz is required for the historical latent model. Package the
matching assets with the model:

```bash
python scripts/export/export_model_bundle.py \
  --train-run-dir runs/train_legacy_vae \
  --checkpoint latest \
  --output-dir exports/stereo2spatial-v1 \
  --vae-checkpoint-path path/to/ear_vae_v2_48k.pyt \
  --vae-config-path path/to/ear_vae_v2.json
```

The resulting bundle includes a `vae/` directory and can be passed directly to
`infer.py` without separate VAE arguments.

```text
exports/stereo2spatial-v1/
|-- config.json
|-- model.safetensors
`-- vae/
    |-- ear_vae_v2_48k.pyt
    `-- ear_vae_v2.json
```

Legacy export includes VAE assets by default and therefore requires both VAE
paths. `--no-include-vae` deliberately creates a non-self-contained bundle.

For an unexported historical checkpoint, provide the training config, model
checkpoint, and VAE assets explicitly:

```bash
python infer.py \
  --config runs/old_v1/resolved_config.json \
  --checkpoint runs/old_v1/checkpoints/step_0040000 \
  --vae-checkpoint-path path/to/ear_vae_v2_48k.pyt \
  --vae-config-path path/to/ear_vae_v2.json \
  --input-audio path/to/input.wav \
  --output-audio path/to/output_7_1_4.wav \
  --device cuda
```

The CLI auto-discovers `resolved_config.json` when a checkpoint is inside the
usual `run/checkpoints/step_*` tree. Use `--config` whenever that relationship is
missing or ambiguous.

Legacy inference and export enforce 48 kHz because the codec/downsampling ratio
defines the 50 fps latent timeline.

## Dataset Formats

### Waveform artifacts

Waveform datasets support `bundle`, `split`, and `flac` artifact modes. Bundle
and split samples expose these keys:

- `target_signal`
- `source_stereo_signal`
- `source_mono_signal`
- `source_downmix_signal`

Stored tensors use `[C, S]`, where `C` is channel count and `S` is waveform
samples. The loader reshapes them to `[C, P, T]` using `model.patch_size`; patch
size is a model/training choice rather than a property baked into the dataset.
FLAC mode loads song-local audio paths from the manifest and is waveform-only.

Preprocess waveform data from the repository root:

```bash
python scripts/data/preprocess_dataset.py \
  --dataset-root dataset/stereo2spatial_dataset \
  --input-root path/to/atmos_sources \
  --sample-artifact-mode bundle
```

### v1 latent artifacts

The v1 loader accepts the historical latent equivalents:

- `target_latent`
- `source_stereo_latent`
- `source_mono_latent`
- `source_downmix_latent`

Latents are normalized to `[C, D, T]`; two-dimensional `[D, T]` conditioning
latents are accepted and receive a singleton channel dimension. Bundle and split
artifact modes are supported. The manifest must identify `sample_dir` and
`target_latent_shape`; optional timing metadata is used to resolve latent FPS.

Waveform-domain augmentation and losses are intentionally rejected for
`legacy_vae`, including amplitude lift, codec/resample augmentation, MR-STFT,
perceptual, binaural, and downmix-consistency losses. Base reconstruction and
architecture-neutral regularizers continue to operate directly on latents.

See [configs/README.md](configs/README.md) for the complete configuration field
reference and preset guidance.

## Training

The trainer supports:

- strided-crop and full-song sequence modes
- recurrent window memory and TBPTT
- clean-endpoint rectified-flow training
- EMA and weight-only initialization
- scheduled sampling
- optional GAN, routing, correlation, downmix, perceptual, and binaural losses
- waveform and legacy latent datasets through the same batch contract

For replicated-model multi-GPU training with Accelerate:

```bash
python -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes 2 \
  --gpu_ids 0,1 \
  -m stereo2spatial.cli.train \
  --config configs/train_headphone_virtualizer.yaml
```

This is data parallelism: each GPU owns a model replica and gradients are
synchronized. It is not model parallelism.

## Inference Defaults

Runtime values use these defaults:

| Setting | Resolution order |
| --- | --- |
| Sample rate | CLI -> bundle -> `data.training_sample_rate` -> `data.sample_rate`; v1 is forced to 48 kHz |
| Window/overlap | CLI -> bundle -> `training.window_seconds` / `training.overlap_seconds` |
| Solver/steps/tolerances | CLI -> bundle -> validation-generation settings; `auto` resolves to Heun |
| Sampling order | timestep-major; explicit `--sampling-order window-major` selects compatibility mode |

Bundles preserve the effective training sample rate,
`training.window_seconds`, `training.overlap_seconds`, and validation-generation
solver, step count, and tolerances. Short inputs keep the trained fixed window and
use a valid mask for padded context.

### Timestep-major traversal

Timestep-major is the default for single-file, folder, dynamic-batched, and
validation-generation inference. For each model-field evaluation it:

1. initializes memory for the song;
2. sweeps windows left-to-right at one shared solver time;
3. overlap-adds one song-level clean field in float32;
4. advances the global solver state; and
5. discards memory before the next field evaluation.

This matches normal full-song training and keeps conditioning, including mix
style, consistent across a long render. Window-major traversal remains available
for compatibility:

```bash
python infer.py ... --sampling-order window-major
```

### Dynamic folder inference

Use dynamic batching when rendering multiple files:

```bash
python infer.py \
  --checkpoint exports/stereo2spatial-waveform \
  --input-audio path/to/input_folder \
  --output-audio path/to/output_folder \
  --device cuda \
  --dynamic-batching \
  --max-batch-size 4 \
  --max-active-requests 4
```

The scheduler exposes only the next ready window for each song, preserving its
memory dependency, and batches compatible windows from different songs. A single
active song therefore runs at model batch size one; batching scales with the
number of simultaneously active songs.

Each active timestep-major request retains its song-level solver state and uses a
full-song float32 field accumulator. `--max-active-requests` is therefore the
primary VRAM control for long files, especially high-channel-count output. Start
conservatively and increase it while monitoring peak memory; it does not need to
exceed `--max-batch-size` for normal folder workloads.

For `legacy_vae`, one GPU VAE instance is shared. Keep
`--preprocess-workers 1 --postprocess-workers 1`. Both dynamic and sequential
legacy inference use timestep-major by default. Dynamic inference does not
currently support configs with `training.flow_one_step: true`.

### Mix style and solver selection

Waveform models with mix-style conditioning can use a named preset:

```bash
python infer.py ... --mix-style-preset balanced
```

List the available presets with `python infer.py --list-mix-style-presets`.

The inference stack supports Euler, Heun, midpoint RK2, RES6S, UniPC, and the
configured `torchdiffeq` methods. Dynamic batching currently supports the four
fixed-step controllers: Euler, Heun, midpoint RK2, and RES6S. Corrected RES6S is
an endpoint-safe six-stage third-order integrator; Heun uses midpoint RK2 for its
terminal step to avoid the ill-conditioned near-`t=1` velocity probe.

Use `--report-json path/to/report.json` for a single input or a report directory
for folder input. Reports use normalized underscore values such as
`sampling_order: "timestep_major"` and include the resolved architecture,
representation, checkpoint, weights source, solver, window, dtype, and output
shape. Dynamic scheduler batch statistics are printed to the console.

## Native CUDA Deployment

The supported application deployment path is the exported bundle plus the
PyTorch/CUDA inference APIs or CLI. Relevant runtime switches include:

- `--inference-dtype float32|float16|bfloat16|auto`
- `--compile-model`
- `--compile-mode default|reduce-overhead|max-autotune|...`
- `--sdpa-backend auto|flash|efficient|math`
- `--dynamic-batching`
- `--cuda-graphs`
- `--cuda-graph-buckets 1,2,4,8`

Choose dtype and graph/compile settings for the deployment GPU, then verify
quality against float32 using the same seed and solver configuration.

## Repository Layout

- `stereo2spatial/modeling/`: waveform and v1-compatible model definitions
- `stereo2spatial/codecs/`: EAR-VAE integration for legacy inference
- `stereo2spatial/training/`: datasets, losses, checkpointing, validation, loop
- `stereo2spatial/inference/`: samplers, dynamic batching, audio I/O, runner
- `scripts/data/`: waveform preprocessing, QC, and maintenance
- `scripts/atmos/`: Atmos acquisition and conversion
- `scripts/export/`: native inference-bundle export
- `configs/`: runnable waveform and legacy presets
- `tests/`: unit, integration, checkpoint-compatibility, and inference coverage

Additional design detail is available in
[docs/architecture.md](docs/architecture.md), and operational script guidance is
in [scripts/README.md](scripts/README.md).
