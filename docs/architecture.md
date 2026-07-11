# stereo2spatial Architecture

`stereo2spatial` has one current training and inference stack with two model
representations. New models operate directly on waveform patches. Historical v1
models operate on precomputed EAR-VAE latents while retaining their original
checkpoint parameter graph.

## Architecture Matrix

| Concern | Current waveform model | v1-compatible model |
| --- | --- | --- |
| Canonical config | `model.architecture: waveform` | `model.architecture: legacy_vae` |
| Model class | `SpatialDiT` | `LegacySpatialDiT` |
| Signal representation | `[C, P, T]` waveform patches | `[C, 64, T]` EAR-VAE latents |
| Typical conditioning | stereo, `cond_channels: 2` | mono latent, `cond_channels: 1` |
| Training artifacts | waveform `bundle`, `split`, or `flac` | latent `bundle` or `split` |
| Audio codec | none | EAR-VAE v2, 48 kHz / 50 latent fps |
| Checkpoints | current waveform family | historical v1 family |

Only the representation-specific model and dataset boundary differ. Optimizer,
scheduler, trainer loop, EMA, checkpoint selection, clean-endpoint losses,
window planning, solvers, timestep-major inference, dynamic batching, and audio
output are shared.

## Selection and Checkpoint Compatibility

`stereo2spatial/modeling/factory.py` is the single architecture factory used by
training, inference, and export.

Canonical values are `waveform` and `legacy_vae`. Compatibility aliases such as
`no_vae` and `ear_vae_latent_v1` normalize to those values, but new configs
should use the canonical names.

Historical config migration is intentionally narrow:

- `latent_dim` present and `patch_size` absent selects `legacy_vae`;
- otherwise the default is `waveform`;
- if both fields are present, set `architecture` explicitly.

Checkpoint state keys are inspected before inference, bundle export, and
weight-only fine-tuning initialization, so waveform and v1 parameter families
cannot be cross-loaded through those paths. Full training resume delegates to
Accelerate's strict state restoration and requires the same architecture and
trainer state as the original run.

## Core Modules

- `stereo2spatial/modeling/factory.py`: architecture selection and construction
- `stereo2spatial/modeling/spatial_dit.py`: current waveform model
- `stereo2spatial/modeling/legacy_spatial_dit.py`: v1-compatible latent model
- `stereo2spatial/modeling/layers.py`: current transformer layers
- `stereo2spatial/modeling/legacy_layers.py`: historical fused-attention layout
- `stereo2spatial/codecs/ear_vae/codec.py`: v1 audio/latent boundary
- `stereo2spatial/training/dataset.py`: waveform datasets
- `stereo2spatial/training/latent_dataset.py`: historical latent datasets
- `stereo2spatial/training/components.py`: representation-specific component selection
- `stereo2spatial/training/losses_batch.py`: shared batch preparation and losses
- `stereo2spatial/training/losses_full_song.py`: long-sequence training
- `stereo2spatial/inference/sampling.py`: window-major compatibility sampler
- `stereo2spatial/inference/timestep_sampling.py`: default song-level sampler
- `stereo2spatial/inference/solvers.py`: shared fixed-step controllers
- `stereo2spatial/inference/offline_batch.py`: dynamic folder scheduler
- `stereo2spatial/inference/export_bundle.py`: native deployment bundles
- `stereo2spatial/inference/runner.py`: audio-to-audio orchestration

## Data and Batch Interface

The shared trainer consumes:

- `target_signal`: `[B, C_target, P, T]`
- `cond_signal`: `[B, C_cond, P, T]`
- `valid_mask`: `[B, T]`

Here `P` is either waveform patch size or latent feature width. This shared
interface keeps representation checks at component construction rather than
spreading architecture branches through the trainer.

### Waveform data

Stored waveform tensors use `[C, S]`, where `S` is audio samples. Bundle/split
artifacts use:

| Tensor | Meaning | Typical shape |
| --- | --- | --- |
| `target_signal` | Spatial target | `[12, S]` |
| `source_stereo_signal` | Stereo conditioning | `[2, S]` |
| `source_mono_signal` | Duplicated mono conditioning | `[2, S]` |
| `source_downmix_signal` | Stereo spatial downmix | `[2, S]` |

The loader patchifies them to `[C, P, T]`. FLAC mode instead reads song-local
audio paths from the manifest and applies the same patchification at runtime.

### v1 latent data

Historical samples use `target_latent`, `source_stereo_latent`,
`source_mono_latent`, and `source_downmix_latent`. Tensors are normalized to
`[C, D, T]`; `[D, T]` conditioning is accepted as one channel. `D` is 64 for the
historical EAR-VAE, and `T` normally advances at 50 fps.

The manifest identifies `sample_dir` and `target_latent_shape`. Old Windows or
POSIX sample paths are rebased under the configured dataset root. Bundle and
split layouts, multiple dataset roots, exclusions, strided crops, full-song
loading, mono/downmix conditioning selection, and deterministic epoch planning
are supported.

## Shared Model Contract

Both models accept:

- noisy state `zt`: `[B, C_target, P, T]`
- conditioning `z_cond`: `[B, C_cond, P, T]`
- float32 timestep `t`: `[B]`
- optional `valid_mask`: `[B, T]`
- optional memory: `[B, M, H]`

Both return a clean endpoint prediction, optionally paired with updated memory.
Mix-style and amplitude conditioning are waveform-only extensions; config
validation disables both for `legacy_vae`.

### Current `SpatialDiT`

The waveform model flattens channel and patch dimensions into one coarse token
per temporal frame. Target and conditioning streams have separate projections.
Timestep-conditioned transformer blocks self-attend, cross-attend to
conditioning, and thread recurrent memory tokens.

Optional waveform-level refinement divides each coarse patch into microtokens,
combines them with coarse semantic tokens and conditioning microtokens, and adds
a fine waveform residual before the final temporal output head. RoPE and valid
masks are shared across coarse and fine paths.

### v1 `LegacySpatialDiT`

The v1 model uses `P = latent_dim` and preserves the historical fused
`nn.MultiheadAttention`, normalization, memory, and `final_proj` state names and
shapes. Its stored head predicts velocity. The public boundary converts it to a
clean endpoint without parameters:

```text
x1_prediction = zt + (1 - t) * velocity_prediction
```

This lets historical weights load strictly while the current clean-prediction
losses and solvers remain unchanged.

## Training

Training samples one clean target `z1`, noise `z0`, and timestep `t`, then forms
the rectified-flow interpolation. For long sequences, all windows in one pass
share the same timestep and song-level noisy state. Memory initializes once and
moves left-to-right through the window sweep.

Current waveform configs can use:

- MSE/L1/Charbonnier clean reconstruction
- MR-STFT and perceptual losses
- downmix consistency
- binaural ILD/IPD/correlation and mid/side terms
- routing/correlation regularizers
- optional adversarial training

`legacy_vae` uses clean reconstruction and architecture-neutral regularizers in
latent space. Config validation rejects waveform-semantic transforms and losses
for this architecture.

`resume_from_checkpoint` restores full current-run state. Weight-only
`init_from_checkpoint` supports fine-tuning but still requires the configured and
checkpoint architecture families to match.

## Inference Pipelines

Waveform:

```text
audio -> channel mapping -> waveform patching -> flow solver -> unpatch -> WAV
```

v1-compatible:

```text
48 kHz audio -> EAR-VAE encode -> latent flow solver
             -> per-target-channel EAR-VAE decode -> multichannel WAV
```

The v1 VAE is shared across files in a session. Dynamic legacy preprocessing and
postprocessing therefore require one worker each.

Runtime defaults come from exported bundle recommendations or the resolved
training config. Explicit CLI values take precedence. Short inputs retain the
trained fixed window; zero padding is excluded by the valid mask.

### Default timestep-major traversal

Timestep-major keeps one global solver state per song. Each requested clean field
performs this sequence:

1. reset memory to the model's initial memory;
2. slice the global state into fixed windows;
3. sweep windows left-to-right at one shared solver time;
4. carry memory only within that sweep;
5. overlap-add window predictions into one float32 global clean field;
6. advance the song-level solver state; and
7. discard sweep memory before the next field request.

This mirrors normal full-song training. It is the default for sequential,
dynamic folder, and validation-generation inference.

`window-major` is retained explicitly for compatibility. It solves one window's
entire trajectory before committing endpoint memory to the next window.

### Dynamic batching

Each active song owns its global solver state, current field accumulator, current
window index, and sweep memory. The scheduler exposes only the next ready window
from that song, so memory order cannot be violated. Compatible ready windows from
different songs are stacked into one model call.

Consequences:

- one active song has effective model batch size one;
- occupancy scales with active songs, not windows within one song;
- `max_active_requests` is the primary VRAM limit because every active song owns
  global state and a float32 field accumulator;
- fixed window shape allows optional CUDA Graph buckets;
- dynamic mode supports Euler, Heun, midpoint RK2, and RES6S;
- one-step flow configs are not supported by the dynamic scheduler.

### Solvers and endpoint handling

Models predict the clean endpoint, so the solver converts it to velocity using
`(x1_prediction - zt) / (1 - t)`. Timestep tensors stay float32 even when model
activations use fp16/bfloat16.

The historical RES6S implementation applied exponential-integrator weights as a
generic RK update and failed even on a constant field. RES6S now composes two
Bogacki-Shampine RK3 half-steps: six stages, third-order convergence, weights
summing to one, and no stage at the ill-conditioned step endpoint.

Heun uses its trapezoidal update except on the terminal step, where midpoint RK2
keeps the final velocity probe away from `t ~= 1`. The final clean prediction and
memory output come from one model call at the accepted solver state.

## Export Bundles and Native Deployment

The native bundle contains:

```text
config.json
model.safetensors
```

Self-contained v1 bundles also contain the 48 kHz EAR-VAE assets under `vae/`.
Bundle metadata records architecture, representation dimensions, output layout,
sample rate, window/overlap, and validation-generation solver recommendations.

The supported application runtime is PyTorch/CUDA. Export selects one EMA or
student state and writes it to `model.safetensors`; raw Accelerate checkpoints
can instead select the source at load time. Sessions can choose inference dtype,
`torch.compile` mode, SDPA backend, dynamic batching, and CUDA Graph buckets
without changing the exported weight format.

## Compatibility Invariants

- Current waveform behavior remains the default when `architecture` is absent
  from a normal patch-based config.
- v1 checkpoint parameters are preserved; adaptation adds no learned state.
- Architecture mismatches fail before loading or export.
- Representation-specific restrictions are validated before training starts.
- Noise is song-level across overlaps; masked tails are zero padded.
- Timestep-major memory resets between solver field evaluations.
- Sequential and dynamic fixed-step outputs agree up to normal batched numerical
  round-off.
