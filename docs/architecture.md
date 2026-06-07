# stereo2spatial Architecture

`stereo2spatial` is a conditional waveform-patch flow model for stereo to
spatial music generation. Dataset artifacts store continuous raw waveforms, and
the training loader groups samples into fixed-size patches at load time.

Core implementation files:

- `stereo2spatial/modeling/spatial_dit.py`
- `stereo2spatial/modeling/layers.py`
- `stereo2spatial/training/dataset.py`
- `stereo2spatial/training/losses_batch.py`
- `stereo2spatial/training/losses_full_song.py`
- `stereo2spatial/training/scheduled_sampling.py`
- `stereo2spatial/training/discriminator.py`
- `stereo2spatial/inference/sampling.py`
- `stereo2spatial/inference/runner.py`

## Data Interface

Dataset samples store waveform tensors in `[C, S]` layout:

- `C`: audio channel count
- `S`: waveform samples

The loader converts each tensor to `[C, P, T]`, where `P` is configured by
`model.patch_size` and `T` is the patch frame index.

The standard sample artifacts are:

| Tensor | Meaning | Default shape |
| --- | --- | --- |
| `target_signal` | Spatial target waveform | `[12, S]` |
| `source_stereo_signal` | Stereo conditioning waveform | `[2, S]` |
| `source_mono_signal` | Duplicated mono conditioning waveform | `[2, S]` |
| `source_downmix_signal` | AC3-style stereo downmix waveform | `[2, S]` |
| `valid_mask` | Valid non-padding patch frames | `[T]` or `[B, T]` |

The default target channel order is:

`FL, FR, FC, LFE, BL, BR, SL, SR, TFL, TFR, TBL, TBR`

## Model

`SpatialDiT` predicts the clean target waveform endpoint from a noised target
state `zt`, a conditioning signal `z_cond`, and scalar timestep `t`.

Inputs:

- `zt`: `[B, C_target, P, T]`
- `z_cond`: `[B, C_cond, P, T]`
- `t`: `[B]`
- optional `valid_mask`: `[B, T]`
- optional recurrent memory: `[B, M, H]`

Per patch frame, channel and patch dimensions are flattened:

- target token width = `target_channels * patch_size`
- conditioning token width = `cond_channels * patch_size`

Separate input projections lift target and conditioning streams into the
transformer hidden width. The network applies timestep-conditioned transformer
blocks, cross-attends from target tokens to conditioning tokens, optionally
threads recurrent memory tokens, and projects back to waveform-patch space.

## Training

Training uses rectified-flow style interpolation between noise and the clean
target waveform. The network predicts the clean endpoint; sampling converts that
prediction into velocity as needed for ODE solvers.

Losses operate directly on clean waveform predictions:

- flow reconstruction loss
- multi-resolution STFT spectral convergence and log-magnitude loss
- optional correlation/routing losses
- optional GAN losses
- optional AC3-style downmix consistency against the conditioning signal

The dataset supports strided crops and full-song windows. Long sequences can be
trained with windowed execution and recurrent memory.

## Inference

Inference reads mono or stereo audio, maps it to the configured conditioning
channel count, patchifies samples into `[C, P, T]`, samples spatial waveform
patches with overlap-add chunking, then unpatches and writes the target
multichannel WAV.
