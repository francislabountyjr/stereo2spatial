# Scripts

Operational utilities live under this folder. Run them from repository root so
relative paths and config defaults resolve consistently.

## Layout

- `scripts/data/`: dataset preprocessing, QC, waveform inspection, and maintenance
- `scripts/atmos/`: Atmos acquisition and conversion helpers
- `scripts/export/`: checkpoint-to-bundle export helpers for inference release

Keep model and training logic in `stereo2spatial/`. Treat the scripts here as
workflow tooling around that core package.

## Most Common Entry Points

### Dataset preparation

- `scripts/data/preprocess_dataset.py`
  - Preprocess raw audio into waveform training artifacts plus `manifest.jsonl`
- `scripts/data/normalize_mix_style.py`
  - Normalize raw target-derived mix-style controls into training conditioning vectors
  - Supports `bundle` and `split` artifact modes
- `scripts/data/build_qc_dataset_subset.py`
  - Build a smaller subset for QA or rapid iteration
- `scripts/data/decode_sample_for_qc.py`
  - Write one waveform sample back to audio for spot checks

### Dataset maintenance

- `scripts/data/delete_dataset_samples.py`
  - Remove bad samples and optionally rewrite the manifest
- `scripts/data/update_qc_album_label.py`
  - Update QC metadata labels
- `scripts/data/detect_upmix.py`
  - Analyze candidates for likely upmixed content

### Atmos tooling

- `scripts/atmos/download_atmos.py`
  - Acquire Atmos sources
- `scripts/atmos/convert_atmos.py`
  - Convert Atmos material into the target layout used by the dataset pipeline

### Inference bundle export

- `scripts/export/export_model_bundle.py`
  - Package a training checkpoint into:
    - `config.json`
    - `model.safetensors`
    - optional `vae/` assets for the v1-compatible architecture
  - This is the recommended format for local inference and Hugging Face upload

Example:

```bash
python scripts/export/export_model_bundle.py --train-run-dir runs/train_with_gan --checkpoint latest --output-dir exports/stereo2spatial-waveform
```

You can then infer directly from that exported bundle:

```bash
python infer.py --checkpoint exports/stereo2spatial-waveform --input-audio path/to/input.wav --output-audio path/to/output_spatial.wav --device cuda
```

For a self-contained v1-compatible export:

```bash
python scripts/export/export_model_bundle.py \
  --train-run-dir runs/train_legacy_vae \
  --checkpoint latest \
  --output-dir exports/stereo2spatial-v1 \
  --vae-checkpoint-path path/to/ear_vae_v2_48k.pyt \
  --vae-config-path path/to/ear_vae_v2.json
```

That bundle adds `vae/ear_vae_v2_48k.pyt` and `vae/ear_vae_v2.json` beside the
model files. Native PyTorch/CUDA bundle inference is the supported application
deployment path.

## Atmos Layout Overrides

`scripts/atmos/convert_atmos.py` and `scripts/data/preprocess_dataset.py`
expose `--target-output-layout` and `--target-input-layout`.

`7.1.4` is only a default. Override it when preparing other layouts such as
`5.1 rear`, `5.1 side`, `Headphone Virtualizer`, `7.1`, or `5.1.2`. The
`5.1 rear` layout uses `FL, FR, FC, LFE, BL, BR` and WAVEX speaker mask `0x3f`.
`Headphone Virtualizer` is a direct binaural stereo target, so the training
config should use `target_channels: 2` and disable downmix consistency.

Example:

```bash
python scripts/atmos/convert_atmos.py --target-output-layout "5.1 rear" --target-input-layout 2.0
```
