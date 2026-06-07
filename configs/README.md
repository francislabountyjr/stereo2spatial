# Training Configs

This directory contains runnable training presets. Use these files as the
starting point for new experiments instead of building configs from scratch.

## Presets

| Config | Use when | Key traits |
| --- | --- | --- |
| `train.yaml` | You want a non-GAN stage 1 baseline | Strided crop training, no GAN, no scheduled sampling |
| `train_with_gan.yaml` | You want stage 1 with adversarial refinement | Stage 1 regime plus discriminator and auxiliary losses |
| `train_stage_2.yaml` | You want longer-context stage 2 refinement | `full_song` training, batch size 1, EMA enabled, scheduled sampling enabled |
| `train_with_gan_stage_2.yaml` | You want stage 2 plus adversarial refinement | Stage 2 long-context regime with GAN enabled |
| `test_5_1_rear.yaml` | You want a 2080 Ti friendly 5.1 rear smoke run | 6-channel rear layout, smaller DiT, no compile |
| `test_headphone_virtualizer.yaml` | You want direct binaural stereo smoke training | 2-channel headphone target, smaller DiT, no downmix loss |
| `train_headphone_virtualizer.yaml` | You want direct binaural stereo training | 2-channel headphone target, EMA, waveform micro-patch refinement, song-local FLAC loading |

In practice:

- stage 1 configs are the right place to start from scratch
- stage 2 configs are refinement configs for longer-context behavior
- GAN presets trade extra complexity and memory use for sharper waveform detail

## Top-Level Sections

Every training config resolves into these top-level sections:

- `seed`: reproducibility seed
- `output_dir`: run directory for checkpoints and resolved config
- `data`: dataset paths, waveform timing, augmentation, and dataloader settings
- `model`: SpatialDiT architecture
- `training`: training loop behavior, sequence regime, GAN, EMA, scheduled
  sampling, flow schedule, and validation
- `optimizer`: optimizer family and hyperparameters
- `scheduler`: learning-rate schedule

## What The Main Sections Control

### `data`

Important fields:

- `datasets`: optional list of `{dataset_root, manifest_path}` pairs for
  training from multiple drives/directories
- `dataset_root`: root folder for continuous waveform samples when using one
  dataset, or a list when paired with a `manifest_path` list
- `manifest_path`: JSONL manifest describing sample directories when using one
  dataset, or a list when paired with a `dataset_root` list
- `sample_artifact_mode`: `bundle` or `split`
- `segment_seconds`: base segment length written by preprocessing
- `sequence_seconds`: nominal loaded sequence length
- `stride_seconds`: stride used when walking long songs
- `sample_rate`: waveform sample rate
- `mono_probability` / `downmix_probability`: conditioning augmentation
- `batch_size`, `num_workers`, `prefetch_factor`, `pin_memory`,
  `persistent_workers`: dataloader throughput controls
- `source_resample_augmentation`: optional source-only sample-rate roundtrip
- `source_codec_augmentation`: optional source-only MP3/AAC/Opus encode-decode
  roundtrip with step-ramped probability. It is intended for normal consumer
  delivery-format robustness, not clipping/noise/stereo-damage augmentation.

Rules worth remembering:

- `mono_probability + downmix_probability` must stay `<= 1`
- `sample_artifact_mode` must be `bundle` or `split`
- if multiple datasets are configured, roots and manifests are paired by list
  order and loaded into one unified segment schedule
- `5.1 rear` uses channel order `FL, FR, FC, LFE, BL, BR` and WAVEX speaker
  mask `0x3f`; `5.1 side` remains available as `FL, FR, FC, LFE, SL, SR`
- `Headphone Virtualizer` is treated as a binaural stereo target. It keeps
  ordinary 2-channel WAV output, disables downmix consistency in the provided
  smoke config, and uses stereo-derived mix-style surrogate features.
- Layout-inactive mix-style controls are omitted from normalized training
  vectors. `5.1 rear` uses 11 controls; `Headphone Virtualizer` uses 10.

### `model`

Important fields:

- `target_channels`: spatial output channel count
- `cond_channels`: conditioning channel count; current stack expects `2`
- `patch_size`: waveform samples per transformer patch
- `hidden_dim`, `num_layers`, `num_heads`, `mlp_ratio`, `dropout`: transformer
  size controls
- `timestep_embed_dim`, `timestep_scale`, `max_period`: timestep embedding
  behavior
- `num_memory_tokens`: recurrent memory-token count
- `mix_style_dim`: number of normalized mix-style conditioning controls
- `waveform_level_depth`: number of PixelDiT-style waveform-token refinement
  blocks. `0` disables the fine pathway for ablation.
- `waveform_micro_patch_size`: raw samples per waveform microtoken inside each
  coarse patch. This must divide `patch_size`.
- `waveform_hidden_dim`: hidden width of each waveform microtoken. The presets
  use a compact width because the microtoken sequence is dense.
- `waveform_num_heads`: temporal attention heads used after waveform token
  compaction. Null falls back to `num_heads`.
- `waveform_mlp_ratio`: MLP expansion ratio inside waveform-token blocks.
- `activation_checkpointing`: recompute coarse and waveform transformer block
  activations during backward to reduce VRAM use. This trades extra compute for
  memory headroom and is most useful for larger models or long windows.

Change `target_channels` and exported `channel_order` together if you are
targeting a different layout.

### `training`

This section carries most of the high-leverage settings.

Core loop and checkpointing:

- `max_steps`
- `grad_accum_steps`
- `mixed_precision`: `no`, `fp16`, or `bf16`
- `compile_model` / `compile_mode`
- `log_every`
- `checkpoint_every`
- `max_checkpoints_to_keep`
- `resume_from_checkpoint`
- `init_from_checkpoint`

Sequence regime:

- `sequence_mode`: `strided_crops` or `full_song`
- `sequence_seconds_choices`: crop lengths used during strided-crop training
- `randomize_sequence_per_batch`
- `window_seconds` / `overlap_seconds`: chunking inside longer sequences
- `tbptt_windows`: truncated-BPTT chunk count
- `full_song_max_seconds`
- `require_batch_size_one_for_full_song`

Stage 1 vs stage 2, in this repo:

- stage 1 presets use `strided_crops` and randomized sequence lengths
- stage 2 presets switch to `full_song`, longer context, lower LR, and batch
  size 1

GAN controls:

- `use_gan`
- `gan_d_lr`, `gan_d_beta1`, `gan_d_beta2`
- `gan_d_base_channels`, `gan_d_num_layers`, `gan_d_fine_layers`,
  `gan_d_coarse_layers`
- `gan_lambda_adv`
- `gan_adv_warmup_steps`
- `gan_r1_gamma`, `gan_r1_every`
- `gan_ms_w_fine`, `gan_ms_w_coarse`

Aux losses:

- `routing_kl_weight`, `routing_kl_temperature`, `routing_kl_eps`
- `corr_weight`, `corr_eps`, `corr_offdiag_only`, `corr_use_correlation`
- `downmix_consistency_weight`, `downmix_consistency_loss`,
  `downmix_channel_order`
- `mix_style_dropout_probability`: probability of withholding per-song mix-style
  controls during training
- `mrstft_loss_weight`, `mrstft_fft_sizes`, `mrstft_hop_lengths`,
  `mrstft_win_lengths`, `mrstft_sc_weight`, `mrstft_log_mag_weight`
- `perceptual_loss_weight`, `perceptual_n_fft`, `perceptual_hop_length`,
  `perceptual_win_length`, `perceptual_n_mels`, `perceptual_f_min`,
  `perceptual_f_max`, `perceptual_band_weight`,
  `perceptual_band_low_hz`, `perceptual_band_high_hz`, `perceptual_eps`.
  This is a broad stereo/render-path log-mel loss. It compares direct 2.0
  targets as stereo, or spatial targets after configured stereo downmix; it
  does not add a center/mid-channel vocal constraint.
- `binaural_ild_loss_weight`, `binaural_ipd_loss_weight`,
  `binaural_ccf_loss_weight`, `binaural_loss_warmup_steps`,
  `binaural_loss_eps`: optional direct-headphone cue losses for two-channel
  targets. ILD compares left/right level ratios, IPD compares interaural phase
  in STFT space, and CCF compares short-window left/right correlation profiles.
  They are ignored for non-2-channel targets.

Scheduled sampling:

- `scheduled_sampling_max_step_offset`
- `scheduled_sampling_probability`
- `scheduled_sampling_prob_start` / `scheduled_sampling_prob_end`
- `scheduled_sampling_ramp_steps`
- `scheduled_sampling_start_step`
- `scheduled_sampling_ramp_shape`: `linear` or `cosine`
- `scheduled_sampling_strategy`: `uniform`, `biased_early`, or `biased_late`
- `scheduled_sampling_sampler`: `euler`, `heun`, or `unipc`
- `scheduled_sampling_reflexflow*`

Flow schedule:

- `flow_timestep_sampling`: `uniform`, `logit_normal`, `beta`, or `custom`
- `flow_fast_schedule`
- `flow_logit_mean`, `flow_logit_std`
- `flow_beta_alpha`, `flow_beta_beta`
- `flow_custom_timesteps`
- `flow_schedule_shift`
- `flow_schedule_auto_shift`
- `flow_schedule_base_seq_len`, `flow_schedule_max_seq_len`
- `flow_schedule_base_shift`, `flow_schedule_max_shift`
- `flow_loss_weighting`: `none`, `sigma_sqrt`, or `cosmap`

EMA controls:

- `use_ema`
- `ema_decay`
- `ema_device`: `accelerator` or `cpu`
- `ema_cpu_only`

Validation controls:

- `run_validation`
- `validation_dataset_root`
- `validation_dataset_path`
- `validation_steps`
- `run_validation_generations`
- `num_valid_generations`
- `validation_generation_seed`
- `validation_generation_input_path`
- `validation_generation_output_path`
- `validation_generation_solver`: inference solver for generated previews
  (`heun`, `euler`, `unipc`, `res6s`, `res_6s`, `dopri5`, `midpoint`, `rk4`,
  `explicit_adams`, `implicit_adams`, or `auto`)
- `validation_generation_solver_steps`
- `validation_generation_solver_rtol` / `validation_generation_solver_atol`
- `validation_generation_chunk_seconds`: null uses `data.segment_seconds`
- `validation_generation_overlap_seconds`

### `optimizer`

- `type`: `muon`, `adamw`, or `adam`
- `lr`
- `weight_decay`
- `beta1`, `beta2`
- `eps`
- `adamw_fused`
- `adamw_foreach`
- `muon_ns_steps`
- `muon_nesterov`

### `scheduler`

- `type`: `cosine` or `constant`
- `warmup_steps`
- `min_lr`

## Common Edits

### Change dataset location

Edit:

- `data.dataset_root`
- `data.manifest_path`
- validation dataset paths if validation is enabled

### Fit training into memory

Usually adjust these first:

- lower `data.batch_size`
- increase `training.grad_accum_steps`
- enable `model.activation_checkpointing`
- reduce `model.hidden_dim`, `model.num_layers`, or `model.num_heads`
- disable GAN if you do not need adversarial training
- disable `compile_model` if compile startup cost is not worth it

### Switch from short-context to long-context training

Move from a stage 1 preset toward a stage 2 preset:

- set `training.sequence_mode: full_song`
- use `data.batch_size: 1`
- set `training.full_song_max_seconds`
- keep `training.require_batch_size_one_for_full_song: true`
- consider enabling EMA and scheduled sampling

### Turn on validation audio previews

Set:

- `training.run_validation_generations: true`
- `training.validation_generation_input_path`
- `training.validation_generation_output_path`

Validation generation now runs the current waveform-patch model directly and
writes rendered multichannel WAVs.

### Export for local inference or Hugging Face

After training, use:

```bash
python scripts/export/export_model_bundle.py --train-run-dir runs/train_stage_2 --checkpoint latest --output-dir exports/stereo2spatial-stage2
```

That produces the bundle format consumed directly by `infer.py`.
