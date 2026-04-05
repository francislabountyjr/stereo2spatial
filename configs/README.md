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

In practice:

- stage 1 configs are the right place to start from scratch
- stage 2 configs are refinement configs for longer-context behavior
- GAN presets trade extra complexity and memory use for sharper decoded results

## Top-Level Sections

Every training config resolves into these top-level sections:

- `seed`: reproducibility seed
- `output_dir`: run directory for checkpoints and resolved config
- `data`: dataset paths, latent timing, augmentation, and dataloader settings
- `model`: SpatialDiT architecture
- `training`: training loop behavior, sequence regime, GAN, EMA, scheduled
  sampling, flow schedule, and validation
- `optimizer`: optimizer family and hyperparameters
- `scheduler`: learning-rate schedule

## What The Main Sections Control

### `data`

Important fields:

- `dataset_root`: root folder for latent samples
- `manifest_path`: JSONL manifest describing sample directories
- `sample_artifact_mode`: `bundle` or `split`
- `segment_seconds`: base segment length written by preprocessing
- `sequence_seconds`: nominal loaded sequence length
- `stride_seconds`: stride used when walking long songs
- `latent_fps`: latent frame rate, either numeric or `auto`
- `mono_probability` / `downmix_probability`: conditioning augmentation
- `batch_size`, `num_workers`, `prefetch_factor`, `pin_memory`,
  `persistent_workers`: dataloader throughput controls

Rules worth remembering:

- `mono_probability + downmix_probability` must stay `<= 1`
- `sample_artifact_mode` must be `bundle` or `split`

### `model`

Important fields:

- `target_channels`: spatial output channel count
- `cond_channels`: conditioning channel count; current stack expects `1`
- `latent_dim`: latent feature depth
- `hidden_dim`, `num_layers`, `num_heads`, `mlp_ratio`, `dropout`: transformer
  size controls
- `timestep_embed_dim`, `timestep_scale`, `max_period`: timestep embedding
  behavior
- `num_memory_tokens`: recurrent memory-token count

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
- `validation_generation_vae_checkpoint_path`
- `validation_generation_vae_config_path`

### `optimizer`

- `type`: `adamw` or `adam`
- `lr`
- `weight_decay`
- `beta1`, `beta2`
- `eps`
- `adamw_fused`
- `adamw_foreach`

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

### Turn on decoded validation previews

Set:

- `training.run_validation_generations: true`
- `training.validation_generation_input_path`
- `training.validation_generation_output_path`
- `training.validation_generation_vae_checkpoint_path`
- `training.validation_generation_vae_config_path`

Without the VAE paths, decoded validation previews will fail validation checks.

### Export for local inference or Hugging Face

After training, use:

```bash
python scripts/export/export_model_bundle.py --train-run-dir runs/train_stage_2 --checkpoint latest --output-dir exports/stereo2spatial-stage2
```

That produces the bundle format consumed directly by `infer.py`.
