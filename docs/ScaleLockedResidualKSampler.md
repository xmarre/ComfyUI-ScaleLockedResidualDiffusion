# Scale-Locked Residual KSampler

All-in-one sampler node for large-resolution generation where the model's native-resolution generations are much better than its direct high-resolution generations.

## Core idea

A native / low-resolution planner branch is sampled first. Its denoised trajectory is cached. The final high-resolution branch then uses a custom guider that keeps the base model's high-frequency residual detail but continuously pulls the low-frequency denoised structure back toward the planner trajectory.

## Inputs

- `model`, `positive`, `negative`, `latent_image`: standard Comfy sampler inputs
- `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise`: standard sampler controls
- `target_megapixels`: planner resolution in pixel-space MP
- `lock_strength`: overall lock multiplier
- `lock_strength_start`: early low-band lock amount
- `lock_strength_end`: late low-band lock amount
- `lock_schedule`: low-band schedule shape, evaluated against normalized log-sigma position when planner sigmas are available, with raw step-index fallback otherwise; `flat` keeps the start value for the whole run and ignores the end value
- `lock_schedule_hold`: hold region before `hold_then_drop` releases
- `lock_schedule_power`: curvature control for power-based schedules
- `coarse_cutoff`: strongest coarse-band resolution fraction
- `mid_band_cutoff`: second, looser mid-band resolution fraction
- `mid_band_strength`: overall mid-band lock multiplier
- `mid_band_strength_start` / `mid_band_strength_end`: independent mid-band envelope when `mid_band_schedule` is not `linked`; it multiplies the base `mid_band_strength`
- `mid_band_schedule`: `linked` for legacy behavior, or an independent curve mode
- `mid_band_schedule_hold` / `mid_band_schedule_power`: shape controls for the independent mid-band schedule
- `nested_noise_strength`: amount of extra high-frequency detail noise
- `pin_anchors`: pinned-memory staging for planner anchors when possible
- `sampler_guard`: warn / error / off handling for samplers outside the conservative safe set
- `lock_mask` (optional): spatial mask to focus the lock on anatomy-critical regions

## Outputs

- `output`: final high-res latent
- `lowres_planner`: final low-res planner latent
- `denoised_output`: final denoised x0 high-res latent when available

## Notes

- Lower `coarse_cutoff` = stronger global structure control.
- Lower `mid_band_cutoff` and higher `mid_band_strength` = tighter control over medium-scale body/shape structure.
- `hold_then_drop` with a `lock_schedule_hold` around `0.30` to `0.45` gives a stronger early anchor with a later release knee.
- Use `mid_band_schedule = linked` to preserve the legacy shared curve, or switch it off to let medium structure release earlier than the coarse band.
- Higher `nested_noise_strength` = more detail freedom, but also more chance of drift.
- A `lock_mask` is recommended for body-heavy and anatomy-sensitive generations.
