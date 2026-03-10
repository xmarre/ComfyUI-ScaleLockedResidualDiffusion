# Scale-Locked Residual KSampler

All-in-one sampler node for large-resolution generation where the model's native-resolution generations are much better than its direct high-resolution generations.

## Core idea

A native / low-resolution planner branch is sampled first. Its denoised trajectory is cached. The final high-resolution branch then uses a custom guider that keeps the base model's high-frequency residual detail but continuously pulls the low-frequency denoised structure back toward the planner trajectory.

## Inputs

- `model`, `positive`, `negative`, `latent_image`: standard Comfy sampler inputs
- `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise`: standard sampler controls
- `target_megapixels`: planner resolution in pixel-space MP
- `lock_strength`: overall lock multiplier
- `lock_strength_start`: early-step lock amount
- `lock_strength_end`: late-step lock amount
- `lock_schedule`: linear / cosine / flat
- `coarse_cutoff`: strongest coarse-band resolution fraction
- `mid_band_cutoff`: second, looser mid-band resolution fraction
- `mid_band_strength`: relative strength of the mid-band lock
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
- Higher `nested_noise_strength` = more detail freedom, but also more chance of drift.
- A `lock_mask` is recommended for body-heavy and anatomy-sensitive generations.
