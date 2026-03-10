# ComfyUI-ScaleLockedResidualDiffusion

A custom ComfyUI node pack implementing a practical MVP of Scale-Locked Residual Diffusion for the specific failure mode where a model behaves well around its native / comfortable resolution (for example ~1 MP) but drifts badly in composition, anatomy, or identity at much higher resolutions.

## What it does

Instead of letting the high-resolution branch freely re-plan the image, the node:

1. creates a low-resolution planner pass at a target megapixel level,
2. records the planner's per-step denoised x0 trajectory,
3. builds nested high-resolution noise so the high-res branch shares the same coarse stochastic layout,
4. runs the final high-res sampling with a custom CFG guider that locks only the low-frequency denoised structure toward the planner trajectory while preserving the base model's high-frequency residual detail.

In practice, this is meant to reduce:

- composition drift,
- anatomy drift,
- duplicate limbs / body parts,
- scene re-planning between 1 MP and 4 MP,
- the "looks like a different sample entirely" problem.

## Nodes

### 1. Scale-Locked Residual KSampler

Main all-in-one node.

**Outputs**
- `output`: final high-res latent
- `lowres_planner`: final low-res planner latent
- `denoised_output`: final high-res denoised x0 latent when available

**Important controls**
- `target_megapixels`: planner resolution in pixel-space megapixels (usually `0.8` to `1.5` for Flux-like native planning)
- `lock_strength`: global multiplier for the scale lock
- `lock_strength_start` / `lock_strength_end`: how strongly the low-band lock applies early vs late in denoising
- `lock_schedule`: low-band schedule shape; progress follows normalized log-sigma position when planner sigmas are available, with fallback to raw step-index progress otherwise; `flat` means constant start-value scheduling and ignores `lock_strength_end`
- `lock_schedule_hold` / `lock_schedule_power`: knee and curvature controls for schedules such as `hold_then_drop`, `ease_in`, and `ease_out`
- `coarse_cutoff`: retained spatial fraction for the strongest coarse lock band
- `mid_band_cutoff`: retained spatial fraction for an additional mid-frequency lock band
- `mid_band_strength`: overall mid-band lock multiplier
- `mid_band_strength_start` / `mid_band_strength_end`: independent mid-band envelope when `mid_band_schedule` is not `linked`; this envelope multiplies the base `mid_band_strength`
- `mid_band_schedule`: `linked` preserves the old behavior; other modes decouple the mid band from the low band
- `mid_band_schedule_hold` / `mid_band_schedule_power`: shape controls for independent mid-band schedules
- `nested_noise_strength`: amount of zero-mean high-frequency detail noise added on top of the lifted low-res noise
- `lock_mask` (optional): spatial mask to strengthen the lock only in selected regions (for example body / face / hands)
- `pin_anchors`: store planner anchors in pinned CPU memory when possible for faster non-blocking transfer during the high-res pass
- `sampler_guard`: `warn` / `error` / `off` guard for samplers outside a conservative alignment-safe allowlist

### 2. Scale-Locked Nested Noise Preview

Utility/debug node to inspect the nested-noise construction separately.

## Installation

Clone or copy this directory into your ComfyUI `custom_nodes` folder:

```bash
git clone <this-repo> ComfyUI/custom_nodes/ComfyUI-ScaleLockedResidualDiffusion
```

Then restart ComfyUI.

No extra Python dependencies are required beyond ComfyUI + PyTorch.

## Suggested first settings for Flux.2 Klein 9B style use

For a first test when your high-res target is around 4 MP:

- `target_megapixels = 1.0`
- `lock_strength = 0.85`
- `lock_strength_start = 0.95`
- `lock_strength_end = 0.25`
- `lock_schedule = hold_then_drop`
- `lock_schedule_hold = 0.35`
- `lock_schedule_power = 3.0`
- `coarse_cutoff = 0.33`
- `mid_band_cutoff = 0.60`
- `mid_band_strength = 0.35`
- `mid_band_schedule = linked`
- `sampler_guard = warn`
- `nested_noise_strength = 0.35`

If the result still drifts too much:
- raise `lock_strength` toward `0.95`
- lower `coarse_cutoff` toward `0.25`
- use a `lock_mask` over body / face / hands

If the result feels too constrained / too similar to the low-res planner:
- lower `mid_band_strength`
- raise `mid_band_cutoff`
- lower `lock_strength`
- raise `coarse_cutoff`
- lower `lock_strength_end`

## Recommended workflow pattern

Use this node exactly where you would normally use a KSampler for the high-resolution generation pass.

Typical graph:

1. checkpoint / text encodes
2. empty latent or incoming img2img latent at your final target resolution
3. Scale-Locked Residual KSampler
4. VAE decode / detailers / final upscaling if desired

The node internally creates the planner pass for you, so you do not need to build a separate 1 MP sampler branch unless you want to compare outputs.

## Current limitations

This is a carefully implemented MVP, not a mathematically complete research system.

What is already implemented:
- low-res planner trajectory capture,
- nested-noise initialization,
- denoised-space low-frequency locking,
- denoised-space mid-frequency locking,
- residual-preserving coarse-field replacement,
- optional pinned-memory anchor staging,
- conservative sampler-alignment safety gating,
- optional spatial masking.

What is not implemented yet:
- automatic anatomy / pose / segmentation mask extraction,
- explicit residual-only tiled model execution,
- sigma-perfect trajectory matching for samplers that perform unusual extra model evaluations,
- scheduled cutoff animation for coarse or mid bands,
- multi-stage 1 MP -> 2 MP -> 4 MP progressive ladder inside one node,
- exact support tuning for every possible exotic custom sampler.

## Why this implementation is conservative

This node avoids invasive patching of ComfyUI's internal sampler code. Instead it uses:

- the standard Comfy custom-node registration path,
- the standard custom-sampling guider path,
- standard sigma generation,
- standard sampler objects,
- standard preview callback behavior.

That makes it much easier to maintain and much less likely to break when ComfyUI internals shift.

## Files

- `__init__.py` - node registration
- `nodes.py` - ComfyUI node definitions and runtime integration
- `slrd_core.py` - algorithm core, nested noise, latent resizing, residual locking, trajectory helpers

## Sampler safety note

The current implementation aligns planner anchors to the final pass using outer-step / sigma progression heuristics.
That works best with a conservative subset of samplers whose effective evaluation pattern is close to one visible step <-> one anchor step.

Because Comfy's custom sampling system is flexible and some samplers can perform more complicated internal evaluations,
the node exposes `sampler_guard`:

- `warn`: log a warning for samplers outside the conservative safe set
- `error`: refuse to run those samplers
- `off`: trust the sampler and run anyway
