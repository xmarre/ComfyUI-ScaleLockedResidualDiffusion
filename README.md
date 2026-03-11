# ComfyUI-ScaleLockedResidualDiffusion

A ComfyUI custom node pack implementing a practical MVP of Scale-Locked Residual Diffusion for the specific failure mode where a model behaves well around its native / comfortable resolution (for example ~1 MP) but drifts badly in composition, anatomy, or identity at much higher resolutions.

## What it does

Instead of letting the high-resolution branch freely re-plan the image, the nodes:

1. create a low-resolution planner pass at a target megapixel level,
2. record the planner's per-step denoised x0 trajectory,
3. build nested high-resolution noise so the high-res branch shares the same coarse stochastic layout,
4. run the final high-res sampling with a scale-lock correction that preserves the base model's high-frequency residual detail.

In practice, this is meant to reduce:

- composition drift,
- anatomy drift,
- duplicate limbs / body parts,
- scene re-planning between 1 MP and 4 MP,
- the "looks like a different sample entirely" problem.

## Nodes

### 1. Scale-Locked Residual KSampler

Main all-in-one node. It still owns the full SLRD runtime internally and is the easiest entry point.

**Outputs**
- `output`: final high-res latent
- `lowres_planner`: final low-res planner latent
- `denoised_output`: final high-res denoised x0 latent when available

### 2. Scale-Locked Runtime Context

Builds the reusable SLRD runtime bundle for ComfyUI's modular custom-sampling path.

**Outputs**
- `runtime`: internal SLRD planner/noise context
- `prepared_noise`: a `NOISE` object containing the aligned nested high-res noise
- `lowres_planner`: final low-res planner latent

Use this before `SamplerCustomAdvanced` when you want the modular graph equivalent of the all-in-one sampler.

### 3. Scale-Locked CFG Guider

Public guider node for the standard ComfyUI `GUIDER` contract. This applies the scale-lock denoiser correction using a `runtime` from `Scale-Locked Runtime Context` and returns a fresh patched guider instead of mutating the upstream guider object in place.

Important: this node is only the denoiser-side piece. Full SLRD behavior still depends on the paired runtime context so the high-res noise field is aligned with the low-res planner branch.

### 4. Scale-Locked Residual SamplerCustomAdvanced

One-node version of the modular custom-sampling workflow. Internally it now calls the same runtime builder and guider patcher that power the public guider node.

### 5. Scale-Locked Detailer Hook Provider

Experimental Impact Pack / FaceDetailer integration path.

This node returns a `DETAILER_HOOK` object with a provider-first adapter surface (`get_custom_sampler()` and related aliases) plus a `pre_ksample(...)` fallback path. The goal is to run the full SLRD bundle per crop: planner pass, anchor recording, nested-noise construction, and scale-locked final sample. This has been validated for import/compile in this repo, but not against a live current Impact Pack checkout in this environment.

### 6. Scale-Locked Nested Noise Preview

Utility/debug node to inspect the nested-noise construction separately.

## Suggested modular workflow

For standard ComfyUI custom sampling:

1. Build your base `GUIDER` normally.
2. Build `sigmas` and choose your `sampler` normally.
3. Run `Scale-Locked Runtime Context` with the base `noise`, `guider`, `sampler`, `sigmas`, and target latent.
4. Run `Scale-Locked CFG Guider` on the base guider using the returned `runtime`; this returns a fresh patched guider and leaves the base guider untouched.
5. Feed `prepared_noise` and the patched guider into `SamplerCustomAdvanced`.

That path uses the same SLRD runtime pieces as the all-in-one sampler instead of duplicating planner/noise logic.

## FaceDetailer / Impact Pack note

A public guider node alone does not make FaceDetailer use SLRD. Detailer crop sampling needs the full planner-plus-noise bundle, not only the denoiser correction. `Scale-Locked Detailer Hook Provider` is the experimental integration point for that path.

The current hook implementation is duck-typed rather than source-verified against a live Impact Pack checkout:

- provider-first via `get_custom_sampler()` / provider aliases,
- fallback interception via `pre_ksample(...)`,
- crop execution delegated to the same shared SLRD runtime as the regular sampler nodes.

Because Impact Pack's internal contracts can move, this integration should be treated as unverified runtime glue until it is exercised against the current Impact Pack source.

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
- optional spatial masking,
- modular `GUIDER` exposure,
- Impact-facing detailer hook/provider path.

What is not implemented yet:
- automatic anatomy / pose / segmentation mask extraction,
- explicit residual-only tiled model execution,
- sigma-perfect trajectory matching for samplers that perform unusual extra model evaluations,
- scheduled cutoff animation for coarse or mid bands,
- multi-stage 1 MP -> 2 MP -> 4 MP progressive ladder inside one node,
- exact support tuning for every possible exotic custom sampler,
- verified bindings for every historical Impact Pack hook/provider variant.

## Files

- `__init__.py` - node registration
- `nodes.py` - ComfyUI node definitions, public guider node, and Impact hook/provider nodes
- `slrd_runtime.py` - shared ComfyUI runtime for planner capture, nested noise, guider patching, and final sampling
- `slrd_core.py` - algorithm core, nested noise, latent resizing, residual locking, trajectory helpers
