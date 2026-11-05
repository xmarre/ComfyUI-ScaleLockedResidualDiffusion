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

## Krea 2 Turbo: experimental flow-aligned regeneration

The `Krea 2 Turbo Flow-Aligned Regenerate [Experimental]` node is a compact
Krea-2-specific integration of the existing scale-locked image sampling path.
It uses the native ComfyUI Krea2 model, the Qwen Image VAE and a matching
Krea2 text encoder. It is not a port of MiniMax H3's audio/video/Continuum
runtime and does not load weights independently.

- Input `latent_image` defines the final pixel canvas (8-pixel VAE stride,
  two-pixel latent DiT patch; use multiples of 16 pixels). ComfyUI's standard
  four-channel empty latent is expanded to Krea 2's 16 channels at preflight.
  Native single-frame `B×16×1×H×W` image latents are preserved through
  planning, patch-coupled noise and spatial locking; multi-frame video latents
  are not supported by this image node.
- The node defaults to an eight-step, approximately 1 MP planner pass, then
  eight high-resolution steps: the first ones follow the planner trajectory
  completely, and the remaining ones add detail at the target resolution.
  The optional `steps` input controls both passes (default 8). Connect the same
  step count to a native sampler for matched target-pass comparisons.
- The Krea 2 Turbo sampling profile is ComfyUI CFG **1.0** (no
  classifier-free guidance), Euler, `simple` scheduler, shift **1.15**.
  This is for **Turbo**, not the undistilled RAW checkpoint.
- High-resolution noise is drawn from an exact *Gaussian conditional coupling*:
  within every 2x2 transformer-patch phase the normalized grouped projection
  matches the source Gaussian field, including noninteger upscaling ratios.
  This is a stochastic consistency invariant, not a learned image upscaler.
- `composition_lock_sigma` (optional, default 0.80) sets where the
  high-resolution pass takes over. Target steps that start above this noise
  level use the planner's denoised prediction in every frequency band, so the
  target state is exactly the resized planner state carrying the target noise.
  At those noise levels the high-resolution model would otherwise commit to its
  own composition; mixing only the planner's coarse bands into it can
  superimpose two layouts, visible as duplicated or displaced contours. Lower values
  keep more of the planner image and leave fewer steps for new detail; 1.0
  disables the full lock. With ComfyUI's Krea 2 schedule (Euler/simple,
  shift 1.15), the default locks four of eight steps (six of twelve).
  Locked steps cost no high-resolution model evaluations: the target state at
  the release is computed in closed form from the planner predictions, and
  the sampler starts there. With a target spatial mask or latent noise mask,
  locked steps are evaluated instead, because the mask blends in the
  target prediction.
- **Resolution stages.** Large jumps from the planner to the output are split
  into a ladder of intermediate resolutions. The ladder continues one sampling
  trajectory: the planner owns every step above `composition_lock_sigma`, and
  the remaining steps are divided over the stages in order, with any
  remainder going to the later stages. Each stage starts from all earlier
  predictions (planner and lower stages), resized to its own size, with its
  own patch-coupled noise. Every step of the schedule is therefore evaluated
  exactly once, at the highest resolution reached by then. Higher stages
  start later in the schedule, so they have less freedom to re-plan the
  composition and mostly sharpen it.
  - `max_stage_upscale` (default 4.0, i.e. 2x per side) is the largest
    pixel-area growth per stage. Jumps larger than that get geometrically
    spaced intermediate stages; 0 always jumps straight to the output.
  - `stage_megapixels` lists explicit intermediate sizes, e.g. `4, 8`, and
    overrides `max_stage_upscale`.
  - Each stage needs at least one step after the composition lock. With
    eight steps and the default lock, four steps remain; raise `steps` to
    give every stage more of them.
- After the release, the high-resolution pass refines freely by default.
  `coarse_lock_strength` and `mid_lock_strength` (default 0) optionally keep
  blending the low and middle spatial frequencies of each denoised prediction
  toward the planner. Their bands are defined relative to the target canvas,
  so at large sizes they include fine structures such as fingers, jewelry and
  seams. The planner version of those structures is upscaled and blurred.
  Nonzero values were observed to double fine contours and soften texture,
  and are kept only for experiments. `composition_lock_sigma` is a starting
  value, not a calibrated optimum.
- The `output` is the high-resolution LATENT, `lowres_planner` is the planner
  LATENT, and `denoised_output` is the available final x0 preview LATENT.

For an initial experiment, use a **1024x1024 planner (1.0 MP)** and a
**2048x2048 final latent (4.19 MP)**. Use the same native model, prompt,
seed, LoRAs and reference conditioning in both stages, all supplied through
the single node. The source/target widths need not have an integral ratio.
The official Krea 2 Turbo range is roughly 1K–2K per axis; larger target
canvases are experimental and may run out of memory or generate artifacts.

A meaningful A/B must compare the same high-resolution model, eight steps,
seed, prompt and conditioning both **with and without** the two-pass method.
Connect one step-count input to both nodes when comparing other step budgets.
Judge composition/subject preservation, duplicated objects, typography,
fine-texture detail and wall time/VRAM. Real Krea 2 checks so far are
limited to individual visual comparisons, not a systematic evaluation;
there is no claimed quality or speed improvement. Against a native eight-step target-resolution run, the
default replaces four target-resolution DiT evaluations with eight planner
evaluations; it is intended to improve structural consistency, not
throughput. For RAW, custom samplers and advanced research controls, use
the generic Scale-Locked Residual KSampler instead.

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

This node returns a `DETAILER_HOOK` object that captures the FaceDetailer crop mask in `post_upscale(...)` and drives the masked residual/manifold correction through the hook's sampler path.

The live hook path is `pre_ksample(...)` plus the custom sampler/runtime integration, not the inert `post_encode(...)` / `pre_decode(...)` pair.

This provider no longer advertises sampler-runtime controls that do not participate in its live sampler-driven hook path. The exposed knobs are the ones that still affect the masked latent correction directly:

- residual lock strength and cutoffs,
- optional manifold companding controls,
- optional `lock_mask` / `manifold_mask`, which are intersected with the FaceDetailer support mask.

Compatibility note: this provider's input signature changed when the inert detailer-only sampler/runtime controls were removed. Older saved workflows that used the previous `Scale-Locked Detailer Hook Provider` input surface will need to be re-wired to the current node inputs.

This has been validated for import/compile in this repo, but not against a live current Impact Pack checkout in this environment.

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

A public guider node alone does not make FaceDetailer use SLRD. `Scale-Locked Detailer Hook Provider` is the experimental integration point for applying the masked residual/manifold correction inside the FaceDetailer crop lifecycle.

The current hook implementation is duck-typed rather than source-verified against a live Impact Pack checkout:

- FaceDetailer mask capture via `post_upscale(...)`,
- alias-tolerant request capture via `pre_ksample(...)`,
- masked latent correction via the custom sampler/runtime path.

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
