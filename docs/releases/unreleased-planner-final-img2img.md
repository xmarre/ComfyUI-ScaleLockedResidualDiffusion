# Unreleased: Krea 2 planner-final img2img regeneration

## New optional controls on `Krea 2 Turbo Flow-Aligned Regenerate [Experimental]`

Five controls are appended after the existing optional inputs. Saved
workflows keep their widget values and the existing behavior.

- `regeneration_mode`: `trajectory` (default), `planner_final_img2img`,
  `planner_final_upscale_only` or `planner_final_native_refine`.
- `img2img_start_sigma`: start noise level for `planner_final_img2img` and
  the refinement of `planner_final_native_refine` (default 0.65).
- `handoff_diagnostics`: `off` (default), `log` or `log_and_save_latents`.
- `planner_fidelity_strength`: inactive compatibility control. It must stay 0;
  a nonzero value is rejected before sampling (see below).
- `planner_detail_strength`: inactive compatibility control, kept only for
  saved-workflow widget order. It has no effect.

## The regeneration modes

**`trajectory` (unchanged).** Above `composition_lock_sigma`, the first
high-resolution stage uses the planner's intermediate denoised predictions in
place of target model calls, so it follows the planner's denoising trajectory.
Below that sigma it refines freely. Later resolution stages re-noise the
finished image of the stage below to the lock-release sigma. Outputs are
bitwise identical to the previous version for every sampler, with and without
resolution stages and Spectrum.

**`planner_final_upscale_only` (new).** The planner runs to completion and
its clean latent is enlarged to the output size with `latent_upscale_method`,
through any intermediate stages. No high-resolution denoising runs, so the
target pass makes zero model evaluations and adds no new texture.

**`planner_final_native_refine` (new).** The planner runs to completion, one
img2img refinement runs at most 2048 pixels per axis, and the refined latent
is enlarged to the output size with SesquiLSR, with no diffusion above 2048
pixels. Requires `latent_upscale_method=sesqui_wan21`; `stage_megapixels`
must be empty.

**`planner_final_img2img` (new).** The planner runs to completion. Every
high-resolution stage then starts from a finished image:

1. Source: the finished planner latent for the first stage and the finished
   latent of the previous stage after that. Its intermediate predictions and
   solver history are not used.
2. The source is resized to the stage size in latent output space. Native
   ComfyUI sampling applies `process_latent_in` and the model's flow noise
   scaling, `x = s * noise + (1 - s) * clean`. Each stage uses a fresh,
   seed-deterministic Gaussian field independent of the noise that generated
   its finished source. The field is lifted with the existing patch-coupled
   algorithm; it is not the same realization as trajectory-mode noise.
3. The selected native schedule suffix determines the number of refinement
   steps. Its first sigma is replaced by the requested
   `img2img_start_sigma` value; subsequent sigmas are unchanged. This
   avoids quantizing distinct UI values (e.g. 0.30 and 0.35) to the same
   effective sigma without inserting extra, potentially near-coincident
   solver steps. Both the effective sigma and step count are logged.
4. The selected native solver runs that schedule suffix to zero from a fresh
   start. ER-SDE and SA-Solver keep their patch-coupled stochastic increments,
   indexed by the absolute schedule step, but draw them from a per-stage seed
   separate from the planner's and from the stage's forward noise, so they are
   independent of the noise that produced the finished source.

In this mode, `composition_lock_sigma` is not used, and nonzero
`coarse_lock_strength` or `mid_lock_strength` is rejected. Lower start sigmas
preserve more of the finished image, leave less room for structural change and
also leave fewer high-resolution steps. Raise `steps` for more. Each stage
costs one model evaluation per remaining step (SA-Solver PECE also evaluates
its corrections).

## Retired fidelity projection after real-checkpoint image-quality failure

The previously introduced `planner_fidelity_strength` /
`planner_detail_strength` measurement projection is **disabled**. Real
Krea 2 renders produced severe duplicated anatomy even when their
planner-lowband measurement errors approached zero. At fidelity=1 and
detail=0, the method algebraically cancelled every target model prediction:
the result was simply a lowband-corrected interpolation of the finished
planner latent. In this configuration no high-resolution detail generation
could be retained.

Both widgets remain appended for saved-workflow compatibility, but nonzero
`planner_fidelity_strength` is rejected **before** running the planner.
`planner_detail_strength` is inactive at fidelity zero. Neither active
regeneration mode performs the discarded projection.

The correct next experiment is an *independent* VAE decode of the
interpolated pre-denoising latent. `log_and_save_latents` now saves
`stageN_source_resized.latent` for every stage alongside the planner,
noisy starts, and x0 estimates. This is purely observational and adds
no model calls. It separates a latent-space resize/manifold defect from
subsequent denoiser hallucinations; it does not itself improve image quality.
A reliable alternative may require a trained latent-space upsampler,
an RGB/VAE re-encode handoff, or a reference-trained refinement model.
None of these alternatives has been verified for this Krea 2 checkpoint.

## Handoff diagnostics

The diagnostics work in both modes and do not change sampling. They log,
for each high-resolution step, how far the denoised prediction and the
state's clean component are from the finished planner and from the stage's
source. `log_and_save_latents` also writes `.latent` files under
`output/scale_locked_diagnostics/`, for LoadLatent + VAE Decode. It now also saves the native
`stageN_completed.latent` at every finished stage and `final_output.latent`
from the actual final sample, enabling a complete five-panel diagnostic
comparison without mistaking an early x0 preview for the final output.
Use the same VAE as the generation and compare the raw sampler VAE decode
before any downstream postprocessing.

## Validation and limits

- Real ComfyUI CPU tests with a small random Krea 2 model check the first
  denoiser input and timestep against `s * noise + (1 - s) * clean` for all
  ten samplers. They also check stage chaining, noise coupling, model-call
  counts, seed determinism, Spectrum forecasting in every stage, and cleanup
  after an interrupt.
- These tests do not show that the mode improves real Krea 2 images.
  Compare both modes at an identical seed and settings, using the
  `lowres_planner` output as the fidelity reference.
- Latent noise masks are not supported in `planner_final_img2img`.
