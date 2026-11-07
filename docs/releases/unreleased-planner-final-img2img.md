# Unreleased: Krea 2 planner-final img2img regeneration

## New optional controls on `Krea 2 Turbo Flow-Aligned Regenerate [Experimental]`

The three controls are appended after the existing optional inputs. Saved
workflows keep their widget values and the existing behavior.

- `regeneration_mode`: `trajectory` (default) or `planner_final_img2img`.
- `img2img_start_sigma`: start noise level for `planner_final_img2img`
  (default 0.65).
- `handoff_diagnostics`: `off` (default), `log` or `log_and_save_latents`.

## The two regeneration modes

**`trajectory` (unchanged).** Above `composition_lock_sigma`, the first
high-resolution stage uses the planner's intermediate denoised predictions in
place of target model calls, so it follows the planner's denoising trajectory.
Below that sigma it refines freely. Later resolution stages re-noise the
finished image of the stage below to the lock-release sigma. Outputs are
bitwise identical to the previous version for every sampler, with and without
resolution stages and Spectrum.

**`planner_final_img2img` (new).** The planner runs to completion. Every
high-resolution stage then starts from a finished image:

1. Source: the finished planner latent for the first stage and the finished
   latent of the previous stage after that. Its intermediate predictions and
   solver history are not used.
2. The source is resized to the stage size in latent output space. Native
   ComfyUI sampling applies `process_latent_in` and the model's flow noise
   scaling, `x = s * noise + (1 - s) * clean`. `noise` is the patch-coupled
   Gaussian field the trajectory mode uses for that stage and seed.
3. `s` is the first sigma of the node's normal schedule at or below
   `img2img_start_sigma`. This is the KSamplerAdvanced `start_at_step` rule.
   The schedule is not rescaled, and the effective sigma and the number of
   remaining steps are logged.
4. The selected native solver runs that schedule suffix to zero from a fresh
   start. ER-SDE and SA-Solver keep their patch-coupled stochastic increments,
   which are indexed by the absolute schedule step.

In this mode, `composition_lock_sigma` is not used, and nonzero
`coarse_lock_strength` or `mid_lock_strength` is rejected. Lower start sigmas
preserve more of the finished image, leave less room for structural change and
also leave fewer high-resolution steps. Raise `steps` for more. Each stage
costs one model evaluation per remaining step (SA-Solver PECE also evaluates
its corrections).

## Handoff diagnostics

The diagnostics work in both modes and do not change sampling. They log,
for each high-resolution step, how far the denoised prediction and the
state's clean component are from the finished planner and from the stage's
source. `log_and_save_latents` also writes `.latent` files under
`output/scale_locked_diagnostics/`, for LoadLatent + VAE Decode. Use them to
tell an expected noisy-sigma prediction or a mismatched starting state from
a high-resolution prediction that drifts.

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
