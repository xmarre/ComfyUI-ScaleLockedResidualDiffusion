# Unreleased: Krea 2 planner-final img2img regeneration

## New optional controls on `Krea 2 Turbo Flow-Aligned Regenerate [Experimental]`

The three controls are appended after the existing optional inputs. Saved
workflows keep their widget values and the existing behavior.

- `regeneration_mode`: `trajectory` (default) or `planner_final_img2img`.
- `img2img_start_sigma`: start noise level for `planner_final_img2img`
  (default 0.65).
- `handoff_diagnostics`: `off` (default), `log` or `log_and_save_latents`.
- `planner_fidelity_strength`: 0 (disabled, default) to 1,
  opt-in in `planner_final_img2img` only, constraining target
  predictions and completed stages to preserve the finished planner's
  downsampled latent image.
- `planner_detail_strength`: default 0.20, range 0–1. Active only with
  nonzero planner fidelity. Caps source-relative high-resolution nullspace
  innovation to the selected fraction at every target prediction and at
  completion of each stage. Zero preserves source detail, one reproduces
  the earlier unconstrained-detail projection.

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
   which are indexed by the absolute schedule step.

In this mode, `composition_lock_sigma` is not used, and nonzero
`coarse_lock_strength` or `mid_lock_strength` is rejected. Lower start sigmas
preserve more of the finished image, leave less room for structural change and
also leave fewer high-resolution steps. Raise `steps` for more. Each stage
costs one model evaluation per remaining step (SA-Solver PECE also evaluates
its corrections).

## Experimental planner consistency projection

The high-resolution model can change a completed planner's geometry even
when it begins from correctly re-noised clean latent data. A
`planner_fidelity_strength` above zero constrains its denoised predictions
to preserve the finished planner under the latent area-downsampling operator.
The constraint uses the minimum-L2 pseudoinverse of the separable adaptive
area downsampler. Strength 1 enforces agreement to floating-point precision;
strength 0 is a complete no-op. The completed stage is projected again to
avoid multistep solvers passing residual planner drift to the next stage.

Crucially, exact agreement at the planner resolution leaves an unlimited
high-resolution nullspace. This permitted severe ghosting in real-checkpoint
output. The separate `planner_detail_strength` control now constrains the
**new nullspace innovation relative to the finished preceding stage**,
without changing planner measurement strength. The operator applied to a
prediction X relative to the stage source S is:

`X + fidelity * P(planner-D(X)) - (1-detail) * [X-S-P(D(X-S))]`.

At full fidelity, its downsampled result matches the finished planner;
detail=0 preserves the stage source's existing nullspace, detail=1 restores
the former unbounded experimental projection. The same operation applies to
the completed stage before passing it to the next resolution. It adds no
denoiser evaluations.

This establishes measurable latent-space fidelity and bounded introduction
of new high-resolution latent features. It cannot guarantee the decoded
image preserves every anatomical feature; blur, residual ghosting and other
visual regressions are possible. Real-checkpoint A/B comparisons must
precede release as a nonzero default for fidelity.

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
