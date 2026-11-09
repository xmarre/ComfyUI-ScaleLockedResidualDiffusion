# Unreleased: Krea 2 spectral progressive sampling

## Summary

`Krea 2 Turbo Flow-Aligned Regenerate [Experimental]` gains an optional
`regeneration_mode=spectral_progressive`, and a new
`Krea 2 Spectral Calibration (Scale-Locked)` node. The default `trajectory`
mode, the other regeneration modes, every existing widget position and every
default are unchanged. Saved workflows run exactly as before.

The mode implements the spectral noise expansion, time alignment and
power-spectrum transition scheduling of *Spectral Progressive Diffusion for
Efficient Image and Video Generation* (Xiao, Chao, Yariv, Wetzstein, 2026;
arXiv:2605.18736), whose official reference implementation is MIT licensed
(https://github.com/howardhx/speed). The code in this repository is an
independent implementation; no third-party code is included.

## New optional controls (appended after `latent_upscale_method`)

- `regeneration_mode`: new choice `spectral_progressive` (default remains
  `trajectory`).
- `progressive_transition_sigmas` (text, default empty): one schedule sigma
  per resolution change, strictly decreasing. Must be empty in other modes.
- `spectral_tolerance` (default 0.01): tolerance `delta` for spectral
  transition times.
- `spectral_calibration` (socket, optional): output of the calibration node.

## How the mode works

One denoising trajectory starts on the planner-sized grid (identical to the
planner's first steps in the other modes) and grows to each intermediate
stage and the output size at planned steps. At a transition the noisy state
`x = (1 - sigma) x0 + sigma * noise` keeps its orthonormal 2-D DCT
coefficients; new higher-frequency coefficients receive fresh Gaussian noise
of standard deviation `sigma`; the state is scaled by
`kappa = r / (1 + (r - 1) sigma)` and sampling continues at
`sigma~ = kappa * sigma`, where `r = sqrt(target area / source area)` of the
actual latent grids. The noise of the result is exactly white Gaussian with
variance `sigma~^2` and the low-frequency content keeps its amplitude; image
content above the smaller grid's resolution is absent, so transitions must
happen while it is still noise dominated.

No denoised prediction is used as a starting image and no finished image is
re-noised. This distinguishes the mode from the previously rejected
staggered resolution ladder, which started later stages from blurry
mid-schedule predictions at a lower noise level.

Transition sources, exactly one of which is used and logged:

1. `progressive_transition_sigmas`;
2. a connected `spectral_calibration`: each stage hands over at the first
   schedule sigma at or below `1 / (1 + sqrt(delta / (P (1 + P - delta))))`,
   where `P` is the calibrated output-grid power at the lowest frequency the
   stage cannot represent;
3. for a single transition only, `composition_lock_sigma`.

Impossible schedules (a resolution without steps, or a transition that is
never reached) are rejected before sampling.

## Calibration node

Fits `P(w) = A * w^-beta` to clean latents in the model's internal latent
space with an orthonormal FFT (`w` in cycles per image along the shorter side;
DC excluded; fit band configurable). It runs no model. Calibrations are
rejected when the fit is poor (R² < 0.9, exponent outside 0.5–5, fewer than
8 bins), when the latent format or channel count differs, or when the output's
shorter side differs from the calibration's by more than 2x. Connect a completed
Krea 2 LATENT (e.g. LoadLatent), or Load Image -> VAE Encode with Krea 2's Qwen
Image VAE. Do **not** connect the Empty Latent/Shared target latent used to set
generation dimensions: it contains no image content and is often a four-channel
placeholder. An empty canvas or wrong channel count is rejected before the fit. Within that
range the amplitude is scaled by the squared side ratio.

Measured on 12 DIV2K photographs (1024x1024 crops) encoded with Krea 2's VAE:
`A = 41.1`, `beta = 1.99`, R² 0.989; per-image `A` 17–105 and `beta`
1.70–2.37. Coefficients published for other models' latents do not transfer
and are not offered as presets.

## Solver, Spectrum and RNG behavior

- All ten Krea 2 solvers are supported. Each resolution is a separate native
  sampler call at its own size, so multistep history and SA-Solver PECE
  corrections restart at first order after each transition.
- ER-SDE and SA-Solver keep their seeded patch-coupled stochastic increments.
  Each transition's new coefficients use separate seeds.
- Spectrum forecasting runs one session per resolution and never forecasts
  across a transition.
- In other modes a connected calibration never changes sampling; `trajectory`
  mode logs whether `composition_lock_sigma` hands over before frequencies the
  target cannot recover become informative.

## Validation and limitations

- CPU unit tests cover DCT parity with SciPy, reconstruction, white-noise
  statistics after expansion (including noninteger and non-square ratios),
  amplitude and time alignment, determinism and seed separation, Eq. 9/10
  parity, calibration fitting, transfer and rejection, schedule rejection, and
  node routing and saved-workflow defaults.
- Real ComfyUI CPU integration with a small random Krea 2 model covers every
  solver, exact reproduction of the planner prefix, exact reconstruction of
  the expanded state at the start of the next segment, DiT calls per
  resolution, three-resolution manual and calibrated schedules, pre-sampling
  rejection with zero model calls, interruption and repeatability, and
  Spectrum sessions per resolution.
- Not established: Krea 2 Raw or Turbo image quality, composition
  preservation, anatomy, text rendering, wall-clock speedup or GPU memory on
  real checkpoints. Compare against native sampling and the `trajectory`
  mode with matched seeds before relying on the mode.
