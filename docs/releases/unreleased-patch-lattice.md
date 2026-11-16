# Unreleased: Krea 2 patch-lattice report and opt-in correction

## Background

Krea 2 writes every 2x2 latent patch (16x16 image pixels) from one token. If
the four positions inside a patch receive a constant per-channel offset, the
decoded image carries a faint, perfectly regular 16-pixel grid across the
whole frame. It is most visible in fine texture such as hair and fabric and
in dark areas.

## New optional control on `Krea 2 Turbo Flow-Aligned Regenerate [Experimental]`

`patch_lattice_correction` is appended after the existing optional inputs.
Saved workflows keep their widget values and behavior.

- `off` (default): the sampled latents are returned unmodified, bitwise
  identical to earlier versions.
- `final_latent` (experimental): the constant patch-position offset is
  measured in the finished latent and subtracted from the output and
  `DENOISED_OUTPUT` latents. Exactly three Fourier coefficients per channel
  change (column-alternating, row-alternating and checkerboard); the channel
  mean and all other frequencies are preserved. Sampling, model-call counts,
  seeds and `LOWRES_PLANNER` are unchanged. One log line reports the measured
  strength before and after.

## Handoff diagnostics

With `handoff_diagnostics=log` or `log_and_save_latents`:

- A `Krea 2 patch lattice:` line is logged for the finished planner, each
  enlarged stage source and each completed stage.
- Every step record gains `x0_patch_lattice_coherence`,
  `x0_patch_lattice_max_coherence`, `x0_patch_lattice_chance` and
  `x0_patch_lattice_amplitude`.

Coherence near `chance` means no pattern. The first latent well above chance
shows where it enters: the planner, the latent upscaler, or the denoiser at a
high-resolution stage. Diagnostics still never change the sampled result.

## Evidence and limits

- One 3920x5888 Krea 2 image with the visible grid was analysed. Its luma
  shows a phase-locked 16.000-pixel component in both axes and no other
  non-JPEG periodicity. Wan 2.1 VAE encodings of eight 768-pixel regions
  carry the same per-channel patch-position offsets (coherence up to 0.27
  against a chance level of 0.009).
- Applying the correction to the encodings of four of those regions removed
  83% to 100% of the decoded pattern, with a mean pixel change of 0.6 to 1.0
  of 255 levels.
- Not measured: the correction on latents sampled directly from a real
  checkpoint, and which pipeline factor creates the offset. Grid-shaped
  texture already drawn into image content is not a constant offset and
  remains.
- The bundled SesquiLSR upscaler, run on synthetic latents, produced no
  period-2 pattern at a 1.55x enlargement and a strong one (coherence 0.17
  against 0.003) at exactly 2.0x. Real-latent behavior at 2.0x is not
  measured; the diagnostics line for the enlarged source reports it per run.
