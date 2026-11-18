# Krea 2 patch-lattice diagnostics and correction

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
  mean and other latent frequencies are preserved within floating-point precision.
  Sampling, model-call counts,
  seeds and `LOWRES_PLANNER` are unchanged. One log line reports the measured
  strength before and after.

## Handoff diagnostics

With `handoff_diagnostics=log` or `log_and_save_latents`:

- A `Krea 2 patch lattice:` line is logged for the finished planner, each
  enlarged stage source and each completed stage.
  Each line carries the signed 48-value pattern (`modes`) so runs with
  different checkpoints or LoRAs can be compared, and `cosine_vs_planner`.
- Every step record gains `x0_patch_lattice_coherence`,
  `x0_patch_lattice_max_coherence`, `x0_patch_lattice_chance`,
  `x0_patch_lattice_amplitude` and `x0_patch_lattice_cosine_vs_planner`.

Coherence near `chance` is consistent with ordinary variation across patches.
The first recorded latent well above chance helps localize the pattern to
the planner, latent upscaling, or a refinement prediction. Diagnostics leave
the sampled result unchanged.

## Evidence and limits

- One 3920x5888 Krea 2 image with the visible grid was analysed. Its luma
  shows a phase-locked 16.000-pixel component in both axes and no other
  non-JPEG periodicity. Wan 2.1 VAE encodings of eight 768-pixel regions
  carry the same per-channel patch-position offsets (coherence up to 0.27
  against a chance level of 0.009).
- Applying the correction to the encodings of four of those regions removed
  83% to 100% of the decoded pattern, with a mean pixel change of 0.6 to 1.0
  of 255 levels.
- Diagnostics from one sampling run of a quantized, LoRA-patched Krea 2
  Turbo finetune (ER-SDE, planner 2000 px, stages 2048 px and 4800 px):
  the finished planner already carried the pattern (coherence 0.097, chance
  0.008). Planner predictions resized onto the 2048 px grid did not (0.008
  to 0.028), the first model prediction after the composition-lock release
  did (0.155), the SesquiLSR-enlarged source did not (0.006, chance 0.003),
  and the first 4800 px prediction did again (0.19). Within this run, the
  denoiser predictions carry the pattern at each resolution, including the
  native range. The offset of
  the first prediction divided by its sigma was 0.0177 at sigma 0.637 and
  0.0178 at sigma 0.558 (internal latent units), as a sigma-independent
  constant offset in the predicted velocity would give.
- Wan 2.1 VAE encodings of ten real photographs were near the chance level
  (coherence 0.018, chance 0.011).
- The output-head bias of the stock Krea 2 Turbo checkpoint has a
  patch-position component of 0.0023, an eighth of the measured velocity
  offset, and does not point in the measured direction (cosine -0.06, measured
  on a different image). The stock INT8 ConvRot checkpoint keeps its output
  head unquantized. The bias parameter alone therefore does not explain the
  pattern.
- Not established: whether the base model, a finetune or merge, weight
  quantization or a LoRA produces the offset. Comparing the `planner_final`
  line between checkpoints, and with LoRAs disabled, at otherwise identical
  settings isolates it.
- With `final_latent`, the measured coherence of the 4800 px output latent
  went from 0.11 to 0 at the reported precision. Decoded visual fidelity
  remains unverified. Grid-shaped texture already drawn into image content
  can remain after the subtraction.
- The bundled SesquiLSR upscaler, run on synthetic latents, produced no
  period-2 pattern at a 1.55x enlargement and a strong one (coherence 0.17
  against 0.003) at exactly 2.0x. Real-latent behavior at 2.0x is not
  measured; the diagnostics line for the enlarged source reports it per run.
