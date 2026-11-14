# Automatic Krea 2 trajectory refinement

Select `trajectory_auto` in the Krea 2 node's **Regeneration mode**. Keep the same planner size, output size, stage layout, solver and scheduler as the `trajectory` comparison. The existing composition lock still governs the first pass. Each subsequent restart is selected automatically. The feature applies when the stage layout has at least one intermediate high-resolution pass. Both interpolation and SesquiLSR use the same policy. `trajectory` remains the default.

## Why the handoff matters

The first high-resolution pass uses intermediate planner predictions above the composition-lock threshold. Euler integrates this locked prefix in closed form when possible. Other solvers integrate it with the original solver and substitute the planner predictions, including the corresponding PECE corrections. The first unlocked prediction can introduce new detail, including unwanted geometry.

Later passes enlarge the preceding stage's completed output-space latent. ComfyUI converts that latent to internal model space and starts the native sampler at

`x(s) = (1 - s) * clean + s * noise_scale * noise`.

In `trajectory`, the same release sigma is used at each completed handoff. For example, at 0.5894 and unit noise scale the ratio of noise to clean amplitude coefficients is 1.435. The coefficient ratio is independent of the enlargement method. The marginal patch-coupled noise is Gaussian, and its coarse component shares randomness with the preceding pass. Stochastic increments are drawn from a separate deterministic stream indexed by the absolute schedule step, also coupled across resolutions. Shortening a suffix retains the starting field and the same increment for each retained sigma pair. The solvers begin each pass with fresh history.

The code establishes a repeated opportunity to reinterpret the source, even with interpolation and native evaluations. The earliest incorrect object or feature depends on the image. It could already exist in the planner, be introduced by enlargement or the first unlocked prediction, or develop across later predictions. Both target resolution and planner evidence matter, particularly above the checkpoint's nominal 2048-pixel range.

## Restart selection

The policy measures three mean per-coefficient powers with orthonormal FFTs, omitting DC and the source Nyquist boundary. Measurements use internal Wan 2.1 latent units. The enlargement is reduced back to the source grid for comparison.

| Measurement | Role |
| --- | --- |
| Source power in the outer octave along growing axes | Proxy for the detail a finer grid must recover |
| Round-trip error power on the shared grid | Accounts for source information lost or altered by the actual enlargement |
| Source power inside the inner half of both Nyquist limits | Proxy for the structure the restart should leave visible |

The band limits follow dyadic octave boundaries. Each power is multiplied by the destination/source area ratio, since an orthonormal Fourier coefficient for the same continuous mode scales by the square root of the sample count. The maximum detail/error power and minimum interior power across images determine a shared restart for a batch.

For power `P`, signal and noise contribute equally when `(1 - s)^2 * P = s^2 * noise_scale^2`. The crossing is `s = sqrt(P) / (sqrt(P) + noise_scale)`. The policy takes the lowest native restart at or above the larger of the edge and error crossings. It accepts the reduction only when that step is at or below the interior crossing, which keeps the measured interior band at or above unit SNR. The legacy suffix is retained when the measurements are degenerate, the schedule has no suitable step, or the candidate would reduce the run to a single terminal update. The two-step minimum leaves a nonterminal update before the terminal one. It supplies no guarantee about detail quality.

[The Spectral Progressive paper](https://arxiv.org/html/2605.18736v3) derives per-frequency activation under a Gaussian signal model for resizing a live trajectory. This policy borrows the forward-SNR relation for measuring completed-latent handoffs. The edge-band and round-trip proxies and the unit-SNR decision are experimental. The paper's velocity-error tolerance does not calibrate this policy for Krea 2. As in [SDEdit](https://arxiv.org/abs/2108.01073), less forward corruption exposes more of the clean source to the model. The spectral measurements cannot classify semantic correctness or recover information absent from the source.

| Alternative | Tradeoff |
| --- | --- |
| Spatially variable corruption | Can target individual regions. Native ComfyUI's Krea 2 sampler uses a scalar sigma, and local gradients alone cannot distinguish incorrect geometry from valid detail. |
| Per-step low-band projection or a data-consistency constraint | Can enforce an explicit observation operator, as in [DDNM](https://arxiv.org/abs/2212.00490). The generated lower-grid latent is not a known degradation of the desired high-resolution image. Previous projection produced doubled anatomy even at low latent residual error. |
| Prediction-based acceptance or retries | Can detect a departure from the source after it occurs. Drift alone does not identify correctness, and retries add expensive target evaluations. Opt-in diagnostics provide the evidence for assessing this approach. |

The policy uses no additional denoiser calls. It computes two source-grid FFTs and, when the latent lives on GPU, reads the reduced statistics back once per measured handoff. Spectrum still starts a separate sampler session per stage. With Diff-Aid `model_sigma_max`, retained sigma values retain the same gain semantics. `schedule_start` normalizes against each pass's first sigma, so a changed restart can change its gain window. The policy does not alter attention backend selection.

## Native-range first refinement

`trajectory_auto` leaves the first high-resolution pass unchanged. That pass is the one with the most freedom: it follows the planner only above `composition_lock_sigma` and then refines from that sigma through every remaining step. At 0.59 the noise coefficient is 1.4 times the clean coefficient, so the model decides again everything except the coarse layout, including features the size of a hand, an eye or a garment detail. Later passes that restart lower can only alter progressively finer content. When the first pass runs on a grid larger than the model was trained for, this decision is made outside the model's documented range (2048 pixels per axis for Krea 2 Turbo), the regime in which diffusion transformers are known to duplicate or invent structure.

`trajectory_auto_native` adds one rule to `trajectory_auto`: **a pass may refine from `composition_lock_sigma` only on a grid inside the native range.** It has no setting of its own.

| Layout | First high-resolution pass | Later passes |
| --- | --- | --- |
| Output inside the range | Unchanged trajectory pass. Identical to `trajectory_auto`. | Automatic restarts |
| Output outside the range, `stage_megapixels` empty, an in-range grid larger than the planner exists | Trajectory pass on the largest in-range grid of the output aspect ratio (for a 3344 x 5024 output: 1360 x 2048). Remaining stages are spaced geometrically from that grid with `max_stage_upscale`. | Automatic restarts |
| First grid outside the range (explicit `stage_megapixels`, or a planner that already fills the range) | Completed-latent handoff from the finished planner with an automatic restart, like every later pass. No pass refines from `composition_lock_sigma`. | Automatic restarts |

The rule moves where the free refinement happens. It does not inspect or alter predictions, adds no model evaluation, and uses the same handoff, noise coupling, solver restarts and restart selection as `trajectory_auto`. The planned in-range stage is smaller than a geometrically spaced first stage, so a layout can gain one short pass.

What this can and cannot do:

- It targets structure invented during an out-of-range first pass. A structure already present in the planner is carried through unchanged, and a structure invented by a short later restart is not prevented.
- Detail that the first pass used to synthesize between the in-range grid and its former size now has to come from the short later passes. Whether enough of it survives is an image-quality question that only decoded comparisons answer.
- In the third layout the first pass can become as short as two steps. Prefer an empty `stage_megapixels`, or a smaller planner, so that an in-range trajectory pass exists.
- That an out-of-range first pass is the source of a given defect is a hypothesis until the saved stage latents show it. Decode `planner_final` and `stage0_completed` of a `trajectory_auto` run: the rule is relevant when the defect is absent from the first and present in the second.

Classifying a finished prediction as correct or invented from latent statistics is not part of any mode. Gradient, contour and spectral measures respond to a new structure and to legitimate new detail alike, and a comparison of the last two predictions of a pass cannot see a structure that both already contain.

## Compare decoded outputs

Use fixed seeds, prompts and identical checkpoints, LoRAs, attention patches and resolutions across `trajectory`, `trajectory_auto` and `trajectory_auto_native`. The log line `Krea 2 trajectory_auto_native:` states which first-pass layout applied. Include a `planner_final_img2img` preservation comparison and a `trajectory` comparison with Spectrum disabled when a defect appears during forecasting. Try scenes with faces and hands, repeated patterns, text, fabrics and structured backgrounds.

Enable `handoff_diagnostics=log_and_save_latents` for individual cases. Decode `planner_final`, each stage's `source_resized`, `step*_x0`, and `completed` with the same VAE. A defect already present in the enlarged source implicates the handoff or the preceding stage. A defect first visible in a denoised prediction implicates the current stage. Saved noisy `state` files are for inspecting initialization. A prediction's deviation from the previous stage at the previous grid is useful for finding changes to inspect, but image-space review is required.

Record restart decisions, wall time by stage, actual and forecast counts, fallbacks, attention backend and peak VRAM. For timing and VRAM, repeat with latent dumps disabled. Accept the mode only if decoded results reduce invented structures across matched cases while retaining useful texture, identity, anatomy and sharp contours. CPU and synthetic tests check the schedule and native sampler contract. Real-checkpoint quality and GPU cost need measured comparisons.
