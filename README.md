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
- **CFG, sampler and scheduler are configurable** on the Krea 2 node.
  The unchanged defaults are ComfyUI CFG **1.0** (no classifier-free
  guidance), Euler, `simple` scheduler, shift **1.15**. These defaults are
  for **Turbo**, not the undistilled RAW checkpoint. `cfg` accepts 0–100,
  and `scheduler` exposes ComfyUI's available schedules; the generated
  sigma schedule is validated to be finite, strictly decreasing, within
  the flow domain [0, 1], and ending at zero. Non-default CFG or schedule
  changes the noise trajectory and therefore the composition-lock release.
- **Sampler choice**: `euler` (default), `dpmpp_2m`, `res_multistep`,
  `deis`, `ipndm`, `ipndm_v`, `gradient_estimation`, `er_sde`,
  `sa_solver` and `sa_solver_pece`, limited to native implementations in
  the installed ComfyUI. Every offered solver retains its native update and
  history. SA-Solver and ER-SDE can inject stochastic noise; SA-Solver PECE
  additionally evaluates the corrected state at the same sigma after the
  predictor callback (except steps without a native correction).
  The same selected solver runs the planner and target passes.
  - **Locked steps cost no target-resolution model evaluations with any
    solver.** Euler reaches the composition-lock release in closed form.
    The other solvers integrate the locked steps with their own native
    update, receiving the resized planner prediction in place of a target
    evaluation. The full lock discards the target prediction on those
    steps, so the result is identical to evaluating it, and the solver's
    history across the release is preserved. The progress bar therefore
    still lists locked steps for non-Euler solvers; they complete without a
    model call.
  - **Resolution stages work with every solver.** The first stage reaches
    the release as described above. Each later stage re-noises the
    previous stage's image to the release noise level and starts the
    solver afresh there, so multistep solvers begin that stage at first
    order.
  - **Native ER-SDE** uses ComfyUI's original VP ER-SDE solver and its first-,
    second-, or third-order history. The optional **ER-SDE solver type**,
    **max stage** (1–3), **eta**, and **s_noise** widgets are appended after
    existing inputs and only apply when `sampler_name=er_sde`. Solver types
    are `ER-SDE`, `Reverse-time SDE`, and `ODE`; ODE or eta=0 removes
    stochastic injection. The first flow sigma is adjusted by native ComfyUI
    for finite log-SNR; the lock's step matching uses that effective sigma.
    All ER-SDE noise increments are generated deterministically from the run
    seed and absolute step index. Across resolutions, a patch-phase Gaussian
    coupling makes the target's coarse stochastic increment match the planner
    while retaining independent fine-frequency noise. Later stages restart
    ER-SDE history at the release step; they do not inherit earlier-stage
    multistep derivatives.
  - **Native SA-Solver and SA-Solver PECE.** The optional `sa_eta`,
    `sa_sde_start_percent` (0.2), `sa_sde_end_percent` (0.35), `sa_s_noise`,
    `sa_predictor_order` (3), `sa_corrector_order` (4), and
    `sa_simple_order_2` settings apply only to the SA-Solver choices.
    The default `sa_sde_end_percent=0.35` (model sigma about 0.68) keeps
    stochastic steps in the planner and makes the high-resolution stages
    deterministic. With ComfyUI's native 0.8, every stage step is stochastic,
    and plain SA-Solver produced visible artifacts at 5–6 steps per stage.
    Each stage already starts from fresh release noise.
    `sa_spectrum_forecasts` (default **true**) lets a connected Spectrum
    Krea 2 model forecast SA predictor calls. False runs every native SA
    predictor and PECE correction, reports `native_only=true` and
    `forecast=0`, and serves as a quality reference. The switch changes no
    solver options, model weights, diffusion patches, noise fields,
    resolution-stage configuration or other samplers. Spectrum never consumes
    a raw forecast on a stochastic step, because a forecast cannot see SA's
    fresh noise increment. Forecasts are therefore most effective on the
    deterministic stages. `sa_planner_spectrum_forecasts` (default **true**)
    independently controls Spectrum only during the low-resolution planner
    pass. Set it to **false** while keeping `sa_spectrum_forecasts=true` to
    evaluate all planner denoisers natively, retain the same seeded noise,
    sigma schedule and planner anchors, then restore Spectrum acceleration
    for both high-resolution stages. This isolates cases where the planner
    preview appears over-smoothed despite a detailed final image. It is
    an optional quality diagnostic, not proof that forecast error causes the
    blur: the planner also operates at a lower native resolution. The
    planner and target passes report `native_only` and `forecast` separately,
    so verify planner `native_only=true, forecast=0`, target
    `native_only=false` before interpreting results. With a 14-step
    Spectrum turbo run that previously used 8 actual planner predictions
    and 6 forecasts, the expected comparison uses 14 actual planner
    predictions while retaining target forecast opportunities. It increases
    planner runtime, without changing target-stage numerical code or the
    underlying solver. Tested on a real Krea 2 Turbo checkpoint
    (14 steps, 1872→2672→3824 px, one seed): plain `sa_solver` at these
    defaults looked clean natively and with forecasts. Forecasts cut sampling
    time from 70 s to 45 s. If upscale stages still invent small details
    (for example a partly visible object turning into something else), end
    the window earlier, for example `sa_sde_end_percent=0.30`.
    Keep `sa_s_noise` and `er_sde_s_noise` at 1.0 on stochastic steps. A
    stochastic step shrinks the existing noise and re-adds fresh noise of the
    size the next step expects. Below 1.0 it re-adds too little, so the model
    sees an image cleaner than its noise level and removes real texture. On a
    real Krea 2 Turbo checkpoint, 0.5 made both SA-Solver and ER-SDE planners
    look airbrushed or plastic, and 1.0 restored sharp detail. To reduce
    stochasticity, narrow the window or lower `sa_eta`/`er_sde_eta` instead.
    The configured stochastic window uses native Krea 2 percent-to-sigma conversion.
    `sa_eta=0` gives the native ODE mode. Noise increments use the same
    Gaussian patch coupling as ER-SDE, but are generated by the absolute
    sigma transition because SA-Solver samples noise only inside its selected
    SDE interval. In PECE mode, the planner retains both the predictor and
    corrected-state predictions. The target uses each corresponding planner
    anchor for fully locked predictor/corrector calls; native corrected-state
    evaluations resume outside the lock. Restarted resolution stages use a
    fresh native predictor/corrector history. This is not an Euler
    approximation or a one-evaluation PECE shortcut.
  - Non-Euler solvers, including ER-SDE and both SA-Solvers, are experimental.
    Their trajectories differ from Euler and require visual validation with
    matched seeds. Other ancestral/SDE and multi-evaluation solvers (e.g. Heun)
    remain unoffered.
  - With Spectrum Krea 2 PR #2, all ten sampler choices (Euler, DPM++ 2M,
    RES Multistep, Gradient Estimation, DEIS, iPNDM, iPNDM-V, ER-SDE,
    SA-Solver and SA-Solver PECE) can forecast on audited native paths.
    SA-Solver PECE forecasts eligible *predictor* opportunities and always
    evaluates corrected-state calls natively; reports count the latter
    separately.
    DEIS coefficient helpers and ER-SDE sigma/log-SNR conversion helpers
    are separately checked. ER-SDE accepts the validated native scaler, stage,
    noise strength and noise sampler options; unreviewed options or altered
    native solver/helper sources fail closed. Forecasting may change
    image quality, especially anatomy, and must be visually tested with
    the same seed and conditioning. Spectrum's forecast
    history is built only from unlocked target evaluations, as with Euler.
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
  Locked steps cost no high-resolution model evaluations: with Euler the
  target state at the release is computed in closed form from the planner
  predictions and the sampler starts there; other solvers step through the
  locked steps natively with the planner predictions. With a target spatial mask or latent noise mask,
  locked steps use the evaluated path rather than the no-DiT shortcut.
  Spatial masks blend in target predictions; latent noise masks require
  retaining the sampler's masked execution behavior.
- **Resolution stages.** Large jumps from the planner to the output are split
  into a ladder of intermediate resolutions, so no single handoff upscales by
  more than `max_stage_upscale`. The first stage reaches the
  composition-lock release from the planner predictions, exactly like a
  single-stage run. Each later stage takes the finished image of the stage
  below, resized to its own size, re-noises it to the same release noise
  level with its own patch-coupled noise, and runs the remaining steps. Every
  stage therefore repeats the bounded planner-to-target handoff from a sharp
  image. Handing over mid-schedule predictions instead was observed to fail:
  they are blurry noise-conditional averages, and a later, lower-noise start
  rendered that blur as defocus with speckle.
  - `max_stage_upscale` (default 4.0, i.e. 2x per side) is the largest
    pixel-area growth per stage. Jumps larger than that get geometrically
    spaced intermediate stages; 0 always jumps straight to the output.
  - `stage_megapixels` lists explicit intermediate sizes, e.g. `4, 8`, and
    overrides `max_stage_upscale`.
  - Every stage runs all steps after the release, so each intermediate
    stage adds that many evaluations at its own size. With eight steps and
    the default lock, that is four per stage.
- After the release, the high-resolution pass refines freely by default.
  `coarse_lock_strength` and `mid_lock_strength` (default 0) optionally keep
  blending the low and middle spatial frequencies of each denoised prediction
  toward the planner. Their bands are defined relative to the target canvas,
  so at large sizes they include fine structures such as fingers, jewelry and
  seams. The planner version of those structures is upscaled and blurred.
  Nonzero values were observed to double fine contours and soften texture,
  and are kept only for experiments. `composition_lock_sigma` is a starting
  value, not a calibrated optimum.
- **Regeneration mode** (`regeneration_mode`, optional, default
  `trajectory`) selects how the first high-resolution stage starts.
  - `trajectory` is the behavior described above: above
    `composition_lock_sigma` the first high-resolution stage follows the
    planner's denoising trajectory (its intermediate denoised predictions),
    then refines freely.
  - `planner_final_img2img` runs the planner to completion and treats its
    finished latent as an image to refine. Each high-resolution stage resizes
    the finished latent of the stage below (the planner for the first stage),
    and native ComfyUI sampling re-noises it to the start sigma `s`:
    `x = s * noise + (1 - s) * clean`, with `clean` the resized latent in the
    model's internal space (`process_latent_in`) and `noise` a fresh,
    seed-deterministic Gaussian field for each stage, independent of the
    Gaussian field that generated its clean source. The same patch-coupled
    *algorithm* lifts that fresh field to the stage resolution. The
    selected native solver then runs from `s` to zero from a fresh start: no
    multistep history, PECE correction, planner trajectory prediction or
    composition lock is carried into it. `composition_lock_sigma` is not used
    in this mode, and `coarse_lock_strength` and `mid_lock_strength` must be 0
    (nonzero values are rejected). All ten samplers and Spectrum forecasting
    are supported; latent noise masks are rejected.
  - `img2img_start_sigma` (default 0.65, planner-final mode only) chooses `s`.
    The normal schedule selects how many steps remain, then its first retained
    sigma is replaced with the **requested** sigma (no extra model call).
    All later sigmas remain the original native schedule values. The start
    noise therefore responds to each slider change, instead of silently
    snapping to a discrete schedule point. For example, with 20 steps,
    requests **0.30** and **0.35** both retain 2 refinement steps per
    stage, but start at different noise levels (0.30 versus 0.35), yielding
    different target denoiser inputs. Changing the value across a schedule
    boundary can also change the number of remaining steps. Lower noise
    generally preserves more of the finished image but may leave insufficient
    refinement; raise `steps` if needed. Neither setting guarantees
    correct anatomy. The effective sigma and number of steps are logged.

    | `steps` | start 0.45 | start 0.65 | start 0.75 |
    |---|---|---|---|
    | 14 | sigma 0.450, 2 steps | sigma 0.650, 5 steps | sigma 0.750, 6 steps |
    | 20 | sigma 0.450, 4 steps | sigma 0.650, 7 steps | sigma 0.750, 9 steps |

    Each stage evaluates the model once per remaining step (SA-Solver PECE
    also evaluates its corrections), the same per-stage cost as the trajectory
    mode at an equal release sigma.
  - **Planner-final learned upscale without target denoising**:
    Select `regeneration_mode=planner_final_upscale_only` and
    `latent_upscale_method=sesqui_wan21`. The sampler completes the planner,
    enlarges its clean output-space Wan VAE latent to the requested canvas
    (using the configured intermediate sizes, if any), and returns it directly.
    It **does not re-noise** or run the 3552px target diffusion model. It makes
    no target DiT calls, so it cannot obtain texture from high-resolution
    regeneration; the high-resolution result is limited by the trained latent
    upscaler and VAE decode. It is an optional preservation comparator and
    usable artifact-avoidance mode, **not a proven cure for incorrect anatomy**.
    `img2img_start_sigma` and `composition_lock_sigma` have no effect in
    this mode. Both band lock strengths must be zero. The existing
    `trajectory` and `planner_final_img2img` modes are unchanged.
  - **Retired planner-fidelity experiment**: the optional
    `planner_fidelity_strength` and `planner_detail_strength` widgets are
    retained at their existing positions to avoid remapping older saved
    workflows. `planner_fidelity_strength` **must remain 0**: nonzero values
    fail before sampling. Real Krea 2 images exposed severe ghosting at
    full planner measurement fidelity, including when the added detail
    strength was zero. `planner_detail_strength` is inactive at fidelity 0.
    A correct latent downsampling measurement does **not** guarantee a
    correctly decoded or anatomically faithful image. No projection is
    performed in the active modes.
  - To distinguish latent-resize damage from model-generated damage, enable
    `handoff_diagnostics=log_and_save_latents`: besides the finished planner,
    noisy handoff states and denoised steps, each resolution stage now saves
    `stageN_source_resized.latent` in the same diagnostics directory.
    Decode the resized clean source with the **same VAE** as the final result.
    If it already contains duplicated geometry, that occurs before the
    high-resolution diffusion pass; the latent interpolation itself is
    unsuitable for a fidelity-preserving handoff.
  - The mode has been checked only on a small random CPU model for execution
    and initialization correctness. Whether it preserves a real planner image
    better than the trajectory mode is not yet established: compare both
    modes at the same seed, prompt, sizes, sampler and steps, with the
    `lowres_planner` output as the reference.
- **Latent upscale method** (`latent_upscale_method`, optional, default
  `interpolate`) chooses how a *completed* lower-resolution latent is
  enlarged before the next stage re-noises and refines it.
  - `interpolate` is the original bilinear latent resize. It stays bitwise
    identical to earlier versions.
  - `sesqui_wan21` uses the pretrained SesquiLSR Wan 2.1 latent upscaler
    ([LoganBooker/SesquiLSR](https://github.com/LoganBooker/SesquiLSR),
    MIT), bundled under `third_party/sesqui_lsr` with its license. Krea 2
    uses the Wan 2.1 VAE, and ComfyUI hands Krea 2 LATENTs around as raw VAE
    latents (the latent format's z-score is applied only inside
    `process_latent_in`), which is the space these weights were trained on.
    The upscaler only sees clean, completed latents: the finished planner and
    finished stage outputs returned by the native sampler, never a noisy
    sampler state or an internal model tensor.
  - Where it applies: every `planner_final_img2img` stage (the finished
    planner for the first stage, then each finished stage), and trajectory
    stages after the first. The trajectory mode's first stage follows planner
    predictions instead of enlarging a finished image, so it is unchanged;
    `trajectory` without resolution stages therefore has nothing to enlarge
    and rejects `sesqui_wan21` before sampling.
  - Output sizes are exact. SesquiLSR is trained for 1x to 2x per axis; a
    larger jump (for example a direct 1408 px to 3552 px handoff) is split
    into the fewest latent-only passes that each stay within 2x. No diffusion
    stage, sigma, noise or model call is added; solvers, Spectrum, seeds and
    sampler noise are unchanged.
  - The weights (SHA-256 `d1b4c2c1…2d015a`) are verified before the planner
    runs and are never downloaded at runtime. The upscaler runs in fp32 on
    the sampler's device, one image at a time, with one cached instance per
    device.
  - Measured before denoising, with the real Wan 2.1 VAE (ComfyUI
    `wan_2.1_vae.safetensors`): 8 photos with hands, rings, bracelets and
    faces, Lanczos-downsampled to 384 px, encoded, enlarged in latent space,
    decoded, and compared with the original. Mean over the 8 photos (LPIPS
    uses VGG; lower is better):

    | Per-axis ratio | Method | LPIPS | PSNR dB | SSIM | Edge F1 |
    |---|---|---|---|---|---|
    | 1.83x | `interpolate` | 0.415 | 22.7 | 0.638 | 0.348 |
    | 1.83x | `sesqui_wan21` | **0.197** | **30.3** | **0.875** | **0.866** |
    | 2.00x | `interpolate` | 0.445 | 22.5 | 0.633 | 0.317 |
    | 2.00x | `sesqui_wan21` | **0.224** | **29.9** | **0.862** | **0.832** |
    | 2.50x (2 passes) | `interpolate` | 0.482 | 22.4 | 0.629 | 0.281 |
    | 2.50x (2 passes) | `sesqui_wan21` | **0.311** | **28.5** | **0.818** | **0.752** |

    `sesqui_wan21` was better on every metric for all 8 photos at every
    ratio (1.5x too). Bilinear-enlarged latents decode with a latent-grid
    pattern and smeared, doubled contours on fingers and eyes; the learned
    result decodes close to the original. Direct jumps above 2x lose part of
    the gain, so automatic stages of at most 2x per side (the default
    `max_stage_upscale=4`) remain preferable. **This measures the clean
    source handed to the high-resolution pass, not the final sampled image:**
    whether the improvement survives denoising on a real Krea 2 checkpoint
    has not been measured yet.
  - Cost (CPU, 4 threads, fp32, one image): 2.8 s and +0.8 GiB peak memory
    for 1408 px to 2240 px; 7.8 to 8.0 s and +1.8 GiB for 2240 px to 3552 px
    or 2448x1632 to 4608x3072; 10.7 s and +1.9 GiB for a direct two-pass
    1408 px to 3552 px jump. GPU time was not measured; GPU activation
    memory is expected to be of the same order and is transient. No DiT
    evaluations are added.
- **Handoff diagnostics** (`handoff_diagnostics`, optional, default `off`)
  work in both modes and never change the sampled result. `log` reports, for
  every high-resolution step: the sigma, whether the step was
  composition-locked, how far the denoised prediction is from the finished
  planner on the planner grid (`x0_vs_planner_lowband`), how far the state's
  clean component `(x - sigma * noise) / (1 - sigma)` is from the finished
  planner (`state_clean_vs_planner_lowband`) and from the stage's resized
  source (`state_clean_vs_stage_source`, about 0 at the start of a
  planner-final stage by construction), and the prediction's distance from
  the resized source (`x0_vs_stage_source`). Relative errors are L2-norm
  ratios. The clean-component metrics are exact at the start of each stage
  and for deterministic solvers; steps after stochastic noise injection
  include that noise. `log_and_save_latents` also writes the finished
  planner, each stage's starting state (and, in trajectory mode, the state at
  the lock release), and every denoised prediction as `.latent` files under
  `output/scale_locked_diagnostics/`. Load them with LoadLatent and VAE Decode
  to see the actual tensors that the live preview approximates. Nothing is
  decoded inside the sampler.
- **Five-stage handoff diagnostic viewer:** Import
  [`workflows/krea2-handoff-diagnostics.json`](workflows/krea2-handoff-diagnostics.json)
  after one generation with `handoff_diagnostics=log_and_save_latents`.
  The loader reads the latest **complete** run directly from ComfyUI's
  output directory, without manual file copying. Five labeled VAEDecode /
  PreviewImage pairs show: **A** the finished planner; **B** the clean
  enlarged first-stage source *before denoising*; **C** first-stage initial
  x0 prediction; **D** the completed first high-resolution stage; and
  **E** the actual final sampler latent. The completed-stage and final
  files come directly from the native sampler return values, not x0
  callback estimates. The viewer adds no model calls.
  It defaults to `run_folder=latest`, but accepts a specific run folder.
  The `VAELoader`'s example filename does **not** establish which VAE was
  used in the generation: select the same original VAE for all five
  previews. Differences after VAE Decode can be caused by the decoder
  if the wrong weights are selected. **These comparisons do not reproduce
  downstream SPCA/HDR/other image postprocessors**; compare output E to
  the sampler's direct VAE decode, not a processed final image.
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
throughput. For RAW or arbitrary samplers not supported by the Krea-specific
flow-aligned handoff, use the generic Scale-Locked Residual KSampler.
That generic path has different noise-coupling and resolution-stage
semantics; it is not interchangeable with the Krea-specific integration.

## Resolution Selector (Uncapped)

`Resolution Selector (Uncapped)` (category `utilities`) behaves like ComfyUI's
built-in Resolution Selector: the same aspect ratios, megapixel convention
(1 MP = 1024x1024 pixels), `multiple` rounding and live resolution preview.
The built-in node caps `megapixels` at 16; this one accepts up to 256, the
largest square canvas ComfyUI latents allow (16384 pixels per side). A ratio
and size that would exceed 16384 pixels on either side is rejected with an
error instead of failing later in the empty-latent node. On ComfyUI versions
without the preview widget it falls back to the same calculation without the
preview.

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

No extra Python dependencies are required beyond ComfyUI + PyTorch. The
SesquiLSR Wan 2.1 latent-upscaler weights used by
`latent_upscale_method=sesqui_wan21` are included in this repository; no
separate download or SesquiLSR custom node is needed.

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
- multi-stage progressive ladders in the generic Scale-Locked samplers (the Krea 2 Turbo node supports resolution stages),
- exact support tuning for every possible exotic custom sampler,
- verified bindings for every historical Impact Pack hook/provider variant.

## Files

- `__init__.py` - node registration
- `nodes.py` - ComfyUI node definitions, public guider node, and Impact hook/provider nodes
- `slrd_runtime.py` - shared ComfyUI runtime for planner capture, nested noise, guider patching, and final sampling
- `latent_upscale.py` - completed-latent enlargement backends (`interpolate`, `sesqui_wan21`)
- `third_party/sesqui_lsr/` - vendored SesquiLSR architecture, Wan 2.1 weights, MIT license and provenance notice
- `slrd_core.py` - algorithm core, nested noise, latent resizing, residual locking, trajectory helpers
