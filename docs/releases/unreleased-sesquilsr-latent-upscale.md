# Unreleased: learned SesquiLSR latent upscaling at Krea 2 handoffs

## New optional control on `Krea 2 Turbo Flow-Aligned Regenerate [Experimental]`

`latent_upscale_method` is appended after the existing optional inputs.
Saved workflows keep their widget values and behavior.

- `interpolate` (default): the original bilinear latent resize, bitwise
  identical to earlier versions.
- `sesqui_wan21`: the pretrained SesquiLSR Wan 2.1 latent upscaler.

## What changes with `sesqui_wan21`

Only the enlargement of a completed lower-resolution latent before the next
stage re-noises it:

| Regeneration mode | Learned handoffs |
|---|---|
| `planner_final_img2img` | Every stage: finished planner → first stage, then each finished stage → next stage |
| `trajectory` with resolution stages | Each finished stage → next stage. The first stage still follows planner predictions and is unchanged. |
| `trajectory` without resolution stages | None; `sesqui_wan21` is rejected before sampling |

Unchanged: planner, sigma schedules, start sigma, img2img forward noise and
its seeds, patch-coupled noise, solver algorithms and history, model-call
counts, Spectrum forecasting, output ports and the default behavior.

Latent space: Krea 2 uses the Wan 2.1 VAE and ComfyUI's `Wan21` latent
format. ComfyUI's Wan VAE encoder returns the raw encoder mean, and the
latent format's per-channel z-score is applied only inside
`process_latent_in`. The completed latents handed between stages are
returned by the native sampler after `process_latent_out`, so they are raw
VAE latents, the space SesquiLSR's Wan21 weights were trained on (upstream's
own ComfyUI node uses an identity adaptor for Krea 2 LATENTs for the same
reason).

Scale range: the model is trained for 1x to 2x per axis. Larger handoffs are
split into the fewest latent-only passes that each stay within 2x, ending at
the exact target size. This never adds a diffusion stage.

## Provenance and license

- Upstream: <https://github.com/LoganBooker/SesquiLSR>, commit
  `821e6db7412089f2a5c0a106c42d3c1a396b9515`, MIT License.
- Vendored unmodified: `sesqui_lsr/upscaler.py` (SHA-256
  `f10d9751…cf281`) and `models/upscaler_Wan21.safetensors` (SHA-256
  `d1b4c2c1d80b2af81e6bbfd38501ffb5adffb9f7c0ed2599555b837d0d2d015a`,
  3,091,382 values). Upstream ships its weights in the same MIT-licensed
  repository without a separate model license.
- The weight hash is verified before the planner runs. Nothing is downloaded
  at runtime.

## Measurements

Decoded with the real Wan 2.1 VAE before any denoising. Eight photos with
hands, rings, bracelets and faces (from the `bghira/pseudo-camera-10k`
dataset, as in upstream's benchmark) were Lanczos-downsampled to 384 px,
encoded, enlarged in latent space, decoded and compared with the original.
Means over the 8 photos:

| Per-axis ratio | Method | LPIPS-VGG ↓ | PSNR dB ↑ | SSIM ↑ | Edge F1 ↑ |
|---|---|---|---|---|---|
| 1.50x | VAE round trip (ceiling) | 0.035 | 35.6 | 0.955 | 0.962 |
| 1.50x | `interpolate` (bilinear) | 0.360 | 22.8 | 0.645 | 0.395 |
| 1.50x | bicubic | 0.316 | 23.0 | 0.652 | 0.441 |
| 1.50x | `sesqui_wan21` | **0.141** | **30.7** | **0.896** | **0.905** |
| 1.83x | `interpolate` | 0.415 | 22.7 | 0.638 | 0.348 |
| 1.83x | `sesqui_wan21` | **0.197** | **30.3** | **0.875** | **0.866** |
| 2.00x | `interpolate` | 0.445 | 22.5 | 0.633 | 0.317 |
| 2.00x | `sesqui_wan21` | **0.224** | **29.9** | **0.862** | **0.832** |
| 2.50x (2 passes) | `interpolate` | 0.482 | 22.4 | 0.629 | 0.281 |
| 2.50x (2 passes) | `sesqui_wan21` | **0.311** | **28.5** | **0.818** | **0.752** |

`sesqui_wan21` won on every metric for every photo at every ratio. Visually,
bilinear-enlarged latents decode with a latent-grid pattern and smeared,
doubled finger and eye contours; the learned result decodes close to the
original. LPIPS uses VGG16 rather than upstream's setup, so absolute values
are not comparable with upstream's table.

**Evidence provenance and reproducibility:** these figures were collected
in a separate cloud experiment and documented here; the exact eight image
identifiers, generated panels, and benchmark execution script are not included
in this repository. Consequently the CI suite cannot independently reproduce
the reported image-quality metrics. The CPU native integration tests do
verify the shipped weights, architecture, model-input construction and solver
compatibility; they do **not** verify the PSNR/LPIPS table or image quality
after Krea 2 high-resolution regeneration.

Cost on CPU (4 threads, fp32, one image): 2.8 s / +0.8 GiB peak memory for
1408 px → 2240 px; 7.8–8.0 s / +1.8 GiB for 2240 px → 3552 px or
2448×1632 → 4608×3072; 10.7 s / +1.9 GiB for a direct two-pass
1408 px → 3552 px jump. GPU time was not measured. No DiT evaluations are
added.

Structural checks: the lightweight suite runs the real weights for exact
sizes, BCHW/BC1HW layouts, batches, multi-pass splits, determinism, equality
with direct upstream inference, and rejection of missing or corrupted
weights. The native ComfyUI CPU smoke shows the learned start latent is
used for all 10 samplers in both modes. It also shows the first DiT input
equals `s·noise + (1−s)·process_latent_in(learned start)`, and that noise,
sigmas and model-call counts are unchanged. Spectrum forecasts with no
fallbacks, and interrupts leave no state behind. `interpolate` is bitwise
identical to the previous head across 162 native cases.

## Limits

- A cleaner enlarged latent is not proof that the high-resolution diffusion
  pass will keep the planner's anatomy or details. The decoded-reconstruction
  results above measure the source handed to the denoiser, not the final
  sampled image.
- Real Krea 2 checkpoint generations with `sesqui_wan21` have not been
  compared yet.
- Latent noise masks remain unsupported in `planner_final_img2img`.


## Planner-final upscale-only output mode

A third `regeneration_mode=planner_final_upscale_only` is an opt-in way
to return the completed planner after SesquiLSR enlargement, **without
starting any additional high-resolution diffusion**. It was introduced
after a real Krea 2 generation with `img2img_start_sigma=0.40`, 12 total
planner steps and 3552px final resolution showed that the clean enlarged
initial state matched its source (relative error 5.8e-8), while two native
high-resolution SA-Solver predictions still diverged from that source by
about 0.19 in relative latent L2. Those values describe whole latents,
not the error of any specific anatomical feature.

The mode returns the final enlarged LATENT in both `output` and
`denoised_output`, and the untouched finished planner in `lowres_planner`.
It executes each configured latent upscaling stage without denoiser calls;
the number of high-resolution model evaluations is **zero**. Neither noise
nor sigma-start controls nor learned-model lock projections are applied.
Use the normal VAE to decode the result. The existing
`planner_final_img2img` and `trajectory` modes retain their numerical
behavior and remain the defaults.

This mode isolates and removes the target-model regeneration as a possible
source of new geometry. However, it also forgoes diffusion-generated
high-resolution texture and could expose blur or reconstruction artifacts
from latent upscaling/VAE decoding. Its visual effect with the user's real
checkpoint has **not** been evaluated on a GPU; correct native execution
alone does not establish improved anatomy.
