from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import time
from typing import Any

import math
import torch
import comfy.samplers
import comfy.model_base

from .slrd_core import build_nested_noise, clone_latent, full_lock_prefix_length, init_scale_lock_state, latent_target_hw_from_megapixels, parse_stage_megapixels, plan_stage_megapixels, resize_4d_tensor, resize_mask
from .resolution import DISPLAY_NAME as RESOLUTION_DISPLAY_NAME, NODE_ID as RESOLUTION_NODE_ID, ResolutionSelectorUncapped
from .diagnostic_load import SLRDLoadKrea2HandoffDiagnostics
from .spectral_progressive import CALIBRATION_TYPE, build_calibration, describe_calibration, parse_transition_sigmas
from .slrd_runtime import (
    LOCK_SCHEDULE_OPTIONS,
    MID_SCHEDULE_OPTIONS,
    ScaleLockConfig,
    _resolve_sampler_device,
    apply_scale_lock_to_noise_prediction,
    calculate_sigmas,
    build_runtime_context_from_advanced,
    clean_latent,
    compat_sampler_names,
    compat_scheduler_names,
    create_cfg_guider,
    fix_latent_channels,
    guard_sampler_alignment,
    krea2_sampler_names,
    krea2_er_sde_options,
    krea2_sa_solver_options,
    krea2_native_refine_hw,
    sa_native_only_scope,
    effective_er_sde_sigmas,
    make_lowres_latent,
    prepare_noise,
    run_scale_locked_ksampler,
    sample_with_runtime,
    apply_scale_lock_to_guider,
    clone_guider_for_scale_lock,
    restore_original_predict_noise,
)


logger = logging.getLogger(__name__)


class ScaleLockedResidualKSampler:
    CATEGORY = "sampling/scale_locked"
    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("output", "lowres_planner", "denoised_output")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 24, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (compat_sampler_names(),),
                "scheduler": (compat_scheduler_names(),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 16.0, "step": 0.05, "round": 0.01}),
                "lock_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_strength_start": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_strength_end": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_schedule": (LOCK_SCHEDULE_OPTIONS,),
                "lock_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "lock_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "coarse_cutoff": ("FLOAT", {"default": 0.33, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_cutoff": ("FLOAT", {"default": 0.60, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_schedule": (MID_SCHEDULE_OPTIONS,),
                "mid_band_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "mid_band_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "manifold_enabled": ("BOOLEAN", {"default": False}),
                "manifold_strength": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_strength_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_schedule": (LOCK_SCHEDULE_OPTIONS,),
                "manifold_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "manifold_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "manifold_cutoff": ("FLOAT", {"default": 0.18, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_radial_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "round": 0.001}),
                "manifold_anisotropy": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "manifold_translation_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "round": 0.001}),
                "manifold_anchor_mix": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_mean_anchor_mix": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_contrast_restore": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_energy_tether": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_channel_tether": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_energy_gain_cap": ("FLOAT", {"default": 1.75, "min": 1.0, "max": 4.0, "step": 0.05, "round": 0.01}),
                "manifold_max_shift_px": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 64.0, "step": 0.1, "round": 0.01}),
                "nested_noise_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 4.0, "step": 0.01, "round": 0.001}),
                "add_noise": ("BOOLEAN", {"default": True}),
                "pin_anchors": ("BOOLEAN", {"default": True}),
                "sampler_guard": (["warn", "error", "off"],),
            },
            "optional": {
                "lock_mask": ("MASK",),
                "manifold_mask": ("MASK",),
            },
        }

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        target_megapixels,
        lock_strength,
        lock_strength_start,
        lock_strength_end,
        lock_schedule,
        lock_schedule_hold,
        lock_schedule_power,
        coarse_cutoff,
        mid_band_cutoff,
        mid_band_strength,
        mid_band_strength_start,
        mid_band_strength_end,
        mid_band_schedule,
        mid_band_schedule_hold,
        mid_band_schedule_power,
        manifold_enabled,
        manifold_strength,
        manifold_strength_start,
        manifold_strength_end,
        manifold_schedule,
        manifold_schedule_hold,
        manifold_schedule_power,
        manifold_cutoff,
        manifold_radial_strength,
        manifold_anisotropy,
        manifold_translation_strength,
        manifold_anchor_mix,
        manifold_mean_anchor_mix,
        manifold_contrast_restore,
        manifold_energy_tether,
        manifold_channel_tether,
        manifold_energy_gain_cap,
        manifold_max_shift_px,
        nested_noise_strength,
        add_noise,
        pin_anchors,
        sampler_guard,
        lock_mask=None,
        manifold_mask=None,
    ):
        result = run_scale_locked_ksampler(
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            target_megapixels=target_megapixels,
            nested_noise_strength=nested_noise_strength,
            add_noise=add_noise,
            pin_anchors=pin_anchors,
            sampler_guard=sampler_guard,
            config=_scale_lock_config(
                lock_strength=lock_strength,
                lock_strength_start=lock_strength_start,
                lock_strength_end=lock_strength_end,
                coarse_cutoff=coarse_cutoff,
                mid_band_cutoff=mid_band_cutoff,
                mid_band_strength=mid_band_strength,
                lock_schedule=lock_schedule,
                lock_schedule_hold=lock_schedule_hold,
                lock_schedule_power=lock_schedule_power,
                mid_band_strength_start=mid_band_strength_start,
                mid_band_strength_end=mid_band_strength_end,
                mid_band_schedule=mid_band_schedule,
                mid_band_schedule_hold=mid_band_schedule_hold,
                mid_band_schedule_power=mid_band_schedule_power,
                manifold_enabled=manifold_enabled,
                manifold_strength=manifold_strength,
                manifold_strength_start=manifold_strength_start,
                manifold_strength_end=manifold_strength_end,
                manifold_schedule=manifold_schedule,
                manifold_schedule_hold=manifold_schedule_hold,
                manifold_schedule_power=manifold_schedule_power,
                manifold_cutoff=manifold_cutoff,
                manifold_radial_strength=manifold_radial_strength,
                manifold_anisotropy=manifold_anisotropy,
                manifold_translation_strength=manifold_translation_strength,
                manifold_anchor_mix=manifold_anchor_mix,
                manifold_mean_anchor_mix=manifold_mean_anchor_mix,
                manifold_contrast_restore=manifold_contrast_restore,
                manifold_energy_tether=manifold_energy_tether,
                manifold_channel_tether=manifold_channel_tether,
                manifold_energy_gain_cap=manifold_energy_gain_cap,
                manifold_max_shift_px=manifold_max_shift_px,
                lock_mask=lock_mask,
                manifold_mask=manifold_mask,
            ),
        )
        return (result.output, result.lowres_planner, result.denoised_output)


class Krea2TurboFlowAlignedRegenerate:
    """Experimental Krea 2 Turbo two-pass composition-lock sampler."""

    CATEGORY = "sampling/scale_locked"
    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("output", "lowres_planner", "denoised_output")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "planner_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
                "coarse_lock_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Low-band pull toward the planner after the composition lock releases. "
                               "0 lets the high-resolution pass refine freely; nonzero values can double fine contours and soften detail."}),
                "mid_lock_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Mid-band pull toward the planner after the composition lock releases, relative to the low band. "
                               "0 is recommended; nonzero values can double fine contours and soften detail."}),
            },
            "optional": {
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000,
                                   "tooltip": "Steps in each pass. Connect the same value to a native sampler for matched target-pass comparisons."}),
                "composition_lock_sigma": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Target steps starting above this noise level reuse the complete planner prediction, "
                               "so the high-resolution pass cannot form a second, displaced composition. "
                               "Lower values keep more planner structure; 1.0 disables the full lock."}),
                "max_stage_upscale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 64.0, "step": 0.25,
                    "tooltip": "Largest pixel-area growth per resolution stage (4.0 = 2x per side). Larger jumps are "
                               "split into geometrically spaced intermediate stages. 0 always jumps from the planner "
                               "straight to the output size. Ignored when stage_megapixels is set."}),
                "stage_megapixels": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Optional explicit intermediate stage sizes in megapixels, ascending and between the "
                               "planner and the output, e.g. '4, 8'. Empty uses max_stage_upscale."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                    "tooltip": "ComfyUI CFG. Krea 2 Turbo normally uses 1.0 (no classifier-free guidance). Higher values alter conditioning and require visual validation."}),
                "sampler_name": (krea2_sampler_names(), {"default": "euler",
                    "tooltip": "Solver for planner and target passes. Every option skips the target model during composition-locked steps and supports resolution stages. "
                               "Euler reaches the lock release in closed form; the others run their native multistep update through the locked steps using the planner predictions. "
                               "ER-SDE and SA-Solver use native stochastic updates with patch-coupled noise across resolutions. "
                               "SA-Solver PECE retains separate predictor and correction evaluations. "
                               "Non-Euler solvers are experimental and need visual validation."}),
                "scheduler": (compat_scheduler_names(), {"default": "simple",
                    "tooltip": "ComfyUI sigma schedule applied consistently to planner and target. Changes locked-step count and denoising trajectory. Experimental alternatives need visual validation."}),
                "er_sde_solver_type": (["ER-SDE", "Reverse-time SDE", "ODE"], {"default": "ER-SDE",
                    "tooltip": "Only used with er_sde. Same noise-scaling modes as ComfyUI's native SamplerER_SDE; ODE disables stochastic injection."}),
                "er_sde_max_stage": ("INT", {"default": 3, "min": 1, "max": 3,
                    "tooltip": "Only used with er_sde. Maximum native solver order (1 to 3); history is restarted at each new resolution stage."}),
                "er_sde_eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Only used with er_sde. Native stochastic-strength exponent; zero selects ODE with no stochastic noise."}),
                "er_sde_s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Only used with er_sde. Multiplies its patch-coupled per-step stochastic increments; ODE forces zero. "
                               "Keep 1.0: lower values re-add less noise than each step removes, which gives airbrushed, plastic results."}),
                "sa_eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "SA-Solver only: stochastic strength (0 = ODE)."}),
                "sa_sde_start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "SA-Solver only: start of the stochastic model-sigma interval."}),
                "sa_sde_end_percent": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "SA-Solver only: end of the stochastic model-sigma interval. The default 0.35 (model sigma "
                               "about 0.68 at Krea 2's flow shift) keeps stochastic steps in the planner and makes the "
                               "high-resolution stages deterministic. ComfyUI's native 0.8 keeps every stage step stochastic; "
                               "with few steps per stage that produced visible artifacts with plain SA-Solver. If upscale stages invent small "
                               "details, end the window earlier, for example 0.30."}),
                "sa_s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "SA-Solver only: SDE noise multiplier, patch-coupled across scales. Keep 1.0: lower values re-add less "
                               "noise than each stochastic step removes, which gives airbrushed, plastic results. "
                               "Narrow the SDE window or lower sa_eta to reduce stochasticity instead."}),
                "sa_predictor_order": ("INT", {"default": 3, "min": 1, "max": 6,
                    "tooltip": "SA-Solver only: native Adams predictor order."}),
                "sa_corrector_order": ("INT", {"default": 4, "min": 0, "max": 6,
                    "tooltip": "SA-Solver only: native corrector order. PECE reevaluates corrections when greater than 0."}),
                "sa_simple_order_2": ("BOOLEAN", {"default": False,
                    "tooltip": "SA-Solver only: native simplified second-order coefficients."}),
                "sa_spectrum_forecasts": ("BOOLEAN", {"default": True,
                    "tooltip": "SA-Solver only: allow Spectrum predictor forecasts when a Spectrum Krea 2 model is connected. "
                               "Off runs every SA predictor and PECE correction natively (a quality reference). "
                               "Forecasts on stochastic steps never reuse the forecast's noise estimate; they are most "
                               "effective on the deterministic high-resolution stages set by sa_sde_end_percent."}),
                "sa_planner_spectrum_forecasts": ("BOOLEAN", {"default": True,
                    "tooltip": "SA-Solver only: enable Spectrum forecasts in the low-resolution planner. "
                               "Off evaluates every planner step natively while preserving Spectrum acceleration "
                               "in the intermediate and final high-resolution stages. This is a quality comparison "
                               "for planner texture loss; the planner uses the same native noise and solver."}),
                "regeneration_mode": (["trajectory", "planner_final_img2img", "planner_final_upscale_only", "planner_final_native_refine", "spectral_progressive"], {"default": "trajectory",
                    "tooltip": "trajectory (default, unchanged): the first high-resolution stage follows the planner's "
                               "denoising trajectory above composition_lock_sigma, then refines freely. "
                               "planner_final_img2img: every high-resolution stage starts from the finished image "
                               "below it (the completed planner for the first stage), resized and re-noised to "
                               "img2img_start_sigma, then denoised natively to zero. In this mode "
                               "composition_lock_sigma is not used, and coarse_lock_strength and mid_lock_strength "
                               "must be 0. planner_final_upscale_only completes the same planner, enlarges its clean "
                               "Wan VAE latent, and returns the enlarged latent without high-resolution denoising. "
                               "Use sesqui_wan21 for learned latent scaling. No extra target diffusion texture is "
                               "synthesized; high-resolution denoiser evaluations are exactly zero. "
                               "planner_final_native_refine: finish the planner, run one img2img refinement at "
                               "at most 2048 pixels per axis, then SesquiLSR-upscale the completed latent "
                               "to final dimensions WITHOUT diffusion above 2048. Requires sesqui_wan21; "
                               "stage_megapixels must be empty. This balances native-detail refinement "
                               "against structural drift at unsupported target resolutions. "
                               "spectral_progressive (experimental): no separate planner pass. One denoising "
                               "trajectory starts at the planner size and grows to each stage and the output size "
                               "at transition points, spectrally expanding the noisy state with fresh "
                               "high-frequency noise and the matching sigma alignment. Early steps run at the "
                               "smaller sizes. Transitions come from progressive_transition_sigmas, a connected "
                               "spectral_calibration, or (single transition only) composition_lock_sigma. Band "
                               "lock strengths must be 0; multistep solver history restarts at each transition."}),
                "img2img_start_sigma": ("FLOAT", {"default": 0.65, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "planner_final_img2img only: noise level the finished image is re-noised to before "
                               "each high-resolution stage. The first sigma of the native refinement suffix is "
                               "set to this exact value, while later sigma points remain unchanged. This means "
                               "values like 0.30 and 0.35 have distinct effects even at 20 steps. Lower values can "
                               "also leave fewer high-resolution steps across scheduler boundaries (raise steps for more). Lower values keep "
                               "more of the finished image and allow less structural change; higher values regenerate "
                               "more. Neither guarantees correct anatomy."}),
                "handoff_diagnostics": (["off", "log", "log_and_save_latents"], {"default": "off",
                    "tooltip": "Opt-in handoff diagnostics for both regeneration modes. log reports, per "
                               "high-resolution step, how far the denoised prediction and the state's clean component "
                               "are from the finished planner. log_and_save_latents also writes each stage's starting "
                               "state and every denoised prediction as .latent files under "
                               "output/scale_locked_diagnostics for LoadLatent + VAE Decode. Sampling is unchanged."}),
                "planner_fidelity_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "DISABLED: real-checkpoint testing showed severe duplicated anatomy even when "
                               "latent lowband errors were zero. Retained only to preserve previously saved "
                               "workflow widget ordering. Must remain 0; nonzero is rejected before sampling."}),
                "planner_detail_strength": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "INACTIVE compatibility widget from the disabled planner-fidelity experiment. "
                               "Retained only for saved workflow ordering. It has no effect while fidelity is 0. "
                               "Original trajectory and planner-final img2img modes remain available."}),
                "latent_upscale_method": (["interpolate", "sesqui_wan21"], {"default": "interpolate",
                    "tooltip": "How a completed lower-resolution latent is enlarged before the next stage re-noises "
                               "and refines it. interpolate (default) keeps the previous behavior. sesqui_wan21 "
                               "uses the pretrained SesquiLSR Wan 2.1 latent upscaler (bundled, MIT) on the clean "
                               "VAE latent; jumps above 2x per side use several latent-only passes, never extra "
                               "denoising steps. Applies to every planner_final_img2img stage and to trajectory "
                               "stages after the first; the trajectory mode's first stage follows planner "
                               "predictions and is unchanged. With no intermediate stages, SesquiLSR is "
                               "inactive and the unchanged trajectory runs with an explicit warning. "
                               "Empty stage_megapixels still allows automatic intermediate stages when "
                               "max_stage_upscale is exceeded. A cleaner enlarged latent does not by itself prevent "
                               "the high-resolution model from changing details."}),
                "progressive_transition_sigmas": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "spectral_progressive only: one schedule sigma per resolution change, strictly "
                               "decreasing, e.g. '0.85, 0.6' for planner -> stage -> output. Each transition "
                               "happens at the first step whose sigma is at or below its value. Empty uses the "
                               "connected spectral_calibration, or composition_lock_sigma for a single transition. "
                               "Ignored in other regeneration modes; the saved value is retained when switching modes."}),
                "spectral_tolerance": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.5, "step": 0.001,
                    "tooltip": "Noise-dominance tolerance delta for spectral transition times (used with a "
                               "spectral_calibration). Smaller values hand over to the larger grid earlier and "
                               "keep fewer steps at small sizes."}),
                "spectral_calibration": ("SLRD_SPECTRUM", {
                    "tooltip": "Measured clean-latent power spectrum from the Krea 2 Spectral Calibration node. "
                               "spectral_progressive derives its transition sigmas from it; trajectory mode only "
                               "logs whether composition_lock_sigma hands over before frequencies the target "
                               "cannot recover become informative. Sampling is otherwise unchanged."}),
            },
        }

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        planner_megapixels,
        coarse_lock_strength,
        mid_lock_strength,
        steps=8,
        composition_lock_sigma=0.80,
        max_stage_upscale=4.0,
        stage_megapixels="",
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        er_sde_solver_type="ER-SDE",
        er_sde_max_stage=3,
        er_sde_eta=1.0,
        er_sde_s_noise=1.0,
        sa_eta=1.0,
        sa_sde_start_percent=0.2,
        sa_sde_end_percent=0.35,
        sa_s_noise=1.0,
        sa_predictor_order=3,
        sa_corrector_order=4,
        sa_simple_order_2=False,
        sa_spectrum_forecasts=True,
        sa_planner_spectrum_forecasts=True,
        regeneration_mode="trajectory",
        img2img_start_sigma=0.65,
        handoff_diagnostics="off",
        planner_fidelity_strength=0.0,
        planner_detail_strength=0.20,
        latent_upscale_method="interpolate",
        progressive_transition_sigmas="",
        spectral_tolerance=0.01,
        spectral_calibration=None,
    ):
        if latent_upscale_method not in ("interpolate", "sesqui_wan21"):
            raise ValueError("latent_upscale_method must be interpolate or sesqui_wan21")
        if regeneration_mode not in ("trajectory", "planner_final_img2img", "planner_final_upscale_only",
                                     "planner_final_native_refine", "spectral_progressive"):
            raise ValueError("invalid regeneration_mode; select trajectory, planner_final_img2img, "
                             "planner_final_upscale_only, planner_final_native_refine or spectral_progressive")
        progressive = regeneration_mode == "spectral_progressive"
        # Manual spectral transitions are dormant in every other regeneration mode.
        # Preserve the widget value so switching back restores the schedule.
        transition_sigmas = parse_transition_sigmas(progressive_transition_sigmas) if progressive else []
        if regeneration_mode == "planner_final_native_refine" and latent_upscale_method != "sesqui_wan21":
            raise ValueError("planner_final_native_refine requires latent_upscale_method=sesqui_wan21")
        if handoff_diagnostics not in ("off", "log", "log_and_save_latents"):
            raise ValueError("handoff_diagnostics must be off, log or log_and_save_latents")
        img2img = regeneration_mode in ("planner_final_img2img", "planner_final_native_refine")
        upscale_only = regeneration_mode == "planner_final_upscale_only"
        native_refine = regeneration_mode == "planner_final_native_refine"
        if (isinstance(planner_fidelity_strength, bool)
                or not isinstance(planner_fidelity_strength, (int, float))
                or not math.isfinite(planner_fidelity_strength)
                or not 0.0 <= planner_fidelity_strength <= 1.0):
            raise ValueError("planner_fidelity_strength must be finite and within [0, 1]")
        if planner_fidelity_strength != 0.0:
            raise ValueError(
                "planner_fidelity_strength was disabled after real Krea 2 checkpoint "
                "tests showed severe duplicated anatomy. Set it to 0."
            )
        if (isinstance(planner_detail_strength, bool)
                or not isinstance(planner_detail_strength, (int, float))
                or not math.isfinite(planner_detail_strength)
                or not 0.0 <= planner_detail_strength <= 1.0):
            raise ValueError("planner_detail_strength must be finite and within [0, 1]")
        if img2img:
            if (isinstance(img2img_start_sigma, bool) or not isinstance(img2img_start_sigma, (int, float))
                    or not math.isfinite(img2img_start_sigma) or not 0.0 < img2img_start_sigma < 1.0):
                raise ValueError("img2img_start_sigma must be a finite value strictly between 0 and 1")
        if img2img or upscale_only:
            if coarse_lock_strength != 0.0 or mid_lock_strength != 0.0:
                raise ValueError(
                    "planner-final regeneration does not apply planner trajectory anchors; "
                    "set coarse_lock_strength and mid_lock_strength to 0"
                )
        if not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or not 0.0 <= cfg <= 100.0:
            raise ValueError("Krea 2 CFG must be finite and within [0, 100]")
        if sampler_name not in krea2_sampler_names():
            raise ValueError(
                f"Unsupported Krea 2 sampler {sampler_name!r}; supported: {', '.join(krea2_sampler_names())}"
            )
        if scheduler not in compat_scheduler_names():
            raise ValueError(f"Unsupported Krea 2 scheduler: {scheduler!r}")
        sampler_options = None
        composition_lock_sigma = float(composition_lock_sigma)
        if not math.isfinite(composition_lock_sigma) or not 0.0 <= composition_lock_sigma <= 1.0:
            raise ValueError("composition_lock_sigma must be within [0, 1]")
        if not isinstance(getattr(model, "model", None), comfy.model_base.Krea2):
            raise ValueError("Krea 2 Turbo Flow-Aligned Regenerate requires a native ComfyUI Krea2 MODEL")
        if sampler_name == "er_sde":
            sampler_options = krea2_er_sde_options(
                er_sde_solver_type, er_sde_max_stage, er_sde_eta, er_sde_s_noise
            )
            # Report the validated, effective backend values for each queued
            # invocation. ODE/eta=0 force s_noise=0; the UI widgets alone
            # cannot prove which parameters the sampler actually received.
            er_scaler = sampler_options["noise_scaler"]
            logger.info(
                "Krea 2 ER-SDE invocation: solver_type=%s eta=%.6g "
                "max_stage=%d s_noise=%.6g seed=%d",
                er_scaler.solver_type, er_scaler.eta,
                sampler_options["max_stage"], sampler_options["s_noise"], seed,
            )
        elif sampler_name in ("sa_solver", "sa_solver_pece"):
            if type(sa_spectrum_forecasts) is not bool:
                raise ValueError("sa_spectrum_forecasts must be a boolean")
            if type(sa_planner_spectrum_forecasts) is not bool:
                raise ValueError("sa_planner_spectrum_forecasts must be a boolean")
            sampler_options = krea2_sa_solver_options(
                model, eta=sa_eta, sde_start_percent=sa_sde_start_percent,
                sde_end_percent=sa_sde_end_percent, s_noise=sa_s_noise,
                predictor_order=sa_predictor_order, corrector_order=sa_corrector_order,
                simple_order_2=sa_simple_order_2,
            )
        if not isinstance(latent_image, dict) or not isinstance(latent_image.get("samples"), torch.Tensor):
            raise TypeError("Krea 2 requires a LATENT dictionary with tensor samples")
        # Native ComfyUI uses four-channel EmptyLatentImage placeholders for Krea 2,
        # then expands them to sixteen channels during normal sampler preflight.
        prepared_latent = fix_latent_channels(model, latent_image)
        latent = prepared_latent["samples"]
        image_layout = latent.ndim == 4 or (latent.ndim == 5 and latent.shape[2] == 1)
        if not image_layout or latent.shape[1] != 16:
            raise ValueError("Krea 2 requires a Bx16xHxW or Bx16x1xHxW image LATENT after native empty-latent conversion")
        target_h, target_w = map(int, latent.shape[-2:])
        if target_h % 2 or target_w % 2:
            raise ValueError("Krea 2 requires even latent dimensions (16-pixel output alignment)")
        low_h, low_w = latent_target_hw_from_megapixels(latent, planner_megapixels)
        if low_h > target_h or low_w > target_w or (low_h, low_w) == (target_h, target_w):
            raise ValueError("Final LATENT must be larger than the planner canvas")

        explicit_stages = parse_stage_megapixels(stage_megapixels)
        if native_refine:
            if explicit_stages:
                raise ValueError("planner_final_native_refine computes one 2048px-or-smaller refinement "
                                 "automatically; clear stage_megapixels")
            krea2_native_refine_hw((low_h, low_w), (target_h, target_w))
            stage_mp = []
        elif explicit_stages:
            stage_mp = explicit_stages
        else:
            stage_mp = plan_stage_megapixels(low_h * low_w * 64.0, target_h * target_w * 64.0, max_stage_upscale)
        stage_hw = []
        previous = (low_h, low_w)
        for megapixels in stage_mp:
            hw = latent_target_hw_from_megapixels(latent, megapixels)
            if hw == previous and not explicit_stages:
                continue  # Rounding to the 16-pixel grid merged two automatic stages.
            if hw[0] < previous[0] or hw[1] < previous[1] or hw == previous:
                raise ValueError(
                    "stage_megapixels must increase from the planner size, and each stage must grow on the 16-pixel grid"
                )
            stage_hw.append(hw)
            previous = hw
        if stage_hw and (target_h < previous[0] or target_w < previous[1] or (target_h, target_w) == previous):
            if explicit_stages:
                raise ValueError("stage_megapixels must stay below the output LATENT size")
            stage_hw.pop()
        schedule = [float(v) for v in calculate_sigmas(model, scheduler=scheduler, steps=steps, denoise=1.0).tolist()]
        if (len(schedule) != steps + 1 or any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in schedule)
                or any(a <= b for a, b in zip(schedule, schedule[1:])) or schedule[-1] != 0.0):
            raise ValueError("Krea 2 requires a strictly decreasing flow sigma schedule in [0, 1] ending at zero")
        if img2img:
            start = full_lock_prefix_length(schedule, float(img2img_start_sigma))
            if start == 0 or start >= steps:
                raise ValueError(
                    "img2img_start_sigma leaves no usable refinement steps on this schedule; "
                    "it must lie below the first sigma and at or above the last nonzero sigma"
                )
            logger.info(
                "Krea 2 %s: requested start sigma %.4f -> effective sigma %.6f at suffix step %d of %d "
                "(%d high-resolution steps per stage); later schedule sigmas unchanged; composition_lock_sigma is not used",
                regeneration_mode, float(img2img_start_sigma), float(img2img_start_sigma), start, steps, steps - start,
            )
            if stage_hw:
                logger.info(
                    "Krea 2 resolution stages: planner %sx%s -> %s -> output %sx%s",
                    low_w * 8, low_h * 8, " -> ".join(f"{w * 8}x{h * 8}" for h, w in stage_hw),
                    target_w * 8, target_h * 8,
                )
        elif stage_hw:
            if not upscale_only and not progressive:
                # ER-SDE shifts the initial CONST sigma away from 1.0 for finite
                # log-SNR; lock/release planning must use the sigma actually seen
                # by the native denoiser, not the nominal outer schedule.
                # planner_final_upscale_only never uses the composition lock.
                lock_schedule = (
                    effective_er_sde_sigmas(torch.tensor(schedule, dtype=torch.float64), model).tolist()
                    if sampler_name in ("er_sde", "sa_solver", "sa_solver_pece") else schedule
                )
                release = full_lock_prefix_length(lock_schedule, composition_lock_sigma)
                if release >= len(schedule) - 1:
                    raise ValueError("resolution stages need at least one step after the composition lock; raise steps or composition_lock_sigma")
                if release == 0:
                    raise ValueError(
                        "resolution stages need composition_lock_sigma below the first schedule sigma; "
                        "at 1.0 later stages would restart from noise and discard the enlarged image"
                    )
            logger.info(
                "Krea 2 resolution stages: planner %sx%s -> %s -> output %sx%s",
                low_w * 8, low_h * 8, " -> ".join(f"{w * 8}x{h * 8}" for h, w in stage_hw), target_w * 8, target_h * 8,
            )

        model_sampling = model.get_model_object("model_sampling")
        shift = float(getattr(model_sampling, "shift", float("nan")))
        if not math.isfinite(shift) or abs(shift - 1.15) > 1e-5:
            raise ValueError("Krea 2 Turbo mode requires the native flow sampling shift 1.15")
        if max(target_h, target_w) * 8 > 2048:
            if upscale_only or native_refine:
                logger.info(
                    "Krea 2 final output %sx%s is latent-upscaled; no target model inference "
                    "above the official 2048px per-axis range (mode=%s)",
                    target_w * 8, target_h * 8, regeneration_mode,
                )
            else:
                logger.warning(
                    "Krea 2 Turbo experimental target %sx%s exceeds the official "
                    "2048-pixel-axis inference range; visual fidelity and VRAM use are unvalidated",
                    target_w * 8, target_h * 8,
                )

        config = ScaleLockConfig(
            lock_strength=coarse_lock_strength,
            lock_strength_start=1.0,
            lock_strength_end=0.20,
            cutoff=0.33,
            mid_cutoff=0.60,
            mid_strength=mid_lock_strength,
            schedule="hold_then_drop",
            schedule_power=2.0,
            schedule_hold=0.25,
            mid_strength_start=1.0,
            mid_strength_end=0.0,
            mid_schedule="linked",
            mid_schedule_power=2.0,
            mid_schedule_hold=0.0,
            full_lock_sigma=composition_lock_sigma,
        )
        # Keep the opt-in out of ComfyUI sampler.extra_options and scope it to
        # this invocation. No numerical solver implementation is replaced.
        with sa_native_only_scope(
            sampler_name in ("sa_solver", "sa_solver_pece") and not sa_spectrum_forecasts
        ):
            result = run_scale_locked_ksampler(
                model=model,
                positive=positive,
                negative=negative,
                latent_image=prepared_latent,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=1.0,
                target_megapixels=planner_megapixels,
                nested_noise_strength=1.0,
                add_noise=True,
                pin_anchors=True,
                sampler_guard="error",
                config=config,
                noise_mode="patch_coupled",
                stage_hw=tuple(stage_hw),
                sampler_options=sampler_options,
                sa_planner_native_only=(
                    sampler_name in ("sa_solver", "sa_solver_pece")
                    and not sa_planner_spectrum_forecasts
                ),
                regeneration_mode=regeneration_mode,
                img2img_start_sigma=float(img2img_start_sigma) if img2img else None,
                handoff_diagnostics=handoff_diagnostics,
                planner_fidelity_strength=float(planner_fidelity_strength),
                planner_detail_strength=float(planner_detail_strength),
                latent_upscale_method=latent_upscale_method,
                progressive_transition_sigmas=transition_sigmas,
                spectral_calibration=spectral_calibration,
                spectral_tolerance=float(spectral_tolerance),
            )
        return (result.output, result.lowres_planner, result.denoised_output)


class Krea2SpectralCalibration:
    """Measure the clean-latent power spectrum used for spectral transitions.

    Reads completed latents only; no model evaluation. The fit is made in the
    model's internal latent space, where sampler noise has unit variance.
    """

    CATEGORY = "sampling/scale_locked"
    RETURN_TYPES = (CALIBRATION_TYPE, "STRING")
    RETURN_NAMES = ("spectral_calibration", "report")
    FUNCTION = "calibrate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The Krea 2 model whose latent format the calibration is for."}),
                "latent": ("LATENT", {"tooltip": "Clean, completed latents near the intended output size, e.g. "
                                                 "a finished Krea 2 generation (LoadLatent) or a representative "
                                                 "photograph via Load Image -> VAE Encode with the Krea 2/Qwen Image VAE. "
                                                 "Do not connect the Empty Latent or shared target canvas."}),
            },
            "optional": {
                "fit_low": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.9, "step": 0.01,
                    "tooltip": "Lowest fitted frequency as a fraction of Nyquist."}),
                "fit_high": ("FLOAT", {"default": 0.9, "min": 0.05, "max": 1.0, "step": 0.01,
                    "tooltip": "Highest fitted frequency as a fraction of Nyquist."}),
            },
        }

    def calibrate(self, model, latent, fit_low=0.05, fit_high=0.9):
        if not isinstance(latent, dict) or not isinstance(latent.get("samples"), torch.Tensor):
            raise TypeError("Krea 2 spectral calibration requires a LATENT dictionary with tensor samples")
        base = getattr(model, "model", None)
        latent_format = getattr(base, "latent_format", None)
        if latent_format is None:
            raise ValueError("Krea 2 spectral calibration requires a diffusion MODEL")
        samples = latent["samples"]
        if samples.ndim not in (4, 5) or (samples.ndim == 5 and samples.shape[2] != 1):
            raise ValueError("Krea 2 spectral calibration requires a 4-D or single-frame 5-D image latent")
        # Check the actual content first: EmptyLatentImage normally supplies a
        # four-channel zero canvas that Krea 2 expands only for generation. It
        # contains no image spectrum and must never be used for calibration.
        if not bool(torch.count_nonzero(samples)):
            raise ValueError(
                "Krea 2 spectral calibration cannot use an empty latent canvas (e.g. Empty Latent / "
                "Shared target latent). Connect a completed Krea 2 latent, or Load Image -> VAE Encode "
                "with the Krea 2/Qwen Image VAE; do not connect the generation canvas."
            )
        expected_channels = int(getattr(latent_format, "latent_channels", samples.shape[1]))
        if samples.shape[1] != expected_channels:
            raise ValueError(
                f"Krea 2 spectral calibration expected {expected_channels} latent channels, "
                f"received {samples.shape[1]}. Connect a completed Krea 2 latent, or Load Image -> "
                "VAE Encode with the Krea 2/Qwen Image VAE; the generation canvas is not calibration data."
            )
        samples = samples.detach().to(device="cpu", dtype=torch.float32)
        if samples.ndim == 4:
            # Wan 2.1-format latent statistics broadcast over a temporal axis.
            samples = samples.unsqueeze(2)
        internal = base.process_latent_in(samples)
        calibration = build_calibration(
            internal, latent_format=type(latent_format).__name__,
            fit_low=float(fit_low), fit_high=float(fit_high),
        )
        report = describe_calibration(calibration)
        logger.info("Krea 2 spectral calibration: %s", report)
        return (calibration, report)


class ScaleLockedRuntimeContextBuilder:
    CATEGORY = "sampling/scale_locked"
    RETURN_TYPES = ("SLRD_RUNTIME", "NOISE", "LATENT")
    RETURN_NAMES = ("runtime", "prepared_noise", "lowres_planner")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 16.0, "step": 0.05, "round": 0.01}),
                "nested_noise_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 4.0, "step": 0.01, "round": 0.001}),
                "pin_anchors": ("BOOLEAN", {"default": True}),
            }
        }

    def build(self, noise, guider, sampler, sigmas, latent_image, target_megapixels, nested_noise_strength, pin_anchors):
        runtime = build_runtime_context_from_advanced(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent_image,
            target_megapixels=target_megapixels,
            nested_noise_strength=nested_noise_strength,
            pin_anchors=pin_anchors,
        )
        return (runtime, runtime.prepared_noise(), clean_latent(runtime.lowres_out))


class ScaleLockedCFGGuiderNode:
    CATEGORY = "sampling/scale_locked"
    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER",),
                "runtime": ("SLRD_RUNTIME",),
                "lock_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_strength_start": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_strength_end": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_schedule": (LOCK_SCHEDULE_OPTIONS,),
                "lock_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "lock_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "coarse_cutoff": ("FLOAT", {"default": 0.33, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_cutoff": ("FLOAT", {"default": 0.60, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_schedule": (MID_SCHEDULE_OPTIONS,),
                "mid_band_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "mid_band_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "manifold_enabled": ("BOOLEAN", {"default": False}),
                "manifold_strength": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_strength_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_schedule": (LOCK_SCHEDULE_OPTIONS,),
                "manifold_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "manifold_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "manifold_cutoff": ("FLOAT", {"default": 0.18, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_radial_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "round": 0.001}),
                "manifold_anisotropy": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "manifold_translation_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "round": 0.001}),
                "manifold_anchor_mix": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_mean_anchor_mix": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_contrast_restore": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_energy_tether": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_channel_tether": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_energy_gain_cap": ("FLOAT", {"default": 1.75, "min": 1.0, "max": 4.0, "step": 0.05, "round": 0.01}),
                "manifold_max_shift_px": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 64.0, "step": 0.1, "round": 0.01}),
            },
            "optional": {
                "lock_mask": ("MASK",),
                "manifold_mask": ("MASK",),
            },
        }

    def build(
        self,
        guider,
        runtime,
        lock_strength,
        lock_strength_start,
        lock_strength_end,
        lock_schedule,
        lock_schedule_hold,
        lock_schedule_power,
        coarse_cutoff,
        mid_band_cutoff,
        mid_band_strength,
        mid_band_strength_start,
        mid_band_strength_end,
        mid_band_schedule,
        mid_band_schedule_hold,
        mid_band_schedule_power,
        manifold_enabled,
        manifold_strength,
        manifold_strength_start,
        manifold_strength_end,
        manifold_schedule,
        manifold_schedule_hold,
        manifold_schedule_power,
        manifold_cutoff,
        manifold_radial_strength,
        manifold_anisotropy,
        manifold_translation_strength,
        manifold_anchor_mix,
        manifold_mean_anchor_mix,
        manifold_contrast_restore,
        manifold_energy_tether,
        manifold_channel_tether,
        manifold_energy_gain_cap,
        manifold_max_shift_px,
        lock_mask=None,
        manifold_mask=None,
    ):
        patched_guider = clone_guider_for_scale_lock(guider)
        apply_scale_lock_to_guider(
            patched_guider,
            runtime,
            _scale_lock_config(
                lock_strength=lock_strength,
                lock_strength_start=lock_strength_start,
                lock_strength_end=lock_strength_end,
                coarse_cutoff=coarse_cutoff,
                mid_band_cutoff=mid_band_cutoff,
                mid_band_strength=mid_band_strength,
                lock_schedule=lock_schedule,
                lock_schedule_hold=lock_schedule_hold,
                lock_schedule_power=lock_schedule_power,
                mid_band_strength_start=mid_band_strength_start,
                mid_band_strength_end=mid_band_strength_end,
                mid_band_schedule=mid_band_schedule,
                mid_band_schedule_hold=mid_band_schedule_hold,
                mid_band_schedule_power=mid_band_schedule_power,
                manifold_enabled=manifold_enabled,
                manifold_strength=manifold_strength,
                manifold_strength_start=manifold_strength_start,
                manifold_strength_end=manifold_strength_end,
                manifold_schedule=manifold_schedule,
                manifold_schedule_hold=manifold_schedule_hold,
                manifold_schedule_power=manifold_schedule_power,
                manifold_cutoff=manifold_cutoff,
                manifold_radial_strength=manifold_radial_strength,
                manifold_anisotropy=manifold_anisotropy,
                manifold_translation_strength=manifold_translation_strength,
                manifold_anchor_mix=manifold_anchor_mix,
                manifold_mean_anchor_mix=manifold_mean_anchor_mix,
                manifold_contrast_restore=manifold_contrast_restore,
                manifold_energy_tether=manifold_energy_tether,
                manifold_channel_tether=manifold_channel_tether,
                manifold_energy_gain_cap=manifold_energy_gain_cap,
                manifold_max_shift_px=manifold_max_shift_px,
                lock_mask=lock_mask,
                manifold_mask=manifold_mask,
            ),
        )
        return (patched_guider,)


class ScaleLockedResidualSamplerCustomAdvanced:
    CATEGORY = "sampling/scale_locked"
    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("output", "lowres_planner", "denoised_output")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 16.0, "step": 0.05, "round": 0.01}),
                "lock_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_strength_start": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_strength_end": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_schedule": (LOCK_SCHEDULE_OPTIONS,),
                "lock_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "lock_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "coarse_cutoff": ("FLOAT", {"default": 0.33, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_cutoff": ("FLOAT", {"default": 0.60, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_schedule": (MID_SCHEDULE_OPTIONS,),
                "mid_band_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "mid_band_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "manifold_enabled": ("BOOLEAN", {"default": False}),
                "manifold_strength": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_strength_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_schedule": (LOCK_SCHEDULE_OPTIONS,),
                "manifold_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "manifold_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "manifold_cutoff": ("FLOAT", {"default": 0.18, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_radial_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "round": 0.001}),
                "manifold_anisotropy": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "manifold_translation_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "round": 0.001}),
                "manifold_anchor_mix": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_mean_anchor_mix": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_contrast_restore": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_energy_tether": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_channel_tether": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_energy_gain_cap": ("FLOAT", {"default": 1.75, "min": 1.0, "max": 4.0, "step": 0.05, "round": 0.01}),
                "manifold_max_shift_px": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 64.0, "step": 0.1, "round": 0.01}),
                "nested_noise_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 4.0, "step": 0.01, "round": 0.001}),
                "pin_anchors": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lock_mask": ("MASK",),
                "manifold_mask": ("MASK",),
            },
        }

    def sample(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        target_megapixels,
        lock_strength,
        lock_strength_start,
        lock_strength_end,
        lock_schedule,
        lock_schedule_hold,
        lock_schedule_power,
        coarse_cutoff,
        mid_band_cutoff,
        mid_band_strength,
        mid_band_strength_start,
        mid_band_strength_end,
        mid_band_schedule,
        mid_band_schedule_hold,
        mid_band_schedule_power,
        manifold_enabled,
        manifold_strength,
        manifold_strength_start,
        manifold_strength_end,
        manifold_schedule,
        manifold_schedule_hold,
        manifold_schedule_power,
        manifold_cutoff,
        manifold_radial_strength,
        manifold_anisotropy,
        manifold_translation_strength,
        manifold_anchor_mix,
        manifold_mean_anchor_mix,
        manifold_contrast_restore,
        manifold_energy_tether,
        manifold_channel_tether,
        manifold_energy_gain_cap,
        manifold_max_shift_px,
        nested_noise_strength,
        pin_anchors,
        lock_mask=None,
        manifold_mask=None,
    ):
        runtime = build_runtime_context_from_advanced(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent_image,
            target_megapixels=target_megapixels,
            nested_noise_strength=nested_noise_strength,
            pin_anchors=pin_anchors,
        )
        result = sample_with_runtime(
            guider=guider,
            sampler=sampler,
            runtime=runtime,
            config=_scale_lock_config(
                lock_strength=lock_strength,
                lock_strength_start=lock_strength_start,
                lock_strength_end=lock_strength_end,
                coarse_cutoff=coarse_cutoff,
                mid_band_cutoff=mid_band_cutoff,
                mid_band_strength=mid_band_strength,
                lock_schedule=lock_schedule,
                lock_schedule_hold=lock_schedule_hold,
                lock_schedule_power=lock_schedule_power,
                mid_band_strength_start=mid_band_strength_start,
                mid_band_strength_end=mid_band_strength_end,
                mid_band_schedule=mid_band_schedule,
                mid_band_schedule_hold=mid_band_schedule_hold,
                mid_band_schedule_power=mid_band_schedule_power,
                manifold_enabled=manifold_enabled,
                manifold_strength=manifold_strength,
                manifold_strength_start=manifold_strength_start,
                manifold_strength_end=manifold_strength_end,
                manifold_schedule=manifold_schedule,
                manifold_schedule_hold=manifold_schedule_hold,
                manifold_schedule_power=manifold_schedule_power,
                manifold_cutoff=manifold_cutoff,
                manifold_radial_strength=manifold_radial_strength,
                manifold_anisotropy=manifold_anisotropy,
                manifold_translation_strength=manifold_translation_strength,
                manifold_anchor_mix=manifold_anchor_mix,
                manifold_mean_anchor_mix=manifold_mean_anchor_mix,
                manifold_contrast_restore=manifold_contrast_restore,
                manifold_energy_tether=manifold_energy_tether,
                manifold_channel_tether=manifold_channel_tether,
                manifold_energy_gain_cap=manifold_energy_gain_cap,
                manifold_max_shift_px=manifold_max_shift_px,
                lock_mask=lock_mask,
                manifold_mask=manifold_mask,
            ),
        )
        return (result.output, result.lowres_planner, result.denoised_output)


class ScaleLockedNestedNoisePreview:
    CATEGORY = "sampling/scale_locked"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("nested_noise_latent", "lowres_reference_latent")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 16.0, "step": 0.05, "round": 0.01}),
                "nested_noise_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 4.0, "step": 0.01, "round": 0.001}),
            }
        }

    def build(self, latent_image, seed, target_megapixels, nested_noise_strength):
        high = clone_latent(latent_image)
        low = make_lowres_latent(high, target_megapixels)

        low_samples = low["samples"]
        high_samples = high["samples"]
        low_noise = prepare_noise(low_samples, int(seed), low.get("batch_index", None))
        high_noise = build_nested_noise(
            lowres_noise=low_noise,
            target_shape=tuple(high_samples.shape),
            seed=int(seed) ^ 0x9E3779B97F4A7C15,
            hf_strength=nested_noise_strength,
        )

        out = clone_latent(high)
        out["samples"] = high_noise
        return (out, low)


@dataclass
class _ImpactSampleRequest:
    model: Any
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    positive: Any
    negative: Any
    latent: dict
    denoise: float


_IMPACT_FIELD_ALIASES = {
    "model": ("model",),
    "seed": ("seed",),
    "steps": ("steps",),
    "cfg": ("cfg",),
    "sampler_name": ("sampler_name", "sampler"),
    "scheduler": ("scheduler",),
    "positive": ("positive", "positive_cond", "cond"),
    "negative": ("negative", "negative_cond", "uncond"),
    "latent": ("latent", "latent_image"),
    "denoise": ("denoise",),
}


@dataclass
class _ImpactHookSettings:
    target_megapixels: float
    nested_noise_strength: float
    add_noise: bool
    pin_anchors: bool
    sampler_guard: str
    warn_if_inactive_sampler: bool
    config: ScaleLockConfig


@dataclass
class _ScaleLockedImpactRuntimeState:
    request: _ImpactSampleRequest
    runtime: Any
    config: ScaleLockConfig


def _expand_mask_for_like(mask: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None

    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.ndim == 3:
        m = m.unsqueeze(1)
    elif m.ndim == 4:
        if m.shape[-1] == 1 and m.shape[1] != 1:
            m = m.permute(0, 3, 1, 2)
        elif m.shape[1] != 1:
            m = m.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported mask rank {m.ndim} for shape {tuple(m.shape)}")

    m = m.to(device=like.device, dtype=like.dtype, non_blocking=True)
    if tuple(m.shape[-2:]) != tuple(like.shape[-2:]):
        m = resize_4d_tensor(m, tuple(like.shape[-2:]))
    if m.shape[0] < like.shape[0]:
        repeat = math.ceil(like.shape[0] / max(1, m.shape[0]))
        m = m.repeat(repeat, 1, 1, 1)[: like.shape[0]]
    elif m.shape[0] > like.shape[0]:
        m = m[: like.shape[0]]
    if m.shape[1] == 1:
        m = m.expand(like.shape[0], like.shape[1], like.shape[-2], like.shape[-1]).contiguous()
    elif m.shape[1] != like.shape[1]:
        m = m.mean(dim=1, keepdim=True).expand(like.shape[0], like.shape[1], like.shape[-2], like.shape[-1]).contiguous()
    return m.clamp(0.0, 1.0)


def _combine_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    combined = None
    for mask in masks:
        if mask is None:
            continue
        combined = mask if combined is None else combined * mask
    return None if combined is None else combined.clamp(0.0, 1.0)


_DETAILER_SUPPORT_MASK_EPS = 1e-3


def _detailer_support_mask_for_like(mask: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor | None:
    expanded = _expand_mask_for_like(mask, like)
    if expanded is None:
        return None

    # Impact detailer masks are compositing weights. Preserving their soft latent-cell
    # variation turns them into a spatially varying SLRD strength field, which causes
    # blotchy low-frequency corrections inside smooth regions. SLRD should see a support
    # mask here: on inside the selected region, off outside it.
    return expanded.gt(_DETAILER_SUPPORT_MASK_EPS).to(dtype=like.dtype)


def _canonicalize_impact_noise_mask(mask: Any) -> torch.Tensor | None:
    if not isinstance(mask, torch.Tensor):
        return None
    if mask.ndim == 4:
        if mask.shape[1] == 0:
            return None
        return mask[:, :1, :, :].squeeze(1).contiguous()
    if mask.ndim == 3:
        return mask.contiguous()
    if mask.ndim == 2:
        return mask.unsqueeze(0).contiguous()
    logger.warning(
        "ScaleLockedDetailerHook: ignoring unsupported denoise_mask rank %s with shape %s.",
        mask.ndim,
        tuple(mask.shape),
    )
    return None


def _resolve_impact_sampling_mask(
    latent_samples: torch.Tensor,
    latent_mask: Any,
    denoise_mask: Any,
) -> torch.Tensor | None:
    target_hw = tuple(latent_samples.shape[-2:])
    batch = latent_samples.shape[0]

    def _normalize(mask: Any) -> torch.Tensor | None:
        canonical = _canonicalize_impact_noise_mask(mask)
        if canonical is None:
            return None
        resized = resize_mask(canonical, target_hw, batch, 1)
        if resized is None:
            return None
        return resized[:, :1, :, :].squeeze(1).contiguous()

    sampler_mask = _normalize(denoise_mask)
    if sampler_mask is not None:
        return sampler_mask
    return _normalize(latent_mask)


def _impact_request_tuple_from_kwargs(kwargs: dict[str, Any]) -> tuple[Any, ...]:
    if not kwargs:
        return ()

    data = {}
    for field, aliases in _IMPACT_FIELD_ALIASES.items():
        for alias in aliases:
            if alias in kwargs:
                data[field] = kwargs[alias]
                break
    if not data:
        return ()

    names = ("model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "latent", "denoise")
    return tuple(data.get(name) for name in names)


class _ScaleLockedPlannerNoise:
    def __init__(self, seed: int, disable_noise: bool):
        self.seed = int(seed)
        self.disable_noise = bool(disable_noise)

    def generate_noise(self, latent: dict) -> torch.Tensor:
        return prepare_noise(
            latent["samples"],
            seed=self.seed,
            batch_inds=latent.get("batch_index", None),
            disable_noise=self.disable_noise,
        )


def _is_impact_guider_like(obj) -> bool:
    return (
        obj is not None
        and hasattr(obj, "sample")
        and hasattr(obj, "model_patcher")
        and (hasattr(obj, "set_conds") or hasattr(obj, "inner_set_conds"))
    )


def _clear_impact_ag_guider(owner) -> None:
    if owner is None or not hasattr(owner, "_ag_detailer_guider"):
        return
    try:
        delattr(owner, "_ag_detailer_guider")
        return
    except Exception as exc:
        try:
            setattr(owner, "_ag_detailer_guider", None)
        except Exception as fallback_exc:
            logger.warning(
                "ScaleLockedDetailerHook: failed to clear _ag_detailer_guider on %s (delete=%r, fallback=%r).",
                type(owner).__name__,
                exc,
                fallback_exc,
            )
        else:
            logger.warning(
                "ScaleLockedDetailerHook: delattr(_ag_detailer_guider) failed on %s; replaced with None instead (%r).",
                type(owner).__name__,
                exc,
            )


def _resolve_impact_ag_guider_template(request: "_ImpactSampleRequest", model_wrap):
    """
    Locate an existing impact guider template from the provided request or model wrapper and return it.
    
    If a candidate guider is found on either model_wrap.model_patcher or request.model, the guider is returned and the internal `_ag_detailer_guider` attribute is cleared from both potential owners as a side effect.
    
    Parameters:
        request (_ImpactSampleRequest): Impact request that may contain a `model` owning a guider.
        model_wrap: Model wrapper that may contain a `model_patcher` owning a guider.
    
    Returns:
        The found guider instance, or `None` if no guider template is present.
    """
    owners = (
        getattr(model_wrap, "model_patcher", None),
        getattr(request, "model", None),
    )
    for owner in owners:
        if owner is None:
            continue
        candidate = getattr(owner, "_ag_detailer_guider", None)
        if candidate is None:
            continue
        for clear_owner in owners:
            _clear_impact_ag_guider(clear_owner)
        return candidate
    return None

def _normalize_sampler_extra_args(
    extra_args: dict[str, Any] | None,
    sigmas: torch.Tensor,
    target_device: torch.device,
) -> dict[str, Any]:
    """
    Ensure the provided extra_args include a `model_options.sigmas` tensor moved to the target device.
    
    If `extra_args` contains a `model_options` mapping, this function copies it, sets its `sigmas` entry to `sigmas` converted to `target_device` (non-blocking), and returns a shallow-copied dict with the updated `model_options`. If `extra_args` is None or `model_options` is not a dict, returns a shallow copy of `extra_args` (or an empty dict when None).
    
    Parameters:
        extra_args (dict[str, Any] | None): Extra sampler arguments that may include a `model_options` dict.
        sigmas (torch.Tensor): Sigmas tensor to insert into `model_options`.
        target_device (torch.device): Device to which `sigmas` will be moved.
    
    Returns:
        dict[str, Any]: A normalized copy of `extra_args` with `model_options.sigmas` set to `sigmas` on `target_device` when applicable.
    """
    normalized = {} if extra_args is None else dict(extra_args)
    model_options = normalized.get("model_options", None)
    if isinstance(model_options, dict):
        model_options = dict(model_options)
        model_options["sigmas"] = sigmas.to(device=target_device, non_blocking=True)
        normalized["model_options"] = model_options
    return normalized


class _ScaleLockedPlannerGuiderProxy:
    def __init__(self, model_wrap, extra_args: dict[str, Any] | None):
        """
        Initialize the planner guider proxy which forwards sampling calls to the underlying model wrapper.
        
        Parameters:
            model_wrap: An object that exposes a sampler-compatible interface and may provide a `model_patcher` attribute.
            extra_args (dict[str, Any] | None): Optional model/sampler options to be merged and stored for use when sampling; a shallow copy is made if provided.
        """
        self.model_patcher = getattr(model_wrap, "model_patcher", None)
        self._model_wrap = model_wrap
        self._extra_args = {} if extra_args is None else dict(extra_args)

    def sample(
        self,
        noise,
        latent_samples,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        """
        Prepare inputs on the sampler's device/dtype, normalize extra args, and forward the call to the underlying sampler.
        
        Parameters:
            denoise_mask (torch.Tensor | None): Optional mask moved to the sampler device when provided.
            callback (callable | None): Optional progress callback; if None a no-op callback is used.
            seed: Ignored by this proxy.
        
        Returns:
            The value returned by the underlying sampler.sample(...) call.
        """
        del seed
        if callback is None:
            def callback(*args, **kwargs):
                return None

        target_device = _resolve_sampler_device(self._model_wrap, latent_samples.device)
        target_dtype = latent_samples.dtype
        latent_samples = latent_samples.to(
            device=target_device,
            dtype=target_dtype,
            non_blocking=True,
        )
        noise = noise.to(device=target_device, dtype=target_dtype, non_blocking=True)
        sigmas = sigmas.to(device=target_device, non_blocking=True)
        if isinstance(denoise_mask, torch.Tensor):
            denoise_mask = denoise_mask.to(device=target_device, non_blocking=True)

        sample_extra_args = _normalize_sampler_extra_args(self._extra_args, sigmas, target_device)
        return sampler.sample(
            self._model_wrap,
            sigmas,
            sample_extra_args,
            callback,
            noise,
            latent_image=latent_samples,
            denoise_mask=denoise_mask,
            disable_pbar=disable_pbar,
        )


class _ScaleLockedGuiderProxy:
    def __init__(self, base_guider, runtime, config: ScaleLockConfig):
        """
        Wraps a base guider and initializes its scale-lock state using the provided runtime context and configuration.
        
        Parameters:
            base_guider: The underlying guider object to delegate to; must implement the guider interface used during sampling.
            runtime: Runtime context providing model, anchors_x0, and planner_sigmas required to initialize scale-lock state.
            config (ScaleLockConfig): Configuration object containing all scale-lock parameters (strengths, cutoffs, schedules, manifold controls, spatial masks, etc.) used to initialize the guider's scale-lock behavior.
        """
        self._base_guider = base_guider
        init_scale_lock_state(
            self,
            model=runtime.model,
            anchors_x0_cpu=runtime.anchors_x0,
            planner_sigmas=runtime.planner_sigmas,
            lock_strength=config.lock_strength,
            lock_strength_start=config.lock_strength_start,
            lock_strength_end=config.lock_strength_end,
            cutoff=config.cutoff,
            mid_cutoff=config.mid_cutoff,
            mid_strength=config.mid_strength,
            schedule=config.schedule,
            schedule_power=config.schedule_power,
            schedule_hold=config.schedule_hold,
            mid_strength_start=config.mid_strength_start,
            mid_strength_end=config.mid_strength_end,
            mid_schedule=config.mid_schedule,
            mid_schedule_power=config.mid_schedule_power,
            mid_schedule_hold=config.mid_schedule_hold,
            spatial_mask=config.spatial_mask,
            manifold_enabled=config.manifold_enabled,
            manifold_strength=config.manifold_strength,
            manifold_strength_start=config.manifold_strength_start,
            manifold_strength_end=config.manifold_strength_end,
            manifold_schedule=config.manifold_schedule,
            manifold_schedule_power=config.manifold_schedule_power,
            manifold_schedule_hold=config.manifold_schedule_hold,
            manifold_cutoff=config.manifold_cutoff,
            manifold_radial_strength=config.manifold_radial_strength,
            manifold_anisotropy=config.manifold_anisotropy,
            manifold_translation_strength=config.manifold_translation_strength,
            manifold_anchor_mix=config.manifold_anchor_mix,
            manifold_mean_anchor_mix=config.manifold_mean_anchor_mix,
            manifold_contrast_restore=config.manifold_contrast_restore,
            manifold_energy_tether=config.manifold_energy_tether,
            manifold_channel_tether=config.manifold_channel_tether,
            manifold_energy_gain_cap=config.manifold_energy_gain_cap,
            manifold_max_shift_px=config.manifold_max_shift_px,
            manifold_spatial_mask=config.manifold_spatial_mask,
        )

    def __getattr__(self, name: str):
        """
        Delegate attribute lookup to the wrapped base guider.
        
        Parameters:
            name (str): The attribute name to retrieve from the wrapped guider.
        
        Returns:
            Any: The value of the requested attribute from the wrapped base guider.
        """
        return getattr(self._base_guider, name)

    def __call__(self, x, timestep, model_options=None, seed=None):
        """
        Run the wrapped guider to produce a noise prediction then apply scale-lock adjustments.
        
        Parameters:
            x (torch.Tensor): Input latent tensor for guidance.
            timestep: Timestep value passed to the guider (e.g., scheduler timestep).
            model_options (dict, optional): Additional model options forwarded to the base guider.
            seed (int | None, optional): RNG seed forwarded to the base guider.
        
        Returns:
            torch.Tensor: Noise prediction after applying scale-lock modifications.
        """
        if model_options is None:
            model_options = {}
        base_noise = self._base_guider(x, timestep, model_options=model_options, seed=seed)
        return apply_scale_lock_to_noise_prediction(self, base_noise, x, timestep)


def _sync_impact_guider_conditions(guider, positive, negative) -> None:
    """
    Synchronizes positive and negative conditioning on a guider object.
    
    Attempts to apply the provided positive and negative conditioning to the guider.
    If the guider exposes set_conds, that method is called with (positive, negative).
    If it exposes inner_set_conds, that method is called with {"positive": positive, "negative": negative}.
    If neither method is present, a warning is logged. Failures during the call are caught and logged as warnings.
    
    Parameters:
        guider (object): The guider instance to update; may implement `set_conds` or `inner_set_conds`.
        positive: The positive conditioning to apply (type depends on guider implementation).
        negative: The negative conditioning to apply (type depends on guider implementation).
    """
    try:
        if hasattr(guider, "set_conds"):
            guider.set_conds(positive, negative)
        elif hasattr(guider, "inner_set_conds"):
            guider.inner_set_conds({"positive": positive, "negative": negative})
        else:
            logger.warning(
                "ScaleLockedDetailerHook: guider %s does not expose set_conds/inner_set_conds.",
                type(guider).__name__,
            )
    except Exception as exc:
        logger.warning(
            "ScaleLockedDetailerHook: failed to sync guider conditions on %s: %r",
            type(guider).__name__,
            exc,
        )


def _sync_impact_guider_cfg(guider, cfg: float) -> None:
    try:
        if hasattr(guider, "set_scales"):
            w_ag = getattr(guider, "w_ag", None)
            if w_ag is None:
                w_ag = getattr(guider, "w_autoguide", 2.0)
            guider.set_scales(cfg=float(cfg), w_ag=float(w_ag))
        elif hasattr(guider, "set_cfg"):
            guider.set_cfg(float(cfg))
        else:
            guider.cfg = float(cfg)
    except Exception as exc:
        logger.warning(
            "ScaleLockedDetailerHook: failed to sync guider cfg/scales on %s: %r",
            type(guider).__name__,
            exc,
        )


def _apply_impact_model_options(guider, extra_args: dict[str, Any] | None) -> None:
    if not isinstance(extra_args, dict):
        return
    model_options = extra_args.get("model_options", None)
    if not isinstance(model_options, dict):
        return
    try:
        guider.model_options = dict(model_options)
    except Exception as exc:
        logger.warning(
            "ScaleLockedDetailerHook: failed to copy model_options onto guider %s: %r",
            type(guider).__name__,
            exc,
        )


def _build_impact_effective_guider(
    request: "_ImpactSampleRequest",
    model_wrap,
    extra_args: dict[str, Any] | None,
    guider_template=None,
):
    if guider_template is None:
        if _is_impact_guider_like(model_wrap):
            guider_template = model_wrap
        else:
            guider_template = _resolve_impact_ag_guider_template(request, model_wrap)

    if guider_template is None:
        guider = create_cfg_guider(request.model, request.positive, request.negative, float(request.cfg))
    else:
        guider = clone_guider_for_scale_lock(guider_template)
        _sync_impact_guider_conditions(guider, request.positive, request.negative)
        _sync_impact_guider_cfg(guider, float(request.cfg))

    _apply_impact_model_options(guider, extra_args)

    return guider


class _ScaleLockedImpactSampler:
    def __init__(self, hook: "_ScaleLockedDetailerHook"):
        self._hook = hook

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        """
        Dispatches a scale-locked sampling run through the underlying sampler using an effective guider and device-aware tensors.
        
        Parameters:
            model_wrap: Model wrapper or model object used to resolve device information and guider behavior.
            sigmas (torch.Tensor): Noise schedule tensor for the sampler; will be moved to the target device.
            extra_args (dict | None): Additional sampler/model options; may be normalized to include device-sharded sigmas.
            callback: Progress/callback object passed through to the underlying sampler.
            noise (torch.Tensor): Initial noise tensor for sampling; may be replaced with prepared high-resolution noise if shapes match.
            latent_image (dict | torch.Tensor | None): Optional high-resolution latent to sample from; if a dict is provided its "samples" entry is used.
            denoise_mask (torch.Tensor | None): Optional denoising mask to pass to the sampler.
            disable_pbar (bool): If true, disables progress bar propagation to the underlying sampler.
        
        Raises:
            RuntimeError: If no pending impact sampling request was captured before this sampler was invoked.
        
        Returns:
            The value returned by the underlying sampler's sample(...) call.
        """
        request = self._hook._pending_request
        if request is None:
            raise RuntimeError("ScaleLockedDetailerHook: sampler request was not captured before the custom sampler was used.")
        try:
            self._hook._sampler_ran = True

            planner_latent = clone_latent(request.latent)
            if latent_image is not None:
                planner_latent["samples"] = latent_image

            planner_mask = _resolve_impact_sampling_mask(
                planner_latent["samples"],
                planner_latent.get("noise_mask", None),
                denoise_mask,
            )
            if planner_mask is not None:
                planner_latent["noise_mask"] = planner_mask
            else:
                planner_latent.pop("noise_mask", None)

            base_sampler = comfy.samplers.sampler_object(request.sampler_name)
            effective_guider = _build_impact_effective_guider(request, model_wrap, extra_args)
            state = self._hook._prepare_runtime_state_for_sampler(
                request=request,
                guider=effective_guider,
                sampler=base_sampler,
                sigmas=sigmas,
                live_latent=planner_latent,
                runtime_mask=planner_mask,
            )

            fallback_device = latent_image.device if latent_image is not None else noise.device
            target_device = _resolve_sampler_device(model_wrap, fallback_device)
            target_dtype = latent_image.dtype if latent_image is not None else noise.dtype
            sampling_mask = planner_mask

            prepared_noise = state.runtime.highres_noise
            use_prepared_noise = tuple(prepared_noise.shape) == tuple(noise.shape)

            if use_prepared_noise:
                sampling_noise = prepared_noise.to(
                    device=target_device,
                    dtype=target_dtype,
                    non_blocking=True,
                ).clone()
            else:
                sampling_noise = noise.to(
                    device=target_device,
                    dtype=target_dtype,
                    non_blocking=True,
                )

            sampling_latent = latent_image if latent_image is not None else state.runtime.highres_latent["samples"]
            sampling_latent = sampling_latent.to(
                device=target_device,
                dtype=target_dtype,
                non_blocking=True,
            )

            sigmas = sigmas.to(device=target_device, non_blocking=True)
            if isinstance(sampling_mask, torch.Tensor):
                sampling_mask = sampling_mask.to(device=target_device, non_blocking=True)

            apply_scale_lock_to_guider(effective_guider, state.runtime, state.config)
            try:
                return effective_guider.sample(
                    sampling_noise,
                    sampling_latent,
                    base_sampler,
                    sigmas,
                    denoise_mask=sampling_mask,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=state.runtime.noise_seed,
                )
            finally:
                restore_original_predict_noise(effective_guider)
        finally:
            self._hook._clear_sampler_state()


class _ScaleLockedDetailerHook:
    def __init__(self, settings: _ImpactHookSettings):
        self._settings = settings
        self._pending_request: _ImpactSampleRequest | None = None
        self._step_info: Any | None = None
        self._upscale_mask: torch.Tensor | None = None
        self._active_runtime: _ScaleLockedImpactRuntimeState | None = None
        self._sampler_ran = False
        self._inactive_sampler_warned = False
        self._custom_sampler = _ScaleLockedImpactSampler(self)

    def _clear_sampler_state(self):
        self._pending_request = None
        self._active_runtime = None
        self._sampler_ran = False
        self._inactive_sampler_warned = False

    def _clear_cycle_state(self):
        self._clear_sampler_state()
        self._upscale_mask = None

    def _warn_if_sampler_inactive(self, stage: str):
        if self._pending_request is None or self._sampler_ran or self._inactive_sampler_warned:
            return
        if not self._settings.warn_if_inactive_sampler:
            return
        self._inactive_sampler_warned = True
        logger.warning(
            "ScaleLockedDetailerHook: custom sampler was not selected for this cycle before %s. "
            "An earlier hook in the Impact chain provided the active sampler, so Scale-Locked sampler logic stayed inert.",
            stage,
        )

    def _effective_config_for_samples(
        self,
        samples: torch.Tensor,
        runtime_mask: torch.Tensor | None = None,
    ) -> ScaleLockConfig:
        cfg = self._settings.config

        detailer_support = _detailer_support_mask_for_like(
            runtime_mask if runtime_mask is not None else self._upscale_mask,
            samples,
        )

        # In the detailer hook path, SLRD must not extend beyond the region that the
        # detailer actually denoises/composites for this crop. Treat the live sampler mask
        # or captured upscale mask as the default support mask, then let explicit user masks
        # further restrict that support.
        explicit_lock_mask = _expand_mask_for_like(cfg.spatial_mask, samples)
        lock_mask = _combine_masks(detailer_support, explicit_lock_mask)

        explicit_manifold_source = (
            cfg.manifold_spatial_mask if cfg.manifold_spatial_mask is not None else cfg.spatial_mask
        )
        explicit_manifold_mask = _expand_mask_for_like(explicit_manifold_source, samples)
        manifold_mask = _combine_masks(detailer_support, explicit_manifold_mask)
        return replace(cfg, spatial_mask=lock_mask, manifold_spatial_mask=manifold_mask)

    def _prepare_runtime_state_for_sampler(
        self,
        request: _ImpactSampleRequest,
        guider,
        sampler,
        sigmas,
        live_latent: dict[str, Any] | None = None,
        runtime_mask: torch.Tensor | None = None,
    ) -> _ScaleLockedImpactRuntimeState:
        """
        Builds and registers a scale-locked impact runtime state for the given impact sampling request.
        
        Parameters:
            request (_ImpactSampleRequest): Canonicalized impact sampling request containing model, seed, sampler name, and latent.
            guider: Effective guider used for the planner runtime build.
            sampler: Sampler instance to be used for planning and runtime creation.
            sigmas (torch.Tensor): Noise schedule tensor to use for runtime construction.
            live_latent (dict | None): Optional override latent to use for planning instead of request.latent.
            runtime_mask (torch.Tensor | None): Optional latent-space mask actually used by the detailer sampler for this cycle.
        
        Returns:
            _ScaleLockedImpactRuntimeState: The created runtime state containing the original request, the prepared runtime context, and the effective ScaleLockConfig computed for the planner latent.
        """
        guard_sampler_alignment(request.sampler_name, self._settings.sampler_guard)
        planner_noise = _ScaleLockedPlannerNoise(request.seed, disable_noise=not self._settings.add_noise)
        planner_latent = request.latent if live_latent is None else live_latent
        runtime = build_runtime_context_from_advanced(
            noise=planner_noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=planner_latent,
            target_megapixels=self._settings.target_megapixels,
            nested_noise_strength=self._settings.nested_noise_strength,
            pin_anchors=self._settings.pin_anchors,
        )
        state = _ScaleLockedImpactRuntimeState(
            request=request,
            runtime=runtime,
            config=self._effective_config_for_samples(runtime.highres_latent["samples"], runtime_mask=runtime_mask),
        )
        self._active_runtime = state
        return state

    def get_skip_sampling(self):
        return False

    def set_steps(self, info):
        self._step_info = info

    def post_crop_region(self, w, h, item_bbox, crop_region):
        del w, h, item_bbox
        return crop_region

    def post_detection(self, segs):
        return segs

    def touch_scaled_size(self, w, h):
        return w, h

    def post_upscale(self, pixels, mask=None):
        self._clear_sampler_state()
        self._upscale_mask = None if mask is None else mask.detach().clone()
        return pixels

    def post_encode(self, samples):
        return samples

    def pre_decode(self, samples):
        self._warn_if_sampler_inactive("pre_decode")
        self._clear_sampler_state()
        return samples

    def post_decode(self, pixels):
        self._warn_if_sampler_inactive("post_decode")
        self._clear_sampler_state()
        return pixels

    def cycle_latent(self, latent):
        return latent

    def post_paste(self, image):
        self._warn_if_sampler_inactive("post_paste")
        self._clear_cycle_state()
        return image

    def get_custom_noise(self, seed, noise, is_touched):
        del seed
        return noise, is_touched

    def should_retry_patch(self, image):
        del image
        return False

    def get_custom_sampler(self, *args, **kwargs):
        del args, kwargs
        return self._custom_sampler

    def get_custom_sampler_provider(self, *args, **kwargs):
        return self.get_custom_sampler(*args, **kwargs)

    def get_custom_ksampler_provider(self, *args, **kwargs):
        return self.get_custom_sampler(*args, **kwargs)

    def pre_ksample(self, *args, **kwargs):
        self._clear_sampler_state()
        request = self._remember_request(args, kwargs, strict=False)
        if request is None:
            return _impact_request_tuple_from_kwargs(kwargs) if kwargs else args

        return _impact_request_tuple_from_kwargs(kwargs) if kwargs else args

    def post_ksample(self, *args, **kwargs):
        self._clear_cycle_state()
        if kwargs:
            return kwargs
        if len(args) == 1:
            return args[0]
        return args

    def _remember_request(self, args, kwargs, strict: bool) -> _ImpactSampleRequest | None:
        try:
            request = _coerce_impact_request(args, kwargs, fallback=self._pending_request)
        except TypeError:
            if strict:
                raise
            request = self._pending_request
        if request is not None:
            self._pending_request = request
        return request


class ScaleLockedDetailerHookProvider:
    CATEGORY = "sampling/scale_locked/impact"
    RETURN_TYPES = ("DETAILER_HOOK",)
    RETURN_NAMES = ("detailer_hook",)
    FUNCTION = "build"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        del cls, args, kwargs
        # Force a fresh hook object for each prompt execution.
        return time.monotonic_ns()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 16.0, "step": 0.05, "round": 0.01}),
                "lock_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_strength_start": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_strength_end": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_schedule": (LOCK_SCHEDULE_OPTIONS,),
                "lock_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "lock_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "coarse_cutoff": ("FLOAT", {"default": 0.33, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_cutoff": ("FLOAT", {"default": 0.60, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_schedule": (MID_SCHEDULE_OPTIONS,),
                "mid_band_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "mid_band_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "manifold_enabled": ("BOOLEAN", {"default": False}),
                "manifold_strength": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_strength_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_schedule": (LOCK_SCHEDULE_OPTIONS,),
                "manifold_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "manifold_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "manifold_cutoff": ("FLOAT", {"default": 0.18, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_radial_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "round": 0.001}),
                "manifold_anisotropy": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "manifold_translation_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "round": 0.001}),
                "manifold_anchor_mix": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_mean_anchor_mix": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_contrast_restore": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_energy_tether": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_channel_tether": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "manifold_energy_gain_cap": ("FLOAT", {"default": 1.75, "min": 1.0, "max": 4.0, "step": 0.05, "round": 0.01}),
                "manifold_max_shift_px": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 64.0, "step": 0.1, "round": 0.01}),
                "nested_noise_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 4.0, "step": 0.01, "round": 0.001}),
                "add_noise": ("BOOLEAN", {"default": True}),
                "pin_anchors": ("BOOLEAN", {"default": True}),
                "sampler_guard": (["warn", "error", "off"],),
                "warn_if_inactive_sampler": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lock_mask": ("MASK",),
                "manifold_mask": ("MASK",),
            },
        }

    def build(
        self,
        target_megapixels,
        lock_strength,
        lock_strength_start,
        lock_strength_end,
        lock_schedule,
        lock_schedule_hold,
        lock_schedule_power,
        coarse_cutoff,
        mid_band_cutoff,
        mid_band_strength,
        mid_band_strength_start,
        mid_band_strength_end,
        mid_band_schedule,
        mid_band_schedule_hold,
        mid_band_schedule_power,
        manifold_enabled,
        manifold_strength,
        manifold_strength_start,
        manifold_strength_end,
        manifold_schedule,
        manifold_schedule_hold,
        manifold_schedule_power,
        manifold_cutoff,
        manifold_radial_strength,
        manifold_anisotropy,
        manifold_translation_strength,
        manifold_anchor_mix,
        manifold_mean_anchor_mix,
        manifold_contrast_restore,
        manifold_energy_tether,
        manifold_channel_tether,
        manifold_energy_gain_cap,
        manifold_max_shift_px,
        nested_noise_strength,
        add_noise,
        pin_anchors,
        sampler_guard,
        warn_if_inactive_sampler,
        lock_mask=None,
        manifold_mask=None,
    ):
        settings = _ImpactHookSettings(
            target_megapixels=target_megapixels,
            nested_noise_strength=nested_noise_strength,
            add_noise=add_noise,
            pin_anchors=pin_anchors,
            sampler_guard=sampler_guard,
            warn_if_inactive_sampler=warn_if_inactive_sampler,
            config=_scale_lock_config(
                lock_strength=lock_strength,
                lock_strength_start=lock_strength_start,
                lock_strength_end=lock_strength_end,
                coarse_cutoff=coarse_cutoff,
                mid_band_cutoff=mid_band_cutoff,
                mid_band_strength=mid_band_strength,
                lock_schedule=lock_schedule,
                lock_schedule_hold=lock_schedule_hold,
                lock_schedule_power=lock_schedule_power,
                mid_band_strength_start=mid_band_strength_start,
                mid_band_strength_end=mid_band_strength_end,
                mid_band_schedule=mid_band_schedule,
                mid_band_schedule_hold=mid_band_schedule_hold,
                mid_band_schedule_power=mid_band_schedule_power,
                manifold_enabled=manifold_enabled,
                manifold_strength=manifold_strength,
                manifold_strength_start=manifold_strength_start,
                manifold_strength_end=manifold_strength_end,
                manifold_schedule=manifold_schedule,
                manifold_schedule_hold=manifold_schedule_hold,
                manifold_schedule_power=manifold_schedule_power,
                manifold_cutoff=manifold_cutoff,
                manifold_radial_strength=manifold_radial_strength,
                manifold_anisotropy=manifold_anisotropy,
                manifold_translation_strength=manifold_translation_strength,
                manifold_anchor_mix=manifold_anchor_mix,
                manifold_mean_anchor_mix=manifold_mean_anchor_mix,
                manifold_contrast_restore=manifold_contrast_restore,
                manifold_energy_tether=manifold_energy_tether,
                manifold_channel_tether=manifold_channel_tether,
                manifold_energy_gain_cap=manifold_energy_gain_cap,
                manifold_max_shift_px=manifold_max_shift_px,
                lock_mask=lock_mask,
                manifold_mask=manifold_mask,
            ),
        )
        return (_ScaleLockedDetailerHook(settings),)


def _scale_lock_config(
    *,
    lock_strength,
    lock_strength_start,
    lock_strength_end,
    coarse_cutoff,
    mid_band_cutoff,
    mid_band_strength,
    lock_schedule,
    lock_schedule_hold,
    lock_schedule_power,
    mid_band_strength_start,
    mid_band_strength_end,
    mid_band_schedule,
    mid_band_schedule_hold,
    mid_band_schedule_power,
    lock_mask,
    manifold_enabled=False,
    manifold_strength=0.0,
    manifold_strength_start=1.0,
    manifold_strength_end=0.0,
    manifold_schedule="ease_out",
    manifold_schedule_hold=0.0,
    manifold_schedule_power=2.0,
    manifold_cutoff=0.18,
    manifold_radial_strength=1.0,
    manifold_anisotropy=0.15,
    manifold_translation_strength=1.0,
    manifold_anchor_mix=0.18,
    manifold_mean_anchor_mix=0.12,
    manifold_contrast_restore=0.10,
    manifold_energy_tether=0.0,
    manifold_channel_tether=0.0,
    manifold_energy_gain_cap=1.75,
    manifold_max_shift_px=3.0,
    manifold_mask=None,
):
    return ScaleLockConfig(
        lock_strength=float(lock_strength),
        lock_strength_start=float(lock_strength_start),
        lock_strength_end=float(lock_strength_end),
        cutoff=float(coarse_cutoff),
        mid_cutoff=float(mid_band_cutoff),
        mid_strength=float(mid_band_strength),
        schedule=lock_schedule,
        schedule_power=float(lock_schedule_power),
        schedule_hold=float(lock_schedule_hold),
        mid_strength_start=float(mid_band_strength_start),
        mid_strength_end=float(mid_band_strength_end),
        mid_schedule=mid_band_schedule,
        mid_schedule_power=float(mid_band_schedule_power),
        mid_schedule_hold=float(mid_band_schedule_hold),
        spatial_mask=lock_mask,
        manifold_enabled=bool(manifold_enabled),
        manifold_strength=float(manifold_strength),
        manifold_strength_start=float(manifold_strength_start),
        manifold_strength_end=float(manifold_strength_end),
        manifold_schedule=manifold_schedule,
        manifold_schedule_power=float(manifold_schedule_power),
        manifold_schedule_hold=float(manifold_schedule_hold),
        manifold_cutoff=float(manifold_cutoff),
        manifold_radial_strength=float(manifold_radial_strength),
        manifold_anisotropy=float(manifold_anisotropy),
        manifold_translation_strength=float(manifold_translation_strength),
        manifold_anchor_mix=float(manifold_anchor_mix),
        manifold_mean_anchor_mix=float(manifold_mean_anchor_mix),
        manifold_contrast_restore=float(manifold_contrast_restore),
        manifold_energy_tether=float(manifold_energy_tether),
        manifold_channel_tether=float(manifold_channel_tether),
        manifold_energy_gain_cap=float(manifold_energy_gain_cap),
        manifold_max_shift_px=float(manifold_max_shift_px),
        manifold_spatial_mask=manifold_mask,
    )


def _coerce_impact_request(args, kwargs, fallback=None) -> _ImpactSampleRequest:
    data = {}
    if fallback is not None:
        data.update(fallback.__dict__)

    if len(args) >= 10:
        names = ("model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "latent", "denoise")
        for name, value in zip(names, args[:10]):
            data[name] = value

    for field, aliases in _IMPACT_FIELD_ALIASES.items():
        for alias in aliases:
            if alias in kwargs:
                data[field] = kwargs[alias]
                break

    required = ("model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "latent", "denoise")
    missing = [field for field in required if field not in data or data[field] is None]
    if missing:
        raise TypeError(f"Unable to resolve Impact detailer sample request fields: {', '.join(missing)}")

    latent = data["latent"]
    if not isinstance(latent, dict) or "samples" not in latent:
        raise TypeError("Impact detailer sample request did not provide a LATENT dict with a 'samples' entry.")

    return _ImpactSampleRequest(
        model=data["model"],
        seed=int(data["seed"]),
        steps=int(data["steps"]),
        cfg=float(data["cfg"]),
        sampler_name=str(data["sampler_name"]),
        scheduler=str(data["scheduler"]),
        positive=data["positive"],
        negative=data["negative"],
        latent=_snapshot_impact_latent(latent),
        denoise=float(data["denoise"]),
    )


def _snapshot_impact_latent(latent: dict[str, Any]) -> dict[str, Any]:
    snapshot = clone_latent(latent)
    for key, value in snapshot.items():
        if isinstance(value, torch.Tensor):
            snapshot[key] = value.detach().clone()
        elif isinstance(value, list):
            snapshot[key] = list(value)
        elif isinstance(value, dict):
            snapshot[key] = dict(value)
    return snapshot


NODE_CLASS_MAPPINGS = {
    "ScaleLockedResidualKSampler": ScaleLockedResidualKSampler,
    "Krea2TurboFlowAlignedRegenerate": Krea2TurboFlowAlignedRegenerate,
    RESOLUTION_NODE_ID: ResolutionSelectorUncapped,
    "ScaleLockedRuntimeContextBuilder": ScaleLockedRuntimeContextBuilder,
    "ScaleLockedCFGGuider": ScaleLockedCFGGuiderNode,
    "ScaleLockedResidualSamplerCustomAdvanced": ScaleLockedResidualSamplerCustomAdvanced,
    "ScaleLockedNestedNoisePreview": ScaleLockedNestedNoisePreview,
    "ScaleLockedDetailerHookProvider": ScaleLockedDetailerHookProvider,
    "SLRDLoadKrea2HandoffDiagnostics": SLRDLoadKrea2HandoffDiagnostics,
    "Krea2SpectralCalibration": Krea2SpectralCalibration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaleLockedResidualKSampler": "Scale-Locked Residual KSampler",
    "Krea2TurboFlowAlignedRegenerate": "Krea 2 Turbo Flow-Aligned Regenerate [Experimental]",
    RESOLUTION_NODE_ID: RESOLUTION_DISPLAY_NAME,
    "ScaleLockedRuntimeContextBuilder": "Scale-Locked Runtime Context",
    "ScaleLockedCFGGuider": "Scale-Locked CFG Guider",
    "ScaleLockedResidualSamplerCustomAdvanced": "Scale-Locked Residual SamplerCustomAdvanced",
    "ScaleLockedNestedNoisePreview": "Scale-Locked Nested Noise Preview",
    "ScaleLockedDetailerHookProvider": "Scale-Locked Detailer Hook Provider",
    "SLRDLoadKrea2HandoffDiagnostics": "Scale-Locked Load Krea 2 Handoff Diagnostics",
    "Krea2SpectralCalibration": "Krea 2 Spectral Calibration (Scale-Locked)",
}

