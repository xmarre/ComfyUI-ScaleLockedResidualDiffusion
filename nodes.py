from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import math
import torch
import comfy.samplers

from .slrd_core import build_nested_noise, clone_latent, init_scale_lock_state, resize_4d_tensor
from .slrd_runtime import (
    LOCK_SCHEDULE_OPTIONS,
    MID_SCHEDULE_OPTIONS,
    ScaleLockConfig,
    apply_scale_lock_to_noise_prediction,
    build_runtime_context_from_advanced,
    calculate_sigmas,
    clean_latent,
    compat_sampler_names,
    compat_scheduler_names,
    create_cfg_guider,
    fix_latent_channels,
    guard_sampler_alignment,
    make_lowres_latent,
    prepare_noise,
    run_scale_locked_ksampler,
    sample_with_runtime,
    apply_scale_lock_to_guider,
    clone_guider_for_scale_lock,
)


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


class _ScaleLockedGuiderProxy:
    def __init__(self, base_guider, runtime, config: ScaleLockConfig):
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
        return getattr(self._base_guider, name)

    def __call__(self, x, timestep, model_options=None, seed=None):
        if model_options is None:
            model_options = {}
        base_noise = self._base_guider(x, timestep, model_options=model_options, seed=seed)
        return apply_scale_lock_to_noise_prediction(self, base_noise, x, timestep)


class _ScaleLockedImpactSampler:
    def __init__(self, hook: "_ScaleLockedDetailerHook"):
        self._hook = hook

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        state = self._hook._active_runtime
        if state is None:
            raise RuntimeError("ScaleLockedDetailerHook: runtime state was not prepared before the custom sampler was used.")

        base_sampler = comfy.samplers.sampler_object(state.request.sampler_name)
        proxy = _ScaleLockedGuiderProxy(model_wrap, state.runtime, state.config)
        return base_sampler.sample(
            proxy,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            disable_pbar=disable_pbar,
        )


class _ScaleLockedDetailerHook:
    def __init__(self, settings: _ImpactHookSettings):
        self._settings = settings
        self._pending_request: _ImpactSampleRequest | None = None
        self._step_info: Any | None = None
        self._upscale_mask: torch.Tensor | None = None
        self._active_runtime: _ScaleLockedImpactRuntimeState | None = None
        self._custom_sampler = _ScaleLockedImpactSampler(self)

    def _clear_cycle_state(self):
        self._pending_request = None
        self._active_runtime = None

    def _effective_config_for_samples(self, samples: torch.Tensor) -> ScaleLockConfig:
        cfg = self._settings.config

        face_mask = _expand_mask_for_like(self._upscale_mask, samples)
        lock_mask = _combine_masks(face_mask, _expand_mask_for_like(cfg.spatial_mask, samples))
        manifold_source = cfg.manifold_spatial_mask if cfg.manifold_spatial_mask is not None else cfg.spatial_mask
        manifold_mask = _combine_masks(face_mask, _expand_mask_for_like(manifold_source, samples))
        return replace(cfg, spatial_mask=lock_mask, manifold_spatial_mask=manifold_mask)

    def _prepare_runtime_state(self, request: _ImpactSampleRequest) -> _ScaleLockedImpactRuntimeState:
        guard_sampler_alignment(request.sampler_name, self._settings.sampler_guard)
        planner_noise = _ScaleLockedPlannerNoise(request.seed, disable_noise=not self._settings.add_noise)
        sigmas = calculate_sigmas(request.model, scheduler=request.scheduler, steps=request.steps, denoise=request.denoise)
        guider = create_cfg_guider(request.model, request.positive, request.negative, request.cfg)
        sampler = comfy.samplers.sampler_object(request.sampler_name)
        runtime = build_runtime_context_from_advanced(
            noise=planner_noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=request.latent,
            target_megapixels=self._settings.target_megapixels,
            nested_noise_strength=self._settings.nested_noise_strength,
            pin_anchors=self._settings.pin_anchors,
        )
        state = _ScaleLockedImpactRuntimeState(
            request=request,
            runtime=runtime,
            config=self._effective_config_for_samples(runtime.highres_latent["samples"]),
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
        self._clear_cycle_state()
        self._upscale_mask = None if mask is None else mask.detach().clone()
        return pixels

    def post_encode(self, samples):
        return samples

    def pre_decode(self, samples):
        return samples

    def post_decode(self, pixels):
        return pixels

    def cycle_latent(self, latent):
        return latent

    def post_paste(self, image):
        return image

    def get_custom_noise(self, seed, noise, is_touched):
        del seed
        state = self._active_runtime
        if state is None:
            return noise, is_touched

        prepared = state.runtime.highres_noise
        if tuple(prepared.shape) != tuple(noise.shape):
            return noise, is_touched
        return prepared.to(device=noise.device, dtype=noise.dtype).clone(), True

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
        request = self._remember_request(args, kwargs, strict=False)
        if request is None:
            return _impact_request_tuple_from_kwargs(kwargs) if kwargs else args

        state = self._prepare_runtime_state(request)
        return (
            request.model,
            request.seed,
            request.steps,
            request.cfg,
            request.sampler_name,
            request.scheduler,
            request.positive,
            request.negative,
            state.runtime.highres_latent,
            request.denoise,
        )

    def post_ksample(self, *args, **kwargs):
        self._clear_cycle_state()
        self._upscale_mask = None
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
        lock_mask=None,
        manifold_mask=None,
    ):
        settings = _ImpactHookSettings(
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
        latent=data["latent"],
        denoise=float(data["denoise"]),
    )


NODE_CLASS_MAPPINGS = {
    "ScaleLockedResidualKSampler": ScaleLockedResidualKSampler,
    "ScaleLockedRuntimeContextBuilder": ScaleLockedRuntimeContextBuilder,
    "ScaleLockedCFGGuider": ScaleLockedCFGGuiderNode,
    "ScaleLockedResidualSamplerCustomAdvanced": ScaleLockedResidualSamplerCustomAdvanced,
    "ScaleLockedNestedNoisePreview": ScaleLockedNestedNoisePreview,
    "ScaleLockedDetailerHookProvider": ScaleLockedDetailerHookProvider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaleLockedResidualKSampler": "Scale-Locked Residual KSampler",
    "ScaleLockedRuntimeContextBuilder": "Scale-Locked Runtime Context",
    "ScaleLockedCFGGuider": "Scale-Locked CFG Guider",
    "ScaleLockedResidualSamplerCustomAdvanced": "Scale-Locked Residual SamplerCustomAdvanced",
    "ScaleLockedNestedNoisePreview": "Scale-Locked Nested Noise Preview",
    "ScaleLockedDetailerHookProvider": "Scale-Locked Detailer Hook Provider",
}

