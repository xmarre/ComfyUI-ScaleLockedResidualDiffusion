from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import time
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


def _resolve_impact_ag_guider_template(request: "_ImpactSampleRequest", model_wrap):
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
            if clear_owner is None:
                continue
            try:
                delattr(clear_owner, "_ag_detailer_guider")
            except Exception:
                pass
        return candidate
    return None


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

        try:
            guider.set_conds(request.positive, request.negative)
        except Exception:
            pass

        try:
            if hasattr(guider, "set_scales"):
                w_ag = getattr(guider, "w_ag", None)
                if w_ag is None:
                    w_ag = getattr(guider, "w_autoguide", 2.0)
                guider.set_scales(cfg=float(request.cfg), w_ag=float(w_ag))
            elif hasattr(guider, "set_cfg"):
                guider.set_cfg(float(request.cfg))
            else:
                guider.cfg = float(request.cfg)
        except Exception:
            pass

    if isinstance(extra_args, dict):
        model_options = extra_args.get("model_options", None)
        if isinstance(model_options, dict):
            try:
                guider.model_options = dict(model_options)
            except Exception:
                pass

    return guider


class _ScaleLockedImpactSampler:
    def __init__(self, hook: "_ScaleLockedDetailerHook"):
        self._hook = hook

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        request = self._hook._pending_request
        if request is None:
            raise RuntimeError("ScaleLockedDetailerHook: sampler request was not captured before the custom sampler was used.")
        try:
            self._hook._sampler_ran = True

            planner_latent = clone_latent(request.latent)
            if latent_image is not None:
                planner_latent["samples"] = latent_image
            if isinstance(denoise_mask, torch.Tensor):
                planner_latent["noise_mask"] = denoise_mask

            base_sampler = comfy.samplers.sampler_object(request.sampler_name)
            guider_template = (
                model_wrap
                if _is_impact_guider_like(model_wrap)
                else _resolve_impact_ag_guider_template(request, model_wrap)
            )
            planner_guider = _build_impact_effective_guider(
                request,
                model_wrap,
                extra_args,
                guider_template=guider_template,
            )
            state = self._hook._prepare_runtime_state_for_sampler(
                request=request,
                guider=planner_guider,
                sampler=base_sampler,
                sigmas=sigmas,
                live_latent=planner_latent,
            )

            sample_guider = _build_impact_effective_guider(
                request,
                model_wrap,
                extra_args,
                guider_template=guider_template,
            )
            apply_scale_lock_to_guider(sample_guider, state.runtime, state.config)

            sampling_noise = state.runtime.highres_noise
            if tuple(sampling_noise.shape) != tuple(noise.shape):
                sampling_noise = noise
            sampling_latent = state.runtime.highres_latent["samples"]
            sampling_mask = state.runtime.highres_latent.get("noise_mask", denoise_mask)

            return sample_guider.sample(
                sampling_noise,
                sampling_latent,
                base_sampler,
                state.runtime.sigmas,
                denoise_mask=sampling_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=state.runtime.noise_seed,
            )
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

    def _effective_config_for_samples(self, samples: torch.Tensor) -> ScaleLockConfig:
        cfg = self._settings.config

        face_mask = _expand_mask_for_like(self._upscale_mask, samples)
        lock_mask = _combine_masks(face_mask, _expand_mask_for_like(cfg.spatial_mask, samples))
        manifold_source = cfg.manifold_spatial_mask if cfg.manifold_spatial_mask is not None else cfg.spatial_mask
        manifold_mask = _combine_masks(face_mask, _expand_mask_for_like(manifold_source, samples))
        return replace(cfg, spatial_mask=lock_mask, manifold_spatial_mask=manifold_mask)

    def _prepare_runtime_state_for_sampler(
        self,
        request: _ImpactSampleRequest,
        guider,
        sampler,
        sigmas,
        live_latent: dict[str, Any] | None = None,
    ) -> _ScaleLockedImpactRuntimeState:
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

