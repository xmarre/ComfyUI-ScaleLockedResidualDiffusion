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


def _resolve_sampler_device(model_or_wrap, fallback: torch.device) -> torch.device:
    """
    Determine the device to use for sampling by inspecting the provided model or wrapper.
    
    Parameters:
        model_or_wrap: An object (model or wrapper) to inspect for a `load_device` attribute; common wrapper fields (`inner_model`, `model`, `model_patcher`) are checked as well.
        fallback (torch.device): Device to return if no `load_device` attribute is found on the inspected objects.
    
    Returns:
        torch.device: The first `load_device` found on `model_or_wrap` or its common attributes, otherwise `fallback`.
    """
    for obj in (
        model_or_wrap,
        getattr(model_or_wrap, "inner_model", None),
        getattr(model_or_wrap, "model", None),
        getattr(model_or_wrap, "model_patcher", None),
    ):
        if obj is None:
            continue
        device = getattr(obj, "load_device", None)
        if device is not None:
            return device
    return fallback


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
            callback = lambda *args, **kwargs: None

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
        Dispatches a scale-locked sampling run through the underlying sampler using a guider proxy and device-aware tensors.
        
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

            planner_mask = _canonicalize_impact_noise_mask(planner_latent.get("noise_mask", None))
            if planner_mask is None:
                planner_mask = _canonicalize_impact_noise_mask(denoise_mask)
            if planner_mask is not None:
                planner_latent["noise_mask"] = planner_mask

            base_sampler = comfy.samplers.sampler_object(request.sampler_name)
            state = self._hook._prepare_runtime_state_for_sampler(
                request=request,
                model_wrap=model_wrap,
                sampler=base_sampler,
                sigmas=sigmas,
                extra_args=extra_args,
                live_latent=planner_latent,
            )

            proxy = _ScaleLockedGuiderProxy(model_wrap, state.runtime, state.config)

            sampling_noise = noise
            fallback_device = latent_image.device if latent_image is not None else noise.device
            target_device = _resolve_sampler_device(model_wrap, fallback_device)
            target_dtype = latent_image.dtype if latent_image is not None else noise.dtype

            prepared_noise = state.runtime.highres_noise
            if tuple(prepared_noise.shape) == tuple(noise.shape):
                sampling_noise = prepared_noise.to(
                    device=target_device,
                    dtype=target_dtype,
                    non_blocking=True,
                ).clone()
            else:
                sampling_noise = sampling_noise.to(
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

            sampling_mask = denoise_mask
            sigmas = sigmas.to(device=target_device, non_blocking=True)
            if isinstance(sampling_mask, torch.Tensor):
                sampling_mask = sampling_mask.to(device=target_device, non_blocking=True)

            sample_extra_args = _normalize_sampler_extra_args(extra_args, sigmas, target_device)
            return base_sampler.sample(
                proxy,
                sigmas,
                sample_extra_args,
                callback,
                sampling_noise,
                latent_image=sampling_latent,
                denoise_mask=sampling_mask,
                disable_pbar=disable_pbar,
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
        model_wrap,
        sampler,
        sigmas,
        extra_args: dict[str, Any] | None,
        live_latent: dict[str, Any] | None = None,
    ) -> _ScaleLockedImpactRuntimeState:
        """
        Builds and registers a scale-locked impact runtime state for the given impact sampling request.
        
        Parameters:
            request (_ImpactSampleRequest): Canonicalized impact sampling request containing model, seed, sampler name, and latent.
            model_wrap: Model or model wrapper used to construct the planner guider proxy and to resolve runtime device/dtype.
            sampler: Sampler instance to be used for planning and runtime creation.
            sigmas (torch.Tensor): Noise schedule tensor to use for runtime construction.
            extra_args (dict | None): Optional extra model arguments forwarded into the planner guider proxy (e.g., model_options).
            live_latent (dict | None): Optional override latent to use for planning instead of request.latent.
        
        Returns:
            _ScaleLockedImpactRuntimeState: The created runtime state containing the original request, the prepared runtime context, and the effective ScaleLockConfig computed for the planner latent.
        """
        guard_sampler_alignment(request.sampler_name, self._settings.sampler_guard)
        planner_noise = _ScaleLockedPlannerNoise(request.seed, disable_noise=not self._settings.add_noise)
        planner_latent = request.latent if live_latent is None else live_latent
        planner_guider = _ScaleLockedPlannerGuiderProxy(model_wrap, extra_args)
        runtime = build_runtime_context_from_advanced(
            noise=planner_noise,
            guider=planner_guider,
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

