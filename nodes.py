from __future__ import annotations

import logging
import types
from typing import Optional

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

from .slrd_core import (
    ScaleLockedCFGGuider,
    TrajectoryRecorder,
    build_nested_noise,
    clone_latent,
    init_scale_lock_state,
    latent_target_hw_from_megapixels,
    resolve_scale_lock_step_index,
    resize_latent_dict,
    resize_mask,
    residual_lock_multiband,
    scale_lock_anchor_for,
    scale_lock_mask_for,
    scale_lock_strengths_for_step,
)


_LOGGER = logging.getLogger(__name__)


# Attach our mixin-style state logic to the real CFG guider base.
class _ScaleLockedCFGGuiderImpl(ScaleLockedCFGGuider, comfy.samplers.CFGGuider):
    def __init__(self, model_patcher):
        comfy.samplers.CFGGuider.__init__(self, model_patcher)

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = float(cfg)

    def set_scale_lock(
        self,
        *,
        model,
        anchors_x0_cpu,
        planner_sigmas,
        lock_strength: float,
        lock_strength_start: float,
        lock_strength_end: float,
        cutoff: float,
        mid_cutoff: float,
        mid_strength: float,
        schedule: str,
        schedule_power: float,
        schedule_hold: float,
        mid_strength_start: float,
        mid_strength_end: float,
        mid_schedule: str,
        mid_schedule_power: float,
        mid_schedule_hold: float,
        spatial_mask: Optional[torch.Tensor],
    ):
        init_scale_lock_state(
            self,
            model=model,
            anchors_x0_cpu=anchors_x0_cpu,
            planner_sigmas=planner_sigmas,
            lock_strength=lock_strength,
            lock_strength_start=lock_strength_start,
            lock_strength_end=lock_strength_end,
            cutoff=cutoff,
            mid_cutoff=mid_cutoff,
            mid_strength=mid_strength,
            schedule=schedule,
            schedule_power=schedule_power,
            schedule_hold=schedule_hold,
            mid_strength_start=mid_strength_start,
            mid_strength_end=mid_strength_end,
            mid_schedule=mid_schedule,
            mid_schedule_power=mid_schedule_power,
            mid_schedule_hold=mid_schedule_hold,
            spatial_mask=spatial_mask,
        )

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        del seed
        negative_cond = self.conds.get("negative", None)
        positive_cond = self.conds.get("positive", None)

        out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, positive_cond], x, timestep, model_options)
        base_cfg = comfy.samplers.cfg_function(
            self.inner_model,
            out[1],
            out[0],
            self.cfg,
            x,
            timestep,
            model_options=model_options,
            cond=positive_cond,
            uncond=negative_cond,
        )

        if not getattr(self, "_slrd_anchors_x0_cpu", None):
            return base_cfg

        idx = resolve_scale_lock_step_index(self, timestep)
        low_strength, mid_strength = scale_lock_strengths_for_step(self, idx)
        if low_strength <= 0.0 and mid_strength <= 0.0:
            return base_cfg

        anchor = scale_lock_anchor_for(self, idx, base_cfg)
        corrected = residual_lock_multiband(
            base_cfg,
            anchor,
            low_strength=low_strength,
            mid_strength=mid_strength,
            low_cutoff=self._slrd_cutoff,
            mid_cutoff=self._slrd_mid_cutoff,
        )

        mask = scale_lock_mask_for(self, base_cfg)
        if mask is not None:
            corrected = base_cfg + mask * (corrected - base_cfg)

        return corrected




class _NullPreviewCallback:
    def __call__(self, step, x0, x, total_steps):
        del step, x0, x, total_steps


_CONSERVATIVE_SAFE_SAMPLERS = {
    "ddim",
    "euler",
    "euler_cfg_pp",
    "heun",
    "lcm",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
}

_LOCK_SCHEDULE_OPTIONS = [
    "linear",
    "cosine",
    "flat",
    "smoothstep",
    "smootherstep",
    "ease_in",
    "ease_out",
    "ease_in_out",
    "hold_then_drop",
    "fast_drop",
]

_MID_SCHEDULE_OPTIONS = ["linked", *_LOCK_SCHEDULE_OPTIONS]


def _normalize_sampler_name(name: str) -> str:
    return str(name).strip().lower()


def _guard_sampler_alignment(sampler_name: str, mode: str) -> None:
    mode = str(mode).strip().lower()
    if mode == "off":
        return

    normalized = _normalize_sampler_name(sampler_name)
    if normalized in _CONSERVATIVE_SAFE_SAMPLERS:
        return

    msg = (
        "ScaleLockedResidualKSampler: sampler "
        f"'{sampler_name}' is outside the conservative SLRD alignment-safe allowlist. "
        "The node will still work in many cases, but planner/high-res anchor matching is less trustworthy "
        "for samplers with more complex internal evaluation patterns."
    )
    if mode == "error":
        raise ValueError(msg)
    _LOGGER.warning(msg)


def _compat_sampler_names():
    return getattr(comfy.samplers, "SAMPLER_NAMES", comfy.samplers.KSampler.SAMPLERS)


def _compat_scheduler_names():
    return getattr(comfy.samplers, "SCHEDULER_NAMES", comfy.samplers.KSampler.SCHEDULERS)


def _model_sampling_obj(model):
    if hasattr(model, "get_model_object"):
        return model.get_model_object("model_sampling")
    if hasattr(model, "model") and hasattr(model.model, "model_sampling"):
        return model.model.model_sampling
    raise AttributeError("Unable to resolve model_sampling from the ComfyUI model patcher.")


def _calculate_sigmas(model, scheduler: str, steps: int, denoise: float) -> torch.Tensor:
    total_steps = int(steps)
    if denoise < 1.0:
        if denoise <= 0.0:
            return torch.FloatTensor([])
        total_steps = int(steps / denoise)
    sigmas = comfy.samplers.calculate_sigmas(_model_sampling_obj(model), scheduler, total_steps).cpu()
    if denoise < 1.0:
        sigmas = sigmas[-(steps + 1) :]
    return sigmas


def _prepare_noise(latent_samples: torch.Tensor, seed: int, batch_inds=None, disable_noise: bool = False) -> torch.Tensor:
    if disable_noise:
        return torch.zeros(latent_samples.size(), dtype=latent_samples.dtype, layout=latent_samples.layout, device="cpu")
    return comfy.sample.prepare_noise(latent_samples, seed, batch_inds)


def _fix_latent_channels(model, latent_dict: dict) -> dict:
    out = clone_latent(latent_dict)
    ratio = out.get("downscale_ratio_spacial", None)
    out["samples"] = comfy.sample.fix_empty_latent_channels(model, out["samples"], ratio)
    return out


def _make_lowres_latent(latent: dict, target_megapixels: float) -> dict:
    low_hw = latent_target_hw_from_megapixels(latent["samples"], target_megapixels)
    return resize_latent_dict(latent, low_hw)


def _store_dtype_for(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float32, torch.float16, torch.bfloat16):
        return x.dtype
    return torch.float32


def _prepare_spatial_lock_mask(mask: Optional[torch.Tensor], latent_samples: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return resize_mask(mask, tuple(latent_samples.shape[-2:]), latent_samples.shape[0], latent_samples.shape[1])


def _make_preview_callback(model, steps: int, x0_output: dict):
    try:
        return latent_preview.prepare_callback(model, max(0, steps), x0_output)
    except Exception:
        return _NullPreviewCallback()


def _planner_sigmas_for_recorded_steps(sigmas: torch.Tensor, recorded_steps: int) -> list[float]:
    if recorded_steps <= 0:
        return []

    sigma_values = [float(v) for v in sigmas.detach().flatten().cpu().tolist()]
    if not sigma_values:
        return []

    visible_sigmas = sigma_values[:-1] if len(sigma_values) > 1 else sigma_values
    if not visible_sigmas:
        visible_sigmas = sigma_values

    if recorded_steps <= len(visible_sigmas):
        return visible_sigmas[:recorded_steps]
    return visible_sigmas + [visible_sigmas[-1]] * (recorded_steps - len(visible_sigmas))


def _noise_seed(noise) -> int:
    try:
        return int(getattr(noise, "seed", 0))
    except Exception:
        return 0


def _generate_noise_for_latent(noise, latent: dict) -> torch.Tensor:
    generated = noise.generate_noise(latent)
    if not isinstance(generated, torch.Tensor):
        raise TypeError("ScaleLockedResidualSamplerCustomAdvanced currently supports tensor noise only.")
    return generated

def _scale_lock_predict_noise(guider, base_noise: torch.Tensor, x, timestep):
    if not getattr(guider, "_slrd_anchors_x0_cpu", None):
        return base_noise

    idx = resolve_scale_lock_step_index(guider, timestep)
    low_strength, mid_strength = scale_lock_strengths_for_step(guider, idx)
    if low_strength <= 0.0 and mid_strength <= 0.0:
        return base_noise

    anchor = scale_lock_anchor_for(guider, idx, base_noise)
    corrected = residual_lock_multiband(
        base_noise,
        anchor,
        low_strength=low_strength,
        mid_strength=mid_strength,
        low_cutoff=guider._slrd_cutoff,
        mid_cutoff=guider._slrd_mid_cutoff,
    )

    mask = scale_lock_mask_for(guider, base_noise)
    if mask is not None:
        corrected = base_noise + mask * (corrected - base_noise)

    return corrected


def _patch_guider_with_scale_lock(
    guider,
    *,
    model,
    anchors_x0_cpu,
    planner_sigmas,
    lock_strength: float,
    lock_strength_start: float,
    lock_strength_end: float,
    cutoff: float,
    mid_cutoff: float,
    mid_strength: float,
    schedule: str,
    schedule_power: float,
    schedule_hold: float,
    mid_strength_start: float,
    mid_strength_end: float,
    mid_schedule: str,
    mid_schedule_power: float,
    mid_schedule_hold: float,
    spatial_mask: Optional[torch.Tensor],
):
    init_scale_lock_state(
        guider,
        model=model,
        anchors_x0_cpu=anchors_x0_cpu,
        planner_sigmas=planner_sigmas,
        lock_strength=lock_strength,
        lock_strength_start=lock_strength_start,
        lock_strength_end=lock_strength_end,
        cutoff=cutoff,
        mid_cutoff=mid_cutoff,
        mid_strength=mid_strength,
        schedule=schedule,
        schedule_power=schedule_power,
        schedule_hold=schedule_hold,
        mid_strength_start=mid_strength_start,
        mid_strength_end=mid_strength_end,
        mid_schedule=mid_schedule,
        mid_schedule_power=mid_schedule_power,
        mid_schedule_hold=mid_schedule_hold,
        spatial_mask=spatial_mask,
    )

    original_predict_noise = guider.predict_noise

    def _wrapped_predict_noise(self, x, timestep, model_options={}, seed=None):
        base_noise = original_predict_noise(x, timestep, model_options=model_options, seed=seed)
        return _scale_lock_predict_noise(self, base_noise, x, timestep)

    guider.predict_noise = types.MethodType(_wrapped_predict_noise, guider)
    return original_predict_noise



def _run_lowres_planner_advanced(
    *,
    guider,
    sampler,
    sigmas: torch.Tensor,
    lowres_latent: dict,
    noise,
    pin_anchors: bool,
) -> tuple[dict, list[torch.Tensor], list[float], torch.Tensor]:
    model = guider.model_patcher
    lowres_latent = _fix_latent_channels(model, lowres_latent)
    latent_samples = lowres_latent["samples"]
    noise_tensor = _generate_noise_for_latent(noise, lowres_latent).to(device="cpu", dtype=latent_samples.dtype)

    noise_mask = lowres_latent.get("noise_mask", None)
    recorder = TrajectoryRecorder(
        store_dtype=_store_dtype_for(latent_samples),
        capture_noisy_xt=False,
        pin_memory=pin_anchors,
    )

    samples = guider.sample(
        noise_tensor,
        latent_samples,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=recorder.callback,
        disable_pbar=True,
        seed=_noise_seed(noise),
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    out = clone_latent(lowres_latent)
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples

    planner_sigmas = _planner_sigmas_for_recorded_steps(sigmas, len(recorder.x0_steps))
    recorder.step_sigmas = planner_sigmas
    return out, recorder.x0_steps, planner_sigmas, noise_tensor


def _run_lowres_planner(
    *,
    model,
    positive,
    negative,
    cfg: float,
    sampler_name: str,
    sigmas: torch.Tensor,
    lowres_latent: dict,
    seed: int,
    disable_noise: bool,
    pin_anchors: bool,
) -> tuple[dict, list[torch.Tensor], list[float]]:
    sampler_obj = comfy.samplers.sampler_object(sampler_name)
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)

    lowres_latent = _fix_latent_channels(model, lowres_latent)
    latent_samples = lowres_latent["samples"]
    batch_inds = lowres_latent.get("batch_index", None)
    noise = _prepare_noise(latent_samples, seed=seed, batch_inds=batch_inds, disable_noise=disable_noise)

    noise_mask = lowres_latent.get("noise_mask", None)
    recorder = TrajectoryRecorder(
        store_dtype=_store_dtype_for(latent_samples),
        capture_noisy_xt=False,
        pin_memory=pin_anchors,
    )

    samples = guider.sample(
        noise,
        latent_samples,
        sampler_obj,
        sigmas,
        denoise_mask=noise_mask,
        callback=recorder.callback,
        disable_pbar=True,
        seed=seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    out = clone_latent(lowres_latent)
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples

    planner_sigmas = _planner_sigmas_for_recorded_steps(sigmas, len(recorder.x0_steps))
    recorder.step_sigmas = planner_sigmas
    return out, recorder.x0_steps, planner_sigmas


class ScaleLockedResidualKSampler:
    """
    All-in-one custom node implementing a practical MVP of Scale-Locked Residual Diffusion.

    High-level flow:
      1. Downscale the user latent to a native / low-MP planner scale.
      2. Sample that branch fully while recording the per-step denoised x0 trajectory.
      3. Construct high-res nested noise from the low-res noise field.
      4. Sample the high-res branch with a custom guider that blends low-frequency denoised structure
         toward the planner trajectory while preserving the high-frequency residual predicted by the base model.
    """

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
                "sampler_name": (_compat_sampler_names(),),
                "scheduler": (_compat_scheduler_names(),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 16.0, "step": 0.05, "round": 0.01}),
                "lock_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_strength_start": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_strength_end": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "lock_schedule": (_LOCK_SCHEDULE_OPTIONS,),
                "lock_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "lock_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "coarse_cutoff": ("FLOAT", {"default": 0.33, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_cutoff": ("FLOAT", {"default": 0.60, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_schedule": (_MID_SCHEDULE_OPTIONS,),
                "mid_band_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "mid_band_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "nested_noise_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 4.0, "step": 0.01, "round": 0.001}),
                "add_noise": ("BOOLEAN", {"default": True}),
                "pin_anchors": ("BOOLEAN", {"default": True}),
                "sampler_guard": (["warn", "error", "off"],),
            },
            "optional": {
                "lock_mask": ("MASK",),
            },
        }

    @staticmethod
    def _preview_callback(model, steps: int, x0_output: dict):
        return _make_preview_callback(model, steps, x0_output)

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
        nested_noise_strength,
        add_noise,
        pin_anchors,
        sampler_guard,
        lock_mask=None,
    ):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if denoise <= 0.0:
            highres_latent = _fix_latent_channels(model, latent_image)
            lowres_latent = _make_lowres_latent(highres_latent, target_megapixels)
            out = clone_latent(highres_latent)
            out.pop("downscale_ratio_spacial", None)
            lowres_latent.pop("downscale_ratio_spacial", None)
            return (out, lowres_latent, out)

        _guard_sampler_alignment(sampler_name, sampler_guard)
        highres_latent = _fix_latent_channels(model, latent_image)
        lowres_latent = _make_lowres_latent(highres_latent, target_megapixels)

        sigmas = _calculate_sigmas(model, scheduler=scheduler, steps=steps, denoise=denoise)
        if sigmas.numel() == 0:
            out = clone_latent(highres_latent)
            out.pop("downscale_ratio_spacial", None)
            lowres_out = clone_latent(lowres_latent)
            lowres_out.pop("downscale_ratio_spacial", None)
            return (out, lowres_out, out)

        disable_noise = not bool(add_noise)
        planner_seed = int(seed)
        detail_seed = int(seed) ^ 0x9E3779B97F4A7C15

        # 1) Low-res planner pass.
        lowres_out, anchors_x0, planner_sigmas = _run_lowres_planner(
            model=model,
            positive=positive,
            negative=negative,
            cfg=cfg,
            sampler_name=sampler_name,
            sigmas=sigmas,
            lowres_latent=lowres_latent,
            seed=planner_seed,
            disable_noise=disable_noise,
            pin_anchors=pin_anchors,
        )

        if len(anchors_x0) == 0:
            raise RuntimeError("ScaleLockedResidualKSampler: planner pass did not record any x0 anchors.")

        # 2) High-res nested noise aligned to the planner branch.
        highres_samples = highres_latent["samples"]
        lowres_samples = lowres_latent["samples"]

        lowres_noise = _prepare_noise(
            lowres_samples,
            seed=planner_seed,
            batch_inds=lowres_latent.get("batch_index", None),
            disable_noise=disable_noise,
        )
        if disable_noise:
            highres_noise = torch.zeros(
                highres_samples.size(),
                dtype=highres_samples.dtype,
                layout=highres_samples.layout,
                device="cpu",
            )
        else:
            highres_noise = build_nested_noise(
                lowres_noise=lowres_noise,
                target_shape=tuple(highres_samples.shape),
                seed=detail_seed,
                hf_strength=nested_noise_strength,
            )

        highres_noise = highres_noise.to(device="cpu", dtype=highres_samples.dtype)

        # 3) Scale-locked high-res pass.
        sampler_obj = comfy.samplers.sampler_object(sampler_name)
        guider = _ScaleLockedCFGGuiderImpl(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.set_scale_lock(
            model=model,
            anchors_x0_cpu=anchors_x0,
            planner_sigmas=planner_sigmas,
            lock_strength=lock_strength,
            lock_strength_start=lock_strength_start,
            lock_strength_end=lock_strength_end,
            cutoff=coarse_cutoff,
            mid_cutoff=mid_band_cutoff,
            mid_strength=mid_band_strength,
            schedule=lock_schedule,
            schedule_power=lock_schedule_power,
            schedule_hold=lock_schedule_hold,
            mid_strength_start=mid_band_strength_start,
            mid_strength_end=mid_band_strength_end,
            mid_schedule=mid_band_schedule,
            mid_schedule_power=mid_band_schedule_power,
            mid_schedule_hold=mid_band_schedule_hold,
            spatial_mask=_prepare_spatial_lock_mask(lock_mask, highres_samples),
        )

        x0_output = {}
        callback = self._preview_callback(model, len(sigmas) - 1, x0_output)
        noise_mask = highres_latent.get("noise_mask", None)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = guider.sample(
            highres_noise,
            highres_samples,
            sampler_obj,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=detail_seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = clone_latent(highres_latent)
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples

        if "x0" in x0_output:
            try:
                x0_out = model.model.process_latent_out(x0_output["x0"].cpu())
            except Exception:
                x0_out = x0_output["x0"].detach().cpu()
            denoised = clone_latent(highres_latent)
            denoised.pop("downscale_ratio_spacial", None)
            denoised["samples"] = x0_out
        else:
            denoised = out

        lowres_out_clean = clone_latent(lowres_out)
        lowres_out_clean.pop("downscale_ratio_spacial", None)

        return (out, lowres_out_clean, denoised)


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
                "lock_schedule": (_LOCK_SCHEDULE_OPTIONS,),
                "lock_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "lock_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "coarse_cutoff": ("FLOAT", {"default": 0.33, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_cutoff": ("FLOAT", {"default": 0.60, "min": 0.05, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_strength_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "mid_band_schedule": (_MID_SCHEDULE_OPTIONS,),
                "mid_band_schedule_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "round": 0.001}),
                "mid_band_schedule_power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "round": 0.01}),
                "nested_noise_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 4.0, "step": 0.01, "round": 0.001}),
                "pin_anchors": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lock_mask": ("MASK",),
            },
        }

    @staticmethod
    def _preview_callback(model, steps: int, x0_output: dict):
        return _make_preview_callback(model, steps, x0_output)

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
        nested_noise_strength,
        pin_anchors,
        lock_mask=None,
    ):
        model = guider.model_patcher
        highres_latent = _fix_latent_channels(model, latent_image)
        lowres_latent = _make_lowres_latent(highres_latent, target_megapixels)

        if sigmas.numel() == 0:
            out = clone_latent(highres_latent)
            out.pop("downscale_ratio_spacial", None)
            lowres_out = clone_latent(lowres_latent)
            lowres_out.pop("downscale_ratio_spacial", None)
            return (out, lowres_out, out)

        lowres_out, anchors_x0, planner_sigmas, lowres_noise = _run_lowres_planner_advanced(
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            lowres_latent=lowres_latent,
            noise=noise,
            pin_anchors=pin_anchors,
        )

        if len(anchors_x0) == 0:
            raise RuntimeError("ScaleLockedResidualSamplerCustomAdvanced: planner pass did not record any x0 anchors.")

        highres_samples = highres_latent["samples"]
        if torch.count_nonzero(lowres_noise).item() == 0:
            highres_noise = torch.zeros(
                highres_samples.size(),
                dtype=highres_samples.dtype,
                layout=highres_samples.layout,
                device="cpu",
            )
        else:
            highres_noise = build_nested_noise(
                lowres_noise=lowres_noise,
                target_shape=tuple(highres_samples.shape),
                seed=_noise_seed(noise) ^ 0x9E3779B97F4A7C15,
                hf_strength=nested_noise_strength,
            )
        highres_noise = highres_noise.to(device="cpu", dtype=highres_samples.dtype)

        original_predict_noise = _patch_guider_with_scale_lock(
            guider,
            model=model,
            anchors_x0_cpu=anchors_x0,
            planner_sigmas=planner_sigmas,
            lock_strength=lock_strength,
            lock_strength_start=lock_strength_start,
            lock_strength_end=lock_strength_end,
            cutoff=coarse_cutoff,
            mid_cutoff=mid_band_cutoff,
            mid_strength=mid_band_strength,
            schedule=lock_schedule,
            schedule_power=lock_schedule_power,
            schedule_hold=lock_schedule_hold,
            mid_strength_start=mid_band_strength_start,
            mid_strength_end=mid_band_strength_end,
            mid_schedule=mid_band_schedule,
            mid_schedule_power=mid_band_schedule_power,
            mid_schedule_hold=mid_band_schedule_hold,
            spatial_mask=_prepare_spatial_lock_mask(lock_mask, highres_samples),
        )

        x0_output = {}
        callback = self._preview_callback(model, len(sigmas) - 1, x0_output)
        noise_mask = highres_latent.get("noise_mask", None)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        try:
            samples = guider.sample(
                highres_noise,
                highres_samples,
                sampler,
                sigmas,
                denoise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=_noise_seed(noise),
            )
        finally:
            guider.predict_noise = original_predict_noise
        samples = samples.to(comfy.model_management.intermediate_device())

        out = clone_latent(highres_latent)
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples

        if "x0" in x0_output:
            try:
                x0_out = model.model.process_latent_out(x0_output["x0"].cpu())
            except Exception:
                x0_out = x0_output["x0"].detach().cpu()
            denoised = clone_latent(highres_latent)
            denoised.pop("downscale_ratio_spacial", None)
            denoised["samples"] = x0_out
        else:
            denoised = out

        lowres_out_clean = clone_latent(lowres_out)
        lowres_out_clean.pop("downscale_ratio_spacial", None)

        return (out, lowres_out_clean, denoised)


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
        low = _make_lowres_latent(high, target_megapixels)

        low_samples = low["samples"]
        high_samples = high["samples"]
        low_noise = comfy.sample.prepare_noise(low_samples, int(seed), low.get("batch_index", None))
        high_noise = build_nested_noise(
            lowres_noise=low_noise,
            target_shape=tuple(high_samples.shape),
            seed=int(seed) ^ 0x9E3779B97F4A7C15,
            hf_strength=nested_noise_strength,
        )

        out = clone_latent(high)
        out["samples"] = high_noise
        return (out, low)


NODE_CLASS_MAPPINGS = {
    "ScaleLockedResidualKSampler": ScaleLockedResidualKSampler,
    "ScaleLockedResidualSamplerCustomAdvanced": ScaleLockedResidualSamplerCustomAdvanced,
    "ScaleLockedNestedNoisePreview": ScaleLockedNestedNoisePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaleLockedResidualKSampler": "Scale-Locked Residual KSampler",
    "ScaleLockedResidualSamplerCustomAdvanced": "Scale-Locked Residual SamplerCustomAdvanced",
    "ScaleLockedNestedNoisePreview": "Scale-Locked Nested Noise Preview",
}

