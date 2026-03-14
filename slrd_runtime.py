from __future__ import annotations

import copy
import logging
import types
from dataclasses import dataclass
from typing import Any, Optional

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

from .slrd_core import (
    TrajectoryRecorder,
    build_nested_noise,
    clone_latent,
    init_scale_lock_state,
    latent_manifold_compand,
    latent_target_hw_from_megapixels,
    resolve_scale_lock_step_index,
    resize_latent_dict,
    resize_mask,
    residual_lock_multiband,
    scale_lock_anchor_for,
    scale_lock_manifold_mask_for,
    scale_lock_manifold_strength_for_step,
    scale_lock_mask_for,
    scale_lock_strengths_for_step,
)


_LOGGER = logging.getLogger(__name__)



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

LOCK_SCHEDULE_OPTIONS = [
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

MID_SCHEDULE_OPTIONS = ["linked", *LOCK_SCHEDULE_OPTIONS]


@dataclass
class ScaleLockConfig:
    lock_strength: float
    lock_strength_start: float
    lock_strength_end: float
    cutoff: float
    mid_cutoff: float
    mid_strength: float
    schedule: str
    schedule_power: float
    schedule_hold: float
    mid_strength_start: float
    mid_strength_end: float
    mid_schedule: str
    mid_schedule_power: float
    mid_schedule_hold: float
    spatial_mask: Optional[torch.Tensor] = None
    manifold_enabled: bool = False
    manifold_strength: float = 0.0
    manifold_strength_start: float = 1.0
    manifold_strength_end: float = 0.0
    manifold_schedule: str = "ease_out"
    manifold_schedule_power: float = 2.0
    manifold_schedule_hold: float = 0.0
    manifold_cutoff: float = 0.18
    manifold_radial_strength: float = 1.0
    manifold_anisotropy: float = 0.15
    manifold_translation_strength: float = 1.0
    manifold_anchor_mix: float = 0.18
    manifold_mean_anchor_mix: float = 0.12
    manifold_contrast_restore: float = 0.10
    manifold_energy_tether: float = 0.0
    manifold_channel_tether: float = 0.0
    manifold_energy_gain_cap: float = 1.75
    manifold_max_shift_px: float = 3.0
    manifold_spatial_mask: Optional[torch.Tensor] = None


@dataclass
class ScaleLockedRuntimeContext:
    model: Any
    highres_latent: dict
    lowres_latent: dict
    lowres_out: dict
    sigmas: torch.Tensor
    anchors_x0: list[torch.Tensor]
    planner_sigmas: list[float]
    highres_noise: torch.Tensor
    noise_seed: int

    def prepared_noise(self) -> "ScaleLockedPreparedNoise":
        return ScaleLockedPreparedNoise(self.highres_noise, self.noise_seed)


@dataclass
class ScaleLockedSampleResult:
    output: dict
    lowres_planner: dict
    denoised_output: dict


class ScaleLockedPreparedNoise:
    def __init__(self, noise_tensor: torch.Tensor, seed: int):
        self._noise_tensor = noise_tensor.detach().to(device="cpu").contiguous().clone()
        self.seed = int(seed)

    def generate_noise(self, latent: dict) -> torch.Tensor:
        latent_samples = latent["samples"]
        if tuple(latent_samples.shape) != tuple(self._noise_tensor.shape):
            raise ValueError(
                "ScaleLockedPreparedNoise expected latent shape "
                f"{tuple(self._noise_tensor.shape)} but received {tuple(latent_samples.shape)}."
            )
        return self._noise_tensor.clone()


def _normalize_sampler_name(name: str) -> str:
    return str(name).strip().lower()


def guard_sampler_alignment(sampler_name: str, mode: str) -> None:
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


def compat_sampler_names():
    return getattr(comfy.samplers, "SAMPLER_NAMES", comfy.samplers.KSampler.SAMPLERS)


def compat_scheduler_names():
    return getattr(comfy.samplers, "SCHEDULER_NAMES", comfy.samplers.KSampler.SCHEDULERS)


def clean_latent(latent: dict) -> dict:
    out = clone_latent(latent)
    out.pop("downscale_ratio_spacial", None)
    return out


def _model_sampling_obj(model):
    if hasattr(model, "get_model_object"):
        return model.get_model_object("model_sampling")
    if hasattr(model, "model") and hasattr(model.model, "model_sampling"):
        return model.model.model_sampling
    raise AttributeError("Unable to resolve model_sampling from the ComfyUI model patcher.")


def calculate_sigmas(model, scheduler: str, steps: int, denoise: float) -> torch.Tensor:
    total_steps = int(steps)
    if denoise < 1.0:
        if denoise <= 0.0:
            return torch.FloatTensor([])
        total_steps = int(steps / denoise)
    sigmas = comfy.samplers.calculate_sigmas(_model_sampling_obj(model), scheduler, total_steps).cpu()
    if denoise < 1.0:
        sigmas = sigmas[-(steps + 1) :]
    return sigmas


def prepare_noise(latent_samples: torch.Tensor, seed: int, batch_inds=None, disable_noise: bool = False) -> torch.Tensor:
    if disable_noise:
        return torch.zeros(latent_samples.size(), dtype=latent_samples.dtype, layout=latent_samples.layout, device="cpu")
    return comfy.sample.prepare_noise(latent_samples, seed, batch_inds)


def _resolve_sampler_device(model_or_wrap, fallback: torch.device) -> torch.device:
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


def fix_latent_channels(model, latent_dict: dict) -> dict:
    out = clone_latent(latent_dict)
    ratio = out.get("downscale_ratio_spacial", None)
    out["samples"] = comfy.sample.fix_empty_latent_channels(model, out["samples"], ratio)
    return out


def make_lowres_latent(latent: dict, target_megapixels: float) -> dict:
    low_hw = latent_target_hw_from_megapixels(latent["samples"], target_megapixels)
    return resize_latent_dict(latent, low_hw)


def _store_dtype_for(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float32, torch.float16, torch.bfloat16):
        return x.dtype
    return torch.float32


def prepare_spatial_lock_mask(mask: Optional[torch.Tensor], latent_samples: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return resize_mask(mask, tuple(latent_samples.shape[-2:]), latent_samples.shape[0], latent_samples.shape[1])


def make_preview_callback(model, steps: int, x0_output: dict):
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


def noise_seed(noise) -> int:
    try:
        return int(getattr(noise, "seed", 0))
    except Exception:
        return 0


def generate_noise_for_latent(noise, latent: dict) -> torch.Tensor:
    generated = noise.generate_noise(latent)
    if not isinstance(generated, torch.Tensor):
        raise TypeError("ScaleLockedResidualSamplerCustomAdvanced currently supports tensor noise only.")
    return generated


def _apply_manifold_compand_to_noise_prediction(guider, working_noise: torch.Tensor, anchor: torch.Tensor, idx: int) -> torch.Tensor:
    manifold_strength = scale_lock_manifold_strength_for_step(guider, idx)
    if manifold_strength <= 0.0:
        return working_noise

    mask = scale_lock_manifold_mask_for(guider, working_noise)
    corrected = latent_manifold_compand(
        working_noise,
        anchor,
        mask=mask,
        strength=manifold_strength,
        cutoff=guider._slrd_manifold_cutoff,
        radial_strength=guider._slrd_manifold_radial_strength,
        anisotropy=guider._slrd_manifold_anisotropy,
        translation_strength=guider._slrd_manifold_translation_strength,
        anchor_mix=guider._slrd_manifold_anchor_mix,
        mean_anchor_mix=guider._slrd_manifold_mean_anchor_mix,
        contrast_restore=guider._slrd_manifold_contrast_restore,
        energy_tether=guider._slrd_manifold_energy_tether,
        channel_tether=guider._slrd_manifold_channel_tether,
        energy_gain_cap=guider._slrd_manifold_energy_gain_cap,
        max_shift_px=guider._slrd_manifold_max_shift_px,
    )
    if mask is not None:
        corrected = working_noise + mask * (corrected - working_noise)
    return corrected


def apply_scale_lock_to_noise_prediction(guider, base_noise: torch.Tensor, x, timestep):
    del x
    if not getattr(guider, "_slrd_anchors_x0_cpu", None):
        return base_noise

    idx = resolve_scale_lock_step_index(guider, timestep)
    anchor = scale_lock_anchor_for(guider, idx, base_noise)

    corrected = base_noise
    low_strength, mid_strength = scale_lock_strengths_for_step(guider, idx)
    if low_strength > 0.0 or mid_strength > 0.0:
        corrected = residual_lock_multiband(
            corrected,
            anchor,
            low_strength=low_strength,
            mid_strength=mid_strength,
            low_cutoff=guider._slrd_cutoff,
            mid_cutoff=guider._slrd_mid_cutoff,
        )

        mask = scale_lock_mask_for(guider, corrected)
        if mask is not None:
            corrected = base_noise + mask * (corrected - base_noise)

    corrected = _apply_manifold_compand_to_noise_prediction(guider, corrected, anchor, idx)
    return corrected


def create_cfg_guider(model, positive, negative, cfg):
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    return guider


def clone_guider_for_scale_lock(guider):
    cloned = copy.copy(guider)
    if hasattr(guider, "__dict__"):
        cloned.__dict__ = dict(guider.__dict__)

    original_predict_noise = getattr(cloned, "_slrd_original_predict_noise", None)
    if original_predict_noise is not None:
        if isinstance(original_predict_noise, types.MethodType):
            cloned.predict_noise = types.MethodType(original_predict_noise.__func__, cloned)
        else:
            cloned.predict_noise = original_predict_noise

    stale_keys = [key for key in getattr(cloned, "__dict__", {}) if key.startswith("_slrd_")]
    for key in stale_keys:
        delattr(cloned, key)

    return cloned


def apply_scale_lock_to_guider(guider, runtime: ScaleLockedRuntimeContext, config: ScaleLockConfig):
    spatial_mask = prepare_spatial_lock_mask(config.spatial_mask, runtime.highres_latent["samples"])
    manifold_spatial_mask = prepare_spatial_lock_mask(
        config.manifold_spatial_mask if config.manifold_spatial_mask is not None else config.spatial_mask,
        runtime.highres_latent["samples"],
    )
    init_scale_lock_state(
        guider,
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
        spatial_mask=spatial_mask,
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
        manifold_spatial_mask=manifold_spatial_mask,
    )

    original_predict_noise = getattr(guider, "_slrd_original_predict_noise", guider.predict_noise)
    guider._slrd_original_predict_noise = original_predict_noise

    def _wrapped_predict_noise(self, x, timestep, model_options=None, seed=None):
        if model_options is None:
            model_options = {}
        base_noise = self._slrd_original_predict_noise(x, timestep, model_options=model_options, seed=seed)
        return apply_scale_lock_to_noise_prediction(self, base_noise, x, timestep)

    guider.predict_noise = types.MethodType(_wrapped_predict_noise, guider)
    return guider


def restore_original_predict_noise(guider) -> None:
    original_predict_noise = getattr(guider, "_slrd_original_predict_noise", None)
    if original_predict_noise is not None:
        guider.predict_noise = original_predict_noise


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
    lowres_latent = fix_latent_channels(model, lowres_latent)
    latent_samples = lowres_latent["samples"]
    target_device = _resolve_sampler_device(model, latent_samples.device)
    target_dtype = latent_samples.dtype
    latent_samples = latent_samples.to(
        device=target_device,
        dtype=target_dtype,
        non_blocking=True,
    )
    lowres_latent["samples"] = latent_samples
    noise_tensor = generate_noise_for_latent(noise, lowres_latent).to(
        device=target_device,
        dtype=target_dtype,
        non_blocking=True,
    )

    noise_mask = lowres_latent.get("noise_mask", None)
    if isinstance(noise_mask, torch.Tensor):
        noise_mask = noise_mask.to(device=target_device, non_blocking=True)
    sigmas = sigmas.to(device=target_device, non_blocking=True)
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
        seed=noise_seed(noise),
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
) -> tuple[dict, list[torch.Tensor], list[float], torch.Tensor]:
    sampler_obj = comfy.samplers.sampler_object(sampler_name)
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)

    lowres_latent = fix_latent_channels(model, lowres_latent)
    latent_samples = lowres_latent["samples"]
    target_device = _resolve_sampler_device(model, latent_samples.device)
    target_dtype = latent_samples.dtype
    latent_samples = latent_samples.to(
        device=target_device,
        dtype=target_dtype,
        non_blocking=True,
    )
    lowres_latent["samples"] = latent_samples
    batch_inds = lowres_latent.get("batch_index", None)
    planner_noise = prepare_noise(
        latent_samples,
        seed=seed,
        batch_inds=batch_inds,
        disable_noise=disable_noise,
    ).to(
        device=target_device,
        dtype=target_dtype,
        non_blocking=True,
    )

    noise_mask = lowres_latent.get("noise_mask", None)
    if isinstance(noise_mask, torch.Tensor):
        noise_mask = noise_mask.to(device=target_device, non_blocking=True)
    sigmas = sigmas.to(device=target_device, non_blocking=True)
    recorder = TrajectoryRecorder(
        store_dtype=_store_dtype_for(latent_samples),
        capture_noisy_xt=False,
        pin_memory=pin_anchors,
    )

    samples = guider.sample(
        planner_noise,
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
    return out, recorder.x0_steps, planner_sigmas, planner_noise


def _build_highres_noise(highres_latent: dict, lowres_noise: torch.Tensor, seed: int, hf_strength: float):
    highres_samples = highres_latent["samples"]
    target_device = highres_samples.device
    target_dtype = highres_samples.dtype
    if torch.count_nonzero(lowres_noise).item() == 0:
        return torch.zeros(
            highres_samples.size(),
            dtype=target_dtype,
            layout=highres_samples.layout,
            device=target_device,
        )

    return build_nested_noise(
        lowres_noise=lowres_noise,
        target_shape=tuple(highres_samples.shape),
        seed=seed,
        hf_strength=hf_strength,
    ).to(device=target_device, dtype=target_dtype, non_blocking=True)


def build_runtime_context_from_advanced(
    *,
    noise,
    guider,
    sampler,
    sigmas: torch.Tensor,
    latent_image,
    target_megapixels: float,
    nested_noise_strength: float,
    pin_anchors: bool,
) -> ScaleLockedRuntimeContext:
    model = guider.model_patcher
    highres_latent = fix_latent_channels(model, latent_image)
    target_device = _resolve_sampler_device(model, highres_latent["samples"].device)
    target_dtype = highres_latent["samples"].dtype
    highres_latent["samples"] = highres_latent["samples"].to(
        device=target_device,
        dtype=target_dtype,
        non_blocking=True,
    )
    if isinstance(highres_latent.get("noise_mask"), torch.Tensor):
        highres_latent["noise_mask"] = highres_latent["noise_mask"].to(
            device=target_device,
            non_blocking=True,
        )
    lowres_latent = make_lowres_latent(highres_latent, target_megapixels)

    if sigmas.numel() == 0:
        return ScaleLockedRuntimeContext(
            model=model,
            highres_latent=highres_latent,
            lowres_latent=lowres_latent,
            lowres_out=clean_latent(lowres_latent),
            sigmas=sigmas.to(device=target_device, non_blocking=True),
            anchors_x0=[],
            planner_sigmas=[],
            highres_noise=torch.zeros_like(
                highres_latent["samples"],
                device=target_device,
            ),
            noise_seed=noise_seed(noise),
        )

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

    highres_noise = _build_highres_noise(
        highres_latent=highres_latent,
        lowres_noise=lowres_noise,
        seed=noise_seed(noise) ^ 0x9E3779B97F4A7C15,
        hf_strength=nested_noise_strength,
    )
    return ScaleLockedRuntimeContext(
        model=model,
        highres_latent=highres_latent,
        lowres_latent=lowres_latent,
        lowres_out=lowres_out,
        sigmas=sigmas.to(device=target_device, non_blocking=True),
        anchors_x0=anchors_x0,
        planner_sigmas=planner_sigmas,
        highres_noise=highres_noise,
        noise_seed=noise_seed(noise),
    )


def sample_with_runtime(
    *,
    guider,
    sampler,
    runtime: ScaleLockedRuntimeContext,
    config: ScaleLockConfig,
    restore_after: bool = True,
) -> ScaleLockedSampleResult:
    if runtime.sigmas.numel() == 0:
        out = clean_latent(runtime.highres_latent)
        return ScaleLockedSampleResult(output=out, lowres_planner=clean_latent(runtime.lowres_out), denoised_output=out)

    apply_scale_lock_to_guider(guider, runtime, config)

    x0_output = {}
    callback = make_preview_callback(runtime.model, len(runtime.sigmas) - 1, x0_output)
    noise_mask = runtime.highres_latent.get("noise_mask", None)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    try:
        samples = guider.sample(
            runtime.highres_noise,
            runtime.highres_latent["samples"],
            sampler,
            runtime.sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=runtime.noise_seed,
        )
    finally:
        if restore_after:
            restore_original_predict_noise(guider)
    samples = samples.to(comfy.model_management.intermediate_device())

    out = clean_latent(runtime.highres_latent)
    out["samples"] = samples

    if "x0" in x0_output:
        try:
            x0_out = runtime.model.model.process_latent_out(x0_output["x0"].cpu())
        except Exception:
            x0_out = x0_output["x0"].detach().cpu()
        denoised = clean_latent(runtime.highres_latent)
        denoised["samples"] = x0_out
    else:
        denoised = out

    return ScaleLockedSampleResult(
        output=out,
        lowres_planner=clean_latent(runtime.lowres_out),
        denoised_output=denoised,
    )


def run_scale_locked_ksampler(
    *,
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
    nested_noise_strength,
    add_noise,
    pin_anchors,
    sampler_guard,
    config: ScaleLockConfig,
) -> ScaleLockedSampleResult:
    if steps < 1:
        raise ValueError("steps must be >= 1")

    highres_latent = fix_latent_channels(model, latent_image)
    lowres_latent = make_lowres_latent(highres_latent, target_megapixels)
    if denoise <= 0.0:
        out = clean_latent(highres_latent)
        return ScaleLockedSampleResult(output=out, lowres_planner=clean_latent(lowres_latent), denoised_output=out)

    guard_sampler_alignment(sampler_name, sampler_guard)
    sigmas = calculate_sigmas(model, scheduler=scheduler, steps=steps, denoise=denoise)
    if sigmas.numel() == 0:
        out = clean_latent(highres_latent)
        return ScaleLockedSampleResult(output=out, lowres_planner=clean_latent(lowres_latent), denoised_output=out)

    disable_noise = not bool(add_noise)
    planner_seed = int(seed)
    detail_seed = int(seed) ^ 0x9E3779B97F4A7C15

    lowres_out, anchors_x0, planner_sigmas, lowres_noise = _run_lowres_planner(
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

    runtime = ScaleLockedRuntimeContext(
        model=model,
        highres_latent=highres_latent,
        lowres_latent=lowres_latent,
        lowres_out=lowres_out,
        sigmas=sigmas,
        anchors_x0=anchors_x0,
        planner_sigmas=planner_sigmas,
        highres_noise=_build_highres_noise(
            highres_latent=highres_latent,
            lowres_noise=lowres_noise,
            seed=detail_seed,
            hf_strength=nested_noise_strength,
        ),
        noise_seed=detail_seed,
    )
    guider = create_cfg_guider(model, positive, negative, cfg)
    sampler = comfy.samplers.sampler_object(sampler_name)
    return sample_with_runtime(guider=guider, sampler=sampler, runtime=runtime, config=config)

