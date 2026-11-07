from __future__ import annotations

import contextvars
import copy
from contextlib import contextmanager
import logging
import math
import sys
import types
from dataclasses import dataclass
from typing import Any, Optional

import torch

import comfy.model_management
import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

from .krea2_flow import build_patch_coupled_noise

from .slrd_core import (
    TrajectoryRecorder,
    build_nested_noise,
    clone_latent,
    init_scale_lock_state,
    latent_manifold_compand,
    latent_target_hw_from_megapixels,
    resolve_scale_lock_step_index,
    resize_4d_tensor,
    resize_latent_dict,
    resize_mask,
    residual_lock_multiband,
    scale_lock_anchor_for,
    scale_lock_full_lock_active,
    full_lock_prefix_length,
    flow_euler_locked_clean_state,
    scale_lock_manifold_mask_for,
    scale_lock_manifold_strength_for_step,
    scale_lock_mask_for,
    scale_lock_strengths_for_step,
)


_LOGGER = logging.getLogger(__name__)



_SPECTRUM_RUNTIME = "comfyui_spectrum_krea2.runtime"


def _spectrum_force_native_vars() -> list:
    """FORCE_NATIVE of every loaded Spectrum Krea 2 runtime module.

    ComfyUI loads a custom-node directory under a path-derived module name,
    so the active runtime is ``<custom node path>.comfyui_spectrum_krea2.runtime``
    rather than the bare package name. A bare import would miss it, or load a
    second copy whose context variable the active sampler hook never reads.
    """
    found = []
    for name, module in list(sys.modules.items()):
        if module is None or not (name == _SPECTRUM_RUNTIME or name.endswith("." + _SPECTRUM_RUNTIME)):
            continue
        var = getattr(module, "FORCE_NATIVE", None)
        if isinstance(var, contextvars.ContextVar) and all(var is not v for v in found):
            found.append(var)
    return found


@contextmanager
def sa_native_only_scope(enabled: bool):
    """Suppress only Spectrum forecasts for the scoped SA execution.

    No mutation of the patcher, sampler, or process-global wrappers. The
    contextvar is reset on cancellation, errors, and nested sampling.
    Standalone Scale-Locked works when Spectrum is not installed.
    """
    if not enabled:
        yield
        return
    tokens = [(var, var.set(True)) for var in _spectrum_force_native_vars()]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


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
    # Deterministic, one denoiser evaluation per step, history built only
    # from (x, denoised) pairs: planner anchors map one-to-one onto steps.
    "deis",
    "gradient_estimation",
    "ipndm",
    "ipndm_v",
    "res_multistep",
}

# Native ComfyUI samplers offered by the Krea 2 node. All make exactly one
# denoiser evaluation per outer step, so native solver state survives the
# planner-prediction substitution on fully locked steps. ER-SDE is stochastic:
# its per-step noise is patch-coupled across planner and target canvases.
# CFG++ variants require side-channel unconditional predictions and are excluded.
# Euler alone permits the closed-form flow handoff.
KREA2_SAMPLERS = (
    "euler",
    "dpmpp_2m",
    "res_multistep",
    "deis",
    "ipndm",
    "ipndm_v",
    "gradient_estimation",
    "er_sde",
    "sa_solver",
    "sa_solver_pece",
)


def krea2_sampler_names() -> list[str]:
    """Krea 2 node solvers that the installed ComfyUI provides, Euler first."""
    available = set(compat_sampler_names())
    return [name for name in KREA2_SAMPLERS if name in available]


_ER_SDE_TYPES = ("ER-SDE", "Reverse-time SDE", "ODE")
_SEED_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class Krea2ERSDENoiseScaler:
    """Same three scaler families as ComfyUI's SamplerER_SDE node."""

    solver_type: str
    eta: float

    def __call__(self, x):
        if self.solver_type == "ODE" or self.eta == 0.0:
            return x
        if self.solver_type == "Reverse-time SDE":
            return x ** (self.eta + 1.0)
        return x * ((x ** 0.3).exp() + 10.0) ** self.eta


def krea2_er_sde_options(solver_type="ER-SDE", max_stage=3, eta=1.0, s_noise=1.0) -> dict:
    """Validate parameters before sampling; ODE/eta=0 disables stochasticity."""
    if solver_type not in _ER_SDE_TYPES:
        raise ValueError("ER-SDE solver type must be ER-SDE, Reverse-time SDE or ODE")
    if isinstance(max_stage, bool) or not isinstance(max_stage, int) or not 1 <= max_stage <= 3:
        raise ValueError("ER-SDE max_stage must be an integer in [1, 3]")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) for v in (eta, s_noise)):
        raise ValueError("ER-SDE eta and s_noise must be finite numbers")
    if not (0.0 <= eta <= 10.0 and 0.0 <= s_noise <= 100.0):
        raise ValueError("ER-SDE eta or s_noise is outside its ComfyUI input range")
    if solver_type == "ODE" or eta == 0.0:
        solver_type, s_noise = "ODE", 0.0
    return {
        "max_stage": max_stage,
        "s_noise": float(s_noise),
        "noise_scaler": Krea2ERSDENoiseScaler(solver_type, float(eta)),
    }


def effective_er_sde_sigmas(sigmas: torch.Tensor, model) -> torch.Tensor:
    """Native SNR solver first-sigma correction shared by ER-SDE and SA-Solver."""
    from comfy.k_diffusion.sampling import offset_first_sigma_for_snr
    return offset_first_sigma_for_snr(sigmas, _model_sampling_obj(model))


SA_SOLVER_NAMES = frozenset(("sa_solver", "sa_solver_pece"))
SNR_SOLVER_NAMES = SA_SOLVER_NAMES | {"er_sde"}


def krea2_sa_solver_options(
    model, *, eta=1.0, sde_start_percent=0.2, sde_end_percent=0.8,
    s_noise=1.0, predictor_order=3, corrector_order=4, simple_order_2=False,
) -> dict:
    """ComfyUI SamplerSASolver options, using native model percent->sigma."""
    values = (eta, sde_start_percent, sde_end_percent, s_noise)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("SA-Solver eta, SDE percentages and s_noise must be finite")
    if not (0 <= eta <= 10 and 0 <= s_noise <= 100 and
            0 <= sde_start_percent <= sde_end_percent <= 1):
        raise ValueError("SA-Solver SDE interval and noise values are invalid")
    if (type(predictor_order) is not int or not 1 <= predictor_order <= 6 or
            type(corrector_order) is not int or not 0 <= corrector_order <= 6 or
            type(simple_order_2) is not bool):
        raise ValueError("SA-Solver predictor/corrector order or simple_order_2 is invalid")
    from comfy.k_diffusion.sa_solver import get_tau_interval_func
    model_sampling = _model_sampling_obj(model)
    start_sigma = model_sampling.percent_to_sigma(sde_start_percent)
    end_sigma = model_sampling.percent_to_sigma(sde_end_percent)
    return {
        "tau_func": get_tau_interval_func(start_sigma, end_sigma, eta=float(eta)),
        "s_noise": float(s_noise),
        "predictor_order": predictor_order,
        "corrector_order": corrector_order,
        "simple_order_2": simple_order_2,
    }


def _er_sde_seed(seed: int, step: int, level: int) -> int:
    """Independent, reproducible Gaussian fields across steps and patch levels."""
    x = (int(seed) ^ 0xD1B54A32D192ED03 ^
         ((int(step) + 1) * 0x9E3779B97F4A7C15) ^
         ((int(level) + 2) * 0x94D049BB133111EB)) & _SEED_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _SEED_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _SEED_MASK
    return (x ^ (x >> 31)) & _SEED_MASK


class Krea2PatchCoupledERSDENoise:
    """One iid Gaussian increment per native ER-SDE noise step.

    Every pass regenerates the same coarse increment from the absolute outer
    step index. Target increments are lifted through the same patch-phase
    hierarchy as the resolution stages, preserving a Gaussian marginal at
    every scale and coupling their coarse stochastic layout. No state escapes
    this sampler invocation.
    """

    def __init__(self, low_shape, stage_shapes, stage_index, seed, start_step, sigmas, device, dtype):
        self.low_shape = tuple(low_shape)
        self.stage_shapes = tuple(tuple(shape) for shape in stage_shapes)
        self.stage_index = int(stage_index)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.sigmas = tuple(float(s) for s in sigmas.detach().cpu().tolist())
        self.device = torch.device(device)
        self.dtype = dtype
        self.cursor = 0
        if not (-1 <= self.stage_index < len(self.stage_shapes)):
            raise ValueError("ER-SDE noise stage index is invalid")

    def __call__(self, sigma, sigma_next):
        i = self.cursor
        if i + 1 >= len(self.sigmas):
            raise RuntimeError("ER-SDE requested extra stochastic noise outside its sigma schedule")
        # The native solver offsets the first flow sigma to avoid infinite
        # log-SNR. The expected schedule was corrected before constructing us.
        if (not math.isclose(float(sigma), self.sigmas[i], rel_tol=1e-5, abs_tol=1e-6)
                or not math.isclose(float(sigma_next), self.sigmas[i + 1],
                                    rel_tol=1e-5, abs_tol=1e-6)):
            raise RuntimeError("ER-SDE noise request is out of native sigma order")
        self.cursor += 1
        step = self.start_step + i
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_er_sde_seed(self.seed, step, -1))
        field = torch.randn(self.low_shape, generator=generator,
                            device="cpu", dtype=torch.float32).to(self.device)
        for level in range(self.stage_index + 1):
            field = build_patch_coupled_noise(
                field, self.stage_shapes[level], _er_sde_seed(self.seed, step, level)
            )
        return field.to(device=self.device, dtype=self.dtype)



class Krea2PatchCoupledSANoise(Krea2PatchCoupledERSDENoise):
    """Native SA-Solver may request stochastic increments only in its SDE window.

    Locate the actual sigma pair instead of advancing a dense step cursor;
    this preserves common coarse noise even when stochastic intervals skip
    steps, restart at release, or disable noise entirely.
    """

    def __call__(self, sigma, sigma_next):
        for i in range(self.cursor, len(self.sigmas) - 1):
            if (math.isclose(float(sigma), self.sigmas[i], rel_tol=1e-5, abs_tol=1e-6)
                    and math.isclose(float(sigma_next), self.sigmas[i + 1],
                                     rel_tol=1e-5, abs_tol=1e-6)):
                self.cursor = i
                return super().__call__(sigma, sigma_next)
        raise RuntimeError("SA-Solver stochastic increment was requested out of sigma order")


def _krea2_sampler_object(
    sampler_name: str, sampler_options: Optional[dict], *,
    model, low_shape: tuple, stage_shapes: tuple, stage_index: int,
    seed: int, start_step: int, sigmas: torch.Tensor, noise_like: torch.Tensor,
):
    # The generic SLRD node preserves its caller-selected native sampler.
    # Only the Krea 2 node passes validated options and enables patch coupling.
    if sampler_name not in SNR_SOLVER_NAMES or sampler_options is None:
        return comfy.samplers.sampler_object(sampler_name)
    options = dict(sampler_options)
    if sampler_name in SA_SOLVER_NAMES and sampler_name == "sa_solver":
        options["use_pece"] = False
    if options.get("s_noise", 1.0) > 0.0:
        noise_cls = (Krea2PatchCoupledSANoise if sampler_name in SA_SOLVER_NAMES
                     else Krea2PatchCoupledERSDENoise)
        options["noise_sampler"] = noise_cls(
            low_shape=low_shape, stage_shapes=stage_shapes, stage_index=stage_index,
            seed=seed, start_step=start_step,
            sigmas=effective_er_sde_sigmas(sigmas, model),
            device=_resolve_sampler_device(model, noise_like.device),
            dtype=noise_like.dtype,
        )
    # Keep ComfyUI's native stochastic Adams/ER-SDE integration and history.
    return comfy.samplers.ksampler(sampler_name, options)


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
    # Target steps whose input sigma exceeds this value use the planner
    # prediction in every frequency band. None disables the full lock.
    full_lock_sigma: Optional[float] = None


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
    pece_corrected_anchors: Optional[list[Optional[torch.Tensor]]] = None

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

    if scale_lock_full_lock_active(guider, timestep):
        mask = scale_lock_mask_for(guider, base_noise)
        if mask is None:
            # The anchor may alias stored planner state; never hand it out.
            return anchor.clone()
        return base_noise + mask * (anchor - base_noise)

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


def locked_planner_prediction(guider, x: torch.Tensor, timestep) -> Optional[torch.Tensor]:
    """Planner prediction for a fully locked step, or None if the model must run.

    ``apply_scale_lock_to_noise_prediction`` discards the target prediction
    entirely during an unmasked full lock, so returning this directly yields
    the same denoised value without the target model evaluation. Masked locks
    blend the target prediction and return None.
    """
    if not getattr(guider, "_slrd_anchors_x0_cpu", None):
        return None
    if not scale_lock_full_lock_active(guider, timestep):
        return None
    if scale_lock_mask_for(guider, x) is not None:
        return None
    idx = resolve_scale_lock_step_index(guider, timestep)
    # The anchor may alias stored planner state; never hand it out.
    return scale_lock_anchor_for(guider, idx, x).clone()


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


def apply_scale_lock_to_guider(
    guider,
    runtime: ScaleLockedRuntimeContext,
    config: ScaleLockConfig,
    skip_locked_model_calls: bool = False,
):
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
        full_lock_sigma=config.full_lock_sigma,
    )

    original_predict_noise = getattr(guider, "_slrd_original_predict_noise", guider.predict_noise)
    guider._slrd_original_predict_noise = original_predict_noise
    corrected = runtime.pece_corrected_anchors
    guider._slrd_pece_corrected_anchors = corrected
    guider._slrd_pece_counts = {} if corrected is not None else None
    # A sampler-side latent noise mask is distinct from the scale-lock's
    # spatial mask. Preserve the evaluated path for masked sampling even
    # when the full lock would otherwise discard the target prediction.
    guider._slrd_skip_locked_model_calls = (
        bool(skip_locked_model_calls)
        and not isinstance(runtime.highres_latent.get("noise_mask"), torch.Tensor)
    )

    def _wrapped_predict_noise(self, x, timestep, model_options=None, seed=None):
        if model_options is None:
            model_options = {}
        original_anchors = self._slrd_anchors_x0_cpu
        corrected = getattr(self, "_slrd_pece_corrected_anchors", None)
        if corrected is not None:
            index = resolve_scale_lock_step_index(self, timestep)
            counts = self._slrd_pece_counts
            calls = counts.get(index, 0) + 1
            if calls > 2:
                raise RuntimeError("SA-PECE target made more than two predictions at one sigma")
            counts[index] = calls
            if calls == 2:
                correction = corrected[index]
                if correction is None:
                    raise RuntimeError("SA-PECE corrected planner anchor missing for target")
                # This invocation sees the corrected planner prediction only
                # at the matching sigma. Restore the shared predictor anchors
                # even if CFG/transformer evaluation raises.
                revised = list(original_anchors)
                revised[index] = correction
                self._slrd_anchors_x0_cpu = revised
        try:
            if getattr(self, "_slrd_skip_locked_model_calls", False):
                locked = locked_planner_prediction(self, x, timestep)
                if locked is not None:
                    return locked
            base_noise = self._slrd_original_predict_noise(x, timestep, model_options=model_options, seed=seed)
            return apply_scale_lock_to_noise_prediction(self, base_noise, x, timestep)
        finally:
            self._slrd_anchors_x0_cpu = original_anchors

    guider.predict_noise = types.MethodType(_wrapped_predict_noise, guider)
    return guider


def restore_original_predict_noise(guider) -> None:
    original_predict_noise = getattr(guider, "_slrd_original_predict_noise", None)
    if original_predict_noise is not None:
        guider.predict_noise = original_predict_noise


def _planner_callback(model, recorder: TrajectoryRecorder, sigmas: torch.Tensor):
    """Record planner anchors and report progress like the other passes."""
    preview = make_preview_callback(model, max(0, len(sigmas) - 1), {})

    def callback(step, x0, x, total_steps):
        recorder.callback(step, x0, x, total_steps)
        preview(step, x0, x, total_steps)

    return callback


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
        callback=_planner_callback(model, recorder, sigmas),
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
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
    sampler_options: Optional[dict] = None,
) -> tuple[dict, list[torch.Tensor], list[float], torch.Tensor,
           Optional[list[Optional[torch.Tensor]]]]:
    sampler_obj = None
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
    sampler_obj = _krea2_sampler_object(
        sampler_name, sampler_options, model=model, low_shape=tuple(planner_noise.shape),
        stage_shapes=(), stage_index=-1, seed=seed, start_step=0,
        sigmas=sigmas, noise_like=planner_noise,
    )
    recorder = TrajectoryRecorder(
        store_dtype=_store_dtype_for(latent_samples),
        capture_noisy_xt=False,
        pin_memory=pin_anchors,
    )


    pece_corrections = None
    if sampler_name == "sa_solver_pece":
        # Native SA-Solver emits its callback before the corrected-state
        # denoiser evaluation. Capture that separate prediction here; the
        # callback-only trajectory recorder intentionally stores predictors.
        pece_corrections = [None] * (len(sigmas) - 1)
        calls_per_step = [0] * len(pece_corrections)
        effective = effective_er_sde_sigmas(sigmas, model).detach().cpu().tolist()
        predict_native = guider.predict_noise

        def _capture_pece_predictor_and_corrector(self, x, timestep, model_options=None, seed=None):
            sigma = float(timestep.flatten()[0]) if torch.is_tensor(timestep) else float(timestep)
            matches = [i for i, expected in enumerate(effective[:-1])
                       if math.isclose(sigma, float(expected), rel_tol=1e-5, abs_tol=1e-6)]
            if len(matches) != 1:
                raise RuntimeError("SA-PECE planner evaluation did not match its sigma schedule")
            index = matches[0]
            calls_per_step[index] += 1
            if calls_per_step[index] > 2:
                raise RuntimeError("SA-PECE planner unexpectedly evaluated a step more than twice")
            result = predict_native(x, timestep, model_options=model_options, seed=seed)
            if calls_per_step[index] == 2:
                pece_corrections[index] = result.detach().to(
                    device="cpu", dtype=_store_dtype_for(latent_samples)
                ).contiguous().clone()
            return result

        guider.predict_noise = types.MethodType(_capture_pece_predictor_and_corrector, guider)

    samples = guider.sample(
        planner_noise,
        latent_samples,
        sampler_obj,
        sigmas,
        denoise_mask=noise_mask,
        callback=_planner_callback(model, recorder, sigmas),
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    out = clone_latent(lowres_latent)
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples

    recorded_sigmas = effective_er_sde_sigmas(sigmas, model) if sampler_name in SNR_SOLVER_NAMES else sigmas
    planner_sigmas = _planner_sigmas_for_recorded_steps(recorded_sigmas, len(recorder.x0_steps))
    recorder.step_sigmas = planner_sigmas
    if pece_corrections is not None:
        order = int((sampler_options or {}).get("corrector_order", 4))
        expected_counts = [1] + [2 if order > 0 else 1] * (len(sigmas) - 2)
        if calls_per_step != expected_counts:
            raise RuntimeError("SA-PECE planner correction evaluation count changed")
        if any((pece_corrections[i] is None) != (expected_counts[i] == 1)
               for i in range(len(calls_per_step))):
            raise RuntimeError("SA-PECE corrected planner prediction is missing")
    return out, recorder.x0_steps, planner_sigmas, planner_noise, pece_corrections


def _build_highres_noise(
    highres_latent: dict,
    lowres_noise: torch.Tensor,
    seed: int,
    hf_strength: float,
    mode: str = "legacy",
):
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

    if mode == "patch_coupled":
        return build_patch_coupled_noise(
            lowres_noise=lowres_noise,
            target_shape=tuple(highres_samples.shape),
            seed=seed,
        ).to(device=target_device, dtype=target_dtype, non_blocking=True)
    if mode != "legacy":
        raise ValueError(f"unsupported nested-noise mode: {mode}")

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


def _full_lock_handoff(runtime: ScaleLockedRuntimeContext, config: ScaleLockConfig):
    """Target state at the composition-lock release, without model calls.

    Locked steps discard the target prediction, so evaluating the model there
    is wasted work. For flow-matching Euler the state after those steps is
    ``sigma * noise + (1 - sigma) * clean`` with ``clean`` a fixed convex
    combination of the resized planner predictions. Returns
    ``(locked_steps, clean)`` in the model's internal latent space, or None
    when no step is locked or the masked fallback path must run instead.
    """
    sigma_values = [float(v) for v in runtime.sigmas.detach().cpu().tolist()]
    locked_steps = full_lock_prefix_length(sigma_values, config.full_lock_sigma)
    if locked_steps == 0:
        return None
    if config.spatial_mask is not None or isinstance(runtime.highres_latent.get("noise_mask"), torch.Tensor):
        # Masked locks blend with the target prediction; it must be evaluated.
        return None
    if config.manifold_enabled:
        return None
    if len(runtime.anchors_x0) < locked_steps:
        raise RuntimeError("planner recorded fewer anchors than locked target steps")

    target = runtime.highres_latent["samples"]
    initial = target
    if torch.count_nonzero(initial) > 0:
        # Mirrors CFGGuider.inner_sample, which leaves empty latents unshifted.
        initial = runtime.model.model.process_latent_in(initial)
    predictions = (
        resize_4d_tensor(
            runtime.anchors_x0[i].to(device=target.device, dtype=torch.float32, non_blocking=True),
            tuple(target.shape[-2:]),
        )
        for i in range(locked_steps)
    )
    clean = flow_euler_locked_clean_state(initial, predictions, sigma_values[: locked_steps + 1])
    return locked_steps, clean


def sample_with_runtime(
    *,
    guider,
    sampler,
    runtime: ScaleLockedRuntimeContext,
    config: ScaleLockConfig,
    restore_after: bool = True,
    analytic_full_lock: bool = False,
    skip_locked_model_calls: bool = False,
    diagnostics: Optional["Krea2HandoffDiagnostics"] = None,
) -> ScaleLockedSampleResult:
    if runtime.sigmas.numel() == 0:
        out = clean_latent(runtime.highres_latent)
        return ScaleLockedSampleResult(output=out, lowres_planner=clean_latent(runtime.lowres_out), denoised_output=out)

    sigmas = runtime.sigmas
    latent_samples = runtime.highres_latent["samples"]
    handoff = _full_lock_handoff(runtime, config) if analytic_full_lock else None
    if handoff is not None:
        locked_steps, clean = handoff
        if locked_steps == len(sigmas) - 1:
            # Every step is locked: the result is the final clean state.
            out = clean_latent(runtime.highres_latent)
            out["samples"] = runtime.model.model.process_latent_out(clean).to(
                comfy.model_management.intermediate_device()
            )
            return ScaleLockedSampleResult(
                output=out, lowres_planner=clean_latent(runtime.lowres_out), denoised_output=out
            )
        # The sampler re-applies process_latent_in and the flow noise scaling,
        # reproducing sigma * noise + (1 - sigma) * clean at the release step.
        sigmas = sigmas[locked_steps:]
        latent_samples = runtime.model.model.process_latent_out(clean).to(dtype=latent_samples.dtype)

    apply_scale_lock_to_guider(guider, runtime, config, skip_locked_model_calls=skip_locked_model_calls)

    x0_output = {}
    callback = make_preview_callback(runtime.model, len(sigmas) - 1, x0_output)
    if diagnostics is not None:
        diagnostics.set_planner(runtime.lowres_out["samples"])
        callback = diagnostics.wrap(
            callback, stage=0, stage_hw=tuple(latent_samples.shape[-2:]), sigmas=sigmas,
            noise=runtime.highres_noise, source=runtime.lowres_out["samples"], lock_sigma=config.full_lock_sigma,
        )
    noise_mask = runtime.highres_latent.get("noise_mask", None)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    try:
        samples = guider.sample(
            runtime.highres_noise,
            latent_samples,
            sampler,
            sigmas,
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


def _stage_seed(detail_seed: int, stage_index: int) -> int:
    return (int(detail_seed) ^ ((stage_index + 1) * 0xD1B54A32D192ED03)) & 0xFFFFFFFFFFFFFFFF


def _run_resolution_ladder(
    *,
    model,
    positive,
    negative,
    cfg,
    sampler_name: str,
    sigmas: torch.Tensor,
    highres_latent: dict,
    lowres_latent: dict,
    lowres_out: dict,
    anchors_x0: list,
    planner_sigmas: list,
    lowres_noise: torch.Tensor,
    detail_seed: int,
    stochastic_seed: int,
    nested_noise_strength: float,
    noise_mode: str,
    config: ScaleLockConfig,
    stage_hw: tuple,
    sampler_options: Optional[dict] = None,
    pece_corrected_anchors: Optional[list] = None,
    diagnostics: Optional["Krea2HandoffDiagnostics"] = None,
) -> ScaleLockedSampleResult:
    """Refine through increasing resolutions, each stage finishing its image.

    The first stage reaches the composition-lock release exactly like the
    single-stage path: in closed form for Euler, and for other solvers by
    integrating the locked prefix natively with the planner predictions in
    place of target model calls, which preserves their multistep history.
    Every later stage re-noises the finished image of the stage below, resized to
    its own size, to the same release noise level with its own patch-coupled
    noise, and runs the remaining steps. Each stage therefore repeats the
    planner-to-2x handoff at a bounded scale ratio, starting from a sharp
    image rather than from mid-noise predictions, whose posterior-mean blur
    a late, low-noise start cannot remove. A later stage is a fresh solver
    start at the release sigma, so multistep solvers begin it at first order.
    """
    if config.full_lock_sigma is None:
        raise ValueError("resolution stages require the composition lock")
    if not isinstance(_model_sampling_obj(model), comfy.model_sampling.CONST):
        raise ValueError("resolution stages require flow model sampling")
    euler = _normalize_sampler_name(sampler_name) == "euler"
    if (
        isinstance(highres_latent.get("noise_mask"), torch.Tensor)
        or config.spatial_mask is not None
        or config.manifold_enabled
    ):
        raise ValueError("resolution stages do not support latent noise masks, spatial lock masks or manifold correction")

    target = highres_latent["samples"]
    effective_sigmas = effective_er_sde_sigmas(sigmas, model) if sampler_name in SNR_SOLVER_NAMES else sigmas
    sigma_values = [float(v) for v in effective_sigmas.detach().cpu().tolist()]
    total_steps = len(sigma_values) - 1
    release = full_lock_prefix_length(sigma_values, config.full_lock_sigma)
    if release >= total_steps:
        raise ValueError("resolution stages need at least one step after the composition lock")
    sizes = [tuple(map(int, hw)) for hw in stage_hw] + [tuple(map(int, target.shape[-2:]))]
    previous = tuple(map(int, lowres_latent["samples"].shape[-2:]))
    for hw in sizes:
        if hw[0] < previous[0] or hw[1] < previous[1] or hw == previous or hw[0] % 2 or hw[1] % 2:
            raise ValueError("resolution stages must grow monotonically in both dimensions on the 16-pixel grid")
        previous = hw

    stage_noise = lowres_noise
    shapes = tuple(tuple(lowres_noise.shape[:-2]) + tuple(hw) for hw in sizes)
    sampler = comfy.samplers.sampler_object(sampler_name) if sampler_name not in SNR_SOLVER_NAMES else None
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    x0_output: dict = {}
    finished = None
    if diagnostics is not None:
        diagnostics.set_planner(lowres_out["samples"])

    for index, hw in enumerate(sizes):
        final = index == len(sizes) - 1
        stage_latent = highres_latent if final else resize_latent_dict(highres_latent, hw)
        stage_samples = stage_latent["samples"]
        seed = detail_seed if final else _stage_seed(detail_seed, index)
        stage_noise = _build_highres_noise(
            highres_latent=stage_latent,
            lowres_noise=stage_noise,
            seed=seed,
            hf_strength=nested_noise_strength,
            mode=noise_mode,
        )

        stage_sigmas = sigmas[release:]
        if finished is None:
            start_latent = stage_samples
            if not euler:
                # Native solver over the full schedule; locked steps take the
                # planner prediction and cost no model evaluation.
                stage_sigmas = sigmas
            elif release > 0:
                initial = stage_samples
                if torch.count_nonzero(initial) > 0:
                    initial = model.model.process_latent_in(initial)
                predictions = (
                    resize_4d_tensor(
                        anchors_x0[k].to(device=stage_samples.device, dtype=torch.float32, non_blocking=True), hw
                    )
                    for k in range(release)
                )
                clean = flow_euler_locked_clean_state(initial, predictions, sigma_values[: release + 1])
                start_latent = model.model.process_latent_out(clean).to(dtype=stage_samples.dtype)
        else:
            # Output-space latent: CFGGuider.inner_sample applies process_latent_in
            # and the flow noise scaling re-noises it to the release sigma.
            start_latent = resize_4d_tensor(
                finished.to(device=stage_samples.device, dtype=torch.float32), hw
            ).to(dtype=stage_samples.dtype)

        if sampler_name in SNR_SOLVER_NAMES:
            sampler = _krea2_sampler_object(
                sampler_name, sampler_options, model=model, low_shape=tuple(lowres_noise.shape),
                stage_shapes=shapes, stage_index=index, seed=stochastic_seed,
                start_step=0 if finished is None else release,
                sigmas=stage_sigmas, noise_like=stage_noise,
            )

        stage_runtime = ScaleLockedRuntimeContext(
            model=model,
            highres_latent=stage_latent,
            lowres_latent=lowres_latent,
            lowres_out=lowres_out,
            sigmas=sigmas,
            anchors_x0=anchors_x0,
            planner_sigmas=planner_sigmas,
            highres_noise=stage_noise,
            noise_seed=seed,
            pece_corrected_anchors=pece_corrected_anchors,
        )
        guider = create_cfg_guider(model, positive, negative, cfg)
        apply_scale_lock_to_guider(guider, stage_runtime, config, skip_locked_model_calls=True)
        stage_x0: dict = {}
        callback = make_preview_callback(model, len(stage_sigmas) - 1, stage_x0)
        if diagnostics is not None:
            callback = diagnostics.wrap(
                callback, stage=index, stage_hw=hw,
                sigmas=effective_er_sde_sigmas(stage_sigmas, model) if sampler_name in SNR_SOLVER_NAMES else stage_sigmas,
                noise=stage_noise, source=lowres_out["samples"] if finished is None else finished,
                lock_sigma=config.full_lock_sigma,
            )
        try:
            finished = guider.sample(
                stage_noise,
                start_latent,
                sampler,
                stage_sigmas,
                denoise_mask=None,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )
        finally:
            restore_original_predict_noise(guider)
        if final:
            x0_output = stage_x0

    samples = finished.to(comfy.model_management.intermediate_device())
    out = clean_latent(highres_latent)
    out["samples"] = samples
    if "x0" in x0_output:
        try:
            x0_out = model.model.process_latent_out(x0_output["x0"].cpu())
        except Exception:
            x0_out = x0_output["x0"].detach().cpu()
        denoised = clean_latent(highres_latent)
        denoised["samples"] = x0_out
    else:
        denoised = out
    return ScaleLockedSampleResult(output=out, lowres_planner=clean_latent(lowres_out), denoised_output=denoised)


REGENERATION_MODES = ("trajectory", "planner_final_img2img")
HANDOFF_DIAGNOSTIC_MODES = ("off", "log", "log_and_save_latents")


def img2img_start_index(sigma_values, start_sigma: float) -> int:
    """Schedule index where planner-final img2img refinement starts.

    The refinement resumes the native schedule at its first sigma that does
    not exceed ``start_sigma``, the same discrete suffix rule as ComfyUI's
    KSamplerAdvanced ``start_at_step`` and the composition-lock release. The
    schedule itself is never rescaled; the returned sigma is exact.
    """
    values = [float(v) for v in sigma_values]
    if isinstance(start_sigma, bool) or not isinstance(start_sigma, (int, float)) \
            or not math.isfinite(start_sigma) or not 0.0 < float(start_sigma) < 1.0:
        raise ValueError("img2img_start_sigma must be a finite value strictly between 0 and 1")
    steps = len(values) - 1
    index = full_lock_prefix_length(values, float(start_sigma))
    if index == 0:
        raise ValueError(
            "img2img_start_sigma is at or above the first schedule sigma; that discards the planner "
            "image entirely. Lower img2img_start_sigma."
        )
    if index >= steps:
        raise ValueError(
            "img2img_start_sigma is below every nonzero schedule sigma, leaving no refinement step; "
            "raise img2img_start_sigma or steps"
        )
    return index


class Krea2HandoffDiagnostics:
    """Opt-in observer for the planner-to-target handoff.

    It reads the preview callback's denoised prediction and state, never the
    sampler inputs, so sampling is bitwise unchanged. Metrics compare against
    the finished planner latent in the model's internal space:

    * ``x0_vs_planner_lowband``: the prediction area-reduced to the planner
      grid against the finished planner (structural drift).
    * ``state_clean_vs_planner_lowband``: the clean component
      ``(x - sigma * noise_scale * noise) / (1 - sigma)`` of the state against
      the finished planner. Exact for the injected start noise of
      deterministic solvers; stochastic steps add fresh noise.
    * ``state_clean_vs_stage_source``: the same clean component against the
      stage's resized source image. Planner-final img2img starts at ~0 by
      construction; trajectory mode shows how far its release state is
      from the finished planner.
    * ``x0_vs_stage_source``: the prediction against the stage's own resized
      source image at full resolution.

    Optional ``.latent`` files (ComfyUI SaveLatent format, output space) hold
    each stage's starting state and every denoised prediction, for decoding
    with LoadLatent + VAE Decode. Nothing is decoded here.
    """

    def __init__(self, model, mode: str, regeneration_mode: str, seed: int):
        if mode not in HANDOFF_DIAGNOSTIC_MODES or mode == "off":
            raise ValueError(f"unsupported handoff diagnostics mode: {mode!r}")
        self.model = model
        self.regeneration_mode = regeneration_mode
        self.save = mode == "log_and_save_latents"
        self.seed = int(seed)
        self.reference = None
        self.directory = None
        self.records: list[dict] = []
        self.saved: list[str] = []

    def set_planner(self, planner_out: torch.Tensor) -> None:
        with torch.no_grad():
            self.reference = self.model.model.process_latent_in(
                planner_out.detach().to(device="cpu", dtype=torch.float32)
            )
        if self.save:
            self._save("planner_final", planner_out)

    def _output_dir(self):
        if self.directory is None:
            import os
            import time
            import folder_paths
            name = f"krea2_handoff_{time.strftime('%Y%m%d-%H%M%S')}_seed{self.seed}_{self.regeneration_mode}"
            self.directory = os.path.join(folder_paths.get_output_directory(), "scale_locked_diagnostics", name)
            os.makedirs(self.directory, exist_ok=True)
        return self.directory

    def _save(self, name: str, output_space: torch.Tensor) -> None:
        import os
        import safetensors.torch
        path = os.path.join(self._output_dir(), name + ".latent")
        safetensors.torch.save_file({
            "latent_tensor": output_space.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            "latent_format_version_0": torch.tensor([]),
        }, path)
        self.saved.append(path)

    @staticmethod
    def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).norm() / b.norm().clamp_min(1e-12))

    def wrap(self, callback, *, stage: int, stage_hw, sigmas: torch.Tensor, noise: torch.Tensor,
             source: Optional[torch.Tensor], lock_sigma: Optional[float]):
        if self.reference is None:
            raise RuntimeError("handoff diagnostics need the finished planner first")
        sigma_values = [float(v) for v in sigmas.detach().cpu().tolist()]
        planner_hw = tuple(self.reference.shape[-2:])
        noise_scale = float(getattr(_model_sampling_obj(self.model), "noise_scale", 1.0))
        source_in = None
        if source is not None:
            with torch.no_grad():
                source_in = self.model.model.process_latent_in(
                    resize_4d_tensor(source.detach().to(device="cpu", dtype=torch.float32), tuple(stage_hw))
                )
        state = {"previous_locked": None}

        def observe(step, x0, x, total_steps):
            sigma = sigma_values[step] if 0 <= step < len(sigma_values) else float("nan")
            locked = lock_sigma is not None and sigma > lock_sigma
            with torch.no_grad():
                pred = x0.detach().to(device="cpu", dtype=torch.float32)
                record = {
                    "regeneration_mode": self.regeneration_mode, "stage": stage,
                    "stage_hw": list(stage_hw), "step": int(step), "total_steps": int(total_steps),
                    "sigma": sigma, "locked": bool(locked),
                    "x0_vs_planner_lowband": self._rel(resize_4d_tensor(pred, planner_hw), self.reference),
                }
                if 0.0 < sigma < 0.99:
                    clean = (x.detach().to(device="cpu", dtype=torch.float32)
                             - sigma * noise_scale * noise.detach().to(device="cpu", dtype=torch.float32)
                             ) / (1.0 - sigma)
                    record["state_clean_vs_planner_lowband"] = self._rel(
                        resize_4d_tensor(clean, planner_hw), self.reference)
                    if source_in is not None:
                        record["state_clean_vs_stage_source"] = self._rel(clean, source_in)
                if source_in is not None:
                    record["x0_vs_stage_source"] = self._rel(pred, source_in)
                self.records.append(record)
                _LOGGER.info("Krea 2 handoff diagnostics: %s", record)
                if self.save:
                    released = state["previous_locked"] is True and not locked
                    if step == 0 or released:
                        self._save(f"stage{stage}_step{step:03d}_state",
                                   self.model.model.process_latent_out(x.detach().float().cpu()))
                    self._save(f"stage{stage}_step{step:03d}_x0", self.model.model.process_latent_out(pred))
            state["previous_locked"] = locked
            callback(step, x0, x, total_steps)

        return observe


def _run_planner_final_img2img(
    *,
    model,
    positive,
    negative,
    cfg,
    sampler_name: str,
    sigmas: torch.Tensor,
    highres_latent: dict,
    lowres_latent: dict,
    lowres_out: dict,
    lowres_noise: torch.Tensor,
    detail_seed: int,
    stochastic_seed: int,
    nested_noise_strength: float,
    noise_mode: str,
    stage_hw: tuple,
    start_sigma: float,
    sampler_options: Optional[dict] = None,
    diagnostics: Optional[Krea2HandoffDiagnostics] = None,
) -> ScaleLockedSampleResult:
    """Img2img refinement of the finished planner image at each stage.

    Every stage resizes the finished latent of the stage below (the planner
    for the first one), and the native sampler re-noises it to the exact
    schedule sigma ``sigmas[k]`` through ``CFGGuider.inner_sample`` and the
    model's ``noise_scaling``: ``x = sigma_k * s * noise + (1 - sigma_k) * clean``
    with ``clean = process_latent_in(resized)`` and the patch-coupled noise
    used by the trajectory path. The native solver then runs ``sigmas[k:]``
    from a fresh start, so no multistep history, PECE correction or planner
    trajectory anchor carries into it. Composition-lock settings are unused.
    """
    if not isinstance(_model_sampling_obj(model), comfy.model_sampling.CONST):
        raise ValueError("planner_final_img2img requires flow model sampling")
    if isinstance(highres_latent.get("noise_mask"), torch.Tensor):
        raise ValueError("planner_final_img2img does not support latent noise masks")
    if torch.count_nonzero(lowres_noise).item() == 0:
        raise ValueError("planner_final_img2img requires Gaussian noise (add_noise)")
    sigma_values = [float(v) for v in sigmas.detach().cpu().tolist()]
    start = img2img_start_index(sigma_values, start_sigma)
    target = highres_latent["samples"]
    sizes = [tuple(map(int, hw)) for hw in stage_hw] + [tuple(map(int, target.shape[-2:]))]
    previous = tuple(map(int, lowres_latent["samples"].shape[-2:]))
    for hw in sizes:
        if hw[0] < previous[0] or hw[1] < previous[1] or hw == previous or hw[0] % 2 or hw[1] % 2:
            raise ValueError("resolution stages must grow monotonically in both dimensions on the 16-pixel grid")
        previous = hw

    stage_sigmas = sigmas[start:]
    _LOGGER.info(
        "Krea 2 planner_final_img2img: requested start sigma %.4f -> schedule sigma %.6f "
        "(step %d of %d, %d refinement steps per stage, %d stage(s))",
        float(start_sigma), sigma_values[start], start, len(sigma_values) - 1,
        len(sigma_values) - 1 - start, len(sizes),
    )
    shapes = tuple(tuple(lowres_noise.shape[:-2]) + tuple(hw) for hw in sizes)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    source = lowres_out["samples"]  # finished planner, output latent space
    stage_noise = lowres_noise
    x0_output: dict = {}
    if diagnostics is not None:
        diagnostics.set_planner(source)

    for index, hw in enumerate(sizes):
        final = index == len(sizes) - 1
        stage_latent = highres_latent if final else resize_latent_dict(highres_latent, hw)
        stage_samples = stage_latent["samples"]
        seed = detail_seed if final else _stage_seed(detail_seed, index)
        stage_noise = _build_highres_noise(
            highres_latent=stage_latent,
            lowres_noise=stage_noise,
            seed=seed,
            hf_strength=nested_noise_strength,
            mode=noise_mode,
        )
        # Output-space latent; CFGGuider.inner_sample applies process_latent_in.
        start_latent = resize_4d_tensor(
            source.to(device=stage_samples.device, dtype=torch.float32), hw
        ).to(dtype=stage_samples.dtype)
        if sampler_name in SNR_SOLVER_NAMES:
            sampler = _krea2_sampler_object(
                sampler_name, sampler_options, model=model, low_shape=tuple(lowres_noise.shape),
                stage_shapes=shapes, stage_index=index, seed=stochastic_seed,
                start_step=start, sigmas=stage_sigmas, noise_like=stage_noise,
            )
        else:
            sampler = comfy.samplers.sampler_object(sampler_name)
        guider = create_cfg_guider(model, positive, negative, cfg)
        stage_x0: dict = {}
        callback = make_preview_callback(model, len(stage_sigmas) - 1, stage_x0)
        if diagnostics is not None:
            callback = diagnostics.wrap(
                callback, stage=index, stage_hw=hw, sigmas=stage_sigmas, noise=stage_noise,
                source=source, lock_sigma=None,
            )
        source = guider.sample(
            stage_noise,
            start_latent,
            sampler,
            stage_sigmas,
            denoise_mask=None,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        if final:
            x0_output = stage_x0

    samples = source.to(comfy.model_management.intermediate_device())
    out = clean_latent(highres_latent)
    out["samples"] = samples
    if "x0" in x0_output:
        try:
            x0_out = model.model.process_latent_out(x0_output["x0"].cpu())
        except Exception:
            x0_out = x0_output["x0"].detach().cpu()
        denoised = clean_latent(highres_latent)
        denoised["samples"] = x0_out
    else:
        denoised = out
    return ScaleLockedSampleResult(output=out, lowres_planner=clean_latent(lowres_out), denoised_output=denoised)


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
    noise_mode: str = "legacy",
    stage_hw: tuple = (),
    sampler_options: Optional[dict] = None,
    sa_planner_native_only: bool = False,
    regeneration_mode: str = "trajectory",
    img2img_start_sigma: Optional[float] = None,
    handoff_diagnostics: str = "off",
) -> ScaleLockedSampleResult:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if regeneration_mode not in REGENERATION_MODES:
        raise ValueError(f"regeneration_mode must be one of {', '.join(REGENERATION_MODES)}")
    if handoff_diagnostics not in HANDOFF_DIAGNOSTIC_MODES:
        raise ValueError(f"handoff_diagnostics must be one of {', '.join(HANDOFF_DIAGNOSTIC_MODES)}")
    img2img = regeneration_mode == "planner_final_img2img"
    if img2img:
        if (config.lock_strength != 0.0 or config.mid_strength != 0.0 or config.spatial_mask is not None
                or config.manifold_enabled):
            raise ValueError(
                "planner_final_img2img does not apply planner trajectory anchors; "
                "set coarse_lock_strength and mid_lock_strength to 0"
            )
        if denoise != 1.0 or not add_noise:
            raise ValueError("planner_final_img2img requires a full planner pass with added noise")

    highres_latent = fix_latent_channels(model, latent_image)
    lowres_latent = make_lowres_latent(highres_latent, target_megapixels)
    if denoise <= 0.0:
        out = clean_latent(highres_latent)
        return ScaleLockedSampleResult(output=out, lowres_planner=clean_latent(lowres_latent), denoised_output=out)

    # The Krea 2 ER-SDE node injects a verified single-prediction native
    # sampler and patch-coupled stochastic increments. The generic SLRD
    # alignment guard remains conservative for arbitrary ER-SDE samplers.
    if not (sampler_name in SNR_SOLVER_NAMES and sampler_options is not None):
        guard_sampler_alignment(sampler_name, sampler_guard)
    sigmas = calculate_sigmas(model, scheduler=scheduler, steps=steps, denoise=denoise)
    if sigmas.numel() == 0:
        out = clean_latent(highres_latent)
        return ScaleLockedSampleResult(output=out, lowres_planner=clean_latent(lowres_latent), denoised_output=out)

    if img2img:
        # Reject an unusable start sigma before spending the planner pass.
        img2img_start_index(sigmas.tolist(), img2img_start_sigma)
    disable_noise = not bool(add_noise)
    planner_seed = int(seed)
    detail_seed = int(seed) ^ 0x9E3779B97F4A7C15

    # A native-only planner is an invocation-scoped quality comparator.
    # Reset the context before either target stage: target Spectrum forecasts
    # remain enabled, and no sampler or MODEL wrapper is modified.
    with sa_native_only_scope(
        bool(sa_planner_native_only)
        and sampler_name in ("sa_solver", "sa_solver_pece")
    ):
        lowres_out, anchors_x0, planner_sigmas, lowres_noise, pece_corrections = _run_lowres_planner(
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
            sampler_options=sampler_options,
        )
    if len(anchors_x0) == 0:
        raise RuntimeError("ScaleLockedResidualKSampler: planner pass did not record any x0 anchors.")

    diagnostics = (None if handoff_diagnostics == "off" else
                   Krea2HandoffDiagnostics(model, handoff_diagnostics, regeneration_mode, int(seed)))
    if img2img:
        return _run_planner_final_img2img(
            model=model, positive=positive, negative=negative, cfg=cfg,
            sampler_name=sampler_name, sigmas=sigmas, highres_latent=highres_latent,
            lowres_latent=lowres_latent, lowres_out=lowres_out, lowres_noise=lowres_noise,
            detail_seed=detail_seed, stochastic_seed=planner_seed,
            nested_noise_strength=nested_noise_strength, noise_mode=noise_mode,
            stage_hw=tuple(stage_hw), start_sigma=img2img_start_sigma,
            sampler_options=sampler_options, diagnostics=diagnostics,
        )

    if stage_hw:
        return _run_resolution_ladder(
            model=model, positive=positive, negative=negative, cfg=cfg,
            sampler_name=sampler_name, sigmas=sigmas, highres_latent=highres_latent,
            lowres_latent=lowres_latent, lowres_out=lowres_out, anchors_x0=anchors_x0,
            planner_sigmas=planner_sigmas, lowres_noise=lowres_noise, detail_seed=detail_seed,
            stochastic_seed=planner_seed, nested_noise_strength=nested_noise_strength, noise_mode=noise_mode,
            config=config, stage_hw=tuple(stage_hw), sampler_options=sampler_options,
            pece_corrected_anchors=pece_corrections, diagnostics=diagnostics,
        )

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
            mode=noise_mode,
        ),
        noise_seed=detail_seed,
        pece_corrected_anchors=pece_corrections,
    )
    guider = create_cfg_guider(model, positive, negative, cfg)
    sampler = _krea2_sampler_object(
        sampler_name, sampler_options, model=model, low_shape=tuple(lowres_noise.shape),
        stage_shapes=(tuple(runtime.highres_noise.shape),), stage_index=0,
        seed=planner_seed, start_step=0, sigmas=sigmas, noise_like=runtime.highres_noise,
    )
    # The closed-form lock handoff is exact only for the flow-matching Euler update.
    analytic_full_lock = (
        config.full_lock_sigma is not None
        and _normalize_sampler_name(sampler_name) == "euler"
        and isinstance(_model_sampling_obj(model), comfy.model_sampling.CONST)
    )
    # Other solvers integrate the locked prefix natively; answering those steps
    # with the planner prediction skips only target evaluations whose output the
    # full lock would discard, so solver history and the result are unchanged.
    return sample_with_runtime(
        guider=guider, sampler=sampler, runtime=runtime, config=config,
        analytic_full_lock=analytic_full_lock,
        skip_locked_model_calls=config.full_lock_sigma is not None,
        diagnostics=diagnostics,
    )

