"""CPU checks for the full planner lock used by the Krea 2 Turbo node.

ComfyUI is replaced by import stubs. The Euler loop below matches ComfyUI's
``sample_euler`` update for flow models. These checks establish trajectory
algebra and gating only; they do not establish image quality.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "slrd_composition_lock_test"


def _load_runtime():
    for name in ("comfy", "comfy.model_management", "comfy.sample", "comfy.samplers", "comfy.utils", "latent_preview"):
        sys.modules.setdefault(name, types.ModuleType(name))
    comfy = sys.modules["comfy"]
    for child in ("model_management", "sample", "samplers", "utils"):
        setattr(comfy, child, sys.modules[f"comfy.{child}"])
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE] = package
    for name in ("krea2_flow", "slrd_core", "slrd_runtime"):
        spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{PACKAGE}.slrd_runtime"], sys.modules[f"{PACKAGE}.slrd_core"]


runtime, core = _load_runtime()


def _config(full_lock_sigma):
    return runtime.ScaleLockConfig(
        lock_strength=0.75, lock_strength_start=1.0, lock_strength_end=0.2,
        cutoff=0.33, mid_cutoff=0.6, mid_strength=0.2, schedule="hold_then_drop",
        schedule_power=2.0, schedule_hold=0.25, mid_strength_start=1.0,
        mid_strength_end=0.0, mid_schedule="linked", mid_schedule_power=2.0,
        mid_schedule_hold=0.0, full_lock_sigma=full_lock_sigma,
    )


def _flow_sigmas(steps, shift=1.15):
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return shift * t / (1.0 + (shift - 1.0) * t)


def _euler(x, sigmas, predict):
    for i in range(len(sigmas) - 1):
        denoised = predict(x, sigmas[i])
        x = x + (x - denoised) / sigmas[i] * (sigmas[i + 1] - sigmas[i])
    return x


class _Guider:
    def __init__(self, model_prediction):
        self._model_prediction = model_prediction
        self.calls = 0

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        self.calls += 1
        return self._model_prediction(x, timestep)


def _locked_guider(model_prediction, anchors, planner_sigmas, highres, full_lock_sigma, spatial_mask=None):
    guider = _Guider(model_prediction)
    config = _config(full_lock_sigma)
    config.spatial_mask = spatial_mask
    context = runtime.ScaleLockedRuntimeContext(
        model=None, highres_latent={"samples": highres}, lowres_latent=None,
        lowres_out=None, sigmas=None, anchors_x0=anchors,
        planner_sigmas=planner_sigmas, highres_noise=None, noise_seed=0,
    )
    return runtime.apply_scale_lock_to_guider(guider, context, config)


def _planner(low_shape, sigmas, seed=5):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(low_shape[1], low_shape[1], generator=generator, dtype=torch.float64) * 0.3
    bias = torch.randn(low_shape, generator=generator, dtype=torch.float64)

    def model(x, sigma):
        return torch.einsum("oc,bc...->bo...", weight, x) * float(sigma) + bias * (1.0 - float(sigma))

    noise = torch.randn(low_shape, generator=generator, dtype=torch.float64)
    anchors = []

    def record(x, sigma):
        out = model(x, sigma)
        anchors.append(out.clone())
        return out

    final = _euler(noise, sigmas, record)
    return noise, anchors, final


@pytest.mark.parametrize("layout", ["bchw", "bc1hw"])
def test_locked_target_state_is_planner_state_with_target_noise(layout):
    sigmas = _flow_sigmas(8)
    low_shape, high_shape = (1, 16, 8, 10), (1, 16, 14, 18)
    if layout == "bc1hw":
        low_shape, high_shape = (1, 16, 1, 8, 10), (1, 16, 1, 14, 18)
    _, anchors, _ = _planner(low_shape, sigmas)
    planner_sigmas = [float(s) for s in sigmas[:-1]]
    release = 0.6
    locked_steps = sum(float(s) > release for s in sigmas[:-1])
    assert 0 < locked_steps < len(sigmas) - 1

    target_noise = torch.randn(high_shape, generator=torch.Generator().manual_seed(9), dtype=torch.float64)

    def displaced_model(x, sigma):
        # A target prediction unrelated to the planner composition.
        return torch.roll(x, shifts=3, dims=-1) * 0.5 + 2.0

    guider = _locked_guider(displaced_model, anchors, planner_sigmas, torch.zeros(high_shape, dtype=torch.float64), release)
    x = target_noise
    for i in range(locked_steps):
        denoised = guider.predict_noise(x, sigmas[i])
        assert torch.equal(denoised, core.resize_4d_tensor(anchors[i], high_shape[-2:]))
        x = x + (x - denoised) / sigmas[i] * (sigmas[i + 1] - sigmas[i])

    # Flow Euler keeps x_k = sigma_k * noise + (1 - sigma_k) * A_k. While locked,
    # A_k is the planner's own accumulated prediction, resized to the target.
    low_x = torch.zeros(low_shape, dtype=torch.float64)
    planner_clean = _euler(low_x, sigmas[: locked_steps + 1], lambda _x, s: anchors[[float(v) for v in sigmas].index(float(s))])
    sigma = float(sigmas[locked_steps])
    expected = sigma * target_noise + core.resize_4d_tensor(planner_clean, high_shape[-2:])
    assert torch.allclose(x, expected, atol=1e-10, rtol=1e-10)
    assert guider.calls == locked_steps


def test_release_threshold_is_strict_and_none_disables():
    sigmas = _flow_sigmas(8)
    low_shape, high_shape = (1, 16, 8, 10), (1, 16, 16, 20)
    _, anchors, _ = _planner(low_shape, sigmas)
    planner_sigmas = [float(s) for s in sigmas[:-1]]

    def model(x, sigma):
        return torch.full_like(x, 7.0)

    x = torch.zeros(high_shape, dtype=torch.float64)
    full = _locked_guider(model, anchors, planner_sigmas, x, 1.0)
    # sigma == threshold is not above it, so 1.0 disables the full lock.
    assert not torch.equal(full.predict_noise(x, sigmas[0]), core.resize_4d_tensor(anchors[0], high_shape[-2:]))

    zero = _locked_guider(model, anchors, planner_sigmas, x, 0.0)
    for i in range(len(planner_sigmas)):
        assert torch.equal(zero.predict_noise(x, sigmas[i]), core.resize_4d_tensor(anchors[i], high_shape[-2:]))

    legacy = _locked_guider(model, anchors, planner_sigmas, x, None)
    reference = _Guider(model)
    core.init_scale_lock_state(
        reference, model=None, anchors_x0_cpu=anchors, planner_sigmas=planner_sigmas,
        lock_strength=0.75, lock_strength_start=1.0, lock_strength_end=0.2, cutoff=0.33,
        mid_cutoff=0.6, mid_strength=0.2, schedule="hold_then_drop", schedule_power=2.0,
        schedule_hold=0.25, mid_strength_start=1.0, mid_strength_end=0.0, mid_schedule="linked",
        mid_schedule_power=2.0, mid_schedule_hold=0.0, spatial_mask=None,
    )
    for i in range(len(planner_sigmas)):
        expected = runtime.apply_scale_lock_to_noise_prediction(reference, model(x, sigmas[i]), x, sigmas[i])
        assert torch.equal(legacy.predict_noise(x, sigmas[i]), expected)


def test_locked_prediction_does_not_alias_stored_anchor_and_respects_mask():
    sigmas = _flow_sigmas(4)
    shape = (1, 16, 8, 10)
    _, anchors, _ = _planner(shape, sigmas)
    planner_sigmas = [float(s) for s in sigmas[:-1]]

    def model(x, sigma):
        return torch.full_like(x, 3.0)

    x = torch.zeros(shape, dtype=torch.float64)
    guider = _locked_guider(model, anchors, planner_sigmas, x, 0.5)
    out = guider.predict_noise(x, sigmas[0])
    assert torch.equal(out, anchors[0]) and out.data_ptr() != anchors[0].data_ptr()
    out.add_(1.0)
    assert not torch.equal(out, anchors[0])

    mask = torch.zeros((1, 1, 8, 10), dtype=torch.float64)
    mask[..., :, :5] = 1.0
    masked = _locked_guider(model, anchors, planner_sigmas, x, 0.5, spatial_mask=mask)
    out = masked.predict_noise(x, sigmas[0])
    assert torch.equal(out[..., :5], anchors[0][..., :5])
    assert torch.equal(out[..., 5:], torch.full_like(out[..., 5:], 3.0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_invalid_threshold_rejected(value):
    with pytest.raises(ValueError, match="full_lock_sigma"):
        core.init_scale_lock_state(
            _Guider(None), model=None, anchors_x0_cpu=[], planner_sigmas=None,
            lock_strength=1.0, lock_strength_start=1.0, lock_strength_end=1.0, cutoff=0.33,
            mid_cutoff=0.6, mid_strength=0.0, schedule="flat", schedule_power=2.0,
            schedule_hold=0.0, mid_strength_start=1.0, mid_strength_end=0.0, mid_schedule="linked",
            mid_schedule_power=2.0, mid_schedule_hold=0.0, spatial_mask=None, full_lock_sigma=value,
        )
