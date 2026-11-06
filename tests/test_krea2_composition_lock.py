"""CPU checks for the full planner lock used by the Krea 2 Turbo node.

ComfyUI is replaced by import stubs. The Euler loop below matches ComfyUI's
``sample_euler`` update for flow models. These checks establish trajectory
algebra and gating only; they do not establish image quality.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "slrd_composition_lock_test"


def _load_runtime():
    for name in ("comfy", "comfy.model_management", "comfy.model_sampling", "comfy.sample", "comfy.samplers", "comfy.utils", "latent_preview"):
        sys.modules.setdefault(name, types.ModuleType(name))
    comfy = sys.modules["comfy"]
    for child in ("model_management", "model_sampling", "sample", "samplers", "utils"):
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
    # ComfyUI's flux_time_shift form, as used by Krea 2's model sampling.
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    sigmas = torch.zeros_like(t)
    live = t > 0
    sigmas[live] = math.exp(shift) / (math.exp(shift) + (1.0 / t[live] - 1.0))
    return sigmas


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


def _locked_guider(model_prediction, anchors, planner_sigmas, highres, full_lock_sigma, spatial_mask=None,
                   skip_locked_model_calls=False, noise_mask=None):
    guider = _Guider(model_prediction)
    config = _config(full_lock_sigma)
    config.spatial_mask = spatial_mask
    context = runtime.ScaleLockedRuntimeContext(
        model=None, highres_latent={"samples": highres, **({"noise_mask": noise_mask} if noise_mask is not None else {})}, lowres_latent=None,
        lowres_out=None, sigmas=None, anchors_x0=anchors,
        planner_sigmas=planner_sigmas, highres_noise=None, noise_seed=0,
    )
    return runtime.apply_scale_lock_to_guider(guider, context, config,
                                              skip_locked_model_calls=skip_locked_model_calls)


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
    release = 0.8
    locked_steps = sum(float(s) > release for s in sigmas[:-1])
    assert locked_steps == 4

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


@pytest.mark.parametrize("steps, release", [(8, 0.8), (12, 0.8), (8, 0.6), (5, 0.0)])
def test_closed_form_handoff_matches_locked_euler_steps(steps, release):
    sigmas = _flow_sigmas(steps)
    low_shape, high_shape = (1, 16, 8, 10), (1, 16, 14, 18)
    _, anchors, _ = _planner(low_shape, sigmas)
    planner_sigmas = [float(s) for s in sigmas[:-1]]
    locked = core.full_lock_prefix_length([float(s) for s in sigmas], release)
    assert locked == sum(float(s) > release for s in sigmas[:-1])

    target_noise = torch.randn(high_shape, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    guider = _locked_guider(lambda x, s: x * 0.0 + 5.0, anchors, planner_sigmas,
                            torch.zeros(high_shape, dtype=torch.float64), release)
    x = target_noise
    for i in range(locked):
        denoised = guider.predict_noise(x, sigmas[i])
        x = x + (x - denoised) / sigmas[i] * (sigmas[i + 1] - sigmas[i])

    clean = core.flow_euler_locked_clean_state(
        torch.zeros(high_shape, dtype=torch.float64),
        (core.resize_4d_tensor(anchors[i], high_shape[-2:]) for i in range(locked)),
        [float(s) for s in sigmas[: locked + 1]],
    )
    sigma = float(sigmas[locked])
    # Same tolerance class as float32 accumulation in the closed form.
    assert torch.allclose(sigma * target_noise + (1.0 - sigma) * clean.double(), x, atol=1e-5, rtol=1e-5)


def test_closed_form_handoff_rejects_invalid_schedules():
    zero = torch.zeros(1, 16, 4, 4)
    with pytest.raises(ValueError, match="one more sigma"):
        core.flow_euler_locked_clean_state(zero, [zero], [1.0])
    with pytest.raises(ValueError, match="decrease"):
        core.flow_euler_locked_clean_state(zero, [zero], [0.5, 0.7])
    assert core.full_lock_prefix_length([1.0, 0.5, 0.0], None) == 0
    assert core.full_lock_prefix_length([1.0, 0.9, 0.0], 0.0) == 2


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
    assert torch.allclose(out[..., :5], anchors[0][..., :5], atol=1e-12, rtol=0.0)
    assert torch.allclose(out[..., 5:], torch.full_like(out[..., 5:], 3.0), atol=1e-12, rtol=0.0)


def _dpmpp_2m(x, sigmas, predict):
    # ComfyUI k_diffusion.sampling.sample_dpmpp_2m update, verbatim.
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    for i in range(len(sigmas) - 1):
        denoised = predict(x, sigmas[i])
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


def _ipndm(x, sigmas, predict, max_order=4):
    # ComfyUI k_diffusion.sampling.sample_ipndm update, verbatim.
    buffer_model = []
    for i in range(len(sigmas) - 1):
        t_cur, t_next = sigmas[i], sigmas[i + 1]
        denoised = predict(x, t_cur)
        d_cur = (x - denoised) / t_cur
        order = min(max_order, i + 1)
        if t_next == 0:
            x_next = denoised
        elif order == 1:
            x_next = x + (t_next - t_cur) * d_cur
        elif order == 2:
            x_next = x + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:
            x_next = x + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        else:
            x_next = x + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        if len(buffer_model) == max_order - 1:
            buffer_model = buffer_model[1:] + [d_cur]
        else:
            buffer_model.append(d_cur)
        x = x_next
    return x


@pytest.mark.parametrize("solver", [_euler, _dpmpp_2m, _ipndm])
@pytest.mark.parametrize("steps, release", [(8, 0.8), (12, 0.8), (18, 0.8), (6, 0.0), (8, 1.0)])
def test_skipped_locked_evaluations_preserve_native_multistep_trajectory(solver, steps, release):
    """Answering locked steps from the planner equals evaluating and discarding.

    The solver runs every step either way, so its multistep history across the
    lock release is identical; only the discarded target evaluations vanish.
    """
    sigmas = _flow_sigmas(steps)
    low_shape, high_shape = (1, 16, 8, 10), (1, 16, 14, 18)
    _, anchors, _ = _planner(low_shape, sigmas)
    planner_sigmas = [float(s) for s in sigmas[:-1]]
    locked = core.full_lock_prefix_length([float(s) for s in sigmas], release)

    def target_model(x, sigma):
        return torch.roll(x, shifts=2, dims=-1) * 0.4 + torch.sin(x) * float(sigma)

    noise = torch.randn(high_shape, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    reference = _locked_guider(target_model, anchors, planner_sigmas, noise, release)
    fast = _locked_guider(target_model, anchors, planner_sigmas, noise, release, skip_locked_model_calls=True)
    expected = solver(noise.clone(), sigmas, reference.predict_noise)
    actual = solver(noise.clone(), sigmas, fast.predict_noise)
    assert torch.equal(actual, expected)
    assert reference.calls == steps
    assert fast.calls == steps - locked


def test_skipped_locked_evaluation_still_runs_model_for_masked_lock():
    sigmas = _flow_sigmas(4)
    shape = (1, 16, 8, 10)
    _, anchors, _ = _planner(shape, sigmas)
    planner_sigmas = [float(s) for s in sigmas[:-1]]
    mask = torch.zeros((1, 1, 8, 10), dtype=torch.float64)
    mask[..., :, :5] = 1.0
    x = torch.zeros(shape, dtype=torch.float64)

    def model(x, sigma):
        return torch.full_like(x, 3.0)

    masked = _locked_guider(model, anchors, planner_sigmas, x, 0.5, spatial_mask=mask, skip_locked_model_calls=True)
    out = masked.predict_noise(x, sigmas[0])
    assert masked.calls == 1
    assert torch.allclose(out[..., 5:], torch.full_like(out[..., 5:], 3.0), atol=1e-12, rtol=0.0)

    unmasked = _locked_guider(model, anchors, planner_sigmas, x, 0.5, skip_locked_model_calls=True)
    out = unmasked.predict_noise(x, sigmas[0])
    assert unmasked.calls == 0
    assert torch.equal(out, anchors[0]) and out.data_ptr() != anchors[0].data_ptr()


def test_sampler_noise_mask_prevents_locked_model_call_shortcut():
    """A latent noise_mask must use the evaluated path, independent of the
    scale-lock's separate spatial_mask, as promised by the node interface."""
    sigmas = _flow_sigmas(4)
    shape = (1, 16, 8, 10)
    _, anchors, _ = _planner(shape, sigmas)
    planner_sigmas = [float(s) for s in sigmas[:-1]]
    latent = torch.zeros(shape, dtype=torch.float64)
    denoise_mask = torch.ones((1, 1, 8, 10), dtype=torch.float64)
    model = lambda x, sigma: torch.full_like(x, 3.0)

    masked = _locked_guider(
        model, anchors, planner_sigmas, latent, 0.5,
        skip_locked_model_calls=True, noise_mask=denoise_mask)
    result = masked.predict_noise(latent, sigmas[0])
    assert masked.calls == 1
    assert torch.equal(result, anchors[0])  # The full lock still wins.
    assert result.data_ptr() != anchors[0].data_ptr()

    plain = _locked_guider(
        model, anchors, planner_sigmas, latent, 0.5,
        skip_locked_model_calls=True)
    assert torch.equal(plain.predict_noise(latent, sigmas[0]), anchors[0])
    assert plain.calls == 0


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


def test_stage_planning_is_geometric_and_bounded():
    planner, target = 3.0e6, 14.0e6
    stages = core.plan_stage_megapixels(planner, target, 4.0)
    assert len(stages) == 1
    assert stages[0] == pytest.approx((3.0 * 14.0) ** 0.5)
    sizes = [planner / 1e6, *stages, target / 1e6]
    assert all(b / a <= 4.0 + 1e-9 for a, b in zip(sizes, sizes[1:]))
    assert len(core.plan_stage_megapixels(1.0e6, 64.0e6, 4.0)) == 2
    # Exactly the limit needs no intermediate stage, despite rounding in the logarithm.
    assert core.plan_stage_megapixels(1048576.0, 4194304.0, 4.0) == []
    assert core.plan_stage_megapixels(1.0e6, 14.0e6, 0.0) == []
    with pytest.raises(ValueError, match="max_stage_upscale"):
        core.plan_stage_megapixels(1.0e6, 14.0e6, 1.0)


def test_stage_megapixels_parsing():
    assert core.parse_stage_megapixels("") == []
    assert core.parse_stage_megapixels(" 4, 8.5 ;12 ") == [4.0, 8.5, 12.0]
    with pytest.raises(ValueError, match="not a number"):
        core.parse_stage_megapixels("4, big")
    with pytest.raises(ValueError, match="positive"):
        core.parse_stage_megapixels("4, -1")
