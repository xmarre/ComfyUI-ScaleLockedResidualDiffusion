"""CPU numerical tests for Krea 2's phase-coupled noise transport."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "krea2_flow", Path(__file__).resolve().parents[1] / "krea2_flow.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
build_patch_coupled_noise = module.build_patch_coupled_noise
PatchMeanProjection = module.PatchMeanProjection


def grouped_phase_means(target, source_hw):
    """Independent slice-based oracle, including unequal parent group sizes."""
    image = target.squeeze(2) if target.ndim == 5 else target
    sh, sw = source_hw
    th, tw = image.shape[-2:]
    result = image.new_empty((*image.shape[:2], sh, sw))
    for y in range(sh // 2):
        ys = [i for i in range(th // 2) if i * (sh // 2) // (th // 2) == y]
        for x in range(sw // 2):
            xs = [i for i in range(tw // 2) if i * (sw // 2) // (tw // 2) == x]
            for py in range(2):
                for px in range(2):
                    group = image[..., 2 * ys[0] + py:2 * ys[-1] + py + 1:2,
                                  2 * xs[0] + px:2 * xs[-1] + px + 1:2]
                    result[..., 2 * y + py, 2 * x + px] = group.mean((-2, -1))
    return result.unsqueeze(2) if target.ndim == 5 else result


@pytest.mark.parametrize("low, high", [((8, 10), (16, 20)), ((8, 10), (14, 24)),
                                         ((8, 10), (8, 14)), ((10, 6), (16, 10))])
@pytest.mark.parametrize("temporal", [False, True])
def test_exact_patch_mean_lock_preserves_detail_and_is_idempotent(low, high, temporal):
    shape = (2, 16, 1) if temporal else (2, 16)
    generator = torch.Generator().manual_seed(94)
    base = torch.randn((*shape, *high), generator=generator, dtype=torch.float64)
    planner = torch.randn((*shape, *low), generator=generator, dtype=torch.float64)
    projection = PatchMeanProjection(low, high)
    original, original_planner = base.clone(), planner.clone()
    corrected = projection.correct(base, planner, 1.0)
    assert torch.allclose(grouped_phase_means(corrected, low), planner, atol=1e-12, rtol=1e-12)
    assert torch.allclose(projection.correct(corrected, planner, 1.0), corrected, atol=1e-12, rtol=1e-12)
    # The zero-mean residual within each parent phase group remains available
    # to the target denoiser. The correction cannot add a competing residual.
    before_detail = base - projection.lift(grouped_phase_means(base, low))
    after_detail = corrected - projection.lift(grouped_phase_means(corrected, low))
    assert torch.allclose(after_detail, before_detail, atol=1e-12, rtol=1e-12)
    assert torch.equal(base, original)
    assert torch.equal(planner, original_planner)


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_mid_band_candidate_cannot_override_requested_coarse_constraint(strength):
    g = torch.Generator().manual_seed(73)
    base = torch.randn((1, 16, 14, 24), generator=g, dtype=torch.float64)
    planner = torch.randn((1, 16, 8, 10), generator=g, dtype=torch.float64)
    candidate = torch.randn(base.shape, generator=g, dtype=torch.float64) * 3.0
    projection = PatchMeanProjection((8, 10), (14, 24))
    result = projection.correct(base, planner, strength, candidate)
    expected = torch.lerp(grouped_phase_means(base, (8, 10)), planner, strength)
    assert torch.allclose(grouped_phase_means(result, (8, 10)), expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(result - projection.lift(expected),
                          candidate - projection.lift(grouped_phase_means(candidate, (8, 10))),
                          atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype, tolerance", [(torch.float32, 2e-6),
                                             (torch.float16, 2e-3), (torch.bfloat16, 2e-2)])
def test_patch_lock_preserves_model_dtype_and_zero_lock_identity(dtype, tolerance):
    g = torch.Generator().manual_seed(23)
    base = torch.randn((1, 16, 1, 14, 24), generator=g).to(dtype)
    planner = torch.randn((1, 16, 1, 8, 10), generator=g).to(dtype)
    projection = PatchMeanProjection((8, 10), (14, 24))
    assert projection.correct(base, planner, 0.0) is base
    result = projection.correct(base, planner, 1.0)
    assert result.dtype == dtype
    assert bool(torch.isfinite(result).all())
    assert torch.allclose(grouped_phase_means(result.float(), (8, 10)), planner.float(),
                          atol=tolerance, rtol=tolerance)


def test_patch_projection_uses_the_noise_ownership_partition():
    g = torch.Generator().manual_seed(18)
    source = torch.randn((1, 16, 8, 10), generator=g, dtype=torch.float64)
    target = build_patch_coupled_noise(source, (1, 16, 14, 24), seed=89)
    projection = PatchMeanProjection((8, 10), (14, 24))
    # Noise has normalized-sum constraints, signals have mean constraints.
    # Derive unequal group counts independently from the owner definition.
    cy = torch.bincount(torch.arange(7) * 4 // 7, minlength=4)
    cx = torch.bincount(torch.arange(12) * 5 // 12, minlength=5)
    counts = (cy[:, None] * cx[None, :]).repeat_interleave(2, 0).repeat_interleave(2, 1)
    assert torch.allclose(projection.reduce(target) * counts.sqrt(), source, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("low, high", [((7, 8), (14, 16)), ((8, 8), (6, 16)), ((0, 8), (8, 16))])
def test_invalid_projection_geometry(low, high):
    with pytest.raises(ValueError):
        PatchMeanProjection(low, high)


def test_patch_lock_validates_layout_and_strength():
    projection = PatchMeanProjection((8, 10), (14, 24))
    base = torch.zeros(1, 16, 14, 24)
    planner = torch.zeros(1, 16, 8, 10)
    for strength in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="strength"):
            projection.correct(base, planner, strength)
    with pytest.raises(ValueError, match="layouts"):
        projection.correct(base, planner.repeat(2, 1, 1, 1), 1.0)
    with pytest.raises(ValueError, match="image latents"):
        projection.correct(base.unsqueeze(2).repeat(1, 1, 2, 1, 1),
                           planner.unsqueeze(2).repeat(1, 1, 2, 1, 1), 1.0)


@pytest.mark.parametrize(
    "low, high",
    [
        ((1, 16, 8, 10), (1, 16, 16, 20)),
        ((2, 16, 8, 10), (2, 16, 14, 24)),
        ((1, 16, 8, 10), (1, 16, 8, 14)),
        ((1, 16, 10, 6), (1, 16, 16, 10)),
    ],
)
def test_phase_projection(low, high):
    source = torch.randn(low, generator=torch.Generator().manual_seed(42))
    original = source.clone()
    target = build_patch_coupled_noise(source, high, seed=123)
    assert target.shape == high
    assert torch.equal(source, original)

    sy, sx = low[-2] // 2, low[-1] // 2
    ty, tx = high[-2] // 2, high[-1] // 2
    owner_y = torch.arange(ty) * sy // ty
    owner_x = torch.arange(tx) * sx // tx
    counts = (
        torch.bincount(owner_y, minlength=sy)[:, None]
        * torch.bincount(owner_x, minlength=sx)[None, :]
    )
    for py in range(2):
        for px in range(2):
            phase = target[..., py::2, px::2]
            row_sums = torch.zeros((*phase.shape[:2], sy, tx))
            row_sums.index_add_(2, owner_y, phase)
            sums = torch.zeros((*phase.shape[:2], sy, sx))
            sums.index_add_(3, owner_x, row_sums)
            assert torch.allclose(
                sums / counts.sqrt(), source[..., py::2, px::2],
                atol=1e-5, rtol=1e-5,
            )


def test_determinism_and_marginal_statistics():
    source = torch.randn((3, 16, 60, 60), generator=torch.Generator().manual_seed(888))
    first = build_patch_coupled_noise(source, (3, 16, 120, 120), seed=9)
    second = build_patch_coupled_noise(source, (3, 16, 120, 120), seed=9)
    third = build_patch_coupled_noise(source, (3, 16, 120, 120), seed=10)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert abs(float(first.mean())) < 0.025
    assert 0.97 < float(first.std()) < 1.03


def test_dtype_preserved():
    source = torch.randn((1, 16, 8, 10)).to(torch.bfloat16)
    out = build_patch_coupled_noise(source, (1, 16, 14, 22), seed=2)
    assert out.dtype == torch.bfloat16
    assert bool(torch.isfinite(out).all())


@pytest.mark.parametrize(
    "low, target",
    [
        (torch.ones((1, 16, 9, 10)), (1, 16, 16, 20)),
        (torch.ones((1, 16, 8, 10)), (1, 16, 6, 20)),
        (torch.ones((1, 16, 8, 10)), (1, 16, 8, 10)),
        (torch.ones((1, 16, 8, 10)), (1, 8, 16, 20)),
    ],
)
def test_invalid_geometries_rejected(low, target):
    with pytest.raises((ValueError, TypeError)):
        build_patch_coupled_noise(low, target, seed=1)


def test_nonfinite_source_rejected():
    source = torch.ones((1, 16, 8, 10))
    source[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_patch_coupled_noise(source, (1, 16, 16, 20), seed=1)


def test_single_frame_noise_preserves_native_layout_and_four_dimensional_coupling():
    source = torch.randn((2, 16, 1, 8, 10), generator=torch.Generator().manual_seed(73))
    target = build_patch_coupled_noise(source, (2, 16, 1, 14, 24), seed=19)
    expected = build_patch_coupled_noise(source.squeeze(2), (2, 16, 14, 24), seed=19)
    assert target.shape == (2, 16, 1, 14, 24)
    assert torch.equal(target.squeeze(2), expected)


def test_multiple_temporal_slices_are_not_image_noise():
    with pytest.raises(ValueError, match="single temporal slice"):
        build_patch_coupled_noise(torch.zeros(1, 16, 2, 8, 10), (1, 16, 2, 16, 20), seed=1)


def test_single_frame_spatial_helpers_preserve_native_latent_layout():
    import sys
    core_spec = importlib.util.spec_from_file_location(
        "slrd_core_image_test", Path(__file__).resolve().parents[1] / "slrd_core.py"
    )
    core = importlib.util.module_from_spec(core_spec)
    sys.modules[core_spec.name] = core
    core_spec.loader.exec_module(core)
    source = torch.randn((2, 16, 1, 8, 10), generator=torch.Generator().manual_seed(23))
    image = source.squeeze(2)
    for size in ((4, 6), (14, 24), (8, 10)):
        out = core.resize_4d_tensor(source, size)
        assert out.shape == (2, 16, 1, *size)
        assert torch.equal(out.squeeze(2), core.resize_4d_tensor(image, size))
    assert core.latent_target_hw_from_megapixels(source, 1.0) == core.latent_target_hw_from_megapixels(image, 1.0)
    with pytest.raises(ValueError, match="image latent"):
        core.latent_target_hw_from_megapixels(source.repeat(1, 1, 2, 1, 1), 1.0)
