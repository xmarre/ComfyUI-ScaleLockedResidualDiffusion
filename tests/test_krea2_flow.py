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


def _with_lattice(shape, seed, strength=0.25):
    g = torch.Generator().manual_seed(seed)
    content = torch.randn(shape, generator=g, dtype=torch.float64)
    offsets = strength * torch.randn(shape[:-2] + (2, 2), generator=g, dtype=torch.float64)
    offsets = offsets - offsets.mean(dim=(-2, -1), keepdim=True)
    tiled = offsets.repeat(*([1] * (len(shape) - 2)), shape[-2] // 2, shape[-1] // 2)
    return content, offsets, content + tiled


@pytest.mark.parametrize("shape", [(1, 16, 12, 20), (2, 16, 30, 14), (1, 16, 1, 16, 18)])
def test_patch_lattice_removal_is_the_exact_projection_onto_three_fourier_modes(shape):
    content, offsets, corrupted = _with_lattice(shape, 3)
    fixed = module.remove_patch_lattice(corrupted)
    assert fixed.shape == corrupted.shape and fixed.dtype == corrupted.dtype
    assert module.patch_lattice_modes(fixed).abs().max() < 1e-12
    # Idempotent, and linear: the injected offset is removed exactly, leaving
    # the content minus its own (chance-level) lattice modes.
    assert torch.allclose(module.remove_patch_lattice(fixed), fixed, atol=1e-12)
    assert torch.allclose(fixed, module.remove_patch_lattice(content), atol=1e-12)
    # Only the three period-2 Fourier coefficients of each plane change.
    delta = torch.fft.fft2(corrupted - fixed)
    h, w = shape[-2:]
    keep = torch.zeros(h, w, dtype=torch.bool)
    keep[0, w // 2] = keep[h // 2, 0] = keep[h // 2, w // 2] = True
    assert delta[..., ~keep].abs().max() < 1e-9
    assert torch.allclose(fixed.mean(dim=(-2, -1)), corrupted.mean(dim=(-2, -1)), atol=1e-12)
    # The measured modes reproduce the per-position offsets.
    modes = module.patch_lattice_modes(corrupted) - module.patch_lattice_modes(content)
    cols, rows, diag = modes.unbind(-1)
    assert torch.allclose(cols + rows + diag, offsets[..., 0, 0], atol=1e-12)
    assert torch.allclose(-cols + rows - diag, offsets[..., 0, 1], atol=1e-12)
    assert torch.allclose(cols - rows - diag, offsets[..., 1, 0], atol=1e-12)
    assert torch.allclose(-cols - rows + diag, offsets[..., 1, 1], atol=1e-12)


def test_patch_lattice_removal_leaves_grid_free_content_and_other_periods_alone():
    g = torch.Generator().manual_seed(4)
    content = torch.randn(1, 16, 96, 128, generator=g)
    fixed = module.remove_patch_lattice(content)
    # Three of 12288 coefficients per plane: a chance-level change.
    assert (fixed - content).pow(2).mean().sqrt() < 0.03 * content.std()
    # A pattern shifted off the lattice period (three latent pixels) is kept.
    period3 = torch.zeros(1, 16, 96, 126)
    period3[..., ::3] = 1.0
    assert torch.allclose(module.remove_patch_lattice(period3), period3, atol=1e-6)
    # A one-pixel shift of the grid origin flips the sign of the column and row modes.
    _, _, corrupted = _with_lattice((1, 16, 16, 16), 5)
    modes = module.patch_lattice_modes(corrupted - module.remove_patch_lattice(corrupted))
    shifted = module.patch_lattice_modes(torch.roll(corrupted - module.remove_patch_lattice(corrupted), (1, 1), (-2, -1)))
    assert torch.allclose(shifted, modes * torch.tensor([-1.0, -1.0, 1.0], dtype=modes.dtype), atol=1e-12)


def test_patch_lattice_removal_preserves_dtype_and_does_not_mutate():
    _, _, corrupted = _with_lattice((1, 16, 8, 8), 6)
    for dtype in (torch.float32, torch.float16):
        x = corrupted.to(dtype)
        original = x.clone()
        fixed = module.remove_patch_lattice(x)
        assert fixed.dtype == dtype and torch.equal(x, original) and fixed is not x


def test_patch_lattice_metrics_separate_locked_offsets_from_chance():
    g = torch.Generator().manual_seed(7)
    content = torch.randn(1, 16, 64, 96, generator=g)
    clean = module.patch_lattice_metrics(content)
    assert clean["chance"] == pytest.approx((32 * 48) ** -0.5)
    assert clean["coherence"] < 2 * clean["chance"] and clean["max_coherence"] < 5 * clean["chance"]
    offsets = torch.zeros(1, 16, 2, 2)
    offsets[:, :, 0, 0], offsets[:, :, 1, 1] = 0.3, -0.3     # pure row + column modes
    locked = module.patch_lattice_metrics(content + offsets.repeat(1, 1, 32, 48))
    assert locked["coherence"] > 5 * clean["coherence"] and locked["max_coherence"] > 0.25
    assert locked["amplitude"] == pytest.approx(0.3 / 2 ** 0.5, rel=0.1)
    # A plane that consists only of the pattern is fully coherent.
    pure = module.patch_lattice_metrics(offsets.repeat(1, 1, 32, 48))
    assert pure["max_coherence"] == pytest.approx(1.0)
    assert set(clean) == {"coherence", "max_coherence", "chance", "amplitude"}
    assert all(type(v) is float for v in clean.values())


@pytest.mark.parametrize("shape", [(1, 16, 7, 8), (1, 16, 8, 9), (1, 16, 0, 8)])
def test_patch_lattice_rejects_grids_off_the_patch_geometry(shape):
    for fn in (module.patch_lattice_modes, module.remove_patch_lattice, module.patch_lattice_metrics):
        with pytest.raises(ValueError, match="even latent H/W"):
            fn(torch.zeros(shape))
    with pytest.raises(TypeError):
        module.patch_lattice_modes(torch.zeros(1, 16, 8, 8, dtype=torch.long))
