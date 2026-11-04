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
