"""Mathematical checks for the spectral progressive-resolution primitives.

These establish transform, distribution and scheduling invariants on CPU.
They do not establish image quality or GPU speed.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("slrd_spectral_test", ROOT / "spectral_progressive.py")
sp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sp
_spec.loader.exec_module(sp)


def _flow_sigmas(steps, shift=1.15):
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    sigmas = torch.zeros_like(t)
    live = t > 0
    sigmas[live] = math.exp(shift) / (math.exp(shift) + (1.0 / t[live] - 1.0))
    return sigmas.tolist()


# --- transforms ------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 7, 16, 90, 128])
def test_dct_matrix_is_orthonormal(n):
    c = sp.dct_matrix(n, dtype=torch.float64)
    assert torch.allclose(c @ c.T, torch.eye(n, dtype=torch.float64), atol=1e-12)


def test_dct_matches_scipy_orthonormal_type2():
    scipy_fft = pytest.importorskip("scipy.fft")
    x = torch.randn(2, 3, 18, 26, dtype=torch.float64)
    ours = sp.dct2(x)
    ref = torch.from_numpy(scipy_fft.dctn(x.numpy(), type=2, norm="ortho", axes=(-2, -1)))
    assert torch.allclose(ours, ref, atol=1e-10)
    assert torch.allclose(sp.idct2(ours), x, atol=1e-10)


def test_float32_round_trip_on_large_krea2_axis():
    x = torch.randn(1, 16, 64, 256)
    assert (sp.idct2(sp.dct2(x)) - x).abs().max() < 2e-4


# --- expansion ---------------------------------------------------------------

def test_expansion_shapes_dtype_device_and_5d_layout():
    x = torch.randn(2, 16, 1, 40, 64, dtype=torch.bfloat16)
    out, aligned = sp.expand_flow_state(x, (60, 96), 0.7, seed=3)
    assert out.shape == (2, 16, 1, 60, 96) and out.dtype == torch.bfloat16 and out.device == x.device
    assert 0.7 < aligned < 1.0


def test_expansion_is_deterministic_and_seed_separated():
    x = torch.randn(1, 16, 32, 48)
    a, _ = sp.expand_flow_state(x, (64, 96), 0.6, seed=11)
    b, _ = sp.expand_flow_state(x, (64, 96), 0.6, seed=11)
    c, _ = sp.expand_flow_state(x, (64, 96), 0.6, seed=12)
    assert torch.equal(a, b)
    assert not torch.allclose(a, c)


def test_expansion_keeps_duplicate_comfy_batch_index_noise_identical():
    # ComfyUI's prepare_noise reuses exactly the same noise draw for rows
    # sharing batch_index. The high-frequency expansion must preserve it.
    x = torch.randn(1, 16, 24, 36).repeat(3, 1, 1, 1)
    out, aligned = sp.expand_flow_state(
        x, (48, 72), 0.63, seed=1234, batch_inds=[2, 2, 5]
    )
    assert aligned > 0.63
    assert torch.equal(out[0], out[1])
    assert not torch.equal(out[0], out[2])


def test_expansion_batch_indices_preserve_existing_sequential_draws_and_validate_shape():
    x = torch.randn(3, 16, 20, 30)
    original, _ = sp.expand_flow_state(x, (40, 60), 0.55, seed=88)
    indexed, _ = sp.expand_flow_state(x, (40, 60), 0.55, seed=88, batch_inds=[0, 1, 2])
    assert torch.equal(original, indexed)
    for invalid in ([0, 1], [-1, 0, 1]):
        with pytest.raises(ValueError, match="batch_inds"):
            sp.expand_flow_state(x, (40, 60), 0.55, seed=88, batch_inds=invalid)


def test_expansion_retains_low_band_exactly_and_projection_recovers_source():
    x = torch.randn(1, 4, 30, 44, dtype=torch.float64)
    sigma = 0.55
    out, _ = sp.expand_flow_state(x, (45, 66), sigma, seed=5)
    r = sp.area_ratio((30, 44), (45, 66))
    low = sp.dct2(out)[..., :30, :44] / sp.kappa(sigma, r)
    assert torch.allclose(low, sp.dct2(x), atol=1e-10)
    # DCT truncation is the exact left inverse of the expansion's low band.
    assert torch.allclose(sp.idct2(low), x, atol=1e-10)


@pytest.mark.parametrize("src,dst", [((32, 32), (64, 64)), ((30, 44), (46, 66)), ((40, 24), (56, 40))])
def test_expanded_noise_is_white_with_aligned_variance(src, dst):
    # Pure-noise state: x = sigma * eps. After expansion and kappa the state
    # must be N(0, sigma~^2) and spatially white, also for noninteger ratios.
    sigma = 0.8
    torch.manual_seed(0)
    x = sigma * torch.randn(64, 16, *src, dtype=torch.float64)
    out, aligned = sp.expand_flow_state(x, dst, sigma, seed=9)
    assert aligned == pytest.approx(sp.align_sigma(sigma, sp.area_ratio(src, dst)))
    z = out / aligned
    assert z.mean().abs() < 0.01
    assert z.var().item() == pytest.approx(1.0, abs=0.01)
    # Neighbour correlations of white noise vanish.
    assert (z[..., 1:, :] * z[..., :-1, :]).mean().abs() < 0.01
    assert (z[..., :, 1:] * z[..., :, :-1]).mean().abs() < 0.01
    # Per-coefficient variance is flat across low and new high bands.
    coeff_var = sp.dct2(z).var(dim=(0, 1))
    assert coeff_var[: src[0], : src[1]].mean().item() == pytest.approx(1.0, abs=0.03)
    assert coeff_var[src[0]:, :].mean().item() == pytest.approx(1.0, abs=0.03)


def test_expansion_restores_flow_signal_amplitude_for_band_limited_content():
    # A band-limited clean image x0_hi and its DCT-truncated low-resolution
    # version (same pixel amplitude) give x_t~ = (1-t~) x0_hi + t~ eps~.
    hi, lo = (48, 72), (24, 36)
    coeffs = torch.zeros(1, 2, *hi, dtype=torch.float64)
    coeffs[..., : lo[0], : lo[1]] = torch.randn(1, 2, *lo, dtype=torch.float64)
    x0_hi = sp.idct2(coeffs)
    r = sp.area_ratio(lo, hi)
    x0_lo = sp.idct2(coeffs[..., : lo[0], : lo[1]] / r)  # same pixel amplitude, smaller grid
    eps = torch.randn(1, 2, *lo, dtype=torch.float64)
    sigma = 0.6
    x_lo = (1 - sigma) * x0_lo + sigma * eps
    out, aligned = sp.expand_flow_state(x_lo, hi, sigma, seed=1)
    noise = out - (1 - aligned) * x0_hi
    low_noise = sp.dct2(noise)[..., : lo[0], : lo[1]]
    assert torch.allclose(low_noise, aligned * sp.dct2(eps), atol=1e-10)


def test_noise_scale_enters_new_coefficients():
    x = torch.zeros(1, 1, 16, 16, dtype=torch.float64)
    out, _ = sp.expand_flow_state(x, (32, 32), 0.5, seed=2, noise_scale=3.0)
    out1, _ = sp.expand_flow_state(x, (32, 32), 0.5, seed=2, noise_scale=1.0)
    assert torch.allclose(out, 3.0 * out1, atol=1e-12)


def test_alignment_matches_reference_formulas_and_composes():
    for t in (0.1, 0.5, 0.83):
        for r in (1.25, 1.5, 2.0, 3.7):
            ref_kappa = r / (1.0 + (r - 1.0) * t)
            assert sp.kappa(t, r) == pytest.approx(ref_kappa)
            assert sp.align_sigma(t, r) == pytest.approx(t * ref_kappa)
            # Two expansions equal one with the product ratio.
            assert sp.align_sigma(sp.align_sigma(t, r), 1.3) == pytest.approx(sp.align_sigma(t, r * 1.3))
            assert t < sp.align_sigma(t, r) < 1.0


@pytest.mark.parametrize("bad", [((32, 32), (32, 32)), ((32, 32), (30, 64)), ((0, 4), (8, 8))])
def test_expansion_rejects_non_growing_grids(bad):
    with pytest.raises(ValueError):
        sp.area_ratio(*bad)


@pytest.mark.parametrize("sigma", [0.0, 1.0, -0.1, float("nan")])
def test_expansion_rejects_invalid_sigma(sigma):
    with pytest.raises(ValueError):
        sp.expand_flow_state(torch.randn(1, 1, 8, 8), (16, 16), sigma, seed=0)


# --- scheduling ------------------------------------------------------------

def test_activation_matches_reference_equation_and_is_monotone():
    for p in (0.01, 0.3, 4.0, 200.0):
        for d in (0.001, 0.01, 0.2):
            ref = 1.0 / (1.0 + math.sqrt(d / (p * (1.0 + p - d))))
            assert sp.activation_sigma(p, d) == pytest.approx(ref)
    # Stronger frequencies become informative earlier (higher sigma);
    # a smaller tolerance delays nothing being kept noise dominated longer.
    assert sp.activation_sigma(10.0, 0.01) > sp.activation_sigma(0.1, 0.01)
    assert sp.activation_sigma(1.0, 0.001) > sp.activation_sigma(1.0, 0.1)


def _calibration(a=200.0, beta=2.0, grid=(128, 128)):
    return {"type": sp.CALIBRATION_TYPE, "version": sp.CALIBRATION_VERSION, "A": a, "beta": beta,
            "r2": 0.99, "bins": 50, "grid": grid, "reference_length": min(grid), "channels": 16,
            "samples": 1, "latent_format": "Wan21", "fit_band": (0.05, 0.9)}


def test_spectral_thresholds_reproduce_reference_at_calibration_grid():
    cal = _calibration(a=219.484718, beta=2.422687, grid=(90, 160))
    sizes = [(45, 80), (90, 160)]
    expected = sp.activation_sigma(219.484718 * (0.5 * 45) ** -2.422687, 0.01)
    assert sp.spectral_transition_sigmas(sizes, cal, 0.01) == [pytest.approx(expected)]


def test_amplitude_transfer_scales_with_shorter_side_squared_and_is_bounded():
    cal = _calibration(grid=(128, 128))
    assert sp.calibrated_amplitude(cal, (256, 384)) == pytest.approx(800.0)
    assert sp.calibrated_amplitude(cal, (64, 64)) == pytest.approx(50.0)
    with pytest.raises(ValueError, match="calibrate on latents near"):
        sp.calibrated_amplitude(cal, (288, 288))
    with pytest.raises(ValueError):
        sp.calibrated_amplitude(cal, (62, 62))


def test_transfer_is_consistent_with_resampled_content():
    # The same periodic content on half the grid (exact Fourier truncation,
    # pixel amplitude preserved) has a quarter of the per-coefficient power,
    # so the transferred amplitude equals a direct measurement on the full grid.
    field = _power_law_field((8, 4, 128, 128), 2.0, seed=1)
    spectrum = torch.fft.fftshift(torch.fft.fft2(field, norm="ortho"), dim=(-2, -1))
    cropped = spectrum[..., 32:96, 32:96] * 0.5
    half_field = torch.fft.ifft2(torch.fft.ifftshift(cropped, dim=(-2, -1)), norm="ortho").real
    full = sp.build_calibration(field, latent_format="Wan21", fit_low=0.05, fit_high=0.45)
    half = sp.build_calibration(half_field, latent_format="Wan21", fit_low=0.1, fit_high=0.9)
    assert half["beta"] == pytest.approx(full["beta"], abs=0.1)
    assert sp.calibrated_amplitude(half, (128, 128)) == pytest.approx(full["A"], rel=0.1)


def test_transition_steps_follow_schedule_and_reject_impossible_points():
    schedule = _flow_sigmas(20)
    sizes = [(64, 64), (96, 96), (128, 128)]
    steps = sp.transition_steps(schedule, [0.9, 0.6], sizes)
    assert steps[0] == next(k for k, s in enumerate(schedule) if s <= 0.9)
    assert schedule[steps[0] - 1] > 0.9 >= schedule[steps[0]]
    assert steps[1] > steps[0]
    with pytest.raises(ValueError, match="leaves no step"):
        sp.transition_steps([0.95] + schedule[1:], [0.97], sizes[1:])
    with pytest.raises(ValueError, match="leaves no step"):
        sp.transition_steps(schedule, [0.9, 0.89], sizes)
    with pytest.raises(ValueError, match="never reached"):
        sp.transition_steps(schedule, [1e-6], sizes[1:])
    with pytest.raises(ValueError, match="transition sigma"):
        sp.transition_steps(schedule, [0.9], sizes)
    with pytest.raises(ValueError, match="one step per resolution"):
        sp.transition_steps(_flow_sigmas(2), [0.9, 0.5], sizes)


def test_parse_transition_sigmas():
    assert sp.parse_transition_sigmas(" 0.85, 0.6 ") == [0.85, 0.6]
    assert sp.parse_transition_sigmas("") == []
    for bad in ("0.5, 0.6", "1.0", "0", "a"):
        with pytest.raises(ValueError):
            sp.parse_transition_sigmas(bad)


# --- calibration -----------------------------------------------------------

def _power_law_field(shape, beta, seed=0):
    g = torch.Generator().manual_seed(seed)
    b, c, h, w = shape
    fy = torch.fft.fftfreq(h, dtype=torch.float64).unsqueeze(1)
    fx = torch.fft.fftfreq(w, dtype=torch.float64).unsqueeze(0)
    radius = min(h, w) * torch.sqrt(fy ** 2 + fx ** 2)
    amp = torch.where(radius > 0, radius.clamp(min=1e-9) ** (-beta / 2), torch.zeros_like(radius))
    white = torch.randn(shape, generator=g, dtype=torch.float64)
    return torch.fft.ifft2(torch.fft.fft2(white, norm="ortho") * amp, norm="ortho").real * 4.0


@pytest.mark.parametrize("shape", [(4, 16, 96, 96), (4, 16, 80, 144)])
def test_calibration_recovers_known_power_law(shape):
    beta = 2.3
    cal = sp.build_calibration(_power_law_field(shape, beta), latent_format="Wan21")
    assert cal["beta"] == pytest.approx(beta, abs=0.08)
    # Real-field FFT power of amp^2 * 16 (white unit power), averaged in bins.
    assert cal["A"] == pytest.approx(16.0, rel=0.25)
    assert cal["r2"] > 0.98
    assert cal["grid"] == shape[-2:] and cal["reference_length"] == min(shape[-2:])
    assert sp.validate_calibration(cal, latent_format="Wan21", channels=16) is cal


def test_calibration_rejects_white_noise_flat_content_and_mismatch():
    with pytest.raises(ValueError, match="beta"):
        sp.build_calibration(torch.randn(2, 16, 64, 64), latent_format="Wan21")
    with pytest.raises(ValueError, match="non-finite"):
        sp.radial_power_spectrum(torch.full((1, 16, 64, 64), float("nan")))
    with pytest.raises(ValueError, match="too small"):
        sp.radial_power_spectrum(torch.randn(1, 16, 8, 8))
    cal = sp.build_calibration(_power_law_field((2, 16, 64, 64), 2.0), latent_format="Wan21")
    with pytest.raises(ValueError, match="latent format"):
        sp.validate_calibration(cal, latent_format="Flux")
    with pytest.raises(ValueError, match="channel"):
        sp.validate_calibration(cal, channels=4)
    for key, value in (("r2", 0.5), ("A", float("inf")), ("version", 0), ("type", "x")):
        bad = dict(cal, **{key: value})
        with pytest.raises(ValueError):
            sp.validate_calibration(bad)
