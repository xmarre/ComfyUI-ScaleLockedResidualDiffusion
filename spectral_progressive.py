"""Spectral progressive-resolution primitives for flow-matching latents.

Independent implementation of the mechanisms described in *Spectral
Progressive Diffusion for Efficient Image and Video Generation* (Xiao, Chao,
Yariv, Wetzstein, 2026; arXiv:2605.18736). Equation numbers refer to the
paper as cited by its official MIT-licensed reference implementation
(https://github.com/howardhx/speed). No code is copied from either that
implementation or any other port.

Flow-matching state, with ComfyUI's ``CONST`` noise scaling ``s``::

    x_t = (1 - t) * x0 + t * s * eps

Spectral noise expansion (orthonormal 2-D DCT-II). The coefficients of a
``h x w`` state are placed in the low-frequency corner of an ``H x W``
coefficient array; every new coefficient receives fresh ``N(0, (t * s)^2)``
noise. Orthonormal transforms map white noise to white noise, so the noise
component of the expanded state is exactly white with variance ``(t * s)^2``.
A low-resolution image with the same pixel amplitude as the high-resolution
one has orthonormal coefficients ``1 / r`` times smaller, with
``r = sqrt(H * W / (h * w))``, so the expanded *signal* is attenuated by
``r``. Rescaling by ``kappa = r / (1 + (r - 1) t)`` (Eq. 5) and moving to
``t~ = kappa * t`` (Eq. 6) restores the flow form exactly::

    kappa * expand(x_t) = (1 - t~) * x0_low_band + t~ * s * eps~

Only the missing high-band signal ``(1 - t~) * x0_high_band`` is an
approximation; the transition schedule keeps it below a tolerance.

Transition scheduling (Eqs. 9-10): with a radial power-law spectrum
``P(w) = A * w^-beta`` of clean latents, frequency ``w`` is noise dominated
until ``t_w = 1 / (1 + sqrt(delta / (P (1 + P - delta))))``. A stage whose grid
cannot represent frequencies above ``w_i`` hands over at the first schedule
sigma at or below ``t_{w_i}``.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch


CALIBRATION_TYPE = "SLRD_SPECTRUM"
CALIBRATION_VERSION = 1
# Linear grid-size ratio range within which a calibration may be transferred
# to another canvas by the per-coefficient power scaling below.
CALIBRATION_MAX_TRANSFER = 2.0
MIN_FIT_R2 = 0.90
MIN_FIT_BINS = 8
BETA_RANGE = (0.5, 5.0)


# ---------------------------------------------------------------------------
# Orthonormal separable DCT-II.
# ---------------------------------------------------------------------------

def dct_matrix(n: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II basis ``C`` with ``X = C @ x`` and ``x = C.T @ X``.

    Built per call in float64 and cast; a separable ``n x n`` basis per axis
    (512 x 512 for a 4096-pixel Krea 2 axis) is negligible next to the DiT.
    """
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("DCT length must be a positive integer")
    n = int(n)
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    j = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    basis = torch.cos(math.pi * (2.0 * j + 1.0) * k / (2.0 * n)) * math.sqrt(2.0 / n)
    basis[0] *= 1.0 / math.sqrt(2.0)
    return basis.to(device=device, dtype=dtype)


def dct2(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2-D DCT-II over the last two axes (float32 or wider)."""
    h, w = x.shape[-2:]
    ch = dct_matrix(h, device=x.device, dtype=x.dtype)
    cw = dct_matrix(w, device=x.device, dtype=x.dtype)
    return ch @ x @ cw.transpose(0, 1)


def idct2(coeffs: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`dct2`."""
    h, w = coeffs.shape[-2:]
    ch = dct_matrix(h, device=coeffs.device, dtype=coeffs.dtype)
    cw = dct_matrix(w, device=coeffs.device, dtype=coeffs.dtype)
    return ch.transpose(0, 1) @ coeffs @ cw


# ---------------------------------------------------------------------------
# Expansion and time alignment (Eqs. 5-6).
# ---------------------------------------------------------------------------

def _check_unit_interval(name: str, value, *, closed_low=True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not ((0.0 <= value) if closed_low else (0.0 < value)) or value >= 1.0:
        raise ValueError(f"{name} must lie in {'[0' if closed_low else '(0'}, 1)")
    return value


def area_ratio(src_hw, dst_hw) -> float:
    """Effective linear ratio ``r`` of an orthonormal expansion.

    The low-band signal attenuation is ``sqrt(h * w / (H * W))`` for the
    actual integer grids, also for non-uniform or noninteger per-axis ratios.
    """
    (sh, sw), (dh, dw) = (tuple(map(int, src_hw)), tuple(map(int, dst_hw)))
    if min(sh, sw, dh, dw) < 1 or dh < sh or dw < sw or (dh, dw) == (sh, sw):
        raise ValueError("spectral expansion requires a target grid that grows and does not shrink")
    return math.sqrt((dh * dw) / (sh * sw))


def kappa(sigma: float, r: float) -> float:
    """State rescaling after expansion by ratio ``r`` at flow time ``sigma`` (Eq. 5)."""
    sigma = _check_unit_interval("sigma", sigma)
    if not math.isfinite(r) or r < 1.0:
        raise ValueError("expansion ratio must be finite and at least 1")
    return r / (1.0 + (r - 1.0) * sigma)


def align_sigma(sigma: float, r: float) -> float:
    """Flow time of the expanded state (Eq. 6): ``r t / (1 + (r - 1) t)``."""
    return float(sigma) * kappa(sigma, r)


def expand_flow_state(
    x: torch.Tensor,
    target_hw,
    sigma: float,
    *,
    seed: int,
    noise_scale: float = 1.0,
    batch_inds=None,
) -> tuple[torch.Tensor, float]:
    """Spectrally expand a noisy flow state ``x`` (internal model space).

    ``x`` is ``B x C x h x w`` or a single-frame ``B x C x 1 x h x w``. The
    fresh high-frequency coefficients come from a CPU generator seeded with
    ``seed`` (device independent, reproducible). If ``batch_inds`` is supplied,
    repeat indices receive identical high-frequency coefficients, following
    ComfyUI's initial-noise batch-index semantics. Returns the expanded state
    in ``x.dtype`` on ``x.device`` and the aligned sigma. The computation runs
    in float32 (float64 for float64 input); no tensor larger than three target-sized latents is held.
    """
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("flow state must be a floating tensor")
    if x.ndim == 5:
        if x.shape[2] != 1:
            raise ValueError("spectral expansion supports single-frame image latents only")
        out, aligned = expand_flow_state(
            x.squeeze(2), target_hw, sigma, seed=seed,
            noise_scale=noise_scale, batch_inds=batch_inds,
        )
        return out.unsqueeze(2), aligned
    if x.ndim != 4:
        raise ValueError("flow state must be BxCxHxW or Bx Cx1xHxW")
    sigma = _check_unit_interval("transition sigma", sigma, closed_low=False)
    if not math.isfinite(noise_scale) or noise_scale <= 0.0:
        raise ValueError("noise_scale must be positive")
    sh, sw = map(int, x.shape[-2:])
    th, tw = map(int, target_hw)
    r = area_ratio((sh, sw), (th, tw))
    k = kappa(sigma, r)
    work = torch.float64 if x.dtype == torch.float64 else torch.float32
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    coeff_shape = tuple(x.shape[:2]) + (th, tw)
    if batch_inds is None:
        coeffs_cpu = torch.randn(
            coeff_shape, generator=generator, device="cpu", dtype=torch.float32
        )
    else:
        # Match comfy.sample.prepare_noise_inner: each noise index denotes one
        # reproducible draw, even when indices are repeated or noncontiguous.
        indices = [int(i) for i in batch_inds]
        if len(indices) != x.shape[0] or any(i < 0 for i in indices):
            raise ValueError("batch_inds must contain one nonnegative index per batch row")
        selected = set(indices)
        by_index = {}
        for i in range(max(indices) + 1):
            draw = torch.randn(
                (1,) + coeff_shape[1:], generator=generator,
                device="cpu", dtype=torch.float32,
            )
            if i in selected:
                by_index[i] = draw
        coeffs_cpu = torch.cat([by_index[i] for i in indices], dim=0)
    coeffs = coeffs_cpu.to(device=x.device, dtype=work)
    coeffs.mul_(sigma * float(noise_scale))
    coeffs[..., :sh, :sw] = dct2(x.to(dtype=work))
    out = idct2(coeffs)
    del coeffs
    out.mul_(k)
    return out.to(dtype=x.dtype), sigma * k


# ---------------------------------------------------------------------------
# Spectrum, activation and transition planning (Eqs. 9-10).
# ---------------------------------------------------------------------------

def activation_sigma(power: float, delta: float) -> float:
    """Flow time at which a frequency of clean-latent power ``power`` stops
    being noise dominated for tolerance ``delta`` (Eq. 9)."""
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("spectral power must be positive and finite")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) \
            or not 0.0 < float(delta) < 1.0:
        raise ValueError("spectral tolerance delta must lie in (0, 1)")
    return 1.0 / (1.0 + math.sqrt(float(delta) / (power * (1.0 + power - float(delta)))))


def reference_length(hw) -> int:
    return int(min(map(int, hw)))


def calibrated_amplitude(calibration: dict, grid_hw) -> float:
    """Power-law amplitude transferred to a grid of a different size.

    Frequencies are measured in cycles per image normalized to the shorter
    side ``L`` (Nyquist ``L / 2``). For fixed content, orthonormal
    per-coefficient power at a fixed image frequency scales with ``L^2``
    (Parseval: the per-pixel latent variance does not depend on resolution),
    so ``A`` scales by ``(L / L_cal)^2``. Transfer is restricted to
    ``CALIBRATION_MAX_TRANSFER``; beyond it, recalibrate.
    """
    ratio = reference_length(grid_hw) / float(calibration["reference_length"])
    if not (1.0 / CALIBRATION_MAX_TRANSFER - 1e-9 <= ratio <= CALIBRATION_MAX_TRANSFER + 1e-9):
        raise ValueError(
            f"spectral calibration measured on a {calibration['grid'][1] * 8}x{calibration['grid'][0] * 8} "
            f"canvas cannot be transferred to {int(grid_hw[1]) * 8}x{int(grid_hw[0]) * 8} "
            f"(shorter-side ratio {ratio:.3g} outside [{1 / CALIBRATION_MAX_TRANSFER:.3g}, "
            f"{CALIBRATION_MAX_TRANSFER:.3g}]); calibrate on latents near the output resolution"
        )
    return float(calibration["A"]) * ratio * ratio


def cutoff_frequency(src_hw, dst_hw) -> float:
    """Lowest frequency on ``dst_hw`` that grid ``src_hw`` cannot represent,
    in the normalized units of :func:`radial_power_spectrum`."""
    (sh, sw), (dh, dw) = (tuple(map(int, src_hw)), tuple(map(int, dst_hw)))
    return 0.5 * reference_length((dh, dw)) * min(sh / dh, sw / dw)


def handoff_activation_sigma(src_hw, dst_hw, calibration: dict, delta: float) -> float:
    """Activation time, on the ``dst_hw`` clock, of the first frequency that
    the source grid cannot represent."""
    amplitude = calibrated_amplitude(calibration, dst_hw)
    omega = cutoff_frequency(src_hw, dst_hw)
    return activation_sigma(amplitude * omega ** (-float(calibration["beta"])), delta)


def spectral_transition_sigmas(sizes: Sequence, calibration: dict, delta: float) -> list[float]:
    """Transition thresholds for progressive stages ``sizes`` (first to final).

    As in the official reference implementation, every threshold is the
    activation time on the final grid of the first frequency above stage
    ``i``'s Nyquist limit, compared against the schedule sigma of stage ``i``.
    Under the same power model this clock is conservative: a stage at linear
    scale ``s`` reaches the same full-resolution noise level only at the lower
    stage sigma ``align_sigma^-1``, so these thresholds hand over earlier,
    never later, than a full-resolution-equivalent criterion.
    """
    sizes = [tuple(map(int, hw)) for hw in sizes]
    final = sizes[-1]
    return [handoff_activation_sigma(hw, final, calibration, delta) for hw in sizes[:-1]]


def transition_steps(schedule: Sequence[float], thresholds: Sequence[float], sizes: Sequence) -> list[int]:
    """Step index of each transition: the first schedule sigma at or below its threshold.

    Every stage, including the final resolution, must receive at least one
    step; a threshold that would skip a stage or never be reached raises
    instead of silently reshaping the requested schedule.
    """
    values = [float(v) for v in schedule]
    steps = len(values) - 1
    if steps < len(sizes):
        raise ValueError(
            f"spectral_progressive needs at least one step per resolution ({len(sizes)}); raise steps"
        )
    if len(thresholds) != len(sizes) - 1:
        raise ValueError(
            f"spectral_progressive needs {len(sizes) - 1} transition sigma(s) for "
            f"{len(sizes)} resolutions, got {len(thresholds)}"
        )
    indices = []
    previous = 0
    for i, threshold in enumerate(thresholds):
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("transition sigmas must lie strictly between 0 and 1")
        index = next((k for k in range(steps) if values[k] <= threshold), steps)
        label = f"transition {i + 1} ({sizes[i][1] * 8}x{sizes[i][0] * 8} -> {sizes[i + 1][1] * 8}x{sizes[i + 1][0] * 8})"
        if index <= previous:
            raise ValueError(
                f"{label} at sigma {threshold:.4f} leaves no step at the lower resolution "
                f"(schedule starts at {values[previous]:.4f}); lower the transition sigma, "
                "remove a stage or raise steps"
            )
        if index >= steps:
            raise ValueError(
                f"{label} at sigma {threshold:.4f} is never reached before the last step; "
                "raise the transition sigma or steps"
            )
        indices.append(index)
        previous = index
    return indices


def parse_transition_sigmas(text: str) -> list[float]:
    if not isinstance(text, str):
        raise ValueError("progressive_transition_sigmas must be text")
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("progressive_transition_sigmas must be comma-separated numbers, e.g. '0.85, 0.6'") from exc
    if any(not math.isfinite(v) or not 0.0 < v < 1.0 for v in values):
        raise ValueError("progressive_transition_sigmas must lie strictly between 0 and 1")
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("progressive_transition_sigmas must strictly decrease")
    return values


# ---------------------------------------------------------------------------
# Calibration from clean latents (no model evaluations).
# ---------------------------------------------------------------------------

def radial_power_spectrum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Radially binned per-coefficient power of clean latents.

    ``x`` is ``B x C x H x W`` (or single-frame 5-D) in the model's internal
    latent space, where sampler noise has unit variance per element. Uses an
    orthonormal 2-D FFT, so unit white noise has unit power at every
    frequency. Frequency ``w = L * |f|`` with ``f`` in cycles per latent pixel
    and ``L`` the shorter side, i.e. cycles per image along the shorter side,
    Nyquist ``L / 2``. Returns bin centres (``1, 2, ...``), the mean power per
    bin averaged over batch and channels, and the coefficient count per bin.
    DC is excluded.
    """
    if x.ndim == 5:
        if x.shape[2] != 1:
            raise ValueError("calibration supports single-frame image latents only")
        x = x.squeeze(2)
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("calibration latents must be floating BxCxHxW tensors")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("calibration latents contain non-finite values")
    h, w = map(int, x.shape[-2:])
    if min(h, w) < 16:
        raise ValueError("calibration latents are too small for a spectral fit")
    spectrum = torch.fft.fft2(x.to(torch.float64), norm="ortho").abs().square().mean(dim=(0, 1))
    length = min(h, w)
    fy = torch.fft.fftfreq(h, dtype=torch.float64).unsqueeze(1)
    fx = torch.fft.fftfreq(w, dtype=torch.float64).unsqueeze(0)
    radius = length * torch.sqrt(fy * fy + fx * fx)
    bins = torch.round(radius).to(torch.long).flatten()
    power = spectrum.flatten().cpu()
    bins = bins.cpu()
    keep = (bins >= 1) & (bins <= length // 2)
    bins, power = bins[keep], power[keep]
    count = torch.bincount(bins, minlength=length // 2 + 1)[1:]
    total = torch.bincount(bins, weights=power, minlength=length // 2 + 1)[1:]
    centres = torch.arange(1, length // 2 + 1, dtype=torch.float64)
    valid = count > 0
    return centres[valid], (total[valid] / count[valid]), count[valid]


def fit_power_law(
    omega: torch.Tensor, power: torch.Tensor, *, lo: float, hi: float,
) -> tuple[float, float, float, int]:
    """Least-squares fit of ``log P = log A - beta log w`` over ``lo <= w <= hi``.

    Returns ``(A, beta, r2, bins)``.
    """
    sel = (omega >= lo) & (omega <= hi) & (power > 0)
    n = int(sel.sum())
    if n < MIN_FIT_BINS:
        raise ValueError(f"spectral fit band contains {n} frequency bins; at least {MIN_FIT_BINS} are required")
    lx = torch.log(omega[sel].to(torch.float64))
    ly = torch.log(power[sel].to(torch.float64))
    mx, my = lx.mean(), ly.mean()
    sxx = ((lx - mx) ** 2).sum()
    slope = float(((lx - mx) * (ly - my)).sum() / sxx)
    intercept = float(my - slope * mx)
    residual = ly - (intercept + slope * lx)
    sst = float(((ly - my) ** 2).sum())
    r2 = 1.0 - float((residual ** 2).sum()) / sst if sst > 0 else 0.0
    return math.exp(intercept), -slope, r2, n


def build_calibration(
    internal_latents: torch.Tensor, *, latent_format: str, fit_low: float = 0.05, fit_high: float = 0.9,
) -> dict:
    """Measure the spectrum of clean internal-space latents and validate the fit."""
    if not (0.0 < fit_low < fit_high <= 1.0):
        raise ValueError("fit band must satisfy 0 < fit_low < fit_high <= 1 (fractions of Nyquist)")
    x = internal_latents.squeeze(2) if internal_latents.ndim == 5 else internal_latents
    omega, power, count = radial_power_spectrum(x)
    h, w = map(int, x.shape[-2:])
    nyquist = min(h, w) / 2.0
    amplitude, beta, r2, bins = fit_power_law(omega, power, lo=fit_low * nyquist, hi=fit_high * nyquist)
    calibration = {
        "type": CALIBRATION_TYPE,
        "version": CALIBRATION_VERSION,
        "A": amplitude,
        "beta": beta,
        "r2": r2,
        "bins": bins,
        "grid": (h, w),
        "reference_length": min(h, w),
        "channels": int(x.shape[1]),
        "samples": int(x.shape[0]),
        "latent_format": str(latent_format),
        "fit_band": (float(fit_low), float(fit_high)),
    }
    validate_calibration(calibration)
    return calibration


def validate_calibration(calibration, *, latent_format: Optional[str] = None, channels: Optional[int] = None) -> dict:
    """Reject malformed, poorly fitted or mismatched calibrations."""
    if not isinstance(calibration, dict) or calibration.get("type") != CALIBRATION_TYPE:
        raise ValueError("spectral_calibration must come from the Krea 2 Spectral Calibration node")
    if calibration.get("version") != CALIBRATION_VERSION:
        raise ValueError("spectral_calibration was produced by an incompatible version; recalibrate")
    a, beta, r2 = calibration.get("A"), calibration.get("beta"), calibration.get("r2")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (a, beta, r2)):
        raise ValueError("spectral_calibration contains non-finite fit values")
    if a <= 0.0:
        raise ValueError("spectral_calibration amplitude must be positive")
    if not BETA_RANGE[0] <= beta <= BETA_RANGE[1]:
        raise ValueError(
            f"spectral_calibration exponent beta={beta:.3g} is outside the plausible range "
            f"[{BETA_RANGE[0]}, {BETA_RANGE[1]}]; the latents do not follow a power law"
        )
    if r2 < MIN_FIT_R2:
        raise ValueError(
            f"spectral_calibration power-law fit is poor (R^2={r2:.3f} < {MIN_FIT_R2}); "
            "use clean, completed latents of natural content"
        )
    grid = calibration.get("grid")
    if (not isinstance(grid, (tuple, list)) or len(grid) != 2 or any(int(v) < 16 for v in grid)
            or calibration.get("reference_length") != min(map(int, grid))):
        raise ValueError("spectral_calibration grid metadata is invalid")
    if latent_format is not None and calibration.get("latent_format") != latent_format:
        raise ValueError(
            f"spectral_calibration was measured for latent format {calibration.get('latent_format')!r}, "
            f"not {latent_format!r}; recalibrate with this model"
        )
    if channels is not None and calibration.get("channels") != int(channels):
        raise ValueError("spectral_calibration channel count does not match the model latents")
    return calibration


def describe_calibration(calibration: dict) -> str:
    h, w = calibration["grid"]
    return (
        f"P(w) = {calibration['A']:.6g} * w^-{calibration['beta']:.4f} "
        f"(R^2 {calibration['r2']:.4f}, {calibration['bins']} bins, "
        f"{calibration['samples']} latent(s) at {w * 8}x{h * 8}, "
        f"band {calibration['fit_band'][0]:.3g}-{calibration['fit_band'][1]:.3g} of Nyquist, "
        f"latent format {calibration['latent_format']})"
    )
