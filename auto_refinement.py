"""Automatic restart selection for completed Krea 2 trajectory handoffs."""

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class RefinementRestart:
    index: int
    reason: str
    edge_power: float | None = None
    roundtrip_power: float | None = None
    interior_power: float | None = None
    detail_sigma: float | None = None
    structure_sigma: float | None = None
    retimed_sigmas: tuple[float, float, float] | None = None


def _crossing(power: float, noise_scale: float) -> float:
    # (1 - sigma)^2 P = sigma^2 * noise_scale^2 in CONST model space.
    rms = math.sqrt(power)
    return rms / (noise_scale + rms)


def _noise_odds(sigma: float) -> float:
    return sigma / (1.0 - sigma)


def _retime_two_steps(detail: float, structure: float, native_start: float,
                      native_last: float, sigma_min: float, dtype: torch.dtype):
    """Move the two positive knots together in half-log-SNR."""
    if not 0.0 < sigma_min <= native_last < native_start < 1.0 or not 0.0 < detail <= structure < native_start:
        return None
    start_odds = _noise_odds(native_start)
    interval = start_odds / _noise_odds(native_last)
    # Equal distance to the two measured crossings in log noise odds.
    middle = math.sqrt(_noise_odds(detail)) * math.sqrt(_noise_odds(structure))
    # Ensure even the final model evaluation stays within model support.
    first = max(middle, interval * _noise_odds(sigma_min))
    second = first / interval
    values = torch.tensor([first / (1 + first), second / (1 + second), 0.0],
                          device="cpu", dtype=dtype).tolist()
    if not sigma_min <= values[1] < values[0] < native_start or not detail <= values[0] <= structure:
        return None
    # An audited SA noise matcher uses a 1e-6 absolute tolerance. Reject a
    # collapsed or near-coincident interval after conversion to sampler dtype.
    if math.isclose(values[0], values[1], rel_tol=1e-5, abs_tol=1e-6):
        return None
    return tuple(values)


@torch.no_grad()
def choose_refinement_restart(
    schedule: Sequence[float], release: int, source: torch.Tensor,
    roundtrip: torch.Tensor, target_hw: tuple[int, int], noise_scale: float,
    *, retime_sigma_min: float | None = None, schedule_dtype: torch.dtype = torch.float32,
) -> RefinementRestart:
    """Select a native suffix from the actual source and resized handoff.

    Both tensors are in internal model space on the source grid. The second
    is the real enlarged start reduced back to that grid. Use one schedule
    for a batch, respecting the most demanding detail and weakest interior
    band across the batch. The measured adjacent octave is a proxy for the
    unknown finer content, with no extrapolated power law.

    A model minimum sigma opts into shifting a coarse two-step tail when its
    first knot is above the measured structure crossing.
    """
    tail = schedule[release:]
    if (len(tail) < 2 or tail[-1] != 0.0
            or any(not (0.0 < b < a < 1.0) for a, b in zip(tail, tail[1:-1]))
            or not 0.0 < tail[-2] < 1.0):
        return RefinementRestart(release, "invalid_native_suffix")
    # A terminal-to-zero update alone has no nonterminal refinement step.
    # Retain at least one of each, or use the original suffix if it is shorter.
    if len(tail) == 2 or (len(tail) == 3 and retime_sigma_min is None):
        return RefinementRestart(release, "no_shorter_two_step_suffix")
    if not math.isfinite(noise_scale) or noise_scale <= 0.0:
        return RefinementRestart(release, "invalid_noise_scale")
    if source.ndim == 5 and roundtrip.ndim == 5 and source.shape[2] == roundtrip.shape[2] == 1:
        source, roundtrip = source.squeeze(2), roundtrip.squeeze(2)
    if (source.ndim != 4 or source.shape != roundtrip.shape or source.numel() == 0
            or not source.is_floating_point() or not roundtrip.is_floating_point()):
        return RefinementRestart(release, "unsupported_latent_geometry")

    sh, sw = source.shape[-2:]
    th, tw = target_hw
    if th < sh or tw < sw or (th, tw) == (sh, sw):
        return RefinementRestart(release, "invalid_stage_geometry")

    # Full FFT avoids half-spectrum weights. Use actual per-axis source
    # Nyquists and omit their boundary bins (ambiguous on the finer grid).
    # The edge octave only considers axes that gain frequencies.
    y = torch.fft.fftfreq(sh, device=source.device).abs().mul_(2)[:, None]
    x = torch.fft.fftfreq(sw, device=source.device).abs().mul_(2)[None, :]
    shared = (y < 1) & (x < 1) & ((y > 0) | (x > 0))
    interior = shared & (y < 0.5) & (x < 0.5)
    edge = shared & (((th > sh) & (y >= 0.5)) | ((tw > sw) & (x >= 0.5)))

    def power_in_bands(latent, *bands):
        latent = latent.float()
        centered = latent - latent.mean(dim=(-2, -1), keepdim=True)
        power = torch.fft.fft2(centered, norm="ortho").abs().square_().mean(dim=1)
        return [(power * band).sum(dim=(-2, -1)) / band.sum() for band in bands]

    edge_power, interior_power = power_in_bands(source, edge, interior)
    (roundtrip_power,) = power_in_bands(source.float() - roundtrip.float(), shared)
    # For the same image-domain Fourier amplitude, orthonormal coefficient
    # power is proportional to the number of spatial samples. White noise
    # retains unit coefficient power on both grids.
    area_ratio = (th * tw) / (sh * sw)
    p_edge, p_roundtrip, p_interior = (
        torch.stack((edge_power.max(), roundtrip_power.max(), interior_power.min())) * area_ratio
    ).tolist()
    if any(not math.isfinite(p) or p < 0.0 for p in (p_edge, p_roundtrip, p_interior)):
        return RefinementRestart(release, "invalid_spectral_measurement")
    if p_edge <= 0.0 or p_interior <= 0.0:
        return RefinementRestart(release, "insufficient_spectral_evidence", p_edge, p_roundtrip, p_interior)

    detail_sigma = _crossing(max(p_edge, p_roundtrip), noise_scale)
    structure_sigma = _crossing(p_interior, noise_scale)
    two_step_index = len(schedule) - 3
    if retime_sigma_min is not None:
        floor_sigma = schedule[two_step_index]
        if detail_sigma <= structure_sigma < floor_sigma:
            retimed = _retime_two_steps(
                detail_sigma, structure_sigma, floor_sigma, schedule[-2], retime_sigma_min, schedule_dtype,
            )
            if retimed is not None:
                return RefinementRestart(
                    two_step_index, "retimed_two_step_suffix", p_edge, p_roundtrip, p_interior,
                    detail_sigma, structure_sigma, retimed,
                )
    # Round toward more refinement when the native schedule is coarse. The
    # retained interior must still have at least unit SNR at that point.
    index = release
    for candidate in range(release + 1, len(schedule) - 2):
        if schedule[candidate] < detail_sigma:
            break
        index = candidate
    if len(tail) == 3:
        reason = "no_safe_two_step_retime" if structure_sigma < tail[0] else "native_two_step_suffix"
    elif index == release:
        reason = "detail_needs_legacy_suffix"
    elif schedule[index] > structure_sigma:
        index, reason = release, "no_native_step_separates_bands"
    else:
        reason = "reduced"
    return RefinementRestart(index, reason, p_edge, p_roundtrip, p_interior, detail_sigma, structure_sigma)
