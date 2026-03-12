from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


def _sigma_scalar(timestep: torch.Tensor | float | int) -> float:
    if isinstance(timestep, (float, int)):
        return float(timestep)
    if timestep.numel() == 0:
        return 0.0
    return float(timestep.detach().flatten()[0].cpu().item())


def clone_latent(latent: dict) -> dict:
    return dict(latent)


def _stage_cpu_tensor(x: torch.Tensor, dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    y = x.detach().to(device="cpu", dtype=dtype).contiguous().clone()
    if pin_memory:
        try:
            y = y.pin_memory()
        except Exception:
            pass
    return y


def _round_to_multiple(v: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, int(v))
    return max(multiple, int(round(v / multiple) * multiple))


def resize_4d_tensor_nearest(x: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size_hw):
        return x
    return F.interpolate(x, size=size_hw, mode="nearest")


def _interp_mode(src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> str:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    return "area" if (dst_h < src_h or dst_w < src_w) else "bilinear"


def resize_4d_tensor(x: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """Resize BCHW tensor with sane defaults for latent tensors."""
    if tuple(x.shape[-2:]) == tuple(size_hw):
        return x
    mode = _interp_mode(tuple(x.shape[-2:]), tuple(size_hw))
    if mode == "area":
        return F.interpolate(x, size=size_hw, mode=mode)
    return F.interpolate(x, size=size_hw, mode=mode, align_corners=False)


def resize_mask(mask: Optional[torch.Tensor], size_hw: tuple[int, int], batch: int, channels: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None

    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.ndim == 3:
        m = m.unsqueeze(1)
    elif m.ndim == 4 and m.shape[1] != 1:
        # Unexpected format: collapse to a single channel mask.
        m = m.mean(dim=1, keepdim=True)

    m = m.float()
    mode = _interp_mode(tuple(m.shape[-2:]), tuple(size_hw))
    if mode == "area":
        m = F.interpolate(m, size=size_hw, mode=mode)
    else:
        m = F.interpolate(m, size=size_hw, mode=mode, align_corners=False)

    if m.shape[0] < batch:
        repeat = math.ceil(batch / max(1, m.shape[0]))
        m = m.repeat(repeat, 1, 1, 1)[:batch]
    elif m.shape[0] > batch:
        m = m[:batch]

    m = m.expand(batch, channels, size_hw[0], size_hw[1]).contiguous()
    return m


def latent_target_hw_from_megapixels(
    latent_bchw: torch.Tensor,
    target_megapixels: float,
    latent_multiple: int = 2,
) -> tuple[int, int]:
    """
    Compute a target latent spatial size that preserves aspect ratio while targeting a pixel-space megapixel count.
    Latent tensors are expected to be 8x downsampled in H/W for standard image models used by ComfyUI.
    """
    _, _, h_lat, w_lat = latent_bchw.shape
    h_px = h_lat * 8
    w_px = w_lat * 8
    aspect = w_px / max(1.0, float(h_px))
    target_px_area = max(target_megapixels, 0.01) * 1_000_000.0

    h_target_px = math.sqrt(target_px_area / max(aspect, 1e-8))
    w_target_px = h_target_px * aspect

    # Keep latent compatibility and avoid absurdly small sizes.
    h_target_lat = _round_to_multiple(int(round(h_target_px / 8.0)), latent_multiple)
    w_target_lat = _round_to_multiple(int(round(w_target_px / 8.0)), latent_multiple)

    return (h_target_lat, w_target_lat)


def resize_latent_dict(latent: dict, size_hw: tuple[int, int]) -> dict:
    out = clone_latent(latent)
    out["samples"] = resize_4d_tensor(latent["samples"], size_hw)
    if "noise_mask" in latent:
        nm = resize_mask(latent.get("noise_mask"), size_hw, out["samples"].shape[0], 1)
        if nm is not None:
            out["noise_mask"] = nm[:, :1, :, :].squeeze(1)
    return out


def build_nested_noise(
    lowres_noise: torch.Tensor,
    target_shape: tuple[int, int, int, int],
    seed: int,
    hf_strength: float,
) -> torch.Tensor:
    """
    Lift low-res noise to high-res while keeping the coarse field aligned.

    The implementation uses:
      base = nearest-upsampled low-res noise
      hf   = random noise with its low-frequency projection removed
    so that the high-res branch shares the same coarse stochastic layout but can still invent fine detail.
    """
    batch, channels, target_h, target_w = target_shape
    device = lowres_noise.device
    dtype = lowres_noise.dtype

    base = resize_4d_tensor_nearest(lowres_noise.float(), (target_h, target_w))
    if hf_strength <= 0.0:
        return base.to(device=device, dtype=dtype)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    hf = torch.randn((batch, channels, target_h, target_w), generator=gen, dtype=torch.float32, device="cpu")

    low_hw = tuple(lowres_noise.shape[-2:])
    hf_low = resize_4d_tensor(hf, low_hw)
    hf_low_up = resize_4d_tensor_nearest(hf_low, (target_h, target_w))
    hf = hf - hf_low_up

    std = hf.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    hf = hf / std

    out = base + float(hf_strength) * hf
    out_std = out.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    out = out / out_std

    return out.to(device=device, dtype=dtype)


def lowpass_latent(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Cheap, sampler-friendly low-pass filter implemented as area downsample + bilinear upsample.

    cutoff is a retained-resolution fraction in (0, 1].
    Example: cutoff=0.33 means keep roughly one third of the spatial resolution for the coarse field.
    """
    cutoff = float(max(0.01, min(1.0, cutoff)))
    h, w = x.shape[-2:]
    if cutoff >= 0.999:
        return x
    coarse_h = max(1, int(round(h * cutoff)))
    coarse_w = max(1, int(round(w * cutoff)))
    y = resize_4d_tensor(x, (coarse_h, coarse_w))
    y = resize_4d_tensor(y, (h, w))
    return y


def residual_lock_multiband(
    base_denoised: torch.Tensor,
    anchor_denoised: torch.Tensor,
    low_strength: float,
    mid_strength: float,
    low_cutoff: float,
    mid_cutoff: float,
) -> torch.Tensor:
    """
    Preserve the highest-frequency residual while independently blending:
      - a low / coarse band
      - a mid-frequency structural band
    toward the planner anchor.
    """
    low_strength = float(max(0.0, min(1.0, low_strength)))
    mid_strength = float(max(0.0, min(1.0, mid_strength)))
    mid_cutoff = float(max(low_cutoff, mid_cutoff))

    base_mid_lp = lowpass_latent(base_denoised, mid_cutoff)
    anchor_mid_lp = lowpass_latent(anchor_denoised, mid_cutoff)
    base_low = lowpass_latent(base_denoised, low_cutoff)
    anchor_low = lowpass_latent(anchor_denoised, low_cutoff)
    base_mid = base_mid_lp - base_low
    anchor_mid = anchor_mid_lp - anchor_low
    base_high = base_denoised - base_mid_lp
    return base_high + torch.lerp(base_low, anchor_low, low_strength) + torch.lerp(base_mid, anchor_mid, mid_strength)


def _base_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1).contiguous()


def warp_4d_tensor(x: torch.Tensor, flow_xy_norm: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
    batch, _, height, width = x.shape
    expected = (batch, height, width, 2)
    if tuple(flow_xy_norm.shape) != expected:
        raise ValueError(f"Expected flow shape {expected}, got {tuple(flow_xy_norm.shape)}")
    base_grid = _base_grid(batch, height, width, x.device, x.dtype)
    sample_grid = (base_grid + flow_xy_norm).clamp(-1.25, 1.25)
    return F.grid_sample(
        x,
        sample_grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )


def _expand_mask_channels(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(f"Expected mask tensor BCHW, got shape {tuple(mask.shape)}")
    if mask.shape[0] < like.shape[0]:
        repeat = math.ceil(like.shape[0] / max(1, mask.shape[0]))
        mask = mask.repeat(repeat, 1, 1, 1)[: like.shape[0]]
    elif mask.shape[0] > like.shape[0]:
        mask = mask[: like.shape[0]]
    if mask.shape[1] == like.shape[1]:
        return mask
    if mask.shape[1] == 1:
        return mask.expand(like.shape[0], like.shape[1], like.shape[-2], like.shape[-1]).contiguous()
    return mask.mean(dim=1, keepdim=True).expand(like.shape[0], like.shape[1], like.shape[-2], like.shape[-1]).contiguous()


def _latent_activity_map(x: torch.Tensor, mask_1ch: torch.Tensor) -> torch.Tensor:
    mass = mask_1ch.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    mean = (x * mask_1ch).sum(dim=(-2, -1), keepdim=True) / mass
    centered = x - mean
    activity = centered.square().mean(dim=1, keepdim=True)
    activity = activity * mask_1ch
    activity = activity + mask_1ch * 1e-8
    return activity


def _masked_spatial_stats(x: torch.Tensor, mask_1ch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, height, width = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    xx = xx.view(1, 1, height, width).expand(batch, -1, -1, -1)
    yy = yy.view(1, 1, height, width).expand(batch, -1, -1, -1)

    activity = _latent_activity_map(x, mask_1ch)
    mass = activity.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    cx = (activity * xx).sum(dim=(-2, -1), keepdim=True) / mass
    cy = (activity * yy).sum(dim=(-2, -1), keepdim=True) / mass

    dx = xx - cx
    dy = yy - cy
    rx = torch.sqrt((activity * dx.square()).sum(dim=(-2, -1), keepdim=True) / mass).clamp_min(1e-4)
    ry = torch.sqrt((activity * dy.square()).sum(dim=(-2, -1), keepdim=True) / mass).clamp_min(1e-4)
    return cx, cy, rx, ry


def estimate_latent_compaction_flow(
    anchor_low: torch.Tensor,
    base_low: torch.Tensor,
    mask_1ch: torch.Tensor,
    strength: float,
    radial_strength: float,
    anisotropy: float,
    translation_strength: float,
    max_shift_px: float,
) -> torch.Tensor:
    batch, _, height, width = base_low.shape
    anchor_cx, anchor_cy, anchor_rx, anchor_ry = _masked_spatial_stats(anchor_low, mask_1ch)
    base_cx, base_cy, base_rx, base_ry = _masked_spatial_stats(base_low, mask_1ch)

    ratio_x = (base_rx / anchor_rx.clamp_min(1e-5)).clamp(0.85, 1.25)
    ratio_y = (base_ry / anchor_ry.clamp_min(1e-5)).clamp(0.85, 1.25)
    outward_x = (ratio_x - 1.0).clamp(min=0.0)
    outward_y = (ratio_y - 1.0).clamp(min=0.0)

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=base_low.device, dtype=base_low.dtype),
        torch.linspace(-1.0, 1.0, width, device=base_low.device, dtype=base_low.dtype),
        indexing="ij",
    )
    xx = xx.view(1, 1, height, width).expand(batch, -1, -1, -1)
    yy = yy.view(1, 1, height, width).expand(batch, -1, -1, -1)

    dx = xx - base_cx
    dy = yy - base_cy
    ex = dx / base_rx.clamp_min(1e-5)
    ey = dy / base_ry.clamp_min(1e-5)
    radius = torch.sqrt(ex.square() + ey.square() + 1e-8)
    edge_envelope = torch.clamp(radius / 1.25, 0.0, 1.0)

    smooth_mask = lowpass_latent(mask_1ch, 0.35).clamp(0.0, 1.0)
    mean_outward = 0.5 * (outward_x + outward_y)
    axis_x = mean_outward * float(radial_strength) + (outward_x - mean_outward) * float(anisotropy)
    axis_y = mean_outward * float(radial_strength) + (outward_y - mean_outward) * float(anisotropy)

    shift_x = ex * edge_envelope * smooth_mask * float(strength) * axis_x
    shift_y = ey * edge_envelope * smooth_mask * float(strength) * axis_y

    trans_x = (base_cx - anchor_cx) * smooth_mask * float(strength) * float(translation_strength)
    trans_y = (base_cy - anchor_cy) * smooth_mask * float(strength) * float(translation_strength)

    max_shift_norm_x = 2.0 * float(max_shift_px) / max(width - 1, 1)
    max_shift_norm_y = 2.0 * float(max_shift_px) / max(height - 1, 1)

    shift_x = (shift_x + trans_x).clamp(-max_shift_norm_x, max_shift_norm_x)
    shift_y = (shift_y + trans_y).clamp(-max_shift_norm_y, max_shift_norm_y)
    return torch.stack([shift_x[:, 0], shift_y[:, 0]], dim=-1)


def _weighted_channel_stats(x: torch.Tensor, mask_1ch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mass = mask_1ch.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    mean = (x * mask_1ch).sum(dim=(-2, -1), keepdim=True) / mass
    var = (((x - mean).square()) * mask_1ch).sum(dim=(-2, -1), keepdim=True) / mass
    std = torch.sqrt(var.clamp_min(1e-8))
    return mean, std


def _weighted_global_std(x: torch.Tensor, mask_1ch: torch.Tensor, mean: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mean is None:
        mean, _ = _weighted_channel_stats(x, mask_1ch)
    mass = mask_1ch.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    denom = mass * float(max(1, x.shape[1]))
    var = (((x - mean).square()) * mask_1ch).sum(dim=(1, 2, 3), keepdim=True) / denom
    return torch.sqrt(var.clamp_min(1e-8))


def _bounded_ratio(target: torch.Tensor, source: torch.Tensor, gain_cap: float) -> torch.Tensor:
    gain_cap = float(max(1.0, gain_cap))
    lo = 1.0 / gain_cap
    hi = gain_cap
    return (target / source.clamp_min(1e-5)).clamp(lo, hi)


def tether_latent_low_frequency_energy(
    base_low: torch.Tensor,
    anchor_low: torch.Tensor,
    mask_1ch: torch.Tensor,
    energy_tether: float,
    channel_tether: float,
    gain_cap: float,
) -> torch.Tensor:
    energy_tether = float(max(0.0, min(1.0, energy_tether)))
    channel_tether = float(max(0.0, min(1.0, channel_tether)))
    if energy_tether <= 0.0 and channel_tether <= 0.0:
        return base_low

    base_mean, _ = _weighted_channel_stats(base_low, mask_1ch)
    anchor_mean, anchor_std = _weighted_channel_stats(anchor_low, mask_1ch)

    regulated = base_low
    if energy_tether > 0.0:
        base_global_std = _weighted_global_std(regulated, mask_1ch, mean=base_mean)
        anchor_global_std = _weighted_global_std(anchor_low, mask_1ch, mean=anchor_mean)
        global_gain = _bounded_ratio(anchor_global_std, base_global_std, gain_cap)
        global_gain = torch.lerp(torch.ones_like(global_gain), global_gain, energy_tether)
        regulated = base_mean + (regulated - base_mean) * global_gain

    if channel_tether > 0.0:
        regulated_mean, regulated_std = _weighted_channel_stats(regulated, mask_1ch)
        channel_gain = _bounded_ratio(anchor_std, regulated_std, gain_cap)
        channel_gain = torch.lerp(torch.ones_like(channel_gain), channel_gain, channel_tether)
        regulated = regulated_mean + (regulated - regulated_mean) * channel_gain

    return regulated


def restore_latent_low_frequency_stats(
    warped_low: torch.Tensor,
    anchor_low: torch.Tensor,
    mask_1ch: torch.Tensor,
    anchor_mix: float,
    mean_anchor_mix: float,
    contrast_restore: float,
) -> torch.Tensor:
    warped_mean, warped_std = _weighted_channel_stats(warped_low, mask_1ch)
    anchor_mean, anchor_std = _weighted_channel_stats(anchor_low, mask_1ch)

    mean_target = torch.lerp(warped_mean, anchor_mean, float(mean_anchor_mix))
    contrast_gain = torch.lerp(
        torch.ones_like(anchor_std),
        anchor_std / warped_std.clamp_min(1e-5),
        float(contrast_restore),
    )
    restored = mean_target + (warped_low - warped_mean) * contrast_gain
    return torch.lerp(restored, anchor_low, float(anchor_mix))


def latent_manifold_compand(
    base_denoised: torch.Tensor,
    anchor_denoised: torch.Tensor,
    mask: Optional[torch.Tensor],
    strength: float,
    cutoff: float,
    radial_strength: float,
    anisotropy: float,
    translation_strength: float,
    anchor_mix: float,
    mean_anchor_mix: float,
    contrast_restore: float,
    energy_tether: float,
    channel_tether: float,
    energy_gain_cap: float,
    max_shift_px: float,
) -> torch.Tensor:
    strength = float(max(0.0, min(1.0, strength)))
    if strength <= 0.0:
        return base_denoised

    cutoff = float(max(0.05, min(1.0, cutoff)))
    work_dtype = base_denoised.dtype
    compute_dtype = torch.float32

    base = base_denoised.to(dtype=compute_dtype)
    anchor = anchor_denoised.to(device=base.device, dtype=compute_dtype)
    if tuple(anchor.shape[-2:]) != tuple(base.shape[-2:]):
        anchor = resize_4d_tensor(anchor, tuple(base.shape[-2:]))

    if mask is None:
        mask_1ch = torch.ones((base.shape[0], 1, base.shape[-2], base.shape[-1]), device=base.device, dtype=compute_dtype)
    else:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim == 4 and mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)
        mask_1ch = _expand_mask_channels(mask.to(device=base.device, dtype=compute_dtype), base)[:, :1].clamp(0.0, 1.0)

    base_low = lowpass_latent(base, cutoff)
    anchor_low = lowpass_latent(anchor, cutoff)
    base_high = base - base_low

    regulated_low = tether_latent_low_frequency_energy(
        base_low,
        anchor_low,
        mask_1ch,
        energy_tether=float(max(0.0, min(1.0, energy_tether))) * strength,
        channel_tether=float(max(0.0, min(1.0, channel_tether))) * strength,
        gain_cap=float(max(1.0, energy_gain_cap)),
    )

    flow = estimate_latent_compaction_flow(
        anchor_low=anchor_low,
        base_low=regulated_low,
        mask_1ch=mask_1ch,
        strength=strength,
        radial_strength=radial_strength,
        anisotropy=anisotropy,
        translation_strength=translation_strength,
        max_shift_px=max_shift_px,
    )
    warped_low = warp_4d_tensor(regulated_low, flow, mode="bilinear")
    restored_low = restore_latent_low_frequency_stats(
        warped_low,
        anchor_low,
        mask_1ch,
        anchor_mix=float(max(0.0, min(1.0, anchor_mix))),
        mean_anchor_mix=float(max(0.0, min(1.0, mean_anchor_mix))),
        contrast_restore=float(max(0.0, min(1.0, contrast_restore))),
    )
    corrected = base_high + restored_low
    return corrected.to(dtype=work_dtype)


def clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def sigma_progress(planner_sigmas: Optional[Sequence[float]], idx: int) -> Optional[float]:
    if planner_sigmas is None:
        return None

    if len(planner_sigmas) == 0:
        return None

    idx = max(0, min(int(idx), len(planner_sigmas) - 1))
    sigma_hi = max(float(planner_sigmas[0]), 1e-6)
    sigma_lo = max(float(planner_sigmas[-1]), 1e-6)
    sigma = max(float(planner_sigmas[idx]), 1e-6)

    hi = math.log(sigma_hi)
    lo = math.log(sigma_lo)
    denom = hi - lo
    if abs(denom) < 1e-6:
        return None

    cur = math.log(sigma)
    return clamp((hi - cur) / denom, 0.0, 1.0)


def schedule_curve(progress: float, mode: str, power: float = 2.0, hold: float = 0.0) -> float:
    p = clamp(progress, 0.0, 1.0)
    power = max(1e-6, float(power))
    hold = clamp(hold, 0.0, 0.95)

    if mode == "flat":
        return 0.0
    if mode == "linear":
        return p
    if mode == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * p)
    if mode == "smoothstep":
        return p * p * (3.0 - 2.0 * p)
    if mode == "smootherstep":
        return p * p * p * (p * (p * 6.0 - 15.0) + 10.0)
    if mode == "ease_in":
        return p ** power
    if mode == "ease_out":
        return 1.0 - (1.0 - p) ** power
    if mode == "ease_in_out":
        if p < 0.5:
            return 0.5 * ((2.0 * p) ** power)
        return 1.0 - 0.5 * ((2.0 * (1.0 - p)) ** power)
    if mode == "hold_then_drop":
        if p <= hold:
            return 0.0
        u = (p - hold) / max(1e-6, 1.0 - hold)
        return u ** power
    if mode == "fast_drop":
        return p ** (1.0 / power)
    return p


def schedule_value(start: float, end: float, progress: float, mode: str, power: float = 2.0, hold: float = 0.0) -> float:
    t = schedule_curve(progress, mode, power=power, hold=hold)
    return float(start + (end - start) * t)


@dataclass
class TrajectoryRecorder:
    store_dtype: torch.dtype = torch.float16
    capture_noisy_xt: bool = False
    pin_memory: bool = False

    def __post_init__(self) -> None:
        self.x0_steps: list[torch.Tensor] = []
        self.xt_steps: list[torch.Tensor] = []
        self.step_sigmas: list[float] = []

    def callback(self, step: int, x0: torch.Tensor, x: torch.Tensor, total_steps: int) -> None:
        del step, total_steps
        self.x0_steps.append(_stage_cpu_tensor(x0, self.store_dtype, self.pin_memory))
        if self.capture_noisy_xt:
            self.xt_steps.append(_stage_cpu_tensor(x, self.store_dtype, self.pin_memory))


class ScaleLockedCFGGuider:
    """
    A custom ComfyUI guider that applies the scale lock in denoised-latent space.

    It subclasses the runtime behavior of comfy.samplers.CFGGuider at usage time
    (the actual inheritance happens in nodes.py where comfy is available).
    This standalone class only implements the extra logic/state we need.
    """

    def _init_scale_lock_state(
        self,
        *,
        model,
        anchors_x0_cpu: Iterable[torch.Tensor],
        planner_sigmas: Optional[Iterable[float]],
        lock_strength: float,
        lock_strength_start: float,
        lock_strength_end: float,
        cutoff: float,
        mid_cutoff: float,
        mid_strength: float,
        schedule: str,
        schedule_power: float,
        schedule_hold: float,
        mid_strength_start: float,
        mid_strength_end: float,
        mid_schedule: str,
        mid_schedule_power: float,
        mid_schedule_hold: float,
        spatial_mask: Optional[torch.Tensor],
        manifold_enabled: bool = False,
        manifold_strength: float = 0.0,
        manifold_strength_start: float = 1.0,
        manifold_strength_end: float = 0.0,
        manifold_schedule: str = "ease_out",
        manifold_schedule_power: float = 2.0,
        manifold_schedule_hold: float = 0.0,
        manifold_cutoff: float = 0.18,
        manifold_radial_strength: float = 1.0,
        manifold_anisotropy: float = 0.15,
        manifold_translation_strength: float = 1.0,
        manifold_anchor_mix: float = 0.18,
        manifold_mean_anchor_mix: float = 0.12,
        manifold_contrast_restore: float = 0.10,
        manifold_energy_tether: float = 0.0,
        manifold_channel_tether: float = 0.0,
        manifold_energy_gain_cap: float = 1.75,
        manifold_max_shift_px: float = 3.0,
        manifold_spatial_mask: Optional[torch.Tensor] = None,
    ) -> None:
        init_scale_lock_state(
            self,
            model=model,
            anchors_x0_cpu=anchors_x0_cpu,
            planner_sigmas=planner_sigmas,
            lock_strength=lock_strength,
            lock_strength_start=lock_strength_start,
            lock_strength_end=lock_strength_end,
            cutoff=cutoff,
            mid_cutoff=mid_cutoff,
            mid_strength=mid_strength,
            schedule=schedule,
            schedule_power=schedule_power,
            schedule_hold=schedule_hold,
            mid_strength_start=mid_strength_start,
            mid_strength_end=mid_strength_end,
            mid_schedule=mid_schedule,
            mid_schedule_power=mid_schedule_power,
            mid_schedule_hold=mid_schedule_hold,
            spatial_mask=spatial_mask,
            manifold_enabled=manifold_enabled,
            manifold_strength=manifold_strength,
            manifold_strength_start=manifold_strength_start,
            manifold_strength_end=manifold_strength_end,
            manifold_schedule=manifold_schedule,
            manifold_schedule_power=manifold_schedule_power,
            manifold_schedule_hold=manifold_schedule_hold,
            manifold_cutoff=manifold_cutoff,
            manifold_radial_strength=manifold_radial_strength,
            manifold_anisotropy=manifold_anisotropy,
            manifold_translation_strength=manifold_translation_strength,
            manifold_anchor_mix=manifold_anchor_mix,
            manifold_mean_anchor_mix=manifold_mean_anchor_mix,
            manifold_contrast_restore=manifold_contrast_restore,
            manifold_energy_tether=manifold_energy_tether,
            manifold_channel_tether=manifold_channel_tether,
            manifold_energy_gain_cap=manifold_energy_gain_cap,
            manifold_max_shift_px=manifold_max_shift_px,
            manifold_spatial_mask=manifold_spatial_mask,
        )

    def _slrd_resolve_step_index(self, timestep: torch.Tensor | float | int) -> int:
        return resolve_scale_lock_step_index(self, timestep)

    def _slrd_strength_for_step(self, idx: int) -> float:
        return scale_lock_strength_for_step(self, idx)

    def _slrd_anchor_for(self, idx: int, like: torch.Tensor) -> torch.Tensor:
        return scale_lock_anchor_for(self, idx, like)

    def _slrd_mask_for(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        return scale_lock_mask_for(self, like)

    def _slrd_manifold_strength_for_step(self, idx: int) -> float:
        return scale_lock_manifold_strength_for_step(self, idx)

    def _slrd_manifold_mask_for(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        return scale_lock_manifold_mask_for(self, like)


def init_scale_lock_state(
    guider,
    *,
    model,
    anchors_x0_cpu: Iterable[torch.Tensor],
    planner_sigmas: Optional[Iterable[float]],
    lock_strength: float,
    lock_strength_start: float,
    lock_strength_end: float,
    cutoff: float,
    mid_cutoff: float,
    mid_strength: float,
    schedule: str,
    schedule_power: float,
    schedule_hold: float,
    mid_strength_start: float,
    mid_strength_end: float,
    mid_schedule: str,
    mid_schedule_power: float,
    mid_schedule_hold: float,
    spatial_mask: Optional[torch.Tensor],
    manifold_enabled: bool = False,
    manifold_strength: float = 0.0,
    manifold_strength_start: float = 1.0,
    manifold_strength_end: float = 0.0,
    manifold_schedule: str = "ease_out",
    manifold_schedule_power: float = 2.0,
    manifold_schedule_hold: float = 0.0,
    manifold_cutoff: float = 0.18,
    manifold_radial_strength: float = 1.0,
    manifold_anisotropy: float = 0.15,
    manifold_translation_strength: float = 1.0,
    manifold_anchor_mix: float = 0.18,
    manifold_mean_anchor_mix: float = 0.12,
    manifold_contrast_restore: float = 0.10,
    manifold_energy_tether: float = 0.0,
    manifold_channel_tether: float = 0.0,
    manifold_energy_gain_cap: float = 1.75,
    manifold_max_shift_px: float = 3.0,
    manifold_spatial_mask: Optional[torch.Tensor] = None,
) -> None:
    guider._slrd_model = model
    guider._slrd_anchors_x0_cpu = list(anchors_x0_cpu)
    guider._slrd_planner_sigmas = [float(x) for x in planner_sigmas] if planner_sigmas is not None else None
    guider._slrd_lock_strength = float(lock_strength)
    guider._slrd_lock_strength_start = float(lock_strength_start)
    guider._slrd_lock_strength_end = float(lock_strength_end)
    guider._slrd_cutoff = float(cutoff)
    guider._slrd_mid_cutoff = float(max(cutoff, mid_cutoff))
    guider._slrd_mid_strength = float(mid_strength)
    guider._slrd_schedule = schedule
    guider._slrd_schedule_power = float(schedule_power)
    guider._slrd_schedule_hold = float(schedule_hold)
    guider._slrd_mid_strength_start = float(mid_strength_start)
    guider._slrd_mid_strength_end = float(mid_strength_end)
    guider._slrd_mid_schedule = mid_schedule
    guider._slrd_mid_schedule_power = float(mid_schedule_power)
    guider._slrd_mid_schedule_hold = float(mid_schedule_hold)
    guider._slrd_seen_sigmas = []
    guider._slrd_prev_match_idx = 0
    guider._slrd_spatial_mask = spatial_mask
    guider._slrd_last_sigma = None
    guider._slrd_manifold_enabled = bool(manifold_enabled)
    guider._slrd_manifold_strength = float(manifold_strength)
    guider._slrd_manifold_strength_start = float(manifold_strength_start)
    guider._slrd_manifold_strength_end = float(manifold_strength_end)
    guider._slrd_manifold_schedule = manifold_schedule
    guider._slrd_manifold_schedule_power = float(manifold_schedule_power)
    guider._slrd_manifold_schedule_hold = float(manifold_schedule_hold)
    guider._slrd_manifold_cutoff = float(max(0.05, min(1.0, manifold_cutoff)))
    guider._slrd_manifold_radial_strength = float(manifold_radial_strength)
    guider._slrd_manifold_anisotropy = float(manifold_anisotropy)
    guider._slrd_manifold_translation_strength = float(manifold_translation_strength)
    guider._slrd_manifold_anchor_mix = float(manifold_anchor_mix)
    guider._slrd_manifold_mean_anchor_mix = float(manifold_mean_anchor_mix)
    guider._slrd_manifold_contrast_restore = float(manifold_contrast_restore)
    guider._slrd_manifold_energy_tether = float(max(0.0, min(1.0, manifold_energy_tether)))
    guider._slrd_manifold_channel_tether = float(max(0.0, min(1.0, manifold_channel_tether)))
    guider._slrd_manifold_energy_gain_cap = float(max(1.0, manifold_energy_gain_cap))
    guider._slrd_manifold_max_shift_px = float(manifold_max_shift_px)
    guider._slrd_manifold_spatial_mask = manifold_spatial_mask if manifold_spatial_mask is not None else spatial_mask


def resolve_scale_lock_step_index(guider, timestep: torch.Tensor | float | int) -> int:
    sigma = _sigma_scalar(timestep)

    if guider._slrd_planner_sigmas:
        best_idx = 0
        best_dist = float("inf")
        for i, planner_sigma in enumerate(guider._slrd_planner_sigmas):
            dist = abs(planner_sigma - sigma)
            if dist < best_dist:
                best_idx = i
                best_dist = dist
        best_idx = max(best_idx, guider._slrd_prev_match_idx)
        best_idx = min(best_idx, len(guider._slrd_anchors_x0_cpu) - 1)
        guider._slrd_prev_match_idx = best_idx
        return best_idx

    if guider._slrd_last_sigma is None:
        guider._slrd_last_sigma = sigma
        step_index = 0
        guider._slrd_seen_sigmas = [sigma]
    else:
        tol = 1e-6 * max(1.0, abs(guider._slrd_last_sigma), abs(sigma))
        if abs(sigma - guider._slrd_last_sigma) > tol:
            guider._slrd_last_sigma = sigma
            guider._slrd_seen_sigmas.append(sigma)
        unique_sigmas = []
        for seen_sigma in guider._slrd_seen_sigmas:
            if all(
                abs(seen_sigma - unique_sigma) > (1e-6 * max(1.0, abs(seen_sigma), abs(unique_sigma)))
                for unique_sigma in unique_sigmas
            ):
                unique_sigmas.append(seen_sigma)
        step_index = max(0, len(unique_sigmas) - 1)
        step_index = max(step_index, guider._slrd_prev_match_idx)
        guider._slrd_prev_match_idx = step_index
    if not guider._slrd_anchors_x0_cpu:
        return 0
    return min(step_index, len(guider._slrd_anchors_x0_cpu) - 1)


def scale_lock_strength_for_step(guider, idx: int) -> float:
    return scale_lock_strengths_for_step(guider, idx)[0]


def scale_lock_progress_for_step(guider, idx: int) -> float:
    sigma_based = sigma_progress(getattr(guider, "_slrd_planner_sigmas", None), idx)
    if sigma_based is not None:
        return sigma_based
    total = max(1, len(guider._slrd_anchors_x0_cpu) - 1)
    return clamp(idx / total, 0.0, 1.0)


def _scheduled_strength(
    base_strength: float,
    start: float,
    end: float,
    progress: float,
    mode: str,
    power: float,
    hold: float,
) -> float:
    scheduled = schedule_value(start, end, progress, mode, power=power, hold=hold)
    return clamp(base_strength * scheduled, 0.0, 1.0)


def scale_lock_strengths_for_step(guider, idx: int) -> tuple[float, float]:
    progress = scale_lock_progress_for_step(guider, idx)
    low_strength = _scheduled_strength(
        guider._slrd_lock_strength,
        guider._slrd_lock_strength_start,
        guider._slrd_lock_strength_end,
        progress,
        guider._slrd_schedule,
        guider._slrd_schedule_power,
        guider._slrd_schedule_hold,
    )
    if getattr(guider, "_slrd_mid_schedule", "linked") == "linked":
        mid_strength = clamp(low_strength * guider._slrd_mid_strength, 0.0, 1.0)
    else:
        mid_strength = _scheduled_strength(
            guider._slrd_mid_strength,
            guider._slrd_mid_strength_start,
            guider._slrd_mid_strength_end,
            progress,
            guider._slrd_mid_schedule,
            guider._slrd_mid_schedule_power,
            guider._slrd_mid_schedule_hold,
        )
    return low_strength, mid_strength


def scale_lock_manifold_strength_for_step(guider, idx: int) -> float:
    if not getattr(guider, "_slrd_manifold_enabled", False):
        return 0.0
    progress = scale_lock_progress_for_step(guider, idx)
    return _scheduled_strength(
        guider._slrd_manifold_strength,
        guider._slrd_manifold_strength_start,
        guider._slrd_manifold_strength_end,
        progress,
        guider._slrd_manifold_schedule,
        guider._slrd_manifold_schedule_power,
        guider._slrd_manifold_schedule_hold,
    )


def scale_lock_anchor_for(guider, idx: int, like: torch.Tensor) -> torch.Tensor:
    anchor = guider._slrd_anchors_x0_cpu[idx].to(device=like.device, dtype=like.dtype, non_blocking=True)
    if tuple(anchor.shape[-2:]) != tuple(like.shape[-2:]):
        anchor = resize_4d_tensor(anchor, tuple(like.shape[-2:]))
    return anchor


def scale_lock_mask_for(guider, like: torch.Tensor) -> Optional[torch.Tensor]:
    if guider._slrd_spatial_mask is None:
        return None
    mask = guider._slrd_spatial_mask.to(device=like.device, dtype=like.dtype, non_blocking=True)
    if tuple(mask.shape[-2:]) != tuple(like.shape[-2:]):
        mask = resize_4d_tensor(mask, tuple(like.shape[-2:]))
    if mask.shape[0] < like.shape[0]:
        repeat = math.ceil(like.shape[0] / max(1, mask.shape[0]))
        mask = mask.repeat(repeat, 1, 1, 1)[: like.shape[0]]
    elif mask.shape[0] > like.shape[0]:
        mask = mask[: like.shape[0]]
    return mask


def scale_lock_manifold_mask_for(guider, like: torch.Tensor) -> Optional[torch.Tensor]:
    manifold_mask = getattr(guider, "_slrd_manifold_spatial_mask", None)
    if manifold_mask is None:
        return None
    mask = manifold_mask.to(device=like.device, dtype=like.dtype, non_blocking=True)
    if tuple(mask.shape[-2:]) != tuple(like.shape[-2:]):
        mask = resize_4d_tensor(mask, tuple(like.shape[-2:]))
    if mask.shape[0] < like.shape[0]:
        repeat = math.ceil(like.shape[0] / max(1, mask.shape[0]))
        mask = mask.repeat(repeat, 1, 1, 1)[: like.shape[0]]
    elif mask.shape[0] > like.shape[0]:
        mask = mask[: like.shape[0]]
    return mask




