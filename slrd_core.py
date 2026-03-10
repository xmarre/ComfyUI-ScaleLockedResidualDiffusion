from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional

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


def schedule_value(start: float, end: float, progress: float, mode: str) -> float:
    progress = float(max(0.0, min(1.0, progress)))
    if mode == "cosine":
        t = 0.5 - 0.5 * math.cos(math.pi * progress)
    elif mode == "flat":
        t = 0.0
    else:
        t = progress
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
        spatial_mask: Optional[torch.Tensor],
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
            spatial_mask=spatial_mask,
        )

    def _slrd_resolve_step_index(self, timestep: torch.Tensor | float | int) -> int:
        return resolve_scale_lock_step_index(self, timestep)

    def _slrd_strength_for_step(self, idx: int) -> float:
        return scale_lock_strength_for_step(self, idx)

    def _slrd_anchor_for(self, idx: int, like: torch.Tensor) -> torch.Tensor:
        return scale_lock_anchor_for(self, idx, like)

    def _slrd_mask_for(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        return scale_lock_mask_for(self, like)


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
    spatial_mask: Optional[torch.Tensor],
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
    guider._slrd_seen_sigmas = []
    guider._slrd_prev_match_idx = 0
    guider._slrd_spatial_mask = spatial_mask
    guider._slrd_last_sigma = None


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
    total = max(1, len(guider._slrd_anchors_x0_cpu) - 1)
    progress = idx / total
    scheduled = schedule_value(guider._slrd_lock_strength_start, guider._slrd_lock_strength_end, progress, guider._slrd_schedule)
    return float(max(0.0, min(1.0, guider._slrd_lock_strength * scheduled)))


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




