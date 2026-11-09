"""Completed-latent enlargement backends for Krea 2 resolution handoffs.

``interpolate`` is the original Scale-Locked behavior (``resize_4d_tensor``).
``sesqui_wan21`` runs the pretrained SesquiLSR Wan 2.1 latent upscaler,
vendored unmodified under ``third_party/sesqui_lsr`` (MIT, upstream commit
821e6db7412089f2a5c0a106c42d3c1a396b9515).

Latent space: SesquiLSR's Wan21 weights are trained on raw Wan VAE latents.
ComfyUI LATENT tensors for Krea 2 (latent format ``Wan21``) are raw VAE
latents: ``VAE.encode`` returns the encoder mean, and the per-channel
z-score lives only in ``process_latent_in``. Upstream's own ComfyUI node uses
an identity adaptor for these latents for that reason. The handoffs here pass
completed ``CFGGuider.sample`` outputs, which ComfyUI has already returned
through ``process_latent_out``; they are clean output-space latents, never a
noisy sampler state or a model-internal tensor.
"""
from __future__ import annotations

import hashlib
import math
import threading
from pathlib import Path

import torch

from .slrd_core import resize_4d_tensor

LATENT_UPSCALE_METHODS = ("interpolate", "sesqui_wan21")

SESQUI_UPSTREAM = "https://github.com/LoganBooker/SesquiLSR"
SESQUI_UPSTREAM_COMMIT = "821e6db7412089f2a5c0a106c42d3c1a396b9515"
SESQUI_WAN21_FILENAME = "upscaler_Wan21.safetensors"
SESQUI_WAN21_SHA256 = "d1b4c2c1d80b2af81e6bbfd38501ffb5adffb9f7c0ed2599555b837d0d2d015a"
SESQUI_WAN21_CHANNELS = 16
# Trained scale range per spatial axis (upstream README / node: 1x to 2x).
SESQUI_MAX_AXIS_SCALE = 2.0

_WEIGHTS_PATH = Path(__file__).resolve().parent / "third_party" / "sesqui_lsr" / SESQUI_WAN21_FILENAME

_lock = threading.Lock()
_state_dict = None
_models: dict = {}


def _verified_state_dict(path: Path = _WEIGHTS_PATH):
    """Load the pinned weights once, after checking their SHA-256."""
    global _state_dict
    with _lock:
        # Preflight runs for each generation. Never allow a previous model
        # load to bypass the current bundled weight integrity check.
        if not path.is_file():
            raise FileNotFoundError(
                f"SesquiLSR Wan21 weights are missing: expected {path}. They ship with this "
                "custom node; reinstall or update the Scale-Locked node files."
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SESQUI_WAN21_SHA256:
            raise ValueError(
                f"SesquiLSR Wan21 weights at {path} failed verification "
                f"(sha256 {digest}, expected {SESQUI_WAN21_SHA256}); refusing to load them."
            )
        if _state_dict is not None:
            return _state_dict
        from safetensors.torch import load_file
        _state_dict = load_file(str(path), device="cpu")
        return _state_dict


def ensure_latent_upscaler(method: str) -> None:
    """Validate a backend before any sampling work (weights present and intact)."""
    if method not in LATENT_UPSCALE_METHODS:
        raise ValueError(f"latent_upscale_method must be one of {', '.join(LATENT_UPSCALE_METHODS)}")
    if method == "sesqui_wan21":
        _verified_state_dict()


def _model_for(device: torch.device):
    """One cached fp32 eval-mode instance per device (about 12 MB each)."""
    key = torch.device(device)
    with _lock:
        model = _models.get(key)
        if model is not None:
            return model
    state = _verified_state_dict()
    # Serialize construction and device transfer for the same cache. Previously
    # competing invocations built and allocated redundant CUDA models before
    # setdefault chose one, creating an avoidable VRAM peak during first use.
    with _lock:
        model = _models.get(key)
        if model is not None:
            return model
        from .third_party.sesqui_lsr import LatentUpscaler
        model = LatentUpscaler(in_channels=SESQUI_WAN21_CHANNELS)
        model.load_state_dict(state)
        model = model.to(device=key, dtype=torch.float32).eval().requires_grad_(False)
        _models[key] = model
        return model


def sesqui_hops(src_hw, dst_hw) -> list[tuple[int, int]]:
    """Exact output sizes of the learned passes for one handoff.

    One pass when both axes grow by at most 2x. A larger jump is split into
    the fewest geometrically spaced latent-only passes that each stay within
    the trained 1x-2x range; the last pass lands exactly on ``dst_hw``. No
    diffusion stage or model evaluation is added.
    """
    (sh, sw), (th, tw) = tuple(map(int, src_hw)), tuple(map(int, dst_hw))
    if min(sh, sw, th, tw) < 1:
        raise ValueError("latent sizes must be positive")
    if th < sh or tw < sw:
        raise ValueError("SesquiLSR only enlarges latents (1x to 2x per axis)")
    ratio = max(th / sh, tw / sw)
    passes = max(1, math.ceil(math.log(ratio, SESQUI_MAX_AXIS_SCALE) - 1e-9))
    hops = []
    ph, pw = sh, sw
    for k in range(1, passes + 1):
        if k == passes:
            hh, ww = th, tw
        else:
            hh = max(ph, min(th, round(sh * (th / sh) ** (k / passes))))
            ww = max(pw, min(tw, round(sw * (tw / sw) ** (k / passes))))
        if hh > SESQUI_MAX_AXIS_SCALE * ph or ww > SESQUI_MAX_AXIS_SCALE * pw:
            raise ValueError(
                f"cannot split latent enlargement {sh}x{sw} -> {th}x{tw} into SesquiLSR passes "
                "of at most 2x per axis"
            )
        hops.append((hh, ww))
        ph, pw = hh, ww
    return hops


def enlarge_completed_latent(source: torch.Tensor, hw, method: str, *, device=None) -> torch.Tensor:
    """Enlarge a completed output-space latent to ``hw`` with ``method``.

    Returns float32 on ``device`` for ``sesqui_wan21`` (default: the
    source's device). ``interpolate`` reproduces ``resize_4d_tensor``.
    Accepts BCHW and single-frame BC1HW, and preserves that layout.
    """
    hw = tuple(map(int, hw))
    if method == "interpolate":
        return resize_4d_tensor(source, hw)
    if method != "sesqui_wan21":
        raise ValueError(f"latent_upscale_method must be one of {', '.join(LATENT_UPSCALE_METHODS)}")
    temporal = source.ndim == 5
    if temporal and source.shape[2] != 1:
        raise ValueError("SesquiLSR handoffs support single-frame BC1HW latents only")
    x = source.squeeze(2) if temporal else source
    if x.ndim != 4 or x.shape[1] != SESQUI_WAN21_CHANNELS:
        raise ValueError(f"SesquiLSR Wan21 expects a 16-channel BCHW/BC1HW latent, got {tuple(source.shape)}")
    if tuple(x.shape[-2:]) == hw:
        out = x.to(device=device or x.device, dtype=torch.float32)
        return out.unsqueeze(2) if temporal else out
    device = torch.device(device) if device is not None else x.device
    hops = sesqui_hops(x.shape[-2:], hw)
    if device.type == "cpu":
        # Preserve the original CPU failure, including its traceback.
        out = _run_hops(x, hops, device)
    else:
        out = _run_hops_or_oom(x, hops, device)
        if out is None:
            # Complete the failed invocation's exception handler before
            # freeing memory or retrying: its traceback may retain the
            # model's intermediate GPU tensors.
            _free_device_memory(device)
            out = _run_hops_or_oom(x, hops, device)
            if out is None:
                _free_device_memory(device)
                import logging
                logging.getLogger(__name__).warning(
                    "SesquiLSR: out of memory on %s for a %dx%d latent; enlarging on the CPU instead",
                    device, hw[0], hw[1],
                )
                out = _run_hops(x, hops, torch.device("cpu")).to(device)
    return out.unsqueeze(2) if temporal else out


_OOM_ERRORS = tuple({getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError),
                     torch.cuda.OutOfMemoryError})


def _run_hops_or_oom(x: torch.Tensor, hops, device: torch.device) -> torch.Tensor | None:
    """End the failed call's traceback lifetime before retrying on any device."""
    try:
        return _run_hops(x, hops, device)
    except _OOM_ERRORS:
        return None


def _run_hops(x: torch.Tensor, hops, device: torch.device) -> torch.Tensor:
    model = _model_for(device)
    results = []
    with torch.no_grad():
        # One image at a time bounds activation memory at large canvases.
        for item in x.detach().to(device=device, dtype=torch.float32).split(1, dim=0):
            for hop in hops:
                item = model(item, hop)
            results.append(item)
    return torch.cat(results, dim=0)


def _free_device_memory(device: torch.device) -> None:
    """Ask ComfyUI to unload managed models from ``device`` and release cached blocks."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    mm.free_memory(1e30, device)
    mm.soft_empty_cache()
