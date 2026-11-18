from __future__ import annotations

import torch


def build_patch_coupled_noise(
    lowres_noise: torch.Tensor,
    target_shape: tuple[int, ...],
    seed: int,
) -> torch.Tensor:
    """Lift low-grid Gaussian noise onto the finer Krea 2 patch lattice.

    Each source 2x2 latent-patch phase is refined independently. Within every
    parent cell's target group, the normalized sum equals the source noise
    component. For iid Gaussian source and innovation this produces iid
    standard Gaussian target noise, including non-integral scale ratios.

    This is a *noise coupling*, not an upscaler or a guarantee of visual quality.
    """
    if isinstance(lowres_noise, torch.Tensor) and lowres_noise.ndim == 5:
        if lowres_noise.shape[2] != 1 or len(target_shape) != 5 or target_shape[2] != 1:
            raise ValueError("Krea 2 image noise requires a single temporal slice")
        return build_patch_coupled_noise(
            lowres_noise.squeeze(2), tuple(target_shape[:2]) + tuple(target_shape[-2:]), seed
        ).unsqueeze(2)
    if not isinstance(lowres_noise, torch.Tensor) or lowres_noise.ndim != 4:
        raise TypeError("source noise must be BxCxHxW")
    if not lowres_noise.is_floating_point() or not bool(torch.isfinite(lowres_noise).all()):
        raise ValueError("source noise must contain finite floating values")

    b, c, sh, sw = map(int, lowres_noise.shape)
    tb, tc, th, tw = map(int, target_shape)
    if (b, c) != (tb, tc):
        raise ValueError("batch/channel dimensions must match")
    if min(sh, sw, th, tw) < 2 or any(v % 2 for v in (sh, sw, th, tw)):
        raise ValueError("Krea 2 patch geometry requires positive even latent H/W")
    if th < sh or tw < sw or (th == sh and tw == sw):
        raise ValueError("target must grow at least one spatial dimension")

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    innovation = torch.randn(
        target_shape, generator=g, device="cpu", dtype=torch.float32
    ).to(lowres_noise.device)
    result = torch.empty_like(innovation)

    source_patch_h, source_patch_w = sh // 2, sw // 2
    target_patch_h, target_patch_w = th // 2, tw // 2
    owner_y = (
        torch.arange(target_patch_h, device=lowres_noise.device, dtype=torch.long)
        * source_patch_h // target_patch_h
    )
    owner_x = (
        torch.arange(target_patch_w, device=lowres_noise.device, dtype=torch.long)
        * source_patch_w // target_patch_w
    )

    def axis_means(x: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.shape[-1]
        bounds = torch.div(
            torch.arange(groups + 1, device=x.device) * length + groups - 1,
            groups, rounding_mode="floor",
        )
        prefix = torch.cat((torch.zeros_like(x[..., :1]), x.cumsum(dim=-1)), dim=-1)
        sums = prefix.index_select(-1, bounds[1:]) - prefix.index_select(-1, bounds[:-1])
        counts = bounds[1:] - bounds[:-1]
        return sums / counts.to(dtype=x.dtype), counts

    for phase_y in range(2):
        for phase_x in range(2):
            z = innovation[..., phase_y::2, phase_x::2]
            x_means, x_counts = axis_means(z, source_patch_w)
            y_means_t, y_counts = axis_means(
                x_means.transpose(-1, -2), source_patch_h
            )
            means = y_means_t.transpose(-1, -2)
            group_counts = (
                (y_counts[:, None] * x_counts[None, :])[owner_y[:, None], owner_x[None, :]]
                .to(dtype=innovation.dtype)
            )
            parent = lowres_noise[..., phase_y::2, phase_x::2].float()
            owner = (..., owner_y[:, None], owner_x[None, :])
            result[..., phase_y::2, phase_x::2] = (
                z - means[owner] + parent[owner] / group_counts.sqrt()
            )

    return result.to(dtype=lowres_noise.dtype)


def _patch_phases(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(x, torch.Tensor) or x.ndim < 2 or not x.is_floating_point():
        raise TypeError("patch lattice analysis requires a floating tensor with trailing HxW")
    h, w = map(int, x.shape[-2:])
    if min(h, w) < 2 or h % 2 or w % 2:
        raise ValueError("Krea 2 patch geometry requires positive even latent H/W")
    return x[..., 0::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 0::2], x[..., 1::2, 1::2]


def patch_lattice_modes(x: torch.Tensor) -> torch.Tensor:
    """Amplitudes of the three period-2 spatial modes of each ``HxW`` plane.

    Krea 2 emits one token per 2x2 latent patch. An offset that depends only
    on the position inside the patch is therefore a pattern with a period of
    two latent pixels (16 image pixels): exactly the column-alternating,
    row-alternating and checkerboard Fourier modes. The returned ``[..., 3]``
    tensor holds their signed amplitudes in that order, so that the pattern is
    ``cols * (-1)**x + rows * (-1)**y + diag * (-1)**(x + y)``.
    """
    a, b, c, d = (p.mean(dim=(-2, -1)) for p in _patch_phases(x))
    return torch.stack((a - b + c - d, a + b - c - d, a - b - c + d), dim=-1) / 4.0


def remove_patch_lattice(x: torch.Tensor) -> torch.Tensor:
    """Return ``x`` without its three period-2 patch-lattice modes.

    Only those three Fourier coefficients of each plane change; the plane mean
    and every other frequency are preserved. Each plane is corrected with its
    own measurement, so batches, channels and a singleton temporal axis need
    no special handling. Content that is not phase-locked to the patch grid
    contributes only its chance-level residue, of order ``1/sqrt(patches)`` of
    its fine-detail amplitude.
    """
    # Half-precision means over a large plane lose the small offset; measure in float32 or wider.
    wide = x if x.dtype in (torch.float32, torch.float64) else x.float()
    modes = patch_lattice_modes(wide).to(dtype=x.dtype)
    cols, rows, diag = (modes[..., k, None, None] for k in range(3))
    out = x.clone()
    for py, sy in ((0, 1.0), (1, -1.0)):
        for px, sx in ((0, 1.0), (1, -1.0)):
            out[..., py::2, px::2] -= sx * cols + sy * rows + sx * sy * diag
    return out


def patch_lattice_metrics(x: torch.Tensor) -> dict:
    """Strength of the period-2 patch lattice in a latent, as plain floats.

    ``coherence`` and ``max_coherence`` are the RMS and the maximum, over all
    planes and the three modes, of ``|mean| / rms`` of the per-patch mode
    coefficients: 1 means every patch carries the identical pattern, and
    content that ignores the patch grid gives about ``chance``
    (``1/sqrt(patches)``). ``amplitude`` is the RMS pattern amplitude in the
    units of ``x``.
    """
    a, b, c, d = (p.float() for p in _patch_phases(x))
    coefficients = torch.stack((a - b + c - d, a + b - c - d, a - b - c + d), dim=-3) / 4.0
    mean = coefficients.mean(dim=(-2, -1))
    rms = coefficients.pow(2).mean(dim=(-2, -1)).sqrt()
    coherence = mean.abs() / rms.clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "coherence": float(coherence.pow(2).mean().sqrt()),
        "max_coherence": float(coherence.max()),
        "chance": float((a.shape[-2] * a.shape[-1]) ** -0.5),
        "amplitude": float(mean.pow(2).sum(dim=-1).mean().sqrt()),
    }
