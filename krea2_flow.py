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
