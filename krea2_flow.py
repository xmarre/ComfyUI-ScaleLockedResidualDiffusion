from __future__ import annotations

import torch


class PatchMeanProjection:
    """Disjoint patch-group means and their phase-preserving right inverse.

    The ownership partition matches ``build_patch_coupled_noise``. Denoised
    signals use arithmetic means to preserve their amplitude; Gaussian noise
    uses normalized sums instead. These normalizations are not interchangeable.
    """

    def __init__(self, source_hw: tuple[int, int], target_hw: tuple[int, int]):
        self.source_hw = tuple(map(int, source_hw))
        self.target_hw = tuple(map(int, target_hw))
        if len(self.source_hw) != 2 or len(self.target_hw) != 2:
            raise ValueError("patch projection requires spatial H/W pairs")
        if any(v < 2 or v % 2 for v in (*self.source_hw, *self.target_hw)):
            raise ValueError("patch projection requires positive even latent H/W")
        if any(t < s for s, t in zip(self.source_hw, self.target_hw)):
            raise ValueError("target patch grid cannot shrink the planner")

    @staticmethod
    def _image(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5 and x.shape[2] == 1:
            return x.squeeze(2)
        if x.ndim != 4:
            raise ValueError("patch projection requires BCHW or BC1HW image latents")
        return x

    @staticmethod
    def _phases(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)

    @staticmethod
    def _unphases(x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, h, w = x.shape
        return x.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * 2, w * 2)

    @staticmethod
    def _axis_means(x: torch.Tensor, groups: int) -> torch.Tensor:
        length = x.shape[-1]
        if length == groups:
            return x
        bounds = torch.div(
            torch.arange(groups + 1, device=x.device) * length + groups - 1,
            groups, rounding_mode="floor",
        )
        prefix = torch.cat((torch.zeros_like(x[..., :1]), x.cumsum(dim=-1)), dim=-1)
        sums = prefix.index_select(-1, bounds[1:]) - prefix.index_select(-1, bounds[:-1])
        return sums / (bounds[1:] - bounds[:-1]).to(dtype=x.dtype)

    def reduce(self, target: torch.Tensor) -> torch.Tensor:
        """Return one arithmetic mean per planner patch phase and parent cell."""
        image = self._image(target)
        if tuple(image.shape[-2:]) != self.target_hw:
            raise ValueError("target prediction shape differs from the projection grid")
        phases = self._phases(image)
        reduced = self._axis_means(phases, self.source_hw[1] // 2)
        reduced = self._axis_means(reduced.transpose(-1, -2), self.source_hw[0] // 2)
        out = self._unphases(reduced.transpose(-1, -2))
        return out.unsqueeze(2) if target.ndim == 5 else out

    def lift(self, planner: torch.Tensor) -> torch.Tensor:
        """Repeat parent values within each phase, without mixing patch phases."""
        image = self._image(planner)
        if tuple(image.shape[-2:]) != self.source_hw:
            raise ValueError("planner prediction shape differs from the projection grid")
        sh, sw = (v // 2 for v in self.source_hw)
        th, tw = (v // 2 for v in self.target_hw)
        owner_y = torch.arange(th, device=image.device) * sh // th
        owner_x = torch.arange(tw, device=image.device) * sw // tw
        phases = self._phases(image).index_select(-2, owner_y).index_select(-1, owner_x)
        out = self._unphases(phases)
        return out.unsqueeze(2) if planner.ndim == 5 else out

    def correct(
        self,
        base: torch.Tensor,
        planner: torch.Tensor,
        strength: float,
        candidate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Replace coarse means while preserving the candidate's detail residual.

        A candidate may contain an optional mid-band edit. Its coarse component
        is removed, so mid-band strength cannot weaken or overshoot this lock.
        At strength one, reduce(output) == planner and the correction is
        idempotent up to floating point rounding. No input tensor is mutated.
        """
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("patch lock strength must be finite and in [0, 1]")
        if candidate is None:
            candidate = base
        if base.shape != candidate.shape or base.shape[:2] != planner.shape[:2] or base.ndim != planner.ndim:
            raise ValueError("planner, base and candidate layouts must match")
        self._image(base)
        self._image(planner)
        if strength == 0.0 and candidate is base:
            return base
        # Accumulate in float32 for low-precision model predictions. Preserve
        # float64 for numerical checks and return the model's original dtype.
        dtype = torch.float64 if base.dtype == torch.float64 else torch.float32
        work = base.to(dtype=dtype)
        anchor = planner.to(device=base.device, dtype=dtype)
        edited = candidate.to(device=base.device, dtype=dtype)
        coarse = self.reduce(work)
        desired = torch.lerp(coarse, anchor, strength)
        edited_coarse = coarse if candidate is base else self.reduce(edited)
        out = edited + self.lift(desired - edited_coarse)
        return out.to(dtype=base.dtype)


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
