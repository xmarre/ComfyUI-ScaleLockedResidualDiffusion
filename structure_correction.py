"""Conservative whole-candidate selection at native Krea 2 stage boundaries."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .structure_screening import MIN_AXIS, _correspondence, _energy, _gradients, reduce_latent


def _upper_fence(values):
    """Tukey's upper inner fence, per image over spatial patches."""
    q1, q3 = torch.quantile(values.flatten(1), values.new_tensor([.25, .75]), dim=1)
    return (q3 + 1.5 * (q3 - q1))[:, None, None, None]


def _addition_outlier(earlier, terminal):
    increase = terminal - earlier
    eps = torch.finfo(increase.dtype).eps
    return ((increase > _upper_fence(increase).clamp_min(eps))
            & (terminal > _upper_fence(terminal).clamp_min(eps))
            & (earlier <= _upper_fence(earlier) + eps))


def _detail_energy(latent, hw):
    # Orthonormal 2x2 Haar AC energy per location. Process one batch item
    # at a time and reduce before comparing, so no full-res bands are stored.
    latent = latent.squeeze(2) if latent.ndim == 5 else latent
    h, w = latent.shape[-2:]
    a = latent[..., :h - h % 2:2, :w - w % 2:2]
    b = latent[..., :h - h % 2:2, 1:w - w % 2:2]
    c = latent[..., 1:h - h % 2:2, :w - w % 2:2]
    d = latent[..., 1:h - h % 2:2, 1:w - w % 2:2]
    # By Parseval: a²+b²+c²+d² - DC². Use differences to avoid
    # cancellation when high-frequency detail is small relative to DC.
    ab, cd = a - b, c - d
    ac = (a - c) + (b - d)
    energy = .5 * (ab.square() + cd.square()) + .25 * ac.square()
    return F.adaptive_avg_pool2d(energy.mean(dim=1, keepdim=True), hw)


@torch.no_grad()
def select_structure_candidate(reference, earlier, terminal, normalize):
    """Return selected batch indices and evidence, leaving every input intact.

    The earlier candidate is an internal clean prediction, and terminal is
    the completed native output-space latent. Normalize only pooled terminal
    evidence for screening. Only shortlisted items need full-grid checks.
    """
    if (reference is None or earlier is None or earlier.shape != terminal.shape
            or min(reference.shape[-2:]) < MIN_AXIS):
        return [], []
    hw = reference.shape[-2:]
    early_grid = reduce_latent(earlier, hw)
    end_grid = reduce_latent(terminal, hw)
    if early_grid is None or end_grid is None or early_grid.shape != reference.shape or end_grid.shape != reference.shape:
        return [], []
    early_grid = early_grid.to(reference.device)
    end_grid = normalize(end_grid.unsqueeze(2)).squeeze(2).to(reference.device)
    finite = torch.isfinite(early_grid).flatten(1).all(1) & torch.isfinite(end_grid).flatten(1).all(1)
    mean_missing, added_delta = [], []
    fine_mask = None
    eps = torch.finfo(reference.dtype).eps
    for scale in (1, 2):
        images = ((reference, early_grid, end_grid) if scale == 1 else
                  tuple(F.avg_pool2d(x, scale) for x in (reference, early_grid, end_grid)))
        gradients = _gradients(images[0])
        energy = _energy(*gradients)
        usable = torch.isfinite(energy).flatten(1).all(1) & (energy.flatten(1).mean(1) > eps)
        finite &= usable
        early_added, early_missing, _ = _correspondence(gradients, energy, images[1])
        end_added, end_missing, _ = _correspondence(gradients, energy, images[2])
        mean_missing.append((early_missing - end_missing).flatten(1).mean(1))
        increase = (end_added - early_added).flatten(1).amax(1)
        added_delta.append(increase)
        outlier = _addition_outlier(early_added, end_added)
        if scale == 1:
            fine_mask = outlier
        else:
            # A reported addition must have support at the same location
            # at both scales. Nearest resampling only maps patch evidence.
            overlap = fine_mask & F.interpolate(outlier.float(), fine_mask.shape[-2:], mode="nearest").bool()

    candidate = finite & overlap.flatten(1).any(1) & (torch.stack(mean_missing).amax(0) <= eps)
    shortlisted = torch.stack((candidate, added_delta[0], added_delta[1]), dim=1).tolist()
    selected, receipt = [], []
    for i, (passes, fine_increase, coarse_increase) in enumerate(shortlisted):
        record = {"item": i, "reason": "insufficient_structural_evidence"}
        if passes:
            record.update(fine_added_increase=fine_increase, coarse_added_increase=coarse_increase)
            # Compare detail across the canvas. The median of patchwise
            # changes limits the influence of the offending region. An
            # earlier estimate with lower typical AC energy is rejected.
            native = normalize(terminal[i:i + 1].to(device=earlier.device, dtype=torch.float32))
            earlier_energy = _detail_energy(earlier[i:i + 1].float(), overlap.shape[-2:])
            native_energy = _detail_energy(native, overlap.shape[-2:])
            delta = (earlier_energy - native_energy).median()
            rounding = torch.finfo(native_energy.dtype).eps * native_energy.amax()
            retained = delta >= -rounding
            valid = (torch.isfinite(earlier_energy).all() & torch.isfinite(native_energy).all())
            if bool(retained & valid):
                selected.append(i)
                record["reason"] = "selected_penultimate_prediction"
            else:
                record["reason"] = "retained_native_detail"
        receipt.append(record)
    return selected, receipt
