"""Conservative whole-candidate selection at native Krea 2 stage boundaries."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .structure_screening import MIN_AXIS, _correspondence, _energy, _gradients, _region, reduce_latent


def _upper_fence(values):
    """Tukey's upper inner fence, per image over spatial patches."""
    q1, q3 = torch.quantile(values.flatten(1), values.new_tensor([.25, .75]), dim=1)
    return (q3 + 1.5 * (q3 - q1))[:, None, None, None]


def _new_outlier(earlier, terminal):
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
    total = latent.new_zeros((1, 1, *hw))
    for channels in latent.split(4, dim=1):
        a = channels[..., :h - h % 2:2, :w - w % 2:2]
        b = channels[..., :h - h % 2:2, 1:w - w % 2:2]
        c = channels[..., 1:h - h % 2:2, :w - w % 2:2]
        d = channels[..., 1:h - h % 2:2, 1:w - w % 2:2]
        # By Parseval: a²+b²+c²+d² - DC². Use differences to avoid
        # cancellation when high-frequency detail is small relative to DC.
        energy = .5 * ((a - b).square() + (c - d).square()) + .25 * ((a - c) + (b - d)).square()
        total += F.adaptive_avg_pool2d(energy.sum(dim=1, keepdim=True), hw)
    return total / latent.shape[1]


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
    worse_missing, added_delta = [], []
    additions, reversals = [], []
    eps = torch.finfo(reference.dtype).eps
    for scale in (1, 2):
        images = ((reference, early_grid, end_grid) if scale == 1 else
                  tuple(F.avg_pool2d(x, scale) for x in (reference, early_grid, end_grid)))
        gradients = _gradients(images[0])
        energy = _energy(*gradients)
        usable = torch.isfinite(energy).flatten(1).all(1) & (energy.flatten(1).mean(1) > eps)
        finite &= usable
        early_added, early_missing, _ = _correspondence(
            gradients, energy, images[1], prefer_lower_unmatched=True,
        )
        end_added, end_missing, _ = _correspondence(
            gradients, energy, images[2], prefer_lower_unmatched=True,
        )
        delta_missing = early_missing - end_missing
        missing_outliers = _new_outlier(end_missing, early_missing)
        worse_missing.append((delta_missing.flatten(1).mean(1) > eps) | missing_outliers.flatten(1).any(1))
        increase = (end_added - early_added).flatten(1).amax(1)
        added_delta.append(increase)
        additions.append(_new_outlier(early_added, end_added))
        reversals.append(_new_outlier(end_added, early_added))
        if scale == 1:
            fine_increase = end_added - early_added

    def shared_outliers(maps):
        fine, coarse = maps
        aligned = F.interpolate(coarse.float(), fine.shape[-2:], mode="nearest-exact").bool()
        return fine & aligned

    localized = shared_outliers(additions)
    opposing = shared_outliers(reversals).flatten(1).any(1) | torch.stack(worse_missing).any(0)
    boxes = _region(torch.where(localized, fine_increase, 0.), hw)[:, 2:]
    shortlisted = torch.cat((
        torch.stack((finite, localized.flatten(1).any(1), opposing, *added_delta), dim=1), boxes,
    ), dim=1).tolist()
    selected, receipt = [], []
    for i, (usable, has_localized_addition, has_opposing_change, fine_peak, coarse_peak, *box) in enumerate(shortlisted):
        record = {"item": i}
        if not usable:
            record["reason"] = "unavailable_evidence"
        elif not has_localized_addition:
            record["reason"] = "no_localized_two_scale_addition"
        elif has_opposing_change:
            record["reason"] = "earlier_candidate_loses_other_structure"
        else:
            record.update(fine_added_increase=fine_peak, coarse_added_increase=coarse_peak, peak_box_xyxy=box)
            # Compare detail across the canvas. The median of patchwise
            # changes limits the influence of the offending region. An
            # earlier estimate with lower typical AC energy is rejected.
            native = normalize(terminal[i:i + 1].to(device=earlier.device, dtype=torch.float32))
            earlier_energy = _detail_energy(earlier[i:i + 1].float(), additions[0].shape[-2:])
            native_energy = _detail_energy(native, additions[0].shape[-2:])
            detail_loss = native_energy - earlier_energy
            rounding = torch.finfo(native_energy.dtype).eps * native_energy.amax()
            # Also protect a newly sharpened region elsewhere in the image.
            # Patch pooling and the spatial search can spread an addition
            # into adjacent patches, so exclude their immediate neighbors.
            region = F.max_pool2d(localized[i:i + 1].float(), 3, stride=1, padding=1).bool()
            local_loss = _new_outlier(earlier_energy, native_energy) & ~region & (detail_loss > rounding)
            retained = (detail_loss.median() <= rounding) & ~local_loss.any()
            valid = (torch.isfinite(earlier_energy).all() & torch.isfinite(native_energy).all())
            if bool(retained & valid):
                selected.append(i)
                record["reason"] = "selected_penultimate_prediction"
            else:
                record["reason"] = "retained_native_detail"
        receipt.append(record)
    return selected, receipt
