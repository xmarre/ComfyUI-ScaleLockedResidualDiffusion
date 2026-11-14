"""Read-only local correspondence for Krea 2 clean-latent predictions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# This is an analysis grid, in latent cells. Neither it nor the patch search
# changes the sampling resolution. The longest-axis cap bounds allocations.
MAX_AXIS = 128
PATCH = 8
STRIDE = PATCH // 2
MAX_PREDICTIONS = 32
MIN_AXIS = 2 * (PATCH + 1)  # both scales need an entire gradient patch
_SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
           (-1, -1), (-1, 1), (1, -1), (1, 1))


def _image_plane(latent):
    if latent.ndim == 5 and latent.shape[2] == 1:
        latent = latent.squeeze(2)
    if latent.ndim != 4 or not latent.is_floating_point() or not latent.numel():
        return None
    return latent


def analysis_hw(source_hw, target_hw):
    """Aspect-preserving grid supported by both latent grids."""
    sh, sw = source_hw
    th, tw = target_hw
    scale = min(1., th / sh, tw / sw, MAX_AXIS / max(sh, sw))
    return max(1, round(sh * scale)), max(1, round(sw * scale))


def reduce_latent(latent, hw):
    """Area reduce on the input device before any cast or transfer."""
    plane = _image_plane(latent)
    if plane is None:
        return None
    return F.adaptive_avg_pool2d(plane.detach(), hw).float()


def _gradients(image):
    center = image[..., :-1, :-1]
    return image[..., :-1, 1:] - center, image[..., 1:, :-1] - center


def _patch_mean(x):
    return F.avg_pool2d(x, PATCH, STRIDE, ceil_mode=True, count_include_pad=False)


def _energy(gx, gy):
    return _patch_mean((gx.square() + gy.square()).mean(dim=1, keepdim=True))


def _correspondence(reference, reference_energy, candidate, *, prefer_lower_unmatched=False):
    """One common subregion displacement for both axes and all channels."""
    rx, ry = reference
    cx, cy = _gradients(candidate)
    h, w = cx.shape[-2:]
    px, py = (F.pad(g, (1, 1, 1, 1), mode="replicate") for g in (cx, cy))
    best = torch.full_like(reference_energy, -1.)
    best_energy = torch.zeros_like(reference_energy)
    floor = reference_energy.mean(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
    for dy, dx in _SHIFTS:
        gx = px[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        gy = py[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        energy = _energy(gx, gy)
        dot = _patch_mean((rx * gx + ry * gy).mean(dim=1, keepdim=True))
        denom = (reference_energy * energy).sqrt()
        corr = torch.where(denom > 0, dot / denom.clamp_min(torch.finfo(torch.float32).tiny), 0.)
        corr = corr.clamp(0., 1.)
        better = corr > best
        if prefer_lower_unmatched:
            # Equal-angle matches have no positional preference. Use the
            # lower-energy candidate as conservative evidence for additions.
            better |= (corr == best) & (energy < best_energy)
        best = torch.where(better, corr, best)
        best_energy = torch.where(better, energy, best_energy)
    # An oriented gradient explained by a positively correlated match is
    # discounted. The energy floor prevents tiny flat-region noise from
    # becoming a high score, without imposing a semantic decision threshold.
    unmatched = (1 - best.square()) / (reference_energy + best_energy + floor)
    ratio = best_energy.mean(dim=(-3, -2, -1)) / floor.flatten()
    return best_energy * unmatched, reference_energy * unmatched, ratio


def _region(values, hw):
    """Return mean, peak and a normalized box for the strongest patch."""
    flat = values.flatten(1)
    peak, index = flat.max(dim=1)
    h, w = hw
    y = torch.div(index, values.shape[-1], rounding_mode="floor") * STRIDE
    x = index.remainder(values.shape[-1]) * STRIDE
    box = torch.stack((x / w, y / h, (x + PATCH + 1).clamp(max=w) / w,
                       (y + PATCH + 1).clamp(max=h) / h), dim=1)
    return torch.cat((flat.mean(dim=1)[:, None], peak[:, None], box), dim=1)


class StageStructureScreen:
    """Owns compact evidence for one stage. Inputs are read-only."""

    def __init__(self, reference, sigmas, lock_sigma, noise_scale):
        self.hw = tuple(reference.shape[-2:])
        self.sigmas = sigmas
        self.noise_scale = noise_scale
        self.reference = reference if min(self.hw) >= MIN_AXIS else None
        self.scales = None
        self.variance = None
        self.entries = []
        self.previous = None
        # Take every candidate on ordinary schedules. Bound the receipt on
        # long schedules and always retain the first two and last unlocked steps.
        end = max(0, len(sigmas) - 1)
        self.first = next((i for i, s in enumerate(sigmas[:-1])
                           if lock_sigma is None or s <= lock_sigma), end)
        count = end - self.first
        self.stride = max(1, math.ceil((count - 2) / (MAX_PREDICTIONS - 3)))

    @torch.no_grad()
    def record(self, kind, latent, step=None):
        if self.reference is None:
            return
        if step is not None:
            last = len(self.sigmas) - 2
            if not self.first <= step <= last:
                return
            if step > self.first + 1 and step != last and (step - self.first - 1) % self.stride:
                return
        reduced = reduce_latent(latent, self.hw)
        if reduced is None or reduced.shape != self.reference.shape:
            return
        if self.scales is None:
            # Keep the feature maps on the reference device for the stage.
            self.scales = []
            for image in (self.reference, F.avg_pool2d(self.reference, 2)):
                gradients = _gradients(image)
                energy = _energy(*gradients)
                self.scales.append((gradients, energy, image.shape[-2:]))
            self.variance = self.reference.var(dim=(-2, -1), unbiased=False).mean(dim=1)
        reduced = reduced.to(self.reference.device)
        sigma = self.sigmas[step] if step is not None else None
        finite = torch.isfinite(reduced).flatten(1).all(dim=1) & torch.isfinite(self.variance)
        has_structure = self.scales[0][1].flatten(1).mean(dim=1) > torch.finfo(torch.float32).eps
        measured = finite & has_structure
        # The reference-grid variance and unattenuated native noise variance
        # give a conservative input-SNR proxy, not a confidence estimate of x0.
        reliable = measured
        snr = torch.full_like(self.variance, float("nan"))
        if sigma is not None and 0. < sigma < 1. and math.isfinite(self.noise_scale) and self.noise_scale > 0.:
            snr = self.variance * ((1. - sigma) / (sigma * self.noise_scale)) ** 2
            reliable = reliable & (snr >= 1.)
        elif sigma is not None:
            reliable = torch.zeros_like(reliable)

        rows = [measured.float()[:, None], reliable.float()[:, None], snr[:, None]]
        previous_maps = []
        for scale, (image, (gradients, energy, hw)) in enumerate(zip(
                (reduced, F.avg_pool2d(reduced, 2)), self.scales)):
            added, missing, relative_energy = _correspondence(gradients, energy, image)
            rows.extend((_region(added, hw), _region(missing, hw), relative_energy[:, None]))
            if step is not None:
                if self.previous is None:
                    persistence = torch.full_like(added, float("nan"))
                else:
                    prior_step, prior_maps, prior_reliable = self.previous
                    adjacent = reliable & prior_reliable & (step == prior_step + 1)
                    persistence = torch.where(adjacent[:, None, None, None],
                                              torch.minimum(added, prior_maps[scale]), float("nan"))
                rows.append(_region(persistence, hw))
                previous_maps.append(added)
            else:
                rows.append(_region(torch.full_like(added, float("nan")), hw))
        if step is not None:
            self.previous = step, previous_maps, reliable
        self.entries.append((kind, step, sigma, torch.cat(rows, dim=1)))

    def report(self):
        if not self.entries:
            status = "grid_too_small" if self.reference is None else "no_measurements"
            return {"grid_hw": list(self.hw), "status": status, "images": []}
        # One compact transfer for a stage. No full-resolution prediction or
        # solver state survives a callback.
        rows = torch.stack([entry[3] for entry in self.entries]).tolist()
        images = []
        for batch_index in range(len(rows[0])):
            candidates = []
            for (kind, step, sigma, _), batch in zip(self.entries, rows):
                measured, reliable, snr, *metrics = batch[batch_index]
                def region(offset):
                    mean, peak, *box = metrics[offset:offset + 6]
                    if not measured or not all(map(math.isfinite, (mean, peak))):
                        return None
                    return {"mean": mean, "peak": peak, "peak_box_xyxy": box if peak > 0. else None}

                scales = []
                for offset in (0, 19):
                    ratio = metrics[offset + 12]
                    scales.append({
                        "added": region(offset), "missing": region(offset + 6),
                        "gradient_energy_ratio": ratio if measured and math.isfinite(ratio) else None,
                        "repeated_added": region(offset + 13),
                    })
                candidates.append({
                    "kind": kind, "step": step, "sigma": sigma, "source_snr_proxy": snr if math.isfinite(snr) else None,
                    "evidence": "usable" if reliable else ("low_snr" if measured else "unavailable"),
                    "fine": scales[0], "coarse": scales[1],
                })
            eligible = [
                c for c in candidates if c["kind"] == "prediction" and c["evidence"] == "usable"
                and all(c[scale][metric] is not None
                        for scale in ("fine", "coarse")
                        for metric in ("added", "missing", "gradient_energy_ratio"))
            ]
            # Prefer the smallest worst-direction loss across both scales;
            # in a tie prefer the lower-noise prediction. Diagnostic only.
            def rank(candidate):
                def loss(scale):
                    return max(
                        scale["added"]["mean"], scale["missing"]["mean"],
                        max(0., 1. - scale["gradient_energy_ratio"]),
                    )
                worst = max(loss(candidate["fine"]), loss(candidate["coarse"]))
                return worst, candidate["sigma"], candidate["step"]

            preferred = min(eligible, default=None, key=rank)
            images.append({"candidates": candidates, "preferred_step": preferred["step"] if preferred else None})
        return {"grid_hw": list(self.hw), "status": "observed", "images": images}
