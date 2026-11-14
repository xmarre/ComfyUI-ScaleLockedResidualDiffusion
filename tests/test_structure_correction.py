"""Known-geometry controls for completed-stage candidate selection."""

import importlib.util
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "slrd_structure_correction_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.structure_correction", ROOT / "structure_correction.py")
correction = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = correction
spec.loader.exec_module(correction)


def scene():
    reference = torch.zeros(1, 3, 48, 64)
    reference[:, :, 12:36, 8:24] = 1
    early = F.interpolate(reference, scale_factor=2, mode="nearest")
    texture = (torch.arange(96)[:, None] + torch.arange(128)[None, :]).remainder(2) * .125
    early = early + texture
    end = early.clone()
    end[..., 40:60, 84:104] += 1
    return reference, early, end


def choose(reference, earlier, terminal):
    return correction.select_structure_candidate(reference, earlier, terminal, lambda x: x)


def test_inserted_geometry_is_rejected_with_existing_detail_retained():
    reference, early, end = scene()
    selected, receipt = choose(reference, early, end)
    assert selected == [0], receipt
    assert torch.equal(early[..., :36, :36], end[..., :36, :36])
    assert not torch.equal(early, end)
    assert end[..., 42:58, 86:102].mean() > early[..., 42:58, 86:102].mean() + .9


def test_fine_texture_alone_preserves_the_terminal_candidate():
    reference, early, _ = scene()
    smooth = F.interpolate(reference, scale_factor=2, mode="nearest")
    assert choose(reference, smooth, early)[0] == []


def test_rejects_earlier_candidate_with_lower_typical_detail():
    reference, early, end = scene()
    smooth = F.interpolate(reference, scale_factor=2, mode="nearest")
    selected, receipt = choose(reference, smooth, end)
    assert selected == []
    assert receipt[0]["reason"] == "retained_native_detail"


def test_rejects_sacrificing_a_sharpened_region_away_from_the_added_shape():
    reference, early, end = scene()
    y, x = torch.meshgrid(torch.arange(24), torch.arange(24), indexing="ij")
    end[..., 4:28, 4:28] += ((x + y).remainder(2) - .5) * .5
    selected, receipt = choose(reference, early, end)
    assert selected == []
    assert receipt[0]["reason"] == "retained_native_detail"


def test_persistent_addition_and_small_translation_are_not_treated_as_late_geometry():
    reference, early, end = scene()
    assert choose(reference, end, end)[0] == []
    assert choose(reference, early, early.roll(2, -1))[0] == []


def test_earlier_prediction_must_preserve_source_edges():
    reference, early, end = scene()
    early[..., 18:80, :54].remainder_(.25)
    assert choose(reference, early, end)[0] == []


def test_batched_affine_layout_keeps_invalid_items_separate():
    reference, early, end = scene()
    batched = torch.cat((early, end, early))
    native = torch.cat((end, end, end))
    native[-1, 0, 0, 0] = float("nan")
    native = (2 * native + 3).unsqueeze(2)
    earlier = batched.unsqueeze(2)
    selected, _ = correction.select_structure_candidate(
        reference.expand(3, -1, -1, -1), earlier, native, lambda x: (x - 3) / 2,
    )
    assert selected == [0]
    assert torch.equal(earlier.squeeze(2), batched)
    assert torch.isnan(native[-1, 0, 0, 0, 0])


def test_no_candidate_or_no_reference_evidence():
    reference, early, end = scene()
    assert choose(reference, None, end) == ([], [])
    assert choose(torch.zeros_like(reference), early, end)[0] == []
