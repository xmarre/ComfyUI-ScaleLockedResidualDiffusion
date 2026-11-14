"""Correspondence on controlled geometry and solver evidence eligibility."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    "slrd_structure_screening_test", Path(__file__).resolve().parents[1] / "structure_screening.py"
)
screening = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screening)


def scene(h=64, w=96, dtype=torch.float32, device="cpu"):
    image = torch.zeros(1, 3, h, w, dtype=dtype, device=device)
    image[:, :, h // 4:h * 3 // 4, w // 8:w // 3] = 1
    image[:, 0, h // 3:h // 2, w // 4:w // 2] = -0.5
    return image


def screen(source, candidates, sigmas, lock=None):
    observer = screening.StageStructureScreen(source, sigmas, lock, 1.)
    for step, candidate in enumerate(candidates):
        observer.record("prediction", candidate, step)
    return observer.report()


@pytest.mark.parametrize("hw", [(64, 96), (78, 54)])
def test_new_shape_scores_above_texture_and_small_displacement(hw):
    source = scene(*hw)
    h, w = hw
    enlarged = F.interpolate(source, scale_factor=2, mode="nearest")
    checker = ((torch.arange(h * 2)[:, None] + torch.arange(w * 2)[None, :]) % 2 * 2 - 1)
    fine_texture = enlarged + 0.25 * checker
    shifted = source.roll(shifts=(1, -1), dims=(-2, -1))
    artifact = source.clone()
    artifact[:, :, h // 3:h * 2 // 3, w * 2 // 3:w * 5 // 6] = 1.
    results = screen(source, [fine_texture, shifted, artifact], [.2, .1, .05, 0.])["images"][0]
    fine, displacement, wrong = results["candidates"]
    for scale in ("fine", "coarse"):
        assert fine[scale]["added"]["peak"] < wrong[scale]["added"]["peak"]
        assert displacement[scale]["added"]["mean"] < wrong[scale]["added"]["mean"]
        x1, y1, x2, y2 = wrong[scale]["added"]["peak_box_xyxy"]
        assert x1 >= .52 and x2 <= 1. and y1 < .8 and y2 > .25
        assert wrong[scale]["added"]["peak"] > wrong[scale]["added"]["mean"]
    assert results["preferred_step"] == 0


def test_shape_removal_and_duplicate_have_different_directions():
    source = scene()
    missing = torch.zeros_like(source)
    added = source + source.roll(48, -1)
    values = screen(source, [missing, added], [.1, .05, 0.])["images"][0]["candidates"]
    for scale in ("fine", "coarse"):
        assert values[0][scale]["missing"]["peak"] > values[0][scale]["added"]["peak"]
        assert values[1][scale]["added"]["peak"] > values[1][scale]["missing"]["peak"]


def test_two_step_candidates_report_repeated_change_only_when_both_have_evidence():
    source = scene()
    added = source + source.roll(48, -1)
    values = screen(source, [added, added], [.17, .1, 0.])["images"][0]
    first, second = values["candidates"]
    assert first["fine"]["repeated_added"] is None
    assert second["fine"]["repeated_added"] == second["fine"]["added"]
    values = screen(source, [added, added], [.8, .1, 0.])["images"][0]
    assert [x["evidence"] for x in values["candidates"]] == ["low_snr", "usable"]
    assert values["candidates"][1]["fine"]["repeated_added"] is None
    assert values["preferred_step"] == 1


def test_preferred_prediction_does_not_reward_contrast_erasure():
    source = scene()
    altered = source + .25 * source.roll(48, -1)
    blurred = source * .10
    values = screen(source, [altered, blurred], [.15, .05, 0.])["images"][0]
    assert values["preferred_step"] == 0
    assert values["candidates"][1]["fine"]["gradient_energy_ratio"] == pytest.approx(.01)


def test_blank_near_constant_invalid_and_textured_batch_items_are_independent():
    clean = scene()
    texture = torch.randn(clean.shape, generator=torch.Generator().manual_seed(34))
    source = torch.cat((torch.zeros_like(clean), clean * 1e-7, texture, clean, clean))
    prediction = source.clone()
    prediction[-1, 0, 0, 0] = float("inf")
    results = screen(source, [prediction], [.1, 0.])
    assert [im["candidates"][0]["evidence"] for im in results["images"]] == [
        "unavailable", "unavailable", "usable", "usable", "unavailable",
    ]
    assert results["images"][2]["candidates"][0]["fine"]["added"]["mean"] < 1e-6
    json.dumps(results, allow_nan=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_dtypes_devices_and_observers_do_not_mutate_or_share_state(dtype, device):
    source = scene(dtype=dtype, device=device)
    candidate = source + source.roll(48, -1)
    before = candidate.clone()
    left = screening.StageStructureScreen(source.float(), [.15, .1, 0.], None, 1.)
    right = screening.StageStructureScreen(source.float(), [.15, .1, 0.], None, 1.)
    left.record("prediction", candidate.unsqueeze(2), 0)
    right.record("prediction", source.unsqueeze(2), 0)
    left.record("prediction", candidate.unsqueeze(2), 1)
    assert torch.equal(candidate, before)
    assert len(left.report()["images"][0]["candidates"]) == 2
    assert right.report()["images"][0]["candidates"][0]["fine"]["added"]["peak"] < 1e-6


def test_geometry_and_long_schedule_bound():
    assert screening.analysis_hw((628, 418), (800, 560)) == (128, 85)
    assert screening.analysis_hw((128, 96), (32, 64)) == (32, 24)
    small = torch.zeros(1, 3, 12, 16)
    assert screen(small, [small], [.1, 0.])["status"] == "grid_too_small"
    source = scene(24, 32)
    sigmas = torch.linspace(1, 0, 257).tolist()
    receipt = screen(source, [source] * 256, sigmas, lock=.75)["images"][0]["candidates"]
    assert [c["step"] for c in receipt[:2]] == [64, 65]
    assert receipt[-1]["step"] == 255
    assert len(receipt) <= screening.MAX_PREDICTIONS
    for previous, current in zip(receipt, receipt[1:]):
        if current["step"] > previous["step"] + 1:
            assert current["fine"]["repeated_added"] is None
