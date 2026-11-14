"""Numerical contracts for automatic completed-latent restarts."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "slrd_auto_refinement_test", Path(__file__).resolve().parents[1] / "auto_refinement.py"
)
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)

SCHEDULE = [1.0, .82, .65, .5, .4, .3, .2, .1, 0.0]
RELEASE = 3


def make_source(edge, *, interior=2.0, shape=(16, 24)):
    """Prescribe cosine amplitudes in the measured source frequency bands."""
    h, w = shape
    x = torch.arange(w, dtype=torch.float64) * (2 * torch.pi / w)
    y = torch.arange(h, dtype=torch.float64)[:, None] * (2 * torch.pi / h)
    return (interior * (x.cos() + y.cos()) +
            edge * ((x * (w // 3)).cos() + (y * (h // 3)).cos()))[None, None]


def choose(source, *, target=(24, 32), schedule=SCHEDULE, roundtrip=None, noise_scale=1.0, **options):
    return policy.choose_refinement_restart(
        schedule, RELEASE, source, source if roundtrip is None else roundtrip, target, noise_scale, **options,
    )


def test_content_and_roundtrip_control_native_suffix():
    smooth = make_source(.15)
    detailed = make_source(.35)
    a = choose(smooth)
    b = choose(detailed)
    c = choose(smooth, roundtrip=smooth + make_source(.0, interior=1.5))
    assert a.reason == b.reason == "reduced"
    assert a.index > b.index > RELEASE
    assert a.index > c.index >= RELEASE
    assert a.detail_sigma < b.detail_sigma < c.detail_sigma
    assert SCHEDULE[a.index] >= a.detail_sigma > SCHEDULE[a.index + 1]
    assert SCHEDULE[a.index] <= a.structure_sigma


def test_resolution_and_noise_scale_change_snr_with_noninteger_rectangular_growth():
    source = make_source(.32)
    a = choose(source, target=(24, 28))
    b = choose(source, target=(32, 38))
    c = choose(source, target=(32, 38), noise_scale=2.0)
    assert a.edge_power * (32 * 38) == pytest.approx(b.edge_power * (24 * 28))
    assert b.detail_sigma > a.detail_sigma
    assert b.index <= a.index
    assert c.detail_sigma == pytest.approx(b.detail_sigma / (2 - b.detail_sigma))
    assert c.index > b.index
    assert choose(source.unsqueeze(2), target=(24, 28)) == a


def test_only_growing_axis_contributes_edge_power():
    source = make_source(.15)
    h, w = source.shape[-2:]
    x = torch.arange(w, dtype=torch.float64) * (2 * torch.pi / w)
    width_detail = (x * (w // 3)).cos().view(1, 1, 1, w)
    a = choose(source, target=(24, 24))
    assert choose(source + width_detail, target=(24, 24)).edge_power == pytest.approx(a.edge_power)
    assert choose(source + width_detail, target=(24, 32)).detail_sigma > a.detail_sigma


def test_batch_uses_strongest_detail_and_weakest_interior():
    smooth = make_source(.15)
    detailed = make_source(.35)
    assert choose(torch.cat((smooth, detailed))).index == choose(detailed).index
    a = choose(torch.cat((smooth, make_source(.35, interior=.03))))
    assert a.index == RELEASE
    assert a.reason == "no_native_step_separates_bands"


def test_discrete_rounding_and_short_schedule_fallback():
    source = make_source(.15)
    short = [1., .82, .65, .5, .2, 0.]
    assert choose(source, schedule=short).reason == "no_shorter_two_step_suffix"
    long = torch.linspace(1.0, 0.0, 1001).tolist()
    a = choose(source, schedule=long)
    assert long[a.index] >= a.detail_sigma > long[a.index + 1]
    assert len(long[a.index:]) >= 3
    assert choose(make_source(1.1)).index == RELEASE


@pytest.mark.parametrize("prefix", [[], [.71, .65]])
def test_retimes_coarse_two_step_floor_in_half_log_snr(prefix):
    first, last = .5582663416862488, .34492799639701843
    source = make_source(.16, interior=.28)
    schedule = [1., .9, .8, *prefix, first, last, 0.]
    decision = choose(source, schedule=schedule, retime_sigma_min=.001)
    assert decision.reason == "retimed_two_step_suffix"
    assert decision.index == len(schedule) - 3
    start, terminal, zero = decision.retimed_sigmas
    odds = lambda s: s / (1 - s)
    assert decision.detail_sigma < start < decision.structure_sigma < first
    assert 0. < terminal < last and zero == 0.
    assert odds(start) / odds(terminal) == pytest.approx(odds(first) / odds(last))
    assert odds(start) ** 2 == pytest.approx(odds(decision.detail_sigma) * odds(decision.structure_sigma))
    assert choose(source * .7, schedule=schedule, retime_sigma_min=.001).retimed_sigmas[0] < start


def test_retime_keeps_native_fallback_for_unsafe_evidence_and_one_step():
    schedule = [1., .9, .8, .55, .34, 0.]
    source = make_source(.16, interior=.28)
    options = dict(schedule=schedule, retime_sigma_min=.001)
    assert choose(make_source(.16, interior=.06), **options).retimed_sigmas is None
    assert choose(torch.zeros_like(source), **options).retimed_sigmas is None
    assert choose(make_source(.16, interior=2.), **options).reason == "native_two_step_suffix"
    assert choose(source, schedule=schedule, retime_sigma_min=.3).reason == "no_safe_two_step_retime"
    assert choose(source, schedule=[1., .9, .8, .55, 0.],
                  retime_sigma_min=.001).reason == "no_shorter_two_step_suffix"


def test_retime_checks_model_minimum_and_rounded_knots():
    result = policy._retime_two_steps(.15, .52, .55826634, .344928, .25, torch.float32)
    assert result is not None and result[1] >= .25
    assert policy._retime_two_steps(.15, .40, .55826634, .344928, .25, torch.float32) is None
    assert policy._retime_two_steps(.1, .4, .558, .55799999, .001, torch.float32) is None


@pytest.mark.parametrize("shape", [(16, 24), (14, 28), (18, 22)])
def test_two_step_floor_avoids_terminal_only_update(shape):
    a = choose(make_source(.001, shape=shape), target=(26, 38))
    assert a.index == len(SCHEDULE) - 3
    assert a.detail_sigma < SCHEDULE[-2]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_evidence_is_rejected(value):
    source = make_source(.15)
    bad = source.clone()
    bad[0, 0, 0, 0] = value
    for a in (choose(bad, roundtrip=source), choose(source, roundtrip=bad)):
        assert a.reason == "invalid_spectral_measurement" and a.index == RELEASE


def test_degenerate_input_and_invalid_configuration_keep_release():
    source = make_source(.15)
    cases = [
        (choose(torch.zeros_like(source)), "insufficient_spectral_evidence"),
        (choose(torch.ones_like(source)), "insufficient_spectral_evidence"),
        (choose(source[:0]), "unsupported_latent_geometry"),
        (choose(source, target=(16, 24)), "invalid_stage_geometry"),
        (choose(source, target=(14, 28)), "invalid_stage_geometry"),
        (choose(source, noise_scale=0.0), "invalid_noise_scale"),
        (choose(source, schedule=[1., .82, .65, .5, .5, .2, 0.]), "invalid_native_suffix"),
        (choose(source, schedule=[1., .82, .65, .5, .3, float("nan"), 0.]), "invalid_native_suffix"),
    ]
    for decision, reason in cases:
        assert decision.reason == reason and decision.index == RELEASE
