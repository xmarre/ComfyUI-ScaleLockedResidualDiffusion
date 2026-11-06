"""Krea 2 node widget migration and sampler routing contracts.

Runs without a ComfyUI installation. It evaluates the production node class
with strict stubs, so legacy workflow inputs and fail-before-work behavior are
verified without introducing a second implementation of the node.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _node():
    source = ast.parse((ROOT / "nodes.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef)
               and n.name == "Krea2TurboFlowAlignedRegenerate")

    class Krea2:
        pass

    calls = []
    model = SimpleNamespace(
        model=Krea2(),
        get_model_object=lambda name: SimpleNamespace(shift=1.15),
    )

    def target_hw(latent, megapixels):
        side = int(round(math.sqrt(megapixels * 1_000_000) / 16)) * 2
        return (side, side)

    def calculate_sigmas(model, scheduler, steps, denoise):
        del model, denoise
        if scheduler == "sgm_uniform":
            return torch.tensor([1., .8, .9, 0.])  # invalid: must fail before sampling
        return torch.linspace(1., 0., steps + 1)

    def prefix(values, threshold):
        return sum(1 for v in values[:-1] if v > threshold)

    def run(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output="target", lowres_planner="planner",
                               denoised_output="denoised")

    namespace = {
        "math": math, "torch": torch,
        "comfy": SimpleNamespace(model_base=SimpleNamespace(Krea2=Krea2)),
        "compat_scheduler_names": lambda: ["simple", "normal", "sgm_uniform"],
        "fix_latent_channels": lambda model, latent: latent,
        "latent_target_hw_from_megapixels": target_hw,
        "parse_stage_megapixels": lambda text: (
            [float(x) for x in text.split(",") if x.strip()] if text else []
        ),
        "plan_stage_megapixels": lambda *args: [],
        "full_lock_prefix_length": prefix,
        "calculate_sigmas": calculate_sigmas,
        "ScaleLockConfig": SimpleNamespace,
        "run_scale_locked_ksampler": run,
        "logger": SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT / "nodes.py"),
                 "exec"), namespace)
    node = namespace[cls.name]()
    latent = {"samples": torch.zeros(1, 16, 200, 200)}
    base_args = (model, "positive", "negative", latent, 42, 1.0, 0.0, 0.0)
    return node, calls, base_args


def test_existing_widget_slots_and_explicit_defaults():
    node, calls, args = _node()
    schema = node.INPUT_TYPES()
    optional = list(schema["optional"])
    assert optional == [
        "steps", "composition_lock_sigma", "max_stage_upscale",
        "stage_megapixels", "cfg", "sampler_name", "scheduler",
    ]
    assert schema["optional"]["cfg"][1]["default"] == 1.0
    assert schema["optional"]["sampler_name"][0] == ["euler", "dpmpp_2m"]
    assert schema["optional"]["sampler_name"][1]["default"] == "euler"
    assert schema["optional"]["scheduler"][1]["default"] == "simple"
    assert node.sample(*args) == ("target", "planner", "denoised")
    assert (calls[-1]["cfg"], calls[-1]["sampler_name"],
            calls[-1]["scheduler"]) == (1.0, "euler", "simple")
    assert calls[-1]["stage_hw"] == ()


def test_configured_cfg_sampler_scheduler_route_to_both_passes():
    node, calls, args = _node()
    node.sample(*args, cfg=2.5, sampler_name="dpmpp_2m", scheduler="normal")
    assert (calls[-1]["cfg"], calls[-1]["sampler_name"],
            calls[-1]["scheduler"]) == (2.5, "dpmpp_2m", "normal")
    node.sample(*args, cfg=0.0, sampler_name="euler", scheduler="normal",
                stage_megapixels="2")
    assert calls[-1]["stage_hw"]
    assert calls[-1]["cfg"] == 0.0


@pytest.mark.parametrize("extra, message", [
    ({"cfg": float("nan")}, "CFG must be finite"),
    ({"cfg": float("inf")}, "CFG must be finite"),
    ({"cfg": -1.0}, "CFG must be finite"),
    ({"cfg": 101.0}, "CFG must be finite"),
    ({"sampler_name": "heun"}, "supports Euler or DPM"),
    ({"scheduler": "not_a_scheduler"}, "Unsupported Krea 2 scheduler"),
    ({"scheduler": "sgm_uniform"}, "strictly decreasing"),
    ({"sampler_name": "dpmpp_2m", "stage_megapixels": "2"},
     "resolution ladders require Euler"),
])
def test_invalid_selection_rejected_before_any_sampling(extra, message):
    node, calls, args = _node()
    with pytest.raises(ValueError, match=message):
        node.sample(*args, **extra)
    assert not calls
