"""Krea 2 node widget migration and sampler routing contracts.

Runs without a ComfyUI installation. It evaluates the production node class
with strict stubs, so legacy workflow inputs and fail-before-work behavior are
verified without introducing a second implementation of the node.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SAMPLERS = ("euler", "dpmpp_2m", "res_multistep", "deis", "ipndm", "ipndm_v", "gradient_estimation", "er_sde", "sa_solver", "sa_solver_pece")


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
        "krea2_sampler_names": lambda: list(SAMPLERS),
        "fix_latent_channels": lambda model, latent: latent,
        "latent_target_hw_from_megapixels": target_hw,
        "parse_stage_megapixels": lambda text: (
            [float(x) for x in text.split(",") if x.strip()] if text else []
        ),
        "plan_stage_megapixels": lambda *args: [],
        "full_lock_prefix_length": prefix,
        "calculate_sigmas": calculate_sigmas,
        "effective_er_sde_sigmas": lambda sigmas, model: sigmas,
        "krea2_er_sde_options": lambda mode, order, eta, noise: {
            "solver_type": mode, "max_stage": order, "eta": eta, "s_noise": noise,
        },
        "krea2_sa_solver_options": lambda model, **kwargs: kwargs,
        "sa_native_only_scope": lambda enabled: nullcontext(),
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
        "er_sde_solver_type", "er_sde_max_stage", "er_sde_eta", "er_sde_s_noise",
        "sa_eta", "sa_sde_start_percent", "sa_sde_end_percent", "sa_s_noise",
        "sa_predictor_order", "sa_corrector_order", "sa_simple_order_2",
        "sa_spectrum_forecasts",
    ]
    assert schema["optional"]["cfg"][1]["default"] == 1.0
    assert schema["optional"]["sampler_name"][0] == list(SAMPLERS)
    assert schema["optional"]["sampler_name"][1]["default"] == "euler"
    assert schema["optional"]["scheduler"][1]["default"] == "simple"
    assert schema["optional"]["er_sde_solver_type"][1]["default"] == "ER-SDE"
    assert schema["optional"]["er_sde_max_stage"][1]["default"] == 3
    assert schema["optional"]["er_sde_eta"][1]["default"] == 1.0
    assert schema["optional"]["er_sde_s_noise"][1]["default"] == 1.0
    assert schema["optional"]["sa_eta"][1]["default"] == 1.0
    assert schema["optional"]["sa_sde_start_percent"][1]["default"] == 0.2
    assert schema["optional"]["sa_sde_end_percent"][1]["default"] == 0.35
    assert schema["optional"]["sa_predictor_order"][1]["default"] == 3
    assert schema["optional"]["sa_corrector_order"][1]["default"] == 4
    assert schema["optional"]["sa_simple_order_2"][1]["default"] is False
    assert schema["optional"]["sa_spectrum_forecasts"][1]["default"] is True
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


@pytest.mark.parametrize("sampler_name", SAMPLERS)
def test_every_offered_solver_routes_single_stage_and_resolution_ladders(sampler_name):
    node, calls, args = _node()
    node.sample(*args, sampler_name=sampler_name, max_stage_upscale=0.0)
    assert calls[-1]["sampler_name"] == sampler_name and calls[-1]["stage_hw"] == ()
    node.sample(*args, sampler_name=sampler_name, stage_megapixels="2")
    assert calls[-1]["sampler_name"] == sampler_name and calls[-1]["stage_hw"]


@pytest.mark.parametrize("extra, message", [
    ({"cfg": float("nan")}, "CFG must be finite"),
    ({"cfg": float("inf")}, "CFG must be finite"),
    ({"cfg": -1.0}, "CFG must be finite"),
    ({"cfg": 101.0}, "CFG must be finite"),
    ({"sampler_name": "heun"}, "Unsupported Krea 2 sampler"),
    ({"sampler_name": "dpmpp_2m_cfg_pp"}, "Unsupported Krea 2 sampler"),
    ({"scheduler": "not_a_scheduler"}, "Unsupported Krea 2 scheduler"),
    ({"scheduler": "sgm_uniform"}, "strictly decreasing"),
])
def test_invalid_selection_rejected_before_any_sampling(extra, message):
    node, calls, args = _node()
    with pytest.raises(ValueError, match=message):
        node.sample(*args, **extra)
    assert not calls


def test_er_sde_options_are_routed_without_changing_other_solvers():
    node, calls, args = _node()
    node.sample(*args, sampler_name="er_sde", er_sde_solver_type="Reverse-time SDE",
                er_sde_max_stage=2, er_sde_eta=.7, er_sde_s_noise=.8,
                stage_megapixels="2")
    assert calls[-1]["sampler_options"] == {
        "solver_type": "Reverse-time SDE", "max_stage": 2,
        "eta": .7, "s_noise": .8,
    }
    assert calls[-1]["stage_hw"]
    node.sample(*args, sampler_name="euler")
    assert calls[-1]["sampler_options"] is None



@pytest.mark.parametrize("sampler_name", ["sa_solver", "sa_solver_pece"])
def test_sa_solver_optional_controls_are_routed_to_native_sampler(sampler_name):
    node, calls, args = _node()
    node.sample(
        *args, sampler_name=sampler_name, sa_eta=.45,
        sa_sde_start_percent=.1, sa_sde_end_percent=.9,
        sa_s_noise=.75, sa_predictor_order=5, sa_corrector_order=2,
        sa_simple_order_2=True,
    )
    assert calls[-1]["sampler_name"] == sampler_name
    assert calls[-1]["sampler_options"] == {
        "eta": .45, "sde_start_percent": .1, "sde_end_percent": .9,
        "s_noise": .75, "predictor_order": 5, "corrector_order": 2,
        "simple_order_2": True,
    }
    node.sample(*args, sampler_name="euler")
    assert calls[-1]["sampler_options"] is None



@pytest.mark.parametrize("sampler_name", ["sa_solver", "sa_solver_pece"])
@pytest.mark.parametrize("forecasts", [False, True])
def test_sa_forecast_option_is_local_to_selected_sampler(sampler_name, forecasts):
    node, calls, args = _node()
    node.sample(*args, sampler_name=sampler_name, sa_spectrum_forecasts=forecasts)
    assert calls[-1]["sampler_name"] == sampler_name
    assert "sa_spectrum_forecasts" not in calls[-1]["sampler_options"]
    # A future model path must never inherit the previous SA runtime toggle.
    node.sample(*args, sampler_name="euler", sa_spectrum_forecasts=forecasts)
    assert calls[-1]["sampler_name"] == "euler"
    assert calls[-1]["sampler_options"] is None


def test_sa_invalid_forecast_control_fails_before_sampling():
    node, calls, args = _node()
    with pytest.raises(ValueError, match="sa_spectrum_forecasts must be a boolean"):
        node.sample(*args, sampler_name="sa_solver", sa_spectrum_forecasts=1)
    assert not calls
