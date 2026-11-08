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
        "krea2_native_refine_hw": lambda planner, target: (256, 256),
        "full_lock_prefix_length": prefix,
        "calculate_sigmas": calculate_sigmas,
        "effective_er_sde_sigmas": lambda sigmas, model: sigmas,
        "krea2_er_sde_options": lambda mode, order, eta, noise: {
            "max_stage": order,
            "s_noise": 0.0 if mode == "ODE" or eta == 0 else float(noise),
            "noise_scaler": SimpleNamespace(
                solver_type="ODE" if mode == "ODE" or eta == 0 else mode,
                eta=float(eta),
            ),
        },
        "krea2_sa_solver_options": lambda model, **kwargs: kwargs,
        "sa_native_only_scope": lambda enabled: nullcontext(),
        "ScaleLockConfig": SimpleNamespace,
        "run_scale_locked_ksampler": run,
        "logger": SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
        # resolution.MAX_MEGAPIXELS with ComfyUI's MAX_RESOLUTION of 16384.
        "MAX_MEGAPIXELS": 256.0,
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
        "sa_spectrum_forecasts", "sa_planner_spectrum_forecasts",
        "regeneration_mode", "img2img_start_sigma", "handoff_diagnostics",
        "planner_fidelity_strength", "planner_detail_strength",
        "latent_upscale_method",
    ]
    assert schema["required"]["planner_megapixels"][1]["max"] == 256.0
    assert schema["optional"]["latent_upscale_method"][0] == ["interpolate", "sesqui_wan21"]
    assert schema["optional"]["latent_upscale_method"][1]["default"] == "interpolate"
    assert schema["optional"]["regeneration_mode"][0] == ["trajectory", "planner_final_img2img", "planner_final_upscale_only", "planner_final_native_refine"]
    assert schema["optional"]["regeneration_mode"][1]["default"] == "trajectory"
    assert schema["optional"]["img2img_start_sigma"][1]["default"] == 0.65
    assert schema["optional"]["handoff_diagnostics"][0] == ["off", "log", "log_and_save_latents"]
    assert schema["optional"]["handoff_diagnostics"][1]["default"] == "off"
    assert schema["optional"]["planner_fidelity_strength"][1]["default"] == 0.0
    assert schema["optional"]["planner_detail_strength"][1]["default"] == 0.20
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
    assert schema["optional"]["sa_planner_spectrum_forecasts"][1]["default"] is True
    assert node.sample(*args) == ("target", "planner", "denoised")
    assert (calls[-1]["cfg"], calls[-1]["sampler_name"],
            calls[-1]["scheduler"]) == (1.0, "euler", "simple")
    assert calls[-1]["stage_hw"] == ()
    assert calls[-1]["regeneration_mode"] == "trajectory"
    assert calls[-1]["img2img_start_sigma"] is None
    assert calls[-1]["handoff_diagnostics"] == "off"
    assert calls[-1]["planner_fidelity_strength"] == 0.0
    assert calls[-1]["planner_detail_strength"] == 0.20


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
        "max_stage": 2, "s_noise": .8,
        "noise_scaler": SimpleNamespace(solver_type="Reverse-time SDE", eta=.7),
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



@pytest.mark.parametrize("sampler_name", ["sa_solver", "sa_solver_pece"])
@pytest.mark.parametrize("planner_forecasts", [False, True])
def test_sa_native_planner_control_does_not_change_target_policy(sampler_name, planner_forecasts):
    node, calls, args = _node()
    node.sample(*args, sampler_name=sampler_name, sa_spectrum_forecasts=True,
                sa_planner_spectrum_forecasts=planner_forecasts)
    assert calls[-1]["sa_planner_native_only"] is not planner_forecasts
    assert "sa_planner_spectrum_forecasts" not in calls[-1]["sampler_options"]
    assert "sa_planner_native_only" not in calls[-1]["sampler_options"]
    # The outer native-only control still applies to every stage. Restoring
    # the nested planner flag cannot spuriously re-enable target forecasting.
    node.sample(*args, sampler_name=sampler_name, sa_spectrum_forecasts=False,
                sa_planner_spectrum_forecasts=True)
    assert calls[-1]["sa_planner_native_only"] is False
    # Non-SA sampler behavior must remain independent of the new UI setting.
    node.sample(*args, sampler_name="er_sde", sa_planner_spectrum_forecasts=False)
    assert calls[-1]["sa_planner_native_only"] is False


def test_sa_invalid_planner_forecast_control_rejected_before_any_sampling():
    node, calls, args = _node()
    with pytest.raises(ValueError, match="sa_planner_spectrum_forecasts must be a boolean"):
        node.sample(*args, sampler_name="sa_solver", sa_planner_spectrum_forecasts=1)
    assert not calls



@pytest.mark.parametrize("sampler_name", SAMPLERS)
def test_planner_final_img2img_routes_with_every_solver_and_ignores_lock_sigma(sampler_name):
    node, calls, args = _node()
    node.sample(*args, sampler_name=sampler_name, regeneration_mode="planner_final_img2img",
                img2img_start_sigma=0.4, stage_megapixels="2")
    call = calls[-1]
    assert call["regeneration_mode"] == "planner_final_img2img"
    assert call["img2img_start_sigma"] == 0.4
    assert call["sampler_name"] == sampler_name and call["stage_hw"]
    # A composition lock that would leave no trajectory step is irrelevant here.
    node.sample(*args, sampler_name=sampler_name, regeneration_mode="planner_final_img2img",
                composition_lock_sigma=0.0, steps=2, img2img_start_sigma=0.6, stage_megapixels="2")
    assert calls[-1]["regeneration_mode"] == "planner_final_img2img"


def test_trajectory_mode_ignores_img2img_sigma():
    node, calls, args = _node()
    node.sample(*args, img2img_start_sigma=0.3)
    first = dict(calls[-1])
    node.sample(*args, img2img_start_sigma=0.9)
    assert calls[-1] == first
    assert first["img2img_start_sigma"] is None and first["regeneration_mode"] == "trajectory"


@pytest.mark.parametrize("extra, message", [
    ({"regeneration_mode": "img2img"}, "invalid regeneration_mode"),
    ({"handoff_diagnostics": "verbose"}, "handoff_diagnostics must be"),
    ({"regeneration_mode": "planner_final_img2img", "img2img_start_sigma": 0.0}, "strictly between"),
    ({"regeneration_mode": "planner_final_img2img", "img2img_start_sigma": 1.0}, "strictly between"),
    ({"regeneration_mode": "planner_final_img2img", "img2img_start_sigma": float("nan")}, "strictly between"),
    ({"regeneration_mode": "planner_final_img2img", "img2img_start_sigma": True}, "strictly between"),
    # linspace(1, 0, 9): 0.05 lies below the last nonzero sigma 0.125.
    ({"regeneration_mode": "planner_final_img2img", "img2img_start_sigma": 0.05}, "no usable refinement"),
])
def test_invalid_img2img_configuration_rejected_before_sampling(extra, message):
    node, calls, args = _node()
    with pytest.raises(ValueError, match=message):
        node.sample(*args, **extra)
    assert not calls


@pytest.mark.parametrize("index", [6, 7])
def test_img2img_rejects_nonzero_trajectory_lock_strengths(index):
    node, calls, args = _node()
    args = list(args)
    args[index] = 0.25
    with pytest.raises(ValueError, match="coarse_lock_strength and mid_lock_strength to 0"):
        node.sample(*args, regeneration_mode="planner_final_img2img")
    assert not calls
    node.sample(*args)  # trajectory mode keeps its existing lock-strength behavior
    assert calls[-1]["regeneration_mode"] == "trajectory"


@pytest.mark.parametrize("strength", [0.25, 0.6, 1.0])
def test_disabled_planner_fidelity_rejected_before_sampling(strength):
    node, calls, args = _node()
    with pytest.raises(ValueError, match="disabled after real Krea 2 checkpoint"):
        node.sample(*args, regeneration_mode="planner_final_img2img",
                    planner_fidelity_strength=strength)
    assert not calls


@pytest.mark.parametrize("strength", [-1.0, 1.1, float("nan"), True])
def test_planner_fidelity_invalid_rejected_before_sampling(strength):
    node, calls, args = _node()
    with pytest.raises(ValueError, match="planner_fidelity_strength"):
        node.sample(*args, regeneration_mode="planner_final_img2img",
                    planner_fidelity_strength=strength)
    assert not calls


def test_planner_fidelity_nonzero_rejected_in_trajectory():
    node, calls, args = _node()
    with pytest.raises(ValueError, match="disabled after real Krea 2 checkpoint"):
        node.sample(*args, planner_fidelity_strength=0.5)
    assert not calls


@pytest.mark.parametrize("detail", [0.0, 0.2, 0.5, 1.0])
def test_planner_detail_strength_routes_without_modifying_other_controls(detail):
    node, calls, args = _node()
    node.sample(*args, regeneration_mode="planner_final_img2img",
                planner_fidelity_strength=0.0, planner_detail_strength=detail)
    assert calls[-1]["planner_detail_strength"] == detail
    assert calls[-1]["planner_fidelity_strength"] == 0.0


@pytest.mark.parametrize("detail", [-0.1, 1.1, float("nan"), True])
def test_invalid_planner_detail_rejected_before_sampling(detail):
    node, calls, args = _node()
    with pytest.raises(ValueError, match="planner_detail_strength"):
        node.sample(*args, regeneration_mode="planner_final_img2img",
                    planner_fidelity_strength=0.0, planner_detail_strength=detail)
    assert not calls



def test_latent_upscale_method_defaults_to_interpolate_and_routes():
    node, calls, args = _node()
    node.sample(*args)
    assert calls[-1]["latent_upscale_method"] == "interpolate"
    node.sample(*args, regeneration_mode="planner_final_img2img", latent_upscale_method="sesqui_wan21",
                stage_megapixels="2")
    assert calls[-1]["latent_upscale_method"] == "sesqui_wan21"
    assert calls[-1]["regeneration_mode"] == "planner_final_img2img"


def test_invalid_latent_upscale_method_rejected_before_sampling():
    node, calls, args = _node()
    with pytest.raises(ValueError, match="latent_upscale_method must be"):
        node.sample(*args, latent_upscale_method="bicubic")
    assert not calls


@pytest.mark.parametrize("method", ["interpolate", "sesqui_wan21"])
def test_upscale_only_node_routes_without_target_refinement(method):
    node, calls, args = _node()
    node.sample(*args, regeneration_mode="planner_final_upscale_only",
                latent_upscale_method=method, img2img_start_sigma=0.0)
    assert calls[-1]["regeneration_mode"] == "planner_final_upscale_only"
    assert calls[-1]["latent_upscale_method"] == method
    assert calls[-1]["img2img_start_sigma"] is None


@pytest.mark.parametrize("mode", ["planner_final_img2img", "planner_final_upscale_only"])
def test_planner_final_modes_reject_nonzero_trajectory_band_strengths(mode):
    node, calls, args = _node()
    with pytest.raises(ValueError, match="set coarse_lock_strength and mid_lock_strength to 0"):
        modified = args[:6] + (0.5,) + args[7:]
        node.sample(*modified, regeneration_mode=mode)
    assert not calls


def test_native_refine_mode_uses_one_internal_stage_and_preserves_saved_widget_order():
    node, calls, args = _node()
    args = list(args)
    args[3] = {"samples": torch.zeros(1, 16, 444, 444)}
    node.sample(*args, regeneration_mode="planner_final_native_refine",
                latent_upscale_method="sesqui_wan21", img2img_start_sigma=0.4,
                max_stage_upscale=1.2)
    call = calls[-1]
    assert call["stage_hw"] == ()
    assert call["img2img_start_sigma"] == 0.4
    assert call["regeneration_mode"] == "planner_final_native_refine"
    assert call["latent_upscale_method"] == "sesqui_wan21"


def test_native_refine_rejects_manual_stage_or_wrong_backend_before_sampling():
    node, calls, args = _node()
    args = list(args)
    args[3] = {"samples": torch.zeros(1, 16, 444, 444)}
    with pytest.raises(ValueError, match="requires latent_upscale_method=sesqui_wan21"):
        node.sample(*args, regeneration_mode="planner_final_native_refine")
    with pytest.raises(ValueError, match="clear stage_megapixels"):
        node.sample(*args, regeneration_mode="planner_final_native_refine",
                    latent_upscale_method="sesqui_wan21", stage_megapixels="4")
    assert not calls


@pytest.mark.parametrize("index", [6, 7])
def test_native_refine_rejects_trajectory_band_locks(index):
    node, calls, args = _node()
    args = list(args)
    args[index] = 0.4
    with pytest.raises(ValueError, match="set coarse_lock_strength and mid_lock_strength to 0"):
        node.sample(*args, regeneration_mode="planner_final_native_refine",
                    latent_upscale_method="sesqui_wan21")
    assert not calls
