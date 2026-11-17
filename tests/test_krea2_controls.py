"""Krea 2 node widget migration and sampler routing contracts.

Runs without a ComfyUI installation. It evaluates the production node class
with strict stubs, so legacy workflow inputs and fail-before-work behavior are
verified without introducing a second implementation of the node.
"""
from __future__ import annotations

import ast
import importlib.util
from contextlib import nullcontext
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
_SPECTRAL_SPEC = importlib.util.spec_from_file_location("slrd_controls_spectral", ROOT / "spectral_progressive.py")
spectral = importlib.util.module_from_spec(_SPECTRAL_SPEC)
_SPECTRAL_SPEC.loader.exec_module(spectral)
_FLOW_SPEC = importlib.util.spec_from_file_location("slrd_controls_krea2_flow", ROOT / "krea2_flow.py")
krea2_flow = importlib.util.module_from_spec(_FLOW_SPEC)
_FLOW_SPEC.loader.exec_module(krea2_flow)
REGENERATION_MODES = ("trajectory", "planner_final_img2img", "planner_final_upscale_only",
                      "planner_final_native_refine", "spectral_progressive", "trajectory_auto", "trajectory_auto_native")
SAMPLERS = ("euler", "dpmpp_2m", "res_multistep", "deis", "ipndm", "ipndm_v", "gradient_estimation", "er_sde", "sa_solver", "sa_solver_pece")


def _node(native_first=None, planned=()):
    source = ast.parse((ROOT / "nodes.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef)
               and n.name == "Krea2TurboFlowAlignedRegenerate")
    # Module-level definitions the production class reads at call time.
    support = [
        n for n in source.body
        if (isinstance(n, ast.FunctionDef) and n.name == "_without_patch_lattice")
        or (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PATCH_LATTICE_CORRECTIONS" for t in n.targets))
    ]
    assert len(support) == 2

    class Krea2:
        pass

    calls = []
    results = []
    log_lines = []
    planning = []
    native_calls = []
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
        if results:
            return results[-1]
        return SimpleNamespace(output="target", lowres_planner="planner",
                               denoised_output="denoised")

    namespace = {
        "math": math, "torch": torch,
        "comfy": SimpleNamespace(model_base=SimpleNamespace(Krea2=Krea2)),
        "compat_scheduler_names": lambda: ["simple", "normal", "sgm_uniform"],
        "krea2_sampler_names": lambda: list(SAMPLERS),
        "REGENERATION_MODES": REGENERATION_MODES,
        "fix_latent_channels": lambda model, latent: latent,
        "latent_target_hw_from_megapixels": target_hw,
        "parse_stage_megapixels": lambda text: (
            [float(x) for x in text.split(",") if x.strip()] if text else []
        ),
        "plan_stage_megapixels": lambda *args: (planning.append(args), list(planned))[1],
        "krea2_native_refine_hw": lambda planner, target: (256, 256),
        "krea2_native_first_stage_hw": lambda planner, target, cap: (
            native_calls.append((planner, target, cap)), native_first
        )[1],
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
        "parse_transition_sigmas": spectral.parse_transition_sigmas,
        "logger": SimpleNamespace(info=lambda *args: log_lines.append(args), warning=lambda *args: None),
        "patch_lattice_metrics": krea2_flow.patch_lattice_metrics,
        "remove_patch_lattice": krea2_flow.remove_patch_lattice,
        # resolution.MAX_MEGAPIXELS on the native 16384-pixel canvas.
        "MAX_MEGAPIXELS": 256.0,
    }
    exec(compile(ast.Module(body=support + [cls], type_ignores=[]), str(ROOT / "nodes.py"),
                 "exec"), namespace)
    node = namespace[cls.name]()
    node.results = results
    node.log_lines = log_lines
    latent = {"samples": torch.zeros(1, 16, 200, 200)}
    base_args = (model, "positive", "negative", latent, 42, 1.0, 0.0, 0.0)
    node.planning = planning
    node.native_calls = native_calls
    return node, calls, base_args


def _calibration_node():
    # Run the production calibration node with a strict model-format stub.
    # Invalid inputs must fail before process_latent_in or the spectral fit.
    source = ast.parse((ROOT / "nodes.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef)
               and n.name == "Krea2SpectralCalibration")

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid calibration input reached model processing")

    namespace = {
        "torch": torch,
        "CALIBRATION_TYPE": spectral.CALIBRATION_TYPE,
        "build_calibration": unexpected,
        "describe_calibration": unexpected,
        "logger": SimpleNamespace(info=unexpected),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT / "nodes.py"),
                 "exec"), namespace)
    model = SimpleNamespace(model=SimpleNamespace(
        latent_format=SimpleNamespace(latent_channels=16),
        process_latent_in=unexpected,
    ))
    return namespace[cls.name](), model


@pytest.mark.parametrize("channels", [4, 16])
def test_spectral_calibration_rejects_empty_generation_canvas(channels):
    node, model = _calibration_node()
    with pytest.raises(ValueError, match="empty latent canvas.*VAE Encode"):
        node.calibrate(model, {"samples": torch.zeros(1, channels, 64, 80)})


def test_spectral_calibration_reports_actual_and_expected_channels():
    node, model = _calibration_node()
    with pytest.raises(ValueError, match="expected 16 latent channels, received 4.*VAE Encode"):
        node.calibrate(model, {"samples": torch.ones(1, 4, 64, 80)})


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
        "progressive_transition_sigmas", "spectral_tolerance", "spectral_calibration",
        "stage_planning", "highres_min_step_fraction",
        "patch_lattice_correction",
    ]
    assert schema["optional"]["patch_lattice_correction"][0] == ["off", "final_latent"]
    assert schema["optional"]["patch_lattice_correction"][1]["default"] == "off"
    assert schema["optional"]["latent_upscale_method"][0] == ["interpolate", "sesqui_wan21"]
    assert schema["optional"]["latent_upscale_method"][1]["default"] == "interpolate"
    assert schema["optional"]["regeneration_mode"][0] == list(REGENERATION_MODES)
    assert schema["optional"]["progressive_transition_sigmas"][1]["default"] == ""
    assert schema["optional"]["spectral_tolerance"][1]["default"] == 0.01
    assert schema["optional"]["spectral_calibration"][0] == "SLRD_SPECTRUM"
    assert schema["optional"]["stage_planning"][0] == ["manual", "auto_two_passes"]
    assert schema["optional"]["stage_planning"][1]["default"] == "manual"
    assert schema["optional"]["highres_min_step_fraction"][1]["default"] == 0.25
    assert schema["optional"]["regeneration_mode"][1]["default"] == "trajectory"
    assert schema["optional"]["steps"][1]["default"] == 12
    assert schema["optional"]["composition_lock_sigma"][1]["default"] == 0.65
    assert schema["optional"]["max_stage_upscale"][1]["default"] == 4.0
    assert schema["required"]["planner_megapixels"][1]["default"] == 1.0
    assert schema["required"]["planner_megapixels"][1]["max"] == 256.0
    assert "4096x4096" in schema["required"]["planner_megapixels"][1]["tooltip"]
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



@pytest.mark.parametrize("sampler_name", SAMPLERS)
@pytest.mark.parametrize("planner_forecasts", [False, True])
def test_planner_native_control_applies_to_every_sampler(sampler_name, planner_forecasts):
    node, calls, args = _node()
    node.sample(*args, sampler_name=sampler_name, sa_planner_spectrum_forecasts=planner_forecasts)
    assert calls[-1]["sa_planner_native_only"] is not planner_forecasts
    assert "sa_planner_spectrum_forecasts" not in (calls[-1]["sampler_options"] or {})
    assert "sa_planner_native_only" not in (calls[-1]["sampler_options"] or {})
    assert "All Spectrum-supported samplers" in (
        node.INPUT_TYPES()["optional"]["sa_planner_spectrum_forecasts"][1]["tooltip"]
    )
    # A following invocation with the default enabled policy must not
    # retain the previous planner's native-only selection.
    node.sample(*args, sampler_name=sampler_name)
    assert calls[-1]["sa_planner_native_only"] is False


@pytest.mark.parametrize("sampler_name", ["sa_solver", "sa_solver_pece"])
def test_sa_full_invocation_native_control_remains_independent_of_planner(sampler_name):
    node, calls, args = _node()
    node.sample(*args, sampler_name=sampler_name, sa_spectrum_forecasts=False,
                sa_planner_spectrum_forecasts=True)
    assert calls[-1]["sa_planner_native_only"] is False




def test_native_spectrum_scopes_preserve_sa_only_and_planner_all_sampler_separation(monkeypatch):
    import contextvars
    import sys
    from contextlib import contextmanager

    source = ast.parse((ROOT / "slrd_runtime.py").read_text())
    functions = [
        n for n in source.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_spectrum_force_native_vars", "sa_native_only_scope")
    ]
    assert len(functions) == 2
    env = {
        "contextvars": contextvars, "sys": sys,
        "contextmanager": contextmanager,
        "_SPECTRUM_RUNTIME": "comfyui_spectrum_krea2.runtime",
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 str(ROOT / "slrd_runtime.py"), "exec"), env)
    native_scope = env["sa_native_only_scope"]
    sa_flag = contextvars.ContextVar("test_sa_native_only", default=False)
    planner_flag = contextvars.ContextVar("test_all_sampler_native_only", default=False)
    monkeypatch.setitem(
        sys.modules, "test_loaded.comfyui_spectrum_krea2.runtime",
        SimpleNamespace(FORCE_NATIVE=sa_flag, FORCE_NATIVE_ALL_SAMPLERS=planner_flag)
    )

    with native_scope(True, all_samplers=True):
        assert planner_flag.get() is True and sa_flag.get() is False
        with native_scope(True):
            assert planner_flag.get() is True and sa_flag.get() is True
        assert planner_flag.get() is True and sa_flag.get() is False
    assert planner_flag.get() is False and sa_flag.get() is False

    with pytest.raises(RuntimeError, match="interrupted"):
        with native_scope(True, all_samplers=True):
            assert planner_flag.get() is True
            raise RuntimeError("interrupted")
    assert planner_flag.get() is False and sa_flag.get() is False


def test_planner_native_scope_rejects_stale_spectrum_runtime(monkeypatch):
    import contextvars
    import sys
    from contextlib import contextmanager

    source = ast.parse((ROOT / "slrd_runtime.py").read_text())
    functions = [
        n for n in source.body if isinstance(n, ast.FunctionDef)
        and n.name in ("_spectrum_force_native_vars", "sa_native_only_scope")
    ]
    env = {
        "contextvars": contextvars, "sys": sys,
        "contextmanager": contextmanager,
        "_SPECTRUM_RUNTIME": "comfyui_spectrum_krea2.runtime",
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 str(ROOT / "slrd_runtime.py"), "exec"), env)
    monkeypatch.setitem(
        sys.modules, "test_legacy.comfyui_spectrum_krea2.runtime",
        SimpleNamespace(FORCE_NATIVE=contextvars.ContextVar("old_sa_only", default=False))
    )
    with pytest.raises(RuntimeError, match="FORCE_NATIVE_ALL_SAMPLERS"):
        with env["sa_native_only_scope"](True, all_samplers=True):
            raise AssertionError("stale Spectrum runtime must be rejected")


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
    ({"regeneration_mode": "img2img"}, "regeneration_mode must be one of"),
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


def test_staged_upscale_only_ignores_composition_lock():
    # planner_final_upscale_only never uses the composition lock, so a lock
    # value that would leave no released step must not reject staged runs.
    node, calls, args = _node()
    node.sample(*args, regeneration_mode="planner_final_upscale_only",
                composition_lock_sigma=0.0, stage_megapixels="2")
    assert calls[-1]["regeneration_mode"] == "planner_final_upscale_only"
    assert calls[-1]["stage_hw"]


def test_staged_trajectory_rejects_lock_without_a_locked_prefix():
    # At composition_lock_sigma=1.0 later stages would restart at sigma 1
    # and discard the enlarged previous stage.
    node, calls, args = _node()
    with pytest.raises(ValueError, match="below the first schedule sigma"):
        node.sample(*args, composition_lock_sigma=1.0, stage_megapixels="2")
    assert not calls
    node.sample(*args, composition_lock_sigma=1.0, max_stage_upscale=0.0)
    assert calls[-1]["stage_hw"] == ()


def test_saved_workflows_keep_defaults_and_disable_progressive_inputs():
    node, calls, args = _node()
    node.sample(*args)
    call = calls[-1]
    assert call["regeneration_mode"] == "trajectory"
    assert call["steps"] == 12
    assert call["config"].full_lock_sigma == 0.65
    assert call["progressive_transition_sigmas"] == []
    assert call["spectral_calibration"] is None
    assert call["spectral_tolerance"] == 0.01



@pytest.mark.parametrize("mode", [
    "trajectory", "planner_final_img2img", "planner_final_upscale_only",
    "planner_final_native_refine",
])
@pytest.mark.parametrize("saved_text", ["0.80, 0.65", "0.50, 0.70", "not-a-schedule"])
def test_other_modes_ignore_stored_spectral_transition_values(mode, saved_text):
    node, calls, args = _node()
    args = list(args)
    kwargs = {"regeneration_mode": mode, "progressive_transition_sigmas": saved_text}
    if mode == "planner_final_native_refine":
        args[3] = {"samples": torch.zeros(1, 16, 444, 444)}
        kwargs["latent_upscale_method"] = "sesqui_wan21"
    node.sample(*args, **kwargs)
    assert calls[-1]["regeneration_mode"] == mode
    assert calls[-1]["progressive_transition_sigmas"] == []


def test_saved_spectral_transition_schedule_reactivates_when_mode_changes_back():
    node, calls, args = _node()
    saved_text = "0.80, 0.65"
    node.sample(*args, regeneration_mode="trajectory", progressive_transition_sigmas=saved_text)
    assert calls[-1]["progressive_transition_sigmas"] == []
    node.sample(*args, regeneration_mode="spectral_progressive",
                stage_megapixels="2", progressive_transition_sigmas=saved_text)
    assert calls[-1]["progressive_transition_sigmas"] == [0.8, 0.65]


def test_spectral_progressive_routes_parsed_sigmas_and_calibration():
    node, calls, args = _node()
    calibration = {"type": "SLRD_SPECTRUM"}
    node.sample(*args, regeneration_mode="spectral_progressive", stage_megapixels="2",
                progressive_transition_sigmas="0.9, 0.6", spectral_tolerance=0.02)
    assert calls[-1]["regeneration_mode"] == "spectral_progressive"
    assert calls[-1]["progressive_transition_sigmas"] == [0.9, 0.6]
    assert calls[-1]["spectral_tolerance"] == 0.02
    node.sample(*args, regeneration_mode="spectral_progressive", spectral_calibration=calibration)
    assert calls[-1]["spectral_calibration"] is calibration
    assert calls[-1]["progressive_transition_sigmas"] == []


def test_spectral_progressive_staged_run_skips_trajectory_release_validation():
    # Staged trajectory runs require a locked prefix; progressive runs plan
    # their own transitions and must not inherit that rejection.
    node, calls, args = _node()
    node.sample(*args, regeneration_mode="spectral_progressive", stage_megapixels="2",
                composition_lock_sigma=1.0, progressive_transition_sigmas="0.9, 0.6")
    assert calls[-1]["regeneration_mode"] == "spectral_progressive"


@pytest.mark.parametrize("text", ["0.5, 0.7", "1.2", "x"])
def test_invalid_progressive_transition_sigmas_rejected_before_sampling(text):
    node, calls, args = _node()
    with pytest.raises(ValueError, match="progressive_transition_sigmas"):
        node.sample(*args, regeneration_mode="spectral_progressive", progressive_transition_sigmas=text)
    assert not calls


def test_trajectory_auto_uses_existing_widgets_without_manual_refinement_sigmas():
    node, calls, args = _node()
    node.sample(*args, regeneration_mode="trajectory_auto", stage_megapixels="2",
                img2img_start_sigma=0.0, progressive_transition_sigmas="invalid inactive setting")
    call = calls[-1]
    assert call["regeneration_mode"] == "trajectory_auto"
    assert call["stage_hw"]
    assert call["img2img_start_sigma"] is None
    assert call["progressive_transition_sigmas"] == []


def test_trajectory_auto_native_plans_the_in_range_grid_as_first_stage():
    # Planner 1 MP -> (124, 124) on the test grid; output (200, 200).
    node, calls, args = _node(native_first=(150, 150), planned=(2.0,))
    node.sample(*args, regeneration_mode="trajectory_auto_native")
    call = calls[-1]
    assert call["regeneration_mode"] == "trajectory_auto_native"
    # The in-range grid leads; later stages are planned from it, not from the planner.
    assert call["stage_hw"] == ((150, 150), (176, 176))
    assert node.native_calls == [((124, 124), (200, 200), 4.0)]
    assert node.planning[-1][0] == 150 * 150 * 64.0
    assert call["img2img_start_sigma"] is None


def test_auto_two_passes_uses_target_grid_instead_of_a_fixed_upscale_cap():
    node, calls, args = _node(native_first=(150, 150))
    node.sample(*args, regeneration_mode="trajectory_auto_native",
                stage_planning="auto_two_passes", max_stage_upscale=2.0)
    assert calls[-1]["stage_hw"] == ((150, 150),)
    assert node.native_calls[-1] == ((124, 124), (200, 200), 0.0)
    assert node.planning[-1][2] == pytest.approx(
        1.02 * max((150 / 124) ** 2, (200 / 150) ** 2)
    )

    # Same planner and native stage; a larger connected target automatically
    # increases the area cap without changing any widget values.
    enlarged = list(args)
    enlarged[3] = {"samples": torch.zeros(1, 16, 300, 300)}
    node.sample(*enlarged, regeneration_mode="trajectory_auto_native",
                stage_planning="auto_two_passes", max_stage_upscale=2.0)
    assert node.native_calls[-1] == ((124, 124), (300, 300), 0.0)
    assert node.planning[-1][2] == pytest.approx(
        1.02 * max((150 / 124) ** 2, (300 / 150) ** 2)
    )


def test_auto_two_passes_respects_explicit_stages_and_other_modes():
    node, calls, args = _node(native_first=(150, 150), planned=(2.0,))
    node.sample(*args, regeneration_mode="trajectory_auto_native",
                stage_planning="auto_two_passes", stage_megapixels="2")
    assert calls[-1]["stage_hw"] == ((176, 176),)
    assert not node.native_calls and not node.planning

    node.sample(*args, regeneration_mode="trajectory",
                stage_planning="auto_two_passes")
    assert node.native_calls == []
    assert node.planning[-1][2] == 4.0
    with pytest.raises(ValueError, match="stage_planning"):
        node.sample(*args, stage_planning="invalid")


def test_auto_two_passes_falls_back_to_single_pass_without_native_stage():
    node, calls, args = _node(native_first=None)
    node.sample(*args, regeneration_mode="trajectory_auto_native",
                stage_planning="auto_two_passes")
    assert calls[-1]["stage_hw"] == ()
    assert node.planning[-1][2] == pytest.approx(1.02 * (200 / 124) ** 2)


def test_trajectory_auto_native_without_an_in_range_grid_keeps_the_planned_layout():
    node, calls, args = _node(native_first=None, planned=(2.0,))
    node.sample(*args, regeneration_mode="trajectory_auto_native")
    assert calls[-1]["stage_hw"] == ((176, 176),)
    assert node.planning[-1][0] == 124 * 124 * 64.0


def test_native_first_stage_is_specific_to_its_mode_and_respects_explicit_stages():
    node, calls, args = _node(native_first=(150, 150), planned=(2.0,))
    node.sample(*args, regeneration_mode="trajectory_auto")
    assert calls[-1]["stage_hw"] == ((176, 176),)
    node.sample(*args, regeneration_mode="trajectory")
    assert calls[-1]["stage_hw"] == ((176, 176),)
    node.sample(*args, regeneration_mode="trajectory_auto_native", stage_megapixels="2")
    assert calls[-1]["stage_hw"] == ((176, 176),)


def test_highres_step_budget_reaches_runtime_and_rejects_invalid_values():
    node, calls, args = _node()
    node.sample(*args, steps=20, regeneration_mode="trajectory_auto_native")
    assert calls[-1]["highres_min_step_fraction"] == 0.25
    assert calls[-1]["steps"] == 20
    node.sample(*args, steps=20, regeneration_mode="trajectory_auto_native",
                highres_min_step_fraction=0.0)
    assert calls[-1]["highres_min_step_fraction"] == 0.0
    for bad in (-0.01, 1.01, float("nan"), float("inf"), True, "0.25"):
        with pytest.raises(ValueError, match="highres_min_step_fraction"):
            node.sample(*args, highres_min_step_fraction=bad)


def _lattice_latent(seed, shape=(1, 16, 24, 28)):
    g = torch.Generator().manual_seed(seed)
    content = torch.randn(shape, generator=g)
    offsets = 0.2 * torch.randn(shape[:-2] + (2, 2), generator=g)
    offsets = offsets - offsets.mean(dim=(-2, -1), keepdim=True)
    lattice = offsets.repeat(*([1] * (len(shape) - 2)), shape[-2] // 2, shape[-1] // 2)
    return content, content + lattice


def test_patch_lattice_correction_is_off_by_default_and_returns_sampler_objects_untouched():
    node, calls, args = _node()
    _, corrupted = _lattice_latent(0)
    output = {"samples": corrupted.clone(), "batch_index": [3]}
    node.results.append(SimpleNamespace(output=output, lowres_planner="planner", denoised_output=output))
    out, planner, denoised = node.sample(*args)
    assert out is output and denoised is output and planner == "planner"
    assert torch.equal(out["samples"], corrupted)
    assert "patch_lattice_correction" not in calls[-1]  # sampling never sees the option
    out, _, _ = node.sample(*args, patch_lattice_correction="off")
    assert out is output and torch.equal(out["samples"], corrupted)


def test_patch_lattice_correction_removes_the_offset_from_both_final_latents_only():
    node, calls, args = _node()
    clean, corrupted = _lattice_latent(1)
    _, corrupted_denoised = _lattice_latent(2)
    output = {"samples": corrupted.clone(), "batch_index": [3]}
    denoised = {"samples": corrupted_denoised.clone()}
    planner = {"samples": corrupted.clone()}
    node.results.append(SimpleNamespace(output=output, lowres_planner=planner, denoised_output=denoised))
    out, planner_out, denoised_out = node.sample(*args, patch_lattice_correction="final_latent")
    # Inputs from the sampler are not mutated; metadata is carried over.
    assert torch.equal(output["samples"], corrupted) and torch.equal(denoised["samples"], corrupted_denoised)
    assert out is not output and out["batch_index"] == [3]
    assert planner_out is planner and torch.equal(planner_out["samples"], corrupted)
    for result in (out, denoised_out):
        assert krea2_flow.patch_lattice_modes(result["samples"]).abs().max() < 1e-6
    # The injected offset is recovered up to the content's own chance-level modes.
    assert (out["samples"] - krea2_flow.remove_patch_lattice(clean)).abs().max() < 1e-5
    assert sum("patch lattice correction" in line[0] for line in node.log_lines) == 2
    assert "patch_lattice_correction" not in calls[-1]

    # A shared output/denoised dictionary stays shared and is corrected once.
    node.log_lines.clear()
    node.results.append(SimpleNamespace(output=output, lowres_planner=planner, denoised_output=output))
    out, _, denoised_out = node.sample(*args, patch_lattice_correction="final_latent")
    assert denoised_out is out and len(node.log_lines) == 1


def test_patch_lattice_correction_rejects_unknown_values_before_sampling():
    node, calls, args = _node()
    for bad in ("on", "", None, True):
        with pytest.raises(ValueError, match="patch_lattice_correction must be one of off, final_latent"):
            node.sample(*args, patch_lattice_correction=bad)
    assert not calls
