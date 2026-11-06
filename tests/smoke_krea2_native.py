"""CPU integration with real ComfyUI and a tiny random native Krea 2 model.

This checks execution/layout only. It does not establish real-checkpoint image
quality or GPU speed. Run with ComfyUI's Python dependencies installed.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--spectrum-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.comfy_root.resolve()))
    sys.argv = [sys.argv[0], "--cpu", "--use-pytorch-cross-attention", "--preview-method", "none"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import comfy.supported_models
    import comfy.model_patcher
    import comfy.model_base
    torch.set_num_threads(1)
    torch.manual_seed(73)
    config = comfy.supported_models.Krea2(dict(
        image_model="krea2", features=32, tdim=8, txtdim=16, heads=2, kvheads=1,
        multiplier=1, layers=2, patch=2, channels=16, txtlayers=2,
        txtheads=2, txtkvheads=1,
    ))
    config.set_inference_dtype(torch.float32, None)
    base = comfy.model_base.Krea2(config, device=torch.device("cpu"))
    with torch.no_grad():
        for param in base.diffusion_model.parameters():
            param.copy_(torch.randn_like(param) * .02)
    model = comfy.model_patcher.ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
    # The separate resolution-selector tests cover the full UI integration.
    # Loading ComfyUI's top-level nodes.py here would pull in unrelated
    # optional video/frontend dependencies. Only its canvas cap is needed
    # by the scale-locked package's resolution selector during import.
    import types
    core_nodes = types.ModuleType("nodes")
    core_nodes.MAX_RESOLUTION = 16384
    sys.modules.setdefault("nodes", core_nodes)
    spec = importlib.util.spec_from_file_location("slrd_native_smoke", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    node = package.NODE_CLASS_MAPPINGS["Krea2TurboFlowAlignedRegenerate"]()
    positive = [[torch.randn(1, 3, 32), {}]]
    negative = [[torch.zeros(1, 3, 32), {}]]
    cases = [("native", model)]
    reports_context = None
    if args.spectrum_root:
        sys.path.insert(0, str(args.spectrum_root.resolve()))
        from comfyui_spectrum_krea2.nodes import SpectrumApplyKrea2
        from comfyui_spectrum_krea2.runtime import REPORTS
        reports_context = REPORTS
        cases.append(("spectrum", SpectrumApplyKrea2().apply(model, preset="turbo")[0]))
    runtime = sys.modules[spec.name + ".slrd_runtime"]
    calls = []
    locked_by_steps = {8: 4, 12: 6}

    def run(selected, steps, analytic=True, count_calls=False):
        reports = []
        token = reports_context.set(reports) if reports_context is not None else None
        original = runtime._full_lock_handoff
        original_locked = runtime.locked_planner_prediction
        if not analytic:
            # Reference: step through the lock with Euler, evaluating the target
            # model on locked steps and discarding its prediction.
            runtime._full_lock_handoff = lambda *args, **kwargs: None
            runtime.locked_planner_prediction = lambda *args, **kwargs: None
        calls.clear()
        # Spectrum declines hooked models, so only count calls without it.
        hook = (base.diffusion_model.register_forward_pre_hook(
            lambda module, args: calls.append(tuple(args[0].shape[-2:]))) if count_calls else None)
        try:
            with torch.inference_mode():
                # Small latent canvases keep genuine CPU model execution practical.
                result = node.sample(selected, positive, negative,
                                     {"samples": torch.zeros(1, 4, 16, 20)}, 42, .01, 0.0, 0.0, steps=steps)
        finally:
            if hook is not None:
                hook.remove()
            runtime._full_lock_handoff = original
            runtime.locked_planner_prediction = original_locked
            if reports_context is not None:
                reports_context.reset(token)
        target_calls = sum(shape == (16, 20) for shape in calls)
        return result, reports, target_calls

    for label, selected, steps in [(label,selected,steps) for label,selected in cases for steps in (8,12)]:
        locked = locked_by_steps[steps]
        result, reports, target_calls = run(selected, steps, count_calls=label == "native")
        if label == "native":
            # Locked target steps are skipped, not evaluated and discarded.
            assert target_calls == steps - locked, (target_calls, steps, locked)
            evaluated, _, evaluated_calls = run(selected, steps, analytic=False, count_calls=True)
            assert evaluated_calls == steps
            for fast, slow in zip(result, evaluated):
                difference = float((fast["samples"] - slow["samples"]).abs().max())
                scale = float(slow["samples"].abs().max())
                assert difference <= 1e-4 * max(1.0, scale), (difference, scale)
        if label == "spectrum":
            assert len(reports) == 2
            assert reports[0]["outer_steps"] == steps
            assert reports[1]["outer_steps"] == steps - locked
            assert reports[1]["denoiser_calls"] == steps - locked
            assert all(r["fallback"] is None for r in reports)
            assert reports[0]["forecast"] > 0
        shapes = [list(x["samples"].shape) for x in result]
        assert shapes[0] == shapes[2] == [1, 16, 1, 16, 20]
        assert all(torch.isfinite(x["samples"]).all() for x in result)
        assert shapes[1][2] == 1
        assert not base.diffusion_model._forward_pre_hooks
        assert not base.diffusion_model.last._forward_pre_hooks
        print(json.dumps({"case": label, "steps_per_pass": steps, "locked_target_steps": locked,
                          "target_model_calls": target_calls, "output_shapes": shapes, "reports": reports}))

    # Resolution ladder: planner 12x14 -> stage 16x20 -> output 24x28 latent cells.
    # Each stage finishes its image and runs every step after the lock.
    for label, selected, steps in [(label,selected,steps) for label,selected in cases for steps in (8,12)]:
        for stage_options in ({"stage_megapixels": "0.02"}, {"max_stage_upscale": 2.0}):
            reports = []
            token = reports_context.set(reports) if reports_context is not None else None
            calls.clear()
            hook = (base.diffusion_model.register_forward_pre_hook(
                lambda module, args: calls.append(tuple(args[0].shape[-2:]))) if label == "native" else None)
            try:
                with torch.inference_mode():
                    result = node.sample(selected, positive, negative, {"samples": torch.zeros(1, 4, 24, 28)},
                                         42, .01, 0.0, 0.0, steps=steps, **stage_options)
            finally:
                if hook is not None:
                    hook.remove()
                if reports_context is not None:
                    reports_context.reset(token)
            free = steps - locked_by_steps[steps]
            if label == "native":
                counts = {shape: calls.count(shape) for shape in set(calls)}
                assert counts == {(12, 14): steps, (16, 20): free, (24, 28): free}, counts
            else:
                assert [r["outer_steps"] for r in reports] == [steps, free, free], reports
                assert all(r["fallback"] is None for r in reports)
            shapes = [list(x["samples"].shape) for x in result]
            assert shapes[0] == shapes[2] == [1, 16, 1, 24, 28]
            assert shapes[1] == [1, 16, 1, 12, 14]
            assert all(torch.isfinite(x["samples"]).all() for x in result)
            assert not base.diffusion_model._forward_pre_hooks
            print(json.dumps({"case": label + "_ladder", "options": stage_options, "steps_per_pass": steps,
                              "stage_steps": free, "output_shapes": shapes, "reports": reports}))


    # The compact Krea sampler's controls must reach genuine ComfyUI
    # guider/sampler execution. Every solver skips the target model on fully
    # locked steps: Euler through the closed-form handoff, the others by
    # integrating the locked prefix natively with the planner predictions.
    spectrum_forecast = {"euler", "dpmpp_2m", "res_multistep", "gradient_estimation",
                         "deis", "ipndm", "ipndm_v"}
    solver_cases = [
        ("euler", "normal", 1.0),
        ("euler", "sgm_uniform", 2.0),
        ("dpmpp_2m", "simple", 1.0),
        ("dpmpp_2m", "normal", 2.0),
    ] + [(name, "simple", 1.0) for name in runtime.krea2_sampler_names()
         if name not in ("euler", "dpmpp_2m")]
    assert set(runtime.krea2_sampler_names()) == set(runtime.KREA2_SAMPLERS), runtime.krea2_sampler_names()

    def controls(selected, sampler_name, scheduler, cfg, steps, count_calls, evaluate_locked=False, **extra):
        reports = []
        token = reports_context.set(reports) if reports_context is not None else None
        calls.clear()
        original = runtime.locked_planner_prediction
        if evaluate_locked:
            # Reference: evaluate the target model on locked steps and discard it.
            runtime.locked_planner_prediction = lambda *args, **kwargs: None
        hook = base.diffusion_model.register_forward_pre_hook(
            lambda module, args: calls.append(tuple(args[0].shape[-2:]))
        ) if count_calls else None
        try:
            with torch.inference_mode():
                latent = extra.pop("latent", torch.zeros(1, 4, 16, 20))
                result = node.sample(
                    selected, positive, negative, {"samples": latent},
                    42, .01, 0.0, 0.0,
                    steps=steps, cfg=cfg, sampler_name=sampler_name,
                    scheduler=scheduler, **extra,
                )
        finally:
            runtime.locked_planner_prediction = original
            if hook is not None:
                hook.remove()
            if reports_context is not None:
                reports_context.reset(token)
        assert all(torch.isfinite(x["samples"]).all() for x in result), sampler_name
        return result, reports, {shape: calls.count(shape) for shape in set(calls)}

    for label, selected in cases:
        for sampler_name, scheduler, cfg in solver_cases:
            steps = 8
            locked = runtime.full_lock_prefix_length(
                runtime.calculate_sigmas(selected, scheduler=scheduler, steps=steps, denoise=1.0).tolist(), 0.8)
            assert 0 < locked < steps, (scheduler, locked)
            count_calls = label == "native"
            result, reports, counts = controls(selected, sampler_name, scheduler, cfg, steps, count_calls,
                                               max_stage_upscale=0.0)
            assert result[0]["samples"].shape == (1, 16, 1, 16, 20)
            if count_calls:
                planner_hw = tuple(result[1]["samples"].shape[-2:])
                assert counts == {planner_hw: steps, (16, 20): steps - locked}, (sampler_name, scheduler, counts)
                if sampler_name != "euler":
                    reference, _, reference_counts = controls(
                        selected, sampler_name, scheduler, cfg, steps, True,
                        evaluate_locked=True, max_stage_upscale=0.0)
                    assert reference_counts[(16, 20)] == steps, reference_counts
                    for fast, slow in zip(result, reference):
                        # Locked steps discard the target prediction, so skipping
                        # it must not change any output.
                        assert torch.equal(fast["samples"], slow["samples"]), sampler_name
            if label == "spectrum":
                assert len(reports) == 2, reports
                assert reports[0]["outer_steps"] == steps
                expected_outer = steps - locked if sampler_name == "euler" else steps
                assert reports[1]["outer_steps"] == expected_outer, reports
                if sampler_name in spectrum_forecast:
                    assert all(report["fallback"] is None for report in reports), reports
                    # Skipped locked steps never reach the diffusion model.
                    assert reports[1]["denoiser_calls"] == steps - locked, reports
                else:
                    assert all(report["fallback"] for report in reports), reports
            print(json.dumps({
                "case": label + "_controls", "sampler": sampler_name,
                "scheduler": scheduler, "cfg": cfg, "locked_target_steps": locked,
                "model_calls": {f"{h}x{w}": n for (h, w), n in counts.items()}, "reports": reports,
            }))

    # The 20-step/0.72-lock case has 12 fully locked and 8 unlocked
    # target steps. All seven native solvers are now forecast-enabled,
    # independently of the lock's zero-DiT shortcut.
    if reports_context is not None:
        from comfyui_spectrum_krea2.nodes import SpectrumApplyKrea2
        aligned = SpectrumApplyKrea2().apply(
            model, preset="turbo", warmup=2, tail=1, bootstrap=False,
            extrapolation_guard_enabled=False,
        )[0]
        for sampler_name in ("res_multistep", "gradient_estimation",
                             "deis", "ipndm", "ipndm_v"):
            outputs, reports, _ = controls(
                aligned, sampler_name, "simple", 1.0, 20, False,
                max_stage_upscale=0.0, composition_lock_sigma=0.72,
            )
            assert len(reports) == 2, (sampler_name, reports)
            assert reports[0]["actual"] == 11 and reports[0]["forecast"] == 9, reports
            assert reports[1]["outer_steps"] == 20, reports
            assert reports[1]["denoiser_calls"] == 8, reports
            assert reports[1]["actual"] == 5 and reports[1]["forecast"] == 3, reports
            assert all(r["fallback"] is None for r in reports), reports
            print(json.dumps({
                "case": "spectrum_20_step_lock",
                "sampler": sampler_name,
                "locked_target_steps": 12,
                "reports": reports,
            }))

    # Resolution ladders with every solver: planner 12x14 -> stage 16x20 ->
    # output 24x28. Locked steps cost no model calls in the first stage, and
    # each later stage restarts the solver at the composition-lock release.
    for label, selected in cases:
        for sampler_name in runtime.krea2_sampler_names():
            steps = 8
            free = steps - locked_by_steps[steps]
            result, reports, counts = controls(
                selected, sampler_name, "simple", 1.0, steps, label == "native",
                latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02")
            assert [list(x["samples"].shape) for x in result] == [
                [1, 16, 1, 24, 28], [1, 16, 1, 12, 14], [1, 16, 1, 24, 28]]
            if label == "native":
                assert counts == {(12, 14): steps, (16, 20): free, (24, 28): free}, (sampler_name, counts)
            else:
                first = free if sampler_name == "euler" else steps
                assert [r["outer_steps"] for r in reports] == [steps, first, free], reports
                if sampler_name in spectrum_forecast:
                    assert all(r["fallback"] is None for r in reports), reports
            print(json.dumps({"case": label + "_ladder_solver", "sampler": sampler_name,
                              "model_calls": {f"{h}x{w}": n for (h, w), n in counts.items()},
                              "reports": reports}))

    try:
        node.sample(model, positive, negative, {"samples": torch.zeros(1, 4, 24, 28)},
                    42, .01, 0.0, 0.0, steps=2, composition_lock_sigma=0.0, stage_megapixels="0.02")
    except ValueError as exc:
        assert "at least one step" in str(exc)
    else:
        raise AssertionError("too few steps for the stages must be rejected before sampling")


if __name__ == "__main__":
    main()
