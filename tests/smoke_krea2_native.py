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
    parser.add_argument("--kj-root", type=Path)
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
        # Load Spectrum exactly as ComfyUI loads a custom-node directory: a
        # path-derived module name, not importable as a top-level package.
        # Cross-package controls must reach this loaded instance.
        spectrum_dir = args.spectrum_root.resolve()
        spectrum_name = str(spectrum_dir).replace(".", "_x_")
        spectrum_spec = importlib.util.spec_from_file_location(
            spectrum_name, spectrum_dir / "__init__.py", submodule_search_locations=[str(spectrum_dir)])
        spectrum_package = importlib.util.module_from_spec(spectrum_spec)
        sys.modules[spectrum_name] = spectrum_package
        spectrum_spec.loader.exec_module(spectrum_package)
        SpectrumApplyKrea2 = sys.modules[spectrum_name + ".comfyui_spectrum_krea2.nodes"].SpectrumApplyKrea2
        REPORTS = sys.modules[spectrum_name + ".comfyui_spectrum_krea2.runtime"].REPORTS
        assert "comfyui_spectrum_krea2.runtime" not in sys.modules
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
                         "deis", "ipndm", "ipndm_v", "er_sde",
                         "sa_solver", "sa_solver_pece"}
    solver_cases = [
        ("euler", "normal", 1.0),
        ("euler", "sgm_uniform", 2.0),
        ("dpmpp_2m", "simple", 1.0),
        ("dpmpp_2m", "normal", 2.0),
    ] + [(name, "simple", 1.0) for name in runtime.krea2_sampler_names()
         if name not in ("euler", "dpmpp_2m")]
    assert set(runtime.krea2_sampler_names()) == set(runtime.KREA2_SAMPLERS), runtime.krea2_sampler_names()

    def solver_calls(sampler_name, steps):
        return steps + max(0, steps - 1) if sampler_name == "sa_solver_pece" else steps

    def target_calls(sampler_name, steps, locked):
        if sampler_name == "sa_solver_pece":
            # First PECE step has no correction, and all locked corrections
            # use the corresponding corrected planner anchor without a DiT.
            return 2 * (steps - locked) if locked else solver_calls(sampler_name, steps)
        return steps - locked

    def controls(selected, sampler_name, scheduler, cfg, steps, count_calls, evaluate_locked=False, **extra):
        if sampler_name in ("sa_solver", "sa_solver_pece"):
            # The existing forecast regression cases remain opt-in. The new
            # Krea 2 UI defaults to a native-only SA A/B control.
            extra.setdefault("sa_spectrum_forecasts", True)
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
                assert counts == {
                    planner_hw: solver_calls(sampler_name, steps),
                    (16, 20): target_calls(sampler_name, steps, locked),
                }, (sampler_name, scheduler, counts)
                if sampler_name != "euler":
                    reference, _, reference_counts = controls(
                        selected, sampler_name, scheduler, cfg, steps, True,
                        evaluate_locked=True, max_stage_upscale=0.0)
                    assert reference_counts[(16, 20)] == solver_calls(sampler_name, steps), reference_counts
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
                    assert reports[1]["denoiser_calls"] == target_calls(sampler_name, steps, locked), reports
                else:
                    assert all(report["fallback"] for report in reports), reports
            print(json.dumps({
                "case": label + "_controls", "sampler": sampler_name,
                "scheduler": scheduler, "cfg": cfg, "locked_target_steps": locked,
                "model_calls": {f"{h}x{w}": n for (h, w), n in counts.items()}, "reports": reports,
            }))

    # The 20-step/0.72-lock case has 12 fully locked and 8 unlocked
    # target steps. All offered native solvers are forecast-enabled,
    # independently of the lock's zero-DiT shortcut.
    if reports_context is not None:
        aligned = SpectrumApplyKrea2().apply(
            model, preset="turbo", warmup=2, tail=1, bootstrap=False,
            extrapolation_guard_enabled=False,
        )[0]
        for sampler_name in ("res_multistep", "gradient_estimation",
                             "deis", "ipndm", "ipndm_v", "er_sde", "sa_solver"):
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
                assert counts == {
                    (12, 14): solver_calls(sampler_name, steps),
                    (16, 20): target_calls(sampler_name, steps, steps - free),
                    (24, 28): solver_calls(sampler_name, free),
                }, (sampler_name, counts)
            else:
                first = free if sampler_name == "euler" else steps
                assert [r["outer_steps"] for r in reports] == [steps, first, free], reports
                if sampler_name in spectrum_forecast:
                    assert all(r["fallback"] is None for r in reports), reports
            print(json.dumps({"case": label + "_ladder_solver", "sampler": sampler_name,
                              "model_calls": {f"{h}x{w}": n for (h, w), n in counts.items()},
                              "reports": reports}))



    # SA-PECE must preserve its second denoiser prediction at each corrected
    # sigma, while native Scale-Locked skips both corresponding locked target
    # calls. Spectrum may forecast only predictors and always runs corrections
    # natively on unlocked steps.
    for label, selected in cases:
        result, reports, counts = controls(
            selected, "sa_solver_pece", "simple", 1.0, 14, label == "native",
            max_stage_upscale=0.0, composition_lock_sigma=0.72,
        )
        sigmas = runtime.calculate_sigmas(selected, scheduler="simple", steps=14, denoise=1.0)
        effective = runtime.effective_er_sde_sigmas(sigmas, selected)
        locked = runtime.full_lock_prefix_length(effective.tolist(), .72)
        if label == "native":
            planner_hw = tuple(result[1]["samples"].shape[-2:])
            assert counts[planner_hw] == 27, counts
            assert counts[(16, 20)] == 2 * (14 - locked), counts
            reference, _, total = controls(
                selected, "sa_solver_pece", "simple", 1.0, 14, True,
                evaluate_locked=True, max_stage_upscale=0.0,
                composition_lock_sigma=0.72,
            )
            assert total[(16, 20)] == 27, total
            assert all(torch.equal(a["samples"], b["samples"])
                       for a, b in zip(result, reference))
        else:
            assert len(reports) == 2 and all(r["fallback"] is None for r in reports), reports
            assert reports[0]["pece_corrector_evaluations"] == 13, reports
            assert reports[1]["denoiser_calls"] == 2 * (14 - locked), reports
            # Conservative default warmup/guard may leave no forecast in
            # the short unlocked target tail. Correct PECE accounting is
            # mandatory; forecast availability is configuration-dependent.
            assert reports[0]["forecast"] > 0, reports
        print(json.dumps({
            "case": label + "_pece", "locked": locked,
            "counts": {f"{h}x{w}": n for (h, w), n in counts.items()},
            "reports": reports,
        }))

    if reports_context is not None:
        fast_pece = SpectrumApplyKrea2().apply(
            model, preset="turbo", warmup=2, tail=1,
            bootstrap=False, extrapolation_guard_enabled=False,
        )[0]
        _, pece_reports, _ = controls(
            fast_pece, "sa_solver_pece", "simple", 1.0, 20, False,
            max_stage_upscale=0.0, composition_lock_sigma=0.72,
        )
        assert len(pece_reports) == 2, pece_reports
        assert all(r["fallback"] is None for r in pece_reports), pece_reports
        assert pece_reports[0]["pece_corrector_evaluations"] == 19
        assert pece_reports[0]["forecast"] > 0 and pece_reports[1]["forecast"] > 0, pece_reports
        assert pece_reports[1]["denoiser_calls"] == 16, pece_reports
        print(json.dumps({"case": "spectrum_pece_20_step_lock", "reports": pece_reports}))

    if reports_context is not None:
        # Opt-in active-PECE acceleration through the resolution ladder: after
        # two actual predicted phases, predictions are solver-owned skips and
        # every unlocked correction is an exact model evaluation.
        balanced = SpectrumApplyKrea2().apply(
            model, preset="turbo", warmup=2, tail=1, bootstrap=False,
            extrapolation_guard_enabled=False, sa_pece_forecast_policy="balanced",
        )[0]
        _, accelerated, _ = controls(
            balanced, "sa_solver_pece", "simple", 1.0, 12, False,
            latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02",
        )
        assert len(accelerated) == 3 and all(r["fallback"] is None for r in accelerated), accelerated
        assert all(r["sa_pece_forecast_policy"] == "balanced" for r in accelerated)
        assert accelerated[0]["sa_pece_skipped_predictions"] == 10, accelerated
        assert accelerated[0]["pece_corrector_evaluations"] == 11, accelerated
        assert all(r["sa_pece_skipped_predictions"] > 0 for r in accelerated[1:]), accelerated
        print(json.dumps({"case": "sa_pece_balanced_ladder", "reports": accelerated}))

        # The optional detail probe must retain the exact same native
        # predictor/corrector call budget and stochastic-noise routing while
        # avoiding the hard secant-to-hold discontinuity at late log-SNR steps.
        detail = SpectrumApplyKrea2().apply(
            model, preset="turbo", warmup=2, tail=1, bootstrap=False,
            extrapolation_guard_enabled=False,
            sa_pece_forecast_policy="balanced_detail",
        )[0]
        _, detail_reports, _ = controls(
            detail, "sa_solver_pece", "simple", 1.0, 12, False,
            latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02",
        )
        assert len(detail_reports) == 3, detail_reports
        for prior, changed in zip(accelerated, detail_reports):
            assert changed["fallback"] is None
            assert changed["actual"] == prior["actual"]
            assert changed["forecast"] == prior["forecast"]
            assert changed["pece_corrector_evaluations"] == prior["pece_corrector_evaluations"]
            assert changed["sa_pece_skipped_predictions"] == prior["sa_pece_skipped_predictions"]
            assert changed["sa_pece_forecast_policy"] == "balanced_detail"
            assert changed["sa_dense_modes"]["soft_extrapolation"] > 0, changed
        # Every previously held final-stage predictor has a bounded, continuous
        # corrected-endpoint surrogate; none comes from a raw stochastic forecast.
        assert accelerated[-1]["sa_dense_modes"]["guard_hold"] == 4, accelerated
        assert detail_reports[-1]["sa_dense_modes"]["guard_hold"] == 0, detail_reports
        assert detail_reports[-1]["sa_dense_modes"]["soft_extrapolation"] == 4, detail_reports
        print(json.dumps({"case": "sa_pece_balanced_detail_ladder",
                          "reports": detail_reports}))


    if reports_context is not None:
        # This is the exact quality-isolation path used by the new widget:
        # native SA/PECE plus the same Scale-Locked stages, but 0 forecasts.
        # It must be bitwise equal to running the native model without
        # installing Spectrum, including the stochastic coupled increments.
        for sampler_name in ("sa_solver", "sa_solver_pece"):
            plain, _, _ = controls(
                model, sampler_name, "simple", 1.0, 12, False,
                latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02",
                sa_spectrum_forecasts=False,
            )
            bypass, reports, _ = controls(
                cases[1][1], sampler_name, "simple", 1.0, 12, False,
                latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02",
                sa_spectrum_forecasts=False,
            )
            assert len(reports) == 3 and all(r["fallback"] is None for r in reports), reports
            assert all(r["native_only"] is True and r["forecast"] == 0 and
                       r["peak_history_bytes"] == 0 for r in reports), reports
            for left, right in zip(plain, bypass):
                torch.testing.assert_close(left["samples"], right["samples"], atol=0, rtol=0)
            enabled, on_reports, _ = controls(
                cases[1][1], sampler_name, "simple", 1.0, 12, False,
                latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02",
                sa_spectrum_forecasts=True,
            )
            assert len(on_reports) == 3 and all(not r["native_only"] for r in on_reports), on_reports
            assert on_reports[0]["forecast"] > 0, on_reports
            assert all(torch.isfinite(t["samples"]).all() for t in enabled)
            print(json.dumps({
                "case": "sa_12step_native_bypass",
                "sampler": sampler_name, "records": reports,
                "forecast_enabled": [r["forecast"] for r in on_reports],
            }))

    if reports_context is not None:
        # Planner-native-only A/B: isolate the low-resolution Spectrum quality
        # without giving up accelerated intermediate/final target stages.
        # Same seed, sampler, Gaussian coupling and planner schedule.
        # The default distance guard rejects every target-stage forecast on
        # this short CPU schedule; use the relaxed configuration of the other
        # ladder forecast checks so the stage acceleration is observable.
        stage_forecasting = SpectrumApplyKrea2().apply(
            model, preset="turbo", warmup=2, tail=1, bootstrap=False,
            extrapolation_guard_enabled=False,
        )[0]
        for sampler_name in ("sa_solver", "sa_solver_pece"):
            native_all, native_reports, _ = controls(
                stage_forecasting, sampler_name, "simple", 1.0, 12, False,
                latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02",
                sa_spectrum_forecasts=False,
            )
            planner_native, hybrid_reports, _ = controls(
                stage_forecasting, sampler_name, "simple", 1.0, 12, False,
                latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02",
                sa_spectrum_forecasts=True, sa_planner_spectrum_forecasts=False,
            )
            assert len(native_reports) == len(hybrid_reports) == 3
            assert all(r["fallback"] is None for r in hybrid_reports), hybrid_reports
            assert hybrid_reports[0]["native_only"] is True, hybrid_reports
            assert hybrid_reports[0]["forecast"] == 0, hybrid_reports
            assert hybrid_reports[0]["peak_history_bytes"] == 0, hybrid_reports
            assert hybrid_reports[0]["actual"] == native_reports[0]["actual"], hybrid_reports
            torch.testing.assert_close(
                planner_native[1]["samples"], native_all[1]["samples"],
                rtol=0, atol=0,
            )
            assert all(r["native_only"] is False for r in hybrid_reports[1:]), hybrid_reports
            assert any(r["forecast"] > 0 for r in hybrid_reports[1:]), hybrid_reports
            assert all(torch.isfinite(x["samples"]).all() for x in planner_native)

            # Context lifetime: the next independent call must regain planner
            # forecasts with no leaked native-only selection.
            _, restored, _ = controls(
                stage_forecasting, sampler_name, "simple", 1.0, 12, False,
                latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02",
                sa_spectrum_forecasts=True, sa_planner_spectrum_forecasts=True,
            )
            assert all(not r["native_only"] for r in restored), restored
            assert restored[0]["forecast"] > 0, restored
            print(json.dumps({
                "case": "sa_native_planner_only",
                "sampler": sampler_name,
                "reports": hybrid_reports,
                "restored_planner_forecasts": restored[0]["forecast"],
            }))

    # ER-SDE's three modes must not contaminate later queued invocations,
    # even when ComfyUI reuses the same node, MODEL patcher, and Spectrum
    # wrapper instance. Test the *complete* three-resolution sampler, not just
    # an options constructor or a toy noise-scaler.
    for label, selected in (
        ("native", model),
        *((("spectrum", SpectrumApplyKrea2().apply(
            model, preset="turbo", warmup=2, tail=1, bootstrap=False,
            extrapolation_guard_enabled=False,
        )[0]),) if reports_context is not None else ()),
    ):
        a = dict(sampler_name="er_sde", er_sde_solver_type="ER-SDE",
                 er_sde_max_stage=3, er_sde_eta=1.0, er_sde_s_noise=1.0)
        ode = dict(sampler_name="er_sde", er_sde_solver_type="ODE",
                   er_sde_max_stage=1, er_sde_eta=1.0, er_sde_s_noise=1.0)
        reverse = dict(sampler_name="er_sde", er_sde_solver_type="Reverse-time SDE",
                       er_sde_max_stage=2, er_sde_eta=0.5, er_sde_s_noise=0.65)
        profiles = (a, ode, reverse,
                    dict(sampler_name="sa_solver", sa_sde_end_percent=.30,
                         sa_s_noise=.5), a)
        outcomes = []
        receipts = []
        for profile in profiles:
            latent_parts, reports, _ = controls(
                selected, profile["sampler_name"], "simple", 1.0, 12, False,
                latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02",
                **{k: v for k, v in profile.items() if k != "sampler_name"},
            )
            outcomes.append(tuple(part["samples"].clone() for part in latent_parts))
            receipts.append(reports)
            if label == "spectrum":
                assert len(reports) == 3, reports
                assert all(r["fallback"] is None for r in reports), reports
                assert all(r["native_only"] is False for r in reports), reports
        for before, after in zip(outcomes[0], outcomes[-1]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        # The mode selector materially affects the native numerical output.
        # A genuine ODE switch must not merely reuse the preceding ER-SDE
        # option dictionary or stochastic sampler instance.
        assert not torch.equal(outcomes[0][0], outcomes[1][0]), label
        assert not torch.equal(outcomes[0][0], outcomes[2][0]), label
        assert not torch.equal(outcomes[1][0], outcomes[2][0]), label

        # ER-SDE controls are intentionally inert when a different solver is
        # selected. A user switching only er_sde_solver_type during plain SA
        # will see no output change; pin that behavior to avoid diagnosing it
        # as leaked numerical settings.
        plain_sa, plain_reports, _ = controls(
            selected, "sa_solver", "simple", 1.0, 12, False,
            latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02",
            er_sde_solver_type="ER-SDE",
        )
        inert_sa, inert_reports, _ = controls(
            selected, "sa_solver", "simple", 1.0, 12, False,
            latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02",
            er_sde_solver_type="ODE", er_sde_eta=0.0,
            er_sde_max_stage=1, er_sde_s_noise=0.0,
        )
        for baseline, changed in zip(plain_sa, inert_sa):
            torch.testing.assert_close(
                baseline["samples"], changed["samples"], atol=0, rtol=0
            )
        if label == "spectrum":
            assert [(r["actual"], r["forecast"], r["fallback"])
                    for r in plain_reports] == [
                    (r["actual"], r["forecast"], r["fallback"])
                    for r in inert_reports]
        assert not base.diffusion_model._forward_pre_hooks
        assert not base.diffusion_model.last._forward_pre_hooks
        print(json.dumps({
            "case": "er_sde_cross_run_state_isolation",
            "model": label,
            "modes": [x["er_sde_solver_type"] if x["sampler_name"] == "er_sde"
                      else x["sampler_name"] for x in profiles],
            "identical_return_to_first": True,
            "spectrum_reports": [
                [{"actual": p["actual"], "forecast": p["forecast"],
                  "fallback": p["fallback"]} for p in rr]
                for rr in receipts if rr
            ],
        }))

    # Native ER-SDE's deterministic ODE mode and stochastic reverse-time mode
    # use the same wrapper, stage planner and evaluation-skipping contract.
    for er_mode, eta, noise in (("ODE", 1.0, 1.0), ("Reverse-time SDE", .5, .6)):
        result, reports, counts = controls(
            model, "er_sde", "simple", 1.0, 8, True,
            max_stage_upscale=0.0,
            er_sde_solver_type=er_mode, er_sde_eta=eta, er_sde_s_noise=noise,
        )
        assert counts[tuple(result[1]["samples"].shape[-2:])] == 8, counts
        assert counts[(16, 20)] == 4, counts
        assert all(torch.isfinite(part["samples"]).all() for part in result), er_mode



    # Native SA-Solver supports ODE/no-noise, stochastic SDE windows,
    # simplified-order coefficients and PECE. Exercise the unmodified
    # ComfyUI numerical path with these non-default controls.
    for solver, eta, s_noise, predictor, corrector, simple in (
        ("sa_solver", 0.0, 1.0, 3, 4, False),
        ("sa_solver", 0.7, 0.45, 2, 2, True),
        ("sa_solver_pece", 0.0, 0.0, 3, 4, False),
        ("sa_solver_pece", 0.45, 0.6, 2, 2, True),
    ):
        result, _, counts = controls(
            model, solver, "simple", 1.0, 8, True,
            max_stage_upscale=0.0, sa_eta=eta,
            sa_sde_start_percent=.1, sa_sde_end_percent=.9,
            sa_s_noise=s_noise, sa_predictor_order=predictor,
            sa_corrector_order=corrector, sa_simple_order_2=simple,
        )
        planner_hw = tuple(result[1]["samples"].shape[-2:])
        assert counts[planner_hw] == solver_calls(solver, 8), counts
        assert counts[(16, 20)] == target_calls(solver, 8, locked_by_steps[8]), counts
        print(json.dumps({"case": "sa_native_modes", "sampler": solver,
                          "eta": eta, "s_noise": s_noise, "simple": simple}))

    if args.spectrum_root and args.kj_root:
        # Exercise the real upstream KJ PreviewOverride __call__ in the
        # complete Scale-Locked -> Spectrum -> native ER-SDE two-pass route.
        # Only preview transport/decoder dependencies are stubbed to avoid
        # external UI side effects in this CPU execution-contract test.
        import ast
        import logging
        import time
        import comfy.patcher_extension
        kj_path = args.kj_root.resolve() / "nodes/preview_override_node.py"
        parsed = ast.parse(kj_path.read_text(), filename=str(kj_path))
        klass_ast = next(n for n in parsed.body if isinstance(n, ast.ClassDef)
                         and n.name == "_PreviewOverrideWrapper")
        mod = types.ModuleType("slrd_kj_fixture.preview_override_node")
        mod.__file__ = str(kj_path)
        class NoPreviewEncoder:
            def submit(self, fn):
                return True
            def shutdown(self, drain_timeout=5.0):
                pass
        mod.__dict__.update({
            "torch": torch, "comfy": __import__("comfy"),
            "logging": logging, "time": time, "PromptServer": None,
            "load_tiny_vae_decoder": None,
            "_get_core_previewer": lambda *a: None,
            "_is_ltx_latent_format": lambda *a: False,
            "_is_av_model": lambda *a: False,
            "_normalize_packed_x0": lambda x, *a: x,
            "_frames_per_token": lambda *a: (1, 1),
            "_AsyncPreviewEncoder": NoPreviewEncoder,
            "latent_preview": types.SimpleNamespace(
                Latent2RGBPreviewer=lambda *a, **kw: None,
            ),
        })
        sys.modules[mod.__name__] = mod
        exec(compile(ast.Module(body=[klass_ast], type_ignores=[]), str(kj_path), "exec"),
             mod.__dict__)
        assert callable(mod._PreviewOverrideWrapper(512, None, 80, False, tiny_vae="none"))

        selected = SpectrumApplyKrea2().apply(
            model, preset="turbo", warmup=2, tail=1,
            bootstrap=False, extrapolation_guard_enabled=False,
        )[0]
        with_preview = selected.clone()
        with_preview.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "kj_preview_override",
            mod._PreviewOverrideWrapper(512, None, 80, False, tiny_vae="none"),
        )

        def run_kj(candidate):
            receipts = []
            tok = reports_context.set(receipts)
            try:
                with torch.inference_mode():
                    result = node.sample(
                        candidate, positive, negative,
                        {"samples": torch.zeros(1, 4, 16, 20)},
                        42, .01, 0.0, 0.0, steps=14, sampler_name="er_sde",
                        max_stage_upscale=0.0,
                    )
            finally:
                reports_context.reset(tok)
            return result, receipts

        baseline, base_reports = run_kj(selected)
        tested, kj_reports = run_kj(with_preview)
        assert len(kj_reports) == len(base_reports) == 2
        for base_report, kj_report in zip(base_reports, kj_reports):
            assert kj_report["fallback"] is None, kj_report
            assert kj_report["forecast"] > 0, kj_report
            assert (kj_report["actual"], kj_report["forecast"]) == (
                base_report["actual"], base_report["forecast"],
            )
        for original, actual in zip(baseline, tested):
            torch.testing.assert_close(
                original["samples"], actual["samples"], atol=0, rtol=0,
            )
        print(json.dumps({"case": "kj_preview_spectrum_slrd_er_sde",
                          "reports": kj_reports}))

    # ---- planner_final_img2img: every high-resolution stage is img2img of
    # the finished image below it, run by the genuine native ComfyUI path.
    import tempfile
    import comfy.model_management
    import folder_paths
    import safetensors.torch

    detail_seed = 42 ^ 0x9E3779B97F4A7C15
    img2img_sigma = .55
    model_sampling = model.get_model_object("model_sampling")

    def img2img(selected, sampler_name, steps, *, latent_hw=(16, 20), capture_inputs=False,
                mode="planner_final_img2img", fail=None, start_sigma=None, **extra):
        if sampler_name in ("sa_solver", "sa_solver_pece"):
            extra.setdefault("sa_spectrum_forecasts", True)
        reports, stages, inputs = [], [], []
        token = reports_context.set(reports) if reports_context is not None else None
        original_guider = runtime.create_cfg_guider
        original_preview = runtime.make_preview_callback

        def capture_guider(*a, **kw):
            guider = original_guider(*a, **kw)
            native_sample = guider.sample

            def sample(noise, latent_image, sampler, sigmas, **kws):
                out = native_sample(noise, latent_image, sampler, sigmas, **kws)
                stages.append({"noise": noise.detach().clone(), "latent": latent_image.detach().clone(),
                               "sigmas": sigmas.detach().cpu().clone(), "out": out.detach().clone()})
                return out
            guider.sample = sample
            return guider

        def failing_preview(*a, **kw):
            inner = original_preview(*a, **kw)

            def callback(step, x0, x, total):
                if tuple(x.shape[-2:]) == fail[0] and step == fail[1]:
                    raise comfy.model_management.InterruptProcessingException()
                inner(step, x0, x, total)
            return callback

        runtime.create_cfg_guider = capture_guider
        if fail is not None:
            runtime.make_preview_callback = failing_preview
        hook = base.diffusion_model.register_forward_pre_hook(
            lambda module, args: inputs.append((args[0].detach().clone(), args[1].detach().clone()))
        ) if capture_inputs else None
        try:
            with torch.inference_mode():
                result = node.sample(
                    selected, positive, negative, {"samples": torch.zeros(1, 4, *latent_hw)},
                    42, .01, 0.0, 0.0, steps=steps, sampler_name=sampler_name,
                    regeneration_mode=mode,
                    img2img_start_sigma=img2img_sigma if start_sigma is None else start_sigma,
                    **extra,
                )
        finally:
            runtime.create_cfg_guider = original_guider
            runtime.make_preview_callback = original_preview
            if hook is not None:
                hook.remove()
            if reports_context is not None:
                reports_context.reset(token)
        assert all(torch.isfinite(x["samples"]).all() for x in result), sampler_name
        assert not base.diffusion_model._forward_pre_hooks
        return result, reports, stages, inputs

    for sampler_name in runtime.KREA2_SAMPLERS:
        steps = 8
        schedule = runtime.calculate_sigmas(model, scheduler="simple", steps=steps, denoise=1.0)
        start, expected_sigmas = runtime.img2img_refinement_sigmas(schedule, img2img_sigma)
        assert 0 < start < steps
        free = len(expected_sigmas) - 1
        assert abs(float(expected_sigmas[0]) - img2img_sigma) < 1e-7
        for stage_options, sizes in (({"max_stage_upscale": 0.0}, [(16, 20)]),
                                     ({"stage_megapixels": "0.02"}, [(16, 20), (24, 28)])):
            latent_hw = sizes[-1]
            result, _, stages, inputs = img2img(model, sampler_name, steps, latent_hw=latent_hw,
                                                capture_inputs=True, **stage_options)
            planner = result[1]["samples"]
            planner_hw = tuple(planner.shape[-2:])
            # The planner is identical to the trajectory mode's finished planner.
            trajectory = controls(model, sampler_name, "simple", 1.0, steps, False,
                                  latent=torch.zeros(1, 4, *latent_hw), **stage_options)[0]
            assert torch.equal(trajectory[1]["samples"], planner), sampler_name
            planner_noise = runtime.prepare_noise(torch.zeros(planner.shape), 42)
            roots = []
            assert len(stages) == len(sizes), (sampler_name, len(stages))
            for index, (stage, hw) in enumerate(zip(stages, sizes)):
                final = index == len(sizes) - 1
                seed = detail_seed if final else runtime._stage_seed(detail_seed, index)
                root = runtime.prepare_noise(
                    torch.zeros(planner.shape), runtime._img2img_root_seed(42, index))
                # A finished source depends on its generating noise. Reusing
                # that noise violates the independent forward corruption
                # required for img2img; distinct stages need distinct roots.
                assert not torch.equal(root, planner_noise)
                assert all(not torch.equal(root, previous) for previous in roots)
                roots.append(root)
                noise = runtime._build_highres_noise(
                    {"samples": torch.zeros(planner.shape[:-2] + hw)}, root, seed, 1.0, "patch_coupled")
                assert torch.equal(stage["noise"].cpu(), noise), (sampler_name, index)
                assert torch.equal(stage["sigmas"], expected_sigmas.to(stage["sigmas"].dtype)), sampler_name
                # Source: the finished planner, then each finished stage below.
                source = planner if index == 0 else stages[index - 1]["out"]
                expected_latent = runtime.resize_4d_tensor(source.float(), hw)
                assert torch.equal(stage["latent"].float(), expected_latent), (sampler_name, index)
                # First native denoiser input = sigma_k * noise + (1 - sigma_k) * clean.
                first = next(x for x, t in inputs if tuple(x.shape[-2:]) == hw)
                timestep = next(t for x, t in inputs if tuple(x.shape[-2:]) == hw)
                sigma = expected_sigmas[0]
                expected_x = sigma * noise + (1 - sigma) * base.process_latent_in(expected_latent)
                torch.testing.assert_close(first.reshape(expected_x.shape), expected_x, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(timestep.flatten()[0].float(),
                                           model_sampling.timestep(sigma).float(), atol=1e-6, rtol=1e-6)
            assert torch.equal(result[0]["samples"], stages[-1]["out"].to(result[0]["samples"].dtype))
            counts = {}
            for x, _ in inputs:
                counts[tuple(x.shape[-2:])] = counts.get(tuple(x.shape[-2:]), 0) + 1
            expected_counts = {planner_hw: solver_calls(sampler_name, steps)}
            expected_counts.update({hw: solver_calls(sampler_name, free) for hw in sizes})
            assert counts == expected_counts, (sampler_name, counts, expected_counts)
            again = img2img(model, sampler_name, steps, latent_hw=latent_hw, **stage_options)[0]
            for a, b in zip(result, again):
                assert torch.equal(a["samples"], b["samples"]), sampler_name
            print(json.dumps({"case": "img2img_native", "sampler": sampler_name, "options": stage_options,
                              "start_index": start, "start_sigma": float(expected_sigmas[0]),
                              "model_calls": {f"{h}x{w}": n for (h, w), n in counts.items()}}))

    # Planner-final upscale-only bypasses all target DiT inference, unlike
    # img2img. This is the lossless-with-respect-to-the-upscaler baseline for
    # diagnosing anatomy drift in the final diffusion pass.
    for method in ("interpolate", "sesqui_wan21"):
        for options, latent_hw, hw_list in (
            ({"max_stage_upscale": 0.0}, (16, 20), [(16, 20)]),
            ({"stage_megapixels": "0.02"}, (24, 28), [(16, 20), (24, 28)]),
        ):
            result, _, target_stages, inputs = img2img(
                model, "euler", 8, mode="planner_final_upscale_only",
                latent_upscale_method=method, latent_hw=latent_hw,
                capture_inputs=True, **options,
            )
            assert not target_stages  # native target guider never created
            planner = result[1]["samples"]
            assert tuple(result[0]["samples"].shape[-2:]) == tuple(latent_hw)
            assert torch.equal(result[0]["samples"], result[2]["samples"])
            assert all(tuple(x.shape[-2:]) == tuple(planner.shape[-2:]) for x, _ in inputs)
            expected = planner
            for hw in hw_list:
                expected = runtime._enlarge_stage_source(
                    model, expected, hw, result[0]["samples"], method)
            torch.testing.assert_close(
                result[0]["samples"], expected.to(result[0]["samples"].device),
                rtol=0, atol=0,
            )
            repeated = img2img(model, "euler", 8, mode="planner_final_upscale_only",
                              latent_upscale_method=method, latent_hw=latent_hw,
                              **options)[0]
            for left, right in zip(result, repeated):
                assert torch.equal(left["samples"], right["samples"])
            print(json.dumps({"case": "planner_final_upscale_only_native",
                              "method": method, "options": options,
                              "target_dit_evaluations": 0}))

    # Native-range refinement: planner -> one in-range img2img pass ->
    # trained SesquiLSR final enlargement, with no final-resolution DiT.
    original_native_limit = runtime.KREA2_NATIVE_MAX_AXIS_PX
    runtime.KREA2_NATIVE_MAX_AXIS_PX = 256  # small real native model for CPU CI
    try:
        assert runtime.krea2_native_refine_hw((12, 12), (48, 48)) == (32, 32)
        for invalid_planner, invalid_target in (
            ((32, 32), (48, 48)),
            ((12, 12), (24, 24)),
        ):
            try:
                runtime.krea2_native_refine_hw(invalid_planner, invalid_target)
            except ValueError:
                pass
            else:
                raise AssertionError("unsupported native-refine geometry was accepted")
        for sampler_name in ("euler", "sa_solver"):
            result, _, stages, inputs = img2img(
                model, sampler_name, 8, latent_hw=(48, 48),
                mode="planner_final_native_refine",
                latent_upscale_method="sesqui_wan21",
                capture_inputs=True)
            assert len(stages) == 1
            assert tuple(stages[0]["latent"].shape[-2:]) == (32, 32)
            assert all(tuple(x.shape[-2:]) != (48, 48) for x, _ in inputs)
            planner = result[1]["samples"]
            assert all(tuple(x.shape[-2:]) in (tuple(planner.shape[-2:]), (32, 32))
                       for x, _ in inputs)
            expected = runtime._enlarge_stage_source(
                model, stages[0]["out"], (48, 48),
                torch.zeros_like(result[0]["samples"]), "sesqui_wan21")
            torch.testing.assert_close(
                result[0]["samples"], expected.cpu(), atol=0, rtol=0)
            assert torch.equal(result[0]["samples"], result[2]["samples"])
            again, _, _, _ = img2img(
                model, sampler_name, 8, latent_hw=(48, 48),
                mode="planner_final_native_refine",
                latent_upscale_method="sesqui_wan21")
            for a, b in zip(result, again):
                assert torch.equal(a["samples"], b["samples"])
            print(json.dumps({"case": "native_range_refinement",
                              "sampler": sampler_name,
                              "target_dit_above_256px": 0,
                              "native_stage_hw": list(stages[0]["latent"].shape[-2:])}))
    finally:
        runtime.KREA2_NATIVE_MAX_AXIS_PX = original_native_limit

    # The user's 20-step 0.30 vs 0.35 example must not alias to an identical
    # discrete start sigma (both previously snapped to 0.259758).
    for sampler_name in ("euler", "sa_solver"):
        low, _, low_stages, low_inputs = img2img(
            model, sampler_name, 20, start_sigma=0.30, capture_inputs=True)
        high, _, high_stages, high_inputs = img2img(
            model, sampler_name, 20, start_sigma=0.35, capture_inputs=True)
        assert torch.equal(low[1]["samples"], high[1]["samples"]), sampler_name
        assert len(low_stages) == len(high_stages) == 1
        assert len(low_stages[0]["sigmas"]) == len(high_stages[0]["sigmas"])
        assert torch.equal(low_stages[0]["sigmas"][1:], high_stages[0]["sigmas"][1:])
        assert abs(float(low_stages[0]["sigmas"][0]) - 0.30) < 1e-6
        assert abs(float(high_stages[0]["sigmas"][0]) - 0.35) < 1e-6
        assert torch.equal(low_stages[0]["noise"], high_stages[0]["noise"])
        target_hw = tuple(low_stages[0]["latent"].shape[-2:])
        low_first = next(x for x, _ in low_inputs if tuple(x.shape[-2:]) == target_hw)
        high_first = next(x for x, _ in high_inputs if tuple(x.shape[-2:]) == target_hw)
        assert not torch.equal(low_first, high_first), sampler_name

    # Retired fidelity projection must reject all nonzero settings before
    # planner execution; legacy detail widget values remain a no-op at
    # fidelity=0 for saved workflows.
    for detail in (0.0, 0.2, 1.0):
        result, _, _, _ = img2img(
            model, "euler", 8, planner_fidelity_strength=0.0,
            planner_detail_strength=detail)
        if detail == 0.0:
            baseline_zero_fidelity = result
        else:
            for a, b in zip(baseline_zero_fidelity, result):
                assert torch.equal(a["samples"], b["samples"])

    # Explicit trajectory mode ignores the img2img sigma and equals the default.
    for sampler_name in ("euler", "sa_solver"):
        explicit = img2img(model, sampler_name, 8, mode="trajectory", stage_megapixels="0.02",
                           latent_hw=(24, 28))[0]
        default = controls(model, sampler_name, "simple", 1.0, 8, False,
                           latent=torch.zeros(1, 4, 24, 28), stage_megapixels="0.02")[0]
        for a, b in zip(explicit, default):
            assert torch.equal(a["samples"], b["samples"]), sampler_name

    # Diagnostics are observational: outputs are bitwise unchanged, and the
    # saved .latent files load as ComfyUI SaveLatent files.
    created = []
    original_diagnostics = runtime.Krea2HandoffDiagnostics

    class CapturedDiagnostics(original_diagnostics):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created.append(self)

    runtime.Krea2HandoffDiagnostics = CapturedDiagnostics
    previous_output = folder_paths.get_output_directory()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            folder_paths.set_output_directory(tmp)
            for mode in ("planner_final_img2img", "trajectory"):
                for sampler_name in ("euler", "dpmpp_2m", "sa_solver_pece"):
                    plain = img2img(model, sampler_name, 8, mode=mode, stage_megapixels="0.02",
                                    latent_hw=(24, 28))[0]
                    created.clear()
                    observed, _, stages, _ = img2img(model, sampler_name, 8, mode=mode, stage_megapixels="0.02",
                                                     latent_hw=(24, 28),
                                                     handoff_diagnostics="log_and_save_latents")
                    for a, b in zip(plain, observed):
                        assert torch.equal(a["samples"], b["samples"]), (mode, sampler_name)
                    diag = created[-1]
                    records = diag.records
                    assert {r["stage"] for r in records} == {0, 1}, records
                    if mode == "planner_final_img2img":
                        assert all(not r["locked"] for r in records)
                        firsts = [r for r in records if r["step"] == 0]
                        assert len(firsts) == 2 and all(r["state_clean_vs_stage_source"] < 1e-4 for r in firsts), firsts
                        state = safetensors.torch.load_file(
                            next(p for p in diag.saved if p.endswith("stage0_step000_state.latent")))
                        assert "latent_format_version_0" in state
                        source_path = next(p for p in diag.saved
                                           if p.endswith("stage0_source_resized.latent"))
                        source_saved = safetensors.torch.load_file(source_path)["latent_tensor"]
                        source_expected = runtime.resize_4d_tensor(
                            observed[1]["samples"].float(),
                            tuple(stages[0]["latent"].shape[-2:]),
                        )
                        torch.testing.assert_close(source_saved, source_expected)
                        assert any(p.endswith("stage1_source_resized.latent")
                                   for p in diag.saved)
                        expected_state = base.process_latent_out(
                            stages[0]["sigmas"][0] * stages[0]["noise"]
                            + (1 - stages[0]["sigmas"][0]) * base.process_latent_in(stages[0]["latent"].float()))
                        torch.testing.assert_close(state["latent_tensor"], expected_state, atol=1e-4, rtol=1e-4)
                    else:
                        assert any(r["locked"] for r in records) or sampler_name == "euler"
                    assert any(p.endswith("planner_final.latent") for p in diag.saved)
                    assert any(p.endswith("stage0_completed.latent") for p in diag.saved)
                    assert any(p.endswith("stage1_completed.latent") for p in diag.saved)
                    final_saved = safetensors.torch.load_file(
                        next(p for p in diag.saved if p.endswith("final_output.latent"))
                    )["latent_tensor"]
                    torch.testing.assert_close(final_saved, observed[0]["samples"].float())
                    print(json.dumps({"case": "handoff_diagnostics", "mode": mode, "sampler": sampler_name,
                                      "files": len(diag.saved),
                                      "first_records": [r for r in records if r["stage"] == 0][:2]}))
    finally:
        folder_paths.set_output_directory(previous_output)
        runtime.Krea2HandoffDiagnostics = original_diagnostics

    # Disable the mathematically degenerate fidelity projection before
    # running the planner, regardless of retained saved-workflow controls.
    calls.clear()
    hook = base.diffusion_model.register_forward_pre_hook(
        lambda module, args: calls.append(1))
    try:
        node.sample(model, positive, negative,
                    {"samples": torch.zeros(1, 4, 16, 20)}, 42, .01, 0.0, 0.0,
                    steps=8, regeneration_mode="planner_final_img2img",
                    planner_fidelity_strength=1.0)
    except ValueError as exc:
        assert "disabled after real Krea 2 checkpoint" in str(exc), exc
    else:
        raise AssertionError("visually broken planner-fidelity projection must be disabled")
    finally:
        hook.remove()
    assert not calls

    # Invalid img2img configurations fail before any model evaluation.
    for extra, message in (({"img2img_start_sigma": .01}, "no usable refinement"),
                           ({"img2img_start_sigma": 1.0}, "strictly between")):
        calls.clear()
        hook = base.diffusion_model.register_forward_pre_hook(lambda module, args: calls.append(1))
        try:
            node.sample(model, positive, negative, {"samples": torch.zeros(1, 4, 16, 20)}, 42, .01, 0.0, 0.0,
                        steps=8, regeneration_mode="planner_final_img2img", **extra)
        except ValueError as exc:
            assert message in str(exc), exc
        else:
            raise AssertionError(extra)
        finally:
            hook.remove()
        assert not calls
    calls.clear()
    hook = base.diffusion_model.register_forward_pre_hook(lambda module, args: calls.append(1))
    try:
        node.sample(model, positive, negative,
                    {"samples": torch.zeros(1, 4, 16, 20), "noise_mask": torch.ones(1, 16, 20)},
                    42, .01, 0.0, 0.0, steps=8, regeneration_mode="planner_final_img2img")
    except ValueError as exc:
        assert "noise masks" in str(exc), exc
    else:
        raise AssertionError("latent noise masks must be rejected in img2img mode")
    finally:
        hook.remove()
    assert not calls, "the planner must not run for a rejected img2img configuration"
    try:
        node.sample(model, positive, negative, {"samples": torch.zeros(1, 4, 16, 20)}, 42, .01, 0.2, 0.0,
                    steps=8, regeneration_mode="planner_final_img2img")
    except ValueError as exc:
        assert "coarse_lock_strength" in str(exc)
    else:
        raise AssertionError("trajectory lock strengths must be rejected in img2img mode")

    # Interruption inside a high-resolution stage leaves no hooks or Spectrum
    # state behind; the next run reproduces the uninterrupted result exactly.
    for label, selected in cases:
        for sampler_name in ("euler", "sa_solver"):
            options = {"stage_megapixels": "0.02", "latent_hw": (24, 28)}
            if sampler_name == "sa_solver":
                options["sa_spectrum_forecasts"] = False
            reference, ref_reports, _, _ = img2img(selected, sampler_name, 8, **options)
            try:
                img2img(selected, sampler_name, 8, fail=((24, 28), 1), **options)
            except comfy.model_management.InterruptProcessingException:
                pass
            else:
                raise AssertionError("interrupt was not raised")
            assert not base.diffusion_model._forward_pre_hooks
            assert not base.diffusion_model.last._forward_pre_hooks
            repeated, rep_reports, _, _ = img2img(selected, sampler_name, 8, **options)
            for a, b in zip(reference, repeated):
                assert torch.equal(a["samples"], b["samples"]), (label, sampler_name)
            if label == "spectrum":
                assert ref_reports == rep_reports, (ref_reports, rep_reports)
                if sampler_name == "sa_solver":
                    # The scoped native-only flag was restored after the interrupt.
                    forecasting = img2img(selected, sampler_name, 8, stage_megapixels="0.02",
                                          latent_hw=(24, 28))[1]
                    assert not any(r.get("native_only") for r in forecasting), forecasting

    # Spectrum stays operational in the planner and in every img2img stage.
    if reports_context is not None:
        aligned = SpectrumApplyKrea2().apply(
            model, preset="turbo", warmup=2, tail=1, bootstrap=False,
            extrapolation_guard_enabled=False,
        )[0]
        steps = 20
        schedule = runtime.calculate_sigmas(model, scheduler="simple", steps=steps, denoise=1.0)
        free = steps - runtime.img2img_start_index(schedule.tolist(), img2img_sigma)
        for sampler_name in runtime.KREA2_SAMPLERS:
            for stage_options, n_stages in (({"max_stage_upscale": 0.0}, 1), ({"stage_megapixels": "0.02"}, 2)):
                latent_hw = (16, 20) if n_stages == 1 else (24, 28)
                result, reports, _, _ = img2img(aligned, sampler_name, steps, latent_hw=latent_hw, **stage_options)
                assert len(reports) == 1 + n_stages, (sampler_name, reports)
                assert reports[0]["outer_steps"] == steps
                assert [r["outer_steps"] for r in reports[1:]] == [free] * n_stages, reports
                assert all(r["fallback"] is None for r in reports), reports
                assert all(r["forecast"] > 0 for r in reports), (sampler_name, reports)
                assert not base.diffusion_model._forward_pre_hooks
                print(json.dumps({"case": "img2img_spectrum", "sampler": sampler_name, "stages": n_stages,
                                  "stage_steps": free,
                                  "reports": [{k: r[k] for k in ("outer_steps", "actual", "forecast",
                                                                  "denoiser_calls", "fallback")} for r in reports]}))

    # ---- latent_upscale_method=sesqui_wan21: completed-latent handoffs use
    # the pretrained SesquiLSR Wan21 upscaler. An independently loaded copy of
    # the vendored upstream architecture and weights is the reference.
    import hashlib
    sesqui_dir = Path(__file__).resolve().parents[1] / "third_party" / "sesqui_lsr"
    ref_spec = importlib.util.spec_from_file_location("sesqui_reference_upscaler", sesqui_dir / "upscaler.py")
    ref_module = importlib.util.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(ref_module)
    weights = sesqui_dir / "upscaler_Wan21.safetensors"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == \
        "d1b4c2c1d80b2af81e6bbfd38501ffb5adffb9f7c0ed2599555b837d0d2d015a"
    reference_upscaler = ref_module.LatentUpscaler(in_channels=16)
    reference_upscaler.load_state_dict(safetensors.torch.load_file(str(weights)))
    reference_upscaler.eval()
    latent_upscale = sys.modules[spec.name + ".latent_upscale"]

    def sesqui_reference(source, hw):
        x = source.detach().float().cpu()
        temporal = x.ndim == 5
        x = x.squeeze(2) if temporal else x
        with torch.no_grad():
            for hop in latent_upscale.sesqui_hops(x.shape[-2:], hw):
                x = reference_upscaler(x, hop)
        return x.unsqueeze(2) if temporal else x

    def first_inputs(inputs, hw):
        return next((x, t) for x, t in inputs if tuple(x.shape[-2:]) == hw)

    for sampler_name in runtime.KREA2_SAMPLERS:
        steps = 8
        schedule = runtime.calculate_sigmas(model, scheduler="simple", steps=steps, denoise=1.0)
        _, expected_sigmas = runtime.img2img_refinement_sigmas(schedule, img2img_sigma)
        free = len(expected_sigmas) - 1
        for stage_options, sizes in (({"max_stage_upscale": 0.0}, [(16, 20)]),
                                     ({"stage_megapixels": "0.02"}, [(16, 20), (24, 28)])):
            latent_hw = sizes[-1]
            plain, _, plain_stages, _ = img2img(model, sampler_name, steps, latent_hw=latent_hw, **stage_options)
            learned, _, stages, inputs = img2img(model, sampler_name, steps, latent_hw=latent_hw,
                                                 capture_inputs=True, latent_upscale_method="sesqui_wan21",
                                                 **stage_options)
            planner = learned[1]["samples"]
            assert torch.equal(planner, plain[1]["samples"]), sampler_name  # planner untouched
            assert len(stages) == len(sizes)
            for index, (stage, hw) in enumerate(zip(stages, sizes)):
                source = planner if index == 0 else stages[index - 1]["out"]
                expected = sesqui_reference(source, hw)
                torch.testing.assert_close(stage["latent"].float().cpu(), expected, rtol=1e-5, atol=1e-5)
                assert not torch.allclose(stage["latent"].float().cpu(),
                                          runtime.resize_4d_tensor(source.float().cpu(), hw), atol=1e-3)
                # Noise, sigma schedule and native initialization are unchanged.
                assert torch.equal(stage["noise"], plain_stages[index]["noise"]), sampler_name
                assert torch.equal(stage["sigmas"], plain_stages[index]["sigmas"]), sampler_name
                first, timestep = first_inputs(inputs, hw)
                sigma = stage["sigmas"][0]
                expected_x = sigma * stage["noise"].float().cpu() + (1 - sigma) * base.process_latent_in(
                    stage["latent"].float().cpu())
                torch.testing.assert_close(first.reshape(expected_x.shape), expected_x, atol=1e-5, rtol=1e-5)
            counts = {}
            for x, _ in inputs:
                counts[tuple(x.shape[-2:])] = counts.get(tuple(x.shape[-2:]), 0) + 1
            expected_counts = {tuple(planner.shape[-2:]): solver_calls(sampler_name, steps)}
            expected_counts.update({hw: solver_calls(sampler_name, free) for hw in sizes})
            assert counts == expected_counts, (sampler_name, counts)
            again = img2img(model, sampler_name, steps, latent_hw=latent_hw,
                            latent_upscale_method="sesqui_wan21", **stage_options)[0]
            for a, b in zip(learned, again):
                assert torch.equal(a["samples"], b["samples"]), sampler_name
            print(json.dumps({"case": "sesqui_img2img_native", "sampler": sampler_name, "options": stage_options,
                              "model_calls": {f"{h}x{w}": n for (h, w), n in counts.items()}}))

    # Trajectory ladder: the first stage follows planner predictions exactly
    # as before; only the completed stage-to-stage handoff is learned.
    for sampler_name in ("euler", "dpmpp_2m", "er_sde", "sa_solver_pece"):
        plain, _, plain_stages, _ = img2img(model, sampler_name, 8, mode="trajectory", latent_hw=(24, 28),
                                            stage_megapixels="0.02")
        learned, _, stages, _ = img2img(model, sampler_name, 8, mode="trajectory", latent_hw=(24, 28),
                                        stage_megapixels="0.02", latent_upscale_method="sesqui_wan21")
        assert len(stages) == len(plain_stages) == 2
        assert torch.equal(stages[0]["out"], plain_stages[0]["out"]), sampler_name
        assert torch.equal(stages[0]["latent"], plain_stages[0]["latent"]), sampler_name
        torch.testing.assert_close(stages[1]["latent"].float().cpu(),
                                   sesqui_reference(stages[0]["out"], (24, 28)), rtol=1e-5, atol=1e-5)
        assert torch.equal(stages[1]["sigmas"], plain_stages[1]["sigmas"])
        assert torch.equal(stages[1]["noise"], plain_stages[1]["noise"])

    # A direct jump above 2x per side uses two latent-only learned passes and
    # still exactly one diffusion stage with the same model-call count.
    result, _, stages, inputs = img2img(model, "euler", 8, latent_hw=(32, 36), capture_inputs=True,
                                        max_stage_upscale=0.0, latent_upscale_method="sesqui_wan21")
    planner_hw = tuple(result[1]["samples"].shape[-2:])
    assert max(32 / planner_hw[0], 36 / planner_hw[1]) > 2.0, planner_hw
    assert len(latent_upscale.sesqui_hops(planner_hw, (32, 36))) == 2
    assert len(stages) == 1 and tuple(stages[0]["latent"].shape[-2:]) == (32, 36)
    torch.testing.assert_close(stages[0]["latent"].float().cpu(),
                               sesqui_reference(result[1]["samples"], (32, 36)), rtol=1e-5, atol=1e-5)
    _, expected_sigmas = runtime.img2img_refinement_sigmas(
        runtime.calculate_sigmas(model, scheduler="simple", steps=8, denoise=1.0), img2img_sigma)
    assert sum(tuple(x.shape[-2:]) == (32, 36) for x, _ in inputs) == len(expected_sigmas) - 1

    # Regression: a learned-upscaler preflight previously rejected valid
    # single-stage trajectory graphs when sesqui_wan21 was selected. The first
    # target stage follows planner predictions, not a completed clean latent.
    # An unused selection must preserve native execution and report a no-op.
    original_ensure = runtime.ensure_latent_upscaler
    def unexpected_weight_preflight(method):
        raise AssertionError("single-stage trajectory must not load unused upscaler weights")
    runtime.ensure_latent_upscaler = unexpected_weight_preflight
    try:
        for label, selected in cases:
            for sampler_name in ("euler", "sa_solver"):
                baseline, baseline_reports, baseline_stages, baseline_inputs = img2img(
                    selected, sampler_name, 8, mode="trajectory",
                    latent_hw=(16, 20), max_stage_upscale=0.0,
                    capture_inputs=True,
                )
                selected_learned, learned_reports, learned_stages, learned_inputs = img2img(
                    selected, sampler_name, 8, mode="trajectory",
                    latent_hw=(16, 20), max_stage_upscale=0.0,
                    latent_upscale_method="sesqui_wan21",
                    capture_inputs=True,
                )
                assert len(baseline_stages) == len(learned_stages) == 1
                for reference, observed in zip(baseline, selected_learned):
                    assert torch.equal(reference["samples"], observed["samples"]), (label, sampler_name)
                assert baseline_reports == learned_reports, (label, sampler_name)
                assert len(baseline_inputs) == len(learned_inputs)
                assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                           for a, b in zip(baseline_inputs, learned_inputs))
                print(json.dumps({"case": "sesqui_trajectory_single_stage_compat",
                                  "model": label, "sampler": sampler_name,
                                  "target_calls_unchanged": True, "upscaler_loaded": False}))
    finally:
        runtime.ensure_latent_upscaler = original_ensure

    # Automatic stage planning must still use SesquiLSR for a subsequent
    # finished-latent handoff when the requested area growth exceeds the cap.
    auto, _, auto_stages, _ = img2img(
        model, "euler", 8, mode="trajectory", latent_hw=(32, 36),
        max_stage_upscale=4.0, latent_upscale_method="sesqui_wan21",
    )
    assert len(auto_stages) > 1, "expected automatic intermediate stage"
    torch.testing.assert_close(
        auto_stages[-1]["latent"].float().cpu(),
        sesqui_reference(auto_stages[-2]["out"], (32, 36)),
        rtol=1e-5, atol=1e-5,
    )
    assert tuple(auto[0]["samples"].shape[-2:]) == (32, 36)

    # Diagnostics measure against the actual learned start latent.
    created.clear()
    runtime.Krea2HandoffDiagnostics = CapturedDiagnostics
    try:
        img2img(model, "euler", 8, latent_hw=(24, 28), stage_megapixels="0.02",
                latent_upscale_method="sesqui_wan21", handoff_diagnostics="log")
    finally:
        runtime.Krea2HandoffDiagnostics = original_diagnostics
    firsts = [r for r in created[-1].records if r["step"] == 0]
    assert len(firsts) == 2 and all(r["state_clean_vs_stage_source"] < 1e-4 for r in firsts), firsts

    # Interrupts inside a learned-handoff stage leave no hooks or Spectrum
    # state behind, and Spectrum keeps forecasting in every stage.
    for label, selected in cases:
        options = {"stage_megapixels": "0.02", "latent_hw": (24, 28), "latent_upscale_method": "sesqui_wan21"}
        reference, ref_reports, _, _ = img2img(selected, "euler", 8, **options)
        try:
            img2img(selected, "euler", 8, fail=((24, 28), 1), **options)
        except comfy.model_management.InterruptProcessingException:
            pass
        else:
            raise AssertionError("interrupt was not raised")
        assert not base.diffusion_model._forward_pre_hooks
        repeated, rep_reports, _, _ = img2img(selected, "euler", 8, **options)
        for a, b in zip(reference, repeated):
            assert torch.equal(a["samples"], b["samples"]), label
        if label == "spectrum":
            assert ref_reports == rep_reports
    if reports_context is not None:
        for sampler_name in ("euler", "er_sde", "sa_solver", "sa_solver_pece"):
            for mode in ("planner_final_img2img", "trajectory"):
                _, reports, _, _ = img2img(aligned, sampler_name, 20, latent_hw=(24, 28), mode=mode,
                                           stage_megapixels="0.02", latent_upscale_method="sesqui_wan21")
                assert len(reports) == 3, reports
                assert all(r["fallback"] is None for r in reports), reports
                assert all(r["forecast"] > 0 for r in reports), (sampler_name, mode, reports)
                print(json.dumps({"case": "sesqui_spectrum", "sampler": sampler_name, "mode": mode,
                                  "reports": [{k: r[k] for k in ("outer_steps", "actual", "forecast", "fallback")}
                                              for r in reports]}))

    try:
        node.sample(model, positive, negative, {"samples": torch.zeros(1, 4, 24, 28)},
                    42, .01, 0.0, 0.0, steps=2, composition_lock_sigma=0.0, stage_megapixels="0.02")
    except ValueError as exc:
        assert "at least one step" in str(exc)
    else:
        raise AssertionError("too few steps for the stages must be rejected before sampling")

    review_regressions(node, model, positive, negative, runtime, base, spec.name)
    progressive_regressions(node, model, positive, negative, runtime, base, package,
                            dict(cases).get("spectrum"), reports_context)


def review_regressions(node, model, positive, negative, runtime, base, package_name):
    """Diagnostics on zero-DiT paths, and SA-PECE corrected anchors under a noise mask."""
    import folder_paths
    import safetensors.torch
    import torch
    import tempfile
    loader = sys.modules[package_name + ".diagnostic_load"]
    previous_output = folder_paths.get_output_directory()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            folder_paths.set_output_directory(tmp)
            # Every target step locked (Euler, no stages, no mask): the target
            # pass returns before sampling and must still save a loadable run.
            common = dict(steps=8, sampler_name="euler", max_stage_upscale=0.0, composition_lock_sigma=0.0)
            with torch.inference_mode():
                plain = node.sample(model, positive, negative, {"samples": torch.zeros(1, 4, 16, 20)},
                                    42, .01, 0.0, 0.0, **common)
                observed = node.sample(model, positive, negative, {"samples": torch.zeros(1, 4, 16, 20)},
                                       42, .01, 0.0, 0.0, handoff_diagnostics="log_and_save_latents", **common)
            assert torch.equal(plain[0]["samples"], observed[0]["samples"])
            directory, paths = loader._resolve_diagnostic_run("latest")
            final = safetensors.torch.load_file(str(paths[-1]))["latent_tensor"]
            torch.testing.assert_close(final, observed[0]["samples"].float())
            print(json.dumps({"case": "fully_locked_diagnostics", "run": directory.name}))
            # Upscale-only makes no target prediction; its run must still load.
            with torch.inference_mode():
                enlarged = node.sample(model, positive, negative, {"samples": torch.zeros(1, 4, 24, 28)},
                                       42, .01, 0.0, 0.0, steps=8, sampler_name="euler",
                                       regeneration_mode="planner_final_upscale_only", stage_megapixels="0.02",
                                       composition_lock_sigma=0.0, handoff_diagnostics="log_and_save_latents")
            upscaled_dir, upscaled_paths = loader._resolve_diagnostic_run("latest")
            assert upscaled_dir != directory
            torch.testing.assert_close(safetensors.torch.load_file(str(upscaled_paths[-1]))["latent_tensor"],
                                       enlarged[0]["samples"].float())
            assert not any(p.name.endswith(".partial") for p in upscaled_dir.iterdir())
            print(json.dumps({"case": "upscale_only_diagnostics", "run": upscaled_dir.name}))
    finally:
        folder_paths.set_output_directory(previous_output)

    # SA-PECE planner with a partial latent noise mask: corrected anchors must
    # carry KSamplerX0Inpaint's blend, i.e. equal the latent where mask == 0.
    captured = {}
    original_planner = runtime._run_lowres_planner

    def capture_planner(**kwargs):
        result = original_planner(**kwargs)
        captured["latent"] = kwargs["lowres_latent"]
        captured["corrections"] = result[4]
        return result

    torch.manual_seed(5)
    latent = torch.randn(1, 16, 16, 20) * 0.5
    mask = torch.ones(1, 16, 20)
    mask[..., :, :10] = 0.0
    runtime._run_lowres_planner = capture_planner
    try:
        with torch.inference_mode():
            node.sample(model, positive, negative, {"samples": latent, "noise_mask": mask},
                        42, .01, 0.3, 0.0, steps=8, sampler_name="sa_solver_pece", max_stage_upscale=0.0,
                        sa_spectrum_forecasts=False)
    finally:
        runtime._run_lowres_planner = original_planner
    low = captured["latent"]
    import comfy.utils
    low_mask = comfy.utils.reshape_mask(low["noise_mask"], tuple(low["samples"].shape))
    kept = base.process_latent_in(low["samples"].float())
    corrections = [c for c in captured["corrections"] if c is not None]
    assert corrections
    for correction in corrections:
        region = low_mask < 0.5
        assert bool(region.any())
        torch.testing.assert_close(correction.float()[region], kept[region], atol=2e-3, rtol=2e-3)
    print(json.dumps({"case": "sa_pece_masked_corrections", "checked": len(corrections)}))


def progressive_regressions(node, model, positive, negative, runtime, base, package, spectrum_model, reports_context):
    """spectral_progressive through genuine ComfyUI guiders, samplers and Krea 2 DiT calls."""
    import torch
    import comfy.model_management
    spectral = sys.modules[package.__name__ + ".spectral_progressive"]

    def run(selected, sampler_name, steps, *, latent_hw=(24, 28), fail=None, record=None, **extra):
        if sampler_name in ("sa_solver", "sa_solver_pece"):
            extra.setdefault("sa_spectrum_forecasts", True)
        calls, reports = [], []
        token = reports_context.set(reports) if reports_context is not None else None
        original_preview = runtime.make_preview_callback

        def preview(*a, **kw):
            inner = original_preview(*a, **kw)

            def callback(step, x0, x, total):
                if record is not None:
                    record.append((step, tuple(x.shape[-2:]), x.detach().clone()))
                if fail is not None and tuple(x.shape[-2:]) == fail[0] and step == fail[1]:
                    raise comfy.model_management.InterruptProcessingException()
                inner(step, x0, x, total)
            return callback

        runtime.make_preview_callback = preview
        hook = (base.diffusion_model.register_forward_pre_hook(
            lambda module, args: calls.append(tuple(args[0].shape[-2:]))) if selected is model else None)
        try:
            with torch.inference_mode():
                result = node.sample(selected, positive, negative, {"samples": torch.zeros(1, 4, *latent_hw)},
                                     42, .01, 0.0, 0.0, steps=steps, sampler_name=sampler_name,
                                     regeneration_mode="spectral_progressive", **extra)
        finally:
            runtime.make_preview_callback = original_preview
            if hook is not None:
                hook.remove()
            if reports_context is not None:
                reports_context.reset(token)
        assert not base.diffusion_model._forward_pre_hooks
        assert all(torch.isfinite(x["samples"]).all() for x in result), sampler_name
        counts = {shape: calls.count(shape) for shape in set(calls)}
        return result, counts, reports

    def expected_split(steps, lock=0.80, sampler_name="euler"):
        schedule = runtime.calculate_sigmas(model, scheduler="simple", steps=steps, denoise=1.0)
        if sampler_name in runtime.SNR_SOLVER_NAMES:
            schedule = runtime.effective_er_sde_sigmas(schedule, model)
        return next(i for i, v in enumerate(schedule.tolist()) if v <= lock)

    # Every offered solver: single transition at the composition-lock release.
    planner_hw = None
    for sampler_name in runtime.krea2_sampler_names():
        steps = 8
        k = expected_split(steps, sampler_name=sampler_name)
        result, counts, _ = run(model, sampler_name, steps)
        repeat, _, _ = run(model, sampler_name, steps)
        for a, b in zip(result, repeat):
            assert torch.equal(a["samples"], b["samples"]), sampler_name
        planner_hw = tuple(result[1]["samples"].shape[-2:])
        assert list(result[0]["samples"].shape) == [1, 16, 1, 24, 28]
        if sampler_name == "sa_solver_pece":
            # Each segment is a fresh PECE run: no correction on its first step.
            expected = {planner_hw: 2 * k - 1, (24, 28): 2 * (steps - k) - 1}
        else:
            expected = {planner_hw: k, (24, 28): steps - k}
        assert counts == expected, (sampler_name, counts, expected)
        print(json.dumps({"case": "progressive_solver", "sampler": sampler_name, "split_step": k,
                          "dit_calls": {f"{h}x{w}": c for (h, w), c in counts.items()}}))

    # The first segment reproduces the planner prefix; the second segment
    # starts from exactly the expanded, kappa-scaled state at the aligned sigma.
    record = []
    captured = {}
    original_expand = runtime.expand_flow_state

    def spy(x, hw, sigma, **kw):
        out, aligned = original_expand(x, hw, sigma, **kw)
        captured.update(source=x.detach().clone(), out=out.detach().clone(), sigma=sigma, aligned=aligned, kw=kw)
        return out, aligned
    runtime.expand_flow_state = spy
    try:
        run(model, "euler", 8, record=record)
    finally:
        runtime.expand_flow_state = original_expand
    k = expected_split(8)
    second = [x for step, hw, x in record if hw == (24, 28)]
    assert len(second) == 8 - k
    torch.testing.assert_close(second[0].float(), captured["out"].float(), atol=1e-5, rtol=1e-5)
    r = spectral.area_ratio(planner_hw, (24, 28))
    assert abs(captured["aligned"] - spectral.align_sigma(captured["sigma"], r)) < 1e-12
    src = captured["source"].float().squeeze(2)
    low = spectral.dct2(captured["out"].float().squeeze(2))[..., :planner_hw[0], :planner_hw[1]]
    torch.testing.assert_close(low / spectral.kappa(captured["sigma"], r), spectral.dct2(src), atol=1e-4, rtol=1e-4)
    trajectory_record = []
    original_preview = runtime.make_preview_callback

    def planner_preview(*a, **kw):
        inner = original_preview(*a, **kw)

        def callback(step, x0, x, total):
            trajectory_record.append((step, tuple(x.shape[-2:]), x.detach().clone()))
            inner(step, x0, x, total)
        return callback
    runtime.make_preview_callback = planner_preview
    try:
        with torch.inference_mode():
            node.sample(model, positive, negative, {"samples": torch.zeros(1, 4, 24, 28)}, 42, .01, 0.0, 0.0, steps=8)
    finally:
        runtime.make_preview_callback = original_preview
    planner_states = [x for step, hw, x in trajectory_record if hw == planner_hw][:k]
    progressive_states = [x for step, hw, x in record if hw == planner_hw]
    assert len(progressive_states) == k
    for a, b in zip(planner_states, progressive_states):
        assert torch.equal(a, b)
    print(json.dumps({"case": "progressive_handoff_state", "split_step": k, "aligned_sigma": captured["aligned"]}))

    # Mode switching must not require clearing a stored manual spectral
    # schedule or silently change the non-spectral numerical result.
    for mode in ("trajectory", "planner_final_img2img"):
        options = dict(steps=8, regeneration_mode=mode, sampler_name="euler",
                       max_stage_upscale=0.0)
        with torch.inference_mode():
            reference = node.sample(
                model, positive, negative, {"samples": torch.zeros(1, 4, 24, 28)},
                42, .01, 0.0, 0.0, progressive_transition_sigmas="", **options,
            )
            with_saved_schedule = node.sample(
                model, positive, negative, {"samples": torch.zeros(1, 4, 24, 28)},
                42, .01, 0.0, 0.0, progressive_transition_sigmas="0.80, 0.65", **options,
            )
        for actual, expected in zip(with_saved_schedule, reference):
            torch.testing.assert_close(actual["samples"], expected["samples"], rtol=0, atol=0)
        print(json.dumps({"case": "inactive_progressive_transitions", "mode": mode}))

    # Three resolutions with manual transitions and with a measured calibration.
    for options in ({"stage_megapixels": "0.02", "progressive_transition_sigmas": "0.9, 0.6"},):
        result, counts, _ = run(model, "dpmpp_2m", 12, **options)
        schedule = runtime.calculate_sigmas(model, scheduler="simple", steps=12, denoise=1.0).tolist()
        s1 = next(i for i, v in enumerate(schedule) if v <= 0.9)
        s2 = next(i for i, v in enumerate(schedule) if v <= 0.6)
        assert counts == {planner_hw: s1,
                          (16, 20): s2 - s1, (24, 28): 12 - s2}, counts
        print(json.dumps({"case": "progressive_manual_stages", "dit_calls": {f"{h}x{w}": c for (h, w), c in counts.items()}}))

    # Calibration from a synthetic power-law clean latent (no model calls).
    g = torch.Generator().manual_seed(5)
    fy = torch.fft.fftfreq(48, dtype=torch.float64).unsqueeze(1)
    fx = torch.fft.fftfreq(56, dtype=torch.float64).unsqueeze(0)
    radius = 48 * torch.sqrt(fy ** 2 + fx ** 2)
    amp = torch.where(radius > 0, radius.clamp(min=1e-9) ** -1.1, torch.zeros_like(radius))
    internal = torch.fft.ifft2(torch.fft.fft2(torch.randn(2, 16, 48, 56, generator=g, dtype=torch.float64), norm="ortho")
                               * amp, norm="ortho").real.float() * 6.0
    clean = base.process_latent_out(internal.unsqueeze(2))
    calibrator = package.NODE_CLASS_MAPPINGS["Krea2SpectralCalibration"]()
    # Native EmptyLatentImage supplies four zero channels. Calibration must
    # reject this canvas with wiring guidance rather than treating it as data.
    for wrong, message in (
        (torch.zeros(1, 4, 48, 56), "empty latent canvas"),
        (torch.ones(1, 4, 48, 56), "expected 16 latent channels, received 4"),
    ):
        try:
            calibrator.calibrate(model, {"samples": wrong})
        except ValueError as exc:
            assert message in str(exc) and "VAE Encode" in str(exc), str(exc)
        else:
            raise AssertionError("calibration accepted an invalid generation canvas")
    calibration, report = calibrator.calibrate(model, {"samples": clean})
    assert abs(calibration["beta"] - 2.2) < 0.15, report
    assert calibration["latent_format"] == "Wan21" and calibration["grid"] == (48, 56)
    # The 24x28 test canvas is within the 2x transfer range of a 48x56 fit.
    sizes = [planner_hw, (24, 28)]
    thresholds = spectral.spectral_transition_sigmas(sizes, calibration, 0.01)
    schedule = runtime.calculate_sigmas(model, scheduler="simple", steps=16, denoise=1.0).tolist()
    split = spectral.transition_steps(schedule, thresholds, sizes)[0]
    result, counts, _ = run(model, "euler", 16, spectral_calibration=calibration, spectral_tolerance=0.01)
    assert counts == {planner_hw: split, (24, 28): 16 - split}, (counts, thresholds)
    print(json.dumps({"case": "progressive_spectral", "report": report, "thresholds": thresholds, "split_step": split}))

    # Unsupported configurations fail before any model evaluation.
    rejected = [
        ({"coarse_lock_strength": 0.2}, "coarse_lock_strength"),
        ({"latent_upscale_method": "sesqui_wan21"}, "interpolate"),
        ({"handoff_diagnostics": "log_and_save_latents"}, "latents"),
        ({"progressive_transition_sigmas": "0.9", "spectral_calibration": calibration}, "both"),
        ({"stage_megapixels": "0.02"}, "intermediate"),
        ({"progressive_transition_sigmas": "0.001"}, "never reached"),
        ({"progressive_transition_sigmas": "0.9, 0.5"}, "transition sigma"),
        ({"composition_lock_sigma": 1.0}, "composition_lock_sigma"),
        ({"spectral_calibration": dict(calibration, latent_format="Flux")}, "latent format"),
        ({"spectral_calibration": dict(calibration, grid=(160, 160), reference_length=160)}, "transferred"),
    ]
    for extra, needle in rejected:
        calls = []
        hook = base.diffusion_model.register_forward_pre_hook(lambda module, args: calls.append(1))
        try:
            kwargs = dict(steps=8, regeneration_mode="spectral_progressive")
            if "coarse_lock_strength" in extra:
                args = (model, positive, negative, {"samples": torch.zeros(1, 4, 24, 28)}, 42, .01, extra.pop("coarse_lock_strength"), 0.0)
            else:
                args = (model, positive, negative, {"samples": torch.zeros(1, 4, 24, 28)}, 42, .01, 0.0, 0.0)
            node.sample(*args, **kwargs, **extra)
        except ValueError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError(f"spectral_progressive accepted {extra}")
        finally:
            hook.remove()
        assert not calls, extra
    mask_latent = {"samples": torch.zeros(1, 4, 24, 28), "noise_mask": torch.ones(1, 24, 28)}
    try:
        node.sample(model, positive, negative, mask_latent, 42, .01, 0.0, 0.0, steps=8,
                    regeneration_mode="spectral_progressive")
    except ValueError as exc:
        assert "noise mask" in str(exc)
    else:
        raise AssertionError("noise masks must be rejected")
    try:
        node.sample(model, positive, negative, {"samples": torch.zeros(1, 4, 24, 28)}, 42, .01, 0.0, 0.0,
                    steps=8, progressive_transition_sigmas="0.8")
    except ValueError as exc:
        assert "only used by" in str(exc)
    else:
        raise AssertionError("transition sigmas outside spectral_progressive must be rejected")

    # Interruption in the second segment leaves no hooks or state behind.
    reference, _, _ = run(model, "euler", 8)
    try:
        run(model, "euler", 8, fail=((24, 28), 1))
    except comfy.model_management.InterruptProcessingException:
        pass
    else:
        raise AssertionError("interrupt was not raised")
    again, _, _ = run(model, "euler", 8)
    for a, b in zip(reference, again):
        assert torch.equal(a["samples"], b["samples"])

    # Spectrum: one forecast session per resolution, never across a transition.
    if spectrum_model is not None:
        for sampler_name in ("euler", "er_sde", "sa_solver"):
            k = expected_split(16, sampler_name=sampler_name)
            _, _, reports = run(spectrum_model, sampler_name, 16)
            assert [r["outer_steps"] for r in reports] == [k, 16 - k], reports
            assert all(r["fallback"] is None for r in reports), reports
            print(json.dumps({"case": "progressive_spectrum", "sampler": sampler_name,
                              "reports": [{key: r[key] for key in ("outer_steps", "actual", "forecast", "fallback")}
                                          for r in reports]}))


if __name__ == "__main__":
    main()
