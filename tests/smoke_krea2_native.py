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
        from comfyui_spectrum_krea2.nodes import SpectrumApplyKrea2
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
        from comfyui_spectrum_krea2.nodes import SpectrumApplyKrea2
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

    try:
        node.sample(model, positive, negative, {"samples": torch.zeros(1, 4, 24, 28)},
                    42, .01, 0.0, 0.0, steps=2, composition_lock_sigma=0.0, stage_megapixels="0.02")
    except ValueError as exc:
        assert "at least one step" in str(exc)
    else:
        raise AssertionError("too few steps for the stages must be rejected before sampling")


if __name__ == "__main__":
    main()
