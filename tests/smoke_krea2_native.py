"""CPU integration with real ComfyUI and a tiny random native Krea 2 model.

This checks execution/layout only. It does not establish real-checkpoint image
quality or GPU speed. Run with ComfyUI's Python dependencies installed.
"""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def check_generic_isolation(runtime_module, model):
    """Reusing a guider must not carry the Krea projection into generic nodes."""
    import torch
    g = torch.Generator().manual_seed(12)
    base = torch.randn((1, 16, 1, 14, 24), generator=g)
    anchor = torch.randn((1, 16, 1, 8, 10), generator=g)
    config = runtime_module.ScaleLockConfig(
        lock_strength=1.0, lock_strength_start=1.0, lock_strength_end=1.0,
        cutoff=.33, mid_cutoff=.6, mid_strength=0.0, schedule="flat",
        schedule_power=2.0, schedule_hold=0.0, mid_strength_start=1.0,
        mid_strength_end=0.0, mid_schedule="linked", mid_schedule_power=2.0,
        mid_schedule_hold=0.0, structure_mode="patch_mean",
    )
    context = runtime_module.ScaleLockedRuntimeContext(
        model=model, highres_latent={"samples": torch.zeros_like(base)},
        lowres_latent={"samples": torch.zeros_like(anchor)},
        lowres_out={"samples": anchor}, sigmas=torch.tensor([1.0, 0.0]),
        anchors_x0=[anchor], planner_sigmas=[1.0], highres_noise=torch.zeros_like(base), noise_seed=0,
    )
    original = lambda x, timestep, **kwargs: base
    guider = SimpleNamespace(predict_noise=original)
    runtime_module.apply_scale_lock_to_guider(guider, context, config)
    assert guider._slrd_patch_projection is not None
    result = guider.predict_noise(base, torch.tensor([1.0]))
    assert torch.allclose(guider._slrd_patch_projection.reduce(result), anchor, atol=2e-6)
    runtime_module.restore_original_predict_noise(guider)
    assert guider.predict_noise is original

    generic = replace(config, structure_mode="multiband", lock_strength=.75, mid_strength=.2)
    assert runtime_module.ScaleLockConfig.__dataclass_fields__["structure_mode"].default == "multiband"
    runtime_module.apply_scale_lock_to_guider(guider, context, generic)
    assert guider._slrd_patch_projection is None
    expected = runtime_module.residual_lock_multiband(
        base, runtime_module.resize_latent_dict({"samples": anchor}, base.shape[-2:])["samples"],
        .75, .75 * .2, .33, .6,
    )
    assert torch.equal(guider.predict_noise(base, torch.tensor([1.0])), expected)
    runtime_module.restore_original_predict_noise(guider)
    assert guider.predict_noise is original
    try:
        runtime_module.apply_scale_lock_to_guider(guider, context, replace(config, spatial_mask=torch.ones(14, 24)))
    except ValueError:
        pass
    else:
        raise AssertionError("patch-mean mode must reject unsupported masking")
    assert guider.predict_noise is original
    print(json.dumps({"case": "generic_guider_isolation", "passed": True}))


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
    spec = importlib.util.spec_from_file_location("slrd_native_smoke", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    runtime_module = sys.modules[f"{spec.name}.slrd_runtime"]
    check_generic_isolation(runtime_module, model)
    node = package.NODE_CLASS_MAPPINGS["Krea2TurboFlowAlignedRegenerate"]()
    defaults = node.INPUT_TYPES()["required"]
    assert defaults["coarse_lock_strength"][1]["default"] == 1.0
    assert defaults["mid_lock_strength"][1]["default"] == 0.0
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
    runs = [(label, selected, steps, 1.0, 0.0) for label, selected in cases for steps in (8, 12)]
    runs += [("native", model, 8, coarse, mid) for coarse, mid in ((1.0, .25), (.5, 0.0), (0.0, 0.0))]
    for label, selected, steps, coarse, mid in runs:
        reports = []
        checks = []
        original_lock = runtime_module.apply_scale_lock_to_noise_prediction

        def checked_lock(guider, base_prediction, x, timestep):
            result = original_lock(guider, base_prediction, x, timestep)
            projection = guider._slrd_patch_projection
            assert projection is not None
            idx = guider._slrd_prev_match_idx
            strengths = runtime_module.scale_lock_strengths_for_step(guider, idx)
            assert strengths == (coarse, coarse * mid), strengths
            planner = guider._slrd_anchors_x0_cpu[idx].to(base_prediction)
            expected = torch.lerp(projection.reduce(base_prediction), planner, coarse)
            assert torch.allclose(projection.reduce(result), expected, atol=2e-5, rtol=2e-5)
            if coarse == 0.0 and mid == 0.0:
                assert result is base_prediction
            checks.append(idx)
            return result

        runtime_module.apply_scale_lock_to_noise_prediction = checked_lock
        token = reports_context.set(reports) if reports_context is not None else None
        try:
            with torch.inference_mode():
                # Small latent canvases keep genuine CPU model execution practical.
                result = node.sample(selected, positive, negative,
                                     {"samples": torch.zeros(1, 4, 16, 20)}, 42, .01, coarse, mid, steps=steps)
        finally:
            runtime_module.apply_scale_lock_to_noise_prediction = original_lock
            if reports_context is not None:
                reports_context.reset(token)
        assert checks == list(range(steps)), checks
        if label == "spectrum":
            assert len(reports) == 2
            assert all(r["outer_steps"] == steps for r in reports)
            assert all(r["forecast"] > 0 and r["fallback"] is None for r in reports)
        shapes = [list(x["samples"].shape) for x in result]
        assert shapes[0] == shapes[2] == [1, 16, 1, 16, 20]
        assert all(torch.isfinite(x["samples"]).all() for x in result)
        assert shapes[1][2] == 1
        assert not base.diffusion_model.last._forward_pre_hooks
        print(json.dumps({"case": label, "steps_per_pass": steps, "coarse": coarse, "mid": mid,
                          "constrained_steps": checks, "output_shapes": shapes, "reports": reports}))


if __name__ == "__main__":
    main()
