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
    for label, selected, steps in [(label,selected,steps) for label,selected in cases for steps in (8,12)]:
        reports = []
        token = reports_context.set(reports) if reports_context is not None else None
        with torch.inference_mode():
            # Small latent canvases keep genuine CPU model execution practical.
            result = node.sample(selected, positive, negative,
                                 {"samples": torch.zeros(1, 4, 16, 20)}, 42, .01, 0.0, 0.0, steps=steps)
        if reports_context is not None:
            reports_context.reset(token)
        if label == "spectrum":
            assert len(reports) == 2
            assert all(r["outer_steps"] == steps for r in reports)
            assert all(r["forecast"] > 0 and r["fallback"] is None for r in reports)
        shapes = [list(x["samples"].shape) for x in result]
        assert shapes[0] == shapes[2] == [1, 16, 1, 16, 20]
        assert all(torch.isfinite(x["samples"]).all() for x in result)
        assert shapes[1][2] == 1
        assert not base.diffusion_model.last._forward_pre_hooks
        print(json.dumps({"case": label, "steps_per_pass": steps, "output_shapes": shapes, "reports": reports}))


if __name__ == "__main__":
    main()
