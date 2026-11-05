"""Compare the uncapped Resolution Selector with ComfyUI's built-in node.

Run with ComfyUI's Python dependencies installed:
    python tests/smoke_resolution_selector.py --comfy-root /path/to/ComfyUI
"""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.comfy_root.resolve()))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy_extras.nodes_resolution import ResolutionSelector, AspectRatio
    from comfy_api.internal import _ComfyNodeInternal

    spec = importlib.util.spec_from_file_location("slrd_resolution_smoke", root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    node = package.NODE_CLASS_MAPPINGS["ScaleLockedResolutionSelector"]
    assert issubclass(node, _ComfyNodeInternal), "expected the live-preview node on this ComfyUI"

    info = node.GET_NODE_INFO_V1()
    core_info = ResolutionSelector.GET_NODE_INFO_V1()
    assert info["input"]["required"]["megapixels"][1]["max"] == 256.0
    assert core_info["input"]["required"]["megapixels"][1]["max"] == 16.0
    # Same widgets, options and preview wiring as the core node, apart from the cap.
    assert list(info["input"]["required"]) == list(core_info["input"]["required"])
    assert info["input"]["required"]["aspect_ratio"][0] == core_info["input"]["required"]["aspect_ratio"][0]
    assert info["input"].get("optional", {}).keys() == core_info["input"].get("optional", {}).keys()
    for section in ("required", "optional"):
        for name, spec_ in core_info["input"].get(section, {}).items():
            if name != "megapixels":
                assert info["input"][section][name] == spec_, (name, info["input"][section][name], spec_)
    assert info["output"] == core_info["output"] and info["output_name"] == core_info["output_name"]

    compared = 0
    for ratio in AspectRatio:
        for megapixels in (0.1, 1.0, 2.5, 8.0, 16.0):
            for multiple in (8, 16, 64, 128):
                ours = node.EXECUTE_NORMALIZED(aspect_ratio=ratio.value, megapixels=megapixels, multiple=multiple).args
                core = ResolutionSelector.EXECUTE_NORMALIZED(aspect_ratio=ratio.value, megapixels=megapixels,
                                                            multiple=multiple).args
                assert tuple(ours) == tuple(core), (ratio, megapixels, multiple, ours, core)
                compared += 1
    above = node.EXECUTE_NORMALIZED(aspect_ratio="1:1 (Square)", megapixels=64.0, multiple=16).args
    assert tuple(above) == (8192, 8192), above
    try:
        node.EXECUTE_NORMALIZED(aspect_ratio="21:9 (Ultrawide)", megapixels=256.0, multiple=16)
    except ValueError as exc:
        assert "canvas limit" in str(exc)
    else:
        raise AssertionError("canvases beyond ComfyUI's per-side limit must be rejected")
    classic = sys.modules[spec.name + ".resolution"].ResolutionSelectorUncappedClassic
    assert classic.INPUT_TYPES()["required"]["megapixels"][1]["max"] == 256.0
    assert classic().execute("16:9 (Widescreen)", 2.5, 16) == tuple(
        ResolutionSelector.EXECUTE_NORMALIZED(aspect_ratio="16:9 (Widescreen)", megapixels=2.5, multiple=16).args)
    print(f"matched core Resolution Selector on {compared} cases; 64 MP square -> {above[0]}x{above[1]}")


if __name__ == "__main__":
    main()
