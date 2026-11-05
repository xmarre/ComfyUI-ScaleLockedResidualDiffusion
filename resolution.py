"""Resolution Selector without the core 16-megapixel input cap.

Same aspect ratios, megapixel convention (1 MP = 1024 x 1024 pixels), rounding
and live preview as ComfyUI's built-in Resolution Selector. Only the megapixel
limit differs: the outputs may use the full ComfyUI canvas range.
"""
from __future__ import annotations

import math

from nodes import MAX_RESOLUTION

ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}
# Largest square canvas ComfyUI latents accept, in 1024 x 1024 pixel megapixels.
MAX_MEGAPIXELS = float((MAX_RESOLUTION // 1024) ** 2)

NODE_ID = "ScaleLockedResolutionSelector"
DISPLAY_NAME = "Resolution Selector (Uncapped)"
DESCRIPTION = (
    "Calculate width and height from an aspect ratio and a megapixel target, like the built-in "
    "Resolution Selector but without its 16-megapixel limit. 1 MP = 1024x1024 pixels."
)


def compute_resolution(aspect_ratio: str, megapixels: float, multiple: int) -> tuple[int, int]:
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"unknown aspect ratio {aspect_ratio!r}")
    megapixels = float(megapixels)
    multiple = int(multiple)
    if not math.isfinite(megapixels) or megapixels <= 0.0:
        raise ValueError("megapixels must be positive")
    if multiple < 1:
        raise ValueError("multiple must be positive")
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    scale = math.sqrt(megapixels * 1024 * 1024 / (w_ratio * h_ratio))
    width = max(multiple, round(w_ratio * scale / multiple) * multiple)
    height = max(multiple, round(h_ratio * scale / multiple) * multiple)
    if max(width, height) > MAX_RESOLUTION:
        raise ValueError(
            f"{aspect_ratio} at {megapixels:g} MP gives {width}x{height}, beyond ComfyUI's "
            f"{MAX_RESOLUTION}-pixel canvas limit per side"
        )
    return width, height


_TOOLTIPS = {
    "aspect_ratio": "The aspect ratio for the output dimensions.",
    "megapixels": "Target total megapixels. 1.0 MP = 1024x1024 for square. No 16 MP cap.",
    "multiple": "Nearest multiple of the result to set the selected resolution to.",
    "width": "Calculated width in pixels multiplied by the selected multiple.",
    "height": "Calculated height in pixels multiplied by the selected multiple.",
}


def _build_v3_node():
    from comfy_api.latest import io

    if not hasattr(io, "ResolutionPreview"):
        return None

    class ResolutionSelectorUncapped(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=NODE_ID,
                display_name=DISPLAY_NAME,
                category="utilities",
                description=DESCRIPTION,
                inputs=[
                    io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS),
                                   default="1:1 (Square)", tooltip=_TOOLTIPS["aspect_ratio"]),
                    io.Float.Input("megapixels", default=1.0, min=0.1, max=MAX_MEGAPIXELS, step=0.1,
                                   tooltip=_TOOLTIPS["megapixels"]),
                    io.ResolutionPreview.Input("preview", tooltip="Live preview of the calculated output resolution."),
                    io.Int.Input(id="multiple", default=8, min=8, max=128, step=4,
                                 tooltip=_TOOLTIPS["multiple"], advanced=True),
                ],
                outputs=[
                    io.Int.Output("width", tooltip=_TOOLTIPS["width"]),
                    io.Int.Output("height", tooltip=_TOOLTIPS["height"]),
                ],
            )

        @classmethod
        def execute(cls, aspect_ratio: str, megapixels: float, multiple: int, preview=None) -> io.NodeOutput:
            return io.NodeOutput(*compute_resolution(aspect_ratio, megapixels, multiple))

    return ResolutionSelectorUncapped


class ResolutionSelectorUncappedClassic:
    """Fallback for ComfyUI builds without the resolution preview widget."""

    CATEGORY = "utilities"
    DESCRIPTION = DESCRIPTION
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (list(ASPECT_RATIOS), {"default": "1:1 (Square)", "tooltip": _TOOLTIPS["aspect_ratio"]}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": MAX_MEGAPIXELS, "step": 0.1,
                                         "tooltip": _TOOLTIPS["megapixels"]}),
                "multiple": ("INT", {"default": 8, "min": 8, "max": 128, "step": 4, "tooltip": _TOOLTIPS["multiple"]}),
            }
        }

    def execute(self, aspect_ratio, megapixels, multiple):
        return compute_resolution(aspect_ratio, megapixels, multiple)


try:
    ResolutionSelectorUncapped = _build_v3_node() or ResolutionSelectorUncappedClassic
except ImportError:
    ResolutionSelectorUncapped = ResolutionSelectorUncappedClassic
