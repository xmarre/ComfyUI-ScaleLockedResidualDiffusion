"""Vendored SesquiLSR latent upscaler architecture (MIT, see LICENSE here).

Source: https://github.com/LoganBooker/SesquiLSR at commit
821e6db7412089f2a5c0a106c42d3c1a396b9515. ``upscaler.py`` and
``upscaler_Wan21.safetensors`` are unmodified copies of that commit's
``sesqui_lsr/upscaler.py`` and ``models/upscaler_Wan21.safetensors``.
"""

from .upscaler import LatentUpscaler

__all__ = ["LatentUpscaler"]
