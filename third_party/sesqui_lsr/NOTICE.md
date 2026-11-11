# SesquiLSR (vendored)

- Upstream: https://github.com/LoganBooker/SesquiLSR
- Commit: `821e6db7412089f2a5c0a106c42d3c1a396b9515`
- License: MIT, copyright (c) 2026 SesquiLSR Contributors (`LICENSE` in this
  directory, copied verbatim).
- `upscaler.py`: verbatim copy of upstream `sesqui_lsr/upscaler.py`.
- `upscaler_Wan21.safetensors`: verbatim copy of upstream
  `models/upscaler_Wan21.safetensors`, SHA-256
  `d1b4c2c1d80b2af81e6bbfd38501ffb5adffb9f7c0ed2599555b837d0d2d015a`.
  Upstream distributes its weights inside the same MIT-licensed repository and
  does not publish a separate model license.

The Scale-Locked integration (`latent_upscale.py`) verifies the weight file's
SHA-256 before loading it and never downloads code or weights at runtime.
