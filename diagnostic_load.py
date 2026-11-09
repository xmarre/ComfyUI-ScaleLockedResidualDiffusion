"""Read planner, first-stage handoff, and actual final Krea 2 latent from output.

All latents are saved from one diagnostic generation in ComfyUI output-space
format. The completed-stage and final-output files are native sampler
outputs, not intermediate x0 preview estimates. All decodes must use the
same VAE as the generation. This node performs no denoiser calls.
"""

from __future__ import annotations

from pathlib import Path
import re

import torch


_DIAGNOSTIC_FILES = (
    # Keep the first three outputs in place for imported workflow compatibility.
    "planner_final.latent",
    "stage0_source_resized.latent",
    "stage0_step000_x0.latent",
    "stage0_completed.latent",
    "final_output.latent",
)
_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\Z")


def _resolve_diagnostic_run(run_folder: str) -> tuple[Path, tuple[Path, ...]]:
    import folder_paths

    root = Path(folder_paths.get_output_directory()) / "scale_locked_diagnostics"
    selected = run_folder.strip()
    if not selected:
        raise ValueError("run_folder must be 'latest' or a diagnostic run directory name")
    if selected == "latest":
        if not root.is_dir():
            raise FileNotFoundError(
                "No Scale-Locked diagnostic output exists. Run the Krea 2 sampler "
                "with handoff_diagnostics=log_and_save_latents first."
            )
        complete = [
            directory for directory in root.iterdir()
            if directory.is_dir() and not directory.is_symlink()
            and all((directory / name).is_file() and not (directory / name).is_symlink()
                    for name in _DIAGNOSTIC_FILES)
        ]
        if not complete:
            raise FileNotFoundError(
                "No complete diagnostic run has planner, clean stage source, "
                "first x0, completed first stage, and actual final output. "
                "Run Krea 2 sampling with handoff_diagnostics=log_and_save_latents."
            )
        directory = max(complete, key=lambda d: ((d / _DIAGNOSTIC_FILES[0]).stat().st_mtime_ns, d.name))
    else:
        if not _RUN_NAME_PATTERN.fullmatch(selected) or selected in (".", ".."):
            raise ValueError("run_folder must be one directory name under scale_locked_diagnostics")
        directory = root / selected
        if not directory.is_dir() or directory.is_symlink():
            raise FileNotFoundError(f"Scale-Locked diagnostic run not found: {selected}")
    paths = tuple(directory / name for name in _DIAGNOSTIC_FILES)
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing diagnostic latent: {path.name} (run: {directory.name})")
    return directory, paths


class SLRDLoadKrea2HandoffDiagnostics:
    """Directly load comparable handoff and native final latents from one run."""

    CATEGORY = "sampling/scale_locked/diagnostics"
    FUNCTION = "load"
    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT", "LATENT")
    RETURN_NAMES = (
        "planner_final", "stage0_source_resized", "stage0_step000_x0",
        "stage0_completed", "final_output",
    )
    DESCRIPTION = (
        "Load the planner, resized source, first x0, completed first stage, "
        "and ACTUAL final sampler latent from a single Krea 2 diagnostic run. "
        "All outputs are raw output-space latents; use the VAE actually used "
        "for the original generation, before downstream image postprocessing."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_folder": ("STRING", {
            "default": "latest",
            "multiline": False,
            "tooltip": "'latest' selects the newest diagnostic run with all five required latents. "
                       "Or enter the run folder name to select an older one. "
                       "Run the Krea 2 sampler with handoff_diagnostics=log_and_save_latents "
                       "to capture the completed first stage and actual final output.",
        })}}

    @classmethod
    def IS_CHANGED(cls, run_folder: str):
        directory, paths = _resolve_diagnostic_run(run_folder)
        return (directory.name, tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths))

    @classmethod
    def VALIDATE_INPUTS(cls, run_folder: str):
        try:
            _resolve_diagnostic_run(run_folder)
        except (OSError, ValueError) as exc:
            return str(exc)
        return True

    def load(self, run_folder: str):
        import logging
        import safetensors.torch

        directory, paths = _resolve_diagnostic_run(run_folder)
        out = []
        for path in paths:
            payload = safetensors.torch.load_file(str(path), device="cpu")
            if "latent_format_version_0" not in payload or "latent_tensor" not in payload:
                raise ValueError(f"Unrecognized ComfyUI diagnostic latent format: {path.name}")
            samples = payload["latent_tensor"]
            if samples.ndim not in (4, 5) or (samples.ndim == 5 and samples.shape[2] != 1):
                raise ValueError(f"Unsupported diagnostic latent layout: {path.name} {tuple(samples.shape)}")
            if not torch.isfinite(samples).all():
                raise ValueError(f"Non-finite values in diagnostic latent: {path.name}")
            out.append({"samples": samples.float()})
        logging.getLogger(__name__).info(
            "Scale-Locked Krea 2 diagnostic loader: run=%s files=%s",
            directory.name, ", ".join(p.name for p in paths),
        )
        return tuple(out)
