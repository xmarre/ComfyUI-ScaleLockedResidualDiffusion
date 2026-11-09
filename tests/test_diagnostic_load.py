"""Directly loading Scale-Locked Krea 2 diagnostics from output is safe and cache-aware."""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import safetensors.torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("slrd_diag_loader", ROOT / "diagnostic_load.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Loader = module.SLRDLoadKrea2HandoffDiagnostics


def write_run(root, name, value, all_three=True):
    folder = root / "scale_locked_diagnostics" / name
    folder.mkdir(parents=True)
    filenames = module._DIAGNOSTIC_FILES if all_three else module._DIAGNOSTIC_FILES[:2]
    for i, filename in enumerate(filenames):
        samples = torch.full((1, 16, 1, 8 + i, 8 + i), value + i, dtype=torch.float32)
        safetensors.torch.save_file(
            {"latent_tensor": samples, "latent_format_version_0": torch.empty(0)},
            str(folder / filename),
        )
    return folder


def test_latest_complete_and_explicit_run(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(get_output_directory=lambda: str(tmp_path)),
    )
    # Names sort opposite to age: "latest" must follow planner modification
    # time, not lexical order.
    earlier = write_run(tmp_path, "z_earlier", 1)
    newer = write_run(tmp_path, "a_latest_good", 10)
    incomplete = write_run(tmp_path, "incomplete", 100, all_three=False)
    planner = module._DIAGNOSTIC_FILES[0]
    os.utime(earlier / planner, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer / planner, ns=(2_000_000_000, 2_000_000_000))
    os.utime(incomplete / planner, ns=(3_000_000_000, 3_000_000_000))
    assert Loader.VALIDATE_INPUTS("latest") is True

    values = Loader().load("latest")
    assert [int(x["samples"].flatten()[0]) for x in values] == [10, 11, 12, 13, 14]
    assert [int(x["samples"].flatten()[0]) for x in Loader().load("z_earlier")] == [1, 2, 3, 4, 5]
    signature = Loader.IS_CHANGED("latest")
    assert signature[0] == "a_latest_good"
    assert len(signature[1]) == 5
    assert all(x["samples"].shape[2] == 1 for x in values)


def test_selected_run_signature_tracks_every_file(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(get_output_directory=lambda: str(tmp_path)),
    )
    folder = write_run(tmp_path, "selected", 1)
    for index, filename in enumerate(module._DIAGNOSTIC_FILES):
        before = Loader.IS_CHANGED("selected")
        samples = torch.full((1, 16, 1, 9 + index, 9 + index), 50.0)
        safetensors.torch.save_file(
            {"latent_tensor": samples, "latent_format_version_0": torch.empty(0)},
            str(folder / filename),
        )
        stamp = 5_000_000_000 + index
        os.utime(folder / filename, ns=(stamp, stamp))
        after = Loader.IS_CHANGED("selected")
        assert after != before, filename
        assert after[1][index] != before[1][index]


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", ".", "..", "foo/bar", "foo\\bar"])
def test_path_escape_rejected(tmp_path, monkeypatch, name):
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(get_output_directory=lambda: str(tmp_path)),
    )
    with pytest.raises(ValueError):
        Loader().load(name)


def test_missing_diagnostics_explains_sampling_requirement(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(get_output_directory=lambda: str(tmp_path)),
    )
    with pytest.raises(FileNotFoundError, match="handoff_diagnostics=log_and_save_latents"):
        Loader().load("latest")


def test_comfyui_latent_format_validation(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(get_output_directory=lambda: str(tmp_path)),
    )
    run = write_run(tmp_path, "bad_format", 1)
    safetensors.torch.save_file(
        {"latent_tensor": torch.zeros(1, 16, 1, 8, 8)},
        str(run / "planner_final.latent"),
    )
    with pytest.raises(ValueError, match="Unrecognized ComfyUI diagnostic latent format"):
        Loader().load("bad_format")
