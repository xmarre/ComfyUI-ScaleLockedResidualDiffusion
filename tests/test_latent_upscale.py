"""SesquiLSR Wan21 handoff backend: real pinned weights, exact sizes, layouts.

Runs the vendored upstream architecture with the vendored pretrained weights
on CPU. It checks execution and integration contracts only; decoded image
quality is measured separately with a real Wan VAE.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "slrd_latent_upscale_test"
UPSTREAM_UPSCALER_SHA256 = "f10d975148cbd6678fe33b0ef6dc5c48bde4797d682b45bcb63dbe940e1cf281"


def _load():
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE] = package
    for name in ("slrd_core", "latent_upscale"):
        spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{PACKAGE}.latent_upscale"], sys.modules[f"{PACKAGE}.slrd_core"]


lu, core = _load()


def _latent(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Raw Wan VAE latents: roughly the Wan21 latent-format channel statistics.
    mean = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                         0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921])
    std = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160])
    view = (1, 16) + (1,) * (len(shape) - 2)
    return torch.randn(*shape, generator=g) * std.view(view) + mean.view(view)


def test_vendored_files_are_the_pinned_upstream_files():
    third = ROOT / "third_party" / "sesqui_lsr"
    assert hashlib.sha256((third / "upscaler.py").read_bytes()).hexdigest() == UPSTREAM_UPSCALER_SHA256
    weights = (third / lu.SESQUI_WAN21_FILENAME).read_bytes()
    assert hashlib.sha256(weights).hexdigest() == lu.SESQUI_WAN21_SHA256
    assert "MIT License" in (third / "LICENSE").read_text()


def test_real_weights_load_into_the_upstream_architecture():
    lu.ensure_latent_upscaler("sesqui_wan21")
    model = lu._model_for(torch.device("cpu"))
    assert sum(t.numel() for t in model.state_dict().values()) == 3091382  # 3091378 parameters + 4 buffer values
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert lu._model_for(torch.device("cpu")) is model  # one cached instance per device


def test_matches_direct_upstream_inference_and_is_deterministic():
    from safetensors.torch import load_file
    upscaler_mod = sys.modules.get(f"{PACKAGE}.third_party.sesqui_lsr.upscaler")
    if upscaler_mod is None:
        lu._model_for(torch.device("cpu"))
        upscaler_mod = sys.modules[f"{PACKAGE}.third_party.sesqui_lsr.upscaler"]
    reference = upscaler_mod.LatentUpscaler(in_channels=16)
    reference.load_state_dict(load_file(str(lu._WEIGHTS_PATH)))
    reference.eval()
    x = _latent(1, 16, 24, 20, seed=1)
    with torch.no_grad():
        expected = reference(x, (40, 34))
    got = lu.enlarge_completed_latent(x, (40, 34), "sesqui_wan21")
    assert torch.equal(got, expected)
    assert torch.equal(got, lu.enlarge_completed_latent(x, (40, 34), "sesqui_wan21"))
    bicubic = F.interpolate(x, size=(40, 34), mode="bicubic", align_corners=False)
    assert not torch.allclose(got, bicubic, atol=1e-3)


@pytest.mark.parametrize("src, dst", [((24, 20), (36, 30)), ((24, 20), (48, 40)),
                                      ((24, 20), (30, 40)), ((24, 20), (24, 26)),
                                      ((22, 30), (56, 74))])
@pytest.mark.parametrize("temporal", [False, True])
def test_exact_target_sizes_layouts_and_dtype(src, dst, temporal):
    shape = (2, 16, 1) + src if temporal else (2, 16) + src
    x = _latent(*shape, seed=2)
    out = lu.enlarge_completed_latent(x, dst, "sesqui_wan21")
    assert out.dtype == torch.float32 and out.device == x.device
    assert tuple(out.shape) == shape[:-2] + dst
    assert torch.isfinite(out).all()
    # Batch items are processed independently.
    single = lu.enlarge_completed_latent(x[1:], dst, "sesqui_wan21")
    torch.testing.assert_close(out[1:], single, rtol=1e-5, atol=1e-5)


def test_interpolate_is_the_original_resize():
    x = _latent(1, 16, 1, 24, 20, seed=3)
    assert torch.equal(lu.enlarge_completed_latent(x, (36, 30), "interpolate"),
                       core.resize_4d_tensor(x, (36, 30)))


def test_same_size_is_identity_and_never_runs_the_model():
    x = _latent(1, 16, 24, 20, seed=4)
    assert torch.equal(lu.enlarge_completed_latent(x, (24, 20), "sesqui_wan21"), x)


@pytest.mark.parametrize("src, dst, passes", [
    ((24, 20), (48, 40), 1), ((24, 20), (36, 30), 1), ((24, 20), (49, 40), 2),
    ((176, 176), (444, 444), 2), ((24, 20), (200, 20), 4),
])
def test_hops_stay_within_trained_scale_and_end_exactly(src, dst, passes):
    hops = lu.sesqui_hops(src, dst)
    assert len(hops) == passes and hops[-1] == dst
    previous = src
    for hop in hops:
        assert previous[0] <= hop[0] <= 2 * previous[0]
        assert previous[1] <= hop[1] <= 2 * previous[1]
        previous = hop


@pytest.mark.parametrize("bad", [
    lambda: lu.sesqui_hops((24, 20), (20, 20)),
    lambda: lu.enlarge_completed_latent(torch.randn(1, 4, 8, 8), (12, 12), "sesqui_wan21"),
    lambda: lu.enlarge_completed_latent(_latent(1, 16, 2, 8, 8), (12, 12), "sesqui_wan21"),
    lambda: lu.enlarge_completed_latent(_latent(1, 16, 8, 8), (12, 12), "lanczos"),
    lambda: lu.ensure_latent_upscaler("bicubic"),
])
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        bad()


def test_missing_or_corrupted_weights_fail_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(lu, "_state_dict", None)
    with pytest.raises(FileNotFoundError, match="weights are missing"):
        lu._verified_state_dict(tmp_path / "absent.safetensors")
    corrupt = tmp_path / lu.SESQUI_WAN21_FILENAME
    corrupt.write_bytes(lu._WEIGHTS_PATH.read_bytes()[:-1] + b"\0")
    with pytest.raises(ValueError, match="failed verification"):
        lu._verified_state_dict(corrupt)
    assert lu._state_dict is None


def test_cached_state_does_not_skip_weight_integrity_preflight(tmp_path, monkeypatch):
    """Replacing or removing weights after the first run must fail before sampling."""
    monkeypatch.setattr(lu, "_state_dict", {"cached": torch.tensor(1)})
    missing = tmp_path / "missing.safetensors"
    with pytest.raises(FileNotFoundError, match="weights are missing"):
        lu._verified_state_dict(missing)
    damaged = tmp_path / "damaged.safetensors"
    damaged.write_bytes(b"corrupt after a successful previous run")
    with pytest.raises(ValueError, match="failed verification"):
        lu._verified_state_dict(damaged)


def test_concurrent_first_use_constructs_one_model(monkeypatch):
    """Parallel workflows must not allocate duplicate GPU models on cache miss."""
    import concurrent.futures
    import threading
    import time

    from importlib import import_module
    model_module = import_module(f"{PACKAGE}.third_party.sesqui_lsr")
    constructions = []
    class SlowModel:
        def __init__(self, *, in_channels):
            constructions.append(in_channels)
            time.sleep(0.03)  # reliably exposes build-outside-lock races
        def load_state_dict(self, state):
            return self
        def to(self, **kwargs):
            return self
        def eval(self):
            return self
        def requires_grad_(self, enabled):
            return self

    monkeypatch.setattr(lu, "_models", {})
    monkeypatch.setattr(lu, "_verified_state_dict", lambda: {})
    monkeypatch.setattr(model_module, "LatentUpscaler", SlowModel)

    start = threading.Barrier(8)
    def load():
        start.wait()
        return lu._model_for(torch.device("cpu"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: load(), range(8)))
    assert constructions == [lu.SESQUI_WAN21_CHANNELS]
    assert all(model is results[0] for model in results)


@pytest.mark.parametrize("failures, expect_cpu", [(1, False), (2, True)])
def test_gpu_oom_frees_comfy_memory_then_retries_or_falls_back_to_cpu(monkeypatch, failures, expect_cpu):
    # A large handoff can exhaust GPU memory while the DiT is still loaded.
    # The backend frees ComfyUI-managed models and retries once, then runs on
    # the CPU instead of aborting the generation.
    devices, freed = [], []
    oom = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)

    def fake_run(x, hops, device):
        devices.append(device.type)
        if device.type != "cpu" and devices.count(device.type) <= failures:
            raise oom("simulated")
        return torch.zeros((x.shape[0], x.shape[1]) + tuple(hops[-1]), device=device)

    monkeypatch.setattr(lu, "_run_hops", fake_run)
    monkeypatch.setattr(lu, "_free_device_memory", lambda device: freed.append(device.type))
    out = lu.enlarge_completed_latent(_latent(1, 16, 8, 8), (12, 12), "sesqui_wan21",
                                      device=torch.device("meta"))
    assert out.shape == (1, 16, 12, 12) and out.device.type == "meta"
    assert freed == ["meta"] * failures
    assert devices == (["meta", "meta", "cpu"] if expect_cpu else ["meta", "meta"])



@pytest.mark.parametrize("failures", [1, 2])
def test_oom_traceback_releases_failed_hop_tensors_before_recovery(monkeypatch, failures):
    """Retry and CPU fallback must not hold tensors from failed GPU frames."""
    import gc
    import weakref

    oom = lu._OOM_ERRORS[0]
    failed_allocations = []
    devices = []

    def run_hops(x, hops, device):
        devices.append(device.type)
        # A failed CUDA model forward can leave intermediate tensors reachable
        # from its traceback frame until that exception handler has exited.
        if device.type != "cpu" and devices.count(device.type) <= failures:
            intermediate = torch.ones(8)
            failed_allocations.append(weakref.ref(intermediate))
            raise oom("simulated OOM with live intermediate")
        gc.collect()
        assert all(ref() is None for ref in failed_allocations), (
            "failed GPU forward tensors still retained by an OOM traceback"
        )
        return torch.zeros((x.shape[0], x.shape[1]) + tuple(hops[-1]), device=device)

    monkeypatch.setattr(lu, "_run_hops", run_hops)
    monkeypatch.setattr(lu, "_free_device_memory", lambda device: None)
    result = lu.enlarge_completed_latent(_latent(1, 16, 8, 8), (12, 12),
                                         "sesqui_wan21", device=torch.device("meta"))
    assert result.shape == (1, 16, 12, 12)
    assert devices == (["meta", "meta", "cpu"] if failures == 2 else ["meta", "meta"])



def test_cpu_oom_is_not_retried(monkeypatch):
    oom = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)

    def fake_run(x, hops, device):
        raise oom("simulated")

    monkeypatch.setattr(lu, "_run_hops", fake_run)
    monkeypatch.setattr(lu, "_free_device_memory", lambda device: pytest.fail("no retry on CPU"))
    with pytest.raises(oom):
        lu.enlarge_completed_latent(_latent(1, 16, 8, 8), (12, 12), "sesqui_wan21",
                                    device=torch.device("cpu"))
