import sys
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from services import stt


def _patch_torch(monkeypatch, available):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: available)
    )

    def fake_lazy_imports():
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        stt.torch = fake_torch

    monkeypatch.setattr(stt, "_lazy_imports", fake_lazy_imports)


def test_pick_device_prefers_cuda(monkeypatch):
    _patch_torch(monkeypatch, available=True)
    device, compute = stt._pick_device_and_compute(None, None)
    assert device == "cuda"
    assert compute == "float16"


def test_pick_device_user_overrides(monkeypatch):
    _patch_torch(monkeypatch, available=True)
    device, compute = stt._pick_device_and_compute("cpu", None)
    assert device == "cpu"
    assert compute == "float32"

    device, compute = stt._pick_device_and_compute(None, "int8")
    assert device == "cuda"
    assert compute == "int8"


def test_pick_device_no_cuda(monkeypatch):
    _patch_torch(monkeypatch, available=False)
    device, compute = stt._pick_device_and_compute(None, None)
    assert device == "cpu"
    assert compute == "float32"
