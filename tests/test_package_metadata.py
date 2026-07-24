from __future__ import annotations

from importlib.metadata import EntryPoint
from pathlib import Path

from rextio.devices import DEVICE_PROVIDER_API_VERSION
from rextio_device_cuda import __version__
from rextio_device_cuda.provider import (
    CAPABILITY_TENSORFLOW_TFE_LINUX_X86_64,
    PROVIDER_ID,
    provider,
)

_ROOT = Path(__file__).resolve().parents[1]


def test_package_and_provider_identity() -> None:
    assert __version__ == "0.1.0"
    instance = provider()
    assert instance.manifest().provider_id == PROVIDER_ID
    assert instance.manifest().api_version == DEVICE_PROVIDER_API_VERSION


def test_entry_point_factory_shape() -> None:
    entry = EntryPoint(
        name="rextio-device-cuda",
        value="rextio_device_cuda.provider:provider",
        group="rextio.device_providers",
    )

    loaded = entry.load()

    assert loaded().__class__.__name__ == "CudaDeviceProvider"


def test_tensorflow_tfe_capability_is_documented_without_support_claim() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    matrix = (_ROOT / "docs/support-matrix.md").read_text(encoding="utf-8")
    evidence = (_ROOT / "docs/evidence/build-only.md").read_text(encoding="utf-8")

    assert CAPABILITY_TENSORFLOW_TFE_LINUX_X86_64 in readme
    assert "TensorFlow 2.21.0" in readme
    assert "support_claim: false" in readme
    assert "TensorFlow TFE 2.21.0" in matrix
    assert "not certified" in matrix
    assert "TensorFlow/TFE" in evidence
    assert "certification_ready=false" in evidence


def test_probe_uses_only_the_shared_driver_loader() -> None:
    probe_source = (
        _ROOT / "crates/rextio-cuda-driver-probe/src/main.rs"
    ).read_text(encoding="utf-8")

    assert "use rextio_cuda_driver_loader::DriverLibrary;" in probe_source
    assert "#[cfg(any())]" not in probe_source
    assert "LoadLibraryExW(" not in probe_source
    assert "dlopen(" not in probe_source


def test_manual_scripts_force_the_documented_rust_toolchain() -> None:
    linux = (_ROOT / "scripts/validate-linux-nvidia.sh").read_text(encoding="utf-8")
    windows = (_ROOT / "scripts/validate-windows-nvidia.ps1").read_text(
        encoding="utf-8"
    )

    assert 'RUST_TOOLCHAIN="1.93.1"' in linux
    assert 'rustc +"${RUST_TOOLCHAIN}" --version' in linux
    assert 'cargo +"${RUST_TOOLCHAIN}" build' in linux
    assert '$RustToolchain = "1.93.1"' in windows
    assert 'rustc "+$RustToolchain" --version' in windows
    assert 'cargo "+$RustToolchain" build' in windows


def test_runtime_loader_lifetime_and_manifest_eof_contracts() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    manifest = (_ROOT / "MANIFEST.in").read_bytes()

    assert "shared driver loader" in readme
    assert "image handle outlives the runtime API" in readme
    assert "The runtime resolves the reviewed CUDA Driver image dynamically" not in readme
    assert manifest.endswith(b"\n")
    assert not manifest.endswith(b"\n\n")


def test_framework_reuse_capability_does_not_import_torch() -> None:
    provider_source = (
        _ROOT / "src/rextio_device_cuda/provider.py"
    ).read_text(encoding="utf-8")

    assert "import torch" not in provider_source
    assert "from torch" not in provider_source
