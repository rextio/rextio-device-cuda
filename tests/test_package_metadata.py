from __future__ import annotations

from importlib.metadata import EntryPoint

from rextio.devices import DEVICE_PROVIDER_API_VERSION
from rextio_device_cuda import __version__
from rextio_device_cuda.provider import PROVIDER_ID, provider


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
