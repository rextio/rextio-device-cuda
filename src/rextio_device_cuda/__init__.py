"""First-party NVIDIA CUDA device provider for Rextio."""

from rextio_device_cuda.__about__ import __version__
from rextio_device_cuda.config import CudaProviderConfig
from rextio_device_cuda.provider import CudaDeviceProvider, provider

__all__ = [
    "CudaDeviceProvider",
    "CudaProviderConfig",
    "__version__",
    "provider",
]
