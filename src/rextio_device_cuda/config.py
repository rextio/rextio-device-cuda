"""Explicit constructor configuration for the CUDA provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CudaProviderConfig:
    """Private inputs that are never inferred from ambient process state."""

    probe_path: Path | None = None
    toolkit_root: Path | None = None
    device_ordinal: int | None = None
    sm: str | None = None
    minimum_driver_version: int = 12_000
    minimum_toolkit_version: tuple[int, int] = (12, 0)

    def __post_init__(self) -> None:
        """Validate bounded configuration without touching the filesystem."""
        for field_name in ("probe_path", "toolkit_root"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"{field_name} must be pathlib.Path or None")
        if self.device_ordinal is not None and (
            type(self.device_ordinal) is not int or not 0 <= self.device_ordinal <= 1_023
        ):
            raise ValueError("device_ordinal must be an integer in [0, 1023]")
        if self.sm is not None and (
            not isinstance(self.sm, str)
            or len(self.sm) not in {5, 6}
            or not self.sm.startswith("sm_")
            or not self.sm[3:].isdigit()
        ):
            raise ValueError("sm must use the sm_NN or sm_NNN spelling")
        if type(self.minimum_driver_version) is not int or not (
            1_000 <= self.minimum_driver_version <= 99_999
        ):
            raise ValueError("minimum_driver_version must be a bounded CUDA driver integer")
        if (
            not isinstance(self.minimum_toolkit_version, tuple)
            or len(self.minimum_toolkit_version) != 2
            or any(type(part) is not int or part < 0 or part > 99 for part in self.minimum_toolkit_version)
        ):
            raise ValueError("minimum_toolkit_version must be a (major, minor) integer tuple")
