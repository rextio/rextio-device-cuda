"""Verify that wheel/sdist contain the independently audited Rust sources."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

_WHEEL_SUFFIXES = {
    "rextio_device_cuda/rust/rextio-cuda-runtime/Cargo.toml",
    "rextio_device_cuda/rust/rextio-cuda-runtime/src/lib.rs",
}
_SDIST_SUFFIXES = {
    "crates/rextio-cuda-driver-loader/src/lib.rs",
    "crates/rextio-cuda-driver-probe/Cargo.toml",
    "crates/rextio-cuda-driver-probe/src/main.rs",
    "crates/rextio-cuda-runtime-smoke/src/main.rs",
    "src/rextio_device_cuda/rust/rextio-cuda-runtime/Cargo.toml",
    "src/rextio_device_cuda/rust/rextio-cuda-runtime/src/lib.rs",
    "scripts/validate-linux-nvidia.sh",
    "scripts/validate-windows-nvidia.ps1",
}


def _contains_suffix(names: set[str], suffix: str) -> bool:
    return any(name == suffix or name.endswith(f"/{suffix}") for name in names)


def main() -> int:
    """Check exact required source classes and reject tool-state leakage."""
    wheel = Path(sys.argv[1])
    sdist = Path(sys.argv[2])
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_names = {member.name for member in archive.getmembers()}
    missing_wheel = sorted(
        suffix for suffix in _WHEEL_SUFFIXES if not _contains_suffix(wheel_names, suffix)
    )
    missing_sdist = sorted(
        suffix for suffix in _SDIST_SUFFIXES if not _contains_suffix(sdist_names, suffix)
    )
    leaked = sorted(
        name
        for name in wheel_names | sdist_names
        if "/.git/" in f"/{name}/"
        or "/.claude" in f"/{name}"
        or "/.codex" in f"/{name}"
        or "__pycache__" in name
    )
    if missing_wheel or missing_sdist or leaked:
        raise SystemExit(
            f"distribution contract failed: wheel={missing_wheel}, "
            f"sdist={missing_sdist}, leaked={leaked}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
