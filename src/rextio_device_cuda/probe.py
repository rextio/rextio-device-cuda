"""Bounded CUDA inventory and explicitly rooted toolkit inspection."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping, Protocol, runtime_checkable

from rextio_device_cuda.config import CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR

_PROBE_NAME = "rextio-cuda-driver-probe"
_PROBE_SCHEMA = "1"
_MAX_REPORT_BYTES = 65_536
_MAX_VERSION_FILE_BYTES = 65_536
_PROCESS_CLEANUP_SECONDS = 5.0
_REASON_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_SM_PATTERN = re.compile(r"^sm_([0-9]{2,3})$")
_VERSION_PATTERN = re.compile(r"^([0-9]{1,2})\.([0-9]{1,2})(?:\.([0-9]{1,3}))?$")
_DEVICE_NAME_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]{1,160}$")
_ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "probe",
        "support_claim",
        "target",
        "platform_supported",
        "status",
        "reason_code",
        "driver_loaded",
        "driver_version",
        "device_count",
        "cuda_result",
        "devices",
    }
)
_ALLOWED_TARGET_KEYS = frozenset({"os", "arch", "environment"})
_ALLOWED_DEVICE_KEYS = frozenset(
    {"ordinal", "name", "compute_major", "compute_minor", "sm"}
)
_TARGET_FACTS = {
    "x86_64-unknown-linux-gnu": ("linux", "x86_64", "gnu"),
    "aarch64-unknown-linux-gnu": ("linux", "aarch64", "gnu"),
    "x86_64-pc-windows-msvc": ("windows", "x86_64", "msvc"),
}


class CudaProbeError(RuntimeError):
    """A stable fail-closed probe or toolkit inspection failure."""

    def __init__(self, reason_code: str) -> None:
        """Create an error containing only a path-free stable reason."""
        if _REASON_PATTERN.fullmatch(reason_code) is None:
            raise ValueError("probe reason code must be a bounded uppercase identifier")
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class ProbeTarget:
    """Target identity emitted by the reviewed Rust probe."""

    os: str
    arch: str
    environment: str

    def __post_init__(self) -> None:
        """Reject unbounded or path-like target identity strings."""
        for value in (self.os, self.arch, self.environment):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 32
                or not value.isascii()
                or not all(character.isalnum() or character in "_-" for character in value)
            ):
                raise CudaProbeError("PROBE_SCHEMA_INVALID")


@dataclass(frozen=True)
class CudaDeviceRecord:
    """One bounded CUDA device inventory row."""

    ordinal: int
    name: str
    compute_major: int
    compute_minor: int
    sm: str

    def __post_init__(self) -> None:
        """Enforce the same invariants for injected and parsed reports."""
        if type(self.ordinal) is not int or not 0 <= self.ordinal <= 1_023:
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        if (
            not isinstance(self.name, str)
            or _DEVICE_NAME_PATTERN.fullmatch(self.name) is None
            or PurePosixPath(self.name).is_absolute()
            or PureWindowsPath(self.name).is_absolute()
        ):
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        if (
            type(self.compute_major) is not int
            or not 1 <= self.compute_major <= 99
            or type(self.compute_minor) is not int
            or not 0 <= self.compute_minor <= 99
            or self.sm != f"sm_{self.compute_major}{self.compute_minor}"
        ):
            raise CudaProbeError("PROBE_SCHEMA_INVALID")


@dataclass(frozen=True)
class CudaProbeReport:
    """Validated, path-free output from ``rextio-cuda-driver-probe``."""

    target: ProbeTarget
    platform_supported: bool
    status: str
    reason_code: str | None
    driver_loaded: bool
    driver_version: int | None
    device_count: int | None
    cuda_result: int | None
    devices: tuple[CudaDeviceRecord, ...]

    def __post_init__(self) -> None:
        """Keep direct runner injection as strict as parsed subprocess output."""
        if not isinstance(self.target, ProbeTarget):
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        if type(self.platform_supported) is not bool or type(self.driver_loaded) is not bool:
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        if self.status not in {"probe-complete", "unavailable", "unsupported", "error"}:
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        if self.reason_code is not None and _REASON_PATTERN.fullmatch(self.reason_code) is None:
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        if self.driver_version is not None and (
            type(self.driver_version) is not int
            or not 1_000 <= self.driver_version <= 99_999
        ):
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        if self.device_count is not None and (
            type(self.device_count) is not int or not 0 <= self.device_count <= 1_024
        ):
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        if self.cuda_result is not None and (
            type(self.cuda_result) is not int
            or not -2_147_483_648 <= self.cuda_result <= 2_147_483_647
        ):
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        if not isinstance(self.devices, tuple) or any(
            not isinstance(device, CudaDeviceRecord) for device in self.devices
        ):
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        if any(device.ordinal != index for index, device in enumerate(self.devices)):
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        if self.status == "probe-complete":
            if (
                self.reason_code is not None
                or not self.platform_supported
                or not self.driver_loaded
                or self.driver_version is None
                or self.device_count != len(self.devices)
                or self.device_count == 0
                or self.cuda_result != 0
            ):
                raise CudaProbeError("PROBE_SCHEMA_INVALID")
        elif self.reason_code is None or self.devices or self.device_count not in {None, 0}:
            raise CudaProbeError("PROBE_SCHEMA_INVALID")


@dataclass(frozen=True)
class CudaToolkitReport:
    """Bounded facts from one explicitly configured CUDA toolkit root."""

    version: str
    runtime_version: str
    components: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate public construction without relying on parser assertions."""
        if (
            not isinstance(self.version, str)
            or _VERSION_PATTERN.fullmatch(self.version) is None
            or not isinstance(self.runtime_version, str)
            or _VERSION_PATTERN.fullmatch(self.runtime_version) is None
        ):
            raise CudaProbeError("TOOLKIT_VERSION_INVALID")
        if self.version_tuple != self.runtime_version_tuple:
            raise CudaProbeError("TOOLKIT_RUNTIME_VERSION_MISMATCH")
        if self.version_tuple[:2] < CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR:
            raise CudaProbeError("TOOLKIT_VERSION_TOO_OLD")
        if (
            not isinstance(self.components, tuple)
            or not self.components
            or any(
                not isinstance(component, str)
                or not component
                or len(component) > 64
                or not component.isascii()
                or not all(
                    character.isalnum() or character in "._-"
                    for character in component
                )
                for component in self.components
            )
            or self.components != tuple(sorted(set(self.components)))
        ):
            raise CudaProbeError("TOOLKIT_COMPONENT_INVALID")

    @property
    def version_tuple(self) -> tuple[int, int, int]:
        """Return a comparable semantic version triple."""
        return self._parse_version(self.version)

    @property
    def runtime_version_tuple(self) -> tuple[int, int, int]:
        """Return the runtime version as a comparable semantic triple."""
        return self._parse_version(self.runtime_version)

    @staticmethod
    def _parse_version(value: str) -> tuple[int, int, int]:
        match = _VERSION_PATTERN.fullmatch(value)
        if match is None:
            raise CudaProbeError("TOOLKIT_VERSION_INVALID")
        return (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3) or "0"),
        )


@runtime_checkable
class ProbeRunner(Protocol):
    """Injectable driver-inventory boundary used by production and tests."""

    def run(self) -> CudaProbeReport:
        """Return one validated report or raise ``CudaProbeError``."""
        ...


@runtime_checkable
class ToolkitInspector(Protocol):
    """Injectable explicitly rooted toolkit boundary."""

    def inspect(self, target_triple: str) -> CudaToolkitReport:
        """Return bounded toolkit facts or raise ``CudaProbeError``."""
        ...


def _exact_mapping(
    value: object,
    *,
    keys: frozenset[str],
    reason_code: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CudaProbeError(reason_code)
    if not all(isinstance(key, str) for key in value):
        raise CudaProbeError(reason_code)
    return value


def _bounded_int(
    value: object,
    *,
    minimum: int,
    maximum: int,
    optional: bool = False,
) -> int | None:
    if value is None and optional:
        return None
    if type(value) is not int or not minimum <= value <= maximum:
        raise CudaProbeError("PROBE_SCHEMA_INVALID")
    return value


def parse_probe_report(payload: bytes | str) -> CudaProbeReport:
    """Validate the exact path-free probe schema and all cross-field invariants."""
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_REPORT_BYTES:
        raise CudaProbeError("PROBE_OUTPUT_INVALID")
    try:
        decoded = raw.decode("utf-8")
        document = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CudaProbeError("PROBE_OUTPUT_INVALID") from None
    root = _exact_mapping(
        document,
        keys=_ALLOWED_KEYS,
        reason_code="PROBE_SCHEMA_INVALID",
    )
    if (
        root["schema_version"] != _PROBE_SCHEMA
        or root["probe"] != _PROBE_NAME
        or root["support_claim"] is not False
    ):
        raise CudaProbeError("PROBE_SCHEMA_INVALID")

    target_raw = _exact_mapping(
        root["target"],
        keys=_ALLOWED_TARGET_KEYS,
        reason_code="PROBE_SCHEMA_INVALID",
    )
    target_parts: list[str] = []
    for key in ("os", "arch", "environment"):
        value = target_raw[key]
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 32
            or not value.isascii()
            or not all(character.isalnum() or character in "_-" for character in value)
        ):
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        target_parts.append(value)
    target = ProbeTarget(*target_parts)

    if type(root["platform_supported"]) is not bool or type(root["driver_loaded"]) is not bool:
        raise CudaProbeError("PROBE_SCHEMA_INVALID")
    status = root["status"]
    if status not in {"probe-complete", "unavailable", "unsupported", "error"}:
        raise CudaProbeError("PROBE_SCHEMA_INVALID")
    reason_code = root["reason_code"]
    if reason_code is not None and (
        not isinstance(reason_code, str) or _REASON_PATTERN.fullmatch(reason_code) is None
    ):
        raise CudaProbeError("PROBE_SCHEMA_INVALID")

    driver_version = _bounded_int(
        root["driver_version"], minimum=1_000, maximum=99_999, optional=True
    )
    device_count = _bounded_int(
        root["device_count"], minimum=0, maximum=1_024, optional=True
    )
    cuda_result = _bounded_int(
        root["cuda_result"], minimum=-2_147_483_648, maximum=2_147_483_647, optional=True
    )
    devices_raw = root["devices"]
    if not isinstance(devices_raw, list) or len(devices_raw) > 1_024:
        raise CudaProbeError("PROBE_SCHEMA_INVALID")
    devices: list[CudaDeviceRecord] = []
    for expected_ordinal, item in enumerate(devices_raw):
        row = _exact_mapping(
            item,
            keys=_ALLOWED_DEVICE_KEYS,
            reason_code="PROBE_SCHEMA_INVALID",
        )
        ordinal = _bounded_int(row["ordinal"], minimum=0, maximum=1_023)
        major = _bounded_int(row["compute_major"], minimum=1, maximum=99)
        minor = _bounded_int(row["compute_minor"], minimum=0, maximum=99)
        if ordinal is None or major is None or minor is None:
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        name = row["name"]
        sm = row["sm"]
        if (
            ordinal != expected_ordinal
            or not isinstance(name, str)
            or _DEVICE_NAME_PATTERN.fullmatch(name) is None
            or PurePosixPath(name).is_absolute()
            or PureWindowsPath(name).is_absolute()
            or not isinstance(sm, str)
            or _SM_PATTERN.fullmatch(sm) is None
            or sm != f"sm_{major}{minor}"
        ):
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        devices.append(
            CudaDeviceRecord(
                ordinal=ordinal,
                name=name,
                compute_major=major,
                compute_minor=minor,
                sm=sm,
            )
        )

    platform_supported = root["platform_supported"]
    driver_loaded = root["driver_loaded"]
    if status == "probe-complete":
        if (
            reason_code is not None
            or platform_supported is not True
            or driver_loaded is not True
            or driver_version is None
            or device_count is None
            or device_count < 1
            or cuda_result != 0
            or device_count != len(devices)
        ):
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
    else:
        if reason_code is None or devices:
            raise CudaProbeError("PROBE_SCHEMA_INVALID")
        if device_count not in {None, 0}:
            raise CudaProbeError("PROBE_SCHEMA_INVALID")

    return CudaProbeReport(
        target=target,
        platform_supported=platform_supported,
        status=status,
        reason_code=reason_code,
        driver_loaded=driver_loaded,
        driver_version=driver_version,
        device_count=device_count,
        cuda_result=cuda_result,
        devices=tuple(devices),
    )


class SubprocessProbeRunner:
    """Run one explicitly configured probe executable without ``PATH`` lookup."""

    def __init__(
        self,
        executable: Path,
        *,
        arguments: tuple[str, ...] = (),
        timeout_seconds: float = 10.0,
    ) -> None:
        """Record an absolute executable path; filesystem checks occur at run time."""
        if (
            not isinstance(arguments, tuple)
            or any(
                not isinstance(argument, str)
                or not argument
                or len(argument) > 4_096
                or "\x00" in argument
                for argument in arguments
            )
        ):
            raise ValueError("probe arguments must be bounded non-empty strings")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not 0 < timeout_seconds <= 60
        ):
            raise ValueError("probe timeout must be in (0, 60] seconds")
        self._executable = executable
        self._arguments = arguments
        self._timeout_seconds = float(timeout_seconds)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> bool:
        """Best-effort kill and reap without introducing an unbounded wait."""
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=_PROCESS_CLEANUP_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=_PROCESS_CLEANUP_SECONDS)
            except subprocess.TimeoutExpired:
                return False
        return process.poll() is not None

    @classmethod
    def _terminate_and_join(
        cls,
        process: subprocess.Popen[bytes],
        reader: threading.Thread,
    ) -> bool:
        """Best-effort kill/reap/join without introducing an unbounded wait."""
        if not cls._terminate_process(process):
            return False
        reader.join(timeout=_PROCESS_CLEANUP_SECONDS)
        return not reader.is_alive()

    def run(self) -> CudaProbeReport:
        """Run the probe with a bounded output/time budget and parse its report."""
        if not self._executable.is_absolute():
            raise CudaProbeError("PROBE_PATH_NOT_ABSOLUTE")
        try:
            executable = self._executable.resolve(strict=True)
        except OSError:
            raise CudaProbeError("PROBE_NOT_FOUND") from None
        if not executable.is_file() or (os.name != "nt" and not os.access(executable, os.X_OK)):
            raise CudaProbeError("PROBE_NOT_EXECUTABLE")
        environment: dict[str, str] = {}
        if os.name == "nt":
            for key in ("SYSTEMROOT", "WINDIR"):
                value = os.environ.get(key)
                if value:
                    environment[key] = value
        try:
            process = subprocess.Popen(
                [str(executable), *self._arguments],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
            )
        except OSError:
            raise CudaProbeError("PROBE_EXECUTION_FAILED") from None
        stdout = process.stdout
        if stdout is None:
            try:
                process.kill()
                process.wait(timeout=_PROCESS_CLEANUP_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise CudaProbeError("PROBE_EXECUTION_FAILED")

        output = bytearray()
        overflow = threading.Event()
        reader_failed = threading.Event()

        def _read_bounded_stdout() -> None:
            try:
                while True:
                    remaining = _MAX_REPORT_BYTES + 1 - len(output)
                    if remaining <= 0:
                        overflow.set()
                        return
                    chunk = stdout.read(min(8_192, remaining))
                    if not chunk:
                        return
                    output.extend(chunk)
                    if len(output) > _MAX_REPORT_BYTES:
                        overflow.set()
                        return
            except (OSError, ValueError):
                reader_failed.set()

        reader = threading.Thread(
            target=_read_bounded_stdout,
            name="rextio-cuda-probe-stdout",
            # The reviewed probe never spawns descendants. Daemonizing still
            # prevents a hostile descendant retaining stdout from turning a
            # bounded probe failure into an interpreter-wide deadlock.
            daemon=True,
        )
        try:
            reader.start()
        except RuntimeError:
            self._terminate_process(process)
            try:
                stdout.close()
            except (OSError, ValueError):
                pass
            raise CudaProbeError("PROBE_EXECUTION_FAILED") from None

        def _cleanup() -> None:
            if self._terminate_and_join(process, reader):
                stdout.close()

        deadline = time.monotonic() + self._timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _cleanup()
                raise CudaProbeError("PROBE_TIMEOUT")
            if overflow.wait(timeout=min(0.01, remaining)):
                _cleanup()
                raise CudaProbeError("PROBE_OUTPUT_TOO_LARGE")
            if reader_failed.is_set():
                _cleanup()
                raise CudaProbeError("PROBE_EXECUTION_FAILED")
            return_code = process.poll()
            if return_code is not None:
                reader.join(timeout=max(0.0, deadline - time.monotonic()))
                if reader.is_alive():
                    _cleanup()
                    raise CudaProbeError("PROBE_TIMEOUT")
                break
        stdout.close()
        if overflow.is_set():
            raise CudaProbeError("PROBE_OUTPUT_TOO_LARGE")
        if reader_failed.is_set():
            raise CudaProbeError("PROBE_EXECUTION_FAILED")
        if return_code != 0:
            raise CudaProbeError("PROBE_EXIT_NONZERO")
        return parse_probe_report(bytes(output))


def expected_probe_target(target_triple: str) -> ProbeTarget:
    """Return the exact probe identity for one supported Rust target triple."""
    expected = _TARGET_FACTS.get(target_triple)
    if expected is None:
        raise CudaProbeError("UNSUPPORTED_TARGET")
    return ProbeTarget(*expected)


def _read_version(root: Path) -> str:
    version_json = root / "version.json"
    try:
        if version_json.stat().st_size > _MAX_VERSION_FILE_BYTES:
            raise CudaProbeError("TOOLKIT_VERSION_INVALID")
        document = json.loads(version_json.read_text(encoding="utf-8"))
        version = document["cuda"]["version"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        raise CudaProbeError("TOOLKIT_VERSION_UNAVAILABLE") from None
    if not isinstance(version, str) or _VERSION_PATTERN.fullmatch(version) is None:
        raise CudaProbeError("TOOLKIT_VERSION_INVALID")
    return version


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class FilesystemToolkitInspector:
    """Inspect a user-selected toolkit root without PATH or registry discovery."""

    def __init__(self, root: Path) -> None:
        """Record the explicit root; no search is performed."""
        self._root = root

    def inspect(self, target_triple: str) -> CudaToolkitReport:
        """Require the bounded header/runtime/link surface for the target."""
        if not self._root.is_absolute():
            raise CudaProbeError("TOOLKIT_PATH_NOT_ABSOLUTE")
        try:
            root = self._root.resolve(strict=True)
        except OSError:
            raise CudaProbeError("TOOLKIT_NOT_FOUND") from None
        if not root.is_dir():
            raise CudaProbeError("TOOLKIT_NOT_FOUND")
        version = _read_version(root)
        header = root / "include" / "cuda.h"
        candidates: dict[str, tuple[Path, ...]]
        if target_triple == "x86_64-pc-windows-msvc":
            runtime_dlls = tuple(sorted((root / "bin").glob("cudart64_*.dll")))
            candidates = {
                "cuda-header": (header,),
                "cuda-runtime": runtime_dlls,
                "cuda-import-library": (root / "lib" / "x64" / "cudart.lib",),
            }
        elif target_triple == "x86_64-unknown-linux-gnu":
            candidates = {
                "cuda-header": (header,),
                "cuda-runtime": (
                    root / "lib64" / "libcudart.so",
                    root / "targets" / "x86_64-linux" / "lib" / "libcudart.so",
                ),
            }
        elif target_triple == "aarch64-unknown-linux-gnu":
            candidates = {
                "cuda-header": (header,),
                "cuda-runtime": (
                    root / "lib64" / "libcudart.so",
                    root / "targets" / "aarch64-linux" / "lib" / "libcudart.so",
                ),
            }
        else:
            raise CudaProbeError("UNSUPPORTED_TARGET")

        components: list[str] = []
        for component, paths in candidates.items():
            accepted = False
            for path in paths:
                try:
                    resolved = path.resolve(strict=True)
                except OSError:
                    continue
                if resolved.is_file() and _inside_root(resolved, root):
                    accepted = True
                    break
            if not accepted:
                raise CudaProbeError("TOOLKIT_COMPONENT_MISSING")
            components.append(component)
        return CudaToolkitReport(
            version=version,
            runtime_version=version,
            components=tuple(sorted(components)),
        )
