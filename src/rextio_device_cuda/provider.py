"""NVIDIA CUDA Device Provider API 1 implementation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from rextio.artifacts import (
    ArtifactKind,
    CertificationTier,
    RuntimeRequirement,
    TargetCapability,
)
from rextio.devices import (
    DEVICE_PROVIDER_API_VERSION,
    DeviceBuildContribution,
    DevicePreflightRequest,
    DevicePreflightResult,
    DevicePreflightStatus,
    DeviceProviderManifest,
    DeviceResourceAccess,
    DeviceResourceContract,
    DeviceResourceOwner,
    normalize_device_id,
)

from rextio_device_cuda.__about__ import __version__
from rextio_device_cuda.config import CudaProviderConfig
from rextio_device_cuda.probe import (
    CudaProbeError,
    FilesystemToolkitInspector,
    ProbeRunner,
    SubprocessProbeRunner,
    ToolkitInspector,
    expected_probe_target,
)

if TYPE_CHECKING:
    from rextio.artifacts import DeviceRequirement

PROVIDER_ID = "rextio-device-cuda"
CAPABILITY_LINUX_X86_64 = "cuda-linux-x86_64"
CAPABILITY_LINUX_AARCH64 = "cuda-linux-aarch64"
CAPABILITY_WINDOWS_X86_64 = "cuda-windows-x86_64"

_SM_PATTERN = re.compile(r"^sm_[0-9]{2,3}$")
_ARCHITECTURES = (
    "sm_60",
    "sm_61",
    "sm_70",
    "sm_72",
    "sm_75",
    "sm_80",
    "sm_86",
    "sm_87",
    "sm_89",
    "sm_90",
)
_CAPABILITY_TARGETS = {
    CAPABILITY_LINUX_X86_64: "x86_64-unknown-linux-gnu",
    CAPABILITY_LINUX_AARCH64: "aarch64-unknown-linux-gnu",
    CAPABILITY_WINDOWS_X86_64: "x86_64-pc-windows-msvc",
}
_ALLOWED_OPTION_KEYS = frozenset(
    {"probe_executable", "toolkit_root", "device_ordinal", "sm"}
)
_ARTIFACT_KINDS = (
    ArtifactKind.HOST_EXECUTABLE,
    ArtifactKind.HOST_EXTENSION,
    ArtifactKind.RUST_CRATE,
)
_EVIDENCE = (
    "docs/evidence/build-only.md",
    "docs/support-matrix.md",
)


def _capability(capability_id: str, target_triple: str) -> TargetCapability:
    return TargetCapability(
        id=capability_id,
        target_triples=(target_triple,),
        artifact_kinds=_ARTIFACT_KINDS,
        accelerator_backends=("cuda",),
        minimum_runtime_version="12.0",
        minimum_driver_version="12000",
        architectures=_ARCHITECTURES,
        certification_tier=CertificationTier.BUILD_ONLY,
        evidence_references=_EVIDENCE,
    )


def _request_fingerprint(request: DevicePreflightRequest) -> str:
    return json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"))


def _cuda_requirement(request: DevicePreflightRequest) -> tuple[DeviceRequirement, int, str]:
    candidates: list[tuple[DeviceRequirement, int, str]] = []
    for requirement in request.artifact_profile.device_requirements:
        try:
            device = normalize_device_id(
                requirement.logical_device,
                backend=requirement.backend,
            )
        except ValueError:
            continue
        if device.kind == "gpu" and device.backend == "cuda":
            if len(requirement.architectures) != 1:
                raise CudaProbeError("EXPLICIT_SM_REQUIRED")
            sm = requirement.architectures[0]
            if _SM_PATTERN.fullmatch(sm) is None:
                raise CudaProbeError("INVALID_SM")
            candidates.append((requirement, device.index, sm))
    if len(candidates) != 1:
        raise CudaProbeError("EXACTLY_ONE_CUDA_DEVICE_REQUIRED")
    requirement, ordinal, sm = candidates[0]
    if requirement.reuse_domain_runtime:
        raise CudaProbeError("FRAMEWORK_RUNTIME_REUSE_UNSUPPORTED")
    if requirement.runtime not in {None, "cuda", "cuda-driver"}:
        raise CudaProbeError("CUDA_RUNTIME_INCOMPATIBLE")
    if set(requirement.features) - {"driver-api"}:
        raise CudaProbeError("CUDA_FEATURE_UNSUPPORTED")
    if set(requirement.memory_spaces) - {"device"}:
        raise CudaProbeError("CUDA_MEMORY_SPACE_UNSUPPORTED")
    if requirement.layouts:
        raise CudaProbeError("CUDA_LAYOUT_UNSUPPORTED")
    return requirement, ordinal, sm


def _private_option(request: DevicePreflightRequest, key: str) -> str | None:
    """Read an additive API-1 options record without requiring it at import time."""
    options = getattr(request, "options", None)
    getter = getattr(options, "get", None)
    if getter is None:
        return None
    value = getter(key)
    return value if isinstance(value, str) else None


def _validate_private_option_keys(request: DevicePreflightRequest) -> None:
    """Reject unknown private inputs before any provider-owned observation."""
    options = getattr(request, "options", None)
    keys = getattr(options, "keys", ())
    if not isinstance(keys, tuple) or any(not isinstance(key, str) for key in keys):
        raise CudaProbeError("PROVIDER_OPTIONS_INVALID")
    if set(keys) - _ALLOWED_OPTION_KEYS:
        raise CudaProbeError("PROVIDER_OPTION_UNKNOWN")


def _coalesce_private_value(
    request_value: str | None,
    constructor_value: str | None,
    *,
    conflict_reason: str,
) -> str | None:
    if (
        request_value is not None
        and constructor_value is not None
        and request_value != constructor_value
    ):
        raise CudaProbeError(conflict_reason)
    return request_value if request_value is not None else constructor_value


def _driver_version_tuple(value: int) -> tuple[int, int]:
    return (value // 1_000, (value % 1_000) // 10)


class CudaDeviceProvider:
    """Bounded E1 NVIDIA CUDA provider; never a Python lowering plugin."""

    def __init__(
        self,
        config: CudaProviderConfig | None = None,
        *,
        probe_runner: ProbeRunner | None = None,
        toolkit_inspector: ToolkitInspector | None = None,
    ) -> None:
        """Create a provider with explicit or injectable inspection boundaries."""
        self._config = config or CudaProviderConfig()
        self._probe_runner = probe_runner
        self._toolkit_inspector = toolkit_inspector
        self._ready_requests: set[str] = set()
        self._ready_lock = Lock()

    def manifest(self) -> DeviceProviderManifest:
        """Declare build-only target compatibility without probing the host."""
        return DeviceProviderManifest(
            provider_id=PROVIDER_ID,
            display_name=f"Rextio NVIDIA CUDA provider {__version__}",
            provider_version=__version__,
            backend="cuda",
            api_version=DEVICE_PROVIDER_API_VERSION,
            capabilities=tuple(
                _capability(capability_id, target)
                for capability_id, target in _CAPABILITY_TARGETS.items()
            ),
            runtime_requirements=(
                RuntimeRequirement(name="nvidia-cuda-driver", version=">=12.0"),
            ),
        )

    def _runner(self, path: str | None) -> ProbeRunner:
        if self._probe_runner is not None:
            return self._probe_runner
        if path is None:
            raise CudaProbeError("PROBE_NOT_CONFIGURED")
        return SubprocessProbeRunner(Path(path))

    def _inspector(self, root: str | None) -> ToolkitInspector | None:
        if self._toolkit_inspector is not None:
            return self._toolkit_inspector
        if root is None:
            return None
        return FilesystemToolkitInspector(Path(root))

    def preflight(self, request: DevicePreflightRequest) -> DevicePreflightResult:
        """Verify exact target, device, SM, driver, runtime, and toolkit facts."""
        if request.selection.provider_id != PROVIDER_ID:
            return self._failure("PROVIDER_ID_MISMATCH", incompatible=True)
        expected_target = _CAPABILITY_TARGETS.get(request.selection.capability_id)
        if expected_target is None:
            return self._failure("CAPABILITY_UNKNOWN", incompatible=True)
        if request.artifact_profile.target_triple != expected_target:
            return self._failure("TARGET_MISMATCH", incompatible=True)
        try:
            _validate_private_option_keys(request)
            _, required_ordinal, required_sm = _cuda_requirement(request)
            probe_path = _coalesce_private_value(
                _private_option(request, "probe_executable"),
                str(self._config.probe_path) if self._config.probe_path is not None else None,
                conflict_reason="PROBE_CONFIGURATION_CONFLICT",
            )
            toolkit_root = _coalesce_private_value(
                _private_option(request, "toolkit_root"),
                str(self._config.toolkit_root)
                if self._config.toolkit_root is not None
                else None,
                conflict_reason="TOOLKIT_CONFIGURATION_CONFLICT",
            )
            ordinal_text = _coalesce_private_value(
                _private_option(request, "device_ordinal"),
                (
                    str(self._config.device_ordinal)
                    if self._config.device_ordinal is not None
                    else None
                ),
                conflict_reason="DEVICE_CONFIGURATION_CONFLICT",
            )
            selected_sm = _coalesce_private_value(
                _private_option(request, "sm"),
                self._config.sm,
                conflict_reason="SM_CONFIGURATION_CONFLICT",
            )
            if ordinal_text is None:
                raise CudaProbeError("DEVICE_ORDINAL_NOT_CONFIGURED")
            if not ordinal_text.isascii() or not ordinal_text.isdigit():
                raise CudaProbeError("DEVICE_ORDINAL_INVALID")
            ordinal = int(ordinal_text)
            if ordinal > 1_023:
                raise CudaProbeError("DEVICE_ORDINAL_INVALID")
            if selected_sm is None:
                raise CudaProbeError("SM_NOT_CONFIGURED")
            if _SM_PATTERN.fullmatch(selected_sm) is None:
                raise CudaProbeError("INVALID_SM")
            if ordinal != required_ordinal or selected_sm != required_sm:
                raise CudaProbeError("DEVICE_REQUIREMENT_MISMATCH")
            expected_probe = expected_probe_target(expected_target)
            report = self._runner(probe_path).run()
            if report.target != expected_probe:
                raise CudaProbeError("PROBE_TARGET_MISMATCH")
            if report.status != "probe-complete":
                raise CudaProbeError(report.reason_code or "PROBE_NOT_READY")
            if report.driver_version is None:
                raise CudaProbeError("DRIVER_VERSION_UNAVAILABLE")
            if report.driver_version < self._config.minimum_driver_version:
                raise CudaProbeError("DRIVER_VERSION_TOO_OLD")
            if ordinal >= len(report.devices):
                raise CudaProbeError("CUDA_DEVICE_NOT_FOUND")
            device = report.devices[ordinal]
            if device.ordinal != ordinal or device.sm != selected_sm:
                raise CudaProbeError("CUDA_SM_MISMATCH")
            inspector = self._inspector(toolkit_root)
            toolkit = inspector.inspect(expected_target) if inspector is not None else None
            if toolkit is not None:
                toolkit_version = toolkit.version_tuple
                if toolkit_version[:2] < self._config.minimum_toolkit_version:
                    raise CudaProbeError("TOOLKIT_VERSION_TOO_OLD")
                if _driver_version_tuple(report.driver_version) < toolkit_version[:2]:
                    raise CudaProbeError("DRIVER_TOOLKIT_INCOMPATIBLE")
        except CudaProbeError as exc:
            return self._failure(exc.reason_code)

        with self._ready_lock:
            self._ready_requests.add(_request_fingerprint(request))
        return DevicePreflightResult(
            provider_id=PROVIDER_ID,
            status=DevicePreflightStatus.READY,
            observations=(
                ("device.count", str(report.device_count)),
                ("driver.version", str(report.driver_version)),
                ("probe.schema", "1"),
                ("selected.device", str(ordinal)),
                ("selected.sm", selected_sm),
                ("target.arch", report.target.arch),
                ("target.os", report.target.os),
                (
                    "toolkit.runtime",
                    toolkit.runtime_version if toolkit is not None else "not-configured",
                ),
                (
                    "toolkit.version",
                    toolkit.version if toolkit is not None else "not-configured",
                ),
            ),
            support_claim=False,
        )

    def _failure(
        self,
        reason_code: str,
        *,
        incompatible: bool = False,
    ) -> DevicePreflightResult:
        return DevicePreflightResult(
            provider_id=PROVIDER_ID,
            status=(
                DevicePreflightStatus.INCOMPATIBLE
                if incompatible
                else DevicePreflightStatus.UNAVAILABLE
            ),
            reason_codes=(reason_code,),
            support_claim=False,
        )

    def build_contribution(
        self,
        request: DevicePreflightRequest,
    ) -> DeviceBuildContribution:
        """Return deterministic inputs only for an identical ready request."""
        with self._ready_lock:
            ready = _request_fingerprint(request) in self._ready_requests
        if not ready:
            raise RuntimeError("CUDA provider build contribution requires successful preflight")
        return DeviceBuildContribution(
            package_references=(
                "generated/device-providers/rextio-device-cuda/"
                "rextio-cuda-runtime/Cargo.toml",
            ),
            generated_helper_ids=("rextio_cuda_runtime_v1",),
            runtime_check_ids=("rextio_cuda_driver_inventory_v1",),
            resource_contracts=(
                DeviceResourceContract(
                    resource_kind="raw.cuda.context",
                    owner=DeviceResourceOwner.PROVIDER,
                    access=DeviceResourceAccess.OWNED,
                    may_allocate=True,
                ),
                DeviceResourceContract(
                    resource_kind="raw.cuda.dedicated-stream",
                    owner=DeviceResourceOwner.PROVIDER,
                    access=DeviceResourceAccess.OWNED,
                    may_allocate=True,
                    may_synchronize=True,
                ),
                DeviceResourceContract(
                    resource_kind="raw.cuda.device-memory",
                    owner=DeviceResourceOwner.PROVIDER,
                    access=DeviceResourceAccess.OWNED,
                    may_allocate=True,
                ),
                DeviceResourceContract(
                    resource_kind="framework.tensor",
                    owner=DeviceResourceOwner.FRAMEWORK,
                    access=DeviceResourceAccess.BORROW_VALIDATE,
                ),
                DeviceResourceContract(
                    resource_kind="framework.allocator",
                    owner=DeviceResourceOwner.FRAMEWORK,
                    access=DeviceResourceAccess.BORROW_VALIDATE,
                ),
                DeviceResourceContract(
                    resource_kind="framework.current-stream",
                    owner=DeviceResourceOwner.FRAMEWORK,
                    access=DeviceResourceAccess.BORROW_VALIDATE,
                ),
                DeviceResourceContract(
                    resource_kind="framework.event",
                    owner=DeviceResourceOwner.FRAMEWORK,
                    access=DeviceResourceAccess.BORROW_VALIDATE,
                ),
            ),
        )


def provider() -> CudaDeviceProvider:
    """Return the Device Provider API 1 entry-point object."""
    return CudaDeviceProvider()
