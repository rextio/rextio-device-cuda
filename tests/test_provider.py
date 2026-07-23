from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock

import pytest

from rextio.artifacts import (
    ArtifactKind,
    ArtifactProfile,
    DeviceRequirement,
    RuntimeRequirement,
)
from rextio.devices import (
    DevicePreflightRequest,
    DevicePreflightStatus,
    DeviceProviderOptions,
    DeviceProviderSelection,
    DeviceResourceAccess,
    DeviceResourceOwner,
    resolve_device_plan,
)
from rextio_device_cuda.config import (
    CUDA_DRIVER_VERSION_FLOOR,
    CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR,
    CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR_TEXT,
    CudaProviderConfig,
)
from rextio_device_cuda.probe import (
    CudaDeviceRecord,
    CudaProbeError,
    CudaProbeReport,
    CudaToolkitReport,
    ProbeTarget,
)
from rextio_device_cuda.provider import (
    CAPABILITY_LIBTORCH_LINUX_X86_64,
    CAPABILITY_LINUX_X86_64,
    LIBTORCH_RUNTIME,
    LIBTORCH_VERSION,
    PROVIDER_ID,
    TCH_VERSION,
    CudaDeviceProvider,
)


@dataclass
class FixedRunner:
    report: CudaProbeReport
    calls: int = 0

    def run(self) -> CudaProbeReport:
        self.calls += 1
        return self.report


@dataclass
class FixedToolkit:
    report: CudaToolkitReport
    calls: int = 0

    def inspect(self, target_triple: str) -> CudaToolkitReport:
        assert target_triple == "x86_64-unknown-linux-gnu"
        self.calls += 1
        return self.report


class OverlappingRunner:
    """Deterministically overlap two probes for one request fingerprint."""

    def __init__(self) -> None:
        self.first_started = Event()
        self.second_started = Event()
        self.release_first = Event()
        self.release_second = Event()
        self._calls = 0
        self._lock = Lock()

    def run(self) -> CudaProbeReport:
        with self._lock:
            self._calls += 1
            call = self._calls
        if call == 1:
            self.first_started.set()
            if not self.release_first.wait(timeout=5):
                raise AssertionError("first overlapping probe was not released")
            return probe_report()
        if call == 2:
            self.second_started.set()
            if not self.release_second.wait(timeout=5):
                raise AssertionError("second overlapping probe was not released")
            return probe_report(sm="sm_90")
        raise AssertionError("unexpected overlapping probe call")


def probe_report(
    *,
    sm: str = "sm_80",
    driver_version: int = 12_080,
) -> CudaProbeReport:
    major = int(sm[3:-1])
    minor = int(sm[-1])
    return CudaProbeReport(
        target=ProbeTarget(os="linux", arch="x86_64", environment="gnu"),
        platform_supported=True,
        status="probe-complete",
        reason_code=None,
        driver_loaded=True,
        driver_version=driver_version,
        device_count=1,
        cuda_result=0,
        devices=(
            CudaDeviceRecord(
                ordinal=0,
                name="NVIDIA Test GPU",
                compute_major=major,
                compute_minor=minor,
                sm=sm,
            ),
        ),
    )


def profile(*, sm: str = "sm_80") -> ArtifactProfile:
    return ArtifactProfile(
        kind=ArtifactKind.RUST_CRATE,
        target_triple="x86_64-unknown-linux-gnu",
        packaging_backend="cargo",
        device_requirements=(
            DeviceRequirement(
                logical_device="cuda:0",
                backend="cuda",
                runtime="cuda-driver",
                features=("driver-api",),
                memory_spaces=("device",),
                architectures=(sm,),
            ),
        ),
    )


def request(*, sm: str = "sm_80") -> DevicePreflightRequest:
    return DevicePreflightRequest(
        artifact_profile=profile(sm=sm),
        selection=DeviceProviderSelection(
            provider_id=PROVIDER_ID,
            capability_id=CAPABILITY_LINUX_X86_64,
        ),
    )


def libtorch_profile(
    *,
    architectures: tuple[str, ...] = (),
    logical_device: str = "gpu:0",
    backend: str = "cuda",
    runtime: str = LIBTORCH_RUNTIME,
    features: tuple[str, ...] = ("inference", "no-grad"),
    layouts: tuple[str, ...] = ("strided",),
    memory_spaces: tuple[str, ...] = ("device",),
    reuse_domain_runtime: bool = True,
    runtime_requirements: tuple[RuntimeRequirement, ...] | None = None,
) -> ArtifactProfile:
    return ArtifactProfile(
        kind=ArtifactKind.HOST_EXTENSION,
        target_triple="x86_64-unknown-linux-gnu",
        packaging_backend="cargo",
        python_fallback_backend="cpython",
        runtime_requirements=(
            runtime_requirements
            if runtime_requirements is not None
            else (
                RuntimeRequirement(
                    "libtorch",
                    LIBTORCH_VERSION,
                    ("cuda", "pytorch-wheel"),
                ),
                RuntimeRequirement("tch", TCH_VERSION, ("cuda",)),
            )
        ),
        device_requirements=(
            DeviceRequirement(
                logical_device=logical_device,
                backend=backend,
                runtime=runtime,
                features=features,
                layouts=layouts,
                memory_spaces=memory_spaces,
                architectures=architectures,
                reuse_domain_runtime=reuse_domain_runtime,
            ),
        ),
    )


def libtorch_request(
    *,
    artifact_profile: ArtifactProfile | None = None,
) -> DevicePreflightRequest:
    return DevicePreflightRequest(
        artifact_profile=artifact_profile or libtorch_profile(),
        selection=DeviceProviderSelection(
            provider_id=PROVIDER_ID,
            capability_id=CAPABILITY_LIBTORCH_LINUX_X86_64,
        ),
    )


def configured_provider(*, report_sm: str = "sm_80") -> CudaDeviceProvider:
    return CudaDeviceProvider(
        CudaProviderConfig(device_ordinal=0, sm="sm_80"),
        probe_runner=FixedRunner(probe_report(sm=report_sm)),
        toolkit_inspector=FixedToolkit(
            CudaToolkitReport(
                version="12.8.0",
                runtime_version="12.8.0",
                components=("cuda-header", "cuda-runtime"),
            )
        ),
    )


def test_manifest_is_build_only_and_target_bounded() -> None:
    provider = CudaDeviceProvider()
    manifest = provider.manifest()

    assert manifest.provider_id == PROVIDER_ID
    assert manifest.backend == "cuda"
    assert {item.id for item in manifest.capabilities} == {
        "cuda-libtorch-linux-x86_64",
        "cuda-linux-aarch64",
        "cuda-linux-x86_64",
        "cuda-windows-x86_64",
    }
    assert {item.certification_tier.value for item in manifest.capabilities} == {"build-only"}
    assert {item.minimum_driver_version for item in manifest.capabilities} == {
        str(CUDA_DRIVER_VERSION_FLOOR)
    }
    assert {item.minimum_runtime_version for item in manifest.capabilities} == {
        CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR_TEXT,
        LIBTORCH_VERSION,
    }
    libtorch_capability = next(
        item for item in manifest.capabilities if item.id == CAPABILITY_LIBTORCH_LINUX_X86_64
    )
    assert libtorch_capability.artifact_kinds == (ArtifactKind.HOST_EXTENSION,)
    assert tuple(
        requirement.to_dict() for requirement in libtorch_capability.device_requirements
    ) == (
        {
            "logical_device": "gpu:0",
            "backend": "cuda",
            "runtime": "libtorch",
            "features": ["inference", "no-grad"],
            "layouts": ["strided"],
            "memory_spaces": ["device"],
            "architectures": [],
            "reuse_domain_runtime": True,
        },
    )
    assert CudaProviderConfig().minimum_driver_version == CUDA_DRIVER_VERSION_FLOOR
    assert CudaProviderConfig().minimum_toolkit_version == CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR
    assert not hasattr(provider, "claim")
    assert not hasattr(provider, "lower")


def test_config_accepts_exact_manifest_driver_floor() -> None:
    config = CudaProviderConfig(minimum_driver_version=CUDA_DRIVER_VERSION_FLOOR)

    assert config.minimum_driver_version == 12_000


@pytest.mark.parametrize("minimum_driver_version", [11_999, 11_000])
def test_config_rejects_driver_floor_below_manifest(
    minimum_driver_version: int,
) -> None:
    with pytest.raises(ValueError, match="at or above 12000"):
        CudaProviderConfig(minimum_driver_version=minimum_driver_version)


def test_preflight_enforces_the_same_driver_floor_as_manifest() -> None:
    config = CudaProviderConfig(
        device_ordinal=0,
        sm="sm_80",
        minimum_driver_version=CUDA_DRIVER_VERSION_FLOOR,
    )
    exact_floor = CudaDeviceProvider(
        config,
        probe_runner=FixedRunner(probe_report(driver_version=CUDA_DRIVER_VERSION_FLOOR)),
    )
    below_floor = CudaDeviceProvider(
        config,
        probe_runner=FixedRunner(probe_report(driver_version=CUDA_DRIVER_VERSION_FLOOR - 1)),
    )

    assert exact_floor.preflight(request()).status is DevicePreflightStatus.READY
    rejected = below_floor.preflight(request())
    assert rejected.status is DevicePreflightStatus.UNAVAILABLE
    assert rejected.reason_codes == ("DRIVER_VERSION_TOO_OLD",)


def test_config_accepts_exact_manifest_toolkit_runtime_floor() -> None:
    config = CudaProviderConfig(minimum_toolkit_version=CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR)

    assert config.minimum_toolkit_version == (12, 0)


@pytest.mark.parametrize("minimum_toolkit_version", [(11, 8), (1, 0)])
def test_config_rejects_toolkit_floor_below_manifest(
    minimum_toolkit_version: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="at or above 12.0"):
        CudaProviderConfig(minimum_toolkit_version=minimum_toolkit_version)


def test_toolkit_report_accepts_exact_manifest_runtime_floor() -> None:
    report = CudaToolkitReport(
        version="12.0.0",
        runtime_version="12.0.0",
        components=("cuda-header", "cuda-runtime"),
    )

    assert report.version_tuple[:2] == CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR
    assert report.runtime_version_tuple == report.version_tuple


@pytest.mark.parametrize(
    ("version", "runtime_version", "reason_code"),
    [
        ("12.8.0", "1.0.0", "TOOLKIT_RUNTIME_VERSION_MISMATCH"),
        ("12.8.0", "12.7.0", "TOOLKIT_RUNTIME_VERSION_MISMATCH"),
        ("11.8.0", "11.8.0", "TOOLKIT_VERSION_TOO_OLD"),
    ],
)
def test_toolkit_report_rejects_mismatch_and_below_floor_versions(
    version: str,
    runtime_version: str,
    reason_code: str,
) -> None:
    with pytest.raises(CudaProbeError, match=reason_code):
        CudaToolkitReport(
            version=version,
            runtime_version=runtime_version,
            components=("cuda-header", "cuda-runtime"),
        )


def test_preflight_accepts_exact_manifest_toolkit_runtime_floor() -> None:
    provider = CudaDeviceProvider(
        CudaProviderConfig(
            device_ordinal=0,
            sm="sm_80",
            minimum_driver_version=CUDA_DRIVER_VERSION_FLOOR,
            minimum_toolkit_version=CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR,
        ),
        probe_runner=FixedRunner(probe_report(driver_version=CUDA_DRIVER_VERSION_FLOOR)),
        toolkit_inspector=FixedToolkit(
            CudaToolkitReport(
                version="12.0.0",
                runtime_version="12.0.0",
                components=("cuda-header", "cuda-runtime"),
            )
        ),
    )

    assert provider.preflight(request()).status is DevicePreflightStatus.READY


def test_resolve_device_plan_records_path_free_lock_and_resource_boundaries() -> None:
    provider = configured_provider()
    plan = resolve_device_plan(
        artifact_profile=profile(),
        selection=DeviceProviderSelection(
            provider_id=PROVIDER_ID,
            capability_id=CAPABILITY_LINUX_X86_64,
        ),
        providers={PROVIDER_ID: provider},
    )

    assert plan is not None
    assert plan.preflight.status is DevicePreflightStatus.READY
    assert plan.preflight.support_claim is False
    assert (
        "policy.minimum-driver-version",
        str(CUDA_DRIVER_VERSION_FLOOR),
    ) in plan.preflight.observations
    assert (
        "policy.minimum-toolkit-version",
        CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR_TEXT,
    ) in plan.preflight.observations
    assert plan.contribution.native_libraries == ()
    assert plan.contribution.cargo_features == ()
    assert plan.contribution.package_references == (
        "generated/device-providers/rextio-device-cuda/rextio-cuda-runtime/Cargo.toml",
    )
    assert plan.contribution.generated_helper_ids == ("rextio_cuda_runtime_v1",)
    assert plan.contribution.runtime_check_ids == ("rextio_cuda_driver_inventory_v1",)
    serialized = plan.to_dict()
    assert serialized["report"]["certification_tier"] == "build-only"
    assert serialized["report"]["support_claim"] is False
    assert "/private" not in str(serialized)
    assert "C:\\" not in str(serialized)
    resources = {item.resource_kind: item for item in plan.contribution.resource_contracts}
    assert resources["raw.cuda.context"].owner is DeviceResourceOwner.PROVIDER
    assert resources["raw.cuda.device-memory"].access is DeviceResourceAccess.OWNED
    assert resources["framework.tensor"].owner is DeviceResourceOwner.FRAMEWORK
    assert resources["framework.current-stream"].may_synchronize is False


def test_missing_explicit_probe_fails_closed() -> None:
    result = CudaDeviceProvider(CudaProviderConfig(device_ordinal=0, sm="sm_80")).preflight(
        request()
    )

    assert result.status is DevicePreflightStatus.UNAVAILABLE
    assert result.reason_codes == ("PROBE_NOT_CONFIGURED",)
    assert result.support_claim is False


def test_unknown_private_option_fails_before_probe_without_echoing_value() -> None:
    runner = FixedRunner(probe_report())
    provider = CudaDeviceProvider(
        CudaProviderConfig(device_ordinal=0, sm="sm_80"),
        probe_runner=runner,
    )
    req = DevicePreflightRequest(
        artifact_profile=profile(),
        selection=DeviceProviderSelection(
            provider_id=PROVIDER_ID,
            capability_id=CAPABILITY_LINUX_X86_64,
        ),
        options=DeviceProviderOptions(values=(("unexpected_option", "/private/machine/secret"),)),
    )

    result = provider.preflight(req)

    assert result.status is DevicePreflightStatus.UNAVAILABLE
    assert result.reason_codes == ("PROVIDER_OPTION_UNKNOWN",)
    assert result.observations == ()
    assert "/private/machine/secret" not in str(result.to_dict())
    assert runner.calls == 0


def test_sm_mismatch_fails_before_build_contribution() -> None:
    provider = configured_provider(report_sm="sm_90")
    req = request()

    result = provider.preflight(req)

    assert result.status is DevicePreflightStatus.UNAVAILABLE
    assert result.reason_codes == ("CUDA_SM_MISMATCH",)
    with pytest.raises(RuntimeError, match="requires successful preflight"):
        provider.build_contribution(req)


def test_direct_preflight_rejects_architecture_outside_manifest() -> None:
    provider = CudaDeviceProvider(
        CudaProviderConfig(device_ordinal=0, sm="sm_99"),
        probe_runner=FixedRunner(probe_report(sm="sm_99")),
    )

    result = provider.preflight(request(sm="sm_99"))

    assert result.status is DevicePreflightStatus.UNAVAILABLE
    assert result.reason_codes == ("CUDA_ARCHITECTURE_UNSUPPORTED",)


def test_framework_runtime_reuse_is_not_claimed_by_raw_e1_provider() -> None:
    raw_profile = profile()
    reused = ArtifactProfile(
        kind=raw_profile.kind,
        target_triple=raw_profile.target_triple,
        packaging_backend=raw_profile.packaging_backend,
        device_requirements=(
            DeviceRequirement(
                logical_device="cuda:0",
                backend="cuda",
                runtime="cuda-driver",
                architectures=("sm_80",),
                reuse_domain_runtime=True,
            ),
        ),
    )
    req = DevicePreflightRequest(
        artifact_profile=reused,
        selection=DeviceProviderSelection(
            provider_id=PROVIDER_ID,
            capability_id=CAPABILITY_LINUX_X86_64,
        ),
    )

    result = configured_provider().preflight(req)

    assert result.status is DevicePreflightStatus.UNAVAILABLE
    assert result.reason_codes == ("FRAMEWORK_RUNTIME_REUSE_UNSUPPORTED",)


def test_libtorch_runtime_reuse_is_a_separate_build_only_capability() -> None:
    provider = configured_provider()
    result = provider.preflight(libtorch_request())

    assert result.status is DevicePreflightStatus.READY
    assert result.support_claim is False
    assert ("framework.runtime", "libtorch") in result.observations
    assert ("framework.runtime-pin", "2.11.0") in result.observations
    assert ("framework.binding-pin", "0.24.0") in result.observations
    assert ("framework.reuse", "required-unverified") in result.observations
    assert (
        "policy.minimum-driver-version",
        str(CUDA_DRIVER_VERSION_FLOOR),
    ) in result.observations
    assert not any(key.startswith("toolkit.") for key, _ in result.observations)
    assert not any(key == "policy.minimum-toolkit-version" for key, _ in result.observations)


def test_libtorch_runtime_reuse_may_cross_check_one_domain_sm() -> None:
    provider = configured_provider()

    accepted = provider.preflight(
        libtorch_request(artifact_profile=libtorch_profile(architectures=("sm_80",)))
    )
    rejected = provider.preflight(
        libtorch_request(artifact_profile=libtorch_profile(architectures=("sm_90",)))
    )

    assert accepted.status is DevicePreflightStatus.READY
    assert rejected.status is DevicePreflightStatus.UNAVAILABLE
    assert rejected.reason_codes == ("DEVICE_REQUIREMENT_MISMATCH",)


def test_libtorch_contribution_only_borrows_framework_owned_resources() -> None:
    provider = configured_provider()
    req = libtorch_request()
    assert provider.preflight(req).status is DevicePreflightStatus.READY

    contribution = provider.build_contribution(req)

    assert contribution.cargo_features == ()
    assert contribution.native_libraries == ()
    assert contribution.package_references == ()
    assert contribution.generated_helper_ids == ()
    assert contribution.runtime_check_ids == ()
    assert {resource.resource_kind for resource in contribution.resource_contracts} == {
        "framework.tensor",
        "framework.allocator",
        "framework.current-stream",
    }
    assert all(
        resource.owner is DeviceResourceOwner.FRAMEWORK
        and resource.access is DeviceResourceAccess.BORROW_VALIDATE
        and not resource.may_allocate
        and not resource.may_replace
        and not resource.may_synchronize
        for resource in contribution.resource_contracts
    )
    assert not any(
        resource.resource_kind.startswith("raw.cuda.")
        for resource in contribution.resource_contracts
    )


@pytest.mark.parametrize(
    ("artifact_profile", "reason_code"),
    [
        (
            libtorch_profile(logical_device="gpu:1"),
            "LIBTORCH_CUDA_DEVICE_UNSUPPORTED",
        ),
        (
            libtorch_profile(backend="rocm"),
            "LIBTORCH_CUDA_DEVICE_UNSUPPORTED",
        ),
        (
            libtorch_profile(runtime="cuda-driver"),
            "LIBTORCH_RUNTIME_REQUIRED",
        ),
        (
            libtorch_profile(reuse_domain_runtime=False),
            "LIBTORCH_RUNTIME_REUSE_REQUIRED",
        ),
        (
            libtorch_profile(features=("inference",)),
            "LIBTORCH_FEATURES_INCOMPATIBLE",
        ),
        (
            libtorch_profile(layouts=("contiguous",)),
            "LIBTORCH_LAYOUT_INCOMPATIBLE",
        ),
        (
            libtorch_profile(memory_spaces=("host",)),
            "LIBTORCH_MEMORY_SPACE_INCOMPATIBLE",
        ),
        (
            libtorch_profile(architectures=("sm_80", "sm_90")),
            "AT_MOST_ONE_SM_ALLOWED",
        ),
        (
            libtorch_profile(architectures=("sm_99",)),
            "CUDA_ARCHITECTURE_UNSUPPORTED",
        ),
        (
            libtorch_profile(
                runtime_requirements=(
                    RuntimeRequirement(
                        "libtorch",
                        "2.10.0",
                        ("cuda", "pytorch-wheel"),
                    ),
                    RuntimeRequirement("tch", TCH_VERSION, ("cuda",)),
                )
            ),
            "LIBTORCH_RUNTIME_PINS_REQUIRED",
        ),
    ],
)
def test_libtorch_capability_rejects_non_frozen_framework_contract(
    artifact_profile: ArtifactProfile,
    reason_code: str,
) -> None:
    provider = configured_provider()

    result = provider.preflight(libtorch_request(artifact_profile=artifact_profile))

    assert result.status is DevicePreflightStatus.UNAVAILABLE
    assert result.reason_codes == (reason_code,)


def test_libtorch_capability_requires_exactly_one_device_requirement() -> None:
    valid = libtorch_profile()
    duplicate = ArtifactProfile(
        kind=valid.kind,
        target_triple=valid.target_triple,
        packaging_backend=valid.packaging_backend,
        python_fallback_backend=valid.python_fallback_backend,
        runtime_requirements=valid.runtime_requirements,
        device_requirements=(
            *valid.device_requirements,
            DeviceRequirement("cpu"),
        ),
    )

    result = configured_provider().preflight(libtorch_request(artifact_profile=duplicate))

    assert result.status is DevicePreflightStatus.UNAVAILABLE
    assert result.reason_codes == ("EXACTLY_ONE_CUDA_DEVICE_REQUIRED",)


def test_libtorch_capability_rejects_raw_only_toolkit_option_before_probe() -> None:
    runner = FixedRunner(probe_report())
    provider = CudaDeviceProvider(
        CudaProviderConfig(device_ordinal=0, sm="sm_80"),
        probe_runner=runner,
    )
    req = DevicePreflightRequest(
        artifact_profile=libtorch_profile(),
        selection=DeviceProviderSelection(
            provider_id=PROVIDER_ID,
            capability_id=CAPABILITY_LIBTORCH_LINUX_X86_64,
        ),
        options=DeviceProviderOptions(values=(("toolkit_root", "/opt/cuda"),)),
    )

    result = provider.preflight(req)

    assert result.status is DevicePreflightStatus.UNAVAILABLE
    assert result.reason_codes == ("PROVIDER_OPTION_UNKNOWN",)
    assert runner.calls == 0


def test_libtorch_capability_requires_explicit_provider_sm() -> None:
    runner = FixedRunner(probe_report())
    provider = CudaDeviceProvider(
        CudaProviderConfig(device_ordinal=0),
        probe_runner=runner,
    )

    result = provider.preflight(libtorch_request())

    assert result.status is DevicePreflightStatus.UNAVAILABLE
    assert result.reason_codes == ("SM_NOT_CONFIGURED",)
    assert runner.calls == 0


def test_libtorch_capability_rejects_wrong_target_and_capability_before_probe() -> None:
    valid = libtorch_profile()
    wrong_target = ArtifactProfile(
        kind=valid.kind,
        target_triple="x86_64-pc-windows-msvc",
        packaging_backend=valid.packaging_backend,
        python_fallback_backend=valid.python_fallback_backend,
        runtime_requirements=valid.runtime_requirements,
        device_requirements=valid.device_requirements,
    )
    runner = FixedRunner(probe_report())
    provider = CudaDeviceProvider(
        CudaProviderConfig(device_ordinal=0, sm="sm_80"),
        probe_runner=runner,
    )

    target_result = provider.preflight(libtorch_request(artifact_profile=wrong_target))
    unknown_capability = DevicePreflightRequest(
        artifact_profile=valid,
        selection=DeviceProviderSelection(
            provider_id=PROVIDER_ID,
            capability_id="cuda-libtorch-unknown",
        ),
    )
    capability_result = provider.preflight(unknown_capability)

    assert target_result.status is DevicePreflightStatus.INCOMPATIBLE
    assert target_result.reason_codes == ("TARGET_MISMATCH",)
    assert capability_result.status is DevicePreflightStatus.INCOMPATIBLE
    assert capability_result.reason_codes == ("CAPABILITY_UNKNOWN",)
    assert runner.calls == 0


def test_libtorch_capability_rejects_non_extension_artifact_before_probe() -> None:
    valid = libtorch_profile()
    rust_crate = ArtifactProfile(
        kind=ArtifactKind.RUST_CRATE,
        target_triple=valid.target_triple,
        packaging_backend="cargo",
        runtime_requirements=valid.runtime_requirements,
        device_requirements=valid.device_requirements,
    )
    runner = FixedRunner(probe_report())
    provider = CudaDeviceProvider(
        CudaProviderConfig(device_ordinal=0, sm="sm_80"),
        probe_runner=runner,
    )

    result = provider.preflight(libtorch_request(artifact_profile=rust_crate))

    assert result.status is DevicePreflightStatus.INCOMPATIBLE
    assert result.reason_codes == ("LIBTORCH_ARTIFACT_KIND_UNSUPPORTED",)
    assert runner.calls == 0


def test_libtorch_capability_rejects_constructor_toolkit_root_before_probe() -> None:
    runner = FixedRunner(probe_report())
    provider = CudaDeviceProvider(
        CudaProviderConfig(
            toolkit_root=Path("/opt/cuda"),
            device_ordinal=0,
            sm="sm_80",
        ),
        probe_runner=runner,
    )

    result = provider.preflight(libtorch_request())

    assert result.status is DevicePreflightStatus.UNAVAILABLE
    assert result.reason_codes == ("LIBTORCH_TOOLKIT_ROOT_UNSUPPORTED",)
    assert runner.calls == 0


def test_failed_repeat_preflight_invalidates_prior_ready_request() -> None:
    runner = FixedRunner(probe_report())
    provider = CudaDeviceProvider(
        CudaProviderConfig(device_ordinal=0, sm="sm_80"),
        probe_runner=runner,
        toolkit_inspector=FixedToolkit(
            CudaToolkitReport(
                version="12.8.0",
                runtime_version="12.8.0",
                components=("cuda-header", "cuda-runtime"),
            )
        ),
    )
    req = request()

    assert provider.preflight(req).status is DevicePreflightStatus.READY
    provider.build_contribution(req)

    runner.report = probe_report(sm="sm_90")
    failed = provider.preflight(req)

    assert failed.status is DevicePreflightStatus.UNAVAILABLE
    assert failed.reason_codes == ("CUDA_SM_MISMATCH",)
    with pytest.raises(
        RuntimeError,
        match="requires successful preflight",
    ):
        provider.build_contribution(req)


def test_newer_failed_preflight_prevents_older_success_from_restoring_ready() -> None:
    runner = OverlappingRunner()
    provider = CudaDeviceProvider(
        CudaProviderConfig(device_ordinal=0, sm="sm_80"),
        probe_runner=runner,
        toolkit_inspector=FixedToolkit(
            CudaToolkitReport(
                version="12.8.0",
                runtime_version="12.8.0",
                components=("cuda-header", "cuda-runtime"),
            )
        ),
    )
    req = request()
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        first = executor.submit(provider.preflight, req)
        assert runner.first_started.wait(timeout=5)

        second = executor.submit(provider.preflight, req)
        assert runner.second_started.wait(timeout=5)

        runner.release_first.set()
        assert first.result(timeout=5).status is DevicePreflightStatus.READY

        runner.release_second.set()
        failed = second.result(timeout=5)
        assert failed.status is DevicePreflightStatus.UNAVAILABLE
        assert failed.reason_codes == ("CUDA_SM_MISMATCH",)
    finally:
        runner.release_first.set()
        runner.release_second.set()
        executor.shutdown(wait=True, cancel_futures=True)

    with pytest.raises(
        RuntimeError,
        match="requires successful preflight",
    ):
        provider.build_contribution(req)
