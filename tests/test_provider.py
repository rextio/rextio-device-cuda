from __future__ import annotations

from dataclasses import dataclass

import pytest

from rextio.artifacts import ArtifactKind, ArtifactProfile, DeviceRequirement
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
    CAPABILITY_LINUX_X86_64,
    PROVIDER_ID,
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
        "cuda-linux-aarch64",
        "cuda-linux-x86_64",
        "cuda-windows-x86_64",
    }
    assert {item.certification_tier.value for item in manifest.capabilities} == {
        "build-only"
    }
    assert {item.minimum_driver_version for item in manifest.capabilities} == {
        str(CUDA_DRIVER_VERSION_FLOOR)
    }
    assert {item.minimum_runtime_version for item in manifest.capabilities} == {
        CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR_TEXT
    }
    assert CudaProviderConfig().minimum_driver_version == CUDA_DRIVER_VERSION_FLOOR
    assert (
        CudaProviderConfig().minimum_toolkit_version
        == CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR
    )
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
    config = CudaProviderConfig(
        minimum_toolkit_version=CUDA_TOOLKIT_RUNTIME_VERSION_FLOOR
    )

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
        probe_runner=FixedRunner(
            probe_report(driver_version=CUDA_DRIVER_VERSION_FLOOR)
        ),
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
    assert plan.contribution.native_libraries == ()
    assert plan.contribution.cargo_features == ()
    assert plan.contribution.package_references == (
        "generated/device-providers/rextio-device-cuda/"
        "rextio-cuda-runtime/Cargo.toml",
    )
    assert plan.contribution.generated_helper_ids == ("rextio_cuda_runtime_v1",)
    assert plan.contribution.runtime_check_ids == (
        "rextio_cuda_driver_inventory_v1",
    )
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
    result = CudaDeviceProvider(
        CudaProviderConfig(device_ordinal=0, sm="sm_80")
    ).preflight(request())

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
        options=DeviceProviderOptions(
            values=(("unexpected_option", "/private/machine/secret"),)
        ),
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
