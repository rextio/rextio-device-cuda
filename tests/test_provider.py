from __future__ import annotations

from dataclasses import dataclass

import pytest

from rextio.artifacts import ArtifactKind, ArtifactProfile, DeviceRequirement
from rextio.devices import (
    DevicePreflightRequest,
    DevicePreflightStatus,
    DeviceProviderSelection,
    DeviceResourceAccess,
    DeviceResourceOwner,
    resolve_device_plan,
)
from rextio_device_cuda.config import CudaProviderConfig
from rextio_device_cuda.probe import (
    CudaDeviceRecord,
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


def probe_report(*, sm: str = "sm_80") -> CudaProbeReport:
    major = int(sm[3:-1])
    minor = int(sm[-1])
    return CudaProbeReport(
        target=ProbeTarget(os="linux", arch="x86_64", environment="gnu"),
        platform_supported=True,
        status="probe-complete",
        reason_code=None,
        driver_loaded=True,
        driver_version=12080,
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
    assert not hasattr(provider, "claim")
    assert not hasattr(provider, "lower")


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


def test_sm_mismatch_fails_before_build_contribution() -> None:
    provider = configured_provider(report_sm="sm_90")
    req = request()

    result = provider.preflight(req)

    assert result.status is DevicePreflightStatus.UNAVAILABLE
    assert result.reason_codes == ("CUDA_SM_MISMATCH",)
    with pytest.raises(RuntimeError, match="requires successful preflight"):
        provider.build_contribution(req)


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

