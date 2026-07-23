from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from rextio.artifacts import ArtifactKind, ArtifactProfile
from rextio.build.orchestrator import _resolve_build_device_plans
from rextio.devices import (
    DEVICE_PROVIDER_ENTRY_POINT,
    DeviceProviderError,
    DeviceProviderOptions,
    DeviceProviderSelection,
    load_selected_device_provider,
    resolve_device_plan,
)
from rextio_device_cuda.config import CudaProviderConfig
from rextio_device_cuda.provider import (
    CAPABILITY_LINUX_X86_64,
    PROVIDER_ID,
    CudaDeviceProvider,
)

from test_provider import FixedRunner, FixedToolkit, probe_report, profile
from rextio_device_cuda.probe import CudaToolkitReport


@dataclass(frozen=True)
class FakeDistribution:
    name: str = "rextio-device-cuda"
    version: str = "0.1.0"


class FakeEntryPoint:
    name = PROVIDER_ID
    group = DEVICE_PROVIDER_ENTRY_POINT
    dist = FakeDistribution()

    def __init__(self, payload: CudaDeviceProvider) -> None:
        self._payload = payload

    def load(self) -> CudaDeviceProvider:
        return self._payload


def ready_provider() -> CudaDeviceProvider:
    return CudaDeviceProvider(
        CudaProviderConfig(),
        probe_runner=FixedRunner(probe_report()),
        toolkit_inspector=FixedToolkit(
            CudaToolkitReport(
                version="12.8.0",
                runtime_version="12.8.0",
                components=("cuda-header", "cuda-runtime"),
            )
        ),
    )


def selection() -> DeviceProviderSelection:
    return DeviceProviderSelection(
        provider_id=PROVIDER_ID,
        capability_id=CAPABILITY_LINUX_X86_64,
    )


def test_selected_provider_resolves_standalone_plan_with_redacted_options() -> None:
    provider, source = load_selected_device_provider(
        selection(),
        entry_points=(FakeEntryPoint(ready_provider()),),
    )
    plan = resolve_device_plan(
        artifact_profile=profile(),
        selection=selection(),
        providers={PROVIDER_ID: provider},
        provider_sources={PROVIDER_ID: source},
        options=DeviceProviderOptions(
            values=(
                ("device_ordinal", "0"),
                ("sm", "sm_80"),
            )
        ),
    )
    assert plan is not None

    record = plan.to_dict()
    assert record["manifest"]["provider_id"] == PROVIDER_ID
    assert record["contribution"]["native_libraries"] == ["cuda"]
    assert record["contribution"]["package_references"]
    assert record["report"]["certification_tier"] == "build-only"
    assert record["report"]["support_claim"] is False
    assert record["lock"]["option_keys"] == ["device_ordinal", "sm"]
    assert record["lock"]["options_sha256"]


def test_current_core_materialization_gate_fails_before_generated_writes(
    tmp_path: Path,
) -> None:
    raw = profile()
    host_profile = ArtifactProfile(
        kind=ArtifactKind.HOST_EXTENSION,
        target_triple=raw.target_triple,
        packaging_backend="cargo",
        python_fallback_backend="cpython",
        device_requirements=raw.device_requirements,
    )

    with pytest.raises(DeviceProviderError, match="cannot be represented"):
        _resolve_build_device_plans(
            (host_profile,),
            selection=selection(),
            options=DeviceProviderOptions(
                values=(("device_ordinal", "0"), ("sm", "sm_80"))
            ),
            entry_points=(FakeEntryPoint(ready_provider()),),
        )

    assert not (tmp_path / ".rextio").exists()


def test_missing_production_probe_fails_before_generated_writes(tmp_path: Path) -> None:
    provider = CudaDeviceProvider(
        CudaProviderConfig(device_ordinal=0, sm="sm_80")
    )

    with pytest.raises(DeviceProviderError, match="PROBE_NOT_CONFIGURED"):
        resolve_device_plan(
            artifact_profile=profile(),
            selection=selection(),
            providers={PROVIDER_ID: provider},
        )

    assert not (tmp_path / ".rextio").exists()


def test_unsupported_target_fails_before_probe_or_generated_writes(
    tmp_path: Path,
) -> None:
    unsupported = profile()
    unsupported = unsupported.__class__(
        kind=unsupported.kind,
        target_triple="aarch64-apple-darwin",
        packaging_backend=unsupported.packaging_backend,
        device_requirements=unsupported.device_requirements,
    )
    runner = FixedRunner(probe_report())
    provider = CudaDeviceProvider(
        CudaProviderConfig(device_ordinal=0, sm="sm_80"),
        probe_runner=runner,
    )

    with pytest.raises(DeviceProviderError, match="incompatible"):
        resolve_device_plan(
            artifact_profile=unsupported,
            selection=selection(),
            providers={PROVIDER_ID: provider},
        )

    assert runner.calls == 0
    assert not (tmp_path / ".rextio").exists()
