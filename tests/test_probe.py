from __future__ import annotations

import json

import pytest

from rextio_device_cuda.probe import CudaProbeError, parse_probe_report


def valid_report() -> dict[str, object]:
    return {
        "schema_version": "1",
        "probe": "rextio-cuda-driver-probe",
        "support_claim": False,
        "target": {"os": "linux", "arch": "x86_64", "environment": "gnu"},
        "platform_supported": True,
        "status": "probe-complete",
        "reason_code": None,
        "driver_loaded": True,
        "driver_version": 12080,
        "device_count": 1,
        "cuda_result": 0,
        "devices": [
            {
                "ordinal": 0,
                "name": "NVIDIA Test GPU",
                "compute_major": 8,
                "compute_minor": 0,
                "sm": "sm_80",
            }
        ],
    }


def test_parse_probe_report_accepts_exact_bounded_schema() -> None:
    report = parse_probe_report(json.dumps(valid_report()))

    assert report.driver_version == 12080
    assert report.devices[0].sm == "sm_80"
    assert report.target.arch == "x86_64"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda report: report.update(support_claim=True), "PROBE_SCHEMA_INVALID"),
        (lambda report: report.update(extra="value"), "PROBE_SCHEMA_INVALID"),
        (
            lambda report: report["devices"][0].update(name="/private/driver.so"),
            "PROBE_SCHEMA_INVALID",
        ),
        (
            lambda report: report["devices"][0].update(sm="sm_90"),
            "PROBE_SCHEMA_INVALID",
        ),
        (
            lambda report: report.update(device_count=2),
            "PROBE_SCHEMA_INVALID",
        ),
    ],
)
def test_parse_probe_report_rejects_malformed_or_path_bearing_rows(
    mutate: object,
    reason: str,
) -> None:
    report = valid_report()
    mutate(report)  # type: ignore[operator]

    with pytest.raises(CudaProbeError, match=reason):
        parse_probe_report(json.dumps(report))


def test_parse_probe_report_enforces_output_budget() -> None:
    with pytest.raises(CudaProbeError, match="PROBE_OUTPUT_INVALID"):
        parse_probe_report(b"x" * 65_537)

