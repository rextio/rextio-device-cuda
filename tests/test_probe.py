from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from rextio_device_cuda.probe import (
    CudaProbeError,
    SubprocessProbeRunner,
    parse_probe_report,
)


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


def test_subprocess_runner_reaps_nonzero_probe() -> None:
    runner = SubprocessProbeRunner(
        Path(sys.executable),
        arguments=("-c", "raise SystemExit(7)"),
        timeout_seconds=2,
    )

    with pytest.raises(CudaProbeError, match="PROBE_EXIT_NONZERO"):
        runner.run()


def test_subprocess_runner_kills_reaps_and_joins_on_timeout() -> None:
    runner = SubprocessProbeRunner(
        Path(sys.executable),
        arguments=("-c", "import time; time.sleep(30)"),
        timeout_seconds=0.1,
    )
    started = time.monotonic()

    with pytest.raises(CudaProbeError, match="PROBE_TIMEOUT"):
        runner.run()

    assert time.monotonic() - started < 5


def test_subprocess_runner_kills_on_max_plus_one_without_buffering_rest() -> None:
    runner = SubprocessProbeRunner(
        Path(sys.executable),
        arguments=(
            "-c",
            "import os,time; os.write(1,b'x'*65537); time.sleep(30)",
        ),
        timeout_seconds=10,
    )
    started = time.monotonic()

    with pytest.raises(CudaProbeError, match="PROBE_OUTPUT_TOO_LARGE"):
        runner.run()

    assert time.monotonic() - started < 5
