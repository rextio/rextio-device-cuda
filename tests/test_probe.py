from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import pytest

import rextio_device_cuda.probe as probe_module
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


def test_terminate_process_retries_wait_after_first_timeout() -> None:
    class TwoStageProcess:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.wait_count = 0
            self.reaped = False

        def poll(self) -> int | None:
            self.calls.append("poll")
            return 0 if self.reaped else None

        def kill(self) -> None:
            self.calls.append("kill")

        def wait(self, *, timeout: float) -> int:
            self.calls.append(f"wait:{timeout}")
            self.wait_count += 1
            if self.wait_count == 1:
                raise subprocess.TimeoutExpired(cmd="probe", timeout=timeout)
            self.reaped = True
            return 0

    process = TwoStageProcess()

    terminated = SubprocessProbeRunner._terminate_process(
        cast(subprocess.Popen[bytes], process)
    )

    assert terminated is True
    assert process.reaped is True
    assert process.calls == [
        "poll",
        "kill",
        "wait:5.0",
        "kill",
        "wait:5.0",
        "poll",
    ]


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


def test_subprocess_runner_reaps_process_when_reader_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes: list[subprocess.Popen[bytes]] = []
    real_popen = probe_module.subprocess.Popen

    def tracked_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    def fail_start(_thread: object) -> None:
        raise RuntimeError("injected thread-start failure")

    monkeypatch.setattr(probe_module.subprocess, "Popen", tracked_popen)
    monkeypatch.setattr(probe_module.threading.Thread, "start", fail_start)
    runner = SubprocessProbeRunner(
        Path(sys.executable),
        arguments=("-c", "import time; time.sleep(30)"),
        timeout_seconds=2,
    )

    with pytest.raises(CudaProbeError, match="PROBE_EXECUTION_FAILED"):
        runner.run()

    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdout is not None
    assert processes[0].stdout.closed
