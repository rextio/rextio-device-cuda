"""Run standalone, path-redacted CUDA provider inventory validation."""

from __future__ import annotations

import argparse
import json
import platform
import re
from pathlib import Path

from rextio.artifacts import ArtifactKind, ArtifactProfile, DeviceRequirement
from rextio.devices import (
    DeviceProviderOptions,
    DeviceProviderSelection,
    resolve_device_plan,
)
from rextio_device_cuda.provider import PROVIDER_ID, CudaDeviceProvider

_SM = re.compile(r"^sm_[0-9]{2,3}$")
_TARGETS = {
    ("linux", "x86_64"): (
        "x86_64-unknown-linux-gnu",
        "cuda-linux-x86_64",
    ),
    ("linux", "amd64"): (
        "x86_64-unknown-linux-gnu",
        "cuda-linux-x86_64",
    ),
    ("linux", "aarch64"): (
        "aarch64-unknown-linux-gnu",
        "cuda-linux-aarch64",
    ),
    ("windows", "amd64"): (
        "x86_64-pc-windows-msvc",
        "cuda-windows-x86_64",
    ),
    ("windows", "x86_64"): (
        "x86_64-pc-windows-msvc",
        "cuda-windows-x86_64",
    ),
}


def main() -> int:
    """Validate inventory and write path-free, non-certifying evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-executable", required=True)
    parser.add_argument("--toolkit-root")
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--sm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.device_ordinal <= 1_023:
        parser.error("--device-ordinal must be in [0, 1023]")
    if _SM.fullmatch(args.sm) is None:
        parser.error("--sm must use sm_NN or sm_NNN")
    target = _TARGETS.get((platform.system().lower(), platform.machine().lower()))
    if target is None:
        parser.error("host is not a declared Linux/Windows CUDA target")
    target_triple, capability_id = target
    values = [
        ("device_ordinal", str(args.device_ordinal)),
        ("probe_executable", args.probe_executable),
        ("sm", args.sm),
    ]
    if args.toolkit_root:
        values.append(("toolkit_root", args.toolkit_root))
    profile = ArtifactProfile(
        kind=ArtifactKind.RUST_CRATE,
        target_triple=target_triple,
        packaging_backend="cargo",
        device_requirements=(
            DeviceRequirement(
                logical_device=f"cuda:{args.device_ordinal}",
                backend="cuda",
                runtime="cuda-driver",
                features=("driver-api",),
                memory_spaces=("device",),
                architectures=(args.sm,),
            ),
        ),
    )
    selection = DeviceProviderSelection(PROVIDER_ID, capability_id)
    plan = resolve_device_plan(
        artifact_profile=profile,
        selection=selection,
        providers={PROVIDER_ID: CudaDeviceProvider()},
        options=DeviceProviderOptions(values=tuple(values)),
    )
    if plan is None:
        raise RuntimeError("explicit CUDA selection unexpectedly resolved to no plan")
    evidence = {
        "schema_version": 1,
        "validation_kind": "cuda-inventory-preflight-only",
        "support_claim": False,
        "kernel_executed": False,
        "certification_ready": False,
        "device_provider": plan.to_dict(),
    }
    output = Path(args.output)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("CUDA inventory preflight completed; support_claim=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
