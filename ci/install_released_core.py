"""Install an exact released Core version with bounded PyPI propagation retries."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def install_released_core(version: str, *, attempts: int, delay_seconds: float) -> None:
    """Install ``rextio==version`` or preserve pip's final nonzero exit code."""
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        f"rextio=={version}",
    ]
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            return
        if attempt == attempts:
            raise SystemExit(completed.returncode)
        print(
            f"Exact Core {version} is not visible yet; retrying in "
            f"{delay_seconds:g}s ({attempt}/{attempts}).",
            flush=True,
        )
        time.sleep(delay_seconds)


def main() -> None:
    """Parse command-line options and install the requested Core release."""
    parser = argparse.ArgumentParser(
        description="Install an exact released Core version despite short PyPI propagation delays."
    )
    parser.add_argument("version")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--delay-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be non-negative")
    install_released_core(
        args.version,
        attempts=args.attempts,
        delay_seconds=args.delay_seconds,
    )


if __name__ == "__main__":
    main()
