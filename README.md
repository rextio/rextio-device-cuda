# rextio-device-cuda

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio icon">
</p>

<p align="center"><strong>Explicit, fail-closed CUDA capability discovery for Rextio.</strong></p>

<p align="center">
  English · <a href="https://github.com/rextio/rextio-device-cuda/blob/main/README.ko.md">한국어</a> · <a href="https://github.com/rextio/rextio-device-cuda/blob/main/README.zh-hans.md">简体中文</a> · <a href="https://github.com/rextio/rextio-device-cuda/blob/main/README.zh-hant.md">繁體中文</a> · <a href="https://github.com/rextio/rextio-device-cuda/blob/main/README.ja.md">日本語</a>
</p>

`rextio-device-cuda` is the first-party NVIDIA CUDA Device Provider for Rextio. It validates an explicitly selected driver, device, architecture, and runtime contract, then exposes provider capabilities to a domain plugin.

> **Published non-certifying Alpha 0.1.0** (2026-07-26). Requires Python 3.11+ and `rextio>=0.1.6,<0.2`. All preflight and report records serialize `support_claim: false`.
>
> This is a capability provider—not a Python lowering plugin. It derives no NumPy, pandas, PyTorch, TensorFlow, or other Python semantics, executes no product GPU kernel, and currently makes no CUDA support, certification, or performance claim.

## What is proven today

- Device Provider API 1 manifests, explicit option validation, deterministic lock/resource records, and packaged Rust runtime source.
- Path-safe, explicitly configured CUDA Driver API inventory with fixed driver/toolkit floors.
- Mocked/hosted contract tests and Rust RAII tests for provider-owned context, stream, and device allocation lifetimes.
- Borrow/validate-only framework reuse contracts for the pinned libtorch and TensorFlow TFE lanes.

Mock reports, hosted CI, inventory, and manual lifetime smoke are **not GPU execution certification**. Manual smoke records `support_claim=false`, `kernel_executed=false`, and `certification_ready=false`.

## How it works

```text
domain plugin requirement
        +
explicit provider selection/options
        ↓
driver/device/SM preflight → capability contribution → Core composition
```

The domain plugin owns Python semantics and must emit a matching CUDA `DeviceRequirement`. The provider validates the requested capability and contributes only the resources permitted by that capability. Configuration alone never turns CPU Python into GPU code.

## Quick start

```bash
python -m pip install "rextio-device-cuda==0.1.0"
```

The PyPI wheel registers the provider and includes its Rust runtime source, but it does **not** install `rextio-cuda-driver-probe`. A raw-driver lane therefore needs a trusted source checkout and an explicit probe build:

```bash
git clone --branch 0.1.0 --depth 1 https://github.com/rextio/rextio-device-cuda.git
cd rextio-device-cuda
cargo +1.93.1 build --locked --release -p rextio-cuda-driver-probe
```

Use the resulting absolute path below (`target/release/rextio-cuda-driver-probe.exe` on Windows).

```toml
# rextio.toml
[target]
device_provider = "rextio-device-cuda"
device_capability = "cuda-linux-x86_64"

[target.device_options]
probe_executable = "/absolute/path/to/rextio-cuda-driver-probe"
device_ordinal = "0"
sm = "sm_80"
# toolkit_root = "/absolute/path/to/cuda"  # optional validation for raw-driver lanes
```

The raw option values stay in memory. Public locks and reports record option names plus a SHA-256 digest, never the configured paths. The entry point is `rextio-device-cuda` in `rextio.device_providers`.

## Declared capabilities

| Capability / target | Contract | Current status |
| --- | --- | --- |
| `cuda-linux-x86_64` | raw-driver CUDA, explicit probe/device/SM, optional toolkit root | build-only, not certified |
| `cuda-linux-aarch64` | explicit probe/device/SM, optional toolkit root | build-only cross-check, not certified |
| `cuda-windows-x86_64` | explicit probe/device/SM, optional toolkit root | build-only, not certified |
| `cuda-libtorch-linux-x86_64` | PyTorch/libtorch 2.11.0 + tch 0.24.0 runtime reuse | borrow/validate only, not PyTorch CUDA support |
| `cuda-tensorflow-tfe-linux-x86_64` | TensorFlow 2.21.0 + CPython 3.11 TFE runtime reuse | borrow/validate only, not TensorFlow CUDA support |

Supported architecture vocabulary: `sm_60`, `sm_61`, `sm_70`, `sm_72`, `sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_89`, and `sm_90`. A declaration is compatibility metadata, not proof that an installed system can execute it.

The minimum raw-driver contract is CUDA driver `12000` (12.0) and toolkit/runtime 12.0. Explicit configuration may raise these floors, never lower them. Toolkit and runtime reports must name one identical version at or above the floor. Framework-reuse lanes reject `toolkit_root` because they do not own a standalone toolkit.

## Framework reuse boundaries

### libtorch lane

The domain requirement must select `gpu:0`, `backend="cuda"`, `runtime="libtorch"`, inference + no-grad, strided device memory, libtorch/PyTorch 2.11.0, tch 0.24.0, and the selected provider SM. The provider contributes framework tensor, allocator, and current-stream borrow/validate records only.

It does not import torch, inspect loaded libtorch/ATen images, prove one-image ABI identity, allocate or synchronize framework resources, or execute an operation. This provider alone cannot make a `rextio-torch` route CUDA-capable.

### TensorFlow TFE lane

The domain requirement must select `gpu:0`, `runtime="tensorflow-tfe"`, eager + inference + no-grad, dense device memory, TensorFlow 2.21.0, CPython 3.11, and the selected provider SM. The provider contributes framework tensor and eager-context borrow/validate records only.

It does not import TensorFlow, resolve TFE/private bridge symbols, inspect runtime images, claim current-stream/event control, allocate or synchronize framework resources, or execute a GPU kernel. This provider alone cannot make a `rextio-tensorflow` route CUDA-capable.

## Unsupported and deferred

- macOS, every 32-bit target, Windows ARM, and non-NVIDIA accelerators.
- CUDA kernel generation, automatic provider selection, implicit hardware discovery, and performance claims.
- Python-domain framework lowering, one-image ABI proof, and formal real-GPU support/certification.
- General Core materialization of the packaged raw-driver Rust helper. Core 0.1.6 rejects unmaterialized raw-runtime inputs before generated writes.
- Adopting, replacing, allocating, or synchronizing framework-owned resources.

Provider-owned raw contexts, streams, and allocations are RAII-managed and deliberately thread-affine (`!Send` / `!Sync`) in this Alpha. The host-owned shared driver loader must outlive every injected API and child resource.

Any future Core materializer must prove that the image handle outlives the runtime API, contexts, streams, and allocations.

## Probe and security boundary

- There is no `PATH`, registry, common-directory, or ambient-environment discovery. Production preflight requires an explicit absolute probe path.
- Linux tries reviewed absolute `libcuda.so.1` paths, canonicalizes the selected image, and rejects mutable/unreviewed roots. Windows loads `nvcuda.dll` from System32.
- Probe output is bounded and exact-schema. Selecting a probe does not bind its content identity and is not certification evidence.
- Absolute paths do not neutralize `LD_PRELOAD`, transitive dependencies, same-UID compromise, or a hostile OS. Run validation only on a trusted host/filesystem.

See [SECURITY.md](SECURITY.md) before running a probe and [the support matrix](docs/support-matrix.md) for the remaining certification gates.

## Manual NVIDIA inventory

Trusted Linux host:

```bash
./scripts/validate-linux-nvidia.sh sm_80 0 /usr/local/cuda
```

Trusted Windows PowerShell host:

```powershell
.\scripts\validate-windows-nvidia.ps1 -Sm sm_80 -DeviceOrdinal 0 `
  -ToolkitRoot "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
```

Both scripts require exact Rust 1.93.1, build the local probe, perform provider resolution, and run a separate raw-driver RAII lifetime smoke. They do not execute a CUDA product kernel or promote support.

## Development

```bash
python -m pip install "rextio==0.1.6"
python -m pytest -q
python -m ruff check src tests
python -m mypy src
cargo +1.93.1 test --workspace
cargo +1.93.1 clippy --workspace --all-targets -- -D warnings
```

See [CHANGELOG.md](CHANGELOG.md) and the [build-only evidence statement](docs/evidence/build-only.md).

## License

MIT
