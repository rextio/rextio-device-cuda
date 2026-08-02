# rextio-device-cuda

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio 아이콘">
</p>

<p align="center"><strong>Rextio를 위한 명시적이고 fail-closed인 CUDA capability discovery.</strong></p>

<p align="center">
  <a href="README.md">English</a> · 한국어 · <a href="README.zh-hans.md">简体中文</a> · <a href="README.zh-hant.md">繁體中文</a> · <a href="README.ja.md">日本語</a>
</p>

`rextio-device-cuda`는 Rextio의 first-party NVIDIA CUDA Device Provider입니다. 명시적으로 선택된 driver, device, architecture, runtime 계약을 검증한 뒤 provider capability를 domain plugin에 노출합니다.

> **공개된 비인증 Alpha 0.1.0** (2026-07-26). Python 3.11+와 `rextio>=0.1.6,<0.2`가 필요합니다. 모든 preflight/report record는 `support_claim: false`로 직렬화됩니다.
>
> 이것은 capability provider이지 Python lowering plugin이 아닙니다. NumPy, pandas, PyTorch, TensorFlow 또는 다른 Python semantics를 도출하지 않고 제품 GPU kernel을 실행하지 않으며 현재 CUDA support, certification, performance를 주장하지 않습니다.

## 현재 증명된 것

- Device Provider API 1 manifest, 명시적 option validation, deterministic lock/resource record, packaged Rust runtime source.
- 고정 driver/toolkit floor를 사용하는 path-safe 명시적 CUDA Driver API inventory.
- Provider-owned context, stream, device allocation lifetime에 대한 mocked/hosted contract test와 Rust RAII test.
- 고정된 libtorch와 TensorFlow TFE lane을 위한 borrow/validate-only framework reuse contract.

Mock report, hosted CI, inventory, manual lifetime smoke는 **GPU execution certification이 아닙니다**. Manual smoke는 `support_claim=false`, `kernel_executed=false`, `certification_ready=false`를 기록합니다.

## 동작 방식

```text
domain plugin requirement
        +
explicit provider selection/options
        ↓
driver/device/SM preflight → capability contribution → Core composition
```

Domain plugin이 Python semantics를 소유하고 일치하는 CUDA `DeviceRequirement`를 내야 합니다. Provider는 요청 capability를 검증하고 해당 capability가 허용하는 resource만 제공합니다. 설정만으로 CPU Python이 GPU code가 되지 않습니다.

## 빠른 시작

```bash
python -m pip install "rextio-device-cuda==0.1.0"
```

PyPI wheel은 provider를 등록하고 Rust runtime source를 포함하지만 `rextio-cuda-driver-probe`를 설치하지는 않습니다. 따라서 raw-driver lane에는 신뢰할 수 있는 source checkout과 명시적인 probe build가 필요합니다.

```bash
git clone --branch 0.1.0 --depth 1 https://github.com/rextio/rextio-device-cuda.git
cd rextio-device-cuda
cargo +1.93.1 build --locked --release -p rextio-cuda-driver-probe
```

아래에는 생성된 파일의 절대 경로를 사용하십시오(Windows에서는 `target/release/rextio-cuda-driver-probe.exe`).

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

Raw option 값은 memory에 남습니다. 공개 lock/report는 설정 path가 아니라 option name과 SHA-256 digest만 기록합니다. Entry point는 `rextio.device_providers` group의 `rextio-device-cuda`입니다.

## 선언된 capability

| Capability / target | 계약 | 현재 상태 |
| --- | --- | --- |
| `cuda-linux-x86_64` | raw-driver CUDA, 명시적 probe/device/SM, 선택 toolkit root | build-only, 미인증 |
| `cuda-linux-aarch64` | 명시적 probe/device/SM, 선택 toolkit root | build-only cross-check, 미인증 |
| `cuda-windows-x86_64` | 명시적 probe/device/SM, 선택 toolkit root | build-only, 미인증 |
| `cuda-libtorch-linux-x86_64` | PyTorch/libtorch 2.11.0 + tch 0.24.0 runtime reuse | borrow/validate only, PyTorch CUDA support 아님 |
| `cuda-tensorflow-tfe-linux-x86_64` | TensorFlow 2.21.0 + CPython 3.11 TFE runtime reuse | borrow/validate only, TensorFlow CUDA support 아님 |

지원 architecture vocabulary: `sm_60`, `sm_61`, `sm_70`, `sm_72`, `sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_89`, `sm_90`. 선언은 compatibility metadata이지 설치된 system이 실행할 수 있다는 증명이 아닙니다.

최소 raw-driver 계약은 CUDA driver `12000`(12.0) 및 toolkit/runtime 12.0입니다. 명시적 config는 floor를 높일 수 있지만 낮출 수 없습니다. Toolkit과 runtime report는 floor 이상인 동일 version 하나를 제시해야 합니다. Framework-reuse lane은 standalone toolkit을 소유하지 않으므로 `toolkit_root`를 거부합니다.

## Framework reuse 경계

### libtorch lane

Domain requirement는 `gpu:0`, `backend="cuda"`, `runtime="libtorch"`, inference + no-grad, strided device memory, libtorch/PyTorch 2.11.0, tch 0.24.0, 선택된 provider SM을 요구해야 합니다. Provider는 framework tensor, allocator, current-stream borrow/validate record만 제공합니다.

Torch import, loaded libtorch/ATen image 검사, one-image ABI identity 증명, framework resource allocation/synchronization, operation 실행은 하지 않습니다. 이 provider만으로 `rextio-torch` route를 CUDA-capable하게 만들 수 없습니다.

### TensorFlow TFE lane

Domain requirement는 `gpu:0`, `runtime="tensorflow-tfe"`, eager + inference + no-grad, dense device memory, TensorFlow 2.21.0, CPython 3.11, 선택된 provider SM을 요구해야 합니다. Provider는 framework tensor와 eager-context borrow/validate record만 제공합니다.

TensorFlow import, TFE/private bridge symbol resolution, runtime image 검사, current-stream/event control 주장, framework resource allocation/synchronization, GPU kernel 실행은 하지 않습니다. 이 provider만으로 `rextio-tensorflow` route를 CUDA-capable하게 만들 수 없습니다.

## 미지원 및 보류

- macOS, 모든 32-bit target, Windows ARM, non-NVIDIA accelerator.
- CUDA kernel generation, automatic provider selection, implicit hardware discovery, performance claim.
- Python-domain framework lowering, one-image ABI proof, formal real-GPU support/certification.
- Packaged raw-driver Rust helper의 일반 Core materialization. Core 0.1.6은 materialize되지 않은 raw-runtime input을 generated write 전에 거부합니다.
- Framework-owned resource의 adoption, replacement, allocation, synchronization.

Provider-owned raw context, stream, allocation은 RAII로 관리되며 이 Alpha에서는 의도적으로 thread-affine (`!Send` / `!Sync`)입니다. Host-owned shared driver loader는 모든 injected API와 child resource보다 오래 살아야 합니다.

향후 Core materializer는 image handle이 runtime API, context, stream, allocation보다 오래 유지됨을 증명해야 합니다.

## Probe와 security 경계

- `PATH`, registry, common-directory, ambient-environment discovery가 없습니다. Production preflight는 명시적 absolute probe path를 요구합니다.
- Linux는 검토된 absolute `libcuda.so.1` path를 시도하고 선택 image를 canonicalize하며 mutable/unreviewed root를 거부합니다. Windows는 System32의 `nvcuda.dll`을 load합니다.
- Probe output은 bounded exact-schema입니다. Probe 선택은 content identity를 bind하지 않으며 certification evidence가 아닙니다.
- Absolute path는 `LD_PRELOAD`, transitive dependency, same-UID compromise, hostile OS를 무력화하지 않습니다. 신뢰하는 host/filesystem에서만 검증하세요.

Probe 실행 전 [SECURITY.md](SECURITY.md), 남은 certification gate는 [support matrix](docs/support-matrix.md)를 참고하세요.

## Manual NVIDIA inventory

신뢰하는 Linux host:

```bash
./scripts/validate-linux-nvidia.sh sm_80 0 /usr/local/cuda
```

신뢰하는 Windows PowerShell host:

```powershell
.\scripts\validate-windows-nvidia.ps1 -Sm sm_80 -DeviceOrdinal 0 `
  -ToolkitRoot "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
```

두 script는 정확한 Rust 1.93.1을 요구하고 local probe를 build하며 provider resolution과 별도 raw-driver RAII lifetime smoke를 실행합니다. CUDA product kernel을 실행하거나 support를 promote하지 않습니다.

## 개발

```bash
python -m pip install "rextio==0.1.6"
python -m pytest -q
python -m ruff check src tests
python -m mypy src
cargo +1.93.1 test --workspace
cargo +1.93.1 clippy --workspace --all-targets -- -D warnings
```

[CHANGELOG.md](CHANGELOG.md)와 [build-only evidence statement](docs/evidence/build-only.md)를 참고하세요.

## 라이선스

MIT
