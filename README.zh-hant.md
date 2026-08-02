# rextio-device-cuda

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio 圖示">
</p>

<p align="center"><strong>為 Rextio 提供明確、fail-closed 的 CUDA capability discovery。</strong></p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.ko.md">한국어</a> · <a href="README.zh-hans.md">简体中文</a> · 繁體中文 · <a href="README.ja.md">日本語</a>
</p>

`rextio-device-cuda` 是 Rextio 的 first-party NVIDIA CUDA Device Provider。它驗證明確選取的 driver、device、architecture 與 runtime 契約，再向 domain plugin 暴露 provider capability。

> **已發布、非認證 Alpha 0.1.0**（2026-07-26）。需要 Python 3.11+ 與 `rextio>=0.1.6,<0.2`。所有 preflight/report 記錄都序列化為 `support_claim: false`。
>
> 這是 capability provider，不是 Python lowering plugin。它不推導 NumPy、pandas、PyTorch、TensorFlow 或其他 Python 語意，不執行產品 GPU kernel，目前也不作 CUDA support、certification 或 performance 聲明。

## 目前已證明內容

- Device Provider API 1 manifest、明確 option validation、deterministic lock/resource record 與封裝的 Rust runtime source。
- 採用固定 driver/toolkit floor、路徑安全且明確設定的 CUDA Driver API inventory。
- 針對 provider-owned context、stream、device allocation lifetime 的 mocked/hosted contract test 與 Rust RAII test。
- 面向固定 libtorch 與 TensorFlow TFE lane 的 borrow/validate-only framework reuse contract。

Mock report、hosted CI、inventory 和 manual lifetime smoke **不是 GPU execution certification**。Manual smoke 記錄 `support_claim=false`、`kernel_executed=false` 與 `certification_ready=false`。

## 運作方式

```text
domain plugin requirement
        +
explicit provider selection/options
        ↓
driver/device/SM preflight → capability contribution → Core composition
```

Domain plugin 擁有 Python 語意，且必須發出相符的 CUDA `DeviceRequirement`。Provider 驗證請求 capability，只貢獻該 capability 允許的 resource。僅設定絕不會把 CPU Python 變成 GPU code。

## 快速開始

```bash
python -m pip install "rextio-device-cuda==0.1.0"
```

PyPI wheel 會註冊 provider 並包含 Rust runtime source，但不會安裝 `rextio-cuda-driver-probe`。因此，raw-driver lane 需要可信的 source checkout，並明確建置 probe：

```bash
git clone --branch 0.1.0 --depth 1 https://github.com/rextio/rextio-device-cuda.git
cd rextio-device-cuda
cargo +1.93.1 build --locked --release -p rextio-cuda-driver-probe
```

在下方使用產出檔案的絕對路徑（Windows 上為 `target/release/rextio-cuda-driver-probe.exe`）。

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

Raw option 值留在記憶體中。公開 lock/report 只記錄 option name 與 SHA-256 digest，不記錄設定路徑。Entry point 是 `rextio.device_providers` group 中的 `rextio-device-cuda`。

## 宣告的 capability

| Capability / target | 契約 | 目前狀態 |
| --- | --- | --- |
| `cuda-linux-x86_64` | raw-driver CUDA、明確 probe/device/SM、選用 toolkit root | build-only，未認證 |
| `cuda-linux-aarch64` | 明確 probe/device/SM、選用 toolkit root | build-only cross-check，未認證 |
| `cuda-windows-x86_64` | 明確 probe/device/SM、選用 toolkit root | build-only，未認證 |
| `cuda-libtorch-linux-x86_64` | PyTorch/libtorch 2.11.0 + tch 0.24.0 runtime reuse | 僅 borrow/validate，不是 PyTorch CUDA support |
| `cuda-tensorflow-tfe-linux-x86_64` | TensorFlow 2.21.0 + CPython 3.11 TFE runtime reuse | 僅 borrow/validate，不是 TensorFlow CUDA support |

支援的 architecture vocabulary：`sm_60`、`sm_61`、`sm_70`、`sm_72`、`sm_75`、`sm_80`、`sm_86`、`sm_87`、`sm_89`、`sm_90`。宣告只是 compatibility metadata，不是已安裝系統能執行它的證明。

最低 raw-driver 契約是 CUDA driver `12000`（12.0）與 toolkit/runtime 12.0。明確 config 可提高 floor，絕不能降低。Toolkit 與 runtime report 必須提出同一個且不低於 floor 的版本。Framework-reuse lane 因不擁有 standalone toolkit 而拒絕 `toolkit_root`。

## Framework reuse 邊界

### libtorch lane

Domain requirement 必須選取 `gpu:0`、`backend="cuda"`、`runtime="libtorch"`、inference + no-grad、strided device memory、libtorch/PyTorch 2.11.0、tch 0.24.0 與所選 provider SM。Provider 只貢獻 framework tensor、allocator、current-stream 的 borrow/validate record。

它不 import torch、不檢查已載入 libtorch/ATen image、不證明 one-image ABI identity、不配置或同步 framework resource，也不執行 operation。該 provider 單獨無法讓 `rextio-torch` route 具備 CUDA capability。

### TensorFlow TFE lane

Domain requirement 必須選取 `gpu:0`、`runtime="tensorflow-tfe"`、eager + inference + no-grad、dense device memory、TensorFlow 2.21.0、CPython 3.11 與所選 provider SM。Provider 只貢獻 framework tensor 和 eager-context 的 borrow/validate record。

它不 import TensorFlow、不解析 TFE/private bridge symbol、不檢查 runtime image、不宣告 current-stream/event control、不配置或同步 framework resource，也不執行 GPU kernel。該 provider 單獨無法讓 `rextio-tensorflow` route 具備 CUDA capability。

## 不支援與延後內容

- macOS、所有 32-bit target、Windows ARM 與非 NVIDIA accelerator。
- CUDA kernel generation、automatic provider selection、implicit hardware discovery 與 performance claim。
- Python-domain framework lowering、one-image ABI proof、正式 real-GPU support/certification。
- 封裝 raw-driver Rust helper 的通用 Core materialization。Core 0.1.6 會在 generated write 前拒絕未 materialize 的 raw-runtime input。
- 採用、取代、配置或同步 framework-owned resource。

Provider-owned raw context、stream 與 allocation 由 RAII 管理，並在此 Alpha 中刻意 thread-affine（`!Send` / `!Sync`）。Host-owned shared driver loader 必須比所有 injected API 與 child resource 存活更久。

未來的 Core materializer 必須證明 image handle 的生命週期長於 runtime API、context、stream 與 allocation。

## Probe 與安全邊界

- 不做 `PATH`、registry、common-directory 或 ambient-environment discovery。Production preflight 要求明確 absolute probe path。
- Linux 嘗試已審查 absolute `libcuda.so.1` path，canonicalize 所選 image，並拒絕 mutable/unreviewed root。Windows 從 System32 載入 `nvcuda.dll`。
- Probe output 有界且使用 exact schema。選取 probe 不綁定其 content identity，也不是 certification evidence。
- Absolute path 無法消除 `LD_PRELOAD`、transitive dependency、same-UID compromise 或 hostile OS。僅在可信 host/filesystem 上驗證。

執行 probe 前請閱讀 [SECURITY.md](SECURITY.md)，剩餘 certification gate 見 [support matrix](docs/support-matrix.md)。

## Manual NVIDIA inventory

可信 Linux host：

```bash
./scripts/validate-linux-nvidia.sh sm_80 0 /usr/local/cuda
```

可信 Windows PowerShell host：

```powershell
.\scripts\validate-windows-nvidia.ps1 -Sm sm_80 -DeviceOrdinal 0 `
  -ToolkitRoot "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
```

兩個 script 都要求精確 Rust 1.93.1，build 本機 probe，執行 provider resolution 與獨立 raw-driver RAII lifetime smoke。它們不執行 CUDA product kernel，也不提升 support。

## 開發

```bash
python -m pip install "rextio==0.1.6"
python -m pytest -q
python -m ruff check src tests
python -m mypy src
cargo +1.93.1 test --workspace
cargo +1.93.1 clippy --workspace --all-targets -- -D warnings
```

參見 [CHANGELOG.md](CHANGELOG.md)與 [build-only evidence statement](docs/evidence/build-only.md)。

## 授權條款

MIT
