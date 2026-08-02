# rextio-device-cuda

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio 图标">
</p>

<p align="center"><strong>为 Rextio 提供显式、fail-closed 的 CUDA capability discovery。</strong></p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.ko.md">한국어</a> · 简体中文 · <a href="README.zh-hant.md">繁體中文</a> · <a href="README.ja.md">日本語</a>
</p>

`rextio-device-cuda` 是 Rextio 的 first-party NVIDIA CUDA Device Provider。它验证显式选择的 driver、device、architecture 和 runtime 契约，再向 domain plugin 暴露 provider capability。

> **已发布、非认证 Alpha 0.1.0**（2026-07-26）。需要 Python 3.11+ 和 `rextio>=0.1.6,<0.2`。所有 preflight/report 记录都序列化为 `support_claim: false`。
>
> 这是 capability provider，不是 Python lowering plugin。它不推导 NumPy、pandas、PyTorch、TensorFlow 或其他 Python 语义，不执行产品 GPU kernel，当前也不作 CUDA support、certification 或 performance 声明。

## 当前已证明内容

- Device Provider API 1 manifest、显式 option validation、deterministic lock/resource record 和打包的 Rust runtime source。
- 采用固定 driver/toolkit floor、路径安全且显式配置的 CUDA Driver API inventory。
- 针对 provider-owned context、stream、device allocation lifetime 的 mocked/hosted contract test 与 Rust RAII test。
- 面向固定 libtorch 和 TensorFlow TFE lane 的 borrow/validate-only framework reuse contract。

Mock report、hosted CI、inventory 和 manual lifetime smoke **不是 GPU execution certification**。Manual smoke 记录 `support_claim=false`、`kernel_executed=false` 和 `certification_ready=false`。

## 工作原理

```text
domain plugin requirement
        +
explicit provider selection/options
        ↓
driver/device/SM preflight → capability contribution → Core composition
```

Domain plugin 拥有 Python 语义，并必须发出匹配的 CUDA `DeviceRequirement`。Provider 验证请求 capability，只贡献该 capability 允许的 resource。仅配置绝不会把 CPU Python 变成 GPU code。

## 快速开始

```bash
python -m pip install "rextio-device-cuda==0.1.0"
```

PyPI wheel 会注册 provider 并包含 Rust runtime source，但不会安装 `rextio-cuda-driver-probe`。因此，raw-driver lane 需要可信的 source checkout，并显式构建 probe：

```bash
git clone --branch 0.1.0 --depth 1 https://github.com/rextio/rextio-device-cuda.git
cd rextio-device-cuda
cargo +1.93.1 build --locked --release -p rextio-cuda-driver-probe
```

在下面使用生成文件的绝对路径（Windows 上为 `target/release/rextio-cuda-driver-probe.exe`）。

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

Raw option 值留在内存中。公开 lock/report 只记录 option name 和 SHA-256 digest，不记录配置路径。Entry point 是 `rextio.device_providers` group 中的 `rextio-device-cuda`。

## 声明的 capability

| Capability / target | 契约 | 当前状态 |
| --- | --- | --- |
| `cuda-linux-x86_64` | raw-driver CUDA、显式 probe/device/SM、可选 toolkit root | build-only，未认证 |
| `cuda-linux-aarch64` | 显式 probe/device/SM、可选 toolkit root | build-only cross-check，未认证 |
| `cuda-windows-x86_64` | 显式 probe/device/SM、可选 toolkit root | build-only，未认证 |
| `cuda-libtorch-linux-x86_64` | PyTorch/libtorch 2.11.0 + tch 0.24.0 runtime reuse | 仅 borrow/validate，不是 PyTorch CUDA support |
| `cuda-tensorflow-tfe-linux-x86_64` | TensorFlow 2.21.0 + CPython 3.11 TFE runtime reuse | 仅 borrow/validate，不是 TensorFlow CUDA support |

支持的 architecture vocabulary：`sm_60`、`sm_61`、`sm_70`、`sm_72`、`sm_75`、`sm_80`、`sm_86`、`sm_87`、`sm_89`、`sm_90`。声明只是 compatibility metadata，不是已安装系统能执行它的证明。

最低 raw-driver 契约是 CUDA driver `12000`（12.0）与 toolkit/runtime 12.0。显式 config 可提高 floor，绝不能降低。Toolkit 与 runtime report 必须给出同一个且不低于 floor 的版本。Framework-reuse lane 因不拥有 standalone toolkit 而拒绝 `toolkit_root`。

## Framework reuse 边界

### libtorch lane

Domain requirement 必须选择 `gpu:0`、`backend="cuda"`、`runtime="libtorch"`、inference + no-grad、strided device memory、libtorch/PyTorch 2.11.0、tch 0.24.0 和所选 provider SM。Provider 只贡献 framework tensor、allocator、current-stream 的 borrow/validate record。

它不 import torch、不检查已加载 libtorch/ATen image、不证明 one-image ABI identity、不分配或同步 framework resource，也不执行 operation。该 provider 单独无法让 `rextio-torch` route 具备 CUDA capability。

### TensorFlow TFE lane

Domain requirement 必须选择 `gpu:0`、`runtime="tensorflow-tfe"`、eager + inference + no-grad、dense device memory、TensorFlow 2.21.0、CPython 3.11 和所选 provider SM。Provider 只贡献 framework tensor 和 eager-context 的 borrow/validate record。

它不 import TensorFlow、不解析 TFE/private bridge symbol、不检查 runtime image、不声明 current-stream/event control、不分配或同步 framework resource，也不执行 GPU kernel。该 provider 单独无法让 `rextio-tensorflow` route 具备 CUDA capability。

## 不支持与延后内容

- macOS、所有 32-bit target、Windows ARM 和非 NVIDIA accelerator。
- CUDA kernel generation、automatic provider selection、implicit hardware discovery 和 performance claim。
- Python-domain framework lowering、one-image ABI proof、正式 real-GPU support/certification。
- 打包 raw-driver Rust helper 的通用 Core materialization。Core 0.1.6 会在 generated write 前拒绝未 materialize 的 raw-runtime input。
- 采用、替换、分配或同步 framework-owned resource。

Provider-owned raw context、stream 与 allocation 由 RAII 管理，并在此 Alpha 中刻意 thread-affine（`!Send` / `!Sync`）。Host-owned shared driver loader 必须比所有 injected API 和 child resource 存活更久。

未来的 Core materializer 必须证明 image handle 的生命周期长于 runtime API、context、stream 和 allocation。

## Probe 与安全边界

- 不做 `PATH`、registry、common-directory 或 ambient-environment discovery。Production preflight 要求显式 absolute probe path。
- Linux 尝试已审查 absolute `libcuda.so.1` path，canonicalize 所选 image，并拒绝 mutable/unreviewed root。Windows 从 System32 加载 `nvcuda.dll`。
- Probe output 有界且使用 exact schema。选择 probe 不绑定其 content identity，也不是 certification evidence。
- Absolute path 无法消除 `LD_PRELOAD`、transitive dependency、same-UID compromise 或 hostile OS。仅在可信 host/filesystem 上验证。

运行 probe 前请阅读 [SECURITY.md](SECURITY.md)，剩余 certification gate 见 [support matrix](docs/support-matrix.md)。

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

两个 script 都要求精确 Rust 1.93.1，build 本地 probe，执行 provider resolution 和独立 raw-driver RAII lifetime smoke。它们不执行 CUDA product kernel，也不提升 support。

## 开发

```bash
python -m pip install "rextio==0.1.6"
python -m pytest -q
python -m ruff check src tests
python -m mypy src
cargo +1.93.1 test --workspace
cargo +1.93.1 clippy --workspace --all-targets -- -D warnings
```

参见 [CHANGELOG.md](CHANGELOG.md)和 [build-only evidence statement](docs/evidence/build-only.md)。

## 许可证

MIT
