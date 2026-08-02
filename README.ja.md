# rextio-device-cuda

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio アイコン">
</p>

<p align="center"><strong>Rextio のための明示的で fail-closed な CUDA capability discovery。</strong></p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.ko.md">한국어</a> · <a href="README.zh-hans.md">简体中文</a> · <a href="README.zh-hant.md">繁體中文</a> · 日本語
</p>

`rextio-device-cuda` は Rextio の first-party NVIDIA CUDA Device Provider です。明示的に選択された driver、device、architecture、runtime 契約を検証し、provider capability を domain plugin に公開します。

> **公開済み・非認証 Alpha 0.1.0**（2026-07-26）。Python 3.11+ と `rextio>=0.1.6,<0.2` が必要です。すべての preflight/report レコードは `support_claim: false` としてシリアライズされます。
>
> これは capability provider であり、Python lowering plugin ではありません。NumPy、pandas、PyTorch、TensorFlow、その他の Python semantics を導出せず、製品 GPU kernel を実行せず、現在 CUDA support、certification、performance を主張しません。

## 現在証明されていること

- Device Provider API 1 manifest、明示的 option validation、deterministic lock/resource record、packaged Rust runtime source。
- 固定 driver/toolkit floor を用いる path-safe で明示設定された CUDA Driver API inventory。
- Provider-owned context、stream、device allocation lifetime に対する mocked/hosted contract test と Rust RAII test。
- 固定 libtorch および TensorFlow TFE lane の borrow/validate-only framework reuse contract。

Mock report、hosted CI、inventory、manual lifetime smoke は **GPU execution certification ではありません**。Manual smoke は `support_claim=false`、`kernel_executed=false`、`certification_ready=false` を記録します。

## 仕組み

```text
domain plugin requirement
        +
explicit provider selection/options
        ↓
driver/device/SM preflight → capability contribution → Core composition
```

Domain plugin が Python semantics を所有し、一致する CUDA `DeviceRequirement` を出す必要があります。Provider は要求 capability を検証し、その capability が許す resource だけを提供します。設定だけで CPU Python が GPU code になることはありません。

## クイックスタート

```bash
python -m pip install "rextio-device-cuda==0.1.0"
```

PyPI wheel は provider を登録し Rust runtime source を含みますが、`rextio-cuda-driver-probe` はインストールしません。そのため raw-driver lane には、信頼できる source checkout と明示的な probe build が必要です。

```bash
git clone --branch 0.1.0 --depth 1 https://github.com/rextio/rextio-device-cuda.git
cd rextio-device-cuda
cargo +1.93.1 build --locked --release -p rextio-cuda-driver-probe
```

以下では生成物の絶対パスを使用してください（Windows では `target/release/rextio-cuda-driver-probe.exe`）。

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

Raw option 値は memory に残ります。公開 lock/report は設定 path ではなく option name と SHA-256 digest だけを記録します。Entry point は `rextio.device_providers` group の `rextio-device-cuda` です。

## 宣言済み capability

| Capability / target | 契約 | 現在の状態 |
| --- | --- | --- |
| `cuda-linux-x86_64` | raw-driver CUDA、明示 probe/device/SM、任意 toolkit root | build-only、未認証 |
| `cuda-linux-aarch64` | 明示 probe/device/SM、任意 toolkit root | build-only cross-check、未認証 |
| `cuda-windows-x86_64` | 明示 probe/device/SM、任意 toolkit root | build-only、未認証 |
| `cuda-libtorch-linux-x86_64` | PyTorch/libtorch 2.11.0 + tch 0.24.0 runtime reuse | borrow/validate only、PyTorch CUDA support ではない |
| `cuda-tensorflow-tfe-linux-x86_64` | TensorFlow 2.21.0 + CPython 3.11 TFE runtime reuse | borrow/validate only、TensorFlow CUDA support ではない |

サポート architecture vocabulary：`sm_60`、`sm_61`、`sm_70`、`sm_72`、`sm_75`、`sm_80`、`sm_86`、`sm_87`、`sm_89`、`sm_90`。宣言は compatibility metadata であり、インストール済み system が実行可能という証明ではありません。

最小 raw-driver 契約は CUDA driver `12000`（12.0）と toolkit/runtime 12.0 です。明示 config は floor を上げられますが下げられません。Toolkit と runtime report は floor 以上の同一 version を 1 つ示す必要があります。Framework-reuse lane は standalone toolkit を所有しないため `toolkit_root` を拒否します。

## Framework reuse 境界

### libtorch lane

Domain requirement は `gpu:0`、`backend="cuda"`、`runtime="libtorch"`、inference + no-grad、strided device memory、libtorch/PyTorch 2.11.0、tch 0.24.0、選択 provider SM を指定する必要があります。Provider は framework tensor、allocator、current-stream の borrow/validate record だけを提供します。

Torch import、loaded libtorch/ATen image 検査、one-image ABI identity 証明、framework resource の allocation/synchronization、operation 実行は行いません。この provider だけで `rextio-torch` route を CUDA-capable にできません。

### TensorFlow TFE lane

Domain requirement は `gpu:0`、`runtime="tensorflow-tfe"`、eager + inference + no-grad、dense device memory、TensorFlow 2.21.0、CPython 3.11、選択 provider SM を指定する必要があります。Provider は framework tensor と eager-context の borrow/validate record だけを提供します。

TensorFlow import、TFE/private bridge symbol 解決、runtime image 検査、current-stream/event control 主張、framework resource の allocation/synchronization、GPU kernel 実行は行いません。この provider だけで `rextio-tensorflow` route を CUDA-capable にできません。

## 未対応・保留

- macOS、全 32-bit target、Windows ARM、non-NVIDIA accelerator。
- CUDA kernel generation、automatic provider selection、implicit hardware discovery、performance claim。
- Python-domain framework lowering、one-image ABI proof、正式な real-GPU support/certification。
- Packaged raw-driver Rust helper の一般 Core materialization。Core 0.1.6 は materialize されていない raw-runtime input を generated write 前に拒否します。
- Framework-owned resource の adoption、replacement、allocation、synchronization。

Provider-owned raw context、stream、allocation は RAII 管理され、この Alpha では意図的に thread-affine（`!Send` / `!Sync`）です。Host-owned shared driver loader はすべての injected API と child resource より長く存続する必要があります。

将来の Core materializer は、image handle が runtime API、context、stream、allocation より長く存続することを証明しなければなりません。

## Probe と security 境界

- `PATH`、registry、common-directory、ambient-environment discovery はありません。Production preflight は明示 absolute probe path を要求します。
- Linux はレビュー済み absolute `libcuda.so.1` path を試し、選択 image を canonicalize し、mutable/unreviewed root を拒否します。Windows は System32 の `nvcuda.dll` を load します。
- Probe output は bounded exact-schema です。Probe 選択は content identity を bind せず、certification evidence ではありません。
- Absolute path は `LD_PRELOAD`、transitive dependency、same-UID compromise、hostile OS を無効化しません。信頼できる host/filesystem だけで検証してください。

Probe 実行前に [SECURITY.md](SECURITY.md)、残る certification gate は [support matrix](docs/support-matrix.md)を参照してください。

## Manual NVIDIA inventory

信頼できる Linux host：

```bash
./scripts/validate-linux-nvidia.sh sm_80 0 /usr/local/cuda
```

信頼できる Windows PowerShell host：

```powershell
.\scripts\validate-windows-nvidia.ps1 -Sm sm_80 -DeviceOrdinal 0 `
  -ToolkitRoot "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
```

両 script は厳密な Rust 1.93.1 を要求し、local probe を build し、provider resolution と別の raw-driver RAII lifetime smoke を実行します。CUDA product kernel を実行せず、support を promote しません。

## 開発

```bash
python -m pip install "rextio==0.1.6"
python -m pytest -q
python -m ruff check src tests
python -m mypy src
cargo +1.93.1 test --workspace
cargo +1.93.1 clippy --workspace --all-targets -- -D warnings
```

[CHANGELOG.md](CHANGELOG.md)と [build-only evidence statement](docs/evidence/build-only.md)を参照してください。

## ライセンス

MIT
