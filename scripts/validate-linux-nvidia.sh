#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This validation script requires Linux." >&2
  exit 2
fi
case "$(uname -m)" in
  x86_64|aarch64) ;;
  *)
    echo "Only Linux x86_64 and aarch64 are declared." >&2
    exit 2
    ;;
esac

SM="${1:?usage: validate-linux-nvidia.sh sm_NN [device] [toolkit-root] [output]}"
DEVICE="${2:-0}"
TOOLKIT="${3:-}"
OUTPUT="${4:-cuda-inventory-linux.json}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"

cargo build --locked --release \
  -p rextio-cuda-driver-probe \
  -p rextio-cuda-runtime-smoke
PROBE="${ROOT}/target/release/rextio-cuda-driver-probe"
SMOKE="${ROOT}/target/release/rextio-cuda-runtime-smoke"

ARGS=(
  "${ROOT}/scripts/validate_preflight.py"
  --probe-executable "${PROBE}"
  --device-ordinal "${DEVICE}"
  --sm "${SM}"
  --output "${OUTPUT}"
)
if [[ -n "${TOOLKIT}" ]]; then
  ARGS+=(--toolkit-root "${TOOLKIT}")
fi
python3 "${ARGS[@]}"
SMOKE_OUTPUT="${OUTPUT%.json}-runtime-smoke.json"
"${SMOKE}" --device "${DEVICE}" > "${SMOKE_OUTPUT}"
echo "Inventory and RAII smoke evidence written."
echo "No kernel ran; this is not certification."
