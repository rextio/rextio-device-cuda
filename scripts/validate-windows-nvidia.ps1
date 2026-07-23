param(
  [Parameter(Mandatory = $true)]
  [ValidatePattern("^sm_[0-9]{2,3}$")]
  [string]$Sm,
  [ValidateRange(0, 1023)]
  [int]$DeviceOrdinal = 0,
  [string]$ToolkitRoot = "",
  [string]$Output = "cuda-inventory-windows.json"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $Root
try {
  $RustToolchain = "1.93.1"
  $RustcVersion = (& rustc "+$RustToolchain" --version)
  if ($LASTEXITCODE -ne 0 -or $RustcVersion -notmatch "^rustc 1\.93\.1 ") {
    throw "rustc 1.93.1 is required."
  }
  $CargoVersion = (& cargo "+$RustToolchain" --version)
  if ($LASTEXITCODE -ne 0 -or $CargoVersion -notmatch "^cargo 1\.93\.1 ") {
    throw "cargo 1.93.1 is required."
  }

  cargo "+$RustToolchain" build --locked --release `
    -p rextio-cuda-driver-probe `
    -p rextio-cuda-runtime-smoke
  $Probe = Join-Path $Root "target\release\rextio-cuda-driver-probe.exe"
  $Smoke = Join-Path $Root "target\release\rextio-cuda-runtime-smoke.exe"
  $Arguments = @(
    (Join-Path $Root "scripts\validate_preflight.py"),
    "--probe-executable", $Probe,
    "--device-ordinal", "$DeviceOrdinal",
    "--sm", $Sm,
    "--output", $Output
  )
  if ($ToolkitRoot) {
    $Arguments += @("--toolkit-root", $ToolkitRoot)
  }
  python @Arguments
  $SmokeOutput = $Output -replace "\.json$", "-runtime-smoke.json"
  & $Smoke --device "$DeviceOrdinal" | Out-File -FilePath $SmokeOutput -Encoding utf8
  if ($LASTEXITCODE -ne 0) {
    throw "CUDA RAII smoke failed; inspect the bounded smoke JSON."
  }
  Write-Host "Inventory and RAII smoke evidence written."
  Write-Host "No kernel ran; this is not certification."
}
finally {
  Pop-Location
}
