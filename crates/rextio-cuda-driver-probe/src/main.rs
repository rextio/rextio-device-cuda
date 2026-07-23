//! Narrow CUDA Driver API inventory probe.
//!
//! Ported from Rextio Core's reviewed `tools/cuda-driver-probe` at commit
//! `ea56822`. The security boundary is intentionally unchanged:
//! - Windows x86_64 loads only System32 `nvcuda.dll` with `LoadLibraryExW`.
//! - Linux x86_64/aarch64 tries only reviewed absolute `libcuda.so.1` paths,
//!   then rejects mutable or out-of-root canonical images.
//!
//! It never creates a context, allocates memory, creates a stream, or runs a
//! kernel. A successful inventory is still not a device-support claim.

#![cfg_attr(
    not(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )),
    allow(dead_code, unused_imports)
)]

use std::ffi::{c_char, c_void};
use std::fmt::Write as _;

use rextio_cuda_driver_loader::DriverLibrary;

const SCHEMA_VERSION: &str = "1";
const PROBE_NAME: &str = "rextio-cuda-driver-probe";

#[derive(Debug, Eq, PartialEq)]
struct DeviceRecord {
    ordinal: i32,
    name: String,
    compute_major: i32,
    compute_minor: i32,
}

#[derive(Debug, Eq, PartialEq)]
struct Report {
    status: &'static str,
    reason_code: Option<&'static str>,
    platform_supported: bool,
    driver_loaded: bool,
    driver_version: Option<i32>,
    device_count: Option<i32>,
    cuda_result: Option<i32>,
    devices: Vec<DeviceRecord>,
}

impl Report {
    #[cfg(not(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    fn unsupported() -> Self {
        Self {
            status: "unsupported",
            reason_code: Some("UNSUPPORTED_TARGET"),
            platform_supported: false,
            driver_loaded: false,
            driver_version: None,
            device_count: None,
            cuda_result: None,
            devices: Vec::new(),
        }
    }

    fn unavailable(reason_code: &'static str, driver_loaded: bool) -> Self {
        Self {
            status: "unavailable",
            reason_code: Some(reason_code),
            platform_supported: true,
            driver_loaded,
            driver_version: None,
            device_count: None,
            cuda_result: None,
            devices: Vec::new(),
        }
    }

    fn cuda_failure(reason_code: &'static str, result: i32) -> Self {
        Self {
            status: "unavailable",
            reason_code: Some(reason_code),
            platform_supported: true,
            driver_loaded: true,
            driver_version: None,
            device_count: None,
            cuda_result: Some(result),
            devices: Vec::new(),
        }
    }

    fn to_json(&self) -> String {
        let mut output = String::from("{");
        field(&mut output, "schema_version", &quoted(SCHEMA_VERSION), true);
        field(&mut output, "probe", &quoted(PROBE_NAME), false);
        field(&mut output, "support_claim", "false", false);
        output.push_str(",\"target\":{");
        field(&mut output, "os", &quoted(std::env::consts::OS), true);
        field(&mut output, "arch", &quoted(std::env::consts::ARCH), false);
        field(
            &mut output,
            "environment",
            &quoted(target_environment()),
            false,
        );
        output.push('}');
        field(
            &mut output,
            "platform_supported",
            if self.platform_supported {
                "true"
            } else {
                "false"
            },
            false,
        );
        field(&mut output, "status", &quoted(self.status), false);
        field(
            &mut output,
            "reason_code",
            &self.reason_code.map_or_else(|| "null".to_owned(), quoted),
            false,
        );
        field(
            &mut output,
            "driver_loaded",
            if self.driver_loaded { "true" } else { "false" },
            false,
        );
        field(
            &mut output,
            "driver_version",
            &optional_i32(self.driver_version),
            false,
        );
        field(
            &mut output,
            "device_count",
            &optional_i32(self.device_count),
            false,
        );
        field(
            &mut output,
            "cuda_result",
            &optional_i32(self.cuda_result),
            false,
        );
        output.push_str(",\"devices\":[");
        for (index, device) in self.devices.iter().enumerate() {
            if index != 0 {
                output.push(',');
            }
            output.push('{');
            field(&mut output, "ordinal", &device.ordinal.to_string(), true);
            field(&mut output, "name", &quoted(&device.name), false);
            field(
                &mut output,
                "compute_major",
                &device.compute_major.to_string(),
                false,
            );
            field(
                &mut output,
                "compute_minor",
                &device.compute_minor.to_string(),
                false,
            );
            field(
                &mut output,
                "sm",
                &quoted(&format!(
                    "sm_{}{}",
                    device.compute_major, device.compute_minor
                )),
                false,
            );
            output.push('}');
        }
        output.push_str("]}");
        output
    }
}

fn field(output: &mut String, key: &str, value: &str, first: bool) {
    if !first {
        output.push(',');
    }
    let _ = write!(output, "{}:{}", quoted(key), value);
}

fn optional_i32(value: Option<i32>) -> String {
    value.map_or_else(|| "null".to_owned(), |item| item.to_string())
}

fn quoted(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            value if value.is_control() => {
                let _ = write!(output, "\\u{:04x}", value as u32);
            }
            value => output.push(value),
        }
    }
    output.push('"');
    output
}

fn target_environment() -> &'static str {
    if cfg!(target_env = "msvc") {
        "msvc"
    } else if cfg!(target_env = "gnu") {
        "gnu"
    } else if cfg!(target_env = "musl") {
        "musl"
    } else {
        "unknown"
    }
}

type CuDevice = i32;
type CuResult = i32;
type CuInit = unsafe extern "system" fn(u32) -> CuResult;
type CuDriverGetVersion = unsafe extern "system" fn(*mut i32) -> CuResult;
type CuDeviceGetCount = unsafe extern "system" fn(*mut i32) -> CuResult;
type CuDeviceGet = unsafe extern "system" fn(*mut CuDevice, i32) -> CuResult;
type CuDeviceGetName = unsafe extern "system" fn(*mut c_char, i32, CuDevice) -> CuResult;
type CuDeviceComputeCapability =
    unsafe extern "system" fn(*mut i32, *mut i32, CuDevice) -> CuResult;

struct DriverSymbols {
    init: CuInit,
    driver_version: CuDriverGetVersion,
    device_count: CuDeviceGetCount,
    device_get: CuDeviceGet,
    device_name: CuDeviceGetName,
    compute_capability: CuDeviceComputeCapability,
}

fn inventory(symbols: DriverSymbols) -> Report {
    // This process is the standalone probe. These variables reduce incidental
    // CUDA initialization but are not treated as a sandbox.
    std::env::set_var("CUDA_FORCE_PRELOAD_LIBRARIES", "0");
    std::env::set_var("CUDA_DISABLE_JIT", "1");
    // SAFETY: all pointers were resolved from one live, reviewed driver image.
    let init = unsafe { (symbols.init)(0) };
    if init != 0 {
        return Report::cuda_failure("CU_INIT_FAILED", init);
    }
    let mut driver_version = 0;
    // SAFETY: output pointer is valid.
    let result = unsafe { (symbols.driver_version)(&mut driver_version) };
    if result != 0 {
        return Report::cuda_failure("CU_DRIVER_VERSION_FAILED", result);
    }
    let mut device_count = 0;
    // SAFETY: output pointer is valid.
    let result = unsafe { (symbols.device_count)(&mut device_count) };
    if result != 0 {
        return Report::cuda_failure("CU_DEVICE_COUNT_FAILED", result);
    }
    if !(0..=1024).contains(&device_count) {
        return Report::unavailable("INVALID_DEVICE_COUNT", true);
    }
    if device_count == 0 {
        return Report {
            status: "unavailable",
            reason_code: Some("NO_CUDA_DEVICES"),
            platform_supported: true,
            driver_loaded: true,
            driver_version: Some(driver_version),
            device_count: Some(0),
            cuda_result: Some(0),
            devices: Vec::new(),
        };
    }
    let mut devices = Vec::with_capacity(device_count as usize);
    for ordinal in 0..device_count {
        let mut device = 0;
        // SAFETY: ordinal is bounded by the same driver-reported count.
        let result = unsafe { (symbols.device_get)(&mut device, ordinal) };
        if result != 0 {
            return Report::cuda_failure("CU_DEVICE_GET_FAILED", result);
        }
        let mut name = [0_u8; 256];
        // SAFETY: writable buffer and exact byte length are supplied.
        let result =
            unsafe { (symbols.device_name)(name.as_mut_ptr().cast(), name.len() as i32, device) };
        if result != 0 {
            return Report::cuda_failure("CU_DEVICE_NAME_FAILED", result);
        }
        let mut major = 0;
        let mut minor = 0;
        // SAFETY: both output pointers are valid.
        let result = unsafe { (symbols.compute_capability)(&mut major, &mut minor, device) };
        if result != 0 {
            return Report::cuda_failure("CU_DEVICE_CAPABILITY_FAILED", result);
        }
        if !(1..=99).contains(&major) || !(0..=99).contains(&minor) {
            return Report::unavailable("INVALID_COMPUTE_CAPABILITY", true);
        }
        let bytes: Vec<u8> = name.into_iter().take_while(|value| *value != 0).collect();
        devices.push(DeviceRecord {
            ordinal,
            name: sanitize_name(&String::from_utf8_lossy(&bytes)),
            compute_major: major,
            compute_minor: minor,
        });
    }
    Report {
        status: "probe-complete",
        reason_code: None,
        platform_supported: true,
        driver_loaded: true,
        driver_version: Some(driver_version),
        device_count: Some(device_count),
        cuda_result: Some(0),
        devices,
    }
}

fn sanitize_name(value: &str) -> String {
    let value: String = value
        .chars()
        .filter(|character| !character.is_control())
        .take(160)
        .collect();
    let value = value.trim();
    if value.is_empty() || value.starts_with('/') || value.starts_with('\\') {
        "unknown".to_owned()
    } else {
        value.to_owned()
    }
}

#[cfg(any(
    all(target_os = "windows", target_arch = "x86_64"),
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn run_probe() -> Report {
    let library = match DriverLibrary::load() {
        Ok(library) => library,
        Err(error) => return Report::unavailable(error.reason_code(), false),
    };
    macro_rules! resolve {
        ($name:literal, $kind:ty) => {{
            let pointer = match library.symbol(concat!($name, "\0").as_bytes()) {
                Ok(pointer) => pointer,
                Err(error) => return Report::unavailable(error.reason_code(), true),
            };
            // SAFETY: the fixed symbol name and type match the CUDA Driver API.
            unsafe { std::mem::transmute::<*mut c_void, $kind>(pointer.as_ptr()) }
        }};
    }
    inventory(DriverSymbols {
        init: resolve!("cuInit", CuInit),
        driver_version: resolve!("cuDriverGetVersion", CuDriverGetVersion),
        device_count: resolve!("cuDeviceGetCount", CuDeviceGetCount),
        device_get: resolve!("cuDeviceGet", CuDeviceGet),
        device_name: resolve!("cuDeviceGetName", CuDeviceGetName),
        compute_capability: resolve!("cuDeviceComputeCapability", CuDeviceComputeCapability),
    })
}

#[cfg(not(any(
    all(target_os = "windows", target_arch = "x86_64"),
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
)))]
fn run_probe() -> Report {
    Report::unsupported()
}

fn main() {
    println!("{}", run_probe().to_json());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_is_path_free_and_never_support_claim() {
        let json = Report::unavailable("DRIVER_LIBRARY_NOT_FOUND", false).to_json();
        assert!(json.contains("\"support_claim\":false"));
        assert!(json.contains("\"reason_code\":\"DRIVER_LIBRARY_NOT_FOUND\""));
        assert!(!json.contains(std::env::temp_dir().to_string_lossy().as_ref()));
    }

    #[test]
    fn json_escaping_is_deterministic() {
        assert_eq!(quoted("a\n\"b\\c"), "\"a\\n\\\"b\\\\c\"");
    }

    #[cfg(not(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    #[test]
    fn unsupported_host_does_not_load_any_driver() {
        let report = run_probe();
        assert_eq!(report.status, "unsupported");
        assert!(!report.driver_loaded);
    }
}
