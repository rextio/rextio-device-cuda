//! Manual-only real-driver lifetime smoke for provider-owned raw resources.
//!
//! This binary is excluded from support claims and ordinary CI execution. It
//! loads the same reviewed driver image as the inventory probe, resolves the
//! runtime's nine symbols from that one live image, restores current-context
//! state around every operation, and emits path-free JSON.

use std::ffi::c_void;
use std::sync::Arc;

use rextio_cuda_driver_loader::DriverLibrary;
use rextio_cuda_runtime::{CuContext, CuDevice, CuDevicePtr, CuResult, CuStream, DriverApi};

type CuInit = unsafe extern "system" fn(u32) -> CuResult;
type CuDeviceGet = unsafe extern "system" fn(*mut CuDevice, i32) -> CuResult;
type CuCtxCreate = unsafe extern "system" fn(*mut CuContext, u32, CuDevice) -> CuResult;
type CuCtxDestroy = unsafe extern "system" fn(CuContext) -> CuResult;
type CuCtxPushCurrent = unsafe extern "system" fn(CuContext) -> CuResult;
type CuCtxPopCurrent = unsafe extern "system" fn(*mut CuContext) -> CuResult;
type CuStreamCreate = unsafe extern "system" fn(*mut CuStream, u32) -> CuResult;
type CuStreamDestroy = unsafe extern "system" fn(CuStream) -> CuResult;
type CuStreamSynchronize = unsafe extern "system" fn(CuStream) -> CuResult;
type CuMemAlloc = unsafe extern "system" fn(*mut CuDevicePtr, usize) -> CuResult;
type CuMemFree = unsafe extern "system" fn(CuDevicePtr) -> CuResult;

#[derive(Debug)]
struct SmokeError {
    reason_code: &'static str,
    cuda_result: Option<i32>,
}

fn resolve<T: Copy>(library: &DriverLibrary, name: &'static [u8]) -> Result<T, SmokeError> {
    let pointer = library.symbol(name).map_err(|error| SmokeError {
        reason_code: error.reason_code(),
        cuda_result: None,
    })?;
    // SAFETY: every call below pairs a fixed CUDA symbol with its exact
    // function-pointer type; function and data pointers share width here.
    if std::mem::size_of::<T>() != std::mem::size_of::<*mut c_void>() {
        return Err(SmokeError {
            reason_code: "FUNCTION_POINTER_SIZE_MISMATCH",
            cuda_result: None,
        });
    }
    // SAFETY: size was checked and caller fixes T at the corresponding alias.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&pointer.as_ptr()) })
}

fn run(device_ordinal: i32) -> Result<(), SmokeError> {
    let library = DriverLibrary::load().map_err(|error| SmokeError {
        reason_code: error.reason_code(),
        cuda_result: None,
    })?;
    let init: CuInit = resolve(&library, b"cuInit\0")?;
    let device_get: CuDeviceGet = resolve(&library, b"cuDeviceGet\0")?;
    let ctx_create: CuCtxCreate = resolve(&library, b"cuCtxCreate_v2\0")?;
    let ctx_destroy: CuCtxDestroy = resolve(&library, b"cuCtxDestroy_v2\0")?;
    let ctx_push: CuCtxPushCurrent = resolve(&library, b"cuCtxPushCurrent_v2\0")?;
    let ctx_pop: CuCtxPopCurrent = resolve(&library, b"cuCtxPopCurrent_v2\0")?;
    let stream_create: CuStreamCreate = resolve(&library, b"cuStreamCreate\0")?;
    let stream_destroy: CuStreamDestroy = resolve(&library, b"cuStreamDestroy_v2\0")?;
    let stream_sync: CuStreamSynchronize = resolve(&library, b"cuStreamSynchronize\0")?;
    let mem_alloc: CuMemAlloc = resolve(&library, b"cuMemAlloc_v2\0")?;
    let mem_free: CuMemFree = resolve(&library, b"cuMemFree_v2\0")?;

    // SAFETY: pointer came from the reviewed live image with exact signature.
    let result = unsafe { init(0) };
    if result != 0 {
        return Err(SmokeError {
            reason_code: "CU_INIT_FAILED",
            cuda_result: Some(result),
        });
    }
    let mut device = 0;
    // SAFETY: output pointer is valid and ordinal was CLI-bounded.
    let result = unsafe { device_get(&mut device, device_ordinal) };
    if result != 0 {
        return Err(SmokeError {
            reason_code: "CU_DEVICE_GET_FAILED",
            cuda_result: Some(result),
        });
    }
    // SAFETY: all nine runtime symbols came from `library`, which remains live
    // until after api and all resources below are explicitly released.
    let api = Arc::new(unsafe {
        DriverApi::from_symbols(
            ctx_create,
            ctx_destroy,
            ctx_push,
            ctx_pop,
            stream_create,
            stream_destroy,
            stream_sync,
            mem_alloc,
            mem_free,
        )
    });
    let mut context = api.create_context(device, 0).map_err(|error| SmokeError {
        reason_code: "CONTEXT_CREATE_OR_DETACH_FAILED",
        cuda_result: Some(error.code()),
    })?;
    let stream = context
        .create_dedicated_stream(0)
        .map_err(|error| SmokeError {
            reason_code: "STREAM_CREATE_FAILED",
            cuda_result: Some(error.code()),
        })?;
    let allocation = context.allocate(4096).map_err(|error| SmokeError {
        reason_code: "DEVICE_ALLOCATION_FAILED",
        cuda_result: Some(error.code()),
    })?;
    stream.synchronize().map_err(|error| SmokeError {
        reason_code: "STREAM_SYNCHRONIZE_FAILED",
        cuda_result: Some(error.code()),
    })?;
    allocation.close().map_err(|error| SmokeError {
        reason_code: "DEVICE_FREE_FAILED",
        cuda_result: Some(error.code()),
    })?;
    stream.close().map_err(|error| SmokeError {
        reason_code: "STREAM_DESTROY_FAILED",
        cuda_result: Some(error.code()),
    })?;
    context.close().map_err(|error| SmokeError {
        reason_code: "CONTEXT_DESTROY_FAILED",
        cuda_result: Some(error.code()),
    })?;
    drop(api);
    drop(library);
    Ok(())
}

fn target_environment() -> &'static str {
    if cfg!(target_env = "msvc") {
        "msvc"
    } else if cfg!(target_env = "gnu") {
        "gnu"
    } else {
        "unknown"
    }
}

fn report(status: &str, reason: Option<&str>, result: Option<i32>, ordinal: i32) {
    let reason = reason.map_or_else(|| "null".to_owned(), |value| format!("\"{value}\""));
    let result = result.map_or_else(|| "null".to_owned(), |value| value.to_string());
    println!(
        concat!(
            "{{\"schema_version\":\"1\",",
            "\"probe\":\"rextio-cuda-runtime-smoke\",",
            "\"support_claim\":false,",
            "\"target\":{{\"os\":\"{}\",\"arch\":\"{}\",\"environment\":\"{}\"}},",
            "\"status\":\"{}\",\"reason_code\":{},\"cuda_result\":{},",
            "\"device_ordinal\":{},\"allocation_bytes\":4096,",
            "\"kernel_executed\":false,\"certification_ready\":false}}"
        ),
        std::env::consts::OS,
        std::env::consts::ARCH,
        target_environment(),
        status,
        reason,
        result,
        ordinal,
    );
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let ordinal = if arguments.len() == 3 && arguments[1] == "--device" {
        arguments[2].parse::<i32>().ok()
    } else {
        None
    };
    let Some(ordinal) = ordinal.filter(|value| (0..=1023).contains(value)) else {
        report("error", Some("INVALID_ARGUMENTS"), None, -1);
        std::process::exit(2);
    };
    match run(ordinal) {
        Ok(()) => report("smoke-complete", None, Some(0), ordinal),
        Err(error) => {
            report(
                "unavailable",
                Some(error.reason_code),
                error.cuda_result,
                ordinal,
            );
            std::process::exit(1);
        }
    }
}
