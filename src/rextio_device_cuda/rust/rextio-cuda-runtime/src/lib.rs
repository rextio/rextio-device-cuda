//! Provider-owned raw CUDA Driver API resource lifetime primitives.
//!
//! This crate intentionally does not wrap framework tensors, allocators,
//! current streams, or events. PyTorch/TensorFlow adapters must validate and
//! borrow those framework-owned resources without transferring ownership.
//!
//! A caller resolves the exact driver symbols through its reviewed loader,
//! then constructs [`DriverApi`] once. No symbol search or PATH lookup occurs
//! in this crate.

#![forbid(unsafe_op_in_unsafe_fn)]

use std::cell::Cell;
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

/// CUDA Driver API result code.
pub type CuResult = i32;
/// CUDA device ordinal.
pub type CuDevice = i32;
/// Opaque CUDA context handle.
pub type CuContext = *mut c_void;
/// Opaque CUDA stream handle.
pub type CuStream = *mut c_void;
/// CUDA device address.
pub type CuDevicePtr = u64;

/// Successful CUDA Driver API result.
pub const CUDA_SUCCESS: CuResult = 0;

type CuCtxCreate = unsafe extern "system" fn(*mut CuContext, u32, CuDevice) -> CuResult;
type CuCtxDestroy = unsafe extern "system" fn(CuContext) -> CuResult;
type CuCtxPushCurrent = unsafe extern "system" fn(CuContext) -> CuResult;
type CuCtxPopCurrent = unsafe extern "system" fn(*mut CuContext) -> CuResult;
type CuStreamCreate = unsafe extern "system" fn(*mut CuStream, u32) -> CuResult;
type CuStreamDestroy = unsafe extern "system" fn(CuStream) -> CuResult;
type CuStreamSynchronize = unsafe extern "system" fn(CuStream) -> CuResult;
type CuMemAlloc = unsafe extern "system" fn(*mut CuDevicePtr, usize) -> CuResult;
type CuMemFree = unsafe extern "system" fn(CuDevicePtr) -> CuResult;

/// One CUDA Driver API failure without unstable driver text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaError {
    operation: &'static str,
    code: CuResult,
}

impl CudaError {
    /// Stable operation identifier.
    pub fn operation(self) -> &'static str {
        self.operation
    }

    /// Numeric CUDA result.
    pub fn code(self) -> CuResult {
        self.code
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} failed with CUDA result {}",
            self.operation, self.code
        )
    }
}

impl std::error::Error for CudaError {}

fn check(operation: &'static str, result: CuResult) -> Result<(), CudaError> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError {
            operation,
            code: result,
        })
    }
}

/// Exact function table whose lifetime dominates all provider-owned resources.
///
/// The dynamic-library owner must retain its library handle for at least as
/// long as this value and every resource holding an `Arc<DriverApi>`.
pub struct DriverApi {
    cu_ctx_create: CuCtxCreate,
    cu_ctx_destroy: CuCtxDestroy,
    cu_ctx_push_current: CuCtxPushCurrent,
    cu_ctx_pop_current: CuCtxPopCurrent,
    cu_stream_create: CuStreamCreate,
    cu_stream_destroy: CuStreamDestroy,
    cu_stream_synchronize: CuStreamSynchronize,
    cu_mem_alloc: CuMemAlloc,
    cu_mem_free: CuMemFree,
}

impl DriverApi {
    /// Construct from symbols resolved from one provenance-checked driver image.
    ///
    /// # Safety
    ///
    /// Every pointer must have the exact CUDA Driver API signature shown here.
    /// The image containing them must remain loaded until the returned API and
    /// all resources created from it have been dropped.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_symbols(
        cu_ctx_create: CuCtxCreate,
        cu_ctx_destroy: CuCtxDestroy,
        cu_ctx_push_current: CuCtxPushCurrent,
        cu_ctx_pop_current: CuCtxPopCurrent,
        cu_stream_create: CuStreamCreate,
        cu_stream_destroy: CuStreamDestroy,
        cu_stream_synchronize: CuStreamSynchronize,
        cu_mem_alloc: CuMemAlloc,
        cu_mem_free: CuMemFree,
    ) -> Self {
        Self {
            cu_ctx_create,
            cu_ctx_destroy,
            cu_ctx_push_current,
            cu_ctx_pop_current,
            cu_stream_create,
            cu_stream_destroy,
            cu_stream_synchronize,
            cu_mem_alloc,
            cu_mem_free,
        }
    }

    /// Create a provider-owned raw context for one validated device ordinal.
    pub fn create_context(
        self: &Arc<Self>,
        device: CuDevice,
        flags: u32,
    ) -> Result<OwnedRawContext, CudaError> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: output pointer is valid and the function contract is frozen by
        // the unsafe `from_symbols` constructor.
        check("cuCtxCreate_v2", unsafe {
            (self.cu_ctx_create)(&mut raw, flags, device)
        })?;
        let raw = NonNull::new(raw).ok_or(CudaError {
            operation: "cuCtxCreate_v2:null",
            code: CUDA_SUCCESS,
        })?;
        let mut popped = std::ptr::null_mut();
        // cuCtxCreate makes the new context current. Pop it immediately so a
        // safe constructor restores the caller's prior current-context stack.
        // SAFETY: output pointer is valid and the function table is trusted.
        let pop_result = unsafe { (self.cu_ctx_pop_current)(&mut popped) };
        if pop_result != CUDA_SUCCESS || popped != raw.as_ptr() {
            // SAFETY: raw came from successful creation; best-effort cleanup is
            // required even when current-stack restoration failed.
            unsafe {
                (self.cu_ctx_destroy)(raw.as_ptr());
            }
            return Err(CudaError {
                operation: if pop_result == CUDA_SUCCESS {
                    "cuCtxPopCurrent_v2:mismatch"
                } else {
                    "cuCtxPopCurrent_v2"
                },
                code: pop_result,
            });
        }
        Ok(OwnedRawContext {
            inner: Rc::new(ContextInner {
                api: Arc::clone(self),
                raw: Cell::new(Some(raw)),
            }),
        })
    }
}

struct ContextInner {
    api: Arc<DriverApi>,
    raw: Cell<Option<NonNull<c_void>>>,
}

impl ContextInner {
    fn with_current<T>(
        &self,
        operation: impl FnOnce(&DriverApi) -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        let raw = self.raw.get().ok_or(CudaError {
            operation: "cuCtxPushCurrent_v2:context-closed",
            code: CUDA_SUCCESS,
        })?;
        // SAFETY: handle is live and this bounded API is same-thread only.
        check("cuCtxPushCurrent_v2", unsafe {
            (self.api.cu_ctx_push_current)(raw.as_ptr())
        })?;
        let result = operation(&self.api);
        let mut popped = std::ptr::null_mut();
        // SAFETY: output pointer is valid and balances the successful push.
        let pop_result = unsafe { (self.api.cu_ctx_pop_current)(&mut popped) };
        if pop_result != CUDA_SUCCESS {
            return Err(CudaError {
                operation: "cuCtxPopCurrent_v2",
                code: pop_result,
            });
        }
        if popped != raw.as_ptr() {
            return Err(CudaError {
                operation: "cuCtxPopCurrent_v2:mismatch",
                code: CUDA_SUCCESS,
            });
        }
        result
    }

    fn destroy(&self) -> Result<(), CudaError> {
        let Some(raw) = self.raw.take() else {
            return Ok(());
        };
        // SAFETY: handle came from this API and ContextInner outlives every
        // provider-owned child through their Rc clones.
        check("cuCtxDestroy_v2", unsafe {
            (self.api.cu_ctx_destroy)(raw.as_ptr())
        })
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

/// Provider-owned raw CUDA context, destroyed after every child.
///
/// This Alpha is deliberately thread-affine:
///
/// ```compile_fail
/// use rextio_cuda_runtime::OwnedRawContext;
/// fn require_send<T: Send>() {}
/// require_send::<OwnedRawContext>();
/// ```
pub struct OwnedRawContext {
    inner: Rc<ContextInner>,
}

impl OwnedRawContext {
    /// Non-owning raw handle for provider-generated helper calls.
    pub fn as_raw(&self) -> CuContext {
        self.inner
            .raw
            .get()
            .map_or(std::ptr::null_mut(), NonNull::as_ptr)
    }

    /// Create a provider-owned stream tied to this context's lifetime.
    ///
    /// This is a dedicated provider stream, never a framework current stream.
    pub fn create_dedicated_stream(&self, flags: u32) -> Result<OwnedRawStream, CudaError> {
        if self.inner.raw.get().is_none() {
            return Err(CudaError {
                operation: "cuStreamCreate:context-closed",
                code: CUDA_SUCCESS,
            });
        }
        let mut raw = std::ptr::null_mut();
        self.inner.with_current(|api| {
            // SAFETY: with_current installed the owned context for this thread.
            check("cuStreamCreate", unsafe {
                (api.cu_stream_create)(&mut raw, flags)
            })
        })?;
        let raw = NonNull::new(raw).ok_or(CudaError {
            operation: "cuStreamCreate:null",
            code: CUDA_SUCCESS,
        })?;
        Ok(OwnedRawStream {
            context: Rc::clone(&self.inner),
            raw: Some(raw),
        })
    }

    /// Allocate provider-owned memory tied to this context's lifetime.
    pub fn allocate(&self, bytes: usize) -> Result<OwnedRawDeviceAllocation, CudaError> {
        if self.inner.raw.get().is_none() {
            return Err(CudaError {
                operation: "cuMemAlloc_v2:context-closed",
                code: CUDA_SUCCESS,
            });
        }
        if bytes == 0 {
            return Err(CudaError {
                operation: "cuMemAlloc_v2:zero",
                code: CUDA_SUCCESS,
            });
        }
        let mut raw = 0;
        self.inner.with_current(|api| {
            // SAFETY: with_current installed the owned context for this thread.
            check("cuMemAlloc_v2", unsafe {
                (api.cu_mem_alloc)(&mut raw, bytes)
            })
        })?;
        if raw == 0 {
            return Err(CudaError {
                operation: "cuMemAlloc_v2:null",
                code: CUDA_SUCCESS,
            });
        }
        Ok(OwnedRawDeviceAllocation {
            context: Rc::clone(&self.inner),
            raw: Some(raw),
            bytes,
        })
    }

    /// Destroy now, but fail while a stream or allocation still owns the context.
    pub fn close(&mut self) -> Result<(), CudaError> {
        if Rc::strong_count(&self.inner) != 1 {
            return Err(CudaError {
                operation: "cuCtxDestroy_v2:children-live",
                code: -1,
            });
        }
        self.inner.destroy()
    }
}

/// Provider-owned dedicated CUDA stream, never a framework current stream.
///
/// ```compile_fail
/// use rextio_cuda_runtime::OwnedRawStream;
/// fn require_send<T: Send>() {}
/// require_send::<OwnedRawStream>();
/// ```
pub struct OwnedRawStream {
    context: Rc<ContextInner>,
    raw: Option<NonNull<c_void>>,
}

impl OwnedRawStream {
    /// Non-owning raw handle for provider-generated helper calls.
    pub fn as_raw(&self) -> CuStream {
        self.raw.map_or(std::ptr::null_mut(), NonNull::as_ptr)
    }

    /// Synchronize only this provider-owned dedicated stream.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        let raw = self.raw.ok_or(CudaError {
            operation: "cuStreamSynchronize:closed",
            code: CUDA_SUCCESS,
        })?;
        self.context.with_current(|api| {
            // SAFETY: with_current installed the owning context.
            check("cuStreamSynchronize", unsafe {
                (api.cu_stream_synchronize)(raw.as_ptr())
            })
        })
    }

    /// Destroy now and report a driver error.
    pub fn close(mut self) -> Result<(), CudaError> {
        self.destroy()
    }

    fn destroy(&mut self) -> Result<(), CudaError> {
        let Some(raw) = self.raw.take() else {
            return Ok(());
        };
        self.context.with_current(|api| {
            // SAFETY: handle came from this context and is consumed once.
            check("cuStreamDestroy_v2", unsafe {
                (api.cu_stream_destroy)(raw.as_ptr())
            })
        })
    }
}

impl Drop for OwnedRawStream {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

/// Provider-owned raw CUDA allocation, freed exactly once.
///
/// ```compile_fail
/// use rextio_cuda_runtime::OwnedRawDeviceAllocation;
/// fn require_send<T: Send>() {}
/// require_send::<OwnedRawDeviceAllocation>();
/// ```
pub struct OwnedRawDeviceAllocation {
    context: Rc<ContextInner>,
    raw: Option<CuDevicePtr>,
    bytes: usize,
}

impl OwnedRawDeviceAllocation {
    /// Raw CUDA device address.
    pub fn as_raw(&self) -> CuDevicePtr {
        self.raw.unwrap_or(0)
    }

    /// Allocation size recorded at successful creation.
    pub fn len(&self) -> usize {
        self.bytes
    }

    /// Allocations are never zero length.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Free now and report a driver error.
    pub fn close(mut self) -> Result<(), CudaError> {
        self.free()
    }

    fn free(&mut self) -> Result<(), CudaError> {
        let Some(raw) = self.raw.take() else {
            return Ok(());
        };
        self.context.with_current(|api| {
            // SAFETY: address came from this context and is consumed once.
            check("cuMemFree_v2", unsafe { (api.cu_mem_free)(raw) })
        })
    }
}

impl Drop for OwnedRawDeviceAllocation {
    fn drop(&mut self) {
        let _ = self.free();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());
    static CONTEXT_DROPS: AtomicUsize = AtomicUsize::new(0);
    static STREAM_DROPS: AtomicUsize = AtomicUsize::new(0);
    static ALLOCATION_DROPS: AtomicUsize = AtomicUsize::new(0);
    static CONTEXT_PUSHES: AtomicUsize = AtomicUsize::new(0);
    static CONTEXT_POPS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "system" fn create_context(
        output: *mut CuContext,
        _flags: u32,
        _device: CuDevice,
    ) -> CuResult {
        // SAFETY: test caller supplies the valid output pointer.
        unsafe { output.write(1_usize as CuContext) };
        CUDA_SUCCESS
    }

    unsafe extern "system" fn destroy_context(_context: CuContext) -> CuResult {
        CONTEXT_DROPS.fetch_add(1, Ordering::SeqCst);
        CUDA_SUCCESS
    }

    unsafe extern "system" fn push_context(context: CuContext) -> CuResult {
        if context.is_null() {
            return 1;
        }
        CONTEXT_PUSHES.fetch_add(1, Ordering::SeqCst);
        CUDA_SUCCESS
    }

    unsafe extern "system" fn pop_context(output: *mut CuContext) -> CuResult {
        CONTEXT_POPS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: test caller supplies the valid output pointer.
        unsafe { output.write(1_usize as CuContext) };
        CUDA_SUCCESS
    }

    unsafe extern "system" fn create_stream(output: *mut CuStream, _flags: u32) -> CuResult {
        // SAFETY: test caller supplies the valid output pointer.
        unsafe { output.write(2_usize as CuStream) };
        CUDA_SUCCESS
    }

    unsafe extern "system" fn destroy_stream(_stream: CuStream) -> CuResult {
        STREAM_DROPS.fetch_add(1, Ordering::SeqCst);
        CUDA_SUCCESS
    }

    unsafe extern "system" fn synchronize_stream(_stream: CuStream) -> CuResult {
        CUDA_SUCCESS
    }

    unsafe extern "system" fn allocate(output: *mut CuDevicePtr, _bytes: usize) -> CuResult {
        // SAFETY: test caller supplies the valid output pointer.
        unsafe { output.write(0x1000) };
        CUDA_SUCCESS
    }

    unsafe extern "system" fn free(_address: CuDevicePtr) -> CuResult {
        ALLOCATION_DROPS.fetch_add(1, Ordering::SeqCst);
        CUDA_SUCCESS
    }

    fn api() -> Arc<DriverApi> {
        // SAFETY: each local fake has the declared signature and static lifetime.
        Arc::new(unsafe {
            DriverApi::from_symbols(
                create_context,
                destroy_context,
                push_context,
                pop_context,
                create_stream,
                destroy_stream,
                synchronize_stream,
                allocate,
                free,
            )
        })
    }

    #[test]
    fn owned_resources_release_exactly_once() {
        let _guard = TEST_LOCK.lock().unwrap();
        CONTEXT_DROPS.store(0, Ordering::SeqCst);
        STREAM_DROPS.store(0, Ordering::SeqCst);
        ALLOCATION_DROPS.store(0, Ordering::SeqCst);
        CONTEXT_PUSHES.store(0, Ordering::SeqCst);
        CONTEXT_POPS.store(0, Ordering::SeqCst);
        let api = api();
        let context = api.create_context(0, 0).unwrap();
        let stream = context.create_dedicated_stream(0).unwrap();
        let allocation = context.allocate(4096).unwrap();
        assert!(!context.as_raw().is_null());
        assert!(!stream.as_raw().is_null());
        assert_eq!(allocation.as_raw(), 0x1000);
        assert_eq!(allocation.len(), 4096);
        stream.synchronize().unwrap();
        drop((allocation, stream, context));
        assert_eq!(CONTEXT_DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(STREAM_DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(ALLOCATION_DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(CONTEXT_PUSHES.load(Ordering::SeqCst), 5);
        assert_eq!(CONTEXT_POPS.load(Ordering::SeqCst), 6);
    }

    #[test]
    fn explicit_close_does_not_double_release() {
        let _guard = TEST_LOCK.lock().unwrap();
        CONTEXT_DROPS.store(0, Ordering::SeqCst);
        let mut context = api().create_context(0, 0).unwrap();
        context.close().unwrap();
        drop(context);
        assert_eq!(CONTEXT_DROPS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn zero_length_allocation_fails_without_driver_call() {
        let _guard = TEST_LOCK.lock().unwrap();
        let context = api().create_context(0, 0).unwrap();
        let error = context.allocate(0).err().unwrap();
        assert_eq!(error.operation(), "cuMemAlloc_v2:zero");
    }

    #[test]
    fn child_resources_keep_context_alive_and_release_before_it() {
        let _guard = TEST_LOCK.lock().unwrap();
        CONTEXT_DROPS.store(0, Ordering::SeqCst);
        STREAM_DROPS.store(0, Ordering::SeqCst);
        ALLOCATION_DROPS.store(0, Ordering::SeqCst);
        let context = api().create_context(0, 0).unwrap();
        let stream = context.create_dedicated_stream(0).unwrap();
        let allocation = context.allocate(4096).unwrap();

        drop(context);
        assert_eq!(CONTEXT_DROPS.load(Ordering::SeqCst), 0);
        drop(allocation);
        assert_eq!(ALLOCATION_DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(CONTEXT_DROPS.load(Ordering::SeqCst), 0);
        drop(stream);
        assert_eq!(STREAM_DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(CONTEXT_DROPS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn explicit_context_close_fails_while_child_is_live() {
        let _guard = TEST_LOCK.lock().unwrap();
        CONTEXT_DROPS.store(0, Ordering::SeqCst);
        let mut context = api().create_context(0, 0).unwrap();
        let stream = context.create_dedicated_stream(0).unwrap();

        let error = context.close().unwrap_err();
        assert_eq!(error.operation(), "cuCtxDestroy_v2:children-live");
        assert_eq!(CONTEXT_DROPS.load(Ordering::SeqCst), 0);
        drop(stream);
        context.close().unwrap();
        assert_eq!(CONTEXT_DROPS.load(Ordering::SeqCst), 1);
    }
}
