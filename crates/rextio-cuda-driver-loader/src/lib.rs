//! Reviewed, PATH-free CUDA driver-image selection shared by manual tools.

#![forbid(unsafe_op_in_unsafe_fn)]

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

/// Path-free stable loader failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadError(&'static str);

impl LoadError {
    /// Stable public reason code.
    pub fn reason_code(self) -> &'static str {
        self.0
    }
}

/// One provenance-checked CUDA driver image.
///
/// The handle is deliberately thread-affine in this bounded Alpha.
pub struct DriverLibrary {
    #[cfg(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    handle: NonNull<c_void>,
    _thread_affine: PhantomData<Rc<()>>,
}

impl DriverLibrary {
    /// Load the only reviewed driver image for this target.
    pub fn load() -> Result<Self, LoadError> {
        platform::load()
    }

    /// Resolve one fixed, null-terminated ASCII symbol name.
    pub fn symbol(&self, name: &'static [u8]) -> Result<NonNull<c_void>, LoadError> {
        if name.len() < 2
            || name.last() != Some(&0)
            || name[..name.len() - 1].contains(&0)
            || !name[..name.len() - 1].is_ascii()
        {
            return Err(LoadError("INVALID_SYMBOL_NAME"));
        }
        platform::symbol(self, name)
    }
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
mod platform {
    use super::{DriverLibrary, LoadError};
    use std::ffi::c_void;
    use std::marker::PhantomData;
    use std::ptr::NonNull;

    type HModule = *mut c_void;
    const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x00000800;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> HModule;
        fn GetProcAddress(module: HModule, name: *const u8) -> *mut c_void;
        fn FreeLibrary(module: HModule) -> i32;
    }

    pub(super) fn load() -> Result<DriverLibrary, LoadError> {
        let mut name: Vec<u16> = "nvcuda.dll".encode_utf16().collect();
        name.push(0);
        // SAFETY: name is null-terminated; System32-only selection excludes
        // current-directory and PATH DLL lookup.
        let handle = unsafe {
            LoadLibraryExW(
                name.as_ptr(),
                std::ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        };
        Ok(DriverLibrary {
            handle: NonNull::new(handle).ok_or(LoadError("NVCUDA_DLL_NOT_FOUND"))?,
            _thread_affine: PhantomData,
        })
    }

    pub(super) fn symbol(
        library: &DriverLibrary,
        name: &'static [u8],
    ) -> Result<NonNull<c_void>, LoadError> {
        // SAFETY: caller validated the static null-terminated name and handle.
        let pointer = unsafe { GetProcAddress(library.handle.as_ptr(), name.as_ptr()) };
        NonNull::new(pointer).ok_or(LoadError("REQUIRED_SYMBOL_MISSING"))
    }

    pub(super) fn close(library: &mut DriverLibrary) {
        // SAFETY: owned successful LoadLibraryExW handle is released once.
        unsafe {
            FreeLibrary(library.handle.as_ptr());
        }
    }
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod platform {
    use super::{DriverLibrary, LoadError};
    use std::ffi::{c_char, c_void, CString};
    use std::fs;
    use std::marker::PhantomData;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Component, Path, PathBuf};
    use std::ptr::NonNull;

    #[cfg(target_arch = "x86_64")]
    const CANDIDATES: &[&str] = &[
        "/usr/lib/wsl/lib/libcuda.so.1",
        "/usr/local/nvidia/lib64/libcuda.so.1",
        "/usr/local/nvidia/lib/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib/libcuda.so.1",
    ];
    #[cfg(target_arch = "aarch64")]
    const CANDIDATES: &[&str] = &[
        "/usr/local/nvidia/lib64/libcuda.so.1",
        "/usr/local/nvidia/lib/libcuda.so.1",
        "/usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1",
        "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib/libcuda.so.1",
    ];
    const ROOTS: &[&str] = &[
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/nvidia/lib",
        "/usr/local/nvidia/lib64",
    ];
    const RTLD_NOW: i32 = 0x0002;
    const RTLD_LOCAL: i32 = 0x0000;

    #[link(name = "dl")]
    extern "C" {
        fn dlopen(name: *const c_char, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> i32;
    }

    pub(super) fn load() -> Result<DriverLibrary, LoadError> {
        let mut found = false;
        for candidate in CANDIDATES {
            let path = Path::new(candidate);
            if path.symlink_metadata().is_err() {
                continue;
            }
            found = true;
            let Ok(canonical) = fs::canonicalize(path) else {
                continue;
            };
            if !acceptable(&canonical) {
                continue;
            }
            let Ok(name) = CString::new(canonical.to_string_lossy().as_bytes()) else {
                continue;
            };
            // SAFETY: reviewed absolute canonical path and eager local flags.
            let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
            if let Some(handle) = NonNull::new(handle) {
                return Ok(DriverLibrary {
                    handle,
                    _thread_affine: PhantomData,
                });
            }
        }
        Err(LoadError(if found {
            "LIBCUDA_SO_LOAD_FAILED"
        } else {
            "LIBCUDA_SO_NOT_FOUND"
        }))
    }

    pub(super) fn symbol(
        library: &DriverLibrary,
        name: &'static [u8],
    ) -> Result<NonNull<c_void>, LoadError> {
        // SAFETY: caller validated name; handle remains live.
        let pointer = unsafe { dlsym(library.handle.as_ptr(), name.as_ptr().cast()) };
        NonNull::new(pointer).ok_or(LoadError("REQUIRED_SYMBOL_MISSING"))
    }

    pub(super) fn close(library: &mut DriverLibrary) {
        // SAFETY: owned successful dlopen handle is released once.
        unsafe {
            dlclose(library.handle.as_ptr());
        }
    }

    fn acceptable(path: &Path) -> bool {
        let Some(text) = path.to_str() else {
            return false;
        };
        if !path.is_absolute()
            || !ROOTS
                .iter()
                .any(|root| text == *root || text.starts_with(&format!("{root}/")))
        {
            return false;
        }
        let Ok(metadata) = fs::metadata(path) else {
            return false;
        };
        if !metadata.is_file() || metadata.permissions().mode() & 0o022 != 0 {
            return false;
        }
        let mut current = PathBuf::new();
        for component in path.components() {
            match component {
                Component::RootDir => current.push(Component::RootDir.as_os_str()),
                Component::Normal(part) => {
                    current.push(part);
                    let Ok(metadata) = fs::metadata(&current) else {
                        return false;
                    };
                    if metadata.is_dir() && metadata.permissions().mode() & 0o022 != 0 {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        true
    }

    #[cfg(test)]
    mod tests {
        use super::acceptable;
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        #[test]
        fn mutable_temporary_image_is_rejected() {
            let path =
                std::env::temp_dir().join(format!("rextio-cuda-loader-{}", std::process::id()));
            fs::write(&path, b"fixture").unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o666)).unwrap();
            assert!(!acceptable(&path));
            fs::remove_file(path).unwrap();
        }
    }
}

#[cfg(not(any(
    all(target_os = "windows", target_arch = "x86_64"),
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
)))]
mod platform {
    use super::{DriverLibrary, LoadError};
    use std::ffi::c_void;
    use std::ptr::NonNull;

    pub(super) fn load() -> Result<DriverLibrary, LoadError> {
        Err(LoadError("UNSUPPORTED_TARGET"))
    }

    pub(super) fn symbol(
        _library: &DriverLibrary,
        _name: &'static [u8],
    ) -> Result<NonNull<c_void>, LoadError> {
        Err(LoadError("UNSUPPORTED_TARGET"))
    }

    pub(super) fn close(_library: &mut DriverLibrary) {}
}

impl Drop for DriverLibrary {
    fn drop(&mut self) {
        platform::close(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_symbol_names_fail_without_loader_access() {
        let library = DriverLibrary {
            #[cfg(any(
                all(target_os = "windows", target_arch = "x86_64"),
                all(
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            handle: NonNull::dangling(),
            _thread_affine: PhantomData,
        };
        assert_eq!(
            library.symbol(b"cuInit").unwrap_err().reason_code(),
            "INVALID_SYMBOL_NAME"
        );
        std::mem::forget(library);
    }
}
