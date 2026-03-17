//! # Axon Plugin SDK
//!
//! Write WASM plugins for Axon ML pipelines in pure Rust.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use axon_plugin::{axon_plugin, Value};
//!
//! fn label_map(input: Value) -> Result<Value, String> {
//!     let mut output = input;
//!     // Map class IDs to human-readable labels.
//!     if let Some(class_id) = output.get("class_id").and_then(|v| v.as_u64()) {
//!         let label = match class_id {
//!             0 => "person",
//!             1 => "bicycle",
//!             2 => "car",
//!             _ => "unknown",
//!         };
//!         output["label"] = Value::String(label.to_string());
//!     }
//!     Ok(output)
//! }
//!
//! axon_plugin!(label_map);
//! ```
//!
//! ## Build
//!
//! ```bash
//! cargo build --target wasm32-unknown-unknown --release
//! ```
//!
//! ## Use in pipeline
//!
//! ```toml
//! [post]
//! steps = [
//!   { op = "wasm.run", module = "target/wasm32-unknown-unknown/release/my_plugin.wasm" },
//! ]
//! ```
//!
//! ## ABI
//!
//! The macro generates three WASM exports that Axon's WasmKernel calls:
//!
//! - `alloc(size: i32) -> i32` — allocate memory for input
//! - `dealloc(ptr: i32, size: i32)` — free memory (called after transform completes)
//! - `transform(ptr: i32, len: i32) -> i64` — process JSON input, return packed (ptr, len)
//!
//! You never need to touch these — the macro handles everything.

/// Re-export for convenience in plugin code.
pub use serde_json;
pub use serde_json::Value;

/// Generate the WASM plugin entry point.
///
/// Wraps your transform function with the required ABI boilerplate:
/// memory allocation, JSON serialization, and the `transform` export.
///
/// Your function must have signature:
/// `fn(serde_json::Value) -> Result<serde_json::Value, String>`
///
/// # Example
///
/// ```rust,ignore
/// use axon_plugin::{axon_plugin, Value};
///
/// fn my_transform(input: Value) -> Result<Value, String> {
///     let mut out = input;
///     out["processed"] = true.into();
///     Ok(out)
/// }
///
/// axon_plugin!(my_transform);
/// ```
#[macro_export]
macro_rules! axon_plugin {
    ($transform_fn:ident) => {
        /// Allocate `size` bytes in WASM linear memory.
        #[no_mangle]
        pub extern "C" fn alloc(size: i32) -> i32 {
            let layout = core::alloc::Layout::from_size_align(size as usize, 1).unwrap();
            unsafe { std::alloc::alloc(layout) as i32 }
        }

        /// Free `size` bytes at `ptr` in WASM linear memory.
        #[no_mangle]
        pub extern "C" fn dealloc(ptr: i32, size: i32) {
            let layout = core::alloc::Layout::from_size_align(size as usize, 1).unwrap();
            unsafe { std::alloc::dealloc(ptr as *mut u8, layout) }
        }

        /// Process JSON input and return JSON output.
        ///
        /// Axon writes input JSON bytes at `ptr..ptr+len`.
        /// Returns packed `(output_ptr << 32) | output_len`.
        #[no_mangle]
        pub extern "C" fn transform(ptr: i32, len: i32) -> i64 {
            // Read input bytes from WASM memory.
            let input_bytes =
                unsafe { core::slice::from_raw_parts(ptr as *const u8, len as usize) };

            // Deallocate input memory now that we've read it.
            dealloc(ptr, len);

            // Parse JSON input.
            let input: $crate::serde_json::Value = match $crate::serde_json::from_slice(input_bytes)
            {
                Ok(v) => v,
                Err(e) => {
                    let err = $crate::serde_json::json!({"error": format!("parse error: {e}")});
                    let bytes = $crate::serde_json::to_vec(&err).unwrap();
                    return _axon_write_output(&bytes);
                }
            };

            // Call user transform function.
            let output = match $transform_fn(input) {
                Ok(v) => v,
                Err(e) => $crate::serde_json::json!({"error": e}),
            };

            // Serialize output and write to WASM memory.
            let output_bytes = $crate::serde_json::to_vec(&output).unwrap();
            _axon_write_output(&output_bytes)
        }

        /// Write output bytes to WASM memory and return packed (ptr, len).
        #[doc(hidden)]
        fn _axon_write_output(bytes: &[u8]) -> i64 {
            let out_ptr = alloc(bytes.len() as i32);
            unsafe {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), out_ptr as *mut u8, bytes.len());
            }
            ((out_ptr as i64) << 32) | (bytes.len() as i64)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // We can't test the WASM exports in native tests, but we can
    // test that the macro compiles and the user function type-checks.

    fn example_transform(input: Value) -> Result<Value, String> {
        let mut out = input;
        out["processed"] = Value::Bool(true);
        Ok(out)
    }

    #[test]
    fn test_transform_fn_signature() {
        let input = serde_json::json!({"text": "hello"});
        let result = example_transform(input).unwrap();
        assert_eq!(result["processed"], true);
        assert_eq!(result["text"], "hello");
    }

    #[test]
    fn test_transform_error() {
        fn failing(_input: Value) -> Result<Value, String> {
            Err("something went wrong".to_string())
        }

        let result = failing(serde_json::json!({}));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "something went wrong");
    }
}
