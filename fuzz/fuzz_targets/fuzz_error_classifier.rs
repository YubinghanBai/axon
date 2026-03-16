//! Fuzz target for the error classification system.
//!
//! Tests that `classify_string_error` never panics on arbitrary input
//! and always produces a valid ErrorKind.
//!
//! Run: `cargo +nightly fuzz run fuzz_error_classifier -j4 -- -max_len=1024`

#![no_main]

use libfuzzer_sys::fuzz_target;
use axon::AxonError;

fuzz_target!(|data: &[u8]| {
    // Convert arbitrary bytes to string (lossy).
    let msg = String::from_utf8_lossy(data).to_string();

    // From<String> triggers classify_string_error.
    let err: AxonError = msg.into();

    // Must always produce valid error code 1-9.
    let code = err.code();
    assert!(code >= 1 && code <= 9, "invalid error code: {code}");

    // Display must not panic.
    let _ = err.to_string();

    // contains() must not panic.
    let _ = err.contains("test");

    // kind() must not panic.
    let _ = err.kind();
});
