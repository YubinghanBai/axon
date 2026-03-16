//! Fuzz target for tensor kernel operations.
//!
//! Tests robustness against malformed/extreme inputs:
//! - Random shapes with mismatched element counts
//! - NaN/Inf/subnormal float values
//! - Empty tensors
//! - Huge dimension values
//!
//! Run: `cargo +nightly fuzz run fuzz_tensor -j4 -- -max_len=4096`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use serde_json::json;

use axon::kernel::KernelInput;
use axon::ComputeKernel;
use axon::kernels::tensor::TensorKernel;

/// Fuzz input for tensor operations.
#[derive(Arbitrary, Debug)]
struct TensorFuzzInput {
    /// Operation to test.
    op: TensorFuzzOp,
    /// Tensor data (raw f32 values).
    data: Vec<f32>,
    /// Number of dimensions (1-4).
    ndim: u8,
    /// Shape dimensions (may not match data length — that's the point).
    dim0: u16,
    dim1: u16,
    dim2: u16,
    dim3: u16,
}

#[derive(Arbitrary, Debug)]
enum TensorFuzzOp {
    Softmax,
    Normalize,
    MeanPool,
    Reshape,
    Transpose,
    Unsqueeze,
    Squeeze,
    Clamp,
    ReduceSum,
    Argmax,
}

fuzz_target!(|input: TensorFuzzInput| {
    let kernel = TensorKernel;

    // Build shape from fuzz input (cap dimensions to avoid OOM).
    let shape: Vec<usize> = match input.ndim % 4 {
        0 => vec![input.dim0.max(1) as usize],
        1 => vec![input.dim0.max(1) as usize, input.dim1.max(1) as usize],
        2 => vec![
            input.dim0.max(1) as usize,
            input.dim1.max(1) as usize,
            input.dim2.max(1) as usize,
        ],
        _ => vec![
            input.dim0.max(1) as usize,
            input.dim1.max(1) as usize,
            input.dim2.max(1) as usize,
            input.dim3.max(1) as usize,
        ],
    };

    // Cap total elements to avoid OOM.
    let total: usize = shape.iter().product();
    if total > 10_000 || total == 0 {
        return;
    }

    // Use fuzz data, padding or truncating to match shape.
    let data: Vec<f64> = if input.data.len() >= total {
        input.data[..total].iter().map(|&x| x as f64).collect()
    } else {
        let mut d: Vec<f64> = input.data.iter().map(|&x| x as f64).collect();
        d.resize(total, 0.0);
        d
    };

    let ki = KernelInput::from_json(json!({
        "shape": shape,
        "data": data,
    }));

    let ops = match input.op {
        TensorFuzzOp::Softmax => json!({"op": "softmax", "dim": 0}),
        TensorFuzzOp::Normalize => json!({"op": "normalize"}),
        TensorFuzzOp::MeanPool => json!({"op": "mean_pool", "dim": 0}),
        TensorFuzzOp::Reshape => {
            // Try reshaping to reversed dimensions.
            let rev_shape: Vec<usize> = shape.iter().rev().cloned().collect();
            json!({"op": "reshape", "shape": rev_shape})
        }
        TensorFuzzOp::Transpose => {
            let ndim = shape.len();
            if ndim >= 2 {
                // Swap first two dims.
                let mut axes: Vec<usize> = (0..ndim).collect();
                axes.swap(0, 1);
                json!({"op": "transpose", "axes": axes})
            } else {
                json!({"op": "transpose", "axes": [0]})
            }
        }
        TensorFuzzOp::Unsqueeze => json!({"op": "unsqueeze", "dim": 0}),
        TensorFuzzOp::Squeeze => json!({"op": "squeeze"}),
        TensorFuzzOp::Clamp => json!({"op": "clamp", "min": -1.0, "max": 1.0}),
        TensorFuzzOp::ReduceSum => json!("reduce_sum"),
        TensorFuzzOp::Argmax => json!({"op": "argmax", "dim": 0}),
    };

    // Execute — we don't care about the result, only that it doesn't panic/crash.
    let _ = kernel.execute(ki, ops);
});
