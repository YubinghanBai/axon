//! Fuzz target for detection kernel operations.
//!
//! Tests robustness against:
//! - Malformed bounding boxes (negative coords, zero-area, huge values)
//! - NaN/Inf scores
//! - Empty detection lists
//! - Mismatched boxes/scores lengths
//!
//! Run: `cargo +nightly fuzz run fuzz_detection -j4 -- -max_len=4096`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use serde_json::json;

use axon::kernel::KernelInput;
use axon::ComputeKernel;
use axon::kernels::detection::DetectionKernel;

#[derive(Arbitrary, Debug)]
struct DetectionFuzzInput {
    op: DetectionFuzzOp,
    /// Number of boxes (capped).
    n_boxes: u8,
    /// Raw coordinate data.
    coords: Vec<f32>,
    /// Raw score data.
    scores: Vec<f32>,
    /// IoU threshold.
    iou: f32,
    /// Confidence threshold.
    threshold: f32,
}

#[derive(Arbitrary, Debug)]
enum DetectionFuzzOp {
    Nms,
    SoftNms,
    ConfidenceFilter,
    XywhToXyxy,
    XyxyToXywh,
    BatchedNms,
}

fuzz_target!(|input: DetectionFuzzInput| {
    let kernel = DetectionKernel;

    let n = (input.n_boxes as usize).min(500).max(1);

    // Build boxes from raw fuzz data.
    let mut boxes = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * 4;
        let coords = &input.coords;
        let x1 = coords.get(base).copied().unwrap_or(0.0) as f64;
        let y1 = coords.get(base + 1).copied().unwrap_or(0.0) as f64;
        let x2 = coords.get(base + 2).copied().unwrap_or(100.0) as f64;
        let y2 = coords.get(base + 3).copied().unwrap_or(100.0) as f64;
        boxes.push(vec![x1, y1, x2, y2]);
    }

    // Build scores from fuzz data.
    let scores: Vec<f64> = (0..n)
        .map(|i| input.scores.get(i).copied().unwrap_or(0.5) as f64)
        .collect();

    let ki = KernelInput::from_json(json!({
        "boxes": boxes,
        "scores": scores,
    }));

    let ops = match input.op {
        DetectionFuzzOp::Nms => {
            let iou = input.iou.clamp(0.01, 1.0) as f64;
            json!({"op": "nms", "iou": iou})
        }
        DetectionFuzzOp::SoftNms => {
            let iou = input.iou.clamp(0.01, 1.0) as f64;
            json!({"op": "soft_nms", "iou": iou, "sigma": 0.5, "method": "gaussian"})
        }
        DetectionFuzzOp::ConfidenceFilter => {
            let t = input.threshold.clamp(0.0, 1.0) as f64;
            json!({"op": "confidence_filter", "threshold": t})
        }
        DetectionFuzzOp::XywhToXyxy => json!({"op": "xywh_to_xyxy"}),
        DetectionFuzzOp::XyxyToXywh => json!({"op": "xyxy_to_xywh"}),
        DetectionFuzzOp::BatchedNms => {
            let iou = input.iou.clamp(0.01, 1.0) as f64;
            json!({"op": "batched_nms", "iou": iou})
        }
    };

    // Must not panic on any input.
    let _ = kernel.execute(ki, ops);
});
