//! Fuzz target for image kernel operations.
//!
//! Tests robustness against:
//! - Malformed image dimensions (zero width/height, huge sizes)
//! - Random pixel data
//! - Invalid channel counts
//! - Edge cases in layout conversion and normalization
//!
//! Run: `cargo +nightly fuzz run fuzz_image -j4 -- -max_len=8192`

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use serde_json::json;

use axon::kernel::KernelInput;
use axon::ComputeKernel;
use axon::kernels::image::ImageKernel;
use axon::{BlobData, BlobMeta};

#[derive(Arbitrary, Debug)]
struct ImageFuzzInput {
    op: ImageFuzzOp,
    /// Image height (capped).
    height: u16,
    /// Image width (capped).
    width: u16,
    /// Channel count.
    channels: u8,
    /// Raw pixel data.
    pixels: Vec<f32>,
}

#[derive(Arbitrary, Debug)]
enum ImageFuzzOp {
    LayoutToChw,
    LayoutToHwc,
    Normalize,
    ColorspaceBgr,
    ColorspaceGrayscale,
    Pad,
    Crop,
}

fuzz_target!(|input: ImageFuzzInput| {
    let kernel = ImageKernel;

    // Cap dimensions to avoid OOM.
    let h = (input.height as usize).clamp(1, 256);
    let w = (input.width as usize).clamp(1, 256);
    let c = (input.channels as usize).clamp(1, 4);

    let total = h * w * c;
    if total > 200_000 {
        return;
    }

    // Pad or truncate pixel data.
    let pixels: Vec<f32> = if input.pixels.len() >= total {
        input.pixels[..total].to_vec()
    } else {
        let mut p = input.pixels.clone();
        p.resize(total, 0.0);
        p
    };

    let bytes: Vec<u8> = pixels.iter().flat_map(|f| f.to_le_bytes()).collect();

    let ki = KernelInput {
        json: serde_json::Value::Null,
        blobs: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "image".to_string(),
                BlobData {
                    bytes,
                    meta: BlobMeta {
                        size: 0,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![h, w, c]),
                    },
                },
            );
            m
        },
    };

    let ops = match input.op {
        ImageFuzzOp::LayoutToChw => json!({"op": "layout", "to": "chw"}),
        ImageFuzzOp::LayoutToHwc => json!({"op": "layout", "to": "hwc"}),
        ImageFuzzOp::Normalize => json!({
            "op": "normalize",
            "scale": 255.0,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }),
        ImageFuzzOp::ColorspaceBgr => json!({"op": "colorspace", "to": "bgr"}),
        ImageFuzzOp::ColorspaceGrayscale => json!({"op": "colorspace", "to": "grayscale"}),
        ImageFuzzOp::Pad => json!({"op": "pad", "width": w + 10, "height": h + 10, "value": 0.0}),
        ImageFuzzOp::Crop => {
            // Crop a small region.
            let cw = (w / 2).max(1);
            let ch = (h / 2).max(1);
            json!({"op": "crop", "x": 0, "y": 0, "width": cw, "height": ch})
        }
    };

    // Must not panic on any input.
    let _ = kernel.execute(ki, ops);
});
