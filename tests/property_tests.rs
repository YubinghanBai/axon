//! Property-based tests for Axon kernels using proptest.
//!
//! Tests mathematical invariants that must hold for ALL valid inputs,
//! not just hand-picked examples. Proptest generates random inputs
//! and shrinks failures to minimal reproducible cases.
//!
//! Run: `cargo test -p axon --all-features --test property_tests`

use proptest::prelude::*;
use serde_json::json;

use axon::kernel::KernelInput;
use axon::ComputeKernel;

// ── Helpers ──────────────────────────────────────────────────────

/// Extract f32 data from JSON KernelOutput.
fn extract_json_data(output: axon::KernelOutput) -> Vec<f64> {
    let json = output.unwrap_json();
    json["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn extract_json_shape(output: &serde_json::Value) -> Vec<usize> {
    output["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}


// ── Tensor kernel tests ──────────────────────────────────────────

#[cfg(feature = "onnx")]
mod tensor_props {
    use super::*;
    use axon::kernels::tensor::TensorKernel;

    // ── Reshape roundtrip ────────────────────────────────────────
    //
    // reshape([M, N] → [N, M]) then reshape([N, M] → [M, N]) == original data

    proptest! {
        #[test]
        fn reshape_roundtrip(
            rows in 1..20usize,
            cols in 1..20usize,
        ) {
            let total = rows * cols;
            let data: Vec<f64> = (0..total).map(|i| i as f64).collect();

            let kernel = TensorKernel;
            let input = KernelInput::from_json(json!({
                "shape": [rows, cols],
                "data": data,
            }));
            let ops = json!({"op": "reshape", "shape": [cols, rows]});
            let mid = kernel.execute(input, ops).unwrap().unwrap_json();

            // Reshape back
            let mid_data: Vec<f64> = mid["data"].as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap()).collect();
            let input2 = KernelInput::from_json(json!({
                "shape": [cols, rows],
                "data": mid_data,
            }));
            let ops2 = json!({"op": "reshape", "shape": [rows, cols]});
            let result = extract_json_data(kernel.execute(input2, ops2).unwrap());

            prop_assert_eq!(result, data, "reshape roundtrip failed");
        }

        // ── Reshape preserves element count ──────────────────────

        #[test]
        fn reshape_preserves_elements(
            rows in 1..50usize,
            cols in 1..50usize,
        ) {
            let total = rows * cols;
            let data: Vec<f64> = (0..total).map(|i| i as f64 * 0.1).collect();

            let kernel = TensorKernel;
            let input = KernelInput::from_json(json!({
                "shape": [rows, cols],
                "data": data,
            }));
            let ops = json!({"op": "reshape", "shape": [total]});
            let result = kernel.execute(input, ops).unwrap().unwrap_json();
            let out_shape = extract_json_shape(&result);

            prop_assert_eq!(out_shape, vec![total]);
        }

        // ── Softmax sums to ~1.0 ────────────────────────────────

        #[test]
        fn softmax_sums_to_one(
            n in 2..200usize,
        ) {
            let data: Vec<f64> = (0..n).map(|i| (i as f64 - n as f64 / 2.0) * 0.1).collect();

            let kernel = TensorKernel;
            let input = KernelInput::from_json(json!({
                "shape": [n],
                "data": data,
            }));
            let ops = json!({"op": "softmax", "dim": 0});
            let result = extract_json_data(kernel.execute(input, ops).unwrap());

            let sum: f64 = result.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "softmax sum = {sum}, expected ~1.0"
            );
        }

        // ── Softmax all values in [0, 1] ────────────────────────

        #[test]
        fn softmax_values_in_unit_range(
            n in 2..200usize,
        ) {
            let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 25.0).collect();

            let kernel = TensorKernel;
            let input = KernelInput::from_json(json!({
                "shape": [n],
                "data": data,
            }));
            let ops = json!({"op": "softmax", "dim": 0});
            let result = extract_json_data(kernel.execute(input, ops).unwrap());

            for (i, &v) in result.iter().enumerate() {
                prop_assert!(
                    v >= 0.0 && v <= 1.0,
                    "softmax[{i}] = {v}, not in [0, 1]"
                );
            }
        }

        // ── Normalize produces unit vectors ──────────────────────

        #[test]
        fn normalize_unit_vector(
            n in 2..500usize,
        ) {
            // Ensure non-zero vector
            let data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.01).collect();

            let kernel = TensorKernel;
            let input = KernelInput::from_json(json!({
                "shape": [n],
                "data": data,
            }));
            let ops = json!({"op": "normalize"});
            let result = extract_json_data(kernel.execute(input, ops).unwrap());

            let magnitude: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
            prop_assert!(
                (magnitude - 1.0).abs() < 1e-3,
                "normalize magnitude = {magnitude}, expected ~1.0"
            );
        }

        // ── Normalize preserves direction ────────────────────────

        #[test]
        fn normalize_preserves_direction(
            n in 2..100usize,
            scale in 0.1f64..100.0,
        ) {
            let data: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
            let scaled: Vec<f64> = data.iter().map(|&x| x * scale).collect();

            let kernel = TensorKernel;

            let r1 = extract_json_data(kernel.execute(
                KernelInput::from_json(json!({"shape": [n], "data": data})),
                json!({"op": "normalize"}),
            ).unwrap());

            let r2 = extract_json_data(kernel.execute(
                KernelInput::from_json(json!({"shape": [n], "data": scaled})),
                json!({"op": "normalize"}),
            ).unwrap());

            for i in 0..n {
                prop_assert!(
                    (r1[i] - r2[i]).abs() < 1e-4,
                    "direction diverged at [{i}]: {} vs {}",
                    r1[i], r2[i]
                );
            }
        }

        // ── Transpose involution: T(T(x)) == x ──────────────────

        #[test]
        fn transpose_involution(
            rows in 1..20usize,
            cols in 1..20usize,
        ) {
            let data: Vec<f64> = (0..rows * cols).map(|i| i as f64).collect();

            let kernel = TensorKernel;
            let input = KernelInput::from_json(json!({
                "shape": [rows, cols],
                "data": data.clone(),
            }));
            let ops = json!({"op": "transpose", "axes": [1, 0]});
            let mid = kernel.execute(input, ops).unwrap().unwrap_json();
            let mid_data: Vec<f64> = mid["data"].as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap()).collect();

            // Transpose back
            let input2 = KernelInput::from_json(json!({
                "shape": [cols, rows],
                "data": mid_data,
            }));
            let ops2 = json!({"op": "transpose", "axes": [1, 0]});
            let result = extract_json_data(kernel.execute(input2, ops2).unwrap());

            prop_assert_eq!(result, data, "transpose involution failed");
        }

        // ── Unsqueeze/squeeze roundtrip ──────────────────────────

        #[test]
        fn unsqueeze_squeeze_roundtrip(
            n in 1..100usize,
        ) {
            let data: Vec<f64> = (0..n).map(|i| i as f64).collect();

            let kernel = TensorKernel;

            // unsqueeze at dim 0: [n] → [1, n]
            let input = KernelInput::from_json(json!({
                "shape": [n],
                "data": data.clone(),
            }));
            let mid = kernel.execute(input, json!({"op": "unsqueeze", "dim": 0}))
                .unwrap().unwrap_json();
            let mid_shape = extract_json_shape(&mid);
            prop_assert_eq!(&mid_shape, &vec![1, n]);

            // squeeze back: [1, n] → [n]
            let mid_data: Vec<f64> = mid["data"].as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap()).collect();
            let input2 = KernelInput::from_json(json!({
                "shape": [1, n],
                "data": mid_data,
            }));
            let result = kernel.execute(input2, json!({"op": "squeeze", "dim": 0}))
                .unwrap().unwrap_json();
            let result_data = extract_json_data(axon::KernelOutput::Json(result.clone()));
            let result_shape = extract_json_shape(&result);

            prop_assert_eq!(result_shape, vec![n]);
            prop_assert_eq!(result_data, data);
        }

        // ── Clamp bounds ─────────────────────────────────────────
        //
        // Note: f64→f32→f64 roundtrip loses precision, so we use wider epsilon.

        #[test]
        fn clamp_respects_bounds(
            n in 1..200usize,
            min_val in -100.0f64..0.0,
            max_val in 0.0f64..100.0,
        ) {
            let data: Vec<f64> = (0..n).map(|i| i as f64 - (n as f64 / 2.0)).collect();

            let kernel = TensorKernel;
            let input = KernelInput::from_json(json!({
                "shape": [n],
                "data": data,
            }));
            let ops = json!({"op": "clamp", "min": min_val, "max": max_val});
            let result = extract_json_data(kernel.execute(input, ops).unwrap());

            // Allow f32 precision tolerance (f64 → f32 → f64 roundtrip).
            let eps = 1e-3;
            for (i, &v) in result.iter().enumerate() {
                prop_assert!(
                    v >= min_val - eps && v <= max_val + eps,
                    "clamp[{i}] = {v}, not in [{min_val}, {max_val}]"
                );
            }
        }

        // ── Reduce sum associativity: sum(shape) == sum(flatten) ─

        #[test]
        fn reduce_sum_matches_flat(
            rows in 1..30usize,
            cols in 1..30usize,
        ) {
            let data: Vec<f64> = (0..rows * cols).map(|i| i as f64 * 0.1).collect();
            let expected_sum: f64 = data.iter().sum();

            let kernel = TensorKernel;
            let input = KernelInput::from_json(json!({
                "shape": [rows, cols],
                "data": data,
            }));
            let ops = json!("reduce_sum");
            let result = extract_json_data(kernel.execute(input, ops).unwrap());

            prop_assert!(
                (result[0] - expected_sum).abs() < 1e-2,
                "reduce_sum = {}, expected ~{expected_sum}",
                result[0]
            );
        }

        // ── Mean pool: mean(data) == sum(data) / N ───────────────

        #[test]
        fn mean_pool_equals_arithmetic_mean(
            n in 2..100usize,
        ) {
            let data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.5).collect();
            let expected: f64 = data.iter().sum::<f64>() / n as f64;

            let kernel = TensorKernel;
            let input = KernelInput::from_json(json!({
                "shape": [1, n],
                "data": data,
            }));
            let ops = json!({"op": "mean_pool", "dim": 1});
            let result = kernel.execute(input, ops).unwrap().unwrap_json();
            let embedding: Vec<f64> = result["embedding"].as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap()).collect();

            prop_assert!(
                (embedding[0] - expected).abs() < 1e-4,
                "mean_pool = {}, expected ~{expected}",
                embedding[0]
            );
        }

        // ── Matmul identity: A × I == A ──────────────────────────

        #[test]
        fn matmul_identity(
            n in 2..10usize,
        ) {
            // Create identity matrix
            let mut identity = vec![0.0f64; n * n];
            for i in 0..n {
                identity[i * n + i] = 1.0;
            }
            let data: Vec<f64> = (0..n * n).map(|i| (i as f64 + 1.0) * 0.5).collect();

            let kernel = TensorKernel;
            let input = KernelInput::from_json(json!({
                "shape": [n, n],
                "data": data.clone(),
            }));
            let ops = json!({
                "op": "matmul",
                "other": {"data": identity, "shape": [n, n]},
            });
            let result = extract_json_data(kernel.execute(input, ops).unwrap());

            for i in 0..data.len() {
                prop_assert!(
                    (result[i] - data[i]).abs() < 1e-3,
                    "A×I != A at [{i}]: {} vs {}",
                    result[i], data[i]
                );
            }
        }
    }
}

// ── Detection kernel tests ───────────────────────────────────────

#[cfg(feature = "vision")]
mod detection_props {
    use super::*;
    use axon::kernels::detection::DetectionKernel;

    proptest! {
        // ── NMS output ≤ input ────────────────────────────────────

        #[test]
        fn nms_reduces_count(
            n in 2..200usize,
        ) {
            let boxes: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let x = (i * 37 % 600) as f64;
                    let y = (i * 53 % 600) as f64;
                    vec![x, y, x + 30.0, y + 30.0]
                })
                .collect();
            let scores: Vec<f64> = (0..n).map(|i| 0.1 + (i * 13 % 90) as f64 / 100.0).collect();

            let kernel = DetectionKernel;
            let input = KernelInput::from_json(json!({
                "boxes": boxes,
                "scores": scores,
            }));
            let ops = json!({"op": "nms", "iou": 0.45});
            let result = kernel.execute(input, ops).unwrap().unwrap_json();

            let out_boxes = result["boxes"].as_array().unwrap();
            prop_assert!(
                out_boxes.len() <= n,
                "NMS output ({}) > input ({n})",
                out_boxes.len()
            );
        }

        // ── Confidence filter threshold ──────────────────────────

        #[test]
        fn confidence_filter_respects_threshold(
            n in 2..200usize,
            threshold in 0.1f64..0.9,
        ) {
            let boxes: Vec<Vec<f64>> = (0..n)
                .map(|i| vec![i as f64 * 10.0, 0.0, i as f64 * 10.0 + 5.0, 5.0])
                .collect();
            let scores: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

            let kernel = DetectionKernel;
            let input = KernelInput::from_json(json!({
                "boxes": boxes,
                "scores": scores,
            }));
            let ops = json!({"op": "confidence_filter", "threshold": threshold});
            let result = kernel.execute(input, ops).unwrap().unwrap_json();

            let out_scores = result["scores"].as_array().unwrap();
            for (i, s) in out_scores.iter().enumerate() {
                let score = s.as_f64().unwrap();
                prop_assert!(
                    score >= threshold - 1e-6,
                    "score[{i}] = {score} < threshold {threshold}"
                );
            }
        }

        // ── xywh_to_xyxy roundtrip ───────────────────────────────

        #[test]
        fn box_format_roundtrip(
            n in 1..50usize,
        ) {
            let boxes: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let cx = 100.0 + i as f64 * 20.0;
                    let cy = 100.0 + i as f64 * 15.0;
                    let w = 50.0 + i as f64 * 2.0;
                    let h = 40.0 + i as f64 * 3.0;
                    vec![cx, cy, w, h]
                })
                .collect();
            let scores: Vec<f64> = (0..n).map(|i| 0.5 + i as f64 * 0.01).collect();

            let kernel = DetectionKernel;

            // xywh → xyxy
            let input1 = KernelInput::from_json(json!({
                "boxes": boxes.clone(),
                "scores": scores.clone(),
            }));
            let mid = kernel.execute(input1, json!({"op": "xywh_to_xyxy"}))
                .unwrap().unwrap_json();

            // xyxy → xywh
            let input2 = KernelInput::from_json(mid.clone());
            let result = kernel.execute(input2, json!({"op": "xyxy_to_xywh"}))
                .unwrap().unwrap_json();

            let result_boxes = result["boxes"].as_array().unwrap();
            for (i, (orig, out)) in boxes.iter().zip(result_boxes.iter()).enumerate() {
                let out_box: Vec<f64> = out.as_array().unwrap()
                    .iter().map(|v| v.as_f64().unwrap()).collect();
                for j in 0..4 {
                    prop_assert!(
                        (orig[j] - out_box[j]).abs() < 1e-4,
                        "box[{i}][{j}]: {} vs {}",
                        orig[j], out_box[j]
                    );
                }
            }
        }

        // ── NMS idempotent: NMS(NMS(x)) == NMS(x) ───────────────

        #[test]
        fn nms_idempotent(
            n in 5..100usize,
        ) {
            let boxes: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let x = (i * 37 % 300) as f64;
                    let y = (i * 53 % 300) as f64;
                    vec![x, y, x + 40.0, y + 40.0]
                })
                .collect();
            let scores: Vec<f64> = (0..n).map(|i| 0.1 + (i * 13 % 90) as f64 / 100.0).collect();

            let kernel = DetectionKernel;
            let ops = json!({"op": "nms", "iou": 0.45});

            // First NMS
            let input1 = KernelInput::from_json(json!({"boxes": boxes, "scores": scores}));
            let r1 = kernel.execute(input1, ops.clone()).unwrap().unwrap_json();
            let count1 = r1["boxes"].as_array().unwrap().len();

            // Second NMS on result
            let input2 = KernelInput::from_json(r1);
            let r2 = kernel.execute(input2, ops).unwrap().unwrap_json();
            let count2 = r2["boxes"].as_array().unwrap().len();

            prop_assert_eq!(count1, count2, "NMS not idempotent: {} vs {}", count1, count2);
        }
    }
}

// ── Image kernel tests ───────────────────────────────────────────

#[cfg(feature = "vision")]
mod image_props {
    use super::*;
    use axon::kernels::image::ImageKernel;

    fn make_image_blob(h: usize, w: usize, c: usize) -> KernelInput {
        let pixels: Vec<f32> = (0..h * w * c).map(|i| (i % 256) as f32).collect();
        let bytes: Vec<u8> = pixels.iter().flat_map(|f| f.to_le_bytes()).collect();
        KernelInput {
            json: serde_json::Value::Null,
            blobs: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "image".to_string(),
                    axon::BlobData {
                        bytes,
                        meta: axon::BlobMeta {
                            size: 0,
                            content_type: "tensor/f32".to_string(),
                            shape: Some(vec![h, w, c]),
                        },
                    },
                );
                m
            },
        }
    }

    fn extract_image_blob(output: axon::KernelOutput) -> (Vec<f32>, Vec<usize>) {
        match output {
            axon::KernelOutput::Blob { data, shape, .. } => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                (floats, shape.unwrap())
            }
            _ => panic!("expected Blob output"),
        }
    }

    proptest! {
        // ── Layout HWC→CHW produces correct shape and element count ──

        #[test]
        fn layout_hwc_to_chw_preserves_data(
            h in 1..30usize,
            w in 1..30usize,
        ) {
            let c = 3;
            let input = make_image_blob(h, w, c);

            let kernel = ImageKernel;
            let result = kernel.execute(input.clone(), json!({"op": "layout", "to": "chw"})).unwrap();
            let (chw_data, chw_shape) = extract_image_blob(result);

            // Shape should be [C, H, W]
            prop_assert_eq!(&chw_shape, &vec![c, h, w], "CHW shape wrong");

            // Element count preserved
            prop_assert_eq!(chw_data.len(), h * w * c);

            // Verify the scatter: chw[ch][y][x] == hwc[y*W*C + x*C + ch]
            let orig_blob = input.first_blob().unwrap();
            let hwc: Vec<f32> = orig_blob.bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let hw = h * w;
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let hwc_idx = (y * w + x) * c + ch;
                        let chw_idx = ch * hw + y * w + x;
                        prop_assert!(
                            (hwc[hwc_idx] - chw_data[chw_idx]).abs() < 1e-6,
                            "mismatch at ch={ch}, y={y}, x={x}: hwc={} chw={}",
                            hwc[hwc_idx], chw_data[chw_idx]
                        );
                    }
                }
            }
        }

        // ── Layout preserves element count for any channel count ─

        #[test]
        fn layout_preserves_element_count(
            h in 1..50usize,
            w in 1..50usize,
            c in 1..5usize,
        ) {
            let input = make_image_blob(h, w, c);
            let kernel = ImageKernel;

            let result = kernel.execute(input, json!({"op": "layout", "to": "chw"})).unwrap();
            let (data, shape) = extract_image_blob(result);

            prop_assert_eq!(data.len(), h * w * c);
            prop_assert_eq!(shape.iter().product::<usize>(), h * w * c);
        }

        // ── BGR conversion swaps R and B channels ────────────────

        #[test]
        fn colorspace_bgr_swaps_channels(
            h in 1..20usize,
            w in 1..20usize,
        ) {
            let c = 3;
            let input = make_image_blob(h, w, c);
            let kernel = ImageKernel;

            let result = kernel.execute(input.clone(), json!({"op": "colorspace", "to": "bgr"})).unwrap();
            let (bgr_data, _) = extract_image_blob(result);

            let orig_blob = input.first_blob().unwrap();
            let rgb: Vec<f32> = orig_blob.bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // Check that R and B are swapped, G is unchanged
            for pixel_idx in 0..(h * w) {
                let base = pixel_idx * 3;
                prop_assert!(
                    (rgb[base] - bgr_data[base + 2]).abs() < 1e-6,
                    "R→B swap failed at pixel {pixel_idx}"
                );
                prop_assert!(
                    (rgb[base + 1] - bgr_data[base + 1]).abs() < 1e-6,
                    "G changed at pixel {pixel_idx}"
                );
                prop_assert!(
                    (rgb[base + 2] - bgr_data[base]).abs() < 1e-6,
                    "B→R swap failed at pixel {pixel_idx}"
                );
            }
        }

        // ── Grayscale reduces to 1 channel ───────────────────────

        #[test]
        fn grayscale_reduces_channels(
            h in 1..20usize,
            w in 1..20usize,
        ) {
            let input = make_image_blob(h, w, 3);
            let kernel = ImageKernel;

            let result = kernel.execute(input, json!({"op": "colorspace", "to": "grayscale"})).unwrap();
            let (gray_data, gray_shape) = extract_image_blob(result);

            prop_assert_eq!(&gray_shape, &vec![h, w, 1]);
            prop_assert_eq!(gray_data.len(), h * w);

            // All values should be non-negative (weighted sum of RGB)
            for (i, &v) in gray_data.iter().enumerate() {
                prop_assert!(v >= 0.0, "gray[{i}] = {v} is negative");
            }
        }
    }
}
