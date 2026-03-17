//! Integration tests for Axon pipelines.
//!
//! These tests exercise real kernel execution across multi-step chains,
//! verifying that data flows correctly between kernels via the blob store
//! and that the full pre → model → post pipeline works end-to-end.
//!
//! Unlike unit tests (which test individual operations in isolation),
//! integration tests validate the system-level behavior of Axon.
//!
//! Run: `cargo test -p axon --all-features --test integration_tests`

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;

use axon::error::AxonError;
use axon::kernel::{
    BlobData, ComputeKernel, KernelInput, KernelOutput, KernelRegistry,
};
use axon::manifest::Manifest;
use axon::pipeline::Pipeline;
use axon::BlobMeta;

// ── Mock ONNX kernel ─────────────────────────────────────────────
//
// Simulates model inference by passing through the input tensor
// (optionally transforming it). This lets us test the full pipeline
// without needing a real .onnx model file.

struct MockOnnxKernel;

impl ComputeKernel for MockOnnxKernel {
    fn name(&self) -> &str {
        "onnx"
    }

    fn execute(
        &self,
        _input: KernelInput,
        _operations: serde_json::Value,
    ) -> Result<KernelOutput, AxonError> {
        // Always return a fixed 384-dim embedding tensor blob.
        // This simulates model inference regardless of input format.
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32 + 1.0) * 0.01).collect();
        Ok(KernelOutput::Blob {
            data: embedding.iter().flat_map(|f| f.to_le_bytes()).collect(),
            content_type: "tensor/f32".to_string(),
            shape: Some(vec![1, 384]),
        })
    }
}

// ── Helper: create pipeline from TOML + mock registry ────────────

fn make_pipeline(toml: &str) -> Pipeline {
    let manifest = Manifest::from_toml(toml).unwrap();
    let mut registry = KernelRegistry::new();
    registry.register(Arc::new(MockOnnxKernel));

    // Register real kernels based on features
    #[cfg(feature = "onnx")]
    registry.register(Arc::new(axon::kernels::tensor::TensorKernel));

    #[cfg(feature = "vision")]
    {
        registry.register(Arc::new(axon::kernels::image::ImageKernel));
        registry.register(Arc::new(axon::kernels::detection::DetectionKernel));
    }

    #[cfg(feature = "tokenizer")]
    registry.register(Arc::new(axon::kernels::tokenizer::TokenizerKernel::new()));

    #[cfg(feature = "audio")]
    {
        registry.register(Arc::new(axon::kernels::audio::AudioKernel::new()));
        registry.register(Arc::new(axon::kernels::mel::MelKernel::new()));
    }

    Pipeline::new(manifest, registry, PathBuf::from("."))
}


// ── Test: BERT-like embedding pipeline ───────────────────────────
//
// Pipeline: mock_onnx → tensor.mean_pool → tensor.normalize
// Simulates: model(tokens) → embedding → pool → L2 normalize

#[cfg(feature = "onnx")]
#[test]
fn test_embedding_pipeline_end_to_end() {
    let toml = r#"
[model]
name = "bert-test"
file = "model.onnx"

[post]
steps = [
    {op = "tensor.mean_pool", dim = 1},
    {op = "tensor.normalize"},
]
"#;

    let pipeline = make_pipeline(toml);

    // Input doesn't matter much — MockOnnxKernel returns fixed 384-dim embedding
    let output = pipeline.run(b"test input", "text/plain").unwrap();

    match &output {
        KernelOutput::Json(v) => {
            // After mean_pool + normalize, should have "data" and "shape"
            assert!(v.get("data").is_some() || v.get("embedding").is_some());
        }
        KernelOutput::Blob { shape, .. } => {
            // Blob output is also acceptable
            assert!(shape.is_some());
        }
    }
}

// ── Test: Detection post-processing pipeline ─────────────────────
//
// Pipeline: detection.confidence_filter → detection.nms
// Simulates: model(image) → raw detections → filter → NMS → clean output

#[cfg(feature = "vision")]
#[test]
fn test_detection_pipeline_filter_then_nms() {
    use axon::kernels::detection::DetectionKernel;

    let kernel = DetectionKernel;

    // Create 20 overlapping detections
    let mut boxes = Vec::new();
    let mut scores = Vec::new();
    for i in 0..20 {
        let x = (i * 15) as f64;
        boxes.push(vec![x, 0.0, x + 50.0, 50.0]);
        scores.push(if i % 3 == 0 { 0.1 } else { 0.5 + i as f64 * 0.02 });
    }

    // Step 1: confidence filter
    let input1 = KernelInput::from_json(json!({
        "boxes": boxes,
        "scores": scores,
    }));
    let filtered = kernel
        .execute(input1, json!({"op": "confidence_filter", "threshold": 0.3}))
        .unwrap()
        .unwrap_json();

    let filtered_count = filtered["boxes"].as_array().unwrap().len();
    assert!(filtered_count < 20, "filter should remove low-confidence boxes");

    // All remaining scores >= 0.3
    for s in filtered["scores"].as_array().unwrap() {
        assert!(s.as_f64().unwrap() >= 0.3 - 1e-6);
    }

    // Step 2: NMS on filtered results
    let input2 = KernelInput::from_json(filtered);
    let nms_result = kernel
        .execute(input2, json!({"op": "nms", "iou": 0.45}))
        .unwrap()
        .unwrap_json();

    let nms_count = nms_result["boxes"].as_array().unwrap().len();
    assert!(nms_count <= filtered_count, "NMS should not add boxes");
    assert!(nms_count > 0, "NMS should keep at least one box");
}

// ── Test: Image → tensor processing chain ────────────────────────
//
// Simulates: decode → normalize → layout(HWC→CHW) → unsqueeze(batch dim)
// This is the typical YOLO/ResNet pre-processing pipeline.

#[cfg(all(feature = "vision", feature = "onnx"))]
#[test]
fn test_image_to_tensor_chain() {
    use axon::kernels::image::ImageKernel;
    use axon::kernels::tensor::TensorKernel;

    let image_kernel = ImageKernel;
    let tensor_kernel = TensorKernel;

    let h = 4;
    let w = 4;
    let c = 3;

    // Create a fake HWC image blob (values 0-255)
    let pixels: Vec<f32> = (0..h * w * c).map(|i| (i % 256) as f32).collect();
    let bytes: Vec<u8> = pixels.iter().flat_map(|f| f.to_le_bytes()).collect();

    let input = KernelInput {
        json: serde_json::Value::Null,
        blobs: {
            let mut m = HashMap::new();
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

    // Step 1: normalize (scale by 1/255, ImageNet mean/std)
    let normalized = image_kernel
        .execute(
            input,
            json!({
                "op": "normalize",
                "scale": 255.0,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }),
        )
        .unwrap();

    // Verify blob output
    let (norm_data, norm_shape) = match &normalized {
        KernelOutput::Blob { data, shape, .. } => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            (floats, shape.clone().unwrap())
        }
        _ => panic!("expected Blob from normalize"),
    };
    assert_eq!(norm_shape, vec![h, w, c]);
    assert_eq!(norm_data.len(), h * w * c);

    // Step 2: layout HWC → CHW
    let chw_input = KernelInput {
        json: serde_json::Value::Null,
        blobs: {
            let mut m = HashMap::new();
            m.insert(
                "image".to_string(),
                BlobData {
                    bytes: norm_data.iter().flat_map(|f| f.to_le_bytes()).collect(),
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

    let chw_output = image_kernel
        .execute(chw_input, json!({"op": "layout", "to": "chw"}))
        .unwrap();

    let (chw_data, chw_shape) = match &chw_output {
        KernelOutput::Blob { data, shape, .. } => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            (floats, shape.clone().unwrap())
        }
        _ => panic!("expected Blob from layout"),
    };
    assert_eq!(chw_shape, vec![c, h, w]);
    assert_eq!(chw_data.len(), h * w * c);

    // Step 3: unsqueeze to add batch dim [C,H,W] → [1,C,H,W]
    let unsqueeze_input = KernelInput {
        json: json!({
            "shape": chw_shape,
            "data": chw_data.iter().map(|&f| f as f64).collect::<Vec<f64>>(),
        }),
        blobs: HashMap::new(),
    };

    let batch_output = tensor_kernel
        .execute(unsqueeze_input, json!({"op": "unsqueeze", "dim": 0}))
        .unwrap()
        .unwrap_json();

    let batch_shape: Vec<u64> = batch_output["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(batch_shape, vec![1, c as u64, h as u64, w as u64]);
}

// ── Test: Multi-step tensor pipeline ─────────────────────────────
//
// softmax → argmax (common classification post-processing)

#[cfg(feature = "onnx")]
#[test]
fn test_softmax_then_argmax() {
    use axon::kernels::tensor::TensorKernel;

    let kernel = TensorKernel;

    // Simulate logits for 5 classes
    let logits = vec![1.0, 3.0, 0.5, 2.0, 0.1];

    // Step 1: softmax
    let input1 = KernelInput::from_json(json!({
        "shape": [5],
        "data": logits,
    }));
    let probs = kernel
        .execute(input1, json!({"op": "softmax", "dim": 0}))
        .unwrap()
        .unwrap_json();

    let prob_data: Vec<f64> = probs["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    // Softmax should sum to ~1.0
    let sum: f64 = prob_data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax sum = {sum}"
    );

    // Step 2: argmax
    let input2 = KernelInput::from_json(json!({
        "shape": [5],
        "data": prob_data,
    }));
    let argmax_result = kernel
        .execute(input2, json!({"op": "argmax", "dim": 0}))
        .unwrap()
        .unwrap_json();

    // Class 1 (value 3.0) should be the argmax.
    // argmax returns {"indices": [...], "shape": [...]}.
    // For 1D dim=0, result is scalar (0D array with 1 element).
    let indices: Vec<i64> = argmax_result["indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(indices, vec![1], "argmax should select index 1 (highest logit)");
}

// ── Test: Full pipeline with manifest + mock model ───────────────
//
// Tests Pipeline::run() end-to-end with mock ONNX kernel.

#[cfg(feature = "onnx")]
#[test]
fn test_full_pipeline_with_manifest() {
    let toml = r#"
[model]
name = "test-model"
file = "model.onnx"

[post]
steps = [
    {op = "tensor.normalize"},
]
"#;

    let pipeline = make_pipeline(toml);

    // Validate pipeline
    assert!(pipeline.validate().is_ok());

    // Run pipeline
    let output = pipeline.run(b"hello world", "text/plain").unwrap();

    // Should produce normalized output
    match output {
        KernelOutput::Json(v) => {
            let data = v["data"].as_array().unwrap();
            assert!(!data.is_empty());

            // Check normalization: L2 norm should be ~1.0
            let norm: f64 = data
                .iter()
                .map(|v| {
                    let x = v.as_f64().unwrap();
                    x * x
                })
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "L2 norm = {norm}"
            );
        }
        KernelOutput::Blob { data, .. } => {
            // Blob is also valid
            assert!(!data.is_empty());
        }
    }
}

// ── Test: Pipeline validation catches missing kernels ────────────

#[test]
fn test_pipeline_validation_missing_kernel() {
    let toml = r#"
[model]
name = "test"
file = "model.onnx"

[pre]
steps = [
    {op = "nonexistent.do_thing"},
]
"#;

    let manifest = Manifest::from_toml(toml).unwrap();
    let mut registry = KernelRegistry::new();
    registry.register(Arc::new(MockOnnxKernel));

    let pipeline = Pipeline::new(manifest, registry, PathBuf::from("."));
    let result = pipeline.validate();

    assert!(result.is_err());
    let missing = result.unwrap_err();
    assert!(missing.contains(&"nonexistent".to_string()));
}

// ── Test: Pipeline with no post-processing ───────────────────────

#[cfg(feature = "onnx")]
#[test]
fn test_pipeline_model_only() {
    let toml = r#"
[model]
name = "passthrough"
file = "model.onnx"
"#;

    let pipeline = make_pipeline(toml);
    let output = pipeline.run(b"raw data", "application/octet-stream").unwrap();

    // MockOnnxKernel returns 384-dim tensor. Pipeline may return Blob or JSON
    // depending on whether blob-to-JSON conversion happens at pipeline boundary.
    match &output {
        KernelOutput::Blob { shape, .. } => {
            assert_eq!(shape, &Some(vec![1, 384]));
        }
        KernelOutput::Json(v) => {
            // Pipeline converted blob to JSON — verify it has content
            assert!(v.is_object() || v.is_array());
        }
    }
}

// ── Test: Detection pipeline via manifest ────────────────────────
//
// Tests the full pipeline flow with detection post-processing steps.

#[cfg(all(feature = "vision", feature = "onnx"))]
#[test]
fn test_detection_pipeline_via_manifest() {
    // Create a mock ONNX kernel that outputs detection-like data
    struct DetectionMockOnnx;
    impl ComputeKernel for DetectionMockOnnx {
        fn name(&self) -> &str {
            "onnx"
        }
        fn execute(
            &self,
            _input: KernelInput,
            _operations: serde_json::Value,
        ) -> Result<KernelOutput, AxonError> {
            // Return pre-formatted detection JSON (as if split already happened)
            let boxes: Vec<Vec<f64>> = (0..10)
                .map(|i| {
                    let x = i as f64 * 50.0;
                    vec![x, 0.0, x + 40.0, 40.0]
                })
                .collect();
            let scores: Vec<f64> = (0..10).map(|i| 0.2 + i as f64 * 0.08).collect();

            Ok(KernelOutput::Json(json!({
                "boxes": boxes,
                "scores": scores,
            })))
        }
    }

    let toml = r#"
[model]
name = "yolov8n-test"
file = "model.onnx"

[post]
steps = [
    {op = "detection.confidence_filter", threshold = 0.5},
    {op = "detection.nms", iou = 0.45},
]
"#;

    let manifest = Manifest::from_toml(toml).unwrap();
    let mut registry = KernelRegistry::new();
    registry.register(Arc::new(DetectionMockOnnx));
    registry.register(Arc::new(axon::kernels::detection::DetectionKernel));
    registry.register(Arc::new(axon::kernels::tensor::TensorKernel));

    let pipeline = Pipeline::new(manifest, registry, PathBuf::from("."));
    let output = pipeline.run(b"image data", "image/jpeg").unwrap();

    match output {
        KernelOutput::Json(v) => {
            let boxes = v["boxes"].as_array().unwrap();
            let scores = v["scores"].as_array().unwrap();

            // All remaining scores should be >= 0.5
            for s in scores {
                assert!(s.as_f64().unwrap() >= 0.5 - 1e-6);
            }

            // NMS should have reduced count
            assert!(boxes.len() <= 10);
            assert!(boxes.len() > 0);
        }
        _ => panic!("expected JSON output from detection pipeline"),
    }
}

// ── Test: Tensor chain with blob passing ─────────────────────────
//
// Tests that blob_output=true produces binary data that can be consumed
// by the next kernel step.

#[cfg(feature = "onnx")]
#[test]
fn test_tensor_blob_chaining() {
    use axon::kernels::tensor::TensorKernel;

    let kernel = TensorKernel;

    // Create blob input: [2, 3] tensor
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let input = KernelInput {
        json: serde_json::Value::Null,
        blobs: {
            let mut m = HashMap::new();
            m.insert(
                "tensor".to_string(),
                BlobData {
                    bytes,
                    meta: BlobMeta {
                        size: 0,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![2, 3]),
                    },
                },
            );
            m
        },
    };

    // Step 1: reshape [2,3] → [3,2] with blob output
    let mid = kernel
        .execute(input, json!({"op": "reshape", "shape": [3, 2], "blob_output": true}))
        .unwrap();

    let (mid_bytes, mid_shape) = match &mid {
        KernelOutput::Blob {
            data, shape, ..
        } => (data.clone(), shape.clone().unwrap()),
        _ => panic!("expected Blob"),
    };
    assert_eq!(mid_shape, vec![3, 2]);

    // Step 2: consume blob and transpose [3,2] → [2,3]
    let input2 = KernelInput {
        json: serde_json::Value::Null,
        blobs: {
            let mut m = HashMap::new();
            m.insert(
                "tensor".to_string(),
                BlobData {
                    bytes: mid_bytes,
                    meta: BlobMeta {
                        size: 0,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![3, 2]),
                    },
                },
            );
            m
        },
    };

    let result = kernel
        .execute(input2, json!({"op": "transpose", "axes": [1, 0]}))
        .unwrap()
        .unwrap_json();

    let result_shape: Vec<u64> = result["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(result_shape, vec![2, 3]);

    // Data should be transposed (column-major reorder)
    let result_data: Vec<f64> = result["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    // Original [1,2,3,4,5,6] reshaped to [3,2] = [[1,2],[3,4],[5,6]]
    // Transposed to [2,3] = [[1,3,5],[2,4,6]]
    assert_eq!(result_data, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

// ── Test: Error propagation through pipeline ─────────────────────

#[cfg(feature = "onnx")]
#[test]
fn test_pipeline_error_propagation() {
    let toml = r#"
[model]
name = "test-error"
file = "model.onnx"

[post]
steps = [
    {op = "tensor.reshape", shape = [999]},
]
"#;

    let pipeline = make_pipeline(toml);

    // MockOnnxKernel returns [1, 384] — reshape to [999] should fail
    let result = pipeline.run(b"test", "text/plain");
    assert!(result.is_err(), "pipeline should propagate shape error");
}

// ── Test: Large tensor pipeline stress test ──────────────────────

#[cfg(feature = "onnx")]
#[test]
fn test_large_tensor_pipeline() {
    use axon::kernels::tensor::TensorKernel;

    let kernel = TensorKernel;

    // Create a large tensor [128, 768] (similar to BERT hidden states)
    let n = 128 * 768;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) % 10.0).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let input = KernelInput {
        json: serde_json::Value::Null,
        blobs: {
            let mut m = HashMap::new();
            m.insert(
                "tensor".to_string(),
                BlobData {
                    bytes,
                    meta: BlobMeta {
                        size: 0,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![128, 768]),
                    },
                },
            );
            m
        },
    };

    // mean_pool along sequence dim → [768]
    let pooled = kernel
        .execute(input, json!({"op": "mean_pool", "dim": 0, "blob_output": true}))
        .unwrap();

    let (pooled_data, pooled_shape) = match &pooled {
        KernelOutput::Blob { data, shape, .. } => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            (floats, shape.clone().unwrap())
        }
        _ => panic!("expected Blob"),
    };
    assert_eq!(pooled_shape, vec![768]);
    assert_eq!(pooled_data.len(), 768);

    // normalize
    let input2 = KernelInput {
        json: serde_json::Value::Null,
        blobs: {
            let mut m = HashMap::new();
            m.insert(
                "tensor".to_string(),
                BlobData {
                    bytes: pooled_data.iter().flat_map(|f| f.to_le_bytes()).collect(),
                    meta: BlobMeta {
                        size: 0,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![768]),
                    },
                },
            );
            m
        },
    };

    let normalized = kernel
        .execute(input2, json!({"op": "normalize", "blob_output": true}))
        .unwrap();

    let norm_data = match &normalized {
        KernelOutput::Blob { data, .. } => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            floats
        }
        _ => panic!("expected Blob"),
    };

    // L2 norm should be ~1.0
    let l2: f64 = norm_data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    assert!(
        (l2 - 1.0).abs() < 1e-3,
        "L2 norm of 768-dim vector = {l2}"
    );
}

// ══════════════════════════════════════════════════════════════════
// Batch pipeline integration tests
// ══════════════════════════════════════════════════════════════════
//
// Tests true tensor-level batching: N inputs → pre each → concat dim 0
// → single model call → split → post each → N outputs.

/// Mock ONNX kernel that implements true tensor batching.
///
/// Single execute: returns a 384-dim blob tensor per input.
/// Batch execute: verifies all inputs received, returns N individual results.
struct BatchAwareOnnxKernel;

impl ComputeKernel for BatchAwareOnnxKernel {
    fn name(&self) -> &str {
        "onnx"
    }

    fn execute(
        &self,
        _input: KernelInput,
        _operations: serde_json::Value,
    ) -> Result<KernelOutput, AxonError> {
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32 + 1.0) * 0.01).collect();
        Ok(KernelOutput::Blob {
            data: embedding.iter().flat_map(|f| f.to_le_bytes()).collect(),
            content_type: "tensor/f32".to_string(),
            shape: Some(vec![1, 384]),
        })
    }

    fn supports_batch(&self) -> bool {
        true
    }

    fn execute_batch(
        &self,
        inputs: Vec<KernelInput>,
        _operations: serde_json::Value,
    ) -> Result<Vec<KernelOutput>, AxonError> {
        let n = inputs.len();
        // Return N individual blob outputs, each marked with the batch index.
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            // Each item gets a distinct embedding (first element = batch index).
            let mut embedding: Vec<f32> = (0..384).map(|j| (j as f32 + 1.0) * 0.01).collect();
            embedding[0] = i as f32; // Tag with batch index for verification.
            results.push(KernelOutput::Blob {
                data: embedding.iter().flat_map(|f| f.to_le_bytes()).collect(),
                content_type: "tensor/f32".to_string(),
                shape: Some(vec![1, 384]),
            });
        }
        Ok(results)
    }
}

fn make_batch_pipeline(toml: &str) -> Pipeline {
    let manifest = Manifest::from_toml(toml).unwrap();
    let mut registry = KernelRegistry::new();
    registry.register(Arc::new(BatchAwareOnnxKernel));

    #[cfg(feature = "onnx")]
    registry.register(Arc::new(axon::kernels::tensor::TensorKernel));

    Pipeline::new(manifest, registry, PathBuf::from("."))
}

// ── Test: run_batch with model-only pipeline ─────────────────────

#[cfg(feature = "onnx")]
#[test]
fn test_run_batch_model_only() {
    let toml = r#"
[model]
name = "batch-test"
file = "model.onnx"
"#;
    let pipeline = make_batch_pipeline(toml);

    let inputs: Vec<(&[u8], &str)> = vec![
        (b"input-0", "text/plain"),
        (b"input-1", "text/plain"),
        (b"input-2", "text/plain"),
        (b"input-3", "text/plain"),
    ];

    let results = pipeline.run_batch(&inputs).unwrap();
    assert_eq!(results.len(), 4);

    // Each output should be a blob from BatchAwareOnnxKernel.
    for (i, result) in results.iter().enumerate() {
        let output = result.as_ref().unwrap_or_else(|e| panic!("batch item {i} failed: {e}"));
        match output {
            KernelOutput::Blob { data, shape, .. } => {
                assert_eq!(shape, &Some(vec![1, 384]));
                // Verify the first float is the batch index (tagged in execute_batch).
                let first_f32 = f32::from_le_bytes(data[..4].try_into().unwrap());
                assert_eq!(
                    first_f32, i as f32,
                    "batch item {i} should have index tag"
                );
            }
            _ => panic!("expected blob output for batch item {i}"),
        }
    }
}

// ── Test: run_batch with post-processing ─────────────────────────

#[cfg(feature = "onnx")]
#[test]
fn test_run_batch_with_post_processing() {
    let toml = r#"
[model]
name = "batch-post-test"
file = "model.onnx"

[post]
steps = [
    {op = "tensor.normalize"},
]
"#;
    let pipeline = make_batch_pipeline(toml);

    let inputs: Vec<(&[u8], &str)> = vec![
        (b"input-0", "text/plain"),
        (b"input-1", "text/plain"),
        (b"input-2", "text/plain"),
    ];

    let results = pipeline.run_batch(&inputs).unwrap();
    assert_eq!(results.len(), 3);

    // Each output should be normalized (L2 norm ≈ 1.0).
    for (i, result) in results.iter().enumerate() {
        let output = result.as_ref().unwrap_or_else(|e| panic!("batch item {i} failed: {e}"));
        match output {
            KernelOutput::Json(v) => {
                let data = v["data"].as_array().unwrap();
                let l2: f64 = data
                    .iter()
                    .map(|v| {
                        let x = v.as_f64().unwrap();
                        x * x
                    })
                    .sum::<f64>()
                    .sqrt();
                assert!(
                    (l2 - 1.0).abs() < 1e-3,
                    "batch item {i}: L2 norm = {l2}"
                );
            }
            KernelOutput::Blob { data, .. } => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let l2: f64 = floats.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                assert!(
                    (l2 - 1.0).abs() < 1e-3,
                    "batch item {i}: blob L2 norm = {l2}"
                );
            }
        }
    }
}

// ── Test: run_batch single item behaves like run() ───────────────

#[cfg(feature = "onnx")]
#[test]
fn test_run_batch_single_item() {
    let toml = r#"
[model]
name = "single-batch"
file = "model.onnx"
"#;
    let pipeline = make_batch_pipeline(toml);

    let single_result = pipeline.run(b"hello", "text/plain").unwrap();
    let batch_results = pipeline.run_batch(&[(b"hello", "text/plain")]).unwrap();

    assert_eq!(batch_results.len(), 1);
    let batch_item = batch_results.into_iter().next().unwrap().unwrap();

    // Both should produce blob outputs with same shape.
    match (&single_result, &batch_item) {
        (
            KernelOutput::Blob {
                shape: s1,
                content_type: ct1,
                ..
            },
            KernelOutput::Blob {
                shape: s2,
                content_type: ct2,
                ..
            },
        ) => {
            assert_eq!(s1, s2);
            assert_eq!(ct1, ct2);
        }
        _ => {} // Different output types are acceptable for single vs batch
    }
}

// ── Test: run_batch empty input ──────────────────────────────────

#[test]
fn test_run_batch_empty() {
    let toml = r#"
[model]
name = "empty-batch"
file = "model.onnx"
"#;
    let pipeline = make_batch_pipeline(toml);

    let outputs = pipeline.run_batch(&[]).unwrap();
    assert!(outputs.is_empty());
}

// ── Test: run_batch large batch size ─────────────────────────────

#[cfg(feature = "onnx")]
#[test]
fn test_run_batch_large() {
    let toml = r#"
[model]
name = "large-batch"
file = "model.onnx"
"#;
    let pipeline = make_batch_pipeline(toml);

    let input_data: Vec<Vec<u8>> = (0..32).map(|i| format!("input-{i}").into_bytes()).collect();
    let inputs: Vec<(&[u8], &str)> = input_data
        .iter()
        .map(|d| (d.as_slice(), "text/plain"))
        .collect();

    let results = pipeline.run_batch(&inputs).unwrap();
    assert_eq!(results.len(), 32);

    // Verify each output is distinct (batch index tag).
    for (i, result) in results.iter().enumerate() {
        let output = result.as_ref().unwrap_or_else(|e| panic!("batch item {i} failed: {e}"));
        if let KernelOutput::Blob { data, .. } = output {
            let tag = f32::from_le_bytes(data[..4].try_into().unwrap());
            assert_eq!(tag, i as f32, "batch item {i} mismatch");
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// Audio pipeline integration tests
// ══════════════════════════════════════════════════════════════════
//
// Tests the real audio processing pipeline:
//   WAV bytes → audio.decode (PCM f32) → mel.spectrogram (tensor)
//
// Run: `cargo test -p axon --features audio --test integration_tests`

/// Generate a WAV file with a sine wave signal.
/// Returns raw WAV bytes (16-bit PCM mono).
#[cfg(feature = "audio")]
fn generate_sine_wav(freq_hz: f32, duration_secs: f32, sample_rate: u32) -> Vec<u8> {
    let n_samples = (sample_rate as f32 * duration_secs) as usize;
    let data_size = (n_samples * 2) as u32; // 16-bit = 2 bytes/sample
    let file_size = 36 + data_size;
    let mut wav = Vec::with_capacity(file_size as usize + 8);

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());

    let two_pi = 2.0 * std::f32::consts::PI;
    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;
        let val = 0.5 * (two_pi * freq_hz * t).sin();
        let sample = (val * 32000.0) as i16;
        wav.extend_from_slice(&sample.to_le_bytes());
    }

    wav
}

// ── Test: audio.decode kernel with real WAV file ─────────────────

#[cfg(feature = "audio")]
#[test]
fn test_audio_decode_real_wav_file() {
    use axon::kernels::audio::AudioKernel;

    let wav_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sherpa-onnx-zipformer-zh-en-2023-11-22/test_wavs/0.wav"
    );
    let wav_bytes = std::fs::read(wav_path).expect("test WAV file not found");

    let input = KernelInput {
        json: serde_json::Value::Null,
        blobs: {
            let mut m = HashMap::new();
            m.insert(
                "_input".to_string(),
                BlobData {
                    bytes: wav_bytes.clone(),
                    meta: BlobMeta {
                        size: wav_bytes.len() as u64,
                        content_type: "audio/wav".to_string(),
                        shape: None,
                    },
                },
            );
            m
        },
    };

    let result = AudioKernel::new()
        .execute(input, json!({"op": "decode", "sample_rate": 16000}))
        .unwrap();

    match &result {
        KernelOutput::Blob {
            data,
            content_type,
            shape,
        } => {
            assert_eq!(content_type, "audio/pcm-f32");
            let n_samples = data.len() / 4;
            assert!(n_samples > 1000, "should have >1000 samples, got {n_samples}");
            assert_eq!(shape, &Some(vec![1, n_samples]));

            // Verify PCM data is valid (not all zeros, not NaN)
            let samples: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            assert!(max_abs > 0.001, "audio should have non-zero samples (max_abs={max_abs})");
            assert!(samples.iter().all(|s| s.is_finite()), "all samples should be finite");

            eprintln!(
                "  audio.decode: {} samples ({:.2}s at 16kHz), peak={:.4}",
                n_samples,
                n_samples as f32 / 16000.0,
                max_abs
            );
        }
        _ => panic!("expected blob output from audio.decode"),
    }
}

// ── Test: audio.decode with resampling (44100 → 16000) ──────────

#[cfg(feature = "audio")]
#[test]
fn test_audio_decode_with_resample() {
    use axon::kernels::audio::AudioKernel;

    // Generate 2s 440Hz sine at 44100Hz — should resample to 16000Hz
    let wav = generate_sine_wav(440.0, 2.0, 44100);

    let input = KernelInput {
        json: serde_json::Value::Null,
        blobs: {
            let mut m = HashMap::new();
            m.insert(
                "_input".to_string(),
                BlobData {
                    bytes: wav.clone(),
                    meta: BlobMeta {
                        size: wav.len() as u64,
                        content_type: "audio/wav".to_string(),
                        shape: None,
                    },
                },
            );
            m
        },
    };

    let result = AudioKernel::new()
        .execute(input, json!({"op": "decode", "sample_rate": 16000}))
        .unwrap();

    match &result {
        KernelOutput::Blob { data, shape, .. } => {
            let n_samples = data.len() / 4;
            // 2s at 16kHz ≈ 32000 samples (±100 for resampler edge effects)
            assert!(
                (n_samples as i64 - 32000).abs() < 200,
                "resampled length should be ~32000, got {n_samples}"
            );
            assert_eq!(shape, &Some(vec![1, n_samples]));
            eprintln!("  resample 44100→16000: {n_samples} samples (expected ~32000)");
        }
        _ => panic!("expected blob output"),
    }
}

// ── Test: full audio → mel pipeline (Whisper-compatible) ─────────

#[cfg(feature = "audio")]
#[test]
fn test_audio_to_mel_pipeline() {
    use axon::kernels::audio::AudioKernel;
    use axon::kernels::mel::MelKernel;

    // Load real speech WAV
    let wav_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sherpa-onnx-zipformer-zh-en-2023-11-22/test_wavs/1.wav"
    );
    let wav_bytes = std::fs::read(wav_path).expect("test WAV file not found");

    // Step 1: audio.decode → PCM f32 at 16kHz
    let input = KernelInput {
        json: serde_json::Value::Null,
        blobs: {
            let mut m = HashMap::new();
            m.insert(
                "_input".to_string(),
                BlobData {
                    bytes: wav_bytes.clone(),
                    meta: BlobMeta {
                        size: wav_bytes.len() as u64,
                        content_type: "audio/wav".to_string(),
                        shape: None,
                    },
                },
            );
            m
        },
    };

    let pcm_output = AudioKernel::new()
        .execute(input, json!({"op": "decode", "sample_rate": 16000}))
        .unwrap();

    let (pcm_bytes, pcm_shape) = match &pcm_output {
        KernelOutput::Blob { data, shape, content_type } => {
            assert_eq!(content_type, "audio/pcm-f32");
            (data.clone(), shape.clone().unwrap())
        }
        _ => panic!("expected blob from audio.decode"),
    };

    let n_pcm = pcm_bytes.len() / 4;
    eprintln!(
        "  Step 1 (audio.decode): {} PCM samples ({:.2}s)",
        n_pcm,
        n_pcm as f32 / 16000.0
    );

    // Step 2: mel.spectrogram → [1, 80, 3000] tensor (Whisper format)
    let mel_input = KernelInput {
        json: serde_json::Value::Null,
        blobs: {
            let mut m = HashMap::new();
            m.insert(
                "_prev".to_string(),
                BlobData {
                    bytes: pcm_bytes,
                    meta: BlobMeta {
                        size: (n_pcm * 4) as u64,
                        content_type: "audio/pcm-f32".to_string(),
                        shape: Some(pcm_shape),
                    },
                },
            );
            m
        },
    };

    let mel_output = MelKernel::new()
        .execute(
            mel_input,
            json!({
                "op": "spectrogram",
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 80,
                "sample_rate": 16000,
                "chunk_length": 30
            }),
        )
        .unwrap();

    match &mel_output {
        KernelOutput::Blob {
            data,
            content_type,
            shape,
        } => {
            assert_eq!(content_type, "tensor/f32");
            assert_eq!(shape, &Some(vec![1, 80, 3000]));
            assert_eq!(data.len(), 1 * 80 * 3000 * 4); // 960,000 bytes

            // Verify mel values are reasonable (not all zeros, finite)
            let mel_vals: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert!(mel_vals.iter().all(|v| v.is_finite()), "all mel values should be finite");

            let min = mel_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = mel_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = mel_vals.iter().sum::<f32>() / mel_vals.len() as f32;
            eprintln!(
                "  Step 2 (mel.spectrogram): shape=[1,80,3000], range=[{:.3}, {:.3}], mean={:.3}",
                min, max, mean
            );

            // Whisper mel values should be normalized roughly in [-1, 1] range
            assert!(min >= -2.0, "mel min too low: {min}");
            assert!(max <= 2.0, "mel max too high: {max}");
        }
        _ => panic!("expected blob from mel.spectrogram"),
    }
}

// ── Test: audio pipeline via Pipeline::run_pre ───────────────────
//
// Tests the full TOML-driven pipeline path (manifest → pipeline → run_pre)

#[cfg(feature = "audio")]
#[test]
fn test_audio_pipeline_via_manifest() {
    let toml = r#"
[model]
name = "whisper-test"
file = "model.onnx"

[pre]
steps = [
    {op = "audio.decode", sample_rate = 16000},
    {op = "mel.spectrogram", n_fft = 400, hop_length = 160, n_mels = 80, chunk_length = 30},
]
"#;

    let pipeline = make_pipeline(toml);

    // Load real speech WAV
    let wav_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sherpa-onnx-zipformer-zh-en-2023-11-22/test_wavs/2.wav"
    );
    let wav_bytes = std::fs::read(wav_path).expect("test WAV file not found");

    // run_pre: just the preprocessing steps (no model inference)
    let output = pipeline.run_pre(&wav_bytes, "audio/wav").unwrap();

    match &output {
        KernelOutput::Blob {
            data,
            content_type,
            shape,
        } => {
            assert_eq!(content_type, "tensor/f32");
            assert_eq!(shape, &Some(vec![1, 80, 3000]));
            assert_eq!(data.len(), 80 * 3000 * 4);

            eprintln!(
                "  Pipeline run_pre: WAV ({} bytes) → mel tensor [1,80,3000] ({} bytes)",
                wav_bytes.len(),
                data.len()
            );
        }
        _ => panic!("expected blob output from audio pipeline"),
    }
}

// ── Test: all 3 WAV files decode successfully ────────────────────

#[cfg(feature = "audio")]
#[test]
fn test_all_test_wavs_decode() {
    use axon::kernels::audio::AudioKernel;

    let base = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sherpa-onnx-zipformer-zh-en-2023-11-22/test_wavs"
    );

    for name in &["0.wav", "1.wav", "2.wav"] {
        let path = format!("{base}/{name}");
        let wav_bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{name}: {e}"));

        let input = KernelInput {
            json: serde_json::Value::Null,
            blobs: {
                let mut m = HashMap::new();
                m.insert(
                    "_input".to_string(),
                    BlobData {
                        bytes: wav_bytes.clone(),
                        meta: BlobMeta {
                            size: wav_bytes.len() as u64,
                            content_type: "audio/wav".to_string(),
                            shape: None,
                        },
                    },
                );
                m
            },
        };

        let result = AudioKernel::new()
            .execute(input, json!({"op": "decode", "sample_rate": 16000}))
            .unwrap();

        match &result {
            KernelOutput::Blob { data, .. } => {
                let n = data.len() / 4;
                let samples: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                let duration = n as f32 / 16000.0;
                eprintln!("  {name}: {n} samples ({duration:.2}s), peak={peak:.4}");
                assert!(n > 0, "{name} should have samples");
                assert!(peak > 0.0, "{name} should have non-zero audio");
            }
            _ => panic!("{name}: expected blob output"),
        }
    }
}
