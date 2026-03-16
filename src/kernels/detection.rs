//! Object detection post-processing kernel: NMS, confidence filtering, box format conversion.
//!
//! Enabled alongside `vision` feature (shares vision dependencies).
//!
//! High-performance pure Rust implementation with minimal allocations:
//!   ONNX(detections blob) → BlobStore → detection_kernel(format + filter) → JSON detections
//!
//! Supported operations:
//!   - `confidence_filter`: filter by confidence threshold
//!   - `nms`: non-maximum suppression with IoU threshold
//!   - `xywh_to_xyxy`: convert [center_x, center_y, width, height] → [x1, y1, x2, y2]
//!   - `xyxy_to_xywh`: convert [x1, y1, x2, y2] → [center_x, center_y, width, height]
//!   - `split`: split raw ONNX tensor into boxes and scores with optional transpose
//!   - `format`: format final output (json or coco format)
//!
//! Config (operations):
//!   - String: `"nms"` (defaults: iou=0.45)
//!   - Object: `{"op": "nms", "iou": 0.45}`
//!   - Object: `{"op": "confidence_filter", "threshold": 0.25}`
//!   - Object: `{"op": "split", "boxes": "0:4", "scores": "4:", "transpose": true}`

use serde_json::{json, Value};
use tracing::debug;

use crate::error::AxonError;
use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};

/// Object detection post-processing kernel.
pub struct DetectionKernel;

/// Soft-NMS decay method (Bodla et al., 2017).
#[derive(Clone, Copy, Debug)]
enum SoftNmsMethod {
    /// Linear decay: s *= (1 - IoU) when IoU > threshold
    Linear,
    /// Gaussian decay: s *= exp(-IoU² / sigma)
    Gaussian,
}

/// Parsed detection operation.
enum DetectionOp {
    /// Filter detections by confidence threshold.
    ConfidenceFilter { threshold: f32 },
    /// Non-maximum suppression.
    Nms { iou_threshold: f32 },
    /// Convert [center_x, center_y, width, height] → [x1, y1, x2, y2].
    XywhToXyxy,
    /// Convert [x1, y1, x2, y2] → [center_x, center_y, width, height].
    XyxyToXywh,
    /// Split raw ONNX tensor into boxes and scores.
    Split {
        boxes_range: (usize, usize),
        scores_range: (usize, usize),
        transpose: bool,
    },
    /// Format final detection output.
    Format {
        output_format: String,
        labels: Option<Vec<String>>,
    },
    /// Soft-NMS: decay scores instead of hard suppression.
    SoftNms {
        iou_threshold: f32,
        sigma: f32,
        method: SoftNmsMethod,
        min_score: f32,
    },
    /// Per-class NMS using coordinate offset trick (torchvision.ops.batched_nms).
    BatchedNms { iou_threshold: f32 },
}

/// Parsed config with operation + output mode.
struct DetectionConfig {
    op: DetectionOp,
}

impl ComputeKernel for DetectionKernel {
    fn name(&self) -> &str {
        "detection"
    }

    fn execute(
        &self,
        input: KernelInput,
        operations: Value,
    ) -> Result<KernelOutput, AxonError> {
        let config = parse_config(&operations)?;

        match config.op {
            DetectionOp::ConfidenceFilter { threshold } => {
                confidence_filter(&input.json, threshold)
            }
            DetectionOp::Nms { iou_threshold } => nms_operation(&input.json, iou_threshold),
            DetectionOp::XywhToXyxy => xywh_to_xyxy_op(&input.json),
            DetectionOp::XyxyToXywh => xyxy_to_xywh_op(&input.json),
            DetectionOp::Split {
                boxes_range,
                scores_range,
                transpose,
            } => split_operation(&input, boxes_range, scores_range, transpose),
            DetectionOp::Format {
                output_format,
                labels,
            } => format_operation(&input.json, &output_format, labels),
            DetectionOp::SoftNms { iou_threshold, sigma, method, min_score } => {
                soft_nms_operation(&input.json, iou_threshold, sigma, method, min_score)
            }
            DetectionOp::BatchedNms { iou_threshold } => {
                batched_nms_operation(&input.json, iou_threshold)
            }
        }.map_err(Into::into)
    }
}

// ── Config Parsing ──────────────────────────────────────────────────

fn parse_config(operations: &Value) -> Result<DetectionConfig, String> {
    let op = match operations {
        Value::String(s) => {
            // Default operations
            match s.as_str() {
                "confidence_filter" => DetectionOp::ConfidenceFilter { threshold: 0.25 },
                "nms" => DetectionOp::Nms {
                    iou_threshold: 0.45,
                },
                "xywh_to_xyxy" => DetectionOp::XywhToXyxy,
                "xyxy_to_xywh" => DetectionOp::XyxyToXywh,
                "soft_nms" => DetectionOp::SoftNms {
                    iou_threshold: 0.3,
                    sigma: 0.5,
                    method: SoftNmsMethod::Gaussian,
                    min_score: 0.001,
                },
                "batched_nms" => DetectionOp::BatchedNms { iou_threshold: 0.45 },
                _ => return Err(format!("detection: unknown operation '{}'", s)),
            }
        }
        Value::Object(map) => {
            let op_name = map
                .get("op")
                .and_then(|v| v.as_str())
                .ok_or("detection: missing 'op' field in config")?;

            match op_name {
                "confidence_filter" => {
                    let threshold = map
                        .get("threshold")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.25) as f32;
                    DetectionOp::ConfidenceFilter { threshold }
                }
                "nms" => {
                    let iou_threshold = map
                        .get("iou")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.45) as f32;
                    DetectionOp::Nms { iou_threshold }
                }
                "xywh_to_xyxy" => DetectionOp::XywhToXyxy,
                "xyxy_to_xywh" => DetectionOp::XyxyToXywh,
                "split" => {
                    let boxes_str = map
                        .get("boxes")
                        .and_then(|v| v.as_str())
                        .ok_or("detection: split requires 'boxes' range string (e.g. '0:4')")?;
                    let scores_str = map
                        .get("scores")
                        .and_then(|v| v.as_str())
                        .ok_or("detection: split requires 'scores' range string (e.g. '4:')")?;
                    let transpose = map
                        .get("transpose")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);

                    let boxes_range = parse_range(boxes_str)?;
                    let scores_range = parse_range(scores_str)?;

                    DetectionOp::Split {
                        boxes_range,
                        scores_range,
                        transpose,
                    }
                }
                "format" => {
                    let output_format = map
                        .get("output")
                        .and_then(|v| v.as_str())
                        .unwrap_or("json")
                        .to_string();
                    let labels = map
                        .get("labels")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        });

                    DetectionOp::Format {
                        output_format,
                        labels,
                    }
                }
                "soft_nms" => {
                    let iou_threshold = map
                        .get("iou")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.3) as f32;
                    let sigma = map
                        .get("sigma")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5) as f32;
                    let method = match map
                        .get("method")
                        .and_then(|v| v.as_str())
                        .unwrap_or("gaussian")
                    {
                        "linear" => SoftNmsMethod::Linear,
                        "gaussian" => SoftNmsMethod::Gaussian,
                        m => return Err(format!("detection: unknown soft_nms method '{}'", m)),
                    };
                    let min_score = map
                        .get("min_score")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.001) as f32;
                    DetectionOp::SoftNms { iou_threshold, sigma, method, min_score }
                }
                "batched_nms" => {
                    let iou_threshold = map
                        .get("iou")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.45) as f32;
                    DetectionOp::BatchedNms { iou_threshold }
                }
                _ => return Err(format!("detection: unknown operation '{}'", op_name)),
            }
        }
        _ => return Err("detection: operations must be string or object".to_string()),
    };

    Ok(DetectionConfig { op })
}

fn parse_range(s: &str) -> Result<(usize, usize), String> {
    let parts: Vec<&str> = s.split(':').collect();
    match parts.as_slice() {
        [start_str, end_str] => {
            let start = if start_str.is_empty() {
                0
            } else {
                start_str
                    .parse::<usize>()
                    .map_err(|_| format!("detection: invalid range start '{}'", start_str))?
            };
            let end = if end_str.is_empty() {
                usize::MAX
            } else {
                end_str
                    .parse::<usize>()
                    .map_err(|_| format!("detection: invalid range end '{}'", end_str))?
            };
            Ok((start, end))
        }
        _ => Err(format!(
            "detection: range '{}' must be in format 'start:end'",
            s
        )),
    }
}

// ── Operations ──────────────────────────────────────────────────────

/// Filter detections by confidence threshold.
fn confidence_filter(input: &Value, threshold: f32) -> Result<KernelOutput, String> {
    let boxes = input
        .get("boxes")
        .and_then(|v| v.as_array())
        .ok_or("detection: missing 'boxes' array in input")?;

    let scores = input
        .get("scores")
        .and_then(|v| v.as_array())
        .ok_or("detection: missing 'scores' array in input")?;

    if boxes.len() != scores.len() {
        return Err(format!(
            "detection: boxes ({}) and scores ({}) length mismatch",
            boxes.len(),
            scores.len()
        ));
    }

    let mut filtered_boxes = Vec::new();
    let mut filtered_scores = Vec::new();

    for (i, score) in scores.iter().enumerate() {
        let score_val = score.as_f64().unwrap_or(0.0) as f32;
        if score_val >= threshold {
            filtered_boxes.push(boxes[i].clone());
            filtered_scores.push(Value::Number(
                serde_json::Number::from_f64(score_val as f64).unwrap(),
            ));
        }
    }

    debug!(
        "confidence_filter: {} → {} detections (threshold={})",
        boxes.len(),
        filtered_boxes.len(),
        threshold
    );

    Ok(KernelOutput::Json(json!({
        "boxes": filtered_boxes,
        "scores": filtered_scores,
        "count": filtered_boxes.len(),
    })))
}

/// Non-maximum suppression.
fn nms_operation(input: &Value, iou_threshold: f32) -> Result<KernelOutput, String> {
    let boxes = input
        .get("boxes")
        .and_then(|v| v.as_array())
        .ok_or("detection: missing 'boxes' array in input")?;

    let scores = input
        .get("scores")
        .and_then(|v| v.as_array())
        .ok_or("detection: missing 'scores' array in input")?;

    if boxes.len() != scores.len() {
        return Err(format!(
            "detection: boxes ({}) and scores ({}) length mismatch",
            boxes.len(),
            scores.len()
        ));
    }

    // Convert boxes to f32 array.
    let boxes_f32: Result<Vec<[f32; 4]>, String> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let arr = b
                .as_array()
                .ok_or(format!("detection: box {} is not an array", i))?;
            if arr.len() != 4 {
                return Err(format!(
                    "detection: box {} has {} elements, expected 4",
                    i,
                    arr.len()
                ));
            }
            Ok([
                arr[0].as_f64().unwrap_or(0.0) as f32,
                arr[1].as_f64().unwrap_or(0.0) as f32,
                arr[2].as_f64().unwrap_or(0.0) as f32,
                arr[3].as_f64().unwrap_or(0.0) as f32,
            ])
        })
        .collect();
    let boxes_f32 = boxes_f32?;

    // Convert scores to f32.
    let scores_f32: Vec<f32> = scores
        .iter()
        .map(|s| s.as_f64().unwrap_or(0.0) as f32)
        .collect();

    // Run NMS.
    let kept_indices = nms_impl(&boxes_f32, &scores_f32, iou_threshold);

    // Build output.
    let mut output_boxes = Vec::new();
    let mut output_scores = Vec::new();
    let mut output_indices = Vec::new();

    for &idx in &kept_indices {
        output_boxes.push(boxes[idx].clone());
        output_scores.push(Value::Number(
            serde_json::Number::from_f64(scores_f32[idx] as f64).unwrap(),
        ));
        output_indices.push(Value::Number(
            serde_json::Number::from(idx as i64),
        ));
    }

    debug!(
        "nms: {} → {} detections (iou_threshold={})",
        boxes.len(),
        kept_indices.len(),
        iou_threshold
    );

    Ok(KernelOutput::Json(json!({
        "boxes": output_boxes,
        "scores": output_scores,
        "indices": output_indices,
        "count": kept_indices.len(),
    })))
}

/// Convert [center_x, center_y, width, height] → [x1, y1, x2, y2].
fn xywh_to_xyxy_op(input: &Value) -> Result<KernelOutput, String> {
    let boxes = input
        .get("boxes")
        .and_then(|v| v.as_array())
        .ok_or("detection: missing 'boxes' array in input")?;

    let converted: Result<Vec<Value>, String> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let arr = b
                .as_array()
                .ok_or(format!("detection: box {} is not an array", i))?;
            if arr.len() != 4 {
                return Err(format!(
                    "detection: box {} has {} elements, expected 4",
                    i,
                    arr.len()
                ));
            }
            let cx = arr[0].as_f64().unwrap_or(0.0) as f32;
            let cy = arr[1].as_f64().unwrap_or(0.0) as f32;
            let w = arr[2].as_f64().unwrap_or(0.0) as f32;
            let h = arr[3].as_f64().unwrap_or(0.0) as f32;

            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            Ok(Value::Array(vec![
                Value::Number(serde_json::Number::from_f64(x1 as f64).unwrap()),
                Value::Number(serde_json::Number::from_f64(y1 as f64).unwrap()),
                Value::Number(serde_json::Number::from_f64(x2 as f64).unwrap()),
                Value::Number(serde_json::Number::from_f64(y2 as f64).unwrap()),
            ]))
        })
        .collect();

    let converted = converted?;

    // Pass through scores and other fields from input.
    let mut result = json!({ "boxes": converted });
    if let Some(scores) = input.get("scores") {
        result["scores"] = scores.clone();
    }
    if let Some(num_classes) = input.get("num_classes") {
        result["num_classes"] = num_classes.clone();
    }
    if let Some(class_ids) = input.get("class_ids") {
        result["class_ids"] = class_ids.clone();
    }

    Ok(KernelOutput::Json(result))
}

/// Convert [x1, y1, x2, y2] → [center_x, center_y, width, height].
fn xyxy_to_xywh_op(input: &Value) -> Result<KernelOutput, String> {
    let boxes = input
        .get("boxes")
        .and_then(|v| v.as_array())
        .ok_or("detection: missing 'boxes' array in input")?;

    let converted: Result<Vec<Value>, String> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let arr = b
                .as_array()
                .ok_or(format!("detection: box {} is not an array", i))?;
            if arr.len() != 4 {
                return Err(format!(
                    "detection: box {} has {} elements, expected 4",
                    i,
                    arr.len()
                ));
            }
            let x1 = arr[0].as_f64().unwrap_or(0.0) as f32;
            let y1 = arr[1].as_f64().unwrap_or(0.0) as f32;
            let x2 = arr[2].as_f64().unwrap_or(0.0) as f32;
            let y2 = arr[3].as_f64().unwrap_or(0.0) as f32;

            let cx = (x1 + x2) / 2.0;
            let cy = (y1 + y2) / 2.0;
            let w = x2 - x1;
            let h = y2 - y1;

            Ok(Value::Array(vec![
                Value::Number(serde_json::Number::from_f64(cx as f64).unwrap()),
                Value::Number(serde_json::Number::from_f64(cy as f64).unwrap()),
                Value::Number(serde_json::Number::from_f64(w as f64).unwrap()),
                Value::Number(serde_json::Number::from_f64(h as f64).unwrap()),
            ]))
        })
        .collect();

    let converted = converted?;

    // Pass through scores and other fields from input.
    let mut result = json!({ "boxes": converted });
    if let Some(scores) = input.get("scores") {
        result["scores"] = scores.clone();
    }
    if let Some(num_classes) = input.get("num_classes") {
        result["num_classes"] = num_classes.clone();
    }
    if let Some(class_ids) = input.get("class_ids") {
        result["class_ids"] = class_ids.clone();
    }

    Ok(KernelOutput::Json(result))
}

/// Convert raw blob bytes to Vec<f32>.
/// Supports f32 and f64 tensor content types.
fn blob_bytes_to_f32(bytes: &[u8], content_type: &str) -> Result<Vec<f32>, String> {
    match content_type {
        "tensor/f32" => {
            if bytes.len() % 4 != 0 {
                return Err(format!(
                    "detection: f32 blob length {} not divisible by 4",
                    bytes.len()
                ));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        "tensor/f64" => {
            if bytes.len() % 8 != 0 {
                return Err(format!(
                    "detection: f64 blob length {} not divisible by 8",
                    bytes.len()
                ));
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect())
        }
        _ => Err(format!(
            "detection: unsupported blob content type '{}', expected tensor/f32 or tensor/f64",
            content_type
        )),
    }
}

/// Split raw ONNX tensor into boxes and scores.
///
/// Supports both blob input (from ONNX kernel) and JSON input.
/// Blob path: reads raw f32 bytes + shape from blob metadata.
/// JSON path: reads `{"data": [...], "shape": [...]}` from JSON.
fn split_operation(
    input: &KernelInput,
    boxes_range: (usize, usize),
    scores_range: (usize, usize),
    transpose: bool,
) -> Result<KernelOutput, String> {
    // Prefer blob input (zero-copy from ONNX kernel), fall back to JSON.
    let (data_f32, shape) = if let Some(blob) = input.first_blob() {
        let floats = blob_bytes_to_f32(&blob.bytes, &blob.meta.content_type)?;
        let shape = blob
            .meta
            .shape
            .clone()
            .ok_or("detection: split blob missing shape metadata")?;
        debug!(
            shape = ?shape,
            n_floats = floats.len(),
            "detection: split using blob input"
        );
        (floats, shape)
    } else {
        // Fallback: parse from JSON.
        let data = input
            .json
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or("detection: split requires 'data' array (or blob input)")?;

        let shape_val = input
            .json
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or("detection: split requires 'shape' array")?;

        let shape: Vec<usize> = shape_val
            .iter()
            .filter_map(|v| v.as_u64().map(|u| u as usize))
            .collect();

        let data_f32: Vec<f32> = data
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        (data_f32, shape)
    };

    if shape.len() != 3 {
        return Err(format!(
            "detection: expected shape with 3 dimensions, got {}",
            shape.len()
        ));
    }

    let batch = shape[0];
    let dim1 = shape[1]; // For YOLO [1, 84, 8400]: 84 (features)
    let dim2 = shape[2]; // For YOLO [1, 84, 8400]: 8400 (detections)

    // Determine logical layout.
    // transpose=true: physical [batch, features, num_det] → logical [num_det, features]
    // transpose=false: physical [batch, num_det, features]
    let (num_detections, features) = if transpose {
        (dim2, dim1) // [1, 84, 8400] → 8400 detections, 84 features
    } else {
        (dim1, dim2) // [1, num_det, features]
    };

    if data_f32.len() != batch * dim1 * dim2 {
        return Err(format!(
            "detection: data size {} doesn't match shape {:?}",
            data_f32.len(),
            shape
        ));
    }

    // Index into physical row-major [dim1, dim2] data.
    // transpose: element at logical [det, feat] is at physical [feat, det] → feat * dim2 + det
    // no transpose: element at logical [det, feat] is at physical [det, feat] → det * dim2 + feat
    let phys_idx = |det_idx: usize, feat_idx: usize| -> usize {
        if transpose {
            feat_idx * dim2 + det_idx
        } else {
            det_idx * dim2 + feat_idx
        }
    };

    // Extract boxes and scores.
    let mut boxes = Vec::new();
    let mut scores = Vec::new();

    for det_idx in 0..num_detections {
        // Extract boxes in the specified range.
        let mut box_data = Vec::new();
        for feat_idx in boxes_range.0..boxes_range.1.min(features) {
            let idx = phys_idx(det_idx, feat_idx);
            if idx < data_f32.len() {
                box_data.push(data_f32[idx]);
            }
        }
        if !box_data.is_empty() {
            boxes.push(Value::Array(
                box_data
                    .iter()
                    .map(|&v| Value::Number(serde_json::Number::from_f64(v as f64).unwrap()))
                    .collect(),
            ));
        }

        // Extract scores in the specified range.
        let mut score_data = Vec::new();
        for feat_idx in scores_range.0..scores_range.1.min(features) {
            let idx = phys_idx(det_idx, feat_idx);
            if idx < data_f32.len() {
                score_data.push(data_f32[idx]);
            }
        }
        if !score_data.is_empty() {
            // For scores, use the max confidence (common in YOLO).
            let max_score = score_data
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            scores.push(Value::Number(
                serde_json::Number::from_f64(max_score as f64).unwrap(),
            ));
        }
    }

    debug!(
        "split: extracted {} boxes and {} scores from shape {:?}",
        boxes.len(),
        scores.len(),
        shape
    );

    Ok(KernelOutput::Json(json!({
        "boxes": boxes,
        "scores": scores,
        "num_classes": (scores_range.1 - scores_range.0),
    })))
}

/// Format final detection output.
fn format_operation(
    input: &Value,
    output_format: &str,
    labels: Option<Vec<String>>,
) -> Result<KernelOutput, String> {
    let boxes = input
        .get("boxes")
        .and_then(|v| v.as_array())
        .ok_or("detection: missing 'boxes' array in input")?;

    let scores = input
        .get("scores")
        .and_then(|v| v.as_array())
        .ok_or("detection: missing 'scores' array in input")?;

    let class_ids = input
        .get("class_ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|u| u as usize))
                .collect::<Vec<_>>()
        });

    if boxes.len() != scores.len() {
        return Err(format!(
            "detection: boxes ({}) and scores ({}) length mismatch",
            boxes.len(),
            scores.len()
        ));
    }

    match output_format {
        "json" => {
            let mut detections = Vec::new();
            for (i, (box_val, score_val)) in boxes.iter().zip(scores.iter()).enumerate() {
                let mut det = serde_json::Map::new();

                if let Some(arr) = box_val.as_array() {
                    det.insert("box".to_string(), Value::Array(arr.clone()));
                } else {
                    det.insert("box".to_string(), box_val.clone());
                }

                if let Some(score) = score_val.as_f64() {
                    det.insert("confidence".to_string(), Value::Number(
                        serde_json::Number::from_f64(score).unwrap(),
                    ));
                }

                if let Some(class_ids) = &class_ids {
                    if i < class_ids.len() {
                        let class_id = class_ids[i];
                        det.insert("class_id".to_string(), Value::Number(
                            serde_json::Number::from(class_id as i64),
                        ));
                        if let Some(labels) = &labels {
                            if class_id < labels.len() {
                                det.insert("label".to_string(), Value::String(labels[class_id].clone()));
                            }
                        }
                    }
                }

                detections.push(Value::Object(det));
            }
            Ok(KernelOutput::Json(Value::Array(detections)))
        }
        "coco" => {
            let mut coco_dets = Vec::new();
            for (i, (box_val, score_val)) in boxes.iter().zip(scores.iter()).enumerate() {
                let mut det = serde_json::Map::new();

                if let Some(arr) = box_val.as_array() {
                    if arr.len() >= 4 {
                        let x = arr[0].as_f64().unwrap_or(0.0);
                        let y = arr[1].as_f64().unwrap_or(0.0);
                        let w = arr[2].as_f64().unwrap_or(0.0);
                        let h = arr[3].as_f64().unwrap_or(0.0);
                        det.insert(
                            "bbox".to_string(),
                            Value::Array(vec![
                                Value::Number(serde_json::Number::from_f64(x).unwrap()),
                                Value::Number(serde_json::Number::from_f64(y).unwrap()),
                                Value::Number(serde_json::Number::from_f64(w).unwrap()),
                                Value::Number(serde_json::Number::from_f64(h).unwrap()),
                            ]),
                        );
                    }
                }

                if let Some(score) = score_val.as_f64() {
                    det.insert("score".to_string(), Value::Number(
                        serde_json::Number::from_f64(score).unwrap(),
                    ));
                }

                if let Some(class_ids) = &class_ids {
                    if i < class_ids.len() {
                        det.insert("category_id".to_string(), Value::Number(
                            serde_json::Number::from(class_ids[i] as i64),
                        ));
                    }
                }

                coco_dets.push(Value::Object(det));
            }
            Ok(KernelOutput::Json(Value::Array(coco_dets)))
        }
        _ => Err(format!(
            "detection: unknown output format '{}', use 'json' or 'coco'",
            output_format
        )),
    }
}

/// Soft non-maximum suppression (Bodla et al., 2017).
///
/// Instead of hard elimination, decays overlapping box scores:
///   - Linear:   s_i *= (1 - IoU) when IoU > threshold
///   - Gaussian: s_i *= exp(-IoU² / sigma)
///
/// Preserves more detections in crowded scenes, typically +1-2% mAP improvement.
fn soft_nms_operation(
    input: &Value,
    iou_threshold: f32,
    sigma: f32,
    method: SoftNmsMethod,
    min_score: f32,
) -> Result<KernelOutput, String> {
    let boxes_json = input
        .get("boxes")
        .and_then(|v| v.as_array())
        .ok_or("detection: missing 'boxes' array in input")?;

    let scores_json = input
        .get("scores")
        .and_then(|v| v.as_array())
        .ok_or("detection: missing 'scores' array in input")?;

    if boxes_json.len() != scores_json.len() {
        return Err(format!(
            "detection: boxes ({}) and scores ({}) length mismatch",
            boxes_json.len(),
            scores_json.len()
        ));
    }

    let boxes_f32: Result<Vec<[f32; 4]>, String> = boxes_json
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let arr = b
                .as_array()
                .ok_or(format!("detection: box {} is not an array", i))?;
            if arr.len() != 4 {
                return Err(format!("detection: box {} has {} elements, expected 4", i, arr.len()));
            }
            Ok([
                arr[0].as_f64().unwrap_or(0.0) as f32,
                arr[1].as_f64().unwrap_or(0.0) as f32,
                arr[2].as_f64().unwrap_or(0.0) as f32,
                arr[3].as_f64().unwrap_or(0.0) as f32,
            ])
        })
        .collect();
    let boxes_f32 = boxes_f32?;

    let mut scores: Vec<f32> = scores_json
        .iter()
        .map(|s| s.as_f64().unwrap_or(0.0) as f32)
        .collect();

    let n = boxes_f32.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut kept = Vec::new();
    let mut kept_scores = Vec::new();

    while !order.is_empty() {
        // Find index with maximum score among remaining
        let best_pos = order
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                scores[**a]
                    .partial_cmp(&scores[**b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(pos, _)| pos)
            .unwrap();

        let best_idx = order.remove(best_pos);
        kept.push(best_idx);
        kept_scores.push(scores[best_idx]);

        // Decay scores of remaining boxes
        order.retain(|&i| {
            let iou_val = iou(&boxes_f32[best_idx], &boxes_f32[i]);

            match method {
                SoftNmsMethod::Linear => {
                    if iou_val > iou_threshold {
                        scores[i] *= 1.0 - iou_val;
                    }
                }
                SoftNmsMethod::Gaussian => {
                    scores[i] *= (-iou_val * iou_val / sigma).exp();
                }
            }

            scores[i] >= min_score
        });
    }

    debug!(
        "soft_nms: {} → {} detections (method={:?}, sigma={})",
        n,
        kept.len(),
        method,
        sigma
    );

    let mut output_boxes = Vec::new();
    let mut output_scores = Vec::new();
    let mut output_indices = Vec::new();

    for (pos, &idx) in kept.iter().enumerate() {
        output_boxes.push(boxes_json[idx].clone());
        output_scores.push(Value::Number(
            serde_json::Number::from_f64(kept_scores[pos] as f64).unwrap(),
        ));
        output_indices.push(Value::Number(serde_json::Number::from(idx as i64)));
    }

    Ok(KernelOutput::Json(json!({
        "boxes": output_boxes,
        "scores": output_scores,
        "indices": output_indices,
        "count": kept.len(),
    })))
}

/// Per-class NMS using coordinate offset trick (torchvision.ops.batched_nms).
///
/// Shifts each class's boxes into non-overlapping coordinate space so standard
/// NMS naturally handles per-class suppression in a single pass:
///   offset = class_id * (max_coordinate + 1)
///   shifted_box = box + offset
fn batched_nms_operation(
    input: &Value,
    iou_threshold: f32,
) -> Result<KernelOutput, String> {
    let boxes_json = input
        .get("boxes")
        .and_then(|v| v.as_array())
        .ok_or("detection: missing 'boxes' array in input")?;

    let scores_json = input
        .get("scores")
        .and_then(|v| v.as_array())
        .ok_or("detection: missing 'scores' array in input")?;

    let class_ids_json = input
        .get("class_ids")
        .and_then(|v| v.as_array())
        .ok_or("detection: batched_nms requires 'class_ids' array")?;

    let n = boxes_json.len();
    if scores_json.len() != n || class_ids_json.len() != n {
        return Err(format!(
            "detection: boxes ({}), scores ({}), class_ids ({}) length mismatch",
            n, scores_json.len(), class_ids_json.len()
        ));
    }

    let boxes_f32: Result<Vec<[f32; 4]>, String> = boxes_json
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let arr = b
                .as_array()
                .ok_or(format!("detection: box {} is not an array", i))?;
            if arr.len() != 4 {
                return Err(format!("detection: box {} has {} elements, expected 4", i, arr.len()));
            }
            Ok([
                arr[0].as_f64().unwrap_or(0.0) as f32,
                arr[1].as_f64().unwrap_or(0.0) as f32,
                arr[2].as_f64().unwrap_or(0.0) as f32,
                arr[3].as_f64().unwrap_or(0.0) as f32,
            ])
        })
        .collect();
    let boxes_f32 = boxes_f32?;

    let scores_f32: Vec<f32> = scores_json
        .iter()
        .map(|s| s.as_f64().unwrap_or(0.0) as f32)
        .collect();

    let class_ids: Vec<usize> = class_ids_json
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect();

    // Find max coordinate for offset calculation
    let max_coord = boxes_f32
        .iter()
        .flat_map(|b| b.iter())
        .copied()
        .fold(0.0f32, f32::max);

    // Apply class offset to create non-overlapping coordinate spaces
    let offset_boxes: Vec<[f32; 4]> = boxes_f32
        .iter()
        .zip(class_ids.iter())
        .map(|(b, &cid)| {
            let offset = (cid as f32) * (max_coord + 1.0);
            [b[0] + offset, b[1] + offset, b[2] + offset, b[3] + offset]
        })
        .collect();

    // Run standard NMS on offset boxes
    let kept_indices = nms_impl(&offset_boxes, &scores_f32, iou_threshold);

    debug!(
        "batched_nms: {} → {} detections (iou_threshold={})",
        n,
        kept_indices.len(),
        iou_threshold
    );

    // Build output with original (non-offset) boxes
    let mut output_boxes = Vec::new();
    let mut output_scores = Vec::new();
    let mut output_indices = Vec::new();
    let mut output_class_ids = Vec::new();

    for &idx in &kept_indices {
        output_boxes.push(boxes_json[idx].clone());
        output_scores.push(Value::Number(
            serde_json::Number::from_f64(scores_f32[idx] as f64).unwrap(),
        ));
        output_indices.push(Value::Number(serde_json::Number::from(idx as i64)));
        output_class_ids.push(Value::Number(serde_json::Number::from(class_ids[idx] as i64)));
    }

    Ok(KernelOutput::Json(json!({
        "boxes": output_boxes,
        "scores": output_scores,
        "indices": output_indices,
        "class_ids": output_class_ids,
        "count": kept_indices.len(),
    })))
}

// ── NMS Implementation ──────────────────────────────────────────────

/// Non-maximum suppression — optimized.
///
/// Improvements over naive O(n²):
/// 1. Pre-compute box areas (avoid redundant area recalculation)
/// 2. Coordinate-based early rejection (no overlap → skip IoU)
/// 3. Only iterate unsuppressed candidates in inner loop
/// 4. Inline IoU with pre-computed areas
///
/// Returns indices of kept detections in score-descending order.
fn nms_impl(boxes: &[[f32; 4]], scores: &[f32], iou_threshold: f32) -> Vec<usize> {
    let n = boxes.len();
    if n == 0 || scores.is_empty() {
        return Vec::new();
    }

    // Pre-compute areas once — O(n) instead of O(n²) redundant calculations.
    let areas: Vec<f32> = boxes
        .iter()
        .map(|b| (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0))
        .collect();

    // Score-sorted indices (descending).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept = Vec::with_capacity(n.min(256));
    let mut suppressed = vec![false; n];

    for &i in &order {
        if suppressed[i] {
            continue;
        }
        kept.push(i);

        let bi = &boxes[i];
        let area_i = areas[i];

        // Suppress overlapping boxes. Only check those after current
        // in sorted order (lower score) that aren't already suppressed.
        for &j in &order {
            if suppressed[j] || j == i {
                continue;
            }

            let bj = &boxes[j];

            // Fast coordinate rejection: if boxes don't overlap at all,
            // IoU is 0 — skip the full calculation.
            if bj[0] >= bi[2] || bj[2] <= bi[0] || bj[1] >= bi[3] || bj[3] <= bi[1] {
                continue;
            }

            // Inline IoU with pre-computed areas.
            let inter_w = bi[2].min(bj[2]) - bi[0].max(bj[0]);
            let inter_h = bi[3].min(bj[3]) - bi[1].max(bj[1]);
            let inter_area = inter_w * inter_h; // Already know > 0 from check above.
            let union_area = area_i + areas[j] - inter_area;
            let iou_val = inter_area / (union_area + 1e-6);

            if iou_val > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    kept
}

/// Compute Intersection over Union for boxes in [x1, y1, x2, y2] format.
#[inline]
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter_x1 = a[0].max(b[0]);
    let inter_y1 = a[1].max(b[1]);
    let inter_x2 = a[2].min(b[2]);
    let inter_y2 = a[3].min(b[3]);

    let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union_area = area_a + area_b - inter_area;

    inter_area / (union_area + 1e-6)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_filter() {
        let input = json!({
            "boxes": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            "scores": [0.9, 0.1, 0.8],
        });

        let result = confidence_filter(&input, 0.5).expect("confidence_filter failed");
        let boxes = result.get("boxes").unwrap().as_array().unwrap();
        let scores = result.get("scores").unwrap().as_array().unwrap();

        assert_eq!(boxes.len(), 2, "Expected 2 boxes after filtering");
        assert_eq!(scores.len(), 2, "Expected 2 scores after filtering");
    }

    #[test]
    fn test_nms_basic() {
        // 3 overlapping boxes, highest score should survive with highest second highest.
        let input = json!({
            "boxes": [
                [10.0, 10.0, 50.0, 50.0],  // score 0.9
                [15.0, 15.0, 55.0, 55.0],  // score 0.8 - overlaps with first, should be suppressed
                [100.0, 100.0, 150.0, 150.0],  // score 0.7 - no overlap, should survive
            ],
            "scores": [0.9, 0.8, 0.7],
        });

        let result = nms_operation(&input, 0.5).expect("nms failed");
        let kept_indices = result
            .get("indices")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_u64())
            .collect::<Vec<_>>();

        assert_eq!(kept_indices.len(), 2, "Expected 2 boxes after NMS");
        assert_eq!(kept_indices[0], 0, "First box should be kept");
        assert_eq!(kept_indices[1], 2, "Third box should be kept");
    }

    #[test]
    fn test_nms_no_overlap() {
        // Non-overlapping boxes should all survive.
        let input = json!({
            "boxes": [
                [10.0, 10.0, 50.0, 50.0],
                [100.0, 100.0, 150.0, 150.0],
                [200.0, 200.0, 250.0, 250.0],
            ],
            "scores": [0.9, 0.8, 0.7],
        });

        let result = nms_operation(&input, 0.5).expect("nms failed");
        let count = result.get("count").unwrap().as_u64().unwrap();

        assert_eq!(count, 3, "All non-overlapping boxes should survive");
    }

    #[test]
    fn test_xywh_to_xyxy() {
        let input = json!({
            "boxes": [[30.0, 30.0, 40.0, 40.0]],  // center (30,30) with w=40, h=40
        });

        let result = xywh_to_xyxy_op(&input).expect("xywh_to_xyxy failed");
        let boxes = result.get("boxes").unwrap().as_array().unwrap();
        let box_arr = boxes[0].as_array().unwrap();

        assert!((box_arr[0].as_f64().unwrap() - 10.0).abs() < 1e-5, "x1 should be 10.0");
        assert!((box_arr[1].as_f64().unwrap() - 10.0).abs() < 1e-5, "y1 should be 10.0");
        assert!((box_arr[2].as_f64().unwrap() - 50.0).abs() < 1e-5, "x2 should be 50.0");
        assert!((box_arr[3].as_f64().unwrap() - 50.0).abs() < 1e-5, "y2 should be 50.0");
    }

    #[test]
    fn test_xyxy_to_xywh() {
        let input = json!({
            "boxes": [[10.0, 10.0, 50.0, 50.0]],  // x1=10, y1=10, x2=50, y2=50
        });

        let result = xyxy_to_xywh_op(&input).expect("xyxy_to_xywh failed");
        let boxes = result.get("boxes").unwrap().as_array().unwrap();
        let box_arr = boxes[0].as_array().unwrap();

        assert!((box_arr[0].as_f64().unwrap() - 30.0).abs() < 1e-5, "cx should be 30.0");
        assert!((box_arr[1].as_f64().unwrap() - 30.0).abs() < 1e-5, "cy should be 30.0");
        assert!((box_arr[2].as_f64().unwrap() - 40.0).abs() < 1e-5, "w should be 40.0");
        assert!((box_arr[3].as_f64().unwrap() - 40.0).abs() < 1e-5, "h should be 40.0");
    }

    #[test]
    fn test_iou_calculation() {
        // Two identical boxes should have IoU = 1.0.
        let box1 = [10.0, 10.0, 50.0, 50.0];
        let box2 = [10.0, 10.0, 50.0, 50.0];
        let iou_val = iou(&box1, &box2);
        assert!((iou_val - 1.0).abs() < 1e-5, "Identical boxes should have IoU = 1.0");

        // Non-overlapping boxes should have IoU = 0.0.
        let box3 = [100.0, 100.0, 150.0, 150.0];
        let iou_val2 = iou(&box1, &box3);
        assert!(iou_val2 < 1e-5, "Non-overlapping boxes should have IoU ≈ 0.0");

        // Partial overlap.
        let box4 = [30.0, 30.0, 70.0, 70.0];  // overlaps with box1
        let iou_val3 = iou(&box1, &box4);
        assert!(
            iou_val3 > 0.0 && iou_val3 < 1.0,
            "Partial overlap should have 0 < IoU < 1"
        );
    }

    #[test]
    fn test_split_operation_basic() {
        // Shape [1, 2, 5]: batch=1, 2 detections, 5 features each (no transpose).
        // Data layout (row-major): det0=[1,2,3,4,5], det1=[6,7,8,9,10]
        // boxes=0..4 → [1,2,3,4] and [6,7,8,9], scores=4..5 → [5] and [10]
        let json = json!({
            "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "shape": [1, 2, 5],
        });
        let input = KernelInput::from_json(json);

        let result = split_operation(&input, (0, 4), (4, 5), false).expect("split failed");
        let boxes = result.get("boxes").unwrap().as_array().unwrap();
        let scores = result.get("scores").unwrap().as_array().unwrap();

        assert_eq!(boxes.len(), 2, "Expected 2 boxes");
        assert_eq!(scores.len(), 2, "Expected 2 scores");
        // Verify first detection box values.
        let box0: Vec<f64> = boxes[0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(box0, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_split_operation_transposed() {
        // Shape [1, 5, 2]: batch=1, 5 features, 2 detections (transpose=true).
        // Physical layout: feat0=[1,2], feat1=[3,4], feat2=[5,6], feat3=[7,8], feat4=[9,10]
        // After transpose: det0=[1,3,5,7,9], det1=[2,4,6,8,10]
        // boxes=0..4 → [1,3,5,7] and [2,4,6,8], scores=4..5 → [9] and [10]
        let json = json!({
            "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "shape": [1, 5, 2],
        });
        let input = KernelInput::from_json(json);

        let result = split_operation(&input, (0, 4), (4, 5), true).expect("split failed");
        let boxes = result.get("boxes").unwrap().as_array().unwrap();
        let scores = result.get("scores").unwrap().as_array().unwrap();

        assert_eq!(boxes.len(), 2, "Expected 2 detections after transpose");
        assert_eq!(scores.len(), 2, "Expected 2 scores after transpose");
        // Verify transposed access: det0 gets features [0,1,2,3] via physical indices [0,2,4,6].
        let box0: Vec<f64> = boxes[0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(box0, vec![1.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_format_json_output() {
        let input = json!({
            "boxes": [[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 150.0, 150.0]],
            "scores": [0.9, 0.8],
            "class_ids": [0, 1],
        });

        let labels = Some(vec!["cat".to_string(), "dog".to_string()]);
        let result = format_operation(&input, "json", labels).expect("format failed");
        let dets = result.as_array().unwrap();

        assert_eq!(dets.len(), 2, "Expected 2 detections");
        assert_eq!(dets[0].get("label").unwrap().as_str().unwrap(), "cat");
        assert_eq!(dets[1].get("label").unwrap().as_str().unwrap(), "dog");
    }

    #[test]
    fn test_soft_nms_gaussian() {
        // Two overlapping boxes - soft NMS should keep both but decay the second
        let input = json!({
            "boxes": [
                [10.0, 10.0, 50.0, 50.0],
                [15.0, 15.0, 55.0, 55.0],
                [100.0, 100.0, 150.0, 150.0],
            ],
            "scores": [0.9, 0.8, 0.7],
        });

        let result = soft_nms_operation(&input, 0.3, 0.5, SoftNmsMethod::Gaussian, 0.001)
            .expect("soft_nms failed");

        let count = result.get("count").unwrap().as_u64().unwrap();
        // Soft NMS should keep all 3 (scores decayed but above min_score)
        assert!(count >= 2, "Soft NMS should keep at least 2 detections, got {}", count);
    }

    #[test]
    fn test_soft_nms_linear() {
        let input = json!({
            "boxes": [
                [10.0, 10.0, 50.0, 50.0],
                [12.0, 12.0, 52.0, 52.0],
            ],
            "scores": [0.9, 0.3],
        });

        let result = soft_nms_operation(&input, 0.3, 0.5, SoftNmsMethod::Linear, 0.2)
            .expect("soft_nms failed");

        let scores = result.get("scores").unwrap().as_array().unwrap();
        // First box should keep its score, second should be decayed
        assert!(scores[0].as_f64().unwrap() > 0.8);
    }

    #[test]
    fn test_batched_nms() {
        // Two boxes overlap but are different classes - both should survive
        let input = json!({
            "boxes": [
                [10.0, 10.0, 50.0, 50.0],
                [15.0, 15.0, 55.0, 55.0],
                [100.0, 100.0, 150.0, 150.0],
            ],
            "scores": [0.9, 0.8, 0.7],
            "class_ids": [0, 1, 0],
        });

        let result = batched_nms_operation(&input, 0.5).expect("batched_nms failed");
        let count = result.get("count").unwrap().as_u64().unwrap();

        // Boxes 0 and 1 are different classes so both survive; box 2 is same class as 0 but no overlap
        assert_eq!(count, 3, "All boxes should survive batched NMS (different classes or no overlap)");
    }

    #[test]
    fn test_batched_nms_same_class_suppression() {
        // Two overlapping boxes, same class - one should be suppressed
        let input = json!({
            "boxes": [
                [10.0, 10.0, 50.0, 50.0],
                [15.0, 15.0, 55.0, 55.0],
            ],
            "scores": [0.9, 0.8],
            "class_ids": [0, 0],
        });

        let result = batched_nms_operation(&input, 0.5).expect("batched_nms failed");
        let count = result.get("count").unwrap().as_u64().unwrap();

        // Same class, overlapping → one suppressed
        assert_eq!(count, 1, "Overlapping same-class boxes should be suppressed");
    }
}
