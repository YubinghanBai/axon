//! ONNX compute kernel: raw tensor inference via ONNX Runtime.
//!
//! Enabled via `cargo build --features onnx`.
//!
//! Raw-only kernel: tensor in → ONNX Runtime → tensor out.
//! No business logic, no pre/post processing.
//! Pre/post processing belongs in DSL composition (quickjs/wasm tasks).
//!
//! Input format: JSON keys matching model input names.
//!   - Nested arrays (auto-detect shape): `{"x": [[1.0, 2.0], [3.0, 4.0]]}`
//!   - Explicit spec: `{"x": {"shape": [2, 2], "data": [1.0, 2.0, 3.0, 4.0]}}`
//!
//! Output format:
//!   `{"outputs": {"name": {"shape": [1, 384], "data": [0.1, ...]}}}`
//!
//! Config (operations):
//!   - String: `"path/to/model.onnx"`
//!   - Object: `{"model": "path/to/model.onnx"}`

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{ArrayD, IxDyn};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use parking_lot::{Mutex, RwLock};
use serde_json::Value;
use tracing::{debug, info};

use crate::error::AxonError;
use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};

/// ONNX compute kernel with session caching.
///
/// Each unique model path is loaded once and cached (double-check locking).
/// Sessions are wrapped in Mutex because `Session::run` requires `&mut self`.
pub struct OnnxKernel {
    sessions: RwLock<HashMap<PathBuf, Arc<Mutex<Session>>>>,
    /// Cache: whether a model supports dynamic batch dimension.
    /// `true` = tensor batching works, `false` = must use serial fallback.
    /// Discovered lazily on first batch attempt per model.
    batch_supported: RwLock<HashMap<PathBuf, bool>>,
}

/// Config parsed from the operations JSON.
#[derive(Debug)]
struct OnnxConfig {
    model_path: PathBuf,
    /// When true, output raw f32 bytes as KernelOutput::Blob
    /// for zero-copy downstream tensor pipelines.
    blob_output: bool,
    /// Execution provider / device: "cpu", "coreml", "cuda", "tensorrt", "directml".
    device: String,
}

impl OnnxKernel {
    pub fn new() -> Result<Self, String> {
        ort::init().commit();
        Ok(Self {
            sessions: RwLock::new(HashMap::new()),
            batch_supported: RwLock::new(HashMap::new()),
        })
    }

    /// Get a cached session or load from file.
    ///
    /// The `device` parameter selects the ONNX Runtime Execution Provider:
    /// - `"cpu"` (default): CPU execution
    /// - `"coreml"`: Apple CoreML (macOS/iOS, Neural Engine + GPU + CPU)
    /// - `"cuda"`: NVIDIA CUDA
    /// - `"tensorrt"`: NVIDIA TensorRT (requires CUDA)
    /// - `"directml"`: DirectX 12 GPU (Windows)
    ///
    /// Cache key includes device to avoid sharing sessions across EPs.
    fn get_or_load(&self, path: &Path, device: &str) -> Result<Arc<Mutex<Session>>, String> {
        // Cache key: path + device suffix to avoid EP conflicts.
        let cache_key = if device == "cpu" {
            path.to_path_buf()
        } else {
            let mut key = path.to_path_buf();
            let new_name = format!(
                "{}.{}",
                key.file_stem().unwrap_or_default().to_string_lossy(),
                device
            );
            key.set_file_name(new_name);
            key
        };

        // Fast path: read lock.
        if let Some(s) = self.sessions.read().get(&cache_key) {
            return Ok(Arc::clone(s));
        }
        // Slow path: write lock + double check.
        let mut cache = self.sessions.write();
        if let Some(s) = cache.get(&cache_key) {
            return Ok(Arc::clone(s));
        }
        info!(model = %path.display(), device, "onnx: loading model");

        let builder = Session::builder()
            .map_err(|e| format!("onnx: session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("onnx: optimization: {e}"))?;

        // Configure execution provider based on device.
        let mut builder = configure_execution_provider(builder, device)?;

        let session = builder
            .commit_from_file(path)
            .map_err(|e| format!("onnx: load '{}': {e}", path.display()))?;
        let session = Arc::new(Mutex::new(session));
        cache.insert(cache_key, Arc::clone(&session));
        Ok(session)
    }
}

impl ComputeKernel for OnnxKernel {
    fn name(&self) -> &str {
        "onnx"
    }

    fn execute(
        &self,
        input: KernelInput,
        operations: Value,
    ) -> Result<KernelOutput, AxonError> {
        let config = parse_config(&operations)?;
        let session_mutex = self.get_or_load(&config.model_path, &config.device)?;

        info!(
            model = %config.model_path.display(),
            has_blobs = input.has_blobs(),
            "onnx: running inference"
        );

        // Lock session (run requires &mut self).
        let mut session = session_mutex.lock();

        // Collect model input/output names before building inputs.
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        // Build input tensors: prefer blob input (raw bytes), fallback to JSON.
        let input_values = if input.has_blobs() && input_names.len() == 1 {
            // Single-input model with blob data: build tensor directly from raw bytes.
            build_inputs_from_blob(&input_names, &input)?
        } else {
            // Standard JSON path (or multi-input model).
            let json = input.into_json();
            build_inputs(&input_names, &json)?
        };

        // Run inference.
        let outputs = session
            .run(input_values)
            .map_err(|e| format!("onnx: inference: {e}"))?;

        extract_single_output(&output_names, &outputs, &config)
    }

    fn supports_batch(&self) -> bool {
        true
    }

    /// True tensor-level batching: concatenate N inputs along dim 0,
    /// run a single session.run(), split outputs back to N results.
    ///
    /// Requirements:
    /// - Single-input model (multi-input falls back to serial)
    /// - All inputs must be blob type with compatible shapes
    /// - Model must accept dynamic batch dimension
    ///
    /// Batch compatibility is cached per model path: on first attempt,
    /// if batching fails (model lacks dynamic batch dim), the model is
    /// marked incompatible and future calls skip directly to serial.
    fn execute_batch(
        &self,
        inputs: Vec<KernelInput>,
        operations: Value,
    ) -> Result<Vec<KernelOutput>, AxonError> {
        let n = inputs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            return Ok(vec![self.execute(inputs.into_iter().next().unwrap(), operations)?]);
        }

        let config = parse_config(&operations)?;

        // Check batch compatibility cache.
        if let Some(&supported) = self.batch_supported.read().get(&config.model_path) {
            if !supported {
                debug!(
                    model = %config.model_path.display(),
                    "onnx: cached as batch-incompatible, using serial"
                );
                return inputs
                    .into_iter()
                    .map(|input| self.execute(input, operations.clone()))
                    .collect();
            }
        }

        let session_mutex = self.get_or_load(&config.model_path, &config.device)?;
        let mut session = session_mutex.lock();

        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        // Only attempt tensor batching for single-input models with all-blob inputs.
        let all_blobs = inputs.iter().all(|inp| inp.has_blobs()) && input_names.len() == 1;
        if !all_blobs {
            debug!("onnx: batch fallback to serial (non-blob or multi-input)");
            drop(session);
            return inputs
                .into_iter()
                .map(|input| self.execute(input, operations.clone()))
                .collect();
        }

        info!(
            model = %config.model_path.display(),
            batch_size = n,
            "onnx: running batched inference"
        );

        // Build concatenated tensor along batch dim 0.
        let batched_inputs = build_batched_inputs_from_blobs(&input_names, &inputs, n)?;

        // Single inference call. Process outputs while session is held,
        // then release session before any serial fallback.
        let batch_result: Result<Vec<KernelOutput>, String> = {
            match session.run(batched_inputs) {
                Ok(outputs) => {
                    let r = split_batch_outputs(&output_names, &outputs, n, &config)
                        .map_err(|e| e.to_string());
                    // outputs dropped here (before session).
                    r
                }
                Err(e) => Err(e.to_string()),
            }
        };
        // Session lock released here.
        drop(session);

        match batch_result {
            Ok(outputs) => {
                self.batch_supported
                    .write()
                    .insert(config.model_path.clone(), true);
                Ok(outputs)
            }
            Err(e) => {
                info!(
                    model = %config.model_path.display(),
                    error = %e,
                    "onnx: batch inference failed, caching as incompatible"
                );
                self.batch_supported
                    .write()
                    .insert(config.model_path.clone(), false);
                // Fall back to serial execution.
                inputs
                    .into_iter()
                    .map(|input| self.execute(input, operations.clone()))
                    .collect()
            }
        }
    }
}

// ── Output extraction helpers ─────────────────────────────────

/// Extract output from a single (non-batched) inference run.
fn extract_single_output(
    output_names: &[String],
    outputs: &ort::session::SessionOutputs<'_>,
    config: &OnnxConfig,
) -> Result<KernelOutput, AxonError> {
    // Blob output mode: emit first f32 output as raw bytes.
    if config.blob_output {
        for (i, name) in output_names.iter().enumerate() {
            if let Ok(view) = outputs[i].try_extract_array::<f32>() {
                let shape: Vec<usize> = view.shape().to_vec();
                let data: Vec<u8> = view
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                debug!(output = %name, shape = ?shape, bytes = data.len(), "onnx: blob output");
                return Ok(KernelOutput::Blob {
                    data,
                    content_type: "tensor/f32".to_string(),
                    shape: Some(shape),
                });
            }
        }
        return Err("onnx: blob_output requested but no f32 outputs found".into());
    }

    // Standard JSON output mode.
    let mut result = serde_json::Map::new();
    for (i, name) in output_names.iter().enumerate() {
        if let Ok(view) = outputs[i].try_extract_array::<f32>() {
            let shape: Vec<usize> = view.shape().to_vec();
            let data: Vec<f32> = view.iter().copied().collect();
            result.insert(
                name.clone(),
                serde_json::json!({"shape": shape, "data": data}),
            );
        } else if let Ok(view) = outputs[i].try_extract_array::<i64>() {
            let shape: Vec<usize> = view.shape().to_vec();
            let data: Vec<i64> = view.iter().copied().collect();
            result.insert(
                name.clone(),
                serde_json::json!({"shape": shape, "data": data}),
            );
        } else {
            return Err(format!(
                "onnx: unsupported output dtype for '{name}' (only f32/i64 supported)"
            )
            .into());
        }
    }

    Ok(KernelOutput::Json(serde_json::json!({"outputs": result})))
}

// ── Batch tensor helpers ──────────────────────────────────────

/// Concatenate N blob inputs into a single batched tensor along dim 0.
///
/// Each blob must have shape `[1, D1, D2, ...]` with matching inner dims.
/// Result: single tensor with shape `[N, D1, D2, ...]`.
fn build_batched_inputs_from_blobs<'v>(
    input_names: &[String],
    inputs: &[KernelInput],
    batch_size: usize,
) -> Result<Vec<(String, SessionInputValue<'v>)>, String> {
    let name = &input_names[0];
    let mut all_bytes: Vec<&[u8]> = Vec::with_capacity(batch_size);
    let mut inner_shape: Option<Vec<usize>> = None;
    let mut content_type: Option<&str> = None;

    for (i, input) in inputs.iter().enumerate() {
        let blob = input
            .first_blob()
            .ok_or_else(|| format!("onnx: batch item {i} has no blob"))?;
        let shape = blob
            .meta
            .shape
            .as_ref()
            .ok_or_else(|| format!("onnx: batch item {i} missing shape metadata"))?;

        // Inner shape = shape[1..] (strip the batch dimension).
        let item_inner: Vec<usize> = if shape.len() > 1 {
            shape[1..].to_vec()
        } else {
            vec![]
        };

        if let Some(ref expected) = inner_shape {
            if &item_inner != expected {
                return Err(format!(
                    "onnx: batch shape mismatch at item {i}: expected inner {:?}, got {:?}",
                    expected, item_inner
                ));
            }
        } else {
            inner_shape = Some(item_inner);
        }

        if let Some(expected_ct) = content_type {
            if blob.meta.content_type != expected_ct {
                return Err(format!(
                    "onnx: batch dtype mismatch at item {i}: expected {expected_ct}, got {}",
                    blob.meta.content_type
                ));
            }
        } else {
            content_type = Some(&blob.meta.content_type);
        }

        all_bytes.push(&blob.bytes);
    }

    let inner_shape = inner_shape.unwrap_or_default();
    let ct = content_type.unwrap_or("tensor/f32");

    // Batched shape: [N, D1, D2, ...]
    let mut batched_shape = vec![batch_size];
    batched_shape.extend_from_slice(&inner_shape);

    // Concatenate raw bytes (works because all tensors share the same inner shape).
    let total_bytes: usize = all_bytes.iter().map(|b| b.len()).sum();
    let mut cat_bytes = Vec::with_capacity(total_bytes);
    for bytes in &all_bytes {
        cat_bytes.extend_from_slice(bytes);
    }

    match ct {
        "tensor/f32" => {
            if cat_bytes.len() % 4 != 0 {
                return Err("onnx: batch f32 blob size not aligned to 4 bytes".into());
            }
            let data: Vec<f32> = cat_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            validate_shape(&batched_shape, data.len())?;
            let array = ArrayD::<f32>::from_shape_vec(IxDyn(&batched_shape), data)
                .map_err(|e| format!("onnx: batch array: {e}"))?;
            let tensor = Tensor::<f32>::from_array(array)
                .map_err(|e| format!("onnx: batch tensor: {e}"))?;
            debug!(name, batch_size, shape = ?batched_shape, "onnx: built batched f32 tensor");
            Ok(vec![(name.clone(), tensor.into())])
        }
        "tensor/i64" => {
            if cat_bytes.len() % 8 != 0 {
                return Err("onnx: batch i64 blob size not aligned to 8 bytes".into());
            }
            let data: Vec<i64> = cat_bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            validate_shape(&batched_shape, data.len())?;
            let array = ArrayD::<i64>::from_shape_vec(IxDyn(&batched_shape), data)
                .map_err(|e| format!("onnx: batch array: {e}"))?;
            let tensor = Tensor::<i64>::from_array(array)
                .map_err(|e| format!("onnx: batch tensor: {e}"))?;
            debug!(name, batch_size, shape = ?batched_shape, "onnx: built batched i64 tensor");
            Ok(vec![(name.clone(), tensor.into())])
        }
        other => Err(format!(
            "onnx: unsupported batch blob content_type '{other}'"
        )),
    }
}

/// Split a batched output tensor back into N individual results.
///
/// For blob_output mode: splits first f32 output into N blobs with shape `[1, ...]`.
/// For JSON mode: splits all outputs into N JSON objects.
fn split_batch_outputs(
    output_names: &[String],
    outputs: &ort::session::SessionOutputs<'_>,
    batch_size: usize,
    config: &OnnxConfig,
) -> Result<Vec<KernelOutput>, AxonError> {
    if config.blob_output {
        // Blob output: split first f32 output into N blobs.
        for (i, name) in output_names.iter().enumerate() {
            if let Ok(view) = outputs[i].try_extract_array::<f32>() {
                let shape = view.shape().to_vec();
                if shape.is_empty() || shape[0] != batch_size {
                    return Err(format!(
                        "onnx: output batch dim mismatch: expected {batch_size}, got {:?}",
                        shape.first()
                    )
                    .into());
                }

                let inner_shape: Vec<usize> = shape[1..].to_vec();
                let item_elements: usize = inner_shape.iter().product::<usize>().max(1);
                let flat: Vec<f32> = view.iter().copied().collect();

                let mut results = Vec::with_capacity(batch_size);
                for b in 0..batch_size {
                    let start = b * item_elements;
                    let end = start + item_elements;
                    let item_data: Vec<u8> = flat[start..end]
                        .iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect();
                    let mut item_shape = vec![1usize];
                    item_shape.extend_from_slice(&inner_shape);
                    results.push(KernelOutput::Blob {
                        data: item_data,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(item_shape),
                    });
                }
                debug!(output = %name, batch_size, "onnx: split batch blob output");
                return Ok(results);
            }
        }
        return Err("onnx: blob_output requested but no f32 outputs in batch".into());
    }

    // JSON output: split each output tensor along batch dim.
    let mut all_results: Vec<serde_json::Map<String, serde_json::Value>> =
        (0..batch_size).map(|_| serde_json::Map::new()).collect();

    for (i, name) in output_names.iter().enumerate() {
        if let Ok(view) = outputs[i].try_extract_array::<f32>() {
            let shape = view.shape().to_vec();
            let inner_shape: Vec<usize> = shape[1..].to_vec();
            let item_elements: usize = inner_shape.iter().product::<usize>().max(1);
            let flat: Vec<f32> = view.iter().copied().collect();

            for b in 0..batch_size {
                let start = b * item_elements;
                let end = start + item_elements;
                let item_data: Vec<f32> = flat[start..end].to_vec();
                let mut item_shape = vec![1usize];
                item_shape.extend_from_slice(&inner_shape);
                all_results[b].insert(
                    name.clone(),
                    serde_json::json!({"shape": item_shape, "data": item_data}),
                );
            }
        } else if let Ok(view) = outputs[i].try_extract_array::<i64>() {
            let shape = view.shape().to_vec();
            let inner_shape: Vec<usize> = shape[1..].to_vec();
            let item_elements: usize = inner_shape.iter().product::<usize>().max(1);
            let flat: Vec<i64> = view.iter().copied().collect();

            for b in 0..batch_size {
                let start = b * item_elements;
                let end = start + item_elements;
                let item_data: Vec<i64> = flat[start..end].to_vec();
                let mut item_shape = vec![1usize];
                item_shape.extend_from_slice(&inner_shape);
                all_results[b].insert(
                    name.clone(),
                    serde_json::json!({"shape": item_shape, "data": item_data}),
                );
            }
        } else {
            return Err(
                format!("onnx: unsupported batch output dtype for '{name}'").into(),
            );
        }
    }

    Ok(all_results
        .into_iter()
        .map(|r| KernelOutput::Json(serde_json::json!({"outputs": r})))
        .collect())
}

// ── Input building ─────────────────────────────────────────────

/// Build ort input values from JSON, matching model input names.
fn build_inputs<'v>(
    input_names: &[String],
    input: &Value,
) -> Result<Vec<(String, SessionInputValue<'v>)>, String> {
    let mut ort_inputs: Vec<(String, SessionInputValue<'v>)> = Vec::new();

    for name in input_names {
        let json_data = find_tensor_data(input, name)
            .ok_or_else(|| format!("onnx: missing input '{name}' in JSON"))?;

        // Determine dtype from JSON values: all integers → i64, otherwise → f32.
        let is_int = is_all_integer(json_data);

        if is_int {
            let shape = infer_shape(json_data)?;
            let mut data = Vec::new();
            flatten_i64(json_data, &mut data);
            validate_shape(&shape, data.len())?;
            let array = ArrayD::<i64>::from_shape_vec(IxDyn(&shape), data)
                .map_err(|e| format!("onnx: array '{name}': {e}"))?;
            let tensor = Tensor::<i64>::from_array(array)
                .map_err(|e| format!("onnx: tensor '{name}': {e}"))?;
            ort_inputs.push((name.clone(), tensor.into()));
        } else {
            let shape = infer_shape(json_data)?;
            let mut data = Vec::new();
            flatten_f32(json_data, &mut data);
            validate_shape(&shape, data.len())?;
            let array = ArrayD::<f32>::from_shape_vec(IxDyn(&shape), data)
                .map_err(|e| format!("onnx: array '{name}': {e}"))?;
            let tensor = Tensor::<f32>::from_array(array)
                .map_err(|e| format!("onnx: tensor '{name}': {e}"))?;
            ort_inputs.push((name.clone(), tensor.into()));
        }

        debug!(name, is_int, "onnx: built input tensor");
    }

    Ok(ort_inputs)
}

/// Build ort input values from raw blob bytes (zero-copy path for tensor pipelines).
///
/// Currently supports single-input models where the blob contains packed f32 or i64 data.
/// The blob's `content_type` and `shape` metadata guide tensor construction.
fn build_inputs_from_blob<'v>(
    input_names: &[String],
    input: &KernelInput,
) -> Result<Vec<(String, SessionInputValue<'v>)>, String> {
    let blob = input.first_blob()
        .ok_or("onnx: has_blobs() but no blob data found")?;

    let shape = blob.meta.shape.as_ref()
        .ok_or("onnx: blob input missing shape metadata")?;

    let name = &input_names[0];

    match blob.meta.content_type.as_str() {
        "tensor/f32" => {
            if blob.bytes.len() % 4 != 0 {
                return Err(format!(
                    "onnx: f32 blob size {} not divisible by 4", blob.bytes.len()
                ));
            }
            let data: Vec<f32> = blob.bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            validate_shape(shape, data.len())?;
            let array = ArrayD::<f32>::from_shape_vec(IxDyn(shape), data)
                .map_err(|e| format!("onnx: blob array '{name}': {e}"))?;
            let tensor = Tensor::<f32>::from_array(array)
                .map_err(|e| format!("onnx: blob tensor '{name}': {e}"))?;
            debug!(name, dtype = "f32", shape = ?shape, "onnx: built tensor from blob");
            Ok(vec![(name.clone(), tensor.into())])
        }
        "tensor/i64" => {
            if blob.bytes.len() % 8 != 0 {
                return Err(format!(
                    "onnx: i64 blob size {} not divisible by 8", blob.bytes.len()
                ));
            }
            let data: Vec<i64> = blob.bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            validate_shape(shape, data.len())?;
            let array = ArrayD::<i64>::from_shape_vec(IxDyn(shape), data)
                .map_err(|e| format!("onnx: blob array '{name}': {e}"))?;
            let tensor = Tensor::<i64>::from_array(array)
                .map_err(|e| format!("onnx: blob tensor '{name}': {e}"))?;
            debug!(name, dtype = "i64", shape = ?shape, "onnx: built tensor from blob");
            Ok(vec![(name.clone(), tensor.into())])
        }
        ct => Err(format!(
            "onnx: unsupported blob content_type '{ct}' (expected tensor/f32 or tensor/i64)"
        )),
    }
}

/// Find tensor data for a model input name in the JSON.
///
/// Searches top-level keys first, then one level deep (namespaced signals).
fn find_tensor_data<'a>(input: &'a Value, name: &str) -> Option<&'a Value> {
    // Direct match at top level.
    if let Some(v) = input.get(name) {
        return Some(v);
    }
    // One level deep (e.g., {"tokens": {"input_ids": [...]}}).
    if let Value::Object(map) = input {
        for (_, inner) in map {
            if let Some(v) = inner.get(name) {
                return Some(v);
            }
        }
    }
    None
}

// ── Tensor parsing helpers ─────────────────────────────────────

/// Infer tensor shape from a nested JSON array.
///
/// Handles both nested arrays and explicit `{"shape": [...], "data": [...]}` format.
fn infer_shape(value: &Value) -> Result<Vec<usize>, String> {
    // Explicit format: {"shape": [1, 3], "data": [...]}
    if let Some(shape_val) = value.get("shape") {
        return shape_val
            .as_array()
            .ok_or_else(|| "onnx: 'shape' must be an array".to_string())?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| "onnx: shape dims must be integers".to_string())
            })
            .collect();
    }
    // Auto-detect from nested arrays.
    let mut shape = Vec::new();
    let mut current = value;
    loop {
        match current {
            Value::Array(arr) => {
                if arr.is_empty() {
                    shape.push(0);
                    break;
                }
                shape.push(arr.len());
                current = &arr[0];
            }
            Value::Number(_) => break,
            _ => return Err("onnx: tensor values must be numbers or arrays".into()),
        }
    }
    Ok(shape)
}

/// Flatten a nested JSON array (or explicit "data" field) into f32 values.
fn flatten_f32(value: &Value, out: &mut Vec<f32>) {
    if let Some(data) = value.get("data") {
        flatten_f32(data, out);
        return;
    }
    match value {
        Value::Array(arr) => {
            for item in arr {
                flatten_f32(item, out);
            }
        }
        Value::Number(n) => {
            out.push(n.as_f64().unwrap_or(0.0) as f32);
        }
        _ => {}
    }
}

/// Flatten a nested JSON array (or explicit "data" field) into i64 values.
fn flatten_i64(value: &Value, out: &mut Vec<i64>) {
    if let Some(data) = value.get("data") {
        flatten_i64(data, out);
        return;
    }
    match value {
        Value::Array(arr) => {
            for item in arr {
                flatten_i64(item, out);
            }
        }
        Value::Number(n) => {
            out.push(n.as_i64().unwrap_or(0));
        }
        _ => {}
    }
}

/// Check if all leaf numbers in a JSON value are integers.
fn is_all_integer(value: &Value) -> bool {
    // Explicit format with "data" field.
    if let Some(data) = value.get("data") {
        return is_all_integer(data);
    }
    match value {
        Value::Array(arr) => arr.iter().all(is_all_integer),
        Value::Number(n) => n.as_i64().is_some() && n.as_f64().map_or(true, |f| f.fract() == 0.0),
        _ => false,
    }
}

/// Validate that shape matches data length.
fn validate_shape(shape: &[usize], data_len: usize) -> Result<(), String> {
    let expected: usize = shape.iter().product();
    if expected != data_len {
        return Err(format!(
            "onnx: shape {shape:?} expects {expected} elements, got {data_len}"
        ));
    }
    Ok(())
}

// ── Config parsing ─────────────────────────────────────────────

fn parse_config(operations: &Value) -> Result<OnnxConfig, String> {
    match operations {
        Value::String(s) => Ok(OnnxConfig {
            model_path: PathBuf::from(s),
            blob_output: false,
            device: "cpu".to_string(),
        }),
        Value::Object(obj) => {
            let model = obj
                .get("model")
                .and_then(|v| v.as_str())
                .ok_or("onnx: operations must have 'model' path")?;
            let blob_output = obj
                .get("blob_output")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let device = obj
                .get("device")
                .and_then(|v| v.as_str())
                .unwrap_or("cpu")
                .to_string();
            Ok(OnnxConfig {
                model_path: PathBuf::from(model),
                blob_output,
                device,
            })
        }
        _ => Err("onnx: operations must be model path string or config object".into()),
    }
}

/// Configure the ONNX Runtime Execution Provider on a session builder.
///
/// Used by both `OnnxKernel` and `GenerateKernel` to share EP configuration logic.
///
/// Supported devices:
/// - `"cpu"`: Default CPU execution (no EP configuration needed)
/// - `"coreml"`: Apple CoreML — uses Neural Engine + GPU on Apple Silicon
/// - `"cuda"`: NVIDIA CUDA GPU
/// - `"tensorrt"`: NVIDIA TensorRT (optimized inference, requires CUDA)
/// - `"directml"`: DirectX 12 GPU (Windows)
///
/// Unknown devices log a warning and fall back to CPU.
/// If an EP is unavailable at runtime, ORT silently falls back to CPU.
pub fn configure_execution_provider(
    builder: ort::session::builder::SessionBuilder,
    device: &str,
) -> Result<ort::session::builder::SessionBuilder, String> {
    match device {
        "cpu" => Ok(builder),
        "coreml" => {
            info!("onnx: configuring CoreML execution provider");
            builder
                .with_execution_providers([
                    ort::execution_providers::CoreMLExecutionProvider::default().build(),
                ])
                .map_err(|e| format!("onnx: CoreML EP: {e}"))
        }
        "cuda" => {
            info!("onnx: configuring CUDA execution provider");
            builder
                .with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default().build(),
                ])
                .map_err(|e| format!("onnx: CUDA EP: {e}"))
        }
        "tensorrt" => {
            info!("onnx: configuring TensorRT execution provider");
            builder
                .with_execution_providers([
                    ort::execution_providers::TensorRTExecutionProvider::default().build(),
                ])
                .map_err(|e| format!("onnx: TensorRT EP: {e}"))
        }
        "directml" => {
            info!("onnx: configuring DirectML execution provider");
            builder
                .with_execution_providers([
                    ort::execution_providers::DirectMLExecutionProvider::default().build(),
                ])
                .map_err(|e| format!("onnx: DirectML EP: {e}"))
        }
        other => {
            tracing::warn!(device = other, "onnx: unknown device, falling back to CPU");
            Ok(builder)
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config parsing ──

    #[test]
    fn test_parse_config_string() {
        let ops = serde_json::json!("models/test.onnx");
        let config = parse_config(&ops).unwrap();
        assert_eq!(config.model_path, PathBuf::from("models/test.onnx"));
    }

    #[test]
    fn test_parse_config_object() {
        let ops = serde_json::json!({"model": "models/test.onnx"});
        let config = parse_config(&ops).unwrap();
        assert_eq!(config.model_path, PathBuf::from("models/test.onnx"));
        assert_eq!(config.device, "cpu");
    }

    #[test]
    fn test_parse_config_with_device() {
        let ops = serde_json::json!({"model": "model.onnx", "device": "coreml"});
        let config = parse_config(&ops).unwrap();
        assert_eq!(config.model_path, PathBuf::from("model.onnx"));
        assert_eq!(config.device, "coreml");
    }

    #[test]
    fn test_parse_config_string_defaults_cpu() {
        let ops = serde_json::json!("model.onnx");
        let config = parse_config(&ops).unwrap();
        assert_eq!(config.device, "cpu");
    }

    #[test]
    fn test_parse_config_missing_model() {
        let ops = serde_json::json!({"timeout": 5});
        let err = parse_config(&ops).unwrap_err();
        assert!(err.contains("model"), "expected 'model' error, got: {err}");
    }

    #[test]
    fn test_parse_config_invalid() {
        let ops = serde_json::json!(42);
        assert!(parse_config(&ops).is_err());
    }

    // ── Shape inference ──

    #[test]
    fn test_infer_shape_1d() {
        let v = serde_json::json!([1.0, 2.0, 3.0]);
        assert_eq!(infer_shape(&v).unwrap(), vec![3]);
    }

    #[test]
    fn test_infer_shape_2d() {
        let v = serde_json::json!([[1, 2], [3, 4], [5, 6]]);
        assert_eq!(infer_shape(&v).unwrap(), vec![3, 2]);
    }

    #[test]
    fn test_infer_shape_3d() {
        let v = serde_json::json!([[[1, 2], [3, 4]]]);
        assert_eq!(infer_shape(&v).unwrap(), vec![1, 2, 2]);
    }

    #[test]
    fn test_infer_shape_explicit() {
        let v = serde_json::json!({"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6]});
        assert_eq!(infer_shape(&v).unwrap(), vec![2, 3]);
    }

    // ── Flatten ──

    #[test]
    fn test_flatten_f32() {
        let v = serde_json::json!([[1.0, 2.0], [3.0, 4.0]]);
        let mut data = Vec::new();
        flatten_f32(&v, &mut data);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_flatten_i64() {
        let v = serde_json::json!([[101, 7592], [2088, 102]]);
        let mut data = Vec::new();
        flatten_i64(&v, &mut data);
        assert_eq!(data, vec![101, 7592, 2088, 102]);
    }

    #[test]
    fn test_flatten_explicit_data() {
        let v = serde_json::json!({"shape": [2, 2], "data": [1.0, 2.0, 3.0, 4.0]});
        let mut data = Vec::new();
        flatten_f32(&v, &mut data);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ── Integer detection ──

    #[test]
    fn test_is_all_integer_yes() {
        let v = serde_json::json!([[101, 7592], [2088, 102]]);
        assert!(is_all_integer(&v));
    }

    #[test]
    fn test_is_all_integer_no() {
        let v = serde_json::json!([[1.5, 2.0], [3.0, 4.0]]);
        assert!(!is_all_integer(&v));
    }

    #[test]
    fn test_is_all_integer_explicit_data() {
        let v = serde_json::json!({"data": [1, 2, 3]});
        assert!(is_all_integer(&v));
    }

    // ── Shape validation ──

    #[test]
    fn test_validate_shape_ok() {
        assert!(validate_shape(&[2, 3], 6).is_ok());
    }

    #[test]
    fn test_validate_shape_mismatch() {
        let err = validate_shape(&[2, 3], 5).unwrap_err();
        assert!(err.contains("6") && err.contains("5"));
    }

    // ── Tensor data lookup ──

    #[test]
    fn test_find_tensor_data_top_level() {
        let input = serde_json::json!({"input_ids": [[101, 102]]});
        let found = find_tensor_data(&input, "input_ids");
        assert!(found.is_some());
        assert_eq!(found.unwrap(), &serde_json::json!([[101, 102]]));
    }

    #[test]
    fn test_find_tensor_data_nested() {
        let input = serde_json::json!({"tokens": {"input_ids": [[101, 102]]}});
        let found = find_tensor_data(&input, "input_ids");
        assert!(found.is_some());
    }

    #[test]
    fn test_find_tensor_data_missing() {
        let input = serde_json::json!({"foo": "bar"});
        assert!(find_tensor_data(&input, "input_ids").is_none());
    }

    // ── Real inference tests (require test_models/) ──

    fn test_model_path(name: &str) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop(); // crates/
        path.pop(); // medulla-rse/
        path.push("test_models");
        path.push(name);
        path
    }

    fn has_test_models() -> bool {
        test_model_path("simple_linear.onnx").exists()
    }

    #[test]
    fn test_real_simple_linear() {
        if !has_test_models() {
            eprintln!("skipping: test_models/ not found");
            return;
        }
        let kernel = OnnxKernel::new().unwrap();
        let input = serde_json::json!({"x": [[1.0, 2.0, 3.0]]});
        let ops = serde_json::json!({"model": test_model_path("simple_linear.onnx").to_str().unwrap()});

        let result = kernel.execute(KernelInput::from_json(input), ops).unwrap();
        // y = x * 2 + 1 → [3.0, 5.0, 7.0]
        let y_data = &result["outputs"]["y"]["data"];
        let y: Vec<f32> = y_data.as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert_eq!(y, vec![3.0, 5.0, 7.0]);

        let y_shape = &result["outputs"]["y"]["shape"];
        assert_eq!(y_shape, &serde_json::json!([1, 3]));
    }

    #[test]
    fn test_real_matmul_predict() {
        if !has_test_models() {
            eprintln!("skipping: test_models/ not found");
            return;
        }
        let kernel = OnnxKernel::new().unwrap();
        // features = [1.0, 0.0, 0.0, 0.0] → prediction = features @ W = [0.5, -0.3]
        let input = serde_json::json!({"features": [[1.0, 0.0, 0.0, 0.0]]});
        let ops = serde_json::json!({"model": test_model_path("matmul.onnx").to_str().unwrap()});

        let result = kernel.execute(KernelInput::from_json(input), ops).unwrap();
        let pred = &result["outputs"]["prediction"]["data"];
        let vals: Vec<f32> = pred.as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert!((vals[0] - 0.5).abs() < 1e-5, "expected ~0.5, got {}", vals[0]);
        assert!((vals[1] - (-0.3)).abs() < 1e-5, "expected ~-0.3, got {}", vals[1]);
    }

    #[test]
    fn test_real_token_embedding() {
        if !has_test_models() {
            eprintln!("skipping: test_models/ not found");
            return;
        }
        let kernel = OnnxKernel::new().unwrap();
        // input_ids = [0, 1, 2] (int64) → gather from embedding table → mean pool
        let input = serde_json::json!({"input_ids": [[0, 1, 2]]});
        let ops = serde_json::json!({"model": test_model_path("token_sum.onnx").to_str().unwrap()});

        let result = kernel.execute(KernelInput::from_json(input), ops).unwrap();
        let emb_shape = &result["outputs"]["embedding"]["shape"];
        assert_eq!(emb_shape, &serde_json::json!([1, 1, 4]));

        let emb_data = &result["outputs"]["embedding"]["data"];
        let vals: Vec<f64> = emb_data.as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(vals.len(), 4, "expected 4-dim embedding");
        // Values should be non-zero (mean of random embeddings)
        assert!(vals.iter().any(|v| v.abs() > 0.01), "embedding should be non-zero");
    }

    #[test]
    fn test_real_explicit_tensor_format() {
        if !has_test_models() {
            eprintln!("skipping: test_models/ not found");
            return;
        }
        let kernel = OnnxKernel::new().unwrap();
        // Use explicit {"shape": ..., "data": ...} format
        let input = serde_json::json!({
            "x": {"shape": [1, 3], "data": [10.0, 20.0, 30.0]}
        });
        let ops = serde_json::json!({"model": test_model_path("simple_linear.onnx").to_str().unwrap()});

        let result = kernel.execute(KernelInput::from_json(input), ops).unwrap();
        // y = x * 2 + 1 → [21.0, 41.0, 61.0]
        let y: Vec<f32> = result["outputs"]["y"]["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert_eq!(y, vec![21.0, 41.0, 61.0]);
    }

    #[test]
    fn test_real_session_caching() {
        if !has_test_models() {
            eprintln!("skipping: test_models/ not found");
            return;
        }
        let kernel = OnnxKernel::new().unwrap();
        let ops = serde_json::json!({"model": test_model_path("simple_linear.onnx").to_str().unwrap()});

        // Run twice — second should use cached session.
        let r1 = kernel.execute(KernelInput::from_json(serde_json::json!({"x": [[1.0, 1.0, 1.0]]})), ops.clone()).unwrap();
        let r2 = kernel.execute(KernelInput::from_json(serde_json::json!({"x": [[2.0, 2.0, 2.0]]})), ops).unwrap();

        let y1: Vec<f32> = r1["outputs"]["y"]["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let y2: Vec<f32> = r2["outputs"]["y"]["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert_eq!(y1, vec![3.0, 3.0, 3.0]);
        assert_eq!(y2, vec![5.0, 5.0, 5.0]);

        // Cache should have exactly 1 entry.
        assert_eq!(kernel.sessions.read().len(), 1);
    }

    #[test]
    fn test_real_missing_input() {
        if !has_test_models() {
            eprintln!("skipping: test_models/ not found");
            return;
        }
        let kernel = OnnxKernel::new().unwrap();
        let input = serde_json::json!({"wrong_name": [[1.0, 2.0, 3.0]]});
        let ops = serde_json::json!({"model": test_model_path("simple_linear.onnx").to_str().unwrap()});

        let err = kernel.execute(KernelInput::from_json(input), ops).unwrap_err();
        assert!(err.contains("missing input"), "expected missing input error, got: {err}");
    }

    #[test]
    fn test_real_model_not_found() {
        let kernel = OnnxKernel::new().unwrap();
        let input = serde_json::json!({"x": [[1.0]]});
        let ops = serde_json::json!({"model": "/nonexistent/model.onnx"});

        let err = kernel.execute(KernelInput::from_json(input), ops).unwrap_err();
        assert!(err.contains("onnx: load"), "expected load error, got: {err}");
    }

    // ── Pipeline: tokenizer → onnx ──

    fn example_model_path(name: &str) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop(); // crates/
        path.pop(); // medulla-rse/
        path.push("examples");
        path.push("models");
        path.push(name);
        path
    }

    fn has_example_models() -> bool {
        example_model_path("small_embed.onnx").exists()
            && example_model_path("tokenizer.json").exists()
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn test_pipeline_tokenizer_to_onnx() {
        if !has_example_models() {
            eprintln!("skipping: examples/models/ not found");
            return;
        }
        let tok = crate::kernels::tokenizer::TokenizerKernel::new();
        let onnx = OnnxKernel::new().unwrap();

        // Step 1: tokenize
        let tok_input = serde_json::json!({"text": "hello world"});
        let tok_ops = serde_json::json!(example_model_path("tokenizer.json").to_str().unwrap());
        let tok_output = tok.execute(KernelInput::from_json(tok_input), tok_ops).unwrap();

        // Should have input_ids and attention_mask
        assert!(tok_output.get("input_ids").is_some(), "missing input_ids");
        assert!(tok_output.get("attention_mask").is_some(), "missing attention_mask");
        let ids = tok_output["input_ids"].as_array().unwrap();
        assert_eq!(ids.len(), 1, "expected batch_size=1");
        let seq_len = ids[0].as_array().unwrap().len();
        assert!(seq_len > 2, "expected at least [CLS] + tokens + [SEP]");

        // Step 2: onnx inference
        let onnx_ops = serde_json::json!(example_model_path("small_embed.onnx").to_str().unwrap());
        let onnx_output = onnx.execute(KernelInput::from_json(tok_output.unwrap_json()), onnx_ops).unwrap();

        // Should have outputs.last_hidden_state
        let hs = &onnx_output["outputs"]["last_hidden_state"];
        let shape: Vec<usize> = hs["shape"].as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(shape.len(), 3, "expected 3D shape [batch, seq, dim]");
        assert_eq!(shape[0], 1, "batch_size=1");
        assert_eq!(shape[1], seq_len, "seq_len should match tokenizer output");
        assert_eq!(shape[2], 16, "embed_dim=16");

        let data: Vec<f64> = hs["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(data.len(), seq_len * 16);
        assert!(data.iter().any(|v| v.abs() > 0.001), "embeddings should be non-zero");
    }

    // NOTE: test_full_embedding_pipeline (tokenizer → onnx → quickjs) lives in
    // crates/drivers/ because QuickJsKernel is a non-ML kernel that stays there.
}
