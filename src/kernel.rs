//! ComputeKernel trait and I/O types for the Axon ML runtime.
//!
//! All ML kernels (onnx, tensor, wasm, audio, mel, tokenizer) implement
//! the `ComputeKernel` trait. Kernels are synchronous and stateless —
//! they run on a blocking thread pool when used inside an async engine.

use std::collections::HashMap;
use std::sync::Arc;

use crate::blob::BlobMeta;
use crate::error::AxonError;

// ── KernelInput ────────────────────────────────────────────────

/// Rich input for compute kernels.
///
/// Always provides a JSON view. Optionally provides raw blob bytes
/// for kernels that support binary input (e.g. tensor_kernel).
#[derive(Clone)]
pub struct KernelInput {
    /// JSON representation of input data.
    ///
    /// - Inline signals: the actual JSON value.
    /// - Blob signals with JSON content: deserialized JSON.
    /// - Blob signals with binary content: the blob fingerprint metadata.
    pub json: serde_json::Value,

    /// Raw blob data keyed by tract name.
    ///
    /// Only populated for blob signals when BlobStore has the data.
    /// JSON-only kernels ignore this; tensor kernels use it for
    /// zero-copy f32 access.
    pub blobs: HashMap<String, BlobData>,
}

/// Raw blob bytes with metadata.
#[derive(Clone)]
pub struct BlobData {
    /// The raw bytes (e.g. packed f32 for tensors).
    pub bytes: Vec<u8>,
    /// Content metadata (size, type, shape).
    pub meta: BlobMeta,
}

impl KernelInput {
    /// Create a KernelInput from a JSON value (no blob data).
    pub fn from_json(json: serde_json::Value) -> Self {
        Self {
            json,
            blobs: HashMap::new(),
        }
    }

    /// Consume self and return just the JSON value.
    /// Convenience for JSON-only kernels.
    pub fn into_json(self) -> serde_json::Value {
        self.json
    }

    /// Get blob data for a specific tract name.
    pub fn blob(&self, name: &str) -> Option<&BlobData> {
        self.blobs.get(name)
    }

    /// Get the first (or only) blob input. Useful when there's a single
    /// upstream blob signal (e.g. ONNX → tensor_kernel pipeline).
    pub fn first_blob(&self) -> Option<&BlobData> {
        self.blobs.values().next()
    }

    /// Check if any blob inputs are available.
    pub fn has_blobs(&self) -> bool {
        !self.blobs.is_empty()
    }
}

// ── KernelOutput ───────────────────────────────────────────────

/// Output from a ComputeKernel execution.
///
/// Kernels explicitly declare whether their output is structured JSON
/// or raw binary bytes. There is no automatic size-based promotion.
#[derive(Debug)]
pub enum KernelOutput {
    /// Structured JSON result (most kernels: polars, quickjs, e2b).
    Json(serde_json::Value),

    /// Raw binary output (tensors, embeddings, images).
    /// The kernel provides content metadata for lazy materialization.
    Blob {
        data: Vec<u8>,
        content_type: String,
        shape: Option<Vec<usize>>,
    },
}

impl KernelOutput {
    /// Unwrap as JSON value. Panics if this is a Blob.
    /// Convenience for tests.
    pub fn unwrap_json(self) -> serde_json::Value {
        match self {
            Self::Json(v) => v,
            Self::Blob { .. } => panic!("expected KernelOutput::Json, got Blob"),
        }
    }

    /// Borrow inner JSON value. Panics if Blob.
    fn as_json(&self) -> &serde_json::Value {
        match self {
            Self::Json(v) => v,
            Self::Blob { .. } => panic!("expected KernelOutput::Json, got Blob"),
        }
    }

    /// Delegate as_array to inner JSON.
    pub fn as_array(&self) -> Option<&Vec<serde_json::Value>> {
        self.as_json().as_array()
    }

    /// Delegate get to inner JSON.
    pub fn get(&self, key: impl AsRef<str>) -> Option<&serde_json::Value> {
        self.as_json().get(key.as_ref())
    }
}

impl std::ops::Index<&str> for KernelOutput {
    type Output = serde_json::Value;
    fn index(&self, key: &str) -> &serde_json::Value {
        &self.as_json()[key]
    }
}

impl std::ops::Index<usize> for KernelOutput {
    type Output = serde_json::Value;
    fn index(&self, idx: usize) -> &serde_json::Value {
        &self.as_json()[idx]
    }
}

impl PartialEq<serde_json::Value> for KernelOutput {
    fn eq(&self, other: &serde_json::Value) -> bool {
        self.as_json() == other
    }
}

// ── ComputeKernel trait ────────────────────────────────────────

/// Compute kernel trait. Each module (onnx, tensor, wasm, audio, etc.) implements this.
///
/// Kernels are synchronous — they run on a blocking thread pool.
///
/// Input: `KernelInput` with JSON view + optional raw blob bytes.
/// Output: `KernelOutput` — JSON or raw bytes.
pub trait ComputeKernel: Send + Sync {
    /// Kernel name (e.g. "onnx", "tensor", "wasm", "audio").
    fn name(&self) -> &str;

    /// Execute computation.
    ///
    /// - `input`: merged data from all `needs` Signals.
    ///   - `input.json`: JSON view (always available).
    ///   - `input.blobs`: raw blob bytes (only for blob signals).
    /// - `operations`: the operation spec from the Blueprint template body.
    fn execute(
        &self,
        input: KernelInput,
        operations: serde_json::Value,
    ) -> Result<KernelOutput, AxonError>;

    /// Whether this kernel supports true tensor-level batching.
    ///
    /// When true, `execute_batch` concatenates inputs along batch dim 0
    /// for a single execution call (e.g. one ONNX session.run for N inputs).
    /// When false, `execute_batch` falls back to serial per-item execution.
    fn supports_batch(&self) -> bool {
        false
    }

    /// Execute a batch of inputs in a single call.
    ///
    /// Default: serial execution. Kernels that support tensor batching
    /// (like ONNX) override this to concatenate along dim 0.
    fn execute_batch(
        &self,
        inputs: Vec<KernelInput>,
        operations: serde_json::Value,
    ) -> Result<Vec<KernelOutput>, AxonError> {
        inputs
            .into_iter()
            .map(|input| self.execute(input, operations.clone()))
            .collect()
    }
}

// ── KernelRegistry ─────────────────────────────────────────────

/// Registry of available compute kernels.
///
/// Used by both the standalone Axon Pipeline and medulla's ComputeDriver.
pub struct KernelRegistry {
    kernels: HashMap<String, Arc<dyn ComputeKernel>>,
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
        }
    }

    /// Register a kernel.
    pub fn register(&mut self, kernel: Arc<dyn ComputeKernel>) {
        self.kernels.insert(kernel.name().to_string(), kernel);
    }

    /// Look up a kernel by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn ComputeKernel>> {
        self.kernels.get(name)
    }

    /// Check if a kernel is available.
    pub fn has(&self, name: &str) -> bool {
        self.kernels.contains_key(name)
    }

    /// List available kernel names.
    pub fn names(&self) -> Vec<&str> {
        self.kernels.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Create a registry pre-loaded with all compiled-in kernels.
    ///
    /// This registers every kernel whose feature flag is enabled at compile time:
    /// - `onnx`: OnnxKernel + TensorKernel
    /// - `wasm`: WasmKernel
    /// - `tokenizer`: TokenizerKernel
    /// - `audio`: AudioKernel + MelKernel
    pub fn with_defaults() -> Self {
        // Delegate to Pipeline's default_registry to avoid duplication.
        crate::pipeline::Pipeline::default_registry()
    }
}
