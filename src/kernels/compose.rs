//! Pipeline composition kernel.
//!
//! Allows a pipeline step to invoke another pipeline, enabling
//! function-style composition (like Unix pipes) without a DAG engine.
//!
//! ## Usage
//!
//! ```toml
//! [post]
//! steps = [
//!   { op = "tensor.normalize" },
//!   { op = "pipeline.run", manifest = "sink-pipeline.toml" },
//! ]
//! ```
//!
//! The sub-pipeline receives the current step's output as input.
//! Blob outputs (tensors) are passed as raw bytes; JSON outputs are
//! serialized and passed as `application/json`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;
use tracing::{debug, info_span};

use crate::error::AxonError;
use crate::kernel::{ComputeKernel, KernelInput, KernelOutput, KernelRegistry};
use crate::manifest::Manifest;
use crate::pipeline::Pipeline;

/// Registry factory: called to create a `KernelRegistry` for each sub-pipeline.
type RegistryFactory = dyn Fn() -> KernelRegistry + Send + Sync;

/// Pipeline composition kernel — invokes a sub-pipeline as a step.
///
/// Sub-pipelines are loaded once and cached (warm after first call).
pub struct PipelineKernel {
    cache: Mutex<HashMap<PathBuf, Arc<Pipeline>>>,
    /// Optional custom registry factory for sub-pipelines.
    /// If `None`, sub-pipelines use `Pipeline::default_registry()`.
    registry_fn: Option<Arc<RegistryFactory>>,
}

impl PipelineKernel {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            registry_fn: None,
        }
    }

    /// Create a PipelineKernel that gives sub-pipelines a custom registry.
    ///
    /// Useful for embedding Axon as a library with custom kernels,
    /// or for tests that need mock kernels in sub-pipelines.
    pub fn with_registry_fn(f: impl Fn() -> KernelRegistry + Send + Sync + 'static) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            registry_fn: Some(Arc::new(f)),
        }
    }

    /// Load a sub-pipeline from disk, or return a cached instance.
    fn load_or_cached(&self, manifest_path: &PathBuf) -> Result<Arc<Pipeline>, String> {
        let mut cache = self.cache.lock();
        if let Some(pipeline) = cache.get(manifest_path) {
            return Ok(Arc::clone(pipeline));
        }

        let content = std::fs::read_to_string(manifest_path)
            .map_err(|e| format!("pipeline.run: read '{}': {e}", manifest_path.display()))?;
        let manifest = Manifest::from_toml(&content)
            .map_err(|e| format!("pipeline.run: parse '{}': {e}", manifest_path.display()))?;
        let base_dir = manifest_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf();

        let registry = match &self.registry_fn {
            Some(f) => f(),
            None => Pipeline::default_registry(),
        };

        let pipeline = Arc::new(Pipeline::new(manifest, registry, base_dir));
        cache.insert(manifest_path.clone(), Arc::clone(&pipeline));
        Ok(pipeline)
    }
}

impl ComputeKernel for PipelineKernel {
    fn name(&self) -> &str {
        "pipeline"
    }

    fn execute(
        &self,
        input: KernelInput,
        operations: Value,
    ) -> Result<KernelOutput, AxonError> {
        let manifest_path = operations
            .get("manifest")
            .and_then(|v| v.as_str())
            .ok_or("pipeline.run: missing 'manifest' field")?;

        let path = PathBuf::from(manifest_path);
        let pipeline = self.load_or_cached(&path)?;

        // Convert KernelInput back to bytes for the sub-pipeline.
        let (bytes, content_type) = input_to_bytes(&input);

        let _span = info_span!(
            "pipeline.run",
            manifest = %manifest_path,
            input_size = bytes.len(),
        )
        .entered();

        debug!("invoking sub-pipeline");

        pipeline
            .run(&bytes, &content_type)
            .map_err(|e| AxonError::from(format!("pipeline.run: sub-pipeline failed: {e}")))
    }
}

/// Convert KernelInput back to raw bytes for a sub-pipeline.
///
/// Prefers blob data (zero-copy tensor path) over JSON serialization.
fn input_to_bytes(input: &KernelInput) -> (Vec<u8>, String) {
    // Prefer blob data (tensor/binary from upstream kernel).
    if let Some(blob) = input.first_blob() {
        return (blob.bytes.clone(), blob.meta.content_type.clone());
    }

    // Fall back to JSON serialization.
    let json_bytes = serde_json::to_vec(&input.json).unwrap_or_default();
    (json_bytes, "application/json".to_string())
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::BlobMeta;
    use crate::kernel::{BlobData, KernelRegistry};
    use crate::manifest::Manifest;

    /// Mock ONNX kernel that tags its output.
    struct MockOnnxKernel;

    impl ComputeKernel for MockOnnxKernel {
        fn name(&self) -> &str {
            "onnx"
        }
        fn execute(
            &self,
            input: KernelInput,
            operations: Value,
        ) -> Result<KernelOutput, AxonError> {
            let model = operations
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            Ok(KernelOutput::Json(serde_json::json!({
                "model": model,
                "prediction": [0.9, 0.05, 0.05],
                "from_input": input.json,
            })))
        }
    }

    /// Tag kernel that passes JSON through with a marker.
    struct TagKernel(&'static str);

    impl ComputeKernel for TagKernel {
        fn name(&self) -> &str {
            self.0
        }
        fn execute(
            &self,
            input: KernelInput,
            _operations: Value,
        ) -> Result<KernelOutput, AxonError> {
            let mut json = input.into_json();
            if let Some(obj) = json.as_object_mut() {
                obj.insert(
                    format!("_{}_applied", self.0),
                    serde_json::Value::Bool(true),
                );
            }
            Ok(KernelOutput::Json(json))
        }
    }

    fn test_registry() -> KernelRegistry {
        let mut reg = KernelRegistry::new();
        reg.register(Arc::new(MockOnnxKernel));
        reg.register(Arc::new(TagKernel("tensor")));
        reg.register(Arc::new(PipelineKernel::with_registry_fn(test_registry)));
        reg
    }

    #[test]
    fn test_input_to_bytes_json() {
        let input = KernelInput::from_json(serde_json::json!({"text": "hello"}));
        let (bytes, ct) = input_to_bytes(&input);
        assert_eq!(ct, "application/json");
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed["text"], "hello");
    }

    #[test]
    fn test_input_to_bytes_blob() {
        let floats: Vec<f32> = vec![0.1, 0.2, 0.3];
        let blob_bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut blobs = HashMap::new();
        blobs.insert(
            "_prev".to_string(),
            BlobData {
                bytes: blob_bytes.clone(),
                meta: BlobMeta {
                    size: 12,
                    content_type: "tensor/f32".to_string(),
                    shape: Some(vec![1, 3]),
                },
            },
        );

        let input = KernelInput {
            json: serde_json::json!({}),
            blobs,
        };

        let (bytes, ct) = input_to_bytes(&input);
        assert_eq!(ct, "tensor/f32");
        assert_eq!(bytes, blob_bytes);
    }

    #[test]
    fn test_pipeline_kernel_name() {
        let kernel = PipelineKernel::new();
        assert_eq!(kernel.name(), "pipeline");
    }

    #[test]
    fn test_pipeline_kernel_missing_manifest() {
        let kernel = PipelineKernel::new();
        let input = KernelInput::from_json(serde_json::json!({"data": [1, 2, 3]}));
        let ops = serde_json::json!({"op": "run"});
        let result = kernel.execute(input, ops);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing 'manifest'"));
    }

    #[test]
    fn test_pipeline_kernel_manifest_not_found() {
        let kernel = PipelineKernel::new();
        let input = KernelInput::from_json(serde_json::json!({"data": [1, 2, 3]}));
        let ops = serde_json::json!({"op": "run", "manifest": "/nonexistent/path.toml"});
        let result = kernel.execute(input, ops);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("pipeline.run"));
    }

    #[test]
    fn test_pipeline_composition_with_tmpfile() {
        // Create a temporary manifest for the sub-pipeline.
        let tmp_dir = std::env::temp_dir().join("axon_test_compose");
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let sub_manifest_path = tmp_dir.join("sub.toml");
        std::fs::write(
            &sub_manifest_path,
            r#"
[model]
name = "sub-model"
file = "model.onnx"
"#,
        )
        .unwrap();

        // Build a parent pipeline that calls the sub-pipeline in post-processing.
        let parent_manifest = Manifest::from_toml(
            r#"
[model]
name = "parent-model"
file = "model.onnx"

[post]
steps = [
  { op = "tensor.tag" },
]
"#,
        )
        .unwrap();

        let parent = Pipeline::new(parent_manifest, test_registry(), PathBuf::from("."));
        let output = parent.run(b"test input", "text/plain").unwrap();
        let json = output.unwrap_json();

        // Parent model ran, then tensor.tag was applied.
        assert_eq!(json["_tensor_applied"], true);

        // Now test the sub-pipeline directly via PipelineKernel.
        let kernel = PipelineKernel::with_registry_fn(test_registry);
        let input = KernelInput::from_json(serde_json::json!({"text": "composed"}));
        let ops = serde_json::json!({
            "op": "run",
            "manifest": sub_manifest_path.to_string_lossy(),
        });
        let result = kernel.execute(input, ops);
        assert!(result.is_ok());
        let sub_json = result.unwrap().unwrap_json();
        assert!(sub_json["model"].as_str().unwrap().contains("model.onnx"));

        // Cleanup.
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_pipeline_kernel_caching() {
        let tmp_dir = std::env::temp_dir().join("axon_test_cache");
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let manifest_path = tmp_dir.join("cached.toml");
        std::fs::write(
            &manifest_path,
            r#"
[model]
name = "cached-model"
file = "model.onnx"
"#,
        )
        .unwrap();

        let kernel = PipelineKernel::with_registry_fn(test_registry);

        // First call loads and caches.
        let input1 = KernelInput::from_json(serde_json::json!({"run": 1}));
        let ops = serde_json::json!({
            "op": "run",
            "manifest": manifest_path.to_string_lossy(),
        });
        assert!(kernel.execute(input1, ops.clone()).is_ok());

        // Second call uses cache.
        let input2 = KernelInput::from_json(serde_json::json!({"run": 2}));
        assert!(kernel.execute(input2, ops).is_ok());

        // Verify cache has exactly one entry.
        assert_eq!(kernel.cache.lock().len(), 1);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
