//! Pipeline executor for Axon ML inference.
//!
//! A Pipeline loads a manifest.toml and chains kernel executions:
//!
//! ```text
//! raw bytes → [pre steps] → model inference → [post steps] → output
//! ```
//!
//! Each step dispatches to a registered `ComputeKernel` via the
//! `KernelRegistry`. Steps are wired by automatically converting
//! each step's output into the next step's input.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let pipeline = Pipeline::load("models/yolov8n/manifest.toml")?;
//! let output = pipeline.run(image_bytes, "image/jpeg")?;
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tracing::{debug, info, info_span};

use crate::blob::{BlobMeta, BlobStore};
use crate::error::AxonError;
use crate::kernel::{BlobData, ComputeKernel, KernelInput, KernelOutput, KernelRegistry};
use crate::manifest::{Manifest, StepConfig};

// ── Error type ─────────────────────────────────────────────────

/// Errors from pipeline loading or execution.
#[derive(Debug)]
pub enum PipelineError {
    /// Failed to read manifest file.
    Io(std::io::Error),
    /// Failed to parse manifest TOML.
    Parse(String),
    /// Kernel referenced in a step not found in registry.
    KernelNotFound { kernel: String, op: String },
    /// Invalid op format (expected "kernel.operation").
    InvalidOp(String),
    /// A pre/post-processing step failed.
    StepFailed {
        phase: &'static str,
        step: usize,
        op: String,
        error: AxonError,
    },
    /// Model inference failed.
    ModelFailed(AxonError),
    /// Pipeline has no steps and no model.
    Empty,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "manifest I/O error: {e}"),
            Self::Parse(e) => write!(f, "manifest parse error: {e}"),
            Self::KernelNotFound { kernel, op } => {
                write!(f, "kernel '{kernel}' not found (required by op '{op}')")
            }
            Self::InvalidOp(op) => {
                write!(f, "invalid op '{op}': expected 'kernel.operation' format")
            }
            Self::StepFailed {
                phase,
                step,
                op,
                error,
            } => write!(f, "{phase} step {step} ({op}) failed: {error}"),
            Self::ModelFailed(e) => write!(f, "model inference failed: {e}"),
            Self::Empty => write!(f, "pipeline has no steps"),
        }
    }
}

impl std::error::Error for PipelineError {}

// ── Resolved step ──────────────────────────────────────────────

/// A step resolved to a concrete kernel + operations JSON.
struct ResolvedStep {
    kernel: Arc<dyn ComputeKernel>,
    operations: serde_json::Value,
    op_name: String,
}

// ── Progress event ─────────────────────────────────────────────

/// Progress event emitted during pipeline execution (for SSE streaming).
#[derive(Debug, Clone)]
pub struct StepProgress {
    /// Phase: "pre", "model", or "post".
    pub phase: String,
    /// Step index within the phase.
    pub step: usize,
    /// Operation name (e.g. "audio.decode").
    pub op: String,
    /// Duration of this step.
    pub duration: std::time::Duration,
}

// ── Pipeline ───────────────────────────────────────────────────

/// ML inference pipeline: pre-processing → model → post-processing.
///
/// The pipeline owns a `KernelRegistry` and a `BlobStore` for
/// passing binary data between steps without serialization overhead.
pub struct Pipeline {
    manifest: Manifest,
    /// Directory containing the manifest file (model paths resolve relative to this).
    base_dir: PathBuf,
    registry: KernelRegistry,
    blob_store: Arc<BlobStore>,
}

impl Pipeline {
    /// Create a Pipeline from a manifest and registry.
    ///
    /// `base_dir` is used to resolve relative model file paths.
    pub fn new(manifest: Manifest, registry: KernelRegistry, base_dir: PathBuf) -> Self {
        Self {
            manifest,
            base_dir,
            registry,
            blob_store: Arc::new(BlobStore::in_memory()),
        }
    }

    /// Load a Pipeline from a manifest.toml file.
    ///
    /// Builds a default `KernelRegistry` with all compiled-in kernels.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(PipelineError::Io)?;
        let manifest = Manifest::from_toml(&content).map_err(PipelineError::Parse)?;
        let base_dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let registry = Self::default_registry();
        Ok(Self::new(manifest, registry, base_dir))
    }

    /// Access the parsed manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Access the kernel registry.
    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    /// Mutable access to registry for adding custom kernels before run.
    pub fn registry_mut(&mut self) -> &mut KernelRegistry {
        &mut self.registry
    }

    // ── Validation ─────────────────────────────────────────────

    /// Validate the pipeline before execution.
    ///
    /// Checks that all kernels referenced in pre/post steps are registered,
    /// and that the onnx kernel is available for model inference.
    /// Returns a list of missing kernel names, or Ok(()) if valid.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut missing = Vec::new();

        // Check pre steps.
        if let Some(ref pre) = self.manifest.pre {
            for step in &pre.steps {
                let (kernel, _) = step.split_op();
                if !self.registry.has(kernel) && !missing.contains(&kernel.to_string()) {
                    missing.push(kernel.to_string());
                }
            }
        }

        // Check post steps.
        if let Some(ref post) = self.manifest.post {
            for step in &post.steps {
                let (kernel, _) = step.split_op();
                if !self.registry.has(kernel) && !missing.contains(&kernel.to_string()) {
                    missing.push(kernel.to_string());
                }
            }
        }

        // Check onnx kernel for model inference.
        if !self.registry.has("onnx") {
            missing.push("onnx".to_string());
        }

        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing)
        }
    }

    /// List all kernel names required by this pipeline's manifest.
    pub fn required_kernels(&self) -> Vec<String> {
        let mut kernels = Vec::new();
        kernels.push("onnx".to_string()); // model inference always needs onnx

        for phase in [&self.manifest.pre, &self.manifest.post] {
            if let Some(config) = phase {
                for step in &config.steps {
                    let (kernel, _) = step.split_op();
                    let name = kernel.to_string();
                    if !kernels.contains(&name) {
                        kernels.push(name);
                    }
                }
            }
        }
        kernels
    }

    // ── Execution ──────────────────────────────────────────────

    /// Run the full pipeline: pre → model → post.
    ///
    /// - `input`: raw input bytes (e.g. image file, audio file).
    /// - `content_type`: MIME type of input (e.g. "image/jpeg", "audio/wav").
    ///
    /// Returns the final step's `KernelOutput`.
    pub fn run(&self, input: &[u8], content_type: &str) -> Result<KernelOutput, PipelineError> {
        let _pipeline_span = info_span!(
            "pipeline",
            model = %self.manifest.model.name,
            file = %self.manifest.model.file,
            input_size = input.len(),
            content_type,
        )
        .entered();

        info!("pipeline: starting");

        // Resolve all steps upfront (fail fast on missing kernels).
        let pre_steps = self.resolve_steps("pre")?;
        let post_steps = self.resolve_steps("post")?;

        let has_model = self.registry.has("onnx");
        if pre_steps.is_empty() && post_steps.is_empty() && !has_model {
            return Err(PipelineError::Empty);
        }

        // Initial input: raw bytes as a blob.
        let mut current = self.bytes_to_kernel_input(input, content_type);

        // Pre-processing.
        let total_pre = pre_steps.len();
        if !pre_steps.is_empty() {
            let _pre_span = info_span!("pre", steps = total_pre).entered();
            for (i, step) in pre_steps.iter().enumerate() {
                let _step_span = info_span!("step", idx = i, op = %step.op_name).entered();
                debug!("executing");
                let mut ops = step.operations.clone();
                if i == total_pre - 1 && has_model {
                    if let serde_json::Value::Object(ref mut map) = ops {
                        map.entry("blob_output")
                            .or_insert(serde_json::Value::Bool(true));
                    }
                }
                let output = step
                    .kernel
                    .execute(current, ops)
                    .map_err(|e| PipelineError::StepFailed {
                        phase: "pre",
                        step: i,
                        op: step.op_name.clone(),
                        error: e,
                    })?;
                current = self.output_to_input(output);
            }
        }

        // Model inference.
        if has_model {
            let model_path = self.base_dir.join(&self.manifest.model.file);
            let has_post = !post_steps.is_empty();
            let device = self.manifest.model.device.as_deref().unwrap_or("cpu");
            let _model_span = info_span!(
                "model",
                file = %model_path.display(),
                device,
            )
            .entered();

            let model_ops = serde_json::json!({
                "model": model_path.to_string_lossy(),
                "blob_output": has_post,
                "device": device,
            });

            debug!("executing inference");
            let onnx = self
                .registry
                .get("onnx")
                .expect("onnx kernel checked above");
            let output = onnx
                .execute(current, model_ops)
                .map_err(|e| PipelineError::ModelFailed(e))?;
            current = self.output_to_input(output);
        }

        // Post-processing.
        let total_post = post_steps.len();
        if !post_steps.is_empty() {
            let _post_span = info_span!("post", steps = total_post).entered();
            for (i, step) in post_steps.into_iter().enumerate() {
                let _step_span = info_span!("step", idx = i, op = %step.op_name).entered();
                debug!("executing");
                let output = step
                    .kernel
                    .execute(current, step.operations.clone())
                    .map_err(|e| PipelineError::StepFailed {
                        phase: "post",
                        step: i,
                        op: step.op_name.clone(),
                        error: e,
                    })?;

                if i == total_post - 1 {
                    info!(model = %self.manifest.model.name, "pipeline: complete");
                    return Ok(output);
                }
                current = self.output_to_input(output);
            }
        }

        info!(model = %self.manifest.model.name, "pipeline: complete");
        Ok(KernelOutput::Json(current.json))
    }

    /// Run only the pre-processing steps (useful for testing/debugging).
    pub fn run_pre(&self, input: &[u8], content_type: &str) -> Result<KernelOutput, PipelineError> {
        let steps = self.resolve_steps("pre")?;
        self.run_step_chain(steps, self.bytes_to_kernel_input(input, content_type), "pre")
    }

    /// Run only the post-processing steps with the given input.
    pub fn run_post(&self, input: KernelInput) -> Result<KernelOutput, PipelineError> {
        let steps = self.resolve_steps("post")?;
        self.run_step_chain(steps, input, "post")
    }

    /// Run the full pipeline with per-step timing.
    ///
    /// Returns a list of (step_name, duration) pairs for each pipeline step.
    pub fn run_timed(
        &self,
        input: &[u8],
        content_type: &str,
    ) -> Result<Vec<(String, std::time::Duration)>, PipelineError> {
        use std::time::Instant;

        let pre_steps = self.resolve_steps("pre")?;
        let post_steps = self.resolve_steps("post")?;
        let has_model = self.registry.has("onnx");

        let mut timings = Vec::new();
        let mut current = self.bytes_to_kernel_input(input, content_type);

        // Pre-processing.
        let total_pre = pre_steps.len();
        for (i, step) in pre_steps.iter().enumerate() {
            let mut ops = step.operations.clone();
            if i == total_pre - 1 && has_model {
                if let serde_json::Value::Object(ref mut map) = ops {
                    map.entry("blob_output").or_insert(serde_json::Value::Bool(true));
                }
            }
            let start = Instant::now();
            let output = step
                .kernel
                .execute(current, ops)
                .map_err(|e| PipelineError::StepFailed {
                    phase: "pre",
                    step: i,
                    op: step.op_name.clone(),
                    error: e,
                })?;
            timings.push((format!("pre: {}", step.op_name), start.elapsed()));
            current = self.output_to_input(output);
        }

        // Model inference.
        if has_model {
            let model_path = self.base_dir.join(&self.manifest.model.file);
            let has_post = !post_steps.is_empty();
            let device = self.manifest.model.device.as_deref().unwrap_or("cpu");
            let model_ops = serde_json::json!({
                "model": model_path.to_string_lossy(),
                "blob_output": has_post,
                "device": device,
            });
            let onnx = self.registry.get("onnx").expect("onnx kernel");
            let start = Instant::now();
            let output = onnx
                .execute(current, model_ops)
                .map_err(|e| PipelineError::ModelFailed(e))?;
            timings.push((format!("model: onnx ({})", device), start.elapsed()));
            current = self.output_to_input(output);
        }

        // Post-processing.
        for (i, step) in post_steps.into_iter().enumerate() {
            let start = Instant::now();
            let output = step
                .kernel
                .execute(current, step.operations.clone())
                .map_err(|e| PipelineError::StepFailed {
                    phase: "post",
                    step: i,
                    op: step.op_name.clone(),
                    error: e,
                })?;
            timings.push((format!("post: {}", step.op_name), start.elapsed()));
            current = self.output_to_input(output);
        }

        Ok(timings)
    }

    /// Run the full pipeline, calling `on_progress` after each step completes.
    ///
    /// Returns the final output. The callback receives progress events in order.
    /// This enables SSE streaming: the caller can forward events to clients
    /// as each pipeline step finishes.
    pub fn run_with_progress(
        &self,
        input: &[u8],
        content_type: &str,
        on_progress: impl Fn(StepProgress),
    ) -> Result<KernelOutput, PipelineError> {
        use std::time::Instant;

        let pre_steps = self.resolve_steps("pre")?;
        let post_steps = self.resolve_steps("post")?;
        let has_model = self.registry.has("onnx");

        if pre_steps.is_empty() && post_steps.is_empty() && !has_model {
            return Err(PipelineError::Empty);
        }

        let mut current = self.bytes_to_kernel_input(input, content_type);

        // Pre-processing.
        let total_pre = pre_steps.len();
        for (i, step) in pre_steps.iter().enumerate() {
            let mut ops = step.operations.clone();
            if i == total_pre - 1 && has_model {
                if let serde_json::Value::Object(ref mut map) = ops {
                    map.entry("blob_output")
                        .or_insert(serde_json::Value::Bool(true));
                }
            }
            let start = Instant::now();
            let output = step
                .kernel
                .execute(current, ops)
                .map_err(|e| PipelineError::StepFailed {
                    phase: "pre",
                    step: i,
                    op: step.op_name.clone(),
                    error: e,
                })?;
            on_progress(StepProgress {
                phase: "pre".into(),
                step: i,
                op: step.op_name.clone(),
                duration: start.elapsed(),
            });
            current = self.output_to_input(output);
        }

        // Model inference.
        if has_model {
            let model_path = self.base_dir.join(&self.manifest.model.file);
            let has_post = !post_steps.is_empty();
            let device = self.manifest.model.device.as_deref().unwrap_or("cpu");
            let model_ops = serde_json::json!({
                "model": model_path.to_string_lossy(),
                "blob_output": has_post,
                "device": device,
            });
            let onnx = self.registry.get("onnx").expect("onnx kernel");
            let start = Instant::now();
            let output = onnx
                .execute(current, model_ops)
                .map_err(PipelineError::ModelFailed)?;
            on_progress(StepProgress {
                phase: "model".into(),
                step: 0,
                op: format!("onnx ({})", device),
                duration: start.elapsed(),
            });
            current = self.output_to_input(output);
        }

        // Post-processing.
        let total_post = post_steps.len();
        for (i, step) in post_steps.into_iter().enumerate() {
            let start = Instant::now();
            let output = step
                .kernel
                .execute(current, step.operations.clone())
                .map_err(|e| PipelineError::StepFailed {
                    phase: "post",
                    step: i,
                    op: step.op_name.clone(),
                    error: e,
                })?;
            on_progress(StepProgress {
                phase: "post".into(),
                step: i,
                op: step.op_name.clone(),
                duration: start.elapsed(),
            });

            if i == total_post - 1 {
                return Ok(output);
            }
            current = self.output_to_input(output);
        }

        Ok(KernelOutput::Json(current.json))
    }

    // ── Internal helpers ───────────────────────────────────────

    /// Resolve steps for a given phase ("pre" or "post").
    fn resolve_steps(&self, phase: &str) -> Result<Vec<ResolvedStep>, PipelineError> {
        let steps_config = match phase {
            "pre" => self.manifest.pre.as_ref(),
            "post" => self.manifest.post.as_ref(),
            _ => None,
        };

        let Some(config) = steps_config else {
            return Ok(Vec::new());
        };

        config
            .steps
            .iter()
            .map(|step| self.resolve_step(step))
            .collect()
    }

    /// Resolve a single step config to a kernel + operations.
    fn resolve_step(&self, step: &StepConfig) -> Result<ResolvedStep, PipelineError> {
        let (kernel_name, operation) = step.split_op();

        if kernel_name.is_empty() {
            return Err(PipelineError::InvalidOp(step.op.clone()));
        }

        let kernel = self
            .registry
            .get(kernel_name)
            .ok_or_else(|| PipelineError::KernelNotFound {
                kernel: kernel_name.to_string(),
                op: step.op.clone(),
            })?;

        // Build operations: if the kernel expects a simple string op
        // (like TensorKernel), we produce {"op": "mean_pool", ...params}.
        // If no operation part (bare kernel name), pass params only.
        let mut operations = step.to_operations();

        // Resolve relative paths in operations to base_dir for kernels that use file paths.
        // This ensures paths like "models/tokenizer.json" in manifests are resolved
        // relative to the manifest directory, not the CWD.
        if let serde_json::Value::Object(ref mut map) = operations {
            for key in ["tokenizer", "model_path", "model", "manifest"] {
                if let Some(serde_json::Value::String(p)) = map.get(key) {
                    let path = std::path::Path::new(p);
                    if path.is_relative() {
                        let resolved = self.base_dir.join(path);
                        map.insert(key.to_string(), serde_json::Value::String(
                            resolved.to_string_lossy().to_string(),
                        ));
                    }
                }
            }
        }

        debug!(
            kernel = kernel_name,
            operation = operation,
            "pipeline: resolved step"
        );

        Ok(ResolvedStep {
            kernel: Arc::clone(kernel),
            operations,
            op_name: step.op.clone(),
        })
    }

    /// Execute a chain of resolved steps.
    fn run_step_chain(
        &self,
        steps: Vec<ResolvedStep>,
        mut current: KernelInput,
        phase: &'static str,
    ) -> Result<KernelOutput, PipelineError> {
        if steps.is_empty() {
            return Ok(KernelOutput::Json(current.json));
        }

        let total = steps.len();
        for (i, step) in steps.into_iter().enumerate() {
            let output = step
                .kernel
                .execute(current, step.operations.clone())
                .map_err(|e| PipelineError::StepFailed {
                    phase,
                    step: i,
                    op: step.op_name.clone(),
                    error: e,
                })?;

            if i == total - 1 {
                return Ok(output);
            }
            current = self.output_to_input(output);
        }

        unreachable!()
    }

    /// Convert raw input bytes into a `KernelInput` with blob data.
    fn bytes_to_kernel_input(&self, bytes: &[u8], content_type: &str) -> KernelInput {
        // Text inputs: convert to {"text": "..."} for tokenizer kernels.
        if content_type.starts_with("text/") || content_type == "application/json" {
            let text = String::from_utf8_lossy(bytes);
            if content_type == "application/json" {
                // JSON input: parse and pass through.
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    return KernelInput::from_json(json);
                }
            }
            // Plain text: wrap as {"text": "..."}.
            return KernelInput::from_json(serde_json::json!({ "text": text.as_ref() }));
        }

        // Binary inputs: pass as blob.
        let mut blobs = HashMap::new();
        blobs.insert(
            "_input".to_string(),
            BlobData {
                bytes: bytes.to_vec(),
                meta: BlobMeta {
                    size: bytes.len() as u64,
                    content_type: content_type.to_string(),
                    shape: None,
                },
            },
        );
        KernelInput {
            json: serde_json::json!({
                "_type": "blob",
                "content_type": content_type,
                "size": bytes.len(),
            }),
            blobs,
        }
    }

    /// Convert a `KernelOutput` into a `KernelInput` for the next step.
    fn output_to_input(&self, output: KernelOutput) -> KernelInput {
        match output {
            KernelOutput::Json(json) => KernelInput::from_json(json),
            KernelOutput::Blob {
                data,
                content_type,
                shape,
            } => {
                let meta = BlobMeta {
                    size: data.len() as u64,
                    content_type: content_type.clone(),
                    shape: shape.clone(),
                };
                let fingerprint = serde_json::json!({
                    "_type": "blob",
                    "content_type": content_type,
                    "size": data.len(),
                    "shape": shape,
                });

                // Store in BlobStore for potential later access.
                self.blob_store.put(data.clone(), &content_type, shape);

                let mut blobs = HashMap::new();
                blobs.insert(
                    "_prev".to_string(),
                    BlobData {
                        bytes: data,
                        meta,
                    },
                );
                KernelInput {
                    json: fingerprint,
                    blobs,
                }
            }
        }
    }

    /// Build a default `KernelRegistry` with all compiled-in kernels.
    pub fn default_registry() -> KernelRegistry {
        #[allow(unused_mut)]
        let mut reg = KernelRegistry::new();

        // Pipeline composition (always available).
        reg.register(Arc::new(
            crate::kernels::compose::PipelineKernel::new(),
        ));

        #[cfg(feature = "onnx")]
        {
            if let Ok(k) = crate::kernels::onnx::OnnxKernel::new() {
                reg.register(Arc::new(k));
            }
            reg.register(Arc::new(crate::kernels::tensor::TensorKernel));
            reg.register(Arc::new(
                crate::kernels::generate::GenerateKernel::new(),
            ));
        }

        #[cfg(feature = "wasm")]
        {
            if let Ok(k) = crate::kernels::wasm::WasmKernel::new() {
                reg.register(Arc::new(k));
            }
        }

        #[cfg(feature = "tokenizer")]
        {
            reg.register(Arc::new(
                crate::kernels::tokenizer::TokenizerKernel::new(),
            ));
        }

        #[cfg(feature = "audio")]
        {
            reg.register(Arc::new(crate::kernels::audio::AudioKernel::new()));
            reg.register(Arc::new(crate::kernels::mel::MelKernel::new()));
        }

        #[cfg(feature = "vision")]
        {
            reg.register(Arc::new(crate::kernels::image::ImageKernel));
            reg.register(Arc::new(crate::kernels::detection::DetectionKernel));
        }

        #[cfg(feature = "text")]
        {
            reg.register(Arc::new(crate::kernels::text::TextKernel));
        }

        #[cfg(feature = "sherpa")]
        {
            reg.register(Arc::new(crate::kernels::sherpa::SherpaKernel));
        }

        #[cfg(feature = "sink")]
        {
            reg.register(Arc::new(crate::kernels::sink::SinkKernel::new()));
        }

        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test kernel that passes JSON through with a tag.
    struct TagKernel {
        name: &'static str,
    }

    impl ComputeKernel for TagKernel {
        fn name(&self) -> &str {
            self.name
        }

        fn execute(
            &self,
            input: KernelInput,
            operations: serde_json::Value,
        ) -> Result<KernelOutput, AxonError> {
            let mut json = input.into_json();
            if let Some(obj) = json.as_object_mut() {
                obj.insert(
                    format!("_{}_applied", self.name),
                    serde_json::Value::Bool(true),
                );
                if let Some(op) = operations.get("op") {
                    obj.insert(
                        format!("_{}_op", self.name),
                        op.clone(),
                    );
                }
            }
            Ok(KernelOutput::Json(json))
        }
    }

    /// A test "model" kernel that simulates inference.
    struct MockOnnxKernel;

    impl ComputeKernel for MockOnnxKernel {
        fn name(&self) -> &str {
            "onnx"
        }

        fn execute(
            &self,
            input: KernelInput,
            operations: serde_json::Value,
        ) -> Result<KernelOutput, AxonError> {
            let model = operations
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            Ok(KernelOutput::Json(serde_json::json!({
                "model": model,
                "prediction": [0.9, 0.05, 0.05],
                "input_keys": input.json.as_object().map(|m| m.keys().cloned().collect::<Vec<_>>()),
            })))
        }
    }

    fn test_registry() -> KernelRegistry {
        let mut reg = KernelRegistry::new();
        reg.register(Arc::new(TagKernel { name: "image" }));
        reg.register(Arc::new(TagKernel { name: "tensor" }));
        reg.register(Arc::new(TagKernel { name: "detection" }));
        reg.register(Arc::new(MockOnnxKernel));
        reg
    }

    #[test]
    fn test_pipeline_pre_only() {
        let manifest = Manifest::from_toml(
            r#"
[model]
name = "test"
file = "model.onnx"

[pre]
steps = [
  { op = "image.decode" },
  { op = "image.resize", target = 640 },
]
"#,
        )
        .unwrap();

        let pipeline = Pipeline::new(manifest, test_registry(), PathBuf::from("."));
        let output = pipeline.run(b"fake image bytes", "image/jpeg").unwrap();

        // Pre steps applied, then model ran, output is model result (no post steps).
        let json = output.unwrap_json();
        assert!(json["model"].as_str().unwrap().ends_with("model.onnx"));
        assert!(json["prediction"].is_array());
    }

    #[test]
    fn test_pipeline_full_chain() {
        let manifest = Manifest::from_toml(
            r#"
[model]
name = "classifier"
file = "model.onnx"

[pre]
steps = [
  { op = "image.decode" },
]

[post]
steps = [
  { op = "detection.format", output = "json" },
]
"#,
        )
        .unwrap();

        let pipeline = Pipeline::new(manifest, test_registry(), PathBuf::from("."));
        let output = pipeline.run(b"fake", "image/jpeg").unwrap();

        let json = output.unwrap_json();
        // Post step applied detection.format tag.
        assert_eq!(json["_detection_applied"], true);
        assert_eq!(json["_detection_op"], "format");
    }

    #[test]
    fn test_pipeline_kernel_not_found() {
        let manifest = Manifest::from_toml(
            r#"
[model]
name = "test"
file = "model.onnx"

[pre]
steps = [
  { op = "nonexistent.something" },
]
"#,
        )
        .unwrap();

        let pipeline = Pipeline::new(manifest, test_registry(), PathBuf::from("."));
        let err = pipeline.run(b"data", "application/octet-stream").unwrap_err();
        assert!(matches!(err, PipelineError::KernelNotFound { .. }));
        assert!(err.to_string().contains("nonexistent"));
    }

    #[test]
    fn test_pipeline_no_pre_no_post() {
        let manifest = Manifest::from_toml(
            r#"
[model]
name = "simple"
file = "model.onnx"
"#,
        )
        .unwrap();

        let pipeline = Pipeline::new(manifest, test_registry(), PathBuf::from("."));
        let output = pipeline.run(b"data", "application/octet-stream").unwrap();

        // Straight to model inference.
        let json = output.unwrap_json();
        assert!(json["model"].as_str().unwrap().ends_with("model.onnx"));
    }

    #[test]
    fn test_pipeline_blob_chaining() {
        /// A kernel that produces blob output.
        struct BlobProducer;
        impl ComputeKernel for BlobProducer {
            fn name(&self) -> &str {
                "blobber"
            }
            fn execute(
                &self,
                _input: KernelInput,
                _ops: serde_json::Value,
            ) -> Result<KernelOutput, AxonError> {
                Ok(KernelOutput::Blob {
                    data: vec![1.0f32, 2.0, 3.0]
                        .iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect(),
                    content_type: "tensor/f32".to_string(),
                    shape: Some(vec![1, 3]),
                })
            }
        }

        /// A kernel that reads blob input.
        struct BlobConsumer;
        impl ComputeKernel for BlobConsumer {
            fn name(&self) -> &str {
                "consumer"
            }
            fn execute(
                &self,
                input: KernelInput,
                _ops: serde_json::Value,
            ) -> Result<KernelOutput, AxonError> {
                let blob = input.first_blob().ok_or("expected blob input")?;
                let floats: Vec<f32> = blob
                    .bytes
                    .chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Ok(KernelOutput::Json(
                    serde_json::json!({"received_floats": floats}),
                ))
            }
        }

        let mut reg = KernelRegistry::new();
        reg.register(Arc::new(BlobProducer));
        reg.register(Arc::new(BlobConsumer));
        reg.register(Arc::new(MockOnnxKernel));

        let manifest = Manifest::from_toml(
            r#"
[model]
name = "test"
file = "model.onnx"

[pre]
steps = [
  { op = "blobber.produce" },
  { op = "consumer.read" },
]
"#,
        )
        .unwrap();

        let pipeline = Pipeline::new(manifest, reg, PathBuf::from("."));
        let output = pipeline.run_pre(b"input", "application/octet-stream").unwrap();

        let json = output.unwrap_json();
        assert_eq!(json["received_floats"], serde_json::json!([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_pipeline_step_error() {
        struct FailKernel;
        impl ComputeKernel for FailKernel {
            fn name(&self) -> &str {
                "fail"
            }
            fn execute(
                &self,
                _input: KernelInput,
                _ops: serde_json::Value,
            ) -> Result<KernelOutput, AxonError> {
                Err("intentional failure".into())
            }
        }

        let mut reg = KernelRegistry::new();
        reg.register(Arc::new(FailKernel));
        reg.register(Arc::new(MockOnnxKernel));

        let manifest = Manifest::from_toml(
            r#"
[model]
name = "test"
file = "model.onnx"

[pre]
steps = [
  { op = "fail.now" },
]
"#,
        )
        .unwrap();

        let pipeline = Pipeline::new(manifest, reg, PathBuf::from("."));
        let err = pipeline.run(b"data", "text/plain").unwrap_err();
        assert!(matches!(err, PipelineError::StepFailed { phase: "pre", step: 0, .. }));
        assert!(err.to_string().contains("intentional failure"));
    }
}
