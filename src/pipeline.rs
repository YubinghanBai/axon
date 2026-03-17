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
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use tracing::{debug, info, info_span, warn};

use crate::audit::{AuditEntry, AuditLogger};
use crate::blob::{BlobMeta, BlobStore};
use crate::canary::CanaryRouter;
use crate::coalesce::Singleflight;
use crate::context::SessionStore;
use crate::deadline::{DeadlineExceeded, RequestBudget};
use crate::error::AxonError;
use crate::guard::{GuardError, InputGuard};
use crate::healthcheck::{HealthAlert, OutputHealthCheck};
use crate::kernel::{BlobData, ComputeKernel, KernelInput, KernelOutput, KernelRegistry};
use crate::manifest::{Manifest, StepConfig};
use crate::shadow::ShadowRunner;

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
    /// Server overloaded — batch queue is full. Maps to HTTP 503.
    Overloaded,
    /// Input validation failed. Maps to HTTP 400.
    GuardFailed(GuardError),
    /// Request deadline exceeded. Maps to HTTP 408.
    DeadlineExceeded(DeadlineExceeded),
    /// Output health check failed (NaN, Inf, out-of-range).
    HealthCheckFailed(HealthAlert),
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
            Self::Overloaded => write!(f, "server overloaded (batch queue full)"),
            Self::GuardFailed(e) => write!(f, "input validation failed: {e}"),
            Self::DeadlineExceeded(e) => write!(f, "{e}"),
            Self::HealthCheckFailed(a) => write!(f, "output health check failed: {a}"),
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

// ── Inference metadata ────────────────────────────────────────

/// Per-request timing metadata for observability and cost attribution.
#[derive(Debug, Clone)]
pub struct InferenceMeta {
    /// Total request latency in microseconds.
    pub total_us: u64,
    /// Pre-processing latency in microseconds.
    pub pre_us: u64,
    /// Model inference latency in microseconds (0 if cache hit).
    pub inference_us: u64,
    /// Post-processing latency in microseconds.
    pub post_us: u64,
    /// Device used for inference.
    pub device: String,
    /// Whether the inference cache was hit.
    pub cache_hit: bool,
    /// Whether cascade fallback was triggered.
    pub cascade_used: bool,
}

// ── Pipeline ───────────────────────────────────────────────────

// ── Inference cache ───────────────────────────────────────────

/// Cached inference result (clone-friendly wrapper for KernelOutput).
#[derive(Clone)]
enum CachedResult {
    Json(serde_json::Value),
    Blob {
        data: Arc<Vec<u8>>,
        content_type: String,
        shape: Option<Vec<usize>>,
    },
}

impl CachedResult {
    fn from_output(output: &KernelOutput) -> Self {
        match output {
            KernelOutput::Json(v) => Self::Json(v.clone()),
            KernelOutput::Blob {
                data,
                content_type,
                shape,
            } => Self::Blob {
                data: Arc::new(data.clone()),
                content_type: content_type.clone(),
                shape: shape.clone(),
            },
        }
    }

    fn to_output(&self) -> KernelOutput {
        match self {
            Self::Json(v) => KernelOutput::Json(v.clone()),
            Self::Blob {
                data,
                content_type,
                shape,
            } => KernelOutput::Blob {
                data: (**data).clone(),
                content_type: content_type.clone(),
                shape: shape.clone(),
            },
        }
    }
}

// ── Circuit breaker ──────────────────────────────────────────

/// Circuit breaker state for device fallback.
struct CircuitBreaker {
    fallback_device: String,
    threshold: u32,
    recovery_timeout: Duration,
    /// Consecutive failure count.
    failures: AtomicU32,
    /// 0=Closed (normal), 1=Open (fallback), 2=HalfOpen (probing).
    state: AtomicU8,
    /// When the circuit was opened (for recovery timeout).
    opened_at: parking_lot::Mutex<Option<Instant>>,
}

const CB_CLOSED: u8 = 0;
const CB_OPEN: u8 = 1;
const CB_HALF_OPEN: u8 = 2;

impl CircuitBreaker {
    fn new(fallback_device: String, threshold: u32, recovery_timeout: Duration) -> Self {
        Self {
            fallback_device,
            threshold,
            recovery_timeout,
            failures: AtomicU32::new(0),
            state: AtomicU8::new(CB_CLOSED),
            opened_at: parking_lot::Mutex::new(None),
        }
    }

    /// Get the device to use for this request.
    fn effective_device<'a>(&'a self, primary_device: &'a str) -> &'a str {
        let state = self.state.load(Ordering::Acquire);
        match state {
            CB_OPEN => {
                // Check if recovery timeout has elapsed.
                let opened = self.opened_at.lock();
                if let Some(at) = *opened {
                    if at.elapsed() >= self.recovery_timeout {
                        drop(opened);
                        self.state.store(CB_HALF_OPEN, Ordering::Release);
                        return primary_device;
                    }
                }
                // Still in open state — use fallback.
                // Leak the &str lifetime by returning from struct field.
                // Safe because CircuitBreaker lives as long as Pipeline.
                &self.fallback_device
            }
            CB_HALF_OPEN => primary_device, // Probing: try primary.
            _ => primary_device,            // Closed: normal.
        }
    }

    /// Record a successful inference.
    fn record_success(&self) {
        let prev = self.state.load(Ordering::Acquire);
        if prev == CB_HALF_OPEN {
            info!("circuit breaker: recovery successful, closing");
        }
        self.failures.store(0, Ordering::Release);
        self.state.store(CB_CLOSED, Ordering::Release);
    }

    /// Record a failed inference. Returns true if circuit just opened.
    fn record_failure(&self) -> bool {
        let count = self.failures.fetch_add(1, Ordering::AcqRel) + 1;
        let state = self.state.load(Ordering::Acquire);

        if state == CB_HALF_OPEN {
            // Probe failed — reopen.
            self.state.store(CB_OPEN, Ordering::Release);
            *self.opened_at.lock() = Some(Instant::now());
            warn!("circuit breaker: probe failed, reopening");
            return false;
        }

        if count >= self.threshold && state == CB_CLOSED {
            self.state.store(CB_OPEN, Ordering::Release);
            *self.opened_at.lock() = Some(Instant::now());
            warn!(
                failures = count,
                fallback = %self.fallback_device,
                "circuit breaker: opened, switching to fallback device"
            );
            return true;
        }
        false
    }
}

// ── Adaptive cascade ─────────────────────────────────────────

/// Adaptive cascade threshold controller.
///
/// Automatically adjusts the confidence threshold to maintain a target
/// fallback rate. Uses AIMD-style adjustment: increase threshold when
/// fallback rate is below target, decrease when above.
struct AdaptiveCascade {
    target_rate: f64,
    adjust_interval: u64,
    /// Current adaptive threshold.
    threshold: parking_lot::Mutex<f64>,
    /// Recent fallback count.
    fallback_count: AtomicU32,
    /// Recent total count.
    total_count: AtomicU32,
}

impl AdaptiveCascade {
    fn new(initial_threshold: f64, target_rate: f64, adjust_interval: u64) -> Self {
        Self {
            target_rate: target_rate.clamp(0.0, 1.0),
            adjust_interval: adjust_interval.max(10),
            threshold: parking_lot::Mutex::new(initial_threshold),
            fallback_count: AtomicU32::new(0),
            total_count: AtomicU32::new(0),
        }
    }

    /// Get the current adaptive threshold.
    fn threshold(&self) -> f64 {
        *self.threshold.lock()
    }

    /// Record a cascade decision and maybe adjust threshold.
    fn record(&self, used_fallback: bool) {
        if used_fallback {
            self.fallback_count.fetch_add(1, Ordering::Relaxed);
        }
        let total = self.total_count.fetch_add(1, Ordering::Relaxed) + 1;

        if total as u64 >= self.adjust_interval {
            let fallbacks = self.fallback_count.swap(0, Ordering::Relaxed);
            self.total_count.store(0, Ordering::Relaxed);

            let actual_rate = fallbacks as f64 / total as f64;
            let mut threshold = self.threshold.lock();

            if actual_rate > self.target_rate {
                // Too many fallbacks → lower threshold (accept more from primary).
                *threshold = (*threshold - 0.02).clamp(0.1, 0.99);
                debug!(
                    actual_rate,
                    target = self.target_rate,
                    new_threshold = *threshold,
                    "adaptive cascade: lowering threshold"
                );
            } else if actual_rate < self.target_rate {
                // Too few fallbacks → raise threshold (be stricter).
                *threshold = (*threshold + 0.01).clamp(0.1, 0.99);
                debug!(
                    actual_rate,
                    target = self.target_rate,
                    new_threshold = *threshold,
                    "adaptive cascade: raising threshold"
                );
            }
        }
    }
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
    /// Inference result cache (content-addressed by preprocessed tensor hash).
    inference_cache: Option<moka::sync::Cache<[u8; 32], CachedResult>>,
    /// Circuit breaker for device fallback on failures.
    circuit_breaker: Option<CircuitBreaker>,
    /// Session context store for stateful pipelines.
    context_store: Arc<SessionStore>,
    /// Input validation guard.
    guard: Option<InputGuard>,
    /// Audit trail logger.
    audit: Option<AuditLogger>,
    /// Request coalescer (singleflight).
    coalescer: Singleflight,
    /// Shadow inference runner.
    shadow: Option<ShadowRunner>,
    /// Canary traffic router.
    canary: Option<CanaryRouter>,
    /// Output health check validator.
    healthcheck: OutputHealthCheck,
    /// Adaptive cascade threshold controller.
    adaptive_cascade: Option<AdaptiveCascade>,
}

impl Pipeline {
    /// Create a Pipeline from a manifest and registry.
    ///
    /// `base_dir` is used to resolve relative model file paths.
    pub fn new(manifest: Manifest, registry: KernelRegistry, base_dir: PathBuf) -> Self {
        // Build inference cache if configured.
        let inference_cache = manifest.cache.as_ref().map(|cfg| {
            moka::sync::Cache::builder()
                .max_capacity(cfg.max_entries)
                .time_to_live(Duration::from_secs(cfg.ttl_seconds))
                .build()
        });

        // Build circuit breaker if configured.
        let circuit_breaker = manifest.resilience.as_ref().map(|cfg| {
            CircuitBreaker::new(
                cfg.fallback_device.clone(),
                cfg.failure_threshold,
                Duration::from_secs(cfg.recovery_timeout_seconds),
            )
        });

        // Build input guard if configured.
        let guard = manifest.guard.as_ref().map(InputGuard::from_config);

        // Build audit logger if configured.
        let audit = manifest.audit.as_ref().and_then(|cfg| {
            match AuditLogger::new(&cfg.path, cfg.sample_rate) {
                Ok(logger) => Some(logger),
                Err(e) => {
                    warn!("failed to create audit logger at '{}': {e}", cfg.path);
                    None
                }
            }
        });

        // Build shadow runner if configured.
        let shadow = manifest.shadow.as_ref().and_then(|cfg| {
            let shadow_model_path = base_dir.join(&cfg.model);
            let device = cfg
                .device
                .clone()
                .or_else(|| manifest.model.device.clone())
                .unwrap_or_else(|| "cpu".to_string());
            match ShadowRunner::new(shadow_model_path, device, cfg.sample_rate, &cfg.log_path) {
                Ok(runner) => Some(runner),
                Err(e) => {
                    warn!("failed to create shadow runner: {e}");
                    None
                }
            }
        });

        // Build canary router if configured.
        let canary = manifest.canary.as_ref().map(|cfg| {
            let primary_device = manifest.model.device.as_deref().unwrap_or("cpu");
            CanaryRouter::from_config(cfg, &base_dir, primary_device)
        });

        // Build output health check.
        let healthcheck = OutputHealthCheck::from_config(&manifest.healthcheck);

        // Build adaptive cascade controller if configured.
        let adaptive_cascade = manifest.cascade.as_ref().and_then(|cfg| {
            cfg.target_fallback_rate.map(|rate| {
                AdaptiveCascade::new(cfg.confidence_threshold, rate, cfg.adjust_interval)
            })
        });

        Self {
            manifest,
            base_dir,
            registry,
            blob_store: Arc::new(BlobStore::in_memory()),
            inference_cache,
            circuit_breaker,
            context_store: Arc::new(SessionStore::default_store()),
            guard,
            audit,
            coalescer: Singleflight::new(),
            shadow,
            canary,
            healthcheck,
            adaptive_cascade,
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

        for config in [&self.manifest.pre, &self.manifest.post].into_iter().flatten() {
            for step in &config.steps {
                let (kernel, _) = step.split_op();
                let name = kernel.to_string();
                if !kernels.contains(&name) {
                    kernels.push(name);
                }
            }
        }
        kernels
    }

    // ── Warmup ─────────────────────────────────────────────────

    /// Pre-warm the pipeline by loading the ONNX session and running
    /// dummy inference to trigger kernel compilation / workspace allocation.
    ///
    /// Call this at startup before exposing the HTTP port.
    /// Errors are logged but not fatal — warmup is best-effort.
    pub fn warmup(&self) {
        let model_path = self.base_dir.join(&self.manifest.model.file);
        let device = self.manifest.model.device.as_deref().unwrap_or("cpu");

        let Some(onnx) = self.registry.get("onnx") else {
            return;
        };

        // Run dummy inference to trigger session load + JIT compilation.
        // We call the onnx kernel directly with a minimal tensor input.
        let dummy_input = KernelInput::from_json(serde_json::json!({
            "x": { "shape": [1, 1], "data": [0.0] }
        }));
        let ops = serde_json::json!({
            "model": model_path.to_string_lossy(),
            "device": device,
        });

        // First call: loads session + optimizes graph + allocates workspace.
        let _ = onnx.execute(dummy_input.clone(), ops.clone());
        // Second call: ensures JIT-compiled kernels are cached.
        let _ = onnx.execute(dummy_input, ops);

        info!(
            model = %self.manifest.model.name,
            device,
            "pipeline warmup complete"
        );
    }

    // ── Execution ──────────────────────────────────────────────

    /// Run the full pipeline: pre → model → post.
    ///
    /// - `input`: raw input bytes (e.g. image file, audio file).
    /// - `content_type`: MIME type of input (e.g. "image/jpeg", "audio/wav").
    ///
    /// Returns the final step's `KernelOutput` (discards metadata).
    pub fn run(&self, input: &[u8], content_type: &str) -> Result<KernelOutput, PipelineError> {
        self.run_with_session(input, content_type, None, None)
            .map(|(output, _meta)| output)
    }

    /// Run the pipeline with an optional session ID and deadline for stateful execution.
    ///
    /// Features applied automatically based on manifest configuration:
    /// - **Input guard**: validate input before any compute
    /// - **Request deadline**: short-circuit if client timeout has passed
    /// - **Semantic cache**: skip inference on content-addressed cache hit (blake3)
    /// - **Request coalescing**: deduplicate identical in-flight requests
    /// - **Circuit breaker**: automatic device fallback on consecutive failures
    /// - **Canary routing**: route a percentage of traffic to a canary model
    /// - **Cascade inference**: confidence-gated early exit with fallback model
    /// - **Shadow inference**: fire-and-forget A/B comparison on background thread
    /// - **Output health check**: NaN/Inf/range validation on model outputs
    /// - **Stateful context**: per-session state injected into post-processing
    /// - **Cost metering**: per-request timing metadata in InferenceMeta
    /// - **Audit trail**: append-only JSONL logging of every request
    pub fn run_with_session(
        &self,
        input: &[u8],
        content_type: &str,
        session_id: Option<&str>,
        deadline: Option<&RequestBudget>,
    ) -> Result<(KernelOutput, InferenceMeta), PipelineError> {
        let total_start = Instant::now();
        let _pipeline_span = info_span!(
            "pipeline",
            model = %self.manifest.model.name,
            file = %self.manifest.model.file,
            input_size = input.len(),
            content_type,
        )
        .entered();

        info!("pipeline: starting");

        // ── Input guard ───────────────────────────────────────
        if let Some(ref guard) = self.guard {
            guard.validate(input, content_type).map_err(PipelineError::GuardFailed)?;
        }

        // ── Deadline check (before any compute) ─────────────
        if let Some(budget) = deadline {
            budget.check("pre").map_err(PipelineError::DeadlineExceeded)?;
        }

        let pre_steps = self.resolve_steps("pre")?;
        let post_steps = self.resolve_steps("post")?;

        let has_model = self.registry.has("onnx");
        if pre_steps.is_empty() && post_steps.is_empty() && !has_model {
            return Err(PipelineError::Empty);
        }

        // Track timing for InferenceMeta.
        let mut meta_device = self.manifest.model.device.as_deref().unwrap_or("cpu").to_string();
        let meta_cache_hit = false;
        let mut meta_cascade_used = false;

        // Initial input: raw bytes as a blob.
        let mut current = self.bytes_to_kernel_input(input, content_type);

        // ── Pre-processing ────────────────────────────────────
        let pre_start = Instant::now();
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
        let pre_us = pre_start.elapsed().as_micros() as u64;

        // ── Semantic cache check ──────────────────────────────
        let cache_key = match (has_model, self.inference_cache.as_ref()) {
            (true, Some(cache)) => {
                let key = Self::hash_input(&current);
                if let Some(cached) = cache.get(&key) {
                    debug!("inference cache hit");
                    current = self.output_to_input(cached.to_output());
                    let post_start = Instant::now();
                    let result = self.run_post_phase(post_steps, current, session_id);
                    let post_us = post_start.elapsed().as_micros() as u64;
                    let meta = InferenceMeta {
                        total_us: total_start.elapsed().as_micros() as u64,
                        pre_us,
                        inference_us: 0,
                        post_us,
                        device: meta_device,
                        cache_hit: true,
                        cascade_used: false,
                    };
                    self.audit_request(input, &result, &meta, session_id);
                    return result.map(|out| (out, meta));
                }
                Some(key)
            }
            _ => None,
        };

        // ── Model inference (with circuit breaker + coalescing) ──
        let inference_start = Instant::now();
        if let Some(budget) = deadline {
            budget.check("model").map_err(PipelineError::DeadlineExceeded)?;
        }
        if has_model {
            let model_path = self.base_dir.join(&self.manifest.model.file);
            let has_post = !post_steps.is_empty() || session_id.is_some();
            let primary_device = self.manifest.model.device.as_deref().unwrap_or("cpu");

            // Circuit breaker: may override device to fallback.
            let device = match &self.circuit_breaker {
                Some(cb) => cb.effective_device(primary_device).to_string(),
                None => primary_device.to_string(),
            };
            meta_device = device.clone();

            let _model_span = info_span!(
                "model",
                file = %model_path.display(),
                device = %device,
            )
            .entered();

            // ── Canary routing ──────────────────────────────────
            let input_hash_str = crate::audit::hash_bytes(input);
            let (effective_model, effective_device) = if let Some(ref canary) = self.canary {
                if canary.should_canary(&input_hash_str, session_id) {
                    info!(canary_model = %canary.model_path().display(), "canary: routing to canary model");
                    (canary.model_path().to_path_buf(), canary.device().to_string())
                } else {
                    (model_path.clone(), device.clone())
                }
            } else {
                (model_path.clone(), device.clone())
            };

            let model_ops = serde_json::json!({
                "model": effective_model.to_string_lossy(),
                "blob_output": has_post,
                "device": &effective_device,
            });

            // Save pre-processed input for potential cascade re-inference and shadow.
            let pre_processed = if self.manifest.cascade.is_some() || self.shadow.is_some() {
                Some(current.clone())
            } else {
                None
            };

            debug!("executing inference");
            let onnx = self
                .registry
                .get("onnx")
                .expect("onnx kernel checked above");

            // ── Request coalescing ────────────────────────────
            // If we have a cache key, use it for dedup. Otherwise compute directly.
            let output = if let Some(ref key) = cache_key {
                let onnx_ref = Arc::clone(onnx);
                let current_clone = current;
                let model_ops_clone = model_ops;
                let (result, _was_leader) = self.coalescer.dedupe(*key, || {
                    onnx_ref.execute(current_clone, model_ops_clone)
                        .map_err(PipelineError::ModelFailed)
                });
                match result {
                    Ok(out) => {
                        if let Some(ref cb) = self.circuit_breaker {
                            cb.record_success();
                        }
                        out
                    }
                    Err(PipelineError::ModelFailed(e)) => {
                        if let Some(ref cb) = self.circuit_breaker {
                            cb.record_failure();
                        }
                        return Err(PipelineError::ModelFailed(e));
                    }
                    Err(e) => return Err(e),
                }
            } else {
                match onnx.execute(current, model_ops) {
                    Ok(out) => {
                        if let Some(ref cb) = self.circuit_breaker {
                            cb.record_success();
                        }
                        out
                    }
                    Err(e) => {
                        if let Some(ref cb) = self.circuit_breaker {
                            cb.record_failure();
                        }
                        return Err(PipelineError::ModelFailed(e));
                    }
                }
            };

            // ── Cascade inference ─────────────────────────────
            let output = if let Some(ref cascade) = self.manifest.cascade {
                let confidence = Self::extract_max_confidence(&output);
                // Use adaptive threshold if configured, else fixed.
                let effective_threshold = match &self.adaptive_cascade {
                    Some(ac) => ac.threshold(),
                    None => cascade.confidence_threshold,
                };
                let should_fallback = confidence.is_some_and(|c| c < effective_threshold);
                // Record for adaptive adjustment.
                if let Some(ref ac) = self.adaptive_cascade {
                    ac.record(should_fallback);
                }
                if should_fallback {
                    meta_cascade_used = true;
                    let fallback_path = self.base_dir.join(&cascade.fallback);
                    let fallback_device = cascade
                        .fallback_device
                        .as_deref()
                        .unwrap_or(primary_device);
                    warn!(
                        confidence = ?confidence,
                        threshold = cascade.confidence_threshold,
                        fallback = %cascade.fallback,
                        "cascade: below threshold, running fallback model"
                    );
                    let fallback_ops = serde_json::json!({
                        "model": fallback_path.to_string_lossy(),
                        "blob_output": has_post,
                        "device": fallback_device,
                    });
                    onnx.execute(pre_processed.as_ref().unwrap().clone(), fallback_ops)
                        .map_err(PipelineError::ModelFailed)?
                } else {
                    output
                }
            } else {
                output
            };

            // Cache the inference result.
            if let (Some(key), Some(cache)) = (cache_key, &self.inference_cache) {
                cache.insert(key, CachedResult::from_output(&output));
                debug!("inference result cached");
            }

            // ── Shadow inference ──────────────────────────────
            if let Some(ref shadow) = self.shadow {
                if let Some(ref shadow_input) = pre_processed {
                    let inference_us = inference_start.elapsed().as_micros() as u64;
                    shadow.maybe_run(shadow_input, onnx, &output, inference_us);
                }
            }

            // ── Output health check ───────────────────────────
            if let Some(alert) = self.healthcheck.check(&output) {
                warn!(rule = %alert.rule, message = %alert.message, "output health check failed");
                return Err(PipelineError::HealthCheckFailed(alert));
            }

            current = self.output_to_input(output);
        }
        let inference_us = inference_start.elapsed().as_micros() as u64;

        // ── Deadline check (before post) ─────────────────────
        if let Some(budget) = deadline {
            budget.check("post").map_err(PipelineError::DeadlineExceeded)?;
        }

        // ── Post-processing (with session context) ────────────
        let post_start = Instant::now();
        let result = self.run_post_phase(post_steps, current, session_id);
        let post_us = post_start.elapsed().as_micros() as u64;

        let meta = InferenceMeta {
            total_us: total_start.elapsed().as_micros() as u64,
            pre_us,
            inference_us,
            post_us,
            device: meta_device,
            cache_hit: meta_cache_hit,
            cascade_used: meta_cascade_used,
        };

        // ── Audit trail ───────────────────────────────────────
        self.audit_request(input, &result, &meta, session_id);

        result.map(|out| (out, meta))
    }

    /// Execute post-processing steps with optional session context injection.
    fn run_post_phase(
        &self,
        post_steps: Vec<ResolvedStep>,
        mut current: KernelInput,
        session_id: Option<&str>,
    ) -> Result<KernelOutput, PipelineError> {
        // Inject session context before post-processing.
        if let Some(sid) = session_id {
            if let Some(ctx) = self.context_store.get(sid) {
                if let Some(obj) = current.json.as_object_mut() {
                    obj.insert("_context".to_string(), ctx);
                }
            }
        }

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
                    // Extract and save session context from final output.
                    if let Some(sid) = session_id {
                        Self::save_session_context(&self.context_store, sid, &output);
                    }
                    info!(model = %self.manifest.model.name, "pipeline: complete");
                    return Ok(output);
                }
                current = self.output_to_input(output);
            }
        }

        // No post steps — still save context if present.
        if let Some(sid) = session_id {
            let output = KernelOutput::Json(current.json.clone());
            Self::save_session_context(&self.context_store, sid, &output);
        }

        info!(model = %self.manifest.model.name, "pipeline: complete");
        Ok(KernelOutput::Json(current.json))
    }

    /// Compute blake3 hash of a KernelInput for cache keying.
    fn hash_input(input: &KernelInput) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(input.json.to_string().as_bytes());
        let mut keys: Vec<&String> = input.blobs.keys().collect();
        keys.sort();
        for key in keys {
            if let Some(blob) = input.blobs.get(key) {
                hasher.update(key.as_bytes());
                hasher.update(&blob.bytes);
            }
        }
        *hasher.finalize().as_bytes()
    }

    /// Extract the maximum confidence score from a model output.
    ///
    /// Checks common JSON patterns (`confidence`, `results[].confidence`,
    /// `prediction[]`) and interprets blob data as f32 tensor.
    fn extract_max_confidence(output: &KernelOutput) -> Option<f64> {
        match output {
            KernelOutput::Json(v) => {
                // Direct confidence field.
                if let Some(c) = v.get("confidence").and_then(|v| v.as_f64()) {
                    return Some(c);
                }
                // Array of detections with confidence.
                let arr = v
                    .as_array()
                    .or_else(|| v.get("results").and_then(|v| v.as_array()));
                if let Some(arr) = arr {
                    let max = arr
                        .iter()
                        .filter_map(|item| item.get("confidence").and_then(|v| v.as_f64()))
                        .fold(f64::NEG_INFINITY, f64::max);
                    if max > f64::NEG_INFINITY {
                        return Some(max);
                    }
                }
                // Prediction array (softmax output).
                if let Some(arr) = v.get("prediction").and_then(|v| v.as_array()) {
                    let max = arr
                        .iter()
                        .filter_map(|v| v.as_f64())
                        .fold(f64::NEG_INFINITY, f64::max);
                    if max > f64::NEG_INFINITY {
                        return Some(max);
                    }
                }
                None
            }
            KernelOutput::Blob { data, .. } => {
                if data.len() >= 4 && data.len() % 4 == 0 {
                    let max = data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .fold(f32::NEG_INFINITY, f32::max);
                    if max.is_finite() {
                        return Some(max as f64);
                    }
                }
                None
            }
        }
    }

    /// Extract `_context` from output and save to session store.
    fn save_session_context(store: &SessionStore, session_id: &str, output: &KernelOutput) {
        if let KernelOutput::Json(v) = output {
            if let Some(ctx) = v.get("_context") {
                store.put(session_id, ctx.clone());
            }
        }
    }

    /// Access the session context store.
    pub fn context_store(&self) -> &Arc<SessionStore> {
        &self.context_store
    }

    /// Log an audit entry if audit trail is configured. Never fails the pipeline.
    fn audit_request(
        &self,
        input: &[u8],
        result: &Result<KernelOutput, PipelineError>,
        meta: &InferenceMeta,
        session_id: Option<&str>,
    ) {
        if let Some(ref audit) = self.audit {
            let (status, output_hash) = match result {
                Ok(output) => ("ok", crate::audit::hash_output(output)),
                Err(_) => ("error", "error".to_string()),
            };
            audit.log(&AuditEntry {
                timestamp: crate::audit::now_iso8601(),
                pipeline: self.manifest.model.name.clone(),
                input_hash: crate::audit::hash_bytes(input),
                output_hash,
                status,
                latency_us: meta.total_us,
                device: meta.device.clone(),
                cache_hit: meta.cache_hit,
                session_id: session_id.map(|s| s.to_string()),
            });
        }
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
                .map_err(PipelineError::ModelFailed)?;
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

    // ── Batch execution ─────────────────────────────────────────

    /// Run the full pipeline for N inputs as a true tensor batch.
    ///
    /// Phases:
    /// 1. **Pre-processing**: each input processed independently (partial failure isolated)
    /// 2. **Model inference**: all valid pre-processed tensors batched → single session.run()
    /// 3. **Post-processing**: each output processed independently (partial failure isolated)
    ///
    /// Returns `Ok(Vec<Result<...>>)` where each inner Result corresponds to one input.
    /// Pre/post failures are isolated per-item; model batch failure propagates as outer Err
    /// so the caller (batch dispatcher) can fall back to serial execution.
    pub fn run_batch(
        &self,
        inputs: &[(&[u8], &str)],
    ) -> Result<Vec<Result<(KernelOutput, InferenceMeta), PipelineError>>, PipelineError> {
        let n = inputs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            return Ok(vec![self.run_with_session(inputs[0].0, inputs[0].1, None, None)]);
        }

        let _span = info_span!(
            "pipeline_batch",
            model = %self.manifest.model.name,
            batch_size = n,
        )
        .entered();

        info!("pipeline: starting batch");

        let pre_steps = self.resolve_steps("pre")?;
        let post_steps = self.resolve_steps("post")?;
        let has_model = self.registry.has("onnx");

        if pre_steps.is_empty() && post_steps.is_empty() && !has_model {
            return Err(PipelineError::Empty);
        }

        // Phase 1: Pre-process each input independently (partial failure isolation).
        let total_pre = pre_steps.len();
        let mut pre_results: Vec<Result<KernelInput, PipelineError>> = Vec::with_capacity(n);
        for (bytes, ct) in inputs {
            let result = (|| -> Result<KernelInput, PipelineError> {
                let mut current = self.bytes_to_kernel_input(bytes, ct);
                for (i, step) in pre_steps.iter().enumerate() {
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
                Ok(current)
            })();
            pre_results.push(result);
        }

        // Separate successful from failed, tracking original indices.
        let mut valid_inputs: Vec<KernelInput> = Vec::new();
        let mut valid_indices: Vec<usize> = Vec::new();
        let mut results: Vec<Option<Result<(KernelOutput, InferenceMeta), PipelineError>>> =
            (0..n).map(|_| None).collect();
        let batch_start = Instant::now();

        for (i, r) in pre_results.into_iter().enumerate() {
            match r {
                Ok(input) => {
                    valid_inputs.push(input);
                    valid_indices.push(i);
                }
                Err(e) => {
                    debug!(item = i, error = %e, "pipeline: pre-processing failed (isolated)");
                    results[i] = Some(Err(e));
                }
            }
        }

        if valid_inputs.is_empty() {
            info!(model = %self.manifest.model.name, "pipeline: all items failed pre-processing");
            return Ok(results.into_iter().map(|r| r.unwrap()).collect());
        }

        let valid_count = valid_inputs.len();

        // Phase 2: Model inference (batched if possible).
        // Model batch failure → outer Err (caller handles serial fallback).
        let model_outputs = if has_model {
            let onnx = self
                .registry
                .get("onnx")
                .expect("onnx kernel checked above");
            let model_path = self.base_dir.join(&self.manifest.model.file);
            let has_post = !post_steps.is_empty();
            let device = self.manifest.model.device.as_deref().unwrap_or("cpu");
            let model_ops = serde_json::json!({
                "model": model_path.to_string_lossy(),
                "blob_output": has_post,
                "device": device,
            });

            if onnx.supports_batch() && valid_count > 1 {
                debug!(batch_size = valid_count, "pipeline: tensor-level batch inference");
                onnx.execute_batch(valid_inputs, model_ops)
                    .map_err(PipelineError::ModelFailed)?
            } else {
                debug!(batch_size = valid_count, "pipeline: serial model inference");
                let mut outputs = Vec::with_capacity(valid_count);
                for input in valid_inputs {
                    let output = onnx
                        .execute(input, model_ops.clone())
                        .map_err(PipelineError::ModelFailed)?;
                    outputs.push(output);
                }
                outputs
            }
        } else {
            valid_inputs
                .into_iter()
                .map(|ki| KernelOutput::Json(ki.json))
                .collect()
        };

        // Phase 3: Post-process each output independently (partial failure isolation).
        let device = self.manifest.model.device.as_deref().unwrap_or("cpu").to_string();
        let inference_elapsed = batch_start.elapsed();
        let total_post = post_steps.len();
        for (vi, model_output) in valid_indices.iter().zip(model_outputs) {
            if post_steps.is_empty() {
                let meta = InferenceMeta {
                    total_us: batch_start.elapsed().as_micros() as u64,
                    pre_us: 0,
                    inference_us: inference_elapsed.as_micros() as u64,
                    post_us: 0,
                    device: device.clone(),
                    cache_hit: false,
                    cascade_used: false,
                };
                results[*vi] = Some(Ok((model_output, meta)));
                continue;
            }

            let post_start = Instant::now();
            let post_result = (|| -> Result<KernelOutput, PipelineError> {
                let mut current = self.output_to_input(model_output);
                let mut last: Option<KernelOutput> = None;
                for (i, step) in post_steps.iter().enumerate() {
                    let step_output = step
                        .kernel
                        .execute(current, step.operations.clone())
                        .map_err(|e| PipelineError::StepFailed {
                            phase: "post",
                            step: i,
                            op: step.op_name.clone(),
                            error: e,
                        })?;
                    if i == total_post - 1 {
                        last = Some(step_output);
                        break;
                    }
                    current = self.output_to_input(step_output);
                }
                Ok(last.expect("post_steps is not empty"))
            })();

            if let Err(ref e) = post_result {
                debug!(item = *vi, error = %e, "pipeline: post-processing failed (isolated)");
            }
            let post_us = post_start.elapsed().as_micros() as u64;
            let meta = InferenceMeta {
                total_us: batch_start.elapsed().as_micros() as u64,
                pre_us: 0,
                inference_us: inference_elapsed.as_micros() as u64,
                post_us,
                device: device.clone(),
                cache_hit: false,
                cascade_used: false,
            };
            results[*vi] = Some(post_result.map(|out| (out, meta)));
        }

        info!(model = %self.manifest.model.name, batch_size = n, "pipeline: batch complete");
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
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

    // ── Semantic cache tests ──────────────────────────────────

    #[test]
    fn test_semantic_cache_hit() {
        let manifest = Manifest::from_toml(
            r#"
[model]
name = "cached"
file = "model.onnx"

[cache]
ttl_seconds = 60
max_entries = 100
"#,
        )
        .unwrap();

        let pipeline = Pipeline::new(manifest, test_registry(), PathBuf::from("."));

        // First call: cache miss.
        let out1 = pipeline.run(b"same input", "text/plain").unwrap();
        // Second call with same input: cache hit (same result).
        let out2 = pipeline.run(b"same input", "text/plain").unwrap();
        assert_eq!(out1.unwrap_json(), out2.unwrap_json());

        // Different input: cache miss (different result key set, same model though).
        let out3 = pipeline.run(b"different input", "text/plain").unwrap();
        // Both are valid model outputs.
        assert!(out3.unwrap_json()["prediction"].is_array());
    }

    #[test]
    fn test_semantic_cache_no_config() {
        // Without [cache] config, no caching behavior.
        let manifest = Manifest::from_toml(
            r#"
[model]
name = "uncached"
file = "model.onnx"
"#,
        )
        .unwrap();

        let pipeline = Pipeline::new(manifest, test_registry(), PathBuf::from("."));
        assert!(pipeline.inference_cache.is_none());

        // Should still work normally.
        let out = pipeline.run(b"data", "text/plain").unwrap();
        assert!(out.unwrap_json()["prediction"].is_array());
    }

    // ── Circuit breaker tests ─────────────────────────────────

    #[test]
    fn test_circuit_breaker_state_machine() {
        let cb = CircuitBreaker::new("cpu".to_string(), 3, Duration::from_secs(60));

        // Initial state: closed.
        assert_eq!(cb.state.load(Ordering::Acquire), CB_CLOSED);
        assert_eq!(cb.effective_device("gpu"), "gpu");

        // Record failures below threshold.
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state.load(Ordering::Acquire), CB_CLOSED);

        // Third failure opens the circuit.
        assert!(cb.record_failure());
        assert_eq!(cb.state.load(Ordering::Acquire), CB_OPEN);
        assert_eq!(cb.effective_device("gpu"), "cpu"); // fallback

        // Success resets to closed.
        cb.state.store(CB_HALF_OPEN, Ordering::Release);
        cb.record_success();
        assert_eq!(cb.state.load(Ordering::Acquire), CB_CLOSED);
        assert_eq!(cb.failures.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_circuit_breaker_half_open_probe_failure() {
        let cb = CircuitBreaker::new("cpu".to_string(), 2, Duration::from_millis(10));

        // Open the circuit.
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state.load(Ordering::Acquire), CB_OPEN);

        // Simulate half-open state (probe attempt).
        cb.state.store(CB_HALF_OPEN, Ordering::Release);

        // Probe fails — should reopen.
        cb.record_failure();
        assert_eq!(cb.state.load(Ordering::Acquire), CB_OPEN);
    }

    // ── Cascade inference tests ───────────────────────────────

    #[test]
    fn test_extract_confidence_json() {
        // Direct confidence field.
        let output = KernelOutput::Json(serde_json::json!({"confidence": 0.85}));
        assert_eq!(Pipeline::extract_max_confidence(&output), Some(0.85));

        // Prediction array (softmax).
        let output = KernelOutput::Json(serde_json::json!({"prediction": [0.9, 0.05, 0.05]}));
        assert_eq!(Pipeline::extract_max_confidence(&output), Some(0.9));

        // Array of detections.
        let output = KernelOutput::Json(serde_json::json!({
            "results": [
                {"confidence": 0.7, "class": "cat"},
                {"confidence": 0.95, "class": "dog"},
            ]
        }));
        assert_eq!(Pipeline::extract_max_confidence(&output), Some(0.95));

        // No confidence info.
        let output = KernelOutput::Json(serde_json::json!({"data": "no confidence"}));
        assert_eq!(Pipeline::extract_max_confidence(&output), None);
    }

    #[test]
    fn test_extract_confidence_blob() {
        // f32 tensor blob.
        let floats: Vec<f32> = vec![0.1, 0.3, 0.85, 0.2];
        let data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let output = KernelOutput::Blob {
            data,
            content_type: "application/octet-stream".to_string(),
            shape: Some(vec![1, 4]),
        };
        let conf = Pipeline::extract_max_confidence(&output).unwrap();
        assert!((conf - 0.85).abs() < 0.001);
    }

    // ── Session context tests ─────────────────────────────────

    #[test]
    fn test_session_context_flow() {
        // Use a post-processing kernel that echoes _context through.
        let manifest = Manifest::from_toml(
            r#"
[model]
name = "stateful"
file = "model.onnx"

[post]
steps = [
  { op = "detection.format" },
]
"#,
        )
        .unwrap();

        let pipeline = Pipeline::new(manifest, test_registry(), PathBuf::from("."));

        // First call with session: no context injected (new session).
        let (out1, _meta1) = pipeline
            .run_with_session(b"frame1", "text/plain", Some("cam-1"), None)
            .unwrap();
        let json1 = out1.unwrap_json();
        // TagKernel just passes through, so output has model fields + _detection_applied.
        assert!(json1["_detection_applied"].as_bool().unwrap());

        // Manually set session context.
        pipeline
            .context_store()
            .put("cam-1", serde_json::json!({"tracks": [1, 2]}));

        // Second call: _context should be injected into post-processing input.
        let (out2, _meta2) = pipeline
            .run_with_session(b"frame2", "text/plain", Some("cam-1"), None)
            .unwrap();
        let json2 = out2.unwrap_json();
        assert!(json2["_detection_applied"].as_bool().unwrap());

        // Different session has no context.
        let (out3, _meta3) = pipeline
            .run_with_session(b"frame1", "text/plain", Some("cam-2"), None)
            .unwrap();
        let json3 = out3.unwrap_json();
        assert!(json3["_detection_applied"].as_bool().unwrap());
    }

    #[test]
    fn test_session_none_no_context() {
        let manifest = Manifest::from_toml(
            r#"
[model]
name = "test"
file = "model.onnx"
"#,
        )
        .unwrap();

        let pipeline = Pipeline::new(manifest, test_registry(), PathBuf::from("."));

        // Run without session — should work normally.
        let (out, meta) = pipeline.run_with_session(b"data", "text/plain", None, None).unwrap();
        assert!(out.unwrap_json()["prediction"].is_array());
        assert!(meta.total_us > 0);
        assert_eq!(meta.cache_hit, false);
    }

    // ── Hash determinism test ─────────────────────────────────

    #[test]
    fn test_hash_input_deterministic() {
        let input1 = KernelInput::from_json(serde_json::json!({"key": "value"}));
        let input2 = KernelInput::from_json(serde_json::json!({"key": "value"}));
        let input3 = KernelInput::from_json(serde_json::json!({"key": "different"}));

        assert_eq!(Pipeline::hash_input(&input1), Pipeline::hash_input(&input2));
        assert_ne!(Pipeline::hash_input(&input1), Pipeline::hash_input(&input3));
    }
}
