//! Manifest types for Axon ML pipelines.
//!
//! A manifest.toml describes an ML inference pipeline:
//!
//! ```toml
//! [model]
//! name = "yolov8n"
//! file = "yolov8n.onnx"
//!
//! [model.input]
//! shape = [1, 3, 640, 640]
//! dtype = "f32"
//!
//! [pre]
//! steps = [
//!   { op = "image.decode" },
//!   { op = "image.resize", target = 640, mode = "letterbox" },
//!   { op = "tensor.cast", dtype = "f32" },
//! ]
//!
//! [post]
//! steps = [
//!   { op = "tensor.transpose", axes = [0, 2, 1] },
//!   { op = "detection.nms", iou = 0.45 },
//! ]
//! ```
//!
//! Each step's `op` field uses `kernel.operation` format.
//! Extra fields become the operation parameters passed to the kernel.

use serde::Deserialize;

/// Top-level manifest describing an ML pipeline.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    /// Model configuration (path, shapes, metadata).
    pub model: ModelConfig,

    /// Pre-processing steps (run before model inference).
    #[serde(default)]
    pub pre: Option<StepsConfig>,

    /// Post-processing steps (run after model inference).
    #[serde(default)]
    pub post: Option<StepsConfig>,

    /// Cascade inference: run a fast model first, fall back to a heavier
    /// model only when confidence is below threshold.
    #[serde(default)]
    pub cascade: Option<CascadeConfig>,

    /// Inference result caching (content-addressable, skip model on cache hit).
    #[serde(default)]
    pub cache: Option<CacheConfig>,

    /// Resilience / circuit breaker: automatic device fallback on failures.
    #[serde(default)]
    pub resilience: Option<ResilienceConfig>,

    /// Input guard: pre-processing validation rules.
    #[serde(default)]
    pub guard: Option<GuardConfig>,

    /// Audit trail: append-only JSONL logging of inference requests.
    #[serde(default)]
    pub audit: Option<AuditConfig>,

    /// Shadow inference: A/B comparison with a shadow model.
    #[serde(default)]
    pub shadow: Option<ShadowConfig>,

    /// Canary traffic routing for safe model rollouts.
    #[serde(default)]
    pub canary: Option<CanaryConfig>,

    /// Output health check: NaN/Inf/range validation on kernel outputs.
    #[serde(default)]
    pub healthcheck: Option<HealthCheckConfig>,
}

/// Cascade inference configuration.
///
/// Two modes: fixed threshold or adaptive.
///
/// Fixed threshold:
/// ```toml
/// [cascade]
/// confidence_threshold = 0.95
/// fallback = "resnet50.onnx"
/// ```
///
/// Adaptive threshold (adjusts to maintain target fallback rate):
/// ```toml
/// [cascade]
/// confidence_threshold = 0.95
/// fallback = "resnet50.onnx"
/// target_fallback_rate = 0.15
/// adjust_interval = 100
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CascadeConfig {
    /// Minimum confidence score to accept the primary model's output.
    /// If the max output score is below this, the fallback model is used.
    /// When adaptive mode is enabled, this is the initial threshold.
    pub confidence_threshold: f64,
    /// Path to the fallback model file (relative to manifest dir).
    pub fallback: String,
    /// Device for fallback model (defaults to primary model's device).
    #[serde(default)]
    pub fallback_device: Option<String>,
    /// Target fallback rate (0.0–1.0). Enables adaptive threshold mode.
    /// The threshold auto-adjusts to maintain this cascade trigger rate.
    #[serde(default)]
    pub target_fallback_rate: Option<f64>,
    /// How often to adjust the threshold (every N requests). Default: 100.
    #[serde(default = "default_adjust_interval")]
    pub adjust_interval: u64,
}

fn default_adjust_interval() -> u64 {
    100
}

/// Inference cache configuration.
///
/// ```toml
/// [cache]
/// ttl_seconds = 3600
/// max_entries = 10000
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CacheConfig {
    /// Time-to-live for cached results in seconds.
    #[serde(default = "default_cache_ttl")]
    pub ttl_seconds: u64,
    /// Maximum number of cached entries.
    #[serde(default = "default_cache_max")]
    pub max_entries: u64,
}

fn default_cache_ttl() -> u64 {
    3600
}
fn default_cache_max() -> u64 {
    10000
}

/// Resilience / circuit breaker configuration.
///
/// ```toml
/// [resilience]
/// fallback_device = "cpu"
/// failure_threshold = 3
/// recovery_timeout_seconds = 60
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ResilienceConfig {
    /// Device to fall back to when the primary device fails repeatedly.
    pub fallback_device: String,
    /// Number of consecutive failures before opening the circuit.
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: u32,
    /// Seconds before attempting recovery on the primary device.
    #[serde(default = "default_recovery_timeout")]
    pub recovery_timeout_seconds: u64,
}

fn default_failure_threshold() -> u32 {
    3
}
fn default_recovery_timeout() -> u64 {
    60
}

/// Model metadata and file path.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Human-readable model name.
    pub name: String,

    /// Path to model file (relative to manifest directory).
    pub file: String,

    /// Model version string.
    #[serde(default)]
    pub version: Option<String>,

    /// Execution provider / device for ONNX Runtime.
    ///
    /// Supported values: `"cpu"` (default), `"coreml"`, `"cuda"`, `"tensorrt"`, `"directml"`.
    /// Falls back to CPU if the requested EP is unavailable.
    #[serde(default)]
    pub device: Option<String>,

    /// Optional description for the model.
    #[serde(default)]
    pub description: Option<String>,

    /// Expected input tensor spec.
    #[serde(default)]
    pub input: Option<TensorSpec>,

    /// Expected output tensor spec.
    #[serde(default)]
    pub output: Option<TensorSpec>,
}

/// Tensor shape and data type specification.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorSpec {
    /// Tensor dimensions, e.g. `[1, 3, 640, 640]`.
    pub shape: Vec<usize>,

    /// Data type: "f32", "f16", "i32", "i64", "u8".
    pub dtype: String,
}

/// A list of pipeline steps.
#[derive(Debug, Clone, Deserialize)]
pub struct StepsConfig {
    pub steps: Vec<StepConfig>,
}

/// A single pipeline step.
///
/// The `op` field uses `kernel.operation` format (e.g. `"tensor.mean_pool"`).
/// All other fields are collected as operation parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct StepConfig {
    /// Operation identifier: `"kernel_name.operation_name"`.
    pub op: String,

    /// All other fields become operation parameters.
    #[serde(flatten)]
    pub params: serde_json::Map<String, serde_json::Value>,
}

/// Input guard configuration.
///
/// ```toml
/// [guard]
/// max_size = "10MB"
/// allowed_content_types = ["image/jpeg", "image/png"]
/// required_json_fields = ["image", "model_id"]
/// ```
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GuardConfig {
    /// Maximum input size (e.g. "10MB", "1GB", "1024").
    #[serde(default)]
    pub max_size: Option<String>,
    /// Allowed MIME content types.
    #[serde(default)]
    pub allowed_content_types: Option<Vec<String>>,
    /// Required JSON fields (checked only for JSON/text content types).
    #[serde(default)]
    pub required_json_fields: Option<Vec<String>>,
}

/// Audit trail configuration.
///
/// ```toml
/// [audit]
/// path = "/var/log/axon/inference.jsonl"
/// sample_rate = 1.0
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct AuditConfig {
    /// Path to the JSONL audit log file.
    pub path: String,
    /// Sample rate: 0.0–1.0 (default 1.0, log everything).
    #[serde(default = "default_sample_rate")]
    pub sample_rate: f64,
}

fn default_sample_rate() -> f64 {
    1.0
}

/// Shadow inference configuration.
///
/// ```toml
/// [shadow]
/// model = "models/v2/model.onnx"
/// sample_rate = 0.1
/// log_path = "/var/log/axon/shadow.jsonl"
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ShadowConfig {
    /// Path to the shadow model file (relative to manifest directory).
    pub model: String,
    /// Shadow model device (defaults to primary model's device).
    #[serde(default)]
    pub device: Option<String>,
    /// Sample rate: 0.0–1.0 (default 0.1).
    #[serde(default = "default_shadow_sample_rate")]
    pub sample_rate: f64,
    /// Path to the comparison JSONL log file.
    pub log_path: String,
}

fn default_shadow_sample_rate() -> f64 {
    0.1
}

/// Canary traffic routing configuration.
///
/// ```toml
/// [canary]
/// model = "models/v2/model.onnx"
/// weight = 5
/// sticky_sessions = true
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CanaryConfig {
    /// Path to the canary model file (relative to manifest directory).
    pub model: String,
    /// Percentage of traffic routed to canary (0-100).
    pub weight: u8,
    /// Canary model device (defaults to primary model's device).
    #[serde(default)]
    pub device: Option<String>,
    /// If true, same session always sees same model version (deterministic routing).
    #[serde(default)]
    pub sticky_sessions: Option<bool>,
}

/// Output health check configuration.
///
/// ```toml
/// [healthcheck]
/// nan_check = true
/// output_range = [0.0, 1.0]
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct HealthCheckConfig {
    /// Whether to check for NaN and Infinity values in outputs.
    #[serde(default)]
    pub nan_check: Option<bool>,
    /// Optional [min, max] bounds for numeric output values.
    #[serde(default)]
    pub output_range: Option<[f64; 2]>,
}

impl Manifest {
    /// Parse a manifest from a TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self, String> {
        toml::from_str(toml_str).map_err(|e| format!("manifest parse error: {e}"))
    }
}

impl StepConfig {
    /// Split `op` into (kernel_name, operation_name).
    ///
    /// `"tensor.mean_pool"` → `("tensor", "mean_pool")`
    /// `"onnx"` → `("onnx", "")`
    pub fn split_op(&self) -> (&str, &str) {
        match self.op.split_once('.') {
            Some((kernel, operation)) => (kernel, operation),
            None => (self.op.as_str(), ""),
        }
    }

    /// Build the operations JSON to pass to the kernel.
    ///
    /// Merges `{"op": "operation_name"}` with all extra params.
    pub fn to_operations(&self) -> serde_json::Value {
        let (_kernel, operation) = self.split_op();
        let mut map = self.params.clone();
        if !operation.is_empty() {
            map.insert("op".to_string(), serde_json::Value::String(operation.to_string()));
        }
        if map.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::Object(map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_manifest() {
        let toml = r#"
[model]
name = "test"
file = "model.onnx"
"#;
        let m = Manifest::from_toml(toml).unwrap();
        assert_eq!(m.model.name, "test");
        assert_eq!(m.model.file, "model.onnx");
        assert!(m.pre.is_none());
        assert!(m.post.is_none());
    }

    #[test]
    fn test_parse_full_manifest() {
        let toml = r#"
[model]
name = "yolov8n"
file = "yolov8n.onnx"
version = "8.0.0"

[model.input]
shape = [1, 3, 640, 640]
dtype = "f32"

[model.output]
shape = [1, 84, 8400]
dtype = "f32"

[pre]
steps = [
  { op = "image.decode" },
  { op = "image.resize", target = 640, mode = "letterbox" },
  { op = "tensor.cast", dtype = "f32" },
  { op = "tensor.unsqueeze", dim = 0 },
]

[post]
steps = [
  { op = "tensor.transpose", axes = [0, 2, 1] },
  { op = "detection.nms", iou = 0.45 },
  { op = "detection.format", output = "json" },
]
"#;
        let m = Manifest::from_toml(toml).unwrap();
        assert_eq!(m.model.name, "yolov8n");
        assert_eq!(m.model.version.as_deref(), Some("8.0.0"));

        let input = m.model.input.as_ref().unwrap();
        assert_eq!(input.shape, vec![1, 3, 640, 640]);
        assert_eq!(input.dtype, "f32");

        let pre = m.pre.as_ref().unwrap();
        assert_eq!(pre.steps.len(), 4);
        assert_eq!(pre.steps[0].op, "image.decode");
        assert_eq!(pre.steps[1].op, "image.resize");
        assert_eq!(pre.steps[1].params["target"], 640);
        assert_eq!(pre.steps[1].params["mode"], "letterbox");

        let post = m.post.as_ref().unwrap();
        assert_eq!(post.steps.len(), 3);
        assert_eq!(post.steps[1].op, "detection.nms");
        assert_eq!(post.steps[1].params["iou"], 0.45);
    }

    #[test]
    fn test_parse_manifest_with_device() {
        let toml = r#"
[model]
name = "yolov8n"
file = "yolov8n.onnx"
device = "coreml"
"#;
        let m = Manifest::from_toml(toml).unwrap();
        assert_eq!(m.model.device.as_deref(), Some("coreml"));
    }

    #[test]
    fn test_parse_manifest_device_defaults_none() {
        let toml = r#"
[model]
name = "test"
file = "model.onnx"
"#;
        let m = Manifest::from_toml(toml).unwrap();
        assert!(m.model.device.is_none());
    }

    #[test]
    fn test_split_op() {
        let step = StepConfig {
            op: "tensor.mean_pool".to_string(),
            params: serde_json::Map::new(),
        };
        assert_eq!(step.split_op(), ("tensor", "mean_pool"));

        let step = StepConfig {
            op: "onnx".to_string(),
            params: serde_json::Map::new(),
        };
        assert_eq!(step.split_op(), ("onnx", ""));
    }

    #[test]
    fn test_to_operations() {
        let mut params = serde_json::Map::new();
        params.insert("dim".to_string(), serde_json::json!(1));
        params.insert("blob_output".to_string(), serde_json::json!(true));

        let step = StepConfig {
            op: "tensor.mean_pool".to_string(),
            params,
        };
        let ops = step.to_operations();
        assert_eq!(ops["op"], "mean_pool");
        assert_eq!(ops["dim"], 1);
        assert_eq!(ops["blob_output"], true);
    }

    #[test]
    fn test_to_operations_no_params() {
        let step = StepConfig {
            op: "image.decode".to_string(),
            params: serde_json::Map::new(),
        };
        let ops = step.to_operations();
        // Has "op" key even with no extra params.
        assert_eq!(ops["op"], "decode");
    }
}
