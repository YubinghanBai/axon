//! Shadow inference for A/B model comparison.
//!
//! Runs a shadow model in parallel with the primary model for comparison
//! logging without affecting primary latency. Fire-and-forget on a
//! detached thread.
//!
//! Configured via `[shadow]` in manifest.toml:
//!
//! ```toml
//! [shadow]
//! model = "models/v2/model.onnx"
//! sample_rate = 0.1
//! log_path = "/var/log/axon/shadow.jsonl"
//! ```

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;

use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};

// ── ShadowComparison ────────────────────────────────────────────

/// Logged comparison between primary and shadow model outputs.
#[derive(Debug, Serialize)]
struct ShadowComparison {
    timestamp: String,
    input_hash: String,
    primary_output_hash: String,
    shadow_output_hash: String,
    outputs_match: bool,
    primary_latency_us: u64,
    shadow_latency_us: u64,
}

// ── ShadowRunner ────────────────────────────────────────────────

/// Runs shadow inference on a background thread for A/B comparison.
///
/// The shadow runner uses the same ONNX kernel as the primary pipeline
/// but with a different model file. Results are compared and logged,
/// but never affect the primary response.
pub struct ShadowRunner {
    model_path: PathBuf,
    device: String,
    sample_rate: f64,
    log_path: PathBuf,
}

impl ShadowRunner {
    /// Create a new ShadowRunner.
    ///
    /// `model_path` is the absolute path to the shadow model file.
    /// `device` is the execution provider (e.g. "cpu").
    /// `sample_rate` is 0.0–1.0.
    /// `log_path` is where comparison JSONL is written.
    pub fn new(
        model_path: PathBuf,
        device: String,
        sample_rate: f64,
        log_path: &str,
    ) -> Result<Self, std::io::Error> {
        // Ensure parent directory exists.
        if let Some(parent) = std::path::Path::new(log_path).parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // Verify we can open the file for writing.
        let _file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        Ok(Self {
            model_path,
            device,
            sample_rate: sample_rate.clamp(0.0, 1.0),
            log_path: PathBuf::from(log_path),
        })
    }

    /// Get the log path as a string for background threads.
    fn log_path(&self) -> PathBuf {
        self.log_path.clone()
    }

    /// Maybe run shadow inference in the background.
    ///
    /// Checks sample_rate, clones the input, spawns a detached thread,
    /// runs the shadow model, and logs the comparison. Primary response
    /// is never affected.
    pub fn maybe_run(
        &self,
        input: &KernelInput,
        onnx: &Arc<dyn ComputeKernel>,
        primary_result: &KernelOutput,
        primary_latency_us: u64,
    ) {
        // Sampling check.
        if self.sample_rate < 1.0 {
            let hash = crate::audit::hash_bytes(input.json.to_string().as_bytes());
            let sample_val = u32::from_str_radix(&hash[..hash.len().min(8)], 16)
                .unwrap_or(0) as f64
                / u32::MAX as f64;
            if sample_val >= self.sample_rate {
                return;
            }
        }

        // Clone what we need for the background thread.
        let input_clone = input.clone();
        let onnx_clone = Arc::clone(onnx);
        let model_path = self.model_path.clone();
        let device = self.device.clone();
        let primary_hash = crate::audit::hash_output(primary_result);
        let input_hash = crate::audit::hash_bytes(input.json.to_string().as_bytes());
        let log_path = self.log_path();

        std::thread::spawn(move || {
            let start = std::time::Instant::now();

            let model_ops = serde_json::json!({
                "model": model_path.to_string_lossy(),
                "blob_output": false,
                "device": &device,
            });

            let shadow_result = onnx_clone.execute(input_clone, model_ops);
            let shadow_latency_us = start.elapsed().as_micros() as u64;

            let (shadow_hash, outputs_match) = match &shadow_result {
                Ok(output) => {
                    let h = crate::audit::hash_output(output);
                    let m = h == primary_hash;
                    (h, m)
                }
                Err(_) => ("error".to_string(), false),
            };

            let comparison = ShadowComparison {
                timestamp: crate::audit::now_iso8601(),
                input_hash,
                primary_output_hash: primary_hash,
                shadow_output_hash: shadow_hash,
                outputs_match,
                primary_latency_us,
                shadow_latency_us,
            };

            // Log comparison to file. Errors silently ignored.
            if let Ok(line) = serde_json::to_string(&comparison) {
                if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&log_path) {
                    let _ = writeln!(file, "{}", line);
                }
            }
        });
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::error::AxonError;
    use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};

    struct MockShadowKernel;
    impl ComputeKernel for MockShadowKernel {
        fn name(&self) -> &str {
            "onnx"
        }
        fn execute(
            &self,
            _input: KernelInput,
            _operations: serde_json::Value,
        ) -> Result<KernelOutput, AxonError> {
            Ok(KernelOutput::Json(
                serde_json::json!({"shadow": true, "prediction": [0.85, 0.1, 0.05]}),
            ))
        }
    }

    #[test]
    fn test_shadow_runner_logs_comparison() {
        let dir = std::env::temp_dir().join("axon_shadow_test");
        let _ = std::fs::remove_dir_all(&dir);
        let log_path = dir.join("shadow.jsonl");
        let model_path = PathBuf::from("shadow_model.onnx");

        let runner = ShadowRunner::new(
            model_path,
            "cpu".to_string(),
            1.0, // 100% sample
            log_path.to_str().unwrap(),
        )
        .unwrap();

        let onnx: Arc<dyn ComputeKernel> = Arc::new(MockShadowKernel);
        let input = KernelInput::from_json(serde_json::json!({"test": true}));
        let primary_result = KernelOutput::Json(
            serde_json::json!({"primary": true, "prediction": [0.9, 0.05, 0.05]}),
        );

        runner.maybe_run(&input, &onnx, &primary_result, 1000);

        // Wait for background thread to complete.
        std::thread::sleep(std::time::Duration::from_millis(100));

        let content = std::fs::read_to_string(&log_path).unwrap();
        assert!(!content.is_empty(), "shadow comparison should be logged");

        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert!(parsed["primary_output_hash"].is_string());
        assert!(parsed["shadow_output_hash"].is_string());
        assert_eq!(parsed["primary_latency_us"], 1000);
        assert!(parsed["shadow_latency_us"].is_number());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_shadow_zero_sample_rate() {
        let dir = std::env::temp_dir().join("axon_shadow_zero");
        let _ = std::fs::remove_dir_all(&dir);
        let log_path = dir.join("shadow.jsonl");

        let runner = ShadowRunner::new(
            PathBuf::from("model.onnx"),
            "cpu".to_string(),
            0.0, // Never sample.
            log_path.to_str().unwrap(),
        )
        .unwrap();

        let onnx: Arc<dyn ComputeKernel> = Arc::new(MockShadowKernel);
        let input = KernelInput::from_json(serde_json::json!({"test": true}));
        let primary = KernelOutput::Json(serde_json::json!({"p": 1}));

        runner.maybe_run(&input, &onnx, &primary, 500);
        std::thread::sleep(std::time::Duration::from_millis(50));

        let content = std::fs::read_to_string(&log_path).unwrap_or_default();
        assert!(content.is_empty(), "0% sample rate should not log");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_shadow_primary_unaffected_by_failure() {
        struct FailingKernel;
        impl ComputeKernel for FailingKernel {
            fn name(&self) -> &str {
                "onnx"
            }
            fn execute(
                &self,
                _input: KernelInput,
                _ops: serde_json::Value,
            ) -> Result<KernelOutput, AxonError> {
                Err(AxonError::runtime("shadow model crashed"))
            }
        }

        let dir = std::env::temp_dir().join("axon_shadow_fail");
        let _ = std::fs::remove_dir_all(&dir);
        let log_path = dir.join("shadow.jsonl");

        let runner = ShadowRunner::new(
            PathBuf::from("model.onnx"),
            "cpu".to_string(),
            1.0,
            log_path.to_str().unwrap(),
        )
        .unwrap();

        let onnx: Arc<dyn ComputeKernel> = Arc::new(FailingKernel);
        let input = KernelInput::from_json(serde_json::json!({"test": true}));
        let primary = KernelOutput::Json(serde_json::json!({"result": "good"}));

        // This should not panic or affect the caller.
        runner.maybe_run(&input, &onnx, &primary, 1000);
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Shadow failure is logged with "error" hash.
        let content = std::fs::read_to_string(&log_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed["shadow_output_hash"], "error");
        assert_eq!(parsed["outputs_match"], false);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
