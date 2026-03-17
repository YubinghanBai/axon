//! Output health check for Axon inference pipelines.
//!
//! Lightweight per-request validation of model outputs. Catches NaN/Inf,
//! out-of-range values, and other obvious failures at ~100ns cost.
//! Not a drift detector — just a smoke test for broken outputs.

use crate::kernel::KernelOutput;
use crate::manifest::HealthCheckConfig;
use std::fmt;

/// Alert raised when output health check detects a problem.
#[derive(Debug, Clone)]
pub struct HealthAlert {
    /// Human-readable rule that triggered this alert (e.g. "nan_detection").
    pub rule: String,
    /// Descriptive error message.
    pub message: String,
}

impl fmt::Display for HealthAlert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.rule, self.message)
    }
}

/// Output health check validator.
///
/// Configured by a `HealthCheckConfig` from the manifest, this validator
/// performs lightweight checks on kernel outputs:
/// - NaN/Inf detection
/// - Range bounds checking
/// - Binary blob f32 validation
#[derive(Debug, Clone)]
pub struct OutputHealthCheck {
    /// Whether to check for NaN and Infinity values.
    pub nan_check: bool,
    /// Optional (min, max) bounds for numeric values.
    pub range: Option<(f64, f64)>,
}

impl OutputHealthCheck {
    /// Create a health check from a manifest config.
    ///
    /// If the config is `None` or both checks are disabled, returns a permissive
    /// validator that never alerts.
    pub fn from_config(config: &Option<HealthCheckConfig>) -> Self {
        match config {
            Some(cfg) => Self {
                nan_check: cfg.nan_check.unwrap_or(false),
                range: cfg.output_range.map(|r| (r[0], r[1])),
            },
            None => Self {
                nan_check: false,
                range: None,
            },
        }
    }

    /// Validate a kernel output.
    ///
    /// Scans JSON values recursively or interprets blobs as f32 arrays.
    /// Returns the first alert found, or `None` if all checks pass.
    pub fn check(&self, output: &KernelOutput) -> Option<HealthAlert> {
        match output {
            KernelOutput::Json(value) => self.scan_json(value),
            KernelOutput::Blob { data, .. } => self.check_blob(data),
        }
    }

    /// Check a single f64 value against configured rules.
    fn check_f64(&self, val: f64) -> Option<HealthAlert> {
        if self.nan_check {
            if val.is_nan() {
                return Some(HealthAlert {
                    rule: "nan_detection".to_string(),
                    message: "output contains NaN value".to_string(),
                });
            }
            if val.is_infinite() {
                return Some(HealthAlert {
                    rule: "inf_detection".to_string(),
                    message: "output contains Infinity value".to_string(),
                });
            }
        }

        if let Some((min, max)) = self.range {
            if val < min || val > max {
                return Some(HealthAlert {
                    rule: "range_violation".to_string(),
                    message: format!("value {} outside range [{}, {}]", val, min, max),
                });
            }
        }

        None
    }

    /// Recursively scan a JSON value for numeric violations.
    fn scan_json(&self, value: &serde_json::Value) -> Option<HealthAlert> {
        match value {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    self.check_f64(f)
                } else {
                    None
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    if let Some(alert) = self.scan_json(item) {
                        return Some(alert);
                    }
                }
                None
            }
            serde_json::Value::Object(obj) => {
                for (_, v) in obj {
                    if let Some(alert) = self.scan_json(v) {
                        return Some(alert);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Check a blob as a packed f32 array.
    ///
    /// Blobs must have length >= 4 and divisible by 4 to be valid f32 data.
    /// Interprets bytes as little-endian f32 values.
    fn check_blob(&self, data: &[u8]) -> Option<HealthAlert> {
        if data.len() < 4 || data.len() % 4 != 0 {
            return None; // Not a valid f32 array; skip checks
        }

        for chunk in data.chunks(4) {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let f = f32::from_le_bytes(bytes) as f64;
            if let Some(alert) = self.check_f64(f) {
                return Some(alert);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test NaN detection in blob output (f32 tensor).
    /// Note: JSON cannot represent NaN, so NaN detection only applies to blob data.
    #[test]
    fn test_nan_detection() {
        let config = Some(HealthCheckConfig {
            nan_check: Some(true),
            output_range: None,
        });
        let check = OutputHealthCheck::from_config(&config);

        let mut data = Vec::new();
        data.extend_from_slice(&(0.5f32).to_le_bytes());
        data.extend_from_slice(&f32::NAN.to_le_bytes());
        data.extend_from_slice(&(0.3f32).to_le_bytes());

        let output = KernelOutput::Blob {
            data,
            content_type: "application/octet-stream".to_string(),
            shape: Some(vec![3]),
        };

        let alert = check.check(&output);
        assert!(alert.is_some());
        assert_eq!(alert.unwrap().rule, "nan_detection");
    }

    /// Test Infinity detection in blob output (f32 tensor).
    #[test]
    fn test_inf_detection() {
        let config = Some(HealthCheckConfig {
            nan_check: Some(true),
            output_range: None,
        });
        let check = OutputHealthCheck::from_config(&config);

        let mut data = Vec::new();
        data.extend_from_slice(&(0.5f32).to_le_bytes());
        data.extend_from_slice(&f32::INFINITY.to_le_bytes());
        data.extend_from_slice(&(0.3f32).to_le_bytes());

        let output = KernelOutput::Blob {
            data,
            content_type: "application/octet-stream".to_string(),
            shape: Some(vec![3]),
        };

        let alert = check.check(&output);
        assert!(alert.is_some());
        assert_eq!(alert.unwrap().rule, "inf_detection");
    }

    /// Test range violation detection.
    #[test]
    fn test_range_violation() {
        let config = Some(HealthCheckConfig {
            nan_check: Some(false),
            output_range: Some([0.0, 1.0]),
        });
        let check = OutputHealthCheck::from_config(&config);

        let output = KernelOutput::Json(serde_json::json!({
            "logits": [0.3, 1.5, 0.7]
        }));

        let alert = check.check(&output);
        assert!(alert.is_some());
        assert_eq!(alert.unwrap().rule, "range_violation");
    }

    /// Test that healthy outputs pass all checks.
    #[test]
    fn test_healthy_output() {
        let config = Some(HealthCheckConfig {
            nan_check: Some(true),
            output_range: Some([0.0, 1.0]),
        });
        let check = OutputHealthCheck::from_config(&config);

        let output = KernelOutput::Json(serde_json::json!({
            "predictions": [0.1, 0.5, 0.9],
            "metadata": {
                "confidence": 0.95
            }
        }));

        let alert = check.check(&output);
        assert!(alert.is_none());
    }

    /// Test NaN detection in blob f32 data.
    #[test]
    fn test_blob_nan() {
        let config = Some(HealthCheckConfig {
            nan_check: Some(true),
            output_range: None,
        });
        let check = OutputHealthCheck::from_config(&config);

        // Create a blob with f32 values: 0.5, NaN, 0.3
        let mut data = Vec::new();
        data.extend_from_slice(&(0.5f32).to_le_bytes());
        data.extend_from_slice(&f32::NAN.to_le_bytes());
        data.extend_from_slice(&(0.3f32).to_le_bytes());

        let output = KernelOutput::Blob {
            data,
            content_type: "application/octet-stream".to_string(),
            shape: Some(vec![3]),
        };

        let alert = check.check(&output);
        assert!(alert.is_some());
        assert_eq!(alert.unwrap().rule, "nan_detection");
    }

    /// Test that a config with no checks enabled never alerts.
    #[test]
    fn test_no_checks_configured() {
        let config = Some(HealthCheckConfig {
            nan_check: Some(false),
            output_range: None,
        });
        let check = OutputHealthCheck::from_config(&config);

        let output = KernelOutput::Json(serde_json::json!({
            "bad": [f64::NAN, f64::INFINITY, 999.0]
        }));

        let alert = check.check(&output);
        assert!(alert.is_none());
    }
}
