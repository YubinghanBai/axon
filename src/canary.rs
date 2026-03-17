//! Canary traffic routing for safe model rollouts.
//!
//! Routes a configurable percentage of traffic to a canary model,
//! returning the canary's result to the client (unlike shadow inference
//! which is fire-and-forget). Supports sticky sessions for consistent
//! user experience during A/B tests.
//!
//! Configured via `[canary]` in manifest.toml:
//!
//! ```toml
//! [canary]
//! model = "models/v2/model.onnx"
//! weight = 5
//! sticky_sessions = true
//! ```

use std::path::{Path, PathBuf};

use crate::manifest::CanaryConfig;

/// Canary router for A/B model deployment with traffic splitting.
///
/// Routes a configurable percentage of requests to a canary model,
/// with support for sticky sessions (deterministic per-session routing).
#[derive(Debug, Clone)]
pub struct CanaryRouter {
    /// Absolute path to canary model file.
    model_path: PathBuf,
    /// Execution provider for canary model (e.g. "cpu", "coreml").
    device: String,
    /// Percentage of traffic routed to canary (0-100).
    weight: u8,
    /// If true, same session always sees same model.
    sticky: bool,
}

impl CanaryRouter {
    /// Create a new CanaryRouter from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Canary configuration from manifest
    /// * `base_dir` - Base directory for resolving relative paths
    /// * `primary_device` - Primary model's device (used as fallback)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = CanaryConfig {
    ///     model: "v2/model.onnx".to_string(),
    ///     weight: 10,
    ///     device: None,
    ///     sticky_sessions: Some(true),
    /// };
    /// let router = CanaryRouter::from_config(&config, Path::new("/models"), "cpu");
    /// assert_eq!(router.model_path(), Path::new("/models/v2/model.onnx"));
    /// ```
    pub fn from_config(config: &CanaryConfig, base_dir: &Path, primary_device: &str) -> Self {
        Self {
            model_path: base_dir.join(&config.model),
            device: config
                .device
                .clone()
                .unwrap_or_else(|| primary_device.to_string()),
            weight: config.weight.min(100),
            sticky: config.sticky_sessions.unwrap_or(false),
        }
    }

    /// Determine if a request should route to the canary model.
    ///
    /// Uses deterministic hashing based on input or session to ensure
    /// consistent routing. Returns true if request should use canary,
    /// false to use primary model.
    ///
    /// # Arguments
    ///
    /// * `input_hash` - Hash of the request input
    /// * `session_id` - Optional session ID for sticky routing
    ///
    /// # Behavior
    ///
    /// - If `weight == 0`, always returns false
    /// - If `weight >= 100`, always returns true
    /// - If sticky && session_id exists: hashes session_id deterministically
    /// - Otherwise: hashes input_hash deterministically
    pub fn should_canary(&self, input_hash: &str, session_id: Option<&str>) -> bool {
        match self.weight {
            0 => false,
            100 => true,
            _ => {
                // Use session ID if sticky sessions are enabled and available
                let hash_source = if self.sticky {
                    session_id.unwrap_or(input_hash)
                } else {
                    input_hash
                };

                // Parse first 8 hex chars as u32, mod 100
                let hash_val = u32::from_str_radix(
                    &hash_source[..hash_source.len().min(8)],
                    16,
                )
                .unwrap_or(0);
                let bucket = (hash_val % 100) as u8;

                bucket < self.weight
            }
        }
    }

    /// Get the absolute path to the canary model file.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Get the execution device for the canary model.
    pub fn device(&self) -> &str {
        &self.device
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_weight_never_canary() {
        let config = CanaryConfig {
            model: "canary.onnx".to_string(),
            weight: 0,
            device: None,
            sticky_sessions: None,
        };
        let router = CanaryRouter::from_config(&config, Path::new("/models"), "cpu");

        // No matter the input hash, should never route to canary
        assert!(!router.should_canary("ffffffff", None));
        assert!(!router.should_canary("00000000", None));
        assert!(!router.should_canary("12345678", Some("session123")));
    }

    #[test]
    fn test_full_weight_always_canary() {
        let config = CanaryConfig {
            model: "canary.onnx".to_string(),
            weight: 100,
            device: None,
            sticky_sessions: None,
        };
        let router = CanaryRouter::from_config(&config, Path::new("/models"), "cpu");

        // Weight >= 100 always routes to canary
        assert!(router.should_canary("ffffffff", None));
        assert!(router.should_canary("00000000", None));
        assert!(router.should_canary("12345678", Some("session123")));
    }

    #[test]
    fn test_deterministic_routing() {
        let config = CanaryConfig {
            model: "canary.onnx".to_string(),
            weight: 50,
            device: None,
            sticky_sessions: Some(false),
        };
        let router = CanaryRouter::from_config(&config, Path::new("/models"), "cpu");

        // Same input hash should always give same result
        let input = "abcd1234";
        let result1 = router.should_canary(input, None);
        let result2 = router.should_canary(input, None);
        let result3 = router.should_canary(input, None);

        assert_eq!(result1, result2);
        assert_eq!(result2, result3);
    }

    #[test]
    fn test_sticky_session() {
        let config = CanaryConfig {
            model: "canary.onnx".to_string(),
            weight: 50,
            device: None,
            sticky_sessions: Some(true),
        };
        let router = CanaryRouter::from_config(&config, Path::new("/models"), "cpu");

        // With sticky sessions, session_id determines routing, not input
        let session = "user_session_123";
        let result1 = router.should_canary("aaaaaaaa", Some(session));
        let result2 = router.should_canary("zzzzzzzz", Some(session)); // Different input

        // Same session should get same routing regardless of input
        assert_eq!(result1, result2);

        // Different session should potentially get different routing
        // (though it might coincidentally be the same)
        let _other_result = router.should_canary("aaaaaaaa", Some("other_session_456"));
    }

    #[test]
    fn test_weight_distribution() {
        let config = CanaryConfig {
            model: "canary.onnx".to_string(),
            weight: 50,
            device: None,
            sticky_sessions: Some(false),
        };
        let router = CanaryRouter::from_config(&config, Path::new("/models"), "cpu");

        // Generate 1000 random-ish hashes and check distribution
        let mut canary_count = 0;
        for i in 0..1000 {
            let hash = format!("{:08x}", i * 7919); // Prime multiplier for pseudo-randomness
            if router.should_canary(&hash, None) {
                canary_count += 1;
            }
        }

        // Should be approximately 50% (allow 35-65% range)
        let percentage = (canary_count as f64 / 1000.0) * 100.0;
        assert!(percentage >= 35.0, "got {:.1}%, expected ~50%", percentage);
        assert!(percentage <= 65.0, "got {:.1}%, expected ~50%", percentage);
    }

    #[test]
    fn test_config_weight_clamping() {
        let config = CanaryConfig {
            model: "canary.onnx".to_string(),
            weight: 150, // Exceeds 100
            device: None,
            sticky_sessions: None,
        };
        let router = CanaryRouter::from_config(&config, Path::new("/models"), "cpu");

        // Weight should be clamped to 100
        assert!(router.should_canary("00000000", None));
    }

    #[test]
    fn test_model_path_construction() {
        let config = CanaryConfig {
            model: "v2/model.onnx".to_string(),
            weight: 10,
            device: None,
            sticky_sessions: None,
        };
        let base = Path::new("/opt/models");
        let router = CanaryRouter::from_config(&config, base, "cpu");

        assert_eq!(router.model_path(), Path::new("/opt/models/v2/model.onnx"));
    }

    #[test]
    fn test_device_override() {
        let config = CanaryConfig {
            model: "canary.onnx".to_string(),
            weight: 50,
            device: Some("cuda".to_string()),
            sticky_sessions: None,
        };
        let router = CanaryRouter::from_config(&config, Path::new("/models"), "cpu");

        assert_eq!(router.device(), "cuda");
    }

    #[test]
    fn test_device_fallback_to_primary() {
        let config = CanaryConfig {
            model: "canary.onnx".to_string(),
            weight: 50,
            device: None,
            sticky_sessions: None,
        };
        let router = CanaryRouter::from_config(&config, Path::new("/models"), "tensorrt");

        assert_eq!(router.device(), "tensorrt");
    }
}
