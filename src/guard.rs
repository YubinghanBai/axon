//! Input validation gate for Axon pipelines.
//!
//! Validates requests before any compute happens. Catches bad inputs
//! at the gate with human-readable error messages.
//!
//! Configured via `[guard]` in manifest.toml:
//!
//! ```toml
//! [guard]
//! max_size = "10MB"
//! allowed_content_types = ["image/jpeg", "image/png", "application/json"]
//! required_json_fields = ["image", "model_id"]
//! ```

use crate::manifest::GuardConfig;

// ── Guard rules ─────────────────────────────────────────────────

/// A single validation rule applied to incoming requests.
pub enum GuardRule {
    /// Maximum input size in bytes.
    MaxSize(usize),
    /// Allowed MIME content types.
    ContentType(Vec<String>),
    /// Required JSON fields (checked only for JSON/text inputs).
    JsonSchema { required_fields: Vec<String> },
}

/// Validation error with rule name and human-readable message.
#[derive(Debug)]
pub struct GuardError {
    pub rule: String,
    pub message: String,
}

impl std::fmt::Display for GuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.rule, self.message)
    }
}

// ── InputGuard ──────────────────────────────────────────────────

/// Pre-processing validation gate. Runs before any compute.
pub struct InputGuard {
    rules: Vec<GuardRule>,
}

impl InputGuard {
    /// Build an InputGuard from manifest config.
    pub fn from_config(config: &GuardConfig) -> Self {
        let mut rules = Vec::new();

        if let Some(ref max_size) = config.max_size {
            rules.push(GuardRule::MaxSize(parse_size(max_size)));
        }
        if let Some(ref types) = config.allowed_content_types {
            rules.push(GuardRule::ContentType(types.clone()));
        }
        if let Some(ref fields) = config.required_json_fields {
            rules.push(GuardRule::JsonSchema {
                required_fields: fields.clone(),
            });
        }

        Self { rules }
    }

    /// Validate input against all configured rules.
    ///
    /// Returns `Ok(())` if all rules pass, or `Err(GuardError)` on the
    /// first rule that fails.
    pub fn validate(&self, input: &[u8], content_type: &str) -> Result<(), GuardError> {
        for rule in &self.rules {
            match rule {
                GuardRule::MaxSize(max) => {
                    if input.len() > *max {
                        return Err(GuardError {
                            rule: "max_size".into(),
                            message: format!(
                                "input size {} bytes exceeds maximum {} bytes",
                                input.len(),
                                max
                            ),
                        });
                    }
                }
                GuardRule::ContentType(allowed) => {
                    let base_type = content_type
                        .split(';')
                        .next()
                        .unwrap_or(content_type)
                        .trim();
                    if !allowed.iter().any(|a| a == base_type) {
                        return Err(GuardError {
                            rule: "content_type".into(),
                            message: format!(
                                "content type '{}' not allowed, expected one of: {}",
                                base_type,
                                allowed.join(", ")
                            ),
                        });
                    }
                }
                GuardRule::JsonSchema { required_fields } => {
                    if content_type.contains("json") || content_type.starts_with("text/") {
                        if let Ok(json) = serde_json::from_slice::<serde_json::Value>(input) {
                            if let Some(obj) = json.as_object() {
                                for field in required_fields {
                                    if !obj.contains_key(field) {
                                        return Err(GuardError {
                                            rule: "required_field".into(),
                                            message: format!(
                                                "missing required JSON field: '{}'",
                                                field
                                            ),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Parse human-readable size string (e.g. "10MB") into bytes.
fn parse_size(s: &str) -> usize {
    let s = s.trim();
    let (num_str, multiplier) = if let Some(n) = s.strip_suffix("GB").or_else(|| s.strip_suffix("gb")) {
        (n, 1024 * 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("MB").or_else(|| s.strip_suffix("mb")) {
        (n, 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("KB").or_else(|| s.strip_suffix("kb")) {
        (n, 1024)
    } else if let Some(n) = s.strip_suffix('B').or_else(|| s.strip_suffix('b')) {
        (n, 1)
    } else {
        (s, 1)
    };
    num_str.trim().parse::<usize>().unwrap_or(0) * multiplier
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn guard_with_rules(config: GuardConfig) -> InputGuard {
        InputGuard::from_config(&config)
    }

    #[test]
    fn test_max_size_pass() {
        let guard = guard_with_rules(GuardConfig {
            max_size: Some("1KB".into()),
            ..Default::default()
        });
        assert!(guard.validate(b"hello", "text/plain").is_ok());
    }

    #[test]
    fn test_max_size_fail() {
        let guard = guard_with_rules(GuardConfig {
            max_size: Some("10B".into()),
            ..Default::default()
        });
        let big = vec![0u8; 20];
        let err = guard.validate(&big, "text/plain").unwrap_err();
        assert_eq!(err.rule, "max_size");
        assert!(err.message.contains("20 bytes"));
    }

    #[test]
    fn test_content_type_pass() {
        let guard = guard_with_rules(GuardConfig {
            allowed_content_types: Some(vec![
                "image/jpeg".into(),
                "image/png".into(),
            ]),
            ..Default::default()
        });
        assert!(guard.validate(b"data", "image/jpeg").is_ok());
        assert!(guard.validate(b"data", "image/png").is_ok());
    }

    #[test]
    fn test_content_type_fail() {
        let guard = guard_with_rules(GuardConfig {
            allowed_content_types: Some(vec!["image/jpeg".into()]),
            ..Default::default()
        });
        let err = guard.validate(b"data", "text/plain").unwrap_err();
        assert_eq!(err.rule, "content_type");
        assert!(err.message.contains("text/plain"));
    }

    #[test]
    fn test_content_type_strips_params() {
        let guard = guard_with_rules(GuardConfig {
            allowed_content_types: Some(vec!["application/json".into()]),
            ..Default::default()
        });
        assert!(guard.validate(b"{}", "application/json; charset=utf-8").is_ok());
    }

    #[test]
    fn test_required_json_fields_pass() {
        let guard = guard_with_rules(GuardConfig {
            required_json_fields: Some(vec!["image".into(), "model_id".into()]),
            ..Default::default()
        });
        let input = br#"{"image": "data", "model_id": "yolo"}"#;
        assert!(guard.validate(input, "application/json").is_ok());
    }

    #[test]
    fn test_required_json_fields_fail() {
        let guard = guard_with_rules(GuardConfig {
            required_json_fields: Some(vec!["image".into(), "model_id".into()]),
            ..Default::default()
        });
        let input = br#"{"image": "data"}"#;
        let err = guard.validate(input, "application/json").unwrap_err();
        assert_eq!(err.rule, "required_field");
        assert!(err.message.contains("model_id"));
    }

    #[test]
    fn test_required_json_fields_skipped_for_binary() {
        let guard = guard_with_rules(GuardConfig {
            required_json_fields: Some(vec!["image".into()]),
            ..Default::default()
        });
        // Binary content type → JSON schema check is skipped.
        assert!(guard.validate(b"\x00\x01\x02", "image/jpeg").is_ok());
    }

    #[test]
    fn test_parse_size() {
        assert_eq!(parse_size("10MB"), 10 * 1024 * 1024);
        assert_eq!(parse_size("1GB"), 1024 * 1024 * 1024);
        assert_eq!(parse_size("512KB"), 512 * 1024);
        assert_eq!(parse_size("100B"), 100);
        assert_eq!(parse_size("42"), 42);
    }

    #[test]
    fn test_multiple_rules() {
        let guard = guard_with_rules(GuardConfig {
            max_size: Some("1MB".into()),
            allowed_content_types: Some(vec!["image/jpeg".into()]),
            required_json_fields: None,
        });
        // Right type, small enough.
        assert!(guard.validate(b"data", "image/jpeg").is_ok());
        // Wrong type.
        assert!(guard.validate(b"data", "text/plain").is_err());
    }

    #[test]
    fn test_empty_guard() {
        let guard = guard_with_rules(GuardConfig::default());
        assert!(guard.validate(b"anything", "anything/whatever").is_ok());
    }
}
