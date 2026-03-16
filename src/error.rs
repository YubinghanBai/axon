//! Typed error types for the Axon ML runtime.
//!
//! `AxonError` provides structured error categories with numeric error codes
//! for programmatic matching, kernel context for debugging, and backward
//! compatibility via `From<String>` and `From<&str>` conversions.
//!
//! ## Error codes
//!
//! | Code | Kind | Description |
//! |------|------|-------------|
//! | AX001 | Config | Bad configuration, parameters, or operation spec |
//! | AX002 | Shape | Tensor shape mismatch or invalid dimensions |
//! | AX003 | Input | Missing or invalid input data |
//! | AX004 | Runtime | Runtime execution failure |
//! | AX005 | Unsupported | Unsupported operation or feature not enabled |
//! | AX006 | Io | I/O error (file, network) |
//! | AX007 | ResourceLimit | Resource limit exceeded (memory, fuel, timeout) |
//! | AX008 | Model | Model loading or inference failure |
//! | AX009 | DataFormat | Data format or encoding error |

/// Error category for programmatic matching.
///
/// Each variant has a stable numeric code (1-9) used in error messages
/// as `AX001`–`AX009`. Match on `ErrorKind` to handle error categories
/// without parsing strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// AX001: Bad configuration, parameters, or operation spec.
    Config = 1,
    /// AX002: Tensor shape mismatch or invalid dimensions.
    Shape = 2,
    /// AX003: Missing or invalid input data.
    Input = 3,
    /// AX004: Runtime execution failure (general catch-all).
    Runtime = 4,
    /// AX005: Unsupported operation or feature not enabled.
    Unsupported = 5,
    /// AX006: I/O error (file read, network, etc.).
    Io = 6,
    /// AX007: Resource limit exceeded (memory, fuel, timeout).
    ResourceLimit = 7,
    /// AX008: Model loading or inference failure.
    Model = 8,
    /// AX009: Data format or encoding error (base64, JSON, etc.).
    DataFormat = 9,
}

impl ErrorKind {
    /// Numeric error code for this category.
    pub fn code(self) -> u16 {
        self as u16
    }

    /// Short label for this category.
    pub fn label(self) -> &'static str {
        match self {
            Self::Config => "config",
            Self::Shape => "shape",
            Self::Input => "input",
            Self::Runtime => "runtime",
            Self::Unsupported => "unsupported",
            Self::Io => "io",
            Self::ResourceLimit => "resource_limit",
            Self::Model => "model",
            Self::DataFormat => "data_format",
        }
    }
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Rich error type for Axon kernel execution.
///
/// Carries a categorized error kind, human-readable message, optional
/// kernel name for context, and optional source error for chaining.
///
/// # Programmatic matching
///
/// ```rust,ignore
/// match err.kind() {
///     ErrorKind::Shape => eprintln!("fix your tensor dimensions"),
///     ErrorKind::Config => eprintln!("check your manifest.toml"),
///     _ => eprintln!("kernel error: {err}"),
/// }
/// ```
#[derive(Debug)]
pub struct AxonError {
    kind: ErrorKind,
    message: String,
    kernel: Option<&'static str>,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl AxonError {
    /// Create an error with full context.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            kernel: None,
            source: None,
        }
    }

    /// Attach kernel name for debugging context.
    pub fn in_kernel(mut self, kernel: &'static str) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Attach a source error for chaining.
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// Error category for programmatic matching.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Numeric error code (1–9).
    pub fn code(&self) -> u16 {
        self.kind.code()
    }

    /// Which kernel produced this error, if known.
    pub fn kernel(&self) -> Option<&str> {
        self.kernel
    }

    /// The error message (without code/kernel prefix).
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Check if the error message contains a substring.
    ///
    /// Convenience for test assertions and error matching.
    /// Searches the raw message, not the formatted Display output.
    pub fn contains(&self, pattern: &str) -> bool {
        self.message.contains(pattern)
    }

    // ── Convenience constructors ─────────────────────────────────

    /// Configuration error (AX001).
    pub fn config(msg: impl Into<String>) -> Self {
        Self::new(ErrorKind::Config, msg)
    }

    /// Shape mismatch error (AX002).
    pub fn shape(msg: impl Into<String>) -> Self {
        Self::new(ErrorKind::Shape, msg)
    }

    /// Input error (AX003).
    pub fn input(msg: impl Into<String>) -> Self {
        Self::new(ErrorKind::Input, msg)
    }

    /// Runtime execution error (AX004).
    pub fn runtime(msg: impl Into<String>) -> Self {
        Self::new(ErrorKind::Runtime, msg)
    }

    /// Unsupported operation (AX005).
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::new(ErrorKind::Unsupported, msg)
    }

    /// Resource limit exceeded (AX007).
    pub fn resource_limit(msg: impl Into<String>) -> Self {
        Self::new(ErrorKind::ResourceLimit, msg)
    }

    /// Model error (AX008).
    pub fn model(msg: impl Into<String>) -> Self {
        Self::new(ErrorKind::Model, msg)
    }

    /// Data format error (AX009).
    pub fn data_format(msg: impl Into<String>) -> Self {
        Self::new(ErrorKind::DataFormat, msg)
    }
}

impl std::fmt::Display for AxonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[AX{:03}]", self.kind.code())?;
        if let Some(kernel) = self.kernel {
            write!(f, " {kernel}:")?;
        }
        write!(f, " {}", self.message)
    }
}

impl std::error::Error for AxonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

// ── Backward-compatible conversions ─────────────────────────────
//
// These allow internal helpers to keep returning `Result<_, String>` —
// the `?` operator auto-converts via these impls. The smart classifier
// inspects known prefixes to route errors to the correct category.

impl From<String> for AxonError {
    fn from(s: String) -> Self {
        classify_string_error(s)
    }
}

impl From<&str> for AxonError {
    fn from(s: &str) -> Self {
        classify_string_error(s.to_string())
    }
}

impl From<std::io::Error> for AxonError {
    fn from(e: std::io::Error) -> Self {
        Self {
            kind: ErrorKind::Io,
            message: e.to_string(),
            kernel: None,
            source: Some(Box::new(e)),
        }
    }
}

/// Smart classifier: inspect error message prefixes to route to the
/// correct `ErrorKind`. This avoids needing to change every internal
/// helper function while still producing categorized errors.
fn classify_string_error(msg: String) -> AxonError {
    // Extract kernel name if message starts with "kernel_name: ..."
    let kernel = extract_kernel_prefix(&msg);

    let kind = if msg.contains("shape") && (msg.contains("expects") || msg.contains("mismatch") || msg.contains("invalid")) {
        ErrorKind::Shape
    } else if msg.contains("unknown op") || msg.contains("unsupported") || msg.contains("unknown operation") {
        ErrorKind::Unsupported
    } else if msg.contains("requires") || msg.contains("missing") || msg.contains("expected") && msg.contains("input") {
        ErrorKind::Input
    } else if msg.contains("operations must") || msg.contains("config") || msg.contains("invalid op") {
        ErrorKind::Config
    } else if msg.contains("inference") || msg.contains("model") {
        ErrorKind::Model
    } else if msg.contains("base64") || msg.contains("decode error") || msg.contains("parse error") {
        ErrorKind::DataFormat
    } else if msg.contains("fuel") || msg.contains("memory") && msg.contains("limit") {
        ErrorKind::ResourceLimit
    } else {
        ErrorKind::Runtime
    };

    AxonError {
        kind,
        message: msg,
        kernel,
        source: None,
    }
}

/// Try to extract kernel name from "kernel_name: ..." prefix.
fn extract_kernel_prefix(msg: &str) -> Option<&'static str> {
    let known_kernels: &[&str] = &[
        "tensor", "image", "detection", "onnx", "wasm", "audio", "mel", "tokenizer",
    ];
    for &k in known_kernels {
        if msg.starts_with(k) && msg.get(k.len()..k.len() + 1) == Some(":") {
            // Return 'static str from the known list.
            return Some(k);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_error_code() {
        let err = AxonError::config("bad param");
        assert_eq!(err.code(), 1);
        assert_eq!(err.kind(), ErrorKind::Config);
        assert!(err.to_string().contains("[AX001]"));
    }

    #[test]
    fn test_error_with_kernel() {
        let err = AxonError::shape("dim 5 out of range").in_kernel("tensor");
        assert_eq!(err.kernel(), Some("tensor"));
        assert!(err.to_string().contains("tensor:"));
        assert!(err.to_string().contains("[AX002]"));
    }

    #[test]
    fn test_classify_shape_error() {
        let err: AxonError = "tensor: shape [2,3] expects 6 elements, got 4".into();
        assert_eq!(err.kind(), ErrorKind::Shape);
        assert_eq!(err.kernel(), Some("tensor"));
    }

    #[test]
    fn test_classify_unsupported() {
        let err: AxonError = "detection: unknown operation 'foo'".into();
        assert_eq!(err.kind(), ErrorKind::Unsupported);
        assert_eq!(err.kernel(), Some("detection"));
    }

    #[test]
    fn test_classify_input() {
        let err: AxonError = "image: decode requires blob input with raw bytes".into();
        assert_eq!(err.kind(), ErrorKind::Input);
        assert_eq!(err.kernel(), Some("image"));
    }

    #[test]
    fn test_classify_config() {
        let err: AxonError = "image: operations must have 'op' field".into();
        assert_eq!(err.kind(), ErrorKind::Config);
    }

    #[test]
    fn test_classify_model() {
        let err: AxonError = "onnx: inference failed: bad input".into();
        assert_eq!(err.kind(), ErrorKind::Model);
        assert_eq!(err.kernel(), Some("onnx"));
    }

    #[test]
    fn test_classify_runtime_fallback() {
        let err: AxonError = "something went wrong".into();
        assert_eq!(err.kind(), ErrorKind::Runtime);
        assert_eq!(err.kernel(), None);
    }

    #[test]
    fn test_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = AxonError::from(io_err);
        assert_eq!(err.kind(), ErrorKind::Io);
        assert!(err.source().is_some());
    }

    #[test]
    fn test_error_chaining() {
        let inner = std::io::Error::new(std::io::ErrorKind::Other, "disk full");
        let err = AxonError::runtime("failed to save blob").with_source(inner);
        assert!(err.source().is_some());
    }
}
