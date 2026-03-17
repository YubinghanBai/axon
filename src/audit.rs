//! Append-only audit trail for inference requests.
//!
//! Writes JSONL entries for every inference request (or a sampled subset)
//! for compliance, debugging, and cost attribution.
//!
//! Configured via `[audit]` in manifest.toml:
//!
//! ```toml
//! [audit]
//! path = "/var/log/axon/inference.jsonl"
//! sample_rate = 1.0
//! ```

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};

use serde::Serialize;

// ── AuditEntry ──────────────────────────────────────────────────

/// A single audit log entry written as one JSONL line.
#[derive(Debug, Serialize)]
pub struct AuditEntry {
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// Pipeline name.
    pub pipeline: String,
    /// Blake3 hash of raw input bytes (hex).
    pub input_hash: String,
    /// Blake3 hash of output (hex).
    pub output_hash: String,
    /// "ok" or "error".
    pub status: &'static str,
    /// Total request latency in microseconds.
    pub latency_us: u64,
    /// Device used for inference.
    pub device: String,
    /// Whether the inference cache was hit.
    pub cache_hit: bool,
    /// Session ID if stateful.
    pub session_id: Option<String>,
}

// ── AuditLogger ─────────────────────────────────────────────────

/// Append-only JSONL audit logger.
///
/// Thread-safe via `parking_lot::Mutex`. Writes are fire-and-forget —
/// audit failures never block or fail the inference pipeline.
pub struct AuditLogger {
    writer: parking_lot::Mutex<BufWriter<File>>,
    sample_rate: f64,
}

impl AuditLogger {
    /// Create a new AuditLogger.
    ///
    /// Opens (or creates) the log file in append mode.
    /// `sample_rate` is 0.0–1.0 (1.0 = log everything).
    pub fn new(path: &str, sample_rate: f64) -> Result<Self, std::io::Error> {
        // Ensure parent directory exists.
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            writer: parking_lot::Mutex::new(BufWriter::new(file)),
            sample_rate: sample_rate.clamp(0.0, 1.0),
        })
    }

    /// Log an audit entry. Fire-and-forget: never fails the pipeline.
    ///
    /// Respects sample_rate: if rate < 1.0, entries are randomly sampled.
    pub fn log(&self, entry: &AuditEntry) {
        // Sampling: skip if random value exceeds sample rate.
        if self.sample_rate < 1.0 {
            // Simple deterministic sampling based on input hash.
            // Parse first 8 hex chars of input_hash as u32, normalize to [0,1).
            let sample_val = u32::from_str_radix(
                &entry.input_hash[..entry.input_hash.len().min(8)],
                16,
            )
            .unwrap_or(0) as f64
                / u32::MAX as f64;
            if sample_val >= self.sample_rate {
                return;
            }
        }

        // Serialize and write. Errors are silently ignored.
        if let Ok(line) = serde_json::to_string(entry) {
            let mut writer = self.writer.lock();
            let _ = writeln!(writer, "{}", line);
            let _ = writer.flush();
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// Compute blake3 hash of bytes and return as hex string.
pub fn hash_bytes(data: &[u8]) -> String {
    let hash = blake3::hash(data);
    hash.to_hex().to_string()
}

/// Compute blake3 hash of a KernelOutput for audit logging.
pub fn hash_output(output: &crate::kernel::KernelOutput) -> String {
    match output {
        crate::kernel::KernelOutput::Json(v) => hash_bytes(v.to_string().as_bytes()),
        crate::kernel::KernelOutput::Blob { data, .. } => hash_bytes(data),
    }
}

/// Get current timestamp in ISO 8601 format.
pub fn now_iso8601() -> String {
    // Use std::time for a simple UTC timestamp.
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Simple ISO 8601 without external crate.
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;
    let millis = dur.subsec_millis();

    // Approximate date from days since epoch (good enough for logging).
    let (year, month, day) = days_to_date(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, hours, minutes, seconds, millis
    )
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_date(days: u64) -> (u64, u64, u64) {
    // Algorithm from Howard Hinnant's date library.
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_logger_writes_jsonl() {
        let dir = std::env::temp_dir().join("axon_audit_test");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("test.jsonl");

        let logger = AuditLogger::new(path.to_str().unwrap(), 1.0).unwrap();

        let entry = AuditEntry {
            timestamp: "2026-01-01T00:00:00.000Z".into(),
            pipeline: "yolo".into(),
            input_hash: "abc123def456".into(),
            output_hash: "789xyz".into(),
            status: "ok",
            latency_us: 42000,
            device: "cpu".into(),
            cache_hit: false,
            session_id: None,
        };

        logger.log(&entry);

        // Read back and verify.
        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed["pipeline"], "yolo");
        assert_eq!(parsed["status"], "ok");
        assert_eq!(parsed["latency_us"], 42000);
        assert_eq!(parsed["cache_hit"], false);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_audit_logger_creates_parent_dirs() {
        let dir = std::env::temp_dir().join("axon_audit_nested/sub1/sub2");
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join("axon_audit_nested"));
        let path = dir.join("audit.jsonl");

        let logger = AuditLogger::new(path.to_str().unwrap(), 1.0);
        assert!(logger.is_ok());

        let _ = std::fs::remove_dir_all(std::env::temp_dir().join("axon_audit_nested"));
    }

    #[test]
    fn test_audit_sampling() {
        let dir = std::env::temp_dir().join("axon_audit_sample");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("sampled.jsonl");

        // 0% sample rate → nothing logged.
        let logger = AuditLogger::new(path.to_str().unwrap(), 0.0).unwrap();

        let entry = AuditEntry {
            timestamp: "2026-01-01T00:00:00.000Z".into(),
            pipeline: "test".into(),
            input_hash: "ffffffff".into(),
            output_hash: "00000000".into(),
            status: "ok",
            latency_us: 1000,
            device: "cpu".into(),
            cache_hit: false,
            session_id: None,
        };

        logger.log(&entry);

        let content = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(content.is_empty(), "0% sample rate should log nothing");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_hash_bytes() {
        let h1 = hash_bytes(b"hello");
        let h2 = hash_bytes(b"hello");
        let h3 = hash_bytes(b"world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_eq!(h1.len(), 64); // blake3 hex is 64 chars
    }

    #[test]
    fn test_now_iso8601() {
        let ts = now_iso8601();
        assert!(ts.contains('T'));
        assert!(ts.ends_with('Z'));
        assert!(ts.len() >= 20);
    }
}
