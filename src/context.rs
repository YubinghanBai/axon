//! Session context store for stateful pipeline execution.
//!
//! Enables cross-request state for streaming workloads like video tracking
//! (DeepSORT), speaker diarization, and time-series anomaly detection.
//!
//! ## How it works
//!
//! Each HTTP request can carry an `X-Session-Id` header. Requests with the
//! same session ID share a persistent context (JSON key-value store) that
//! survives across requests. The context is injected into post-processing
//! steps as `_context` and extracted from their output for the next request.
//!
//! ## Example: video object tracking
//!
//! ```text
//! Frame 1 [session: cam-A]:
//!   YOLO → detections → WASM reads _context (empty) → creates tracks → writes _context
//!
//! Frame 2 [session: cam-A]:
//!   YOLO → detections → WASM reads _context (tracks from frame 1) → matches → updates
//!
//! Frame 1 [session: cam-B]:
//!   YOLO → detections → WASM reads _context (empty, different session) → creates tracks
//! ```

use std::time::Duration;

use moka::sync::Cache;

/// Thread-safe, TTL-based session context store.
///
/// Backed by moka LRU cache: idle sessions are automatically evicted
/// after `ttl` to prevent unbounded memory growth.
pub struct SessionStore {
    store: Cache<String, serde_json::Value>,
}

impl SessionStore {
    /// Create a new session store.
    ///
    /// - `ttl`: time-to-idle — sessions evicted after this much inactivity.
    /// - `max_sessions`: maximum concurrent sessions (LRU eviction).
    pub fn new(ttl: Duration, max_sessions: u64) -> Self {
        Self {
            store: Cache::builder()
                .max_capacity(max_sessions)
                .time_to_idle(ttl)
                .build(),
        }
    }

    /// Default session store: 10 min TTL, 10k max sessions.
    pub fn default_store() -> Self {
        Self::new(Duration::from_secs(600), 10_000)
    }

    /// Get the context for a session. Returns `None` for new sessions.
    pub fn get(&self, session_id: &str) -> Option<serde_json::Value> {
        self.store.get(session_id)
    }

    /// Store/update the context for a session.
    pub fn put(&self, session_id: &str, context: serde_json::Value) {
        self.store.insert(session_id.to_string(), context);
    }

    /// Remove a session's context explicitly.
    pub fn remove(&self, session_id: &str) {
        self.store.invalidate(session_id);
    }

    /// Number of active sessions (approximate — moka is eventually consistent).
    pub fn session_count(&self) -> u64 {
        self.store.run_pending_tasks();
        self.store.entry_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_store_basic() {
        let store = SessionStore::default_store();

        // New session returns None.
        assert!(store.get("cam-1").is_none());

        // Put and get.
        store.put("cam-1", serde_json::json!({"tracks": [1, 2, 3]}));
        let ctx = store.get("cam-1").unwrap();
        assert_eq!(ctx["tracks"], serde_json::json!([1, 2, 3]));

        // Different sessions are isolated.
        assert!(store.get("cam-2").is_none());

        assert_eq!(store.session_count(), 1);
    }

    #[test]
    fn test_session_store_update() {
        let store = SessionStore::default_store();

        store.put("s1", serde_json::json!({"frame": 1}));
        store.put("s1", serde_json::json!({"frame": 2}));

        let ctx = store.get("s1").unwrap();
        assert_eq!(ctx["frame"], 2);
    }

    #[test]
    fn test_session_store_remove() {
        let store = SessionStore::default_store();

        store.put("s1", serde_json::json!({"x": 1}));
        assert!(store.get("s1").is_some());

        store.remove("s1");
        assert!(store.get("s1").is_none());
    }
}
