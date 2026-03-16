//! Model Warm Pool — LRU-cached on-demand pipeline loading.
//!
//! Instead of loading all models into memory at startup, the pool
//! registers manifest paths and loads pipelines lazily on first request.
//! Least-recently-used pipelines are evicted when the pool exceeds
//! `max_models`, freeing ORT sessions and GPU memory.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let mut pool = ModelPool::new(5); // keep at most 5 models warm
//! pool.register("yolo", "models/yolov8n.toml");
//! pool.register("bert", "models/bert.toml");
//!
//! let pipeline = pool.get("yolo")?; // loaded on first call, cached
//! let output = pipeline.run(input, "image/jpeg")?;
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use moka::sync::Cache;
use tracing::{debug, info, warn};

use crate::pipeline::{Pipeline, PipelineError};

/// LRU-cached pool of loaded ML pipelines.
///
/// Pipelines are loaded on-demand and cached. When the pool exceeds
/// `max_models`, the least-recently-used pipeline is evicted (its ORT
/// session and memory are freed).
pub struct ModelPool {
    /// LRU cache: name → loaded pipeline.
    cache: Cache<String, Arc<Pipeline>>,
    /// All registered manifests: name → path.
    manifests: HashMap<String, PathBuf>,
}

impl ModelPool {
    /// Create a new pool with the given capacity.
    pub fn new(max_models: u64) -> Self {
        let cache = Cache::builder().max_capacity(max_models).build();
        Self {
            cache,
            manifests: HashMap::new(),
        }
    }

    /// Register a pipeline manifest (does NOT load the model).
    pub fn register(&mut self, name: impl Into<String>, manifest_path: impl AsRef<Path>) {
        let name = name.into();
        let path = manifest_path.as_ref().to_path_buf();
        info!(pipeline = %name, path = %path.display(), "pool: registered");
        self.manifests.insert(name, path);
    }

    /// Get a pipeline by name, loading it on-demand if needed.
    ///
    /// Returns a cached `Arc<Pipeline>` on cache hit, or loads from
    /// disk on cache miss. LRU eviction happens automatically.
    pub fn get(&self, name: &str) -> Result<Arc<Pipeline>, PipelineError> {
        // Cache hit — fast path.
        if let Some(pipeline) = self.cache.get(name) {
            debug!(pipeline = %name, "pool: cache hit");
            return Ok(pipeline);
        }

        // Cache miss — load from disk.
        let path = self.manifests.get(name).ok_or_else(|| {
            PipelineError::Parse(format!("pool: unknown pipeline '{name}'"))
        })?;

        info!(
            pipeline = %name,
            path = %path.display(),
            loaded = self.cache.entry_count(),
            max = self.cache.policy().max_capacity().unwrap_or(0),
            "pool: loading (cache miss)"
        );

        let pipeline = Pipeline::load(path)?;

        if let Err(missing) = pipeline.validate() {
            warn!(
                pipeline = %name,
                missing = ?missing,
                "pool: pipeline has missing kernels"
            );
        }

        let pipeline = Arc::new(pipeline);
        self.cache.insert(name.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    /// Check if a pipeline name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.manifests.contains_key(name)
    }

    /// List all registered pipeline names.
    pub fn names(&self) -> Vec<&str> {
        self.manifests.keys().map(|s| s.as_str()).collect()
    }

    /// Number of currently loaded (warm) pipelines.
    pub fn loaded_count(&self) -> u64 {
        self.cache.entry_count()
    }

    /// Maximum number of models the pool can hold.
    pub fn max_capacity(&self) -> u64 {
        self.cache.policy().max_capacity().unwrap_or(0)
    }

    /// Preload specific pipelines into the cache.
    ///
    /// Returns errors for any that failed to load.
    pub fn preload(&self, names: &[&str]) -> Vec<(String, PipelineError)> {
        let mut errors = Vec::new();
        for name in names {
            if let Err(e) = self.get(name) {
                errors.push((name.to_string(), e));
            }
        }
        errors
    }

    /// Evict a specific pipeline from the cache (frees memory).
    pub fn evict(&self, name: &str) {
        self.cache.invalidate(name);
        debug!(pipeline = %name, "pool: evicted");
    }

    /// Evict all pipelines from the cache.
    pub fn evict_all(&self) {
        self.cache.invalidate_all();
        debug!("pool: evicted all");
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a temp manifest and return its path.
    fn write_temp_manifest(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(format!("{name}.toml"));
        std::fs::write(
            &path,
            format!(
                r#"
[model]
name = "{name}"
file = "model.onnx"
"#
            ),
        )
        .unwrap();
        path
    }

    #[test]
    fn test_pool_register_and_names() {
        let mut pool = ModelPool::new(5);
        pool.register("a", "/tmp/a.toml");
        pool.register("b", "/tmp/b.toml");
        let mut names = pool.names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_pool_contains() {
        let mut pool = ModelPool::new(5);
        pool.register("yolo", "/tmp/yolo.toml");
        assert!(pool.contains("yolo"));
        assert!(!pool.contains("bert"));
    }

    #[test]
    fn test_pool_unknown_pipeline() {
        let pool = ModelPool::new(5);
        let result = pool.get("nonexistent");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("unknown"));
    }

    #[test]
    fn test_pool_load_and_cache() {
        let tmp = std::env::temp_dir().join("axon_test_pool");
        std::fs::create_dir_all(&tmp).unwrap();
        let manifest_path = write_temp_manifest(&tmp, "test-pool");

        let mut pool = ModelPool::new(5);
        pool.register("test", &manifest_path);

        assert_eq!(pool.loaded_count(), 0);

        // First call loads.
        let result = pool.get("test");
        // Will fail validation (no onnx kernel) but should load the manifest.
        // The pipeline struct is created even without onnx — it just can't run inference.
        assert!(result.is_ok());
        pool.cache.run_pending_tasks();
        assert_eq!(pool.loaded_count(), 1);

        // Second call hits cache.
        let result2 = pool.get("test");
        assert!(result2.is_ok());
        pool.cache.run_pending_tasks();
        assert_eq!(pool.loaded_count(), 1);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_pool_eviction() {
        let tmp = std::env::temp_dir().join("axon_test_pool_evict");
        std::fs::create_dir_all(&tmp).unwrap();

        // Pool with capacity 2.
        let mut pool = ModelPool::new(2);

        for name in ["a", "b", "c"] {
            let path = write_temp_manifest(&tmp, name);
            pool.register(name, &path);
        }

        // Load a and b.
        pool.get("a").unwrap();
        pool.get("b").unwrap();
        pool.cache.run_pending_tasks();
        assert_eq!(pool.loaded_count(), 2);

        // Loading c should evict the LRU (a).
        pool.get("c").unwrap();
        // moka eviction is async — run pending to flush.
        pool.cache.run_pending_tasks();
        assert!(pool.loaded_count() <= 2);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_pool_manual_evict() {
        let tmp = std::env::temp_dir().join("axon_test_pool_manual");
        std::fs::create_dir_all(&tmp).unwrap();
        let path = write_temp_manifest(&tmp, "evictme");

        let mut pool = ModelPool::new(5);
        pool.register("evictme", &path);
        pool.get("evictme").unwrap();
        pool.cache.run_pending_tasks();
        assert_eq!(pool.loaded_count(), 1);

        pool.evict("evictme");
        pool.cache.run_pending_tasks();
        assert_eq!(pool.loaded_count(), 0);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_pool_max_capacity() {
        let pool = ModelPool::new(10);
        assert_eq!(pool.max_capacity(), 10);
    }
}
