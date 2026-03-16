//! Blob types and content-addressable storage for the Axon ML runtime.
//!
//! BlobId = BLAKE3(content). Same content always produces the same ID,
//! giving natural deduplication with no reference counting or GC.
//!
//! Two tiers:
//! - moka LRU cache (bounded in-memory, configurable max capacity)
//! - Optional disk directory (content-addressed filenames: `{blake3_hex}.blob`)

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use moka::sync::Cache;
use serde::{Deserialize, Serialize};

// ── Blob Identifier ────────────────────────────────────────────

/// Content-addressed identifier for a blob in the BlobStore.
///
/// 256-bit BLAKE3 hash of the raw byte content. This means:
/// - Same content always produces the same BlobId (natural deduplication).
/// - BlobId doubles as the integrity checksum.
/// - Disk filename = hex(BlobId), no mapping table needed.
///
/// Serializes as a 64-character hex string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlobId(pub [u8; 32]);

impl BlobId {
    /// Compute BlobId from content bytes (BLAKE3 hash).
    pub fn from_content(data: &[u8]) -> Self {
        Self(*blake3::hash(data).as_bytes())
    }

    /// Create a BlobId for testing only (random, not content-derived).
    #[cfg(test)]
    pub fn new_test() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(1);
        let mut bytes = [0u8; 32];
        let c = CTR.fetch_add(1, Ordering::Relaxed);
        bytes[..8].copy_from_slice(&c.to_le_bytes());
        Self(bytes)
    }

    /// Parse from a 64-character hex string.
    pub fn from_hex(s: &str) -> Result<Self, &'static str> {
        if s.len() != 64 {
            return Err("BlobId hex string must be 64 characters");
        }
        let mut bytes = [0u8; 32];
        for i in 0..32 {
            bytes[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
                .map_err(|_| "invalid hex character")?;
        }
        Ok(Self(bytes))
    }
}

impl fmt::Display for BlobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Serialize for BlobId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for BlobId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

// ── Blob Metadata ──────────────────────────────────────────────

/// Metadata describing a blob's content.
///
/// Stored alongside the BlobRef in Signals and Ledger fingerprints.
/// Consumers use this to decide how to interpret the raw bytes
/// (lazy materialization).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlobMeta {
    /// Size in bytes.
    pub size: u64,

    /// Content type hint (e.g. "tensor/f32", "image/png", "application/octet-stream").
    pub content_type: String,

    /// Tensor shape, if applicable (e.g. [1, 128, 768]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<usize>>,
}

// ── Blob Reference ─────────────────────────────────────────────

/// Reference to raw bytes stored in the BlobStore.
///
/// This is the "pointer" that flows through the signal graph inside
/// `Payload::Blob`. The actual data lives externally in BlobStore;
/// consumers retrieve it lazily when they need the bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlobRef {
    /// Unique blob identifier (also the filename in disk-backed BlobStore).
    pub id: BlobId,

    /// Metadata about the blob's content.
    pub meta: BlobMeta,
}

impl BlobRef {
    /// Create a Ledger-safe fingerprint: JSON with blob metadata, no raw data.
    ///
    /// Used by Cortex when writing blob Signals to the Ledger.
    /// The Ledger stores this instead of the (potentially huge) raw bytes.
    pub fn ledger_fingerprint(&self) -> serde_json::Value {
        serde_json::json!({
            "_blob": {
                "id": self.id.to_string(),
                "size": self.meta.size,
                "content_type": self.meta.content_type,
                "shape": self.meta.shape,
            }
        })
    }
}

// ── BlobStore ──────────────────────────────────────────────────

/// Default maximum number of blobs in the memory cache.
const DEFAULT_MAX_CACHE_ENTRIES: u64 = 1024;

/// Raw byte storage for large payloads (tensors, images, audio, etc.).
///
/// Thread-safe via moka's concurrent cache. Blobs are immutable once written.
/// Content-addressed: BlobId = BLAKE3 hash of the raw bytes.
pub struct BlobStore {
    /// Bounded LRU cache. Eviction is automatic when capacity is exceeded.
    cache: Cache<BlobId, Arc<Vec<u8>>>,

    /// Optional disk directory for durability.
    /// Filenames are `{blake3_hex}.blob`.
    disk_dir: Option<PathBuf>,
}

impl BlobStore {
    /// Create an in-memory-only BlobStore. Blobs are lost on process exit.
    pub fn in_memory() -> Self {
        Self {
            cache: Cache::new(DEFAULT_MAX_CACHE_ENTRIES),
            disk_dir: None,
        }
    }

    /// Create an in-memory-only BlobStore with a custom cache capacity.
    pub fn in_memory_with_capacity(max_entries: u64) -> Self {
        Self {
            cache: Cache::new(max_entries),
            disk_dir: None,
        }
    }

    /// Create a BlobStore backed by a directory on disk.
    ///
    /// Blobs are written to both cache and disk. On cache miss, disk is used
    /// as fallback and the entry is warmed back into the cache.
    pub fn with_dir(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            cache: Cache::new(DEFAULT_MAX_CACHE_ENTRIES),
            disk_dir: Some(dir),
        })
    }

    /// Store raw bytes. Returns a BlobRef with the content-derived BlobId.
    ///
    /// Idempotent: storing the same content twice returns the same BlobId
    /// without duplicating data. The caller provides content metadata.
    pub fn put(
        &self,
        data: Vec<u8>,
        content_type: impl Into<String>,
        shape: Option<Vec<usize>>,
    ) -> BlobRef {
        let id = BlobId::from_content(&data);
        let size = data.len() as u64;
        let content_type = content_type.into();

        // Dedup: skip write if already present in cache or on disk.
        if !self.contains(&id) {
            // Write to disk if configured.
            if let Some(ref dir) = self.disk_dir {
                let path = dir.join(format!("{id}.blob"));
                if let Err(e) = std::fs::write(&path, &data) {
                    tracing::warn!(blob_id = %id, "blob disk write failed: {e}");
                }
            }

            // Insert into cache.
            self.cache.insert(id, Arc::new(data));
        }

        BlobRef {
            id,
            meta: BlobMeta {
                size,
                content_type,
                shape,
            },
        }
    }

    /// Read raw bytes by BlobId.
    ///
    /// Tries cache first, then disk. Returns None if the blob is not found.
    pub fn get(&self, id: &BlobId) -> Option<Vec<u8>> {
        // Fast path: cache hit.
        if let Some(data) = self.cache.get(id) {
            return Some((*data).clone());
        }

        // Slow path: disk fallback.
        if let Some(ref dir) = self.disk_dir {
            let path = dir.join(format!("{id}.blob"));
            if let Ok(data) = std::fs::read(&path) {
                // Warm the cache for subsequent reads.
                self.cache.insert(*id, Arc::new(data.clone()));
                return Some(data);
            }
        }

        None
    }

    /// Check if a blob exists (in cache or on disk).
    pub fn contains(&self, id: &BlobId) -> bool {
        if self.cache.contains_key(id) {
            return true;
        }
        if let Some(ref dir) = self.disk_dir {
            return dir.join(format!("{id}.blob")).exists();
        }
        false
    }

    /// Number of blobs currently held in the memory cache.
    pub fn cache_count(&self) -> u64 {
        self.cache.entry_count()
    }

    /// Invalidate a cache entry (blob remains on disk if persisted).
    /// Primarily for testing.
    pub fn evict_from_cache(&self, id: &BlobId) {
        self.cache.invalidate(id);
    }
}

impl fmt::Debug for BlobStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlobStore")
            .field("cache_count", &self.cache_count())
            .field("disk_dir", &self.disk_dir)
            .finish()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_id_content_addressed() {
        let data = b"hello world";
        let a = BlobId::from_content(data);
        let b = BlobId::from_content(data);
        assert_eq!(a, b, "same content must produce same BlobId");

        let c = BlobId::from_content(b"different content");
        assert_ne!(a, c, "different content must produce different BlobId");
    }

    #[test]
    fn blob_id_roundtrip() {
        let id = BlobId::from_content(b"test data");
        let hex = id.to_string();
        assert_eq!(hex.len(), 64);

        let parsed = BlobId::from_hex(&hex).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn blob_id_serde_json() {
        let id = BlobId::from_content(b"serde test");
        let json = serde_json::to_string(&id).unwrap();
        assert!(json.starts_with('"'));
        assert_eq!(json.len(), 66); // 64 hex chars + 2 quotes

        let parsed: BlobId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn blob_ref_ledger_fingerprint() {
        let blob_ref = BlobRef {
            id: BlobId::from_content(b"tensor data"),
            meta: BlobMeta {
                size: 49152,
                content_type: "tensor/f32".to_string(),
                shape: Some(vec![1, 128, 96]),
            },
        };

        let fp = blob_ref.ledger_fingerprint();
        assert_eq!(fp["_blob"]["size"], 49152);
        assert_eq!(fp["_blob"]["content_type"], "tensor/f32");
        assert_eq!(fp["_blob"]["shape"], serde_json::json!([1, 128, 96]));
    }

    #[test]
    fn put_and_get_in_memory() {
        let store = BlobStore::in_memory();
        let data = vec![1, 2, 3, 4, 5];
        let blob_ref = store.put(data.clone(), "application/octet-stream", None);

        assert_eq!(blob_ref.meta.size, 5);
        assert_eq!(blob_ref.meta.content_type, "application/octet-stream");
        assert!(blob_ref.meta.shape.is_none());

        let retrieved = store.get(&blob_ref.id).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn content_addressable_dedup() {
        let store = BlobStore::in_memory();
        let data = vec![10, 20, 30];

        let ref1 = store.put(data.clone(), "test", None);
        let ref2 = store.put(data.clone(), "test", None);

        assert_eq!(ref1.id, ref2.id);
        store.cache.run_pending_tasks();
        assert_eq!(store.cache_count(), 1);
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let store = BlobStore::in_memory();
        let fake_id = BlobId::from_content(b"nonexistent data");
        assert!(store.get(&fake_id).is_none());
    }

    #[test]
    fn metadata_preserved() {
        let store = BlobStore::in_memory();
        let shape = vec![1, 128, 768];
        let blob_ref = store.put(
            vec![0u8; 393216],
            "tensor/f32",
            Some(shape.clone()),
        );

        assert_eq!(blob_ref.meta.size, 393216);
        assert_eq!(blob_ref.meta.content_type, "tensor/f32");
        assert_eq!(blob_ref.meta.shape, Some(shape));
    }
}
