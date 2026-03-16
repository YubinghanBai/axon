//! Zero-ETL vector DB sink kernel.
//!
//! Writes embedding vectors directly to vector databases via HTTP REST API.
//! No intermediate JSON/file conversion — Vec<f32> goes straight to the DB.
//!
//! Supported targets:
//! - `qdrant` — Qdrant REST API (PUT /collections/{name}/points)
//! - `weaviate` — Weaviate REST API (POST /v1/objects)
//! - `chromadb` — ChromaDB REST API (POST /api/v1/collections/{id}/add)
//! - `postgres` — PostgreSQL/pgvector via PostgREST/Supabase REST API
//!
//! ## Pipeline usage
//!
//! ```toml
//! [post]
//! steps = [
//!   { op = "tensor.mean_pool", dim = 1 },
//!   { op = "tensor.normalize" },
//!   { op = "sink.qdrant", url = "http://localhost:6333", collection = "docs", id = 1 },
//! ]
//! ```
//!
//! ### PostgreSQL/pgvector via PostgREST
//!
//! ```toml
//! [post]
//! steps = [
//!   { op = "tensor.mean_pool", dim = 1 },
//!   { op = "tensor.normalize" },
//!   { op = "sink.postgres", url = "https://xxx.supabase.co/rest/v1", collection = "embeddings", api_key = "..." },
//! ]
//! ```

use serde_json::{Value, json};
use tracing::{debug, info};

use crate::error::AxonError;
use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};

/// Sink kernel for writing vectors to external databases.
pub struct SinkKernel {
    client: reqwest::blocking::Client,
}

impl SinkKernel {
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("sink: failed to build HTTP client"),
        }
    }
}

impl ComputeKernel for SinkKernel {
    fn name(&self) -> &str {
        "sink"
    }

    fn execute(
        &self,
        input: KernelInput,
        operations: Value,
    ) -> Result<KernelOutput, AxonError> {
        let config = parse_config(&operations)?;
        let vector = extract_vector(&input)?;

        debug!(
            target = %config.target,
            url = %config.url,
            dim = vector.len(),
            "sink: writing vector"
        );

        let result = match config.target {
            SinkTarget::Qdrant => write_qdrant(&self.client, &config, &vector),
            SinkTarget::Weaviate => write_weaviate(&self.client, &config, &vector),
            SinkTarget::ChromaDb => write_chromadb(&self.client, &config, &vector),
            SinkTarget::Postgres => write_postgres(&self.client, &config, &vector),
        }?;

        info!(
            target = %config.target,
            dim = vector.len(),
            "sink: write complete"
        );

        Ok(KernelOutput::Json(result))
    }
}

// ── Config ─────────────────────────────────────────────────────

#[derive(Debug)]
enum SinkTarget {
    Qdrant,
    Weaviate,
    ChromaDb,
    Postgres,
}

impl std::fmt::Display for SinkTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Qdrant => write!(f, "qdrant"),
            Self::Weaviate => write!(f, "weaviate"),
            Self::ChromaDb => write!(f, "chromadb"),
            Self::Postgres => write!(f, "postgres"),
        }
    }
}

#[derive(Debug)]
struct SinkConfig {
    target: SinkTarget,
    url: String,
    collection: String,
    /// Point/object ID (string or integer depending on target).
    id: Option<Value>,
    /// Optional metadata/payload to attach.
    payload: Option<Value>,
    /// API key for authentication.
    api_key: Option<String>,
    /// Weaviate class name (defaults to collection).
    class: Option<String>,
    /// PostgreSQL vector column name (defaults to "embedding").
    vector_column: Option<String>,
}

fn parse_config(operations: &Value) -> Result<SinkConfig, String> {
    let obj = operations
        .as_object()
        .ok_or("sink: operations must be a JSON object")?;

    let op = obj
        .get("op")
        .and_then(|v| v.as_str())
        .ok_or("sink: missing 'op' field")?;

    let target = match op {
        "sink.qdrant" | "qdrant" => SinkTarget::Qdrant,
        "sink.weaviate" | "weaviate" => SinkTarget::Weaviate,
        "sink.chromadb" | "chromadb" => SinkTarget::ChromaDb,
        "sink.postgres" | "postgres" => SinkTarget::Postgres,
        other => return Err(format!("sink: unknown target '{other}'")),
    };

    let url = obj
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or("sink: missing 'url' field")?
        .trim_end_matches('/')
        .to_string();

    let collection = obj
        .get("collection")
        .and_then(|v| v.as_str())
        .ok_or("sink: missing 'collection' field")?
        .to_string();

    let id = obj.get("id").cloned();
    let payload = obj.get("payload").cloned();
    let api_key = obj
        .get("api_key")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let class = obj
        .get("class")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let vector_column = obj
        .get("vector_column")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Ok(SinkConfig {
        target,
        url,
        collection,
        id,
        payload,
        api_key,
        class,
        vector_column,
    })
}

// ── Vector extraction ──────────────────────────────────────────

/// Extract f32 vector from pipeline input.
///
/// Supports two formats:
/// 1. Blob input: raw f32 bytes from tensor kernel (zero-copy path)
/// 2. JSON input: {"data": [0.1, 0.2, ...]} from tensor.normalize
fn extract_vector(input: &KernelInput) -> Result<Vec<f32>, String> {
    // Try blob first (zero-copy from tensor pipeline).
    if let Some(blob) = input.first_blob() {
        if blob.bytes.len() % 4 != 0 {
            return Err(format!(
                "sink: blob size {} not aligned to f32 (4 bytes)",
                blob.bytes.len()
            ));
        }
        let floats: Vec<f32> = blob
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        return Ok(floats);
    }

    // Try JSON: look for "data" array of floats.
    if let Some(data) = input.json.get("data") {
        if let Some(arr) = data.as_array() {
            let floats: Result<Vec<f32>, _> = arr
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| f as f32)
                        .ok_or("sink: non-numeric value in data array")
                })
                .collect();
            return floats.map_err(|e| e.to_string());
        }
    }

    // Try flattened JSON: look for "outputs" -> first key -> "data"
    if let Some(outputs) = input.json.get("outputs").and_then(|v| v.as_object()) {
        for (_key, val) in outputs {
            if let Some(arr) = val.get("data").and_then(|v| v.as_array()) {
                let floats: Result<Vec<f32>, _> = arr
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .map(|f| f as f32)
                            .ok_or("sink: non-numeric value")
                    })
                    .collect();
                return floats.map_err(|e| e.to_string());
            }
        }
    }

    Err("sink: no vector data found (expected blob, 'data' array, or 'outputs' object)".to_string())
}

// ── Target writers ─────────────────────────────────────────────

/// Write vector to Qdrant.
/// PUT /collections/{collection}/points
fn write_qdrant(
    client: &reqwest::blocking::Client,
    config: &SinkConfig,
    vector: &[f32],
) -> Result<Value, String> {
    let id = config
        .id
        .as_ref()
        .cloned()
        .unwrap_or_else(|| json!(uuid_v4_simple()));

    let mut point = json!({
        "id": id,
        "vector": vector,
    });

    if let Some(ref payload) = config.payload {
        point["payload"] = payload.clone();
    }

    let url = format!("{}/collections/{}/points", config.url, config.collection);
    let body = json!({ "points": [point] });

    let mut req = client.put(&url).json(&body);
    if let Some(ref key) = config.api_key {
        req = req.header("api-key", key);
    }

    let resp = req
        .send()
        .map_err(|e| format!("sink: qdrant request failed: {e}"))?;

    let status = resp.status();
    let resp_body: Value = resp
        .json()
        .unwrap_or_else(|_| json!({"status": status.as_u16()}));

    if !status.is_success() {
        return Err(format!(
            "sink: qdrant error {}: {}",
            status,
            resp_body
        ));
    }

    Ok(json!({
        "sink": "qdrant",
        "collection": config.collection,
        "id": id,
        "dim": vector.len(),
        "status": "ok",
    }))
}

/// Write vector to Weaviate.
/// POST /v1/objects
fn write_weaviate(
    client: &reqwest::blocking::Client,
    config: &SinkConfig,
    vector: &[f32],
) -> Result<Value, String> {
    let class = config
        .class
        .as_deref()
        .unwrap_or(&config.collection);

    let mut body = json!({
        "class": class,
        "vector": vector,
    });

    if let Some(ref payload) = config.payload {
        body["properties"] = payload.clone();
    }

    // Add ID if specified.
    if let Some(ref id) = config.id {
        if let Some(id_str) = id.as_str() {
            body["id"] = json!(id_str);
        }
    }

    let url = format!("{}/v1/objects", config.url);

    let mut req = client.post(&url).json(&body);
    if let Some(ref key) = config.api_key {
        req = req.header("Authorization", format!("Bearer {key}"));
    }

    let resp = req
        .send()
        .map_err(|e| format!("sink: weaviate request failed: {e}"))?;

    let status = resp.status();
    let resp_body: Value = resp
        .json()
        .unwrap_or_else(|_| json!({"status": status.as_u16()}));

    if !status.is_success() {
        return Err(format!(
            "sink: weaviate error {}: {}",
            status,
            resp_body
        ));
    }

    let returned_id = resp_body
        .get("id")
        .cloned()
        .unwrap_or_else(|| json!("unknown"));

    Ok(json!({
        "sink": "weaviate",
        "class": class,
        "id": returned_id,
        "dim": vector.len(),
        "status": "ok",
    }))
}

/// Write vector to ChromaDB.
/// POST /api/v1/collections/{collection_id}/add
fn write_chromadb(
    client: &reqwest::blocking::Client,
    config: &SinkConfig,
    vector: &[f32],
) -> Result<Value, String> {
    let id = config
        .id
        .as_ref()
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid_v4_simple());

    let mut body = json!({
        "ids": [id],
        "embeddings": [vector],
    });

    if let Some(ref payload) = config.payload {
        if let Some(obj) = payload.as_object() {
            body["metadatas"] = json!([obj]);
        }
    }

    let url = format!(
        "{}/api/v1/collections/{}/add",
        config.url, config.collection
    );

    let mut req = client.post(&url).json(&body);
    if let Some(ref key) = config.api_key {
        req = req.header("Authorization", format!("Bearer {key}"));
    }

    let resp = req
        .send()
        .map_err(|e| format!("sink: chromadb request failed: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        let err_body = resp.text().unwrap_or_default();
        return Err(format!("sink: chromadb error {}: {}", status, err_body));
    }

    Ok(json!({
        "sink": "chromadb",
        "collection": config.collection,
        "id": id,
        "dim": vector.len(),
        "status": "ok",
    }))
}

/// Write vector to PostgreSQL/pgvector via PostgREST or Supabase REST API.
/// POST /{table}
///
/// The vector is sent in pgvector text format: `[0.1,0.2,0.3]`.
/// Works with any PostgREST-compatible endpoint (Supabase, standalone PostgREST, etc.).
///
/// Config fields:
/// - `url`: PostgREST base URL (e.g. `https://xxx.supabase.co/rest/v1`)
/// - `collection`: table name (e.g. `embeddings`)
/// - `vector_column`: column name for the vector (default: `embedding`)
/// - `api_key`: Supabase anon key or service role key
/// - `payload`: additional columns to insert as key-value pairs
/// - `id`: optional row ID
fn write_postgres(
    client: &reqwest::blocking::Client,
    config: &SinkConfig,
    vector: &[f32],
) -> Result<Value, String> {
    let vector_col = config.vector_column.as_deref().unwrap_or("embedding");

    // pgvector text format: '[0.1,0.2,0.3]'
    let vector_str = format!(
        "[{}]",
        vector
            .iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    let mut row = serde_json::Map::new();
    row.insert(vector_col.to_string(), json!(vector_str));

    // Merge payload fields into the row.
    if let Some(ref payload) = config.payload {
        if let Some(obj) = payload.as_object() {
            for (k, v) in obj {
                row.insert(k.clone(), v.clone());
            }
        }
    }

    // Add ID if specified.
    if let Some(ref id) = config.id {
        row.insert("id".to_string(), id.clone());
    }

    let url = format!("{}/{}", config.url, config.collection);
    let body = Value::Object(row);

    let mut req = client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("Prefer", "return=representation")
        .json(&body);

    if let Some(ref key) = config.api_key {
        // Supabase uses both apikey header and Bearer auth.
        req = req
            .header("apikey", key.as_str())
            .header("Authorization", format!("Bearer {key}"));
    }

    let resp = req
        .send()
        .map_err(|e| format!("sink: postgres request failed: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        let err_body = resp.text().unwrap_or_default();
        return Err(format!("sink: postgres error {}: {}", status, err_body));
    }

    Ok(json!({
        "sink": "postgres",
        "table": config.collection,
        "vector_column": vector_col,
        "dim": vector.len(),
        "status": "ok",
    }))
}

// ── Helpers ────────────────────────────────────────────────────

/// Generate a simple UUID v4-like string (no external dep needed).
fn uuid_v4_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:032x}", t)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::kernel::BlobData;
    use crate::blob::BlobMeta;

    #[test]
    fn test_parse_config_qdrant() {
        let ops = json!({
            "op": "sink.qdrant",
            "url": "http://localhost:6333",
            "collection": "test_collection",
            "id": 42,
        });
        let config = parse_config(&ops).unwrap();
        assert!(matches!(config.target, SinkTarget::Qdrant));
        assert_eq!(config.url, "http://localhost:6333");
        assert_eq!(config.collection, "test_collection");
        assert_eq!(config.id, Some(json!(42)));
    }

    #[test]
    fn test_parse_config_weaviate() {
        let ops = json!({
            "op": "sink.weaviate",
            "url": "http://localhost:8080",
            "collection": "Article",
            "class": "MyClass",
            "api_key": "secret",
        });
        let config = parse_config(&ops).unwrap();
        assert!(matches!(config.target, SinkTarget::Weaviate));
        assert_eq!(config.class, Some("MyClass".to_string()));
        assert_eq!(config.api_key, Some("secret".to_string()));
    }

    #[test]
    fn test_parse_config_chromadb() {
        let ops = json!({
            "op": "sink.chromadb",
            "url": "http://localhost:8000",
            "collection": "embeddings",
            "id": "doc-001",
        });
        let config = parse_config(&ops).unwrap();
        assert!(matches!(config.target, SinkTarget::ChromaDb));
        assert_eq!(config.id, Some(json!("doc-001")));
    }

    #[test]
    fn test_parse_config_missing_url() {
        let ops = json!({
            "op": "sink.qdrant",
            "collection": "test",
        });
        assert!(parse_config(&ops).is_err());
    }

    #[test]
    fn test_parse_config_missing_collection() {
        let ops = json!({
            "op": "sink.qdrant",
            "url": "http://localhost:6333",
        });
        assert!(parse_config(&ops).is_err());
    }

    #[test]
    fn test_parse_config_unknown_target() {
        let ops = json!({
            "op": "sink.pinecone",
            "url": "http://localhost",
            "collection": "test",
        });
        assert!(parse_config(&ops).is_err());
    }

    #[test]
    fn test_parse_config_url_trailing_slash() {
        let ops = json!({
            "op": "sink.qdrant",
            "url": "http://localhost:6333/",
            "collection": "test",
        });
        let config = parse_config(&ops).unwrap();
        assert_eq!(config.url, "http://localhost:6333");
    }

    #[test]
    fn test_extract_vector_from_blob() {
        let floats: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut blobs = HashMap::new();
        blobs.insert(
            "_prev".to_string(),
            BlobData {
                bytes,
                meta: BlobMeta {
                    size: 16,
                    content_type: "tensor/f32".to_string(),
                    shape: Some(vec![1, 4]),
                },
            },
        );

        let input = KernelInput {
            json: json!({}),
            blobs,
        };

        let vector = extract_vector(&input).unwrap();
        assert_eq!(vector.len(), 4);
        assert!((vector[0] - 0.1).abs() < 1e-6);
        assert!((vector[3] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_extract_vector_from_json_data() {
        let input = KernelInput::from_json(json!({
            "data": [0.1, 0.2, 0.3],
        }));
        let vector = extract_vector(&input).unwrap();
        assert_eq!(vector.len(), 3);
    }

    #[test]
    fn test_extract_vector_from_outputs() {
        let input = KernelInput::from_json(json!({
            "outputs": {
                "embedding": {
                    "data": [0.5, 0.6, 0.7, 0.8]
                }
            }
        }));
        let vector = extract_vector(&input).unwrap();
        assert_eq!(vector.len(), 4);
    }

    #[test]
    fn test_extract_vector_no_data() {
        let input = KernelInput::from_json(json!({"foo": "bar"}));
        assert!(extract_vector(&input).is_err());
    }

    #[test]
    fn test_extract_vector_blob_misaligned() {
        let mut blobs = HashMap::new();
        blobs.insert(
            "_prev".to_string(),
            BlobData {
                bytes: vec![1, 2, 3], // 3 bytes, not aligned to f32
                meta: BlobMeta {
                    size: 3,
                    content_type: "tensor/f32".to_string(),
                    shape: None,
                },
            },
        );
        let input = KernelInput {
            json: json!({}),
            blobs,
        };
        assert!(extract_vector(&input).is_err());
    }

    #[test]
    fn test_parse_config_with_payload() {
        let ops = json!({
            "op": "sink.qdrant",
            "url": "http://localhost:6333",
            "collection": "test",
            "payload": {"source": "doc.pdf", "page": 5},
        });
        let config = parse_config(&ops).unwrap();
        assert!(config.payload.is_some());
        let payload = config.payload.unwrap();
        assert_eq!(payload["source"], "doc.pdf");
        assert_eq!(payload["page"], 5);
    }

    #[test]
    fn test_uuid_v4_simple() {
        let id1 = uuid_v4_simple();
        let id2 = uuid_v4_simple();
        assert_eq!(id1.len(), 32);
        // Should be different (nanosecond precision).
        // Note: might rarely collide on very fast CPUs.
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_parse_config_postgres() {
        let ops = json!({
            "op": "sink.postgres",
            "url": "https://xxx.supabase.co/rest/v1",
            "collection": "embeddings",
            "api_key": "eyJhbG...",
            "vector_column": "vec",
        });
        let config = parse_config(&ops).unwrap();
        assert!(matches!(config.target, SinkTarget::Postgres));
        assert_eq!(config.url, "https://xxx.supabase.co/rest/v1");
        assert_eq!(config.collection, "embeddings");
        assert_eq!(config.vector_column, Some("vec".to_string()));
        assert_eq!(config.api_key, Some("eyJhbG...".to_string()));
    }

    #[test]
    fn test_parse_config_postgres_defaults() {
        let ops = json!({
            "op": "sink.postgres",
            "url": "http://localhost:3000",
            "collection": "items",
        });
        let config = parse_config(&ops).unwrap();
        assert!(matches!(config.target, SinkTarget::Postgres));
        // vector_column defaults to None (write_postgres uses "embedding").
        assert_eq!(config.vector_column, None);
    }

    #[test]
    fn test_parse_config_postgres_with_payload() {
        let ops = json!({
            "op": "sink.postgres",
            "url": "http://localhost:3000",
            "collection": "documents",
            "payload": {"title": "test doc", "source": "pdf"},
            "id": 42,
        });
        let config = parse_config(&ops).unwrap();
        assert!(matches!(config.target, SinkTarget::Postgres));
        assert_eq!(config.id, Some(json!(42)));
        let payload = config.payload.unwrap();
        assert_eq!(payload["title"], "test doc");
    }

    #[test]
    fn test_sink_kernel_name() {
        let kernel = SinkKernel::new();
        assert_eq!(kernel.name(), "sink");
    }
}
