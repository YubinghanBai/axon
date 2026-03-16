//! Tokenizer compute kernel: HuggingFace tokenizer for text ↔ token IDs.
//!
//! Enabled via `cargo build --features tokenizer`.
//!
//! Wraps the `tokenizers` crate for fast, native tokenization.
//! Output format chains directly to compute://onnx input.
//!
//! Tasks:
//!   - `encode` (default): text → token IDs + attention mask
//!   - `decode`: token IDs → text
//!
//! Encode input:
//!   - Single: `{"text": "hello world"}`
//!   - Batch:  `{"texts": ["hello", "world"]}`
//!
//! Encode output:
//!   `{"input_ids": [[101, 7592, ...]], "attention_mask": [[1, 1, ...]]}`
//!
//! Decode input:
//!   - Single: `{"ids": [101, 7592, 2088, 102]}`
//!   - Batch:  `{"ids": [[101, 7592], [101, 2088]]}`
//!
//! Decode output:
//!   - Single: `{"text": "hello world"}`
//!   - Batch:  `{"texts": ["hello", "world"]}`
//!
//! Config (operations):
//!   - String: `"path/to/tokenizer.json"`
//!   - Object: `{"tokenizer": "path/to/tokenizer.json", "max_length": 512,
//!              "task": "encode", "add_special_tokens": true}`

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::error::AxonError;
use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};

/// Default max sequence length (no truncation).
const DEFAULT_MAX_LENGTH: usize = 0;

/// Tokenizer task: encode (text→IDs) or decode (IDs→text).
#[derive(Debug, Clone, Copy, PartialEq)]
enum TokenizerTask {
    Encode,
    Decode,
}

/// Tokenizer compute kernel with tokenizer caching.
///
/// Each unique tokenizer.json path is loaded once and cached.
pub struct TokenizerKernel {
    tokenizers: parking_lot::RwLock<HashMap<PathBuf, Arc<Tokenizer>>>,
}

/// Config parsed from the operations JSON.
#[derive(Debug)]
struct TokenizerConfig {
    tokenizer_path: PathBuf,
    /// Max sequence length. 0 = no truncation.
    max_length: usize,
    /// Task: encode or decode.
    task: TokenizerTask,
    /// Whether to add special tokens ([CLS], [SEP], etc.).
    /// Default true for encode, ignored for decode.
    add_special_tokens: bool,
    /// Skip special tokens when decoding. Default true.
    skip_special_tokens: bool,
    /// Return character-level offsets for each token. Default false.
    /// When true, output includes `"offsets": [[[start, end], ...]]`.
    /// Essential for NER pipelines that need token→character mapping.
    return_offsets: bool,
}

impl TokenizerKernel {
    pub fn new() -> Self {
        Self {
            tokenizers: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Get a cached tokenizer or load from file.
    fn get_or_load(&self, path: &Path) -> Result<Arc<Tokenizer>, String> {
        // Fast path: read lock.
        if let Some(t) = self.tokenizers.read().get(path) {
            return Ok(Arc::clone(t));
        }
        // Slow path: write lock + double check.
        let mut cache = self.tokenizers.write();
        if let Some(t) = cache.get(path) {
            return Ok(Arc::clone(t));
        }
        info!(path = %path.display(), "tokenizer: loading");
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| format!("tokenizer: load '{}': {e}", path.display()))?;
        let tokenizer = Arc::new(tokenizer);
        cache.insert(path.to_path_buf(), Arc::clone(&tokenizer));
        Ok(tokenizer)
    }

    /// Encode text(s) to token IDs.
    fn encode(
        &self,
        input: &serde_json::Value,
        config: &TokenizerConfig,
        tokenizer: &Tokenizer,
    ) -> Result<KernelOutput, String> {
        let texts = extract_texts(input)?;
        let batch_size = texts.len();

        debug!(
            batch_size,
            max_length = config.max_length,
            add_special_tokens = config.add_special_tokens,
            "tokenizer: encoding"
        );

        // Encode all texts.
        let encodings = if batch_size == 1 {
            vec![tokenizer
                .encode(texts[0].as_str(), config.add_special_tokens)
                .map_err(|e| format!("tokenizer: encode: {e}"))?]
        } else {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            tokenizer
                .encode_batch(refs, config.add_special_tokens)
                .map_err(|e| format!("tokenizer: encode_batch: {e}"))?
        };

        // Determine max length for padding.
        let raw_max = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        let max_len = if config.max_length > 0 {
            raw_max.min(config.max_length)
        } else {
            raw_max
        };

        // Build output: truncate + pad to uniform length.
        let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut all_masks: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut all_type_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);

        let mut all_offsets: Vec<Vec<[usize; 2]>> = if config.return_offsets {
            Vec::with_capacity(batch_size)
        } else {
            Vec::new()
        };

        for enc in &encodings {
            let mut ids: Vec<u32> = enc.get_ids().to_vec();
            let mut mask: Vec<u32> = enc.get_attention_mask().to_vec();

            // Truncate.
            ids.truncate(max_len);
            mask.truncate(max_len);

            // Pad (single allocation, no push loop).
            ids.resize(max_len, 0);
            mask.resize(max_len, 0);

            // token_type_ids (required by BERT-family models).
            let mut type_ids: Vec<u32> = enc.get_type_ids().to_vec();
            type_ids.truncate(max_len);
            type_ids.resize(max_len, 0);

            if config.return_offsets {
                let mut offsets: Vec<[usize; 2]> = enc
                    .get_offsets()
                    .iter()
                    .map(|&(s, e)| [s, e])
                    .collect();
                offsets.truncate(max_len);
                offsets.resize(max_len, [0, 0]);
                all_offsets.push(offsets);
            }

            all_ids.push(ids);
            all_masks.push(mask);
            all_type_ids.push(type_ids);
        }

        info!(batch_size, seq_len = max_len, "tokenizer: encode done");

        let mut result = serde_json::json!({
            "input_ids": all_ids,
            "attention_mask": all_masks,
            "token_type_ids": all_type_ids,
        });

        if config.return_offsets {
            result["offsets"] = serde_json::json!(all_offsets);
        }

        Ok(KernelOutput::Json(result))
    }

    /// Decode token IDs to text.
    fn decode(
        &self,
        input: &serde_json::Value,
        config: &TokenizerConfig,
        tokenizer: &Tokenizer,
    ) -> Result<KernelOutput, String> {
        let id_sequences = extract_ids(input)?;
        let batch_size = id_sequences.len();

        debug!(batch_size, skip_special = config.skip_special_tokens, "tokenizer: decoding");

        if batch_size == 1 {
            let text = tokenizer
                .decode(&id_sequences[0], config.skip_special_tokens)
                .map_err(|e| format!("tokenizer: decode: {e}"))?;

            info!("tokenizer: decode done (single)");
            Ok(KernelOutput::Json(serde_json::json!({ "text": text })))
        } else {
            let refs: Vec<&[u32]> = id_sequences.iter().map(|v| v.as_slice()).collect();
            let texts = tokenizer
                .decode_batch(&refs, config.skip_special_tokens)
                .map_err(|e| format!("tokenizer: decode_batch: {e}"))?;

            info!(batch_size, "tokenizer: decode done (batch)");
            Ok(KernelOutput::Json(serde_json::json!({ "texts": texts })))
        }
    }
}

impl ComputeKernel for TokenizerKernel {
    fn name(&self) -> &str {
        "tokenizer"
    }

    fn execute(
        &self,
        input: KernelInput,
        operations: serde_json::Value,
    ) -> Result<KernelOutput, AxonError> {
        let input = input.into_json();
        let config = parse_tokenizer_config(&operations)?;
        let tokenizer = self.get_or_load(&config.tokenizer_path)?;

        match config.task {
            TokenizerTask::Encode => self.encode(&input, &config, &tokenizer),
            TokenizerTask::Decode => self.decode(&input, &config, &tokenizer),
        }.map_err(Into::into)
    }
}

// ── Input extraction ───────────────────────────────────────────

/// Extract text(s) from the input JSON (for encode).
///
/// Supports:
///   - `{"text": "hello"}` → single text
///   - `{"texts": ["hello", "world"]}` → batch
///   - Plain string → single text
fn extract_texts(input: &serde_json::Value) -> Result<Vec<String>, String> {
    // {"text": "..."}
    if let Some(text) = input.get("text").and_then(|v| v.as_str()) {
        return Ok(vec![text.to_string()]);
    }
    // {"texts": ["...", "..."]}
    if let Some(texts) = input.get("texts").and_then(|v| v.as_array()) {
        let result: Result<Vec<String>, String> = texts
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| "tokenizer: texts array must contain strings".to_string())
            })
            .collect();
        return result;
    }
    // Plain string value.
    if let Some(text) = input.as_str() {
        return Ok(vec![text.to_string()]);
    }
    // Try one level deep (namespaced signal).
    if let serde_json::Value::Object(map) = input {
        for (_, inner) in map {
            if let Some(text) = inner.get("text").and_then(|v| v.as_str()) {
                return Ok(vec![text.to_string()]);
            }
            if let Some(texts) = inner.get("texts").and_then(|v| v.as_array()) {
                let result: Result<Vec<String>, String> = texts
                    .iter()
                    .map(|v| {
                        v.as_str()
                            .map(|s| s.to_string())
                            .ok_or_else(|| {
                                "tokenizer: texts array must contain strings".to_string()
                            })
                    })
                    .collect();
                return result;
            }
        }
    }
    Err("tokenizer: input must have 'text' (string) or 'texts' (string array)".into())
}

/// Extract token ID sequence(s) from the input JSON (for decode).
///
/// Supports:
///   - `{"ids": [101, 7592, 2088, 102]}` → single sequence
///   - `{"ids": [[101, 7592], [101, 2088]]}` → batch
///   - Flat array at top level
fn extract_ids(input: &serde_json::Value) -> Result<Vec<Vec<u32>>, String> {
    let ids_value = input.get("ids")
        .or_else(|| input.get("input_ids"))
        .or_else(|| input.get("token_ids"))
        .unwrap_or(input);

    match ids_value {
        serde_json::Value::Array(arr) if arr.is_empty() => {
            Err("tokenizer: decode input is empty".into())
        }
        serde_json::Value::Array(arr) => {
            // Check first element: if it's a number, it's a single sequence.
            // If it's an array, it's a batch.
            if arr[0].is_number() {
                let ids = parse_id_array(arr)?;
                Ok(vec![ids])
            } else if arr[0].is_array() {
                arr.iter()
                    .map(|v| {
                        v.as_array()
                            .ok_or_else(|| "tokenizer: batch ids must be arrays".to_string())
                            .and_then(|a| parse_id_array(a))
                    })
                    .collect()
            } else {
                Err("tokenizer: ids must be numbers or arrays of numbers".into())
            }
        }
        _ => {
            // Try one level deep (namespaced signal).
            if let serde_json::Value::Object(map) = input {
                for (_, inner) in map {
                    if let Ok(ids) = extract_ids(inner) {
                        return Ok(ids);
                    }
                }
            }
            Err("tokenizer: decode input must have 'ids' (array of ints)".into())
        }
    }
}

/// Parse a JSON array of numbers into Vec<u32>.
fn parse_id_array(arr: &[serde_json::Value]) -> Result<Vec<u32>, String> {
    arr.iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as u32)
                .ok_or_else(|| "tokenizer: ids must be positive integers".to_string())
        })
        .collect()
}

// ── Config parsing ─────────────────────────────────────────────

fn parse_tokenizer_config(operations: &serde_json::Value) -> Result<TokenizerConfig, String> {
    match operations {
        serde_json::Value::String(s) => Ok(TokenizerConfig {
            tokenizer_path: PathBuf::from(s),
            max_length: DEFAULT_MAX_LENGTH,
            task: TokenizerTask::Encode,
            add_special_tokens: true,
            skip_special_tokens: true,
            return_offsets: false,
        }),
        serde_json::Value::Object(obj) => {
            let path = obj
                .get("tokenizer")
                .and_then(|v| v.as_str())
                .ok_or("tokenizer: config must have 'tokenizer' path")?;
            let max_length = obj
                .get("max_length")
                .and_then(|v| v.as_u64())
                .unwrap_or(DEFAULT_MAX_LENGTH as u64) as usize;

            let task = match obj.get("task").and_then(|v| v.as_str()) {
                Some("decode") => TokenizerTask::Decode,
                Some("encode") | None => TokenizerTask::Encode,
                Some(other) => return Err(format!("tokenizer: unknown task '{other}' (use 'encode' or 'decode')")),
            };

            let add_special_tokens = obj
                .get("add_special_tokens")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            let skip_special_tokens = obj
                .get("skip_special_tokens")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            let return_offsets = obj
                .get("return_offsets")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            Ok(TokenizerConfig {
                tokenizer_path: PathBuf::from(path),
                max_length,
                task,
                add_special_tokens,
                skip_special_tokens,
                return_offsets,
            })
        }
        _ => Err("tokenizer: operations must be path string or config object".into()),
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config parsing ──

    #[test]
    fn test_parse_config_string() {
        let ops = serde_json::json!("models/tokenizer.json");
        let config = parse_tokenizer_config(&ops).unwrap();
        assert_eq!(
            config.tokenizer_path,
            PathBuf::from("models/tokenizer.json")
        );
        assert_eq!(config.max_length, 0);
        assert_eq!(config.task, TokenizerTask::Encode);
        assert!(config.add_special_tokens);
    }

    #[test]
    fn test_parse_config_object() {
        let ops = serde_json::json!({
            "tokenizer": "models/tokenizer.json",
            "max_length": 512
        });
        let config = parse_tokenizer_config(&ops).unwrap();
        assert_eq!(
            config.tokenizer_path,
            PathBuf::from("models/tokenizer.json")
        );
        assert_eq!(config.max_length, 512);
    }

    #[test]
    fn test_parse_config_object_defaults() {
        let ops = serde_json::json!({"tokenizer": "tok.json"});
        let config = parse_tokenizer_config(&ops).unwrap();
        assert_eq!(config.max_length, 0);
        assert_eq!(config.task, TokenizerTask::Encode);
        assert!(config.add_special_tokens);
        assert!(config.skip_special_tokens);
    }

    #[test]
    fn test_parse_config_decode_task() {
        let ops = serde_json::json!({
            "tokenizer": "tok.json",
            "task": "decode",
            "skip_special_tokens": false
        });
        let config = parse_tokenizer_config(&ops).unwrap();
        assert_eq!(config.task, TokenizerTask::Decode);
        assert!(!config.skip_special_tokens);
    }

    #[test]
    fn test_parse_config_no_special_tokens() {
        let ops = serde_json::json!({
            "tokenizer": "tok.json",
            "add_special_tokens": false
        });
        let config = parse_tokenizer_config(&ops).unwrap();
        assert!(!config.add_special_tokens);
    }

    #[test]
    fn test_parse_config_unknown_task() {
        let ops = serde_json::json!({"tokenizer": "tok.json", "task": "foo"});
        let err = parse_tokenizer_config(&ops).unwrap_err();
        assert!(err.contains("unknown task"));
    }

    #[test]
    fn test_parse_config_missing_path() {
        let ops = serde_json::json!({"max_length": 512});
        assert!(parse_tokenizer_config(&ops).is_err());
    }

    #[test]
    fn test_parse_config_invalid() {
        let ops = serde_json::json!(42);
        assert!(parse_tokenizer_config(&ops).is_err());
    }

    // ── Text extraction ──

    #[test]
    fn test_extract_single_text() {
        let input = serde_json::json!({"text": "hello world"});
        let texts = extract_texts(&input).unwrap();
        assert_eq!(texts, vec!["hello world"]);
    }

    #[test]
    fn test_extract_batch_texts() {
        let input = serde_json::json!({"texts": ["hello", "world"]});
        let texts = extract_texts(&input).unwrap();
        assert_eq!(texts, vec!["hello", "world"]);
    }

    #[test]
    fn test_extract_plain_string() {
        let input = serde_json::json!("hello world");
        let texts = extract_texts(&input).unwrap();
        assert_eq!(texts, vec!["hello world"]);
    }

    #[test]
    fn test_extract_nested_text() {
        let input = serde_json::json!({"query": {"text": "hello"}});
        let texts = extract_texts(&input).unwrap();
        assert_eq!(texts, vec!["hello"]);
    }

    #[test]
    fn test_extract_no_text() {
        let input = serde_json::json!({"x": 42});
        assert!(extract_texts(&input).is_err());
    }

    // ── ID extraction (decode) ──

    #[test]
    fn test_extract_ids_single() {
        let input = serde_json::json!({"ids": [101, 7592, 2088, 102]});
        let ids = extract_ids(&input).unwrap();
        assert_eq!(ids, vec![vec![101, 7592, 2088, 102]]);
    }

    #[test]
    fn test_extract_ids_batch() {
        let input = serde_json::json!({"ids": [[101, 7592], [101, 2088]]});
        let ids = extract_ids(&input).unwrap();
        assert_eq!(ids, vec![vec![101, 7592], vec![101, 2088]]);
    }

    #[test]
    fn test_extract_ids_input_ids_key() {
        let input = serde_json::json!({"input_ids": [[101, 7592, 102]]});
        let ids = extract_ids(&input).unwrap();
        assert_eq!(ids, vec![vec![101, 7592, 102]]);
    }

    #[test]
    fn test_extract_ids_flat_array() {
        let input = serde_json::json!([101, 7592, 2088, 102]);
        let ids = extract_ids(&input).unwrap();
        assert_eq!(ids, vec![vec![101, 7592, 2088, 102]]);
    }

    #[test]
    fn test_extract_ids_nested() {
        let input = serde_json::json!({"tokens": {"ids": [101, 102]}});
        let ids = extract_ids(&input).unwrap();
        assert_eq!(ids, vec![vec![101, 102]]);
    }

    #[test]
    fn test_extract_ids_empty() {
        let input = serde_json::json!({"ids": []});
        assert!(extract_ids(&input).is_err());
    }
}
