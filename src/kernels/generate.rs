//! Autoregressive generation kernel for encoder-decoder models.
//!
//! Runs greedy decoding with an ONNX decoder model, feeding encoder
//! hidden states from the previous pipeline step.
//!
//! Config (operations):
//!   - `model`: path to decoder ONNX model
//!   - `max_tokens`: max tokens to generate (default: 128)
//!   - `start_tokens`: initial decoder input token IDs (default: [])
//!   - `eos_token`: end-of-sequence token ID (default: -1, no stopping)
//!   - `encoder_input`: name of encoder output in blob metadata (default: auto-detect)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{ArrayD, IxDyn};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use parking_lot::Mutex;
use tracing::{debug, info};

use crate::error::AxonError;
use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};

/// Autoregressive generation kernel.
pub struct GenerateKernel {
    /// Cached ONNX sessions keyed by model path.
    sessions: Mutex<HashMap<PathBuf, Arc<Mutex<Session>>>>,
}

impl GenerateKernel {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    fn get_or_load(&self, path: &Path, device: &str) -> Result<Arc<Mutex<Session>>, String> {
        // Cache key includes device to avoid EP conflicts.
        let cache_key = if device == "cpu" {
            path.to_path_buf()
        } else {
            let mut key = path.to_path_buf();
            let new_name = format!(
                "{}.{}",
                key.file_stem().unwrap_or_default().to_string_lossy(),
                device
            );
            key.set_file_name(new_name);
            key
        };

        let mut cache = self.sessions.lock();
        if let Some(session) = cache.get(&cache_key) {
            return Ok(Arc::clone(session));
        }

        info!(model = %path.display(), device, "generate: loading decoder model");
        let builder = Session::builder()
            .map_err(|e| format!("generate: session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("generate: optimization: {e}"))?;

        // Reuse the same EP configuration as OnnxKernel.
        let mut builder = super::onnx::configure_execution_provider(builder, device)
            .map_err(|e| format!("generate: {e}"))?;

        let session = builder
            .commit_from_file(path)
            .map_err(|e| format!("generate: load '{}': {e}", path.display()))?;

        let arc = Arc::new(Mutex::new(session));
        cache.insert(cache_key, Arc::clone(&arc));
        Ok(arc)
    }
}

impl ComputeKernel for GenerateKernel {
    fn name(&self) -> &str {
        "generate"
    }

    fn execute(
        &self,
        input: KernelInput,
        operations: serde_json::Value,
    ) -> Result<KernelOutput, AxonError> {
        let model_path = operations
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or("generate: operations must have 'model' path")?;

        let max_tokens = operations
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;

        let eos_token = operations
            .get("eos_token")
            .and_then(|v| v.as_i64())
            .unwrap_or(-1);

        let start_tokens: Vec<i64> = operations
            .get("start_tokens")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64())
                    .collect()
            })
            .unwrap_or_default();

        let device = operations
            .get("device")
            .and_then(|v| v.as_str())
            .unwrap_or("cpu");

        if start_tokens.is_empty() {
            return Err("generate: 'start_tokens' must be a non-empty array of token IDs".into());
        }

        info!(
            model = model_path,
            max_tokens,
            device,
            start_tokens_len = start_tokens.len(),
            "generate: starting autoregressive decoding"
        );

        let session_mutex = self.get_or_load(Path::new(model_path), device)?;
        let mut session = session_mutex.lock();

        // Get encoder hidden states from input blob.
        let (encoder_data, encoder_shape) = if let Some(blob) = input.first_blob() {
            if blob.meta.content_type != "tensor/f32" {
                return Err(format!(
                    "generate: expected tensor/f32 blob, got '{}'",
                    blob.meta.content_type
                )
                .into());
            }
            let shape = blob
                .meta
                .shape
                .clone()
                .ok_or("generate: encoder blob missing shape metadata")?;
            let data: Vec<f32> = blob
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            (data, shape)
        } else {
            // Try JSON input: look for encoder output.
            let json = &input.json;
            let encoder_output = json
                .get("outputs")
                .and_then(|o| o.get("last_hidden_state"))
                .or_else(|| json.get("last_hidden_state"))
                .ok_or("generate: no encoder hidden states found (expected blob or JSON with 'last_hidden_state')")?;

            let shape: Vec<usize> = encoder_output
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or("generate: encoder output missing 'shape'")?
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as usize))
                .collect();

            let data: Vec<f32> = encoder_output
                .get("data")
                .and_then(|d| d.as_array())
                .ok_or("generate: encoder output missing 'data'")?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            (data, shape)
        };

        // Build encoder_hidden_states tensor ONCE (reused across all decode steps).
        let encoder_array = ArrayD::<f32>::from_shape_vec(IxDyn(&encoder_shape), encoder_data)
            .map_err(|e| format!("generate: encoder tensor: {e}"))?;
        let encoder_tensor = Tensor::<f32>::from_array(encoder_array)
            .map_err(|e| format!("generate: encoder tensor alloc: {e}"))?;

        // Discover decoder input names.
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        // Find which input names correspond to input_ids and encoder_hidden_states.
        let ids_input_name = input_names
            .iter()
            .find(|n| n.contains("input_ids") || n.contains("decoder_input_ids"))
            .cloned()
            .unwrap_or_else(|| "input_ids".to_string());

        let encoder_input_name = input_names
            .iter()
            .find(|n| {
                n.contains("encoder_hidden_states")
                    || n.contains("encoder_output")
                    || n.contains("hidden_states")
            })
            .cloned()
            .unwrap_or_else(|| "encoder_hidden_states".to_string());

        debug!(
            ids_name = %ids_input_name,
            encoder_name = %encoder_input_name,
            all_inputs = ?input_names,
            "generate: decoder input mapping"
        );

        // Greedy decoding loop.
        let mut generated_tokens = start_tokens.clone();

        for step in 0..max_tokens {
            let seq_len = generated_tokens.len();

            // Build input_ids tensor: [1, seq_len].
            let ids_array = ArrayD::<i64>::from_shape_vec(
                IxDyn(&[1, seq_len]),
                generated_tokens.clone(),
            )
            .map_err(|e| format!("generate: input_ids tensor: {e}"))?;
            let ids_tensor = Tensor::<i64>::from_array(ids_array)
                .map_err(|e| format!("generate: input_ids alloc: {e}"))?;

            // Build decoder inputs — reuse cached encoder_tensor (clone is cheap: ORT ref-count).
            let mut decoder_inputs: Vec<(String, ort::session::SessionInputValue<'_>)> =
                Vec::new();

            for name in &input_names {
                if name == &ids_input_name {
                    decoder_inputs.push((name.clone(), ids_tensor.clone().into()));
                } else if name == &encoder_input_name {
                    decoder_inputs.push((name.clone(), encoder_tensor.clone().into()));
                }
                // Skip other inputs (e.g., past_key_values) — not supported yet.
            }

            if decoder_inputs.len() < 2 {
                return Err(format!(
                    "generate: decoder model has {} inputs ({:?}) but couldn't map input_ids and encoder_hidden_states",
                    input_names.len(),
                    input_names
                )
                .into());
            }

            // Run decoder.
            let outputs = session
                .run(decoder_inputs)
                .map_err(|e| format!("generate: decoder step {step}: {e}"))?;

            // Extract logits from first output.
            let logits = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| format!("generate: extract logits: {e}"))?;

            let logits_shape = logits.shape();
            // logits shape: [batch, seq_len, vocab_size]
            if logits_shape.len() != 3 {
                return Err(format!(
                    "generate: expected 3D logits [batch, seq, vocab], got {:?}",
                    logits_shape
                )
                .into());
            }

            let vocab_size = logits_shape[2];

            // Get logits for the last position: logits[0, -1, :]
            let last_pos = logits_shape[1] - 1;
            let last_logits: Vec<f32> = (0..vocab_size)
                .map(|v| logits[[0, last_pos, v]])
                .collect();

            // Greedy: argmax.
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0);

            debug!(step, next_token, "generate: decoded token");

            // Check EOS.
            if next_token == eos_token {
                info!(
                    steps = step + 1,
                    total_tokens = generated_tokens.len(),
                    "generate: reached EOS"
                );
                break;
            }

            generated_tokens.push(next_token);
        }

        // Strip start tokens from output (only return generated tokens).
        let output_tokens: Vec<i64> = generated_tokens[start_tokens.len()..].to_vec();

        info!(
            generated = output_tokens.len(),
            "generate: decoding complete"
        );

        Ok(KernelOutput::Json(serde_json::json!({
            "token_ids": output_tokens,
            "all_tokens": generated_tokens,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_kernel_name() {
        let kernel = GenerateKernel::new();
        assert_eq!(kernel.name(), "generate");
    }

    #[test]
    fn test_generate_missing_model() {
        let kernel = GenerateKernel::new();
        let ops = serde_json::json!({"max_tokens": 10});
        let err = kernel
            .execute(KernelInput::from_json(serde_json::json!({})), ops)
            .unwrap_err();
        assert!(err.contains("model"), "expected model error: {err}");
    }

    #[test]
    fn test_generate_missing_start_tokens() {
        let kernel = GenerateKernel::new();
        let ops = serde_json::json!({"model": "/nonexistent.onnx"});
        let err = kernel
            .execute(KernelInput::from_json(serde_json::json!({})), ops)
            .unwrap_err();
        assert!(
            err.contains("start_tokens"),
            "expected start_tokens error: {err}"
        );
    }
}
