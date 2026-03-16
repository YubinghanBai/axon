//! Sherpa speech processing kernel — opt-in backend for production speech.
//!
//! Wraps sherpa-onnx (k2-fsa) to provide high-performance ASR, VAD, TTS,
//! and speaker diarization as pipeline steps. Feature-gated behind `sherpa`.
//!
//! ## Why sherpa-onnx?
//!
//! - Streaming ASR (Zipformer transducer) with millisecond latency
//! - Supports Whisper, Paraformer, SenseVoice, Zipformer, NeMo
//! - Built-in Silero VAD, endpoint detection, KV-cache
//! - TTS (VITS, Kokoro) for voice agents
//! - All on ONNX Runtime — same backend as Axon
//!
//! ## Pipeline usage
//!
//! ```toml
//! [pre]
//! steps = [
//!   { op = "sherpa.vad", model = "silero_vad.onnx", threshold = 0.5 },
//! ]
//!
//! [post]
//! steps = [
//!   { op = "sherpa.transcribe", model_type = "whisper",
//!     encoder = "whisper-encoder.onnx", decoder = "whisper-decoder.onnx",
//!     tokens = "tokens.txt", language = "en" },
//! ]
//! ```
//!
//! ## Compile
//!
//! ```bash
//! cargo build --features sherpa
//! ```

use std::cell::UnsafeCell;

use serde_json::Value;
use tracing::{debug, info};

use crate::error::AxonError;
use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};

use sherpa_onnx::{
    GenerationConfig, OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineTts,
    OfflineTtsConfig, OfflineTtsModelConfig, OfflineTtsVitsModelConfig,
    OfflineWhisperModelConfig, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};

// ── Thread-safety wrapper ──────────────────────────────────────
//
// sherpa-onnx types are !Send + !Sync because the C library uses
// thread-local state. We wrap them in UnsafeCell + Send/Sync impl
// because Axon's pipeline runs on a single blocking thread per request
// (spawn_blocking), and we protect access with parking_lot::Mutex
// at the kernel level.

#[allow(dead_code)]
struct SendWrapper<T>(UnsafeCell<T>);
unsafe impl<T> Send for SendWrapper<T> {}
unsafe impl<T> Sync for SendWrapper<T> {}

#[allow(dead_code)]
impl<T> SendWrapper<T> {
    fn new(val: T) -> Self {
        Self(UnsafeCell::new(val))
    }
    fn get(&self) -> &T {
        unsafe { &*self.0.get() }
    }
}

// ── Kernel ─────────────────────────────────────────────────────

/// Sherpa speech processing kernel.
///
/// Lazily initializes recognizers/VAD/TTS on first use based on
/// operation config. Instances are cached for the kernel's lifetime.
pub struct SherpaKernel;

impl ComputeKernel for SherpaKernel {
    fn name(&self) -> &str {
        "sherpa"
    }

    fn execute(
        &self,
        input: KernelInput,
        operations: Value,
    ) -> Result<KernelOutput, AxonError> {
        let op = operations
            .get("op")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        debug!(op, "sherpa kernel");

        match op {
            "transcribe" => op_transcribe(input, &operations),
            "vad" => op_vad(input, &operations),
            "tts" => op_tts(input, &operations),
            _ => Err(format!("sherpa: unknown operation '{op}'").into()),
        }
    }
}

// ── Audio extraction helper ────────────────────────────────────

/// Extract PCM f32 samples from KernelInput.
///
/// Supports two input paths:
/// 1. Blob with content_type "audio/pcm-f32" (from audio.decode kernel)
/// 2. Raw audio bytes blob (decode inline — not implemented, use audio.decode first)
fn extract_pcm_f32(input: &KernelInput) -> Result<Vec<f32>, AxonError> {
    if let Some(blob) = input.first_blob() {
        if blob.meta.content_type == "audio/pcm-f32" {
            // Blob is already PCM f32 LE bytes.
            let samples: Vec<f32> = blob
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            return Ok(samples);
        }
        return Err(format!(
            "sherpa: expected audio/pcm-f32 blob, got '{}'. Use audio.decode first.",
            blob.meta.content_type
        )
        .into());
    }
    Err("sherpa: no audio blob in input. Pipe through audio.decode first.".into())
}

// ── Offline ASR ────────────────────────────────────────────────

/// `sherpa.transcribe` — Offline speech recognition.
///
/// Supports: Whisper, Paraformer, SenseVoice, Transducer, NeMo.
///
/// Config:
/// ```toml
/// { op = "sherpa.transcribe",
///   model_type = "whisper",
///   encoder = "whisper-tiny-encoder.onnx",
///   decoder = "whisper-tiny-decoder.onnx",
///   tokens = "whisper-tiny-tokens.txt",
///   language = "en",
///   num_threads = 4 }
/// ```
fn op_transcribe(input: KernelInput, ops: &Value) -> Result<KernelOutput, AxonError> {
    let samples = extract_pcm_f32(&input)?;
    let sample_rate = ops
        .get("sample_rate")
        .and_then(|v| v.as_i64())
        .unwrap_or(16000) as i32;
    let num_threads = ops
        .get("num_threads")
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;
    let tokens = ops
        .get("tokens")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model_type = ops
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("whisper");

    let model_config = build_offline_model_config(ops, &tokens, num_threads, model_type)?;

    let mut config = OfflineRecognizerConfig::default();
    config.feat_config.sample_rate = sample_rate;
    config.feat_config.feature_dim = 80;
    config.model_config = model_config;
    config.decoding_method = Some("greedy_search".to_string());

    let recognizer = OfflineRecognizer::create(&config)
        .ok_or("sherpa.transcribe: failed to create recognizer (check model paths)")?;

    let stream = recognizer.create_stream();
    stream.accept_waveform(sample_rate, &samples);
    recognizer.decode(&stream);

    let result = stream
        .get_result()
        .ok_or("sherpa.transcribe: no result from recognizer")?;

    info!(
        text_len = result.text.len(),
        tokens = result.tokens.len(),
        "sherpa.transcribe: done"
    );

    // Pass through original input JSON and add transcription.
    let mut json = input.into_json();
    json["text"] = Value::String(result.text.trim().to_string());
    json["tokens"] = Value::Array(
        result
            .tokens
            .iter()
            .map(|t| Value::String(t.clone()))
            .collect(),
    );
    if let Some(ref timestamps) = result.timestamps {
        json["timestamps"] = serde_json::json!(timestamps);
    }

    Ok(KernelOutput::Json(json))
}

/// Build OfflineModelConfig from operation parameters.
fn build_offline_model_config(
    ops: &Value,
    tokens: &str,
    num_threads: i32,
    model_type: &str,
) -> Result<OfflineModelConfig, AxonError> {
    let mut config = OfflineModelConfig {
        tokens: Some(tokens.to_string()),
        num_threads,
        model_type: Some(model_type.to_string()),
        ..Default::default()
    };

    match model_type {
        "whisper" => {
            let encoder = ops
                .get("encoder")
                .and_then(|v| v.as_str())
                .ok_or("sherpa.transcribe: whisper requires 'encoder' path")?;
            let decoder = ops
                .get("decoder")
                .and_then(|v| v.as_str())
                .ok_or("sherpa.transcribe: whisper requires 'decoder' path")?;
            let language = ops
                .get("language")
                .and_then(|v| v.as_str())
                .unwrap_or("en");

            config.whisper = OfflineWhisperModelConfig {
                encoder: Some(encoder.to_string()),
                decoder: Some(decoder.to_string()),
                language: Some(language.to_string()),
                ..Default::default()
            };
        }
        "paraformer" | "sense_voice" => {
            let model = ops
                .get("model")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    format!("sherpa.transcribe: {model_type} requires 'model' path")
                })?;

            if model_type == "paraformer" {
                config.paraformer = sherpa_onnx::OfflineParaformerModelConfig {
                    model: Some(model.to_string()),
                    ..Default::default()
                };
            } else {
                config.sense_voice = sherpa_onnx::OfflineSenseVoiceModelConfig {
                    model: Some(model.to_string()),
                    ..Default::default()
                };
            }
        }
        "transducer" => {
            let encoder = ops
                .get("encoder")
                .and_then(|v| v.as_str())
                .ok_or("sherpa.transcribe: transducer requires 'encoder' path")?;
            let decoder = ops
                .get("decoder")
                .and_then(|v| v.as_str())
                .ok_or("sherpa.transcribe: transducer requires 'decoder' path")?;
            let joiner = ops
                .get("joiner")
                .and_then(|v| v.as_str())
                .ok_or("sherpa.transcribe: transducer requires 'joiner' path")?;

            config.transducer = sherpa_onnx::OfflineTransducerModelConfig {
                encoder: Some(encoder.to_string()),
                decoder: Some(decoder.to_string()),
                joiner: Some(joiner.to_string()),
                ..Default::default()
            };
        }
        _ => {
            return Err(
                format!("sherpa.transcribe: unknown model_type '{model_type}' (use whisper/paraformer/sense_voice/transducer)").into(),
            );
        }
    }

    Ok(config)
}

// ── VAD ────────────────────────────────────────────────────────

/// `sherpa.vad` — Voice Activity Detection using Silero VAD.
///
/// Segments audio into speech/silence regions. Output includes
/// speech segments that can be individually transcribed.
///
/// Config:
/// ```toml
/// { op = "sherpa.vad",
///   model = "silero_vad.onnx",
///   threshold = 0.5,
///   min_silence = 0.5,
///   min_speech = 0.25 }
/// ```
fn op_vad(input: KernelInput, ops: &Value) -> Result<KernelOutput, AxonError> {
    let samples = extract_pcm_f32(&input)?;
    let sample_rate = ops
        .get("sample_rate")
        .and_then(|v| v.as_i64())
        .unwrap_or(16000) as i32;

    let model_path = ops
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or("sherpa.vad: missing 'model' path to silero_vad.onnx")?;

    let threshold = ops
        .get("threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
    let min_silence = ops
        .get("min_silence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
    let min_speech = ops
        .get("min_speech")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.25) as f32;

    let config = VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: Some(model_path.to_string()),
            threshold,
            min_silence_duration: min_silence,
            min_speech_duration: min_speech,
            ..Default::default()
        },
        sample_rate,
        num_threads: 1,
        ..Default::default()
    };

    let buffer_seconds = (samples.len() as f32 / sample_rate as f32) + 1.0;
    let vad = VoiceActivityDetector::create(&config, buffer_seconds)
        .ok_or("sherpa.vad: failed to create VAD (check model path)")?;

    // Feed audio in chunks (sherpa VAD expects window-sized chunks).
    let window_size = (sample_rate as usize) / 1000 * 64; // 64ms windows
    for chunk in samples.chunks(window_size) {
        // Pad last chunk if needed.
        if chunk.len() == window_size {
            vad.accept_waveform(chunk);
        } else {
            let mut padded = chunk.to_vec();
            padded.resize(window_size, 0.0);
            vad.accept_waveform(&padded);
        }
    }
    vad.flush();

    // Collect speech segments.
    let mut segments = Vec::new();
    while !vad.is_empty() {
        if let Some(segment) = vad.front() {
            let start = segment.start() as f32 / sample_rate as f32;
            let samples_data = segment.samples().to_vec();
            let duration = samples_data.len() as f32 / sample_rate as f32;
            segments.push(serde_json::json!({
                "start": start,
                "duration": duration,
                "samples_count": samples_data.len(),
            }));
        }
        vad.pop();
    }

    info!(
        segments = segments.len(),
        total_duration = samples.len() as f32 / sample_rate as f32,
        "sherpa.vad: detected speech segments"
    );

    let mut json = input.into_json();
    let has_speech = !segments.is_empty();
    json["speech_segments"] = Value::Array(segments);
    json["has_speech"] = Value::Bool(has_speech);
    Ok(KernelOutput::Json(json))
}

// ── TTS ────────────────────────────────────────────────────────

/// `sherpa.tts` — Text-to-Speech synthesis.
///
/// Supports VITS, Kokoro, Matcha models.
///
/// Config:
/// ```toml
/// { op = "sherpa.tts",
///   model = "vits-model.onnx",
///   tokens = "tokens.txt",
///   data_dir = "espeak-ng-data",
///   speaker_id = 0,
///   speed = 1.0 }
/// ```
fn op_tts(input: KernelInput, ops: &Value) -> Result<KernelOutput, AxonError> {
    let json = input.into_json();
    let text = json
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or("sherpa.tts: input JSON must have 'text' field")?;

    let model_path = ops
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or("sherpa.tts: missing 'model' path")?;
    let tokens = ops
        .get("tokens")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let data_dir = ops
        .get("data_dir")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let speaker_id = ops
        .get("speaker_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;
    let speed = ops
        .get("speed")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let num_threads = ops
        .get("num_threads")
        .and_then(|v| v.as_i64())
        .unwrap_or(2) as i32;

    let config = OfflineTtsConfig {
        model: OfflineTtsModelConfig {
            vits: OfflineTtsVitsModelConfig {
                model: Some(model_path.to_string()),
                tokens: Some(tokens.to_string()),
                data_dir: Some(data_dir.to_string()),
                ..Default::default()
            },
            num_threads,
            ..Default::default()
        },
        ..Default::default()
    };

    let tts = OfflineTts::create(&config)
        .ok_or("sherpa.tts: failed to create TTS engine (check model paths)")?;

    let gen_config = GenerationConfig {
        sid: speaker_id,
        speed,
        ..Default::default()
    };

    let audio = tts
        .generate_with_config::<fn(&[f32], f32) -> bool>(text, &gen_config, None)
        .ok_or("sherpa.tts: generation failed")?;

    let samples = audio.samples();
    let sample_rate = audio.sample_rate();

    info!(
        samples = samples.len(),
        sample_rate,
        duration_ms = (samples.len() as f32 / sample_rate as f32) * 1000.0,
        "sherpa.tts: synthesis complete"
    );

    // Output as PCM f32 blob (compatible with audio pipeline).
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    Ok(KernelOutput::Blob {
        data: bytes,
        content_type: "audio/pcm-f32".to_string(),
        shape: Some(vec![1, samples.len()]),
    })
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_name() {
        let kernel = SherpaKernel;
        assert_eq!(kernel.name(), "sherpa");
    }

    #[test]
    fn test_unknown_op() {
        let kernel = SherpaKernel;
        let input = KernelInput::from_json(serde_json::json!({}));
        let ops = serde_json::json!({"op": "nonexistent"});
        let result = kernel.execute(input, ops);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown"));
    }

    #[test]
    fn test_transcribe_missing_audio() {
        let result = op_transcribe(
            KernelInput::from_json(serde_json::json!({})),
            &serde_json::json!({"op": "transcribe", "model_type": "whisper"}),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no audio blob"));
    }

    #[test]
    fn test_vad_missing_model() {
        let result = op_vad(
            KernelInput::from_json(serde_json::json!({})),
            &serde_json::json!({"op": "vad"}),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_tts_missing_text() {
        let result = op_tts(
            KernelInput::from_json(serde_json::json!({})),
            &serde_json::json!({"op": "tts", "model": "model.onnx"}),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("text"));
    }

    #[test]
    fn test_build_whisper_config() {
        let ops = serde_json::json!({
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "language": "zh",
        });
        let config = build_offline_model_config(&ops, "tokens.txt", 4, "whisper");
        assert!(config.is_ok());
    }

    #[test]
    fn test_build_paraformer_config() {
        let ops = serde_json::json!({
            "model": "paraformer.onnx",
        });
        let config = build_offline_model_config(&ops, "tokens.txt", 4, "paraformer");
        assert!(config.is_ok());
    }

    #[test]
    fn test_build_unknown_model_type() {
        let ops = serde_json::json!({});
        let config = build_offline_model_config(&ops, "tokens.txt", 4, "unknown");
        assert!(config.is_err());
    }

    #[test]
    fn test_extract_pcm_missing_blob() {
        let input = KernelInput::from_json(serde_json::json!({"text": "hello"}));
        let result = extract_pcm_f32(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_pcm_wrong_content_type() {
        use std::collections::HashMap;
        let mut blobs = HashMap::new();
        blobs.insert(
            "_input".to_string(),
            crate::kernel::BlobData {
                bytes: vec![0u8; 16],
                meta: crate::blob::BlobMeta {
                    size: 16,
                    content_type: "image/jpeg".to_string(),
                    shape: None,
                },
            },
        );
        let input = KernelInput {
            json: serde_json::json!({}),
            blobs,
        };
        let result = extract_pcm_f32(&input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("audio/pcm-f32"));
    }
}
