//! Feature-gated compute kernels for Axon ML runtime.
//!
//! Each kernel implements the `ComputeKernel` trait and handles
//! a specific stage in the ML inference pipeline.

// Pipeline composition (always available — no feature gate)
pub mod compose;

// ML inference
#[cfg(feature = "onnx")]
pub mod onnx;

// Tensor post-processing (mean_pool, normalize, reshape, etc.)
#[cfg(feature = "onnx")]
pub mod tensor;

// Autoregressive generation (encoder-decoder models like Whisper)
#[cfg(feature = "onnx")]
pub mod generate;

// Sandboxed WASM execution for custom pre/post processing
#[cfg(feature = "wasm")]
pub mod wasm;

// Audio decoding + resampling (symphonia + rubato)
#[cfg(feature = "audio")]
pub mod audio;

// Mel spectrogram computation (rustfft)
#[cfg(feature = "audio")]
pub mod mel;

// HuggingFace tokenizer (encode/decode)
#[cfg(feature = "tokenizer")]
pub mod tokenizer;

// Image processing (decode, resize, normalize, colorspace, layout, pad)
#[cfg(feature = "vision")]
pub mod image;

// Object detection post-processing (NMS, confidence filter, box format conversion)
#[cfg(feature = "vision")]
pub mod detection;

// Text processing (regex-based)
#[cfg(feature = "text")]
pub mod text;

// Sherpa speech engine (ASR, VAD, TTS, diarization)
#[cfg(feature = "sherpa")]
pub mod sherpa;

// Vector DB sinks (Zero-ETL: embedding → direct DB write)
#[cfg(feature = "sink")]
pub mod sink;
