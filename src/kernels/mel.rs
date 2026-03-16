//! Mel spectrogram kernel: PCM f32 → log-mel spectrogram tensor.
//!
//! URI: `compute://mel`
//!
//! Input: upstream blob of raw f32 LE PCM samples (from compute://audio)
//!
//! Config:
//!   {"n_fft": 400, "hop_length": 160, "n_mels": 80}   — Whisper defaults
//!   {"n_fft": 512, "hop_length": 128, "n_mels": 80}   — Paraformer
//!   {"n_fft": 400, "hop_length": 160, "n_mels": 128}  — Whisper large-v3
//!   {"chunk_length": 30}                               — seconds (default: 30)
//!   {"filters": "path/to/mel_filters.bin"}             — custom mel filter weights
//!
//! Output: KernelOutput::Blob
//!   raw f32 LE bytes, shape [1, n_mels, n_frames]
//!   content_type: "tensor/f32"
//!
//! Algorithm (adapted from HuggingFace candle / whisper.cpp):
//!   1. Pad/truncate PCM to chunk_length seconds
//!   2. STFT: Hann window → rustfft → power spectrum
//!   3. Apply mel filter bank (Slaney scale, computed or loaded)
//!   4. Log10 + normalize
//!
//! Libraries:
//!   - rustfft: pure Rust FFT with SIMD (AVX2/NEON on native)

use crate::error::AxonError;
use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};
use rustfft::FftPlanner;
use num_complex::Complex32;
use tracing::info;

pub struct MelKernel;

impl MelKernel {
    pub fn new() -> Self {
        Self
    }
}

impl ComputeKernel for MelKernel {
    fn name(&self) -> &str {
        "mel"
    }

    fn execute(
        &self,
        input: KernelInput,
        operations: serde_json::Value,
    ) -> Result<KernelOutput, AxonError> {
        // Parse config with defaults matching Whisper.
        let n_fft = operations.get("n_fft").and_then(|v| v.as_u64()).unwrap_or(400) as usize;
        let hop_length = operations.get("hop_length").and_then(|v| v.as_u64()).unwrap_or(160) as usize;
        let n_mels = operations.get("n_mels").and_then(|v| v.as_u64()).unwrap_or(80) as usize;
        let sample_rate = operations.get("sample_rate").and_then(|v| v.as_u64()).unwrap_or(16000) as usize;
        let chunk_length = operations.get("chunk_length").and_then(|v| v.as_u64()).unwrap_or(30) as usize;

        // Load PCM f32 samples from blob input.
        let blob = input.first_blob().ok_or("mel: requires blob input (raw f32 PCM)")?;
        if blob.bytes.len() % 4 != 0 {
            return Err(format!("mel: blob size {} not aligned to f32", blob.bytes.len()).into());
        }
        let samples: Vec<f32> = blob
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        info!(
            n_fft,
            hop_length,
            n_mels,
            sample_rate,
            chunk_length,
            input_samples = samples.len(),
            "mel: computing spectrogram"
        );

        let chunk_samples = sample_rate * chunk_length;
        let n_frames = chunk_samples / hop_length;

        // Pad/truncate to chunk_length seconds.
        let mut audio = vec![0.0f32; chunk_samples];
        let copy_len = samples.len().min(chunk_samples);
        audio[..copy_len].copy_from_slice(&samples[..copy_len]);

        // Build mel filter bank (Slaney scale).
        let filters_path = operations.get("filters").and_then(|v| v.as_str());
        let filters = if let Some(path) = filters_path {
            load_mel_filters(path, n_mels, n_fft)?
        } else {
            compute_mel_filters(sample_rate, n_fft, n_mels)
        };

        // Compute log-mel spectrogram.
        let mel = log_mel_spectrogram(&audio, &filters, n_fft, hop_length, n_mels, n_frames);

        info!(
            output_shape = ?[1, n_mels, n_frames],
            output_bytes = mel.len() * 4,
            "mel: done"
        );

        // Output as [1, n_mels, n_frames] f32 blob.
        let bytes: Vec<u8> = mel.iter().flat_map(|f| f.to_le_bytes()).collect();
        Ok(KernelOutput::Blob {
            data: bytes,
            content_type: "tensor/f32".to_string(),
            shape: Some(vec![1, n_mels, n_frames]),
        })
    }
}

// ── Log-mel spectrogram ──────────────────────────────────────
//
// Adapted from candle-transformers/src/models/whisper/audio.rs
// Uses rustfft for native SIMD-accelerated FFT.

fn log_mel_spectrogram(
    samples: &[f32],
    filters: &[f32],
    fft_size: usize,
    fft_step: usize,
    n_mel: usize,
    target_frames: usize,
) -> Vec<f32> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let fft_size_f = fft_size as f32;

    // Hann window.
    let hann: Vec<f32> = (0..fft_size)
        .map(|i| 0.5 * (1.0 - (two_pi * i as f32 / fft_size_f).cos()))
        .collect();

    // Compute n_len with candle-style padding.
    let n_len = samples.len() / fft_step;
    let pad = 100 * 30 / 2; // matching candle: 100 * CHUNK_LENGTH / 2
    let n_len = if n_len % pad != 0 {
        (n_len / pad + 1) * pad
    } else {
        n_len
    };
    let n_len = n_len + pad;

    // Pad samples.
    let mut padded = samples.to_vec();
    let to_add = n_len * fft_step - samples.len();
    padded.resize(padded.len() + to_add, 0.0);

    let n_fft = 1 + fft_size / 2;

    // Pre-plan FFT (rustfft: SIMD-accelerated on native).
    let mut planner = FftPlanner::<f32>::new();
    let fft_plan = planner.plan_fft_forward(fft_size);

    // Reusable buffers.
    let mut fft_buf = vec![Complex32::new(0.0, 0.0); fft_size];
    let mut power = vec![0.0f32; n_fft];

    let mut mel = vec![0.0f32; n_len * n_mel];
    let n_samples = padded.len();
    let end = std::cmp::min(n_samples / fft_step + 1, n_len);

    for i in 0..end {
        let offset = i * fft_step;

        // Apply Hann window → complex buffer.
        let copy_len = std::cmp::min(fft_size, n_samples - offset);
        for j in 0..copy_len {
            fft_buf[j] = Complex32::new(hann[j] * padded[offset + j], 0.0);
        }
        for j in copy_len..fft_size {
            fft_buf[j] = Complex32::new(0.0, 0.0);
        }

        // In-place FFT (SIMD on native).
        fft_plan.process(&mut fft_buf);

        // Power spectrum: |X[k]|² + symmetric fold.
        for k in 0..n_fft {
            power[k] = fft_buf[k].norm_sqr();
        }
        for j in 1..fft_size / 2 {
            power[j] += fft_buf[fft_size - j].norm_sqr();
        }

        // Mel filter bank application.
        for j in 0..n_mel {
            let mut sum = 0.0f32;
            let filter_row = j * n_fft;
            // 4x unrolled inner loop.
            let mut k = 0;
            while k + 3 < n_fft {
                sum += power[k] * filters[filter_row + k]
                    + power[k + 1] * filters[filter_row + k + 1]
                    + power[k + 2] * filters[filter_row + k + 2]
                    + power[k + 3] * filters[filter_row + k + 3];
                k += 4;
            }
            while k < n_fft {
                sum += power[k] * filters[filter_row + k];
                k += 1;
            }
            mel[j * n_len + i] = sum.max(1e-10).log10();
        }
    }

    // Normalize (matching candle/whisper.cpp).
    let mmax = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max) - 8.0;
    for m in mel.iter_mut() {
        *m = m.max(mmax) / 4.0 + 1.0;
    }

    // Extract [n_mel, target_frames] from [n_mel, n_len].
    let mut output = vec![0.0f32; n_mel * target_frames];
    for m in 0..n_mel {
        let src_start = m * n_len;
        let dst_start = m * target_frames;
        let copy = target_frames.min(n_len);
        output[dst_start..dst_start + copy]
            .copy_from_slice(&mel[src_start..src_start + copy]);
    }

    output
}

// ── Mel filter bank (Slaney scale) ──────────────────────────
//
// Matches librosa.filters.mel(sr, n_fft, n_mels) with Slaney normalization.
// This is the correct formula used by OpenAI Whisper.

fn compute_mel_filters(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_fft_bins = n_fft / 2 + 1;
    let f_min = 0.0f32;
    let f_max = sample_rate as f32 / 2.0;

    // Slaney mel scale (NOT HTK).
    let mel_min = hz_to_mel_slaney(f_min);
    let mel_max = hz_to_mel_slaney(f_max);

    let n_points = n_mels + 2;
    let mel_points: Vec<f32> = (0..n_points)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f32 / (n_points - 1) as f32))
        .collect();

    // Hz → FFT bin index.
    let bin_points: Vec<f32> = mel_points
        .iter()
        .map(|&f| f * n_fft as f32 / sample_rate as f32)
        .collect();

    // Triangular filters with Slaney normalization.
    let mut filters = vec![0.0f32; n_mels * n_fft_bins];

    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for bin in 0..n_fft_bins {
            let freq = bin as f32;
            if freq >= left && freq < center && center > left {
                filters[m * n_fft_bins + bin] = (freq - left) / (center - left);
            } else if freq >= center && freq <= right && right > center {
                filters[m * n_fft_bins + bin] = (right - freq) / (right - center);
            }
        }

        // Slaney normalization: 2 / (f_upper - f_lower).
        let enorm = 2.0 / (mel_points[m + 2] - mel_points[m]);
        for bin in 0..n_fft_bins {
            filters[m * n_fft_bins + bin] *= enorm;
        }
    }

    filters
}

/// Load mel filters from a binary file (for models with custom filters).
/// Expected format: flat f32 LE, [n_mels, n_fft/2+1].
fn load_mel_filters(path: &str, n_mels: usize, n_fft: usize) -> Result<Vec<f32>, String> {
    let n_fft_bins = n_fft / 2 + 1;
    let expected_bytes = n_mels * n_fft_bins * 4;

    let data = std::fs::read(path)
        .map_err(|e| format!("mel: failed to read filters from {path}: {e}"))?;

    if data.len() != expected_bytes {
        return Err(format!(
            "mel: filter file size mismatch: expected {} bytes ([{}, {}] x f32), got {}",
            expected_bytes, n_mels, n_fft_bins, data.len()
        ));
    }

    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

// ── Slaney mel scale ─────────────────────────────────────────
//
// This is the mel scale used by librosa (default) and OpenAI Whisper.
// NOT the HTK formula (2595 * log10(1 + f/700)).
//
// Slaney: linear below 1kHz, logarithmic above.
//   f < 1000 Hz: mel = 3 * f / 200
//   f >= 1000 Hz: mel = 15 + 27 * ln(f / 1000) / ln(6.4)

const MEL_BREAK_FREQUENCY_HERTZ: f32 = 1000.0;
const LN_6_4: f32 = 1.8562980; // ln(6.4), precomputed

fn hz_to_mel_slaney(hz: f32) -> f32 {
    if hz < MEL_BREAK_FREQUENCY_HERTZ {
        hz * 3.0 / 200.0
    } else {
        15.0 + 27.0 * (hz / MEL_BREAK_FREQUENCY_HERTZ).ln() / LN_6_4
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let break_mel = 15.0; // hz_to_mel_slaney(1000.0) = 3 * 1000 / 200 = 15.0
    if mel < break_mel {
        mel * 200.0 / 3.0
    } else {
        MEL_BREAK_FREQUENCY_HERTZ * ((mel - 15.0) * LN_6_4 / 27.0).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_kernel_name() {
        let kernel = MelKernel::new();
        assert_eq!(kernel.name(), "mel");
    }

    #[test]
    fn test_slaney_mel_scale_roundtrip() {
        // Test that hz → mel → hz is identity.
        for &hz in &[0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel_slaney(hz);
            let recovered = mel_to_hz_slaney(mel);
            assert!(
                (recovered - hz).abs() < 0.1,
                "roundtrip failed for {hz}Hz: got {recovered}"
            );
        }
    }

    #[test]
    fn test_slaney_vs_htk_difference() {
        // Slaney and HTK should differ, especially at low frequencies.
        let hz = 500.0;
        let slaney = hz_to_mel_slaney(hz); // 3 * 500 / 200 = 7.5
        let htk = 2595.0 * (1.0 + hz / 700.0).log10(); // ≈ 607.4
        assert!((slaney - 7.5).abs() < 0.01, "Slaney(500) should be 7.5, got {slaney}");
        assert!(htk > 600.0, "HTK(500) should be ~607, got {htk}");
        // They are VERY different — confirms the bug in our old implementation.
    }

    #[test]
    fn test_compute_mel_filters_shape() {
        let filters = compute_mel_filters(16000, 400, 80);
        assert_eq!(filters.len(), 80 * 201);
        // Check sparsity: most values should be zero (triangular filters).
        let nonzero = filters.iter().filter(|&&v| v > 0.0).count();
        assert!(nonzero < filters.len() / 3, "mel filters should be sparse");
        assert!(nonzero > 0, "mel filters should have non-zero entries");
    }

    #[test]
    fn test_mel_spectrogram_output_shape() {
        // 1 second of silence at 16kHz → mel spectrogram.
        let n_fft = 400;
        let hop = 160;
        let n_mels = 80;
        let sample_rate = 16000;
        let chunk_length = 30;
        let chunk_samples = sample_rate * chunk_length;
        let n_frames = chunk_samples / hop;

        let samples = vec![0.0f32; chunk_samples];
        let filters = compute_mel_filters(sample_rate, n_fft, n_mels);
        let mel = log_mel_spectrogram(&samples, &filters, n_fft, hop, n_mels, n_frames);

        assert_eq!(mel.len(), n_mels * n_frames); // 80 * 3000 = 240,000
    }

    #[test]
    fn test_mel_kernel_blob_input() {
        // Create PCM blob: 480000 samples (30s at 16kHz) of silence.
        let n_samples = 480000;
        let pcm_bytes: Vec<u8> = vec![0.0f32; n_samples]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let input = KernelInput {
            json: serde_json::json!({}),
            blobs: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "pcm".to_string(),
                    crate::kernel::BlobData {
                        bytes: pcm_bytes,
                        meta: crate::blob::BlobMeta {
                            size: (n_samples * 4) as u64,
                            content_type: "audio/pcm-f32".to_string(),
                            shape: Some(vec![1, n_samples]),
                        },
                    },
                );
                m
            },
        };

        let ops = serde_json::json!({"n_fft": 400, "hop_length": 160, "n_mels": 80});
        let result = MelKernel::new().execute(input, ops).unwrap();

        match result {
            KernelOutput::Blob { data, content_type, shape } => {
                assert_eq!(content_type, "tensor/f32");
                assert_eq!(shape, Some(vec![1, 80, 3000]));
                assert_eq!(data.len(), 80 * 3000 * 4); // 960,000 bytes
            }
            _ => panic!("expected blob output"),
        }
    }
}
