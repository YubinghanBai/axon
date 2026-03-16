//! Benchmarks for audio kernel operations (decode + resample + mel spectrogram).
//!
//! Run: `cargo bench -p axon --features audio --bench audio_ops`
//!
//! Requires real WAV files in the sherpa test directory. Falls back to
//! synthetic audio if not found.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "audio")]
mod benches {
    use super::*;
    use axon::kernel::{KernelInput, KernelOutput};
    use axon::kernels::audio::AudioKernel;
    use axon::kernels::mel::MelKernel;
    use axon::ComputeKernel;
    use serde_json::json;
    use std::collections::HashMap;

    /// Path to sherpa test WAVs (relative to crate root).
    const TEST_WAV_DIR: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sherpa-onnx-zipformer-zh-en-2023-11-22/test_wavs"
    );

    fn load_test_wav() -> Option<Vec<u8>> {
        let path = format!("{TEST_WAV_DIR}/0.wav");
        std::fs::read(&path).ok()
    }

    /// Generate a synthetic 16kHz mono WAV (PCM16) of `duration_secs` seconds.
    fn synthetic_wav(duration_secs: f32, sample_rate: u32) -> Vec<u8> {
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let mut samples = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            // 440Hz sine wave
            let t = i as f32 / sample_rate as f32;
            let s = (t * 440.0 * 2.0 * std::f32::consts::PI).sin();
            samples.push((s * 32767.0) as i16);
        }

        // Build minimal WAV header.
        let data_size = (num_samples * 2) as u32;
        let file_size = 36 + data_size;
        let mut wav = Vec::with_capacity(44 + data_size as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for s in &samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        wav
    }

    fn make_audio_input(wav_bytes: &[u8]) -> KernelInput {
        let mut blobs = HashMap::new();
        blobs.insert(
            "_input".to_string(),
            axon::BlobData {
                bytes: wav_bytes.to_vec(),
                meta: axon::BlobMeta {
                    size: wav_bytes.len() as u64,
                    content_type: "audio/wav".to_string(),
                    shape: None,
                },
            },
        );
        KernelInput {
            json: serde_json::Value::Null,
            blobs,
        }
    }

    fn make_pcm_input(pcm_f32: &[f32], sample_rate: usize) -> KernelInput {
        let bytes: Vec<u8> = pcm_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        let num_samples = pcm_f32.len();
        let mut blobs = HashMap::new();
        blobs.insert(
            "_prev".to_string(),
            axon::BlobData {
                bytes,
                meta: axon::BlobMeta {
                    size: (num_samples * 4) as u64,
                    content_type: "audio/pcm-f32le".to_string(),
                    shape: Some(vec![1, num_samples]),
                },
            },
        );
        KernelInput {
            json: json!({"sample_rate": sample_rate}),
            blobs,
        }
    }

    pub fn bench_audio_decode(c: &mut Criterion) {
        let kernel = AudioKernel::new();
        let mut group = c.benchmark_group("audio_decode");

        // Try real WAV first, fall back to synthetic.
        let wav_data = load_test_wav()
            .unwrap_or_else(|| synthetic_wav(3.0, 16000));

        group.bench_function("decode_wav", |bench| {
            let input = make_audio_input(&wav_data);
            let ops = json!({"sample_rate": 16000});
            bench.iter(|| {
                kernel
                    .execute(black_box(input.clone()), black_box(ops.clone()))
                    .unwrap();
            });
        });

        // Benchmark decode + resample (44.1kHz → 16kHz).
        let wav_44k = synthetic_wav(3.0, 44100);
        group.bench_function("decode_resample_44k_to_16k", |bench| {
            let input = make_audio_input(&wav_44k);
            let ops = json!({"sample_rate": 16000});
            bench.iter(|| {
                kernel
                    .execute(black_box(input.clone()), black_box(ops.clone()))
                    .unwrap();
            });
        });

        group.finish();
    }

    pub fn bench_audio_decode_durations(c: &mut Criterion) {
        let kernel = AudioKernel::new();
        let mut group = c.benchmark_group("audio_decode_duration");

        for &secs in &[1.0f32, 5.0, 10.0, 30.0] {
            let wav = synthetic_wav(secs, 16000);
            group.bench_with_input(
                BenchmarkId::new("16khz", format!("{secs}s")),
                &wav,
                |bench, wav| {
                    let input = make_audio_input(wav);
                    let ops = json!({"sample_rate": 16000});
                    bench.iter(|| {
                        kernel
                            .execute(black_box(input.clone()), black_box(ops.clone()))
                            .unwrap();
                    });
                },
            );
        }
        group.finish();
    }

    pub fn bench_mel_spectrogram(c: &mut Criterion) {
        let kernel = MelKernel::new();
        let mut group = c.benchmark_group("mel_spectrogram");

        // Simulate PCM output from audio.decode for various durations.
        for &secs in &[1.0f32, 5.0, 10.0, 30.0] {
            let num_samples = (16000.0 * secs) as usize;
            // Synthetic sine wave PCM.
            let pcm: Vec<f32> = (0..num_samples)
                .map(|i| (i as f32 / 16000.0 * 440.0 * 2.0 * std::f32::consts::PI).sin())
                .collect();

            let ops = json!({
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 80,
                "sample_rate": 16000
            });

            group.bench_with_input(
                BenchmarkId::new("80_mels", format!("{secs}s")),
                &pcm,
                |bench, pcm| {
                    let input = make_pcm_input(pcm, 16000);
                    bench.iter(|| {
                        kernel
                            .execute(black_box(input.clone()), black_box(ops.clone()))
                            .unwrap();
                    });
                },
            );
        }
        group.finish();
    }

    pub fn bench_full_audio_to_mel(c: &mut Criterion) {
        let audio_kernel = AudioKernel::new();
        let mel_kernel = MelKernel::new();

        let wav = load_test_wav()
            .unwrap_or_else(|| synthetic_wav(5.0, 16000));

        c.bench_function("full_pipeline_wav_to_mel", |bench| {
            let input = make_audio_input(&wav);
            let audio_ops = json!({"sample_rate": 16000});
            let mel_ops = json!({
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 80,
                "sample_rate": 16000
            });

            bench.iter(|| {
                // Step 1: audio decode
                let pcm_output = audio_kernel
                    .execute(black_box(input.clone()), black_box(audio_ops.clone()))
                    .unwrap();

                // Step 2: convert output to mel input
                let pcm_input = match pcm_output {
                    KernelOutput::Blob {
                        data,
                        content_type,
                        shape,
                    } => {
                        let mut blobs = HashMap::new();
                        blobs.insert(
                            "_prev".to_string(),
                            axon::BlobData {
                                bytes: data.clone(),
                                meta: axon::BlobMeta {
                                    size: data.len() as u64,
                                    content_type,
                                    shape,
                                },
                            },
                        );
                        KernelInput {
                            json: json!({"sample_rate": 16000}),
                            blobs,
                        }
                    }
                    _ => panic!("expected blob output from audio kernel"),
                };

                // Step 3: mel spectrogram
                mel_kernel
                    .execute(black_box(pcm_input), black_box(mel_ops.clone()))
                    .unwrap();
            });
        });
    }
}

#[cfg(feature = "audio")]
criterion_group!(
    audio_benches,
    benches::bench_audio_decode,
    benches::bench_audio_decode_durations,
    benches::bench_mel_spectrogram,
    benches::bench_full_audio_to_mel,
);

#[cfg(feature = "audio")]
criterion_main!(audio_benches);

#[cfg(not(feature = "audio"))]
fn main() {
    eprintln!("audio_ops benchmarks require --features audio");
}
