//! Audio decode + resample kernel: raw audio bytes → PCM f32 mono.
//!
//! URI: `compute://audio`
//!
//! Input (JSON):
//!   {"file_base64": "<base64 WAV/MP3/AAC bytes>"}
//!   or upstream blob containing raw audio bytes
//!
//! Config:
//!   {"sample_rate": 16000}    — target sample rate (default: 16000)
//!
//! Output: KernelOutput::Blob
//!   raw f32 LE bytes, shape [1, N] where N = number of samples
//!   content_type: "audio/pcm-f32"
//!
//! Libraries:
//!   - symphonia: pure Rust audio decoder (WAV/MP3/AAC/FLAC)
//!   - rubato: pure Rust sinc resampler

use crate::error::AxonError;
use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};
use base64::Engine as _;
use tracing::info;

pub struct AudioKernel;

impl AudioKernel {
    pub fn new() -> Self {
        Self
    }
}

impl ComputeKernel for AudioKernel {
    fn name(&self) -> &str {
        "audio"
    }

    fn execute(
        &self,
        input: KernelInput,
        operations: serde_json::Value,
    ) -> Result<KernelOutput, AxonError> {
        let target_rate = operations
            .get("sample_rate")
            .and_then(|v| v.as_u64())
            .unwrap_or(16000) as usize;

        // Get audio bytes: from blob input or base64 JSON.
        let audio_bytes = if let Some(blob) = input.first_blob() {
            blob.bytes.clone()
        } else {
            let json = input.into_json();
            let b64 = json
                .get("file_base64")
                .or_else(|| json.get("audio_base64"))
                .and_then(|v| v.as_str())
                .ok_or("missing 'file_base64' or 'audio_base64'")?;
            base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| format!("base64: {e}"))?
        };

        // Decode audio → PCM f32 mono + source sample rate.
        let (samples, src_rate) = decode_audio(&audio_bytes)?;
        info!(
            src_rate,
            target_rate,
            samples = samples.len(),
            "audio: decoded"
        );

        // Resample if needed.
        let output_samples = if src_rate == target_rate {
            samples
        } else {
            resample(&samples, src_rate, target_rate)?
        };

        info!(
            output_samples = output_samples.len(),
            "audio: resampled"
        );

        // Output as raw f32 LE blob.
        let bytes: Vec<u8> = output_samples
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let n_samples = output_samples.len();

        Ok(KernelOutput::Blob {
            data: bytes,
            content_type: "audio/pcm-f32".to_string(),
            shape: Some(vec![1, n_samples]),
        })
    }
}

// ── Audio decoding via symphonia ─────────────────────────────

fn decode_audio(data: &[u8]) -> Result<(Vec<f32>, usize), String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let cursor = std::io::Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let probed = symphonia::default::get_probe()
        .format(&Hint::new(), mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("probe: {e}"))?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or("no audio track found")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("unknown sample rate")? as usize;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("decoder: {e}"))?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(e) => return Err(format!("packet: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder
            .decode(&packet)
            .map_err(|e| format!("decode: {e}"))?;

        let spec = *decoded.spec();
        let n_frames = decoded.frames();
        let mut sample_buf = SampleBuffer::<f32>::new(n_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let interleaved = sample_buf.samples();

        // Mix to mono.
        if channels > 1 {
            for frame in 0..n_frames {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += interleaved[frame * channels + ch];
                }
                all_samples.push(sum / channels as f32);
            }
        } else {
            all_samples.extend_from_slice(interleaved);
        }
    }

    if all_samples.is_empty() {
        return Err("no audio samples decoded".into());
    }

    Ok((all_samples, sample_rate))
}

// ── Resampling via rubato ────────────────────────────────────

fn resample(samples: &[f32], from_rate: usize, to_rate: usize) -> Result<Vec<f32>, String> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
        WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_rate as f64 / from_rate as f64;
    let chunk_size = 1024;

    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)
        .map_err(|e| format!("resampler init: {e}"))?;

    let mut output = Vec::new();
    let mut pos = 0;

    while pos + chunk_size <= samples.len() {
        let chunk = vec![samples[pos..pos + chunk_size].to_vec()];
        let out = resampler
            .process(&chunk, None)
            .map_err(|e| format!("resample: {e}"))?;
        output.extend_from_slice(&out[0]);
        pos += chunk_size;
    }

    // Tail: pad to chunk_size, process, keep proportional output.
    if pos < samples.len() {
        let remaining = samples.len() - pos;
        let mut last_chunk = vec![0.0f32; chunk_size];
        last_chunk[..remaining].copy_from_slice(&samples[pos..]);
        let out = resampler
            .process(&vec![last_chunk], None)
            .map_err(|e| format!("resample tail: {e}"))?;
        let keep = ((remaining as f64 * ratio).ceil() as usize).min(out[0].len());
        output.extend_from_slice(&out[0][..keep]);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_kernel_name() {
        let kernel = AudioKernel::new();
        assert_eq!(kernel.name(), "audio");
    }

    #[test]
    fn test_decode_wav() {
        // Generate a minimal valid WAV: 16kHz mono, 100 samples of silence.
        let wav = generate_test_wav(100, 16000);
        let (samples, rate) = decode_audio(&wav).unwrap();
        assert_eq!(rate, 16000);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_resample_44100_to_16000() {
        // 1 second of silence at 44100Hz.
        let samples = vec![0.0f32; 44100];
        let out = resample(&samples, 44100, 16000).unwrap();
        // Should be approximately 16000 samples (±10 for edge effects).
        assert!((out.len() as i64 - 16000).abs() < 100);
    }

    #[test]
    fn test_audio_kernel_base64_wav() {
        let wav = generate_test_wav(160, 16000);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
        let input = KernelInput::from_json(serde_json::json!({"file_base64": b64}));
        let ops = serde_json::json!({"sample_rate": 16000});

        let result = AudioKernel::new().execute(input, ops).unwrap();
        match result {
            KernelOutput::Blob { data, content_type, shape } => {
                assert_eq!(content_type, "audio/pcm-f32");
                assert_eq!(shape, Some(vec![1, 160]));
                assert_eq!(data.len(), 160 * 4); // 160 f32 samples
            }
            _ => panic!("expected blob output"),
        }
    }

    /// Generate a minimal WAV file: PCM 16-bit mono.
    fn generate_test_wav(n_samples: usize, sample_rate: u32) -> Vec<u8> {
        let data_size = (n_samples * 2) as u32; // 16-bit = 2 bytes per sample
        let file_size = 36 + data_size;
        let mut wav = Vec::with_capacity(file_size as usize + 8);

        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        // fmt chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        wav.extend_from_slice(&vec![0u8; data_size as usize]); // silence

        wav
    }
}
