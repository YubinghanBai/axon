//! Axon CLI — standalone ML inference runtime.
//!
//! Usage:
//!   axon run <manifest.toml> --input <file> [--output <file>]
//!   axon bench <manifest.toml> --input <file> [--iterations <n>]
//!   axon init <model.onnx> [--output <manifest.toml>]
//!   axon info <manifest.toml>
//!   axon kernels

use std::path::PathBuf;
use std::process;
use std::time::Instant;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use axon::{KernelRegistry, Pipeline};

#[derive(Parser)]
#[command(
    name = "axon",
    about = "Standalone ML inference runtime",
    version,
    long_about = "Axon runs ML inference pipelines defined in manifest.toml files.\n\n\
                  Pre-processing → ONNX model inference → Post-processing,\n\
                  all in pure Rust with zero Python dependencies."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run an ML inference pipeline.
    Run {
        /// Path to manifest.toml.
        manifest: PathBuf,

        /// Input file (image, audio, text).
        #[arg(short, long)]
        input: PathBuf,

        /// Output file (default: stdout as JSON).
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Benchmark a pipeline with per-step timing.
    Bench {
        /// Path to manifest.toml.
        manifest: PathBuf,

        /// Input file (image, audio, text).
        #[arg(short, long)]
        input: PathBuf,

        /// Number of iterations (default: 10).
        #[arg(short = 'n', long, default_value = "10")]
        iterations: usize,
    },

    /// Auto-generate a manifest.toml from an ONNX model file.
    Init {
        /// Path to ONNX model file.
        model: PathBuf,

        /// Output manifest file (default: <model_name>.toml).
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Show pipeline info from a manifest.
    Info {
        /// Path to manifest.toml.
        manifest: PathBuf,
    },

    /// Download a model from HuggingFace Hub.
    Download {
        /// Model name or HuggingFace repo ID.
        /// Built-in aliases: yolov8n, bert, whisper-tiny
        /// Or use full repo ID: sentence-transformers/all-MiniLM-L6-v2
        model: String,

        /// Output directory for downloaded files (default: models/).
        #[arg(short, long, default_value = "models")]
        output_dir: PathBuf,

        /// Skip manifest generation after download.
        #[arg(long)]
        no_manifest: bool,
    },

    /// Serve pipelines via HTTP REST API.
    #[cfg(feature = "serve")]
    Serve {
        /// Named pipelines: name=path/to/manifest.toml (can be repeated).
        #[arg(short, long, value_parser = parse_pipeline_arg)]
        pipeline: Vec<(String, String)>,

        /// Port to listen on (default: 8080).
        #[arg(long, default_value = "8080")]
        port: u16,

        /// Host to bind to (default: 0.0.0.0).
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Max request body size in MB (default: 50).
        #[arg(long, default_value = "50")]
        max_body_mb: usize,

        /// Enable dynamic batching: collect concurrent requests into batches.
        /// Reduces ORT session contention under high concurrency.
        #[arg(long)]
        batch_size: Option<usize>,

        /// Batch collection timeout in milliseconds (default: 10).
        /// After the first request, wait up to this long for more requests.
        #[arg(long, default_value = "10")]
        batch_timeout_ms: u64,

        /// Enable OpenTelemetry OTLP tracing export.
        /// Sends spans to OTEL_EXPORTER_OTLP_ENDPOINT (default: http://localhost:4318).
        /// View in Jaeger, Grafana Tempo, Zipkin, etc.
        #[arg(long)]
        #[cfg(feature = "otel")]
        otel: bool,
    },

    /// List available compute kernels.
    Kernels,
}

/// Parse "name=path" pipeline argument.
fn parse_pipeline_arg(s: &str) -> Result<(String, String), String> {
    let (name, path) = s
        .split_once('=')
        .ok_or_else(|| format!("expected NAME=PATH format, got '{s}'"))?;
    if name.is_empty() {
        return Err("pipeline name cannot be empty".to_string());
    }
    if path.is_empty() {
        return Err("pipeline path cannot be empty".to_string());
    }
    Ok((name.to_string(), path.to_string()))
}

fn main() {
    let cli = Cli::parse();

    // Skip default tracing init when OTel will set up its own subscriber.
    let skip_default_tracing = {
        #[cfg(feature = "otel")]
        {
            if let Command::Serve { otel, .. } = &cli.command {
                *otel
            } else {
                false
            }
        }
        #[cfg(not(feature = "otel"))]
        {
            false
        }
    };

    if !skip_default_tracing {
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
            )
            .with_target(false)
            .init();
    }

    let result = match cli.command {
        Command::Run {
            manifest,
            input,
            output,
        } => cmd_run(&manifest, &input, output.as_deref()),
        Command::Bench {
            manifest,
            input,
            iterations,
        } => cmd_bench(&manifest, &input, iterations),
        Command::Init { model, output } => cmd_init(&model, output.as_deref()),
        Command::Download {
            model,
            output_dir,
            no_manifest,
        } => cmd_download(&model, &output_dir, !no_manifest),
        Command::Info { manifest } => cmd_info(&manifest),
        #[cfg(all(feature = "serve", not(feature = "otel")))]
        Command::Serve {
            pipeline,
            port,
            host,
            max_body_mb,
            batch_size,
            batch_timeout_ms,
        } => cmd_serve(pipeline, port, &host, max_body_mb, batch_size, batch_timeout_ms),
        #[cfg(feature = "otel")]
        Command::Serve {
            pipeline,
            port,
            host,
            max_body_mb,
            batch_size,
            batch_timeout_ms,
            otel,
        } => {
            if otel {
                cmd_serve_with_otel(pipeline, port, &host, max_body_mb, batch_size, batch_timeout_ms)
            } else {
                cmd_serve(pipeline, port, &host, max_body_mb, batch_size, batch_timeout_ms)
            }
        },
        Command::Kernels => cmd_kernels(),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

// ── Run ────────────────────────────────────────────────────────

fn cmd_run(
    manifest_path: &PathBuf,
    input_path: &PathBuf,
    output_path: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = Pipeline::load(manifest_path).map_err(|e| format!("load: {e}"))?;

    if let Err(missing) = pipeline.validate() {
        eprintln!(
            "warning: missing kernels: {}. Pipeline may fail at runtime.",
            missing.join(", ")
        );
    }

    let input_bytes = std::fs::read(input_path)
        .map_err(|e| format!("read '{}': {e}", input_path.display()))?;

    let content_type = guess_content_type(input_path);

    let output = pipeline
        .run(&input_bytes, &content_type)
        .map_err(|e| format!("pipeline: {e}"))?;

    let json = match output {
        axon::KernelOutput::Json(v) => serde_json::to_string_pretty(&v)?,
        axon::KernelOutput::Blob {
            data,
            content_type,
            shape,
        } => {
            if let Some(path) = output_path {
                std::fs::write(path, &data)?;
                eprintln!(
                    "wrote {} bytes ({content_type}) to {}",
                    data.len(),
                    path.display()
                );
                return Ok(());
            }
            serde_json::to_string_pretty(&serde_json::json!({
                "_type": "blob",
                "size": data.len(),
                "content_type": content_type,
                "shape": shape,
            }))?
        }
    };

    if let Some(path) = output_path {
        std::fs::write(path, &json)?;
        eprintln!("wrote output to {}", path.display());
    } else {
        println!("{json}");
    }
    Ok(())
}

// ── Bench ──────────────────────────────────────────────────────

fn cmd_bench(
    manifest_path: &PathBuf,
    input_path: &PathBuf,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = Pipeline::load(manifest_path).map_err(|e| format!("load: {e}"))?;

    if let Err(missing) = pipeline.validate() {
        return Err(format!("missing kernels: {}", missing.join(", ")).into());
    }

    let input_bytes = std::fs::read(input_path)
        .map_err(|e| format!("read '{}': {e}", input_path.display()))?;

    let content_type = guess_content_type(input_path);

    println!("Pipeline: {}", manifest_path.display());
    println!("Input:    {} ({content_type}, {} bytes)", input_path.display(), input_bytes.len());
    println!("Iterations: {iterations}");
    println!();

    // Warmup run (includes model loading).
    let warmup_start = Instant::now();
    let _ = pipeline
        .run(&input_bytes, &content_type)
        .map_err(|e| format!("pipeline: {e}"))?;
    let warmup_time = warmup_start.elapsed();
    println!("Warmup (incl. model load): {:.1}ms", warmup_time.as_secs_f64() * 1000.0);
    println!();

    // Timed runs.
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = pipeline
            .run(&input_bytes, &content_type)
            .map_err(|e| format!("pipeline: {e}"))?;
        times.push(start.elapsed());
    }

    // Per-step timing (single detailed run).
    println!("Step-by-step timing (single run):");
    let step_times = pipeline
        .run_timed(&input_bytes, &content_type)
        .map_err(|e| format!("pipeline: {e}"))?;

    let mut total_steps = std::time::Duration::ZERO;
    for (name, dur) in &step_times {
        let ms = dur.as_secs_f64() * 1000.0;
        total_steps += *dur;
        println!("  {:<30} {:>8.2}ms", name, ms);
    }
    println!("  {:<30} {:>8.2}ms", "TOTAL", total_steps.as_secs_f64() * 1000.0);

    // Summary statistics.
    println!();
    let times_ms: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let min = times_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sorted = times_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];

    println!("Summary ({iterations} iterations):");
    println!("  mean:  {mean:.2}ms");
    println!("  min:   {min:.2}ms");
    println!("  max:   {max:.2}ms");
    println!("  p50:   {p50:.2}ms");
    println!("  p95:   {p95:.2}ms");

    Ok(())
}

// ── Init ───────────────────────────────────────────────────────

fn cmd_init(
    model_path: &PathBuf,
    output_path: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use ort::session::Session;
    use ort::session::builder::GraphOptimizationLevel;
    ort::init().commit();

    // Load model to inspect metadata.
    let session = Session::builder()
        .map_err(|e| format!("session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| format!("optimization: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("load model '{}': {e}", model_path.display()))?;

    let inputs = session.inputs();
    let outputs = session.outputs();

    // Extract model name from filename.
    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    // Display model info.
    println!("Model: {}", model_path.display());
    println!();
    println!("Inputs ({}):", inputs.len());
    for input in inputs {
        let (dtype, shape) = describe_outlet(input.dtype());
        println!("  {} : {} {:?}", input.name(), dtype, shape);
    }
    println!();
    println!("Outputs ({}):", outputs.len());
    for output in outputs {
        let (dtype, shape) = describe_outlet(output.dtype());
        println!("  {} : {} {:?}", output.name(), dtype, shape);
    }
    println!();

    // Detect model type from input shape heuristics.
    let model_type = detect_model_type(inputs, outputs);
    println!("Detected type: {model_type}");
    println!();

    // Generate manifest.
    let manifest = generate_manifest(model_name, model_path, inputs, outputs, &model_type);

    // Write or print.
    let output_file = output_path.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        PathBuf::from(format!("{model_name}.toml"))
    });

    std::fs::write(&output_file, &manifest)?;
    println!("Wrote manifest to: {}", output_file.display());
    println!();
    println!("--- {} ---", output_file.display());
    print!("{manifest}");

    Ok(())
}

/// Describe a ValueType as (dtype_string, shape_vec).
fn describe_outlet(vt: &ort::value::ValueType) -> (String, Vec<String>) {
    match vt {
        ort::value::ValueType::Tensor { ty, shape, .. } => {
            let dtype = format!("{ty:?}").to_lowercase();
            let dims: Vec<String> = shape
                .iter()
                .map(|&d| if d < 0 { "?".to_string() } else { d.to_string() })
                .collect();
            (dtype, dims)
        }
        _ => ("unknown".to_string(), vec![]),
    }
}

/// Detect model type from input/output shapes.
fn detect_model_type(
    inputs: &[ort::value::Outlet],
    outputs: &[ort::value::Outlet],
) -> String {
    // Check input shapes for heuristic detection.
    if let Some(input) = inputs.first() {
        if let ort::value::ValueType::Tensor { shape, ty, .. } = input.dtype() {
            let dims: Vec<i64> = shape.iter().copied().collect();

            // Vision model: [batch, channels, H, W] with channels=3
            if dims.len() == 4 && (dims[1] == 3 || dims[3] == 3) {
                // Check output for detection vs classification.
                if let Some(output) = outputs.first() {
                    if let ort::value::ValueType::Tensor { shape: out_shape, .. } = output.dtype() {
                        let out_dims: Vec<i64> = out_shape.iter().copied().collect();
                        // YOLO-style: [1, num_classes+4, num_detections]
                        if out_dims.len() == 3 && out_dims[1] > 4 {
                            return "detection".to_string();
                        }
                        // Classification: [1, num_classes]
                        if out_dims.len() == 2 {
                            return "classification".to_string();
                        }
                    }
                }
                return "vision".to_string();
            }

            // Audio model: input named with "mel", "audio", or shape [1, n_mels, time]
            if input.name().contains("mel")
                || input.name().contains("audio")
                || input.name().contains("input_features")
            {
                return "audio".to_string();
            }

            // NLP model: input named "input_ids" or int64 type with 2D shape
            if input.name().contains("input_ids")
                || input.name().contains("token")
                || (dims.len() == 2 && format!("{ty:?}").contains("Int64"))
            {
                return "nlp".to_string();
            }

            // Embedding: 2D or 3D float output
            if dims.len() <= 3 {
                return "embedding".to_string();
            }
        }
    }

    "generic".to_string()
}

/// Generate manifest TOML for a detected model type.
fn generate_manifest(
    model_name: &str,
    model_path: &PathBuf,
    inputs: &[ort::value::Outlet],
    outputs: &[ort::value::Outlet],
    model_type: &str,
) -> String {
    let model_file = model_path.file_name().unwrap_or_default().to_string_lossy();

    // Get input shape info for comments.
    let input_info = if let Some(input) = inputs.first() {
        let (dtype, shape) = describe_outlet(input.dtype());
        format!("{}: {} [{}]", input.name(), dtype, shape.join(", "))
    } else {
        "unknown".to_string()
    };

    let output_info = if let Some(output) = outputs.first() {
        let (dtype, shape) = describe_outlet(output.dtype());
        format!("{}: {} [{}]", output.name(), dtype, shape.join(", "))
    } else {
        "unknown".to_string()
    };

    let mut manifest = format!(
        r#"# Axon Pipeline: {model_name}
#
# Auto-generated by `axon init`
# Input:  {input_info}
# Output: {output_info}
#
# Usage:
#   axon run {model_name}.toml --input <file>

[model]
name = "{model_name}"
file = "{model_file}"
"#
    );

    // Add input/output specs.
    if let Some(input) = inputs.first() {
        if let ort::value::ValueType::Tensor { ty, shape, .. } = input.dtype() {
            let dims: Vec<i64> = shape.iter().copied().collect();
            let dtype = format!("{ty:?}").to_lowercase();
            // Only include spec if shape is fully static.
            if dims.iter().all(|&d| d > 0) {
                let shape_str: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
                manifest.push_str(&format!(
                    "\n[model.input]\nshape = [{}]\ndtype = \"{dtype}\"\n",
                    shape_str.join(", ")
                ));
            }
        }
    }

    if let Some(output) = outputs.first() {
        if let ort::value::ValueType::Tensor { ty, shape, .. } = output.dtype() {
            let dims: Vec<i64> = shape.iter().copied().collect();
            let dtype = format!("{ty:?}").to_lowercase();
            if dims.iter().all(|&d| d > 0) {
                let shape_str: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
                manifest.push_str(&format!(
                    "\n[model.output]\nshape = [{}]\ndtype = \"{dtype}\"\n",
                    shape_str.join(", ")
                ));
            }
        }
    }

    // Generate pre/post steps based on model type.
    match model_type {
        "detection" => {
            let input_size = if let Some(input) = inputs.first() {
                if let ort::value::ValueType::Tensor { shape, .. } = input.dtype() {
                    let dims: Vec<i64> = shape.iter().copied().collect();
                    if dims.len() == 4 && dims[2] > 0 { dims[2] } else { 640 }
                } else { 640 }
            } else { 640 };

            // Detect YOLO-style output shape for split config.
            let (boxes_end, transpose) = if let Some(output) = outputs.first() {
                if let ort::value::ValueType::Tensor { shape, .. } = output.dtype() {
                    let dims: Vec<i64> = shape.iter().copied().collect();
                    // [1, 84, 8400] → features=84, 4 box coords + 80 classes
                    if dims.len() == 3 {
                        let features = dims[1];
                        // Transpose when dim1 < dim2 (features < detections).
                        // YOLO [1, 84, 8400]: features=84 < detections=8400 → transpose=true.
                        (4, features < dims[2])
                    } else { (4, true) }
                } else { (4, true) }
            } else { (4, true) };

            manifest.push_str(&format!(
                r#"
[pre]
steps = [
  {{ op = "image.decode" }},
  {{ op = "image.resize", target = {input_size}, mode = "letterbox" }},
  {{ op = "image.normalize", scale = 255.0 }},
  {{ op = "image.layout", to = "chw" }},
  {{ op = "tensor.unsqueeze", dim = 0 }},
]

[post]
steps = [
  {{ op = "detection.split", boxes = "0:{boxes_end}", scores = "{boxes_end}:", transpose = {transpose} }},
  {{ op = "detection.xywh_to_xyxy" }},
  {{ op = "detection.confidence_filter", threshold = 0.25 }},
  {{ op = "detection.nms", iou = 0.45 }},
  {{ op = "detection.format", output = "json" }},
]
"#
            ));
        }
        "classification" => {
            let input_size = if let Some(input) = inputs.first() {
                if let ort::value::ValueType::Tensor { shape, .. } = input.dtype() {
                    let dims: Vec<i64> = shape.iter().copied().collect();
                    if dims.len() == 4 && dims[2] > 0 { dims[2] } else { 224 }
                } else { 224 }
            } else { 224 };

            manifest.push_str(&format!(
                r#"
[pre]
steps = [
  {{ op = "image.decode" }},
  {{ op = "image.resize", target = {input_size}, mode = "stretch" }},
  {{ op = "image.normalize", scale = 255.0 }},
  {{ op = "image.layout", to = "chw" }},
  {{ op = "tensor.unsqueeze", dim = 0 }},
]

[post]
steps = [
  {{ op = "tensor.softmax", dim = 1 }},
  {{ op = "tensor.argmax", dim = 1 }},
]
"#
            ));
        }
        "nlp" => {
            // Find tokenizer path hint.
            manifest.push_str(
                r#"
[pre]
steps = [
  # TODO: set tokenizer path to your tokenizer.json
  { op = "tokenizer.encode", tokenizer = "tokenizer.json", max_length = 128, padding = true },
]

[post]
steps = [
  { op = "tensor.mean_pool", dim = 1 },
  { op = "tensor.normalize" },
]
"#,
            );
        }
        "audio" => {
            manifest.push_str(
                r#"
[pre]
steps = [
  { op = "audio.decode", sample_rate = 16000, channels = 1 },
  { op = "mel.spectrogram", n_fft = 400, hop_length = 160, n_mels = 80 },
  { op = "tensor.reshape", shape = [1, 80, 3000] },
]

# [post]
# steps = [
#   # Add post-processing steps for your model's output format.
# ]
"#,
            );
        }
        _ => {
            manifest.push_str(
                r#"
# [pre]
# steps = [
#   # Add pre-processing steps for your input format.
#   # Examples:
#   #   { op = "image.decode" }
#   #   { op = "audio.decode", sample_rate = 16000 }
#   #   { op = "tokenizer.encode", tokenizer = "tokenizer.json" }
# ]

# [post]
# steps = [
#   # Add post-processing steps for your model's output.
# ]
"#,
            );
        }
    }

    manifest
}

// ── Download ──────────────────────────────────────────────────

/// Built-in model registry: alias → (repo_id, files_to_download, optional_tokenizer)
struct ModelSpec {
    repo_id: &'static str,
    files: &'static [&'static str],
    tokenizer: Option<&'static str>,
}

fn builtin_models() -> Vec<(&'static str, ModelSpec)> {
    vec![
        (
            "yolov8n",
            ModelSpec {
                repo_id: "ultralytics/yolov8",
                files: &["yolov8n.onnx"],
                tokenizer: None,
            },
        ),
        (
            "yolov8s",
            ModelSpec {
                repo_id: "ultralytics/yolov8",
                files: &["yolov8s.onnx"],
                tokenizer: None,
            },
        ),
        (
            "bert",
            ModelSpec {
                repo_id: "sentence-transformers/all-MiniLM-L6-v2",
                files: &["onnx/model.onnx"],
                tokenizer: Some("tokenizer.json"),
            },
        ),
        (
            "all-MiniLM-L6-v2",
            ModelSpec {
                repo_id: "sentence-transformers/all-MiniLM-L6-v2",
                files: &["onnx/model.onnx"],
                tokenizer: Some("tokenizer.json"),
            },
        ),
        (
            "whisper-tiny",
            ModelSpec {
                repo_id: "Xenova/whisper-tiny",
                files: &["onnx/encoder_model.onnx", "onnx/decoder_model.onnx"],
                tokenizer: Some("tokenizer.json"),
            },
        ),
        (
            "whisper-base",
            ModelSpec {
                repo_id: "Xenova/whisper-base",
                files: &["onnx/encoder_model.onnx", "onnx/decoder_model.onnx"],
                tokenizer: Some("tokenizer.json"),
            },
        ),
    ]
}

fn cmd_download(
    model_name: &str,
    output_dir: &PathBuf,
    generate_manifest: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use reqwest::blocking::Client;
    use std::io::Write;

    let mut builder = Client::builder().user_agent("axon-ml/0.1");

    // Support HF_TOKEN for private/gated models.
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|e| format!("invalid HF_TOKEN: {e}"))?,
        );
        builder = builder.default_headers(headers);
        println!("Using HF_TOKEN for authentication");
    }

    let client = builder.build()?;

    // Resolve model name → repo + files.
    let builtins = builtin_models();
    let (repo_id, files, tokenizer_file) = if let Some((_, spec)) =
        builtins.iter().find(|(name, _)| *name == model_name)
    {
        println!("Resolved '{model_name}' → {}", spec.repo_id);
        (
            spec.repo_id.to_string(),
            spec.files.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            spec.tokenizer.map(|s| s.to_string()),
        )
    } else if model_name.contains('/') {
        // Full repo ID — auto-discover ONNX files.
        println!("Querying HuggingFace: {model_name}");
        let (files, tokenizer) = discover_repo_files(&client, model_name)?;
        (model_name.to_string(), files, tokenizer)
    } else {
        return Err(format!(
            "Unknown model '{}'. Built-in aliases: {}\nOr use a full HuggingFace repo ID: owner/model-name",
            model_name,
            builtins.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
        )
        .into());
    };

    if files.is_empty() {
        return Err(format!("No ONNX files found in repo '{repo_id}'").into());
    }

    // Create output directory.
    std::fs::create_dir_all(output_dir)?;
    println!("Output directory: {}", output_dir.display());
    println!();

    // Download each file.
    let mut downloaded_onnx: Vec<PathBuf> = Vec::new();
    let mut all_files = files.clone();
    if let Some(ref tok) = tokenizer_file {
        all_files.push(tok.clone());
    }

    for file in &all_files {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, file
        );

        // Local filename: flatten path (onnx/model.onnx → model.onnx).
        let local_name = std::path::Path::new(file)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let dest = output_dir.join(&local_name);

        // Skip if already exists.
        if dest.exists() {
            let size = std::fs::metadata(&dest)?.len();
            println!("  ✓ {} ({}) — already exists, skipping", local_name, format_bytes(size));
            if local_name.ends_with(".onnx") {
                downloaded_onnx.push(dest);
            }
            continue;
        }

        print!("  ↓ {} ", local_name);
        std::io::stdout().flush()?;

        let response = client
            .get(&url)
            .send()
            .map_err(|e| format!("download {file}: {e}"))?;

        if !response.status().is_success() {
            println!("FAILED ({})", response.status());
            eprintln!("    URL: {url}");
            continue;
        }

        let total_size = response.content_length();
        if let Some(total) = total_size {
            print!("({})", format_bytes(total));
            std::io::stdout().flush()?;
        }

        // Stream download to file.
        let mut out_file = std::fs::File::create(&dest)?;
        let mut downloaded: u64 = 0;
        let mut reader = response;
        let mut buf = vec![0u8; 256 * 1024]; // 256KB chunks
        loop {
            let n = std::io::Read::read(&mut reader, &mut buf)
                .map_err(|e| format!("download {file}: {e}"))?;
            if n == 0 {
                break;
            }
            out_file.write_all(&buf[..n])?;
            downloaded += n as u64;

            // Print progress.
            if let Some(total) = total_size {
                let pct = (downloaded as f64 / total as f64 * 100.0) as u32;
                print!("\r  ↓ {} ({}) ...{}%", local_name, format_bytes(total), pct);
                std::io::stdout().flush()?;
            }
        }
        drop(out_file);

        println!(
            "\r  ✓ {} ({})       ",
            local_name,
            format_bytes(downloaded)
        );

        if local_name.ends_with(".onnx") {
            downloaded_onnx.push(dest);
        }
    }

    println!();

    // Generate manifest for each downloaded ONNX model.
    let mut generated_manifests = Vec::new();
    if generate_manifest && !downloaded_onnx.is_empty() {
        for onnx_path in &downloaded_onnx {
            let manifest_name = onnx_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            // Place manifest in the output directory alongside the model.
            let manifest_path = output_dir.join(format!("{manifest_name}.toml"));

            println!("Generating manifest: {}", manifest_path.display());
            if let Err(e) = cmd_init(onnx_path, Some(&manifest_path)) {
                eprintln!("  warning: manifest generation failed: {e}");
            } else {
                generated_manifests.push(manifest_path);
            }
        }
    }

    println!();
    println!("Done! Run your pipeline with:");
    if let Some(manifest) = generated_manifests.first() {
        println!("  axon run {} --input <your-file>", manifest.display());
    } else if let Some(onnx) = downloaded_onnx.first() {
        println!("  axon init {} && axon run <manifest.toml> --input <your-file>", onnx.display());
    }

    Ok(())
}

/// Query HuggingFace API to discover ONNX files in a repo.
fn discover_repo_files(
    client: &reqwest::blocking::Client,
    repo_id: &str,
) -> Result<(Vec<String>, Option<String>), Box<dyn std::error::Error>> {
    let url = format!("https://huggingface.co/api/models/{repo_id}");
    let resp = client.get(&url).send()?;

    if !resp.status().is_success() {
        return Err(format!(
            "HuggingFace API error for '{}': {} {}",
            repo_id,
            resp.status(),
            resp.text().unwrap_or_default()
        )
        .into());
    }

    let body: serde_json::Value = resp.json()?;
    let siblings = body["siblings"]
        .as_array()
        .ok_or("unexpected API response: missing 'siblings'")?;

    let mut onnx_files = Vec::new();
    let mut tokenizer = None;

    for file in siblings {
        if let Some(name) = file["rfilename"].as_str() {
            if name.ends_with(".onnx") {
                onnx_files.push(name.to_string());
            }
            if name == "tokenizer.json" {
                tokenizer = Some(name.to_string());
            }
        }
    }

    // Prefer onnx/ subdirectory model if available.
    if onnx_files.len() > 1 {
        let onnx_dir: Vec<_> = onnx_files
            .iter()
            .filter(|f| f.starts_with("onnx/"))
            .cloned()
            .collect();
        if !onnx_dir.is_empty() {
            return Ok((onnx_dir, tokenizer));
        }
    }

    Ok((onnx_files, tokenizer))
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

// ── Serve ──────────────────────────────────────────────────────

#[cfg(feature = "serve")]
fn make_batch_config(batch_size: Option<usize>, batch_timeout_ms: u64) -> Option<axon::batch::BatchConfig> {
    batch_size.map(|size| {
        eprintln!("batching: enabled (max_batch={size}, timeout={batch_timeout_ms}ms)");
        axon::batch::BatchConfig {
            max_batch_size: size,
            timeout: std::time::Duration::from_millis(batch_timeout_ms),
        }
    })
}

#[cfg(feature = "serve")]
fn cmd_serve(
    pipelines: Vec<(String, String)>,
    port: u16,
    host: &str,
    max_body_mb: usize,
    batch_size: Option<usize>,
    batch_timeout_ms: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = axon::serve::ServeConfig {
        pipelines,
        port,
        host: host.to_string(),
        max_body_size: max_body_mb * 1024 * 1024,
        batch: make_batch_config(batch_size, batch_timeout_ms),
    };

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(axon::serve::run_server(config))?;
    Ok(())
}

#[cfg(feature = "otel")]
fn cmd_serve_with_otel(
    pipelines: Vec<(String, String)>,
    port: u16,
    host: &str,
    max_body_mb: usize,
    batch_size: Option<usize>,
    batch_timeout_ms: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize OTel before building the server — tracing subscriber must
    // be set before any spans are created.
    let provider = axon::otel::init_tracing_with_otel()?;
    eprintln!("otel: OTLP tracing enabled (service=axon)");

    let config = axon::serve::ServeConfig {
        pipelines,
        port,
        host: host.to_string(),
        max_body_size: max_body_mb * 1024 * 1024,
        batch: make_batch_config(batch_size, batch_timeout_ms),
    };

    let runtime = tokio::runtime::Runtime::new()?;
    let result = runtime.block_on(axon::serve::run_server(config));

    // Flush remaining spans before exit.
    axon::otel::shutdown_tracing(provider);

    result.map_err(|e| e.into())
}

// ── Info ───────────────────────────────────────────────────────

fn cmd_info(manifest_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(manifest_path)?;
    let manifest = axon::Manifest::from_toml(&content).map_err(|e| format!("parse: {e}"))?;

    println!("Model:   {}", manifest.model.name);
    println!("File:    {}", manifest.model.file);
    if let Some(ref v) = manifest.model.version {
        println!("Version: {v}");
    }
    if let Some(ref spec) = manifest.model.input {
        println!("Input:   {:?} ({})", spec.shape, spec.dtype);
    }
    if let Some(ref spec) = manifest.model.output {
        println!("Output:  {:?} ({})", spec.shape, spec.dtype);
    }

    if let Some(ref pre) = manifest.pre {
        println!("\nPre-processing ({} steps):", pre.steps.len());
        for (i, step) in pre.steps.iter().enumerate() {
            println!("  {i}. {}", step.op);
        }
    }

    if let Some(ref post) = manifest.post {
        println!("\nPost-processing ({} steps):", post.steps.len());
        for (i, step) in post.steps.iter().enumerate() {
            println!("  {i}. {}", step.op);
        }
    }

    let pipeline = Pipeline::new(
        manifest,
        KernelRegistry::with_defaults(),
        manifest_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf(),
    );
    let required = pipeline.required_kernels();
    println!("\nRequired kernels: {}", required.join(", "));

    if let Err(missing) = pipeline.validate() {
        println!("Missing kernels:  {} (compile with matching features)", missing.join(", "));
    } else {
        println!("Status: all kernels available");
    }

    Ok(())
}

// ── Kernels ────────────────────────────────────────────────────

fn cmd_kernels() -> Result<(), Box<dyn std::error::Error>> {
    let reg = KernelRegistry::with_defaults();
    let mut names = reg.names();
    names.sort();

    println!("Available kernels ({}):", names.len());
    for name in &names {
        println!("  - {name}");
    }

    println!("\nCompile with features to enable more:");
    println!("  --features onnx       → onnx, tensor");
    println!("  --features wasm       → wasm");
    println!("  --features tokenizer  → tokenizer");
    println!("  --features audio      → audio, mel");
    println!("  --features text       → text (regex ops)");
    println!("  --features sherpa     → sherpa (ASR, VAD, TTS)");

    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────

fn guess_content_type(path: &PathBuf) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        Some("bmp") => "image/bmp",
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        Some("flac") => "audio/flac",
        Some("ogg") => "audio/ogg",
        Some("json") => "application/json",
        Some("csv") => "text/csv",
        Some("txt") => "text/plain",
        _ => "application/octet-stream",
    }
    .to_string()
}
