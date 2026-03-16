# Axon

**Pure Rust ML inference runtime — from raw bytes to prediction in one binary.**

[![CI](https://github.com/anthropics/medulla-rse/actions/workflows/ci.yml/badge.svg)](https://github.com/anthropics/medulla-rse/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Axon is a zero-dependency (no Python, no C++ build chain) ML inference runtime.
Define your entire pipeline in a `manifest.toml` — pre-processing, ONNX model,
post-processing — and serve it via REST, gRPC, or CLI. One `cargo install`, one binary.

---

## At a Glance

- **50+ built-in ops** — image, audio, tensor, detection, tokenizer, text, WASM
- **HTTP + gRPC** serving with SSE streaming, dynamic batching, model versioning
- **Prometheus `/metrics`** + **OpenTelemetry OTLP** tracing — production-grade observability
- **Speech engine** — ASR / VAD / TTS via Sherpa-ONNX (streaming + offline)
- **WASM sandbox** — run untrusted pre/post-processing with fuel + memory limits
- **Vector DB sinks** — zero-ETL push to Qdrant, Weaviate, ChromaDB, Postgres
- **GPU acceleration** — CoreML, CUDA, TensorRT, DirectML execution providers
- **Content-addressed blob store** — BLAKE3-hashed, LRU-cached, optional disk tier
- **Model pool** — LRU warm-cached pipelines with preload + on-demand loading
- **Typed error codes** — `[AX001]`–`[AX009]` for programmatic error handling
- **Pipeline composition** — invoke sub-pipelines as steps
- **Autoregressive generation** — greedy decoding with KV-cache (encoder-decoder models)
- **4 fuzz targets** — continuous fuzzing for tensor, image, detection, error paths

---

## Competitive Comparison

| Capability | **Axon** | Triton | vLLM | Ollama | tract/candle |
|---|---|---|---|---|---|
| Pre/post-processing | 50+ built-in ops | Custom backends | None | None | DIY |
| Pipeline config | `manifest.toml` | `config.pbtxt` | Code | Modelfile | Code |
| Install | `cargo install` | Docker + NVIDIA | pip + CUDA | Go binary | `cargo add` |
| Binary size | ~15MB | 2GB+ image | 500MB+ | ~100MB | ~5MB |
| REST API | Built-in (axum) | Built-in | OpenAI-compat | OpenAI-compat | None |
| gRPC API | KServe V2 (tonic) | KServe V2 | None | None | None |
| SSE streaming | Built-in | Partial | Built-in | Built-in | None |
| Dynamic batching | Built-in | Built-in | Continuous | None | None |
| Prometheus metrics | Built-in `/metrics` | Built-in | Built-in | None | None |
| OpenTelemetry | OTLP span export | Custom | None | None | None |
| Model versioning | `name@version` | Built-in | None | Tags | None |
| Model pool / cache | LRU with preload | Per-model | KV cache | GGUF cache | None |
| GPU acceleration | CoreML/CUDA/TensorRT/DirectML | CUDA/TensorRT | CUDA | Metal/CUDA | CPU only |
| Speech (ASR/VAD/TTS) | Sherpa-ONNX native | Plugin | None | Whisper only | None |
| Vector DB sink | Qdrant/Weaviate/Chroma/PG | None | None | None | None |
| WASM sandbox | wasmtime (fuel+mem limits) | None | None | None | None |
| Autoregressive gen | Greedy + KV-cache | Custom | PagedAttention | llama.cpp | None |
| Blob store | BLAKE3 content-addressed | None | None | None | None |
| Error codes | Typed AX001–AX009 | gRPC status | Python exceptions | Go errors | anyhow |
| Fuzzing | 4 fuzz targets | None | None | None | None |
| Language | Rust | C++/Python | Python | Go/C++ | Rust |

---

## Quick Start

### Install

```bash
# Build with common features
cargo build -p axon --features "onnx,vision,audio,tokenizer,cli" --release

# Full production build (all features)
cargo build -p axon --features "onnx,vision,audio,tokenizer,text,wasm,cli,serve,grpc,otel,sink" --release

# Or install globally
cargo install --path crates/axon --features "onnx,vision,audio,tokenizer,cli,serve"
```

### CLI Commands

```bash
# ── Download ──────────────────────────────────────────────
# Download from HuggingFace with built-in aliases (auto-generates manifest)
axon download bert              # → sentence-transformers/all-MiniLM-L6-v2
axon download yolov8n           # → ultralytics/yolov8
axon download whisper-tiny      # → openai/whisper-tiny
axon download owner/repo-name   # Any HF repo (auto-discovers ONNX + tokenizer)
# Supports HF_TOKEN for private/gated models

# ── Init ──────────────────────────────────────────────────
# Auto-generate manifest from ONNX model (detects type: vision/nlp/audio/etc.)
axon init models/custom-model.onnx
axon init models/custom-model.onnx --output my-pipeline.toml

# ── Run ───────────────────────────────────────────────────
# Run inference (content-type auto-detected from file extension)
axon run examples/yolov8-detection.toml --input photo.jpg
axon run examples/bert-embedding.toml --input document.txt
axon run examples/whisper-transcription.toml --input recording.wav
axon run manifest.toml --input data.bin --output result.json

# ── Bench ─────────────────────────────────────────────────
# Benchmark with warmup, per-step timing, p50/p95 latency
axon bench examples/yolov8-detection.toml --input photo.jpg -n 20

# ── Info ──────────────────────────────────────────────────
# Inspect pipeline: model, I/O shapes, pre/post steps, required kernels
axon info examples/yolov8-detection.toml

# ── Kernels ───────────────────────────────────────────────
# List all compiled-in compute kernels
axon kernels
```

---

## Serve Mode

Multi-pipeline HTTP server with production features out of the box.

```bash
# Basic — serve multiple pipelines
axon serve \
  --pipeline yolo=examples/yolov8-detection.toml \
  --pipeline whisper=examples/whisper-transcription.toml \
  --port 8080

# With dynamic batching (collect up to 8 requests, 50ms window)
axon serve --pipeline yolo=manifest.toml --batch-size 8 --batch-timeout-ms 50

# With model versioning (name@version syntax)
axon serve \
  --pipeline yolo@v1=models/yolov8n-v1.toml \
  --pipeline yolo@v2=models/yolov8n-v2.toml

# With gRPC on separate port
axon serve --pipeline yolo=manifest.toml --grpc-port 8081

# With OpenTelemetry OTLP tracing
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317 \
  axon serve --pipeline yolo=manifest.toml --otel

# Full production setup
axon serve \
  --pipeline yolo@v2=manifest.toml \
  --batch-size 16 --batch-timeout-ms 20 \
  --grpc-port 8081 \
  --otel \
  --max-body-mb 100 \
  --port 8080
```

### REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/{name}` | Run inference (multipart file upload or raw body) |
| `POST` | `/{name}/stream` | SSE streaming with step-level progress events |
| `POST` | `/{name}/v/{version}` | Run inference on a specific model version |
| `GET` | `/{name}/info` | Pipeline metadata (model, steps, kernels) |
| `GET` | `/{name}/versions` | List available versions for a pipeline |
| `GET` | `/pipelines` | List all loaded pipelines with metadata |
| `GET` | `/metrics` | Prometheus text exposition format |
| `GET` | `/health` | Health check, uptime, pipeline list, batching status |

```bash
# Inference examples
curl -X POST http://localhost:8080/yolo -F "input=@photo.jpg"
curl -X POST http://localhost:8080/whisper --data-binary @audio.wav -H "Content-Type: audio/wav"
curl -X POST http://localhost:8080/bert -d '{"text": "hello world"}' -H "Content-Type: application/json"

# SSE streaming (real-time step progress)
curl -N -X POST http://localhost:8080/whisper/stream --data-binary @audio.wav -H "Content-Type: audio/wav"
# event: start
# data: {"pipeline":"whisper"}
# event: step
# data: {"phase":"pre","step":0,"op":"audio.decode","latency_ms":12.3}
# event: step
# data: {"phase":"model","step":0,"op":"onnx (cpu)","latency_ms":312.5}
# event: result
# data: {"result":{...},"total_latency_ms":420.3}

# Versioned inference
curl -X POST http://localhost:8080/yolo/v/v2 -F "input=@photo.jpg"

# Prometheus scrape
curl http://localhost:8080/metrics
# axon_requests_total 1042
# axon_requests_ok 1038
# axon_requests_errors 4
# axon_latency_avg_ms 23.45
# axon_uptime_seconds 3600
```

### gRPC API (KServe V2)

Dual-port architecture — HTTP and gRPC run simultaneously on separate ports.

Proto definition: [`proto/axon_infer.proto`](proto/axon_infer.proto)

| RPC | Description |
|-----|-------------|
| `ServerLive` | Liveness probe |
| `ServerReady` | Readiness probe |
| `ModelReady(name)` | Check if a specific pipeline is loaded |
| `ModelMetadata(name)` | Pipeline info: platform, kernels, pre/post steps |
| `ModelInfer(model_name, raw_input, content_type)` | Run inference, returns JSON or raw bytes + shape |

```bash
# Build with gRPC support
cargo build -p axon --features "grpc,onnx,vision,cli" --release

# Python client example
# channel = grpc.insecure_channel("localhost:8081")
# stub = AxonInferenceStub(channel)
# resp = stub.ModelInfer(ModelInferRequest(
#     model_name="yolo", raw_input=img_bytes, content_type="image/jpeg"
# ))
```

### Dynamic Batching

Collects concurrent requests into batches to maximize throughput.

```
Request 1 ─┐                ┌─ Response 1
Request 2 ─┤  BatchDispatcher  ├─ Response 2
Request 3 ─┤  (max_size=8,    ├─ Response 3
Request 4 ─┘  timeout=50ms)  └─ Response 4
```

- Requests queue via `mpsc` channel until `max_batch_size` or `timeout` reached
- Each request gets its own `oneshot` reply channel
- Prevents ORT session contention under concurrent load
- Responses include `"batched": true` header for transparency

---

## Observability

### Prometheus Metrics (`GET /metrics`)

Lock-free atomic counters, zero extra dependencies.

| Metric | Type | Description |
|--------|------|-------------|
| `axon_requests_total` | counter | Total inference requests |
| `axon_requests_ok` | counter | Successful inferences |
| `axon_requests_errors` | counter | Failed inferences |
| `axon_latency_avg_ms` | gauge | Average inference latency (ms) |
| `axon_uptime_seconds` | gauge | Server uptime |

### OpenTelemetry Tracing (`--otel` flag)

Full distributed tracing via OTLP export to any OTel-compatible backend
(Jaeger, Grafana Tempo, Datadog, etc.).

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317 axon serve --pipeline yolo=manifest.toml --otel
```

**Span hierarchy:**
```
axon_request (pipeline, body_size, content_type)
  └── pre_step[0] (op=image.decode, duration)
  └── pre_step[1] (op=image.resize, duration)
  └── model (onnx, device, duration)
  └── post_step[0] (op=detection.nms, duration)
```

Layered subscriber: console `fmt` output + OTLP span export. Graceful shutdown flushes remaining spans.

---

## Blob Store

Content-addressed binary storage for zero-copy pipeline data flow.

- **BLAKE3 hashing** — 256-bit content IDs (64-char hex)
- **Two-tier cache** — moka LRU in-memory (default 1024 entries) + optional disk
- **BlobMeta** — `{size, content_type, shape?}` metadata per blob
- **Zero-copy pipelines** — binary data flows between kernels without serialization
- Auto blob routing: pipeline sets `blob_output` on last pre-step and model step

---

## Model Pool

LRU warm-cached pipeline loading for multi-model serving.

```rust
use axon::ModelPool;

let pool = ModelPool::new(10); // max 10 models cached
pool.register("yolo", "manifests/yolo.toml");
pool.register("bert", "manifests/bert.toml");
pool.preload(&["yolo"])?; // warm up on startup

let pipeline = pool.get("bert")?; // load on first access, cached after
let output = pipeline.run(&input, "image/jpeg")?;

pool.evict("bert"); // manual eviction
```

- On-demand loading — models loaded on first request
- LRU eviction — least-recently-used freed when capacity exceeded
- Thread-safe — `parking_lot` read-write locks for concurrent access

---

## Vector DB Sinks (Zero-ETL)

Push embeddings directly from inference pipeline to vector databases.

```toml
[post]
steps = [
  { op = "tensor.mean_pool", dim = 1 },
  { op = "tensor.normalize" },
  { op = "sink.qdrant", url = "http://localhost:6333", collection = "embeddings", id = 42 },
]
```

| Sink | Target |
|------|--------|
| `sink.qdrant` | Qdrant vector DB |
| `sink.weaviate` | Weaviate |
| `sink.chromadb` | ChromaDB |
| `sink.postgres` | pgvector / Supabase |

---

## WASM Sandbox

Run untrusted pre/post-processing code in a sandboxed wasmtime runtime.

```toml
[post]
steps = [
  { op = "wasm.run", module = "custom_postprocess.wasm", fuel = 5000000, memory = 33554432 },
]
```

| Guard | Default | Description |
|-------|---------|-------------|
| CPU fuel | 10M instructions | Prevents infinite loops |
| Memory | 64 MB | Max linear memory |
| Stack | 512 KB | Stack overflow protection |
| I/O | Disabled | No filesystem, network, or syscalls |

**Module ABI:** exports `memory`, `alloc(size) → ptr`, `transform(ptr, len) → packed(out_ptr, out_len)`

---

## Pipeline Composition

Invoke sub-pipelines as steps within a parent pipeline.

```toml
[pre]
steps = [
  { op = "pipeline.run", manifest = "preprocess.toml" },
]
```

---

## Autoregressive Generation

Greedy decoding with KV-cache support for encoder-decoder models (Whisper, etc.).

```toml
[post]
steps = [
  { op = "generate.greedy", model = "decoder.onnx", max_tokens = 128, eos_token = 50257 },
  { op = "tokenizer.decode", tokenizer = "tokenizer.json", skip_special_tokens = true },
]
```

---

## Rust API

```rust
use axon::{Pipeline, ModelPool, KernelRegistry};

// ── One-shot inference ───────────────────────────────────
let pipeline = Pipeline::load("manifest.toml")?;
let output = pipeline.run(&image_bytes, "image/jpeg")?;

// ── Pre-processing only ──────────────────────────────────
let preprocessed = pipeline.run_pre(&audio_bytes, "audio/wav")?;

// ── With per-step timing ─────────────────────────────────
let (output, timings) = pipeline.run_timed(&input, "image/jpeg")?;
for (step_name, duration) in &timings {
    println!("{step_name}: {:.1}ms", duration.as_secs_f64() * 1000.0);
}

// ── With progress callback (for SSE / UI) ────────────────
pipeline.run_with_progress(&audio_bytes, "audio/wav", |step| {
    println!("[{}] step {}: {} ({:.1}ms)",
        step.phase, step.step, step.op,
        step.duration.as_secs_f64() * 1000.0);
})?;

// ── Pipeline validation ──────────────────────────────────
if let Err(missing) = pipeline.validate() {
    eprintln!("Missing kernels: {}", missing.join(", "));
}

// ── Custom kernel registration ───────────────────────────
let mut registry = KernelRegistry::new();
registry.register(Arc::new(MyCustomKernel));
let pipeline = Pipeline::new(manifest, registry, base_dir);
```

---

## Manifest Format

```toml
[model]
name = "yolov8n"                        # Human-readable name
file = "models/yolov8n.onnx"            # Path relative to manifest dir
version = "8.0.0"                        # Optional version string
device = "cpu"                           # cpu | coreml | cuda | tensorrt | directml
description = "YOLOv8 nano detection"    # Optional

[model.input]
shape = [1, 3, 640, 640]                # Expected input tensor shape
dtype = "f32"                            # f32 | f16 | i32 | i64 | u8

[model.output]
shape = [1, 84, 8400]                   # Expected output tensor shape
dtype = "f32"

[pre]
steps = [
  { op = "image.decode" },
  { op = "image.resize", target = 640, mode = "letterbox" },
  { op = "image.normalize", scale = 255.0, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] },
  { op = "image.layout", to = "chw" },
  { op = "tensor.unsqueeze", dim = 0 },
]

[post]
steps = [
  { op = "detection.split", boxes = "0:4", scores = "4:", transpose = true },
  { op = "detection.xywh_to_xyxy" },
  { op = "detection.confidence_filter", threshold = 0.25 },
  { op = "detection.nms", iou = 0.45 },
  { op = "detection.format", output = "json", labels = ["person", "car", "dog"] },
]
```

---

## Verified Pipelines

| Pipeline | Input | Model | Output |
|----------|-------|-------|--------|
| **YOLOv8 Detection** | JPEG/PNG | `yolov8n.onnx` (13MB) | `[{box, confidence, class}]` |
| **YOLOv8 CoreML** | JPEG/PNG | `yolov8n.onnx` + CoreML | Same, GPU-accelerated |
| **BERT Embedding** | Plain text | `all-MiniLM-L6-v2.onnx` | Normalized f32 vector |
| **Whisper Transcription** | WAV audio | encoder + decoder (145MB) | `{"text": "..."}` |
| **Whisper Int8** | WAV audio | quantized encoder + decoder | `{"text": "..."}` |
| **Sherpa ASR** | WAV audio | Zipformer / Paraformer / SenseVoice | `{"text": "..."}` |
| **Sherpa VAD** | WAV audio | Silero VAD | Speech segments |
| **Sherpa TTS** | Text JSON | VITS / Kokoro | PCM audio |

---

## Kernel Reference

### Tensor — feature: `onnx`

| Op | Description |
|----|-------------|
| `mean_pool` | Average across dimension (embedding extraction) |
| `normalize` | L2 normalization |
| `reshape` | Change tensor shape |
| `transpose` | Reorder dimensions |
| `softmax` | Probability normalization (numerically stable) |
| `argmax` | Index of max value |
| `topk` | Top-K values and indices |
| `unsqueeze` | Add dimension of size 1 |
| `squeeze` | Remove size-1 dimensions |
| `clamp` | Clamp values to range |
| `concat` | Concatenate tensors along axis |
| `slice` | Extract sub-tensor with per-dimension ranges |
| `gather` | Index-based gather (torch.gather equivalent) |
| `cast` | Type conversion (f32/f64/i64/f16) |
| `reduce_sum` | Sum reduction |
| `reduce_max` | Max reduction |
| `reduce_min` | Min reduction |
| `reduce_prod` | Product reduction |
| `where` | Conditional element selection |
| `matmul` | Matrix multiplication (2D) |

### Image — feature: `vision`

| Op | Description |
|----|-------------|
| `image.decode` | Auto-detect format (JPEG/PNG/WebP/BMP/GIF) → RGB f32 |
| `image.resize` | Scale to target (stretch / letterbox / crop) |
| `image.normalize` | Pixel normalization (scale, mean, std) |
| `image.colorspace` | RGB ↔ BGR, grayscale conversion |
| `image.layout` | HWC ↔ CHW memory layout conversion |
| `image.pad` | Pad to target size |
| `image.crop` | Extract region of interest |
| `image.draw_bbox` | Draw bounding boxes (visualization) |

### Detection — feature: `vision`

| Op | Description |
|----|-------------|
| `detection.split` | Split raw ONNX output into boxes + scores |
| `detection.confidence_filter` | Filter by confidence threshold |
| `detection.nms` | Non-maximum suppression |
| `detection.soft_nms` | Soft-NMS (Gaussian / Linear decay) |
| `detection.batched_nms` | Per-class NMS (torchvision offset trick) |
| `detection.xywh_to_xyxy` | Center → corner box format |
| `detection.xyxy_to_xywh` | Corner → center box format |
| `detection.format` | Format output as JSON / COCO |

### Audio — feature: `audio`

| Op | Description |
|----|-------------|
| `audio.decode` | WAV/MP3/FLAC/OGG → mono f32 PCM with resample |
| `mel.spectrogram` | Log-mel spectrogram (Whisper-compatible: 80 bins, 25ms window, configurable n_fft/hop/n_mels) |

### Tokenizer — feature: `tokenizer`

| Op | Description |
|----|-------------|
| `tokenizer.encode` | Text → token IDs + attention mask (HuggingFace, batch support, offset tracking) |
| `tokenizer.decode` | Token IDs → text (batch support, skip_special_tokens) |

### Text — feature: `text`

| Op | Description |
|----|-------------|
| `text.lower` | Lowercase |
| `text.upper` | Uppercase |
| `text.trim` | Strip whitespace |
| `text.truncate` | Truncate to N chars |
| `text.replace` | String replacement |
| `text.regex_replace` | Regex replacement |
| `text.regex_extract` | Extract regex matches |
| `text.regex_match` | Boolean regex match |
| `text.split` | Split by separator |
| `text.concat` | Concatenate fields |

### Speech — feature: `sherpa`

| Op | Description |
|----|-------------|
| `sherpa.transcribe` | ASR — transducer / whisper / paraformer / sensevoice / nemo |
| `sherpa.vad` | Voice Activity Detection (Silero VAD) |
| `sherpa.tts` | Text-to-speech (VITS / Kokoro, configurable speed + speaker) |

### Sink — feature: `sink`

| Op | Description |
|----|-------------|
| `sink.qdrant` | Push embedding to Qdrant |
| `sink.weaviate` | Push embedding to Weaviate |
| `sink.chromadb` | Push embedding to ChromaDB |
| `sink.postgres` | Push embedding to pgvector / Supabase |

### Advanced

| Op | Feature | Description |
|----|---------|-------------|
| `generate.greedy` | `onnx` | Autoregressive greedy decoding with KV-cache |
| `wasm.run` | `wasm` | Execute WASM module in sandboxed runtime |
| `pipeline.run` | (core) | Invoke sub-pipeline as a step |

---

## Error Codes

Axon uses typed error codes for programmatic error handling.

| Code | Kind | Description |
|------|------|-------------|
| `AX001` | Config | Bad configuration or operation spec |
| `AX002` | Shape | Tensor shape mismatch or invalid dimensions |
| `AX003` | Input | Missing or invalid input data |
| `AX004` | Runtime | Runtime execution failure |
| `AX005` | Unsupported | Operation not available or feature not compiled |
| `AX006` | IO | File or network I/O error |
| `AX007` | ResourceLimit | Memory, fuel, or timeout exceeded |
| `AX008` | Model | Model loading or inference failure |
| `AX009` | DataFormat | Encoding or format error |

```rust
match result {
    Err(e) if e.kind() == ErrorKind::Shape => { /* handle shape mismatch */ }
    Err(e) => eprintln!("{e}"),  // prints: [AX002] tensor: shape mismatch [1,3,224,224] vs [1,3,640,640]
    Ok(output) => { /* ... */ }
}
```

---

## Feature Flags

| Feature | Dependencies | Description |
|---------|-------------|-------------|
| `onnx` | ort, ndarray, half | ONNX model inference + native tensor ops |
| `vision` | image, fast_image_resize, ndarray | Image decode/resize/normalize + detection post-processing |
| `audio` | symphonia, rubato, rustfft, num-complex | Audio decode + resample + mel spectrogram |
| `tokenizer` | tokenizers | HuggingFace tokenizer encode/decode |
| `text` | regex | Text processing (regex, lower, upper, split, concat) |
| `sherpa` | sherpa-onnx | Sherpa-ONNX speech engine (ASR, VAD, TTS) |
| `wasm` | wasmtime, anyhow | WASM sandbox runtime with fuel/memory limits |
| `sink` | reqwest | Vector DB sinks (Qdrant, Weaviate, ChromaDB, pgvector) |
| `cli` | clap, tracing-subscriber, reqwest | CLI interface (run, bench, init, download, info, kernels) |
| `serve` | axum, tokio, futures, multer, async-stream | HTTP REST server with SSE, batching, versioning |
| `grpc` | tonic, prost, tokio-stream | gRPC server (KServe V2 protocol, dual-port) |
| `otel` | opentelemetry, opentelemetry-otlp, tracing-opentelemetry | OpenTelemetry OTLP tracing |
| `coreml` | ort/coreml | Apple CoreML / Neural Engine GPU |
| `cuda` | ort/cuda | NVIDIA CUDA GPU |
| `tensorrt` | ort/tensorrt | NVIDIA TensorRT optimized inference |
| `directml` | ort/directml | DirectX 12 GPU (Windows) |

### GPU Acceleration

```toml
[model]
name = "yolov8n"
file = "models/yolov8n.onnx"
device = "coreml"   # or "cuda", "tensorrt", "directml"
```

```bash
cargo build -p axon --features "onnx,vision,cli,coreml" --release   # macOS
cargo build -p axon --features "onnx,vision,cli,cuda" --release     # Linux + NVIDIA
```

---

## Architecture

```
                       manifest.toml
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
      ┌─────────┐    ┌──────────┐    ┌──────────┐
      │   Pre   │ →  │  Model   │ →  │   Post   │
      │  Steps  │    │  (ONNX)  │    │   Steps  │
      └─────────┘    └──────────┘    └──────────┘
           │              │               │
     ┌─────┴──────────────┴───────────────┴──────┐
     │              KernelRegistry               │
     │     ComputeKernel trait (Send + Sync)     │
     ├───────┬───────┬───────┬───────┬───────────┤
     │Tensor │Image  │Audio  │Sherpa │WASM  Text │
     │       │Detect │Mel    │  ASR  │Sink  Gen  │
     └───────┴───────┴───────┴───────┴───────────┘
                            │
     ┌──────────────────────┴───────────────────────┐
     │              BlobStore (BLAKE3)               │
     │    moka LRU cache ←→ optional disk tier       │
     └───────────────────────────────────────────────┘
                            │
     ┌──────────────────────┴───────────────────────┐
     │              Serve Layer                      │
     │  HTTP (axum)  │  gRPC (tonic)  │  SSE stream  │
     │  /metrics     │  batching      │  versioning   │
     │  multipart    │  KServe V2     │  model pool   │
     ├───────────────┴────────────────┴──────────────┤
     │           Observability                       │
     │  Prometheus counters  │  OpenTelemetry OTLP   │
     └───────────────────────────────────────────────┘
```

---

## Testing & Benchmarks

```bash
# ── Tests ─────────────────────────────────────────────────
# Full test suite (170+ unit + integration + property tests)
cargo test -p axon --features "onnx,vision,audio,tokenizer"

# Serve tests (REST, SSE, metrics, versioning, batching — 21 tests)
cargo test -p axon --features serve --lib -- serve

# Audio integration tests (real WAV files)
cargo test -p axon --features audio -- audio

# ── Benchmarks (criterion) ────────────────────────────────
cargo bench -p axon --features onnx --bench tensor_ops       # matmul, softmax, mean_pool, gather
cargo bench -p axon --features vision --bench image_ops      # decode, resize, normalize
cargo bench -p axon --features vision --bench detection_ops  # NMS, confidence filter
cargo bench -p axon --features audio --bench audio_ops       # decode, resample, mel spectrogram

# ── Fuzzing (4 targets, requires nightly) ─────────────────
cd crates/axon
cargo +nightly fuzz run fuzz_tensor -- -max_total_time=60
cargo +nightly fuzz run fuzz_image -- -max_total_time=60
cargo +nightly fuzz run fuzz_detection -- -max_total_time=60
cargo +nightly fuzz run fuzz_error_classifier -- -max_total_time=60
```

---

## License

Apache-2.0 — See [LICENSE](LICENSE) for details.
