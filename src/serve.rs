//! HTTP server for Axon ML pipelines (`axon serve`).
//!
//! Loads one or more pipelines at startup and serves them via REST API.
//! Models are loaded once (warm) — no cold start per request.
//!
//! ## Endpoints
//!
//! - `POST /{name}` — run inference (accepts file upload or raw body)
//! - `POST /{name}/stream` — SSE streaming inference with step-level progress
//! - `GET  /{name}/info` — pipeline metadata
//! - `GET  /metrics` — Prometheus-compatible metrics
//! - `GET  /health` — server health check
//! - `GET  /pipelines` — list loaded pipelines
//! - `POST /{name}/v/{version}` — run inference on a specific model version
//! - `GET  /{name}/versions` — list available versions for a pipeline
//!
//! ## Example
//!
//! ```bash
//! axon serve \
//!   --pipeline yolo=examples/yolov8-detection.toml \
//!   --pipeline whisper=examples/whisper-transcription.toml \
//!   --port 8080
//!
//! curl -X POST http://localhost:8080/yolo -F "input=@photo.jpg"
//! curl -X POST http://localhost:8080/whisper --data-binary @audio.wav -H "Content-Type: audio/wav"
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use futures::stream::Stream;
use serde_json::json;
use tokio::net::TcpListener;
use tracing::info_span;

use crate::batch::{BatchConfig, BatchDispatcher};
use crate::pipeline::Pipeline;

// ── Metrics (lightweight Prometheus-compatible) ───────────────

use std::sync::atomic::{AtomicU64, Ordering};

/// Global inference metrics (lock-free atomics).
#[derive(Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub requests_ok: AtomicU64,
    pub requests_err: AtomicU64,
    /// Sum of latencies in microseconds (for computing average).
    pub latency_sum_us: AtomicU64,
}

impl Metrics {
    pub fn record_ok(&self, latency: std::time::Duration) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_ok.fetch_add(1, Ordering::Relaxed);
        self.latency_sum_us
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
    }

    pub fn record_err(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_err.fetch_add(1, Ordering::Relaxed);
    }

    /// Render as Prometheus text exposition format.
    pub fn to_prometheus(&self, uptime_secs: u64) -> String {
        let total = self.requests_total.load(Ordering::Relaxed);
        let ok = self.requests_ok.load(Ordering::Relaxed);
        let err = self.requests_err.load(Ordering::Relaxed);
        let lat_sum_us = self.latency_sum_us.load(Ordering::Relaxed);
        let avg_ms = if ok > 0 {
            lat_sum_us as f64 / ok as f64 / 1000.0
        } else {
            0.0
        };

        format!(
            "# HELP axon_requests_total Total inference requests.\n\
             # TYPE axon_requests_total counter\n\
             axon_requests_total {total}\n\
             # HELP axon_requests_ok Successful inference requests.\n\
             # TYPE axon_requests_ok counter\n\
             axon_requests_ok {ok}\n\
             # HELP axon_requests_errors Failed inference requests.\n\
             # TYPE axon_requests_errors counter\n\
             axon_requests_errors {err}\n\
             # HELP axon_latency_avg_ms Average inference latency (ms).\n\
             # TYPE axon_latency_avg_ms gauge\n\
             axon_latency_avg_ms {avg_ms:.2}\n\
             # HELP axon_uptime_seconds Server uptime.\n\
             # TYPE axon_uptime_seconds gauge\n\
             axon_uptime_seconds {uptime_secs}\n"
        )
    }
}

static METRICS: std::sync::LazyLock<Metrics> = std::sync::LazyLock::new(Metrics::default);

// ── App State ──────────────────────────────────────────────────

/// Shared application state holding all loaded pipelines.
#[derive(Clone)]
pub struct AppState {
    pub pipelines: Arc<HashMap<String, Arc<Pipeline>>>,
    /// Optional batch dispatchers (one per pipeline, when batching is enabled).
    pub batchers: Option<Arc<HashMap<String, Arc<BatchDispatcher>>>>,
    /// Versioned pipelines: name → {version → Pipeline}.
    /// Populated when `--pipeline name@version=path` syntax is used.
    pub versions: Arc<HashMap<String, HashMap<String, Arc<Pipeline>>>>,
    pub started_at: Instant,
}

// ── Error type ─────────────────────────────────────────────────

/// Serve-specific error for pipeline loading.
#[derive(Debug)]
pub enum ServeError {
    /// Pipeline failed to load.
    LoadFailed { name: String, error: String },
    /// No pipelines specified.
    NoPipelines,
    /// Bind/listen error.
    Bind(String),
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoadFailed { name, error } => {
                write!(f, "failed to load pipeline '{name}': {error}")
            }
            Self::NoPipelines => write!(f, "no pipelines specified"),
            Self::Bind(e) => write!(f, "bind error: {e}"),
        }
    }
}

impl std::error::Error for ServeError {}

// ── Server builder ─────────────────────────────────────────────

/// Configuration for the axon serve command.
pub struct ServeConfig {
    /// Named pipelines: name → manifest path.
    pub pipelines: Vec<(String, String)>,
    /// Port to listen on.
    pub port: u16,
    /// Host to bind to.
    pub host: String,
    /// Max request body size in bytes (default: 50MB).
    pub max_body_size: usize,
    /// Dynamic batching config. `None` disables batching (direct execution).
    pub batch: Option<BatchConfig>,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            pipelines: Vec::new(),
            port: 8080,
            host: "0.0.0.0".to_string(),
            max_body_size: 50 * 1024 * 1024,
            batch: None,
        }
    }
}

/// Load all pipelines and build the axum router.
///
/// This is separated from `run_server` so tests can build the router
/// without actually binding to a port.
pub fn build_router(config: &ServeConfig) -> Result<(Router, AppState), ServeError> {
    if config.pipelines.is_empty() {
        return Err(ServeError::NoPipelines);
    }

    let mut pipelines = HashMap::new();
    let mut versions: HashMap<String, HashMap<String, Arc<Pipeline>>> = HashMap::new();

    for (raw_name, manifest_path) in &config.pipelines {
        // Parse "name@version" syntax. Unversioned → version "latest".
        let (base_name, version) = if let Some((n, v)) = raw_name.split_once('@') {
            (n.to_string(), v.to_string())
        } else {
            (raw_name.clone(), "latest".to_string())
        };

        let pipeline = Pipeline::load(manifest_path)
            .map_err(|e| ServeError::LoadFailed {
                name: raw_name.clone(),
                error: e.to_string(),
            })?;

        // Validate all kernels are available.
        if let Err(missing) = pipeline.validate() {
            return Err(ServeError::LoadFailed {
                name: raw_name.clone(),
                error: format!("missing kernels: {}", missing.join(", ")),
            });
        }

        let pipeline = Arc::new(pipeline);

        tracing::info!(
            pipeline = %base_name,
            version = %version,
            model = %pipeline.manifest().model.name,
            "loaded pipeline"
        );

        // Register in version map.
        versions
            .entry(base_name.clone())
            .or_default()
            .insert(version, Arc::clone(&pipeline));

        // The last-loaded version (or "latest") wins as the default.
        pipelines.insert(base_name, pipeline);
    }

    let pipelines = Arc::new(pipelines);
    let versions = Arc::new(versions);

    // Build batch dispatchers if batching is enabled.
    let batchers = config.batch.as_ref().map(|batch_config| {
        let mut dispatchers = HashMap::new();
        for (name, pipeline) in pipelines.iter() {
            let dispatcher = BatchDispatcher::new(
                Arc::clone(pipeline),
                name.clone(),
                batch_config.clone(),
            );
            dispatchers.insert(name.clone(), Arc::new(dispatcher));
        }
        Arc::new(dispatchers)
    });

    let state = AppState {
        pipelines,
        batchers,
        versions,
        started_at: Instant::now(),
    };

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/pipelines", get(pipelines_handler))
        .route("/{name}", post(inference_handler))
        .route("/{name}/stream", post(stream_handler))
        .route("/{name}/info", get(info_handler))
        .route("/{name}/v/{version}", post(versioned_inference_handler))
        .route("/{name}/versions", get(versions_handler))
        .layer(DefaultBodyLimit::max(config.max_body_size))
        .with_state(state.clone());

    Ok((app, state))
}

/// Start the server (blocking on the async runtime).
pub async fn run_server(config: ServeConfig) -> Result<(), ServeError> {
    let (app, state) = build_router(&config)?;

    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|e| ServeError::Bind(format!("{addr}: {e}")))?;

    let pipeline_count = state.pipelines.len();
    let names: Vec<&str> = state.pipelines.keys().map(|s| s.as_str()).collect();
    tracing::info!(
        addr = %addr,
        pipelines = pipeline_count,
        names = ?names,
        "axon serve started"
    );

    // Print human-readable startup message.
    eprintln!("axon serve listening on {addr}");
    eprintln!("pipelines: {}", names.join(", "));
    if state.batchers.is_some() {
        eprintln!("batching: enabled");
    }
    for name in &names {
        eprintln!("  POST http://{addr}/{name}");
    }
    eprintln!();

    axum::serve(listener, app)
        .await
        .map_err(|e| ServeError::Bind(e.to_string()))
}

// ── Handlers ───────────────────────────────────────────────────

/// GET /health — server health check.
async fn health_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let uptime_secs = state.started_at.elapsed().as_secs();
    let names: Vec<&str> = state.pipelines.keys().map(|s| s.as_str()).collect();
    Json(json!({
        "status": "ok",
        "uptime_seconds": uptime_secs,
        "pipelines": names,
        "batching": state.batchers.is_some(),
    }))
}

/// GET /metrics — Prometheus-compatible metrics endpoint.
async fn metrics_handler(State(state): State<AppState>) -> Response {
    let uptime_secs = state.started_at.elapsed().as_secs();
    let body = METRICS.to_prometheus(uptime_secs);
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
        .into_response()
}

/// GET /pipelines — list all loaded pipelines with metadata.
async fn pipelines_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mut entries = Vec::new();
    for (name, pipeline) in state.pipelines.iter() {
        let manifest = pipeline.manifest();
        entries.push(json!({
            "name": name,
            "model": manifest.model.name,
            "file": manifest.model.file,
            "device": manifest.model.device,
            "kernels": pipeline.required_kernels(),
        }));
    }
    Json(json!({ "pipelines": entries }))
}

/// GET /{name}/info — pipeline details.
async fn info_handler(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Response {
    let Some(pipeline) = state.pipelines.get(&name) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("pipeline '{name}' not found") })),
        )
            .into_response();
    };

    let manifest = pipeline.manifest();
    let pre_steps: Vec<&str> = manifest
        .pre
        .as_ref()
        .map(|p| p.steps.iter().map(|s| s.op.as_str()).collect())
        .unwrap_or_default();
    let post_steps: Vec<&str> = manifest
        .post
        .as_ref()
        .map(|p| p.steps.iter().map(|s| s.op.as_str()).collect())
        .unwrap_or_default();

    Json(json!({
        "name": name,
        "model": {
            "name": manifest.model.name,
            "file": manifest.model.file,
            "device": manifest.model.device,
        },
        "pre_steps": pre_steps,
        "post_steps": post_steps,
        "kernels": pipeline.required_kernels(),
    }))
    .into_response()
}

/// POST /{name} — run inference.
///
/// Accepts either:
/// - Multipart form with field named "input" (file upload)
/// - Raw body with Content-Type header
async fn inference_handler(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(pipeline) = state.pipelines.get(&name) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("pipeline '{name}' not found") })),
        )
            .into_response();
    };

    // Determine content type.
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream");

    // Handle multipart: extract the "input" field.
    if content_type.starts_with("multipart/form-data") {
        let batcher = state
            .batchers
            .as_ref()
            .and_then(|b| b.get(&name).cloned());
        return handle_multipart(pipeline.clone(), batcher, headers, body).await;
    }

    // Raw body inference.
    let input_bytes = body.to_vec();
    if input_bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "empty request body" })),
        )
            .into_response();
    }

    let body_size = input_bytes.len();
    let pipeline_name = name.clone();

    // Route through batch dispatcher if batching is enabled.
    if let Some(ref batchers) = state.batchers {
        if let Some(batcher) = batchers.get(&name) {
            return run_pipeline_batched(batcher, input_bytes, content_type.to_string(), body_size).await;
        }
    }

    // Direct execution (no batching).
    run_pipeline(pipeline.clone(), input_bytes, content_type.to_string(), pipeline_name, body_size).await
}

/// Handle multipart form upload — extract the "input" field.
async fn handle_multipart(
    pipeline: Arc<Pipeline>,
    batcher: Option<Arc<BatchDispatcher>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Re-parse the multipart body.
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let boundary = content_type
        .split("boundary=")
        .nth(1)
        .unwrap_or("")
        .to_string();

    if boundary.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "multipart boundary not found" })),
        )
            .into_response();
    }

    let stream = futures::stream::once(async move { Ok::<_, std::io::Error>(body) });
    let mut multipart = multer::Multipart::new(stream, boundary);

    while let Ok(Some(field)) = multipart.next_field().await {
        let field_name = field.name().unwrap_or("").to_string();
        if field_name != "input" {
            continue;
        }

        let file_content_type = field
            .content_type()
            .map(|m| m.to_string())
            .unwrap_or_else(|| {
                // Guess from filename.
                field
                    .file_name()
                    .and_then(|f| guess_content_type_from_ext(f))
                    .unwrap_or("application/octet-stream".to_string())
            });

        let data = match field.bytes().await {
            Ok(b) => b.to_vec(),
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": format!("failed to read field: {e}") })),
                )
                    .into_response();
            }
        };

        if data.is_empty() {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "empty input field" })),
            )
                .into_response();
        }

        let body_size = data.len();
        if let Some(ref batcher) = batcher {
            return run_pipeline_batched(batcher, data, file_content_type, body_size).await;
        }
        return run_pipeline(pipeline, data, file_content_type, "multipart".to_string(), body_size).await;
    }

    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "error": "multipart form must contain an 'input' field" })),
    )
        .into_response()
}

/// POST /{name}/stream — run inference with Server-Sent Events progress.
///
/// Returns an SSE stream with events:
/// - `step`: emitted after each pipeline step completes
/// - `result`: final inference result (JSON or blob metadata)
/// - `error`: if the pipeline fails
///
/// Example client:
/// ```bash
/// curl -N -X POST http://localhost:8080/whisper/stream \
///   --data-binary @audio.wav -H "Content-Type: audio/wav"
/// ```
///
/// Event format:
/// ```text
/// event: step
/// data: {"phase":"pre","step":0,"op":"audio.decode","latency_ms":42.1}
///
/// event: step
/// data: {"phase":"model","step":0,"op":"onnx (cpu)","latency_ms":312.5}
///
/// event: result
/// data: {"result":{...},"total_latency_ms":420.3}
/// ```
async fn stream_handler(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(pipeline) = state.pipelines.get(&name).cloned() else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("pipeline '{name}' not found") })),
        )
            .into_response();
    };

    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_string();

    let input_bytes = body.to_vec();
    if input_bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "empty request body" })),
        )
            .into_response();
    }

    let stream = make_sse_stream(pipeline, input_bytes, content_type, name);
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

/// Create an SSE stream that emits progress events during pipeline execution.
fn make_sse_stream(
    pipeline: Arc<Pipeline>,
    input: Vec<u8>,
    content_type: String,
    pipeline_name: String,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Event>(32);

    tokio::task::spawn_blocking(move || {
        let start = Instant::now();

        // Send start event.
        let _ = tx.blocking_send(
            Event::default()
                .event("start")
                .data(json!({"pipeline": pipeline_name}).to_string()),
        );

        // Run pipeline with progress callback.
        let result = pipeline.run_with_progress(&input, &content_type, |progress| {
            let event = Event::default()
                .event("step")
                .data(
                    json!({
                        "phase": progress.phase,
                        "step": progress.step,
                        "op": progress.op,
                        "latency_ms": progress.duration.as_secs_f64() * 1000.0,
                    })
                    .to_string(),
                );
            let _ = tx.blocking_send(event);
        });

        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Send result or error event.
        let final_event = match result {
            Ok(output) => {
                let data = match output {
                    crate::KernelOutput::Json(value) => {
                        json!({
                            "result": value,
                            "total_latency_ms": format!("{total_ms:.1}"),
                        })
                    }
                    crate::KernelOutput::Blob {
                        data,
                        content_type,
                        shape,
                    } => {
                        json!({
                            "result": {
                                "_type": "blob",
                                "size": data.len(),
                                "content_type": content_type,
                                "shape": shape,
                            },
                            "total_latency_ms": format!("{total_ms:.1}"),
                        })
                    }
                };
                Event::default().event("result").data(data.to_string())
            }
            Err(err) => Event::default().event("error").data(
                json!({
                    "error": err.to_string(),
                    "total_latency_ms": format!("{total_ms:.1}"),
                })
                .to_string(),
            ),
        };
        let _ = tx.blocking_send(final_event);
    });

    // Convert mpsc receiver to a Stream.
    async_stream::stream! {
        while let Some(event) = rx.recv().await {
            yield Ok(event);
        }
    }
}

/// POST /{name}/v/{version} — run inference on a specific pipeline version.
async fn versioned_inference_handler(
    State(state): State<AppState>,
    axum::extract::Path((name, version)): axum::extract::Path<(String, String)>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Look up the specific version.
    let pipeline = state
        .versions
        .get(&name)
        .and_then(|v| v.get(&version))
        .cloned();

    let Some(pipeline) = pipeline else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("pipeline '{name}' version '{version}' not found") })),
        )
            .into_response();
    };

    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream");

    let input_bytes = body.to_vec();
    if input_bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "empty request body" })),
        )
            .into_response();
    }

    let body_size = input_bytes.len();
    let pipeline_name = format!("{name}@{version}");

    run_pipeline(pipeline, input_bytes, content_type.to_string(), pipeline_name, body_size).await
}

/// GET /{name}/versions — list available versions for a pipeline.
async fn versions_handler(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Response {
    let Some(version_map) = state.versions.get(&name) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("pipeline '{name}' not found") })),
        )
            .into_response();
    };

    let versions: Vec<&str> = version_map.keys().map(|s| s.as_str()).collect();
    Json(json!({
        "name": name,
        "versions": versions,
    }))
    .into_response()
}

/// Run the pipeline on a blocking thread and return the response.
async fn run_pipeline(
    pipeline: Arc<Pipeline>,
    input: Vec<u8>,
    content_type: String,
    pipeline_name: String,
    body_size: usize,
) -> Response {
    let start = Instant::now();

    // Pipeline::run is synchronous — run on blocking thread pool.
    // The span is created and entered on the blocking thread (sync context),
    // so the Entered guard is safe (no Send requirement).
    let result = tokio::task::spawn_blocking(move || {
        let _span = info_span!(
            "axon_request",
            pipeline = %pipeline_name,
            body_size,
            content_type = %content_type,
        )
        .entered();
        pipeline.run(&input, &content_type)
    })
    .await;

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

    match result {
        Ok(Ok(output)) => {
            METRICS.record_ok(elapsed);
            match output {
                crate::KernelOutput::Json(value) => {
                    let resp = json!({
                        "result": value,
                        "latency_ms": format!("{elapsed_ms:.1}"),
                    });
                    Json(resp).into_response()
                }
                crate::KernelOutput::Blob {
                    data,
                    content_type,
                    shape,
                } => {
                    let mut headers = HeaderMap::new();
                    headers.insert(
                        "content-type",
                        content_type.parse().unwrap_or_else(|_| {
                            "application/octet-stream".parse().unwrap()
                        }),
                    );
                    headers.insert(
                        "x-axon-latency-ms",
                        format!("{elapsed_ms:.1}").parse().unwrap(),
                    );
                    if let Some(ref s) = shape {
                        let shape_str: Vec<String> = s.iter().map(|d| d.to_string()).collect();
                        if let Ok(v) = shape_str.join(",").parse() {
                            headers.insert("x-axon-shape", v);
                        }
                    }
                    (StatusCode::OK, headers, data).into_response()
                }
            }
        }
        Ok(Err(pipeline_err)) => {
            METRICS.record_err();
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": pipeline_err.to_string(),
                    "latency_ms": format!("{elapsed_ms:.1}"),
                })),
            )
                .into_response()
        }
        Err(join_err) => {
            METRICS.record_err();
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": format!("inference task panicked: {join_err}"),
                })),
            )
                .into_response()
        }
    }
}

/// Run the pipeline through the batch dispatcher.
async fn run_pipeline_batched(
    batcher: &BatchDispatcher,
    input: Vec<u8>,
    content_type: String,
    _body_size: usize,
) -> Response {
    let start = Instant::now();

    let result = batcher.submit(input, content_type).await;
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

    match result {
        Ok(output) => {
            METRICS.record_ok(elapsed);
            match output {
            crate::KernelOutput::Json(value) => {
                let resp = json!({
                    "result": value,
                    "latency_ms": format!("{elapsed_ms:.1}"),
                    "batched": true,
                });
                Json(resp).into_response()
            }
            crate::KernelOutput::Blob {
                data,
                content_type,
                shape,
            } => {
                let mut headers = HeaderMap::new();
                headers.insert(
                    "content-type",
                    content_type.parse().unwrap_or_else(|_| {
                        "application/octet-stream".parse().unwrap()
                    }),
                );
                headers.insert(
                    "x-axon-latency-ms",
                    format!("{elapsed_ms:.1}").parse().unwrap(),
                );
                headers.insert("x-axon-batched", "true".parse().unwrap());
                if let Some(ref s) = shape {
                    let shape_str: Vec<String> = s.iter().map(|d| d.to_string()).collect();
                    if let Ok(v) = shape_str.join(",").parse() {
                        headers.insert("x-axon-shape", v);
                    }
                }
                (StatusCode::OK, headers, data).into_response()
            }
            }
        }
        Err(pipeline_err) => {
            METRICS.record_err();
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": pipeline_err.to_string(),
                    "latency_ms": format!("{elapsed_ms:.1}"),
                })),
            )
                .into_response()
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Guess content type from file extension.
fn guess_content_type_from_ext(filename: &str) -> Option<String> {
    let ext = filename.rsplit('.').next()?;
    let ct = match ext {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "ogg" => "audio/ogg",
        "json" => "application/json",
        "txt" => "text/plain",
        _ => return None,
    };
    Some(ct.to_string())
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// Build a test router with a mock pipeline (no real ONNX model needed).
    fn test_app() -> Router {
        // Create a minimal pipeline with mock onnx kernel.
        let manifest = crate::Manifest::from_toml(
            r#"
[model]
name = "test-model"
file = "model.onnx"
"#,
        )
        .unwrap();

        let mut reg = crate::KernelRegistry::new();
        reg.register(Arc::new(MockOnnxKernel));

        let pipeline = Pipeline::new(
            manifest,
            reg,
            std::path::PathBuf::from("."),
        );

        let pipeline = Arc::new(pipeline);
        let mut pipelines = HashMap::new();
        pipelines.insert("test".to_string(), Arc::clone(&pipeline));

        let mut versions = HashMap::new();
        let mut ver_map = HashMap::new();
        ver_map.insert("latest".to_string(), Arc::clone(&pipeline));
        versions.insert("test".to_string(), ver_map);

        let state = AppState {
            pipelines: Arc::new(pipelines),
            batchers: None,
            versions: Arc::new(versions),
            started_at: Instant::now(),
        };

        Router::new()
            .route("/health", get(health_handler))
            .route("/metrics", get(metrics_handler))
            .route("/pipelines", get(pipelines_handler))
            .route("/{name}", post(inference_handler))
            .route("/{name}/stream", post(stream_handler))
            .route("/{name}/info", get(info_handler))
            .route("/{name}/v/{version}", post(versioned_inference_handler))
            .route("/{name}/versions", get(versions_handler))
            .with_state(state)
    }

    /// Mock ONNX kernel that returns a simple JSON prediction.
    struct MockOnnxKernel;

    impl crate::ComputeKernel for MockOnnxKernel {
        fn name(&self) -> &str {
            "onnx"
        }

        fn execute(
            &self,
            _input: crate::KernelInput,
            operations: serde_json::Value,
        ) -> Result<crate::KernelOutput, crate::AxonError> {
            let model = operations
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            Ok(crate::KernelOutput::Json(json!({
                "model": model,
                "prediction": [0.9, 0.05, 0.05],
            })))
        }
    }

    async fn body_to_json(body: Body) -> serde_json::Value {
        let bytes = body.collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["status"], "ok");
        assert!(json["pipelines"].as_array().unwrap().contains(&json!("test")));
    }

    #[tokio::test]
    async fn test_pipelines_endpoint() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/pipelines")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        let pipelines = json["pipelines"].as_array().unwrap();
        assert_eq!(pipelines.len(), 1);
        assert_eq!(pipelines[0]["name"], "test");
        assert_eq!(pipelines[0]["model"], "test-model");
    }

    #[tokio::test]
    async fn test_info_endpoint() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/test/info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["name"], "test");
        assert_eq!(json["model"]["name"], "test-model");
    }

    #[tokio::test]
    async fn test_info_not_found() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/nonexistent/info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_inference_raw_body() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/test")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(vec![1, 2, 3, 4]))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert!(json["result"]["prediction"].is_array());
        assert!(json["latency_ms"].is_string());
    }

    #[tokio::test]
    async fn test_inference_empty_body() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/test")
                    .header("content-type", "application/octet-stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["error"], "empty request body");
    }

    #[tokio::test]
    async fn test_inference_pipeline_not_found() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/nonexistent")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(vec![1, 2, 3]))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_build_router_no_pipelines() {
        let config = ServeConfig::default();
        let result = build_router(&config);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_build_router_bad_manifest() {
        let config = ServeConfig {
            pipelines: vec![("bad".to_string(), "/nonexistent/path.toml".to_string())],
            ..Default::default()
        };
        let result = build_router(&config);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_inference_text_input() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/test")
                    .header("content-type", "text/plain")
                    .body(Body::from("hello world"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert!(json["result"]["prediction"].is_array());
    }

    #[tokio::test]
    async fn test_inference_json_input() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/test")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"text": "hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert!(json["result"]["prediction"].is_array());
    }

    /// Build a test router with batching enabled.
    fn test_app_batched() -> Router {
        let manifest = crate::Manifest::from_toml(
            r#"
[model]
name = "test-model"
file = "model.onnx"
"#,
        )
        .unwrap();

        let mut reg = crate::KernelRegistry::new();
        reg.register(Arc::new(MockOnnxKernel));

        let pipeline = Arc::new(Pipeline::new(
            manifest,
            reg,
            std::path::PathBuf::from("."),
        ));

        let mut pipelines = HashMap::new();
        pipelines.insert("test".to_string(), pipeline.clone());
        let pipelines = Arc::new(pipelines);

        let batch_config = BatchConfig {
            max_batch_size: 4,
            timeout: std::time::Duration::from_millis(50),
        };

        let mut dispatchers = HashMap::new();
        dispatchers.insert(
            "test".to_string(),
            Arc::new(BatchDispatcher::new(pipeline, "test".to_string(), batch_config)),
        );

        let state = AppState {
            pipelines,
            batchers: Some(Arc::new(dispatchers)),
            versions: Arc::new(HashMap::new()),
            started_at: Instant::now(),
        };

        Router::new()
            .route("/health", get(health_handler))
            .route("/pipelines", get(pipelines_handler))
            .route("/{name}", post(inference_handler))
            .route("/{name}/info", get(info_handler))
            .with_state(state)
    }

    #[tokio::test]
    async fn test_batched_inference() {
        let app = test_app_batched();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/test")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(vec![1, 2, 3, 4]))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert!(json["result"]["prediction"].is_array());
        assert_eq!(json["batched"], true);
    }

    #[tokio::test]
    async fn test_batched_health_reports_batching() {
        let app = test_app_batched();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["batching"], true);
    }

    #[tokio::test]
    async fn test_stream_endpoint() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/test/stream")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(vec![1, 2, 3, 4]))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        // SSE responses have text/event-stream content type.
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(ct.contains("text/event-stream"), "expected SSE content type, got: {ct}");

        // Read the full SSE body and verify it contains expected events.
        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8_lossy(&body_bytes);

        assert!(body_str.contains("event: start"), "should have start event");
        assert!(body_str.contains("event: result"), "should have result event");
    }

    #[tokio::test]
    async fn test_stream_not_found() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/nonexistent/stream")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(vec![1, 2, 3]))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_versioned_inference() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/test/v/latest")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(vec![1, 2, 3]))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert!(json["result"]["prediction"].is_array());
    }

    #[tokio::test]
    async fn test_versioned_inference_not_found() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/test/v/v99")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(vec![1, 2, 3]))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_versions_list() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/test/versions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["name"], "test");
        let versions = json["versions"].as_array().unwrap();
        assert!(versions.contains(&json!("latest")));
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let app = test_app();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(ct.contains("text/plain"), "expected text/plain, got: {ct}");

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8_lossy(&body_bytes);
        assert!(body_str.contains("axon_requests_total"), "should contain requests_total metric");
        assert!(body_str.contains("axon_uptime_seconds"), "should contain uptime metric");
    }

    #[tokio::test]
    async fn test_guess_content_type() {
        assert_eq!(
            guess_content_type_from_ext("photo.jpg"),
            Some("image/jpeg".to_string())
        );
        assert_eq!(
            guess_content_type_from_ext("audio.wav"),
            Some("audio/wav".to_string())
        );
        assert_eq!(
            guess_content_type_from_ext("data.bin"),
            None
        );
    }
}
