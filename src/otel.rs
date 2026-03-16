//! OpenTelemetry observability for Axon pipelines.
//!
//! Provides distributed tracing via OTLP export. When enabled, every pipeline
//! execution generates spans visible in Jaeger, Zipkin, Grafana Tempo, etc.
//!
//! ## Span hierarchy
//!
//! ```text
//! axon_request (HTTP handler)
//!   └── pipeline (model=yolov8n, input_size=...)
//!         ├── pre (steps=5)
//!         │   ├── step (op=image.decode)
//!         │   ├── step (op=image.resize)
//!         │   ├── step (op=image.normalize)
//!         │   ├── step (op=image.layout)
//!         │   └── step (op=tensor.unsqueeze)
//!         ├── model (file=yolov8n.onnx, device=coreml)
//!         └── post (steps=5)
//!             ├── step (op=detection.split)
//!             ├── step (op=detection.xywh_to_xyxy)
//!             ├── step (op=detection.confidence_filter)
//!             ├── step (op=detection.nms)
//!             └── step (op=detection.format)
//! ```
//!
//! ## Usage
//!
//! ```bash
//! # Start with OTLP tracing (sends to localhost:4317 by default)
//! axon serve --pipeline yolo=manifest.toml --otel
//!
//! # Custom OTLP endpoint
//! OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317 axon serve --pipeline yolo=manifest.toml --otel
//! ```

use opentelemetry::trace::TracerProvider;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Initialize the tracing subscriber with OpenTelemetry OTLP export.
///
/// This sets up a layered subscriber:
/// 1. `fmt` layer — human-readable logs to stderr (always on)
/// 2. `opentelemetry` layer — OTLP span export (when `--otel` is enabled)
///
/// The OTLP endpoint defaults to `http://localhost:4317` and can be overridden
/// via the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable.
///
/// Returns the tracer provider handle for graceful shutdown.
pub fn init_tracing_with_otel() -> Result<SdkTracerProvider, Box<dyn std::error::Error>> {
    // Build OTLP exporter (uses env var OTEL_EXPORTER_OTLP_ENDPOINT or defaults).
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .build()?;

    // Build tracer provider with batch span processor.
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(opentelemetry_sdk::Resource::builder()
            .with_service_name("axon")
            .build())
        .build();

    let tracer = provider.tracer("axon");

    // Build the OpenTelemetry tracing layer.
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // Build the fmt layer for console output.
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .compact();

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .with(otel_layer)
        .init();

    Ok(provider)
}

/// Initialize tracing with only the fmt layer (no OTel export).
///
/// Used when `--otel` is not specified.
pub fn init_tracing_fmt_only() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

/// Gracefully shut down the tracer provider, flushing remaining spans.
pub fn shutdown_tracing(provider: SdkTracerProvider) {
    if let Err(e) = provider.shutdown() {
        eprintln!("otel: shutdown error: {e}");
    }
}
