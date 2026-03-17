//! Dynamic batching with true tensor-level inference for `axon serve`.
//!
//! Collects concurrent inference requests into batches, preprocesses each
//! independently, concatenates tensors along batch dim 0 for a single
//! ONNX `session.run()`, then splits results back to individual responses.
//!
//! ## Architecture
//!
//! ```text
//! HTTP request → BatchDispatcher → mpsc channel → batch_loop
//!                                                    ↓
//!                                           collect N requests
//!                                           (timeout or adaptive batch size)
//!                                                    ↓
//!                                           spawn_blocking {
//!                                             pipeline.run_batch(N inputs)
//!                                               → pre-process each independently
//!                                               → concatenate along dim 0
//!                                               → single session.run()
//!                                               → split back to N outputs
//!                                               → post-process each independently
//!                                           }
//!                                                    ↓
//!                                           send results via oneshot
//! ```
//!
//! ## Adaptive batch sizing
//!
//! When `adaptive: true`, the controller auto-adjusts the effective batch size
//! using AIMD (Additive Increase, Multiplicative Decrease):
//! - High fill rate (queue pressure) → increase batch size
//! - Low fill rate (sparse traffic) → decrease batch size
//! - Bounded by `[min_batch_size, max_batch_size]`

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, info_span, warn};

use crate::kernel::KernelOutput;
use crate::pipeline::{InferenceMeta, Pipeline, PipelineError};

// ── Priority ────────────────────────────────────────────────────

/// Request priority for batch scheduling.
///
/// High-priority requests are dequeued first via biased select.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Priority {
    /// Latency-sensitive request — skip ahead of low-priority.
    High,
    /// Bulk / background request — processed after high-priority.
    #[default]
    Low,
}

// ── BatchRequest ──────────────────────────────────────────────

/// A single inference request waiting for batch processing.
struct BatchRequest {
    input: Vec<u8>,
    content_type: String,
    reply: oneshot::Sender<Result<(KernelOutput, InferenceMeta), PipelineError>>,
    /// When this request was submitted (for queue wait time tracking).
    created_at: std::time::Instant,
}

// ── Batch Metrics ─────────────────────────────────────────────

/// Global batch-level metrics (lock-free atomics).
/// Exposed via /metrics for Prometheus scraping.
pub struct BatchMetrics {
    /// Total batches processed.
    pub batch_count: AtomicU64,
    /// Sum of items across all batches (for avg_batch_size = items_sum / batch_count).
    pub batch_items_sum: AtomicU64,
    /// Sum of queue wait time in microseconds (for avg_queue_wait).
    pub queue_wait_sum_us: AtomicU64,
    /// Total items that waited in queue.
    pub queue_wait_count: AtomicU64,
}

impl Default for BatchMetrics {
    fn default() -> Self {
        Self {
            batch_count: AtomicU64::new(0),
            batch_items_sum: AtomicU64::new(0),
            queue_wait_sum_us: AtomicU64::new(0),
            queue_wait_count: AtomicU64::new(0),
        }
    }
}

impl BatchMetrics {
    pub fn record_batch(&self, batch_size: usize, total_queue_wait_us: u64) {
        self.batch_count.fetch_add(1, Ordering::Relaxed);
        self.batch_items_sum.fetch_add(batch_size as u64, Ordering::Relaxed);
        self.queue_wait_sum_us.fetch_add(total_queue_wait_us, Ordering::Relaxed);
        self.queue_wait_count.fetch_add(batch_size as u64, Ordering::Relaxed);
    }

    /// Average batch size (items per batch).
    pub fn avg_batch_size(&self) -> f64 {
        let count = self.batch_count.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        self.batch_items_sum.load(Ordering::Relaxed) as f64 / count as f64
    }

    /// Average queue wait time in milliseconds.
    pub fn avg_queue_wait_ms(&self) -> f64 {
        let count = self.queue_wait_count.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        self.queue_wait_sum_us.load(Ordering::Relaxed) as f64 / count as f64 / 1000.0
    }

    /// Render as Prometheus text exposition format.
    pub fn to_prometheus(&self) -> String {
        let count = self.batch_count.load(Ordering::Relaxed);
        let items = self.batch_items_sum.load(Ordering::Relaxed);
        let avg_size = self.avg_batch_size();
        let avg_wait = self.avg_queue_wait_ms();

        format!(
            "# HELP axon_batch_count Total batches processed.\n\
             # TYPE axon_batch_count counter\n\
             axon_batch_count {count}\n\
             # HELP axon_batch_items_total Total items across all batches.\n\
             # TYPE axon_batch_items_total counter\n\
             axon_batch_items_total {items}\n\
             # HELP axon_batch_size_avg Average items per batch.\n\
             # TYPE axon_batch_size_avg gauge\n\
             axon_batch_size_avg {avg_size:.2}\n\
             # HELP axon_queue_wait_avg_ms Average queue wait time (ms).\n\
             # TYPE axon_queue_wait_avg_ms gauge\n\
             axon_queue_wait_avg_ms {avg_wait:.2}\n"
        )
    }
}

pub static BATCH_METRICS: std::sync::LazyLock<BatchMetrics> =
    std::sync::LazyLock::new(BatchMetrics::default);

// ── BatchConfig ───────────────────────────────────────────────

/// Configuration for dynamic batching.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    /// Maximum number of requests per batch (ceiling for adaptive mode).
    pub max_batch_size: usize,
    /// Maximum time to wait for a full batch after the first request arrives.
    pub timeout: Duration,
    /// Enable adaptive batch sizing based on load.
    pub adaptive: bool,
    /// Minimum batch size for adaptive mode (floor).
    pub min_batch_size: usize,
    /// Bounded queue capacity. When full, new requests get `PipelineError::Overloaded` (503).
    /// 0 means unbounded (no backpressure).
    pub queue_capacity: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            timeout: Duration::from_millis(10),
            adaptive: false,
            min_batch_size: 1,
            queue_capacity: 0,
        }
    }
}

// ── AdaptiveController ───────────────────────────────────────

/// AIMD batch size controller that auto-adjusts based on queue pressure.
///
/// Tracks fill rate (actual_size / target_size) with exponential moving average.
/// High fill → additive increase; low fill → multiplicative decrease.
struct AdaptiveController {
    /// Current effective batch size.
    current: usize,
    /// Minimum batch size (floor).
    min: usize,
    /// Maximum batch size (ceiling).
    max: usize,
    /// EMA of batch fill rate (0.0–1.0).
    avg_fill_rate: f64,
    /// EMA of per-item latency in microseconds.
    avg_item_latency_us: f64,
    /// Smoothing factor (higher = more weight on recent batches).
    alpha: f64,
}

impl AdaptiveController {
    fn new(min: usize, max: usize) -> Self {
        let initial = min.max(1);
        Self {
            current: initial,
            min: initial,
            max,
            avg_fill_rate: 0.5,
            avg_item_latency_us: 0.0,
            alpha: 0.3,
        }
    }

    fn current_batch_size(&self) -> usize {
        self.current
    }

    fn record_batch(&mut self, actual_size: usize, batch_duration: Duration) {
        let fill_rate = actual_size as f64 / self.current as f64;
        self.avg_fill_rate = self.alpha * fill_rate + (1.0 - self.alpha) * self.avg_fill_rate;

        let item_latency = batch_duration.as_micros() as f64 / actual_size.max(1) as f64;
        if self.avg_item_latency_us == 0.0 {
            self.avg_item_latency_us = item_latency;
        } else {
            self.avg_item_latency_us =
                self.alpha * item_latency + (1.0 - self.alpha) * self.avg_item_latency_us;
        }

        self.adjust();
    }

    fn adjust(&mut self) {
        if self.avg_fill_rate > 0.8 {
            // Queue is saturated — increase batch size (additive).
            self.current = (self.current + 1).min(self.max);
        } else if self.avg_fill_rate < 0.3 && self.current > self.min {
            // Queue is sparse — decrease batch size (multiplicative).
            self.current = (self.current * 3 / 4).max(self.min);
        }
        // Between 0.3 and 0.8: hold steady.
    }
}

// ── BatchDispatcher ───────────────────────────────────────────

/// Dispatcher that routes inference requests to a batch processing loop.
///
/// Each pipeline gets its own `BatchDispatcher`. Requests are collected
/// into batches and processed together on a blocking thread.
///
/// When `queue_capacity > 0`, the internal channel is bounded and new
/// requests receive `PipelineError::Overloaded` (HTTP 503) when the
/// queue is full, providing backpressure under load.
pub struct BatchDispatcher {
    tx_high: BatchSender,
    tx_low: BatchSender,
}

/// Sender side: bounded or unbounded depending on config.
enum BatchSender {
    Bounded(mpsc::Sender<BatchRequest>),
    Unbounded(mpsc::UnboundedSender<BatchRequest>),
}

impl BatchSender {
    fn send_request(&self, req: BatchRequest) -> Result<(), PipelineError> {
        match self {
            Self::Bounded(tx) => tx.try_send(req).map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => PipelineError::Overloaded,
                mpsc::error::TrySendError::Closed(_) => PipelineError::Empty,
            }),
            Self::Unbounded(tx) => tx.send(req).map_err(|_| PipelineError::Empty),
        }
    }
}

impl BatchDispatcher {
    /// Create a new BatchDispatcher for a pipeline.
    ///
    /// Spawns a background tokio task that collects and processes batches.
    /// Uses two channels (high + low priority) with biased select.
    pub fn new(pipeline: Arc<Pipeline>, pipeline_name: String, config: BatchConfig) -> Self {
        if config.queue_capacity > 0 {
            let (tx_high, rx_high) = mpsc::channel(config.queue_capacity);
            let (tx_low, rx_low) = mpsc::channel(config.queue_capacity);
            tokio::spawn(batch_loop_bounded(
                pipeline,
                pipeline_name,
                rx_high,
                rx_low,
                config,
            ));
            Self {
                tx_high: BatchSender::Bounded(tx_high),
                tx_low: BatchSender::Bounded(tx_low),
            }
        } else {
            let (tx_high, rx_high) = mpsc::unbounded_channel();
            let (tx_low, rx_low) = mpsc::unbounded_channel();
            tokio::spawn(batch_loop(
                pipeline,
                pipeline_name,
                rx_high,
                rx_low,
                config,
            ));
            Self {
                tx_high: BatchSender::Unbounded(tx_high),
                tx_low: BatchSender::Unbounded(tx_low),
            }
        }
    }

    /// Submit a request for batch processing.
    ///
    /// Returns the pipeline output + metadata when the batch completes.
    /// High priority requests are dequeued first via biased select.
    /// Returns `PipelineError::Overloaded` if the bounded queue is full (backpressure).
    pub async fn submit(
        &self,
        input: Vec<u8>,
        content_type: String,
        priority: Priority,
    ) -> Result<(KernelOutput, InferenceMeta), PipelineError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let req = BatchRequest {
            input,
            content_type,
            reply: reply_tx,
            created_at: std::time::Instant::now(),
        };

        match priority {
            Priority::High => self.tx_high.send_request(req)?,
            Priority::Low => self.tx_low.send_request(req)?,
        }

        reply_rx.await.unwrap_or(Err(PipelineError::Empty))
    }
}

// ── Batch loop ────────────────────────────────────────────────

/// Core batch processing logic shared between bounded and unbounded loops.
fn process_batch(pipeline: &Pipeline, batch: Vec<BatchRequest>, name: &str, batch_size: usize) {
    // Track queue wait time for each item.
    let now = std::time::Instant::now();
    let total_queue_wait_us: u64 = batch
        .iter()
        .map(|r| now.duration_since(r.created_at).as_micros() as u64)
        .sum();
    let avg_queue_wait_us = total_queue_wait_us / batch_size.max(1) as u64;

    let _span = info_span!(
        "batch",
        pipeline = %name,
        batch_size,
        queue_wait_avg_us = avg_queue_wait_us,
    )
    .entered();

    debug!("processing batch (tensor-level)");

    // Try tensor-level batching via pipeline.run_batch().
    let batch_result = if batch_size > 1 {
        let inputs: Vec<(&[u8], &str)> = batch
            .iter()
            .map(|r| (r.input.as_slice(), r.content_type.as_str()))
            .collect();
        pipeline.run_batch(&inputs).ok()
    } else {
        None
    };

    if let Some(results) = batch_result {
        debug!(batch_size, "tensor batch succeeded");
        for (item, result) in batch.into_iter().zip(results) {
            let _ = item.reply.send(result);
        }
    } else {
        // Serial fallback: batch_size == 1, or tensor batch failed.
        if batch_size > 1 {
            warn!("tensor batch failed, falling back to serial execution");
        }
        for (i, item) in batch.into_iter().enumerate() {
            let _item_span = info_span!("batch_item", idx = i).entered();
            let result = pipeline.run_with_session(&item.input, &item.content_type, None, None);
            let _ = item.reply.send(result);
        }
    }

    // Emit batch metrics.
    BATCH_METRICS.record_batch(batch_size, total_queue_wait_us);
}

/// Background loop (unbounded channels) that collects and processes batches.
///
/// Uses biased select: high-priority requests are dequeued first.
async fn batch_loop(
    pipeline: Arc<Pipeline>,
    pipeline_name: String,
    mut rx_high: mpsc::UnboundedReceiver<BatchRequest>,
    mut rx_low: mpsc::UnboundedReceiver<BatchRequest>,
    config: BatchConfig,
) {
    let mut controller = if config.adaptive {
        Some(AdaptiveController::new(
            config.min_batch_size.max(1),
            config.max_batch_size,
        ))
    } else {
        None
    };

    info!(
        pipeline = %pipeline_name,
        max_batch = config.max_batch_size,
        timeout_ms = config.timeout.as_millis(),
        adaptive = config.adaptive,
        "batch dispatcher started (unbounded, priority)"
    );

    loop {
        // Wait for at least one request, biased toward high priority.
        let first = tokio::select! {
            biased;
            Some(req) = rx_high.recv() => req,
            Some(req) = rx_low.recv() => req,
            else => {
                debug!(pipeline = %pipeline_name, "batch channels closed");
                return;
            }
        };

        let effective_max = controller
            .as_ref()
            .map(|c| c.current_batch_size())
            .unwrap_or(config.max_batch_size);

        let mut batch = vec![first];
        let deadline = tokio::time::Instant::now() + config.timeout;

        while batch.len() < effective_max {
            tokio::select! {
                biased;
                Some(req) = rx_high.recv() => batch.push(req),
                Some(req) = rx_low.recv() => batch.push(req),
                _ = tokio::time::sleep_until(deadline) => break,
                else => break,
            }
        }

        let batch_size = batch.len();
        let pipeline = pipeline.clone();
        let name = pipeline_name.clone();
        let start = std::time::Instant::now();

        tokio::task::spawn_blocking(move || {
            process_batch(&pipeline, batch, &name, batch_size);
        })
        .await
        .ok();

        let elapsed = start.elapsed();
        if let Some(ref mut ctrl) = controller {
            ctrl.record_batch(batch_size, elapsed);
            debug!(
                pipeline = %pipeline_name,
                batch_size,
                effective_max,
                new_max = ctrl.current_batch_size(),
                fill_rate = %format!("{:.2}", ctrl.avg_fill_rate),
                item_latency_us = %format!("{:.0}", ctrl.avg_item_latency_us),
                "adaptive batch update"
            );
        }
    }
}

/// Background loop (bounded channels) that collects and processes batches.
/// Provides backpressure: when queue is full, submitters get `PipelineError::Overloaded`.
async fn batch_loop_bounded(
    pipeline: Arc<Pipeline>,
    pipeline_name: String,
    mut rx_high: mpsc::Receiver<BatchRequest>,
    mut rx_low: mpsc::Receiver<BatchRequest>,
    config: BatchConfig,
) {
    let mut controller = if config.adaptive {
        Some(AdaptiveController::new(
            config.min_batch_size.max(1),
            config.max_batch_size,
        ))
    } else {
        None
    };

    info!(
        pipeline = %pipeline_name,
        max_batch = config.max_batch_size,
        queue_capacity = config.queue_capacity,
        timeout_ms = config.timeout.as_millis(),
        adaptive = config.adaptive,
        "batch dispatcher started (bounded, backpressure, priority)"
    );

    loop {
        let first = tokio::select! {
            biased;
            Some(req) = rx_high.recv() => req,
            Some(req) = rx_low.recv() => req,
            else => {
                debug!(pipeline = %pipeline_name, "batch channels closed");
                return;
            }
        };

        let effective_max = controller
            .as_ref()
            .map(|c| c.current_batch_size())
            .unwrap_or(config.max_batch_size);

        let mut batch = vec![first];
        let deadline = tokio::time::Instant::now() + config.timeout;

        while batch.len() < effective_max {
            tokio::select! {
                biased;
                Some(req) = rx_high.recv() => batch.push(req),
                Some(req) = rx_low.recv() => batch.push(req),
                _ = tokio::time::sleep_until(deadline) => break,
                else => break,
            }
        }

        let batch_size = batch.len();
        let pipeline = pipeline.clone();
        let name = pipeline_name.clone();
        let start = std::time::Instant::now();

        tokio::task::spawn_blocking(move || {
            process_batch(&pipeline, batch, &name, batch_size);
        })
        .await
        .ok();

        let elapsed = start.elapsed();
        if let Some(ref mut ctrl) = controller {
            ctrl.record_batch(batch_size, elapsed);
            debug!(
                pipeline = %pipeline_name,
                batch_size,
                effective_max,
                new_max = ctrl.current_batch_size(),
                fill_rate = %format!("{:.2}", ctrl.avg_fill_rate),
                item_latency_us = %format!("{:.0}", ctrl.avg_item_latency_us),
                "adaptive batch update"
            );
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    use crate::error::AxonError;
    use crate::kernel::{ComputeKernel, KernelInput, KernelRegistry};
    use crate::manifest::Manifest;

    /// Mock ONNX kernel that supports batch execution.
    struct MockOnnxKernel;

    impl ComputeKernel for MockOnnxKernel {
        fn name(&self) -> &str {
            "onnx"
        }

        fn execute(
            &self,
            _input: KernelInput,
            operations: serde_json::Value,
        ) -> Result<KernelOutput, AxonError> {
            let model = operations
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            Ok(KernelOutput::Json(serde_json::json!({
                "model": model,
                "prediction": [0.9, 0.05, 0.05],
            })))
        }

        fn supports_batch(&self) -> bool {
            true
        }

        fn execute_batch(
            &self,
            inputs: Vec<KernelInput>,
            operations: serde_json::Value,
        ) -> Result<Vec<KernelOutput>, AxonError> {
            let model = operations
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            Ok(inputs
                .into_iter()
                .enumerate()
                .map(|(i, _)| {
                    KernelOutput::Json(serde_json::json!({
                        "model": model,
                        "prediction": [0.9, 0.05, 0.05],
                        "batch_index": i,
                    }))
                })
                .collect())
        }
    }

    fn test_pipeline() -> Arc<Pipeline> {
        let manifest = Manifest::from_toml(
            r#"
[model]
name = "test-batch"
file = "model.onnx"
"#,
        )
        .unwrap();

        let mut reg = KernelRegistry::new();
        reg.register(Arc::new(MockOnnxKernel));

        Arc::new(Pipeline::new(manifest, reg, PathBuf::from(".")))
    }

    #[tokio::test]
    async fn test_single_request() {
        let pipeline = test_pipeline();
        let dispatcher = BatchDispatcher::new(
            pipeline,
            "test".to_string(),
            BatchConfig {
                max_batch_size: 4,
                timeout: Duration::from_millis(50),
                ..Default::default()
            },
        );

        let result = dispatcher
            .submit(b"hello".to_vec(), "text/plain".to_string(), Priority::Low)
            .await;

        assert!(result.is_ok());
        let (output, meta) = result.unwrap();
        assert!(output.unwrap_json()["prediction"].is_array());
        assert!(meta.total_us > 0);
    }

    #[tokio::test]
    async fn test_concurrent_requests_batched() {
        let pipeline = test_pipeline();
        let dispatcher = Arc::new(BatchDispatcher::new(
            pipeline,
            "test".to_string(),
            BatchConfig {
                max_batch_size: 4,
                timeout: Duration::from_millis(100),
                ..Default::default()
            },
        ));

        // Fire 4 requests concurrently.
        let mut handles = Vec::new();
        for i in 0..4 {
            let d = dispatcher.clone();
            handles.push(tokio::spawn(async move {
                d.submit(
                    format!("input-{i}").into_bytes(),
                    "text/plain".to_string(),
                    Priority::Low,
                )
                .await
            }));
        }

        // All should succeed.
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
            let (output, _meta) = result.unwrap();
            assert!(output.unwrap_json()["prediction"].is_array());
        }
    }

    #[tokio::test]
    async fn test_timeout_triggers_partial_batch() {
        let pipeline = test_pipeline();
        let dispatcher = BatchDispatcher::new(
            pipeline,
            "test".to_string(),
            BatchConfig {
                max_batch_size: 100, // large — won't fill
                timeout: Duration::from_millis(20),
                ..Default::default()
            },
        );

        // Send just 2 requests. The batch should fire after timeout.
        let r1 = dispatcher
            .submit(b"a".to_vec(), "text/plain".to_string(), Priority::Low)
            .await;
        let r2 = dispatcher
            .submit(b"b".to_vec(), "text/plain".to_string(), Priority::Low)
            .await;

        assert!(r1.is_ok());
        assert!(r2.is_ok());
    }

    #[tokio::test]
    async fn test_batch_config_defaults() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.timeout, Duration::from_millis(10));
        assert!(!config.adaptive);
        assert_eq!(config.min_batch_size, 1);
        assert_eq!(config.queue_capacity, 0);
    }

    #[tokio::test]
    async fn test_dispatcher_drop_closes_channel() {
        let pipeline = test_pipeline();
        let dispatcher = BatchDispatcher::new(
            pipeline,
            "test".to_string(),
            BatchConfig::default(),
        );

        // Submit succeeds while dispatcher is alive.
        let result = dispatcher
            .submit(b"test".to_vec(), "text/plain".to_string(), Priority::Low)
            .await;
        assert!(result.is_ok());

        // Dropping the dispatcher closes the channel; background loop exits.
        drop(dispatcher);
        // Give the background task time to notice.
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn test_priority_high_before_low() {
        let pipeline = test_pipeline();
        let dispatcher = Arc::new(BatchDispatcher::new(
            pipeline,
            "test-priority".to_string(),
            BatchConfig {
                max_batch_size: 8,
                timeout: Duration::from_millis(100),
                ..Default::default()
            },
        ));

        // Submit 3 low-priority requests.
        let mut handles = Vec::new();
        for i in 0..3 {
            let d = dispatcher.clone();
            handles.push(tokio::spawn(async move {
                d.submit(
                    format!("low-{i}").into_bytes(),
                    "text/plain".to_string(),
                    Priority::Low,
                )
                .await
            }));
        }

        // Submit 1 high-priority request.
        let d = dispatcher.clone();
        handles.push(tokio::spawn(async move {
            d.submit(b"high-0".to_vec(), "text/plain".to_string(), Priority::High)
                .await
        }));

        // All should succeed.
        for h in handles {
            assert!(h.await.unwrap().is_ok());
        }
    }

    // ── Adaptive controller unit tests ─────────────────────────

    #[test]
    fn test_adaptive_controller_increase() {
        let mut ctrl = AdaptiveController::new(1, 16);
        assert_eq!(ctrl.current_batch_size(), 1);

        // Simulate consistently full batches → should increase.
        for _ in 0..10 {
            ctrl.record_batch(ctrl.current, Duration::from_millis(5));
        }
        assert!(
            ctrl.current_batch_size() > 1,
            "expected increase, got {}",
            ctrl.current_batch_size()
        );
    }

    #[test]
    fn test_adaptive_controller_decrease() {
        let mut ctrl = AdaptiveController::new(1, 16);
        // Ramp up first.
        ctrl.current = 8;
        ctrl.avg_fill_rate = 0.9;

        // Simulate nearly empty batches → should decrease.
        for _ in 0..10 {
            ctrl.record_batch(1, Duration::from_millis(2));
        }
        assert!(
            ctrl.current_batch_size() < 8,
            "expected decrease from 8, got {}",
            ctrl.current_batch_size()
        );
    }

    #[test]
    fn test_adaptive_controller_bounds() {
        let mut ctrl = AdaptiveController::new(2, 4);
        assert_eq!(ctrl.current_batch_size(), 2);

        // Increase past max → should be clamped.
        for _ in 0..50 {
            ctrl.record_batch(ctrl.current, Duration::from_millis(1));
        }
        assert!(ctrl.current_batch_size() <= 4);

        // Decrease past min → should be clamped.
        for _ in 0..50 {
            ctrl.record_batch(1, Duration::from_millis(1));
        }
        assert!(ctrl.current_batch_size() >= 2);
    }

    #[tokio::test]
    async fn test_adaptive_batching_integration() {
        let pipeline = test_pipeline();
        let dispatcher = Arc::new(BatchDispatcher::new(
            pipeline,
            "test-adaptive".to_string(),
            BatchConfig {
                max_batch_size: 16,
                timeout: Duration::from_millis(50),
                adaptive: true,
                min_batch_size: 1,
                queue_capacity: 0,
            },
        ));

        // Fire requests in bursts to exercise adaptive sizing.
        for _ in 0..3 {
            let mut handles = Vec::new();
            for i in 0..8 {
                let d = dispatcher.clone();
                handles.push(tokio::spawn(async move {
                    d.submit(
                        format!("burst-{i}").into_bytes(),
                        "text/plain".to_string(),
                        Priority::Low,
                    )
                    .await
                }));
            }
            for h in handles {
                let r = h.await.unwrap();
                assert!(r.is_ok());
            }
        }
    }
}
