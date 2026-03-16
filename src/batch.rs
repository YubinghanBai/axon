//! Dynamic batching for `axon serve`.
//!
//! Collects concurrent inference requests into batches, runs them through
//! the pipeline with controlled concurrency, and returns individual results.
//!
//! ## Architecture
//!
//! ```text
//! HTTP request → BatchDispatcher → mpsc channel → batch_loop
//!                                                    ↓
//!                                           collect N requests
//!                                           (timeout or max_batch_size)
//!                                                    ↓
//!                                           spawn_blocking {
//!                                             for each: pipeline.run()
//!                                           }
//!                                                    ↓
//!                                           send results via oneshot
//! ```
//!
//! ### Design rationale
//!
//! This v1 uses **request-level batching**: controlled serial execution that
//! prevents ORT session contention and provides predictable throughput.
//!
//! True tensor batching (concatenate preprocessed tensors → single ORT call)
//! requires models with dynamic batch dimension and changes to the ONNX kernel.
//! The channel-based infrastructure here is designed to support that upgrade path.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, info_span};

use crate::kernel::KernelOutput;
use crate::pipeline::{Pipeline, PipelineError};

// ── BatchRequest ──────────────────────────────────────────────

/// A single inference request waiting for batch processing.
struct BatchRequest {
    input: Vec<u8>,
    content_type: String,
    reply: oneshot::Sender<Result<KernelOutput, PipelineError>>,
}

// ── BatchConfig ───────────────────────────────────────────────

/// Configuration for dynamic batching.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    /// Maximum number of requests per batch.
    pub max_batch_size: usize,
    /// Maximum time to wait for a full batch after the first request arrives.
    pub timeout: Duration,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            timeout: Duration::from_millis(10),
        }
    }
}

// ── BatchDispatcher ───────────────────────────────────────────

/// Dispatcher that routes inference requests to a batch processing loop.
///
/// Each pipeline gets its own `BatchDispatcher`. Requests are collected
/// into batches and processed together on a blocking thread.
pub struct BatchDispatcher {
    tx: mpsc::UnboundedSender<BatchRequest>,
}

impl BatchDispatcher {
    /// Create a new BatchDispatcher for a pipeline.
    ///
    /// Spawns a background tokio task that collects and processes batches.
    pub fn new(pipeline: Arc<Pipeline>, pipeline_name: String, config: BatchConfig) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(batch_loop(pipeline, pipeline_name, rx, config));
        Self { tx }
    }

    /// Submit a request for batch processing.
    ///
    /// Returns the pipeline output when the batch containing this request completes.
    pub async fn submit(
        &self,
        input: Vec<u8>,
        content_type: String,
    ) -> Result<KernelOutput, PipelineError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(BatchRequest {
                input,
                content_type,
                reply: reply_tx,
            })
            .map_err(|_| PipelineError::Empty)?;

        reply_rx.await.unwrap_or(Err(PipelineError::Empty))
    }
}

// ── Batch loop ────────────────────────────────────────────────

/// Background loop that collects and processes batches.
///
/// Waits for the first request, then collects more within the timeout window
/// (up to max_batch_size). Processes the entire batch on a blocking thread.
async fn batch_loop(
    pipeline: Arc<Pipeline>,
    pipeline_name: String,
    mut rx: mpsc::UnboundedReceiver<BatchRequest>,
    config: BatchConfig,
) {
    info!(
        pipeline = %pipeline_name,
        max_batch = config.max_batch_size,
        timeout_ms = config.timeout.as_millis(),
        "batch dispatcher started"
    );

    loop {
        // Wait for the first request (no timeout — block until work arrives).
        let first = match rx.recv().await {
            Some(req) => req,
            None => {
                debug!(pipeline = %pipeline_name, "batch channel closed");
                return;
            }
        };

        let mut batch = vec![first];
        let deadline = tokio::time::Instant::now() + config.timeout;

        // Collect more requests until timeout or max batch size.
        while batch.len() < config.max_batch_size {
            tokio::select! {
                biased;
                next = rx.recv() => {
                    match next {
                        Some(req) => batch.push(req),
                        None => break,
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    break;
                }
            }
        }

        let batch_size = batch.len();
        let pipeline = pipeline.clone();
        let name = pipeline_name.clone();

        // Process batch on blocking thread pool.
        tokio::task::spawn_blocking(move || {
            let _span = info_span!(
                "batch",
                pipeline = %name,
                batch_size,
            )
            .entered();

            debug!("processing batch");

            for (i, item) in batch.into_iter().enumerate() {
                let _item_span = info_span!("batch_item", idx = i).entered();
                let result = pipeline.run(&item.input, &item.content_type);
                let _ = item.reply.send(result);
            }
        })
        .await
        .ok();
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

    /// Mock ONNX kernel for testing.
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
            },
        );

        let result = dispatcher
            .submit(b"hello".to_vec(), "text/plain".to_string())
            .await;

        assert!(result.is_ok());
        let output = result.unwrap().unwrap_json();
        assert!(output["prediction"].is_array());
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
            },
        ));

        // Fire 4 requests concurrently.
        let mut handles = Vec::new();
        for i in 0..4 {
            let d = dispatcher.clone();
            handles.push(tokio::spawn(async move {
                d.submit(format!("input-{i}").into_bytes(), "text/plain".to_string())
                    .await
            }));
        }

        // All should succeed.
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
            let output = result.unwrap().unwrap_json();
            assert!(output["prediction"].is_array());
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
            },
        );

        // Send just 2 requests. The batch should fire after timeout.
        let r1 = dispatcher
            .submit(b"a".to_vec(), "text/plain".to_string())
            .await;
        let r2 = dispatcher
            .submit(b"b".to_vec(), "text/plain".to_string())
            .await;

        assert!(r1.is_ok());
        assert!(r2.is_ok());
    }

    #[tokio::test]
    async fn test_batch_config_defaults() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.timeout, Duration::from_millis(10));
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
            .submit(b"test".to_vec(), "text/plain".to_string())
            .await;
        assert!(result.is_ok());

        // Dropping the dispatcher closes the channel; background loop exits.
        drop(dispatcher);
        // Give the background task time to notice.
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
