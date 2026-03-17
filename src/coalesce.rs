//! Request coalescing (singleflight) for Axon pipelines.
//!
//! Deduplicates identical in-flight inference requests. If request B
//! arrives while identical request A is still running, B waits for A's
//! result instead of running inference again.
//!
//! Uses blake3 hash of the pre-processed input as the dedup key.
//! Thread-safe via `parking_lot::Mutex` + `Condvar`.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

// ── Singleflight ────────────────────────────────────────────────

/// Thread-safe request coalescer.
///
/// Multiple threads calling `dedupe()` with the same key will share
/// a single computation. The first caller runs the closure; subsequent
/// callers block until the result is ready and receive a clone.
pub struct Singleflight {
    in_flight: Mutex<HashMap<[u8; 32], Arc<FlightSlot>>>,
}

/// Shared slot for an in-flight computation.
struct FlightSlot {
    /// The result, set once the computation completes.
    result: Mutex<Option<FlightResult>>,
    /// Signaled when the computation completes.
    done: Condvar,
}

/// Clone-friendly result stored in a flight slot.
#[derive(Clone)]
enum FlightResult {
    Ok(SharedOutput),
    Err(String),
}

/// Clone-friendly wrapper for KernelOutput stored in flight slots.
#[derive(Clone)]
enum SharedOutput {
    Json(serde_json::Value),
    Blob {
        data: Arc<Vec<u8>>,
        content_type: String,
        shape: Option<Vec<usize>>,
    },
}

impl SharedOutput {
    fn from_output(output: &crate::kernel::KernelOutput) -> Self {
        match output {
            crate::kernel::KernelOutput::Json(v) => Self::Json(v.clone()),
            crate::kernel::KernelOutput::Blob {
                data,
                content_type,
                shape,
            } => Self::Blob {
                data: Arc::new(data.clone()),
                content_type: content_type.clone(),
                shape: shape.clone(),
            },
        }
    }

    fn to_output(&self) -> crate::kernel::KernelOutput {
        match self {
            Self::Json(v) => crate::kernel::KernelOutput::Json(v.clone()),
            Self::Blob {
                data,
                content_type,
                shape,
            } => crate::kernel::KernelOutput::Blob {
                data: (**data).clone(),
                content_type: content_type.clone(),
                shape: shape.clone(),
            },
        }
    }
}

impl Default for Singleflight {
    fn default() -> Self {
        Self::new()
    }
}

impl Singleflight {
    /// Create a new Singleflight coalescer.
    pub fn new() -> Self {
        Self {
            in_flight: Mutex::new(HashMap::new()),
        }
    }

    /// Deduplicate a computation by key.
    ///
    /// If no computation is in-flight for `key`, runs `compute` and shares
    /// the result with any waiters. If a computation is already running,
    /// blocks until it completes and returns a clone of the result.
    ///
    /// Returns `(KernelOutput, bool)` where the bool indicates whether
    /// this call was the one that ran the computation (true) or waited (false).
    pub fn dedupe<F>(
        &self,
        key: [u8; 32],
        compute: F,
    ) -> (Result<crate::kernel::KernelOutput, crate::pipeline::PipelineError>, bool)
    where
        F: FnOnce() -> Result<crate::kernel::KernelOutput, crate::pipeline::PipelineError>,
    {
        // Check if there's an in-flight computation for this key.
        let slot = {
            let mut map = self.in_flight.lock();
            if let Some(existing) = map.get(&key) {
                // Someone else is computing — wait for their result.
                Arc::clone(existing)
            } else {
                // We're first — register our slot.
                let slot = Arc::new(FlightSlot {
                    result: Mutex::new(None),
                    done: Condvar::new(),
                });
                map.insert(key, Arc::clone(&slot));
                drop(map);

                // Run the computation.
                let result = compute();

                // Store the result.
                let flight_result = match &result {
                    Ok(output) => FlightResult::Ok(SharedOutput::from_output(output)),
                    Err(e) => FlightResult::Err(e.to_string()),
                };
                *slot.result.lock() = Some(flight_result);
                slot.done.notify_all();

                // Remove from in-flight map.
                self.in_flight.lock().remove(&key);

                return (result, true);
            }
        };

        // Wait for the existing computation to complete.
        let mut result_guard = slot.result.lock();
        while result_guard.is_none() {
            slot.done.wait(&mut result_guard);
        }

        let flight_result = result_guard.as_ref().unwrap().clone();
        let result = match flight_result {
            FlightResult::Ok(shared) => Ok(shared.to_output()),
            FlightResult::Err(msg) => Err(crate::pipeline::PipelineError::ModelFailed(
                crate::error::AxonError::runtime(msg),
            )),
        };
        (result, false)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_single_request_runs_compute() {
        let sf = Singleflight::new();
        let key = [0u8; 32];
        let (result, was_leader) = sf.dedupe(key, || {
            Ok(crate::kernel::KernelOutput::Json(serde_json::json!({"answer": 42})))
        });
        assert!(was_leader);
        assert!(result.is_ok());
        let json = result.unwrap().unwrap_json();
        assert_eq!(json["answer"], 42);
    }

    #[test]
    fn test_concurrent_dedup() {
        let sf = Arc::new(Singleflight::new());
        let counter = Arc::new(AtomicUsize::new(0));
        let key = [1u8; 32];

        let mut handles = Vec::new();
        for _ in 0..10 {
            let sf = sf.clone();
            let counter = counter.clone();
            handles.push(std::thread::spawn(move || {
                let (result, _was_leader) = sf.dedupe(key, || {
                    counter.fetch_add(1, Ordering::SeqCst);
                    // Simulate slow computation.
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    Ok(crate::kernel::KernelOutput::Json(
                        serde_json::json!({"computed": true}),
                    ))
                });
                result
            }));
        }

        let mut results = Vec::new();
        for h in handles {
            results.push(h.join().unwrap());
        }

        // All 10 should succeed.
        for r in &results {
            assert!(r.is_ok());
        }

        // Compute should have run exactly once.
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "compute should run exactly once for identical keys"
        );
    }

    #[test]
    fn test_different_keys_run_independently() {
        let sf = Arc::new(Singleflight::new());
        let counter = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for i in 0..5 {
            let sf = sf.clone();
            let counter = counter.clone();
            handles.push(std::thread::spawn(move || {
                let mut key = [0u8; 32];
                key[0] = i as u8; // Different keys.
                sf.dedupe(key, || {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(crate::kernel::KernelOutput::Json(
                        serde_json::json!({"key": i}),
                    ))
                })
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Each key should trigger its own computation.
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_error_propagation() {
        let sf = Arc::new(Singleflight::new());
        let key = [2u8; 32];

        let mut handles = Vec::new();
        for _ in 0..5 {
            let sf = sf.clone();
            handles.push(std::thread::spawn(move || {
                let (result, _) = sf.dedupe(key, || {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    Err(crate::pipeline::PipelineError::ModelFailed(
                        crate::error::AxonError::runtime("model crashed"),
                    ))
                });
                result
            }));
        }

        for h in handles {
            let result = h.join().unwrap();
            assert!(result.is_err());
        }
    }
}
