# Axon Architecture

Design decisions and trade-off analysis for production-critical subsystems.

## Batch Execution Model

```
HTTP request → BatchDispatcher → mpsc channel → batch_loop
                                                    ↓
                                            collect N requests
                                            (timeout or adaptive size)
                                                    ↓
                                            spawn_blocking {
                                              pipeline.run_batch(N inputs)
                                                Phase 1: pre-process each independently
                                                Phase 2: concatenate along dim 0 → single session.run()
                                                Phase 3: post-process each independently
                                            }
                                                    ↓
                                            send results via oneshot
```

### Three-Phase Pipeline Batch

1. **Pre-processing** (per-item, isolated): Each input runs through pre-steps independently. Failures are captured per-item without aborting the batch.
2. **Model inference** (batched): All valid pre-processed tensors are concatenated along batch dimension 0 into a single ONNX input, producing one `session.run()` call. Falls back to serial if model doesn't support dynamic batch dim.
3. **Post-processing** (per-item, isolated): Each output runs through post-steps independently. Failures are isolated per-item.

## Key Design Decisions

### Why tensor-level batching over request-level batching

Request-level batching (running `pipeline.run()` N times serially in a batch loop) wastes GPU cycles: each call has kernel launch overhead, memory allocation, and session lock acquisition. True tensor batching concatenates inputs into a single tensor → one kernel launch, one memory allocation, one lock acquisition for N items. The throughput difference is significant (2-5x on GPU workloads).

### Why AIMD for adaptive sizing

AIMD (Additive Increase, Multiplicative Decrease) is battle-tested in TCP congestion control. It's simple, stable, and converges quickly. Alternatives considered:
- **PID controller**: More responsive but harder to tune, risk of oscillation.
- **Fixed sizing**: No adaptation to load patterns.
- **Gradient-based**: Over-engineered for batch size tuning.

AIMD with EMA smoothing (alpha=0.3) provides stable adaptation without oscillation.

### Why partial failure isolation

In production, a single corrupted image shouldn't kill a batch of 7 valid requests. Per-item error isolation means:
- Pre-processing failure → that item gets an error, rest proceed to inference.
- Post-processing failure → that item gets an error, rest return successfully.
- Model inference failure → outer error, batch dispatcher falls back to serial.

The return type `Result<Vec<Result<KernelOutput, PipelineError>>, PipelineError>` makes the contract explicit: outer Err = fatal (step resolution, empty pipeline); inner Err = per-item failure.

### Why try-and-cache for batch compatibility

Not all ONNX models support dynamic batch dimensions. Rather than requiring users to configure this per-model:
1. First batch attempt tries tensor batching.
2. If ONNX runtime rejects it (shape mismatch), the model is cached as batch-incompatible.
3. Future calls for that model skip directly to serial execution.

This is zero-configuration — correct behavior is discovered automatically with one-time cost.

### Why bounded channel backpressure

Unbounded channels + unlimited request acceptance = OOM under load spikes. Bounded channels with `try_send` provide:
- Immediate 503 response (fail fast) instead of queuing indefinitely.
- Bounded memory usage regardless of load.
- Client can retry or route to another instance.

This is opt-in (`queue_capacity: 0` = unbounded, default behavior preserved).

## Server Lifecycle (Cloud-Native)

### Startup sequence

```
1. Load pipelines (parse manifests, validate kernels)
2. Warmup: for each pipeline, load ONNX session + run 2 dummy inferences
   → Session::new() loads weights + Level3 graph optimization
   → First session.run() triggers JIT kernel compilation (TensorRT/CUDA)
   → Second session.run() confirms stable execution
3. Set ready=true
4. Bind HTTP port → start serving
```

Key: HTTP port is NOT exposed until warmup completes. K8s readiness probe returns 503 during this phase.

### Shutdown sequence (graceful)

```
1. SIGTERM/SIGINT received
2. Set ready=false → K8s readiness probe fails → K8s stops routing new traffic
3. Axum stops accepting new connections (with_graceful_shutdown)
4. In-flight requests complete normally
5. Batch dispatcher channels close → batch_loops process remaining queued items
6. All oneshot replies sent → clients get responses
7. ORT sessions dropped → process exits
```

Zero request loss during rolling updates or scale-down.

### K8s probe design

| Probe | Endpoint | When it fails | K8s action |
|-------|----------|---------------|------------|
| Liveness | `/health/live` | Process deadlocked (never, for now) | Restart pod |
| Readiness | `/health/ready` | During warmup or shutdown | Stop routing traffic |

Why two probes: a pod that's warming up is not dead (don't restart it), just not ready (don't send it traffic). A single `/health` endpoint can't express this distinction.

## Observability

### Queue wait time tracking

Each `BatchRequest` carries a `created_at: Instant`. When the batch is processed, we compute per-item queue wait time and emit it as a tracing span field (`queue_wait_avg_us`). This is the single most important metric for diagnosing "why is my P99 latency high?" — if queue wait >> inference time, you need more capacity or smaller batch timeouts.

### Batch efficiency metrics

The `axon_batch_size_avg` metric is the key operational indicator:
- Consistently 1.0–1.5 → traffic too low for batching to help, or batch timeout too short
- Consistently at `max_batch_size` → system saturated, consider scaling out
- Between 3–8 → healthy, batching providing good throughput gains

## Semantic Inference Cache

```
pre-processing → blake3(preprocessed input) → cache lookup
                                                 ↓ HIT
                                          skip model, return cached
                                                 ↓ MISS
                                          model inference → cache store → post-processing
```

Content-addressed caching at the tensor level. After pre-processing, blake3 hashes the `KernelInput` (JSON + blob data in deterministic key order). On cache hit, model inference is skipped entirely — the cached result feeds directly into post-processing.

Backed by moka LRU cache with configurable TTL and max entries. Configured via `[cache]` in manifest.toml. Zero overhead when not configured.

**Why blake3**: 2-3x faster than SHA-256, SIMD-accelerated. The hash includes both JSON metadata and raw blob bytes for correctness.

**Why tensor-level not request-level**: Hashing raw request bytes would miss cache on trivially different inputs (JPEG re-encoding, HTTP header differences). Hashing after pre-processing means semantically identical inputs always hit cache.

## Cascade Inference

```
primary model (fast) → extract max confidence
                          ↓ ≥ threshold
                        return result
                          ↓ < threshold
                        fallback model (accurate) → return result
```

Confidence-gated early exit. A lightweight primary model runs first. If the maximum confidence score exceeds the threshold, its result is accepted. Otherwise, the heavier fallback model runs using the same pre-processed input.

Configured via `[cascade]` in manifest.toml: `confidence_threshold`, `fallback` model path, optional `fallback_device`.

Confidence extraction checks multiple JSON patterns (direct `confidence` field, `results[].confidence` array, `prediction[]` softmax output) and interprets blob data as f32 tensor. If confidence can't be determined, the primary result is accepted (conservative).

## Circuit Breaker (Device Fallback)

```
CLOSED → (N consecutive failures) → OPEN → (recovery timeout) → HALF_OPEN
   ↑                                                                  ↓
   └──────────── success ──────────────────────────────────────────────┘
                                                                      ↓ failure
                                                                    OPEN
```

Automatic device fallback on consecutive inference failures. Lock-free state machine: `AtomicU8` for state, `AtomicU32` for failure count. Only mutex is `opened_at` for recovery timing (rarely contended). Configured via `[resilience]` in manifest.toml.

## Stateful Session Context

```
Frame 1 [session: cam-A]:  post gets _context (empty) → saves _context
Frame 2 [session: cam-A]:  post gets _context (from frame 1) → updates _context
Frame 1 [session: cam-B]:  post gets _context (empty, isolated)
```

Cross-request state for streaming workloads. HTTP `X-Session-Id` header → `SessionStore` (moka cache, 10 min TTL, 10k max). Context is injected as `_context` before post-processing and extracted after.

## Pipeline Hot-Reload

```
POST /admin/reload/{name} → parse manifest → validate → warmup → RwLock write swap → 200
```

Zero-downtime model updates. Pipelines map is `Arc<parking_lot::RwLock<HashMap>>`. Read lock per-request (sub-microsecond). Write lock only during hot-reload after full warmup. In-flight requests on old pipeline complete uninterrupted via `Arc<Pipeline>` clone.

## Input Guard

```
HTTP request → InputGuard.validate() → pipeline
                    ↓ FAIL
               400 { error, rule, message }
```

Pre-processing validation gate at the top of the pipeline. Catches bad inputs before any compute runs. Rules are evaluated sequentially; first failure short-circuits with a structured error.

Three rule types:
- **MaxSize**: Rejects inputs exceeding a byte threshold. Parsed from human-readable strings ("10MB", "512KB").
- **ContentType**: Allowlist of MIME types. Rejects requests with unlisted Content-Type headers.
- **JsonSchema**: Validates required top-level JSON fields exist.

Configured via `[guard]` in manifest.toml. When no guard config is present, validation is skipped (zero overhead). Guard errors map to HTTP 400 with a JSON body containing `error`, `rule`, and `message` fields — distinct from 500 model failures.

**Why validate before pre-processing**: Pre-processing steps (WASM kernels, image decode) are expensive. Rejecting a 500MB upload at the gate is orders of magnitude cheaper than discovering it fails mid-pipeline.

## Audit Trail

```
pipeline completes → AuditLogger.log(entry) → append JSONL line
                         ↓ (sampling)
                    skip if hash-based sample misses
```

Append-only JSONL log of every inference request for compliance, debugging, and cost attribution. Each entry contains: ISO 8601 timestamp, pipeline name, blake3 input/output hashes, status (ok/error), latency in microseconds, device used, cache hit flag, and optional session ID.

Thread-safe via `parking_lot::Mutex<BufWriter<File>>`. Writes are fire-and-forget: audit failures never affect the primary response. Deterministic sampling based on input hash prefix ensures consistent sampling across retries.

Configured via `[audit]` in manifest.toml with `path` and `sample_rate` (0.0–1.0).

**Why deterministic sampling over random**: The same input always either gets logged or doesn't. This makes debugging reproducible — if a user reports an issue, you know whether the audit trail captured it based on the input alone.

## Cost Metering

```
pipeline.run_with_session() → (KernelOutput, InferenceMeta)
                                                  ↓
                                    X-Axon-Compute-Us: 1234
                                    X-Axon-Cache-Hit: true
                                    X-Axon-Device: cpu
```

Per-request timing metadata returned alongside inference output. `InferenceMeta` breaks down latency into phases: `pre_us`, `inference_us`, `post_us`, `total_us`, plus `device`, `cache_hit`, and `cascade_used` flags.

The `run()` convenience method discards meta for backward compatibility. `run_with_session()` returns the full `(KernelOutput, InferenceMeta)` tuple. HTTP responses include meta as response headers. Batch results propagate per-item meta.

**Why phase-level timing**: "Model inference took 50ms" is not actionable. "Pre-processing: 45ms, inference: 3ms, post: 2ms" immediately reveals the bottleneck. Phase timing is essential for optimizing pipeline configuration.

## Priority Batching

```
HTTP request → X-Priority: high → tx_high ──┐
                                              ├── batch_loop (biased select)
HTTP request → (default)       → tx_low  ──┘
                                              ↓
                                    drain high first, then low
```

Two-tier priority queue for latency-sensitive requests. The batch dispatcher maintains separate high and low priority channels. `tokio::select! { biased; }` drains the high channel first, ensuring priority requests are never starved behind bulk inference.

Priority is set via the `X-Priority: high` HTTP header. Any other value (or absence) defaults to low priority. Both channels share the same pipeline and batch processing logic.

**Why two tiers not N**: Priority inversion and starvation analysis become intractable with N priority levels. Two tiers (interactive vs. batch) cover the real-world use case. If you need more granularity, use separate pipeline deployments.

## Request Coalescing (Singleflight)

```
Thread A [key=abc] → compute() → result
Thread B [key=abc] → wait...   → clone(result)
Thread C [key=abc] → wait...   → clone(result)
Thread D [key=xyz] → compute() → result (independent)
```

Deduplicates identical in-flight inference requests. If request B arrives while identical request A is still running, B blocks and receives a clone of A's result instead of running inference again. Uses blake3 hash of pre-processed input as the dedup key.

Implementation uses `parking_lot::Mutex` + `Condvar` for synchronous notification. The first caller (leader) runs the computation; subsequent callers (waiters) block on the condvar. Results are stored as clone-friendly `SharedOutput` wrappers. Errors are propagated via String conversion since `PipelineError` is not Clone.

Integrated after cache check, before model inference. Cache handles temporal dedup (same input seen before); coalescing handles spatial dedup (same input in-flight right now). Both use the same blake3 key.

**Why Condvar over async channels**: Pipeline execution is synchronous (runs in `spawn_blocking`). Using async primitives would require an async runtime inside a blocking thread. Condvar is the natural fit.

## Shadow Inference

```
primary model → result ──────────────────────→ return to caller
                   ↓ (sampled)
              std::thread::spawn
                   ↓
              shadow model → compare outputs → append JSONL
```

Runs a shadow model in parallel with the primary model for A/B comparison logging without affecting primary latency. Fire-and-forget on a detached `std::thread`. The shadow thread clones the input, runs the shadow model through the same ONNX kernel interface, and logs a `ShadowComparison` entry with both output hashes, latency, and match status.

Shadow failures are silently logged with `"error"` as the shadow output hash. The primary response is never affected by shadow execution — no panics, no latency impact, no error propagation.

Configured via `[shadow]` in manifest.toml with `model` path, `sample_rate`, `log_path`, and optional `device`.

**Why detached thread over tokio::spawn**: Shadow inference uses the synchronous ONNX Runtime API. A detached `std::thread` avoids blocking the async runtime's thread pool and ensures zero impact on primary request scheduling.

## Request Deadline Propagation

```
X-Timeout-Ms: 200 → RequestBudget { deadline: now + 200ms }
                         ↓
    pre phase: budget.check("pre") → ok
    model phase: budget.check("model") → ok or DeadlineExceeded
    post phase: budget.check("post") → ok or DeadlineExceeded
```

Prevents wasted compute on requests whose client-side timeout has already expired. In microservice architectures, a request may spend significant time in queue before reaching inference. If the client has already given up, running inference wastes GPU cycles.

Budget checks are inserted at three phase boundaries: before pre-processing, before model inference, and before post-processing. If the deadline has passed, the pipeline short-circuits with `DeadlineExceeded` (HTTP 408) instead of consuming compute.

Set via `X-Timeout-Ms` header (integer milliseconds). Zero overhead when the header is absent.

**Why phase boundaries not per-step**: Checking between every pre/post step adds overhead proportional to step count. Three checks at phase boundaries catch the vast majority of cases with constant O(1) cost (three `Instant::now()` comparisons).

## Canary Traffic Routing

```
request → hash(input or session) % 100
                    ↓
              < weight (e.g., 5)
                    ↓ YES                    ↓ NO
              canary model               primary model
              X-Axon-Model: canary       X-Axon-Model: primary
```

Routes a configurable percentage of live traffic to a canary model, returning the canary's result to the client. Unlike shadow inference (fire-and-forget, never affects response), canary routing is the real thing — users see the canary model's output.

Deterministic routing via input hash ensures consistent behavior across retries. Sticky sessions (optional) ensure the same session always sees the same model version, preventing confusion in stateful workloads.

Configured via `[canary]` in manifest.toml. Complements shadow inference: shadow → canary → full rollout is the natural progression for safe model deployment.

**Why deterministic hash not random**: Random routing would make debugging impossible — "was this user's bad result from primary or canary?" With hash-based routing, you can reproduce the exact routing decision from the input or session ID.

## Output Health Check

```
model output → OutputHealthCheck.check()
                    ↓ ALERT
              HealthCheckFailed { rule, message }
                    ↓ OK
              continue to post-processing
```

Lightweight per-request validation of model outputs. Not a drift detector — just a smoke test for obviously broken outputs. Checks run at ~100ns cost (a few f32 comparisons).

Two check types:
- **NaN/Inf detection**: Scans blob data as f32 arrays for NaN and Infinity values. Catches numerical instability from fp16 precision issues or corrupted weights.
- **Range bounds**: Validates all numeric values fall within expected bounds (e.g., [0.0, 1.0] for probability outputs).

Configured via `[healthcheck]` in manifest.toml. When not configured, zero overhead.

**Why not full drift detection**: Dataset-level drift detection (KL divergence, PSI, embedding drift) requires historical data, offline computation, and model-type-specific logic. That belongs in MLOps platforms (Arize, Evidently). The inference engine's job is catching per-request failures: "is this output obviously broken?" — the same principle as K8s liveness probes.

## Adaptive Cascade Threshold

```
fixed mode:    confidence < 0.95 → fallback
                                   (threshold never changes)

adaptive mode: confidence < threshold → fallback
                                          ↓
                              track actual fallback rate
                              every N requests: adjust threshold
                              actual_rate > target → lower threshold
                              actual_rate < target → raise threshold
```

Automatically adjusts the cascade confidence threshold to maintain a target fallback rate. Uses AIMD-style adjustment (same principle as batch sizing): additive increase (+0.01) when fallback rate is too low, multiplicative decrease (-0.02) when too high.

Solves the problem of fixed thresholds becoming stale as input distributions shift. Day traffic (clear images) and night traffic (blurry images) need different thresholds — adaptive mode handles this without redeployment.

Configured via `target_fallback_rate` and `adjust_interval` in `[cascade]`. When `target_fallback_rate` is not set, falls back to fixed threshold mode (backward compatible).

**Why AIMD not PID**: Same rationale as adaptive batch sizing — AIMD is simple, stable, and converges. PID controllers are more responsive but require tuning and risk oscillation, which is unacceptable in production inference.

## What We Deliberately Did NOT Implement

### Zero-copy batching via shared memory / arena allocators
**Why not**: The tensor data is already in contiguous `Vec<u8>` blobs. Concatenation is a single memcpy per item. The overhead is negligible compared to model inference latency (microseconds vs milliseconds). Arena allocators add complexity without measurable benefit at current batch sizes (<64).

### Shape alignment and padding
**Why not**: ONNX models with dynamic batch dim accept varying batch sizes natively. Padding would waste compute on dummy inputs. If shapes are genuinely incompatible (different feature dims), the batch correctly falls back to serial.

### OOM protection with memory bucketing
**Why not**: Batch sizes are bounded by `max_batch_size` (typically 8-32). With bounded queues, total memory is bounded. OOM protection at the batch level is solving a problem that doesn't exist when queue capacity is properly configured.

### ONNX dynamic axes introspection
**Why not**: Querying ONNX model metadata for dynamic dimensions is unreliable across providers (CoreML, TensorRT report differently). Try-and-cache is more robust and works with any provider.

### Async pipeline parallelism (3-stage overlap)
**Why not**: Overlapping pre-process / inference / post-process across batches would require 3 independent worker pools with inter-stage channels — doubling the scheduler complexity. In typical ML workloads, inference dominates latency by 10-100x, so overlap with pre/post provides <5% throughput improvement. Current architecture already maximizes GPU utilization via tensor batching. If profiling shows pre/post as bottleneck, this can be added without breaking existing APIs.

## Testing Strategy

- **Unit tests** (batch.rs): AdaptiveController bounds, increase/decrease behavior, config defaults, batch metrics, priority high-before-low ordering.
- **Pipeline tests** (pipeline.rs): Semantic cache hit/miss, circuit breaker state machine (closed→open→half-open→closed, probe failure), cascade confidence extraction (JSON patterns, f32 blob tensor), session context flow (inject/extract, isolation), hash determinism, InferenceMeta population.
- **Guard tests** (guard.rs): MaxSize validation, ContentType allowlist, JsonSchema required fields, parse_size ("10MB", "512KB"), combined rules, edge cases.
- **Audit tests** (audit.rs): JSONL write and read-back, sampling behavior, hash utilities.
- **Coalesce tests** (coalesce.rs): Singleflight dedup (10 concurrent threads, compute runs once), error propagation, independent keys, basic dedup.
- **Shadow tests** (shadow.rs): Comparison logging at 100% sample, zero sample rate suppression, primary unaffected by shadow failure.
- **Canary tests** (canary.rs): Zero/full weight, deterministic routing, sticky sessions, weight distribution (50% ±15%), device override/fallback, path construction.
- **Deadline tests** (deadline.rs): Budget expiry, remaining duration, check phase propagation, header parsing (valid integer, invalid string).
- **Health check tests** (healthcheck.rs): NaN/Inf detection in blobs, range violation, healthy output pass, unconfigured passthrough.
- **Integration tests**: Batch with model-only, batch with post-processing, single-item batch, empty batch, large batch (32 items).
- **Property tests**: Existing property tests cover kernel round-trip behavior.
- **Serve tests**: Batched inference, health check, K8s liveness/readiness probes, hot-reload endpoint, session-aware inference, metrics endpoint, priority header extraction, guard error → 400, cost metering response headers.
