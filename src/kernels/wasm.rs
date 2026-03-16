//! WASM compute kernel: sandboxed execution of pre-compiled WebAssembly modules.
//!
//! Enabled via `cargo build --features wasm`.
//!
//! ABI — WASM module must export:
//!   - `memory`: linear memory
//!   - `alloc(size: i32) -> i32`: allocate input buffer, return pointer
//!   - `transform(ptr: i32, len: i32) -> i64`: process JSON, return packed (out_ptr << 32 | out_len)
//!
//! Security (three fences):
//!   - CPU: fuel consumption (default 10M instructions, configurable)
//!   - Memory: ResourceLimiter (default 64MB, configurable)
//!   - Stack: max_wasm_stack (512KB)
//!   - WASM sandbox: no file system, no network, no syscalls by default
//!
//! Operations format (from Blueprint DSL body):
//!   - String: `"path/to/module.wasm"` (all defaults)
//!   - Multiline: `"module: path\nfuel: 5000000\nmemory: 128"`
//!   - Object: `{"module": "path", "fuel": N, "memory": N}`

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use tracing::{debug, info};
use wasmtime::*;

use crate::error::AxonError;
use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};

/// Default fuel limit: 10 million instructions.
const DEFAULT_FUEL: u64 = 10_000_000;

/// Default memory limit: 64 MB.
const DEFAULT_MEMORY_MB: usize = 64;

/// Maximum output size: 16 MB.
const MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;

/// WASM module configuration parsed from operations.
#[derive(Debug)]
struct WasmConfig {
    module_path: PathBuf,
    fuel: u64,
    memory_mb: usize,
    /// When true, output raw bytes as KernelOutput::Blob (no JSON parsing).
    blob_output: bool,
    /// Content-type for blob output (default: "tensor/f32").
    output_content_type: String,
    /// Shape hint for blob output (optional — WASM module may encode this).
    output_shape: Option<Vec<usize>>,
}

/// Resource limiter for WASM linear memory growth.
struct WasmLimiter {
    max_bytes: usize,
}

impl ResourceLimiter for WasmLimiter {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> anyhow::Result<bool> {
        Ok(desired <= self.max_bytes)
    }

    fn table_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> anyhow::Result<bool> {
        Ok(desired <= 10_000)
    }
}

/// Store state: holds the resource limiter.
struct StoreState {
    limiter: WasmLimiter,
}

/// WASM compute kernel.
///
/// Compiles and caches `.wasm` modules, executes them with fuel + memory limits.
/// Each invocation gets a fresh `Store` (full isolation between calls).
pub struct WasmKernel {
    engine: Engine,
    cache: RwLock<HashMap<PathBuf, Module>>,
}

impl WasmKernel {
    /// Create a new WasmKernel with default engine configuration.
    pub fn new() -> Result<Self, String> {
        let mut config = Config::new();
        config.consume_fuel(true);
        config.max_wasm_stack(512 * 1024); // 512KB stack

        let engine =
            Engine::new(&config).map_err(|e| format!("wasm engine init failed: {e}"))?;

        Ok(Self {
            engine,
            cache: RwLock::new(HashMap::new()),
        })
    }

    /// Load a module from disk, caching the compiled result.
    ///
    /// Uses double-check locking to avoid compiling the same module twice.
    fn get_or_compile(&self, path: &Path) -> Result<Module, String> {
        // Fast path: read lock.
        if let Some(module) = self.cache.read().get(path) {
            return Ok(module.clone());
        }

        // Slow path: write lock + double-check.
        let mut cache = self.cache.write();
        if let Some(module) = cache.get(path) {
            return Ok(module.clone());
        }

        info!(path = %path.display(), "wasm: compiling module");
        let bytes = std::fs::read(path)
            .map_err(|e| format!("wasm: cannot read {}: {e}", path.display()))?;
        let module = Module::new(&self.engine, &bytes)
            .map_err(|e| format!("wasm: compile error for {}: {e}", path.display()))?;
        cache.insert(path.to_path_buf(), module.clone());
        Ok(module)
    }

    /// Compile a module from raw bytes (WAT or WASM binary). Test helper.
    #[cfg(test)]
    fn compile_bytes(&self, bytes: &[u8]) -> Result<Module, String> {
        Module::new(&self.engine, bytes)
            .map_err(|e| format!("wasm: compile error: {e}"))
    }

    /// Execute a compiled module with the given input bytes and resource limits.
    ///
    /// Input bytes are written directly into WASM linear memory.
    /// Output bytes are read back and interpreted based on `blob_output` flag.
    fn run_module(
        &self,
        module: &Module,
        input_bytes: &[u8],
        config: &WasmConfig,
    ) -> Result<KernelOutput, String> {
        // Fresh store per invocation → full isolation.
        let state = StoreState {
            limiter: WasmLimiter {
                max_bytes: config.memory_mb * 1024 * 1024,
            },
        };
        let mut store = Store::new(&self.engine, state);
        store
            .set_fuel(config.fuel)
            .map_err(|e| format!("wasm: set fuel: {e}"))?;
        store.limiter(|s| &mut s.limiter);

        // Instantiate — no imports for pure compute modules.
        let instance = Instance::new(&mut store, module, &[])
            .map_err(|e| format!("wasm: instantiate: {e}"))?;

        // Resolve required exports.
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or("wasm: module must export 'memory'")?;

        let alloc_fn = instance
            .get_typed_func::<i32, i32>(&mut store, "alloc")
            .map_err(|e| format!("wasm: module must export 'alloc(i32) -> i32': {e}"))?;

        let transform_fn = instance
            .get_typed_func::<(i32, i32), i64>(&mut store, "transform")
            .map_err(|e| {
                format!("wasm: module must export 'transform(i32, i32) -> i64': {e}")
            })?;

        // Allocate input buffer in WASM linear memory.
        let input_len = input_bytes.len() as i32;
        let input_ptr = alloc_fn
            .call(&mut store, input_len)
            .map_err(|e| classify_trap(e, config.fuel))?;

        // Write input bytes into WASM memory.
        memory
            .write(&mut store, input_ptr as usize, input_bytes)
            .map_err(|e| format!("wasm: write input to memory: {e}"))?;

        // Call transform.
        debug!(input_len, "wasm: calling transform");
        let result_packed = transform_fn
            .call(&mut store, (input_ptr, input_len))
            .map_err(|e| classify_trap(e, config.fuel))?;

        // Unpack result: high 32 bits = ptr, low 32 bits = len.
        let result_ptr = (result_packed >> 32) as usize;
        let result_len = (result_packed & 0xFFFF_FFFF) as usize;

        if result_len == 0 {
            return Err("wasm: module returned empty output".into());
        }
        if result_len > MAX_OUTPUT_BYTES {
            return Err(format!(
                "wasm: output too large ({result_len} bytes, max {MAX_OUTPUT_BYTES})"
            ));
        }

        // Read result from WASM memory.
        let mut result_buf = vec![0u8; result_len];
        memory
            .read(&store, result_ptr, &mut result_buf)
            .map_err(|e| format!("wasm: read output from memory: {e}"))?;

        // Output interpretation: blob or JSON.
        if config.blob_output {
            debug!(len = result_len, content_type = %config.output_content_type, "wasm: blob output");
            return Ok(KernelOutput::Blob {
                data: result_buf,
                content_type: config.output_content_type.clone(),
                shape: config.output_shape.clone(),
            });
        }

        // Parse as JSON.
        serde_json::from_slice(&result_buf).map_err(|e| {
            let preview = String::from_utf8_lossy(&result_buf[..result_len.min(200)]);
            format!("wasm: invalid JSON output: {e} (preview: {preview})")
        }).map(KernelOutput::Json)
    }
}

impl ComputeKernel for WasmKernel {
    fn name(&self) -> &str {
        "wasm"
    }

    fn execute(
        &self,
        input: KernelInput,
        operations: serde_json::Value,
    ) -> Result<KernelOutput, AxonError> {
        let config = parse_wasm_config(&operations)?;

        info!(
            module = %config.module_path.display(),
            fuel = config.fuel,
            memory_mb = config.memory_mb,
            blob_output = config.blob_output,
            has_blobs = input.has_blobs(),
            "wasm: executing module"
        );

        let module = self.get_or_compile(&config.module_path)?;

        // Build input bytes for the WASM module.
        let input_bytes = if input.has_blobs() {
            // Blob input: pack as binary envelope.
            // Format: [4-byte LE meta_len][JSON metadata][raw blob bytes]
            // WASM module reads meta_len from first 4 bytes, then JSON, then raw data.
            let blob = input.first_blob()
                .ok_or("wasm: has_blobs() but first_blob() is None")?;

            let meta = serde_json::json!({
                "shape": blob.meta.shape,
                "content_type": &blob.meta.content_type,
                "size": blob.meta.size,
                "json": input.json,
            });
            let meta_bytes = serde_json::to_vec(&meta)
                .map_err(|e| format!("wasm: serialize blob metadata: {e}"))?;

            let meta_len = (meta_bytes.len() as u32).to_le_bytes();
            let mut envelope = Vec::with_capacity(4 + meta_bytes.len() + blob.bytes.len());
            envelope.extend_from_slice(&meta_len);
            envelope.extend_from_slice(&meta_bytes);
            envelope.extend_from_slice(&blob.bytes);

            debug!(
                meta_len = meta_bytes.len(),
                blob_len = blob.bytes.len(),
                total_len = envelope.len(),
                "wasm: packed binary envelope"
            );
            envelope
        } else {
            // JSON input (standard path).
            let json = input.into_json();
            serde_json::to_vec(&json)
                .map_err(|e| format!("wasm: serialize input: {e}"))?
        };

        self.run_module(&module, &input_bytes, &config).map_err(Into::into)
    }
}

/// Classify a WASM trap into a user-friendly error message.
fn classify_trap(error: anyhow::Error, fuel_limit: u64) -> String {
    let msg = format!("{error:#}");
    if msg.contains("fuel") || msg.contains("Fuel") {
        format!(
            "wasm: fuel exhausted (limit: {fuel_limit} instructions). \
             Increase fuel in blueprint or optimize module."
        )
    } else if msg.contains("stack overflow") || msg.contains("call stack") {
        "wasm: stack overflow. Reduce recursion depth.".into()
    } else if msg.contains("out of bounds") {
        "wasm: memory access out of bounds.".into()
    } else {
        format!("wasm: execution failed: {msg}")
    }
}

// ── Config parsing ──────────────────────────────────────────────

/// Parse WASM configuration from the operations value.
fn parse_wasm_config(operations: &serde_json::Value) -> Result<WasmConfig, String> {
    match operations {
        serde_json::Value::String(s) => parse_config_string(s),
        serde_json::Value::Object(obj) => {
            let module_path = obj
                .get("module")
                .and_then(|v| v.as_str())
                .ok_or("wasm: operations object must have 'module' field")?;
            let blob_output = obj
                .get("blob_output")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let output_content_type = obj
                .get("output_content_type")
                .and_then(|v| v.as_str())
                .unwrap_or("tensor/f32")
                .to_string();
            let output_shape = obj
                .get("output_shape")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                });
            Ok(WasmConfig {
                module_path: PathBuf::from(module_path),
                fuel: obj.get("fuel").and_then(|v| v.as_u64()).unwrap_or(DEFAULT_FUEL),
                memory_mb: obj
                    .get("memory")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(DEFAULT_MEMORY_MB as u64) as usize,
                blob_output,
                output_content_type,
                output_shape,
            })
        }
        _ => Err("wasm: operations must be a module path string or config object".into()),
    }
}

/// Parse config from a string: single line = path, multi-line = key:value pairs.
fn parse_config_string(s: &str) -> Result<WasmConfig, String> {
    let trimmed = s.trim();

    // Single line without "module:" → treat entire string as module path.
    if !trimmed.contains('\n') && !trimmed.starts_with("module:") {
        return Ok(WasmConfig {
            module_path: PathBuf::from(trimmed),
            fuel: DEFAULT_FUEL,
            memory_mb: DEFAULT_MEMORY_MB,
            blob_output: false,
            output_content_type: "tensor/f32".to_string(),
            output_shape: None,
        });
    }

    // Multi-line key:value format.
    let mut module_path = None;
    let mut fuel = DEFAULT_FUEL;
    let mut memory_mb = DEFAULT_MEMORY_MB;
    let mut blob_output = false;

    for line in trimmed.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim().to_lowercase();
            let value = value.trim();
            match key.as_str() {
                "module" => module_path = Some(PathBuf::from(value)),
                "fuel" => {
                    fuel = value
                        .replace('_', "")
                        .parse::<u64>()
                        .map_err(|_| format!("wasm: invalid fuel value: {value}"))?;
                }
                "memory" => {
                    memory_mb = value
                        .parse::<usize>()
                        .map_err(|_| format!("wasm: invalid memory value: {value}"))?;
                }
                "blob_output" => {
                    blob_output = value == "true";
                }
                _ => {} // ignore unknown keys
            }
        }
    }

    Ok(WasmConfig {
        module_path: module_path.ok_or("wasm: missing 'module:' in config")?,
        fuel,
        memory_mb,
        blob_output,
        output_content_type: "tensor/f32".to_string(),
        output_shape: None,
    })
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel() -> WasmKernel {
        WasmKernel::new().expect("engine init")
    }

    fn default_config() -> WasmConfig {
        WasmConfig {
            module_path: PathBuf::new(),
            fuel: DEFAULT_FUEL,
            memory_mb: DEFAULT_MEMORY_MB,
            blob_output: false,
            output_content_type: "tensor/f32".to_string(),
            output_shape: None,
        }
    }

    /// WAT echo module: returns input JSON unchanged.
    const ECHO_WAT: &str = r#"
        (module
            (memory (export "memory") 2)
            (global $offset (mut i32) (i32.const 1024))

            (func (export "alloc") (param $size i32) (result i32)
                (local $ptr i32)
                (local.set $ptr (global.get $offset))
                (global.set $offset (i32.add (global.get $offset) (local.get $size)))
                (local.get $ptr)
            )

            (func (export "transform") (param $ptr i32) (param $len i32) (result i64)
                ;; Echo: return the same (ptr, len) as packed i64.
                (i64.or
                    (i64.shl (i64.extend_i32_u (local.get $ptr)) (i64.const 32))
                    (i64.extend_i32_u (local.get $len))
                )
            )
        )
    "#;

    /// WAT infinite loop module: will exhaust fuel.
    const LOOP_WAT: &str = r#"
        (module
            (memory (export "memory") 2)
            (global $offset (mut i32) (i32.const 1024))

            (func (export "alloc") (param $size i32) (result i32)
                (local $ptr i32)
                (local.set $ptr (global.get $offset))
                (global.set $offset (i32.add (global.get $offset) (local.get $size)))
                (local.get $ptr)
            )

            (func (export "transform") (param $ptr i32) (param $len i32) (result i64)
                (loop $inf (br $inf))
                (i64.const 0)
            )
        )
    "#;

    // ── Execution tests ──

    #[test]
    fn test_echo_module() {
        let k = kernel();
        let module = k.compile_bytes(ECHO_WAT.as_bytes()).unwrap();
        let input = serde_json::json!({"hello": "world", "n": 42});
        let input_bytes = serde_json::to_vec(&input).unwrap();
        let result = k
            .run_module(&module, &input_bytes, &default_config())
            .unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn test_echo_array_input() {
        let k = kernel();
        let module = k.compile_bytes(ECHO_WAT.as_bytes()).unwrap();
        let input = serde_json::json!([1, 2, 3, "four"]);
        let input_bytes = serde_json::to_vec(&input).unwrap();
        let result = k
            .run_module(&module, &input_bytes, &default_config())
            .unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn test_fuel_exhaustion() {
        let k = kernel();
        let module = k.compile_bytes(LOOP_WAT.as_bytes()).unwrap();
        let input_bytes = serde_json::to_vec(&serde_json::json!({"x": 1})).unwrap();
        let mut config = default_config();
        config.fuel = 1_000;
        let err = k
            .run_module(&module, &input_bytes, &config)
            .unwrap_err();
        assert!(
            err.contains("fuel"),
            "expected fuel error, got: {err}"
        );
    }

    #[test]
    fn test_missing_memory_export() {
        let k = kernel();
        let wat = r#"(module)"#;
        let module = k.compile_bytes(wat.as_bytes()).unwrap();
        let input_bytes = serde_json::to_vec(&serde_json::json!({"x": 1})).unwrap();
        let err = k
            .run_module(&module, &input_bytes, &default_config())
            .unwrap_err();
        assert!(err.contains("memory"), "expected memory error, got: {err}");
    }

    #[test]
    fn test_missing_alloc_export() {
        let k = kernel();
        let wat = r#"(module (memory (export "memory") 1))"#;
        let module = k.compile_bytes(wat.as_bytes()).unwrap();
        let input_bytes = serde_json::to_vec(&serde_json::json!({"x": 1})).unwrap();
        let err = k
            .run_module(&module, &input_bytes, &default_config())
            .unwrap_err();
        assert!(err.contains("alloc"), "expected alloc error, got: {err}");
    }

    #[test]
    fn test_missing_transform_export() {
        let k = kernel();
        let wat = r#"
            (module
                (memory (export "memory") 1)
                (func (export "alloc") (param i32) (result i32) (i32.const 0))
            )
        "#;
        let module = k.compile_bytes(wat.as_bytes()).unwrap();
        let input_bytes = serde_json::to_vec(&serde_json::json!({"x": 1})).unwrap();
        let err = k
            .run_module(&module, &input_bytes, &default_config())
            .unwrap_err();
        assert!(
            err.contains("transform"),
            "expected transform error, got: {err}"
        );
    }

    #[test]
    fn test_blob_output() {
        let k = kernel();
        let module = k.compile_bytes(ECHO_WAT.as_bytes()).unwrap();
        // When blob_output=true, the raw bytes from WASM are returned as-is.
        let raw_bytes: Vec<u8> = vec![1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let mut config = default_config();
        config.blob_output = true;
        config.output_content_type = "tensor/f32".to_string();
        config.output_shape = Some(vec![1, 3]);

        // Echo module returns exact same bytes — perfect for testing blob output path.
        let result = k.run_module(&module, &raw_bytes, &config).unwrap();
        match result {
            KernelOutput::Blob { data, content_type, shape } => {
                assert_eq!(content_type, "tensor/f32");
                assert_eq!(shape, Some(vec![1, 3]));
                let floats: Vec<f32> = data.chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                assert_eq!(floats, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("expected KernelOutput::Blob"),
        }
    }

    // ── Config parsing tests ──

    #[test]
    fn test_parse_config_simple_path() {
        let ops = serde_json::json!("plugins/my_module.wasm");
        let config = parse_wasm_config(&ops).unwrap();
        assert_eq!(config.module_path, PathBuf::from("plugins/my_module.wasm"));
        assert_eq!(config.fuel, DEFAULT_FUEL);
        assert_eq!(config.memory_mb, DEFAULT_MEMORY_MB);
    }

    #[test]
    fn test_parse_config_object() {
        let ops = serde_json::json!({
            "module": "plugins/heavy.wasm",
            "fuel": 5_000_000_u64,
            "memory": 128_u64,
        });
        let config = parse_wasm_config(&ops).unwrap();
        assert_eq!(config.module_path, PathBuf::from("plugins/heavy.wasm"));
        assert_eq!(config.fuel, 5_000_000);
        assert_eq!(config.memory_mb, 128);
    }

    #[test]
    fn test_parse_config_multiline() {
        let ops =
            serde_json::json!("module: plugins/transform.wasm\nfuel: 2_000_000\nmemory: 32");
        let config = parse_wasm_config(&ops).unwrap();
        assert_eq!(
            config.module_path,
            PathBuf::from("plugins/transform.wasm")
        );
        assert_eq!(config.fuel, 2_000_000);
        assert_eq!(config.memory_mb, 32);
    }

    #[test]
    fn test_parse_config_defaults() {
        let ops = serde_json::json!({"module": "test.wasm"});
        let config = parse_wasm_config(&ops).unwrap();
        assert_eq!(config.fuel, DEFAULT_FUEL);
        assert_eq!(config.memory_mb, DEFAULT_MEMORY_MB);
    }

    #[test]
    fn test_parse_config_missing_module() {
        let ops = serde_json::json!({"fuel": 1000});
        let err = parse_wasm_config(&ops).unwrap_err();
        assert!(err.contains("module"));
    }

    #[test]
    fn test_parse_config_blob_output() {
        let ops = serde_json::json!({
            "module": "plugin.wasm",
            "blob_output": true,
            "output_content_type": "tensor/f32",
            "output_shape": [1, 3, 640, 640],
        });
        let config = parse_wasm_config(&ops).unwrap();
        assert!(config.blob_output);
        assert_eq!(config.output_content_type, "tensor/f32");
        assert_eq!(config.output_shape, Some(vec![1, 3, 640, 640]));
    }

    // ── Real YOLO WASM plugin tests ──

    fn plugin_path(name: &str) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop(); // crates/
        path.pop(); // medulla-rse/
        path.push("plugins");
        path.push("bin");
        path.push(name);
        path
    }

    fn has_yolo_plugins() -> bool {
        plugin_path("yolo_postprocess.wasm").exists()
    }

    #[test]
    fn test_yolo_postprocess_plugin() {
        if !has_yolo_plugins() {
            eprintln!("skipping: plugins/bin/*.wasm not found");
            return;
        }
        let k = kernel();
        let module = k.get_or_compile(&plugin_path("yolo_postprocess.wasm")).unwrap();

        // Synthetic YOLOv8 output: [1, 84, 8400]
        // Create one clear detection at position (320, 320) with size (100, 100)
        // class 0 (person) with high confidence.
        let n_attrs = 84;
        let n_preds = 8400;
        let mut yolo_data = vec![0.0f32; n_attrs * n_preds];

        // Set prediction 0: cx=320, cy=320, w=100, h=100
        yolo_data[0 * n_preds + 0] = 320.0; // cx
        yolo_data[1 * n_preds + 0] = 320.0; // cy
        yolo_data[2 * n_preds + 0] = 100.0; // w
        yolo_data[3 * n_preds + 0] = 100.0; // h
        yolo_data[4 * n_preds + 0] = 0.95;  // class 0 score (person)

        // Set prediction 1: overlapping box → should be suppressed by NMS
        yolo_data[0 * n_preds + 1] = 325.0;
        yolo_data[1 * n_preds + 1] = 325.0;
        yolo_data[2 * n_preds + 1] = 100.0;
        yolo_data[3 * n_preds + 1] = 100.0;
        yolo_data[4 * n_preds + 1] = 0.85;

        // Set prediction 2: different class (cat=15), non-overlapping
        yolo_data[0 * n_preds + 2] = 100.0;
        yolo_data[1 * n_preds + 2] = 100.0;
        yolo_data[2 * n_preds + 2] = 50.0;
        yolo_data[3 * n_preds + 2] = 50.0;
        yolo_data[(4 + 15) * n_preds + 2] = 0.80; // class 15 (cat)

        // Build binary envelope.
        let raw_bytes: Vec<u8> = yolo_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let meta = serde_json::json!({
            "shape": [1, 84, 8400],
            "content_type": "tensor/f32",
            "size": raw_bytes.len(),
            "json": {},
        });
        let meta_bytes = serde_json::to_vec(&meta).unwrap();
        let meta_len = (meta_bytes.len() as u32).to_le_bytes();

        let mut envelope = Vec::new();
        envelope.extend_from_slice(&meta_len);
        envelope.extend_from_slice(&meta_bytes);
        envelope.extend_from_slice(&raw_bytes);

        // 2.8MB binary envelope + 8400 predictions → needs more fuel.
        let mut config = default_config();
        config.fuel = 500_000_000;
        config.memory_mb = 64;

        let result = k.run_module(&module, &envelope, &config).unwrap();
        let json = result.unwrap_json();

        let detections = json["detections"].as_array().unwrap();
        let count = json["count"].as_u64().unwrap();

        // Should have 2 detections: person (NMS suppresses duplicate) + cat
        assert_eq!(count, 2, "expected 2 detections after NMS, got {count}");
        assert_eq!(detections.len(), 2);

        // First detection: person (highest confidence)
        assert_eq!(detections[0]["class"], "person");
        assert!(detections[0]["confidence"].as_f64().unwrap() > 0.9);

        // Second detection: cat
        assert_eq!(detections[1]["class"], "cat");
        assert!(detections[1]["confidence"].as_f64().unwrap() > 0.7);

        eprintln!("YOLO postprocess result: {}", serde_json::to_string_pretty(&json).unwrap());
    }

    #[test]
    fn test_yolo_preprocess_plugin() {
        if !has_yolo_plugins() {
            eprintln!("skipping: plugins/bin/*.wasm not found");
            return;
        }
        let k = kernel();
        let module = k.get_or_compile(&plugin_path("yolo_preprocess.wasm")).unwrap();

        // Create a small 4x4 RGB image with known pixel values.
        let w = 4usize;
        let h = 4usize;
        let mut pixels = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                pixels[idx] = (x * 60) as u8;     // R
                pixels[idx + 1] = (y * 60) as u8;  // G
                pixels[idx + 2] = 128;              // B
            }
        }

        // Base64 encode.
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&pixels);

        let input = serde_json::json!({
            "pixels_base64": b64,
            "width": w,
            "height": h,
            "input_size": 8,  // tiny for testing
        });
        let input_bytes = serde_json::to_vec(&input).unwrap();

        // Use blob_output=true with high fuel for image processing.
        let mut config = default_config();
        config.blob_output = true;
        config.output_content_type = "tensor/f32".to_string();
        config.output_shape = Some(vec![1, 3, 8, 8]);
        config.fuel = 500_000_000;
        config.memory_mb = 128;

        let result = k.run_module(&module, &input_bytes, &config).unwrap();
        match result {
            KernelOutput::Blob { data, content_type, .. } => {
                assert_eq!(content_type, "tensor/f32");
                // Output = pure f32 tensor [1, 3, 8, 8] = 3 * 8 * 8 * 4 = 768 bytes
                assert_eq!(data.len(), 768, "expected exactly 768 bytes for [1,3,8,8]");

                // All values should be in [0, 1] range (normalized).
                let floats: Vec<f32> = data.chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                assert_eq!(floats.len(), 3 * 8 * 8);
                for (i, &f) in floats.iter().enumerate() {
                    assert!(f >= 0.0 && f <= 1.0,
                        "float[{i}] = {f} out of [0,1] range");
                }
            }
            _ => panic!("expected KernelOutput::Blob"),
        }
    }
}
