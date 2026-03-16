//! # Axon — Standalone ML Inference Runtime
//!
//! Axon is the compute engine extracted from the Medulla workflow engine.
//! It provides ONNX model inference, native tensor operations, WASM-sandboxed
//! pre/post processing, and a content-addressed blob pipeline — all in pure Rust,
//! with zero Python dependencies.
//!
//! ## Crate design constraint
//!
//! **Axon must NEVER depend on any `medulla-*` crate.** Medulla depends on Axon,
//! not the other way around. This ensures Axon can be used as a standalone library
//! and CLI tool independent of the Medulla workflow engine.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use axon::{KernelRegistry, KernelInput, KernelOutput};
//!
//! let registry = KernelRegistry::new();
//! // register kernels...
//! let kernel = registry.get("onnx").unwrap();
//! let output = kernel.execute(KernelInput::from_json(input), ops)?;
//! ```

pub mod blob;
pub mod error;
pub mod kernel;
pub mod kernels;
pub mod manifest;
pub mod pipeline;
pub mod pool;
#[cfg(feature = "serve")]
pub mod batch;
#[cfg(feature = "serve")]
pub mod serve;
#[cfg(feature = "grpc")]
pub mod grpc;
#[cfg(feature = "otel")]
pub mod otel;

// ── Convenience re-exports ─────────────────────────────────────

pub use blob::{BlobId, BlobMeta, BlobRef, BlobStore};
pub use error::AxonError;
pub use kernel::{BlobData, ComputeKernel, KernelInput, KernelOutput, KernelRegistry};
pub use manifest::Manifest;
pub use pipeline::{Pipeline, PipelineError};
pub use pool::ModelPool;
