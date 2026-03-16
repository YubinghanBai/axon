//! gRPC inference service for Axon (KServe V2 inspired).
//!
//! Provides a tonic-based gRPC server that runs alongside the HTTP server
//! on a separate port. Uses the same `Pipeline` infrastructure as REST.
//!
//! ## Usage
//!
//! ```bash
//! axon serve --pipeline whisper=manifest.toml --grpc-port 8081
//!
//! # Python client:
//! # channel = grpc.insecure_channel("localhost:8081")
//! # stub = AxonInferenceStub(channel)
//! # resp = stub.ModelInfer(ModelInferRequest(model_name="whisper", raw_input=audio_bytes, content_type="audio/wav"))
//! ```
//!
//! ## Compile
//!
//! ```bash
//! cargo build --features grpc
//! ```

use std::time::Instant;

use tonic::{Request, Response, Status};
use tracing::info_span;

use crate::serve::AppState;

// Generated from proto/axon_infer.proto by tonic-build.
pub mod pb {
    tonic::include_proto!("axon.v2");
}

use pb::axon_inference_server::AxonInference;
use pb::*;

/// gRPC service implementation backed by Axon pipelines.
pub struct AxonGrpcService {
    state: AppState,
}

impl AxonGrpcService {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl AxonInference for AxonGrpcService {
    async fn server_live(
        &self,
        _req: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _req: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        Ok(Response::new(ServerReadyResponse { ready: true }))
    }

    async fn model_ready(
        &self,
        req: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        let name = &req.get_ref().name;
        let ready = self.state.pipelines.contains_key(name);
        Ok(Response::new(ModelReadyResponse { ready }))
    }

    async fn model_metadata(
        &self,
        req: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let name = &req.get_ref().name;
        let pipeline = self
            .state
            .pipelines
            .get(name)
            .ok_or_else(|| Status::not_found(format!("pipeline '{name}' not found")))?;

        let manifest = pipeline.manifest();
        let pre_steps: Vec<String> = manifest
            .pre
            .as_ref()
            .map(|p| p.steps.iter().map(|s| s.op.clone()).collect())
            .unwrap_or_default();
        let post_steps: Vec<String> = manifest
            .post
            .as_ref()
            .map(|p| p.steps.iter().map(|s| s.op.clone()).collect())
            .unwrap_or_default();

        Ok(Response::new(ModelMetadataResponse {
            name: name.clone(),
            platform: "axon".to_string(),
            kernels: pipeline.required_kernels(),
            pre_steps,
            post_steps,
        }))
    }

    async fn model_infer(
        &self,
        req: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let req = req.into_inner();

        let pipeline = self
            .state
            .pipelines
            .get(&req.model_name)
            .ok_or_else(|| {
                Status::not_found(format!("pipeline '{}' not found", req.model_name))
            })?
            .clone();

        if req.raw_input.is_empty() {
            return Err(Status::invalid_argument("raw_input is empty"));
        }

        let content_type = if req.content_type.is_empty() {
            "application/octet-stream".to_string()
        } else {
            req.content_type.clone()
        };

        let model_name = req.model_name.clone();
        let request_id = req.id.clone();
        let input = req.raw_input;
        let start = Instant::now();

        // Run pipeline on blocking thread.
        let result = tokio::task::spawn_blocking(move || {
            let _span = info_span!(
                "grpc_infer",
                pipeline = %model_name,
                content_type = %content_type,
                size = input.len(),
            )
            .entered();
            pipeline.run(&input, &content_type)
        })
        .await
        .map_err(|e| Status::internal(format!("task panicked: {e}")))?;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(output) => {
                let mut resp = ModelInferResponse {
                    model_name: req.model_name,
                    id: request_id,
                    latency_ms,
                    ..Default::default()
                };

                match output {
                    crate::KernelOutput::Json(value) => {
                        resp.json_output = serde_json::to_string(&value)
                            .unwrap_or_else(|_| "{}".to_string());
                    }
                    crate::KernelOutput::Blob {
                        data,
                        content_type,
                        shape,
                    } => {
                        resp.raw_output = data;
                        resp.output_content_type = content_type;
                        resp.shape = shape
                            .unwrap_or_default()
                            .iter()
                            .map(|&d| d as i64)
                            .collect();
                    }
                }

                Ok(Response::new(resp))
            }
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }
}

/// Start the gRPC server on the given port.
///
/// This should be `tokio::spawn`'d alongside the HTTP server.
pub async fn run_grpc_server(
    state: AppState,
    host: &str,
    port: u16,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = format!("{host}:{port}").parse()?;
    let service = AxonGrpcService::new(state);

    tracing::info!(%addr, "gRPC server started");
    eprintln!("gRPC listening on {addr}");

    tonic::transport::Server::builder()
        .add_service(pb::axon_inference_server::AxonInferenceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
