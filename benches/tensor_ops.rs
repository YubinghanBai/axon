//! Benchmarks for tensor kernel operations.
//!
//! Run: `cargo bench -p axon --features onnx --bench tensor_ops`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "onnx")]
mod benches {
    use super::*;
    use axon::kernel::{KernelInput, KernelOutput};
    use axon::kernels::tensor::TensorKernel;
    use axon::ComputeKernel;
    use serde_json::json;

    fn make_blob_input(data: &[f32], shape: Vec<usize>) -> KernelInput {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        KernelInput {
            json: serde_json::Value::Null,
            blobs: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "data".to_string(),
                    axon::BlobData {
                        bytes,
                        meta: axon::BlobMeta {
                            size: 0,
                            content_type: "tensor/f32".to_string(),
                            shape: Some(shape),
                        },
                    },
                );
                m
            },
        }
    }

    pub fn bench_matmul(c: &mut Criterion) {
        let kernel = TensorKernel;
        let mut group = c.benchmark_group("matmul");

        for size in [32, 64, 128, 256] {
            let a: Vec<f32> = (0..size * size).map(|i| i as f32 * 0.001).collect();
            let b: Vec<f32> = (0..size * size).map(|i| i as f32 * 0.002).collect();

            let ops = json!({
                "op": "matmul",
                "other": {"data": b, "shape": [size, size]},
                "blob_output": true,
            });

            group.bench_with_input(BenchmarkId::new("square", size), &size, |bench, &sz| {
                let input = make_blob_input(&a, vec![sz, sz]);
                bench.iter(|| {
                    kernel.execute(black_box(input.clone()), black_box(ops.clone())).unwrap();
                });
            });
        }
        group.finish();
    }

    pub fn bench_softmax(c: &mut Criterion) {
        let kernel = TensorKernel;
        let mut group = c.benchmark_group("softmax");

        for n in [100, 1000, 10000] {
            let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
            let ops = json!({"op": "softmax", "dim": 0, "blob_output": true});

            group.bench_with_input(BenchmarkId::new("1d", n), &n, |bench, _| {
                let input = make_blob_input(&data, vec![n]);
                bench.iter(|| {
                    kernel.execute(black_box(input.clone()), black_box(ops.clone())).unwrap();
                });
            });
        }
        group.finish();
    }

    pub fn bench_mean_pool(c: &mut Criterion) {
        let kernel = TensorKernel;
        let mut group = c.benchmark_group("mean_pool");

        // Typical BERT: [1, 128, 384]
        let data: Vec<f32> = (0..1 * 128 * 384).map(|i| (i as f32) * 0.001).collect();
        let ops = json!({"op": "mean_pool", "dim": 1, "blob_output": true});

        group.bench_function("bert_1x128x384", |bench| {
            let input = make_blob_input(&data, vec![1, 128, 384]);
            bench.iter(|| {
                kernel.execute(black_box(input.clone()), black_box(ops.clone())).unwrap();
            });
        });
        group.finish();
    }

    pub fn bench_normalize(c: &mut Criterion) {
        let kernel = TensorKernel;

        // 384-dim embedding
        let data: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let ops = json!({"op": "normalize", "blob_output": true});

        c.bench_function("normalize_384d", |bench| {
            let input = make_blob_input(&data, vec![384]);
            bench.iter(|| {
                kernel.execute(black_box(input.clone()), black_box(ops.clone())).unwrap();
            });
        });
    }

    pub fn bench_gather(c: &mut Criterion) {
        let kernel = TensorKernel;

        let data: Vec<f32> = (0..100 * 50).map(|i| i as f32).collect();
        let indices: Vec<i64> = (0..100).flat_map(|_| vec![0i64, 10, 20, 30, 40]).collect();
        let ops = json!({
            "op": "gather",
            "dim": 1,
            "index": {"data": indices, "shape": [100, 5]},
            "blob_output": true,
        });

        c.bench_function("gather_100x50_to_100x5", |bench| {
            let input = make_blob_input(&data, vec![100, 50]);
            bench.iter(|| {
                kernel.execute(black_box(input.clone()), black_box(ops.clone())).unwrap();
            });
        });
    }
}

#[cfg(feature = "onnx")]
criterion_group!(
    tensor_benches,
    benches::bench_matmul,
    benches::bench_softmax,
    benches::bench_mean_pool,
    benches::bench_normalize,
    benches::bench_gather,
);

#[cfg(feature = "onnx")]
criterion_main!(tensor_benches);

#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("tensor_ops benchmarks require --features onnx");
}
