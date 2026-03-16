//! Benchmarks for image kernel operations.
//!
//! Run: `cargo bench -p axon --features vision --bench image_ops`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "vision")]
mod benches {
    use super::*;
    use axon::kernel::KernelInput;
    use axon::kernels::image::ImageKernel;
    use axon::ComputeKernel;
    use serde_json::json;

    fn make_image_input(h: usize, w: usize, c: usize) -> KernelInput {
        let pixels: Vec<f32> = (0..h * w * c).map(|i| (i % 256) as f32).collect();
        let bytes: Vec<u8> = pixels.iter().flat_map(|f| f.to_le_bytes()).collect();
        KernelInput {
            json: serde_json::Value::Null,
            blobs: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "image".to_string(),
                    axon::BlobData {
                        bytes,
                        meta: axon::BlobMeta {
                            size: 0,
                            content_type: "tensor/f32".to_string(),
                            shape: Some(vec![h, w, c]),
                        },
                    },
                );
                m
            },
        }
    }

    pub fn bench_layout_hwc_to_chw(c: &mut Criterion) {
        let kernel = ImageKernel;
        let mut group = c.benchmark_group("layout_hwc_to_chw");

        for (h, w) in [(224, 224), (640, 640), (1080, 1920)] {
            let ops = json!({"op": "layout", "to": "chw"});

            group.bench_with_input(
                BenchmarkId::new("rgb", format!("{h}x{w}")),
                &(h, w),
                |bench, &(h, w)| {
                    let input = make_image_input(h, w, 3);
                    bench.iter(|| {
                        kernel
                            .execute(black_box(input.clone()), black_box(ops.clone()))
                            .unwrap();
                    });
                },
            );
        }
        group.finish();
    }

    pub fn bench_normalize(c: &mut Criterion) {
        let kernel = ImageKernel;
        let ops = json!({
            "op": "normalize",
            "scale": 255.0,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        });

        c.bench_function("normalize_640x640", |bench| {
            let input = make_image_input(640, 640, 3);
            bench.iter(|| {
                kernel
                    .execute(black_box(input.clone()), black_box(ops.clone()))
                    .unwrap();
            });
        });
    }

    pub fn bench_colorspace(c: &mut Criterion) {
        let kernel = ImageKernel;
        let ops = json!({"op": "colorspace", "to": "grayscale"});

        c.bench_function("grayscale_640x640", |bench| {
            let input = make_image_input(640, 640, 3);
            bench.iter(|| {
                kernel
                    .execute(black_box(input.clone()), black_box(ops.clone()))
                    .unwrap();
            });
        });
    }
}

#[cfg(feature = "vision")]
criterion_group!(
    image_benches,
    benches::bench_layout_hwc_to_chw,
    benches::bench_normalize,
    benches::bench_colorspace,
);

#[cfg(feature = "vision")]
criterion_main!(image_benches);

#[cfg(not(feature = "vision"))]
fn main() {
    eprintln!("image_ops benchmarks require --features vision");
}
