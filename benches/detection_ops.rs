//! Benchmarks for detection kernel operations.
//!
//! Run: `cargo bench -p axon --features vision --bench detection_ops`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "vision")]
mod benches {
    use super::*;
    use axon::kernel::KernelInput;
    use axon::kernels::detection::DetectionKernel;
    use axon::ComputeKernel;
    use serde_json::json;

    /// Generate N random-ish detection boxes with scores.
    fn make_detection_input(n: usize) -> KernelInput {
        let mut boxes = Vec::with_capacity(n);
        let mut scores = Vec::with_capacity(n);

        for i in 0..n {
            // Spread boxes across a 640x640 image with some overlap.
            let x = (i * 37 % 600) as f64;
            let y = (i * 53 % 600) as f64;
            let w = 20.0 + (i * 7 % 40) as f64;
            let h = 20.0 + (i * 11 % 40) as f64;
            boxes.push(vec![x, y, x + w, y + h]);
            scores.push(0.1 + (i * 13 % 90) as f64 / 100.0);
        }

        KernelInput::from_json(json!({
            "boxes": boxes,
            "scores": scores,
        }))
    }

    pub fn bench_nms(c: &mut Criterion) {
        let kernel = DetectionKernel;
        let mut group = c.benchmark_group("nms");
        let ops = json!({"op": "nms", "iou": 0.45});

        for n in [100, 500, 1000, 5000] {
            group.bench_with_input(BenchmarkId::new("boxes", n), &n, |bench, &n| {
                let input = make_detection_input(n);
                bench.iter(|| {
                    kernel
                        .execute(black_box(input.clone()), black_box(ops.clone()))
                        .unwrap();
                });
            });
        }
        group.finish();
    }

    pub fn bench_soft_nms(c: &mut Criterion) {
        let kernel = DetectionKernel;
        let ops = json!({"op": "soft_nms", "iou": 0.3, "sigma": 0.5, "method": "gaussian"});

        c.bench_function("soft_nms_1000", |bench| {
            let input = make_detection_input(1000);
            bench.iter(|| {
                kernel
                    .execute(black_box(input.clone()), black_box(ops.clone()))
                    .unwrap();
            });
        });
    }

    pub fn bench_confidence_filter(c: &mut Criterion) {
        let kernel = DetectionKernel;
        let ops = json!({"op": "confidence_filter", "threshold": 0.25});

        c.bench_function("confidence_filter_5000", |bench| {
            let input = make_detection_input(5000);
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
    detection_benches,
    benches::bench_nms,
    benches::bench_soft_nms,
    benches::bench_confidence_filter,
);

#[cfg(feature = "vision")]
criterion_main!(detection_benches);

#[cfg(not(feature = "vision"))]
fn main() {
    eprintln!("detection_ops benchmarks require --features vision");
}
