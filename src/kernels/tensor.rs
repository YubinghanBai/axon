//! Tensor compute kernel: native Rust tensor operations via ndarray.
//!
//! Enabled alongside `onnx` feature (shares ndarray dependency).
//!
//! Designed for zero-copy blob pipelines:
//!   ONNX(blob output) → BlobStore → tensor_kernel(raw f32 bytes) → JSON embedding
//!
//! Falls back to JSON parsing when blob input is not available.
//!
//! Supported operations:
//!   - `mean_pool`: average across a dimension (embedding extraction)
//!   - `normalize`: L2 normalization
//!   - `reshape`: change tensor shape
//!   - `transpose`: reorder dimensions
//!   - `softmax`: probability normalization
//!   - `argmax`: index of max value
//!   - `topk`: top-K values and indices
//!   - `unsqueeze`: add a dimension
//!   - `squeeze`: remove size-1 dimensions
//!   - `clamp`: clamp values to a range
//!
//! Config (operations):
//!   - String: `"mean_pool"` (defaults: dim=1)
//!   - Object: `{"op": "mean_pool", "dim": 1, "blob_output": true}`

use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};
use serde_json::Value;
use tracing::debug;

use crate::error::AxonError;
use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};

/// Native Rust tensor operations kernel.
pub struct TensorKernel;

/// Reduction mode for reduce operations.
#[derive(Clone, Copy, Debug)]
enum ReduceMode {
    Sum,
    Max,
    Min,
    Prod,
}

/// Target data type for cast operation.
#[derive(Clone, Debug)]
enum CastDtype {
    F32,
    F64,
    I64,
    F16,
}

/// Parsed tensor operation.
enum TensorOp {
    /// Average values across a dimension. Reduces shape[dim].
    MeanPool { dim: usize },
    /// L2 normalization along the last dimension.
    Normalize,
    /// Reshape tensor to new dimensions. Total elements must match.
    Reshape { shape: Vec<usize> },
    /// Reorder dimensions. axes.len() must equal ndim.
    Transpose { axes: Vec<usize> },
    /// Softmax along a dimension (numerically stable).
    Softmax { dim: i64 },
    /// Index of maximum value along a dimension.
    Argmax { dim: i64 },
    /// Top-K values and indices along a dimension.
    TopK { k: usize, dim: i64 },
    /// Add a dimension of size 1 at position.
    Unsqueeze { dim: i64 },
    /// Remove all dimensions of size 1, or a specific one.
    Squeeze { dim: Option<i64> },
    /// Clamp values to [min, max].
    Clamp { min: f32, max: f32 },
    /// Concatenate with additional tensors along a dimension.
    Concat {
        dim: i64,
        others: Vec<(Vec<f32>, Vec<usize>)>,
    },
    /// Slice tensor with per-dimension [start, end) ranges (supports negative indices).
    Slice { ranges: Vec<(i64, i64)> },
    /// Gather elements along a dimension (torch.gather semantics).
    Gather {
        dim: i64,
        index_data: Vec<i64>,
        index_shape: Vec<usize>,
    },
    /// Cast to a different numeric type.
    Cast { to: CastDtype },
    /// Reduce along a dimension: sum, max, min, prod.
    Reduce {
        mode: ReduceMode,
        dim: Option<i64>,
        keepdim: bool,
    },
    /// Conditional selection: where(condition > 0, self, y).
    WhereSelect {
        condition: Vec<f32>,
        y_data: Vec<f32>,
        y_shape: Vec<usize>,
    },
    /// Matrix multiplication (2D).
    MatMul {
        other_data: Vec<f32>,
        other_shape: Vec<usize>,
    },
}

/// Parsed config with operation + output mode.
struct TensorConfig {
    op: TensorOp,
    /// When true, output raw f32 bytes as KernelOutput::Blob.
    blob_output: bool,
}

impl ComputeKernel for TensorKernel {
    fn name(&self) -> &str {
        "tensor"
    }

    fn execute(
        &self,
        input: KernelInput,
        operations: Value,
    ) -> Result<KernelOutput, AxonError> {
        let config = parse_config(&operations)?;

        // Get tensor data: prefer raw blob bytes, fallback to JSON.
        let (data, shape) = if let Some(blob) = input.first_blob() {
            let floats = bytes_to_f32(&blob.bytes, &blob.meta.content_type)?;
            let shape = blob.meta.shape.clone()
                .ok_or("tensor: blob missing shape metadata")?;
            debug!(shape = ?shape, n_floats = floats.len(), "tensor: using raw blob input");
            (floats, shape)
        } else {
            // Fallback: parse tensor from JSON.
            debug!("tensor: parsing tensor from JSON input");
            parse_tensor_from_json(&input.json)?
        };

        // Validate shape matches data.
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(format!(
                "tensor: shape {:?} expects {} elements, got {}",
                shape, expected, data.len()
            ).into());
        }

        match config.op {
            TensorOp::MeanPool { dim } => mean_pool(&data, &shape, dim, config.blob_output),
            TensorOp::Normalize => normalize(&data, &shape, config.blob_output),
            TensorOp::Reshape { shape: new_shape } => reshape(&data, &shape, &new_shape, config.blob_output),
            TensorOp::Transpose { axes } => transpose(&data, &shape, &axes, config.blob_output),
            TensorOp::Softmax { dim } => softmax(&data, &shape, dim, config.blob_output),
            TensorOp::Argmax { dim } => argmax(&data, &shape, dim),
            TensorOp::TopK { k, dim } => topk(&data, &shape, k, dim),
            TensorOp::Unsqueeze { dim } => unsqueeze(&data, &shape, dim, config.blob_output),
            TensorOp::Squeeze { dim } => squeeze(&data, &shape, dim, config.blob_output),
            TensorOp::Clamp { min, max } => clamp(&data, &shape, min, max, config.blob_output),
            TensorOp::Concat { dim, others } => concat(&data, &shape, dim, &others, config.blob_output),
            TensorOp::Slice { ranges } => tensor_slice(&data, &shape, &ranges, config.blob_output),
            TensorOp::Gather { dim, index_data, index_shape } => gather(&data, &shape, dim, &index_data, &index_shape, config.blob_output),
            TensorOp::Cast { to } => cast(&data, &shape, &to),
            TensorOp::Reduce { mode, dim, keepdim } => reduce(&data, &shape, mode, dim, keepdim, config.blob_output),
            TensorOp::WhereSelect { condition, y_data, y_shape } => where_select(&data, &shape, &condition, &y_data, &y_shape, config.blob_output),
            TensorOp::MatMul { other_data, other_shape } => matmul(&data, &shape, &other_data, &other_shape, config.blob_output),
        }.map_err(Into::into)
    }
}

// ── Operations ─────────────────────────────────────────────────

/// Mean pool: average across `dim` using ndarray with f64 accumulation.
///
/// For a [batch, seq_len, hidden_dim] tensor with dim=1:
///   output shape = [batch, hidden_dim]
///   each output[b][h] = mean(input[b][0..seq_len][h])
///
/// Uses ndarray's `mean_axis` for SIMD acceleration and numerical stability.
fn mean_pool(
    data: &[f32],
    shape: &[usize],
    dim: usize,
    blob_output: bool,
) -> Result<KernelOutput, String> {
    if dim >= shape.len() {
        return Err(format!(
            "tensor: mean_pool dim={} out of range for shape {:?}",
            dim, shape
        ));
    }
    if shape[dim] == 0 {
        return Err("tensor: mean_pool over empty dimension".into());
    }

    // Build ndarray view over the data.
    let view = ArrayViewD::<f32>::from_shape(IxDyn(shape), data)
        .map_err(|e| format!("tensor: ndarray view: {e}"))?;

    // Cast to f64 for accumulation precision, then mean along axis.
    let f64_array = view.mapv(|x| x as f64);
    let pooled = f64_array
        .mean_axis(Axis(dim))
        .ok_or("tensor: mean_pool failed (empty axis)")?;

    // Convert back to f32.
    let out: Vec<f32> = pooled.iter().map(|&x| x as f32).collect();
    let out_shape: Vec<usize> = pooled.shape().to_vec();

    if blob_output {
        return Ok(KernelOutput::Blob {
            data: out.iter().flat_map(|f| f.to_le_bytes()).collect(),
            content_type: "tensor/f32".to_string(),
            shape: Some(out_shape),
        });
    }

    Ok(KernelOutput::Json(serde_json::json!({
        "embedding": out,
        "shape": out_shape,
        "dim": out_shape.last().unwrap_or(&0),
    })))
}

/// L2 normalization along the last dimension with f64 accumulation.
fn normalize(
    data: &[f32],
    shape: &[usize],
    blob_output: bool,
) -> Result<KernelOutput, String> {
    if shape.is_empty() {
        return Err("tensor: normalize on empty shape".into());
    }

    let last_dim = *shape.last().unwrap();
    if last_dim == 0 {
        return Err("tensor: normalize over empty last dimension".into());
    }

    let mut out = data.to_vec();
    let n_vectors = data.len() / last_dim;

    for i in 0..n_vectors {
        let start = i * last_dim;
        let end = start + last_dim;
        let slice = &mut out[start..end];

        // f64 accumulation for numerical stability.
        let norm: f64 = slice.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        if norm > 1e-12 {
            let inv_norm = (1.0 / norm) as f32;
            for x in slice.iter_mut() {
                *x *= inv_norm;
            }
        }
    }

    if blob_output {
        return Ok(KernelOutput::Blob {
            data: out.iter().flat_map(|f| f.to_le_bytes()).collect(),
            content_type: "tensor/f32".to_string(),
            shape: Some(shape.to_vec()),
        });
    }

    Ok(KernelOutput::Json(serde_json::json!({
        "data": out,
        "shape": shape,
    })))
}

/// Reshape tensor (total elements must match).
fn reshape(
    data: &[f32],
    old_shape: &[usize],
    new_shape: &[usize],
    blob_output: bool,
) -> Result<KernelOutput, String> {
    let old_size: usize = old_shape.iter().product();
    let new_size: usize = new_shape.iter().product();
    if old_size != new_size {
        return Err(format!(
            "tensor: reshape {:?} → {:?}: element count mismatch ({} vs {})",
            old_shape, new_shape, old_size, new_size
        ));
    }

    if blob_output {
        return Ok(KernelOutput::Blob {
            data: data.iter().flat_map(|f| f.to_le_bytes()).collect(),
            content_type: "tensor/f32".to_string(),
            shape: Some(new_shape.to_vec()),
        });
    }

    Ok(KernelOutput::Json(serde_json::json!({
        "data": data,
        "shape": new_shape,
    })))
}

/// Transpose: reorder dimensions.
///
/// Zero-copy when possible via ndarray's `permuted_axes`.
fn transpose(
    data: &[f32],
    shape: &[usize],
    axes: &[usize],
    blob_output: bool,
) -> Result<KernelOutput, String> {
    if axes.len() != shape.len() {
        return Err(format!(
            "tensor: transpose axes {:?} length doesn't match shape {:?}",
            axes, shape
        ));
    }

    let view = ArrayViewD::<f32>::from_shape(IxDyn(shape), data)
        .map_err(|e| format!("tensor: ndarray view: {e}"))?;

    let transposed = view.permuted_axes(IxDyn(axes));
    let out_shape: Vec<usize> = transposed.shape().to_vec();
    // Materialize into contiguous memory (permuted_axes returns a view).
    let out: Vec<f32> = transposed.iter().copied().collect();

    emit_tensor(&out, &out_shape, blob_output)
}

/// Softmax: numerically stable exp(x - max) / sum(exp(x - max)).
///
/// Operates along `dim` (supports negative indexing).
/// Uses f64 accumulation for precision.
fn softmax(
    data: &[f32],
    shape: &[usize],
    dim: i64,
    blob_output: bool,
) -> Result<KernelOutput, String> {
    let ndim = shape.len();
    let axis = resolve_dim(dim, ndim)?;
    let axis_len = shape[axis];

    if axis_len == 0 {
        return Err("tensor: softmax over empty dimension".into());
    }

    let view = ArrayViewD::<f32>::from_shape(IxDyn(shape), data)
        .map_err(|e| format!("tensor: ndarray view: {e}"))?;

    // Compute in f64 for numerical stability.
    let f64_view = view.mapv(|x| x as f64);

    // For each slice along `axis`: subtract max, exp, divide by sum.
    // Use insert_axis + broadcasting to handle all dimensionalities correctly.
    let max_vals = f64_view
        .map_axis(Axis(axis), |lane| {
            lane.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        });

    // Broadcast max back: insert the reduced axis, then subtract + exp.
    let max_broadcast = max_vals.insert_axis(Axis(axis));
    let mut result = (&f64_view - &max_broadcast).mapv(f64::exp);

    // Normalize by sum along axis (broadcast division).
    let sums = result.sum_axis(Axis(axis));
    let sums_broadcast = sums.insert_axis(Axis(axis));
    result /= &sums_broadcast;

    let out: Vec<f32> = result.iter().map(|&x| x as f32).collect();
    emit_tensor(&out, shape, blob_output)
}

/// Argmax: index of maximum value along a dimension.
///
/// Returns integer indices (always JSON output — indices aren't tensors).
fn argmax(
    data: &[f32],
    shape: &[usize],
    dim: i64,
) -> Result<KernelOutput, String> {
    let ndim = shape.len();
    let axis = resolve_dim(dim, ndim)?;

    let view = ArrayViewD::<f32>::from_shape(IxDyn(shape), data)
        .map_err(|e| format!("tensor: ndarray view: {e}"))?;

    let indices = view.map_axis(Axis(axis), |lane| {
        lane.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i64)
            .unwrap_or(0)
    });

    let out_shape: Vec<usize> = indices.shape().to_vec();
    let out_data: Vec<i64> = indices.iter().copied().collect();

    Ok(KernelOutput::Json(serde_json::json!({
        "indices": out_data,
        "shape": out_shape,
    })))
}

/// Top-K: find the K largest values and their indices along a dimension.
///
/// Uses partial sort for O(n log k) performance.
fn topk(
    data: &[f32],
    shape: &[usize],
    k: usize,
    dim: i64,
) -> Result<KernelOutput, String> {
    let ndim = shape.len();
    let axis = resolve_dim(dim, ndim)?;
    let axis_len = shape[axis];

    if k > axis_len {
        return Err(format!(
            "tensor: topk k={k} > axis length {axis_len}"
        ));
    }

    let view = ArrayViewD::<f32>::from_shape(IxDyn(shape), data)
        .map_err(|e| format!("tensor: ndarray view: {e}"))?;

    let mut all_values: Vec<Vec<f32>> = Vec::new();
    let mut all_indices: Vec<Vec<i64>> = Vec::new();

    // For each lane along the axis, find top-K via partial sort.
    view.map_axis(Axis(axis), |lane| {
        let mut indexed: Vec<(usize, f32)> = lane
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        let len = indexed.len();
        let kk = k.min(len);
        if kk > 0 {
            // Partial sort: O(n) to partition top-K elements to the front.
            indexed.select_nth_unstable_by(kk - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        indexed.truncate(kk);
        // Sort the top-K by value descending.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let values: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
        let indices: Vec<i64> = indexed.iter().map(|(i, _)| *i as i64).collect();
        all_values.push(values);
        all_indices.push(indices);
    });

    // Build output shape: replace axis_len with k.
    let mut out_shape = shape.to_vec();
    out_shape[axis] = k;

    Ok(KernelOutput::Json(serde_json::json!({
        "values": all_values,
        "indices": all_indices,
        "shape": out_shape,
        "k": k,
    })))
}

/// Unsqueeze: insert a dimension of size 1.
///
/// `dim` supports negative indexing (-1 = after last dim).
fn unsqueeze(
    data: &[f32],
    shape: &[usize],
    dim: i64,
    blob_output: bool,
) -> Result<KernelOutput, String> {
    let ndim = shape.len();
    // For unsqueeze, valid range is [0, ndim] (can insert after last).
    let pos = if dim >= 0 {
        dim as usize
    } else {
        ((ndim as i64) + dim + 1) as usize
    };

    if pos > ndim {
        return Err(format!(
            "tensor: unsqueeze dim={dim} out of range for {ndim}-D tensor"
        ));
    }

    let mut new_shape = shape.to_vec();
    new_shape.insert(pos, 1);

    emit_tensor(data, &new_shape, blob_output)
}

/// Squeeze: remove dimensions of size 1.
///
/// If `dim` is None, remove all size-1 dims. Otherwise remove a specific one.
fn squeeze(
    data: &[f32],
    shape: &[usize],
    dim: Option<i64>,
    blob_output: bool,
) -> Result<KernelOutput, String> {
    let new_shape: Vec<usize> = match dim {
        None => shape.iter().copied().filter(|&s| s != 1).collect(),
        Some(d) => {
            let ndim = shape.len();
            let axis = resolve_dim(d, ndim)?;
            if shape[axis] != 1 {
                return Err(format!(
                    "tensor: squeeze dim={d} has size {}, not 1",
                    shape[axis]
                ));
            }
            let mut s = shape.to_vec();
            s.remove(axis);
            s
        }
    };

    // Guard against squeezing to scalar (empty shape).
    let new_shape = if new_shape.is_empty() {
        vec![1]
    } else {
        new_shape
    };

    emit_tensor(data, &new_shape, blob_output)
}

/// Clamp: restrict values to [min, max].
///
/// In-place for performance, no allocation beyond output buffer.
fn clamp(
    data: &[f32],
    shape: &[usize],
    min: f32,
    max: f32,
    blob_output: bool,
) -> Result<KernelOutput, String> {
    let out: Vec<f32> = data.iter().map(|&x| x.clamp(min, max)).collect();
    emit_tensor(&out, shape, blob_output)
}

/// Concatenate primary tensor with additional tensors along a dimension.
///
/// The primary tensor is the first in the concatenation sequence.
/// Additional tensors must have matching shapes on all non-concat dimensions.
fn concat(
    data: &[f32],
    shape: &[usize],
    dim: i64,
    others: &[(Vec<f32>, Vec<usize>)],
    blob_output: bool,
) -> Result<KernelOutput, String> {
    if others.is_empty() {
        return emit_tensor(data, shape, blob_output);
    }

    let ndim = shape.len();
    let axis = resolve_dim(dim, ndim)?;

    // Validate all tensors have compatible shapes
    for (i, (_, other_shape)) in others.iter().enumerate() {
        if other_shape.len() != ndim {
            return Err(format!(
                "tensor: concat tensor {} has {} dims, expected {}",
                i + 1,
                other_shape.len(),
                ndim
            ));
        }
        for d in 0..ndim {
            if d != axis && other_shape[d] != shape[d] {
                return Err(format!(
                    "tensor: concat dim {} mismatch at tensor {}: {} vs {}",
                    d,
                    i + 1,
                    shape[d],
                    other_shape[d]
                ));
            }
        }
    }

    // Build ndarray views and concatenate
    let primary_view = ArrayViewD::<f32>::from_shape(IxDyn(shape), data)
        .map_err(|e| format!("tensor: ndarray view: {e}"))?;

    let other_views: Vec<ArrayViewD<f32>> = others
        .iter()
        .map(|(d, s)| {
            ArrayViewD::from_shape(IxDyn(s), d.as_slice())
                .map_err(|e| format!("tensor: ndarray view: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut all_views: Vec<ArrayViewD<f32>> = Vec::with_capacity(1 + others.len());
    all_views.push(primary_view.view());
    for v in &other_views {
        all_views.push(v.view());
    }

    let result = ndarray::concatenate(Axis(axis), &all_views)
        .map_err(|e| format!("tensor: concat failed: {e}"))?;

    let out: Vec<f32> = result.iter().copied().collect();
    let out_shape: Vec<usize> = result.shape().to_vec();

    emit_tensor(&out, &out_shape, blob_output)
}

/// Slice tensor with per-dimension [start, end) ranges.
///
/// Supports negative indices: -1 means last element, -2 second to last, etc.
/// Ranges are clamped to valid bounds.
fn tensor_slice(
    data: &[f32],
    shape: &[usize],
    ranges: &[(i64, i64)],
    blob_output: bool,
) -> Result<KernelOutput, String> {
    let ndim = shape.len();
    if ranges.len() != ndim {
        return Err(format!(
            "tensor: slice got {} ranges for {}-D tensor",
            ranges.len(),
            ndim
        ));
    }

    let view = ArrayViewD::<f32>::from_shape(IxDyn(shape), data)
        .map_err(|e| format!("tensor: ndarray view: {e}"))?;

    // Resolve negative indices and validate
    let resolved: Vec<(isize, isize)> = ranges
        .iter()
        .enumerate()
        .map(|(d, &(start, end))| {
            let dim_size = shape[d] as i64;
            let s = if start < 0 {
                (dim_size + start).max(0)
            } else {
                start.min(dim_size)
            };
            let e = if end < 0 {
                (dim_size + end).max(0)
            } else {
                end.min(dim_size)
            };
            if s > e {
                Err(format!(
                    "tensor: slice dim {} range [{}, {}) invalid",
                    d, s, e
                ))
            } else {
                Ok((s as isize, e as isize))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let sliced = view.slice_each_axis(|ax| {
        let (s, e) = resolved[ax.axis.index()];
        ndarray::Slice::from(s..e)
    });

    let out: Vec<f32> = sliced.iter().copied().collect();
    let out_shape: Vec<usize> = sliced.shape().to_vec();

    emit_tensor(&out, &out_shape, blob_output)
}

/// Gather elements along a dimension (torch.gather semantics).
///
/// For dim=d: out[i0,...,id,...,in] = input[i0,...,index[i0,...,id,...,in],...,in]
///
/// Uses manual stride computation for N-dimensional support.
fn gather(
    data: &[f32],
    shape: &[usize],
    dim: i64,
    index_data: &[i64],
    index_shape: &[usize],
    blob_output: bool,
) -> Result<KernelOutput, String> {
    let ndim = shape.len();
    let axis = resolve_dim(dim, ndim)?;

    if index_shape.len() != ndim {
        return Err(format!(
            "tensor: gather index has {} dims, expected {}",
            index_shape.len(),
            ndim
        ));
    }

    let total_out: usize = index_shape.iter().product();
    if index_data.len() != total_out {
        return Err(format!(
            "tensor: gather index shape {:?} expects {} elements, got {}",
            index_shape, total_out, index_data.len()
        ));
    }

    // Compute strides for source tensor
    let mut data_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        data_strides[d] = data_strides[d + 1] * shape[d + 1];
    }

    let mut idx_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        idx_strides[d] = idx_strides[d + 1] * index_shape[d + 1];
    }

    let mut out = Vec::with_capacity(total_out);

    // Stack-allocated index buffer (avoid per-element heap alloc).
    // Most tensors are ≤8D; fall back to heap for exotic cases.
    const STACK_DIMS: usize = 8;
    let mut stack_idx = [0usize; STACK_DIMS];
    let use_stack = ndim <= STACK_DIMS;

    // Heap fallback for high-dimensional tensors.
    let mut heap_idx = if use_stack { Vec::new() } else { vec![0usize; ndim] };

    for flat_idx in 0..total_out {
        // Convert flat index to multi-dimensional index (stack or heap).
        let multi_idx: &mut [usize] = if use_stack {
            &mut stack_idx[..ndim]
        } else {
            &mut heap_idx
        };
        let mut rem = flat_idx;
        for d in 0..ndim {
            multi_idx[d] = rem / idx_strides[d];
            rem %= idx_strides[d];
        }

        // Replace axis dimension with the gathered index
        let gather_idx = index_data[flat_idx];
        let gather_idx = if gather_idx < 0 {
            (shape[axis] as i64 + gather_idx) as usize
        } else {
            gather_idx as usize
        };

        if gather_idx >= shape[axis] {
            return Err(format!(
                "tensor: gather index {} out of range for dim {} (size {})",
                index_data[flat_idx], axis, shape[axis]
            ));
        }

        multi_idx[axis] = gather_idx;

        let src_flat: usize = multi_idx
            .iter()
            .zip(data_strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();

        out.push(data[src_flat]);
    }

    emit_tensor(&out, index_shape, blob_output)
}

/// Cast tensor to a different numeric type.
///
/// Always outputs as KernelOutput::Blob with the appropriate content_type.
/// Supports: f32, f64, i64, f16 (via half crate).
fn cast(data: &[f32], shape: &[usize], to: &CastDtype) -> Result<KernelOutput, String> {
    match to {
        CastDtype::F32 => Ok(KernelOutput::Blob {
            data: data.iter().flat_map(|f| f.to_le_bytes()).collect(),
            content_type: "tensor/f32".to_string(),
            shape: Some(shape.to_vec()),
        }),
        CastDtype::F64 => {
            let bytes: Vec<u8> = data
                .iter()
                .flat_map(|&f| (f as f64).to_le_bytes())
                .collect();
            Ok(KernelOutput::Blob {
                data: bytes,
                content_type: "tensor/f64".to_string(),
                shape: Some(shape.to_vec()),
            })
        }
        CastDtype::I64 => {
            let bytes: Vec<u8> = data
                .iter()
                .flat_map(|&f| (f as i64).to_le_bytes())
                .collect();
            Ok(KernelOutput::Blob {
                data: bytes,
                content_type: "tensor/i64".to_string(),
                shape: Some(shape.to_vec()),
            })
        }
        CastDtype::F16 => {
            let bytes: Vec<u8> = data
                .iter()
                .flat_map(|&f| half::f16::from_f32(f).to_le_bytes())
                .collect();
            Ok(KernelOutput::Blob {
                data: bytes,
                content_type: "tensor/f16".to_string(),
                shape: Some(shape.to_vec()),
            })
        }
    }
}

/// Reduce along a dimension (sum, max, min, prod) with f64 accumulation.
///
/// If `dim` is None, reduces over all elements (global reduction).
/// If `keepdim` is true, the reduced dimension is kept as size 1.
fn reduce(
    data: &[f32],
    shape: &[usize],
    mode: ReduceMode,
    dim: Option<i64>,
    keepdim: bool,
    blob_output: bool,
) -> Result<KernelOutput, String> {
    let view = ArrayViewD::<f32>::from_shape(IxDyn(shape), data)
        .map_err(|e| format!("tensor: ndarray view: {e}"))?;

    match dim {
        None => {
            // Global reduction
            let result = match mode {
                ReduceMode::Sum => {
                    view.iter().copied().fold(0.0f64, |a, x| a + x as f64) as f32
                }
                ReduceMode::Max => view.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                ReduceMode::Min => view.iter().copied().fold(f32::INFINITY, f32::min),
                ReduceMode::Prod => {
                    view.iter().copied().fold(1.0f64, |a, x| a * x as f64) as f32
                }
            };
            let out_shape = if keepdim {
                vec![1; shape.len()]
            } else {
                vec![1]
            };
            emit_tensor(&[result], &out_shape, blob_output)
        }
        Some(d) => {
            let ndim = shape.len();
            let axis = resolve_dim(d, ndim)?;

            let f64_view = view.mapv(|x| x as f64);

            let reduced: ArrayD<f64> = match mode {
                ReduceMode::Sum => f64_view.sum_axis(Axis(axis)),
                ReduceMode::Prod => f64_view.map_axis(Axis(axis), |lane| {
                    lane.iter().copied().fold(1.0, |a, x| a * x)
                }),
                ReduceMode::Max => f64_view.map_axis(Axis(axis), |lane| {
                    lane.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                }),
                ReduceMode::Min => f64_view.map_axis(Axis(axis), |lane| {
                    lane.iter().copied().fold(f64::INFINITY, f64::min)
                }),
            };

            let out: Vec<f32> = reduced.iter().map(|&x| x as f32).collect();
            let mut out_shape: Vec<usize> = reduced.shape().to_vec();

            if keepdim {
                out_shape.insert(axis, 1);
            }

            emit_tensor(&out, &out_shape, blob_output)
        }
    }
}

/// Conditional element selection: where(condition > 0, x, y).
///
/// All three tensors must have the same total number of elements.
fn where_select(
    x_data: &[f32],
    x_shape: &[usize],
    condition: &[f32],
    y_data: &[f32],
    y_shape: &[usize],
    blob_output: bool,
) -> Result<KernelOutput, String> {
    let total: usize = x_shape.iter().product();
    if condition.len() != total {
        return Err(format!(
            "tensor: where condition has {} elements, expected {}",
            condition.len(),
            total
        ));
    }
    let y_total: usize = y_shape.iter().product();
    if y_total != total {
        return Err(format!(
            "tensor: where y has {} elements, expected {}",
            y_total, total
        ));
    }

    let out: Vec<f32> = condition
        .iter()
        .zip(x_data.iter())
        .zip(y_data.iter())
        .map(|((&c, &x), &y)| if c > 0.0 { x } else { y })
        .collect();

    emit_tensor(&out, x_shape, blob_output)
}

/// Matrix multiplication for 2D tensors.
///
/// Computes A × B where A is [M, K] and B is [K, N], yielding [M, N].
/// Uses ndarray's optimized dot product.
fn matmul(
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
    blob_output: bool,
) -> Result<KernelOutput, String> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(format!(
            "tensor: matmul requires 2D tensors, got {:?} and {:?}",
            a_shape, b_shape
        ));
    }

    let (m, k1) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);

    if k1 != k2 {
        return Err(format!(
            "tensor: matmul inner dims mismatch: [{}, {}] x [{}, {}]",
            m, k1, k2, n
        ));
    }

    // Zero-copy: use ArrayView instead of cloning into owned Array2.
    let a = ndarray::ArrayView2::<f32>::from_shape((m, k1), a_data)
        .map_err(|e| format!("tensor: ndarray a: {e}"))?;
    let b = ndarray::ArrayView2::<f32>::from_shape((k2, n), b_data)
        .map_err(|e| format!("tensor: ndarray b: {e}"))?;

    let c = a.dot(&b);
    let out: Vec<f32> = c.iter().copied().collect();
    let out_shape = vec![m, n];

    emit_tensor(&out, &out_shape, blob_output)
}

/// Resolve a potentially negative dimension index to a positive one.
fn resolve_dim(dim: i64, ndim: usize) -> Result<usize, String> {
    let axis = if dim >= 0 {
        dim as usize
    } else {
        ((ndim as i64) + dim) as usize
    };
    if axis >= ndim {
        return Err(format!(
            "tensor: dim={dim} out of range for {ndim}-D tensor"
        ));
    }
    Ok(axis)
}

/// Common output emitter for tensor operations.
fn emit_tensor(
    data: &[f32],
    shape: &[usize],
    blob_output: bool,
) -> Result<KernelOutput, String> {
    if blob_output {
        Ok(KernelOutput::Blob {
            data: data.iter().flat_map(|f| f.to_le_bytes()).collect(),
            content_type: "tensor/f32".to_string(),
            shape: Some(shape.to_vec()),
        })
    } else {
        Ok(KernelOutput::Json(serde_json::json!({
            "data": data,
            "shape": shape,
        })))
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Parse raw f32 bytes based on content_type.
fn bytes_to_f32(bytes: &[u8], content_type: &str) -> Result<Vec<f32>, String> {
    match content_type {
        "tensor/f32" => {
            if bytes.len() % 4 != 0 {
                return Err(format!(
                    "tensor: f32 data length {} not divisible by 4",
                    bytes.len()
                ));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        "tensor/f64" => {
            if bytes.len() % 8 != 0 {
                return Err(format!(
                    "tensor: f64 data length {} not divisible by 8",
                    bytes.len()
                ));
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect())
        }
        _ => Err(format!("tensor: unsupported content_type '{content_type}'")),
    }
}

/// Parse tensor from JSON.
///
/// Search order:
///   1. Top level: `{"shape": ..., "data"|"embedding": ...}`
///   2. ONNX format: `{"outputs": {"name": {"shape": ..., "data": ...}}}`
///   3. One level deep (namespaced signals from Cortex context):
///      `{"tract_name": {"shape": ..., "data": ...}}`
///      `{"tract_name": {"outputs": {"name": {"shape": ..., "data": ...}}}}`
///   4. Flat array at top level (assume 1D)
fn parse_tensor_from_json(json: &Value) -> Result<(Vec<f32>, Vec<usize>), String> {
    // Direct format: {"shape": [...], "data"|"embedding": [...]}
    if let Ok(result) = try_extract_tensor(json) {
        return Ok(result);
    }

    // One level deep: namespaced signals from Cortex Rule 2.
    // e.g., {"raw_embeddings": {"outputs": {...}}, "text": "...", "tokens": {...}}
    if let Value::Object(map) = json {
        for (_, inner) in map {
            if let Ok(result) = try_extract_tensor(inner) {
                return Ok(result);
            }
        }
    }

    // Flat array (assume 1D).
    if let Some(arr) = json.as_array() {
        let data: Vec<f32> = arr.iter()
            .map(|v| v.as_f64().ok_or("tensor: non-numeric value in array"))
            .collect::<Result<Vec<f64>, _>>()?
            .into_iter()
            .map(|f| f as f32)
            .collect();
        let shape = vec![data.len()];
        return Ok((data, shape));
    }

    Err("tensor: cannot find tensor data in input JSON".into())
}

/// Try to extract tensor from a single JSON value.
/// Checks direct format and ONNX format.
fn try_extract_tensor(json: &Value) -> Result<(Vec<f32>, Vec<usize>), String> {
    // Direct format: {"shape": [...], "data"|"embedding": [...]}
    if json.get("shape").is_some()
        && (json.get("data").is_some() || json.get("embedding").is_some())
    {
        return extract_shape_data(json);
    }

    // ONNX format: {"outputs": {"name": {"shape": [...], "data": [...]}}}
    if let Some(outputs) = json.get("outputs") {
        if let Value::Object(map) = outputs {
            if let Some((_, tensor)) = map.iter().next() {
                return extract_shape_data(tensor);
            }
        }
    }

    Err("not a tensor".into())
}

/// Extract shape and data from a `{"shape": [...], "data"|"embedding": [...]}` object.
fn extract_shape_data(json: &Value) -> Result<(Vec<f32>, Vec<usize>), String> {
    let shape: Vec<usize> = json
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or("tensor: missing 'shape' array")?
        .iter()
        .map(|v| v.as_u64().ok_or("tensor: shape must be integers").map(|n| n as usize))
        .collect::<Result<Vec<_>, _>>()?;

    // Accept "data" or "embedding" as the data key.
    let data_arr = json
        .get("data")
        .or_else(|| json.get("embedding"))
        .and_then(|v| v.as_array())
        .ok_or("tensor: missing 'data' or 'embedding' array")?;

    let data: Vec<f32> = data_arr
        .iter()
        .map(|v| v.as_f64().ok_or("tensor: data must be numbers").map(|f| f as f32))
        .collect::<Result<Vec<_>, _>>()?;

    Ok((data, shape))
}

/// Parse operation config.
fn parse_config(operations: &Value) -> Result<TensorConfig, String> {
    match operations {
        Value::String(s) => Ok(TensorConfig {
            op: parse_op_string(s)?,
            blob_output: false,
        }),
        Value::Object(obj) => {
            let op_name = obj
                .get("op")
                .and_then(|v| v.as_str())
                .ok_or("tensor: operations must have 'op' field")?;

            let blob_output = obj
                .get("blob_output")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let op = match op_name {
                "mean_pool" => {
                    let dim = obj.get("dim").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                    TensorOp::MeanPool { dim }
                }
                "normalize" => TensorOp::Normalize,
                "reshape" => {
                    let shape = parse_usize_array(obj, "shape", "tensor: reshape")?;
                    TensorOp::Reshape { shape }
                }
                "transpose" => {
                    let axes = parse_usize_array(obj, "axes", "tensor: transpose")?;
                    TensorOp::Transpose { axes }
                }
                "softmax" => {
                    let dim = obj.get("dim").and_then(|v| v.as_i64()).unwrap_or(-1);
                    TensorOp::Softmax { dim }
                }
                "argmax" => {
                    let dim = obj.get("dim").and_then(|v| v.as_i64()).unwrap_or(-1);
                    TensorOp::Argmax { dim }
                }
                "topk" => {
                    let k = obj.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
                    let dim = obj.get("dim").and_then(|v| v.as_i64()).unwrap_or(-1);
                    TensorOp::TopK { k, dim }
                }
                "unsqueeze" => {
                    let dim = obj.get("dim").and_then(|v| v.as_i64()).unwrap_or(0);
                    TensorOp::Unsqueeze { dim }
                }
                "squeeze" => {
                    let dim = obj.get("dim").and_then(|v| v.as_i64());
                    TensorOp::Squeeze { dim }
                }
                "clamp" => {
                    let min = obj.get("min").and_then(|v| v.as_f64()).unwrap_or(f64::NEG_INFINITY) as f32;
                    let max = obj.get("max").and_then(|v| v.as_f64()).unwrap_or(f64::INFINITY) as f32;
                    TensorOp::Clamp { min, max }
                }
                "concat" => {
                    let dim = obj.get("dim").and_then(|v| v.as_i64()).unwrap_or(0);
                    let others = obj
                        .get("others")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .map(|t| extract_shape_data(t))
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .transpose()?
                        .unwrap_or_default();
                    TensorOp::Concat { dim, others }
                }
                "slice" => {
                    let ranges = obj
                        .get("ranges")
                        .and_then(|v| v.as_array())
                        .ok_or("tensor: slice requires 'ranges' array")?
                        .iter()
                        .map(|r| {
                            let arr = r
                                .as_array()
                                .ok_or("tensor: slice range must be [start, end]")?;
                            if arr.len() != 2 {
                                return Err(
                                    "tensor: slice range must be [start, end]".to_string()
                                );
                            }
                            let start = arr[0]
                                .as_i64()
                                .ok_or("tensor: slice range values must be integers")?;
                            let end = arr[1]
                                .as_i64()
                                .ok_or("tensor: slice range values must be integers")?;
                            Ok((start, end))
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    TensorOp::Slice { ranges }
                }
                "gather" => {
                    let dim = obj.get("dim").and_then(|v| v.as_i64()).unwrap_or(0);
                    let index_obj = obj
                        .get("index")
                        .ok_or("tensor: gather requires 'index' object with data and shape")?;
                    let index_shape = index_obj
                        .get("shape")
                        .and_then(|v| v.as_array())
                        .ok_or("tensor: gather index requires 'shape'")?
                        .iter()
                        .map(|v| {
                            v.as_u64()
                                .ok_or("tensor: index shape must be integers")
                                .map(|n| n as usize)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let index_data = index_obj
                        .get("data")
                        .and_then(|v| v.as_array())
                        .ok_or("tensor: gather index requires 'data'")?
                        .iter()
                        .map(|v| v.as_i64().ok_or("tensor: index data must be integers"))
                        .collect::<Result<Vec<_>, _>>()?;
                    TensorOp::Gather {
                        dim,
                        index_data,
                        index_shape,
                    }
                }
                "cast" => {
                    let to = match obj
                        .get("to")
                        .and_then(|v| v.as_str())
                        .ok_or("tensor: cast requires 'to' field")?
                    {
                        "f32" => CastDtype::F32,
                        "f64" => CastDtype::F64,
                        "i64" => CastDtype::I64,
                        "f16" => CastDtype::F16,
                        t => return Err(format!("tensor: unknown cast type '{t}'")),
                    };
                    TensorOp::Cast { to }
                }
                "reduce_sum" | "reduce_max" | "reduce_min" | "reduce_prod" => {
                    let mode = match op_name {
                        "reduce_sum" => ReduceMode::Sum,
                        "reduce_max" => ReduceMode::Max,
                        "reduce_min" => ReduceMode::Min,
                        "reduce_prod" => ReduceMode::Prod,
                        _ => unreachable!(),
                    };
                    let dim = obj.get("dim").and_then(|v| v.as_i64());
                    let keepdim = obj
                        .get("keepdim")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    TensorOp::Reduce { mode, dim, keepdim }
                }
                "where" => {
                    let cond_arr = obj
                        .get("condition")
                        .and_then(|v| v.as_array())
                        .ok_or("tensor: where requires 'condition' array")?;
                    let condition: Vec<f32> = cond_arr
                        .iter()
                        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                        .collect();
                    let y_obj = obj
                        .get("y")
                        .ok_or("tensor: where requires 'y' tensor with data and shape")?;
                    let (y_data, y_shape) = extract_shape_data(y_obj)?;
                    TensorOp::WhereSelect {
                        condition,
                        y_data,
                        y_shape,
                    }
                }
                "matmul" => {
                    let other_obj = obj
                        .get("other")
                        .ok_or("tensor: matmul requires 'other' tensor with data and shape")?;
                    let (other_data, other_shape) = extract_shape_data(other_obj)?;
                    TensorOp::MatMul {
                        other_data,
                        other_shape,
                    }
                }
                _ => return Err(format!("tensor: unknown op '{op_name}'")),
            };

            Ok(TensorConfig { op, blob_output })
        }
        _ => Err("tensor: operations must be string or object".into()),
    }
}

/// Parse shorthand op string like "mean_pool" or "normalize".
fn parse_op_string(s: &str) -> Result<TensorOp, String> {
    match s.split_whitespace().next().unwrap_or("") {
        "mean_pool" => Ok(TensorOp::MeanPool { dim: 1 }),
        "normalize" => Ok(TensorOp::Normalize),
        "softmax" => Ok(TensorOp::Softmax { dim: -1 }),
        "argmax" => Ok(TensorOp::Argmax { dim: -1 }),
        "squeeze" => Ok(TensorOp::Squeeze { dim: None }),
        "reduce_sum" => Ok(TensorOp::Reduce { mode: ReduceMode::Sum, dim: None, keepdim: false }),
        "reduce_max" => Ok(TensorOp::Reduce { mode: ReduceMode::Max, dim: None, keepdim: false }),
        "reduce_min" => Ok(TensorOp::Reduce { mode: ReduceMode::Min, dim: None, keepdim: false }),
        "reduce_prod" => Ok(TensorOp::Reduce { mode: ReduceMode::Prod, dim: None, keepdim: false }),
        other => Err(format!("tensor: unknown op '{other}' (use object format for ops with params)")),
    }
}

/// Parse a usize array from a JSON object field.
fn parse_usize_array(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<Vec<usize>, String> {
    obj.get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{context} requires '{key}' array"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .ok_or_else(|| format!("{context}: {key} must be integers"))
                .map(|n| n as usize)
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::BlobData;
    use crate::blob::BlobMeta;
    use std::collections::HashMap;

    #[test]
    fn test_mean_pool_3d_dim1() {
        let kernel = TensorKernel;
        // Shape [1, 3, 2] — batch=1, seq_len=3, dim=2
        // Mean pool dim=1 → [1, 2]
        let data = serde_json::json!({
            "shape": [1, 3, 2],
            "data": [1.0, 2.0,  3.0, 4.0,  5.0, 6.0]
        });
        let ops = serde_json::json!({"op": "mean_pool", "dim": 1});
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();

        let embedding: Vec<f64> = result["embedding"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        // mean([1,3,5]) = 3.0, mean([2,4,6]) = 4.0
        assert_eq!(embedding, vec![3.0, 4.0]);
        assert_eq!(result["shape"], serde_json::json!([1, 2]));
    }

    #[test]
    fn test_mean_pool_string_op() {
        let kernel = TensorKernel;
        let data = serde_json::json!({
            "shape": [1, 4, 3],
            "data": [
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                10.0, 11.0, 12.0
            ]
        });
        let ops = serde_json::json!("mean_pool");
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();

        let embedding: Vec<f64> = result["embedding"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        // mean of each dim across seq_len=4
        assert!((embedding[0] - 5.5).abs() < 1e-5); // mean(1,4,7,10)
        assert!((embedding[1] - 6.5).abs() < 1e-5); // mean(2,5,8,11)
        assert!((embedding[2] - 7.5).abs() < 1e-5); // mean(3,6,9,12)
    }

    #[test]
    fn test_mean_pool_from_blob() {
        let kernel = TensorKernel;

        // Create raw f32 blob data: shape [1, 2, 3]
        let floats: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let input = KernelInput {
            json: serde_json::Value::Null,
            blobs: {
                let mut m = HashMap::new();
                m.insert("tensor".to_string(), BlobData {
                    bytes,
                    meta: BlobMeta {
                        size: 24,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![1, 2, 3]),
                    },
                });
                m
            },
        };

        let ops = serde_json::json!({"op": "mean_pool", "dim": 1});
        let result = kernel.execute(input, ops).unwrap().unwrap_json();

        let embedding: Vec<f64> = result["embedding"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        // mean([1,4])=2.5, mean([2,5])=3.5, mean([3,6])=4.5
        assert!((embedding[0] - 2.5).abs() < 1e-5);
        assert!((embedding[1] - 3.5).abs() < 1e-5);
        assert!((embedding[2] - 4.5).abs() < 1e-5);
    }

    #[test]
    fn test_mean_pool_blob_output() {
        let kernel = TensorKernel;
        let data = serde_json::json!({
            "shape": [1, 2, 3],
            "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        });
        let ops = serde_json::json!({"op": "mean_pool", "dim": 1, "blob_output": true});
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap();

        match result {
            KernelOutput::Blob { data, content_type, shape } => {
                assert_eq!(content_type, "tensor/f32");
                assert_eq!(shape, Some(vec![1, 3]));
                let floats: Vec<f32> = data.chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                assert!((floats[0] - 2.5).abs() < 1e-5);
                assert!((floats[1] - 3.5).abs() < 1e-5);
                assert!((floats[2] - 4.5).abs() < 1e-5);
            }
            _ => panic!("expected KernelOutput::Blob"),
        }
    }

    #[test]
    fn test_normalize() {
        let kernel = TensorKernel;
        let data = serde_json::json!({
            "shape": [1, 3],
            "data": [3.0, 4.0, 0.0]
        });
        let ops = serde_json::json!({"op": "normalize"});
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();

        let out: Vec<f64> = result["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        // L2 norm of [3, 4, 0] = 5
        assert!((out[0] - 0.6).abs() < 1e-5);
        assert!((out[1] - 0.8).abs() < 1e-5);
        assert!((out[2]).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_f64_precision() {
        // Test that f64 accumulation avoids precision loss with large values.
        let kernel = TensorKernel;
        let data = serde_json::json!({
            "shape": [1, 2],
            "data": [1e18, 1e18]
        });
        let ops = serde_json::json!({"op": "normalize"});
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();

        let out: Vec<f64> = result["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        // Both values equal → each should be 1/sqrt(2) ≈ 0.7071
        let expected = 1.0 / 2.0f64.sqrt();
        assert!((out[0] as f64 - expected).abs() < 1e-3);
        assert!((out[1] as f64 - expected).abs() < 1e-3);
    }

    #[test]
    fn test_reshape() {
        let kernel = TensorKernel;
        let data = serde_json::json!({
            "shape": [2, 3],
            "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        });
        let ops = serde_json::json!({"op": "reshape", "shape": [3, 2]});
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();

        assert_eq!(result["shape"], serde_json::json!([3, 2]));
        let out: Vec<f64> = result["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_mismatch() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [2, 3], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]});
        let ops = serde_json::json!({"op": "reshape", "shape": [2, 2]});
        let err = kernel.execute(KernelInput::from_json(data), ops).unwrap_err();
        assert!(err.contains("mismatch"));
    }

    #[test]
    fn test_mean_pool_dim_out_of_range() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [2, 3], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]});
        let ops = serde_json::json!({"op": "mean_pool", "dim": 5});
        let err = kernel.execute(KernelInput::from_json(data), ops).unwrap_err();
        assert!(err.contains("out of range"));
    }

    #[test]
    fn test_parse_onnx_format() {
        let kernel = TensorKernel;
        let input = serde_json::json!({
            "outputs": {
                "last_hidden_state": {
                    "shape": [1, 2, 4],
                    "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                }
            }
        });
        let ops = serde_json::json!("mean_pool");
        let result = kernel.execute(KernelInput::from_json(input), ops).unwrap().unwrap_json();

        let embedding: Vec<f64> = result["embedding"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        // mean([1,5])=3, mean([2,6])=4, mean([3,7])=5, mean([4,8])=6
        assert_eq!(embedding, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_unknown_op() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [1], "data": [1.0]});
        let err = kernel.execute(KernelInput::from_json(data), serde_json::json!("foobar")).unwrap_err();
        assert!(err.contains("unknown op"));
    }

    #[test]
    fn test_bytes_to_f32_invalid_content_type() {
        let err = bytes_to_f32(&[0u8; 4], "image/png").unwrap_err();
        assert!(err.contains("unsupported"));
    }

    #[test]
    fn test_blob_output_config() {
        let config = parse_config(&serde_json::json!({"op": "normalize", "blob_output": true})).unwrap();
        assert!(config.blob_output);

        let config = parse_config(&serde_json::json!("mean_pool")).unwrap();
        assert!(!config.blob_output);
    }

    #[test]
    fn test_concat() {
        let kernel = TensorKernel;
        let data = serde_json::json!({
            "shape": [2, 3],
            "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        });
        let ops = serde_json::json!({
            "op": "concat",
            "dim": 0,
            "others": [{"data": [7.0, 8.0, 9.0], "shape": [1, 3]}]
        });
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();
        assert_eq!(result["shape"], serde_json::json!([3, 3]));
        let out: Vec<f64> = result["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_concat_dim1() {
        let kernel = TensorKernel;
        let data = serde_json::json!({
            "shape": [2, 2],
            "data": [1.0, 2.0, 3.0, 4.0]
        });
        let ops = serde_json::json!({
            "op": "concat",
            "dim": 1,
            "others": [{"data": [5.0, 6.0], "shape": [2, 1]}]
        });
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();
        assert_eq!(result["shape"], serde_json::json!([2, 3]));
    }

    #[test]
    fn test_slice() {
        let kernel = TensorKernel;
        let data = serde_json::json!({
            "shape": [3, 4],
            "data": [1.0,2.0,3.0,4.0, 5.0,6.0,7.0,8.0, 9.0,10.0,11.0,12.0]
        });
        let ops = serde_json::json!({"op": "slice", "ranges": [[0, 2], [1, 3]]});
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();
        assert_eq!(result["shape"], serde_json::json!([2, 2]));
        let out: Vec<f64> = result["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(out, vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn test_slice_negative_index() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [4], "data": [10.0, 20.0, 30.0, 40.0]});
        // [0, -1) means [0, 3) = first 3 elements
        let ops = serde_json::json!({"op": "slice", "ranges": [[0, -1]]});
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();
        let out: Vec<f64> = result["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(out, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_gather() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [3, 2], "data": [1.0,2.0, 3.0,4.0, 5.0,6.0]});
        let ops = serde_json::json!({
            "op": "gather",
            "dim": 1,
            "index": {"data": [0, 1, 1, 0, 0, 0], "shape": [3, 2]}
        });
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();
        // dim=1: out[i][j] = input[i][index[i][j]]
        let out: Vec<f64> = result["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(out, vec![1.0, 2.0, 4.0, 3.0, 5.0, 5.0]);
    }

    #[test]
    fn test_cast_f64() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [2], "data": [1.5, 2.5]});
        let ops = serde_json::json!({"op": "cast", "to": "f64"});
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap();
        match result {
            KernelOutput::Blob { content_type, shape, data } => {
                assert_eq!(content_type, "tensor/f64");
                assert_eq!(shape, Some(vec![2]));
                assert_eq!(data.len(), 16); // 2 × 8 bytes
                let v0 = f64::from_le_bytes(data[0..8].try_into().unwrap());
                assert!((v0 - 1.5).abs() < 1e-10);
            }
            _ => panic!("expected Blob output"),
        }
    }

    #[test]
    fn test_cast_f16() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [2], "data": [1.0, 0.5]});
        let ops = serde_json::json!({"op": "cast", "to": "f16"});
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap();
        match result {
            KernelOutput::Blob { content_type, shape, data } => {
                assert_eq!(content_type, "tensor/f16");
                assert_eq!(shape, Some(vec![2]));
                assert_eq!(data.len(), 4); // 2 × 2 bytes
                let v0 = half::f16::from_le_bytes([data[0], data[1]]);
                assert!((v0.to_f32() - 1.0).abs() < 1e-3);
            }
            _ => panic!("expected Blob output"),
        }
    }

    #[test]
    fn test_reduce_sum_dim() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [2, 3], "data": [1.0,2.0,3.0, 4.0,5.0,6.0]});
        let ops = serde_json::json!({"op": "reduce_sum", "dim": 1});
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();
        assert_eq!(result["shape"], serde_json::json!([2]));
        let out: Vec<f64> = result["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(out, vec![6.0, 15.0]);
    }

    #[test]
    fn test_reduce_max_global() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [2, 3], "data": [1.0, 5.0, 3.0, 4.0, 2.0, 6.0]});
        let ops = serde_json::json!("reduce_max");
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();
        let out: Vec<f64> = result["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(out, vec![6.0]);
    }

    #[test]
    fn test_reduce_keepdim() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [2, 3], "data": [1.0,2.0,3.0, 4.0,5.0,6.0]});
        let ops = serde_json::json!({"op": "reduce_sum", "dim": 1, "keepdim": true});
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();
        assert_eq!(result["shape"], serde_json::json!([2, 1]));
    }

    #[test]
    fn test_reduce_prod() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [3], "data": [2.0, 3.0, 4.0]});
        let ops = serde_json::json!("reduce_prod");
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();
        let out: Vec<f64> = result["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(out, vec![24.0]);
    }

    #[test]
    fn test_where_select() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [4], "data": [10.0, 20.0, 30.0, 40.0]});
        let ops = serde_json::json!({
            "op": "where",
            "condition": [1.0, 0.0, 1.0, 0.0],
            "y": {"data": [100.0, 200.0, 300.0, 400.0], "shape": [4]}
        });
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();
        let out: Vec<f64> = result["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(out, vec![10.0, 200.0, 30.0, 400.0]);
    }

    #[test]
    fn test_matmul() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [2, 3], "data": [1.0,2.0,3.0, 4.0,5.0,6.0]});
        let ops = serde_json::json!({
            "op": "matmul",
            "other": {"data": [7.0,8.0, 9.0,10.0, 11.0,12.0], "shape": [3, 2]}
        });
        let result = kernel.execute(KernelInput::from_json(data), ops).unwrap().unwrap_json();
        assert_eq!(result["shape"], serde_json::json!([2, 2]));
        let out: Vec<f64> = result["data"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        // [1,2,3]·[7,9,11]=58, [1,2,3]·[8,10,12]=64
        // [4,5,6]·[7,9,11]=139, [4,5,6]·[8,10,12]=154
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_dim_mismatch() {
        let kernel = TensorKernel;
        let data = serde_json::json!({"shape": [2, 3], "data": [1.0,2.0,3.0, 4.0,5.0,6.0]});
        let ops = serde_json::json!({
            "op": "matmul",
            "other": {"data": [1.0, 2.0], "shape": [2, 1]}
        });
        let err = kernel.execute(KernelInput::from_json(data), ops).unwrap_err();
        assert!(err.contains("mismatch"));
    }
}
