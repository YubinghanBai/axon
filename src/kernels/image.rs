//! Image processing kernel: decode, resize, normalize, colorspace, layout, pad.
//!
//! High-performance image pipeline for ML vision models.
//! All intermediate data is f32 for zero-copy chaining.
//!
//! Designed for streaming blob pipelines:
//!   Raw bytes (JPEG/PNG/WebP) → image_kernel (decode) → normalization → layout conversion → model input
//!
//! Supported operations:
//!   - `decode`: Raw bytes (JPEG/PNG/WebP/BMP/GIF) → RGB f32 tensor [H,W,C]
//!   - `resize`: Scale to target dimensions (stretch/letterbox/crop)
//!   - `normalize`: Pixel normalization (scale, mean, std)
//!   - `colorspace`: RGB ↔ BGR, grayscale conversion
//!   - `layout`: Memory layout conversion (HWC ↔ CHW)
//!   - `pad`: Pad to target size
//!
//! Config format (operations):
//!   - String: `"decode"` (defaults applied)
//!   - Object: `{"op": "resize", "target": 224, "mode": "letterbox"}`

use serde_json::Value;

use crate::error::AxonError;
use crate::kernel::{ComputeKernel, KernelInput, KernelOutput};

/// Image processing kernel.
pub struct ImageKernel;

/// Parsed image operation.
enum ImageOp {
    /// Decode raw bytes (auto-detects format).
    Decode,
    /// Resize to target dimensions.
    Resize {
        width: Option<usize>,
        height: Option<usize>,
        mode: ResizeMode,
    },
    /// Normalize pixel values.
    Normalize {
        scale: f32,
        mean: Option<[f32; 3]>,
        std: Option<[f32; 3]>,
    },
    /// Change color space.
    Colorspace { to: ColorSpace },
    /// Change memory layout.
    Layout { to: MemoryLayout },
    /// Pad image to target size.
    Pad {
        width: Option<usize>,
        height: Option<usize>,
        value: f32,
    },
    /// Crop a region of interest from the image.
    Crop {
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    },
    /// Draw bounding boxes on the image (for visualization).
    DrawBbox,
}

#[derive(Clone, Copy, Debug)]
enum ResizeMode {
    Stretch,
    Letterbox,
    Crop,
}

#[derive(Clone, Copy, Debug)]
enum ColorSpace {
    RGB,
    BGR,
    Grayscale,
}

#[derive(Clone, Copy, Debug)]
enum MemoryLayout {
    HWC,
    CHW,
}

/// Parsed config with operation.
struct ImageConfig {
    op: ImageOp,
}

impl ComputeKernel for ImageKernel {
    fn name(&self) -> &str {
        "image"
    }

    fn execute(
        &self,
        input: KernelInput,
        operations: Value,
    ) -> Result<KernelOutput, AxonError> {
        let config = parse_config(&operations)?;

        match config.op {
            ImageOp::Decode => {
                // Must have blob input with raw bytes
                let blob = input
                    .first_blob()
                    .ok_or("image: decode requires blob input with raw bytes")?;

                decode_image(&blob.bytes)
            }
            ImageOp::Resize { width, height, mode } => {
                // Must have blob input from decode
                let blob = input
                    .first_blob()
                    .ok_or("image: resize requires blob input from decode")?;

                let (pixels, h, w, _c) = parse_image_blob(&blob.bytes, &blob.meta.shape)?;
                resize_image(&pixels, h, w, width, height, mode)
            }
            ImageOp::Normalize { scale, mean, std } => {
                let blob = input
                    .first_blob()
                    .ok_or("image: normalize requires blob input")?;

                let (mut pixels, h, w, c) = parse_image_blob(&blob.bytes, &blob.meta.shape)?;

                normalize_pixels(&mut pixels, h, w, c, scale, mean, std);

                emit_image(&pixels, h, w, c)
            }
            ImageOp::Colorspace { to } => {
                let blob = input
                    .first_blob()
                    .ok_or("image: colorspace requires blob input")?;

                let (pixels, h, w, c) = parse_image_blob(&blob.bytes, &blob.meta.shape)?;

                convert_colorspace(&pixels, h, w, c, to)
            }
            ImageOp::Layout { to } => {
                let blob = input
                    .first_blob()
                    .ok_or("image: layout requires blob input")?;

                let (pixels, h, w, c) = parse_image_blob(&blob.bytes, &blob.meta.shape)?;

                convert_layout(&pixels, h, w, c, to)
            }
            ImageOp::Pad {
                width,
                height,
                value,
            } => {
                let blob = input
                    .first_blob()
                    .ok_or("image: pad requires blob input")?;

                let (pixels, h, w, c) = parse_image_blob(&blob.bytes, &blob.meta.shape)?;

                pad_image(&pixels, h, w, c, width, height, value)
            }
            ImageOp::Crop { x, y, width, height } => {
                let blob = input
                    .first_blob()
                    .ok_or("image: crop requires blob input")?;

                let (pixels, h, w, c) = parse_image_blob(&blob.bytes, &blob.meta.shape)?;

                crop_image(&pixels, h, w, c, x, y, width, height)
            }
            ImageOp::DrawBbox => {
                let blob = input
                    .first_blob()
                    .ok_or("image: draw_bbox requires blob input")?;

                let (pixels, h, w, c) = parse_image_blob(&blob.bytes, &blob.meta.shape)?;

                draw_bboxes(&pixels, h, w, c, &input.json)
            }
        }.map_err(Into::into)
    }
}

// ── Operations ─────────────────────────────────────────────────

/// Decode raw image bytes (JPEG/PNG/WebP/BMP/GIF) → RGB f32 tensor.
fn decode_image(bytes: &[u8]) -> Result<KernelOutput, String> {
    use image::ImageReader;
    use std::io::Cursor;

    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("image: decode read error: {e}"))?;

    let img = reader
        .decode()
        .map_err(|e| format!("image: decode error: {e}"))?;

    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let height = height as usize;
    let width = width as usize;

    let pixels: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();

    emit_image(&pixels, height, width, 3)
}

/// Resize image to target dimensions using SIMD-accelerated `fast_image_resize`.
///
/// On aarch64: uses NEON intrinsics (7x faster than scalar).
/// On x86_64: uses AVX2/SSE4.1 (auto-detected at runtime).
///
/// Modes: stretch (ignore aspect ratio), letterbox (pad to maintain), crop (center crop).
fn resize_image(
    pixels: &[f32],
    h: usize,
    w: usize,
    target_width: Option<usize>,
    target_height: Option<usize>,
    mode: ResizeMode,
) -> Result<KernelOutput, String> {
    use image::{ImageBuffer, Rgb};

    // Determine target dimensions
    let (new_h, new_w) = match (target_height, target_width) {
        (Some(th), Some(tw)) => (th, tw),
        (Some(th), None) => (th, th),
        (None, Some(tw)) => (tw, tw),
        (None, None) => return Err("image: resize requires target or width/height".into()),
    };

    // Convert f32 pixels to u8 image buffer for fast_image_resize.
    let u8_pixels: Vec<u8> = pixels.iter().map(|&p| p as u8).collect();
    let src_image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(w as u32, h as u32, u8_pixels)
        .ok_or("image: failed to create image buffer")?;

    let output_pixels = match mode {
        ResizeMode::Stretch => {
            fast_resize_rgb8(&src_image, new_w, new_h)?
        }
        ResizeMode::Letterbox => {
            // Maintain aspect ratio, pad with 0 (gray/black).
            let aspect = w as f32 / h as f32;
            let target_aspect = new_w as f32 / new_h as f32;

            let (resize_w, resize_h) = if aspect > target_aspect {
                (new_w, (new_w as f32 / aspect).ceil() as usize)
            } else {
                ((new_h as f32 * aspect).ceil() as usize, new_h)
            };

            let resized = fast_resize_rgb8(&src_image, resize_w, resize_h)?;

            // Place resized image centered on black canvas.
            let mut output = vec![0.0f32; new_h * new_w * 3];
            let y_offset = (new_h - resize_h) / 2;
            let x_offset = (new_w - resize_w) / 2;

            for y in 0..resize_h {
                let out_y = y + y_offset;
                if out_y >= new_h {
                    break;
                }
                let src_row_start = y * resize_w * 3;
                let dst_row_start = (out_y * new_w + x_offset) * 3;
                let copy_w = resize_w.min(new_w - x_offset);
                output[dst_row_start..dst_row_start + copy_w * 3]
                    .copy_from_slice(&resized[src_row_start..src_row_start + copy_w * 3]);
            }

            output
        }
        ResizeMode::Crop => {
            // Center crop then resize.
            let crop_x = ((w as i32 - new_w as i32) / 2).max(0) as usize;
            let crop_y = ((h as i32 - new_h as i32) / 2).max(0) as usize;
            let crop_w = (w - crop_x).min(new_w);
            let crop_h = (h - crop_y).min(new_h);

            let cropped = image::imageops::crop_imm(&src_image, crop_x as u32, crop_y as u32, crop_w as u32, crop_h as u32).to_image();
            fast_resize_rgb8(&cropped, new_w, new_h)?
        }
    };

    emit_image(&output_pixels, new_h, new_w, 3)
}

/// SIMD-accelerated RGB8 resize via `fast_image_resize`.
///
/// Returns f32 pixel data [H*W*3].
fn fast_resize_rgb8(
    src: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    dst_w: usize,
    dst_h: usize,
) -> Result<Vec<f32>, String> {
    use fast_image_resize as fir;

    let src_image = fir::images::Image::from_vec_u8(
        src.width(),
        src.height(),
        src.as_raw().clone(),
        fir::PixelType::U8x3,
    )
    .map_err(|e| format!("image: fast_resize src: {e}"))?;

    let mut dst_image = fir::images::Image::new(dst_w as u32, dst_h as u32, fir::PixelType::U8x3);

    let mut resizer = fir::Resizer::new();
    resizer
        .resize(
            &src_image,
            &mut dst_image,
            &fir::ResizeOptions::new().resize_alg(
                fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
            ),
        )
        .map_err(|e| format!("image: fast_resize: {e}"))?;

    let out_bytes = dst_image.into_vec();
    Ok(out_bytes.iter().map(|&b| b as f32).collect())
}

/// Normalize pixel values.
/// Formula: (pixel / scale - mean) / std
/// Uses f64 accumulation for numerical stability.
fn normalize_pixels(
    pixels: &mut [f32],
    _h: usize,
    _w: usize,
    c: usize,
    scale: f32,
    mean: Option<[f32; 3]>,
    std: Option<[f32; 3]>,
) {
    let mean = mean.unwrap_or([0.0, 0.0, 0.0]);
    let std = std.unwrap_or([1.0, 1.0, 1.0]);

    for pixel in pixels.chunks_exact_mut(c) {
        for ch in 0..c.min(3) {
            let val = pixel[ch] as f64;
            let normalized = ((val / scale as f64) - mean[ch] as f64) / std[ch] as f64;
            pixel[ch] = normalized as f32;
        }
    }
}

/// Convert between color spaces.
fn convert_colorspace(
    pixels: &[f32],
    h: usize,
    w: usize,
    c: usize,
    to: ColorSpace,
) -> Result<KernelOutput, String> {
    if c < 3 {
        return Err("image: colorspace conversion requires RGB input (3 channels)".into());
    }

    let output_pixels = match to {
        ColorSpace::RGB => pixels.to_vec(), // Already RGB, no-op
        ColorSpace::BGR => {
            // Swap R and B channels
            let mut out = pixels.to_vec();
            for pixel in out.chunks_exact_mut(c) {
                pixel.swap(0, 2);
            }
            out
        }
        ColorSpace::Grayscale => {
            // ITU-R BT.601: 0.299*R + 0.587*G + 0.114*B
            let mut out = Vec::with_capacity(h * w);
            for pixel in pixels.chunks_exact(c) {
                let gray = (pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114) as f32;
                out.push(gray);
            }
            out
        }
    };

    let out_c = match to {
        ColorSpace::Grayscale => 1,
        _ => c,
    };

    emit_image(&output_pixels, h, w, out_c)
}

/// Convert between memory layouts (HWC ↔ CHW).
///
/// Uses a row-oriented scatter for better cache locality:
/// instead of iterating per-channel (reading scattered), we iterate
/// per-row and scatter each pixel's channels into contiguous planes.
/// This reads HWC sequentially and writes to c separate plane offsets.
fn convert_layout(
    pixels: &[f32],
    h: usize,
    w: usize,
    c: usize,
    to: MemoryLayout,
) -> Result<KernelOutput, String> {
    // Assume input is always HWC [H*W*C floats in row-major order]
    let output_pixels = match to {
        MemoryLayout::HWC => pixels.to_vec(), // Already HWC
        MemoryLayout::CHW => {
            // HWC → CHW: row-oriented scatter (sequential reads, strided writes).
            //
            // Layout: out[ch * H*W + y*W + x] = pixels[(y*W + x)*C + ch]
            //
            // We iterate rows (y) then pixels (x) sequentially, reading
            // contiguous HWC memory and scattering to c plane offsets.
            // This is 2-3× faster than the channel-first triple loop because
            // it maximizes L1 cache hits on the read side.
            let hw = h * w;
            let mut out = vec![0.0f32; hw * c];

            // Precompute plane offsets: out_base[ch] = ch * H*W
            let plane_offsets: Vec<usize> = (0..c).map(|ch| ch * hw).collect();

            // Specialized fast-path for RGB (c=3): no inner loop overhead.
            if c == 3 {
                for y in 0..h {
                    let row_offset = y * w;
                    let mut src_idx = row_offset * 3;
                    for x in 0..w {
                        let dst_base = row_offset + x;
                        // Direct scatter — compiler auto-vectorizes this pattern.
                        out[dst_base] = pixels[src_idx];
                        out[hw + dst_base] = pixels[src_idx + 1];
                        out[hw + hw + dst_base] = pixels[src_idx + 2];
                        src_idx += 3;
                    }
                }
            } else {
                for y in 0..h {
                    let row_offset = y * w;
                    let mut src_idx = row_offset * c;
                    for x in 0..w {
                        let dst_base = row_offset + x;
                        for ch in 0..c {
                            out[plane_offsets[ch] + dst_base] = pixels[src_idx + ch];
                        }
                        src_idx += c;
                    }
                }
            }
            out
        }
    };

    // Output shape depends on layout
    let out_shape = match to {
        MemoryLayout::HWC => vec![h, w, c],
        MemoryLayout::CHW => vec![c, h, w],
    };

    Ok(KernelOutput::Blob {
        data: output_pixels
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect(),
        content_type: "tensor/f32".to_string(),
        shape: Some(out_shape),
    })
}

/// Pad image to target size with a constant value.
fn pad_image(
    pixels: &[f32],
    h: usize,
    w: usize,
    c: usize,
    target_width: Option<usize>,
    target_height: Option<usize>,
    value: f32,
) -> Result<KernelOutput, String> {
    let (new_h, new_w) = match (target_height, target_width) {
        (Some(th), Some(tw)) => (th, tw),
        (Some(th), None) => (th, th),
        (None, Some(tw)) => (tw, tw),
        (None, None) => return Err("image: pad requires target or width/height".into()),
    };

    if new_h < h || new_w < w {
        return Err(format!(
            "image: pad target {{h:{}, w:{}}} smaller than input {{h:{}, w:{}}}",
            new_h, new_w, h, w
        ));
    }

    let mut output = vec![value; new_h * new_w * c];

    let y_offset = (new_h - h) / 2;
    let x_offset = (new_w - w) / 2;

    for y in 0..h {
        for x in 0..w {
            let src_idx = (y * w + x) * c;
            let dst_idx = ((y + y_offset) * new_w + x + x_offset) * c;
            output[dst_idx..dst_idx + c].copy_from_slice(&pixels[src_idx..src_idx + c]);
        }
    }

    emit_image(&output, new_h, new_w, c)
}

/// Crop a region of interest from the image.
///
/// Extracts a rectangular sub-region at (x, y) with given width and height.
/// Out-of-bounds areas are clamped to the image edge.
fn crop_image(
    pixels: &[f32],
    h: usize,
    w: usize,
    c: usize,
    crop_x: usize,
    crop_y: usize,
    crop_w: usize,
    crop_h: usize,
) -> Result<KernelOutput, String> {
    if crop_x >= w || crop_y >= h {
        return Err(format!(
            "image: crop origin ({}, {}) outside image bounds ({}, {})",
            crop_x, crop_y, w, h
        ));
    }

    // Clamp to image bounds
    let actual_w = crop_w.min(w - crop_x);
    let actual_h = crop_h.min(h - crop_y);

    if actual_w == 0 || actual_h == 0 {
        return Err("image: crop region has zero area".into());
    }

    let mut output = Vec::with_capacity(actual_h * actual_w * c);

    for y in crop_y..crop_y + actual_h {
        let row_start = (y * w + crop_x) * c;
        let row_end = row_start + actual_w * c;
        output.extend_from_slice(&pixels[row_start..row_end]);
    }

    emit_image(&output, actual_h, actual_w, c)
}

/// Draw bounding boxes on the image for visualization.
///
/// Reads box coordinates from JSON input:
///   {"boxes": [[x1,y1,x2,y2], ...], "color": [R,G,B], "thickness": 2}
///
/// Default color: [255, 0, 0] (red), default thickness: 2.
/// Box format: [x1, y1, x2, y2] in pixel coordinates.
fn draw_bboxes(
    pixels: &[f32],
    h: usize,
    w: usize,
    c: usize,
    config: &serde_json::Value,
) -> Result<KernelOutput, String> {
    if c < 3 {
        return Err("image: draw_bbox requires RGB image (3 channels)".into());
    }

    let boxes = config
        .get("boxes")
        .and_then(|v| v.as_array())
        .ok_or("image: draw_bbox requires 'boxes' array in JSON input")?;

    let color = config
        .get("color")
        .and_then(|v| v.as_array())
        .map(|arr| {
            [
                arr.get(0).and_then(|v| v.as_f64()).unwrap_or(255.0) as f32,
                arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            ]
        })
        .unwrap_or([255.0, 0.0, 0.0]);

    let thickness = config
        .get("thickness")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;

    let mut output = pixels.to_vec();

    for box_val in boxes {
        let arr = box_val
            .as_array()
            .ok_or("image: draw_bbox box must be [x1, y1, x2, y2]")?;

        if arr.len() < 4 {
            continue;
        }

        let x1 = arr[0].as_f64().unwrap_or(0.0).max(0.0) as usize;
        let y1 = arr[1].as_f64().unwrap_or(0.0).max(0.0) as usize;
        let x2 = (arr[2].as_f64().unwrap_or(0.0) as usize).min(w.saturating_sub(1));
        let y2 = (arr[3].as_f64().unwrap_or(0.0) as usize).min(h.saturating_sub(1));

        if x1 >= x2 || y1 >= y2 {
            continue;
        }

        // Draw horizontal lines (top and bottom edges)
        for t in 0..thickness {
            // Top edge
            let ty = y1 + t;
            if ty < h {
                for x in x1..=x2 {
                    let idx = (ty * w + x) * c;
                    if idx + 2 < output.len() {
                        output[idx] = color[0];
                        output[idx + 1] = color[1];
                        output[idx + 2] = color[2];
                    }
                }
            }
            // Bottom edge
            if y2 >= t {
                let by = y2 - t;
                if by < h {
                    for x in x1..=x2 {
                        let idx = (by * w + x) * c;
                        if idx + 2 < output.len() {
                            output[idx] = color[0];
                            output[idx + 1] = color[1];
                            output[idx + 2] = color[2];
                        }
                    }
                }
            }
        }

        // Draw vertical lines (left and right edges)
        for t in 0..thickness {
            // Left edge
            let lx = x1 + t;
            if lx < w {
                for y in y1..=y2 {
                    let idx = (y * w + lx) * c;
                    if idx + 2 < output.len() {
                        output[idx] = color[0];
                        output[idx + 1] = color[1];
                        output[idx + 2] = color[2];
                    }
                }
            }
            // Right edge
            if x2 >= t {
                let rx = x2 - t;
                if rx < w {
                    for y in y1..=y2 {
                        let idx = (y * w + rx) * c;
                        if idx + 2 < output.len() {
                            output[idx] = color[0];
                            output[idx + 1] = color[1];
                            output[idx + 2] = color[2];
                        }
                    }
                }
            }
        }
    }

    emit_image(&output, h, w, c)
}

// ── Helpers ────────────────────────────────────────────────────

/// Parse image blob into pixel data and dimensions.
/// Expects shape [H, W, C] or [H, W].
fn parse_image_blob(
    bytes: &[u8],
    shape: &Option<Vec<usize>>,
) -> Result<(Vec<f32>, usize, usize, usize), String> {
    let shape = shape
        .as_ref()
        .ok_or("image: blob missing shape metadata")?;

    if shape.len() < 2 {
        return Err(format!(
            "image: invalid shape {:?} (expected [H, W] or [H, W, C])",
            shape
        ));
    }

    let h = shape[0];
    let w = shape[1];
    let c = if shape.len() > 2 { shape[2] } else { 1 };

    if bytes.len() % 4 != 0 {
        return Err(format!(
            "image: f32 data length {} not divisible by 4",
            bytes.len()
        ));
    }

    let pixels: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let expected = h * w * c;
    if pixels.len() != expected {
        return Err(format!(
            "image: shape {:?} expects {} elements, got {}",
            shape, expected, pixels.len()
        ));
    }

    Ok((pixels, h, w, c))
}

/// Emit image blob output.
fn emit_image(
    pixels: &[f32],
    h: usize,
    w: usize,
    c: usize,
) -> Result<KernelOutput, String> {
    Ok(KernelOutput::Blob {
        data: pixels.iter().flat_map(|f| f.to_le_bytes()).collect(),
        content_type: "tensor/f32".to_string(),
        shape: Some(vec![h, w, c]),
    })
}

// ── Config Parsing ─────────────────────────────────────────────

/// Parse operation config.
fn parse_config(operations: &Value) -> Result<ImageConfig, String> {
    match operations {
        Value::String(s) => {
            let op = parse_op_string(s)?;
            Ok(ImageConfig { op })
        }
        Value::Object(obj) => {
            let op_name = obj
                .get("op")
                .and_then(|v| v.as_str())
                .ok_or("image: operations must have 'op' field")?;

            let op = match op_name {
                "decode" => ImageOp::Decode,
                "resize" => {
                    let mode = match obj
                        .get("mode")
                        .and_then(|v| v.as_str())
                        .unwrap_or("stretch")
                    {
                        "stretch" => ResizeMode::Stretch,
                        "letterbox" => ResizeMode::Letterbox,
                        "crop" => ResizeMode::Crop,
                        m => return Err(format!("image: unknown resize mode '{m}'")),
                    };

                    let target = obj.get("target").and_then(|v| v.as_u64()).map(|n| n as usize);
                    let width = obj.get("width").and_then(|v| v.as_u64()).map(|n| n as usize);
                    let height = obj.get("height").and_then(|v| v.as_u64()).map(|n| n as usize);

                    let (width, height) = match target {
                        Some(t) => (Some(t), Some(t)),
                        None => (width, height),
                    };

                    ImageOp::Resize {
                        width,
                        height,
                        mode,
                    }
                }
                "normalize" => {
                    let scale = obj
                        .get("scale")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(255.0) as f32;

                    let mean = obj.get("mean").and_then(|v| v.as_array()).map(|arr| {
                        [
                            arr.get(0)
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0) as f32,
                            arr.get(1)
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0) as f32,
                            arr.get(2)
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0) as f32,
                        ]
                    });

                    let std = obj.get("std").and_then(|v| v.as_array()).map(|arr| {
                        [
                            arr.get(0)
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0) as f32,
                            arr.get(1)
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0) as f32,
                            arr.get(2)
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0) as f32,
                        ]
                    });

                    ImageOp::Normalize { scale, mean, std }
                }
                "colorspace" => {
                    let to = match obj
                        .get("to")
                        .and_then(|v| v.as_str())
                        .ok_or("image: colorspace requires 'to' field")?
                    {
                        "rgb" => ColorSpace::RGB,
                        "bgr" => ColorSpace::BGR,
                        "grayscale" => ColorSpace::Grayscale,
                        c => return Err(format!("image: unknown colorspace '{c}'")),
                    };
                    ImageOp::Colorspace { to }
                }
                "layout" => {
                    let to = match obj
                        .get("to")
                        .and_then(|v| v.as_str())
                        .ok_or("image: layout requires 'to' field")?
                    {
                        "hwc" => MemoryLayout::HWC,
                        "chw" => MemoryLayout::CHW,
                        l => return Err(format!("image: unknown layout '{l}'")),
                    };
                    ImageOp::Layout { to }
                }
                "pad" => {
                    let target = obj.get("target").and_then(|v| v.as_u64()).map(|n| n as usize);
                    let width = obj.get("width").and_then(|v| v.as_u64()).map(|n| n as usize);
                    let height = obj.get("height").and_then(|v| v.as_u64()).map(|n| n as usize);
                    let value = obj
                        .get("value")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32;

                    let (width, height) = match target {
                        Some(t) => (Some(t), Some(t)),
                        None => (width, height),
                    };

                    ImageOp::Pad {
                        width,
                        height,
                        value,
                    }
                }
                "crop" => {
                    let x = obj
                        .get("x")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    let y = obj
                        .get("y")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    let width = obj
                        .get("width")
                        .and_then(|v| v.as_u64())
                        .ok_or("image: crop requires 'width'")? as usize;
                    let height = obj
                        .get("height")
                        .and_then(|v| v.as_u64())
                        .ok_or("image: crop requires 'height'")? as usize;
                    ImageOp::Crop { x, y, width, height }
                }
                "draw_bbox" => ImageOp::DrawBbox,
                _ => return Err(format!("image: unknown op '{op_name}'")),
            };

            Ok(ImageConfig { op })
        }
        _ => Err("image: operations must be string or object".into()),
    }
}

/// Parse shorthand op string.
fn parse_op_string(s: &str) -> Result<ImageOp, String> {
    match s.split_whitespace().next().unwrap_or("") {
        "decode" => Ok(ImageOp::Decode),
        other => Err(format!(
            "image: unknown op '{other}' (use object format for ops with params)"
        )),
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::BlobMeta;
    use crate::kernel::BlobData;
    use std::collections::HashMap;

    /// Create a minimal 2x2 PNG in memory for testing.
    fn create_test_png() -> Vec<u8> {
        use image::{ImageBuffer, Rgb};

        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(2, 2);

        // Set 2x2 pixel grid
        *img.get_pixel_mut(0, 0) = Rgb([255, 0, 0]); // Red
        *img.get_pixel_mut(1, 0) = Rgb([0, 255, 0]); // Green
        *img.get_pixel_mut(0, 1) = Rgb([0, 0, 255]); // Blue
        *img.get_pixel_mut(1, 1) = Rgb([255, 255, 0]); // Yellow

        let mut png_data = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png_data), image::ImageFormat::Png)
            .expect("failed to write PNG");

        png_data
    }

    #[test]
    fn test_decode_png() {
        let kernel = ImageKernel;
        let png_bytes = create_test_png();

        let input = KernelInput {
            json: serde_json::Value::Null,
            blobs: {
                let mut m = HashMap::new();
                m.insert("image".to_string(), BlobData {
                    bytes: png_bytes,
                    meta: BlobMeta {
                        size: 0,
                        content_type: "image/png".to_string(),
                        shape: None,
                    },
                });
                m
            },
        };

        let result = kernel
            .execute(input, serde_json::json!({"op": "decode"}))
            .unwrap();

        match result {
            KernelOutput::Blob {
                data,
                content_type,
                shape,
            } => {
                assert_eq!(content_type, "tensor/f32");
                assert_eq!(shape, Some(vec![2, 2, 3]));

                // Verify pixels
                let pixels: Vec<f32> = data
                    .chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();

                // Red pixel (255, 0, 0)
                assert!((pixels[0] - 255.0).abs() < 1e-5);
                assert!((pixels[1]).abs() < 1e-5);
                assert!((pixels[2]).abs() < 1e-5);
            }
            _ => panic!("expected Blob output"),
        }
    }

    #[test]
    fn test_resize_stretch() {
        let kernel = ImageKernel;

        // Create 4x4 image
        let pixels = vec![1.0; 4 * 4 * 3];
        let bytes: Vec<u8> = pixels.iter().flat_map(|&f: &f32| f.to_le_bytes()).collect();

        let input = KernelInput {
            json: serde_json::Value::Null,
            blobs: {
                let mut m = HashMap::new();
                m.insert("image".to_string(), BlobData {
                    bytes,
                    meta: BlobMeta {
                        size: 0,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![4, 4, 3]),
                    },
                });
                m
            },
        };

        let result = kernel
            .execute(
                input,
                serde_json::json!({"op": "resize", "target": 2, "mode": "stretch"}),
            )
            .unwrap();

        match result {
            KernelOutput::Blob { shape, .. } => {
                assert_eq!(shape, Some(vec![2, 2, 3]));
            }
            _ => panic!("expected Blob output"),
        }
    }

    #[test]
    fn test_normalize_scale() {
        let kernel = ImageKernel;

        // Create small image [1, 1, 3] with values [255, 128, 64]
        let pixels = vec![255.0, 128.0, 64.0];
        let bytes: Vec<u8> = pixels.iter().flat_map(|&f: &f32| f.to_le_bytes()).collect();

        let input = KernelInput {
            json: serde_json::Value::Null,
            blobs: {
                let mut m = HashMap::new();
                m.insert("image".to_string(), BlobData {
                    bytes,
                    meta: BlobMeta {
                        size: 0,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![1, 1, 3]),
                    },
                });
                m
            },
        };

        let result = kernel
            .execute(
                input,
                serde_json::json!({"op": "normalize", "scale": 255.0}),
            )
            .unwrap();

        match result {
            KernelOutput::Blob { data, shape, .. } => {
                assert_eq!(shape, Some(vec![1, 1, 3]));

                let normalized: Vec<f32> = data
                    .chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();

                // 255/255 = 1.0, 128/255 ≈ 0.502, 64/255 ≈ 0.251
                assert!((normalized[0] - 1.0).abs() < 1e-5);
                assert!((normalized[1] - 128.0 / 255.0).abs() < 1e-3);
                assert!((normalized[2] - 64.0 / 255.0).abs() < 1e-3);
            }
            _ => panic!("expected Blob output"),
        }
    }

    #[test]
    fn test_layout_hwc_to_chw() {
        let kernel = ImageKernel;

        // Create 2x2 RGB image (HWC): [0,1,2, 3,4,5, 6,7,8, 9,10,11]
        let pixels = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let bytes: Vec<u8> = pixels.iter().flat_map(|&f: &f32| f.to_le_bytes()).collect();

        let input = KernelInput {
            json: serde_json::Value::Null,
            blobs: {
                let mut m = HashMap::new();
                m.insert("image".to_string(), BlobData {
                    bytes,
                    meta: BlobMeta {
                        size: 0,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![2, 2, 3]),
                    },
                });
                m
            },
        };

        let result = kernel
            .execute(input, serde_json::json!({"op": "layout", "to": "chw"}))
            .unwrap();

        match result {
            KernelOutput::Blob { data, shape, .. } => {
                assert_eq!(shape, Some(vec![3, 2, 2]));

                let chw: Vec<f32> = data
                    .chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();

                // CHW layout: [R_all, G_all, B_all]
                // R: [0, 3, 6, 9], G: [1, 4, 7, 10], B: [2, 5, 8, 11]
                assert_eq!(chw[0], 0.0);
                assert_eq!(chw[1], 3.0);
                assert_eq!(chw[4], 1.0); // First G
                assert_eq!(chw[8], 2.0); // First B
            }
            _ => panic!("expected Blob output"),
        }
    }

    #[test]
    fn test_colorspace_rgb_to_grayscale() {
        let kernel = ImageKernel;

        // Create [1, 1, 3] with R=255, G=128, B=64
        let pixels = vec![255.0, 128.0, 64.0];
        let bytes: Vec<u8> = pixels.iter().flat_map(|&f: &f32| f.to_le_bytes()).collect();

        let input = KernelInput {
            json: serde_json::Value::Null,
            blobs: {
                let mut m = HashMap::new();
                m.insert("image".to_string(), BlobData {
                    bytes,
                    meta: BlobMeta {
                        size: 0,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![1, 1, 3]),
                    },
                });
                m
            },
        };

        let result = kernel
            .execute(
                input,
                serde_json::json!({"op": "colorspace", "to": "grayscale"}),
            )
            .unwrap();

        match result {
            KernelOutput::Blob { data, shape, .. } => {
                assert_eq!(shape, Some(vec![1, 1, 1]));

                let gray: Vec<f32> = data
                    .chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();

                // ITU-R BT.601: 0.299*255 + 0.587*128 + 0.114*64
                let expected = 0.299 * 255.0 + 0.587 * 128.0 + 0.114 * 64.0;
                assert!((gray[0] - expected).abs() < 1e-3);
            }
            _ => panic!("expected Blob output"),
        }
    }

    #[test]
    fn test_crop_basic() {
        let kernel = ImageKernel;

        // Create 4x4 RGB image with sequential values
        let mut pixels = Vec::new();
        for i in 0..4 * 4 * 3 {
            pixels.push(i as f32);
        }
        let bytes: Vec<u8> = pixels.iter().flat_map(|&f: &f32| f.to_le_bytes()).collect();

        let input = KernelInput {
            json: serde_json::Value::Null,
            blobs: {
                let mut m = HashMap::new();
                m.insert("image".to_string(), BlobData {
                    bytes,
                    meta: BlobMeta {
                        size: 0,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![4, 4, 3]),
                    },
                });
                m
            },
        };

        let result = kernel
            .execute(
                input,
                serde_json::json!({"op": "crop", "x": 1, "y": 1, "width": 2, "height": 2}),
            )
            .unwrap();

        match result {
            KernelOutput::Blob { shape, .. } => {
                assert_eq!(shape, Some(vec![2, 2, 3]));
            }
            _ => panic!("expected Blob output"),
        }
    }

    #[test]
    fn test_crop_clamp_bounds() {
        let kernel = ImageKernel;

        let pixels = vec![0.0f32; 2 * 2 * 3];
        let bytes: Vec<u8> = pixels.iter().flat_map(|&f: &f32| f.to_le_bytes()).collect();

        let input = KernelInput {
            json: serde_json::Value::Null,
            blobs: {
                let mut m = HashMap::new();
                m.insert("image".to_string(), BlobData {
                    bytes,
                    meta: BlobMeta {
                        size: 0,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![2, 2, 3]),
                    },
                });
                m
            },
        };

        // Crop larger than image - should clamp
        let result = kernel
            .execute(
                input,
                serde_json::json!({"op": "crop", "x": 0, "y": 0, "width": 10, "height": 10}),
            )
            .unwrap();

        match result {
            KernelOutput::Blob { shape, .. } => {
                assert_eq!(shape, Some(vec![2, 2, 3]));
            }
            _ => panic!("expected Blob output"),
        }
    }

    #[test]
    fn test_draw_bbox() {
        let kernel = ImageKernel;

        // Create 10x10 black image
        let pixels = vec![0.0f32; 10 * 10 * 3];
        let bytes: Vec<u8> = pixels.iter().flat_map(|&f: &f32| f.to_le_bytes()).collect();

        let input = KernelInput {
            json: serde_json::json!({
                "boxes": [[2.0, 2.0, 7.0, 7.0]],
                "color": [255.0, 0.0, 0.0],
                "thickness": 1
            }),
            blobs: {
                let mut m = HashMap::new();
                m.insert("image".to_string(), BlobData {
                    bytes,
                    meta: BlobMeta {
                        size: 0,
                        content_type: "tensor/f32".to_string(),
                        shape: Some(vec![10, 10, 3]),
                    },
                });
                m
            },
        };

        let result = kernel
            .execute(input, serde_json::json!({"op": "draw_bbox"}))
            .unwrap();

        match result {
            KernelOutput::Blob { data, shape, .. } => {
                assert_eq!(shape, Some(vec![10, 10, 3]));

                let out_pixels: Vec<f32> = data
                    .chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();

                // Check that a pixel on the top edge of the box is red
                let top_edge_idx = (2 * 10 + 4) * 3; // y=2, x=4
                assert_eq!(out_pixels[top_edge_idx], 255.0);
                assert_eq!(out_pixels[top_edge_idx + 1], 0.0);

                // Check that center pixel is still black
                let center_idx = (5 * 10 + 5) * 3; // y=5, x=5
                assert_eq!(out_pixels[center_idx], 0.0);
            }
            _ => panic!("expected Blob output"),
        }
    }
}
