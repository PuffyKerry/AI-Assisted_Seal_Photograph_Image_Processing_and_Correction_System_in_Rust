// =============================================================================
// web_server/src/convert.rs — Image ↔ ndarray ↔ base64 conversion utilities
// Ported from AI-Model/src/ip_tests.rs (image_to_array3, array3_to_image)
// Plus base64 encode/decode for JSON transport over HTTP
// =============================================================================

use image::{DynamicImage, GenericImageView, RgbImage, Rgb};
use ndarray::Array3;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

/// Convert a DynamicImage to an Array3<f32> [H, W, 3] with values in [0, 1]
pub fn image_to_array3(img: &DynamicImage) -> Array3<f32> {
    let (width, height) = img.dimensions();
    let mut array = Array3::<f32>::zeros((height as usize, width as usize, 3));

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            array[[y as usize, x as usize, 0]] = pixel[0] as f32 / 255.0;
            array[[y as usize, x as usize, 1]] = pixel[1] as f32 / 255.0;
            array[[y as usize, x as usize, 2]] = pixel[2] as f32 / 255.0;
        }
    }
    array
}

/// Convert an Array3<f32> [H, W, 3] (values in [0, 1]) back to an RgbImage
/// Applies display gamma correction (1/2.2) since IP pipeline outputs linear light values
pub fn array3_to_image(matrix: &Array3<f32>) -> RgbImage {
    let (height, width, _) = matrix.dim();
    let mut rgb_img = RgbImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let r = (matrix[[y, x, 0]].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
            let g = (matrix[[y, x, 1]].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
            let b = (matrix[[y, x, 2]].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
            rgb_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    rgb_img
}

/// Decode a base64-encoded image string into an Array3<f32>
pub fn decode_base64_image(b64: &str) -> Result<(DynamicImage, Array3<f32>), String> {
    let bytes = BASE64
        .decode(b64.trim())
        .map_err(|e| format!("Base64 decode error: {}", e))?;

    let img = image::load_from_memory(&bytes)
        .map_err(|e| format!("Image decode error: {}", e))?;

    let array = image_to_array3(&img);
    Ok((img, array))
}

/// Encode an Array3<f32> result into a base64 JPEG string
pub fn encode_image_base64_jpeg(matrix: &Array3<f32>) -> Result<String, String> {
    let rgb_img = array3_to_image(matrix);
    let mut cursor = std::io::Cursor::new(Vec::new());
    rgb_img
        .write_to(&mut cursor, image::ImageFormat::Jpeg)
        .map_err(|e| format!("JPEG encode error: {}", e))?;

    Ok(BASE64.encode(cursor.into_inner()))
}

/// Encode an Array3<f32> result into raw JPEG bytes
pub fn encode_image_jpeg_bytes(matrix: &Array3<f32>) -> Result<Vec<u8>, String> {
    let rgb_img = array3_to_image(matrix);
    let mut cursor = std::io::Cursor::new(Vec::new());
    rgb_img
        .write_to(&mut cursor, image::ImageFormat::Jpeg)
        .map_err(|e| format!("JPEG encode error: {}", e))?;

    Ok(cursor.into_inner())
}

