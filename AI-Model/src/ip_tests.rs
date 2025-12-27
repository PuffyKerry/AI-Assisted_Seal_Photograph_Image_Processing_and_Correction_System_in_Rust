//Unit tests for Image Processing functions. Currently only has tests for DCP-based dehazing as other IP functions are still WIP.
//These tests were previously in main.rs before the ML training pipeline was implemented
//TODO: additional tests

use image::{DynamicImage, GenericImageView, RgbImage, Rgb};
use ndarray::Array3;
use IP_functions::dehaze::{dehaze_default_parameters_test, dehaze_static_test, dehaze_with_params};

//convert input image in DynamicImage format to a 3D array of pixel RGB values for DCP. Moved from main.rs
pub fn image_to_array3(img: &DynamicImage) -> Array3<f32> {
    let (width, height) = img.dimensions();
    let mut array = Array3::<f32>::zeros((height as usize, width as usize, 3));

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            array[[y as usize, x as usize, 0]] = pixel[0] as f32 / 255.0; //scale from RGB values of 0-255 to 0.0-1.0
            array[[y as usize, x as usize, 1]] = pixel[1] as f32 / 255.0;
            array[[y as usize, x as usize, 2]] = pixel[2] as f32 / 255.0;
        }
    }
    array
}

//convert 3D array of pixel RGB values to output image in RgbImage format for saving.
//replaced array2_to_image after I fleshed out the pipeline. Moved here from main.rs
pub fn array3_to_image(matrix: &Array3<f32>) -> RgbImage {
    let (height, width, _) = matrix.dim();
    let mut rgb_img = RgbImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let r = (matrix[[y, x, 0]].clamp(0.0, 1.0).powf(1.0/2.2) * 255.0) as u8; //applying gamma correction powf(1.0/2.2) to brighten the output image due to how monitors will expect gamma-corrected images but the algorithm outputs pure linear light values
            let g = (matrix[[y, x, 1]].clamp(0.0, 1.0).powf(1.0/2.2) * 255.0) as u8;
            let b = (matrix[[y, x, 2]].clamp(0.0, 1.0).powf(1.0/2.2) * 255.0) as u8;
            rgb_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    rgb_img
}

/// Run all IP engine tests - call this from main to test the image processing pipeline
/// Tests each image with both default He et al. paper parameters and custom tuned parameters
pub fn run_all_ip_tests() { //used ai to collate the testing I did manually earlier
    println!("\n========================================");
    println!("  Image Processing Engine Test Suite");
    println!("  DCP Dehazing Manual Testing");
    println!("========================================\n");

    //Test images - three test images for manual evaluation
    let test_images = vec!["fog-137794231410y.jpg", "bansui.jpg", "achuge.jpg"];

    for img_path in test_images.iter() {
        run_single_image_tests(img_path);
        println!(); //spacing between image tests
    }

    println!("========================================");
    println!("  IP Engine Tests Complete");
    println!("========================================\n");
}

/// Run both default and custom parameter tests on a single image
fn run_single_image_tests(img_path: &str) { //ai helped write more tests for me based on the original test I wrote
    println!("------------------------------------------");
    println!("Testing image: {}", img_path);
    println!("------------------------------------------");

    //Extract base name for output file naming
    let base_name = img_path.trim_end_matches(".jpg").trim_end_matches(".png");

    println!("\nAttempting to load: {}", img_path);
    let img = match image::open(img_path) {
        Ok(img) => img,
        Err(e) => {
            println!("Failed to open image {}: {}", img_path, e);
            return;
        }
    };

    let img_matrix = image_to_array3(&img); //convert image to 3D array of pixel RGB values

    //Test 1: Default parameters from He et al. paper
    println!("\n--- Test 1: Default He et al. Paper Parameters ---");
    println!("Dehazing IP pipeline test: Dark Channel Prior-based dehazing with hardcoded patch size of 15 X 15 pixels");
    println!("Parameters: omega=0.95, t0=0.1, guided_radius=60, guided_eps=0.0001");

    let dehazed_default = dehaze_default_parameters_test(&img_matrix, 15);
    let output_default = format!("output_dehazing_dcp_default_params_{}.jpg", base_name);

    let output_img = array3_to_image(&dehazed_default);
    output_img
        .save(&output_default)
        .expect("Failed to save");

    println!("Saved result to {}", output_default);

    //Test 2: Custom tuned parameters for potentially better results
    println!("\n--- Test 2: Custom Tuned Parameters ---");
    println!("Dehazing IP pipeline test: Dark Channel Prior-based dehazing with custom parameters");
    //Custom parameters - these can be adjusted based on testing results
    //omega=0.75 keeps more haze for realism, t0=0.25 prevents noise, smaller guided_radius=15 for performance
    let patch_size = 15;
    let omega = 0.75;       //haze retention factor, keeps 25% haze for realism (paper uses 0.95)
    let t0 = 0.25;          //minimum transmission to prevent noise in thick haze (paper uses 0.1)
    let top_percent = 0.001; //top 0.1% brightest pixels for atmospheric light estimation
    let guided_radius = 15; //guided filter window radius (paper uses 60, reduced for performance)
    let guided_eps = 0.0001; //guided filter regularization (smaller = sharper edges)

    println!("Parameters: patch_size={}, omega={}, t0={}, top_percent={}, guided_radius={}, guided_eps={}",
             patch_size, omega, t0, top_percent, guided_radius, guided_eps);

    let dehazed_custom = dehaze_with_params(&img_matrix, patch_size, omega, t0, top_percent, guided_radius, guided_eps);
    let output_custom = format!("output_dehazing_dcp_custom_params_{}.jpg", base_name);

    let output_img_custom = array3_to_image(&dehazed_custom);
    output_img_custom
        .save(&output_custom)
        .expect("Failed to save");

    println!("Saved result to {}", output_custom);

    //Test 3: Static test parameters (He et al. with some modifications for clarity)
    println!("\n--- Test 3: Static Test Parameters (Modified He et al.) ---");
    println!("Dehazing IP pipeline test: Dark Channel Prior-based dehazing with dehaze_static_test");
    println!("Parameters: omega=0.75, t0=0.25, guided_radius=15 (other params are He et al. defaults)");

    let dehazed_static = dehaze_static_test(&img_matrix, 15);
    let output_static = format!("output_dehazing_dcp_static_params_{}.jpg", base_name);

    let output_img_static = array3_to_image(&dehazed_static);
    output_img_static
        .save(&output_static)
        .expect("Failed to save");

    println!("Saved result to {}", output_static);
}

/// Run a quick single test on one image - for quick debugging
#[allow(dead_code)]
pub fn run_quick_test() { //original test I wrote in main.rs that was kept for reference.
    let img_path = "fog-137794231410y.jpg"; //testing DCP implementation so far on a single test image
    println!("Attempting to load: {}", img_path);
    let img = image::open(img_path).expect("Failed to open image");

    let img_matrix = image_to_array3(&img); //convert image to 3D array of pixel RGB values

    println!("Dehazing IP pipeline test: Dark Channel Prior-based dehazing with hardcoded patch size of 15 X 15 pixels for now");
    let dark_channel_dehazed = dehaze_default_parameters_test(&img_matrix, 15);

    let output_img = array3_to_image(&dark_channel_dehazed);
    output_img
        .save("output_dehazing_dcp_default_params_fog-137794231410y.jpg")
        .expect("Failed to save");

    println!("Saved result to output_dehazing_dcp_default_params_fog-137794231410y.jpg");
}

#[cfg(test)]
mod tests { //tests for helper functions, AI generated
    use super::*;

    #[test]
    fn test_image_to_array3_dimensions() {
        // Create a small test image
        let img = DynamicImage::new_rgb8(10, 20);
        let array = image_to_array3(&img);

        assert_eq!(array.dim(), (20, 10, 3)); // height x width x channels
    }

    #[test]
    fn test_array3_to_image_dimensions() {
        let array = Array3::<f32>::zeros((20, 10, 3));
        let img = array3_to_image(&array);

        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 20);
    }

    #[test]
    fn test_pixel_value_normalization() {
        let mut img = image::RgbImage::new(1, 1);
        img.put_pixel(0, 0, Rgb([128, 64, 255]));
        let dynamic_img = DynamicImage::ImageRgb8(img);

        let array = image_to_array3(&dynamic_img);

        // Check values are normalized to [0, 1]
        assert!((array[[0, 0, 0]] - 128.0/255.0).abs() < 0.01);
        assert!((array[[0, 0, 1]] - 64.0/255.0).abs() < 0.01);
        assert!((array[[0, 0, 2]] - 1.0).abs() < 0.01);
    }
}
