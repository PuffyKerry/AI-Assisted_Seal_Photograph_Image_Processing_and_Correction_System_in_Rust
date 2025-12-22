//Main file for project, runs testing code
//TODO: clean up and make code more modular. Dearly need to make file structure a bit more organized. Very much need to do linear regression now.

mod linear_regression;

use image::{DynamicImage, GenericImageView, GrayImage, Luma, RgbImage, Rgb};
use ndarray::{Array2, Array3};
use IP_functions::dehaze::dehaze_static_test;

fn main() {
    let img_path = "achuge.jpg"; //testing DCP implementation so far on a single test image from the source_database of non-annotated images
    println!("Attempting to load: {}", img_path);
    let img = image::open(img_path).expect("Failed to open image");

    let img_matrix = image_to_array3(&img); //convert image to 3D array of pixel RGB values

    println!("Dehazing IP pipeline test: Dark Channel Prior-based dehazing with hardcoded patch size of 5 X 5 pixels for now");
    let dark_channel_dehazed = dehaze_static_test(&img_matrix, 5);

    let output_img = array3_to_image(&dark_channel_dehazed);
    output_img
        .save("output_dark_channel.png")
        .expect("Failed to save");

    println!("Saved result to output_dark_channel.png");
}

//convert input image in DynamicImage format to a 3D array of pixel RGB values for DCP. May move to other module.
fn image_to_array3(img: &DynamicImage) -> Array3<f32> {
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

//convert 3D array of pixel RGB values to output image in RgbImage format for saving. May move to other module.
//replaced array2_to_image after I fleshed out the pipeline
fn array3_to_image(matrix: &Array3<f32>) -> RgbImage {
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

//convert matrix of calculated dark channel ratios to grayscale image representing haze levels. May move to other module.
//OUTMODED - was used for testing of DCP haze detection
fn array2_to_image(matrix: &Array2<f32>) -> GrayImage {
    let (height, width) = matrix.dim();
    let mut gray_img = GrayImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let val = (matrix[[y, x]] * 255.0) as u8; //scale from 0.0-1.0 back to 0-255 for opacity values
            gray_img.put_pixel(x as u32, y as u32, Luma([val]));
        }
    }
    gray_img
}