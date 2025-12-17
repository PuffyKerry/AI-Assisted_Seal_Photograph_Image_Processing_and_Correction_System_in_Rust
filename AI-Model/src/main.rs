//Main file for project, runs testing code
//TODO: clean up and make code more modular

mod linear_regression;

use image::{DynamicImage, GenericImageView, GrayImage, Luma};
use ndarray::{Array2, Array3};
use IP_functions::dcp::find_dark_channel;

fn main() {
    let img_path = "achuge.jpg"; //testing DCP implementation so far on a single test image from the source_database of non-annotated images
    println!("Attempting to load: {}", img_path);
    let img = image::open(img_path).expect("Failed to open image");
    let (width, height) = img.dimensions();

    let img_matrix = image_to_array3(&img); //convert image to 3D array of pixel RGB values

    println!("Calculating Dark Channel with hardcoded patch size of 5 X 5 pixels for now");
    let dark_channel = find_dark_channel(&img_matrix, 5);

    let output_img = array2_to_image(&dark_channel);
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

//convert matrix of calculated dark channel ratios to grayscale image representing haze levels. May move to other module.
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