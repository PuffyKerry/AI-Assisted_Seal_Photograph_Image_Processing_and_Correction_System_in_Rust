//Main file for project, runs testing code
//Command line interface for demo, testing, and training added 12/26, with much AI assistance for grunt work / printing things / I/O similar to C++
//TODO: clean up and make code more modular. File structure can still be improved a bit. Make run functions more generic (current hard coding is a bit messy)

mod linear_regression;
mod extraction;
mod training;
mod ip_tests;

use std::env;
use std::path::{Path, PathBuf};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use image::{GrayImage, Luma};
use ndarray::{Array2, Array3};
use IP_functions::dehaze::{dehaze_default_parameters_test, dehaze_with_params};
use rand::seq::SliceRandom;
use rand::rng;
use memmap2::Mmap;
use rayon::prelude::*;
use crate::training::{train_haze_regressor, train_haze_regressor_precomputed};
use crate::ip_tests::{image_to_array3, array3_to_image, run_all_ip_tests};
use crate::extraction::extract_mean_dark_channel;

//CLI flags handling was fully AI-generated, did check over it. Rather simple/similar to C++, so I determined there wasn't really a reason I should do it manually from a learning perspective.
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        //Default: run ML training demo
        run_ml_demo();
        return;
    }

    match args[1].as_str() {
        "--help" | "-h" => print_help(),
        "--ip-tests" => run_all_ip_tests(),
        "--train-full" => run_full_dataset_training(),
        "--dehaze" => {
            if args.len() < 3 {
                println!("Error: --dehaze requires an image path");
                println!("Usage: cargo run -p ai-model -- --dehaze path/to/image.jpg");
                return;
            }
            dehaze_single_image(&args[2]);
        }
        "--dehaze-custom" => {
            //Usage: --dehaze-custom FILE omega t0 patch_size guided_radius guided_eps
            if args.len() < 8 {
                println!("Error: --dehaze-custom requires image path and 5 parameters");
                println!("Usage: cargo run -p ai-model -- --dehaze-custom FILE omega t0 patch_size guided_radius guided_eps");
                println!("Example with custom parameters found to be optimal by developer: cargo run -p ai-model -- --dehaze-custom image.jpg 0.75 0.25 15 15 0.0001");
                return;
            }
            let omega: f32 = args[3].parse().unwrap_or_else(|_| { println!("Invalid omega, using 0.75"); 0.75 });
            let t0: f32 = args[4].parse().unwrap_or_else(|_| { println!("Invalid t0, using 0.25"); 0.25 });
            let patch_size: usize = args[5].parse().unwrap_or_else(|_| { println!("Invalid patch_size, using 15"); 15 });
            let guided_radius: usize = args[6].parse().unwrap_or_else(|_| { println!("Invalid guided_radius, using 15"); 15 });
            let guided_eps: f32 = args[7].parse().unwrap_or_else(|_| { println!("Invalid guided_eps, using 0.0001"); 0.0001 });
            dehaze_with_custom_params(&args[2], omega, t0, patch_size, guided_radius, guided_eps);
        }
        "--demo" => run_ml_demo(),
        _ => {
            println!("Unknown option: {}", args[1]);
            print_help();
        }
    }
}

fn print_help() { //Ai-generated to save time. Checked for accuracy.
    println!("=== AI-Assisted Seal Photograph Image Processing System ===\n");
    println!("Usage: cargo run -p ai-model -- [OPTION]\n");
    println!("Options:");
    println!("  (no args)      Run ML training demo on test images");
    println!("  --demo         Same as no args - ML training demo");
    println!("  --ip-tests     Run IP engine tests (dehazing on fog, bansui, achuge)");
    println!("  --train-full   Train on full SealID dataset (requires dataset setup)");
    println!("  --dehaze FILE  Dehaze a specific image file with default parameters");
    println!("  --dehaze-custom FILE omega t0 patch_size guided_radius guided_eps");
    println!("                 Dehaze with custom DCP parameters");
    println!("  --help, -h     Show this help message\n");
    println!("Custom Dehazing Parameters:");
    println!("  omega          Haze retention factor [0-1], lower = more dehaze (default: 0.95)");
    println!("  t0             Min transmission [0-1], higher = less noise (default: 0.1)");
    println!("  patch_size     Dark channel patch size in pixels (default: 15)");
    println!("  guided_radius  Guided filter radius, larger = smoother (default: 60)");
    println!("  guided_eps     Guided filter epsilon, smaller = sharper (default: 0.0001)\n");
    println!("Example:");
    println!("  cargo run -p ai-model -- --dehaze-custom image.jpg 0.75 0.25 15 15 0.0001\n");
    println!("Dataset Setup:");
    println!("  Download SealID from: https://etsin.fairdata.fi/dataset/22b5191e-f24b-4457-93d3-95797c900fc0");
    println!("  Extract to: dataset/SealID/full images/source_database/");
}

//Run ML demo on a few test images
//AI-generated as a truncated version of the full training demo, meant to show that the ML model can be trained and that the IP engine works without spending time/resources on training with the full dataset.
fn run_ml_demo() {
    println!("=== AI-Assisted Seal Photograph Image Processing System ===");
    println!("=== ML Training Demo ===\n");

    //Test images - in production, these would be loaded from a labeled dataset
    let test_images = vec!["fog-137794231410y.jpg", "bansui.jpg"];

    let mut images: Vec<Array3<f32>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();

    //Load available test images and estimate haze labels using DCP features
    for path in test_images.iter() {
        println!("Attempting to load: {}", path);
        match image::open(path) {
            Ok(img) => {
                let img_matrix = image_to_array3(&img); //keeping the old, inefficient, but very step-by-step code that relied on directly calling image_to_array3 here in the demo as it is only 2 images

                //Estimate haze level using DCP features as a proxy (same as full dataset training)
                let mean = extract_mean_dark_channel(&img_matrix, 15);
                let estimated_haze = mean.clamp(0.0, 1.0); //mean dark channel as proxy for haze

                images.push(img_matrix);
                labels.push(estimated_haze);
                println!("  Loaded successfully, estimated haze label: {:.3}", estimated_haze);
            }
            Err(e) => {
                println!("  Failed to load {}: {}", path, e);
            }
        }
    }

    if images.len() < 2 {
        println!("\nWarning: Need at least 2 images for meaningful training demonstration.");
        println!("Falling back to IP engine tests...\n");
        run_all_ip_tests();
        return;
    }

    println!("\n=== Training Linear Regression Haze Regressor ===");
    println!("Training with {} images...", images.len());

    let patch_size = 15;
    let learning_rate = 0.1;
    let epochs = 100;

    let regressor = train_haze_regressor(&images, &labels, patch_size, learning_rate, epochs);

    println!("\n=== Evaluating Trained Model ===");
    let mse = training::evaluate_mse(&regressor, &images, &labels, patch_size);
    println!("\nTraining set MSE: {:.4}", mse);

    println!("\n=== Running Dehazing Pipeline ===");
    //Run dehazing on the first (foggy) image as demonstration
    if !images.is_empty() {
        let dehazed = dehaze_default_parameters_test(&images[0], patch_size);
        let output_img = array3_to_image(&dehazed);
        output_img
            .save("output_dehazing_dcp_ml_demo.jpg")
            .expect("Failed to save");
        println!("Saved dehazed result to output_dehazing_dcp_ml_demo.jpg");
    }

    println!("\n=== Demo Complete ===");
    println!("Model weights: {:?}", regressor.model.weights);
    println!("Model bias: {:.4}", regressor.model.bias);
    println!("Feature normalization mins: {:?}", regressor.feature_mins);
    println!("Feature normalization ranges: {:?}", regressor.feature_ranges);
}

//Train on full SealID dataset
fn run_full_dataset_training() { //I/O code was AI generated, flow was mine
    println!("=== AI-Assisted Seal Photograph Image Processing System ===");
    println!("=== Full Dataset Training ===\n");

    let dataset_path = Path::new("dataset/SealID/full images/source_database");
    if !dataset_path.exists() {
        println!("Error: training dataset not found at {:?}", dataset_path);
        println!("\nPlease download the SealID dataset and extract it to dataset/SealID/ per the instructions in the README");
        println!("Download from: https://etsin.fairdata.fi/dataset/22b5191e-f24b-4457-93d3-95797c900fc0");
        return;
    }

    //Find all image files in the dataset
    let image_paths = find_images_in_directory(dataset_path);
    if image_paths.is_empty() {
        println!("Error: No images found in {:?}", dataset_path);
        return;
    }
    println!("Found {} images in dataset", image_paths.len());

    //Load images with precomputed features in one parallel pass - optimization to avoid computing DCP twice
    let patch_size = 15;
    println!("Loading images and extracting features in parallel...");
    let (images, labels, features) = load_images_with_features(&image_paths, patch_size);
    println!("Successfully loaded {} images with precomputed features", images.len());

    if images.len() < 2 {
        println!("Error: Need at least 2 images to train");
        return;
    }

    println!("\n=== Training on {} images ===", images.len());

    let learning_rate = 0.01; //smaller learning rate for larger dataset for efficiency
    let epochs = 200; //less epochs than optimal due to dataset size making training rather slow

    //Use precomputed features to skip feature extraction during training
    let regressor = train_haze_regressor_precomputed(&features, &labels, learning_rate, epochs);

    println!("\n=== Training Complete ===");
    println!("Model weights: {:?}", regressor.model.weights);
    println!("Model bias: {:.4}", regressor.model.bias);

    //Test on a few random images from the query dataset to show functionality
    println!("\n=== Sample Predictions on Query Set ===");
    let query_path = Path::new("dataset/SealID/full images/source_query/");
    if !query_path.exists() {
        println!("Query dataset path not found: {:?}", query_path);
        return;
    }

    let query_images = find_random_x_images_in_directory(query_path, 5);
    if query_images.is_empty() {
        println!("No images found in query dataset at {:?}", query_path);
        return;
    }

    //Load query images
    let test_images = load_images_parallel(&query_images);
    for cur_path in &query_images {
        println!("Loaded: {}", cur_path.file_name().unwrap_or_default().to_string_lossy());
    }
    let test_labels: Vec<f64> = test_images.iter().map(|img| extract_mean_dark_channel(img, patch_size).clamp(0.0, 1.0)).collect();

    if !test_images.is_empty() {
        let mse = training::evaluate_mse(&regressor, &test_images, &test_labels, patch_size);
        println!("\nQuery set MSE: {:.4}", mse);
    }
}

//Dehaze a single image from command line
fn dehaze_single_image(img_path: &str) { //ai generated, simple function calls, so it's fine.
    println!("=== Dehazing Image: {} ===\n", img_path);

    let img = match image::open(img_path) {
        Ok(img) => img,
        Err(e) => {
            println!("Error: Failed to open image: {}", e);
            return;
        }
    };

    let img_matrix = image_to_array3(&img);

    println!("Image loaded successfully");
    println!("Running Dark Channel Prior dehazing...\n");

    let dehazed = dehaze_default_parameters_test(&img_matrix, 15);

    //Generate output filename
    let input_path = Path::new(img_path);
    let stem = input_path.file_stem().unwrap_or_default().to_str().unwrap_or("output");
    let output_path = format!("output_dehazed_{}.jpg", stem);

    let output_img = array3_to_image(&dehazed);
    output_img
        .save(&output_path)
        .expect("Failed to save");

    println!("Saved dehazed result to {}", output_path);
}

//Dehaze a single image with custom DCP parameters from command line
//just ai putting a wrapper around a function call
fn dehaze_with_custom_params(img_path: &str, omega: f32, t0: f32, patch_size: usize, guided_radius: usize, guided_eps: f32) {
    println!("=== Dehazing Image with Custom Parameters ===");
    println!("Image: {}", img_path);
    println!("Parameters: omega={}, t0={}, patch_size={}, guided_radius={}, guided_eps={}\n",
             omega, t0, patch_size, guided_radius, guided_eps);

    let img = match image::open(img_path) {
        Ok(img) => img,
        Err(e) => {
            println!("Error: Failed to open image: {}", e);
            return;
        }
    };

    let img_matrix = image_to_array3(&img);

    println!("Image loaded successfully");
    println!("Running Dark Channel Prior dehazing with custom parameters...\n");

    //top_percent is hardcoded to 0.001 (top 0.1% brightest pixels for atmospheric light)
    let top_percent = 0.001;
    let dehazed = dehaze_with_params(&img_matrix, patch_size, omega, t0, top_percent, guided_radius, guided_eps);

    //Generate output filename with parameter info
    let input_path = Path::new(img_path);
    let stem = input_path.file_stem().unwrap_or_default().to_str().unwrap_or("output");
    let output_path = format!("output_dehazed_custom_{}.jpg", stem);

    let output_img = array3_to_image(&dehazed);
    output_img
        .save(&output_path)
        .expect("Failed to save");

    println!("Saved dehazed result to {}", output_path);
}

//Find all image files recursively in a directory
fn find_images_in_directory(dir: &Path) -> Vec<PathBuf> { //this was AI, I learned recursion in high school. Also not in use as of now as the model is trained off of only a single directory with no directories inside it.
    let mut images = Vec::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                images.extend(find_images_in_directory(&path));
            } else if let Some(ext) = path.extension() {
                let ext = ext.to_str().unwrap_or("").to_lowercase();
                if ext == "jpg" || ext == "jpeg" || ext == "png" {
                    images.push(path);
                }
            }
        }
    }

    images
}

//fast function for getting a few random images from a directory, for example queries.
fn find_random_x_images_in_directory(dir: &Path, num: usize) -> Vec<PathBuf> {
    let mut images = find_images_in_directory(dir);
    let mut rng = rng();
    images.shuffle(&mut rng);
    images.into_iter().take(num).collect()
}

//Load downsized images in parallel and prints current progress using memory-mapped files for faster I/O, as image_to_array3() was slow due to being single-core and would store repeated copies of already loaded images in memory.
//@param: paths: array of paths to image files passed from main()
//@return: vector of 3D arrays from image_to_array3()
fn load_images_parallel(paths: &[PathBuf]) -> Vec<Array3<f32>> {
    let counter = AtomicUsize::new(0);
    let total = paths.len();

    paths.par_iter().filter_map(|path| {
        let file = fs::File::open(path).ok()?; //memory mapping to avoid redundant loading - optimization
        let mmap = unsafe { //fine as long as file path is checked beforehand
            Mmap::map(&file).ok()?
        };

        let img = image::load_from_memory(&mmap).ok()?; //extract from memory map buffer to load image
        let resized_img = img.resize((img.width() / 4).max(1), (img.height() / 4).max(1), image::imageops::FilterType::CatmullRom); //downsample for faster processing as images are large, truncated ints are fine as the error is very small
        let img_matrix = image_to_array3(&resized_img); //yes, this is literally just a neat wrapper around image_to_array3, but it avoids redundant loading and speeds up I/O significantly for large datasets.

        let curcount = counter.fetch_add(1, Ordering::Relaxed) + 1; //needed to use an atomic counter to minimize race conditions
        if curcount % 5 == 0 || curcount == total {
            println!("Loaded {}/{} images...", curcount, total);
        }

        Some(img_matrix)
    }).collect()
}

//Load images in parallel with precomputed features and labels - optimization to avoid computing DCP features twice
//Computes extract_all_features() during loading so we don't need to recompute during training
//@param: paths: array of paths to image files
//@param: patch_size: patch size for DCP feature extraction
//@return: tuple of (images, labels, feature_matrix) where feature_matrix is [num_images x 5]
fn load_images_with_features(paths: &[PathBuf], patch_size: usize) -> (Vec<Array3<f32>>, Vec<f64>, Array2<f64>) {
    let counter = AtomicUsize::new(0);
    let total = paths.len();

    //parallel load images and extract features in one pass
    let results: Vec<_> = paths.par_iter().filter_map(|path| {
        let file = fs::File::open(path).ok()?;
        let mmap = unsafe { Mmap::map(&file).ok()? };

        let img = image::load_from_memory(&mmap).ok()?;
        let img_matrix = image_to_array3(&img);

        //extract all features during loading to avoid recomputing in training
        let features = extraction::extract_all_features(&img_matrix, patch_size);
        let label = features[0].clamp(0.0, 1.0); //mean_dark_channel is features[0]

        let curcount = counter.fetch_add(1, Ordering::Relaxed) + 1;
        if curcount % 5 == 0 || curcount == total {
            println!("Processed {}/{} images...", curcount, total);
        }

        Some((img_matrix, label, features))
    }).collect();

    //unzip results and build feature matrix
    let num_samples = results.len();
    let mut feature_matrix = Array2::<f64>::zeros((num_samples, 5));
    let mut images = Vec::with_capacity(num_samples);
    let mut labels = Vec::with_capacity(num_samples);

    for (i, (img, label, feat)) in results.into_iter().enumerate() {
        for j in 0..5 {
            feature_matrix[[i, j]] = feat[j];
        }
        images.push(img);
        labels.push(label);
    }

    (images, labels, feature_matrix)
}

//convert matrix of calculated dark channel ratios to grayscale image representing haze levels. May move to other module.
//OUTMODED - was used for testing of DCP haze detection
#[allow(dead_code)]
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