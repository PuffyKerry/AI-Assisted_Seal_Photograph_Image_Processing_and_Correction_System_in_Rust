//Main file for project, runs testing and production code
//Command line interface for demo, testing, and training added 12/26, with much AI assistance for grunt work / printing things / I/O similar to C++
//TODO: clean up and make code more modular. File structure can still be improved a bit. Make run functions more generic (current hard coding is a bit messy)
//Mega-TODO (1/14): clean up code significantly, move helper functions to different files
//TODO 1/29: cut unnecessary functions, or keep them for demonstration? Perhaps move to a different "old code" folder? Need to decide.

//TODO 2/3: take an entire week or so just to clean up the code and make it more modular / robust. AKA do the other TODO's

// Required for wgpu 26 / burn 0.20.1 compatibility - wgpu-core types have deep nesting
#![recursion_limit = "512"]

mod linear_regression;
mod extraction;
mod training;
mod ip_tests;
mod iteration_2_CNN;  //Iteration 2: CNN-based haze detection

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
use burn::backend::ndarray as burn_ndarray;  //for CNN demo device

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
        "--train-full-demo" => run_full_dataset_training(),
        "--train-cnn" => run_cnn_dataset_training(),  //Iteration 2: CNN training on full dataset (CPU)
        "--train-cnn-gpu" => run_cnn_dataset_training_gpu(),  //Iteration 2: CNN training on full dataset (GPU - FAST)
        "--train-cnn-gpu-save" => {
            //Train CNN on GPU and save model for later inference
            if args.len() < 3 {
                println!("Error: --train-cnn-gpu-save requires a model output path");
                println!("Usage: cargo run -p ai-model -- --train-cnn-gpu-save cnn_model");
                return;
            }
            run_cnn_gpu_training_and_save(&args[2]);
        }
        "--process-cnn-gpu" => {
            //Load saved CNN model and process an image on GPU
            if args.len() < 5 {
                println!("Error: --process-cnn-gpu requires model path, input image, and output path");
                println!("Usage: cargo run -p ai-model -- --process-cnn-gpu cnn_model input.jpg output.jpg");
                return;
            }
            run_process_with_cnn_gpu(&args[2], &args[3], &args[4]);
        }
        "--demo-cnn" => run_cnn_demo_standalone(),   //Iteration 2: CNN demo on test images
        "--train-save" => { //Iteration 2: CPU-only training on demo images and save model
            if args.len() < 3 {
                println!("Error: --train-save requires a model output path");
                println!("Usage: cargo run -p ai-model -- --train-save model.json");
                return;
            }
            run_train_and_save(&args[2]);
        }
        "--train-full-save" => {
            if args.len() < 3 {
                println!("Error: --train-full-save requires a model output path");
                println!("Usage: cargo run -p ai-model -- --train-full-save model.json");
                return;
            }
            run_full_training_and_save(&args[2]);
        }
        "--process" => {
            // Load saved model and process an image
            if args.len() < 5 {
                println!("Error: --process requires model path, input image, and output path");
                println!("Usage: cargo run -p ai-model -- --process model.json input.jpg output.jpg");
                return;
            }
            run_process_with_model(&args[2], &args[3], &args[4]);
        }
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
        "--clahe" => {
            if args.len() < 3 {
                println!("Error: --clahe requires an image path");
                println!("Usage: cargo run -p ai-model -- --clahe path/to/image.jpg");
                return;
            }
            ip_tests::clahe_single_image(&args[2]);
        }
        "--clahe-custom" => {
            //Usage: --clahe-custom FILE grid_h grid_w clip_limit
            if args.len() < 6 {
                println!("Error: --clahe-custom requires image path and 3 parameters");
                println!("Usage: cargo run -p ai-model -- --clahe-custom FILE grid_h grid_w clip_limit");
                println!("Example: cargo run -p ai-model -- --clahe-custom image.jpg 8 8 3.0");
                return;
            }
            let grid_h: usize = args[3].parse().unwrap_or_else(|_| { println!("Invalid grid_h, using 8"); 8 });
            let grid_w: usize = args[4].parse().unwrap_or_else(|_| { println!("Invalid grid_w, using 8"); 8 });
            let clip_limit: f32 = args[5].parse().unwrap_or_else(|_| { println!("Invalid clip_limit, using 2.5"); 2.5 });
            ip_tests::clahe_with_custom_params(&args[2], grid_h, grid_w, clip_limit);
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
    println!("  --train-full-demo");
    println!("                 Train linear regression on full SealID dataset");
    println!("  --train-save MODEL_PATH");
    println!("                 Train on demo images and save model to file");
    println!("  --train-full-save MODEL_PATH");
    println!("                 Train on full SealID dataset and save model to file");
    println!("  --process MODEL_PATH INPUT_IMAGE OUTPUT_IMAGE");
    println!("                 Load saved model and dehaze an image");
    println!("  --train-cnn    Train CNN on full SealID dataset (CPU - slow)");
    println!("  --train-cnn-gpu");
    println!("                 Train CNN on full SealID dataset (GPU - FAST, use only for devices with dedicated GPUs (graphics cards))");
    println!("  --train-cnn-gpu-save MODEL_PATH");
    println!("                 Train CNN on GPU and save model for later inference");
    println!("  --process-cnn-gpu MODEL_PATH INPUT_IMAGE OUTPUT_IMAGE");
    println!("                 Load saved CNN model and dehaze an image (GPU)");
    println!("  --demo-cnn     CNN training demo on test images (Iteration 2)");
    println!("  --dehaze FILE  Dehaze a specific image file with default parameters");
    println!("  --dehaze-custom FILE omega t0 patch_size guided_radius guided_eps");
    println!("                 Dehaze with custom DCP parameters");
    println!("  --clahe FILE   Enhance contrast of an image with CLAHE (default parameters)");
    println!("  --clahe-custom FILE grid_h grid_w clip_limit");
    println!("                 Enhance contrast with custom CLAHE parameters");
    println!("  --help, -h     Show this help message\n");
    println!("Model Persistence:");
    println!("  Train and save: cargo run -p ai-model -- --train-save haze_model.json");
    println!("  Process image:  cargo run -p ai-model -- --process haze_model.json foggy.jpg clear.jpg\n");
    println!("Custom Dehazing Parameters:");
    println!("  omega          Haze retention factor [0-1], lower = more dehaze (default: 0.95)");
    println!("  t0             Min transmission [0-1], higher = less noise (default: 0.1)");
    println!("  patch_size     Dark channel patch size in pixels (default: 15)");
    println!("  guided_radius  Guided filter radius, larger = smoother (default: 60)");
    println!("  guided_eps     Guided filter epsilon, smaller = sharper (default: 0.0001)\n");
    println!("CLAHE Parameters:");
    println!("  grid_h         Number of tile rows (default: 8, more = more local contrast)");
    println!("  grid_w         Number of tile columns (default: 8)");
    println!("  clip_limit     Contrast limit multiplier (default: 2.5, range 1.5-4.0)\n");
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

//#[allow(dead_code)] //not yet ready for use, switched to a short demo function instead //IT'S ALIVE, IT'S ALIVE!
//Iteration 2: Train CNN on full SealID dataset
//AI-generated BECAUSE PIPELINE IS ALMOST IDENTICAL TO LINEAR REGRESSION PIPELINE
fn run_cnn_dataset_training() {
    println!("=== AI-Assisted Seal Photograph Image Processing System ===");
    println!("=== Iteration 2: CNN Dataset Training ===\n");

    let dataset_path = Path::new("dataset/SealID/full images/source_database");
    if !dataset_path.exists() {
        println!("Error: training dataset not found at {:?}", dataset_path);
        println!("\nPlease download the SealID dataset and extract it to dataset/SealID/ per the instructions in the README");
        return;
    }

    //Find all image files in the dataset
    let image_paths = find_images_in_directory(dataset_path);
    if image_paths.is_empty() {
        println!("Error: No images found in {:?}", dataset_path);
        return;
    }
    println!("Found {} images in dataset", image_paths.len());

    //Load images with labels (using mean dark channel as proxy for haze)
    let patch_size = 15;
    println!("Loading images in parallel...");
    let images = load_images_parallel(&image_paths);
    println!("Successfully loaded {} images", images.len());

    //Generate labels using mean dark channel
    println!("Generating haze labels...");
    let labels: Vec<f64> = images.iter()
        .map(|img| extract_mean_dark_channel(img, patch_size).clamp(0.0, 1.0))
        .collect();

    if images.len() < 2 {
        println!("Error: Need at least 2 images to train");
        return;
    }

    //Load test images from query set
    let query_path = Path::new("dataset/SealID/full images/source_query/");
    let (test_images, test_labels) = if query_path.exists() {
        let query_paths = find_random_x_images_in_directory(query_path, 10);
        let test_imgs = load_images_parallel(&query_paths);
        let test_lbls: Vec<f64> = test_imgs.iter()
            .map(|img| extract_mean_dark_channel(img, patch_size).clamp(0.0, 1.0))
            .collect();
        (Some(test_imgs), Some(test_lbls))
    } else {
        println!("Query path not found, skipping test evaluation");
        (None, None)
    };

    //Run CNN training (self-contained in iteration_2_cnn module)
    let _trained_model = iteration_2_CNN::run_cnn_training::<&str>(
        &images,
        &labels,
        test_images.as_deref(),
        test_labels.as_deref(),
        50,     //epochs
        8,      //batch_size
        0.001,  //learning_rate
        None,   //save_path
    );

    println!("\n=== CNN Training Complete ===");
}

//Iteration 2: Train CNN on full SealID dataset using GPU acceleration, which is MUCH FASTER than CPU, but is still the same pipeline as run_cnn_dataset_training but uses wgpu backend for GPU compute, use this method on machines that have a GPU if possible.
fn run_cnn_dataset_training_gpu() {
    println!("=== AI-Assisted Seal Photograph Image Processing System ===");
    println!("=== Iteration 2: CNN Dataset Training (GPU) ===\n");

    let dataset_path = Path::new("dataset/SealID/full images/source_database");
    if !dataset_path.exists() {
        println!("Error: training dataset not found at {:?}", dataset_path);
        println!("\nPlease download the SealID dataset and extract it to dataset/SealID/ per the instructions in the README");
        return;
    }

    let image_paths = find_images_in_directory(dataset_path);
    if image_paths.is_empty() {
        println!("Error: No images found in {:?}", dataset_path);
        return;
    }
    println!("Found {} images in dataset", image_paths.len());

    let patch_size = 15;
    println!("Loading images in parallel...");
    let images = load_images_parallel(&image_paths);
    println!("Successfully loaded {} images", images.len());

    println!("Generating haze labels...");
    let labels: Vec<f64> = images.iter()
        .map(|img| extract_mean_dark_channel(img, patch_size).clamp(0.0, 1.0))
        .collect();

    if images.len() < 2 {
        println!("Error: Need at least 2 images to train");
        return;
    }

    //Load test images from query set
    let query_path = Path::new("dataset/SealID/full images/source_query");
    let (test_images, test_labels) = if query_path.exists() {
        let query_paths = find_random_x_images_in_directory(query_path, 10);
        let test_imgs = load_images_parallel(&query_paths);
        let test_lbls: Vec<f64> = test_imgs.iter()
            .map(|img| extract_mean_dark_channel(img, patch_size).clamp(0.0, 1.0))
            .collect();
        (Some(test_imgs), Some(test_lbls))
    } else {
        println!("Query path not found, skipping test evaluation");
        (None, None)
    };

    //Run GPU-accelerated CNN training
    let _trained_model = iteration_2_CNN::run_cnn_training_gpu::<&str>(
        &images,
        &labels,
        test_images.as_deref(),
        test_labels.as_deref(),
        50,     //epochs - can do more since GPU is faster
        0.001,  //learning_rate
        None,   //save_path
    );

    println!("\n=== GPU CNN Training Complete ===");
}

/*
Trains CNN on full SealID dataset using GPU acceleration and saves model to disk for later inference, basically run_cnn_dataset_training_gpu() but with persistence so you can train once and reuse the model later without retraining overhead that makes the tool impractical to actually use
Flow is identical to run_cnn_dataset_training_gpu() with a save path passed to the training function that writes out the model weights
Purpose: I felt it was best to have two training functions (one with persistence and one without) for GPU as I already did for CPU training, just to keep things consistent and avoid confusion. Functions need to be consolidated later. 

@param: model_path: path to save the model file (WITHOUT .mpk extension, burn adds it automatically)
*/
fn run_cnn_gpu_training_and_save(model_path: &str) {
    println!("=== AI-Assisted Seal Photograph Image Processing System ===");
    println!("=== Iteration 2: CNN Dataset Training (GPU) with Model Save ===\n");

    let dataset_path = Path::new("dataset/SealID/full images/source_database");
    if !dataset_path.exists() {
        println!("Error: training dataset not found at {:?}", dataset_path);
        println!("\nPlease download the SealID dataset and extract it to dataset/SealID/ per the instructions in the README");
        return;
    }

    let image_paths = find_images_in_directory(dataset_path);
    if image_paths.is_empty() {
        println!("Error: No images found in {:?}", dataset_path);
        return;
    }
    println!("Found {} images in dataset", image_paths.len());

    let patch_size = 15;
    println!("Loading images in parallel...");
    let images = load_images_parallel(&image_paths);
    println!("Successfully loaded {} images", images.len());

    println!("Generating haze labels...");
    let labels: Vec<f64> = images.iter()
        .map(|img| extract_mean_dark_channel(img, patch_size).clamp(0.0, 1.0))
        .collect();

    if images.len() < 2 {
        println!("Error: Need at least 2 images to train");
        return;
    }

    //Load test images from query set
    let query_path = Path::new("dataset/SealID/full images/source_query");
    let (test_images, test_labels) = if query_path.exists() {
        let query_paths = find_random_x_images_in_directory(query_path, 10);
        let test_imgs = load_images_parallel(&query_paths);
        let test_lbls: Vec<f64> = test_imgs.iter()
            .map(|img| extract_mean_dark_channel(img, patch_size).clamp(0.0, 1.0))
            .collect();
        (Some(test_imgs), Some(test_lbls))
    } else {
        println!("Query path not found, skipping test evaluation");
        (None, None)
    };

    //Run GPU-accelerated CNN training with save path
    let _trained_model = iteration_2_CNN::run_cnn_training_gpu(
        &images,
        &labels,
        test_images.as_deref(),
        test_labels.as_deref(),
        50,     //epochs
        0.001,  //learning_rate
        Some(model_path),  //save_path - this is the key difference
    );

    println!("\n=== GPU CNN Training Complete ===");
    println!("Model saved to: {}.mpk", model_path);
}

/*
Loads a saved CNN model and processes an image using GPU acceleration which is the main inference/production use case for the CNN pipeline, loads the model weights from disk and uses it to predict haze level on a new image then dehazes based on the prediction using suggested DCP parameters, mirrors run_process_with_model() for linear regression but uses CNN instead
This is the "train once with --train-cnn-gpu-save, then use --process-cnn-gpu on new images" workflow for real usage

@param: model_path: path to the saved model file (WITHOUT .mpk extension)
@param: input_path: path to the input image to dehaze
@param: output_path: path to save the dehazed output image
*/
fn run_process_with_cnn_gpu(model_path: &str, input_path: &str, output_path: &str) {
    use burn::backend::wgpu::WgpuDevice;

    println!("=== AI-Assisted Seal Photograph Image Processing System ===");
    println!("=== Process Image with Saved CNN Model (GPU) ===\n");

    //Load the saved CNN model on GPU
    println!("Loading CNN model from: {}.mpk", model_path);
    let model = match iteration_2_CNN::load_pretrained_model_gpu(model_path) {
        Ok(m) => {
            println!("CNN model loaded successfully on GPU!");
            m
        }
        Err(e) => {
            println!("Error loading model: {}", e);
            return;
        }
    };

    //Load the input image
    println!("\nLoading input image: {}", input_path);
    let img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => {
            println!("Error: Failed to open image: {}", e);
            return;
        }
    };

    let img_matrix = image_to_array3(&img);
    println!("Image loaded: {}x{}", img.width(), img.height());

    //Predict haze level using CNN on GPU
    let device = WgpuDevice::default();
    let haze_score = iteration_2_CNN::predict_haze_cnn(&model, &img_matrix, &device);
    println!("\nCNN predicted haze score: {:.4}", haze_score);
    println!("  (0.0 = clear, 1.0 = heavy haze)");

    //Choose dehazing parameters based on CNN haze prediction
    let (omega, t0, patch_size, guided_radius, guided_eps) = iteration_2_CNN::suggest_dcp_parameters(haze_score);

    if haze_score > 0.7 {
        println!("\nHigh haze detected - using aggressive dehazing parameters");
    } else if haze_score > 0.4 {
        println!("\nModerate haze detected - using balanced dehazing parameters");
    } else {
        println!("\nLow haze detected - using gentle dehazing parameters");
    }
    println!("  omega={}, t0={}, patch_size={}, guided_radius={}, guided_eps={}",
                omega,    t0,    patch_size,    guided_radius,    guided_eps);

    println!("\nRunning Dark Channel Prior dehazing...");
    let top_percent = 0.001;
    let dehazed = dehaze_with_params(&img_matrix, patch_size, omega, t0, top_percent, guided_radius, guided_eps);

    let output_img = array3_to_image(&dehazed);
    match output_img.save(output_path) {
        Ok(_) => println!("\nDehazed image saved to: {}", output_path),
        Err(e) => println!("\nError saving output: {}", e),
    }

    println!("\n=== Processing Complete ===");
}

//Iteration 2: CNN demo on test images, mirrors run_ml_demo() for quick testing without full dataset as a barebones proof-of-concept on the same test images to show that the CNN can be trained and produces reasonable predictions in a working pipeline
fn run_cnn_demo_standalone() {
    println!("=== AI-Assisted Seal Photograph Image Processing System ===");
    println!("=== Iteration 2: CNN Training Demo ===\n");

    //Same test images as run_ml_demo() for consistency
    let test_images = vec!["fog-137794231410y.jpg", "bansui.jpg"];

    let mut images: Vec<Array3<f32>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();

    //Load available test images and estimate haze labels using DCP features (same approach as linear regression)
    for path in test_images.iter() {
        println!("Attempting to load: {}", path);
        match image::open(path) {
            Ok(img) => {
                //Downsample to 1/8 resolution for fast CNN demo (CPU training is slow) //this was AI recommendation for my laptop
                //Full dataset training uses 1/4, but demo needs to be quick
                let resized_img = img.resize(
                    (img.width() / 8).max(32),  //min 32px to avoid too-small images
                    (img.height() / 8).max(32),
                    image::imageops::FilterType::Triangle  //faster filter for demo
                );
                let img_matrix = image_to_array3(&resized_img);

                //Estimate haze level using mean dark channel as proxy (smaller patch for small images)
                let mean = extract_mean_dark_channel(&img_matrix, 7); //AI recommended smaller patch due to hardware limits
                let estimated_haze = mean.clamp(0.0, 1.0);

                println!("  Loaded successfully ({}x{} -> {}x{}), estimated haze: {:.3}",
                         img.width(), img.height(), resized_img.width(), resized_img.height(), estimated_haze);
                images.push(img_matrix);
                labels.push(estimated_haze);
            }
            Err(e) => {
                println!("  Failed to load {}: {}", path, e);
            }
        }
    }

    if images.len() < 2 {
        println!("\nWarning: Need at least 2 images for CNN training demonstration.");
        println!("Falling back to IP engine tests...\n");
        run_all_ip_tests();
        return;
    }

    //Run CNN demo training (few epochs, just need to show it works)
    let trained_model = iteration_2_CNN::run_cnn_demo(&images, &labels);

    //Evaluate on training set to show predictions
    println!("\n=== Evaluating Trained CNN ===");
    let device = burn_ndarray::NdArrayDevice::Cpu;

    for (i, (img, &label)) in images.iter().zip(labels.iter()).enumerate() {
        let predicted = iteration_2_CNN::cnn_detection::predict_haze_cnn(&trained_model, img, &device);
        let (omega, t0, patch, radius, eps) = iteration_2_CNN::cnn_detection::suggest_dcp_parameters(predicted);

        println!("Image {}: predicted={:.3}, actual={:.3}", i + 1, predicted, label);
        println!("  Suggested DCP params: omega={}, t0={}, patch={}, radius={}, eps={}",
                 omega, t0, patch, radius, eps);
    }

    //Run dehazing on the first (foggy) image as demonstration with CNN-suggested parameters //AI-generated, this was the only major change from run_ml_demo()
    println!("\n=== Running Dehazing Pipeline with CNN-Suggested Parameters ===");
    if !images.is_empty() {
        let predicted = iteration_2_CNN::cnn_detection::predict_haze_cnn(&trained_model, &images[0], &device);
        let (omega, t0, patch_size, guided_radius, guided_eps) = iteration_2_CNN::cnn_detection::suggest_dcp_parameters(predicted);

        println!("CNN predicted haze level: {:.3}", predicted);
        println!("Using suggested parameters: omega={}, t0={}, patch={}, radius={}, eps={}",
                 omega, t0, patch_size, guided_radius, guided_eps);

        let dehazed = dehaze_with_params(&images[0], patch_size, omega, t0, 0.001, guided_radius, guided_eps);
        let output_img = array3_to_image(&dehazed);
        output_img
            .save("output_dehazing_dcp_cnn_demo.jpg")
            .expect("Failed to save");
        println!("Saved dehazed result to output_dehazing_dcp_cnn_demo.jpg");
    }

    println!("\n=== CNN Demo Complete ===");
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

/*
Train on demo images and save model to a JSON file for later use, which is key for model persistence so the user can train once and then use the model later without retraining which has a large overhead and prevents realistic use.
Basically just run_ml_demo() but with a save call at the end to the functions in training.rs, mirrors the same training flow but outputs to disk instead of just printing weights

@param: model_path: path to save the model JSON file (e.g. "haze_model.json")
*/
fn run_train_and_save(model_path: &str) {
    println!("=== AI-Assisted Seal Photograph Image Processing System ===");
    println!("=== Train and Save Model ===\n");

    //Same test images as run_ml_demo()
    let test_images = vec!["fog-137794231410y.jpg", "bansui.jpg"];

    let mut images: Vec<Array3<f32>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();

    for path in test_images.iter() {
        println!("Attempting to load: {}", path);
        match image::open(path) {
            Ok(img) => {
                let img_matrix = image_to_array3(&img);
                let mean = extract_mean_dark_channel(&img_matrix, 15);
                let estimated_haze = mean.clamp(0.0, 1.0);
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
        println!("\nError: Need at least 2 images for training.");
        return;
    }

    println!("\n=== Training Linear Regression Haze Regressor ===");
    let patch_size = 15;
    let learning_rate = 0.1;
    let epochs = 100;

    let regressor = train_haze_regressor(&images, &labels, patch_size, learning_rate, epochs);

    //Save the model
    match regressor.save(model_path) {
        Ok(_) => println!("\nModel saved successfully to: {}", model_path),
        Err(e) => println!("\nError saving model: {}", e),
    }

    println!("\n=== Training Complete ===");
    println!("Model weights: {:?}", regressor.model.weights);
    println!("Model bias: {:.4}", regressor.model.bias);
}

/*
Trains regressor on full SealID dataset and save model to a JSON file, same as run_full_dataset_training() but with persistence to avert overhead from retraining every single time, also reduces overhead via precomputed features during loading to avoid recomputing DCP as in run_full_dataset_training()
Flow is basically identical to run_full_dataset_training() with a save call tacked on at the end to training.rs's save function

@param: model_path: path to save the model JSON file
*/
fn run_full_training_and_save(model_path: &str) {
    println!("=== AI-Assisted Seal Photograph Image Processing System ===");
    println!("=== Full Dataset Training with Model Save ===\n");

    let dataset_path = Path::new("dataset/SealID/full images/source_database");
    if !dataset_path.exists() {
        println!("Error: training dataset not found at {:?}", dataset_path);
        println!("\nPlease download the SealID dataset and extract it to dataset/SealID/ per the instructions in the README");
        return;
    }

    let image_paths = find_images_in_directory(dataset_path);
    if image_paths.is_empty() {
        println!("Error: No images found in {:?}", dataset_path);
        return;
    }
    println!("Found {} images in dataset", image_paths.len());

    let patch_size = 15;
    println!("Loading images and extracting features in parallel...");
    let (_images, labels, features) = load_images_with_features(&image_paths, patch_size);
    println!("Successfully loaded {} images with precomputed features", labels.len());

    if labels.len() < 2 {
        println!("Error: Need at least 2 images to train");
        return;
    }

    println!("\n=== Training on {} images ===", labels.len());
    let learning_rate = 0.01;
    let epochs = 200;

    let regressor = train_haze_regressor_precomputed(&features, &labels, learning_rate, epochs);

    //Save the model
    match regressor.save(model_path) {
        Ok(_) => println!("\nModel saved successfully to: {}", model_path),
        Err(e) => println!("\nError saving model: {}", e),
    }

    println!("\n=== Training Complete ===");
    println!("Model weights: {:?}", regressor.model.weights);
    println!("Model bias: {:.4}", regressor.model.bias);
}

/*
Loads a saved model and processes an image, which is the main inference function that makes persistence  useful, loads the JSON model file and uses it to predict haze level on a new image then automatically picks DCP parameters based on the prediction and dehazes the image
The parameter selection is based on haze score thresholds: high haze (>0.7) gets aggressive dehazing with lower omega, moderate haze (0.4-0.7) gets balanced params, low haze (<0.4) gets gentle params to avoid overcorrecting, would need to add more variables to the regressor to move beyond a simple heuristic
This is basically the "production" use case of "train once with --train-save or --train-full-save, then use --process on new images without retraining"

@param: model_path: path to the saved model JSON file from training
@param: input_path: path to the input image to dehaze
@param: output_path: path to save the dehazed output image
*/
fn run_process_with_model(model_path: &str, input_path: &str, output_path: &str) {
    println!("=== AI-Assisted Seal Photograph Image Processing System ===");
    println!("=== Process Image with Saved Model ===\n");

    //Load the saved model
    println!("Loading model from: {}", model_path);
    let regressor = match training::HazeRegressor::load(model_path) {
        Ok(r) => {
            println!("Model loaded successfully!");
            println!("  Weights: {:?}", r.model.weights);
            println!("  Bias: {:.4}", r.model.bias);
            r
        }
        Err(e) => {
            println!("Error loading model: {}", e);
            return;
        }
    };

    //Load the input image
    println!("\nLoading input image: {}", input_path);
    let img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => {
            println!("Error: Failed to open image: {}", e);
            return;
        }
    };

    let img_matrix = image_to_array3(&img);
    println!("Image loaded: {}x{}", img.width(), img.height());

    //Predict haze level of input image
    let patch_size = 15;
    let haze_score = training::predict_haze_score(&regressor, &img_matrix, patch_size);
    println!("\nPredicted haze score: {:.4}", haze_score);
    println!("  (0.0 = clear, 1.0 = heavy haze)");

    //Choose dehazing parameters based on haze level, higher haze begets more aggressive dehazing (lower omega, higher t0)
    let (omega, t0, guided_radius, guided_eps) = if haze_score > 0.7 {
        println!("\nHigh haze detected - using aggressive dehazing parameters");
        (0.6_f32, 0.3_f32, 30_usize, 0.0001_f32)
    } else if haze_score > 0.4 {
        println!("\nModerate haze detected - using balanced dehazing parameters");
        (0.75_f32, 0.2_f32, 45_usize, 0.0001_f32)
    } else {
        println!("\nLow haze detected - using gentle dehazing parameters");
        (0.85_f32, 0.15_f32, 60_usize, 0.0001_f32)
    };

    println!("  omega={}, t0={}, patch_size={}, guided_radius={}, guided_eps={}",
                omega,    t0,    patch_size,    guided_radius,    guided_eps); //Whitespace OCD

    println!("\nRunning Dark Channel Prior dehazing...");
    let top_percent = 0.001;
    let dehazed = dehaze_with_params(&img_matrix, patch_size, omega, t0, top_percent, guided_radius, guided_eps);

    let output_img = array3_to_image(&dehazed);
    match output_img.save(output_path) {
        Ok(_) => println!("\nDehazed image saved to: {}", output_path),
        Err(e) => println!("\nError saving output: {}", e),
    }

    println!("\n=== Processing Complete ===");
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
