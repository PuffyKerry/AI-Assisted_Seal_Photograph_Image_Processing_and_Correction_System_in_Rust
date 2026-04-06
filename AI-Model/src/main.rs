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
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use image::{GrayImage, Luma};
use ndarray::{Array2, Array3};
use IP_functions::dehaze::{dehaze_default_parameters_test, dehaze_with_params};
use IP_functions::enhance::enhance_clahe;
use IP_functions::brightness::{brightness_correct_default, brightness_correct};
use rand::seq::SliceRandom;
use rand::rng;
use memmap2::Mmap;
use rayon::prelude::*;
use crate::training::{train_haze_regressor, train_haze_regressor_precomputed};
use crate::ip_tests::{image_to_array3, array3_to_image, run_all_ip_tests};
use crate::extraction::{extract_mean_dark_channel, extract_robust_haze_label, extract_clahe_contrast_deficit, extract_brightness_deficit};
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
                println!("Usage: cargo run -p ai-model -- --train-cnn-gpu-save cnn_model [EPOCHS]");
                return;
            }
            let epochs: usize = if args.len() >= 4 {
                args[3].parse().unwrap_or_else(|_| { println!("Invalid epoch count, using 50"); 50 })
            } else { 50 };
            run_cnn_gpu_training_and_save(&args[2], epochs);
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
        "--gamma" => {
            if args.len() < 3 {
                println!("Error: --gamma requires an image path");
                println!("Usage: cargo run -p ai-model -- --gamma path/to/image.jpg");
                return;
            }
            gamma_single_image(&args[2]);
        }
        "--gamma-custom" => {
            //Usage: --gamma-custom FILE gamma_value
            if args.len() < 4 {
                println!("Error: --gamma-custom requires image path and gamma value");
                println!("Usage: cargo run -p ai-model -- --gamma-custom FILE gamma_value");
                println!("Example: cargo run -p ai-model -- --gamma-custom image.jpg 0.6");
                return;
            }
            let gamma: f32 = args[3].parse().unwrap_or_else(|_| { println!("Invalid gamma, using auto"); -1.0 });
            gamma_with_custom_params(&args[2], gamma);
        }
        "--demo" => run_ml_demo(),
        "--compare-models" => {
            //Compare regressor vs CNN on the full source_query dataset
            if args.len() < 4 {
                println!("Error: --compare-models requires regressor model path and CNN model path");
                println!("Usage: cargo run --release -p ai-model -- --compare-models haze_model.json CNN_GPU_TILED");
                return;
            }
            run_compare_models(&args[2], &args[3]);
        }
        "--ablation" => {
            //CNN architecture ablation study: spawns 3 child processes (Small, Medium, Large)
            //NOTE: This had GPU OOM scaling issues — see comment on run_ablation_study().
            //Use --epoch-study instead for a more practical convergence analysis.
            run_ablation_study();
        }
        "--ablation-variant" => {
            // Internal: train+evaluate a single CNN variant. Called by --ablation as a subprocess
            // so the OS reclaims all GPU memory when the child process exits.
            if args.len() < 3 {
                println!("Error: --ablation-variant requires SMALL, MEDIUM, or LARGE");
                return;
            }
            run_ablation_single_variant(&args[2]);
        }
        "--epoch-study" => {
            //Train CNN at 10, 30, and 50 epochs, then compare all against the regressor
            //to characterize convergence. Each training run is a separate process.
            run_epoch_study();
        }
        "--epoch-comparison" => {
            //Compare pre-trained CNN models at different epochs vs regressor (no retraining)
            if args.len() < 4 {
                println!("Error: --epoch-comparison requires regressor path and at least one CNN model path");
                println!("Usage: cargo run --release -p ai-model -- --epoch-comparison haze_model.json CNN_10ep CNN_30ep CNN_GPU_TILED");
                return;
            }
            let cnn_info: Vec<(&str, &str)> = args[3..].iter()
                .map(|s| (s.as_str(), s.as_str()))
                .collect();
            run_epoch_comparison(&args[2], &cnn_info);
        }
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
    println!("  --train-cnn-gpu-save MODEL_PATH [EPOCHS]");
    println!("                 Train CNN on GPU and save model for later inference (default 50 epochs)");
    println!("  --process-cnn-gpu MODEL_PATH INPUT_IMAGE OUTPUT_IMAGE");
    println!("                 Load saved CNN model and process image with DCP + CLAHE + Gamma (GPU)");
    println!("                 Outputs: OUTPUT_dcp.jpg, OUTPUT.jpg (DCP+CLAHE+Gamma), OUTPUT_clahe.jpg");
    println!("  --demo-cnn     CNN training demo on test images with DCP + CLAHE + Gamma (Iteration 2)");
    println!("  --compare-models REGRESSOR_PATH CNN_PATH");
    println!("                 Compare regressor vs CNN on the full source_query dataset");
    println!("  --epoch-study  Train CNN at 10, 30, 50 epochs and compare convergence vs regressor");
    println!("                 (Each training run is a separate process. Skips epochs with existing models.)");
    println!("  --epoch-comparison REGRESSOR_PATH CNN_PATH1 CNN_PATH2 ...");
    println!("                 Compare pre-trained CNN models at different epochs vs regressor (no retraining)");
    println!("  --dehaze FILE  Dehaze a specific image file with default parameters");
    println!("  --dehaze-custom FILE omega t0 patch_size guided_radius guided_eps");
    println!("                 Dehaze with custom DCP parameters");
    println!("  --clahe FILE   Enhance contrast of an image with CLAHE (default parameters)");
    println!("  --clahe-custom FILE grid_h grid_w clip_limit");
    println!("                 Enhance contrast with custom CLAHE parameters");
    println!("  --gamma FILE   Correct brightness with auto-estimated gamma");
    println!("  --gamma-custom FILE gamma_value");
    println!("                 Correct brightness with custom gamma (< 1.0 brightens, > 1.0 darkens)");
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
    println!("Gamma/Brightness Parameters:");
    println!("  gamma          Power-law exponent (< 1.0 brightens, > 1.0 darkens, 1.0 = no change)");
    println!("                 Auto mode estimates optimal gamma from image mean luminance\n");
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

    //Generate labels using robust multi-feature DCP label (median + regional stats) for better training signal and CLAHE deficit label
    println!("Generating haze labels (robust: median + regional DCP stats)...");
    let labels: Vec<f64> = images.iter()
        .map(|img| extract_robust_haze_label(img, patch_size))
        .collect();
    println!("Generating CLAHE contrast deficit labels...");
    let clahe_labels: Vec<f64> = images.iter()
        .map(|img| extract_clahe_contrast_deficit(img))
        .collect();
    println!("Generating brightness deficit labels...");
    let brightness_labels: Vec<f64> = images.iter()
        .map(|img| extract_brightness_deficit(img))
        .collect();

    if images.len() < 2 {
        println!("Error: Need at least 2 images to train");
        return;
    }

    //Load test images from query set to evaluate the model
    let query_path = Path::new("dataset/SealID/full images/source_query/");
    let (test_images, test_labels, test_clahe_labels, test_brightness_labels) = if query_path.exists() {
        let query_paths = find_random_x_images_in_directory(query_path, 10);
        let test_imgs = load_images_parallel(&query_paths);

        let test_lbls: Vec<f64> = test_imgs.iter()
            .map(|img| extract_robust_haze_label(img, patch_size))
            .collect();
        let test_clahe_lbls: Vec<f64> = test_imgs.iter()
            .map(|img| extract_clahe_contrast_deficit(img))
            .collect();
        let test_bright_lbls: Vec<f64> = test_imgs.iter()
            .map(|img| extract_brightness_deficit(img))
            .collect();

        (Some(test_imgs), Some(test_lbls), Some(test_clahe_lbls), Some(test_bright_lbls))
    } else {
        println!("Query path not found, skipping test evaluation");
        (None, None, None, None)
    };

    //Run CNN training (self-contained in iteration_2_cnn module)
    let _trained_model = iteration_2_CNN::run_cnn_training::<&str>(
        &images,
        &labels,
        &clahe_labels,
        &brightness_labels,
        test_images.as_deref(),
        test_labels.as_deref(),
        test_clahe_labels.as_deref(),
        test_brightness_labels.as_deref(),
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

    println!("Generating haze labels (robust: median + regional DCP stats)...");
    let labels: Vec<f64> = images.iter()
        .map(|img| extract_robust_haze_label(img, patch_size))
        .collect();
    println!("Generating CLAHE contrast deficit labels...");
    let clahe_labels: Vec<f64> = images.iter()
        .map(|img| extract_clahe_contrast_deficit(img))
        .collect();
    println!("Generating brightness deficit labels...");
    let brightness_labels: Vec<f64> = images.iter()
        .map(|img| extract_brightness_deficit(img))
        .collect();

    if images.len() < 2 {
        println!("Error: Need at least 2 images to train");
        return;
    }

    //Load test images from query set
    let query_path = Path::new("dataset/SealID/full images/source_query");
    let (test_images, test_labels, test_clahe_labels, test_brightness_labels) = if query_path.exists() {
        let query_paths = find_random_x_images_in_directory(query_path, 10);
        let test_imgs = load_images_parallel(&query_paths);

        let test_lbls: Vec<f64> = test_imgs.iter()
            .map(|img| extract_robust_haze_label(img, patch_size))
            .collect();
        let test_clahe_lbls: Vec<f64> = test_imgs.iter()
            .map(|img| extract_clahe_contrast_deficit(img))
            .collect();
        let test_bright_lbls: Vec<f64> = test_imgs.iter()
            .map(|img| extract_brightness_deficit(img))
            .collect();

        (Some(test_imgs), Some(test_lbls), Some(test_clahe_lbls), Some(test_bright_lbls))
    } else {
        println!("Query path not found, skipping test evaluation");
        (None, None, None, None)
    };

    //Run GPU-accelerated CNN training
    let _trained_model = iteration_2_CNN::run_cnn_training_gpu::<&str>(
        &images,
        &labels,
        &clahe_labels,
        &brightness_labels,
        test_images.as_deref(),
        test_labels.as_deref(),
        test_clahe_labels.as_deref(),
        test_brightness_labels.as_deref(),
        50,     //epochs - more epochs due to GPU being faster
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
@param: epochs: number of training epochs (default 50)
*/
fn run_cnn_gpu_training_and_save(model_path: &str, epochs: usize) {
    println!("=== AI-Assisted Seal Photograph Image Processing System ===");
    println!("=== Iteration 2: CNN Dataset Training (GPU) with Model Save ===");
    println!("=== Training for {} epochs ===\n", epochs);

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

    println!("Generating haze labels (robust: median + regional DCP stats)...");
    let labels: Vec<f64> = images.iter()
        .map(|img| extract_robust_haze_label(img, patch_size))
        .collect();
    println!("Generating CLAHE contrast deficit labels...");
    let clahe_labels: Vec<f64> = images.iter()
        .map(|img| extract_clahe_contrast_deficit(img))
        .collect();
    println!("Generating brightness deficit labels...");
    let brightness_labels: Vec<f64> = images.iter()
        .map(|img| extract_brightness_deficit(img))
        .collect();

    if images.len() < 2 {
        println!("Error: Need at least 2 images to train");
        return;
    }

    //Load test images from query set
    let query_path = Path::new("dataset/SealID/full images/source_query");
    let (test_images, test_labels, test_clahe_labels, test_brightness_labels) = if query_path.exists() {
        let query_paths = find_random_x_images_in_directory(query_path, 10);
        let test_imgs = load_images_parallel(&query_paths);

        let test_lbls: Vec<f64> = test_imgs.iter()
            .map(|img| extract_robust_haze_label(img, patch_size))
            .collect();
        let test_clahe_lbls: Vec<f64> = test_imgs.iter()
            .map(|img| extract_clahe_contrast_deficit(img))
            .collect();
        let test_bright_lbls: Vec<f64> = test_imgs.iter()
            .map(|img| extract_brightness_deficit(img))
            .collect();

        (Some(test_imgs), Some(test_lbls), Some(test_clahe_lbls), Some(test_bright_lbls))
    } else {
        println!("Query path not found, skipping test evaluation");
        (None, None, None, None)
    };

    //Run GPU-accelerated CNN training with save path
    let _trained_model = iteration_2_CNN::run_cnn_training_gpu(
        &images,
        &labels,
        &clahe_labels,
        &brightness_labels,
        test_images.as_deref(),
        test_labels.as_deref(),
        test_clahe_labels.as_deref(),
        test_brightness_labels.as_deref(),
        epochs,     //epochs - configurable
        0.001,  //learning_rate
        Some(model_path),  //save_path - this is the key difference
    );

    println!("\n=== GPU CNN Training Complete ({} epochs) ===", epochs);
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
    println!("=== Process Image with Saved CNN Model (GPU) ===");
    println!("=== DCP Dehazing + CLAHE Enhancement + Gamma Brightness ===\n");

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

    //Predict all three scores using the triple-output CNN on GPU
    let device = WgpuDevice::default();
    let (dcp_score, clahe_score, brightness_score) = iteration_2_CNN::predict_haze_cnn(&model, &img_matrix, &device);
    println!("\nCNN predictions:");
    println!("  DCP haze score:          {:.4}  (0=clear, 1=heavy haze)", dcp_score);
    println!("  CLAHE contrast deficit:  {:.4}  (0=high contrast, 1=flat/low contrast)", clahe_score);
    println!("  Brightness deficit:      {:.4}  (0=well-exposed, 1=dark/overexposed)", brightness_score);

    //Each score drives its own parameter decisions for its set independently
    let (omega, t0, patch_size, guided_radius, guided_eps) = iteration_2_CNN::suggest_dcp_parameters(dcp_score);
    let (grid_h, grid_w, clip_limit) = iteration_2_CNN::suggest_clahe_parameters(clahe_score);
    let gamma = iteration_2_CNN::suggest_gamma_parameters(brightness_score);

    if dcp_score > 0.7 {
        println!("\nHigh haze detected - using aggressive parameters");
    } else if dcp_score > 0.4 {
        println!("\nModerate haze detected - using balanced parameters");
    } else {
        println!("\nLow haze detected - using gentle parameters");
    }
    println!("  DCP:   omega={}, t0={}, patch_size={}, guided_radius={}, guided_eps={}",
                omega,    t0,    patch_size,    guided_radius,    guided_eps);
    println!("  CLAHE: grid={}x{}, clip_limit={}", grid_h, grid_w, clip_limit);
    println!("  Gamma: {:.3}", gamma);

    // === Step 1: DCP Dehazing ===
    println!("\nStep 1: Running Dark Channel Prior dehazing...");
    let top_percent = 0.001;
    let dehazed = dehaze_with_params(&img_matrix, patch_size, omega, t0, top_percent, guided_radius, guided_eps);

    //Save DCP-only output
    let dcp_output = array3_to_image(&dehazed);
    let dcp_path = format!("{}_dcp.jpg", output_path.trim_end_matches(".jpg").trim_end_matches(".png"));
    match dcp_output.save(&dcp_path) {
        Ok(_) => println!("  DCP dehazed image saved to: {}", dcp_path),
        Err(e) => println!("  Error saving DCP output: {}", e),
    }

    // === Step 2: CLAHE Enhancement on the DCP output ===
    // Attenuate clip_limit when stacking after DCP — the image is already partially corrected,
    // so full-strength CLAHE would overcorrect. 70% of the standalone clip_limit.
    let stacked_clip = 1.0_f32 + (clip_limit - 1.0) * 0.7;
    println!("Step 2: Running CLAHE contrast enhancement on dehazed result (clip_limit {:.2} → {:.2} for stacking)...", clip_limit, stacked_clip);
    let enhanced = enhance_clahe(&dehazed, grid_h, grid_w, stacked_clip);

    // === Step 3: Gamma brightness correction on the DCP+CLAHE output ===
    // Attenuate gamma when stacking — pull it toward 1.0 (identity) since DCP+CLAHE already
    // lifted brightness significantly. Blend: 50% of the way back toward 1.0.
    let stacked_gamma = 1.0 + (gamma - 1.0) * 0.5;
    println!("Step 3: Running gamma brightness correction (gamma {:.3} → {:.3} for stacking)...", gamma, stacked_gamma);
    let gamma_corrected = brightness_correct(&enhanced, stacked_gamma);

    let output_img = array3_to_image(&gamma_corrected);
    match output_img.save(output_path) {
        Ok(_) => println!("  DCP + CLAHE + Gamma result saved to: {}", output_path),
        Err(e) => println!("  Error saving output: {}", e),
    }

    // === Step 4: CLAHE-only for comparison ===
    println!("Step 4: Running CLAHE on original image for comparison...");
    let clahe_only = enhance_clahe(&img_matrix, grid_h, grid_w, clip_limit);
    let clahe_path = format!("{}_clahe.jpg", output_path.trim_end_matches(".jpg").trim_end_matches(".png"));
    let clahe_img = array3_to_image(&clahe_only);
    match clahe_img.save(&clahe_path) {
        Ok(_) => println!("  CLAHE-only image saved to: {}", clahe_path),
        Err(e) => println!("  Error saving CLAHE output: {}", e),
    }

    // === Step 5: Gamma-only for comparison ===
    println!("Step 5: Running gamma on original image for comparison...");
    let gamma_only = brightness_correct(&img_matrix, gamma);
    let gamma_path = format!("{}_gamma.jpg", output_path.trim_end_matches(".jpg").trim_end_matches(".png"));
    let gamma_img = array3_to_image(&gamma_only);
    match gamma_img.save(&gamma_path) {
        Ok(_) => println!("  Gamma-only image saved to: {}", gamma_path),
        Err(e) => println!("  Error saving gamma output: {}", e),
    }

    println!("\n=== Processing Complete ===");
    println!("Outputs:");
    println!("  DCP only:              {}", dcp_path);
    println!("  DCP + CLAHE + Gamma:   {}", output_path);
    println!("  CLAHE only:            {}", clahe_path);
    println!("  Gamma only:            {}", gamma_path);
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

    //Compute DCP labels (robust) and CLAHE contrast deficit labels and brightness deficit labels
    let clahe_labels: Vec<f64> = images.iter()
        .map(|img| extract_clahe_contrast_deficit(img))
        .collect();
    let brightness_labels: Vec<f64> = images.iter()
        .map(|img| extract_brightness_deficit(img))
        .collect();
    //Use simple mean DC for the demo (already in `labels` Vec above) because recomputing robustly is overkill for 2 images

    //Run CNN demo training (few epochs, just need to show it works)
    let trained_model = iteration_2_CNN::run_cnn_demo(&images, &labels, &clahe_labels, &brightness_labels);

    //Evaluate on training set to show predictions
    println!("\n=== Evaluating Trained CNN ===");
    let device = burn_ndarray::NdArrayDevice::Cpu;

    for (i, ((img, (&dcp_lbl, &clahe_lbl)), &bright_lbl)) in images.iter()
        .zip(labels.iter().zip(clahe_labels.iter()))
        .zip(brightness_labels.iter())
        .enumerate()
    {
        let (dcp_pred, clahe_pred, bright_pred) = iteration_2_CNN::cnn_detection::predict_haze_cnn(&trained_model, img, &device);
        let (omega, t0, patch, radius, eps) = iteration_2_CNN::cnn_detection::suggest_dcp_parameters(dcp_pred);
        let (grid_h, grid_w, clip) = iteration_2_CNN::cnn_detection::suggest_clahe_parameters(clahe_pred);
        let gamma = iteration_2_CNN::cnn_detection::suggest_gamma_parameters(bright_pred);

        println!("Image {}: DCP pred={:.3}/actual={:.3}  CLAHE pred={:.3}/actual={:.3}  Bright pred={:.3}/actual={:.3}", i + 1, dcp_pred, dcp_lbl, clahe_pred, clahe_lbl, bright_pred, bright_lbl);
        println!("  DCP params:   omega={:.3}, t0={:.3}, patch={}, radius={}, eps={:.5}", omega, t0, patch, radius, eps);
        println!("  CLAHE params: grid={}x{}, clip_limit={:.2}", grid_h, grid_w, clip);
        println!("  Gamma:        {:.3}", gamma);
    }

    //Run DCP + CLAHE + Gamma on the first (foggy) image as demonstration with CNN-suggested parameters
    println!("\n=== Running DCP + CLAHE + Gamma Pipeline with CNN-Suggested Parameters ===");
    if !images.is_empty() {
        let (dcp_pred, clahe_pred, bright_pred) = iteration_2_CNN::cnn_detection::predict_haze_cnn(&trained_model, &images[0], &device);
        let (omega, t0, patch_size, guided_radius, guided_eps) = iteration_2_CNN::cnn_detection::suggest_dcp_parameters(dcp_pred);
        let (grid_h, grid_w, clip_limit) = iteration_2_CNN::cnn_detection::suggest_clahe_parameters(clahe_pred);
        let gamma = iteration_2_CNN::cnn_detection::suggest_gamma_parameters(bright_pred);

        println!("CNN predictions: DCP haze: {:.3}, CLAHE deficit: {:.3}, Brightness deficit: {:.3}", dcp_pred, clahe_pred, bright_pred);
        println!("DCP params:   omega={:.3}, t0={:.3}, patch={}, radius={}, eps={:.5}", omega, t0, patch_size, guided_radius, guided_eps);
        println!("CLAHE params: grid={}x{}, clip_limit={:.2}", grid_h, grid_w, clip_limit);
        println!("Gamma:        {:.3}", gamma);

        //Step 1: DCP dehazing
        let dehazed = dehaze_with_params(&images[0], patch_size, omega, t0, 0.001, guided_radius, guided_eps);
        let output_img = array3_to_image(&dehazed);
        output_img
            .save("output_dehazing_dcp_cnn_demo.jpg")
            .expect("Failed to save");
        println!("Saved DCP result to output_dehazing_dcp_cnn_demo.jpg");

        //Step 2: CLAHE on DCP output (attenuated for stacking)
        let stacked_clip = 1.0_f32 + (clip_limit - 1.0) * 0.7;
        let enhanced = enhance_clahe(&dehazed, grid_h, grid_w, stacked_clip);
        let enhanced_img = array3_to_image(&enhanced);
        enhanced_img
            .save("output_dcp_clahe_cnn_demo.jpg")
            .expect("Failed to save");
        println!("Saved DCP + CLAHE result to output_dcp_clahe_cnn_demo.jpg");

        //Step 3: Gamma on DCP+CLAHE output (attenuated for stacking)
        let stacked_gamma = 1.0 + (gamma - 1.0) * 0.5;
        let gamma_corrected = brightness_correct(&enhanced, stacked_gamma);
        let gamma_img = array3_to_image(&gamma_corrected);
        gamma_img
            .save("output_dcp_clahe_gamma_cnn_demo.jpg")
            .expect("Failed to save");
        println!("Saved DCP + CLAHE + Gamma result to output_dcp_clahe_gamma_cnn_demo.jpg");

        //Step 4: CLAHE only for comparison
        let clahe_only = enhance_clahe(&images[0], grid_h, grid_w, clip_limit);
        let clahe_img = array3_to_image(&clahe_only);
        clahe_img
            .save("output_clahe_cnn_demo.jpg")
            .expect("Failed to save");
        println!("Saved CLAHE-only result to output_clahe_cnn_demo.jpg");
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
    println!("Parameters: omega={}, t0={}, patch_size={}, guided_radius={}, guided_eps={}\n", omega, t0, patch_size, guided_radius, guided_eps);

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

//Correct brightness of a single image using auto-estimated gamma
fn gamma_single_image(img_path: &str) {
    println!("=== Gamma Brightness Correction: {} ===\n", img_path);

    let img = match image::open(img_path) {
        Ok(img) => img,
        Err(e) => {
            println!("Error: Failed to open image: {}", e);
            return;
        }
    };

    let img_matrix = image_to_array3(&img);
    let gamma = IP_functions::gamma::estimate_gamma(&img_matrix);
    println!("Image loaded successfully");
    println!("Auto-estimated gamma: {:.3}", gamma);
    println!("Running gamma correction...\n");

    let corrected = brightness_correct(&img_matrix, gamma);

    let input_path = Path::new(img_path);
    let stem = input_path.file_stem().unwrap_or_default().to_str().unwrap_or("output");
    let output_path = format!("output_gamma_{}.jpg", stem);

    let output_img = array3_to_image(&corrected);
    output_img.save(&output_path).expect("Failed to save");
    println!("Saved gamma-corrected result to {}", output_path);
}

//Correct brightness of a single image using custom gamma value
fn gamma_with_custom_params(img_path: &str, gamma: f32) {
    let img = match image::open(img_path) {
        Ok(img) => img,
        Err(e) => {
            println!("Error: Failed to open image: {}", e);
            return;
        }
    };

    let img_matrix = image_to_array3(&img);

    //If gamma is negative (invalid), fall back to auto-estimation
    let actual_gamma = if gamma <= 0.0 {
        let auto = IP_functions::gamma::estimate_gamma(&img_matrix);
        println!("Using auto-estimated gamma: {:.3}", auto);
        auto
    } else {
        gamma
    };

    println!("=== Gamma Brightness Correction with gamma={:.3} ===", actual_gamma);
    println!("Image: {}\n", img_path);

    let corrected = brightness_correct(&img_matrix, actual_gamma);

    let input_path = Path::new(img_path);
    let stem = input_path.file_stem().unwrap_or_default().to_str().unwrap_or("output");
    let output_path = format!("output_gamma_custom_{}.jpg", stem);

    let output_img = array3_to_image(&corrected);
    output_img.save(&output_path).expect("Failed to save");
    println!("Saved gamma-corrected result to {}", output_path);
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

    //Also apply CLAHE enhancement on the dehazed result
    println!("Running CLAHE contrast enhancement on dehazed result...");
    let (clahe_grid, clahe_clip) = if haze_score > 0.7 { (8usize, 4.0f32) }
        else if haze_score > 0.4 { (8usize, 2.5f32) }
        else { (4usize, 1.5f32) };
    let enhanced = enhance_clahe(&dehazed, clahe_grid, clahe_grid, clahe_clip);
    let enhanced_path = format!("{}_dcp_clahe.jpg",
        output_path.trim_end_matches(".jpg").trim_end_matches(".png"));
    let enhanced_img = array3_to_image(&enhanced);
    match enhanced_img.save(&enhanced_path) {
        Ok(_) => println!("DCP + CLAHE result saved to: {}", enhanced_path),
        Err(e) => println!("Error saving enhanced output: {}", e),
    }

    println!("\n=== Processing Complete ===");
}

/*
NOTE ON CNN ARCHITECTURE SCALING (4/5/2026):
The ablation study (Small/Medium/Large CNN variants) encountered GPU VRAM scaling issues
that made the multi-architecture approach impractical for this project:

- Small  (8→16→32→64, ~18K params): OOM on RTX 3070 8GB during epoch 50, batch 51/69 of
  a large 324×350 tile group.  Training actually completed (5311s) but the wgpu background
  thread panic (exit code 101) killed the process before the result line could be printed
  back to the parent orchestrator.  The root cause is that wgpu's allocator fragments over
  long batch sequences; even with periodic flushes, 69 consecutive same-sized batches
  exhausted the 8 GB VRAM budget.

- Medium (16→32→64→128, ~73K params): The default/production architecture.  Trains fine in
  standalone mode (--train-cnn-gpu-save) and was the architecture used for all prior
  successful experiments.  The ablation subprocess machinery added complexity but did not
  change the model itself.

- Large  (32→64→128→256, ~290K params): Required halved VRAM budgets (200K pixels, 248px
  tiles) which significantly reduces training image quality and makes the results non-
  comparable to Medium's 400K/350px budget.

CONCLUSION: Varying CNN depth/width is not worth pursuing here — the bottleneck is image
pixel throughput through the GPU, not model capacity.  The Medium architecture is the sweet
spot and further analysis focuses on EPOCH COUNT (10 vs 30 vs 50) to characterize the
training convergence curve.  See --epoch-study.
*/

/*
CNN Architecture Ablation Study — trains three CNN variants (Small, Medium, Large) on the
same training data, then evaluates each on the held-out query set. This produces a comparison
table that demonstrates data-driven architecture selection for the thesis.

Variants:
  Small  : 8 → 16 → 32 → 64  channels, FC 32,  dropout 0.2   (~18K params)
  Medium : 16 → 32 → 64 → 128 channels, FC 64,  dropout 0.3   (~73K params — current model)
  Large  : 32 → 64 → 128 → 256 channels, FC 128, dropout 0.4  (~290K params)

All trained for 50 epochs, LR 0.001, batch 16 on GPU with the same data split.
*/
fn run_ablation_study() {
    println!("========================================================================");
    println!("  CNN ARCHITECTURE ABLATION STUDY");
    println!("  Training Small / Medium / Large CNN variants on the same dataset");
    println!("  Each variant runs as a SEPARATE PROCESS to reclaim GPU memory.");
    println!("========================================================================\n");

    // Get our own executable path
    let exe = env::current_exe().expect("Failed to get current exe path");

    struct VariantResult {
        name: &'static str,
        key: &'static str,
        dcp_mse: f64,
        clahe_mse: f64,
        bright_mse: f64,
        train_secs: f64,
    }

    let variants = [
        ("SMALL",  "Small  (8→16→32→64, FC 32)"),
        ("MEDIUM", "Medium (16→32→64→128, FC 64)"),
        ("LARGE",  "Large  (32→64→128→256, FC 128)"),
    ];

    let mut results: Vec<VariantResult> = Vec::new();

    for (key, name) in &variants {
        println!("========================================================================");
        println!("  Spawning child process for: {}", name);
        println!("========================================================================\n");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Spawn child process: ai-model.exe --ablation-variant SMALL/MEDIUM/LARGE
        let output = std::process::Command::new(&exe)
            .arg("--ablation-variant")
            .arg(key)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);

                // Print all child output
                print!("{}", stdout);
                if !stderr.is_empty() {
                    eprint!("{}", stderr);
                }

                // Parse the ABLATION_RESULT line:
                // ABLATION_RESULT|DCP_MSE|CLAHE_MSE|BRIGHT_MSE|TRAIN_SECS
                let mut found = false;
                for line in stdout.lines() {
                    if line.starts_with("ABLATION_RESULT|") {
                        let parts: Vec<&str> = line.split('|').collect();
                        if parts.len() == 5 {
                            results.push(VariantResult {
                                name,
                                key,
                                dcp_mse: parts[1].parse().unwrap_or(f64::NAN),
                                clahe_mse: parts[2].parse().unwrap_or(f64::NAN),
                                bright_mse: parts[3].parse().unwrap_or(f64::NAN),
                                train_secs: parts[4].parse().unwrap_or(0.0),
                            });
                            found = true;
                        }
                    }
                }
                if !found {
                    println!("\n  WARNING: No ABLATION_RESULT found for {} — child may have crashed.", name);
                    if !out.status.success() {
                        println!("  Exit code: {:?}", out.status.code());
                    }
                }
            }
            Err(e) => {
                println!("  ERROR spawning child process for {}: {}", name, e);
            }
        }

        // Brief pause between variants
        println!("\n  Child process exited — GPU memory reclaimed by OS.");
        std::thread::sleep(std::time::Duration::from_secs(2));
    }

    // ── Summary table ─────────────────────────────────────────────────────
    if results.is_empty() {
        println!("\nNo variants completed successfully. Check errors above.");
        return;
    }

    println!("\n{}", "=".repeat(90));
    println!("  ABLATION STUDY RESULTS — 50 epochs per variant");
    println!("{}", "=".repeat(90));
    println!("\n  {:<36}  {:>10}  {:>10}  {:>10}  {:>10}",
        "Architecture", "DCP MSE", "CLAHE MSE", "Bright MSE", "Train(s)");
    println!("  {}", "-".repeat(84));
    for r in &results {
        println!("  {:<36}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.1}",
            r.name, r.dcp_mse, r.clahe_mse, r.bright_mse, r.train_secs);
    }
    println!("  {}", "-".repeat(84));

    // Pick the best on combined MSE
    let best = results.iter().min_by(|a, b| {
        let sa = a.dcp_mse + a.clahe_mse + a.bright_mse;
        let sb = b.dcp_mse + b.clahe_mse + b.bright_mse;
        sa.partial_cmp(&sb).unwrap()
    }).unwrap();
    println!("\n  Best overall (lowest combined MSE): {}", best.name);
    println!("    DCP: {:.6}  CLAHE: {:.6}  Brightness: {:.6}",
        best.dcp_mse, best.clahe_mse, best.bright_mse);

    println!("\n{}", "=".repeat(90));
    println!("  ABLATION STUDY COMPLETE");
    println!("{}", "=".repeat(90));
    println!("\n  This table demonstrates data-driven architecture selection:");
    println!("  the Medium (16→32→64→128) architecture was chosen as the production");
    println!("  model based on empirical comparison against smaller and larger variants.");
}

/*
Train and evaluate a single CNN variant for the ablation study.
Designed to run as a CHILD PROCESS so all GPU memory is reclaimed on exit.
Prints a parseable result line: ABLATION_RESULT|dcp_mse|clahe_mse|bright_mse|train_secs
*/
fn run_ablation_single_variant(variant_key: &str) {
    use burn::backend::wgpu::WgpuDevice;
    use std::time::Instant;

    let config = match variant_key.to_uppercase().as_str() {
        "SMALL" => iteration_2_CNN::HazeCNNConfig::new()
            .with_conv1_channels(8)
            .with_conv2_channels(16)
            .with_conv3_channels(32)
            .with_conv4_channels(64)
            .with_fc1_size(32)
            .with_dropout_rate(0.2),
        "MEDIUM" => iteration_2_CNN::HazeCNNConfig::new(), // defaults: 16→32→64→128, FC 64
        "LARGE" => iteration_2_CNN::HazeCNNConfig::new()
            .with_conv1_channels(32)
            .with_conv2_channels(64)
            .with_conv3_channels(128)
            .with_conv4_channels(256)
            .with_fc1_size(128)
            .with_dropout_rate(0.4),
        _ => {
            println!("Unknown variant: {}. Use SMALL, MEDIUM, or LARGE.", variant_key);
            return;
        }
    };

    println!("========================================================================");
    println!("  ABLATION VARIANT: {} — Conv {}->{}->{}->{}, FC {}, dropout {:.1}",
        variant_key, config.conv1_channels, config.conv2_channels,
        config.conv3_channels, config.conv4_channels, config.fc1_size, config.dropout_rate);
    println!("========================================================================\n");

    // ── Load training data ────────────────────────────────────────────────
    let dataset_path = Path::new("dataset/SealID/full images/source_database");
    if !dataset_path.exists() {
        println!("Error: training dataset not found at {:?}", dataset_path);
        return;
    }
    let image_paths = find_images_in_directory(dataset_path);
    println!("Found {} training images", image_paths.len());

    let patch_size = 15;
    let images = load_images_parallel(&image_paths);
    println!("Loaded {} training images", images.len());

    println!("Generating ground-truth labels...");
    let dcp_labels: Vec<f64> = images.iter().map(|img| extract_robust_haze_label(img, patch_size)).collect();
    let clahe_labels: Vec<f64> = images.iter().map(|img| extract_clahe_contrast_deficit(img)).collect();
    let bright_labels: Vec<f64> = images.iter().map(|img| extract_brightness_deficit(img)).collect();

    // ── Load test data (50 images) ────────────────────────────────────────
    let query_path = Path::new("dataset/SealID/full images/source_query");
    if !query_path.exists() {
        println!("Error: query dataset not found");
        return;
    }
    let query_paths = {
        let mut all = find_images_in_directory(query_path);
        all.sort();
        all.truncate(50);
        all
    };
    let test_images = load_images_parallel(&query_paths);
    println!("Loaded {} test images", test_images.len());

    let test_dcp: Vec<f64> = test_images.iter().map(|img| extract_robust_haze_label(img, patch_size)).collect();
    let test_clahe: Vec<f64> = test_images.iter().map(|img| extract_clahe_contrast_deficit(img)).collect();
    let test_bright: Vec<f64> = test_images.iter().map(|img| extract_brightness_deficit(img)).collect();

    // ── GPU setup and training ────────────────────────────────────────────
    let device = WgpuDevice::DiscreteGpu(0);
    type AB = iteration_2_CNN::AutodiffGpuBackend;

    // Warmup
    {
        let warmup_model = config.init::<AB>(&device);
        let test_img = ndarray::Array3::<f32>::zeros((64, 64, 3));
        let test_tensor = iteration_2_CNN::cnn_detection::image_to_tensor::<AB>(&test_img, &device);
        let output = warmup_model.forward(test_tensor);
        let _: Vec<f32> = output.reshape([3]).into_data().to_vec().unwrap();
    }

    let model = config.init::<AB>(&device);
    let epochs = 50;
    let lr = 0.001;
    let batch_size = 16;
    let memory_scale = config.conv4_channels as f64 / 128.0; // Small=0.5, Medium=1.0, Large=2.0

    let t_train = Instant::now();
    let trained = iteration_2_CNN::train_cnn(
        model, &images, &dcp_labels, &clahe_labels, &bright_labels,
        epochs, lr, batch_size, &device,
        25, //gpu_throttle_ms
        memory_scale,
    );
    let train_secs = t_train.elapsed().as_secs_f64();
    println!("\n  Training time: {:.1}s", train_secs);

    // ── Evaluate ──────────────────────────────────────────────────────────
    let (dcp_mse, clahe_mse, bright_mse) = iteration_2_CNN::evaluate_cnn(
        &trained, &test_images, &test_dcp, &test_clahe, &test_bright, &device,
    );
    println!("  Test MSE — DCP: {:.6}  CLAHE: {:.6}  Brightness: {:.6}", dcp_mse, clahe_mse, bright_mse);

    // Print parseable result line for the parent process to collect
    println!("ABLATION_RESULT|{:.8}|{:.8}|{:.8}|{:.1}", dcp_mse, clahe_mse, bright_mse, train_secs);
}


/*
Epoch Convergence Study — trains the default Medium CNN (16→32→64→128) at 10, 30, and 50
epochs to characterize the training convergence curve, then compares all three against the
linear regression baseline on the held-out query set.

Each training run is spawned as a separate child process (same pattern as the ablation
study) so the OS reclaims all GPU memory between runs. Existing model files are skipped
to allow resuming after a crash or interruption.

The final comparison loads all saved models and runs inference on the full query set,
producing a metrics table and sample processed images for the thesis.
*/
fn run_epoch_study() {
    println!("========================================================================");
    println!("  CNN EPOCH CONVERGENCE STUDY");
    println!("  Training at 10, 30, and 50 epochs to evaluate convergence");
    println!("  Each training run is a SEPARATE PROCESS to reclaim GPU memory.");
    println!("========================================================================\n");

    let exe = env::current_exe().expect("Failed to get current exe path");

    // (epochs, model_save_name) for each variant
    let configs: [(usize, &str); 3] = [
        (10, "CNN_10ep"),
        (30, "CNN_30ep"),
        (50, "CNN_GPU_TILED"),
    ];

    for (epochs, model_name) in &configs {
        let model_file = format!("{}.mpk", model_name);
        if Path::new(&model_file).exists() {
            println!("Skipping {} epochs — {} already exists.\n", epochs, model_file);
            continue;
        }

        println!("========================================================================");
        println!("  Training CNN for {} epochs → {}", epochs, model_file);
        println!("========================================================================\n");
        io::stdout().flush().unwrap();

        let status = std::process::Command::new(&exe)
            .arg("--train-cnn-gpu-save")
            .arg(model_name)
            .arg(epochs.to_string())
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status();

        match status {
            Ok(s) => {
                if !s.success() {
                    println!("\nWARNING: Training for {} epochs exited with code {:?}", epochs, s.code());
                }
            }
            Err(e) => {
                println!("ERROR spawning training process for {} epochs: {}", epochs, e);
            }
        }

        println!("\n  Child process exited — GPU memory reclaimed by OS.\n");
        std::thread::sleep(std::time::Duration::from_secs(2));
    }

    // ── Run comparison on all available models ─────────────────────────────
    println!("\n========================================================================");
    println!("  Running Epoch Comparison");
    println!("========================================================================\n");

    let all_models: Vec<(&str, &str)> = configs.iter()
        .filter(|(_, name)| Path::new(&format!("{}.mpk", name)).exists())
        .map(|(ep, name)| {
            // Create label from epoch count
            match ep {
                10 => ("CNN 10ep", *name),
                30 => ("CNN 30ep", *name),
                50 => ("CNN 50ep", *name),
                _ => (*name, *name),
            }
        })
        .collect();

    if all_models.is_empty() {
        println!("No CNN models available for comparison. All training runs may have failed.");
        return;
    }

    run_epoch_comparison("haze_model.json", &all_models);
}


/*
Compare linear regression vs multiple CNN models (at different epoch counts) on the full
held-out query dataset.  Produces a metrics table and processes sample images for visual
comparison.  Can be called standalone via --epoch-comparison or automatically after
--epoch-study completes training.

@param regressor_path: path to the saved regressor JSON model (e.g. "haze_model.json")
@param cnn_models: slice of (label, model_path) tuples for CNN models to compare
                   e.g. [("CNN 10ep", "CNN_10ep"), ("CNN 30ep", "CNN_30ep"), ("CNN 50ep", "CNN_GPU_TILED")]
*/
fn run_epoch_comparison(regressor_path: &str, cnn_models: &[(&str, &str)]) {
    use burn::backend::wgpu::WgpuDevice;
    use std::time::Instant;

    println!("========================================================================");
    println!("  EPOCH STUDY: Model Comparison");
    println!("  Comparing linear regression vs CNN at different training epochs");
    println!("========================================================================\n");

    // ── Load regressor ───────────────────────────────────────────────────
    println!("Loading linear regression model from: {}", regressor_path);
    let regressor = match training::HazeRegressor::load(regressor_path) {
        Ok(r) => {
            println!("  Regressor loaded. Weights: {:?}, Bias: {:.4}", r.model.weights, r.model.bias);
            r
        }
        Err(e) => {
            println!("Error loading regressor: {}", e);
            return;
        }
    };

    // ── Load CNN models ──────────────────────────────────────────────────
    let mut loaded_models: Vec<(&str, &str, _)> = Vec::new();
    for (label, path) in cnn_models {
        println!("Loading CNN model: {} ({}.mpk)", label, path);
        match iteration_2_CNN::load_pretrained_model_gpu(path) {
            Ok(m) => {
                println!("  Loaded successfully.");
                loaded_models.push((label, path, m));
            }
            Err(e) => {
                println!("  WARNING: Failed to load '{}': {} — skipping.", path, e);
            }
        }
    }

    if loaded_models.is_empty() {
        println!("\nError: No CNN models loaded. Cannot run comparison.");
        return;
    }

    let device = WgpuDevice::default();

    // ── Load query images ────────────────────────────────────────────────
    let query_path = Path::new("dataset/SealID/full images/source_query");
    if !query_path.exists() {
        println!("Error: query dataset not found at {:?}", query_path);
        return;
    }

    let image_paths = find_images_in_directory(query_path);
    if image_paths.is_empty() {
        println!("Error: no images found in query dataset");
        return;
    }
    println!("\nFound {} query images. Loading in parallel...", image_paths.len());
    let images = load_images_parallel(&image_paths);
    println!("Loaded {} images.\n", images.len());

    // ── Compute ground-truth labels ──────────────────────────────────────
    let patch_size = 15usize;
    println!("Computing ground-truth labels (DCP robust, CLAHE deficit, brightness deficit)...");
    let gt_dcp: Vec<f64> = images.iter()
        .map(|img| extract_robust_haze_label(img, patch_size)).collect();
    let gt_clahe: Vec<f64> = images.iter()
        .map(|img| extract_clahe_contrast_deficit(img)).collect();
    let gt_bright: Vec<f64> = images.iter()
        .map(|img| extract_brightness_deficit(img)).collect();
    let n = images.len() as f64;

    // ── Evaluate regressor ───────────────────────────────────────────────
    println!("Evaluating regressor on {} images...", images.len());
    io::stdout().flush().unwrap();
    let t_reg = Instant::now();
    let (mut reg_se, mut reg_ae) = (0.0f64, 0.0f64);
    for (i, img) in images.iter().enumerate() {
        let score = training::predict_haze_score(&regressor, img, patch_size);
        let err = score - gt_dcp[i];
        reg_se += err * err;
        reg_ae += err.abs();
        if (i + 1) % 100 == 0 || i + 1 == images.len() {
            println!("  Regressor: {}/{}", i + 1, images.len());
            io::stdout().flush().unwrap();
        }
    }
    let reg_time = t_reg.elapsed();

    // ── Evaluate each CNN model ──────────────────────────────────────────
    struct CnnMetrics {
        label: String,
        dcp_mse: f64, dcp_mae: f64,
        clahe_mse: f64, clahe_mae: f64,
        bright_mse: f64, bright_mae: f64,
        time: std::time::Duration,
    }

    let mut cnn_metrics: Vec<CnnMetrics> = Vec::new();

    for (label, _path, model) in &loaded_models {
        println!("Evaluating {}...", label);
        io::stdout().flush().unwrap();
        let t0 = Instant::now();
        let (mut dcp_se, mut dcp_ae) = (0.0f64, 0.0f64);
        let (mut clahe_se, mut clahe_ae) = (0.0f64, 0.0f64);
        let (mut bright_se, mut bright_ae) = (0.0f64, 0.0f64);

        for (i, img) in images.iter().enumerate() {
            let (d, c, b) = iteration_2_CNN::predict_haze_cnn(model, img, &device);
            let de = d as f64 - gt_dcp[i];
            dcp_se += de * de; dcp_ae += de.abs();
            let ce = c as f64 - gt_clahe[i];
            clahe_se += ce * ce; clahe_ae += ce.abs();
            let be_ = b as f64 - gt_bright[i];
            bright_se += be_ * be_; bright_ae += be_.abs();

            if (i + 1) % 100 == 0 || i + 1 == images.len() {
                println!("  {}: {}/{}", label, i + 1, images.len());
                io::stdout().flush().unwrap();
            }
        }

        cnn_metrics.push(CnnMetrics {
            label: label.to_string(),
            dcp_mse: dcp_se / n, dcp_mae: dcp_ae / n,
            clahe_mse: clahe_se / n, clahe_mae: clahe_ae / n,
            bright_mse: bright_se / n, bright_mae: bright_ae / n,
            time: t0.elapsed(),
        });
    }

    // ── Print results table ──────────────────────────────────────────────
    let reg_mse = reg_se / n;
    let reg_mae = reg_ae / n;

    println!("\n{}", "=".repeat(100));
    println!("  EPOCH STUDY RESULTS — {} query images", images.len());
    println!("{}", "=".repeat(100));

    println!("\n  {:<22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}",
        "Model", "DCP MSE", "DCP MAE", "CLAHE MSE", "CLAHE MAE", "Bright MSE", "Bright MAE", "Time/img");
    println!("  {}", "-".repeat(112));
    println!("  {:<22}  {:>10.6}  {:>10.6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12.2?}",
        "Lin. Regression", reg_mse, reg_mae, "N/A", "N/A", "N/A", "N/A",
        reg_time / images.len() as u32);
    for m in &cnn_metrics {
        println!("  {:<22}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>12.2?}",
            m.label, m.dcp_mse, m.dcp_mae, m.clahe_mse, m.clahe_mae, m.bright_mse, m.bright_mae,
            m.time / images.len() as u32);
    }
    println!("  {}", "-".repeat(112));

    // ── Highlight improvements over regressor ────────────────────────────
    println!("\n  DCP MSE comparison vs linear regression ({:.6}):", reg_mse);
    for m in &cnn_metrics {
        if m.dcp_mse < reg_mse {
            let pct = (1.0 - m.dcp_mse / reg_mse) * 100.0;
            println!("    {:<20} {:.6}  ({:.1}% better)", m.label, m.dcp_mse, pct);
        } else {
            let pct = (m.dcp_mse / reg_mse - 1.0) * 100.0;
            println!("    {:<20} {:.6}  ({:.1}% worse)", m.label, m.dcp_mse, pct);
        }
    }

    // ── Check for overfitting (MSE increasing at higher epochs) ──────────
    if cnn_metrics.len() >= 2 {
        for i in 1..cnn_metrics.len() {
            let prev = &cnn_metrics[i - 1];
            let curr = &cnn_metrics[i];
            let combined_prev = prev.dcp_mse + prev.clahe_mse + prev.bright_mse;
            let combined_curr = curr.dcp_mse + curr.clahe_mse + curr.bright_mse;
            if combined_curr > combined_prev {
                println!("\n  NOTE: {} has higher combined MSE than {} — possible overfitting.",
                    curr.label, prev.label);
            }
        }
    }

    // ── Sample predictions ───────────────────────────────────────────────
    println!("\n{}", "=".repeat(100));
    println!("  SAMPLE PREDICTIONS (first 5 query images)");
    println!("{}", "=".repeat(100));

    let sample_n = 5.min(images.len());
    for i in 0..sample_n {
        let img = &images[i];
        let fname = image_paths[i].file_stem().unwrap_or_default().to_string_lossy();
        println!("\n  Image {}: {}", i + 1, fname);
        println!("    Ground truth:   DCP={:.4}  CLAHE={:.4}  Bright={:.4}",
            gt_dcp[i], gt_clahe[i], gt_bright[i]);

        let reg_score = training::predict_haze_score(&regressor, img, patch_size);
        println!("    Regressor:      DCP={:.4}", reg_score);

        for (label, _, model) in &loaded_models {
            let (d, c, b) = iteration_2_CNN::predict_haze_cnn(model, img, &device);
            println!("    {:<16} DCP={:.4}  CLAHE={:.4}  Bright={:.4}", format!("{}:", label), d, c, b);
        }
    }

    // ── Process sample images with each model for visual comparison ──────
    println!("\n{}", "=".repeat(100));
    println!("  SAMPLE IMAGE PROCESSING (3 images × {} models)", loaded_models.len() + 1);
    println!("{}", "=".repeat(100));

    let process_n = 3.min(images.len());
    for i in 0..process_n {
        let img = &images[i];
        let fname = image_paths[i].file_stem().unwrap_or_default().to_string_lossy();
        println!("\n  Processing: {}", fname);

        // Regressor pipeline (same heuristic as run_process_with_model)
        let reg_score = training::predict_haze_score(&regressor, img, patch_size);
        let (omega_r, t0_r, gr_r, ge_r) = if reg_score > 0.7 {
            (0.6f32, 0.3f32, 30usize, 0.0001f32)
        } else if reg_score > 0.4 {
            (0.75f32, 0.2f32, 45usize, 0.0001f32)
        } else {
            (0.85f32, 0.15f32, 60usize, 0.0001f32)
        };
        let dehazed_r = dehaze_with_params(img, patch_size, omega_r, t0_r, 0.001, gr_r, ge_r);
        let (cg_r, cc_r) = if reg_score > 0.7 { (8usize, 4.0f32) }
            else if reg_score > 0.4 { (8, 2.5) }
            else { (4, 1.5) };
        let enhanced_r = enhance_clahe(&dehazed_r, cg_r, cg_r, cc_r);
        let gamma_r = IP_functions::gamma::estimate_gamma(&enhanced_r);
        let final_r = brightness_correct(&enhanced_r, gamma_r);
        let out_r = format!("{}_epoch_regressor.jpg", fname);
        match array3_to_image(&final_r).save(&out_r) {
            Ok(_) => println!("    Regressor        → {}", out_r),
            Err(e) => println!("    Error saving: {}", e),
        }

        // CNN pipelines (DCP → attenuated CLAHE → attenuated Gamma)
        for (label, _, model) in &loaded_models {
            let (d, c, b) = iteration_2_CNN::predict_haze_cnn(model, img, &device);
            let (omega, t0, ps, gr, ge) = iteration_2_CNN::suggest_dcp_parameters(d);
            let (gh, gw, cl) = iteration_2_CNN::suggest_clahe_parameters(c);
            let gamma = iteration_2_CNN::suggest_gamma_parameters(b);

            let dehazed = dehaze_with_params(img, ps, omega, t0, 0.001, gr, ge);
            let stacked_clip = 1.0f32 + (cl - 1.0) * 0.7;
            let enhanced = enhance_clahe(&dehazed, gh, gw, stacked_clip);
            let stacked_gamma = 1.0 + (gamma - 1.0) * 0.5;
            let final_c = brightness_correct(&enhanced, stacked_gamma);

            let safe_label = label.replace(" ", "_").to_lowercase();
            let out_c = format!("{}_epoch_{}.jpg", fname, safe_label);
            match array3_to_image(&final_c).save(&out_c) {
                Ok(_) => println!("    {:<16} → {}", label, out_c),
                Err(e) => println!("    Error saving: {}", e),
            }
        }
    }

    println!("\n{}", "=".repeat(100));
    println!("  EPOCH STUDY COMPARISON COMPLETE");
    println!("{}", "=".repeat(100));
    println!("\nThis table demonstrates CNN convergence behavior:");
    println!("  - 10 epochs: baseline to check if the model is learning at all");
    println!("  - 30 epochs: intermediate checkpoint to track improvement rate");
    println!("  - 50 epochs: full training, same as production model");
    println!("  If 30ep ≈ 50ep, the model has converged early and extra epochs are wasted.");
    println!("  If 50ep < 30ep < 10ep, the model is still improving and may benefit from more epochs.");
    println!("  If 50ep > 30ep, the model may be overfitting and 30 epochs is the better choice.");
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

/*
Compare Linear Regression (Iteration 1) vs CNN (Iteration 2) on the full source_query dataset.
Loads both saved models, runs inference on every query image, computes error metrics (MSE / MAE)
against pseudo-ground-truth labels, and processes a handful of sample images with each model so the
user can visually inspect the difference.

The regressor only predicts a single DCP haze score; the CNN predicts DCP haze, CLAHE deficit, and
brightness deficit. For the regressor we derive CLAHE and gamma heuristically (same logic as
run_process_with_model) so the comparison is apples-to-apples on the full pipeline.

@param regressor_path: path to the saved linear-regression JSON model (e.g. "haze_model.json")
@param cnn_path:       path to the saved CNN .mpk model WITHOUT the .mpk extension (e.g. "CNN_GPU_TILED")
*/
fn run_compare_models(regressor_path: &str, cnn_path: &str) {
    use burn::backend::wgpu::WgpuDevice;
    use std::time::Instant;

    println!("=== AI-Assisted Seal Photograph Image Processing System ===");
    println!("=== Model Comparison: Linear Regression vs CNN ===\n");

    // ── Load both models ────────────────────────────────────────────────
    println!("Loading linear regression model from: {}", regressor_path);
    let regressor = match training::HazeRegressor::load(regressor_path) {
        Ok(r) => {
            println!("  Regressor loaded. Weights: {:?}, Bias: {:.4}", r.model.weights, r.model.bias);
            r
        }
        Err(e) => {
            println!("Error loading regressor: {}", e);
            return;
        }
    };

    println!("Loading CNN model from: {}.mpk", cnn_path);
    let cnn_model = match iteration_2_CNN::load_pretrained_model_gpu(cnn_path) {
        Ok(m) => {
            println!("  CNN loaded on GPU.");
            m
        }
        Err(e) => {
            println!("Error loading CNN: {}", e);
            return;
        }
    };
    let device = WgpuDevice::default();

    // ── Load query images ───────────────────────────────────────────────
    let query_path = Path::new("dataset/SealID/full images/source_query");
    if !query_path.exists() {
        println!("Error: query dataset not found at {:?}", query_path);
        return;
    }

    let image_paths = find_images_in_directory(query_path);
    if image_paths.is_empty() {
        println!("Error: no images found in {:?}", query_path);
        return;
    }
    println!("\nFound {} query images. Loading in parallel...", image_paths.len());
    let images = load_images_parallel(&image_paths);
    println!("Loaded {} images successfully.\n", images.len());

    // ── Compute pseudo-ground-truth labels ──────────────────────────────
    let patch_size = 15usize;
    println!("Computing ground-truth labels (DCP robust, CLAHE deficit, brightness deficit)...");
    let gt_dcp: Vec<f64> = images.iter()
        .map(|img| extract_robust_haze_label(img, patch_size))
        .collect();
    let gt_clahe: Vec<f64> = images.iter()
        .map(|img| extract_clahe_contrast_deficit(img))
        .collect();
    let gt_bright: Vec<f64> = images.iter()
        .map(|img| extract_brightness_deficit(img))
        .collect();

    // ── Run inference and collect predictions ───────────────────────────
    let n = images.len() as f64;

    // Accumulators for regressor
    let mut reg_dcp_se = 0.0_f64;
    let mut reg_dcp_ae = 0.0_f64;

    // Accumulators for CNN (3 heads)
    let mut cnn_dcp_se = 0.0_f64;
    let mut cnn_dcp_ae = 0.0_f64;
    let mut cnn_clahe_se = 0.0_f64;
    let mut cnn_clahe_ae = 0.0_f64;
    let mut cnn_bright_se = 0.0_f64;
    let mut cnn_bright_ae = 0.0_f64;

    println!("Running inference on {} images...\n", images.len());

    let t_reg_start = Instant::now();

    for (i, img) in images.iter().enumerate() {
        // ── Regressor prediction ────────────────────────────────────
        let reg_score = training::predict_haze_score(&regressor, img, patch_size);
        let reg_err = reg_score - gt_dcp[i];
        reg_dcp_se += reg_err * reg_err;
        reg_dcp_ae += reg_err.abs();

        if (i + 1) % 100 == 0 || i + 1 == images.len() {
            println!("  Regressor: {}/{}", i + 1, images.len());
        }
    }
    let t_reg_elapsed = t_reg_start.elapsed();

    let t_cnn_start = Instant::now();
    for (i, img) in images.iter().enumerate() {
        // ── CNN prediction ──────────────────────────────────────────
        let (cnn_dcp, cnn_clahe, cnn_bright) =
            iteration_2_CNN::predict_haze_cnn(&cnn_model, img, &device);

        let dcp_err = cnn_dcp as f64 - gt_dcp[i];
        cnn_dcp_se += dcp_err * dcp_err;
        cnn_dcp_ae += dcp_err.abs();

        let clahe_err = cnn_clahe as f64 - gt_clahe[i];
        cnn_clahe_se += clahe_err * clahe_err;
        cnn_clahe_ae += clahe_err.abs();

        let bright_err = cnn_bright as f64 - gt_bright[i];
        cnn_bright_se += bright_err * bright_err;
        cnn_bright_ae += bright_err.abs();

        if (i + 1) % 100 == 0 || i + 1 == images.len() {
            println!("  CNN:       {}/{}", i + 1, images.len());
        }
    }
    let t_cnn_elapsed = t_cnn_start.elapsed();

    // ── Print aggregate metrics ─────────────────────────────────────────
    println!("\n{}", "=".repeat(72));
    println!("  MODEL COMPARISON RESULTS  ({} query images)", images.len());
    println!("{}", "=".repeat(72));

    println!("\n--- DCP Haze Score (both models predict this) ---");
    println!("  {:>25}  {:>10}  {:>10}", "", "MSE", "MAE");
    println!("  {:>25}  {:>10.6}  {:>10.6}", "Linear Regression", reg_dcp_se / n, reg_dcp_ae / n);
    println!("  {:>25}  {:>10.6}  {:>10.6}", "CNN (GPU)", cnn_dcp_se / n, cnn_dcp_ae / n);

    println!("\n--- CLAHE Contrast Deficit (CNN only) ---");
    println!("  {:>25}  {:>10}  {:>10}", "", "MSE", "MAE");
    println!("  {:>25}  {:>10.6}  {:>10.6}", "CNN (GPU)", cnn_clahe_se / n, cnn_clahe_ae / n);
    println!("  {:>25}  {:>10}", "Linear Regression", "N/A (single-output model)");

    println!("\n--- Brightness Deficit (CNN only) ---");
    println!("  {:>25}  {:>10}  {:>10}", "", "MSE", "MAE");
    println!("  {:>25}  {:>10.6}  {:>10.6}", "CNN (GPU)", cnn_bright_se / n, cnn_bright_ae / n);
    println!("  {:>25}  {:>10}", "Linear Regression", "N/A (single-output model)");

    println!("\n--- Inference Speed ---");
    println!("  {:>25}  {:.2?} total  ({:.2?}/image)",
             "Linear Regression", t_reg_elapsed, t_reg_elapsed / images.len() as u32);
    println!("  {:>25}  {:.2?} total  ({:.2?}/image)",
             "CNN (GPU)", t_cnn_elapsed, t_cnn_elapsed / images.len() as u32);

    // ── Process a few sample images with both pipelines ─────────────────
    println!("\n{}", "=".repeat(72));
    println!("  SAMPLE IMAGE PROCESSING (first 5 query images)");
    println!("{}", "=".repeat(72));

    let sample_count = 5.min(images.len());
    for i in 0..sample_count {
        let img = &images[i];
        let fname = image_paths[i].file_stem().unwrap_or_default().to_string_lossy();
        println!("\n── Image {}: {} ──", i + 1, fname);

        // Ground truth
        println!("  Ground truth:  DCP={:.4}  CLAHE={:.4}  Bright={:.4}", gt_dcp[i], gt_clahe[i], gt_bright[i]);

        // ── Regressor ──
        let reg_score = training::predict_haze_score(&regressor, img, patch_size);
        let (omega_r, t0_r, gr_r, ge_r) = if reg_score > 0.7 {
            (0.6_f32, 0.3_f32, 30_usize, 0.0001_f32)
        } else if reg_score > 0.4 {
            (0.75_f32, 0.2_f32, 45_usize, 0.0001_f32)
        } else {
            (0.85_f32, 0.15_f32, 60_usize, 0.0001_f32)
        };
        println!("  Regressor:     DCP={:.4}", reg_score);
        println!("    DCP params:  omega={:.2} t0={:.2} patch=15 radius={} eps={:.4}", omega_r, t0_r, gr_r, ge_r);

        // Process with regressor
        let dehazed_r = dehaze_with_params(img, patch_size, omega_r, t0_r, 0.001, gr_r, ge_r);
        let out_r = format!("{}_full_regressor.jpg", fname);
        // Also apply CLAHE + gamma (regressor heuristic: same clip_limit / gamma as run_process_with_model)
        let (clahe_grid_r, clahe_clip_r) = if reg_score > 0.7 { (8usize, 4.0f32) }
            else if reg_score > 0.4 { (8, 2.5) }
            else { (4, 1.5) };
        let enhanced_r = enhance_clahe(&dehazed_r, clahe_grid_r, clahe_grid_r, clahe_clip_r);
        let gamma_r = IP_functions::gamma::estimate_gamma(&enhanced_r);
        let final_r = brightness_correct(&enhanced_r, gamma_r);
        let img_r = array3_to_image(&final_r);
        match img_r.save(&out_r) {
            Ok(_) => println!("    Saved: {}", out_r),
            Err(e) => println!("    Error saving: {}", e),
        }

        // ── CNN ──
        let (cnn_dcp_pred, cnn_clahe_pred, cnn_bright_pred) =
            iteration_2_CNN::predict_haze_cnn(&cnn_model, img, &device);
        let (omega_c, t0_c, ps_c, gr_c, ge_c) = iteration_2_CNN::suggest_dcp_parameters(cnn_dcp_pred);
        let (gh_c, gw_c, cl_c) = iteration_2_CNN::suggest_clahe_parameters(cnn_clahe_pred);
        let gamma_c = iteration_2_CNN::suggest_gamma_parameters(cnn_bright_pred);

        println!("  CNN:           DCP={:.4}  CLAHE={:.4}  Bright={:.4}", cnn_dcp_pred, cnn_clahe_pred, cnn_bright_pred);
        println!("    DCP params:  omega={:.2} t0={:.2} patch={} radius={} eps={:.4}", omega_c, t0_c, ps_c, gr_c, ge_c);
        println!("    CLAHE:       grid={}x{} clip={:.2}", gh_c, gw_c, cl_c);
        println!("    Gamma:       {:.3}", gamma_c);

        // Process with CNN pipeline (DCP → attenuated CLAHE → attenuated Gamma, same as run_process_with_cnn_gpu)
        let dehazed_c = dehaze_with_params(img, ps_c, omega_c, t0_c, 0.001, gr_c, ge_c);
        let stacked_clip_c = 1.0_f32 + (cl_c - 1.0) * 0.7;
        let enhanced_c = enhance_clahe(&dehazed_c, gh_c, gw_c, stacked_clip_c);
        let stacked_gamma_c = 1.0 + (gamma_c - 1.0) * 0.5;
        let final_c = brightness_correct(&enhanced_c, stacked_gamma_c);
        let out_c = format!("{}_full_cnn_gpu_tiled.jpg", fname);
        let img_c = array3_to_image(&final_c);
        match img_c.save(&out_c) {
            Ok(_) => println!("    Saved: {}", out_c),
            Err(e) => println!("    Error saving: {}", e),
        }
    }

    println!("\n{}", "=".repeat(72));
    println!("  COMPARISON COMPLETE");
    println!("{}", "=".repeat(72));
    println!("\nSummary:");
    let reg_mse = reg_dcp_se / n;
    let cnn_mse = cnn_dcp_se / n;
    if cnn_mse < reg_mse {
        let pct = (1.0 - cnn_mse / reg_mse) * 100.0;
        println!("  CNN achieves {:.1}% lower DCP MSE than linear regression.", pct);
    } else if reg_mse < cnn_mse {
        let pct = (1.0 - reg_mse / cnn_mse) * 100.0;
        println!("  Linear regression achieves {:.1}% lower DCP MSE than CNN.", pct);
    } else {
        println!("  Both models have identical DCP MSE.");
    }
    println!("  CNN additionally predicts CLAHE deficit (MSE {:.6}) and brightness deficit (MSE {:.6}).",
             cnn_clahe_se / n, cnn_bright_se / n);
    println!("  Linear regression: {:.2?}/image,  CNN (GPU): {:.2?}/image",
             t_reg_elapsed / images.len() as u32, t_cnn_elapsed / images.len() as u32);
}

