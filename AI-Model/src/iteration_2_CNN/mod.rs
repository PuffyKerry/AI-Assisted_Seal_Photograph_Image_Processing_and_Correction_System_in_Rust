//Iteration 2: CNN-based haze detection module
//This module contains the CNN implementation using the burn ML framework to replace the linear regression regressor from Iteration 1

pub mod cnn_detection;

//re-export running functions from main for convenience
pub use cnn_detection::run_cnn_training;
pub use cnn_detection::run_cnn_training_gpu;  //GPU-accelerated training
pub use cnn_detection::run_cnn_demo;
pub use cnn_detection::load_pretrained_model;
pub use cnn_detection::load_pretrained_model_gpu;  //GPU inference
pub use cnn_detection::load_pretrained_model_for_training;
pub use cnn_detection::predict_haze_cnn;  //inference helper
pub use cnn_detection::preprocess_for_cnn; //CLAHE preprocessing used by both training and inference
pub use cnn_detection::suggest_dcp_parameters;  //parameter suggestion based on haze score
pub use cnn_detection::suggest_clahe_parameters; //CLAHE parameter suggestion based on haze score
pub use cnn_detection::suggest_gamma_parameters;  //gamma parameter suggestion based on brightness deficit
pub use cnn_detection::resize_images_to_common_size;  //resize for efficient GPU batching

//re-export types for external use
pub use cnn_detection::{HazeCNN, HazeCNNConfig, CnnBackend, AutodiffCnnBackend, GpuBackend, AutodiffGpuBackend};
pub use cnn_detection::train_cnn;      //training function (generic over backend)
pub use cnn_detection::evaluate_cnn;   //evaluation function (generic over backend)

