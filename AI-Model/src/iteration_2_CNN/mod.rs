//Iteration 2: CNN-based haze detection module
//This module contains the CNN implementation using the burn ML framework to replace the linear regression regressor from Iteration 1

pub mod cnn_detection;

//re-export running functions from main for convenience
pub use cnn_detection::run_cnn_training;
pub use cnn_detection::run_cnn_demo;

