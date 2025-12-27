//Training pipeline for haze classification using linear regression as a regressor
//Outputs a continuous haze score (0.0 = clear, 1.0 = heavy haze) which can be thresholded for classification

///Q: Why a regressor?
///A: Haze levels are continuous, so I decided to use a regressor instead of a classifier such as a decision tree or perceptron.

//TODO: add model serialization/deserialization for persistence

use ndarray::{Array1, Array2, Axis};
use crate::linear_regression::LinearRegression;
use crate::extraction::extract_all_features;

//Wrapper around LinearRegression that stores normalization parameters so that predictions use the same scaling as training
pub struct HazeRegressor {
    pub model: LinearRegression,
    pub feature_mins: Array1<f64>,
    pub feature_ranges: Array1<f64>,
}

impl HazeRegressor { //AI generated after it pointed out the need for normalization via a wrapper on a LinearRegression object. Rather simple, so felt it should be fine to use AI.
    /// Normalize a single feature vector using stored parameters
    pub fn normalize_features(&self, features: &Array1<f64>) -> Array1<f64> {
        let mut normalized = Array1::<f64>::zeros(features.len());
        for j in 0..features.len() {
            if self.feature_ranges[j] > 1e-10 {
                normalized[j] = (features[j] - self.feature_mins[j]) / self.feature_ranges[j];
            } else {
                normalized[j] = 0.5;
            }
        }
        normalized
    }
}

/*
Train a linear regression model to predict haze level from image features with DCP-derived features (mean dark channel, transmission stats, atmospheric intensity) as inputs

@param: images: slice of images as 3D arrays (height x width x 3) with values in [0, 1]
@param: labels: haze labels for each image (0.0 = clear, 1.0 = heavy haze, can use intermediate values)
@param: patch_size: patch size for DCP feature extraction (typically 15), explained in DCP file
@param: learning_rate: gradient descent step size (typically 0.01) (just passed to linear regression model), explained more in linear_regression.rs
@param: epochs: number of training iterations (typically 100-500) (just passed to linear regression model), explained more in linear_regression.rs
@return: trained HazeRegressor with normalization parameters
*/
pub fn train_haze_regressor(images: &[ndarray::Array3<f32>], labels: &[f64], patch_size: usize, learning_rate: f64, epochs: usize) -> HazeRegressor {
    assert_eq!(images.len(), labels.len(), "Number of images must match number of labels");
    assert!(!images.is_empty(), "Must provide at least one training image");

    let num_features = 5; //same as extract_all_features()
    let num_samples = images.len();
    println!("Extracting features from {} images...", num_samples);

    let mut features = Array2::<f64>::zeros((num_samples, num_features));
    for (i, img) in images.iter().enumerate() { //build feature matrix [samples (aka images) x features] to determine haze level of each image
        let feat = extract_all_features(img, patch_size);
        for j in 0..num_features {
            features[[i, j]] = feat[j];
        }
        if (i + 1) % 5 == 0 || i == num_samples - 1 {
            println!("  Processed {}/{} images", i + 1, num_samples);
        }
    }

    let (normalized_features, feature_mins, feature_ranges) = normalize_features_with_params(&features); //normalize features to prevent gradient explosion (simple min-max normalization) - this was recommended by AI
    let targets = Array1::from_vec(labels.to_vec());
    println!("Training linear regression model...");
    let mut model = LinearRegression::new(num_features);
    model.train(&normalized_features, &targets, learning_rate, epochs);

    println!("Training complete!");
    HazeRegressor {
        model,
        feature_mins,
        feature_ranges,
    }
}

/*
Normalize features using min-max scaling to [0, 1] range
Returns the normalized features along with min and range values for each feature
Purpose: Prevents features with larger magnitudes from dominating gradient updates

@param: features: raw feature matrix [samples x features]
@return: tuple of (normalized features, min values, range values)
*/
fn normalize_features_with_params(features: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) { //AI generated due to simplicity.
    let (num_samples, num_features) = features.dim();
    let mut normalized = features.clone();
    let mut mins = Array1::<f64>::zeros(num_features);
    let mut ranges = Array1::<f64>::zeros(num_features);

    for j in 0..num_features {
        let column: Vec<f64> = (0..num_samples).map(|i| features[[i, j]]).collect();
        let min_val = column.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        mins[j] = min_val;
        ranges[j] = range;

        for i in 0..num_samples {
            if range > 1e-10 {
                normalized[[i, j]] = (features[[i, j]] - min_val) / range;
            } else {
                normalized[[i, j]] = 0.5; //if all values are the same, center at 0.5
            }
        }
    }
    (normalized, mins, ranges)
}

/*
Predict haze score for a single image using a trained HazeRegressor with parameters normalized

@param: regressor: trained HazeRegressor with normalization parameters (to avoid issues with gradients)
@param: img_3d: input image as 3D array (height x width x 3) with values in [0, 1]
@param: patch_size: patch size for DCP feature extraction (should match training), explained in DCP file
@return: haze score clamped to [0, 1] range, 0 is no haze, 1 is very high haze.
*/
pub fn predict_haze_score(regressor: &HazeRegressor, img_3d: &ndarray::Array3<f32>, patch_size: usize) -> f64 {
    let features = extract_all_features(img_3d, patch_size);
    let normalized = regressor.normalize_features(&features); //normalize using training parameters

    let feature_matrix = normalized.insert_axis(Axis(0)); //convert 1D feature array to 2D matrix with single row for prediction as matrix multiplication is necessary for linear regression models to predict (and 1D arrays do not support matrix multiplication)
    let prediction = regressor.model.predict(&feature_matrix);

    prediction[0].clamp(0.0, 1.0)
}



/*
Calculate Mean Squared Error on a test set for judging outcomes of this iteration

@param: regressor: trained HazeRegressor
@param: test_images: slice of test images
@param: test_labels: true haze scores for test images
@param: patch_size: patch size for DCP
@return: MSE value (lower is good, higher is bad)
*/
pub fn evaluate_mse(regressor: &HazeRegressor, test_images: &[ndarray::Array3<f32>], test_labels: &[f64], patch_size: usize) -> f64 {
    assert_eq!(test_images.len(), test_labels.len(), "Number of images must match number of labels");

    let mut sum_squared_error = 0.0;
    for (img, &label) in test_images.iter().zip(test_labels.iter()) {
        let score = predict_haze_score(regressor, img, patch_size);
        println!("Predicted score: {:.4}, True label: {:.4}", score, label);
        sum_squared_error += (score - label).powi(2);
    }

    sum_squared_error / test_images.len() as f64
}


#[allow(dead_code)] //OUTMODED / UNUSED - asked AI for examples of a regressor vs a classifier, used a regressor but kept classifier code here for future reference (have had some instances of RustRover losing AI chats before). Very simple, so AI use is fine.
/*
Classify image haze level by thresholding the regression output
Simple binary classification: score > threshold = "High Haze", otherwise "Low Haze"

@param: regressor: trained HazeRegressor
@param: img_3d: input image as 3D array (height x width x 3) with values in [0, 1]
@param: patch_size: patch size for DCP feature extraction (should match training)
@param: threshold: classification threshold (typically 0.5)
@return: "High Haze" or "Low Haze" classification
*/
pub fn classify_haze(
    regressor: &HazeRegressor,
    img_3d: &ndarray::Array3<f32>,
    patch_size: usize,
    threshold: f64,
) -> &'static str {
    let score = predict_haze_score(regressor, img_3d, patch_size);
    if score > threshold {
        "High Haze"
    } else {
        "Low Haze"
    }
}

#[allow(dead_code)] //OUTMODED / UNUSED - asked AI for examples of using a regressor vs a classifier, used the regressor but kept classifier code here for future reference (have had some instances of RustRover losing AI chats before). Very simple, so AI use is fine.
/*
Classify image haze level with detailed output including the raw score
Useful for debugging and understanding model behavior

@param: regressor: trained HazeRegressor
@param: img_3d: input image as 3D array (height x width x 3) with values in [0, 1]
@param: patch_size: patch size for DCP feature extraction (should match training)
@param: threshold: classification threshold (typically 0.5)
@return: tuple of (classification string, raw haze score)
*/
pub fn classify_haze_with_score(
    regressor: &HazeRegressor,
    img_3d: &ndarray::Array3<f32>,
    patch_size: usize,
    threshold: f64,
) -> (&'static str, f64) {
    let score = predict_haze_score(regressor, img_3d, patch_size);
    let classification = if score > threshold { "High Haze" } else { "Low Haze" };
    (classification, score)
}

#[allow(dead_code)]
/*
OUTMODED AFTER BEING REPLACED BY evaluate_mse(), was originally generated by AI when I was asking it to help build a demo.
Evaluate model accuracy on a test set and return the proportion of correctly classified images

@param: regressor: trained HazeRegressor
@param: test_images: slice of test images
@param: test_labels: true labels for test images (0.0 or 1.0)
@param: patch_size: patch size for feature extraction
@param: threshold: classification threshold
@return: accuracy as fraction in [0, 1]
*/
pub fn evaluate_accuracy(
    regressor: &HazeRegressor,
    test_images: &[ndarray::Array3<f32>],
    test_labels: &[f64],
    patch_size: usize,
    threshold: f64,
) -> f64 {
    assert_eq!(test_images.len(), test_labels.len(), "Number of images must match number of labels");

    let mut correct = 0;
    for (img, &label) in test_images.iter().zip(test_labels.iter()) {
        let score = predict_haze_score(regressor, img, patch_size);
        let predicted_class: f64 = if score > threshold { 1.0 } else { 0.0 };
        let true_class: f64 = if label > threshold { 1.0 } else { 0.0 };

        if (predicted_class - true_class).abs() < 0.01 {
            correct += 1;
        }
    }

    correct as f64 / test_images.len() as f64
}