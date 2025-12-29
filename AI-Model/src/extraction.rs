//Helper functions for extracting features from images for ML training, AI generated for the large part as it was mostly simple math or calls to functions
//TODO: None.
use ndarray::{Array1, Array2, Array3};
use IP_functions::dcp::find_dark_channel;
use IP_functions::atmospheric::estimate_atmospheric_light;
use IP_functions::transmission::estimated_transmission_map;

/// Mean dark channel value - higher indicates more haze
pub fn feature_mean_dark_channel(dark_channel: &Array2<f32>) -> f64 {
    dark_channel.iter().map(|&v| v as f64).sum::<f64>() / dark_channel.len() as f64
}

/// Mean transmission - lower indicates more haze
pub fn feature_mean_transmission(transmission: &Array2<f32>) -> f64 {
    transmission.iter().map(|&v| v as f64).sum::<f64>() / transmission.len() as f64
}

/// Standard deviation of transmission - texture/depth variation indicator
pub fn feature_std_transmission(transmission: &Array2<f32>) -> f64 {
    let mean = feature_mean_transmission(transmission);
    let variance: f64 = transmission.iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>() / transmission.len() as f64;
    variance.sqrt()
}

/// Maximum atmospheric light intensity
pub fn feature_atmospheric_intensity(atmospheric: &[f32; 3]) -> f64 {
    atmospheric[0].max(atmospheric[1]).max(atmospheric[2]) as f64
}

/// Ratio of pixels with low transmission (< threshold)
pub fn feature_low_transmission_ratio(transmission: &Array2<f32>, threshold: f32) -> f64 {
    transmission.iter()
        .filter(|&&v| v < threshold)
        .count() as f64 / transmission.len() as f64
}

/// Combine all features into a single vector for regression
pub fn extract_all_features(img_3d: &Array3<f32>, patch_size: usize) -> Array1<f64> {
    let dark_channel = find_dark_channel(img_3d, patch_size);
    let atmospheric = estimate_atmospheric_light(img_3d, &dark_channel, 0.001);
    let transmission = estimated_transmission_map(img_3d, &atmospheric, patch_size, 0.95);

    Array1::from_vec(vec![
        feature_mean_dark_channel(&dark_channel),
        feature_mean_transmission(&transmission),
        feature_std_transmission(&transmission),
        feature_atmospheric_intensity(&atmospheric),
        feature_low_transmission_ratio(&transmission, 0.5),
    ])
}

//Realized that extraction often didn't need more than the mean dark channel value, more optimal.
pub fn extract_mean_dark_channel(img_3d: &Array3<f32>, patch_size: usize) -> f64 {
    feature_mean_dark_channel(&find_dark_channel(img_3d, patch_size))
}