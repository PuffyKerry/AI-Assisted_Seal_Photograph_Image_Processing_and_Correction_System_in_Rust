//Helper functions for extracting features from images for ML training, AI generated for the large part as it was mostly simple math or calls to functions
//TODO: None.
use ndarray::{Array1, Array2, Array3};
use IP_functions::dcp::find_dark_channel;
use IP_functions::atmospheric::estimate_atmospheric_light;
use IP_functions::transmission::estimated_transmission_map;
use IP_functions::enhance::enhance_clahe;
use IP_functions::brightness::brightness_correct;

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


/// Median dark channel value — more robust than mean when bright sky/specular pixels skew the distribution
pub fn feature_median_dark_channel(dark_channel: &Array2<f32>) -> f64 {
    let mut vals: Vec<f32> = dark_channel.iter().cloned().collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    if n == 0 { return 0.0; }
    if n % 2 == 0 {
        (vals[n / 2 - 1] + vals[n / 2]) as f64 / 2.0
    } else {
        vals[n / 2] as f64
    }
}

/// Variance of dark channel — higher variance means spatially uneven / patchy haze
pub fn feature_variance_dark_channel(dark_channel: &Array2<f32>) -> f64 {
    let mean = feature_mean_dark_channel(dark_channel);
    dark_channel.iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / dark_channel.len() as f64
}

/// Split image into 4 quadrants and return (mean-of-quadrant-means, std-of-quadrant-means).
/// High std indicates spatially lopsided haze (e.g. fog only in the upper-right corner).
pub fn feature_regional_dark_channel_stats(dark_channel: &Array2<f32>) -> (f64, f64) {
    let (h, w) = dark_channel.dim();
    if h < 4 || w < 4 {
        let m = feature_mean_dark_channel(dark_channel);
        return (m, 0.0);
    }
    let mh = h / 2;
    let mw = w / 2;
    // (y_start, y_end, x_start, x_end) for each quadrant
    let quadrants = [(0, mh, 0, mw), (0, mh, mw, w), (mh, h, 0, mw), (mh, h, mw, w)];
    let regional_means: Vec<f64> = quadrants.iter().map(|&(y0, y1, x0, x1)| {
        let count = (y1 - y0) * (x1 - x0);
        if count == 0 { return 0.0; }
        let mut sum = 0.0f64;
        for y in y0..y1 {
            for x in x0..x1 {
                sum += dark_channel[[y, x]] as f64;
            }
        }
        sum / count as f64
    }).collect();
    let mean_of_means = regional_means.iter().sum::<f64>() / 4.0;
    let std_of_means = (regional_means.iter()
        .map(|&m| (m - mean_of_means).powi(2))
        .sum::<f64>() / 4.0)
        .sqrt();
    (mean_of_means, std_of_means)
}

/*
Robust haze label combining global mean, median, and regional distribution statistics.
More accurate proxy for haze severity than raw mean dark channel alone:
  - Median is robust to bright sky patches and specular highlights that inflate the mean
  - Regional std captures patchy/spatially-uneven haze that is perceptually disruptive
  - Regional mean is the mean of quadrant means (roughly equal to global mean but can differ near edges)

Weights are designed so each component stays in [0, 1] and the combined label does too.

@param img_3d: input image as Array3<f32> [H, W, 3] with values in [0, 1]
@param patch_size: dark channel patch size (typically 15)
@return: haze label in [0, 1]
*/
pub fn extract_robust_haze_label(img_3d: &Array3<f32>, patch_size: usize) -> f64 {
    let dark_channel = find_dark_channel(img_3d, patch_size);
    let mean_dc   = feature_mean_dark_channel(&dark_channel);
    let median_dc = feature_median_dark_channel(&dark_channel);
    let (_reg_mean, regional_std) = feature_regional_dark_channel_stats(&dark_channel);

    // regional_std is typically small (0.0 – 0.15); scale by 4 to map into a useful [0, 1] range
    let patchy_haze = (regional_std * 4.0).min(1.0);

    // 40% mean (overall haze level) + 40% median (outlier-robust level) + 20% patchiness
    (0.40 * mean_dc + 0.40 * median_dc + 0.20 * patchy_haze).clamp(0.0, 1.0)
}


/*
Measure local contrast deficit as a label for the CNN's CLAHE output head.
Defined as the mean absolute pixel difference between the raw image and its CLAHE-enhanced
version.  When an image has healthy local contrast, CLAHE barely changes it (small deficit).
When the image is hazy/flat/underexposed, CLAHE has to work hard to enhance it (large deficit).

This is intentionally a DIFFERENT signal from the DCP haze score because of the following reasons:
  - An uniform fog raises both the DCP score and the contrast deficit.
  - An uniform spray or snow would also raise both.
  - An overexposed but otherwise clear image may have low DCP score yet moderate contrast deficit.
  - An image with strong color cast can have high deficit but low DCP score.
Predicting both separately lets us tune DCP params and CLAHE params independently.

CLAHE is run at a moderately aggressive clip (3.0) to amplify the signal.
Raw MAD values for natural images are ~0.01-0.15; scaling by 7 maps this to ~[0, 1].

@param img_3d: input image as Array3<f32> [H, W, 3] with values in [0, 1]
@return: contrast deficit score in [0, 1]
*/
pub fn extract_clahe_contrast_deficit(img_3d: &Array3<f32>) -> f64 {
    let (h, w, c) = img_3d.dim();
    if h == 0 || w == 0 || c == 0 { return 0.0; }

    let enhanced = enhance_clahe(img_3d, 8, 8, 3.0); //moderate clip to amplify signal

    let mut total_diff = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                total_diff += (enhanced[[y, x, ch]] - img_3d[[y, x, ch]]).abs() as f64;
            }
        }
    }
    let mean_diff = total_diff / (h * w * c) as f64;
    (mean_diff * 7.0).clamp(0.0, 1.0) //scale: MAD ~0.01-0.15 → ~[0.07, 1.0]
}


/*
Measure brightness deficit as a label for the CNN's gamma/brightness output head.
Defined as the mean absolute pixel difference between the raw image and its
auto-gamma-corrected version.  When an image is already well-exposed, gamma correction
barely changes it (small deficit).  When the image is underexposed (dark) or overexposed
(washed out), gamma correction has to work harder (large deficit).

This is intentionally a DIFFERENT signal from both DCP and CLAHE:
  - A dark but clear image has low DCP score, low CLAHE deficit, but HIGH brightness deficit.
  - A foggy bright image may have high DCP score but low brightness deficit.
  - An overexposed image has low DCP score, moderate CLAHE deficit, and moderate brightness deficit.
Predicting all three separately lets us tune DCP, CLAHE, and gamma params independently.

Raw MAD values for natural images are ~0.01-0.20; scaling by 5 maps this to ~[0, 1].

@param img_3d: input image as Array3<f32> [H, W, 3] with values in [0, 1]
@return: brightness deficit score in [0, 1]
*/
pub fn extract_brightness_deficit(img_3d: &Array3<f32>) -> f64 {
    let (h, w, c) = img_3d.dim();
    if h == 0 || w == 0 || c == 0 { return 0.0; }

    let corrected = brightness_correct(img_3d, IP_functions::gamma::estimate_gamma(img_3d));

    let mut total_diff = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                total_diff += (corrected[[y, x, ch]] - img_3d[[y, x, ch]]).abs() as f64;
            }
        }
    }
    let mean_diff = total_diff / (h * w * c) as f64;
    (mean_diff * 5.0).clamp(0.0, 1.0) //scale: MAD ~0.01-0.20 → ~[0.05, 1.0]
}


