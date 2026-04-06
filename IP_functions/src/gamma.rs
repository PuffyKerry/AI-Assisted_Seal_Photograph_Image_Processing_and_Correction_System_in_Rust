//Gamma correction for brightness adjustment
//Addresses the "Lighting" defect category from the thesis: dawn, dusk, overcast, and other
//low-light or indirect lighting conditions that reduce contrast and brightness.
//
//Gamma correction works by applying a power-law transform to each pixel:
//    output = input ^ gamma
//  - gamma < 1.0: brightens the image (lifts shadows, useful for underexposed/dark photos)
//  - gamma = 1.0: no change (identity transform)
//  - gamma > 1.0: darkens the image (compresses highlights, useful for overexposed photos)
//
//This is the simplest useful image correction algorithm — the entire core is one line of
//math per pixel — but it's surprisingly effective for seal photos taken at dawn/dusk or
//under heavy overcast, where the whole image is uniformly too dark.
//
//Q: Why gamma and not just linear brightness scaling (pixel * factor)?
//A: Linear scaling clips highlights quickly (anything above 1.0/factor gets clamped to white).
//   Gamma preserves the full tonal range because it's a curve, not a line — it lifts shadows
//   more than highlights (when gamma < 1), which is exactly what dark seal photos need.
//
//Q: How does this differ from CLAHE?
//A: CLAHE adjusts LOCAL contrast (different tiles get different transforms). Gamma adjusts
//   GLOBAL brightness uniformly. A dark foggy photo might need both: gamma to lift the overall
//   exposure, then CLAHE to bring out local detail in the seal's fur.
//
//Reference: standard sRGB gamma model, any digital imaging textbook.

use ndarray::{Array2, Array3};


/*
Apply gamma correction to a single channel (grayscale) image.

@param channel: input single-channel image as Array2<f32> with values in [0, 1]
@param gamma: gamma exponent (< 1.0 brightens, > 1.0 darkens, 1.0 = no change)
@return: gamma-corrected single-channel image as Array2<f32> in [0, 1]
*/
pub fn gamma_correct_channel(channel: &Array2<f32>, gamma: f32) -> Array2<f32> {
    channel.mapv(|v| v.clamp(0.0, 1.0).powf(gamma))
}


/*
Estimate optimal gamma value from the image's mean luminance.
Targets a "well-exposed" mean luminance of ~0.45 (slightly below mid-gray to
avoid washing out highlights in outdoor photos).

Formula: gamma = ln(target) / ln(mean_luminance)
This is derived from: target = mean_luminance ^ gamma → solve for gamma.

Clamped to [0.3, 2.5] to prevent extreme corrections on near-black or near-white images.

@param img_3d: input image as Array3<f32> [H, W, 3] with values in [0, 1]
@return: estimated gamma value
*/
pub fn estimate_gamma(img_3d: &Array3<f32>) -> f32 {
    let (h, w, _) = img_3d.dim();
    if h == 0 || w == 0 { return 1.0; }

    //Compute mean luminance using standard sRGB weights
    let mut luminance_sum = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            luminance_sum += (0.299 * img_3d[[y, x, 0]] as f64)
                + (0.587 * img_3d[[y, x, 1]] as f64)
                + (0.114 * img_3d[[y, x, 2]] as f64);
        }
    }
    let mean_lum = (luminance_sum / (h * w) as f64) as f32;

    //Avoid log(0) or log(1) edge cases
    if mean_lum < 0.01 { return 0.5; }  //very dark → moderate brightening (was 0.3, too extreme)
    if mean_lum > 0.95 { return 2.0; }  //very bright → moderate darkening (was 2.5, too extreme)
    if (mean_lum - 0.45).abs() < 0.08 { return 1.0; } //already well-exposed (wider tolerance)

    let target = 0.45f32;
    let gamma = target.ln() / mean_lum.ln();
    gamma.clamp(0.5, 2.0)
}

