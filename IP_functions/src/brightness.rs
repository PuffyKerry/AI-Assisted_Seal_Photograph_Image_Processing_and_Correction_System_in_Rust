//Brightness correction pipeline and entry point for gamma-based brightness adjustment
//Mirrors dehaze.rs and enhance.rs in structure: simple entry points that wire up the
//algorithm with sensible defaults, plus a custom-parameters version for fine-tuning.
//
//Pipeline flow:
//    Input RGB Image [H, W, 3]  →  Estimate gamma (or use provided)  →  Apply gamma per channel  →  Output
//
//Note: Gamma is applied per-channel in RGB space (same approach as CLAHE in enhance.rs).
//A more sophisticated version could convert to LAB and only adjust L, but per-channel RGB
//gamma is standard practice and works well for uniform brightness correction.

use ndarray::Array3;
use crate::gamma::{gamma_correct_channel, estimate_gamma};


/*
Correct brightness using auto-estimated gamma, tuned for seal photographs.
Estimates the optimal gamma from the image's mean luminance and applies it.

@param img_3d: input image as Array3<f32> [height, width, 3] with values in [0, 1]
@return: brightness-corrected image as Array3<f32> [height, width, 3] with values in [0, 1]
*/
pub fn brightness_correct_default(img_3d: &Array3<f32>) -> Array3<f32> {
    let gamma = estimate_gamma(img_3d);
    brightness_correct(img_3d, gamma)
}


/*
Correct brightness using a specific gamma value.

@param img_3d: input image as Array3<f32> [height, width, 3] with values in [0, 1]
@param gamma: gamma exponent (< 1.0 brightens, > 1.0 darkens, 1.0 = no change)
@return: brightness-corrected image as Array3<f32> [height, width, 3] with values in [0, 1]
*/
pub fn brightness_correct(img_3d: &Array3<f32>, gamma: f32) -> Array3<f32> {
    let (height, width, channels) = img_3d.dim();
    assert_eq!(channels, 3, "Input image must have 3 channels (RGB)");

    let mut output = Array3::<f32>::zeros((height, width, 3));

    for c in 0..3 {
        //Extract single channel as Array2
        let mut channel = ndarray::Array2::<f32>::zeros((height, width));
        for y in 0..height {
            for x in 0..width {
                channel[[y, x]] = img_3d[[y, x, c]];
            }
        }

        let corrected = gamma_correct_channel(&channel, gamma);

        //Write back to output
        for y in 0..height {
            for x in 0..width {
                output[[y, x, c]] = corrected[[y, x]];
            }
        }
    }

    output
}

