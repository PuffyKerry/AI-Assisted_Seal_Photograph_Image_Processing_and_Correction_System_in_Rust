//Contrast enhancement pipeline and entry point for CLAHE-based image enhancement
//Mirrors dehaze.rs in structure: provides simple entry points that wire up the algorithm
//with sensible defaults, plus a custom-parameters version for fine-tuning
//
//Pipeline flow (luminance-preserving):
//    Input RGB [H,W,3]  →  Compute luminance  →  CLAHE on luminance  →  Scale RGB by lum ratio  →  Output
//
//IMPORTANT: CLAHE is applied ONLY to the luminance channel, NOT per-channel RGB.
//Per-channel RGB CLAHE independently stretches R, G, B histograms which destroys color
//balance and produces psychedelic color shifts ("acid trip" effect). Luminance-only CLAHE
//enhances contrast while preserving the original color ratios — exactly what we want for
//bringing out seal fur texture without turning the photo into modern art.

use ndarray::{Array2, Array3};
use crate::clahe::clahe_channel;


/*
Enhance local contrast using CLAHE with default parameters tuned for seal photographs.
8x8 grid, clip limit 2.0 — brings out fur texture and detail in hazy conditions without
over-amplifying sensor noise or shifting colors.

@param img_3d: input image as Array3<f32> [height, width, 3] with values in [0, 1]
@return: contrast-enhanced image as Array3<f32> [height, width, 3] with values in [0, 1]
*/
pub fn enhance_clahe_default(img_3d: &Array3<f32>) -> Array3<f32> {
    enhance_clahe(img_3d, 8, 8, 2.0)
}


/*
Enhance local contrast using luminance-only CLAHE with custom parameters.

Applies CLAHE to the luminance channel only, then scales each RGB channel by the ratio
of (new_luminance / old_luminance) to preserve original color balance. This prevents
the per-channel histogram equalization from creating false colors.

@param img_3d: input image as Array3<f32> [height, width, 3] with values in [0, 1]
@param grid_h: number of tile rows (more = more local, fewer = more global)
@param grid_w: number of tile columns
@param clip_limit: contrast limit multiplier (1.0–2.5 typical; higher = stronger)
@return: contrast-enhanced image as Array3<f32> [height, width, 3] with values in [0, 1]
*/
pub fn enhance_clahe(img_3d: &Array3<f32>, grid_h: usize, grid_w: usize, clip_limit: f32) -> Array3<f32> {
    let (height, width, channels) = img_3d.dim();
    assert_eq!(channels, 3, "Input image must have 3 channels (RGB)");

    // Step 1: Compute luminance channel using standard sRGB weights
    let mut luminance = Array2::<f32>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let r = img_3d[[y, x, 0]];
            let g = img_3d[[y, x, 1]];
            let b = img_3d[[y, x, 2]];
            luminance[[y, x]] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }

    // Step 2: Apply CLAHE to the luminance channel only
    let enhanced_lum = clahe_channel(&luminance, grid_h, grid_w, clip_limit);

    // Step 3: Scale original RGB by the ratio (new_lum / old_lum) to preserve color balance
    // This is the key trick: instead of touching R, G, B independently, we just adjust
    // overall brightness per-pixel while keeping the color ratios intact.
    let mut output = Array3::<f32>::zeros((height, width, 3));
    for y in 0..height {
        for x in 0..width {
            let old_lum = luminance[[y, x]];
            let new_lum = enhanced_lum[[y, x]];

            // Avoid division by zero for pure-black pixels
            let scale = if old_lum > 1e-6 {
                new_lum / old_lum
            } else {
                // For near-black pixels, use additive offset instead of scaling
                // (scaling from 0 would be infinite)
                1.0
            };

            for c in 0..3 {
                let val = img_3d[[y, x, c]] * scale;
                output[[y, x, c]] = val.clamp(0.0, 1.0);
            }
        }
    }

    output
}

