//Contrast enhancement pipeline and entry point for CLAHE-based image enhancement
//Mirrors dehaze.rs in structure: provides simple entry points that wire up the algorithm
//with sensible defaults, plus a custom-parameters version for fine-tuning
//
//Pipeline flow:
//    Input RGB Image [H, W, 3]  ->  Split into R, G, B channels  ->  CLAHE per channel  ->  Recombine  ->  Output
//
//Note: Applying CLAHE per-channel in RGB space is the simplest approach. A more sophisticated
//version could convert to LAB and only enhance the L channel, but RGB works well for now.

use ndarray::Array3;
use crate::clahe::clahe_channel;


/*
Enhance local contrast using CLAHE with default parameters tuned for seal photographs.
8x8 grid, clip limit 2.5 — brings out fur texture and detail in hazy conditions without
over-amplifying sensor noise.

@param img_3d: input image as Array3<f32> [height, width, 3] with values in [0, 1]
@return: contrast-enhanced image as Array3<f32> [height, width, 3] with values in [0, 1]
*/
pub fn enhance_clahe_default(img_3d: &Array3<f32>) -> Array3<f32> {
    enhance_clahe(img_3d, 8, 8, 2.5)
}


/*
Enhance local contrast using CLAHE with custom parameters.

@param img_3d: input image as Array3<f32> [height, width, 3] with values in [0, 1]
@param grid_h: number of tile rows (more = more local, fewer = more global)
@param grid_w: number of tile columns
@param clip_limit: contrast limit multiplier (1.5–4.0 typical; higher = stronger)
@return: contrast-enhanced image as Array3<f32> [height, width, 3] with values in [0, 1]
*/
pub fn enhance_clahe(img_3d: &Array3<f32>, grid_h: usize, grid_w: usize, clip_limit: f32) -> Array3<f32> {
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

        let enhanced = clahe_channel(&channel, grid_h, grid_w, clip_limit);

        //Write back to output
        for y in 0..height {
            for x in 0..width {
                output[[y, x, c]] = enhanced[[y, x]];
            }
        }
    }

    output
}

