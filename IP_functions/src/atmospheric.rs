//Implementation of Atmospheric Light Estimation as specified in my Project Plan
//TODO: optimization, further testing

use ndarray::{Array2, Array3};

/*
Estimate the the color of the haze in the image, aka the atmospheric light.
1. Find the top brightest pixels in the dark channel (top 0.1% by default)
2. Among those candidate positions, find the pixel in the original image with the highest intensity
3. Return that pixel's RGB values as the atmospheric light

Purpose: The atmospheric light represents the color of the haze. In the physical haze model,
    I(x) = J(x)t(x) + A(1-t(x)), we need A (atmospheric light) to recover the scene radiance J(x).
    The brightest pixels in the dark channel typically correspond to the haziest regions (sky, distant objects),
    which best represent the pure haze color.

@param: img_3d: the original image as a 3D array (height x width x 3)
@param: dark_channel: the dark channel matrix calculated from find_dark_channel
@param: top_percent: fraction of brightest pixels to consider (e.g., 0.001 = top 0.1%)
@return: RGB values of the atmospheric light as [R, G, B] array
*/
pub fn estimate_atmospheric_light(
    img_3d: &Array3<f32>,
    dark_channel: &Array2<f32>,
    top_percent: f32,
) -> [f32; 3] {
    let (height, width, _) = img_3d.dim();
    let num_pixels = height * width;
    let num_candidates = ((num_pixels as f32) * top_percent).max(1.0) as usize; //ensure at least 1 candidate is drawn from the top_percent brightest pixels
    
    let mut indexed_pixels: Vec<(usize, f32)> = dark_channel //flatten dark channel and pair values to their indexes for sorting purposes
        .iter()
        .enumerate()
        .map(|(i, &val)| (i, val))
        .collect();

    indexed_pixels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); //sort dark channels from brightest to lowest (descending order)
    
    let mut best_intensity = 0.0f32;
    let mut best_pixel = [1.0f32; 3]; //default to white if no pixel found

    for (flat_index, _) in indexed_pixels.into_iter().take(num_candidates) { //find the pixel with highest intensity in original image out of the highest top_percent
        let y = flat_index / width; //convert flat index back to 2D coordinates
        let x = flat_index % width;
        
        let r = img_3d[[y, x, 0]]; //intensity as max RGB value
        let g = img_3d[[y, x, 1]];
        let b = img_3d[[y, x, 2]];
        let intensity = r.max(g).max(b);

        if intensity > best_intensity {
            best_intensity = intensity;
            best_pixel = [r, g, b];
        }
    }
    best_pixel
}