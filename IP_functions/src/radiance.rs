//Last stage of DCP-based dehazing pipeline: recover the haze-free scene radiance
//TODO: optimization, further testing

use ndarray::{Array2, Array3};

/*
Recover the haze-free scene radiance (the dehazed image), uses the output of previous steps to reconstruct the photo with less haze.
For each pixel, apply the radiance recovery formula: J(x) = (I(x) - A) / max(t(x), t0) + A ; where:
    - I(x) is the observed hazy pixel value
    - A is the atmospheric light
    - t(x) is the transmission at pixel x
    - t0 is a lower bound on transmission to prevent noise in thick haze

Purpose: Final step using the inverted haze model to recover the original scene. Rearranges I(x) = J(x)t(x) + A(1-t(x)) to solve for J(x), t0 is a lower bound on transmission to prevent division by very small numbers, which would cause noise/artifacts in heavily hazed regions.
    Applies the calculated values of previous steps to the original image.

@param: img_3d: original image as a 3D array
@param: transmission: the transmission map from estimate_transmission
@param: atmospheric: the atmospheric light RGB values from estimate_atmospheric_light
@param: t0: lower bound for transmission to prevent noise (typically 0.1)
@return: dehazed image as a 3D array
*/
pub fn recover_scene_radiance(img_3d: &Array3<f32>, transmission: &Array2<f32>, atmospheric: &[f32; 3], t0: f32) -> Array3<f32> {
    let (height, width, _) = img_3d.dim();
    let mut output = Array3::<f32>::zeros((height, width, 3));

    for y in 0..height {
        for x in 0..width {
            let t = transmission[[y, x]].max(t0); //clamp transmission to t0 to prevent division by very small values

            for c in 0..3 {
                let recovered = (img_3d[[y, x, c]] - atmospheric[c]) / t + atmospheric[c]; //apply recovery formula: J(x) = (I(x) - A) / t + A
                output[[y, x, c]] = recovered.clamp(0.0, 1.0); //clamp to valid range [0, 1] to prevent out-of-bounds pixel values
            }
        }
    }
    output
}