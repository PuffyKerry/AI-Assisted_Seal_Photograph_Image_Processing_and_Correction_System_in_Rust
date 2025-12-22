//Implementation of transmission map (map of how much light was not affected by the haze) estimation a layer higher than the Dark Channel Prior algorithm, for use in the dehaze function
//TODO: further testing, add dynamic haze retention factor

use ndarray::{Array2, Array3};
use crate::dcp::find_dark_channel;

/*
Estimate the transmission map, which represents how much light from the scene the image was taken reaches the camera without being scattered by haze
1. Normalize the image by the atmospheric light (divide each channel by corresponding the atmospheric light RGB values, aka A value)
    1.1 Call estimate_atmospheric_light to get A value
2. Calculate the dark channel of the normalized image to detect haze
3. Apply the transmission formula: t(x) = 1 - omega * dark_channel_normalized(x)
    Omega keeps a small amount of haze for distant objects to look natural as atmospheric haze always exists to some degree, but is more pronounced for distant objects

Purpose: In the haze model I(x) = J(x)t(x) + A(1-t(x)), t(x) is the transmission
    - t(x) = 1 means no haze, so all scene light reached the camera unaffected
    - t(x) = 0 means complete haze where only atmospheric light reached the camera

@param: img_3d: 3D array of original image
@param: atmospheric: atmospheric light RGB values from estimate_atmospheric_light
@param: patch_size: size of patch for dark channel calculation (should match find_dark_channel)
@param: omega: amount of haze to remove aka the haze retention factor, set to 0.95 to keep 5% of haze for realism, since keeping a small amount of haze for distant objects looks more natural
@return: transmission map as a 2D matrix with values in [0, 1]
*/
pub fn estimated_transmission_map(img_3d: &Array3<f32>, atmospheric: &[f32; 3], patch_size: usize, omega: f32) -> Array2<f32> {
    let (height, width, _) = img_3d.dim();

    let mut normalized = Array3::<f32>::zeros((height, width, 3)); //normalize image by atmospheric light (I/A for each channel)
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                normalized[[y, x, c]] = img_3d[[y, x, c]] / atmospheric[c].max(0.00001); //normalize by dividing by atmospheric light, small epsilon of .max(0.00001) prevents division by zero
            }
        }
    }

    let dark_normalized = find_dark_channel(&normalized, patch_size); //calculate dark channel of normalized image, detection of haze

    //apply transmission formula: t(x) = 1 - omega * dark_channel(I/A)
    dark_normalized.mapv(|val| 1.0 - omega * val)
}
