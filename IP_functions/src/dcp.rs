//Implementation of the Dark Channel Prior algorithm as specified in my Project Plan
//TODO: optimization, further testing

use ndarray::{Array2, Array3}; //linear algebra library is low level enough for this project
use std::cmp::min;

/*
Calculate an image's dark channel by finding the lowest R, G, and B values (separately) in a patch of usize x usize pixels around it
Purpose: low haze is usually signalled by one of three color channels being low-intensity, so if the darkest channel is near zero, low haze, and vice versa. Haze scatters light, so the dark channel will be less dark in images with higher haze
@param: patch_size: size of patch of surrounding pixels to calculate the haze level of the current pixel for. Larger values will result in a more accurate dark channel, but will take longer to calculate
@return: a matrix of the same size as the input image, with each pixel's value being the lowest R, G, and B values in the patch around it
*/
pub fn find_dark_channel(img_3d: &Array3<f32>, patch_size: usize) -> Array2<f32> {
    let (height, width, _) = img_3d.dim();
    let pad = patch_size / 2;
    let mut dark_channel = Array2::<f32>::zeros((height, width)); //convert to matrix

    for y in 0..height {
        for x in 0..width {
            let min_y = y.saturating_sub(pad); //safe subtraction to avoid negative padding in any direction
            let max_y = min(y + pad, height - 1);
            let min_x = x.saturating_sub(pad);
            let max_x = min(x + pad, width - 1);
            let mut min_val: f32 = 1.0;

            for i in min_y..=max_y { //"for every pixel in the patch, check if it has a R, G, and/or B value lower than the current lowest"
                for j in min_x..=max_x {
                    let r = img_3d[[i, j, 0]];
                    let g = img_3d[[i, j, 1]];
                    let b = img_3d[[i, j, 2]];

                    let local_min = r.min(g).min(b);
                    if local_min < min_val {
                        min_val = local_min;
                    }
                }
            }
            dark_channel[[y, x]] = min_val;
        }
    }
    dark_channel
}
//>This felt a lot easier than I expected it to be
//>Decently sure that the above is a bad thing, but perhaps I'm overly pessimistic