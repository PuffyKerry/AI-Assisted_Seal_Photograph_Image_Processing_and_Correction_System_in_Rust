use ndarray::{Array2, Array3};
use std::cmp::min;

pub fn find_dark_channel(image_bytes: &Array3<f32>, patch_size: usize) -> Array2<f32> {
    let (height, width, _) = image_bytes.dim();
    let pad = patch_size / 2;

    let mut dark_channel = Array2::<f32>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let min_y = y.saturating_sub(pad);
            let max_y = min(y + pad, height - 1);
            let min_x = x.saturating_sub(pad);
            let max_x = min(x + pad, width - 1);
            let mut min_val: f32 = 1.0;

            for j in min_y..=max_y {
                for i in min_x..=max_x {
                    let r = image_bytes[[j, i, 0]];
                    let g = image_bytes[[j, i, 1]];
                    let b = image_bytes[[j, i, 2]];

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