//Additional filter in the DCP dehazing pipeline that refines the transmission map by removing halos/borders produced by the patch-based dark channel haze detection algorithm
//TODO: optimization, further testing

use ndarray::Array2;

/*
Apply a guided filter to refine the transmission map and remove halo artifacts, which smooths the input while preserving edges from the guide image.
Works by finding local means and variances using box filters and then finding linear coefficients a and b that minimize reconstruction error which are averaged and applied to get the filtered output map

Purpose: The transmission map from estimate_transmission is "blocky" due to the patch-based dark channel calculation, which causes visible halos around objects in the final dehazed image.
    The guided filter aligns the transmission map edges with the actual image edges, producing a significantly cleaner result with "defined" and "deterministic" edges to objects in the photo

@param: guide: grayscale guide image (typically the original image converted to grayscale)
@param: input: the transmission map to be refined
@param: radius: size of current patch for filtering (larger = smoother)
@param: eps: regularization parameter (larger = more smoothing, less edge preservation)
@return: refined transmission map with edges aligned to the guide image
*/
pub fn guided_filter(guide: &Array2<f32>, input: &Array2<f32>, radius: usize, eps: f32) -> Array2<f32> {
    let (height, width) = guide.dim();

    //box filter computes local mean over a (2 * radius + 1) x (2 * radius +1) patch using integral image (summed area table) for O(1) per-pixel lookups, makes the filter O(height * width) instead of O(height * width * radius^2)
    let box_filter = |arr: &Array2<f32>| -> Array2<f32> {
        let mut integral = Array2::<f64>::zeros((height + 1, width + 1)); //build integral image where each cell contains sum of all pixels above and to the left
        for y in 0..height {
            for x in 0..width {
                integral[[y + 1, x + 1]] = arr[[y, x]] as f64
                    + integral[[y, x + 1]]
                    + integral[[y + 1, x]]
                    - integral[[y, x]];
            }
        }

        let mut output = Array2::<f32>::zeros((height, width)); //use integral image to compute box sum in O(1) per pixel
        for y in 0..height {
            for x in 0..width {
                let y0 = y.saturating_sub(radius);
                let y1 = (y + radius + 1).min(height);
                let x0 = x.saturating_sub(radius);
                let x1 = (x + radius + 1).min(width);

                //sum of rectangle using inclusion-exclusion principle
                let sum = integral[[y1, x1]] - integral[[y0, x1]] - integral[[y1, x0]] + integral[[y0, x0]]; //sum all values in the patch
                let count = ((y1 - y0) * (x1 - x0)) as f64;
                output[[y, x]] = (sum / count) as f32; //local mean
            }
        }
        output
    };

    let mean_g = box_filter(guide); //mean of guide calculated with help of box_filter
    let mean_i = box_filter(input); //mean of input
    let correlation_gi = box_filter(&(guide * input)); //mean of guide * input = correlation
    let correlation_gg = box_filter(&(guide * guide)); //mean of guide * guide = correlation

    let covariance_gp = &correlation_gi - &(&mean_g * &mean_i); //covariance of guide and input (p in the paper)
    let variance_g = &correlation_gg - &(&mean_g * &mean_g); //variance of guide

    //output = a * guide + b
    let a = &covariance_gp / (&variance_g + eps); //cov(I,p) / (var(I) + eps)
    let b = &mean_i - &(&a * &mean_g); //mean(p) - a * mean(I)
    let mean_a = box_filter(&a); //average coefficients over all patches containing each pixel
    let mean_b = box_filter(&b);

    &mean_a * guide + &mean_b //apply linear model to get filtered output
}

/*
Convert RGB to grayscale using standard luminance weights
Purpose: The guided filter needs a grayscale guide image, and luminance-based weights makes a grayscale image that matches human perception of brightness levels pretty well

@param: img_3d: RGB image as a 3D array (height x width x 3)
@return: grayscale image as a 2D matrix with values in [0, 1]
*/
pub fn rgb_to_grayscale(img_3d: &ndarray::Array3<f32>) -> Array2<f32> {
    let (height, width, _) = img_3d.dim();
    let mut grayscale = Array2::<f32>::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            grayscale[[y, x]] = 0.299 * img_3d[[y, x, 0]] + 0.587 * img_3d[[y, x, 1]] + 0.114 * img_3d[[y, x, 2]]; //standard luminance weights for sRGB: 0.299R + 0.587G + 0.114B
        }
    }
    grayscale
}