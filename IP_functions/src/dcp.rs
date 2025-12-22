//Implementation of the Dark Channel Prior algorithm as specified in my Project Plan
//TODO: optimization, further testing

use ndarray::{Array2, Array3, Axis}; //linear algebra library is low level enough for this project
use std::collections::VecDeque;

/*
Calculate an image's dark channel by finding the lowest R, G, and B values (separately) in a patch of usize x usize pixels around it
1. Calculate min RGB value for each pixel to convert from 3D array to 2D matrix
2. Apply a 2D min filter to the matrix
    2.1 Use a horizontal 1D min filter to find the minimum across the row segment [x-pad, x+pad]
    2.2 Use a vertical 1D min filter to find the minimum among the row-segment-minimums (calculated in 2.1) across [y-pad, y+pad]

Purpose: low haze is usually signalled by one of three color channels being low-intensity, so if the darkest channel is near zero, low haze, and vice versa. Haze scatters light, so the dark channel will be less dark in images with higher haze
    O(height * width) time complexity helps for optimization vs previous O(height * width * patch_size^2) time complexity

@param: img_3d: input image as a 3D array (height x width x 3) with values in [0, 1]
@param: patch_size: size of patch of surrounding pixels to calculate the haze level of the current pixel for. Larger values will result in a more accurate dark channel, but will take longer to calculate
@return: a matrix of the same size as the input image, with each pixel's value being the lowest R, G, and B values in the patch around it
*/
pub fn find_dark_channel(img_3d: &Array3<f32>, patch_size: usize) -> Array2<f32> {
    let (height, width, _) = img_3d.dim();
    let pad = patch_size / 2;
    let mut dark_channel = Array2::<f32>::zeros((height, width)); //convert to matrix
    let mut dark_channel_rows = Array2::<f32>::zeros((height, width));

    let min_rgb: Array2<f32> = img_3d.map_axis(Axis(2), |rgb| { //optimization via calculate min RGB values for each pixel once, instead of once every time a pixel appears in a patch
        rgb.iter().fold(1.0f32, |a, &b| a.min(b))
    });

    for y in 0..height { //looking only at the pixels on the same row as the current pixel, find the dark channel within the specified patch size
        let row: Vec<f32> = (0..width).map(|x| min_rgb[[y, x]]).collect();
        let filtered: Vec<f32> = min_filter_1d(&row, pad);
        for x in 0..width {
            dark_channel_rows[[y, x]] = filtered[x];
        }
    }

    for x in 0..width { //looking at the pixels on the same column as the current pixel within the matrix of row-segment-minimums, find the dark channel within the specified patch size
        let column: Vec<f32> = (0..height).map(|y| dark_channel_rows[[y, x]]).collect();
        let filtered: Vec<f32> = min_filter_1d(&column, pad);
        for y in 0..height {
            dark_channel[[y, x]] = filtered[y];
        }
    }
    dark_channel
}
//>This felt a lot easier than I expected it to be
//>Decently sure that the above is a bad thing, but perhaps I'm overly pessimistic
//>The previous implementation was too slow to be feasible, so I did, in fact, find that my first implementation was "too easy"


/*
Filters for the minimum in 1D array using a deque (double-ended queue) from index of min value in current patch to index of max value in current patch, and pop all indexes starting from the back whose values >= the value of a newly added index (Van Herk/Gil-Werman algorithm).
Purpose: runs in O(n) time regardless of patch size vs the much slower O(n^2) implementation earlier. Major optimization improvement.
@param: input: 1D array to find minimum in
@param: pad: half the patch size, full patch = 2 * pad + 1)
@return: filtered array
 */
fn min_filter_1d(input: &[f32], pad: usize) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }

    let mut filtered_array = vec![0.0f32; n];
    let mut deque: VecDeque<usize> = VecDeque::new();
    let window_size = 2 * pad + 1;

    for i in 0..(n + pad) {
        while !deque.is_empty() && *deque.front().unwrap() + window_size <= i { //remove indices that have "fallen" outside the patch from the front
            deque.pop_front();
        }

        if i < n { //only add indices to deque if they are within the patch and exist
            while !deque.is_empty() && input[*deque.back().unwrap()] >= input[i] { //remove indices with values >= current value starting from back of deque (they can never be the minimum while current element is in window)
                deque.pop_back();
            }
            deque.push_back(i);
        }

        if i >= pad && (i - pad) < n { //start populating filtered array with values once the first pixel's patch in this direction has been filled
            filtered_array[i - pad] = input[*deque.front().unwrap()];
        }
    }
    filtered_array
}