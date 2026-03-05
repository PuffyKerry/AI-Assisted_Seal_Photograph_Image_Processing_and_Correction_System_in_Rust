//Implementation of Contrast Limited Adaptive Histogram Equalization (CLAHE) for local contrast enhancement
//Particularly useful for seal photographs taken in hazy, foggy, or underwater conditions where
//contrast is lost across the image non-uniformly (e.g. a seal on a rock in fog where the seal
//is darker and the background is washed out)
//
//CLAHE works by:
//  1. Dividing the image into tiles (grid of sub-regions)
//  2. Computing a histogram for each tile
//  3. Clipping the histogram at a contrast limit to prevent over-amplification of noise
//  4. Redistributing clipped counts evenly across all bins
//  5. Building a CDF (cumulative distribution function) lookup table per tile
//  6. Bilinearly interpolating between neighboring tiles' CDFs for smooth output
//
//Reference: Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization." (1994)
//
//Q: Why CLAHE over regular histogram equalization?
//A: Regular HE applies one global transform — great for simple cases, but it over-amplifies
//   noise in flat regions and can blow out already-bright areas. CLAHE limits amplification
//   per-tile so you get local contrast improvement without destroying global tonality.
//
//Q: Why is this useful for seal photos specifically?
//A: Seal photos often have a dark subject (seal) against a bright hazy background. Global
//   contrast stretch would either clip the seal to black or the sky to white. CLAHE adapts
//   locally so the seal's fur texture is enhanced independently of the background.

use ndarray::Array2;

const NUM_BINS: usize = 256; //standard 8-bit histogram


/*
Core CLAHE algorithm operating on a single-channel (grayscale) image represented as f32 values in [0, 1].

Pipeline:
    Input grayscale [0,1]
        │
        ▼
    Divide into grid_h × grid_w tiles
        │
        ▼
    Per tile: build histogram → clip at limit → redistribute → build CDF LUT
        │
        ▼
    For each pixel: bilinear interpolation of 4 nearest tile LUTs
        │
        ▼
    Output enhanced grayscale [0,1]

@param channel: input single-channel image as Array2<f32> with values in [0, 1]
@param grid_h: number of tile rows (typically 8)
@param grid_w: number of tile columns (typically 8)
@param clip_limit: contrast limit as a multiplier of the average bin count per tile
                   (typically 2.0–4.0; higher = more contrast, lower = gentler)
@return: contrast-enhanced single-channel image as Array2<f32> in [0, 1]
*/
pub fn clahe_channel(
    channel: &Array2<f32>,
    grid_h: usize,
    grid_w: usize,
    clip_limit: f32,
) -> Array2<f32> {
    let (height, width) = channel.dim();

    //Tile dimensions (last tile may be slightly larger due to rounding)
    let tile_h = height / grid_h;
    let tile_w = width / grid_w;

    //Edge case: image smaller than the grid
    if tile_h == 0 || tile_w == 0 {
        //Fall back to global histogram equalization for tiny images
        return global_histogram_equalize(channel);
    }

    // ------ Step 1: Build per-tile CDF lookup tables ------
    //luts[row][col] is a 256-entry LUT mapping input bin → output value in [0, 1]
    let mut luts: Vec<Vec<[f32; NUM_BINS]>> = Vec::with_capacity(grid_h);

    for ty in 0..grid_h {
        let mut row_luts: Vec<[f32; NUM_BINS]> = Vec::with_capacity(grid_w);

        for tx in 0..grid_w {
            //Tile bounds — last tile in each dimension extends to the image edge
            let y0 = ty * tile_h;
            let y1 = if ty == grid_h - 1 { height } else { y0 + tile_h };
            let x0 = tx * tile_w;
            let x1 = if tx == grid_w - 1 { width } else { x0 + tile_w };

            let tile_pixels = (y1 - y0) * (x1 - x0);

            // --- Histogram ---
            let mut hist = [0u32; NUM_BINS];
            for y in y0..y1 {
                for x in x0..x1 {
                    let bin = (channel[[y, x]] * 255.0).clamp(0.0, 255.0) as usize;
                    let bin = bin.min(NUM_BINS - 1);
                    hist[bin] += 1;
                }
            }

            // --- Clip histogram and redistribute ---
            let avg_count = tile_pixels as f32 / NUM_BINS as f32;
            let limit = (clip_limit * avg_count).max(1.0) as u32;

            let mut excess = 0u32;
            for bin in hist.iter_mut() {
                if *bin > limit {
                    excess += *bin - limit;
                    *bin = limit;
                }
            }

            //Spread excess evenly across all bins
            let per_bin = excess / NUM_BINS as u32;
            let remainder = (excess % NUM_BINS as u32) as usize;
            for (i, bin) in hist.iter_mut().enumerate() {
                *bin += per_bin;
                if i < remainder {
                    *bin += 1; //distribute leftover 1 each to first `remainder` bins
                }
            }

            // --- Build CDF lookup table ---
            let mut cdf = [0f32; NUM_BINS];
            let mut cumulative = 0u32;
            for i in 0..NUM_BINS {
                cumulative += hist[i];
                cdf[i] = cumulative as f32 / tile_pixels as f32;
            }

            //Clamp to [0, 1] just in case of floating point drift
            for val in cdf.iter_mut() {
                *val = val.clamp(0.0, 1.0);
            }

            row_luts.push(cdf);
        }

        luts.push(row_luts);
    }

    // ------ Step 2: Bilinear interpolation of tile LUTs for each pixel ------
    let mut output = Array2::<f32>::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let bin = (channel[[y, x]] * 255.0).clamp(0.0, 255.0) as usize;
            let bin = bin.min(NUM_BINS - 1);

            //Find which tile center this pixel is closest to
            //Tile centers are at (ty * tile_h + tile_h/2, tx * tile_w + tile_w/2)
            //We want floating-point tile coordinates for interpolation
            let fy = (y as f32 - tile_h as f32 / 2.0) / tile_h as f32;
            let fx = (x as f32 - tile_w as f32 / 2.0) / tile_w as f32;

            //Clamp to valid tile index range
            let fy = fy.clamp(0.0, (grid_h - 1) as f32);
            let fx = fx.clamp(0.0, (grid_w - 1) as f32);

            let ty0 = (fy as usize).min(grid_h - 1);
            let tx0 = (fx as usize).min(grid_w - 1);
            let ty1 = (ty0 + 1).min(grid_h - 1);
            let tx1 = (tx0 + 1).min(grid_w - 1);

            let dy = fy - ty0 as f32; //fractional part for interpolation weight
            let dx = fx - tx0 as f32;

            //Look up the CDF value from each of the 4 neighboring tiles
            let val_tl = luts[ty0][tx0][bin]; //top-left
            let val_tr = luts[ty0][tx1][bin]; //top-right
            let val_bl = luts[ty1][tx0][bin]; //bottom-left
            let val_br = luts[ty1][tx1][bin]; //bottom-right

            //Bilinear interpolation: blend horizontally then vertically
            let top = val_tl * (1.0 - dx) + val_tr * dx;
            let bot = val_bl * (1.0 - dx) + val_br * dx;
            let val = top * (1.0 - dy) + bot * dy;

            output[[y, x]] = val.clamp(0.0, 1.0);
        }
    }

    output
}


/*
Fallback: simple global histogram equalization for images too small for tiled CLAHE.

@param channel: single-channel image as Array2<f32> in [0, 1]
@return: equalized image as Array2<f32> in [0, 1]
*/
fn global_histogram_equalize(channel: &Array2<f32>) -> Array2<f32> {
    let (height, width) = channel.dim();
    let total = (height * width) as f32;

    let mut hist = [0u32; NUM_BINS];
    for &val in channel.iter() {
        let bin = (val * 255.0).clamp(0.0, 255.0) as usize;
        let bin = bin.min(NUM_BINS - 1);
        hist[bin] += 1;
    }

    //Build CDF
    let mut cdf = [0f32; NUM_BINS];
    let mut cumulative = 0u32;
    for i in 0..NUM_BINS {
        cumulative += hist[i];
        cdf[i] = cumulative as f32 / total;
    }

    //Apply
    let mut output = Array2::<f32>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let bin = (channel[[y, x]] * 255.0).clamp(0.0, 255.0) as usize;
            let bin = bin.min(NUM_BINS - 1);
            output[[y, x]] = cdf[bin];
        }
    }
    output
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clahe_output_range() {
        //Verify output stays in [0, 1]
        let input = Array2::from_shape_fn((64, 64), |(y, x)| {
            ((y * 64 + x) as f32) / (64.0 * 64.0)
        });
        let output = clahe_channel(&input, 4, 4, 2.0);

        for &val in output.iter() {
            assert!(val >= 0.0 && val <= 1.0, "Output pixel {} out of range", val);
        }
    }

    #[test]
    fn test_clahe_preserves_dimensions() {
        let input = Array2::<f32>::zeros((100, 80));
        let output = clahe_channel(&input, 8, 8, 3.0);
        assert_eq!(output.dim(), (100, 80));
    }

    #[test]
    fn test_clahe_tiny_image_fallback() {
        //Image smaller than grid should not panic
        let input = Array2::from_elem((4, 4), 0.5f32);
        let output = clahe_channel(&input, 8, 8, 2.0);
        assert_eq!(output.dim(), (4, 4));
    }

    #[test]
    fn test_global_he_uniform() {
        //Uniform image should stay roughly uniform after global HE
        let input = Array2::from_elem((32, 32), 0.5f32);
        let output = global_histogram_equalize(&input);
        //All pixels map to same bin → CDF jumps to 1.0 at that bin
        for &val in output.iter() {
            assert!((val - 1.0).abs() < 0.01, "Uniform image should map to ~1.0 CDF, got {}", val);
        }
    }
}

