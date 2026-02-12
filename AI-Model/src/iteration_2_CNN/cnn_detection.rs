//Iteration 2 WIP: CNN for haze detection, STILL SOMEWHAT WIP (2/11), see below, needs some refactoring for optimization, decent amount of AI-generated code for boilerplate and basic functions but also in some less-critical areas like the suggest_dcp_parameters function and parts of the training pipeline, will need to be cleaned up for finished Iteration 2.
//Replaces linear regression regressor with a Convolutional Neural Network implemented with the burn framework for improved accuracy
//Purpose: self-contained module for all CNN logic to avoid major changes to other files bc encapsulation makes the iterative design process more understandable
//
//==============================================================================
//                    GPU TRAINING DEBUGGING JOURNEY (2/11/2026)
//==============================================================================
//
//  PURPOSE: need to switch to training CNN's on a GPU for performance reasons, as otherwise evaluating the accuracy of a large model trained on the full dataset will be implausibly time consuming on a CPU-only device.

//  STATUS: GPU training hangs at first epoch during backward pass. Issue is NOT RESOLVED. See ~line 1006 area where loss.backward() gets stuck. Please review the code and help if you have the time to do so.
//
//  TLDR: "It's slow, lemme add more granular progress updates" -> "GPU not used? I should check Task Manager and print out the GPU used by wgpu" -> "GPU detected but wrong one?" ->
//        "Batching issues? " -> "Shader compilation? Try precompiling? Not the issue, commented out" -> "Image size problem perhaps? Start testing" ->
//        "256x256 works, 344 X 484 (small end of images in dataset, rare batch) hangs, 512x512 hangs no matter how many image sizes are tested so it isn't a group number issue or image ratios or height/width being fixed to powers of 2" ->
//        "backward() hangs based on debugging with RustRover's debugger" -> "force syncing GPU isn't working, debugger isn't helping" -> "push to GitHub to update on progress and ask for help while still debugging"
//
//  The Long Version (everything I tried over the last two weeks):

//  1. Granular Printing
    //  1.1 - Thought training was just slow so I added epoch printing to monitor progress, but nothing was printing after "Epoch 1/50 starting...", which was concerning when waiting  20+ minutes to quite literally... 6. LITERAL. HOURS.

//  2. GPU Detection
    //  2.1 - Maybe the GPU isn't being used?? Opened Task Manager and saw GPU spike to 90% during image loading then drop to 1-2% during actual training which led to the below...
    //  2.2 - Added print statements to show which GPU was detected, but wgpu sees my RTX 3070 just fine via Vulkan backend
    //  2.3 - Tried to make absolutely sure I was using the right GPU and not my cpu or something, added WgpuDevice::DiscreteGpu(0) and GPU diagnostics that enumerate all available adapters

//  3. Backend Testing, Batching Implementation and Testing
    //  3.1 - Tried different backends and added batches before trying different batch sizes, thought maybe processing one image at a time was the issue so implemented dimension-grouped batching where images of same size get batched together
    //  3.2 - Thought about and rejected idea of resizing all images to one size for batch efficiency because I didn't want to lose image quality/detail and also that felt like giving up on the actual problem. Jack proposed doing this anyways or some other method to work around the issue if I still can't make any headway.
    //  3.3 - Tested more different backends trying to make sure wgpu is actually working, added WGPU_BACKEND="vulkan" environment variable and checked that the Vulkan adapter was being selected properly

//  4. Training and GPU Connection Diagnostics Monitoring
    //  4.1 - Was still stuck so I added super granular progress updates like "Group 1/116 | Batch 1/1 | 0/430 (0.0%) | Starting..." but the program stayed stuck at the first batch of the first group for hours.
    //  4.2 - Added GPU warmup diagnostics and timing for shader compilation, prints "Requesting DiscreteGpu(0) - this should be your RTX 3070" and shows forward/backward pass compilation times, issue seems to be that something is stuck during shader precompiling.

//  5. Image Size Testing and Debugging
    //  5.1 - Warmup tests showed small sizes work fine (32x32, 64x64, 128x128 all compile and run in <3 seconds each)
    //  5.2 - Found it might be a size issue after testing for:
    //          5.2.1 - Number of test sizes for compilation (more sizes = more shader variants = more compilation time, but that's not the issue it seems)
    //          5.2.2 - Test size ratios (non-square images like 344x484)
    //          5.2.3 - Test size power-of-2 (256x256 works, 512x512 does not)
    //
    //  5.3 - Tried to debug into this mess and found that the issue is at loss.backward() around ~line 1006 (give or take a few), forward pass completes fine but backward pass just... never returns. Tried adding breakpoints but you can't step into compiled crate code (burn/wgpu internals), looked through backward pass code but couldn't see anything obvious like size limits for example.
    //
    //  5.4 - Looked at "force syncing" as a possible solution to memory errors related to accessing a value that hasn't been computed yet when the images are too large to comput in time:
    //          - GPU operations are async so backward() returns immediately but actual computation happens in background
    //          - Tried GradientsParams::from_grads(grads, &model) to force sync by accessing gradient data
    //          - Tried drop(grad_params) to trigger cleanup
    //          - None of this helps because the GPU work itself never completes
    //          - Gradients type doesn't implement .clone() so can't force sync that way, probably not the issue
//
//  CURRENT STATE: 256x256 works perfectly, anything larger (including 512x512 which is also a power of 2) hangs indefinitely on backward pass. This is possibly a wgpu/driver issue with shader compilation for larger convolution kernels but more testing is needed.
//
//  WORKAROUND: Could resize all images to 256x256 or some other smaller size before training (rejected in 3.2) but this would reduce training accuracy for larger images, may be necessary if no other solution is found. May try scaling the entire dataset against the largest image in the dataset normalized against the largest image size accepted?
//
//  If you have any ideas please let me know, I will be working on this in the meantime.
//
//==============================================================================
//
//Architecture:
//  - Input: Variable-sized image from load_images_parallel (original/4 resolution)
//  - Conv2D layers with ReLU activations for spatial feature extraction
//      - Conv2D is the basic building block of CNNs, detects edges and basic textures
//      - Strided convolutions progressively focus on larger patterns
//      - ReLU means rectified linear unit, prevents negative values which can cause gradient vanishing (where a model's earlier layer learns too slowly due to gradients nearing 0 in backpropagation)
//  - Global Average Pooling to handle variable input sizes (key innovation over fixed-size CNNs, converts each feature map to a single average value across all spatial positions)
//  - Fully connected layers for regression output
//  - Output: Single continuous haze score [0.0, 1.0]
//
//TODO: Expand suggested DCP parameters feature beyond current simple heuristic
//TODO: MAJOR OPTIMIZATION FOR MORE THAN PROOF OF CONCEPT TESTING
//TODO: most testing really, but will need to set up environment on a device with more processing power for that.
//MEGA-TODO: clean up this mess of nearly 700 lines of code

///Q: What's a CNN?
///A: A Convolutional Neural Network (CNN) is a type of neural network that uses convolutional layers to extract features from input data. It works by limiting each neuron on layers after the first to only a set bound of spatially close neurons, effectively reducing the number of parameters and computation required to learn the model. For image processing, this has the added advantage of being able to learn spatial hierarchies of features (edges, textures, patterns) that simple linear regression on extracted features cannot capture.
///Q: Why a CNN?
///A: CNN's are uniquely suited to image processing as they can learn spatial hierarchies of features (edges, textures, patterns) that simple linear regression on extracted features cannot capture, and unlike other neural networks have translational invariance that helps with recognizing patterns and shapes at different orientations/locations in the image. This is due to
///Q: Why burn and not candle?
///A: Besides burn being supported by more backends (ndarray, wgpu, LibTorch), burn is significantly safer due to leveraging Rust's typing and requiring generics and trait definitions, meaning that compile-time errors from incorrectly shaped tensors are rarer.
///   Candle is more dynamic and Python-like, which is easier to use but more error-prone and less performant due to runtime checks and dynamic dispatch, which are supposed to be the issues resolved by switching to Rust from Python.
///Q: Why are parameters and return elements now listed vertically? And why did the function descriptions get more organized/more detailed?
///A: I realized that my previous habits (e.g. all parameters on one line, comments on same line as first line of code in chunk) were somewhat detrimental to readability, so I decided to go with the recommended format from Rust documentation guidelines for better clarity.

use burn::{
    prelude::*,
    tensor::{
        Tensor,
        activation::sigmoid,
        backend::AutodiffBackend
    },
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Linear, LinearConfig, Relu, Dropout, DropoutConfig, PaddingConfig2d,
    },
    optim::{AdamConfig, Optimizer, GradientsParams},
    backend::ndarray::{
        NdArray,
        NdArrayDevice::Cpu
    },
    backend::wgpu::{Wgpu, WgpuDevice},
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use ndarray::Array3; //backend linear algebra library for initial testing
use std::{
    path::{Path, PathBuf},
    fs,
    collections::HashMap,
    io::{self, Write},
    time::Instant,
};
use wgpu; //for GPU diagnostics

pub const CNN_INPUT_CHANNELS: usize = 3; //RGB image input


/*
CNN Model for haze detection/regression.
Uses 4 convolutional layers with strided (half each dimension) convolutions for downsampling, followed by global average pooling to handle variable input sizes, then fully connected layers for regression.

Architecture visual:
    Image (H x W x 3) -> Conv1 (H/2 x W/2 x 16) -> Conv2 (H/4 x W/4 x 32) -> Conv3 (H/8 x W/8 x 64) -> Conv4 (H/16 x W/16 x 128)
    -> Global Avg Pool (1 x 1 x 128)
    -> FC1 (64) -> FC2 (1)
    -> Sigmoid (shifts haze score prediction to (0, 1) range)
    -> Haze Score (0,1)

Purpose: CNNs can learn spatial features (haze patterns, texture degradation) that simple linear regression on extracted features cannot capture. The strided convolutions progressively focus on larger patterns, and global average pooling allows any input size.
*/
#[derive(Module, Debug)]
pub struct HazeCNN<B: Backend> {
    conv1: Conv2d<B>,       //3 -> 16 channels, detects edges and basic textures
    conv2: Conv2d<B>,       //16 -> 32 channels, detects patterns made of edges
    conv3: Conv2d<B>,       //32 -> 64 channels, detects complex features like haze gradients
    conv4: Conv2d<B>,       //64 -> 128 channels, high-level haze characteristics
    gap: AdaptiveAvgPool2d, //global average pooling, key for variable input sizes
    fc1: Linear<B>,         //128 -> 64 features
    fc2: Linear<B>,         //64 -> 1 (final haze score)
    dropout: Dropout,       //regularization to prevent overfitting
    activation: Relu,       //ReLU activation between layers
}


/*
Configuration for CNN hyperparameters (architecture of the model).
Default values are reasonable starting points recommended by AI, can be tuned based on dataset characteristics manually or I can add this functionality later.

@field conv1_channels: output channels for first conv layer (default 16)
@field conv2_channels: output channels for second conv layer (default 32)
@field conv3_channels: output channels for third conv layer (default 64)
@field conv4_channels: output channels for fourth conv layer (default 128)
@field fc1_size: size of first fully connected layer (default 64)
@field dropout_rate: dropout probability for regularization (default 0.3)
*/
#[derive(Config, Debug)]
pub struct HazeCNNConfig { //defaults were AI-generated
    #[config(default = "16")]
    conv1_channels: usize,
    #[config(default = "32")]
    conv2_channels: usize,
    #[config(default = "64")]
    conv3_channels: usize,
    #[config(default = "128")]
    conv4_channels: usize,
    #[config(default = "64")]
    fc1_size: usize,
    #[config(default = "0.3")]
    dropout_rate: f64,
}

impl HazeCNNConfig {
    /*
    Initialize the CNN model with random weights based on config parameters.
    All conv(olutional) layers use 3x3 kernels with same padding and stride 2 for downsampling. AKA, all neurons on convolutional layers 2, 3, and 4 only use a 3x3 of neurons from the previous layer to compute their value, and the output spatial dimensions are halved each time due to stride 2.

    @param device: the burn backend device (CPU or GPU) to allocate tensors on
    @return: initialized HazeCNN model ready for training
    */
    pub fn init<B: Backend>(&self, device: &B::Device) -> HazeCNN<B> {
        //Conv layers: 3x3 kernels with same padding, stride 2 halves dimensions each layer
        let conv1 = Conv2dConfig::new([CNN_INPUT_CHANNELS, self.conv1_channels], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_stride([2, 2])
            .init(device);

        let conv2 = Conv2dConfig::new([self.conv1_channels, self.conv2_channels], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_stride([2, 2])
            .init(device);

        let conv3 = Conv2dConfig::new([self.conv2_channels, self.conv3_channels], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_stride([2, 2])
            .init(device);

        let conv4 = Conv2dConfig::new([self.conv3_channels, self.conv4_channels], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_stride([2, 2])
            .init(device);

        //Global Average Pooling: reduces any spatial size to 1x1, helps with handling variable input sizes
        let gap = AdaptiveAvgPool2dConfig::new([1, 1]).init();

        //After GAP: [batch, conv4_channels, 1, 1] -> flatten to [batch, conv4_channels] so that fully connected layers can handle arbitrary spatial sizes
        let fc1 = LinearConfig::new(self.conv4_channels, self.fc1_size).init(device);
        let fc2 = LinearConfig::new(self.fc1_size, 1).init(device);

        let dropout = DropoutConfig::new(self.dropout_rate).init();
        let activation = Relu::new();

        HazeCNN {
            conv1, conv2, conv3, conv4, //convolutional layers
            gap,                        //global average pooling layer
            fc1, fc2,                   //fully connected layers
            dropout,                    //dropout (randomization) layer for regularization
            activation                  //ReLU activation function explained earlier
        }
    }
}

impl<B: Backend> HazeCNN<B> {
    /*
    Forward pass (prediction) through the network that processes input through convolution layers -> global average pool -> fully connected layers -> sigmoid for haze score.

    @param x: input tensor of shape [batch, channels=3, height, width], height/width can vary
    @return: output tensor of shape [batch, 1] containing haze scores in [0, 1]
    */
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        //layer 1: detect low-level features (edges, basic textures)
        let x = self.conv1.forward(x);
        let x = self.activation.forward(x);

        //layer 2: detect mid-level patterns made of edges
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);

        //layer 3: detect higher-level features (haze gradients, texture degradation)
        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);

        //layer 4: detect high-level haze characteristics such as shadows, glows, and blurriness
        let x = self.conv4.forward(x);
        let x = self.activation.forward(x);

        //Global Average Pooling: [batch, 128, H, W] -> [batch, 128, 1, 1]; handles variable input sizes via averaging all spatial positions
        let x = self.gap.forward(x);

        //Flatten: [batch, 128, 1, 1] -> [batch, 128]
        let batch_size = x.dims()[0];
        let num_channels = x.dims()[1];
        let x = x.reshape([batch_size, num_channels]);

        //Fully connected layers with dropout for regularization
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.fc2.forward(x);

        sigmoid(x) //sigmoid constrains output to [0, 1] range for haze score
    }

    /*
    Convenient wrapper for predicting haze of a single image

    @param image: input tensor of shape [1, 3, H, W] (single image batch)
    @return: haze score as f32 in [0, 1]
    */
    pub fn predict_single(&self, image: Tensor<B, 4>) -> f32 {
        let output = self.forward(image);
        let output_data: Vec<f32> = output.to_data().to_vec().unwrap();
        output_data[0]
    }

    /*
    Save the trained model weights to a file for later use.
    Uses burn's NamedMpkFileRecorder for efficient binary serialization with MessagePack format.

    @param path: file path to save the model (without extension, .mpk will be added)
    @return: Result indicating success or error
    */
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let path_buf = path.as_ref().to_path_buf();
        let path_str = path_buf.to_string_lossy().to_string();

        self.clone()
            .save_file(path_buf, &recorder)
            .map_err(|e| format!("Failed to save model to '{}': {}", path_str, e))?;

        println!("Model saved to: {}.mpk", path_str);
        Ok(())
    }

    /*
    Load a previously saved model from a file.
    Creates a new model with random weights, then loads the saved weights from file.

    @param path: file path to load the model from (without extension)
    @param device: burn backend device to allocate tensors on
    @return: Result containing loaded model or error
    */
    pub fn load_model<P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Self, String> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let path_buf = path.as_ref().to_path_buf();
        let path_str = path_buf.to_string_lossy().to_string();

        //Create a default model config and initialize, then load weights
        let config = HazeCNNConfig::new();
        let model = config.init::<B>(device);

        let loaded_model = model
            .load_file(path_buf, &recorder, device)
            .map_err(|e| format!("Failed to load model from '{}': {}", path_str, e))?;

        println!("Model loaded from: {}.mpk", path_str);
        Ok(loaded_model)
    }
}


/*
Convert ndarray image to burn tensor struct format for CNN input, since burn expects format (batch, channels, height, width), but ndarray images are (height, width, channels).

@param img: input image as Array3<f32> of shape [height, width, 3] with values normalized to [0, 1]
@param device: burn backend device to allocate tensor on
@return: tensor shaped [1, 3, height, width] ready for CNN forward pass
*/
pub fn image_to_tensor<B: Backend>(img: &Array3<f32>, device: &B::Device) -> Tensor<B, 4> {
    let (height, width, channels) = img.dim();

    //Reorder from HWC (Height, Weight, Channel) to CHW (Channel, Width, Height) format as burn/PyTorch convention expects channel-first
    let mut data = Vec::with_capacity(channels * height * width);
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                data.push(img[[y, x, c]]);
            }
        }
    }

    //Create tensor and reshape to [batch=1, channels, height, width]
    Tensor::<B, 1>::from_floats(data.as_slice(), device)
        .reshape([1, channels, height, width])
}


/*
Group images by their dimensions so we can batch them together for GPU efficiency, since images with the same dimensions can be stacked into a single tensor and processed in parallel which massively improves GPU utilization compared to one-image-at-a-time processing.
Returns a HashMap where keys are (height, width) tuples and values are vectors of (index, image, label) tuples so we can track which images are in each group.

@param images: slice of images as Array3<f32>
@param labels: slice of corresponding haze labels
@return: HashMap mapping (height, width) to vector of (original_index, image_ref, label)
*/
fn group_images_by_dimensions<'a>(
    images: &'a [Array3<f32>],
    labels: &'a [f64],
) -> HashMap<(usize, usize), Vec<(usize, &'a Array3<f32>, f64)>> {
    let mut groups: HashMap<(usize, usize), Vec<(usize, &'a Array3<f32>, f64)>> = HashMap::new();

    for (idx, (img, &label)) in images.iter().zip(labels.iter()).enumerate() {
        let (h, w, _) = img.dim();
        //Skip images too small for 4 stride-2 convolutions (would collapse to nothing)
        if h < 16 || w < 16 {
            continue;
        }
        groups.entry((h, w)).or_insert_with(Vec::new).push((idx, img, label));
    }

    groups
}


/*
Create a batched tensor from multiple same-sized images for efficient GPU processing, stacks images along the batch dimension so the GPU can process them all in one forward pass instead of one at a time which is super slow due to CPU-GPU transfer overhead dominating tiny workloads.

@param images: slice of images that MUST all have the same dimensions
@param device: burn backend device to allocate tensor on
@return: tensor shaped [batch_size, 3, height, width]
*/
fn images_to_batched_tensor<B: Backend>(images: &[&Array3<f32>], device: &B::Device) -> Tensor<B, 4> {
    if images.is_empty() {
        panic!("Cannot create batched tensor from empty slice");
    }

    let (height, width, channels) = images[0].dim();
    let batch_size = images.len();

    //Pre-allocate for all images in batch
    let mut data = Vec::with_capacity(batch_size * channels * height * width);

    for img in images {
        //Verify dimensions match (should be guaranteed by caller but let's be safe)
        debug_assert_eq!(img.dim(), (height, width, channels), "All images in batch must have same dimensions");

        //Reorder from HWC to CHW for each image
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    data.push(img[[y, x, c]]);
                }
            }
        }
    }

    Tensor::<B, 1>::from_floats(data.as_slice(), device)
        .reshape([batch_size, channels, height, width])
}


/*
OUTMODED/NOT USED FOR NOW, LAST RESORT OPTION FOR THIS PROJECT SO FAR. Resize all images to a common dimension for efficient GPU batching since having 116 dimension groups with most containing only 1-5 images causes excessive CPU-GPU synchronization overhead and basically defeats the purpose of GPU acceleration. By resizing to a common size we get 1 dimension group and can batch all images together for proper GPU parallelism.

@param images: slice of images as Array3<f32> with potentially different dimensions
@param target_size: (height, width) tuple for target dimensions
@return: Vec of resized images all with the same dimensions
*/
pub fn resize_images_to_common_size(images: &[Array3<f32>], target_size: (usize, usize)) -> Vec<Array3<f32>> {
    use rayon::prelude::*;

    let (target_h, target_w) = target_size;
    println!("Resizing {} images to {}x{} for efficient GPU batching...", images.len(), target_h, target_w);
    io::stdout().flush().unwrap();

    let start = Instant::now();

    let result: Vec<Array3<f32>> = images.par_iter().map(|img| {
        let (h, w, c) = img.dim();

        //Skip resize if already correct size
        if h == target_h && w == target_w {
            return img.clone();
        }

        //Convert Array3 to image::RgbImage for resizing
        let mut rgb_img = image::RgbImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let r = (img[[y, x, 0]] * 255.0).clamp(0.0, 255.0) as u8;
                let g = (img[[y, x, 1]] * 255.0).clamp(0.0, 255.0) as u8;
                let b = (img[[y, x, 2]] * 255.0).clamp(0.0, 255.0) as u8;
                rgb_img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
            }
        }

        //Resize using Triangle filter (fast, decent quality)
        let resized = image::imageops::resize(
            &rgb_img,
            target_w as u32,
            target_h as u32,
            image::imageops::FilterType::Triangle
        );

        //Convert back to Array3<f32>
        let mut result = Array3::<f32>::zeros((target_h, target_w, c));
        for y in 0..target_h {
            for x in 0..target_w {
                let pixel = resized.get_pixel(x as u32, y as u32);
                result[[y, x, 0]] = pixel[0] as f32 / 255.0;
                result[[y, x, 1]] = pixel[1] as f32 / 255.0;
                result[[y, x, 2]] = pixel[2] as f32 / 255.0;
            }
        }
        result
    }).collect();

    println!("Resized {} images in {:?}", result.len(), start.elapsed());
    io::stdout().flush().unwrap();

    result
}


/*
Mean Squared Error loss for regression training.
MSE = mean((predictions - targets)^2) (also used in Iteration 1)

@param predictions: model output tensor of shape [batch, 1]
@param targets: ground truth labels tensor of shape [batch, 1]
@return: scalar loss tensor
*/
pub fn mse_loss<B: Backend>(predictions: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
    let diff = predictions - targets;
    let squared = diff.clone() * diff;
    squared.mean()
}


/*
Train the CNN on a set of images and labels using dimension-grouped batching for GPU efficiency. Images with the same dimensions are batched together so the GPU can process them in parallel instead of one at a time which was causing massive underutilization (GPU would spike at model init then sit idle while processing tiny single-image workloads with CPU-GPU transfer overhead dominating).

@param model: initialized HazeCNN model to train
@param train_images: slice of images as Array3<f32>, can have different dimensions due to global average pooling
@param train_labels: slice of haze labels in [0, 1], must match train_images length (mapped)
@param epochs: number of training epochs (number of full passes through dataset in training)
@param learning_rate: Adam optimizer learning rate (Adam is a gradient descent optimization (basically optimization algorithm for the learning rate) with adaptive (adjusts learning rate during runtime) learning rate per parameter, using simple math)
@param batch_size: max images per batch (images of same dimensions are grouped, then split into chunks of this size)
@param device: burn backend device
@return: trained HazeCNN model
*/
pub fn train_cnn<B: Backend>(
    model: HazeCNN<B>,
    train_images: &[Array3<f32>],
    train_labels: &[f64],
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
    device: &B::Device,
) -> HazeCNN<B>
where
    B: AutodiffBackend,
{
    assert_eq!(train_images.len(), train_labels.len(), "Images and labels must have same length");
    let num_samples = train_images.len();

    //Group images by dimensions (exact match) for batching, which is how optimization for GPU utilization works
    let dimension_groups = group_images_by_dimensions(train_images, train_labels);
    let num_groups = dimension_groups.len();
    let valid_samples: usize = dimension_groups.values().map(|v| v.len()).sum();

    println!("=== CNN Training Started ===");
    println!("Training samples: {} ({} valid after size filter)", num_samples, valid_samples);
    println!("Dimension groups: {} (images grouped by size for batching)", num_groups);
    println!("Batch size: {} (max images per GPU batch)", batch_size);
    println!("Epochs: {}, Learning rate: {}", epochs, learning_rate);
    println!();
    io::stdout().flush().unwrap();

    //Print dimension group distribution for debugging
    let mut group_sizes: Vec<_> = dimension_groups.iter()
        .map(|((h, w), imgs)| (*h, *w, imgs.len()))
        .collect();
    group_sizes.sort_by(|a, b| b.2.cmp(&a.2)); //sort by count descending
    println!("Top dimension groups (HxW: count):");
    for (h, w, count) in group_sizes.iter().take(5) {
        println!("  {}x{}: {} images", h, w, count);
    }
    if num_groups > 5 {
        println!("  ... and {} more groups", num_groups - 5);
    }
    println!();
    io::stdout().flush().unwrap();

    //Adam optimizer for adaptive learning rate per parameter
    let optimizer_config = AdamConfig::new();
    let mut optimizer = optimizer_config.init();

    let mut current_model = model;
    let training_start = Instant::now();

    println!("NOTE: First batch will be slow due to GPU shader compilation (one-time cost)");
    io::stdout().flush().unwrap();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut processed_images = 0usize;
        let mut batch_count = 0usize;

        println!("Epoch {}/{} starting...", epoch + 1, epochs);
        io::stdout().flush().unwrap();

        let mut group_idx = 0usize;
        let total_groups = dimension_groups.len();

        //Process each dimension group
        for ((_h, _w), group) in dimension_groups.iter() {
            group_idx += 1;
            //Skip very small images that would collapse to nothing after 4 stride-2 convolutions
            if _h < &16 || _w < &16 {
                continue;
            }

            //Calculate adaptive batch size based on image dimensions to avoid GPU OOM
            //Larger images need smaller batches to fit in VRAM
            //RTX 3070 has 8GB VRAM, estimate ~20MB per 1024x1024 image in training with gradients
            let pixels = _h * _w;
            let adaptive_batch_size = if pixels > 800_000 {
                //Large images (>~900x900): process one at a time to avoid OOM
                1
            } else if pixels > 400_000 {
                //Medium-large images (~600x600 to ~900x900): small batches
                2.min(batch_size)
            } else if pixels > 200_000 {
                //Medium images (~450x450 to ~600x600): moderate batches
                4.min(batch_size)
            } else if pixels > 100_000 {
                //Small-medium images (~300x300 to ~450x450): larger batches
                8.min(batch_size)
            } else {
                //Small images (<~300x300): full batch size
                batch_size
            };

            //Split group into batches of adaptive_batch_size
            let group_batches: Vec<_> = group.chunks(adaptive_batch_size).collect();
            let num_batches_in_group = group_batches.len();

            for (batch_idx, batch_chunk) in group_batches.into_iter().enumerate() {
                let batch_start = Instant::now();

                //Print BEFORE processing so user sees immediate feedback
                println!("  Grp {}/{} ({}x{}) | Batch {}/{} | {}/{} ({:.1}%) | Starting...",
                    group_idx, total_groups, _h, _w,
                    batch_idx + 1, num_batches_in_group,
                    processed_images, valid_samples,
                    (processed_images as f32 / valid_samples as f32) * 100.0);
                io::stdout().flush().unwrap();

                let batch_images: Vec<&Array3<f32>> = batch_chunk.iter().map(|(_, img, _)| *img).collect();
                let batch_labels: Vec<f32> = batch_chunk.iter().map(|(_, _, lbl)| *lbl as f32).collect();
                let current_batch_size = batch_images.len();

                //Convert batch of ndarray images to tensors
                let image_tensor = images_to_batched_tensor::<B>(&batch_images, device);
                let label_tensor = Tensor::<B, 1>::from_floats(batch_labels.as_slice(), device)
                    .reshape([current_batch_size, 1]);

                //Forward pass (making the prediction): batch of images -> batch of haze score predictions
                let predictions = current_model.forward(image_tensor);

                //Compute MSE loss between prediction and ground truth for the batch
                let loss = mse_loss(predictions, label_tensor);
                let loss_value: f32 = loss.clone().into_scalar().elem();
                epoch_loss += loss_value * current_batch_size as f32; //weight by batch size for proper averaging

                //Backward pass (learn based on prediction error):: compute gradients and update weights
                let grads = loss.backward();

                let grads_params = GradientsParams::from_grads(grads, &current_model);
                current_model = optimizer.step(learning_rate, current_model, grads_params);

                processed_images += current_batch_size;
                batch_count += 1;

                //Print completion with timing
                let batch_duration = batch_start.elapsed();
                println!("    -> Done in {:?}, Loss: {:.6}", batch_duration, loss_value);
                io::stdout().flush().unwrap();
            }
        }

        //Print epoch summary with timing (clear line first with spaces to overwrite progress)
        let avg_loss = if processed_images > 0 { epoch_loss / processed_images as f32 } else { 0.0 };
        let epoch_duration = epoch_start.elapsed();
        println!("\rEpoch {}/{} complete: {:?}, {} batches, {} images, MSE Loss = {:.6}                    ",
            epoch + 1, epochs, epoch_duration, batch_count, processed_images, avg_loss);
        io::stdout().flush().unwrap();
    }

    let total_duration = training_start.elapsed();
    println!("\n=== CNN Training Complete in {:?} ===", total_duration);
    io::stdout().flush().unwrap();
    current_model
}


/* AI-generated bc simple
Evaluate trained model on a set of images via computing MSE

@param model: trained HazeCNN model
@param images: slice of test images as Array3<f32>
@param labels: slice of ground truth haze labels
@param device: burn backend device
@return: MSE as f64, or NaN if no valid images
*/
pub fn evaluate_cnn<B: Backend>(
    model: &HazeCNN<B>,
    images: &[Array3<f32>],
    labels: &[f64],
    device: &B::Device,
) -> f64 { //AI-generated code to simply reimplement previous MSE for linear regression
    let mut total_squared_error = 0.0;
    let mut count = 0;

    for (img, &label) in images.iter().zip(labels.iter()) {
        let (h, w, _) = img.dim();
        if h < 16 || w < 16 {
            continue; //skip images too small for the network
        }

        let tensor = image_to_tensor::<B>(img, device);
        let prediction = model.predict_single(tensor);

        let error = prediction as f64 - label;
        total_squared_error += error * error;
        count += 1;
    }

    if count > 0 {
        total_squared_error / count as f64
    } else {
        f64::NAN
    }
}


/* AI-generated function wrapper
Predict haze score for a single image using trained model

@param model: trained HazeCNN model
@param image: input image as Array3<f32>
@param device: burn backend device
@return: predicted haze score in [0, 1]
*/
pub fn predict_haze_cnn<B: Backend>(
    model: &HazeCNN<B>,
    image: &Array3<f32>,
    device: &B::Device,
) -> f32 { //AI-generated wrapper
    let tensor = image_to_tensor::<B>(image, device);
    model.predict_single(tensor)
}


/* AI-generated PLACEHOLDER heuristic
Simple PLACEHOLDER heuristic to suggest DCP dehazing parameters based on CNN-predicted haze level. Higher haze scores need more aggressive dehazing (lower omega, higher t0).
Suggested parameters by haze level:
    High haze (>0.7): omega=0.65 (remove 35% more haze), t0=0.25, larger guided_radius=20
    Medium haze (0.4-0.7): omega=0.75 (balanced), t0=0.2, guided_radius=15
    Low haze (<0.4): omega=0.85 (gentle), t0=0.15, smaller guided_radius=10

@param haze_score: predicted haze level from CNN in [0, 1]
@return: tuple of (omega, t0, patch_size, guided_radius, guided_eps) for dehaze_with_params
*/
pub fn suggest_dcp_parameters(haze_score: f32) -> (f32, f32, usize, usize, f32) { //Again, VERY MUCH A PLACEHOLDER before more complex parameter recommendations
    if haze_score > 0.7 {
        //High haze: aggressive dehazing
        (0.65, 0.25, 15, 20, 0.0001)
    } else if haze_score > 0.4 {
        //Medium haze: balanced parameters
        (0.75, 0.2, 15, 15, 0.0001)
    } else {
        //Low haze: gentle dehazing to avoid artifacts
        (0.85, 0.15, 15, 10, 0.001)
    }
}


//=============================================================================
// High-Level API and Demo Functions
//=============================================================================

//Type aliases for cleaner code, CPU backend, cleaner but slower, AI-generated for readability, will be changed with better machine set up for development.
pub type CnnBackend = NdArray<f32>;
pub type AutodiffCnnBackend = burn::backend::Autodiff<CnnBackend>;

//Type aliases for GPU backend which uses wgpu for GPU acceleration, which is MUCH faster for training, keeps code cleaner
pub type GpuBackend = Wgpu<f32, i32>;
pub type AutodiffGpuBackend = burn::backend::Autodiff<GpuBackend>;

/* PURELY AI-GENERATED TO TEST FOR COMPILATION AND FOR EFFICIENCY
Full CNN training pipeline with optional test set evaluation. //OUTMODED due to switch to GPU's for speed.
Entry point for training from main.rs via --train-cnn flag.


@param train_images: training images as Array3<f32> slices
@param train_labels: training haze labels in [0, 1]
@param test_images: optional test images for evaluation
@param test_labels: optional test labels (must be provided if test_images is Some)
@param epochs: number of training epochs
@param batch_size: unused but kept for burn API consistency (variable sizes prevent batching)
@param learning_rate: Adam optimizer learning rate
@param save_path: optional path to save the trained model (without extension)
@return: trained HazeCNN model
*/
pub fn run_cnn_training<P: AsRef<Path>>(
    train_images: &[Array3<f32>],
    train_labels: &[f64],
    test_images: Option<&[Array3<f32>]>,
    test_labels: Option<&[f64]>,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    save_path: Option<P>,
) -> HazeCNN<AutodiffCnnBackend> { //AI generated running code, series of calls to various components
    println!("\n========================================");
    println!("  Iteration 2: CNN Haze Detection");
    println!("  Variable Input Size Architecture");
    println!("========================================\n");

    let device = Cpu; //hard-coding for now, CPU backend for training, will be changed to GPU backend with wgpu in next update to this iteration
    let config = HazeCNNConfig::new();
    let model = config.init::<AutodiffCnnBackend>(&device);

    //Print model architecture for documentation/debugging //AI generated printing
    println!("Model Architecture:");
    println!("  Input: Variable size RGB image (H x W x 3)");
    println!("  Conv1: 3 -> {} channels, stride 2 (H/2 x W/2)", config.conv1_channels);
    println!("  Conv2: {} -> {} channels, stride 2 (H/4 x W/4)", config.conv1_channels, config.conv2_channels);
    println!("  Conv3: {} -> {} channels, stride 2 (H/8 x W/8)", config.conv2_channels, config.conv3_channels);
    println!("  Conv4: {} -> {} channels, stride 2 (H/16 x W/16)", config.conv3_channels, config.conv4_channels);
    println!("  Global Average Pooling -> {} features", config.conv4_channels);
    println!("  FC1: {} -> {}", config.conv4_channels, config.fc1_size);
    println!("  FC2: {} -> 1 (haze score)", config.fc1_size);
    println!("  Dropout rate: {}", config.dropout_rate);
    println!();

    let trained_model = train_cnn(
        model,
        train_images,
        train_labels,
        epochs,
        learning_rate,
        batch_size,
        &device,
    );

    //Evaluate on test set if provided
    if let (Some(test_imgs), Some(test_lbls)) = (test_images, test_labels) {
        println!("\n=== Evaluating on Test Set ===");
        let mse = evaluate_cnn(&trained_model, test_imgs, test_lbls, &device);
        println!("Test set MSE: {:.6} \n", mse);

        //Show sample predictions with suggested DCP parameters
        println!("Sample predictions with suggested DCP parameters:");
        for (i, (img, &label)) in test_imgs.iter().zip(test_lbls.iter()).take(5).enumerate() {
            let (h, w, _) = img.dim();
            if h < 16 || w < 16 { continue; }

            let pred = predict_haze_cnn(&trained_model, img, &device);
            let (omega, t0, patch, radius, eps) = suggest_dcp_parameters(pred);
            println!("  Image {}: predicted={:.3}, actual={:.3}", i + 1, pred, label);
            println!("           -> suggested: omega={}, t0={}, patch={}, radius={}, eps={}",
                     omega, t0, patch, radius, eps);
        }
    }
    println!("CNN training complete");

    //Save model if path provided
    if let Some(path) = save_path {
        if let Err(e) = trained_model.save_model(path) {
            eprintln!("Warning: {}", e);
        }
    }

    trained_model
}


/* Partially AI-generated
Quick PROOF OF CONCEPT demo of CNN training on a small number of images.
Similar to run_ml_demo() for linear regression, shows the system works without full dataset training.
Note: CPU-only training with burn/ndarray is slow. For production, further testing, and further development, will need to use GPU backend with wgpu

@param train_images: small set of demo images
@param train_labels: corresponding haze labels
@return: "trained" model (not optimized, just for demonstration, so it is very inaccurate and just shows the pipeline is working)
*/
pub fn run_cnn_demo(
    train_images: &[Array3<f32>],
    train_labels: &[f64],
) -> HazeCNN<AutodiffCnnBackend> {
    println!("\n=== CNN Demo Mode ===");
    println!("Training on {} images for quick demonstration on CPU-only devices (currently used for development)", train_images.len());
    println!("Note: CPU training is slow. For production, testing, and further development, will use GPU backend.\n");

    run_cnn_training(
        train_images,
        train_labels,
        None,               //no separate test set for demo
        None,
        5,          //minimal epochs for fast demo (CPU is slow)
        4,       //batch size unused
        0.01,   //higher learning rate to converge faster with fewer epochs for CPU-only test
        None::<&str>,       //no model saving for demo
    )
}


/*
GPU-accelerated CNN training pipeline using wgpu backend. Falls back gracefully if no GPU is available (wgpu will use CPU compute).
Purpose: MUCH faster than CPU training since GPU's are suited to convolutional neural networks which require many fast matrix multiplications.

@param train_images: training images as Array3<f32> slices
@param train_labels: training haze labels in [0, 1]
@param test_images: optional test images for evaluation
@param test_labels: optional test labels (must be provided if test_images is Some)
@param epochs: number of training epochs
@param learning_rate: Adam optimizer learning rate
@param save_path: optional path to save the trained model (without extension)
@return: trained HazeCNN model on GPU backend
*/
pub fn run_cnn_training_gpu<P: AsRef<Path>>(
    train_images: &[Array3<f32>],
    train_labels: &[f64],
    test_images: Option<&[Array3<f32>]>,
    test_labels: Option<&[f64]>,
    epochs: usize,
    learning_rate: f64,
    save_path: Option<P>,
) -> HazeCNN<AutodiffGpuBackend> {
    println!("\n========================================");
    println!("  Iteration 2: CNN Haze Detection");
    println!("  GPU-Accelerated Training (wgpu)");
    println!("========================================\n");
    io::stdout().flush().unwrap();

    // ========== GPU DIAGNOSTICS ==========
    println!("=== GPU DIAGNOSTICS ===");
    io::stdout().flush().unwrap();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    println!("Available GPU adapters:");
    io::stdout().flush().unwrap();

    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    if adapters.is_empty() {
        println!("  WARNING: No GPU adapters found! Training will use CPU (very slow).");
    } else {
        for (i, adapter) in adapters.iter().enumerate() {
            let info = adapter.get_info();
            println!("  [{}] {} ({:?})", i, info.name, info.backend);
            println!("      Device type: {:?}", info.device_type);
            println!("      Driver: {}", info.driver);
            io::stdout().flush().unwrap();
        }
    }
    println!();
    io::stdout().flush().unwrap();
    // ========== END GPU DIAGNOSTICS ==========

    println!("Initializing burn GPU backend...");
    println!("  Requesting DiscreteGpu(0) - this should be your RTX 3070");
    io::stdout().flush().unwrap();

    let gpu_init_start = Instant::now();
    //Use DiscreteGpu(0) to explicitly select the first discrete GPU (e.g., RTX 3070)
    //This avoids accidentally using integrated graphics or CPU fallback
    let device = WgpuDevice::DiscreteGpu(0);
    println!("  Device handle created in {:?}", gpu_init_start.elapsed());
    io::stdout().flush().unwrap();

    //Force GPU initialization by creating and running a small tensor operation
    //This triggers actual GPU/shader compilation BEFORE the main training loop
    println!("  Warming up GPU with progressively larger images...");
    io::stdout().flush().unwrap();
    let warmup_start = Instant::now();
    {
        /*
        //Create a small test tensor and do a forward pass to compile shaders
        let test_data: [f32; 12] = [0.5; 12]; //tiny 2x2x3 image
        let test_tensor: Tensor<AutodiffGpuBackend, 1> = Tensor::from_floats(&test_data[..], &device);
        let _reshaped = test_tensor.reshape([1, 3, 2, 2]);
        //Force sync by reading a value back
        */

        let config = HazeCNNConfig::new();
        let warmup_model = config.init::<AutodiffGpuBackend>(&device);
        /*
        let tiny_img = Array3::<f32>::zeros((32, 32, 3));
        let tiny_tensor = image_to_tensor::<AutodiffGpuBackend>(&tiny_img, &device);
        let _output = warmup_model.forward(tiny_tensor);
        println!("    Forward pass compiled in {:?}", warmup_start.elapsed());
        io::stdout().flush().unwrap();
        */

        //Test progressively larger images to find where it breaks
        let test_sizes = [(32, 32), (64, 64), (128, 128), /*(256, 256),*/ (512, 512)];

        for (h, w) in test_sizes {
            print!("    Testing {}x{}... ", h, w);
            io::stdout().flush().unwrap();
            let size_start = Instant::now();

            let test_img = Array3::<f32>::zeros((h, w, 3));
            let test_tensor = image_to_tensor::<AutodiffGpuBackend>(&test_img, &device);

            print!("forward... ");
            io::stdout().flush().unwrap();
            let output = warmup_model.forward(test_tensor.clone());

            print!("loss... ");
            io::stdout().flush().unwrap();
            let target = Tensor::<AutodiffGpuBackend, 1>::from_floats(&[0.5f32][..], &device).reshape([1, 1]);
            let loss = mse_loss(output, target);

            print!("backward... ");
            io::stdout().flush().unwrap();
            let backward_start = Instant::now();
            print!("vars: {}... ", warmup_model.to_string());
            //==============================================================================
            //  ISSUE BELOW BLOCK COMMENT (2/11/2026): This is where training hangs for images > 256x256 (used AI to write summary of issues here from a summary I gave it myself, and then retyped most of it anyways)
            //
            //  My debugging journey in a nutshell:
            //  "stuck on first epoch" -> "GPU not connected?" -> "batching taking time?" -> "still GPU problem??" -> "precompiling shaders?" -> "image sizes?" -> "force sync GPU?"
            //
            //  loss.backward() triggers async GPU work for backpropagation. For 256x256 and smaller it works fine. For 512x512 or any image with dimensions > 256 (e.g.  it just... get stuck, forever. The forward pass completes, the loss computes, but backward() never returns.
            //
            //  Attempted fixes that didn't work:
            //  - Force sync via GradientsParams::from_grads() (see below) (makes sure previous image is fully processed and gradients are computed before attempting to move onto the next)
            //  - drop(grad_params) to trigger cleanup and force sync
            //  - WGPU_BACKEND="vulkan" environment variable, also tried to set backend to DX12, neither made a difference
            //  - Making sure DiscreteGpu(0) is selected (the right one, it was)
            //  - Dimension-grouped batching, and changing the number of groups or the size of batches
            //  - Granular progress updates in case it was just slow (at least I can see WHERE it hangs now)
            //
            //  The issue is somewhere in wgpu/burn shader compilation or execution for larger convolution operations. Help would be appreciated. I am working on this myself in the meantime nevertheless.
            //==============================================================================
            let grads = loss.backward(); //cursed line here
            print!("backward done in {:?}", backward_start.elapsed());
            let grad_params = GradientsParams::from_grads(grads, &warmup_model); //Force sync by accessing gradient data, this works by forcing the GPU to finish computing gradient data as .backward() is asynchronous, so on paper this should trigger shader compilation and GPU execution for the backward pass, and if it works without OOM or errors then we know the GPU is not having issues with that image size.
            drop(grad_params);//Drop to ensure GPU work completes as a backup to the above

            println!("done in {:?}", size_start.elapsed());
            io::stdout().flush().unwrap();
        }
        /*
        //Now do a backward pass to compile those shaders too
        let backward_start = Instant::now();
        let tiny_tensor2 = image_to_tensor::<AutodiffGpuBackend>(&tiny_img, &device);
        let output2 = warmup_model.forward(tiny_tensor2);
        let target = Tensor::<AutodiffGpuBackend, 1>::from_floats(&[0.5f32][..], &device).reshape([1, 1]);
        let loss = mse_loss(output2, target);
        let _grads = loss.backward();
        println!("    Backward pass compiled in {:?}", backward_start.elapsed());
        io::stdout().flush().unwrap();
         */
    }

    println!("  GPU warmup complete in {:?}", warmup_start.elapsed());
    println!("  Using GPU device: {:?}", device);
    //println!("  NOTE: First epoch will be slower as each new image size compiles shaders.");
    io::stdout().flush().unwrap();

    let config = HazeCNNConfig::new();
    let model = config.init::<AutodiffGpuBackend>(&device);

    println!("\n  Model Architecture:");
    println!("  Input: Variable size RGB image (H x W x 3)");
    println!("  Conv1: 3 -> {} channels, stride 2 (H/2 x W/2)", config.conv1_channels);
    println!("  Conv2: {} -> {} channels, stride 2 (H/4 x W/4)", config.conv1_channels, config.conv2_channels);
    println!("  Conv3: {} -> {} channels, stride 2 (H/8 x W/8)", config.conv2_channels, config.conv3_channels);
    println!("  Conv4: {} -> {} channels, stride 2 (H/16 x W/16)", config.conv3_channels, config.conv4_channels);
    println!("  Global Average Pooling -> {} features", config.conv4_channels);
    println!("  FC1: {} -> {}", config.conv4_channels, config.fc1_size);
    println!("  FC2: {} -> 1 (haze score)", config.fc1_size);
    println!("  Dropout rate: {}", config.dropout_rate);
    println!();
    io::stdout().flush().unwrap();

    let trained_model = train_cnn(
        model,
        train_images,
        train_labels,
        epochs,
        learning_rate,
        16, //batch_size - images of same dimensions are grouped and batched
        &device,
    );

    //Evaluate on test set if provided (see README.md for instructions on setup, test set should be in same directory as training set)
    if let (Some(test_imgs), Some(test_lbls)) = (test_images, test_labels) {
        println!("\n=== Evaluating on Test Set ===");
        io::stdout().flush().unwrap();

        let mse = evaluate_cnn(&trained_model, test_imgs, test_lbls, &device);
        println!("Test set MSE: {:.6} \n", mse);

        println!("Sample predictions with suggested DCP parameters:");
        for (i, (img, &label)) in test_imgs.iter().zip(test_lbls.iter()).take(5).enumerate() {
            let (h, w, _) = img.dim();
            if h < 16 || w < 16 { continue; }

            let pred = predict_haze_cnn(&trained_model, img, &device);
            let (omega, t0, patch, radius, eps) = suggest_dcp_parameters(pred);
            println!("  Image {}: predicted={:.3}, actual={:.3}", i + 1, pred, label);
            println!("           -> suggested: omega={}, t0={}, patch={}, radius={}, eps={}",
                     omega, t0, patch, radius, eps);
        }
        io::stdout().flush().unwrap();
    }
    println!("GPU CNN training complete");
    io::stdout().flush().unwrap();

    //Save model if path provided (recommended for usage in production)
    if let Some(path) = save_path {
        if let Err(e) = trained_model.save_model(path) {
            eprintln!("Warning: {}", e);
        }
    }

    trained_model
}


/*
Load a pre-trained CNN model for inference without training for production use after training. OUTMODED due to switch to GPU usage. Kept as not-dead code for future refactoring/cleanup.

@param model_path: path to the saved model file (WITHOUT .mpk extension)
@return: Result containing loaded model ready for inference, or error message
*/
pub fn load_pretrained_model<P: AsRef<Path>>(model_path: P) -> Result<HazeCNN<CnnBackend>, String> {
    let device = Cpu;
    HazeCNN::<CnnBackend>::load_model(model_path, &device)
}


/*
Load a pre-trained CNN model for inference on GPU for production use after training is complete.

@param model_path: path to the saved model file (WITHOUT .mpk extension)
@return: Result containing loaded model ready for inference on GPU, or error message
*/
pub fn load_pretrained_model_gpu<P: AsRef<Path>>(model_path: P) -> Result<HazeCNN<GpuBackend>, String> {
    let device = WgpuDevice::default();
    HazeCNN::<GpuBackend>::load_model(model_path, &device)
}


/*
Load a pre-trained CNN model with autodiff backend for continued training. OUTMODED due to switch to GPU usage. Kept as not-dead code for future refactoring/cleanup.
Purpose: use to resume training from a checkpoint that was saved during training.

@param model_path: path to the saved model file (without .mpk extension)
@return: Result containing loaded model ready for training, or error message
*/
pub fn load_pretrained_model_for_training<P: AsRef<Path>>(model_path: P) -> Result<HazeCNN<AutodiffCnnBackend>, String> {
    let device = Cpu;
    HazeCNN::<AutodiffCnnBackend>::load_model(model_path, &device)
}


#[cfg(test)]
mod tests { //AI-GENERATED Unit Tests of helper functions for tensor construction and variable input image sizes. WILL ADD MORE AND MORE IN-DEPTH TESTS IN FUTURE.
            //Initial Unit Tests remain CPU-only in case tests are done on a device without a GPU.
    use super::*;

    #[test]
    fn test_image_to_tensor_dimensions() {
        //Create a small test image (20 height x 10 width x 3 channels)
        let img = Array3::<f32>::zeros((20, 10, 3));
        let device = Cpu;

        let tensor = image_to_tensor::<CnnBackend>(&img, &device);
        let dims = tensor.dims();

        assert_eq!(dims[0], 1);   //batch size
        assert_eq!(dims[1], 3);   //channels
        assert_eq!(dims[2], 20);  //height
        assert_eq!(dims[3], 10);  //width
    }

    #[test]
    fn test_cnn_forward_variable_sizes() {
        //Test that CNN can handle different input sizes
        let device = Cpu;
        let config = HazeCNNConfig::new();
        let model = config.init::<CnnBackend>(&device);

        //Test with 64x64 image
        let img1 = Array3::<f32>::zeros((64, 64, 3));
        let tensor1 = image_to_tensor::<CnnBackend>(&img1, &device);
        let output1 = model.forward(tensor1);
        assert_eq!(output1.dims(), [1, 1]);

        //Test with 128x96 image (different aspect ratio)
        let img2 = Array3::<f32>::zeros((128, 96, 3));
        let tensor2 = image_to_tensor::<CnnBackend>(&img2, &device);
        let output2 = model.forward(tensor2);
        assert_eq!(output2.dims(), [1, 1]);
    }

    #[test]
    fn test_model_save_and_load() { //Test if an empty model can be saved and loaded successfully. Should probably write more in-depth tests for model integrity in future.
        let device = Cpu;
        let config = HazeCNNConfig::new();
        let original_model = config.init::<CnnBackend>(&device);

        let test_img = Array3::<f32>::from_elem((64, 64, 3), 0.5); //Build test image
        let original_prediction = predict_haze_cnn(&original_model, &test_img, &device); //Make original model query

        let test_path = "test_model_persistence";
        original_model.save_model(test_path).expect("Failed to save model"); //Save model

        assert!(Path::new(&format!("{}.mpk", test_path)).exists(), "Model file should exist"); //Check for file path

        let loaded_model = HazeCNN::<CnnBackend>::load_model(test_path, &device).expect("Failed to load model"); //Load and query model or throw error
        let loaded_prediction = predict_haze_cnn(&loaded_model, &test_img, &device);

        assert!( //check that predictions match
            (original_prediction - loaded_prediction).abs() < 1e-6,
            "Predictions should match: original={}, loaded={}",
            original_prediction, loaded_prediction
        );

        fs::remove_file(format!("{}.mpk", test_path)).ok(); //Some cleanup to avoid clutter
    }
}