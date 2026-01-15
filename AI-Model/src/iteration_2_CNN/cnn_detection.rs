//Iteration 2 WIP: CNN for haze detection, EXTREMELY WIP (1/14), needs some refactoring for optimization, decent amount of AI-generated code for boilerplate and basic functions but also in some less-critical areas like the suggest_dcp_parameters function and parts of the training pipeline, will need to be cleaned up for finished Iteration 2.
//Replaces linear regression regressor with a Convolutional Neural Network implemented with the burn framework for improved accuracy
//Purpose: self-contained module for all CNN logic to avoid major changes to other files bc encapsulation makes the iterative design process more understandable
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
//TODO: Model persistence (save/load trained weights)
//TODO: Expand suggested DCP parameters feature beyond current simple heuristic
//TODO: MAJOR OPTIMIZATION FOR MORE THAN PROOF OF CONCEPT TESTING
//TODO: most testing really, but will need to set up environment on a device with more processing power for that.

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
    backend::ndarray::NdArray
};
use ndarray::Array3; //backend linear algebra library for initial testing

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
Train the CNN on a set of images and labels. Due to variable image sizes, training processes one image at a time (cannot batch different sizes). This is probably a big issue for testing and development, needs to be fixed for next steps on this iteration.

@param model: initialized HazeCNN model to train
@param train_images: slice of images as Array3<f32>, can have different dimensions due to global average pooling
@param train_labels: slice of haze labels in [0, 1], must match train_images length (mapped)
@param epochs: number of training epochs (number of full passes through dataset in training)
@param learning_rate: Adam optimizer learning rate (Adam is a gradient descent optimization (basically optimization algorithm for the learning rate) with adaptive (adjusts learning rate during runtime) learning rate per parameter, using simple math)
@param _batch_size: unused, kept for burn API consistency (variable sizes prevent batching)
@param device: burn backend device
@return: trained HazeCNN model
*/
pub fn train_cnn<B: Backend>(
    model: HazeCNN<B>,
    train_images: &[Array3<f32>],
    train_labels: &[f64],
    epochs: usize,
    learning_rate: f64,
    _batch_size: usize,
    device: &B::Device,
) -> HazeCNN<B>
where
    B: AutodiffBackend,
{
    assert_eq!(train_images.len(), train_labels.len(), "Images and labels must have same length");
    let num_samples = train_images.len();

    println!("=== CNN Training Started ===");
    println!("Training samples: {}", num_samples);
    println!("Epochs: {}, Learning rate: {}", epochs, learning_rate);
    println!("Note: Training one image at a time due to variable sizes\n");

    //Adam optimizer for adaptive learning rate per parameter
    let optimizer_config = AdamConfig::new();
    let mut optimizer = optimizer_config.init();

    let mut current_model = model;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut valid_samples = 0;

        for (img, &label) in train_images.iter().zip(train_labels.iter()) {
            let (h, w, _) = img.dim();

            //Skip very small images that would collapse to nothing after 4 stride-2 convolutions
            if h < 16 || w < 16 {
                continue;
            }

            //Convert ndarray image to burn tensor
            let image_tensor = image_to_tensor::<B>(img, device);
            let label_tensor = Tensor::<B, 1>::from_floats(&[label as f32][..], device)
                .reshape([1, 1]);

            //Forward pass (making the prediction): image -> haze score prediction
            let prediction = current_model.forward(image_tensor);

            //Compute MSE loss between prediction and ground truth
            let loss = mse_loss(prediction, label_tensor);
            let loss_value: f32 = loss.clone().into_scalar().elem();
            epoch_loss += loss_value;
            valid_samples += 1;

            //Backward pass (learn based on prediction error): compute gradients and update weights
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &current_model);
            current_model = optimizer.step(learning_rate, current_model, grads_params);
        }

        //Print progress every epoch and on final epoch (will be every 5 or 10 when better machine set up for work)
        if epoch % 1 == 0 || epoch == epochs - 1 {
            let avg_loss = if valid_samples > 0 { epoch_loss / valid_samples as f32 } else { 0.0 };
            println!("Epoch {}/{}: MSE Loss = {:.6}", epoch + 1, epochs, avg_loss);
        }
    }

    println!("\n=== CNN Training Complete ===");
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

//Type aliases for cleaner code, AI-generated for readability, will be changed with better machine set up for development.
pub type CnnBackend = NdArray<f32>;
pub type AutodiffCnnBackend = burn::backend::Autodiff<CnnBackend>;


/* PURELY AI-GENERATED TO TEST FOR COMPILATION AND FOR EFFICIENCY
Full CNN training pipeline with optional test set evaluation.
Entry point for training from main.rs via --train-cnn flag.


@param train_images: training images as Array3<f32> slices
@param train_labels: training haze labels in [0, 1]
@param test_images: optional test images for evaluation
@param test_labels: optional test labels (must be provided if test_images is Some)
@param epochs: number of training epochs
@param batch_size: unused but kept for burn API consistency (variable sizes prevent batching)
@param learning_rate: Adam optimizer learning rate
@return: trained HazeCNN model
*/
pub fn run_cnn_training(
    train_images: &[Array3<f32>],
    train_labels: &[f64],
    test_images: Option<&[Array3<f32>]>,
    test_labels: Option<&[f64]>,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
) -> HazeCNN<AutodiffCnnBackend> { //AI generated running code, series of calls to various components
    println!("\n========================================");
    println!("  Iteration 2: CNN Haze Detection");
    println!("  Variable Input Size Architecture");
    println!("========================================\n");

    let device = burn::backend::ndarray::NdArrayDevice::Cpu; //hard-coding for now, CPU backend for training, will be changed to GPU backend with wgpu in next update to this iteration
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
        0.01   //higher learning rate to converge faster with fewer epochs for CPU-only test
    )
}


#[cfg(test)]
mod tests { //AI-GENERATED Unit Tests of helper functions for tensor construction and variable input image sizes. WILL ADD MORE AND MORE IN-DEPTH TESTS IN FUTURE.
    use super::*;

    #[test]
    fn test_image_to_tensor_dimensions() {
        //Create a small test image (20 height x 10 width x 3 channels)
        let img = Array3::<f32>::zeros((20, 10, 3));
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;

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
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
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
}