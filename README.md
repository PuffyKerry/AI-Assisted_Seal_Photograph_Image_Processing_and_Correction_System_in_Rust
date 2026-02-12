# AI-Assisted Seal Photograph Image Processing and Correction System

**What is this?** A Rust-based image processing system for underwater/hazy seal photographs that uses machine learning to detect haze levels and automatically apply dehazing corrections. 
**What project?** Built as an Arizona State University (ASU, *insert something about #1 in INNOVATION here*) Barrett Honors College Honors Thesis (for my B.S. in Computer Science, Cybersecurity concentration) to explore AI-assisted image processing in Rust, expanding my Capstone Project with GDMS to "Develop a Web Server and Packet Sniffer in Rust" as a service mounted to the web server. 
**Why Rust?** Safety and efficiency are key to this project, and exploring the state of Rust's machine learning ecosystem was a good opportunity to learn more about Rust AND ML.
**Why seals?** The SealID dataset provides a good variety of hazy images (snow, mist, fog) for testing haze detection and correction algorithms, plus seals are cute and the dataset was freely available for research purposes, and lastly the scope is rather unique. 

**Two iterations:**
1. **Iteration 1 (Complete):** Linear regression model trained on Dark Channel Prior features to predict haze levels and suggest dehazing parameters, can feed it images via CLI after loading a pretrained / saved / persistent model.
2. **Iteration 2 (WIP):** Convolutional Neural Network for improved accuracy, with CPU and GPU backends... except GPU training is currently broken (see Known Issues below)

Note: some of the README was AI-generated based on my comments in the code. Less of it is AI generated now than before, excluding the summary of my GPU training issues below, which was based on my own recollection of what issues I was having.   

Status as of 2/11/2026: Iteration 1 complete with manual query pipeline. Iteration 2 CNN pipeline has model persistence and manual query feature for both CPU and GPU backends but **GPU training hangs at first epoch** due to what appears to be a wgpu/burn shader compilation issue with larger image sizes. CPU training works but is slow. See Known Issues section for the full debugging saga.

## Known Issues (2/11/2026)

### GPU Training Hangs at First Epoch (AI summary of my long list of steps taken in cnn_detection.rs)

**TLDR:** GPU training hangs during backward pass for images larger than 256x256. Forward pass works, loss computes, but `loss.backward()` never returns. 256x256 works fine, 512x512 hangs indefinitely.

**The Long Debugging Journey (for my capstone sponsor and anyone else who's curious, this is MUCH more in depth in cnn_detection.rs):**

| Version | What I Thought Was Wrong | What I Tried                                                                                    |
|---------|-------------------------|-------------------------------------------------------------------------------------------------|
| 1.1.1 | Training is just slow | Added epoch printing, but nothing printed after "Epoch 1/50 starting..." for 20+ minutes        |
| 1.2.1 | GPU isn't being used | Checked Task Manager, saw GPU spike to 90% during image loading then drop to 1-2% during training |
| 1.2.2 | Wrong GPU detected? | Added GPU detection printing, wgpu sees RTX 3070 via Vulkan just fine                           |
| 1.2.3 | Using integrated graphics? | Added WgpuDevice::DiscreteGpu(0) and GPU diagnostics to enumerate all adapters                  |
| 1.3.1 | One-image-at-a-time overhead | Implemented dimension-grouped batching for images of same size                                  |
| 1.3.2 | Image size variance | Rejected suggestion to resize everything to 256x256 (didn't want to lose quality)               |
| 1.3.3 | wgpu backend issues | Added WGPU_BACKEND="vulkan" env var, verified Vulkan adapter selection                          |
| 1.4.1 | Need more visibility | Added granular progress: "Group 1/116 \| Batch 1/1 \| 0/430 (0.0%)" - still stuck               |
| 1.4.2 | Shader compilation? | Added warmup diagnostics and timing, forward/backward compilation times                         |
| 1.5.1 | Size-dependent issue | Started testing different image sizes in warmup                                                 |
| 1.5.2.1 | Too many shader variants | Tested number of test sizes for compilation                                                     |
| 1.5.2.2 | Non-square ratios | Tested non-square images like 344x484, they are fine if not too big                             |
| 1.5.2.3 | Power-of-2 requirement | Found 256x256 works, 512x512 hangs!                                                             |
| 1.5.3 | Narrowing down location | Found issue is at loss.backward() around line 1006 in cnn_detection.rs                          |
| 1.5.4 | GPU async not syncing | Tried force sync via GradientsParams::from_grads() and drop() - didn't help                     |

**Current workaround:** Use CPU training (slow but works) or resize all images to 256x256 before GPU training (may have to resort to this if I can't find another method)
**Plans:** try to work out why GPU training hangs at first epoch for larger images, or work around this somehow (breaking up images perhaps? or using a different backend?)

**If you figure this out:** Please submit a PR or open an issue, I would genuinely love to know what's wrong here.
**Also, please gaze upon the seals while you're here. They are cute.**
---

TODO:  
  - Figure out why GPU training hangs (see above)
  - More testing for Iteration 2 CNN on better hardware? (currently on RTX 3070 machine)
  - Add more IP functions (contrast adjustment, glare reduction) if possible
  - Improve formatting of README and documentation in general.  
  - Make more improvements to organization if possible.   

## Dataset Setup

The SealID dataset (~2GB, 2000+ images) is required for training but is too large for GitHub. Reasons for using this dataset were outlined in the project plan. This dataset is being used for Iteration 2 of the project for comparison purposes.

**Dataset Source:** https://etsin.fairdata.fi/dataset/22b5191e-f24b-4457-93d3-95797c900fc0
- Credit goes to researchers as mentioned:
  - Nepovinnykh, E., Eerola, T., Biard, V., Mutka, P., Niemi, M., Kunnasranta, M. and Kälviäinen, H., 2022. SealID: Saimaa ringed seal re-identification dataset. Sensors, 22(19), p.7602.
  - Nepovinnykh, E., Chelak, I., Eerola, T. and Kälviäinen, H., 2022. Norppa: Novel ringed seal re-identification by pelage pattern aggregation. arXiv preprint arXiv:2206.02498.
- Dataset needs to be downloaded separately. It is not included in this repository.

**Setup Instructions:**
1. Download the SealID dataset from the link above
2. Extract it to a `dataset/` folder in the project root:
   ```
   project_root/
   ├── dataset/
   │   └── SealID/
   │       ├── full images.zip/
   │       └── patches.zip/
   ├── AI-Model/
   ├── IP_functions/
   └── ...
   ```
3. Unzip/extract the `full images.zip` and `patches.zip` files in the `dataset/SealID/` folder. Place the unzipped folders in the same folder. Final file structure should look like this:
```
   project_root/
   ├── dataset/
   │   └── SealID/
   │       ├── full images.zip/
   |       |-- full images/
   |       |    |-- source_database
   |       |      |-- achuge.jpg
   |       |      |-- adpzbb.jpg
   |       |      |-- ...
   |       |
   │       └── patches.zip/
   |       |-- patches/
   |           |-- ...
   |
   ├── AI-Model/
   ├── IP_functions/
   └── ...
   ```
4. The `dataset/` folder is in `.gitignore` and will not be committed


## Usage

### Command Line Options

```bash
# Run Linear Regression Regressor training demo (default)
cargo run -p ai-model

# Run IP engine tests (dehazing on test images)
cargo run -p ai-model -- --ip-tests

# Train Regressor on full dataset demonstration (requires dataset setup)
cargo run -p ai-model -- --train-full-demo

# Train on demo images and save model for later use
cargo run -p ai-model -- --train-save haze_model.json

# Train on full dataset and save model for later use
cargo run -p ai-model -- --train-full-save haze_model.json

# Load saved model and process a new image FROM THE QUERY DATASET IN THE EXTERNAL SealID DATASET (the whole point of persistence)
cargo run -p ai-model -- --process haze_model.json "dataset/SealID/full images/source_query/input.jpg" output.jpg

# Run Convolutional Neural Network demo (quick test on 2 images)
cargo run -p ai-model -- --demo-cnn

# Train CNN on full dataset (requires dataset setup, slow on CPU)
cargo run -p ai-model -- --train-cnn

# Train CNN on full dataset and save model for later use (CPU)
cargo run -p ai-model -- --train-cnn-save cnn_model

# Load saved CNN model and process a new image (CPU)
cargo run -p ai-model -- --process-cnn cnn_model "dataset/SealID/full images/source_query/input.jpg" output.jpg

# Train CNN on full dataset with GPU acceleration (MUCH faster, requires GPU)
cargo run -p ai-model --release -- --train-cnn-gpu

# Train CNN on GPU and save model for later use (recommended for real usage)
cargo run -p ai-model --release -- --train-cnn-gpu-save cnn_model

# Load saved CNN model and process a new image on GPU
cargo run -p ai-model --release -- --process-cnn-gpu cnn_model "dataset/SealID/full images/source_query/input.jpg" output.jpg

# Dehaze a specific image with default parameters
cargo run -p ai-model -- --dehaze path/to/image.jpg

# Dehaze with custom DCP parameters
# Usage: --dehaze-custom FILE omega t0 patch_size guided_radius guided_eps
cargo run -p ai-model -- --dehaze-custom image.jpg 0.75 0.25 15 15 0.0001

# Show help
cargo run -p ai-model -- --help
```

### ITERATION 1 Model Persistence Workflow
Train once, use many times - the whole point of having a trained model:
```bash
# Step 1: Train and save (do this once)
cargo run -p ai-model -- --train-save haze_model.json
# or for full dataset:
cargo run -p ai-model -- --train-full-save haze_model.json

# Step 2: Process new images with saved model (do this as many times as you want)
cargo run -p ai-model -- --process haze_model.json 'dataset/SealID/full images/query_database/foggy_seal.jpg' clear_seal.jpg
```
The --process command automatically picks dehazing parameters based on predicted haze level:
- High haze (>0.7): Aggressive dehazing (omega=0.6, t0=0.3)
- Moderate haze (0.4-0.7): Balanced dehazing (omega=0.75, t0=0.2)  
- Low haze (<0.4): Gentle dehazing (omega=0.85, t0=0.15)

### ITERATION 2 Model Persistence Workflow (CNN)
Same workflow as Iteration 1, but with the CNN model. GPU version is MUCH faster and recommended if you have a dedicated graphics card:

#### CPU Training (slow, for devices without GPU)
```bash
# Step 1: Train and save CNN (do this once - warning: slow on CPU)
cargo run -p ai-model -- --train-cnn-save cnn_model

# Step 2: Process new images with saved CNN model
cargo run -p ai-model -- --process-cnn cnn_model "dataset/SealID/full images/source_query/input.jpg" output.jpg
```

#### GPU Training (fast, recommended - but currently broken, see Known Issues)
```bash
# Step 1: Train and save CNN on GPU (do this once - much faster... but not working right now, see Known Issues)
cargo run -p ai-model --release -- --train-cnn-gpu-save cnn_model

# Step 2: Process new images with saved CNN model on GPU
cargo run -p ai-model --release -- --process-cnn-gpu cnn_model "dataset/SealID/full images/source_query/input.jpg" output.jpg
```

**WARNING:** GPU training currently hangs at first epoch for images > 256x256. See Known Issues section at top of README for the full debugging saga. CPU training works but is slow beyond practicality for testing/validation/comparison or production use. 

Note: CNN models use `.mpk` format (MessagePack binary) instead of JSON for efficiency. GPU training uses dimension-grouped batching where images of the same dimensions are batched together for processing in parallel which massively improves GPU utilization compared to one-image-at-a-time processing that was causing the GPU to spike once at initialization then sit idle, though other issues are still blocking GPU training, see Known Issues section at top of README.

#### Custom Dehazing Parameters
- `omega`: Haze retention factor [0-1], lower = more dehaze (default: 0.95)
- `t0`: Min transmission [0-1], higher = less noise in thick haze (default: 0.1)  
- `patch_size`: Dark channel patch size in pixels (default: 15)
- `guided_radius`: Guided filter radius, larger = smoother (default: 60)
- `guided_eps`: Guided filter epsilon, smaller = sharper edges (default: 0.0001)

### Running Tests
```bash
cargo test -p ai-model
cargo test -p IP_functions
```

## Image Processing

- Dark Channel Prior is implemented for haze detection. Specifics are explained in comments and in the project plan.  
  - DCP-based dehazing is now implemented (12/21). Works rather well.
- TODO:  
  - Other functions still need to be implemented, especially for contrast adjustment and glare reduction. Note that these are optional per the project plan for Sprint 3 (ending 12/26).
  - Reorganize code and file structure

## Machine Learning

- Iteration 1: Linear Regression implemented as a **haze regressor** (outputs continuous haze score 0.0-1.0)
  - Uses DCP-derived features: mean dark channel, transmission stats (WIP), atmospheric intensity (WIP)
  - Can be thresholded for classification (>0.5 = "High Haze", <=0.5 = "Low Haze")
  - TODONE: (yay)
    - Model persistence implemented (1/29) - save/load trained models to JSON files as per proposer's intention and for overhead/usability reasons
    - Manual query feature implemented (1/29) - --process command loads saved model and dehazes new images with auto-selected parameters (still a basic heuristic, may need to use CNN for better results)

- Iteration 2 (WIP): Convolutional Neural Network implemented as a **haze predictor** (outputs predicted haze score) that **accepts variable image sizes** with a placeholder for DCP parameter recommendations.
  - Architecture: 4 convolutional layers with strided downsampling → Global Average Pooling→ Fully Connected layers -> Sigmoid Function to normalize haze output to [0,1]
  - Uses DCP-derived features: mean dark channel, transmission stats (WIP), atmospheric intensity (WIP)
  - Also trained on SealID dataset for comparison purposes.
  - Handles variable input image sizes using Global Average Pooling
  - Model persistence implemented (1/31) - save/load trained CNN models to .mpk files
  - Manual query feature implemented (1/31) - --process-cnn command loads saved model and dehazes new images
  - GPU acceleration implemented (2/3) - wgpu backend with dimension-grouped batching
  - **GPU TRAINING BROKEN (2/11)** - hangs at first epoch for images > 256x256, see Known Issues section for the full two-week debugging saga that ended with me giving up and pushing to GitHub
  - STATUS: WIP. CPU training works but slow. GPU training hangs on backward pass for larger images.
  - TODO: 
    - **FIX GPU TRAINING** (or just resize everything to 256x256 and accept some accuracy issues)
    - SIGNIFICANT TESTING
      - Architecture adjustments based on testing results?
    - Optimization for lower end machines
    - Further improvements to organization after fixing GPU training
    - Robustness and organization changes due to AI-generated code in some medium-importance functions (detailed in CNN implementation)
    - Full dataset training evaluation on better hardware (stuck on RTX 3070 machine due to the above issue)
  - Papers and Documentation Referenced:
    - burn documentation: https://docs.rs/burn/latest/burn/
      - burn was used due to its flexibility, efficiency, safety/robustness, and wide backend support.
      - This crate does not give prebuilt models or training pipelines, but it does provide a lot of the building blocks like tensors and backend support. 
    - Understanding CNN's better: (very helpful in combination with rust crate documentation)
      - https://www.geeksforgeeks.org/deep-learning/kernels-filters-in-convolutional-neural-network/
      - https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/
      - https://www.geeksforgeeks.org/deep-learning/cnn-introduction-to-pooling-layer/
      - https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/
      - https://optimization.cbe.cornell.edu/index.php?title=Adam
    - Papers on prior use of CNN's for haze detection and DCP-based dehazing (both accessed via ASU Library):
      - Wu, J., Liu, Z., Huang, F. et al. Adaptive haze pixel intensity perception transformer structure for image dehazing networks. Sci Rep 14, 22435 (2024). https://doi.org/10.1038/s41598-024-73866-y
      - Fazlali, H., Shirani, S., McDonald, M. et al. Cloud/haze detection in airborne videos using a convolutional neural network. Multimed Tools Appl 79, 28587–28601 (2020). https://doi.org/10.1007/s11042-020-09359-7