# AI-Assisted Seal Photograph Image Processing and Correction System
# NOTE: DOCUMENTATION IN FILES EXCEPT README.md IS OUTDATED AND MAY BE INCORRECT. WILL NEED UPDATING AS OF 3/4/2026.

## Summary

A Rust-based image processing and machine learning system for enhancing seal photographs taken in hazy, foggy, or underwater conditions. The system can **detect haze levels, contrast deficits, and brightness issues** in images using a trained triple-output CNN or linear regression model, **automatically dehaze** them using the Dark Channel Prior algorithm with ML-suggested parameters, **enhance local contrast** using luminance-preserving CLAHE, and **correct brightness** using auto-estimated gamma correction. Trained on the SealID dataset (~2000+ seal images with varying haze conditions) using GPU-accelerated training with image tiling for large images. Includes a **web server** with an interactive browser UI for uploading and processing images via a REST API.

### Key Features
- **Haze Detection & Dehazing:** Dark Channel Prior (DCP) pipeline with guided filter refinement for automatic haze removal
- **ML-Guided Parameter Selection:** Triple-output CNN (Iteration 2) predicts haze level, CLAHE contrast deficit, and brightness deficit — each drives its own parameter suggestions independently. Linear regression (Iteration 1) predicts haze level only.
- **CLAHE Contrast Enhancement:** Luminance-preserving local contrast enhancement that brings out detail in seal fur, textures, and backgrounds without over-amplifying noise or shifting colors
- **Gamma Brightness Correction:** Power-law gamma correction with auto-estimation from mean luminance — addresses dawn/dusk/overcast lighting conditions
- **Full Processing Pipeline:** DCP → CLAHE → Gamma with attenuated stacking to prevent over-correction when chaining operations
- **GPU-Accelerated CNN Training:** wgpu backend with dimension-grouped batching and image tiling for large images (RTX 3070 tested)
- **Model Persistence:** Train once, save model, process new images without retraining
- **Web Server & REST API:** Interactive browser UI with drag-and-drop upload, per-parameter tooltips, and real-time processing via JSON API endpoints
- **CLI Interface:** Full command-line interface for training, processing, testing, model comparison, and ablation studies
- **Model Comparison & Ablation Study:** Data-driven architecture selection — compare regressor vs CNN, and evaluate Small/Medium/Large CNN variants

### Quick Start
```bash
# Dehaze an image with default parameters
cargo run -p ai-model --release -- --dehaze image.jpg

# Enhance contrast with CLAHE
cargo run -p ai-model --release -- --clahe image.jpg

# Correct brightness with auto-estimated gamma
cargo run -p ai-model --release -- --gamma image.jpg

# Train CNN on GPU and save model (requires dataset)
cargo run -p ai-model --release -- --train-cnn-gpu-save cnn_model

# Process image with trained CNN model (DCP + CLAHE + Gamma pipeline)
cargo run -p ai-model --release -- --process-cnn-gpu cnn_model input.jpg output.jpg

# Start the web server (interactive UI at http://localhost:8080)
cargo run -p web_server
```

---

**What is this?** A Rust-based image processing system for underwater/hazy seal photographs that uses machine learning to detect haze levels, contrast deficits, and brightness issues, then automatically applies dehazing, contrast enhancement, and brightness corrections. Also includes a web server with an interactive browser UI.
**What project?** Built as an Arizona State University (ASU, *insert something about #1 in INNOVATION here*) Barrett Honors College Honors Thesis (for my B.S. in Computer Science, Cybersecurity concentration) to explore AI-assisted image processing in Rust, expanding my Capstone Project with GDMS to "Develop a Web Server and Packet Sniffer in Rust" as a service mounted to the web server. 
**Why Rust?** Safety and efficiency are key to this project, and exploring the state of Rust's machine learning ecosystem was a good opportunity to learn more about Rust AND ML.
**Why seals?** The SealID dataset provides a good variety of hazy images (snow, mist, fog) for testing haze detection and correction algorithms, plus seals are cute and the dataset was freely available for research purposes, and lastly the scope is rather unique. 

**Two iterations of code so far (ML + image processing functions are both improved in each iteration):**
1. **Iteration 1 (Complete):** Linear regression model trained on Dark Channel Prior features to predict haze levels and suggest dehazing parameters, can feed it images via CLI after loading a pretrained / saved / persistent model.
2. **Iteration 2 (Complete — GPU Training, Triple-Output CNN, Web Server):** Convolutional Neural Network with three output heads (DCP haze, CLAHE contrast deficit, brightness deficit) for improved accuracy and multi-defect correction. GPU training with image tiling and memory management. Model comparison and ablation study tools.
3. **Image Processing Functions:** DCP-based dehazing (Iteration 1), CLAHE contrast enhancement (added 3/4/2026), gamma brightness correction (added after 3/4/2026). Full pipeline: DCP → CLAHE → Gamma with attenuated stacking.
4. **Web Server (new):** REST API with interactive browser UI, ported thread pool and strategy-pattern router from General Dynamics capstone project. Endpoints for each IP operation and combined pipeline.

Note: some of the README was AI-generated based on my comments in the code. Less of it is AI generated now than before, excluding the summary of my GPU training issues below, which was based on my own recollection of what issues I was having.   

Status as of 4/4/2026: Iteration 1 complete. Iteration 2 triple-output CNN (DCP + CLAHE + brightness) GPU training **working** with image tiling for oversized images — trains on 430+ images with ~0.002 MSE. CLAHE contrast enhancement added (luminance-preserving). Gamma brightness correction added. Web server with interactive UI and REST API complete. Model comparison (regressor vs CNN) and CNN architecture ablation study (Small/Medium/Large) implemented. Full processing pipeline (DCP → CLAHE → Gamma) with attenuated stacking for both CLI and web API. Next steps: polishing, more testing, update other documentation inside of code files.

## GPU Training Journey (First written 2/11, Irrelevant by 2/19, but useful for reference)

### The Problem: GPU Training Hangs at First Epoch

**TLDR:** GPU training hung during backward pass for images larger than 256x256. Forward pass worked, loss computed, but `loss.backward()` never returned. **RESOLVED 2/19** by upgrading burn 0.16→0.20.1, adding `into_scalar()` to force wgpu fusion flushes before backward pass, using `fork()` to detach autodiff graphs between batches, and tiling oversized images (>400K pixels) into ~350×350 tiles.

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

**Solution (2/18–2/25):** The fix required multiple changes working together:
1. **Upgraded burn 0.16 → 0.20.1** (and wgpu 26 to match) — newer wgpu backend handles larger shader compilations better
2. **`into_scalar()` before `backward()`** — forces the wgpu fusion backend to flush pending GPU work before starting the backward pass, preventing the hang
3. **`fork()` after each optimizer step** — detaches the autodiff graph so gradient history doesn't accumulate in GPU memory across batches
4. **Scoped tensor drops** — forward pass tensors are dropped before the next batch starts, freeing GPU memory
5. **Image tiling** — images >400K pixels are split into ~350×350 non-overlapping tiles (each inherits the parent's haze label since haze is scene-wide), so nothing is skipped

**Training Results (2/25, 50 epochs):**
- 430 images trained per epoch (282 directly + 148 oversized images tiled into smaller patches)
- ~191 seconds per epoch (~3.2 minutes) on RTX 3070
- Final MSE: ~0.002 (test set MSE: ~0.003)
- Total training time: ~2.7 hours
- No OOM crashes across all 50 epochs

**If hitting similar wgpu/burn backward pass hangs:** The key insight is that `into_scalar()` forces a GPU sync that prevents the fusion backend from building up an impossibly large fused kernel. Without it, the backward pass tries to compile a single massive shader that exceeds GPU limits.

**Also, please gaze upon the seals while you're here. They are cute.**
---

TODO:  
  - ~~Figure out why GPU training hangs~~ (FIXED 2/19)
  - ~~Add more IP functions~~ CLAHE added (3/4), gamma brightness correction added, luminance-preserving CLAHE
  - ~~Web server integration~~ DONE — REST API with interactive browser UI
  - ~~Model comparison~~ DONE — `--compare-models` compares regressor vs CNN on full query dataset
  - ~~Ablation study~~ DONE — `--ablation` evaluates Small/Medium/Large CNN architectures
  - More testing for Iteration 2 CNN — full evaluation on larger/different datasets
  - Improve formatting of README and documentation in general
  - Make more improvements to organization if possible
  - Consider: seal-specific features beyond haze detection (species classification is a reach goal)
  - Update documentation in code files (currently outdated as of 3/4/2026)

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
# Outputs: OUTPUT_dcp.jpg, OUTPUT.jpg (DCP+CLAHE+Gamma), OUTPUT_clahe.jpg, OUTPUT_gamma.jpg
cargo run -p ai-model --release -- --process-cnn-gpu cnn_model "dataset/SealID/full images/source_query/input.jpg" output.jpg

# Dehaze a specific image with default parameters
cargo run -p ai-model -- --dehaze path/to/image.jpg

# Dehaze with custom DCP parameters
# Usage: --dehaze-custom FILE omega t0 patch_size guided_radius guided_eps
cargo run -p ai-model -- --dehaze-custom image.jpg 0.75 0.25 15 15 0.0001

# Enhance contrast with CLAHE (default parameters: 8x8 grid, clip_limit=2.0)
cargo run -p ai-model -- --clahe path/to/image.jpg

# Enhance contrast with custom CLAHE parameters
# Usage: --clahe-custom FILE grid_h grid_w clip_limit
cargo run -p ai-model -- --clahe-custom image.jpg 8 8 4.0

# Correct brightness with auto-estimated gamma
cargo run -p ai-model -- --gamma path/to/image.jpg

# Correct brightness with custom gamma value (< 1.0 brightens, > 1.0 darkens)
cargo run -p ai-model -- --gamma-custom image.jpg 0.6

# Compare linear regression vs CNN on the full query dataset
cargo run --release -p ai-model -- --compare-models haze_model.json CNN_GPU_TILED

# Run CNN architecture ablation study (Small/Medium/Large variants)
cargo run --release -p ai-model -- --ablation

# Show help
cargo run -p ai-model -- --help
```

### Web Server

The web server exposes all image processing functions as a REST API with an interactive browser-based upload UI. Architecture is ported from the General Dynamics Capstone Project (same thread pool, strategy-pattern router, TCP handling).

```bash
# Start the web server on default port 8080
cargo run -p web_server

# Start on a custom port
cargo run -p web_server -- --port 3000
```

Then open `http://localhost:8080` in your browser for the interactive UI, or use the JSON API directly:

#### API Endpoints

| Method | Path           | Description                                      |
|--------|----------------|--------------------------------------------------|
| GET    | `/`            | Interactive upload UI (drag-and-drop, parameter controls, tooltips) |
| GET    | `/api/health`  | Health check — returns status and available endpoints |
| POST   | `/api/dehaze`  | DCP dehazing — optional params: `omega`, `t0`, `patch_size`, `guided_radius`, `guided_eps` |
| POST   | `/api/clahe`   | CLAHE contrast enhancement — optional params: `grid_h`, `grid_w`, `clip_limit` |
| POST   | `/api/gamma`   | Gamma brightness correction — optional param: `gamma` (auto-estimated if omitted) |
| POST   | `/api/process` | Full pipeline (DCP → CLAHE → Gamma) with attenuated stacking — returns all individual stages too |

All POST endpoints accept JSON with a base64-encoded image:
```json
{ "image": "<base64-encoded image>", "omega": 0.75, "t0": 0.25, ... }
```
And return JSON with the processed result:
```json
{ "image": "<base64-encoded result>", "width": 1920, "height": 1080, "operation": "dehaze", "params": { ... } }
```

The `/api/process` endpoint returns additional fields: `dehaze_only`, `clahe_only`, `gamma_only` (each base64-encoded) for comparison.

#### Running the Test Suite
```powershell
# Start the server first, then in another terminal:
.\test_all_endpoints.ps1
```
Tests all 11 scenarios: health check, home page, each operation with default and custom parameters, full pipeline, 404 handling, and bad request handling.

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
Same workflow as Iteration 1, but with the CNN model. GPU version is MUCH faster and recommended if you have a dedicated graphics card. The CNN predicts three scores (DCP haze, CLAHE contrast deficit, brightness deficit) and applies a full DCP → CLAHE → Gamma pipeline:

#### CPU Training (slow, for devices without GPU)
```bash
# Step 1: Train and save CNN (do this once - warning: slow on CPU)
cargo run -p ai-model -- --train-cnn-save cnn_model

# Step 2: Process new images with saved CNN model
cargo run -p ai-model -- --process-cnn cnn_model "dataset/SealID/full images/source_query/input.jpg" output.jpg
```

#### GPU Training (fast, recommended)
```bash
# Step 1: Train and save CNN on GPU (do this once - much faster)
cargo run -p ai-model --release -- --train-cnn-gpu-save cnn_model

# Step 2: Process new images with saved CNN model on GPU
# Generates: output_dcp.jpg, output.jpg (full pipeline), output_clahe.jpg, output_gamma.jpg
cargo run -p ai-model --release -- --process-cnn-gpu cnn_model "dataset/SealID/full images/source_query/input.jpg" output.jpg
```

Note: Images larger than 400K pixels (~630×630) are automatically tiled into ~350×350 patches during training so all images contribute. See GPU Training Journey section for details.

Note: CNN models use `.mpk` format (MessagePack binary) instead of JSON for efficiency. GPU training uses dimension-grouped batching where images of the same dimensions are batched together for processing in parallel which massively improves GPU utilization compared to one-image-at-a-time processing that was causing the GPU to spike once at initialization then sit idle, though other issues are still blocking GPU training, see Known Issues section at top of README.

#### Model Comparison
Compare the linear regression (Iteration 1) against the CNN (Iteration 2) on the full query dataset:
```bash
cargo run --release -p ai-model -- --compare-models haze_model.json CNN_GPU_TILED
```
This computes MSE/MAE for both models on all query images, measures inference speed, and processes sample images with both pipelines for visual comparison. The CNN additionally predicts CLAHE deficit and brightness deficit (the regressor only predicts DCP haze).

#### Ablation Study
Evaluate three CNN architecture variants to justify the chosen architecture with data:
```bash
cargo run --release -p ai-model -- --ablation
```
Variants:
- **Small:** 8 → 16 → 32 → 64 channels, FC 32, dropout 0.2 (~18K params)
- **Medium:** 16 → 32 → 64 → 128 channels, FC 64, dropout 0.3 (~73K params — production model)
- **Large:** 32 → 64 → 128 → 256 channels, FC 128, dropout 0.4 (~290K params)

Each variant runs as a **separate child process** so the OS reclaims all GPU memory between runs, preventing OOM crashes. Results are collected and presented in a comparison table.

#### Custom Dehazing Parameters
- `omega`: Haze retention factor [0-1], lower = more dehaze (default: 0.75)
- `t0`: Min transmission [0-1], higher = less noise in thick haze (default: 0.25)  
- `patch_size`: Dark channel patch size in pixels (default: 15)
- `guided_radius`: Guided filter radius, larger = smoother (default: 15)
- `guided_eps`: Guided filter epsilon, smaller = sharper edges (default: 0.0001)

#### Custom CLAHE Parameters
- `grid_h`: Number of tile rows, more = more local contrast (default: 8)
- `grid_w`: Number of tile columns (default: 8)
- `clip_limit`: Contrast limit multiplier, higher = stronger enhancement (default: 2.0, typical range: 1.0–4.0)

#### Custom Gamma Parameters
- `gamma`: Power-law exponent (< 1.0 brightens, > 1.0 darkens, 1.0 = no change)
- Auto mode estimates optimal gamma from image mean luminance using `gamma = ln(target) / ln(mean_lum)` targeting ~0.45 mean luminance
- Clamped to [0.5, 2.0] to prevent extreme corrections

### Running Tests
```bash
cargo test -p ai-model
cargo test -p IP_functions
```

## Project Structure

```
project_root/
├── Cargo.toml                   # Workspace: AI-Model, IP_functions, web_server
├── AI-Model/                    # ML training, inference, and CLI
│   └── src/
│       ├── main.rs              # CLI entry point with all command handlers
│       ├── extraction.rs        # DCP feature extraction + CLAHE/brightness deficit labels
│       ├── training.rs          # Linear regression training + persistence
│       ├── linear_regression.rs # Regressor model implementation
│       ├── ip_tests.rs          # Image processing test functions
│       └── iteration_2_CNN/     # CNN implementation
│           ├── mod.rs           # Public API: training, inference, parameter suggestion
│           └── cnn_detection.rs # CNN architecture, GPU training, tiling, triple-output
├── IP_functions/                # Image processing library
│   └── src/
│       ├── lib.rs               # Module declarations
│       ├── dcp.rs               # Dark Channel Prior computation
│       ├── atmospheric.rs       # Atmospheric light estimation
│       ├── transmission.rs      # Transmission map estimation
│       ├── guided_filter.rs     # Guided filter for transmission refinement
│       ├── radiance.rs          # Radiance recovery (scene reconstruction)
│       ├── dehaze.rs            # DCP dehazing pipeline entry point
│       ├── clahe.rs             # CLAHE algorithm (per-tile histogram equalization)
│       ├── enhance.rs           # Luminance-preserving CLAHE wrapper (prevents color shifts)
│       ├── gamma.rs             # Gamma correction + auto-estimation from mean luminance
│       └── brightness.rs        # Brightness correction pipeline (gamma per channel)
├── web_server/                  # REST API + interactive browser UI
│   └── src/
│       ├── main.rs              # Server entry, thread pool, TCP listener
│       ├── router.rs            # Strategy-pattern router (ported from GD capstone)
│       ├── request.rs           # HTTP request parser
│       ├── response.rs          # HTTP response builder
│       ├── handlers.rs          # API endpoint handlers + embedded HTML UI
│       └── convert.rs           # Image ↔ ndarray ↔ base64 conversions
├── dataset/                     # SealID dataset (not in repo, see Dataset Setup)
└── test_all_endpoints.ps1       # PowerShell test suite for web server API
```

## Image Processing

- **Dark Channel Prior (DCP)** is implemented for haze detection and dehazing. Specifics are explained in comments and in the project plan.  
  - DCP-based dehazing is implemented (12/21) and works well.
  - Pipeline: Dark Channel → Atmospheric Light Estimation → Transmission Map → Guided Filter Refinement → Radiance Recovery
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** is implemented for local contrast enhancement (3/4/2026).
  - **Luminance-preserving:** CLAHE is applied only to the luminance channel, then RGB is scaled by the luminance ratio to preserve original color balance. Per-channel RGB CLAHE was found to destroy color balance ("acid trip" effect).
  - Divides image into tiles, equalizes histogram per tile with a contrast clip limit, then bilinearly interpolates between tiles for smooth output
  - Particularly useful for seal photos where the subject (dark seal) is against a bright hazy background — enhances fur texture and body detail without blowing out the background
  - Default: 8×8 grid, clip limit 2.0; customizable via CLI and web API
- **Gamma Brightness Correction** is implemented for global brightness adjustment.
  - Addresses the "Lighting" defect category: dawn, overcast, and other low-light conditions
  - Power-law transform: `output = input ^ gamma` (gamma < 1 brightens, gamma > 1 darkens)
  - Auto-estimation from mean luminance: `gamma = ln(0.45) / ln(mean_lum)`, clamped to [0.5, 2.0]
  - Differs from CLAHE: gamma adjusts global brightness uniformly, CLAHE adjusts local contrast per tile. A dark foggy photo may need both.
- **Full Pipeline:** DCP → CLAHE → Gamma with **attenuated stacking** — when chaining operations, CLAHE clip limit is reduced to 70% and gamma is pulled 50% toward identity to prevent over-correction. Available via CLI (`--process-cnn-gpu`) and web API (`/api/process`).
- TODO:  
  - More IP functions: white balance correction, unsharp masking/sharpening, noise reduction
  - Reorganize code and file structure

## Machine Learning

- Iteration 1: Linear Regression implemented as a **haze regressor** (outputs continuous haze score 0.0-1.0)
  - Uses DCP-derived features: mean dark channel, transmission stats, atmospheric intensity, low transmission ratio
  - Can be thresholded for classification (>0.5 = "High Haze", <=0.5 = "Low Haze")
  - TODONE: (yay)
    - Model persistence implemented (1/29) - save/load trained models to JSON files as per proposer's intention and for overhead/usability reasons
    - Manual query feature implemented (1/29) - --process command loads saved model and dehazes new images with auto-selected parameters (still a basic heuristic, may need to use CNN for better results)

- Iteration 2 (Complete): Convolutional Neural Network implemented as a **triple-output predictor** that outputs DCP haze score, CLAHE contrast deficit, and brightness deficit — each driving its own parameter suggestions independently. **Accepts variable image sizes** with Global Average Pooling.
  - Architecture: 4 convolutional layers with strided downsampling → Global Average Pooling → Fully Connected layers → 3 sigmoid outputs (DCP haze, CLAHE deficit, brightness deficit)
  - Production architecture: 16 → 32 → 64 → 128 channels, FC 64, dropout 0.3 (~73K params) — selected via ablation study
  - Handles variable input image sizes using Global Average Pooling
  - Model persistence implemented (1/31) - save/load trained CNN models to .mpk files
  - Manual query feature implemented (1/31) - --process-cnn command loads saved model and processes new images
  - GPU acceleration implemented (2/3) - wgpu backend with dimension-grouped batching
  - **GPU TRAINING FIXED (2/19-25)** - required burn upgrade, fusion flush via `into_scalar()`, autodiff graph detachment via `fork()`, and image tiling for oversized images. See GPU Training Journey section.
  - **Triple-output CNN:** Predicts DCP haze score, CLAHE contrast deficit, and brightness deficit simultaneously. Each output drives its own parameter suggestion function independently.
  - **Full processing pipeline:** DCP → CLAHE → Gamma with attenuated stacking. CNN-suggested parameters for each stage.
  - **Training Results:** ~0.002 MSE on training set, ~0.003 MSE on test set, ~2.7 hours for 50 epochs on RTX 3070
  - **Model Comparison:** `--compare-models` computes MSE/MAE for regressor vs CNN on the full query dataset, including inference speed and visual sample outputs
  - **Ablation Study:** (current non-functional) `--ablation` evaluates Small/Medium/Large CNN variants as separate child processes, producing a comparison table for data-driven architecture selection
  - STATUS: Working. GPU training stable. Images >400K pixels are automatically tiled. Web server integration complete.
  - TODO: 
    - SIGNIFICANT TESTING
      - Architecture adjustments based on testing results?
    - Optimization for lower end machines
    - Further improvements to organization after testing
    - Robustness and organization changes due to AI-generated code in some medium-importance functions (detailed in CNN implementation)
    - Evaluate whether tiling vs downscaling vs other approaches give better accuracy for large images
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
    - CLAHE reference:
      - Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization." Graphics Gems IV, Academic Press, 1994, pp. 474–485.
    - Gamma correction:
      - Standard sRGB gamma model, any digital imaging textbook
