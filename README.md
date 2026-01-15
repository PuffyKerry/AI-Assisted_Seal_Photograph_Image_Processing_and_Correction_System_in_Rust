# AI-Assisted Seal Photograph Image Processing and Correction System

Note: much of the README was AI-generated based on my comments in the code.
Status as of 1/14: First HEAVILY WIP version of Iteration 2. CNN for haze detection is implemented, pipeline is mostly working, but needs a lot of cleanup, SIGNIFICANTLY more testing, more documentation, a faster machine for testing/developmen, and perhaps some more/less layers in the architecture depending on further testing.

TODO:  
  - Flesh out Iteration 2 CNN
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

# Train Regressor on full dataset (requires dataset setup)
cargo run -p ai-model -- --train-full

# Run Convolutional Neural Network demo (only option for now)
cargo run -p ai-model -- --demo-cnn

# Running CNN on full dataset IS NOT IMPLEMENTED YET. DO NOT TRY IT. WILL BE IMPLEMENTED LATER IN ITERATION 2. 

# Dehaze a specific image with default parameters
cargo run -p ai-model -- --dehaze path/to/image.jpg

# Dehaze with custom DCP parameters
# Usage: --dehaze-custom FILE omega t0 patch_size guided_radius guided_eps
cargo run -p ai-model -- --dehaze-custom image.jpg 0.75 0.25 15 15 0.0001

# Show help
cargo run -p ai-model -- --help
```

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
  - TODO: (nice-to-haves)
    - Model persistence (save/load trained models)
    - Manual query feature (short wrapper for a function call, minor feature)

- Iteration 2 (WIP): Convolutional Neural Network implemented as a **haze predictor** (outputs predicted haze score) that **accepts variable image sizes** with a placeholder for DCP parameter recommendations.
  - Architecture: 4 convolutional layers with strided downsampling → Global Average Pooling→ Fully Connected layers -> Sigmoid Function to normalize haze output to [0,1]
  - Uses DCP-derived features: mean dark channel, transmission stats (WIP), atmospheric intensity (WIP)
  - Also trained on SealID dataset for comparison purposes.
  - Handles variable input image sizes using Global Average Pooling
  - STATUS: VERY WIP DUE TO LITTLE TESTING AND DATASET SIZE. NOT READY FOR USE. MAY ADJUST ARCHITECTURE BASED ON FURTHER TESTING.
  - TODO: 
    - SIGNIFICANT TESTING
      - Architecture adjustments based on testing results
    - Optimization for lower-end devices
    - Robustness and organization changes due to AI-generated code in some medium-importance functions (detailed in CNN implementation)
    - Full dataset training and evaluation
    - Model persistence (save/load trained models)
    - Manual query feature (short wrapper for a function call, minor feature) separate from regressor