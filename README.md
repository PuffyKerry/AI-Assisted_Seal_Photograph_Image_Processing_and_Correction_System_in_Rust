# AI-Assisted Seal Photograph Image Processing and Correction System

Note: much of the README was AI-generated based on my comments in the code.
Status as of 12/26: Linear regression haze regressor implemented, CLI with custom dehazing parameters added. Demos added. 
- Summary of research slightly delayed due to most of the summary contents being in various places ranging from code comments to emails to the Project Plan. As rationale for each decision is well-documented in each place where the decision's results were implemented, I believe the summary can wait another day, and also due to not having factored in a couple days of delay due to Christmas Eve and Christmas Day.

TODO:  
  - Improve formatting of README and documentation in general.  
  - IMPROVE FILE STRUCTURE ASAP, I really need to make it more understandable.  

## Dataset Setup

The SealID dataset (~2GB, 2000+ images) is required for training but is too large for GitHub.

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
# Run ML training demo (default)
cargo run -p ai-model

# Run IP engine tests (dehazing on test images)
cargo run -p ai-model -- --ip-tests

# Train on full dataset (requires dataset setup)
cargo run -p ai-model -- --train-full

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

- Linear Regression implemented as a **haze regressor** (outputs continuous haze score 0.0-1.0)
  - Uses DCP-derived features: mean dark channel, transmission stats, atmospheric intensity
  - Can be thresholded for classification (>0.5 = "High Haze", <=0.5 = "Low Haze")
- TODO: 
  - Full dataset training and evaluation
  - Model persistence (save/load trained models)
  - Manual query feature (short wrapper for a function call, minor feature)
