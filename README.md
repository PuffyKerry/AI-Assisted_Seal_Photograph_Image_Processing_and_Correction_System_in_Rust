Status as of 12/17: DCP fleshed out and testing of it started. Linear Regression added, not tested yet.  
TODO:  
  - Improve formatting of README and documentation in general.  
  - IMPROVE FILE STRUCTURE ASAP, I really need to make it more understandable.  
  
Dataset is here: https://etsin.fairdata.fi/dataset/22b5191e-f24b-4457-93d3-95797c900fc0  
  - Credit goes to researchers as mentioned:  
    - Nepovinnykh, E., Eerola, T., Biard, V., Mutka, P., Niemi, M., Kunnasranta, M. and Kälviäinen, H., 2022. SealID: Saimaa ringed seal re-identification dataset. Sensors, 22(19), p.7602.
    - Nepovinnykh, E., Chelak, I., Eerola, T. and Kälviäinen, H., 2022. Norppa: Novel ringed seal re-identification by pelage pattern aggregation. arXiv preprint arXiv:2206.02498.  
  - Dataset needs to be downloaded separately. It is not included in this repository.
  - As of now, the dataset is not in use for testing beyond a single image.  
  
Image Processing:  
  - Dark Channel Prior is implemented for haze detection. Specifics are explained in comments and in the project plan.  
    - To run test: cargo run -p ai-model  
      - This is currently a very temporary test, hence it being run in the ai-model crate.  
  - TODO:  
    - Other functions still need to be implemented, especially for dehazing, contrast adjustment, and glare reduction.
    - Reorganize code and file structure  
  
Machine Learning:  
  - Linear Regression is implemented, but not tested yet.
    - To run: not possible yet. Needs testing first.
  - TODO: training testing of model. Refinement based on dataset.  
