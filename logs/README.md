# Experiment Logs

This folder contains comprehensive logs and figures for different experiments conducted with the Point Transformer model. Each experiment explores various configurations and parameters to evaluate performance and behavior.

## Studies

### 1. Study: Model Depth Exploration

- **Objective**: Investigate the impact of varying model depth on performance.
- **Model Depth Range**: 2 to 6

### 2. Study: Influence of the Optimizer
- **Objective**: This study aims to assess the influence of different optimizers on the training dynamics and performance of the Point Transformer model.

- **Optimizers**: 
  - Adam Optimizer
  - SGD Optimizer
  - Madgrad Optimizer
  - Adagrad Optimizer


### 3. Study: Sampling Method

- **Objective**: Evaluate the effect of sampling techniques on model performance.
- **Sampling Techniques**:
  - FPS (Farthest Point Sampling)
  - Density-Based Sampling

### 4. Study: Transition Down Convolution Layer

- **Objective**: Compare the performance of transition down convolution layer modifications.
- **Transition Down Modification**: 
    - PointNet++ method 
    - Original paper

## Accessing Logs

All experiment logs are stored in their respective directories within this folder.