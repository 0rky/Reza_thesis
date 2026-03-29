# Thesis Experiments

This repository contains baseline and model compression experiments for thesis preparation using FashionMNIST and a shared CNN architecture.

## Project Goal

The goal of this project is to compare a baseline CNN against several compression and efficiency techniques:

- Baseline
- Mixed Precision
- Post-Training Dynamic Quantization
- Quantization-Aware Training (QAT)
- Unstructured Weight Pruning
- Structured Pruning

The focus is on understanding how these techniques affect:

- model size
- sparsity
- inference/training behavior
- classification accuracy
- deployability to smaller devices

## Dataset

All experiments use the FashionMNIST dataset with the same preprocessing pipeline:

- Resize to 224 × 224
- Convert to tensor
- Normalize using FashionMNIST mean/std

The dataset is stored locally under:

- `data/sampleNetwork`

This folder is excluded from Git tracking.

## Shared Code Design

To avoid duplication, shared components are centralized:

- `shared/dataset.py`
- `shared/model.py`

These shared files are used by:

- `baseline`
- `mixed_precision`
- `quantization`
- `pruning`
- `structured_pruning`

The `qat` experiment uses its own model definition because quantization-aware training requires:

- `QuantStub`
- `DeQuantStub`
- explicit `ReLU` modules
- module fusion via `fuse_model()`

## Repository Structure

Repository structure:

thesis-experiments/  
├── shared/  
│   ├── dataset.py  
│   └── model.py  
├── data/  
├── baseline/  
│   ├── code/  
│   ├── outputs/  
│   └── checkpoints/  
├── mixed_precision/  
│   ├── code/  
│   ├── outputs/  
│   └── checkpoints/  
├── quantization/  
│   ├── code/  
│   ├── outputs/  
│   └── checkpoints/  
├── qat/  
│   ├── code/  
│   ├── outputs/  
│   └── checkpoints/  
├── pruning/  
│   ├── code/  
│   ├── outputs/  
│   └── checkpoints/  
├── structured_pruning/  
│   ├── code/  
│   ├── outputs/  
│   └── checkpoints/  
├── README.md  
└── .gitignore

## Experiment Descriptions

### 1. Baseline

Standard CNN training used as the reference point for all comparisons.

### 2. Mixed Precision

Uses automatic mixed precision during training to reduce memory pressure and potentially improve training throughput on supported hardware.

### 3. Quantization

Applies post-training dynamic quantization to linear layers after normal training, then compares float and quantized model size and performance.

### 4. QAT

Uses quantization-aware training so the model learns under fake-quantization behavior before conversion to a quantized model.

### 5. Pruning

Applies global unstructured magnitude-based pruning to convolutional and linear weights.

### 6. Structured Pruning

Applies structured pruning across output channels or output units using `ln_structured(..., dim=0)` and evaluates structured sparsity behavior.

## Notes About Tracking

The following are intentionally excluded from Git:

- dataset files
- model weight files
- ONNX exports
- NumPy binary arrays

Tracked outputs mainly include:

- `.txt`
- `.csv`
- `.png`

This keeps the repository lightweight while preserving experiment summaries and plots.

## Running Experiments

Each experiment is run from its own `code/` folder.

Example commands:

`cd baseline/code && python3 train.py`  
`cd mixed_precision/code && python3 train.py`  
`cd quantization/code && python3 train.py`  
`cd qat/code && python3 train.py`  
`cd pruning/code && python3 train.py`  
`cd structured_pruning/code && python3 train.py`

## Current Status

The repository currently includes:

- modularized code organization
- shared dataset/model logic where appropriate
- a separate QAT-specific model implementation
- separated outputs and checkpoints per experiment
- experiment results generated previously and placed in each case folder
