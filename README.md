# Model Compression Experiments

This repository contains two connected parts of the same thesis-oriented project:

1. a **training and evaluation pipeline** for model compression experiments  
2. an **iOS application** for testing selected exported models through on-device inference

The overall goal of the project is to study how model compression techniques affect classification performance, model size, sparsity behavior, and practical deployability to smaller devices such as iPhones.

---

## Repository Overview

This repository is currently organized into two main folders:

- `model-compression-experiments/`  
  Contains the main thesis experiments, shared training code, reports, plots, and experiment outputs.

- `Model Compression Experiments - iOS APP/`  
  Contains the iOS application project used for local on-device inference with selected exported models.

---

## Project Goal

The main purpose of this project is to compare a baseline CNN against several compression and efficiency techniques while also preparing the project for mobile deployment.

The compression methods studied in this repository include:

- Baseline
- Mixed Precision
- Post-Training Dynamic Quantization
- Quantization-Aware Training (QAT)
- Unstructured Weight Pruning
- Structured Pruning

The project focuses on understanding how these techniques affect:

- classification accuracy
- model size
- sparsity
- training and inference behavior
- deployability to smaller devices
- practical mobile-side testing

---

## Part 1: Training and Evaluation Pipeline

The training and evaluation side of the repository is located under:

`model-compression-experiments/`

This part contains the thesis experiments built around a shared CNN architecture and FashionMNIST.

### Dataset

All experiments use the FashionMNIST dataset with the same preprocessing pipeline:

- Resize to 224 × 224
- Convert to tensor
- Normalize using FashionMNIST mean/std

The dataset is stored locally under:

`data/sampleNetwork`

This folder is excluded from Git tracking.

### Shared Code Design

To reduce duplication, common components are centralized in shared files such as:

- `shared/dataset.py`
- `shared/model.py`

These shared components are used by multiple experiment cases, while the QAT case keeps a dedicated model definition because quantization-aware training needs extra modules such as:

- `QuantStub`
- `DeQuantStub`
- explicit `ReLU` modules
- module fusion through `fuse_model()`

### Experiment Cases

The current experiment cases include:

- Baseline
- Mixed Precision
- Quantization
- QAT
- Pruning
- Structured Pruning

Each experiment has its own folder structure, typically including:

- `code/`
- `outputs/`
- `checkpoints/`

### Tracked Outputs

The repository mainly tracks summary and analysis artifacts such as:

- `.txt`
- `.csv`
- `.png`

This keeps the repository lighter while preserving the useful experiment results, reports, and visualizations.

Large training artifacts such as datasets and model checkpoints are intentionally excluded from Git where appropriate.

---

## Part 2: iOS App for On-Device Inference

The mobile deployment side of the repository is located under:

`Model Compression Experiments - iOS APP/`

This iOS application is a companion app for the thesis project and is used to test selected exported models directly on an iPhone.

### Current Purpose of the App

The iOS app was built to connect the research side of the project with a practical mobile deployment scenario.

Its current goals are:

- demonstrate local inference on iOS
- test exported compressed models in a real app environment
- compare selected model variants inside a mobile UI
- prepare the foundation for future privacy-preserving on-device AI workflows

### Current Features

The app currently includes:

- UIKit + Storyboard-based UI
- image selection from photo library
- image capture using camera
- model selection through `UIPickerView`
- local inference using ONNX Runtime
- display of model-related metadata such as:
  - model type
  - model size
  - input size
  - test accuracy
  - confidence
  - inference time

### Current Supported Models in the App

At the current stage, the app includes ONNX-based models that were successfully exported and integrated into the iOS bundle.

These include:

- Baseline
- Mixed Precision
- Quantization Float
- Pruning Dense
- Structured Dense

Not all training-side model variants are deployable inside the app yet.

Some experiment variants still exist only as PyTorch checkpoint files (`.pth`) and require a separate conversion pipeline before they can be integrated into the iOS application.

### Tech Stack for the App

The iOS application currently uses:

- Swift
- UIKit
- Storyboard
- ONNX Runtime for iOS
- Xcode

---

## Current Status

### Training / Thesis Side

The thesis pipeline currently includes:

- modularized code organization
- shared dataset/model logic where appropriate
- separate experiment cases
- generated reports and plots
- experiment outputs for comparison across multiple compression methods

### iOS Side

The iOS app currently includes:

- implemented UI structure
- ONNX Runtime integration
- selected ONNX models added to the app bundle
- local image selection and camera flow
- an inference pipeline under active testing and refinement

---

## Future Work

Planned next steps for the project include:

- improve image preprocessing so mobile inference becomes less sensitive to background and framing
- add support for more compressed model variants
- integrate non-ONNX models using a separate Core ML conversion pipeline
- improve result interpretation in the mobile UI
- benchmark on-device inference speed more systematically
- move toward privacy-preserving real-world mobile use cases
- extend the project from prototype-level integration toward stronger deployment quality

---

## Repository Structure

`./`  
`├── model-compression-experiments/`  
`│   ├── shared/`  
`│   ├── baseline/`  
`│   ├── mixed_precision/`  
`│   ├── quantization/`  
`│   ├── qat/`  
`│   ├── pruning/`  
`│   ├── structured_pruning/`  
`│   ├── reports/`  
`│   ├── README.md`  
`│   └── ...`  
`│`  
`├── Model Compression Experiments - iOS APP/`  
`│   ├── Model Compression Experiments.xcodeproj`  
`│   └── Model Compression Experiments/`  
`│       ├── AppDelegate.swift`  
`│       ├── SceneDelegate.swift`  
`│       ├── HomeViewController.swift`  
`│       ├── ONNXInferenceManager.swift`  
`│       ├── ModelInfo.swift`  
`│       ├── Base.lproj/`  
`│       ├── Assets.xcassets/`  
`│       └── ONNX Files/`  
`│`  
`└── .gitignore`

---

## Notes

This repository is still under active development.

The training side and the iOS side are intentionally kept as separate folders inside the same Git project so that the research workflow and the deployment workflow can evolve together while remaining logically separated.

---

## Author

**Reza Kashkoul**  
M.S. in Computer Science  
Southern Illinois University Edwardsville
