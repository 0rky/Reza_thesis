# Model Compression Experiments – iOS App

This iOS application is a companion app for the **Model Compression Experiments** thesis project.  
Its purpose is to provide a simple on-device interface for testing image classification models that were trained and exported from the main research pipeline.

The app currently focuses on **local inference** on iPhone using pre-exported model files, with a clean UI for selecting a model, choosing or capturing an image, and running inference directly on the device.

---

## Project Purpose

This app was built to connect the research side of the project with a practical mobile deployment scenario.

The main goals are:

- demonstrate local inference on iOS
- test exported compressed models in a real app environment
- compare different model variants from the thesis experiments
- prepare the foundation for future privacy-preserving on-device medical AI workflows

---

## Current Features

- built with **UIKit** and **Storyboard**
- image input from:
  - camera
  - photo library
- model selection through `UIPickerView`
- local inference using **ONNX Runtime**
- model metadata display in the UI, including:
  - model type
  - model size
  - input size
  - test accuracy
  - confidence
  - inference time

---

## Current Supported Models

The current app includes ONNX-based models that were successfully exported from the research experiments:

- Baseline
- Mixed Precision
- Quantization Float
- Pruning Dense
- Structured Dense

These models are included inside the app bundle under:

`Model Compression Experiments/ONNX Files/`

---

## Research Context

The app is based on the larger thesis repository, which contains experiments for:

- baseline training
- mixed precision
- quantization
- quantization-aware training (QAT)
- unstructured pruning
- structured pruning

Not all trained models are currently deployable in the iOS app yet.  
At this stage, the app only includes models that have a working ONNX deployment path.

Some other model variants currently exist only as PyTorch checkpoint files (`.pth`) and require a separate conversion pipeline before they can be integrated into the app.

---

## Tech Stack

- **Swift**
- **UIKit**
- **Storyboard**
- **ONNX Runtime for iOS**
- **Xcode**

---

## App Structure

`Model Compression Experiments - iOS APP/`  
`├── Model Compression Experiments.xcodeproj`  
`└── Model Compression Experiments/`  
`    ├── AppDelegate.swift`  
`    ├── SceneDelegate.swift`  
`    ├── HomeViewController.swift`  
`    ├── ONNXInferenceManager.swift`  
`    ├── ModelInfo.swift`  
`    ├── Info.plist`  
`    ├── Base.lproj/`  
`    ├── Assets.xcassets/`  
`    └── ONNX Files/`

---

## Current UI Flow

1. Select a model from the picker
2. Choose an image from the photo library or capture one with the camera
3. Run inference
4. View model-related information and prediction results on the main screen

---

## Important Notes

- This app is currently a **research prototype**
- the UI and deployment flow are under active development
- the included models were trained in the main thesis pipeline and then prepared for mobile-side testing
- not all compression methods from the thesis are fully integrated into iOS yet
- current inference quality depends heavily on image preprocessing and similarity to the original training distribution

---

## Future Work

Planned next steps include:

- improve image preprocessing so inference is less sensitive to background and framing
- add support for more compressed model variants
- integrate non-ONNX models through a separate Core ML conversion pipeline
- improve result interpretation in the UI
- benchmark inference speed more systematically on-device
- extend the app toward privacy-preserving real-world use cases

---

## Repository Context

This iOS app is part of the broader **Model Compression Experiments** thesis repository, which contains both:

- the model training / evaluation pipeline
- the mobile deployment prototype

The repository is currently organized so that the research experiments and the iOS app are kept as separate folders inside the same Git project.

---

## Status

Current status of the app:

- UI structure implemented
- ONNX Runtime integrated
- several ONNX models added to the app bundle
- local image selection and camera flow implemented
- basic inference pipeline under active testing and refinement

---

## Author

**Reza Kashkoul**  
M.S. in Computer Science  
Southern Illinois University Edwardsville

---
