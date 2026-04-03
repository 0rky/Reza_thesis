//
//  HomeViewController.swift
//  Model Compression Experiments
//
//  Created by Reza Kashkoul on 4/2/26.
//

import UIKit
import AVFoundation
import Photos
import OnnxRuntimeBindings

class HomeViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    // MARK: - IBOutlets
    
    @IBOutlet weak var selectedModelLabel: UILabel!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var pickerView: UIPickerView!
    
    @IBOutlet weak var typeOfModelLabel: UILabel!
    @IBOutlet weak var modelSizeLabel: UILabel!
    @IBOutlet weak var inputSizeLabel: UILabel!
    @IBOutlet weak var testAccuracyLabel: UILabel!
    @IBOutlet weak var confidenceLabel: UILabel!
    @IBOutlet weak var inferenceTimeLabel: UILabel!
    
    // MARK: - Data
    
    private let classNames: [String] = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ]
    
//    private let models: [ModelInfo] = [
//        ModelInfo(
//            displayName: "Baseline",
//            methodType: "Baseline CNN",
//            modelSize: "16.7065 MB",
//            inputSize: "1 x 224 x 224",
//            testAccuracy: "88.95%",
//            checkpointFileName: "sample_network.onnx"
//        ),
//        ModelInfo(
//            displayName: "Mixed Precision",
//            methodType: "AMP / FP16 Training",
//            modelSize: "16.7065 MB",
//            inputSize: "1 x 224 x 224",
//            testAccuracy: "88.73%",
//            checkpointFileName: "mixed_precision.onnx"
//        ),
//        ModelInfo(
//            displayName: "Quantization Float",
//            methodType: "Float Model",
//            modelSize: "16.7066 MB",
//            inputSize: "1 x 224 x 224",
//            testAccuracy: "88.96%",
//            checkpointFileName: "quantization_float.onnx"
//        ),
//        ModelInfo(
//            displayName: "Quantization Dynamic",
//            methodType: "Dynamic Quantization",
//            modelSize: "16.6378 MB",
//            inputSize: "1 x 224 x 224",
//            testAccuracy: "88.93%",
//            checkpointFileName: nil
//        ),
//        ModelInfo(
//            displayName: "QAT Converted",
//            methodType: "Quantization-Aware Training",
//            modelSize: "4.2491 MB",
//            inputSize: "1 x 224 x 224",
//            testAccuracy: "90.79%",
//            checkpointFileName: nil
//        ),
//        ModelInfo(
//            displayName: "Pruning Dense",
//            methodType: "Dense Before Pruning",
//            modelSize: "16.7064 MB",
//            inputSize: "1 x 224 x 224",
//            testAccuracy: "88.63%",
//            checkpointFileName: "pruning_dense.onnx"
//        ),
//        ModelInfo(
//            displayName: "Pruning Pruned",
//            methodType: "Unstructured Pruning",
//            modelSize: "16.7065 MB",
//            inputSize: "1 x 224 x 224",
//            testAccuracy: "88.93%",
//            checkpointFileName: nil
//        ),
//        ModelInfo(
//            displayName: "Structured Dense",
//            methodType: "Dense Before Structured Pruning",
//            modelSize: "16.7065 MB",
//            inputSize: "1 x 224 x 224",
//            testAccuracy: "88.82%",
//            checkpointFileName: "structured_dense.onnx"
//        ),
//        ModelInfo(
//            displayName: "Structured Pruned",
//            methodType: "Structured Pruning",
//            modelSize: "16.7066 MB",
//            inputSize: "1 x 224 x 224",
//            testAccuracy: "89.53%",
//            checkpointFileName: nil
//        )
//    ]
    
    private let models: [ModelInfo] = [
        ModelInfo(
            displayName: "Baseline",
            methodType: "Baseline CNN",
            modelSize: "16.7065 MB",
            inputSize: "1 x 224 x 224",
            testAccuracy: "88.95%",
            checkpointFileName: "sample_network.onnx"
        ),
        ModelInfo(
            displayName: "Mixed Precision",
            methodType: "AMP / FP16 Training",
            modelSize: "16.7065 MB",
            inputSize: "1 x 224 x 224",
            testAccuracy: "88.73%",
            checkpointFileName: "mixed_precision.onnx"
        ),
        ModelInfo(
            displayName: "Quantization Float",
            methodType: "Float Model",
            modelSize: "16.7066 MB",
            inputSize: "1 x 224 x 224",
            testAccuracy: "88.96%",
            checkpointFileName: "quantization_float.onnx"
        ),
        ModelInfo(
            displayName: "Pruning Dense",
            methodType: "Dense Before Pruning",
            modelSize: "16.7064 MB",
            inputSize: "1 x 224 x 224",
            testAccuracy: "88.63%",
            checkpointFileName: "pruning_dense.onnx"
        ),
        ModelInfo(
            displayName: "Structured Dense",
            methodType: "Dense Before Structured Pruning",
            modelSize: "16.7065 MB",
            inputSize: "1 x 224 x 224",
            testAccuracy: "88.82%",
            checkpointFileName: "structured_dense.onnx"
        )
    ]
    
    private var selectedModelIndex: Int = 0
    private var selectedImage: UIImage?
    
    // MARK: - Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupViews()
    }
    
    // MARK: - IBActions
    
    @IBAction func chooseImageButtonAction(_ sender: Any) {
        handlePhotoLibraryAccess()
    }
    
    @IBAction func openCameraButtonAction(_ sender: Any) {
        handleCameraAccess()
    }
    
    @IBAction func runInferenceButtonAction(_ sender: Any) {
        guard let selectedImage else {
            showAlert(title: "No Image Selected", message: "Please choose an image or capture one with the camera.")
            return
        }
        
        let selectedModel = models[selectedModelIndex]
        
        guard let modelFileName = selectedModel.checkpointFileName else {
            showAlert(title: "Model Not Supported Yet", message: "This model is not available in ONNX format yet.")
            return
        }
        
        do {
            try ONNXInferenceManager.shared.loadModel(modelFileName: modelFileName)
            
            let result = try ONNXInferenceManager.shared.runInference(
                image: selectedImage,
                inputName: "input",
                outputName: "output"
            )
            
            let predictedClass = classNames[result.predictedIndex]
            
            confidenceLabel.text = String(format: "%.2f%% (%@)", result.confidence * 100, predictedClass)
            inferenceTimeLabel.text = String(format: "%.2f ms", result.inferenceTime)
            
            showAlert(
                title: "Inference Complete",
                message: "Predicted class: \(predictedClass)\nConfidence: \(String(format: "%.2f%%", result.confidence * 100))"
            )
            
        } catch {
            showAlert(title: "Inference Error", message: error.localizedDescription)
        }    }
}

// MARK: - Setup

private extension HomeViewController {
    
    func setupViews() {
        setupNavigationBar()
        setupPickerView()
        setupImageView()
        setupInitialLabels()
        updateModelUI()
    }
    
    func setupNavigationBar() {
        title = "Model Compression Experiments"
    }
    
    func setupPickerView() {
        pickerView.delegate = self
        pickerView.dataSource = self
        pickerView.selectRow(0, inComponent: 0, animated: false)
    }
    
    func setupImageView() {
        imageView.contentMode = .scaleAspectFit
        imageView.clipsToBounds = true
        imageView.layer.cornerRadius = 10
        imageView.layer.borderWidth = 1
        imageView.layer.borderColor = UIColor.systemGray4.cgColor
        imageView.image = UIImage(systemName: "photo")
        imageView.tintColor = .white
        imageView.backgroundColor = UIColor.white.withAlphaComponent(0.08)
    }
    
    func setupInitialLabels() {
        selectedModelLabel.text = "Choose Model"
        confidenceLabel.text = "-"
        inferenceTimeLabel.text = "-"
    }
    
    func updateModelUI() {
        let model = models[selectedModelIndex]
        
        // This label stays fixed based on your preference
        selectedModelLabel.text = "Choose Model"
        
        typeOfModelLabel.text = model.methodType
        modelSizeLabel.text = model.modelSize
        inputSizeLabel.text = model.inputSize
        testAccuracyLabel.text = model.testAccuracy
        
        confidenceLabel.text = "-"
        inferenceTimeLabel.text = "-"
    }
}

// MARK: - Permissions

private extension HomeViewController {
    
    func handleCameraAccess() {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        
        switch status {
        case .authorized:
            presentCamera()
            
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.presentCamera()
                    } else {
                        self?.showPermissionAlert(
                            title: "Camera Access Denied",
                            message: "Please enable camera access in Settings to use this feature."
                        )
                    }
                }
            }
            
        case .denied, .restricted:
            showPermissionAlert(
                title: "Camera Access Denied",
                message: "Please enable camera access in Settings to use this feature."
            )
            
        @unknown default:
            showAlert(
                title: "Error",
                message: "Unknown camera permission state."
            )
        }
    }
    
    func handlePhotoLibraryAccess() {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        
        switch status {
        case .authorized, .limited:
            presentPhotoLibrary()
            
        case .notDetermined:
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { [weak self] newStatus in
                DispatchQueue.main.async {
                    switch newStatus {
                    case .authorized, .limited:
                        self?.presentPhotoLibrary()
                    case .denied, .restricted:
                        self?.showPermissionAlert(
                            title: "Photo Library Access Denied",
                            message: "Please enable photo library access in Settings to use this feature."
                        )
                    @unknown default:
                        self?.showAlert(
                            title: "Error",
                            message: "Unknown photo library permission state."
                        )
                    }
                }
            }
            
        case .denied, .restricted:
            showPermissionAlert(
                title: "Photo Library Access Denied",
                message: "Please enable photo library access in Settings to use this feature."
            )
            
        @unknown default:
            showAlert(
                title: "Error",
                message: "Unknown photo library permission state."
            )
        }
    }
}

// MARK: - Image Picking

private extension HomeViewController {
    
    func presentPhotoLibrary() {
        guard UIImagePickerController.isSourceTypeAvailable(.photoLibrary) else {
            showAlert(
                title: "Unavailable",
                message: "Photo Library is not available on this device."
            )
            return
        }
        
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = self
        picker.allowsEditing = false
        
        present(picker, animated: true)
    }
    
    func presentCamera() {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            showAlert(
                title: "Unavailable",
                message: "Camera is not available on this device."
            )
            return
        }
        
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = self
        picker.allowsEditing = false
        picker.cameraCaptureMode = .photo
        
        present(picker, animated: true)
    }
}

// MARK: - Alerts

private extension HomeViewController {
    
    func showAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
    
    func showPermissionAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        alert.addAction(UIAlertAction(title: "Settings", style: .default) { _ in
            if let url = URL(string: UIApplication.openSettingsURLString),
               UIApplication.shared.canOpenURL(url) {
                UIApplication.shared.open(url)
            }
        })
        
        present(alert, animated: true)
    }
}

// MARK: - UIPickerViewDataSource

extension HomeViewController: UIPickerViewDataSource {
    
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return models.count
    }
}

// MARK: - UIPickerViewDelegate

extension HomeViewController: UIPickerViewDelegate {
    
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        return models[row].displayName
    }
    
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        selectedModelIndex = row
        updateModelUI()
    }
}

// MARK: - UIImagePickerControllerDelegate

extension HomeViewController {
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true)
    }
    
    func imagePickerController(
        _ picker: UIImagePickerController,
        didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
    ) {
        let originalImage = info[.originalImage] as? UIImage
        
        selectedImage = originalImage
        imageView.image = originalImage
        
        dismiss(animated: true)
    }
}
