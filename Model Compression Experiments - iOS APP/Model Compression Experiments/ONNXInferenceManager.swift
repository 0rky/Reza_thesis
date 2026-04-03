import Foundation
import UIKit
import OnnxRuntimeBindings

final class ONNXInferenceManager {
    
    static let shared = ONNXInferenceManager()
    
    private var session: ORTSession?
    private var env: ORTEnv?
    
    private init() {}
    
    func loadModel(modelFileName: String) throws {
        guard let modelPath = Bundle.main.path(forResource: modelFileName, ofType: nil) else {
            throw NSError(domain: "ONNXInferenceManager", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Model file not found in app bundle: \(modelFileName)"
            ])
        }
        
        env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        let sessionOptions = try ORTSessionOptions()
        session = try ORTSession(env: env!, modelPath: modelPath, sessionOptions: sessionOptions)
    }
    
    func runInference(
        image: UIImage,
        inputName: String = "input",
        outputName: String = "output"
    ) throws -> (predictedIndex: Int, confidence: Float, inferenceTime: Double, probabilities: [Float]) {
        
        guard let session else {
            throw NSError(domain: "ONNXInferenceManager", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Model session is not loaded."
            ])
        }
        
        let inputTensor = try preprocessImage(image)
        
        let shape: [NSNumber] = [1, 1, 224, 224]
        
        let inputValue = try ORTValue(
            tensorData: NSMutableData(data: inputTensor),
            elementType: ORTTensorElementDataType.float,
            shape: shape
        )
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let outputs = try session.run(
            withInputs: [inputName: inputValue],
            outputNames: [outputName],
            runOptions: nil
        )
        
        let inferenceTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        
        guard let outputValue = outputs[outputName] else {
            throw NSError(domain: "ONNXInferenceManager", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Output tensor not found for output name: \(outputName)"
            ])
        }
        
        let outputData = try outputValue.tensorData() as Data
        let floatCount = outputData.count / MemoryLayout<Float>.size
        
        let probabilities: [Float] = outputData.withUnsafeBytes { rawBuffer in
            let buffer = rawBuffer.bindMemory(to: Float.self)
            return Array(buffer.prefix(floatCount))
        }
        
        guard let maxValue = probabilities.max(),
              let maxIndex = probabilities.firstIndex(of: maxValue) else {
            throw NSError(domain: "ONNXInferenceManager", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Failed to parse output probabilities."
            ])
        }
        
        let softmaxValues = softmax(probabilities)
        let confidence = softmaxValues[maxIndex]
        
        return (
            predictedIndex: maxIndex,
            confidence: confidence,
            inferenceTime: inferenceTime,
            probabilities: softmaxValues
        )
    }
}

// MARK: - Preprocessing

private extension ONNXInferenceManager {
    
    func preprocessImage(_ image: UIImage) throws -> Data {
        guard let resized = image.resize(to: CGSize(width: 224, height: 224)),
              let grayscalePixels = resized.normalizedGrayscalePixels(mean: 0.1307, std: 0.3081) else {
            throw NSError(domain: "ONNXInferenceManager", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Image preprocessing failed."
            ])
        }
        
        let data = grayscalePixels.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }
        
        return data
    }
    
    func softmax(_ values: [Float]) -> [Float] {
        guard let maxVal = values.max() else { return values }
        let exps = values.map { expf($0 - maxVal) }
        let sum = exps.reduce(0, +)
        guard sum != 0 else { return values.map { _ in 0 } }
        return exps.map { $0 / sum }
    }
}

// MARK: - UIImage Helpers

private extension UIImage {
    
    func resize(to targetSize: CGSize) -> UIImage? {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
    
    func normalizedGrayscalePixels(mean: Float, std: Float) -> [Float]? {
        guard let cgImage = self.cgImage else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var rawData = [UInt8](repeating: 0, count: height * width * bytesPerPixel)
        
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: &rawData,
                width: width,
                height: height,
                bitsPerComponent: bitsPerComponent,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              ) else {
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        var result = [Float]()
        result.reserveCapacity(width * height)
        
        for y in 0..<height {
            for x in 0..<width {
                let index = y * bytesPerRow + x * bytesPerPixel
                
                let r = Float(rawData[index]) / 255.0
                let g = Float(rawData[index + 1]) / 255.0
                let b = Float(rawData[index + 2]) / 255.0
                
                let gray = 0.299 * r + 0.587 * g + 0.114 * b
                let normalized = (gray - mean) / std
                result.append(normalized)
            }
        }
        
        return result
    }
}
