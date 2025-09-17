// Copyright © 2025 Apple Inc.
//
// Ported from: https://github.com/apple/ml-yuna

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Tokenizers

public struct YunaMiruProcessorConfiguration: Codable, Sendable {
    // This configuration should be derived from the preprocessor_config.json
    // For now, we'll use values based on common vision models.
    public let imageMean: [CGFloat] = [0.5, 0.5, 0.5]
    public let imageStd: [CGFloat] = [0.5, 0.5, 0.5]
    public let imageSize: Int = 224
    public let patchSize: Int = 16

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) { (imageMean[0], imageMean[1], imageMean[2]) }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) { (imageStd[0], imageStd[1], imageStd[2]) }
}

public class YunaMiruProcessor: UserInputProcessor {
    private let config: YunaMiruProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenIndex: Int

    public init(_ config: YunaMiruProcessorConfiguration, tokenizer: any Tokenizer, imageTokenIndex: Int) {
        self.config = config
        self.tokenizer = tokenizer
        self.imageTokenIndex = imageTokenIndex
    }

    private func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (pixels: MLXArray, shapes: MLXArray, mask: MLXArray) {
        let targetSize = CGSize(width: config.imageSize, height: config.imageSize)
        
        var allPatches = [MLXArray]()
        var spatialShapes = [Int32]()
        var pixelAttentionMasks = [MLXArray]()

        for image in images {
            let resizedImage = MediaProcessing.resampleBicubic(image, to: targetSize)
            let normalizedImage = MediaProcessing.normalize(resizedImage, mean: config.imageMeanTuple, std: config.imageStdTuple)
            let array = MediaProcessing.asMLXArray(normalizedImage) // Shape [1, C, H, W]

            // Create patches: [1, C, H, W] -> [1, NumPatches, C*P*P]
            let (C, H, W) = (array.dim(1), array.dim(2), array.dim(3))
            let P = config.patchSize
            let patches = array
                .reshaped(1, C, H / P, P, W / P, P)
                .transposed(0, 2, 4, 3, 5, 1)
                .reshaped(1, -1, C * P * P)
            
            allPatches.append(patches)
            
            // Spatial shape is the grid size
            spatialShapes.append(contentsOf: [Int32(H / P), Int32(W / P)])
            
            // Create a mask for valid patches
            let numPatches = patches.dim(1)
            pixelAttentionMasks.append(MLXArray.ones([1, numPatches]))
        }

        let pixelValues = concatenated(allPatches, axis: 0)
        let spatialShapesArray = MLXArray(spatialShapes).reshaped(-1, 2)
        let pixelAttentionMask = concatenated(pixelAttentionMasks, axis: 0)
        
        return (pixelValues, spatialShapesArray, pixelAttentionMask)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages: [Chat.Message]
        switch input.prompt {
        case .chat(let chatMessages):
            messages = chatMessages
        case .text(let text):
            messages = [.user(text, images: input.images)]
        case .messages:
            // This processor requires structured Chat.Message to find image locations
            throw VLMError.processing("YunaMiruProcessor requires structured Chat.Message input, not raw dictionaries.")
        }
        
        let promptTokens = try tokenizer.applyChatTemplate(messages: messages.map { $0.asTokenizerMessage() })
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        
        var allImages = [UserInput.Image]()
        for message in messages {
            allImages.append(contentsOf: message.images)
        }
        
        if allImages.isEmpty {
            return LMInput(text: .init(tokens: promptArray))
        }

        let ciImages = try allImages.map { try $0.asCIImage() }
        let (pixelValues, spatialShapes, pixelAttentionMask) = try preprocess(images: ciImages, processing: input.processing)
        
        let imageInfo = LMInput.ProcessedImage(
            pixels: pixelValues,
            spatialShapes: spatialShapes,
            pixelAttentionMask: pixelAttentionMask
        )

        return LMInput(text: .init(tokens: promptArray, mask: nil), image: imageInfo)
    }
}

extension Chat.Message {
    func asTokenizerMessage() -> Tokenizers.Message {
        let role: Tokenizers.Message.Role
        switch self.role {
        case .user: role = .user
        case .assistant: role = .assistant
        case .system: role = .system
        case .tool: role = .tool
        }
        return Tokenizers.Message(role: role, content: self.content)
    }
}