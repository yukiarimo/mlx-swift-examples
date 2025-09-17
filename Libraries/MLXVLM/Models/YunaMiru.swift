import CoreImage
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Configuration

public struct YunaMiruConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let intermediateSize: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let maxPositionEmbeddings: Int
        public let ropeTheta: Float
        public let vocabSize: Int
        public let normEps: Float
        public let convBias: Bool
        public let convLCache: Int
        public let blockDim: Int
        public let blockFFDim: Int
        public let blockMultipleOf: Int
        public let blockFFNDimMultiplier: Float
        public let blockAutoAdjustFFDim: Bool
        public let fullAttnIdxs: [Int]

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case maxPositionEmbeddings = "max_position_embeddings"
            case ropeTheta = "rope_theta"
            case vocabSize = "vocab_size"
            case normEps = "norm_eps"
            case convBias = "conv_bias"
            case convLCache = "conv_L_cache"
            case blockDim = "block_dim"
            case blockFFDim = "block_ff_dim"
            case blockMultipleOf = "block_multiple_of"
            case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
            case blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
            case fullAttnIdxs = "full_attn_idxs"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numChannels: Int
        public let imageSize: Int
        public let patchSize: Int
        public let numPatches: Int
        public let layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numChannels = "num_channels"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case numPatches = "num_patches"
            case layerNormEps = "layer_norm_eps"
        }
    }

    public let textConfig: TextConfiguration
    public let visionConfig: VisionConfiguration
    public let modelType: String
    public let downsampleFactor: Int
    public let imageTokenIndex: Int
    public let visionFeatureLayer: Int
    public let projectorHiddenSize: Int
    public let eosTokenId: Int

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case downsampleFactor = "downsample_factor"
        case imageTokenIndex = "image_token_index"
        case visionFeatureLayer = "vision_feature_layer"
        case projectorHiddenSize = "projector_hidden_size"
        case eosTokenId = "eos_token_id"
    }
}

// MARK: - Language Model (Yuna)

private enum Language {

    fileprivate class Attention: Module {
        let nHeads: Int
        let nKVHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        @ModuleInfo var rope: RoPE

        init(_ args: YunaMiruConfiguration.TextConfiguration) {
            self.nHeads = args.numAttentionHeads
            self.nKVHeads = args.numKeyValueHeads
            self.headDim = args.hiddenSize / nHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
            self._kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)

            self._qProj.wrappedValue = Linear(args.hiddenSize, nHeads * headDim, bias: false)
            self._kProj.wrappedValue = Linear(args.hiddenSize, nKVHeads * headDim, bias: false)
            self._vProj.wrappedValue = Linear(args.hiddenSize, nKVHeads * headDim, bias: false)
            self._outProj.wrappedValue = Linear(nHeads * headDim, args.hiddenSize, bias: false)

            self._rope.wrappedValue = RoPE(
                dimensions: headDim, traditional: false, base: args.ropeTheta)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries = qLayerNorm(queries.reshaped(B, L, nHeads, -1)).transposed(0, 2, 1, 3)
            keys = kLayerNorm(keys.reshaped(B, L, nKVHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            if let cache {
                queries = rope(queries, offset: cache.offset)
                keys = rope(keys, offset: cache.offset)
            } else {
                queries = rope(queries)
                keys = rope(keys)
            }

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            return outProj(output)
        }
    }

    fileprivate class ShortConv: Module {
        let lCache: Int
        let hiddenSize: Int

        @ModuleInfo var conv: Conv1d
        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ args: YunaMiruConfiguration.TextConfiguration, layerIdx: Int) {
            self.lCache = args.convLCache
            self.hiddenSize = args.hiddenSize

            self._conv.wrappedValue = Conv1d(
                inputChannels: args.hiddenSize, outputChannels: args.hiddenSize,
                kernelSize: lCache, groups: args.hiddenSize, bias: args.convBias)
            self._inProj.wrappedValue = Linear(
                args.hiddenSize, 3 * args.hiddenSize, bias: args.convBias)
            self._outProj.wrappedValue = Linear(
                args.hiddenSize, args.hiddenSize, bias: args.convBias)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
            let (B, C, x) = inProj(x).split(parts: 3, axis: -1).splat()
            var Bx = B * x
            if let mask {
                Bx = MLX.where(mask.expandedDimensions(axis: -1), Bx, 0)
            }

            let state = cache?[0] ?? MLXArray.zeros([Bx.dim(0), lCache - 1, hiddenSize], dtype: Bx.dtype)
            
            Bx = concatenated([state, Bx], axis: -2)
            if let cache {
                cache[0] = Bx[0..., (Bx.dim(1) - (lCache - 1))..., 0...]
            }
            let convOut = conv(Bx)
            let y = C * convOut
            return outProj(y)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo var w1: Linear
        @ModuleInfo var w3: Linear
        @ModuleInfo var w2: Linear

        init(
            dim: Int, ffDim: Int, multipleOf: Int, autoAdjustFFDim: Bool,
            ffnDimMultiplier: Float?
        ) {
            var adjustedFFDim = ffDim
            if autoAdjustFFDim {
                adjustedFFDim = Int(2 * Float(ffDim) / 3.0)
                if let multiplier = ffnDimMultiplier {
                    adjustedFFDim = Int(multiplier * Float(adjustedFFDim))
                }
                adjustedFFDim = multipleOf * ((adjustedFFDim + multipleOf - 1) / multipleOf)
            }
            self._w1.wrappedValue = Linear(dim, adjustedFFDim, bias: false)
            self._w3.wrappedValue = Linear(dim, adjustedFFDim, bias: false)
            self._w2.wrappedValue = Linear(adjustedFFDim, dim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            w2(silu(w1(x)) * w3(x))
        }
    }

    fileprivate class YunaDecoderLayer: Module {
        let isAttentionLayer: Bool

        @ModuleInfo(key: "self_attn") var selfAttn: Attention?
        @ModuleInfo var conv: ShortConv?
        @ModuleInfo(key: "feed_forward") var feedForward: MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ args: YunaMiruConfiguration.TextConfiguration, layerIdx: Int) {
            self.isAttentionLayer = args.fullAttnIdxs.contains(layerIdx)

            if isAttentionLayer {
                self._selfAttn.wrappedValue = Attention(args)
            } else {
                self._conv.wrappedValue = ShortConv(args, layerIdx: layerIdx)
            }
            self._feedForward.wrappedValue = MLP(
                dim: args.blockDim, ffDim: args.blockFFDim, multipleOf: args.blockMultipleOf,
                autoAdjustFFDim: args.blockAutoAdjustFFDim,
                ffnDimMultiplier: args.blockFFNDimMultiplier)
            self._operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
            self._ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
            let r: MLXArray
            if isAttentionLayer {
                r = selfAttn!(operatorNorm(x), mask: .array(mask!), cache: cache)
            } else {
                r = self.conv!(operatorNorm(x), mask: mask, cache: cache as? MambaCache)
            }
            let h = x + r
            let out = h + feedForward(ffnNorm(h))
            return out
        }
    }

    fileprivate class YunaModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        fileprivate let layers: [YunaDecoderLayer]
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
        let faIdx: Int
        let convIdx: Int

        init(_ args: YunaMiruConfiguration.TextConfiguration) {
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
            self.layers = (0 ..< args.numHiddenLayers).map { i in
                YunaDecoderLayer(args, layerIdx: i)
            }
            self._embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)

            self.faIdx = args.fullAttnIdxs.first ?? 0
            self.convIdx = args.fullAttnIdxs.first ?? 0
        }

        func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> MLXArray {
            var h = inputEmbedding ?? embedTokens(inputs)

            var cache = cache
            if cache == nil {
                cache = Array(repeating: nil, count: layers.count)
            }

            let attnMask = createAttentionMask(h: h, cache: [cache?[faIdx]].compactMap { $0 })
            let convMask = createAttentionMask(h: h, cache: [cache?[convIdx]].compactMap { $0 })

            for (layer, c) in zip(layers, cache!) {
                let mask: MLXArray?
                if case .array(let arr) = (layer.isAttentionLayer ? attnMask : convMask) {
                    mask = arr
                } else {
                    mask = nil
                }
                
                h = layer(h, mask: mask, cache: c)
            }
            return embeddingNorm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: YunaModel

        var kvHeads: [Int]

        init(_ config: YunaMiruConfiguration.TextConfiguration) {
            self.model = YunaModel(config)
            self.kvHeads = (0 ..< config.numHiddenLayers).map { i in
                config.fullAttnIdxs.contains(i) ? config.numKeyValueHeads : 0
            }
        }

        func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> LMOutput {
            let out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            return LMOutput(logits: model.embedTokens.asLinear(out))
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()
            for (name, param) in weights {
                if name.contains("conv.weight") {
                    if param.shape.last! > param.shape[1] {
                        sanitizedWeights[name] = param.transposed(0, 2, 1)
                    } else {
                        sanitizedWeights[name] = param
                    }
                } else {
                    sanitizedWeights[name] = param
                }
            }
            return sanitizedWeights
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            model.layers.map { l in
                l.isAttentionLayer ? KVCacheSimple() : MambaCache()
            }
        }
    }
}

// MARK: - Vision Model

private enum Vision {
    
    // NOTE: This is a simplified bilinear interpolation. The original python uses a custom kernel
    // for bicubic interpolation on MLXArrays which is not available in MLX Swift.
    fileprivate func resizePositionalEmbeddings(
        positionalEmbeddings: MLXArray, spatialShapes: MLXArray, maxLength: Int
    ) -> MLXArray {
        let (batchSize, embedDim) = (spatialShapes.dim(0), positionalEmbeddings.dim(-1))
        let sourceDtype = positionalEmbeddings.dtype
        var resultedPositionalEmbeddings = MLXArray.zeros([batchSize, maxLength, embedDim], dtype: sourceDtype)
        
        // (H, W, C) -> (C, H, W)
        let positionalEmbeddingsCHW = positionalEmbeddings.transposed(2, 0, 1)

        for i in 0 ..< batchSize {
            let (height, width) = (spatialShapes[i, 0].item(Int.self), spatialShapes[i, 1].item(Int.self))
            
            // Simplified resizing: repeat the embedding grid to approximate the target size.
            // This is a placeholder for a proper interpolation implementation.
            let resized = MLX.upsample(positionalEmbeddingsCHW, scaleFactor: [Double(height) / Double(positionalEmbeddings.dim(0)), Double(width) / Double(positionalEmbeddings.dim(1))])
            
            let resizedEmbeddings = resized.reshaped(embedDim, height * width).transposed(1, 0)
            
            resultedPositionalEmbeddings[i, 0 ..< (height * width), 0...] = resizedEmbeddings
            resultedPositionalEmbeddings[i, (height * width) ..< maxLength, 0...] = resizedEmbeddings[0]
        }
        return resultedPositionalEmbeddings
    }


    fileprivate class VisionEmbeddings: Module, UnaryLayer {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
        let positionEmbeddingSize: Int

        init(_ config: YunaMiruConfiguration.VisionConfiguration) {
            self._patchEmbedding.wrappedValue = Linear(
                inputDimensions: config.numChannels * config.patchSize * config.patchSize,
                outputDimensions: config.hiddenSize)
            self.positionEmbeddingSize = Int(sqrt(Double(config.numPatches)))
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: config.numPatches, dimensions: config.hiddenSize)
        }

        func callAsFunction(_ pixelValues: MLXArray, spatialShapes: MLXArray) -> MLXArray {
            let targetDtype = patchEmbedding.weight.dtype
            let patchEmbeds = patchEmbedding(pixelValues.asType(targetDtype))
            
            let positionalEmbeddings = positionEmbedding.weight.reshaped(positionEmbeddingSize, positionEmbeddingSize, -1)
            let resizedPositionalEmbeddings = resizePositionalEmbeddings(
                positionalEmbeddings: positionalEmbeddings,
                spatialShapes: spatialShapes,
                maxLength: pixelValues.dim(1)
            )
            return patchEmbeds + resizedPositionalEmbeddings
        }
    }

    fileprivate class VisionModel: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: Llama.LlamaModelInner
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: YunaMiruConfiguration.VisionConfiguration) {
            self._embeddings.wrappedValue = VisionEmbeddings(config)
            // Re-using Llama's TransformerBlock for the vision encoder part
            let llamaConfig = LlamaConfiguration(
                hiddenSize: config.hiddenSize, hiddenLayers: config.numHiddenLayers,
                intermediateSize: config.intermediateSize, attentionHeads: config.numAttentionHeads,
                rmsNormEps: config.layerNormEps, vocabularySize: 1, kvHeads: config.numAttentionHeads
            )
            self._encoder.wrappedValue = Llama.LlamaModelInner(llamaConfig)
            self._postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        }

        func callAsFunction(
            _ x: MLXArray, outputHiddenStates: Bool = false, spatialShapes: MLXArray
        ) -> (MLXArray, MLXArray, [MLXArray]?) {
            let x = embeddings(x, spatialShapes: spatialShapes)
            let encoderOutputs = encoder(x)
            let lastHiddenState = postLayerNorm(encoderOutputs)
            return (lastHiddenState, x, nil) // hiddenStates not implemented for simplicity
        }
        
        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            weights.filter { !$0.key.contains("position_ids") }
        }
    }
}

// MARK: - Top-Level VLM (YunaMiru)

private class YunaMiruMultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let activation = GELU()

    init(_ config: YunaMiruConfiguration) {
        let inChannels =
            config.visionConfig.hiddenSize * (config.downsampleFactor * config.downsampleFactor)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: inChannels)
        self._linear1.wrappedValue = Linear(inChannels, config.projectorHiddenSize, bias: true)
        self._linear2.wrappedValue = Linear(
            config.projectorHiddenSize, config.textConfig.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(activation(linear1(layerNorm(x))))
    }
}

private class PixelUnshuffleBlock: Module, UnaryLayer {
    let factor: Int
    init(factor: Int) {
        self.factor = factor
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        if w % factor != 0 {
            x = concatenated(
                [x, MLXArray.zeros([n, factor - (w % factor), h, c], dtype: x.dtype)], axis: 1)
            (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        }
        if h % factor != 0 {
            x = concatenated(
                [x, MLXArray.zeros([n, w, factor - (h % factor), c], dtype: x.dtype)], axis: 2)
            (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        }
        var x = x.reshaped(n, w, h / factor, c * factor)
        x = x.transposed(0, 2, 1, 3)
        x = x.reshaped(n, h / factor, w / factor, c * factor * factor)
        x = x.transposed(0, 2, 1, 3)
        return x
    }
}

private func maskedScatter(
    finalEmbedding: MLXArray, imageMaskExpanded: MLXArray, scaledImageFeatures: MLXArray
) -> MLXArray {
    let finalEmbeddingShape = finalEmbedding.shape
    let finalEmbeddingFlattened = finalEmbedding.flattened()

    // Find indices where the mask is true
    let maskIndices = MLX.argwhere(imageMaskExpanded.flattened()).squeezed(axis: -1)
    
    // Ensure we don't try to write more features than we have indices
    let numFeatures = scaledImageFeatures.size
    let numIndices = maskIndices.size
    
    if numFeatures > numIndices {
        fatalError("Mismatch in maskedScatter: \(numFeatures) features but only \(numIndices) positions.")
    }
    let validIndices = maskIndices[0..<numFeatures]

    finalEmbeddingFlattened[validIndices] = scaledImageFeatures.flattened()
    
    return finalEmbeddingFlattened.reshaped(finalEmbeddingShape)
}

public class YunaMiru: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: YunaMiruMultiModalProjector
    @ModuleInfo(key: "pixel_unshuffle") var pixelUnshuffle: PixelUnshuffleBlock

    public let config: YunaMiruConfiguration
    public var vocabularySize: Int { config.textConfig.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: YunaMiruConfiguration) {
        self.config = config
        self._visionTower.wrappedValue = Vision.VisionModel(config.visionConfig)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfig)
        self._multiModalProjector.wrappedValue = YunaMiruMultiModalProjector(config)
        self._pixelUnshuffle.wrappedValue = PixelUnshuffleBlock(factor: config.downsampleFactor)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray, pixelValues: MLXArray, spatialShapes: MLXArray,
        pixelAttentionMask: MLXArray
    ) -> MLXArray {
        let inputsEmbeds = languageModel.model.embedTokens(inputIds)

        let (_, _, visionHiddenStates) = visionTower(
            pixelValues, outputHiddenStates: true, spatialShapes: spatialShapes
        )
        // From python: `hidden_states = self.vision_tower(...)`, where it returns a tuple, and takes the last element
        let hiddenStates = visionHiddenStates!

        let imgFeatureLengths = pixelAttentionMask.sum(axis: 1).asArray(Int.self)
        var imageFeatures = [MLXArray]()

        for imgIdx in 0 ..< hiddenStates.dim(0) {
            var feature = hiddenStates[imgIdx]
            feature = feature[0 ..< imgFeatureLengths[imgIdx], 0...].expandedDimensions(axis: 0)

            let (featureOrgH, featureOrgW) = (
                spatialShapes[imgIdx, 0].item(Int.self), spatialShapes[imgIdx, 1].item(Int.self)
            )
            feature = feature.reshaped(1, featureOrgH, featureOrgW, -1)
            feature = pixelUnshuffle(feature)

            var imgEmbedding = multiModalProjector(feature)
            imgEmbedding = imgEmbedding.reshaped(-1, imgEmbedding.dim(-1))
            imageFeatures.append(imgEmbedding)
        }

        let finalImageFeatures = concatenated(imageFeatures, axis: 0)
        return mergeInputIdsWithImageFeatures(
            imageFeatures: finalImageFeatures, inputsEmbeds: inputsEmbeds, inputIds: inputIds,
            imageTokenIndex: config.imageTokenIndex)
    }
    
    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray, inputsEmbeds: MLXArray, inputIds: MLXArray, imageTokenIndex: Int
    ) -> MLXArray {
        let specialImageMask = (inputIds .== imageTokenIndex)
        var specialImageMaskExpanded = specialImageMask.expandedDimensions(axis: -1)
        specialImageMaskExpanded = MLX.broadcast(specialImageMaskExpanded, to: inputsEmbeds.shape)
        
        let nImageTokens = specialImageMask.sum().item(Int.self)
        let nImageFeatures = imageFeatures.shape[0]

        if nImageTokens != nImageFeatures {
             fatalError("Image features and image tokens do not match: tokens: \(nImageTokens), features \(nImageFeatures)")
        }

        return maskedScatter(finalEmbedding: inputsEmbeds, imageMaskExpanded: specialImageMaskExpanded, scaledImageFeatures: imageFeatures)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        let inputIds = input.text.tokens
        
        guard let pixelValues = input.image?.pixels,
              let spatialShapes = input.image?.spatialShapes,
              let pixelAttentionMask = input.image?.pixelAttentionMask
        else {
            // Text-only branch
            let result = languageModel(inputIds, cache: cache)
            return .logits(result)
        }

        // Multimodal branch
        let inputEmbeddings = getInputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelValues,
            spatialShapes: spatialShapes,
            pixelAttentionMask: pixelAttentionMask
        )
        
        let result = languageModel(inputIds, cache: cache, inputEmbedding: inputEmbeddings)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights
        newWeights = newWeights.mapKeys { key in
            var newKey = key
            if newKey.hasPrefix("model.") { newKey = String(newKey.dropFirst("model.".count)) }
            if newKey.hasPrefix("vision_encoder") { newKey = newKey.replacingOccurrences(of: "vision_encoder", with: "encoder") }
            if newKey.hasPrefix("vision_embeddings") { newKey = newKey.replacingOccurrences(of: "vision_embeddings", with: "embeddings") }
            if newKey.hasPrefix("vision_post_layernorm") { newKey = newKey.replacingOccurrences(of: "vision_post_layernorm", with: "post_layernorm") }
            if newKey.hasPrefix("text_model.") { newKey = newKey.replacingOccurrences(of: "text_model.", with: "language_model.model.") }
            return newKey
        }
        
        newWeights = languageModel.model.layers[0].conv!.sanitize(weights: newWeights)
        newWeights = visionTower.sanitize(weights: newWeights)
        
        return newWeights
    }
    
    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }
}

extension YunaMiru: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        languageModel.model.layers.compactMap {
            if let attn = $0.selfAttn {
                return (attn, ["q_proj", "v_proj"])
            }
            return nil
        }
    }
}