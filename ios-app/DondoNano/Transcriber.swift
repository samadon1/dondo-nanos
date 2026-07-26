import Foundation
import onnxruntime_objc

/// Two-session pipeline: raw audio -> frontend.onnx -> features -> model_int8.onnx -> logits.
/// The frontend ONNX bakes the Kaldi-fbank feature extraction, so the app never does DSP.
/// Decoding is greedy CTC: argmax per frame, drop repeats and the pad token, map through vocab.
final class Transcriber {
    private let env: ORTEnv
    private let frontend: ORTSession
    private let model: ORTSession
    private let idToToken: [Int: String]
    private let padToken: String
    private let wordDelimiter = "|"

    init(frontendPath: URL, modelPath: URL, vocabPath: URL) throws {
        env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        frontend = try ORTSession(env: env, modelPath: frontendPath.path, sessionOptions: opts)
        model = try ORTSession(env: env, modelPath: modelPath.path, sessionOptions: opts)

        let vocab = try JSONDecoder().decode([String: Int].self, from: Data(contentsOf: vocabPath))
        var inv = [Int: String]()
        for (tok, id) in vocab { inv[id] = tok }
        idToToken = inv
        padToken = vocab["[PAD]"] != nil ? "[PAD]" : "<pad>"
    }

    /// samples: 16 kHz mono PCM in [-1, 1].
    func transcribe(samples: [Float]) throws -> String {
        // 1) raw audio -> input_features
        let n = NSNumber(value: samples.count)
        let wavData = NSMutableData(bytes: samples, length: samples.count * MemoryLayout<Float>.size)
        let wav = try ORTValue(tensorData: wavData, elementType: .float, shape: [n])
        let feOut = try frontend.run(withInputs: ["waveform": wav],
                                     outputNames: ["input_features"], runOptions: nil)
        guard let feats = feOut["input_features"] else { return "" }

        // 2) input_features -> logits (re-wrap the tensor for the second session)
        let (feData, feShape) = try tensorData(feats)
        let featIn = try ORTValue(tensorData: NSMutableData(data: feData),
                                  elementType: .float, shape: feShape)
        let out = try model.run(withInputs: ["input_features": featIn],
                                outputNames: ["logits"], runOptions: nil)
        guard let logits = out["logits"] else { return "" }

        let (data, dims) = try tensorData(logits)
        let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let vocabSize = dims.last!.intValue
        let frames = dims[dims.count - 2].intValue

        var ids = [Int](); ids.reserveCapacity(frames)
        for t in 0..<frames {
            var best = 0, bestVal = -Float.greatestFiniteMagnitude
            let base = t * vocabSize
            for v in 0..<vocabSize where floats[base + v] > bestVal { bestVal = floats[base + v]; best = v }
            ids.append(best)
        }
        return ctcDecode(ids)
    }

    private func ctcDecode(_ ids: [Int]) -> String {
        var prev = -1, tokens = [String]()
        for id in ids {
            if id == prev { continue }
            prev = id
            guard let tok = idToToken[id], tok != padToken else { continue }
            tokens.append(tok)
        }
        return tokens.joined()
            .replacingOccurrences(of: wordDelimiter, with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    private func tensorData(_ v: ORTValue) throws -> (Data, [NSNumber]) {
        let info = try v.tensorTypeAndShapeInfo()
        return (try v.tensorData() as Data, info.shape)
    }
}
