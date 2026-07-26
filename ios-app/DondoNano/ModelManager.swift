import Foundation

/// Downloads the two ONNX files and vocab from the HuggingFace repo on first launch,
/// caches them in Documents, and reports readiness.
@MainActor
final class ModelManager: ObservableObject {
    static let repo = "samwell/dondo-nano-twi-ewe"

    @Published var status = "Not downloaded"
    @Published var ready = false

    // frontend: raw audio -> features ; model: features -> logits (int8)
    private let files = ["onnx/frontend.onnx", "onnx/model_int8.onnx", "vocab.json"]

    private var docs: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
    private func local(_ remote: String) -> URL {
        docs.appendingPathComponent((remote as NSString).lastPathComponent)
    }

    func ensureDownloaded() async {
        for remote in files {
            let dest = local(remote)
            if FileManager.default.fileExists(atPath: dest.path) { continue }
            let url = URL(string: "https://huggingface.co/\(Self.repo)/resolve/main/\(remote)")!
            status = "Downloading \((remote as NSString).lastPathComponent)…"
            do {
                let (tmp, _) = try await URLSession.shared.download(from: url)
                try? FileManager.default.removeItem(at: dest)
                try FileManager.default.moveItem(at: tmp, to: dest)
            } catch {
                status = "Download failed: \(error.localizedDescription)"
                return
            }
        }
        status = "Ready"
        ready = true
    }

    var frontendPath: URL { local("onnx/frontend.onnx") }
    var modelPath: URL { local("onnx/model_int8.onnx") }
    var vocabPath: URL { local("vocab.json") }
}
