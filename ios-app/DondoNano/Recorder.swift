import AVFoundation

/// Captures microphone audio and delivers 16 kHz mono float samples.
@MainActor
final class Recorder: ObservableObject {
    @Published var isRecording = false
    private let engine = AVAudioEngine()
    private var samples = [Float]()

    private let targetRate: Double = 16000

    func start() throws {
        samples.removeAll()
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                      sampleRate: targetRate, channels: 1, interleaved: false)!
        let converter = AVAudioConverter(from: inFormat, to: outFormat)!

        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { [weak self] buffer, _ in
            let ratio = outFormat.sampleRate / inFormat.sampleRate
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1
            guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
            var fed = false
            var err: NSError?
            converter.convert(to: out, error: &err) { _, status in
                if fed { status.pointee = .noDataNow; return nil }
                fed = true; status.pointee = .haveData; return buffer
            }
            if let ch = out.floatChannelData {
                let n = Int(out.frameLength)
                self?.samples.append(contentsOf: UnsafeBufferPointer(start: ch[0], count: n))
            }
        }
        try engine.start()
        isRecording = true
    }

    /// Stops capture and returns the recorded 16 kHz mono samples.
    func stop() -> [Float] {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        isRecording = false
        return samples
    }
}
