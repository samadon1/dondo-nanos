import SwiftUI

struct ContentView: View {
    @StateObject private var models = ModelManager()
    @StateObject private var recorder = Recorder()
    @State private var transcript = ""
    @State private var working = false

    var body: some View {
        VStack(spacing: 24) {
            Text("DONDO-nano")
                .font(.largeTitle.bold())
            Text("On-device Twi / Ewe speech recognition")
                .font(.subheadline).foregroundStyle(.secondary)

            Text(models.status)
                .font(.footnote).foregroundStyle(.secondary)

            Button(recorder.isRecording ? "Stop" : "Record") {
                Task { await toggle() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(!models.ready || working)

            if working { ProgressView() }

            ScrollView {
                Text(transcript.isEmpty ? "…" : transcript)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
        }
        .padding()
        .task { await models.ensureDownloaded() }
    }

    private func toggle() async {
        if recorder.isRecording {
            let samples = recorder.stop()
            working = true
            defer { working = false }
            guard samples.count > 16000 else { transcript = "Too short — hold to record a sentence."; return }
            do {
                let t = try Transcriber(frontendPath: models.frontendPath,
                                        modelPath: models.modelPath,
                                        vocabPath: models.vocabPath)
                transcript = try t.transcribe(samples: samples)
            } catch {
                transcript = "Error: \(error.localizedDescription)"
            }
        } else {
            try? recorder.start()
        }
    }
}
