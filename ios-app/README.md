# DONDO-nano iOS example

A minimal SwiftUI app that runs Nano-L12 on-device with ONNX Runtime. On first launch it
downloads two ONNX files from the HuggingFace repo into Documents, then transcribes recorded
audio locally. No feature-extraction code lives in the app.

## How it works

```
record 16kHz mono  ->  frontend.onnx  ->  features  ->  model_int8.onnx  ->  argmax + CTC  ->  text
     AVAudioEngine     (raw -> fbank)      (1,T,160)      (int8 transformer)      vocab.json
```

The Kaldi-fbank frontend (int16 scaling, DC removal, preemphasis, povey window, FFT, mel, log,
CMVN, stride-2 stacking) is **baked into `frontend.onnx`** (~1.8 MB), so the app just feeds raw
microphone samples. The frontend was verified to match the reference processor to ~3e-4, and its
FFT is a constant matmul so it exports and runs anywhere. The int8 transformer stays a separate
file so quantization never touches the frontend.

- **ModelManager.swift** downloads `onnx/frontend.onnx`, `onnx/model_int8.onnx`, and `vocab.json`.
- **Transcriber.swift** runs both ORT sessions and does greedy CTC decoding.
- **Recorder.swift** captures 16 kHz mono PCM.
- **ContentView.swift** ties it together.

## Build and run

Fastest (uses the included `project.yml`):

```bash
brew install xcodegen        # once
cd ios-app
xcodegen generate           # creates DondoNano.xcodeproj (ONNX Runtime package + mic permission wired)
open DondoNano.xcodeproj
```

Then in Xcode: select the **DondoNano** target → Signing & Capabilities → pick your Team (your
Apple ID); choose your connected iPhone as the run destination; press Run. First launch downloads
~320 MB of model files from the Hub, then transcription runs fully on-device.

Manual alternative (no XcodeGen): create a new iOS App (SwiftUI) project, drag in the files under
`DondoNano/`, add the Swift package `https://github.com/microsoft/onnxruntime-swift-package-manager`,
and add `NSMicrophoneUsageDescription` to Info.plist.

Note: this is a scaffold. Depending on the ONNX Runtime package version, the import
(`onnxruntime_objc`) or a couple of `ORTValue`/`ORTSession` call signatures in `Transcriber.swift`
may need a minor adjustment against the installed API.

## Status

- [x] Model download + cache from the Hub (frontend + int8 transformer)
- [x] Feature extraction baked into `frontend.onnx` (no Swift DSP)
- [x] ONNX Runtime two-session pipeline + greedy CTC decode
- [x] Audio capture
- [ ] On-device latency measurement on a real iPhone
- [ ] Core ML backend for the Neural Engine (optional, for battery/speed)
