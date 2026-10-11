import Foundation
import SherpaOnnx
import SherpaOnnxC

actor SpeechRecognizer {
    private var recognizer: OpaquePointer?
    private var detector: OpaquePointer?
    private var remainder: [Float] = []
    private var recent: [Float] = []
    private var partial: [Float] = []
    private var speaking = false

    deinit {
        if let recognizer { SherpaOnnxDestroyOfflineRecognizer(recognizer) }
        if let detector { SherpaOnnxDestroyVoiceActivityDetector(detector) }
    }

    func load() throws {
        guard recognizer == nil else { return }
        let directory = ModelStore.directory
        // The C API copies configuration strings during construction.
        let paths = [directory.appendingPathComponent("ASR/conv_frontend.onnx").path,
                     directory.appendingPathComponent("ASR/encoder.int8.onnx").path,
                     directory.appendingPathComponent("ASR/decoder.int8.onnx").path,
                     directory.appendingPathComponent("ASR/tokenizer").path,
                     directory.appendingPathComponent("silero_vad.onnx").path, "cpu", "", "greedy_search"]
        let strings = paths.map { strdup($0)! }
        defer { strings.forEach { free($0) } }
        var config = SherpaOnnxOfflineRecognizerConfig()
        config.feat_config.sample_rate = 16000
        config.feat_config.feature_dim = 80
        config.model_config.num_threads = 2
        config.model_config.provider = UnsafePointer(strings[5])
        config.model_config.tokens = UnsafePointer(strings[6])
        config.model_config.qwen3_asr.conv_frontend = UnsafePointer(strings[0])
        config.model_config.qwen3_asr.encoder = UnsafePointer(strings[1])
        config.model_config.qwen3_asr.decoder = UnsafePointer(strings[2])
        config.model_config.qwen3_asr.tokenizer = UnsafePointer(strings[3])
        config.model_config.qwen3_asr.max_total_len = 512
        config.model_config.qwen3_asr.max_new_tokens = 256
        config.model_config.qwen3_asr.temperature = 0.000001
        config.model_config.qwen3_asr.top_p = 0.8
        config.model_config.qwen3_asr.seed = 42
        config.model_config.qwen3_asr.hotwords = UnsafePointer(strings[6])
        config.decoding_method = UnsafePointer(strings[7])
        guard let loaded = SherpaOnnxCreateOfflineRecognizer(&config) else {
            throw InterpreterError.message("Qwen 음성 인식 모델을 불러오지 못했습니다.")
        }
        var vadConfig = SherpaOnnxVadModelConfig()
        vadConfig.silero_vad.model = UnsafePointer(strings[4])
        vadConfig.silero_vad.threshold = 0.5
        vadConfig.silero_vad.min_speech_duration = 0.1
        vadConfig.silero_vad.min_silence_duration = 0.45
        vadConfig.silero_vad.window_size = 512
        vadConfig.silero_vad.max_speech_duration = 12
        vadConfig.sample_rate = 16000
        vadConfig.num_threads = 1
        vadConfig.provider = UnsafePointer(strings[5])
        guard let vad = SherpaOnnxCreateVoiceActivityDetector(&vadConfig, 30) else {
            SherpaOnnxDestroyOfflineRecognizer(loaded)
            throw InterpreterError.message("Silero 발화 감지 모델을 불러오지 못했습니다.")
        }
        recognizer = loaded
        detector = vad
    }

    func reset() {
        if let detector { SherpaOnnxVoiceActivityDetectorReset(detector) }
        remainder.removeAll(keepingCapacity: true)
        recent.removeAll(keepingCapacity: true)
        partial.removeAll(keepingCapacity: true)
        speaking = false
    }

    /// Each 512-sample frame is accepted once. Completed VAD segments are consumed once.
    func segment(_ samples: [Float], flush: Bool = false) -> (final: [[Float]], partial: [Float]) {
        guard let detector else { return ([], []) }
        remainder.append(contentsOf: samples)
        var finalized: [[Float]] = []
        var cursor = 0
        while cursor + 512 <= remainder.count {
            let frame = Array(remainder[cursor..<cursor + 512])
            cursor += 512
            recent.append(contentsOf: frame)
            if recent.count > 3200 { recent.removeFirst(recent.count - 3200) }
            frame.withUnsafeBufferPointer { SherpaOnnxVoiceActivityDetectorAcceptWaveform(detector, $0.baseAddress, 512) }
            let detected = SherpaOnnxVoiceActivityDetectorDetected(detector) != 0
            if detected && !speaking {
                partial = recent
                speaking = true
            } else if speaking { partial.append(contentsOf: frame) }
            let completed = takeSegments(detector)
            if !completed.isEmpty {
                finalized += completed
                speaking = detected
                partial = detected ? recent : []
            }
        }
        if cursor > 0 { remainder.removeFirst(cursor) }
        if flush {
            if !remainder.isEmpty {
                remainder.withUnsafeBufferPointer {
                    SherpaOnnxVoiceActivityDetectorAcceptWaveform(detector, $0.baseAddress, Int32($0.count))
                }
                remainder.removeAll()
            }
            SherpaOnnxVoiceActivityDetectorFlush(detector)
            finalized += takeSegments(detector)
            partial.removeAll()
            speaking = false
        }
        // Partial text is display-only and never enters the translation queue.
        return (finalized, Array(partial.suffix(192_000)))
    }

    private func takeSegments(_ detector: OpaquePointer) -> [[Float]] {
        var segments: [[Float]] = []
        while SherpaOnnxVoiceActivityDetectorEmpty(detector) == 0 {
            if let segment = SherpaOnnxVoiceActivityDetectorFront(detector) {
                if let samples = segment.pointee.samples, segment.pointee.n > 0 {
                    segments.append(Array(UnsafeBufferPointer(start: samples, count: Int(segment.pointee.n))))
                }
                SherpaOnnxDestroySpeechSegment(segment)
            }
            SherpaOnnxVoiceActivityDetectorPop(detector)
        }
        return segments
    }

    func transcribe(_ samples: [Float]) throws -> (text: String, language: String?) {
        try Task.checkCancellation()
        guard let recognizer, let stream = SherpaOnnxCreateOfflineStream(recognizer) else {
            throw InterpreterError.message("음성 인식이 준비되지 않았습니다.")
        }
        defer { SherpaOnnxDestroyOfflineStream(stream) }
        samples.withUnsafeBufferPointer { SherpaOnnxAcceptWaveformOffline(stream, 16000, $0.baseAddress, Int32($0.count)) }
        SherpaOnnxDecodeOfflineStream(recognizer, stream)
        try Task.checkCancellation()
        guard let result = SherpaOnnxGetOfflineStreamResult(stream) else {
            throw InterpreterError.message("음성을 인식하지 못했습니다.")
        }
        defer { SherpaOnnxDestroyOfflineRecognizerResult(result) }
        let text = result.pointee.text.map { String(cString: $0) } ?? ""
        let raw = result.pointee.lang.map { String(cString: $0).lowercased() } ?? ""
        let language = ["korean": "ko", "english": "en", "japanese": "ja"][raw] ?? raw
        return (text.trimmingCharacters(in: .whitespacesAndNewlines), language.isEmpty ? nil : language)
    }
}
