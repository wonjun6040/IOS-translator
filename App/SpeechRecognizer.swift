import Foundation
import SherpaOnnx
import SherpaOnnxC
import WhisperKit
import ArgmaxCore
import os

enum SpeechLanguages {
    // Languages available in the conversation picker.
    static let supported: [(code: String, name: String, label: String)] = [
        ("ko", "Korean", "한국어"), ("en", "English", "English"), ("ja", "Japanese", "日本語"),
        ("zh", "Chinese", "중국어"), ("yue", "Cantonese", "광둥어"), ("ar", "Arabic", "아랍어"),
        ("de", "German", "독일어"), ("fr", "French", "프랑스어"), ("es", "Spanish", "스페인어"),
        ("pt", "Portuguese", "포르투갈어"), ("id", "Indonesian", "인도네시아어"),
        ("it", "Italian", "이탈리아어"), ("ru", "Russian", "러시아어"), ("th", "Thai", "태국어"),
        ("vi", "Vietnamese", "베트남어"), ("tr", "Turkish", "튀르키예어"), ("hi", "Hindi", "힌디어"),
        ("ms", "Malay", "말레이어"), ("nl", "Dutch", "네덜란드어"), ("sv", "Swedish", "스웨덴어"),
        ("da", "Danish", "덴마크어"), ("fi", "Finnish", "핀란드어"), ("pl", "Polish", "폴란드어"),
        ("cs", "Czech", "체코어"), ("fil", "Filipino", "필리핀어"), ("fa", "Persian", "페르시아어"),
        ("el", "Greek", "그리스어"), ("hu", "Hungarian", "헝가리어"),
        ("mk", "Macedonian", "마케도니아어"), ("ro", "Romanian", "루마니아어")
    ]

    static func code(for raw: String) -> String? {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if value == "tagalog" || value == "tl" { return "fil" }
        return supported.first { $0.code == value || $0.name.lowercased() == value }?.code
    }

    static func label(for code: String) -> String {
        code == "auto" ? "자동 감지" : supported.first { $0.code == code }?.label ?? code
    }
}

actor SpeechRecognizer {
    static let modelName = "Whisper-large-v3-turbo-626MB"
    private var pipeline: WhisperKit?
    private var decoding = false
    private nonisolated let cancellation = OSAllocatedUnfairLock(initialState: false)

    func load() async throws {
        guard pipeline == nil else { return }
        let directory = ModelStore.directory.appendingPathComponent("ASR/WhisperTurbo")
        // Fail locally if the tokenizer is invalid, before the SDK's Hub fallback.
        _ = try await AutoTokenizerWrapper.from(modelFolder: directory)
        let compute = ModelComputeOptions(melCompute: .cpuOnly,
            audioEncoderCompute: .cpuAndNeuralEngine, textDecoderCompute: .cpuAndNeuralEngine)
        let config = WhisperKitConfig(modelFolder: directory.path, tokenizerFolder: directory,
            computeOptions: compute, verbose: false, logLevel: .error,
            prewarm: false, load: false, download: false)
        let loaded = try await WhisperKit(config)
        loaded.tokenizer = try await ModelUtilities.loadTokenizer(for: .largev3,
            tokenizerFolder: directory, additionalSearchPaths: [directory])
        loaded.textDecoder.isModelMultilingual = true
        try await loaded.loadModels()
        pipeline = loaded
    }

    func transcribe(_ samples: [Float], languageHint: String? = nil,
        onPartial: @escaping @Sendable (String, String?) -> Void = { _, _ in }
    ) async throws -> (text: String, language: String?, timings: [String: Double]) {
        try Task.checkCancellation()
        guard !samples.isEmpty else { return ("", nil, [:]) }
        guard let pipeline else { throw InterpreterError.message("음성 인식 모델이 준비되지 않았습니다.") }
        guard !decoding else { throw InterpreterError.message("이전 음성 인식을 처리하고 있습니다.") }
        decoding = true
        defer { decoding = false }
        cancellation.withLock { $0 = false }
        let started = Date()
        let energy = samples.reduce(0.0) { $0 + Double($1) * Double($1) }
        let rms = sqrt(energy / Double(samples.count))
        let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
        // Quiet microphone recordings failed decoding despite containing speech.
        // Boost only audible input; cap the gain and leave headroom against clipping.
        let gain: Float = rms > 0.0003 && peak > 0
            ? max(1, min(32, min(Float(0.075 / rms), 0.95 / peak))) : 1
        let input = gain > 1 ? samples.map { $0 * gain } : samples
        let preparation = Date().timeIntervalSince(started)
        let hint = languageHint == "fil" ? "tl" : languageHint
        let options = DecodingOptions(verbose: false, task: .transcribe, language: hint,
            temperatureFallbackCount: 0, sampleLength: 256, usePrefillPrompt: true,
            detectLanguage: hint == nil, skipSpecialTokens: true, withoutTimestamps: true,
            // A low-confidence first token prematurely rejected the saved recordings.
            // Silero gates speech; evaluate confidence over the full decoded segment.
            firstTokenLogProbThreshold: nil, concurrentWorkerCount: 1)
        let results = try await withTaskCancellationHandler {
            try await pipeline.transcribe(audioArray: input, decodeOptions: options) { [cancellation] progress in
                guard !cancellation.withLock({ $0 }) else { return false }
                let text = progress.text.replacingOccurrences(of: #"<\|[^|]+\|>"#,
                    with: "", options: .regularExpression).trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty { onPartial(text, languageHint) }
                return true
            }
        } onCancel: { self.cancellation.withLock { $0 = true } }
        try Task.checkCancellation()
        let segments = results.flatMap(\.segments)
        let accepted = segments.filter { $0.avgLogprob >= -1.0 }
        let text = accepted.map(\.text).joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        let language = results.first.flatMap { SpeechLanguages.code(for: $0.language) }
        var timings: [String: Double] = [
            "processing_seconds": Date().timeIntervalSince(started),
            "preparation_seconds": preparation,
            "rejected_segments": Double(segments.count - accepted.count),
            "input_gain": Double(gain),
            "input_rms_dbfs": 20 * log10(max(rms, 1e-9)),
            "normalized_rms_dbfs": 20 * log10(max(rms * Double(gain), 1e-9)),
            "audio_encoding_seconds": results.reduce(0) { $0 + $1.timings.encoding },
            "window_processing_seconds": results.reduce(0) { $0 + $1.timings.decodingLoop }
        ]
        if let segment = results.first?.segments.first {
            timings["average_log_probability"] = Double(segment.avgLogprob)
        }
        if let first = results.first, first.timings.firstTokenTime > first.timings.pipelineStart {
            timings["first_decoder_prediction_seconds"] = first.timings.firstTokenTime - first.timings.pipelineStart
        }
        return (text, language, timings)
    }
}

/// VAD continues consuming microphone frames while Whisper decodes a previous chunk.
actor SpeechDetector {
    private var detector: OpaquePointer?
    private var remainder: [Float] = []
    private var recent: [Float] = []
    private var partial: [Float] = []
    private var speaking = false

    deinit {
        if let detector { SherpaOnnxDestroyVoiceActivityDetector(detector) }
    }

    func load() throws {
        guard detector == nil else { return }
        let path = strdup(ModelStore.directory.appendingPathComponent("silero_vad.onnx").path)!
        let provider = strdup("cpu")!
        defer { free(path); free(provider) }
        var config = SherpaOnnxVadModelConfig()
        config.silero_vad.model = UnsafePointer(path)
        config.silero_vad.threshold = 0.30
        config.silero_vad.min_speech_duration = 0.10
        config.silero_vad.min_silence_duration = 0.90
        config.silero_vad.window_size = 512
        config.silero_vad.max_speech_duration = 12
        config.sample_rate = 16000
        config.num_threads = 1
        config.provider = UnsafePointer(provider)
        guard let loaded = SherpaOnnxCreateVoiceActivityDetector(&config, 30) else {
            throw InterpreterError.message("Silero 발화 감지 모델을 불러오지 못했습니다.")
        }
        detector = loaded
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

}
