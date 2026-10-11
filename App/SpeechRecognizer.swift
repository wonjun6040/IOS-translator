import Foundation
import SherpaOnnx
import SherpaOnnxC
import NaturalLanguage

enum SpeechLanguages {
    // Qwen3-ASR's 30 languages; Chinese regional dialects share the Chinese entry.
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
    private var recognizer: OpaquePointer?

    deinit {
        if let recognizer { SherpaOnnxDestroyOfflineRecognizer(recognizer) }
    }

    func load() throws {
        guard recognizer == nil else { return }
        let directory = ModelStore.directory
        // The C API copies configuration strings during construction.
        let paths = [directory.appendingPathComponent("ASR/conv_frontend.onnx").path,
                     directory.appendingPathComponent("ASR/encoder.int8.onnx").path,
                     directory.appendingPathComponent("ASR/decoder.int8.onnx").path,
                     directory.appendingPathComponent("ASR/tokenizer").path,
                     "cpu", "", "greedy_search"]
        let strings = paths.map { strdup($0)! }
        defer { strings.forEach { free($0) } }
        var config = SherpaOnnxOfflineRecognizerConfig()
        config.feat_config.sample_rate = 16000
        config.feat_config.feature_dim = 80
        config.model_config.num_threads = 2
        config.model_config.provider = UnsafePointer(strings[4])
        config.model_config.tokens = UnsafePointer(strings[5])
        config.model_config.qwen3_asr.conv_frontend = UnsafePointer(strings[0])
        config.model_config.qwen3_asr.encoder = UnsafePointer(strings[1])
        config.model_config.qwen3_asr.decoder = UnsafePointer(strings[2])
        config.model_config.qwen3_asr.tokenizer = UnsafePointer(strings[3])
        config.model_config.qwen3_asr.max_total_len = 1024
        config.model_config.qwen3_asr.max_new_tokens = 256
        config.model_config.qwen3_asr.temperature = 0.000001
        config.model_config.qwen3_asr.top_p = 0.8
        config.model_config.qwen3_asr.seed = 42
        config.model_config.qwen3_asr.hotwords = UnsafePointer(strings[5])
        config.decoding_method = UnsafePointer(strings[6])
        guard let loaded = SherpaOnnxCreateOfflineRecognizer(&config) else {
            throw InterpreterError.message("Qwen 음성 인식 모델을 불러오지 못했습니다.")
        }
        recognizer = loaded
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
        let text = (result.pointee.text.map { String(cString: $0) } ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let raw = result.pointee.lang.map { String(cString: $0).lowercased() } ?? ""
        var language = SpeechLanguages.code(for: raw)
        // sherpa-onnx 1.13.8 strips Qwen's language prefix without filling result.lang.
        // Identify the decoded text locally when that field is absent.
        if language == nil, !text.isEmpty {
            let identifier = NLLanguageRecognizer()
            identifier.processString(text)
            if let candidate = identifier.languageHypotheses(withMaximum: 1).first,
               candidate.value >= 0.65 {
                language = SpeechLanguages.code(for: candidate.key.rawValue)
            }
        }
        return (text, language)
    }
}

/// VAD continues consuming microphone frames while Qwen is decoding a previous chunk.
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
        config.silero_vad.threshold = 0.35
        config.silero_vad.min_speech_duration = 0.15
        config.silero_vad.min_silence_duration = 0.65
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
