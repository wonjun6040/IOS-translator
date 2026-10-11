import Foundation
import SherpaOnnx
import SherpaOnnxC
import LlamaSwift
import os

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
    static let modelName = "Qwen3-ASR-1.7B-Q4_K_M"
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var audioEncoder: OpaquePointer?
    private nonisolated let cancellation = OSAllocatedUnfairLock(initialState: false)
    private let abortFlag: UnsafeMutablePointer<OSAllocatedUnfairLock<Bool>>

    init() {
        abortFlag = .allocate(capacity: 1)
        // A copy shares the same allocated lock/state; C keeps this stable address.
        abortFlag.initialize(to: cancellation)
        llama_backend_init()
    }

    deinit {
        if let audioEncoder { mtmd_free(audioEncoder) }
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
        abortFlag.deinitialize(count: 1)
        abortFlag.deallocate()
        // The backend is shared with translation and lives for the app process.
    }

    func load() throws {
        guard context == nil else { return }
        let directory = ModelStore.directory.appendingPathComponent("ASR")
        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = 0
        guard let loadedModel = llama_model_load_from_file(
            directory.appendingPathComponent("Qwen3-ASR-1.7B.Q4_K_M.gguf").path, modelParams) else {
            throw InterpreterError.message("음성 인식 모델을 불러오지 못했습니다.")
        }
        var params = llama_context_default_params()
        params.n_ctx = 1024
        params.n_batch = 256
        params.n_ubatch = 256
        params.n_threads = 2
        params.n_threads_batch = 4
        params.offload_kqv = false
        params.op_offload = false
        params.abort_callback = { data in
            guard let data else { return false }
            return data.assumingMemoryBound(to: OSAllocatedUnfairLock<Bool>.self).pointee.withLock { $0 }
        }
        params.abort_callback_data = UnsafeMutableRawPointer(abortFlag)
        guard let loadedContext = llama_init_from_model(loadedModel, params) else {
            llama_model_free(loadedModel)
            throw InterpreterError.message("음성 인식에 필요한 메모리를 확보하지 못했습니다.")
        }
        var audioParams = mtmd_context_params_default()
        audioParams.use_gpu = false
        audioParams.n_threads = 4
        audioParams.warmup = false
        guard let loadedEncoder = mtmd_init_from_file(
            directory.appendingPathComponent("Qwen3-ASR-1.7B.mmproj-Q8_0.gguf").path,
            loadedModel, audioParams) else {
            llama_free(loadedContext)
            llama_model_free(loadedModel)
            throw InterpreterError.message("음성 처리 모델을 불러오지 못했습니다.")
        }
        guard mtmd_support_audio(loadedEncoder), mtmd_get_audio_sample_rate(loadedEncoder) == 16000 else {
            mtmd_free(loadedEncoder)
            llama_free(loadedContext)
            llama_model_free(loadedModel)
            throw InterpreterError.message("음성 모델의 입력 형식이 맞지 않습니다.")
        }
        model = loadedModel
        context = loadedContext
        audioEncoder = loadedEncoder
    }

    func transcribe(_ samples: [Float], languageHint: String? = nil,
                   onPartial: @Sendable (String, String?) -> Void = { _, _ in }) async throws
        -> (text: String, language: String?, timings: [String: Double]) {
        try Task.checkCancellation()
        cancellation.withLock { $0 = false }
        return try await withTaskCancellationHandler {
            try decode(samples, languageHint: languageHint, onPartial: onPartial)
        } onCancel: {
            self.cancellation.withLock { $0 = true }
        }
    }

    private func decode(_ samples: [Float], languageHint: String?,
                        onPartial: @Sendable (String, String?) -> Void)
        throws -> (text: String, language: String?, timings: [String: Double]) {
        try Task.checkCancellation()
        guard !samples.isEmpty else { return ("", nil, [:]) }
        let preparationStarted = Date()
        guard let model, let context, let audioEncoder, let vocab = llama_model_get_vocab(model) else {
            throw InterpreterError.message("음성 인식이 준비되지 않았습니다.")
        }
        guard let bitmap = samples.withUnsafeBufferPointer({ mtmd_bitmap_init_from_audio($0.count, $0.baseAddress) }) else {
            throw InterpreterError.message("입력 음성을 처리하지 못했습니다.")
        }
        defer { mtmd_bitmap_free(bitmap) }
        guard let chunks = mtmd_input_chunks_init() else {
            throw InterpreterError.message("입력 음성을 처리하지 못했습니다.")
        }
        defer { mtmd_input_chunks_free(chunks) }
        let languageName = SpeechLanguages.supported.first { $0.code == languageHint }?.name
        // Qwen's own ASR template, including its acoustic language prediction.
        let prefix = languageName.map { "language \($0)<asr_text>" } ?? ""
        let prompt = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
            + String(cString: mtmd_default_marker())
            + "<|im_end|>\n<|im_start|>assistant\n" + prefix
        let result = prompt.withCString { pointer in
            var input = mtmd_input_text(text: pointer, text_len: prompt.utf8.count,
                add_special: false, parse_special: true)
            var bitmaps: [OpaquePointer?] = [bitmap]
            return bitmaps.withUnsafeMutableBufferPointer {
                mtmd_tokenize(audioEncoder, chunks, &input, $0.baseAddress, $0.count)
            }
        }
        let outputLimit = 256
        guard result == 0 else {
            throw InterpreterError.message("입력 음성을 처리하지 못했습니다.")
        }
        guard Int(mtmd_helper_get_n_pos(chunks)) + outputLimit <= Int(llama_n_ctx(context)) else {
            throw InterpreterError.message("입력 음성이 너무 깁니다. 잠시 끊어서 말씀해 주세요.")
        }
        llama_memory_clear(llama_get_memory(context), true)
        defer { llama_memory_clear(llama_get_memory(context), true) }
        let evaluationStarted = Date()
        var position: llama_pos = 0
        var encodingSeconds = Double(0)
        var prefillSeconds = Double(0)
        var audioTokens = 0
        let chunkCount = mtmd_input_chunks_size(chunks)
        for index in 0..<chunkCount {
            try Task.checkCancellation()
            guard let chunk = mtmd_input_chunks_get(chunks, index) else {
                throw InterpreterError.message("입력 음성을 처리하지 못했습니다.")
            }
            let status: Int32
            if mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_AUDIO {
                audioTokens += Int(mtmd_input_chunk_get_n_tokens(chunk))
                let started = Date()
                let encoded = mtmd_encode_chunk(audioEncoder, chunk)
                encodingSeconds += Date().timeIntervalSince(started)
                // Stop a superseded preview before its embeddings reach the LLM.
                try Task.checkCancellation()
                guard encoded == 0, let embeddings = mtmd_get_output_embd(audioEncoder) else {
                    throw InterpreterError.message("음성 인코딩이 실패했습니다.")
                }
                let prefillStarted = Date()
                status = mtmd_helper_decode_image_chunk(audioEncoder, context, chunk, embeddings,
                    position, 0, 256, &position, nil, nil)
                prefillSeconds += Date().timeIntervalSince(prefillStarted)
            } else {
                let started = Date()
                status = mtmd_helper_eval_chunk_single(audioEncoder, context, chunk, position, 0,
                    256, index == chunkCount - 1, &position)
                prefillSeconds += Date().timeIntervalSince(started)
            }
            try Task.checkCancellation()
            guard status == 0 else {
                throw InterpreterError.message("음성 인식 처리가 실패했습니다.")
            }
        }
        let decodingStarted = Date()
        try Task.checkCancellation()
        guard let sampler = llama_sampler_init_greedy() else {
            throw InterpreterError.message("음성 인식 처리가 준비되지 않았습니다.")
        }
        defer { llama_sampler_free(sampler) }
        var batch = llama_batch_init(1, 0, 1)
        defer { llama_batch_free(batch) }
        var bytes: [UInt8] = []
        var completed = false
        var outputTokens = 0
        var publishedAt = Date.distantPast
        var firstTextAt: Date?
        for _ in 0..<outputLimit {
            try Task.checkCancellation()
            let token = llama_sampler_sample(sampler, context, -1)
            if llama_vocab_is_eog(vocab, token) { completed = true; break }
            var piece = [CChar](repeating: 0, count: 256)
            var length = piece.withUnsafeMutableBufferPointer {
                llama_token_to_piece(vocab, token, $0.baseAddress, Int32($0.count), 0, true)
            }
            if length < 0 {
                piece = [CChar](repeating: 0, count: Int(-length))
                length = piece.withUnsafeMutableBufferPointer {
                    llama_token_to_piece(vocab, token, $0.baseAddress, Int32($0.count), 0, true)
                }
            }
            guard length >= 0 else { throw InterpreterError.message("음성 인식 결과를 읽지 못했습니다.") }
            bytes.append(contentsOf: piece.prefix(Int(length)).map { UInt8(bitPattern: $0) })
            outputTokens += 1
            if Date().timeIntervalSince(publishedAt) >= 0.12,
               let decoded = String(bytes: bytes, encoding: .utf8) {
                let output = prefix + decoded
                if let marker = output.range(of: "<asr_text>") {
                    let text = String(output[marker.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                    let language = String(output[..<marker.lowerBound]).replacingOccurrences(of: "language ", with: "")
                    if !text.isEmpty, !text.contains("<|") {
                        if firstTextAt == nil { firstTextAt = Date() }
                        onPartial(text, SpeechLanguages.code(for: language))
                    }
                }
                publishedAt = Date()
            }
            batch.n_tokens = 1
            batch.token[0] = token
            batch.pos[0] = position
            batch.n_seq_id[0] = 1
            batch.seq_id[0]![0] = 0
            batch.logits[0] = 1
            let status = llama_decode(context, batch)
            try Task.checkCancellation()
            guard status == 0 else {
                throw InterpreterError.message("음성 인식 처리가 실패했습니다.")
            }
            position += 1
        }
        guard completed, let decoded = String(bytes: bytes, encoding: .utf8) else {
            throw InterpreterError.message("음성 인식을 끝내지 못했습니다. 다시 말씀해 주세요.")
        }
        var timings = ["preparation_seconds": evaluationStarted.timeIntervalSince(preparationStarted),
            "audio_prefill_seconds": decodingStarted.timeIntervalSince(evaluationStarted),
            "audio_encoding_seconds": encodingSeconds, "model_prefill_seconds": prefillSeconds,
            "audio_tokens": Double(audioTokens), "prompt_positions": Double(mtmd_helper_get_n_pos(chunks)),
            "generation_seconds": Date().timeIntervalSince(decodingStarted), "output_tokens": Double(outputTokens)]
        if let firstTextAt { timings["first_text_seconds"] = firstTextAt.timeIntervalSince(preparationStarted) }
        if decoded.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return ("", languageHint, timings) }
        let output = prefix + decoded
        guard let marker = output.range(of: "<asr_text>") else {
            throw InterpreterError.message("음성 인식 결과의 형식이 맞지 않습니다.")
        }
        let rawLanguage = String(output[..<marker.lowerBound])
            .replacingOccurrences(of: "language ", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let text = String(output[marker.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        return (text, SpeechLanguages.code(for: rawLanguage), timings)
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
