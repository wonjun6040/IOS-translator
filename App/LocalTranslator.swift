import Foundation
import LlamaSwift
import NaturalLanguage

/// Serial CPU inference keeps model work off the UI thread and works while audio runs with the screen locked.
actor LocalTranslator {
    static let modelName = "Hy-MT2-1.8B-Q4_K_M"
    static let supportedLanguageCodes: Set<String> = [
        "zh", "zh-Hant", "en", "fr", "pt", "es", "ja", "tr", "ru", "ar", "ko", "th", "it", "de",
        "vi", "ms", "id", "fil", "hi", "pl", "cs", "nl", "km", "my", "fa", "gu", "ur", "te", "mr",
        "he", "bn", "ta", "uk", "bo", "kk", "mn", "ug", "yue"
    ]
    private static var modelPath: URL {
        ModelStore.directory.appendingPathComponent("Hy-MT2-1.8B-Q4_K_M.gguf")
    }

    private var model: OpaquePointer?
    private var context: OpaquePointer?

    init() { llama_backend_init() }

    deinit {
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
        // The backend is shared with ASR and lives for the app process.
    }

    func load() throws {
        guard context == nil else { return }
        guard FileManager.default.fileExists(atPath: Self.modelPath.path) else { throw InterpreterError.message("로컬 번역 모델이 준비되지 않았습니다.") }
        var modelParams = llama_model_default_params()
        // Metal commands are not allowed once the app goes into the background.
        modelParams.n_gpu_layers = 0
        guard let loadedModel = llama_model_load_from_file(Self.modelPath.path, modelParams) else {
            throw failure("번역 모델을 불러오지 못했습니다.")
        }
        var params = llama_context_default_params()
        params.n_ctx = 1024
        params.n_batch = 256
        params.n_ubatch = 256
        params.n_threads = 2
        params.n_threads_batch = 2
        params.offload_kqv = false
        params.op_offload = false
        guard let loadedContext = llama_init_from_model(loadedModel, params) else {
            llama_model_free(loadedModel)
            throw failure("번역 모델에 필요한 메모리를 확보하지 못했습니다.")
        }
        model = loadedModel
        context = loadedContext
    }

    func translate(text: String, sourceLanguageCode: String, targetLanguageCode: String,
                   onPartial: @Sendable (String) -> Void = { _ in }) throws -> String {
        let names = Dictionary(uniqueKeysWithValues: SpeechLanguages.supported.map { ($0.code, $0.name) })
        guard names[sourceLanguageCode] != nil, let target = names[targetLanguageCode],
              sourceLanguageCode != targetLanguageCode else {
            throw failure("서로 다른 입력·출력 언어를 선택해 주세요.")
        }
        guard Self.supportedLanguageCodes.contains(sourceLanguageCode),
              Self.supportedLanguageCodes.contains(targetLanguageCode) else {
            throw failure("이 언어는 현재 번역 모델이 지원하지 않습니다. 음성 인식 결과만 표시합니다.")
        }
        try Task.checkCancellation()
        let sourceText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !sourceText.isEmpty else { return "" }
        try load()
        guard let model, let context, let vocab = llama_model_get_vocab(model) else {
            throw failure("로컬 번역 모델이 준비되지 않았습니다.")
        }
        // Official single-user Hunyuan chat template, with no system prompt or Qwen thinking tokens.
        let prompt = "<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following text into \(target). Note that you should only output the translated result without any additional explanation:\n\n\(sourceText.replacingOccurrences(of: "<｜", with: "< ｜"))<｜hy_Assistant｜>"
        var tokens = [llama_token](repeating: 0, count: prompt.utf8.count + 1)
        let count = tokens.withUnsafeMutableBufferPointer {
            llama_tokenize(vocab, prompt, Int32(prompt.utf8.count), $0.baseAddress, Int32($0.count), false, true)
        }
        let outputLimit = 256
        guard count > 0, Int(count) + outputLimit <= Int(llama_n_ctx(context)) else {
            throw failure("번역할 발화가 너무 깁니다. 잠시 끊어서 말씀해 주세요.")
        }
        llama_memory_clear(llama_get_memory(context), true)
        defer { llama_memory_clear(llama_get_memory(context), true) }
        var batch = llama_batch_init(256, 0, 1)
        defer { llama_batch_free(batch) }
        for start in stride(from: 0, to: Int(count), by: 256) {
            try Task.checkCancellation()
            let end = min(start + 256, Int(count))
            batch.n_tokens = Int32(end - start)
            for index in start..<end {
                let slot = index - start
                batch.token[slot] = tokens[index]
                batch.pos[slot] = Int32(index)
                batch.n_seq_id[slot] = 1
                batch.seq_id[slot]![0] = 0
                batch.logits[slot] = index == Int(count) - 1 ? 1 : 0
            }
            guard llama_decode(context, batch) == 0 else { throw failure("번역 처리가 실패했습니다.") }
        }
        guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
            throw failure("번역 처리가 준비되지 않았습니다.")
        }
        defer { llama_sampler_free(sampler) }
        // Tencent's recommended decoding settings for the 1.8B model.
        let stages = [llama_sampler_init_top_k(20), llama_sampler_init_top_p(0.6, 1),
            llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 256, 1.05, 0, 0),
            llama_sampler_init_temp(0.7), llama_sampler_init_dist(42)]
        guard stages.allSatisfy({ $0 != nil }) else {
            for stage in stages { if let stage { llama_sampler_free(stage) } }
            throw failure("번역 처리가 준비되지 않았습니다.")
        }
        for stage in stages { llama_sampler_chain_add(sampler, stage!) }
        var bytes: [UInt8] = []
        var completed = false
        var publishedAt = Date.distantPast
        for step in 0..<outputLimit {
            try Task.checkCancellation()
            let token = llama_sampler_sample(sampler, context, -1)
            if llama_vocab_is_eog(vocab, token) {
                completed = true
                break
            }
            var piece = [CChar](repeating: 0, count: 256)
            var length = piece.withUnsafeMutableBufferPointer {
                llama_token_to_piece(vocab, token, $0.baseAddress, Int32($0.count), 0, false)
            }
            if length < 0 {
                piece = [CChar](repeating: 0, count: Int(-length))
                length = piece.withUnsafeMutableBufferPointer {
                    llama_token_to_piece(vocab, token, $0.baseAddress, Int32($0.count), 0, false)
                }
            }
            guard length >= 0 else { throw failure("번역 결과를 읽지 못했습니다.") }
            bytes.append(contentsOf: piece.prefix(Int(length)).map { UInt8(bitPattern: $0) })
            if Date().timeIntervalSince(publishedAt) >= 0.12,
               let text = String(bytes: bytes, encoding: .utf8),
               !text.contains("<think>"), !text.contains("<｜"), !text.contains("<|") {
                onPartial(text.trimmingCharacters(in: .whitespacesAndNewlines))
                publishedAt = Date()
            }
            batch.n_tokens = 1
            batch.token[0] = token
            batch.pos[0] = count + Int32(step)
            batch.n_seq_id[0] = 1
            batch.seq_id[0]![0] = 0
            batch.logits[0] = 1
            guard llama_decode(context, batch) == 0 else { throw failure("번역 처리가 실패했습니다.") }
        }
        guard completed, let decoded = String(bytes: bytes, encoding: .utf8) else {
            throw failure("번역이 끝나지 않았습니다. 이 발화는 읽지 않습니다.")
        }
        let result = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !result.isEmpty, !result.contains("<think>"), !result.contains("<｜"), !result.contains("<|"),
              !(result == sourceText && sourceText.unicodeScalars.contains { CharacterSet.letters.contains($0) }) else {
            throw failure("유효한 번역문을 얻지 못했습니다.")
        }
        let languageRecognizer = NLLanguageRecognizer()
        languageRecognizer.processString(result)
        let hypotheses = languageRecognizer.languageHypotheses(withMaximum: 1)
        if let detected = hypotheses.first, detected.value > 0.85,
           let code = SpeechLanguages.code(for: detected.key.rawValue), code != targetLanguageCode,
           !(targetLanguageCode == "yue" && code == "zh") {
            throw failure("선택한 출력 언어와 번역 결과가 다릅니다. 이 발화는 읽지 않습니다.")
        }
        return result
    }

    private func failure(_ message: String) -> InterpreterError {
        .message(message)
    }
}
