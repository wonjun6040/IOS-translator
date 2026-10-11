import Foundation
import SherpaOnnxC

struct SpeakerAudio {
    let samples: [Float]
    let speaker: Int?
    let similarity: Float?
}

/// IDs identify voices in this app session, not people. The wearer is only
/// identified after voice registration or an explicit assignment by the user.
actor SpeakerRecognizer {
    private var diarizer: OpaquePointer?
    private var extractor: OpaquePointer?
    private var voices: [[Float]] = []
    private var wearer: Int?
    private var profileURL: URL {
        ModelStore.directory.deletingLastPathComponent().appendingPathComponent("VoiceProfile/wearer.json")
    }

    deinit {
        if let diarizer { SherpaOnnxDestroyOfflineSpeakerDiarization(diarizer) }
        if let extractor { SherpaOnnxDestroySpeakerEmbeddingExtractor(extractor) }
    }

    func load() throws {
        guard diarizer == nil else { return }
        let paths = [ModelStore.directory.appendingPathComponent("Speaker/segmentation.int8.onnx").path,
                     ModelStore.directory.appendingPathComponent("Speaker/campplus.onnx").path, "cpu"]
        let strings = paths.map { strdup($0)! }
        defer { strings.forEach { free($0) } }
        var config = SherpaOnnxOfflineSpeakerDiarizationConfig()
        config.segmentation.pyannote.model = UnsafePointer(strings[0])
        config.segmentation.pyannote.window_shift_ratio = 0.5
        config.segmentation.num_threads = 1
        config.segmentation.provider = UnsafePointer(strings[2])
        config.embedding.model = UnsafePointer(strings[1])
        config.embedding.num_threads = 1
        config.embedding.provider = UnsafePointer(strings[2])
        config.clustering.num_clusters = -1
        config.clustering.threshold = 0.5
        config.min_duration_on = 0.2
        config.min_duration_off = 0.3
        guard let loaded = SherpaOnnxCreateOfflineSpeakerDiarization(&config) else {
            throw InterpreterError.message("화자 분할 모델을 불러오지 못했습니다.")
        }
        guard let embedding = SherpaOnnxCreateSpeakerEmbeddingExtractor(&config.embedding) else {
            SherpaOnnxDestroyOfflineSpeakerDiarization(loaded)
            throw InterpreterError.message("목소리 구분 모델을 불러오지 못했습니다.")
        }
        diarizer = loaded
        extractor = embedding
        if let data = try? Data(contentsOf: profileURL),
           let profile = try? JSONDecoder().decode([Float].self, from: data),
           profile.count == Int(SherpaOnnxSpeakerEmbeddingExtractorDim(embedding)) {
            let voice = normalized(profile)
            if !voice.isEmpty { voices.append(voice); wearer = voices.count }
        }
    }

    func wearerID() -> Int? { wearer }

    func saveWearer(_ id: Int) throws {
        guard id > 0, id <= voices.count else { return }
        let directory = profileURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
            attributes: [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication])
        try JSONEncoder().encode(voices[id - 1]).write(to: profileURL, options: .atomic)
        var url = directory
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try url.setResourceValues(values)
        wearer = id
    }

    func forgetWearer() throws {
        if FileManager.default.fileExists(atPath: profileURL.path) {
            try FileManager.default.removeItem(at: profileURL)
        }
        wearer = nil
    }

    func registerWearer(_ samples: [Float]) throws -> Int {
        let parts = try split(samples)
        try Task.checkCancellation()
        let ids = Set(parts.compactMap(\.speaker))
        guard ids.count == 1, let id = ids.first else {
            throw InterpreterError.message("한 사람의 목소리를 확인하지 못했습니다. 조용한 곳에서 혼자 말하며 다시 등록해 주세요.")
        }
        try saveWearer(id)
        return id
    }

    func split(_ samples: [Float]) throws -> [SpeakerAudio] {
        try Task.checkCancellation()
        guard let diarizer else { throw InterpreterError.message("화자 구분이 준비되지 않았습니다.") }
        guard samples.count >= 12_800 else { return [SpeakerAudio(samples: samples, speaker: nil, similarity: nil)] }
        let result = samples.withUnsafeBufferPointer {
            SherpaOnnxOfflineSpeakerDiarizationProcess(diarizer, $0.baseAddress, Int32($0.count))
        }
        guard let result else { throw InterpreterError.message("화자 구분에 실패했습니다.") }
        defer { SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result) }
        let count = Int(SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result))
        guard count > 0, let raw = SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result) else {
            // Preserve weak speech that the speaker VAD did not find.
            let identity = identify(samples, create: true)
            return [SpeakerAudio(samples: samples, speaker: identity?.id, similarity: identity?.score)]
        }
        defer { SherpaOnnxOfflineSpeakerDiarizationDestroySegment(raw) }
        let segments = Array(UnsafeBufferPointer(start: raw, count: count))
        var identities: [Int32: (id: Int, score: Float)] = [:]
        for label in Set(segments.map(\.speaker)) {
            var clean: [Float] = []
            // Build each embedding from speech without overlapping other voices.
            for segment in segments where segment.speaker == label {
                let start = max(0, min(samples.count, Int(segment.start * 16_000)))
                let end = max(start, min(samples.count, Int(segment.end * 16_000)))
                for frameStart in stride(from: start, to: end, by: 512) {
                    let frameEnd = min(end, frameStart + 512)
                    let overlap = segments.contains { other in
                        other.speaker != label && Int(other.start * 16_000) < frameEnd
                            && Int(other.end * 16_000) > frameStart
                    }
                    if !overlap { clean.append(contentsOf: samples[frameStart..<frameEnd]) }
                    if clean.count >= 96_000 { break }
                }
                if clean.count >= 96_000 { break }
            }
            if let identity = identify(clean, create: true) { identities[label] = identity }
        }
        // Keep the original context for a single voice instead of trimming syllables
        // to the speaker model's boundaries.
        if Set(segments.map(\.speaker)).count == 1 {
            let identity = identities[segments[0].speaker]
            return [SpeakerAudio(samples: samples, speaker: identity?.id, similarity: identity?.score)]
        }
        var output: [SpeakerAudio] = []
        for segment in segments {
            let start = max(0, min(samples.count, Int(segment.start * 16_000)))
            let end = max(start, min(samples.count, Int(segment.end * 16_000)))
            // Diarization labels time ranges; it cannot unmix simultaneous voices.
            // Cut out overlapping ranges rather than attribute their mixture.
            var ranges = [start..<end]
            for other in segments where other.speaker != segment.speaker {
                let low = max(0, Int(other.start * 16_000))
                let high = min(samples.count, Int(other.end * 16_000))
                ranges = ranges.flatMap { range -> [Range<Int>] in
                    guard low < range.upperBound, high > range.lowerBound else { return [range] }
                    var remaining: [Range<Int>] = []
                    if range.lowerBound < low { remaining.append(range.lowerBound..<low) }
                    if high < range.upperBound { remaining.append(high..<range.upperBound) }
                    return remaining
                }
            }
            let identity = identities[segment.speaker]
            for range in ranges where range.count >= 6_400 {
                output.append(SpeakerAudio(samples: Array(samples[range]),
                    speaker: identity?.id, similarity: identity?.score))
            }
        }
        try Task.checkCancellation()
        return output
    }

    // An interim match never creates or updates a voice; incomplete speech must
    // not contaminate the conversation's prototypes.
    func interimSpeaker(_ samples: [Float]) -> Int? {
        identify(Array(samples.suffix(48_000)), create: false)?.id
    }

    private func identify(_ samples: [Float], create: Bool) -> (id: Int, score: Float)? {
        guard samples.count >= 12_800, let extractor,
              let stream = SherpaOnnxSpeakerEmbeddingExtractorCreateStream(extractor) else { return nil }
        defer { SherpaOnnxDestroyOnlineStream(stream) }
        samples.withUnsafeBufferPointer {
            SherpaOnnxOnlineStreamAcceptWaveform(stream, 16_000, $0.baseAddress, Int32($0.count))
        }
        SherpaOnnxOnlineStreamInputFinished(stream)
        guard SherpaOnnxSpeakerEmbeddingExtractorIsReady(extractor, stream) != 0,
              let vector = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(extractor, stream) else { return nil }
        defer { SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(vector) }
        let dim = Int(SherpaOnnxSpeakerEmbeddingExtractorDim(extractor))
        let embedding = normalized(Array(UnsafeBufferPointer(start: vector, count: dim)))
        guard !embedding.isEmpty else { return nil }
        let best = voices.enumerated().map { index, voice in
            (index, zip(embedding, voice).reduce(Float(0)) { $0 + $1.0 * $1.1 })
        }.max { $0.1 < $1.1 }
        if let best, best.1 >= 0.60 {
            if create {
                voices[best.0] = normalized(zip(voices[best.0], embedding).map { $0.0 * 0.85 + $0.1 * 0.15 })
            }
            return (best.0 + 1, best.1)
        }
        guard create, voices.count < 32 else { return nil }
        voices.append(embedding)
        return (voices.count, best?.1 ?? 0)
    }

    private func normalized(_ vector: [Float]) -> [Float] {
        let norm = sqrt(vector.reduce(Float(0)) { $0 + $1 * $1 })
        guard norm.isFinite, norm > 0 else { return [] }
        return vector.map { $0 / norm }
    }
}
