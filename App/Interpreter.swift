import Foundation
import Observation
import AVFoundation
import UIKit

enum InterpreterError: LocalizedError {
    case message(String)
    var errorDescription: String? {
        switch self { case .message(let text): text }
    }
}

struct TranslationTurn: Identifiable {
    let id = UUID()
    let original: String
    let source: String
    var target: String
    let isReply: Bool
    var speakerID: Int? = nil
    var translation = ""
    var error: String?
}

@MainActor @Observable
final class Interpreter: NSObject, AVSpeechSynthesizerDelegate {
    let models = ModelStore()
    var sourceLanguage = "auto"
    var targetLanguage = "ko"
    var excludeWearer = false
    private(set) var wearerSpeakerID: Int?
    private(set) var registeringVoice = false
    private(set) var registrationProgress: Double = 0
    var speakTranslation = true {
        didSet { if !speakTranslation { synthesizer.stopSpeaking(at: .immediate) } }
    }
    var microphoneUID = "" {
        didSet {
            do { try audio.selectMicrophone(microphoneUID); refreshRoute() }
            catch { message = error.localizedDescription }
        }
    }
    private(set) var ready = false
    private(set) var preparing = false
    private(set) var recording = false
    private(set) var inputPaused = false
    private(set) var stopping = false
    private(set) var partial = ""
    private(set) var partialLanguage: String?
    private(set) var detectedLanguage: String?
    private(set) var turns: [TranslationTurn] = []
    private(set) var activeTurnID: UUID?
    private(set) var liveTranslation = ""
    private(set) var translating = false
    private(set) var inputs: [AVAudioSessionPortDescription] = []
    private(set) var actualInput = "대기 중"
    private(set) var actualOutput = "대기 중"
    private(set) var bluetoothConnected = false
    private(set) var microphoneMessage = ""
    var message: String? {
        didSet { if let message { log("error", ["message": message]) } }
    }
    private let audio = AudioCapture()
    private let recognition = SpeechRecognizer()
    private let detector = SpeechDetector()
    private let speakers = SpeakerRecognizer()
    private let translator = LocalTranslator()
    private let synthesizer = AVSpeechSynthesizer()
    private var captureTask: Task<Void, Never>?
    private var recognitionTask: Task<Void, Never>?
    private var recognizingPreview = false
    private var translationTask: Task<Void, Never>?
    private var previewSamples: [Float] = []
    private var previewRevision = 0
    private var recognitionDisplayID = UUID()
    private var liveTranslationID = UUID()
    private var gateTask: Task<Void, Never>?
    private var registrationTask: Task<Void, Never>?
    private var registrationID = UUID()
    private var finalized: [[Float]] = []
    private var translations: [TranslationTurn] = []
    private var sessionID = UUID()
    private var spoken: Set<ObjectIdentifier> = []
    private var previewDuration: TimeInterval = 0
    private var previewAt = Date.distantPast
    private var safeDuplex = false
    private var speakerUtteranceID: ObjectIdentifier?
    private var lastLoggedRoute = ""

    override init() {
        super.init()
        synthesizer.delegate = self
        synthesizer.usesApplicationAudioSession = true
        audio.onRouteChanged = { [weak self] in self?.refreshRoute() }
        audio.onInputFormat = { [weak self] rate, channels in
            self?.log("audio_input_format", ["sample_rate": rate, "channels": channels])
        }
        audio.onInputLevel = { [weak self] rms, peak, duration in
            guard let self else { return }
            self.log("audio_input_level", ["rms_dbfs": rms, "peak": peak,
                "audio_seconds": duration, "input": self.actualInput, "suppressed": self.audio.suppressInput])
        }
        audio.onFailure = { [weak self] text in
            guard let self else { return }
            self.message = text
            self.stop()
        }
        do { try audio.configureSession() }
        catch { message = error.localizedDescription }
        refreshRoute()
    }

    func prepare(download: Bool = false) async {
        guard !preparing, !recording else { return }
        preparing = true
        message = nil
        defer { preparing = false }
        do {
            if download || models.recognitionUpdateAvailable { try await models.download() }
            guard models.installed else { return }
            try await detector.load()
            try await recognition.load()
            log("asr_model_loaded", ["model": SpeechRecognizer.modelName])
            try await speakers.load()
            wearerSpeakerID = await speakers.wearerID()
            try await translator.load()
            log("translation_model_loaded", ["model": LocalTranslator.modelName])
            do {
                let bytes = try models.pruneModelCache()
                if bytes > 0 { log("model_cache_pruned", ["bytes": bytes]) }
            } catch { log("model_cache_cleanup_failed", ["message": error.localizedDescription]) }
            ready = true
        } catch { ready = false; message = error.localizedDescription }
    }

    func start() async {
        guard ready, !recording, !stopping, !preparing, !translating, !registeringVoice else { return }
        guard sourceLanguage != targetLanguage else { message = "서로 다른 입력·출력 언어를 선택해 주세요."; return }
        preparing = true
        let operation = UUID()
        sessionID = operation
        let allowed = await AVAudioApplication.requestRecordPermission()
        guard sessionID == operation else { preparing = false; return }
        guard allowed else { preparing = false; message = "마이크 권한이 필요합니다. iPhone 설정에서 허용해 주세요."; return }
        await detector.reset()
        partial = ""
        partialLanguage = nil
        detectedLanguage = nil
        turns.removeAll()
        activeTurnID = nil
        liveTranslation = ""
        recognitionDisplayID = UUID()
        previewSamples.removeAll()
        previewRevision += 1
        message = nil
        finalized.removeAll()
        translations.removeAll()
        previewAt = .distantPast
        previewDuration = 0
        do {
            try audio.start()
            recording = true
            inputPaused = false
            preparing = false
            log("session_start", ["source": sourceLanguage, "target": targetLanguage,
                "device": UIDevice.current.model, "os": UIDevice.current.systemVersion,
                "wearer": wearerSpeakerID ?? 0,
                "exclude_wearer": excludeWearer,
                "asr_model": SpeechRecognizer.modelName, "translation_model": LocalTranslator.modelName])
            captureTask = Task { [weak self] in await self?.captureLoop(id: operation) }
        } catch { preparing = false; message = error.localizedDescription; audio.stop() }
    }

    func stop() {
        if registeringVoice {
            registrationID = UUID()
            registrationTask?.cancel()
            audio.stop()
            if registrationTask == nil { registeringVoice = false }
            return
        }
        guard recording || preparing else { return }
        recording = false
        inputPaused = false
        previewRevision += 1
        previewSamples.removeAll()
        liveTranslation = ""
        stopping = captureTask != nil || recognitionTask != nil
        gateTask?.cancel()
        gateTask = nil
        synthesizer.stopSpeaking(at: .immediate)
        spoken.removeAll()
        if speakerUtteranceID != nil { try? audio.setSpeakerPlayback(false); speakerUtteranceID = nil }
        log("session_stop", ["pending_asr": finalized.count, "pending_translation": translations.count])
        audio.stop()
        // The loop drains captured audio and the VAD endpoint before ending the session.
        if captureTask == nil, recognitionTask == nil { sessionID = UUID(); preparing = false; stopping = false }
    }

    func registerVoice() async {
        guard ready, !recording, !preparing, !stopping, !translating, !registeringVoice else { return }
        registeringVoice = true
        let operation = UUID()
        registrationID = operation
        registrationProgress = 0
        message = nil
        let allowed = await AVAudioApplication.requestRecordPermission()
        guard registrationID == operation, registeringVoice else { return }
        guard allowed else {
            registeringVoice = false
            message = "마이크 권한이 필요합니다."
            return
        }
        await detector.reset()
        guard registrationID == operation, registeringVoice else { return }
        do { try audio.start() }
        catch { registeringVoice = false; message = error.localizedDescription; audio.stop(); return }
        registrationTask = Task { [weak self] in
            guard let self else { return }
            defer {
                self.audio.stop()
                self.registrationTask = nil
                self.registeringVoice = false
            }
            let started = Date()
            var speech: [Float] = []
            while !Task.isCancelled, self.registrationID == operation, Date().timeIntervalSince(started) < 25 {
                let segment = await self.detector.segment(self.audio.takeSamples())
                for samples in segment.final { speech.append(contentsOf: samples) }
                let available = speech.count + segment.partial.count
                self.registrationProgress = min(1, Double(available) / 96_000)
                if available >= 96_000 {
                    let samples = Array((speech + segment.partial).prefix(96_000))
                    do {
                        let identity = try await self.speakers.registerWearer(samples)
                        guard !Task.isCancelled, self.registrationID == operation else { return }
                        self.wearerSpeakerID = identity
                        self.log("wearer_registered", ["speaker": identity, "input": self.actualInput])
                    } catch { if !Task.isCancelled { self.message = error.localizedDescription } }
                    return
                }
                try? await Task.sleep(for: .milliseconds(50))
            }
            if !Task.isCancelled { self.message = "음성이 충분하지 않습니다. 6초 이상 말하며 다시 등록해 주세요." }
        }
    }

    func selectWearer(_ speakerID: Int) {
        Task {
            do {
                try await speakers.saveWearer(speakerID)
                wearerSpeakerID = speakerID
                log("wearer_selected", ["speaker": speakerID])
            } catch { message = error.localizedDescription }
        }
    }

    func forgetWearer() {
        Task {
            do {
                try await speakers.forgetWearer()
                wearerSpeakerID = nil
                excludeWearer = false
                log("wearer_forgotten", [:])
            } catch { message = error.localizedDescription }
        }
    }

    private func captureLoop(id: UUID) async {
        defer {
            captureTask = nil
            if recognitionTask == nil { stopping = false; partial = "" }
        }
        while sessionID == id && !Task.isCancelled {
            let flush = !recording
            let segmentation = await detector.segment(audio.takeSamples(), flush: flush)
            guard sessionID == id else { return }
            if !segmentation.final.isEmpty {
                log("vad_end", ["segments": segmentation.final.map { Double($0.count) / 16000 },
                    "flushed": flush])
                finalized.append(contentsOf: segmentation.final)
                if recognizingPreview { recognitionTask?.cancel() }
                previewRevision += 1
                previewSamples.removeAll()
                recognitionDisplayID = UUID()
                partial = ""
                if translationTask == nil { liveTranslation = "" }
            }
            if recording, !audio.suppressInput {
                previewSamples = segmentation.partial
            }
            recognizePending(id: id)
            if finalized.reduce(0, { $0 + $1.count }) > 960_000 {
                message = "음성 처리가 입력 속도를 따라가지 못했습니다. 마지막 발화를 처리한 뒤 다시 시작해 주세요."
                stop()
            }
            if flush { return }
            if recording { try? await Task.sleep(for: .milliseconds(50)) }
        }
    }

    // Capture and VAD never wait for ASR. Only the latest interim audio is decoded;
    // finalized chunks always have priority and are consumed exactly once.
    private func recognizePending(id: UUID) {
        guard recognitionTask == nil else { return }
        let interval = max(0.75, min(2, previewDuration * 1.25))
        guard !finalized.isEmpty || (recording && previewSamples.count >= 24_000
            && Date().timeIntervalSince(previewAt) >= interval) else { return }
        recognitionTask = Task { [weak self] in
            guard let self else { return }
            defer {
                self.recognitionTask = nil
                self.recognizingPreview = false
                if !self.finalized.isEmpty {
                    self.recognizePending(id: id)
                } else if !self.recording, self.captureTask == nil {
                    self.stopping = false
                    self.partial = ""
                }
            }
            while self.sessionID == id, !Task.isCancelled {
                let final = !self.finalized.isEmpty
                self.recognizingPreview = !final
                let interval = max(0.75, min(2, self.previewDuration * 1.25))
                guard final || (self.recording && self.previewSamples.count >= 24_000
                    && Date().timeIntervalSince(self.previewAt) >= interval) else { return }
                let revision = self.previewRevision
                let samples = final ? self.finalized.removeFirst() : self.previewSamples
                if !final { self.previewSamples.removeAll(); self.previewAt = Date() }
                let started = Date()
                do {
                    let parts: [SpeakerAudio]
                    if final {
                        parts = try await self.speakers.split(samples)
                        self.log("speaker_segments", ["processing_seconds": Date().timeIntervalSince(started),
                            "parts": parts.map { ["speaker": $0.speaker ?? 0,
                                "similarity": $0.similarity ?? -1,
                                "audio_seconds": Double($0.samples.count) / 16_000] as [String: Any] }])
                    } else {
                        let speaker = await self.speakers.interimSpeaker(samples)
                        parts = [SpeakerAudio(samples: samples, speaker: speaker, similarity: nil)]
                    }
                    guard self.sessionID == id else { return }
                    for part in parts {
                        let wearer: Bool? = part.speaker.flatMap { speaker in
                            self.wearerSpeakerID.map { $0 == speaker }
                        }
                        if self.excludeWearer && wearer == true {
                            self.log("wearer_excluded", ["speaker": part.speaker ?? 0,
                                "audio_seconds": Double(part.samples.count) / 16_000])
                            if !final { self.partial = ""; self.liveTranslation = "" }
                            continue
                        }
                        let hint = self.sourceLanguage != "auto"
                            && wearer == false ? self.sourceLanguage : nil
                        let decodeStarted = Date()
                        let displayID = UUID()
                        self.recognitionDisplayID = displayID
                        let result = try await self.recognition.transcribe(part.samples, languageHint: hint) { [weak self] text, language in
                            Task { @MainActor in
                                guard let self, self.sessionID == id, self.recognitionDisplayID == displayID,
                                      (final || self.previewRevision == revision), !self.audio.suppressInput else { return }
                                self.partial = text
                                self.partialLanguage = language
                            }
                        }
                        self.recognitionDisplayID = UUID()
                        guard self.sessionID == id else { return }
                        let duration = Date().timeIntervalSince(started)
                        let energy = part.samples.reduce(Double(0)) { $0 + Double($1) * Double($1) }
                        let rms = sqrt(energy / Double(max(1, part.samples.count)))
                        let peak = part.samples.reduce(Float(0)) { max($0, abs($1)) }
                        self.log(final ? "asr_final" : "asr_interim", ["text": result.text,
                            "language": result.language ?? "unknown", "audio_seconds": Double(part.samples.count) / 16000,
                            "processing_seconds": Date().timeIntervalSince(decodeStarted),
                            "pipeline_seconds": duration, "speaker": part.speaker ?? 0,
                            "language_hint": hint ?? "auto", "pending_asr": self.finalized.count,
                            "input_rms_dbfs": 20 * log10(max(rms, 0.000000001)), "input_peak": peak,
                            "timings": result.timings])
                        if final {
                            self.partial = ""
                            self.partialLanguage = nil
                            if !result.text.isEmpty {
                                if let pair = self.languagePair(detected: result.language, wearer: wearer) {
                                    if !pair.reply {
                                        self.detectedLanguage = pair.source
                                        for index in self.turns.indices where self.turns[index].target == "auto" {
                                            self.turns[index].target = pair.source
                                            self.translations.append(self.turns[index])
                                        }
                                    }
                                    let turn = TranslationTurn(original: result.text, source: pair.source,
                                        target: pair.target, isReply: pair.reply, speakerID: part.speaker)
                                    self.turns.append(turn)
                                    if turn.target != "auto" { self.translations.append(turn) }
                                    self.translatePending(id: id)
                                } else {
                                    self.message = "이 발화의 언어를 확인하지 못했습니다. 상대방 언어를 직접 선택해 주세요."
                                }
                            }
                        } else {
                            self.previewDuration = Date().timeIntervalSince(started)
                            guard self.recording, self.previewRevision == revision,
                                  !self.audio.suppressInput else { continue }
                            self.partial = result.text
                            self.partialLanguage = result.language
                        }
                    }
                } catch is CancellationError {
                    self.recognitionDisplayID = UUID()
                    self.partial = ""
                    self.log("asr_preview_cancelled", ["processing_seconds": Date().timeIntervalSince(started),
                        "pending_asr": self.finalized.count])
                    return
                } catch {
                    self.recognitionDisplayID = UUID()
                    self.partial = ""
                    self.message = error.localizedDescription
                    self.stop()
                    self.finalized.removeAll()
                    return
                }
            }
        }
    }

    private func languagePair(detected: String?, wearer: Bool? = nil) -> (source: String, target: String, reply: Bool)? {
        let language = detected ?? (sourceLanguage == "auto" ? nil : sourceLanguage)
        guard let language else { return nil }
        // A registered voice can speak another language. Identity only decides
        // filtering; foreign speech must still translate into the selected language.
        if language != targetLanguage {
            return (language, targetLanguage, false)
        }
        if wearer == true || wearer == nil {
            let partner = sourceLanguage == "auto" ? detectedLanguage : sourceLanguage
            // The wearer may speak first while the partner's language is still
            // unknown. Keep that transcript and translate after a partner speaks.
            return (language, partner ?? "auto", true)
        }
        return (language, targetLanguage, false)
    }

    private func translatePending(id: UUID) {
        guard translationTask == nil else { return }
        translationTask = Task { [weak self] in
            guard let self else { return }
            self.translating = true
            defer { self.translationTask = nil; self.translating = false }
            while self.sessionID == id, !Task.isCancelled, !self.translations.isEmpty {
                let utterance = self.translations.removeFirst()
                self.activeTurnID = utterance.id
                let liveID = UUID()
                self.liveTranslationID = liveID
                let started = Date()
                self.log("translation_start", ["turn": utterance.id.uuidString,
                    "source": utterance.source, "target": utterance.target, "original": utterance.original])
                do {
                    let result = utterance.source == utterance.target ? utterance.original : try await self.translator.translate(text: utterance.original,
                        sourceLanguageCode: utterance.source, targetLanguageCode: utterance.target) { [weak self] value in
                        Task { @MainActor in
                            guard let self, self.sessionID == id, self.liveTranslationID == liveID,
                                  self.translationTask != nil else { return }
                            self.liveTranslation = value
                        }
                    }
                    guard self.sessionID == id, !Task.isCancelled else { return }
                    self.liveTranslationID = UUID()
                    self.liveTranslation = ""
                    if let index = self.turns.firstIndex(where: { $0.id == utterance.id }) {
                        self.turns[index].translation = result
                    }
                    self.log("translation_final", ["turn": utterance.id.uuidString,
                        "source": utterance.source, "target": utterance.target, "original": utterance.original,
                        "translation": result, "processing_seconds": Date().timeIntervalSince(started)])
                    self.message = nil
                    if self.recording && self.speakTranslation && !utterance.isReply
                        && utterance.source != utterance.target && self.speakerUtteranceID == nil {
                        guard let voice = self.voice(for: utterance.target) else {
                            self.message = "출력 언어의 iOS 음성이 없습니다. 번역문만 표시합니다."
                            continue
                        }
                        let speech = AVSpeechUtterance(string: result)
                        speech.voice = voice
                        self.spoken.insert(ObjectIdentifier(speech))
                        self.updateInputGate(speaking: true)
                        self.synthesizer.speak(speech)
                        self.log("tts_queued", ["turn": utterance.id.uuidString, "text": result,
                            "language": utterance.target, "output": self.actualOutput])
                    }
                } catch {
                    self.liveTranslationID = UUID()
                    self.liveTranslation = ""
                    if !Task.isCancelled {
                        if let index = self.turns.firstIndex(where: { $0.id == utterance.id }) {
                            self.turns[index].error = error.localizedDescription
                        }
                        self.message = "번역 실패 · \(error.localizedDescription)"
                        self.log("translation_error", ["turn": utterance.id.uuidString,
                            "original": utterance.original, "error": error.localizedDescription])
                    }
                }
                self.activeTurnID = nil
            }
        }
    }

    func refreshRoute() {
        let session = AVAudioSession.sharedInstance()
        inputs = session.availableInputs ?? []
        actualInput = session.currentRoute.inputs.map(\.portName).joined(separator: ", ")
        actualOutput = session.currentRoute.outputs.map(\.portName).joined(separator: ", ")
        if actualInput.isEmpty { actualInput = "대기 중" }
        if actualOutput.isEmpty { actualOutput = "대기 중" }
        bluetoothConnected = inputs.contains { $0.portType == .bluetoothHFP }
            || session.currentRoute.outputs.contains { [.bluetoothHFP, .bluetoothA2DP, .bluetoothLE].contains($0.portType) }
        if session.currentRoute.inputs.isEmpty {
            microphoneMessage = "통역을 시작하면 마이크를 확인합니다."
        } else if audio.farFieldInputEnabled {
            microphoneMessage = "AirPods로 상대방 음성을 듣고 있습니다."
        } else if session.currentRoute.inputs.contains(where: { $0.portType == .bluetoothHFP }) {
            microphoneMessage = audio.farFieldInputSupported
                ? "AirPods의 상대방 음성 수집 모드가 아직 활성화되지 않았습니다."
                : "현재 이어폰은 상대방 음성 수집 모드를 지원하지 않습니다. iPhone 마이크를 선택해 주세요."
        } else {
            microphoneMessage = "iPhone 마이크로 상대방 음성을 듣습니다."
        }
        let headphoneOutput = session.currentRoute.outputs.allSatisfy {
            [.bluetoothHFP, .bluetoothA2DP, .bluetoothLE, .headphones].contains($0.portType)
        } && !session.currentRoute.outputs.isEmpty
        safeDuplex = headphoneOutput && audio.farFieldInputEnabled
        let route = "\(actualInput)/\(actualOutput)/\(audio.farFieldInputEnabled)"
        if route != lastLoggedRoute {
            lastLoggedRoute = route
            log("audio_route", ["input": actualInput, "output": actualOutput,
                "far_field_supported": audio.farFieldInputSupported, "far_field_enabled": audio.farFieldInputEnabled,
                "duplex": safeDuplex])
        }
        if !spoken.isEmpty { updateInputGate(speaking: true) }
    }

    func playReply(_ turn: TranslationTurn) {
        guard turn.isReply, !turn.translation.isEmpty, !preparing, !stopping, !registeringVoice else { return }
        guard let voice = voice(for: turn.target) else {
            message = "이 언어의 iOS 음성이 없습니다. 번역문을 상대방에게 보여 주세요."
            return
        }
        synthesizer.stopSpeaking(at: .immediate)
        spoken.removeAll()
        speakerUtteranceID = nil
        do {
            try audio.setSpeakerPlayback(true)
            let speech = AVSpeechUtterance(string: turn.translation)
            speech.voice = voice
            let id = ObjectIdentifier(speech)
            speakerUtteranceID = id
            spoken.insert(id)
            updateInputGate(speaking: true)
            synthesizer.speak(speech)
            log("reply_playback", ["turn": turn.id.uuidString, "text": turn.translation,
                "language": turn.target, "output": actualOutput])
        } catch {
            try? audio.setSpeakerPlayback(false)
            message = "iPhone 스피커 재생 실패 · \(error.localizedDescription)"
        }
    }

    // Private diagnostic files only: no UI, sharing endpoint, or audio recording.
    // Keep at most the current 5 MB file plus its previous rotation, outside backups.
    private func log(_ event: String, _ values: [String: Any] = [:]) {
        let manager = FileManager.default
        var directory = manager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Diagnostics", isDirectory: true)
        do {
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
            var resources = URLResourceValues()
            resources.isExcludedFromBackup = true
            try directory.setResourceValues(resources)
            let url = directory.appendingPathComponent("interpreter.jsonl")
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            if size > 5_000_000 {
                let previous = directory.appendingPathComponent("previous.jsonl")
                if manager.fileExists(atPath: previous.path) { try manager.removeItem(at: previous) }
                try manager.moveItem(at: url, to: previous)
            }
            if !manager.fileExists(atPath: url.path) {
                guard manager.createFile(atPath: url.path, contents: nil,
                    attributes: [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication]) else { return }
            }
            let record: [String: Any] = ["time": ISO8601DateFormatter().string(from: Date()),
                "event": event, "session": sessionID.uuidString, "data": values]
            var data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
            data.append(0x0A)
            let file = try FileHandle(forWritingTo: url)
            defer { try? file.close() }
            try file.seekToEnd()
            try file.write(contentsOf: data)
        } catch { /* Logging must never interrupt interpretation. */ }
    }

    private func voice(for language: String) -> AVSpeechSynthesisVoice? {
        let prefix = language == "fil" ? "tl" : language
        return AVSpeechSynthesisVoice.speechVoices().first {
            $0.language.lowercased().hasPrefix(prefix + "-") || $0.language.lowercased() == prefix
                || (language == "yue" && $0.language.lowercased() == "zh-hk")
        }
    }
    private func updateInputGate(speaking: Bool) {
        gateTask?.cancel()
        if speaking {
            audio.suppressInput = !safeDuplex
            inputPaused = recording && audio.suppressInput
            if audio.suppressInput { partial = "" }
        } else {
            gateTask = Task { [weak self] in
                try? await Task.sleep(for: .milliseconds(350))
                guard !Task.isCancelled, let self else { return }
                self.audio.suppressInput = false
                self.inputPaused = false
            }
        }
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in
            if spoken.contains(id) {
                updateInputGate(speaking: true)
                log("tts_start", ["text": utterance.speechString, "output": actualOutput])
            }
        }
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        playbackEnded(utterance, cancelled: false)
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        playbackEnded(utterance, cancelled: true)
    }
    private nonisolated func playbackEnded(_ utterance: AVSpeechUtterance, cancelled: Bool) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in
            guard spoken.remove(id) != nil else { return }
            log(cancelled ? "tts_cancelled" : "tts_end", ["text": utterance.speechString])
            if speakerUtteranceID == id {
                speakerUtteranceID = nil
                do { try audio.setSpeakerPlayback(false) }
                catch { message = "AirPods 출력을 복구하지 못했습니다. \(error.localizedDescription)" }
            }
            if spoken.isEmpty { updateInputGate(speaking: false) }
        }
    }
}
