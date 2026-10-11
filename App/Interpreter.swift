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

@MainActor @Observable
final class Interpreter: NSObject, AVSpeechSynthesizerDelegate {
    let models = ModelStore()
    var sourceLanguage = "en"
    var targetLanguage = "ko"
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
    private(set) var stopping = false
    private(set) var confirmed = ""
    private(set) var partial = ""
    private(set) var translated = ""
    private(set) var liveTranslation = ""
    private(set) var translating = false
    private(set) var inputs: [AVAudioSessionPortDescription] = []
    private(set) var actualInput = "대기 중"
    private(set) var actualOutput = "대기 중"
    private(set) var bluetoothConnected = false
    private(set) var microphoneMessage = ""
    private(set) var duplexMessage = "스피커 재생 중에는 음성 인식을 잠시 제외합니다."
    var message: String?
    private let audio = AudioCapture()
    private let recognition = SpeechRecognizer()
    private let detector = SpeechDetector()
    private let translator = LocalTranslator()
    private let synthesizer = AVSpeechSynthesizer()
    private var captureTask: Task<Void, Never>?
    private var recognitionTask: Task<Void, Never>?
    private var translationTask: Task<Void, Never>?
    private var previewTranslationTask: Task<Void, Never>?
    private var previewSamples: [Float] = []
    private var previewRevision = 0
    private var previewText = ""
    private var liveTranslationID = UUID()
    private var gateTask: Task<Void, Never>?
    private var finalized: [[Float]] = []
    private var translations: [(text: String, source: String, target: String)] = []
    private var sessionID = UUID()
    private var spoken: Set<ObjectIdentifier> = []
    private var previewDuration: TimeInterval = 0
    private var previewAt = Date.distantPast
    private var safeDuplex = false

    override init() {
        super.init()
        synthesizer.delegate = self
        synthesizer.usesApplicationAudioSession = true
        audio.onRouteChanged = { [weak self] in self?.refreshRoute() }
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
            if download { try await models.download() }
            guard models.installed else { return }
            try await detector.load()
            try await recognition.load()
            try await translator.load()
            ready = true
        } catch { ready = false; message = error.localizedDescription }
    }

    func start() async {
        guard ready, !recording, !stopping, !preparing, !translating else { return }
        guard sourceLanguage != targetLanguage else { message = "서로 다른 입력·출력 언어를 선택해 주세요."; return }
        guard !speakTranslation || voice(for: targetLanguage) != nil else {
            message = "출력 언어의 iOS 음성을 설치해 주세요. 설정 → 손쉬운 사용 → 콘텐츠 말하기 → 음성"
            return
        }
        preparing = true
        let operation = UUID()
        sessionID = operation
        let allowed = await AVAudioApplication.requestRecordPermission()
        guard sessionID == operation else { preparing = false; return }
        guard allowed else { preparing = false; message = "마이크 권한이 필요합니다. iPhone 설정에서 허용해 주세요."; return }
        await detector.reset()
        confirmed = ""
        partial = ""
        translated = ""
        liveTranslation = ""
        previewText = ""
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
            preparing = false
            captureTask = Task { [weak self] in await self?.captureLoop(id: operation) }
        } catch { preparing = false; message = error.localizedDescription; audio.stop() }
    }

    func stop() {
        guard recording || preparing else { return }
        recording = false
        previewTranslationTask?.cancel()
        previewRevision += 1
        previewSamples.removeAll()
        liveTranslation = ""
        stopping = captureTask != nil || recognitionTask != nil
        gateTask?.cancel()
        gateTask = nil
        synthesizer.stopSpeaking(at: .immediate)
        spoken.removeAll()
        audio.stop()
        // The loop drains captured audio and the VAD endpoint before ending the session.
        if captureTask == nil, recognitionTask == nil { sessionID = UUID(); preparing = false; stopping = false }
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
                finalized.append(contentsOf: segmentation.final)
                previewRevision += 1
                previewSamples.removeAll()
                previewTranslationTask?.cancel()
                previewText = ""
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
        guard !finalized.isEmpty || (recording && previewSamples.count >= 9_600
            && Date().timeIntervalSince(previewAt) >= interval) else { return }
        recognitionTask = Task { [weak self] in
            guard let self else { return }
            defer {
                self.recognitionTask = nil
                if !self.recording, self.captureTask == nil {
                    self.stopping = false
                    self.partial = ""
                }
            }
            while self.sessionID == id, !Task.isCancelled {
                let final = !self.finalized.isEmpty
                let interval = max(0.75, min(2, self.previewDuration * 1.25))
                guard final || (self.recording && self.previewSamples.count >= 9_600
                    && Date().timeIntervalSince(self.previewAt) >= interval) else { return }
                let revision = self.previewRevision
                let samples = final ? self.finalized.removeFirst() : self.previewSamples
                if !final { self.previewSamples.removeAll(); self.previewAt = Date() }
                let started = Date()
                do {
                    let result = try await self.recognition.transcribe(samples)
                    guard self.sessionID == id else { return }
                    if final {
                        if !result.text.isEmpty {
                            self.confirmed += (self.confirmed.isEmpty ? "" : "\n") + result.text
                            if let language = result.language, language != self.sourceLanguage {
                                self.message = "입력 언어와 인식된 언어가 다릅니다. 이 발화는 번역·재생하지 않습니다."
                            } else {
                                self.translations.append((result.text, self.sourceLanguage, self.targetLanguage))
                                self.translatePending(id: id)
                            }
                        }
                    } else {
                        self.previewDuration = Date().timeIntervalSince(started)
                        guard self.recording, self.previewRevision == revision,
                              !self.audio.suppressInput else { continue }
                        self.partial = result.text
                        if result.language == nil || result.language == self.sourceLanguage {
                            self.translatePreview(result.text, id: id, revision: revision)
                        }
                    }
                } catch {
                    self.message = error.localizedDescription
                    self.stop()
                    self.finalized.removeAll()
                    return
                }
            }
        }
    }

    private func translatePreview(_ text: String, id: UUID, revision: Int) {
        guard !text.isEmpty, text != previewText, translationTask == nil,
              previewTranslationTask == nil else { return }
        previewText = text
        let source = sourceLanguage
        let target = targetLanguage
        previewTranslationTask = Task { [weak self] in
            guard let self else { return }
            defer { self.previewTranslationTask = nil }
            do {
                let result = try await self.translator.translate(text: text,
                    sourceLanguageCode: source, targetLanguageCode: target) { [weak self] value in
                    Task { @MainActor in
                        guard let self, self.sessionID == id, self.recording,
                              self.previewRevision == revision, self.previewTranslationTask != nil,
                              self.translationTask == nil else { return }
                        self.liveTranslation = value
                    }
                }
                guard !Task.isCancelled, self.sessionID == id, self.recording,
                      self.previewRevision == revision, self.previewTranslationTask != nil,
                      self.translationTask == nil else { return }
                self.liveTranslation = result
            } catch {
                if !Task.isCancelled, self.previewRevision == revision, self.translationTask == nil {
                    self.liveTranslation = ""
                }
            }
        }
    }

    private func translatePending(id: UUID) {
        guard translationTask == nil else { return }
        previewTranslationTask?.cancel()
        translationTask = Task { [weak self] in
            guard let self else { return }
            self.translating = true
            defer { self.translationTask = nil; self.translating = false }
            while self.sessionID == id, !Task.isCancelled, !self.translations.isEmpty {
                let utterance = self.translations.removeFirst()
                let liveID = UUID()
                self.liveTranslationID = liveID
                do {
                    let result = try await self.translator.translate(text: utterance.text,
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
                    self.translated += (self.translated.isEmpty ? "" : "\n") + result
                    if self.recording && self.speakTranslation {
                        guard let voice = self.voice(for: utterance.target) else {
                            self.message = "출력 언어의 iOS 음성이 없습니다. 번역문만 표시합니다."
                            continue
                        }
                        let speech = AVSpeechUtterance(string: result)
                        speech.voice = voice
                        self.spoken.insert(ObjectIdentifier(speech))
                        self.updateInputGate(speaking: true)
                        self.synthesizer.speak(speech)
                    }
                } catch {
                    self.liveTranslationID = UUID()
                    self.liveTranslation = ""
                    if !Task.isCancelled { self.message = "번역 실패 · \(error.localizedDescription)" }
                }
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
        safeDuplex = headphoneOutput && (audio.voiceProcessing || audio.farFieldInputEnabled)
        duplexMessage = safeDuplex
            ? "이어폰 출력 중에도 마이크 입력을 유지합니다."
            : "음성 출력 중에는 인식을 잠시 제외합니다. 이 구간의 다음 발화는 인식하지 못할 수 있습니다."
        if !spoken.isEmpty { updateInputGate(speaking: true) }
    }

    private func voice(for language: String) -> AVSpeechSynthesisVoice? {
        AVSpeechSynthesisVoice.speechVoices().first {
            $0.language.lowercased().hasPrefix(language + "-") || $0.language.lowercased() == language
        }
    }
    private func updateInputGate(speaking: Bool) {
        gateTask?.cancel()
        if speaking {
            audio.suppressInput = !safeDuplex
            if audio.suppressInput { partial = "" }
        } else {
            gateTask = Task { [weak self] in
                try? await Task.sleep(for: .milliseconds(350))
                guard !Task.isCancelled, let self else { return }
                self.audio.suppressInput = false
            }
        }
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in if spoken.contains(id) { updateInputGate(speaking: true) } }
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        playbackEnded(utterance)
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        playbackEnded(utterance)
    }
    private nonisolated func playbackEnded(_ utterance: AVSpeechUtterance) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in
            guard spoken.remove(id) != nil else { return }
            if spoken.isEmpty { updateInputGate(speaking: false) }
        }
    }
}
