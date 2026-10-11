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
    var sourceLanguage = "ko"
    var targetLanguage = "en"
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
    private(set) var translating = false
    private(set) var inputs: [AVAudioSessionPortDescription] = []
    private(set) var actualInput = "대기 중"
    private(set) var actualOutput = "대기 중"
    private(set) var bluetoothConnected = false
    private(set) var duplexMessage = "스피커 재생 중에는 음성 인식을 잠시 제외합니다."
    var message: String?
    private let audio = AudioCapture()
    private let recognition = SpeechRecognizer()
    private let translator = LocalTranslator()
    private let synthesizer = AVSpeechSynthesizer()
    private var captureTask: Task<Void, Never>?
    private var translationTask: Task<Void, Never>?
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
        await recognition.reset()
        confirmed = ""
        partial = ""
        translated = ""
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
        stopping = captureTask != nil
        gateTask?.cancel()
        gateTask = nil
        synthesizer.stopSpeaking(at: .immediate)
        spoken.removeAll()
        audio.stop()
        // The loop drains captured audio and the VAD endpoint before ending the session.
        if captureTask == nil { sessionID = UUID(); preparing = false; stopping = false }
    }

    private func captureLoop(id: UUID) async {
        defer { captureTask = nil; stopping = false; partial = "" }
        while sessionID == id && !Task.isCancelled {
            do {
                let samples = audio.takeSamples()
                let flush = !recording
                let segmentation = await recognition.segment(samples, flush: flush)
                finalized.append(contentsOf: segmentation.final)
                while !finalized.isEmpty {
                    let samples = finalized.removeFirst()
                    let result = try await recognition.transcribe(samples)
                    guard sessionID == id else { return }
                    partial = ""
                    if !result.text.isEmpty {
                        confirmed += (confirmed.isEmpty ? "" : "\n") + result.text
                        if let language = result.language, language != sourceLanguage {
                            message = "입력 언어와 인식된 언어가 다릅니다. 이 발화는 번역·재생하지 않습니다."
                        } else {
                            translations.append((result.text, sourceLanguage, targetLanguage))
                            translatePending(id: id)
                        }
                    }
                }
                if flush { return }
                let duration = Double(segmentation.partial.count) / 16000
                let interval = max(0.9, min(4, previewDuration * 3))
                if segmentation.final.isEmpty, !audio.suppressInput, duration >= 0.7,
                   Date().timeIntervalSince(previewAt) >= interval {
                    previewAt = Date()
                    let started = Date()
                    let result = try await recognition.transcribe(segmentation.partial)
                    previewDuration = Date().timeIntervalSince(started)
                    if sessionID == id, recording, !audio.suppressInput { partial = result.text }
                }
                try await Task.sleep(for: .milliseconds(100))
            } catch {
                message = error.localizedDescription
                recording = false
                audio.stop()
                return
            }
        }
    }

    private func translatePending(id: UUID) {
        guard translationTask == nil else { return }
        translationTask = Task { [weak self] in
            guard let self else { return }
            self.translating = true
            defer { self.translationTask = nil; self.translating = false }
            while self.sessionID == id, !Task.isCancelled, !self.translations.isEmpty {
                let utterance = self.translations.removeFirst()
                do {
                    let result = try await self.translator.translate(text: utterance.text,
                        sourceLanguageCode: utterance.source, targetLanguageCode: utterance.target)
                    guard self.sessionID == id, !Task.isCancelled else { return }
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
        let headphoneOutput = session.currentRoute.outputs.allSatisfy {
            [.bluetoothHFP, .bluetoothA2DP, .bluetoothLE, .headphones].contains($0.portType)
        } && !session.currentRoute.outputs.isEmpty
        safeDuplex = headphoneOutput && audio.voiceProcessing
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
