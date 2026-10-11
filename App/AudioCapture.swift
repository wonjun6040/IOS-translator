import Foundation
import AVFoundation

@MainActor
final class AudioCapture {
    var microphoneUID = ""
    var suppressInput = false
    var onRouteChanged: (() -> Void)?
    var onFailure: ((String) -> Void)?
    private(set) var voiceProcessing = false
    private var engine: AVAudioEngine?
    private var pending: [Float] = []
    private var generation = UUID()
    private var routeFingerprint = ""
    private var observers: [NSObjectProtocol] = []
    private var rebuildTask: Task<Void, Never>?
    private(set) var running = false

    var farFieldInputEnabled: Bool {
        if #available(iOS 26.0, *) {
            return AVAudioSession.sharedInstance().currentRoute.inputs.contains {
                $0.bluetoothMicrophoneExtension?.farFieldCapture.isEnabled == true
            }
        }
        return false
    }

    var farFieldInputSupported: Bool {
        if #available(iOS 26.0, *) {
            return AVAudioSession.sharedInstance().currentRoute.inputs.contains {
                $0.bluetoothMicrophoneExtension?.farFieldCapture.isSupported == true
            }
        }
        return false
    }

    init() {
        let center = NotificationCenter.default
        for name in [AVAudioSession.routeChangeNotification, Notification.Name.AVAudioEngineConfigurationChange] {
            observers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in self?.routeChanged() }
            })
        }
        observers.append(center.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] note in
            let type = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            guard type == AVAudioSession.InterruptionType.began.rawValue else { return }
            Task { @MainActor in self?.onFailure?("전화 등으로 오디오가 중단됐습니다. 다시 시작해 주세요.") }
        })
        observers.append(center.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.onFailure?("iOS 오디오가 재시작됐습니다. 통역을 다시 시작해 주세요.") }
        })
    }
    deinit { observers.forEach { NotificationCenter.default.removeObserver($0) } }

    func configureSession() throws {
        let session = AVAudioSession.sharedInstance()
        var options: AVAudioSession.CategoryOptions = [.allowBluetoothHFP, .allowBluetoothA2DP]
        if #available(iOS 26.2, *) {
            // Public farFieldInput option (1 << 18); Xcode 26.1's SDK lacks its name.
            // https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/farfieldinput
            options.insert(AVAudioSession.CategoryOptions(rawValue: 1 << 18))
        }
        try session.setCategory(.playAndRecord, mode: .default, options: options)
        try session.setActive(true)
        let inputs = session.availableInputs ?? []
        let selected = inputs.first { $0.uid == microphoneUID }
            ?? inputs.first { microphoneUID.isEmpty && $0.portType == .bluetoothHFP }
            ?? inputs.first { $0.portType == .builtInMic }
        try session.setPreferredInput(selected)
        try session.overrideOutputAudioPort(.none)
        let route = session.currentRoute
        // Never force the speaker when headphones or Bluetooth are the active output.
        if selected?.portType == .builtInMic,
           route.outputs.allSatisfy({ $0.portType == .builtInReceiver || $0.portType == .builtInSpeaker }) {
            try session.overrideOutputAudioPort(.speaker)
        }
        onRouteChanged?()
    }

    func start() throws {
        pending.removeAll(keepingCapacity: true)
        suppressInput = false
        try configureSession()
        try installEngine()
        running = true
    }

    func stop() {
        running = false
        generation = UUID()
        rebuildTask?.cancel()
        rebuildTask = nil
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        voiceProcessing = false
        suppressInput = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        onRouteChanged?()
    }

    func takeSamples() -> [Float] {
        let samples = pending
        pending.removeAll(keepingCapacity: true)
        return samples
    }

    func selectMicrophone(_ uid: String) throws {
        microphoneUID = uid
        try configureSession()
        if running { try rebuild() }
    }

    private func installEngine() throws {
        let candidate = AVAudioEngine()
        // VoIP processing targets the wearer's voice. Let the Bluetooth device's
        // far-field capture process surrounding speech instead.
        if AVAudioSession.sharedInstance().currentRoute.inputs.contains(where: { $0.portType == .bluetoothHFP }) {
            try attachTapAndStart(candidate)
            voiceProcessing = false
            engine = candidate
            routeFingerprint = fingerprint()
            onRouteChanged?()
            return
        }
        do {
            try candidate.inputNode.setVoiceProcessingEnabled(true)
            try attachTapAndStart(candidate)
            voiceProcessing = true
            engine = candidate
        } catch {
            candidate.inputNode.removeTap(onBus: 0)
            candidate.stop()
            let fallback = AVAudioEngine()
            try attachTapAndStart(fallback)
            voiceProcessing = false
            engine = fallback
        }
        routeFingerprint = fingerprint()
        onRouteChanged?()
    }

    private func attachTapAndStart(_ candidate: AVAudioEngine) throws {
        let input = candidate.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0,
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: format, to: target) else {
            throw InterpreterError.message("현재 마이크의 오디오 형식을 사용할 수 없습니다.")
        }
        generation = UUID()
        let token = generation
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            let capacity = AVAudioFrameCount(ceil(Double(buffer.frameLength) * 16000 / format.sampleRate) + 16)
            guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
            var supplied = false
            var error: NSError?
            let status = converter.convert(to: output, error: &error) { _, state in
                guard !supplied else { state.pointee = .noDataNow; return nil }
                supplied = true
                state.pointee = .haveData
                return buffer
            }
            guard error == nil, status != .error, let data = output.floatChannelData?[0] else { return }
            let samples = Array(UnsafeBufferPointer(start: data, count: Int(output.frameLength)))
            Task { @MainActor in
                guard let self, self.running, self.generation == token else { return }
                self.pending.append(contentsOf: self.suppressInput ? [Float](repeating: 0, count: samples.count) : samples)
                if self.pending.count > 960_000 {
                    self.onFailure?("처리가 입력 속도를 따라가지 못했습니다. 통역을 중지합니다.")
                }
            }
        }
        candidate.prepare()
        try candidate.start()
    }

    private func fingerprint() -> String {
        let session = AVAudioSession.sharedInstance()
        return (session.currentRoute.inputs + session.currentRoute.outputs).map { $0.uid }.joined(separator: "/")
            + "/\(session.sampleRate)/\(farFieldInputEnabled)"
    }

    private func routeChanged() {
        onRouteChanged?()
        guard running, fingerprint() != routeFingerprint || engine?.isRunning == false else { return }
        rebuildTask?.cancel()
        rebuildTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(200))
            guard !Task.isCancelled, let self, self.running else { return }
            do { try self.configureSession(); try self.rebuild() }
            catch { self.onFailure?("오디오 경로를 다시 연결하지 못했습니다: \(error.localizedDescription)") }
        }
    }

    private func rebuild() throws {
        generation = UUID()
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        try installEngine()
    }
}
