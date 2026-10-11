import Foundation
import AVFoundation

@MainActor
final class AudioCapture {
    private(set) var microphoneUID = ""
    var suppressInput = false
    var onRouteChanged: (() -> Void)?
    var onFailure: ((String) -> Void)?
    var onInputFormat: ((Double, UInt32, Bool) -> Void)?
    var onInputLevel: ((Double, Float, Double, Double?, Float?) -> Void)?
    private var engine: AVAudioEngine?
    private var pending: [Float] = []
    private var generation = UUID()
    private var levelFrames = 0
    private var levelEnergy = Double(0)
    private var levelPeak = Float(0)
    private var nativeLevelSamples = 0
    private var nativeLevelEnergy = Double(0)
    private var nativeLevelPeak = Float(0)
    private var routeFingerprint = ""
    private var observers: [NSObjectProtocol] = []
    private var rebuildTask: Task<Void, Never>?
    private(set) var running = false
    private var speakerPlayback = false

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
        let useBuiltIn = speakerPlayback || microphoneUID == "builtin"
        // HFP couples the Bluetooth microphone and output. A built-in selection
        // must leave HFP/far-field capture, allowing AirPods to remain A2DP output.
        var options: AVAudioSession.CategoryOptions = [.allowBluetoothA2DP]
        if !useBuiltIn { options.insert(.allowBluetoothHFP) }
        if #available(iOS 26.2, *), !useBuiltIn {
            // Public farFieldInput option (1 << 18); Xcode 26.1's SDK lacks its name.
            // https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/farfieldinput
            options.insert(AVAudioSession.CategoryOptions(rawValue: 1 << 18))
        }
        try session.setCategory(.playAndRecord, mode: .default, options: options)
        try session.setActive(true)
        let inputs = session.availableInputs ?? []
        let selected = inputs.first { useBuiltIn && $0.portType == .builtInMic }
            ?? inputs.first { $0.uid == microphoneUID }
            ?? inputs.first { microphoneUID.isEmpty && $0.portType == .bluetoothHFP }
            ?? inputs.first { $0.portType == .builtInMic }
        try session.setPreferredInput(selected)
        try session.overrideOutputAudioPort(.none)
        let route = session.currentRoute
        // Reply playback explicitly uses the phone speaker; microphone selection
        // alone keeps a connected headphone output.
        if speakerPlayback || (selected?.portType == .builtInMic &&
           route.outputs.allSatisfy({ $0.portType == .builtInReceiver || $0.portType == .builtInSpeaker })) {
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

    func setSpeakerPlayback(_ enabled: Bool) throws {
        speakerPlayback = enabled
        try configureSession()
        if running { try rebuild() }
    }

    private func installEngine() throws {
        let candidate = AVAudioEngine()
        // Preferred Bluetooth input can settle after engine construction. Enabling
        // VoIP processing on the temporary built-in route also affects the later
        // AirPods route, suppressing surrounding voices. Capture them unprocessed;
        // supported AirPods supply their own far-field processing.
        try attachTapAndStart(candidate)
        engine = candidate
        routeFingerprint = fingerprint()
        onRouteChanged?()
    }

    private func attachTapAndStart(_ candidate: AVAudioEngine) throws {
        let input = candidate.inputNode
        let format = input.outputFormat(forBus: 0)
        onInputFormat?(format.sampleRate, format.channelCount, input.isVoiceProcessingEnabled)
        guard format.sampleRate > 0, format.channelCount > 0,
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: format, to: target) else {
            throw InterpreterError.message("현재 마이크의 오디오 형식을 사용할 수 없습니다.")
        }
        generation = UUID()
        levelFrames = 0
        levelEnergy = 0
        levelPeak = 0
        nativeLevelSamples = 0
        nativeLevelEnergy = 0
        nativeLevelPeak = 0
        let token = generation
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            // Compare the hardware stream with the resampled stream without
            // retaining microphone audio. An input DSP issue must not be hidden
            // by measuring only the converter's output.
            var nativeCount = 0
            var nativeEnergy = Double(0)
            var nativePeak = Float(0)
            if format.commonFormat == .pcmFormatFloat32, let planes = buffer.floatChannelData {
                let planeCount = format.isInterleaved ? 1 : Int(format.channelCount)
                let planeSamples = Int(buffer.frameLength) * (format.isInterleaved ? Int(format.channelCount) : 1)
                for channel in 0..<planeCount {
                    for index in 0..<planeSamples {
                        let sample = planes[channel][index]
                        nativeEnergy += Double(sample) * Double(sample)
                        nativePeak = max(nativePeak, abs(sample))
                    }
                }
                nativeCount = planeCount * planeSamples
            }
            let nativeMeasurement = (nativeCount, nativeEnergy, nativePeak)
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
                // Measure actual input before playback gating or VAD. No audio is saved.
                self.levelFrames += samples.count
                self.nativeLevelSamples += nativeMeasurement.0
                self.nativeLevelEnergy += nativeMeasurement.1
                self.nativeLevelPeak = max(self.nativeLevelPeak, nativeMeasurement.2)
                for sample in samples {
                    self.levelEnergy += Double(sample) * Double(sample)
                    self.levelPeak = max(self.levelPeak, abs(sample))
                }
                if self.levelFrames >= 32_000 {
                    let rms = sqrt(self.levelEnergy / Double(self.levelFrames))
                    self.onInputLevel?(20 * log10(max(rms, 0.000000001)), self.levelPeak,
                        Double(self.levelFrames) / 16_000,
                        self.nativeLevelSamples > 0
                            ? 20 * log10(max(sqrt(self.nativeLevelEnergy / Double(self.nativeLevelSamples)), 0.000000001)) : nil,
                        self.nativeLevelSamples > 0 ? self.nativeLevelPeak : nil)
                    self.levelFrames = 0
                    self.levelEnergy = 0
                    self.levelPeak = 0
                    self.nativeLevelSamples = 0
                    self.nativeLevelEnergy = 0
                    self.nativeLevelPeak = 0
                }
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
