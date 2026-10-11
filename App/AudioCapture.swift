import Foundation
import AVFoundation
import AudioToolbox

@MainActor
final class AudioCapture {
    private(set) var microphoneUID = ""
    var suppressInput = false
    var inputLocked = false
    var preferPhoneInput = false
    var onRouteChanged: (() -> Void)?
    var onInputSourceChanged: (() -> Void)?
    var onFailure: ((String) -> Void)?
    var onRoutingFallback: ((String) -> Void)?
    var onInputSources: (([String: Double], String) -> Void)?
    var onInputFormat: ((Double, UInt32, Bool) -> Void)?
    var onInputLevel: ((Double, Float, Double, Double?, Float?) -> Void)?
    private var engine: AVAudioEngine?
    private var pending: [[Float]] = []
    private(set) var capturedInputs: [String] = []
    private var inputPorts: [AVAudioSession.Port] = []
    private var activeInput = 0
    private var candidateInput = 0
    private var candidateFrames = 0
    private var sourceEnergy: [Double] = []
    private var sourceFrames = 0
    private(set) var dualRouteEnabled = false
    private var dualInputUnavailable = false
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

    var hasPhoneInput: Bool { inputPorts.contains(.builtInMic) }
    var hasDualInput: Bool { hasPhoneInput && inputPorts.contains(.bluetoothHFP) }
    var selectedInputName: String { capturedInputs.indices.contains(activeInput) ? capturedInputs[activeInput] : "대기 중" }
    var selectedInputIsPhone: Bool { inputPorts.indices.contains(activeInput) && inputPorts[activeInput] == .builtInMic }

    var playbackOutputs: [AVAudioSessionPortDescription] {
        let outputs = AVAudioSession.sharedInstance().currentRoute.outputs
        if speakerPlayback { return outputs.filter { $0.portType == .builtInSpeaker } }
        let headphones = outputs.filter {
            [.bluetoothHFP, .bluetoothA2DP, .bluetoothLE, .headphones].contains($0.portType)
        }
        return headphones.isEmpty ? outputs : headphones
    }

    func speechOutputChannels() throws -> [AVAudioSessionChannelDescription]? {
        let channels = playbackOutputs.flatMap { $0.channels ?? [] }
        // Never let the primary phone speaker become an implicit TTS destination
        // when a multiroute headphone channel cannot be found.
        if dualRouteEnabled && channels.isEmpty {
            throw InterpreterError.message("음성 출력 장치를 확인하지 못했습니다. 번역문만 표시합니다.")
        }
        return channels.isEmpty ? nil : channels
    }

    var highQualityInputEnabled: Bool {
        if #available(iOS 26.0, *) {
            return AVAudioSession.sharedInstance().currentRoute.inputs.contains {
                $0.bluetoothMicrophoneExtension?.highQualityRecording.isEnabled == true
            }
        }
        return false
    }

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
        let farField = AVAudioSession.CategoryOptions(rawValue: 1 << 18)
        // Public iOS 26.2 mode; Xcode 26.1 does not declare its symbol. The same
        // raw value is used by react-native-audio-api's native session bridge.
        let dualMode = AVAudioSession.Mode(rawValue: "AVAudioSessionModeDualRoute")
        if #available(iOS 26.2, *), microphoneUID.isEmpty, !dualInputUnavailable {
            do {
                // Some iOS versions accept far-field only with playAndRecord.
                // Retain dual microphone capture if that combination is rejected.
                do { try session.setCategory(.multiRoute, mode: dualMode, options: [.allowBluetoothHFP, farField]) }
                catch {
                    onRoutingFallback?("dual_far_field: \(error.localizedDescription)")
                    try session.setCategory(.multiRoute, mode: dualMode, options: [.allowBluetoothHFP])
                }
                try session.setActive(true)
                try session.setPreferredInput(nil)
                dualRouteEnabled = session.category == .multiRoute && session.mode == dualMode
                onRouteChanged?()
                return
            } catch {
                // Older/unsupported routes still work through a single input.
                dualRouteEnabled = false
                dualInputUnavailable = true
                onRoutingFallback?("dual_route: \(error.localizedDescription)")
            }
        }
        dualRouteEnabled = false
        let useBuiltIn = speakerPlayback || microphoneUID == "builtin" || (microphoneUID.isEmpty && dualInputUnavailable)
        // HFP couples the Bluetooth microphone and output. A built-in selection
        // must leave HFP/far-field capture, allowing AirPods to remain A2DP output.
        var options: AVAudioSession.CategoryOptions = [.allowBluetoothA2DP]
        if !useBuiltIn { options.insert(.allowBluetoothHFP) }
        if #available(iOS 26.2, *), !useBuiltIn {
            // Public farFieldInput option (1 << 18); Xcode 26.1's SDK lacks its name.
            // https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/farfieldinput
            options.insert(farField)
        }
        try session.setCategory(.playAndRecord, mode: .default, options: options)
        try session.setActive(true)
        let inputs = session.availableInputs ?? []
        let selected = inputs.first { useBuiltIn && $0.portType == .builtInMic }
            ?? inputs.first { $0.uid == microphoneUID }
            ?? inputs.first { microphoneUID.isEmpty && $0.portType == .bluetoothHFP }
            ?? inputs.first { $0.portType == .builtInMic }
        // Automatic far-field capture leaves input routing to iOS. Explicit
        // microphone choices and speaker replies still select their input.
        try session.setPreferredInput(microphoneUID.isEmpty && !useBuiltIn ? nil : selected)
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
        dualInputUnavailable = false
        pending.removeAll()
        inputLocked = false
        preferPhoneInput = false
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
        capturedInputs = []
        inputPorts = []
        inputLocked = false
        preferPhoneInput = false
        suppressInput = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        onRouteChanged?()
    }

    func takeSamples() -> [Float] {
        guard !pending.isEmpty, activeInput < pending.count else { return [] }
        let previous = activeInput
        if running, !inputLocked, preferPhoneInput, let phone = inputPorts.firstIndex(of: .builtInMic) {
            activeInput = phone
            candidateFrames = 0
        } else if running, !inputLocked, pending.count > 1, !suppressInput {
            // Prefer the phone for surrounding voices. An alternative must be
            // at least 6 dB stronger for 150 ms before switching, outside speech.
            let energy = pending.map { samples in
                samples.reduce(Double(0)) { $0 + Double($1) * Double($1) } / Double(max(1, samples.count))
            }
            let best = energy.indices.max { energy[$0] < energy[$1] } ?? activeInput
            if best != activeInput, energy[best] > max(energy[activeInput] * 4, 0.00000001) {
                if candidateInput != best { candidateInput = best; candidateFrames = 0 }
                candidateFrames += pending[best].count
                if candidateFrames >= 2_400 { activeInput = best; candidateFrames = 0 }
            } else { candidateFrames = 0 }
        }
        if previous != activeInput {
            levelFrames = 0
            levelEnergy = 0
            levelPeak = 0
            nativeLevelSamples = 0
            nativeLevelEnergy = 0
            nativeLevelPeak = 0
            onInputSourceChanged?()
        }
        let samples = pending[activeInput]
        for index in pending.indices { pending[index].removeAll(keepingCapacity: true) }
        return suppressInput ? [Float](repeating: 0, count: samples.count) : samples
    }

    func selectMicrophone(_ uid: String) throws {
        microphoneUID = uid
        try configureSession()
        if running { try rebuild() }
    }

    func setSpeakerPlayback(_ enabled: Bool) throws {
        speakerPlayback = enabled
        if dualRouteEnabled { onRouteChanged?(); return }
        try configureSession()
        if running { try rebuild() }
    }

    private func installEngine() throws {
        var candidate = AVAudioEngine()
        // Keep app-level VoIP processing off so it cannot suppress other people
        // or music after the Bluetooth recording route has been negotiated.
        do { try attachTapAndStart(candidate) }
        catch {
            guard dualRouteEnabled else { throw error }
            onRoutingFallback?("input_channels: \(error.localizedDescription)")
            dualInputUnavailable = true
            // Apply on start and reconnect: unusable aggregate inputs recover to
            // phone capture with headphone output, without mixing microphones.
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.allowBluetoothA2DP])
            try session.setActive(true)
            try session.setPreferredInput(session.availableInputs?.first { $0.portType == .builtInMic })
            dualRouteEnabled = false
            candidate = AVAudioEngine()
            try attachTapAndStart(candidate)
        }
        engine = candidate
        routeFingerprint = fingerprint()
        onRouteChanged?()
    }

    private func attachTapAndStart(_ candidate: AVAudioEngine) throws {
        let input = candidate.inputNode
        let hardware = input.inputFormat(forBus: 0)
        guard hardware.sampleRate > 0, hardware.channelCount > 0,
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: hardware.sampleRate,
                  channels: hardware.channelCount, interleaved: false) else {
            throw InterpreterError.message("현재 마이크의 오디오 형식을 사용할 수 없습니다.")
        }
        let route = AVAudioSession.sharedInstance().currentRoute
        var groups: [[Int]] = []
        var offset = 0
        capturedInputs = []
        inputPorts = []
        for port in route.inputs {
            let count = port.channels?.count ?? (dualRouteEnabled ? 0 : Int(format.channelCount))
            guard count > 0 else { throw InterpreterError.message("마이크 입력 채널 정보가 없습니다.") }
            let upper = min(offset + count, Int(format.channelCount))
            let channels = offset < upper ? Array(offset..<upper) : []
            offset += count
            if !channels.isEmpty {
                groups.append(channels)
                capturedInputs.append(port.portName)
                inputPorts.append(port.portType)
            }
        }
        guard !groups.isEmpty, !dualRouteEnabled || offset == Int(format.channelCount) else {
            throw InterpreterError.message("마이크별 입력 채널을 확인하지 못했습니다.")
        }
        if dualRouteEnabled {
            guard let unit = input.audioUnit else {
                throw InterpreterError.message("동시 마이크 입력을 연결하지 못했습니다.")
            }
            // Map every hardware channel into a distinct client channel; never
            // downmix the phone and AirPods into the same recognition signal.
            var map = (0..<Int(format.channelCount)).map { Int32($0) }
            let status = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_ChannelMap,
                kAudioUnitScope_Input, 1, &map, UInt32(map.count * MemoryLayout<Int32>.size))
            guard status == noErr else {
                throw InterpreterError.message("동시 마이크 채널 연결에 실패했습니다 (\(status)).")
            }
        }
        activeInput = inputPorts.firstIndex(of: .builtInMic) ?? 0
        candidateInput = activeInput
        candidateFrames = 0
        pending = Array(repeating: [], count: groups.count)
        sourceEnergy = Array(repeating: 0, count: groups.count)
        sourceFrames = 0
        inputLocked = false
        onInputFormat?(format.sampleRate, format.channelCount, input.isVoiceProcessingEnabled)
        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                  channels: format.channelCount, interleaved: false),
              let converter = AVAudioConverter(from: format, to: target) else {
            throw InterpreterError.message("현재 마이크의 오디오 형식을 사용할 수 없습니다.")
        }
        converter.channelMap = (0..<Int(format.channelCount)).map { NSNumber(value: $0) }
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
            guard buffer.format == format, let native = buffer.floatChannelData else { return }
            let nativeMeasurements = groups.map { channels -> (Int, Double, Float) in
                var energy = Double(0)
                var peak = Float(0)
                for frame in 0..<Int(buffer.frameLength) {
                    let sample = channels.reduce(Float(0)) { $0 + native[$1][frame] } / Float(channels.count)
                    energy += Double(sample) * Double(sample)
                    peak = max(peak, abs(sample))
                }
                return (Int(buffer.frameLength), energy, peak)
            }
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
            guard error == nil, status != .error, let data = output.floatChannelData else { return }
            let sources = groups.map { channels in
                (0..<Int(output.frameLength)).map { frame in
                    channels.reduce(Float(0)) { $0 + data[$1][frame] } / Float(channels.count)
                }
            }
            Task { @MainActor in
                guard let self, self.running, self.generation == token else { return }
                let samples = sources[self.activeInput]
                let nativeMeasurement = nativeMeasurements[self.activeInput]
                // Measure actual input before playback gating or VAD. No audio is saved.
                self.levelFrames += samples.count
                self.nativeLevelSamples += nativeMeasurement.0
                self.nativeLevelEnergy += nativeMeasurement.1
                self.nativeLevelPeak = max(self.nativeLevelPeak, nativeMeasurement.2)
                for sample in samples {
                    self.levelEnergy += Double(sample) * Double(sample)
                    self.levelPeak = max(self.levelPeak, abs(sample))
                }
                for index in sources.indices {
                    self.pending[index].append(contentsOf: sources[index])
                    self.sourceEnergy[index] += sources[index].reduce(Double(0)) { $0 + Double($1) * Double($1) }
                }
                self.sourceFrames += samples.count
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
                    let levels = self.capturedInputs.indices.reduce(into: [String: Double]()) { result, index in
                        result[self.capturedInputs[index]] = 10 * log10(max(self.sourceEnergy[index] / Double(max(1, self.sourceFrames)), 0.000000000000000001))
                    }
                    self.onInputSources?(levels, self.capturedInputs[self.activeInput])
                    self.sourceFrames = 0
                    self.sourceEnergy = Array(repeating: 0, count: sources.count)
                }
                if self.pending.contains(where: { $0.count > 960_000 }) {
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
            + "/\(session.sampleRate)/\(farFieldInputEnabled)/\(highQualityInputEnabled)/\(session.mode.rawValue)"
            + "/\(session.currentRoute.inputs.map { $0.channels?.count ?? 0 })"
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
