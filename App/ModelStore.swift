import Foundation
import Observation

@MainActor @Observable
final class ModelStore: NSObject, URLSessionDownloadDelegate {
    nonisolated static let assets: [(path: String, url: String, size: Int64)] = [
        ("ASR/WhisperTurbo/AudioEncoder.mlmodelc/analytics/coremldata.bin", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/AudioEncoder.mlmodelc/analytics/coremldata.bin", 243),
        ("ASR/WhisperTurbo/AudioEncoder.mlmodelc/coremldata.bin", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/AudioEncoder.mlmodelc/coremldata.bin", 348),
        ("ASR/WhisperTurbo/AudioEncoder.mlmodelc/metadata.json", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/AudioEncoder.mlmodelc/metadata.json", 1922),
        ("ASR/WhisperTurbo/AudioEncoder.mlmodelc/model.mil", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/AudioEncoder.mlmodelc/model.mil", 934263),
        ("ASR/WhisperTurbo/AudioEncoder.mlmodelc/weights/weight.bin", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/AudioEncoder.mlmodelc/weights/weight.bin", 421968768),
        ("ASR/WhisperTurbo/MelSpectrogram.mlmodelc/analytics/coremldata.bin", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/MelSpectrogram.mlmodelc/analytics/coremldata.bin", 243),
        ("ASR/WhisperTurbo/MelSpectrogram.mlmodelc/coremldata.bin", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/MelSpectrogram.mlmodelc/coremldata.bin", 329),
        ("ASR/WhisperTurbo/MelSpectrogram.mlmodelc/metadata.json", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/MelSpectrogram.mlmodelc/metadata.json", 1850),
        ("ASR/WhisperTurbo/MelSpectrogram.mlmodelc/model.mil", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/MelSpectrogram.mlmodelc/model.mil", 10143),
        ("ASR/WhisperTurbo/MelSpectrogram.mlmodelc/weights/weight.bin", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/MelSpectrogram.mlmodelc/weights/weight.bin", 373376),
        ("ASR/WhisperTurbo/TextDecoder.mlmodelc/analytics/coremldata.bin", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/TextDecoder.mlmodelc/analytics/coremldata.bin", 243),
        ("ASR/WhisperTurbo/TextDecoder.mlmodelc/coremldata.bin", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/TextDecoder.mlmodelc/coremldata.bin", 633),
        ("ASR/WhisperTurbo/TextDecoder.mlmodelc/metadata.json", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/TextDecoder.mlmodelc/metadata.json", 4924),
        ("ASR/WhisperTurbo/TextDecoder.mlmodelc/model.mil", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/TextDecoder.mlmodelc/model.mil", 217177),
        ("ASR/WhisperTurbo/TextDecoder.mlmodelc/weights/weight.bin", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/TextDecoder.mlmodelc/weights/weight.bin", 203199860),
        ("ASR/WhisperTurbo/config.json", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/config.json", 1149),
        ("ASR/WhisperTurbo/generation_config.json", "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_626MB/generation_config.json", 2767),
        ("ASR/WhisperTurbo/special_tokens_map.json", "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/special_tokens_map.json", 2186),
        ("ASR/WhisperTurbo/tokenizer.json", "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/tokenizer.json", 2710337),
        ("ASR/WhisperTurbo/tokenizer_config.json", "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/tokenizer_config.json", 282843),
        ("silero_vad.onnx", "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx", 643_854),
        ("Hy-MT2-1.8B-Q4_K_M.gguf", "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/a0c709d9fac510f2c807aa3af52872340dc37a4a/Hy-MT2-1.8B-Q4_K_M.gguf", 1_133_080_448),
        ("Speaker/segmentation.int8.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/9403a6902bb58e3d5ae8c7e77c3422de279db2e0/model.int8.onnx", 1_540_506),
        ("Speaker/campplus.onnx", "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx", 28_281_164)
    ]
    nonisolated static var directory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Models", isDirectory: true)
    }
    var downloading = false
    var progress: Double = 0
    var message = "최초 모델 설치에 약 1.79 GB가 필요합니다."
    var recognitionUpdateAvailable: Bool {
        !installed && Self.assets.filter { !$0.path.hasPrefix("Speaker/") && !$0.path.hasPrefix("ASR/") }.allSatisfy { asset in
            let size = try? Self.directory.appendingPathComponent(asset.path).resourceValues(forKeys: [.fileSizeKey]).fileSize
            return Int64(size ?? 0) == asset.size
        }
    }
    var installed: Bool {
        Self.assets.allSatisfy { asset in
            let size = try? Self.directory.appendingPathComponent(asset.path).resourceValues(forKeys: [.fileSizeKey]).fileSize
            return Int64(size ?? 0) == asset.size
        }
    }
    var requiredDownloadBytes: Int64 {
        Self.assets.reduce(0) { bytes, asset in
            let size = try? Self.directory.appendingPathComponent(asset.path).resourceValues(forKeys: [.fileSizeKey]).fileSize
            return bytes + (Int64(size ?? 0) == asset.size ? 0 : asset.size)
        }
    }

    /// Only discard unused files in the app's model cache after all active models loaded.
    func pruneModelCache() throws -> Int64 {
        let active = Set(Self.assets.map { Self.directory.appendingPathComponent($0.path).standardizedFileURL.path })
        guard installed, let files = FileManager.default.enumerator(at: Self.directory,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else { return 0 }
        var removedBytes: Int64 = 0
        for case let file as URL in files {
            let values = try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values.isRegularFile == true, !active.contains(file.standardizedFileURL.path) else { continue }
            try FileManager.default.removeItem(at: file)
            removedBytes += Int64(values.fileSize ?? 0)
        }
        return removedBytes
    }
    private var session: URLSession!
    private var waiter: CheckedContinuation<Void, Error>?
    private var activePaths: Set<String> = []
    private var downloadedBytes: [String: Int64] = [:]
    private var downloadError: Error?
    private let totalBytes = assets.reduce(Int64(0)) { $0 + $1.size }

    override init() {
        super.init()
        let configuration = URLSessionConfiguration.background(withIdentifier: "com.wonjun6040.iostranslator.models")
        configuration.isDiscretionary = false
        configuration.sessionSendsLaunchEvents = true
        configuration.httpMaximumConnectionsPerHost = 2
        configuration.timeoutIntervalForResource = 45 * 60
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
        Task { await restoreDownloads() }
    }

    private func restoreDownloads() async {
        let tasks = await session.allTasks
        let paths = Set(Self.assets.map(\.path))
        for task in tasks where !paths.contains(task.taskDescription ?? "") { task.cancel() }
        activePaths.formUnion(tasks.filter { paths.contains($0.taskDescription ?? "")
            && ($0.state == .running || $0.state == .suspended) }
            .compactMap(\.taskDescription))
        downloading = !activePaths.isEmpty
        if downloading { message = "다운로드 중 · 화면을 잠가도 이어집니다." }
        updateProgress()
    }

    /// Queue every missing file up front so iOS can keep downloading while the app is suspended.
    func download() async throws {
        await restoreDownloads()
        guard !installed else { return }
        downloadError = nil
        try await withCheckedThrowingContinuation { continuation in
            waiter = continuation
            for asset in Self.assets {
                let file = Self.directory.appendingPathComponent(asset.path)
                let size = try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize
                guard Int64(size ?? 0) != asset.size, !activePaths.contains(asset.path) else { continue }
                let task = session.downloadTask(with: URL(string: asset.url)!)
                task.taskDescription = asset.path
                activePaths.insert(asset.path)
                task.resume()
            }
            downloading = !activePaths.isEmpty
            message = "화면을 잠가도 다운로드가 계속됩니다."
            updateProgress()
            finishIfComplete()
        }
    }

    private func updateProgress() {
        var bytes: Int64 = 0
        for asset in Self.assets {
            let size = try? Self.directory.appendingPathComponent(asset.path).resourceValues(forKeys: [.fileSizeKey]).fileSize
            bytes += Int64(size ?? 0) == asset.size ? asset.size : min(downloadedBytes[asset.path] ?? 0, asset.size)
        }
        progress = min(1, Double(bytes) / Double(totalBytes))
    }

    private func finishIfComplete() {
        guard activePaths.isEmpty else { return }
        downloading = false
        updateProgress()
        let continuation = waiter
        waiter = nil
        if installed {
            message = "모델 설치 완료"
            continuation?.resume()
        } else {
            let error = downloadError ?? InterpreterError.message("일부 모델이 설치되지 않았습니다. 다시 다운로드해 주세요.")
            message = error.localizedDescription
            continuation?.resume(throwing: error)
        }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                               didFinishDownloadingTo location: URL) {
        guard let path = downloadTask.taskDescription,
              let asset = Self.assets.first(where: { $0.path == path }) else { return }
        var failure: Error?
        do {
            guard let response = downloadTask.response as? HTTPURLResponse,
                  (200...299).contains(response.statusCode) else { throw URLError(.badServerResponse) }
            let size = try location.resourceValues(forKeys: [.fileSizeKey]).fileSize
            guard Int64(size ?? 0) == asset.size else {
                throw InterpreterError.message("모델 파일이 불완전합니다. 다시 다운로드해 주세요.")
            }
            var file = Self.directory.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: file)
            try FileManager.default.moveItem(at: location, to: file)
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try file.setResourceValues(values)
        } catch { failure = error }
        let resultError = failure
        Task { @MainActor in
            activePaths.remove(path)
            downloadedBytes.removeValue(forKey: path)
            if let resultError { downloadError = resultError }
            finishIfComplete()
        }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                               didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                               totalBytesExpectedToWrite: Int64) {
        guard let path = downloadTask.taskDescription else { return }
        Task { @MainActor in downloadedBytes[path] = totalBytesWritten; updateProgress() }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error, let path = task.taskDescription else { return }
        Task { @MainActor in
            activePaths.remove(path)
            downloadedBytes.removeValue(forKey: path)
            downloadError = error
            finishIfComplete()
        }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor in
            let completion = AppDelegate.backgroundDownloadCompletion
            AppDelegate.backgroundDownloadCompletion = nil
            completion?()
        }
    }
}
