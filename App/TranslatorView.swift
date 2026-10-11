import SwiftUI
import AVFoundation

struct TranslatorView: View {
    @Bindable var interpreter: Interpreter
    @Environment(\.scenePhase) private var scenePhase
    private let languages = [("ko", "한국어"), ("en", "English"), ("ja", "日本語")]

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                HStack(spacing: 20) {
                    languageMenu("입력 언어", selection: $interpreter.sourceLanguage)
                    Button {
                        let source = interpreter.sourceLanguage
                        interpreter.sourceLanguage = interpreter.targetLanguage
                        interpreter.targetLanguage = source
                    } label: { Image(systemName: "arrow.left.arrow.right") }
                        .accessibilityLabel("언어 바꾸기")
                        .disabled(busy)
                    languageMenu("출력 언어", selection: $interpreter.targetLanguage)
                }
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(spacing: 16) {
                            textCard(language: interpreter.sourceLanguage, text: interpreter.confirmed,
                                partial: interpreter.partial, placeholder: interpreter.recording ? "듣고 있습니다…" : "마이크를 눌러 말해 주세요", translated: false)
                            textCard(language: interpreter.targetLanguage, text: interpreter.translated,
                                partial: "", placeholder: interpreter.translating ? "번역 중…" : "번역이 여기에 표시됩니다", translated: true)
                            Color.clear.frame(height: 1).id("bottom")
                        }
                    }
                    .onChange(of: interpreter.translated) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
                }
                if let message = interpreter.message {
                    Text(message).font(.footnote).foregroundStyle(.orange)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                if !interpreter.ready { modelInstallation }
                routeInformation
                HStack(spacing: 46) {
                    Image(systemName: "checkmark.shield")
                        .foregroundStyle(interpreter.ready ? Color.green : Color.secondary)
                        .accessibilityLabel(interpreter.ready ? "오프라인 모델 준비됨" : "모델 설치 필요")
                    Button {
                        if interpreter.recording { interpreter.stop() }
                        else { Task { await interpreter.start() } }
                    } label: {
                        Image(systemName: interpreter.recording ? "stop.fill" : "mic.fill")
                            .font(.system(size: 28, weight: .medium))
                            .foregroundStyle(.white)
                            .frame(width: 76, height: 76)
                            .background(interpreter.recording ? Color.red : Color.blue, in: Circle())
                    }
                    .accessibilityLabel(interpreter.recording ? "통역 중지" : "통역 시작")
                    .disabled(!interpreter.ready || interpreter.preparing || interpreter.stopping
                        || (!interpreter.recording && interpreter.translating))
                    Button { interpreter.speakTranslation.toggle() } label: {
                        Image(systemName: interpreter.speakTranslation ? "speaker.wave.2.fill" : "speaker.slash")
                            .font(.title3)
                    }
                    .accessibilityLabel(interpreter.speakTranslation ? "음성 출력 끄기" : "음성 출력 켜기")
                }
                Text(interpreter.recording ? "통역 중 · 눌러서 중지" : interpreter.stopping ? "마지막 발화 처리 중…" : "눌러서 통역 시작")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 20)
            .background(Color(.systemGroupedBackground))
            .navigationTitle("번역")
            .navigationBarTitleDisplayMode(.inline)
            .task { await interpreter.prepare() }
            .onChange(of: interpreter.models.downloading) { _, downloading in
                if !downloading, !interpreter.ready, interpreter.models.installed {
                    Task { await interpreter.prepare() }
                }
            }
            .onChange(of: scenePhase) { _, phase in
                if phase == .active, !interpreter.ready, interpreter.models.installed {
                    Task { await interpreter.prepare() }
                }
            }
        }
    }
    private var busy: Bool {
        interpreter.recording || interpreter.preparing || interpreter.stopping || interpreter.translating
    }
    private func languageMenu(_ title: String, selection: Binding<String>) -> some View {
        Menu {
            Picker(title, selection: selection) {
                ForEach(languages, id: \.0) { language in Text(language.1).tag(language.0) }
            }
        } label: {
            VStack(alignment: .leading, spacing: 5) {
                Text(title).font(.caption).foregroundStyle(.secondary)
                HStack(spacing: 6) {
                    Text(name(selection.wrappedValue)).font(.headline)
                    Image(systemName: "chevron.down").font(.caption2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .disabled(busy)
    }
    private func name(_ language: String) -> String { languages.first { $0.0 == language }?.1 ?? language }
    private func textCard(language: String, text: String, partial: String, placeholder: String, translated: Bool) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(name(language)).font(.subheadline.weight(.semibold)).foregroundStyle(.secondary)
            if text.isEmpty && partial.isEmpty {
                Text(placeholder).foregroundStyle(.tertiary)
            } else {
                if !text.isEmpty { Text(text).foregroundStyle(translated ? Color.blue : Color.primary) }
                if !partial.isEmpty { Text(partial).foregroundStyle(.secondary) }
            }
            Spacer(minLength: 0)
        }
        .font(.title2).textSelection(.enabled)
        .padding(22)
        .frame(maxWidth: .infinity, minHeight: 185, alignment: .topLeading)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 22))
    }
    private var modelInstallation: some View {
        VStack(alignment: .leading, spacing: 8) {
            if interpreter.models.downloading {
                ProgressView(value: interpreter.models.progress)
                Text(interpreter.models.message).font(.caption).foregroundStyle(.secondary)
            } else if interpreter.preparing {
                ProgressView("오프라인 모델 준비 중…").font(.footnote)
            } else {
                Text("오프라인 통역 모델 설치").font(.subheadline.weight(.semibold))
                Text("Qwen STT 0.6B · Qwen 번역 0.6B · Silero VAD\n약 1.48 GB · 최초 다운로드에만 인터넷이 필요합니다.")
                    .font(.caption).foregroundStyle(.secondary)
                Button(interpreter.models.installed ? "모델 불러오기" : "모델 다운로드") {
                    Task { await interpreter.prepare(download: !interpreter.models.installed) }
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    private var routeInformation: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Menu {
                    Text("HFP 입력·출력은 iOS가 함께 라우팅합니다. iPhone 마이크를 선택하면 출력도 바뀔 수 있습니다.")
                    Picker("마이크", selection: $interpreter.microphoneUID) {
                        Text("자동 · Bluetooth 우선").tag("")
                        ForEach(interpreter.inputs, id: \.uid) { input in Text(input.portName).tag(input.uid) }
                        if !interpreter.microphoneUID.isEmpty,
                           !interpreter.inputs.contains(where: { $0.uid == interpreter.microphoneUID }) {
                            Text("연결이 해제된 마이크").tag(interpreter.microphoneUID)
                        }
                    }
                } label: { Label(interpreter.actualInput, systemImage: "mic").lineLimit(1) }
                    .disabled(interpreter.preparing || interpreter.stopping)
                Spacer()
                Label(interpreter.bluetoothConnected ? "Bluetooth 연결됨" : "iPhone",
                    systemImage: interpreter.bluetoothConnected ? "airpods" : "iphone")
                    .foregroundStyle(.secondary).lineLimit(1)
            }
            Label("출력 · \(interpreter.actualOutput)", systemImage: "speaker.wave.2")
                .foregroundStyle(.secondary)
            if interpreter.recording {
                Text(interpreter.duplexMessage).foregroundStyle(.secondary)
            }
        }
        .font(.caption)
    }
}
