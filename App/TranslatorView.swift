import SwiftUI
import AVFoundation

struct TranslatorView: View {
    @Bindable var interpreter: Interpreter
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                HStack(spacing: 20) {
                    languageMenu("상대방 언어", selection: $interpreter.sourceLanguage, allowsAuto: true)
                    Button {
                        let source = interpreter.sourceLanguage == "auto"
                            ? (interpreter.detectedLanguage ?? "en") : interpreter.sourceLanguage
                        interpreter.sourceLanguage = interpreter.targetLanguage
                        interpreter.targetLanguage = source
                    } label: { Image(systemName: "arrow.left.arrow.right") }
                        .accessibilityLabel("언어 바꾸기")
                        .disabled(busy || (interpreter.sourceLanguage == "auto" && interpreter.detectedLanguage == nil)
                            || !LocalTranslator.supportedLanguageCodes.contains(interpreter.sourceLanguage == "auto"
                                ? (interpreter.detectedLanguage ?? "") : interpreter.sourceLanguage))
                    languageMenu("내 언어", selection: $interpreter.targetLanguage)
                }
                HStack(spacing: 8) {
                    Circle().fill(interpreter.inputPaused ? Color.orange : interpreter.recording ? Color.green : Color.secondary)
                        .frame(width: 7, height: 7)
                    Text(interpreter.inputPaused ? "번역 음성 재생 중" : interpreter.recording ? "계속 듣고 있습니다" : "실시간 대화")
                    if interpreter.translating { ProgressView().controlSize(.mini); Text("번역 중") }
                    Spacer()
                    if interpreter.sourceLanguage == "auto", let language = interpreter.detectedLanguage {
                        Text("\(name(language)) 감지됨")
                    }
                }
                .font(.caption).foregroundStyle(.secondary)
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 16) {
                            if interpreter.turns.isEmpty && interpreter.partial.isEmpty {
                                VStack(alignment: .leading, spacing: 16) {
                                    Image(systemName: "airpods").font(.system(size: 44)).foregroundStyle(.teal)
                                    Text(interpreter.recording ? "상대방의 말을 기다리고 있습니다" : "자연스럽게 대화하세요")
                                        .font(.title2.weight(.semibold))
                                    Text("상대방의 말을 듣고 내 언어로 번역합니다. 내 답변도 상대방에게 들려줄 수 있습니다.")
                                        .font(.body).foregroundStyle(.secondary)
                                }
                                .padding(24).frame(maxWidth: .infinity, alignment: .leading)
                            }
                            ForEach(interpreter.turns) { turn in conversationCard(turn) }
                            if !interpreter.partial.isEmpty {
                                VStack(alignment: .leading, spacing: 10) {
                                    Text("인식 중 · \(name(interpreter.partialLanguage ?? interpreter.sourceLanguage))")
                                        .font(.caption).foregroundStyle(.secondary)
                                    Text(interpreter.partial).font(.body).foregroundStyle(.secondary)
                                    if interpreter.activeTurnID == nil && !interpreter.liveTranslation.isEmpty {
                                        Text(interpreter.liveTranslation).font(.title3).foregroundStyle(.secondary)
                                    }
                                }
                                .padding(20).frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 20))
                            }
                            Color.clear.frame(height: 1).id("bottom")
                        }
                    }
                    .onChange(of: interpreter.turns.count) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
                    .onChange(of: interpreter.liveTranslation) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
                    .onChange(of: interpreter.partial) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
                }
                if let message = interpreter.message {
                    Text(message).font(.footnote).foregroundStyle(.orange)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                if !interpreter.ready { modelInstallation }
                if interpreter.registeringVoice {
                    ProgressView(value: interpreter.registrationProgress)
                    Text("다른 사람이 말하지 않는 곳에서 혼자 6초 이상 자연스럽게 말해 주세요.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                routeInformation
                HStack(spacing: 16) {
                    Image(systemName: "checkmark.shield")
                        .foregroundStyle(interpreter.ready ? Color.green : Color.secondary)
                        .accessibilityLabel(interpreter.ready ? "오프라인 모델 준비됨" : "모델 설치 필요")
                    Button {
                        if interpreter.recording || interpreter.registeringVoice { interpreter.stop() }
                        else { Task { await interpreter.start() } }
                    } label: {
                        Label(interpreter.registeringVoice ? "등록 중지" : interpreter.recording ? "통역 중지" : "통역 시작",
                            systemImage: interpreter.recording || interpreter.registeringVoice ? "stop.fill" : "mic.fill")
                            .font(.headline)
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity).frame(height: 54)
                            .background(interpreter.recording ? Color.red : Color.teal, in: Capsule())
                    }
                    .accessibilityLabel(interpreter.registeringVoice ? "등록 중지" : interpreter.recording ? "통역 중지" : "통역 시작")
                    .disabled(!interpreter.ready
                        || interpreter.preparing || interpreter.stopping
                        || (!interpreter.recording && interpreter.translating))
                    Button { interpreter.speakTranslation.toggle() } label: {
                        Image(systemName: interpreter.speakTranslation ? "speaker.wave.2.fill" : "speaker.slash")
                            .font(.title3)
                    }
                    .accessibilityLabel(interpreter.speakTranslation ? "음성 출력 끄기" : "음성 출력 켜기")
                }
                Text(interpreter.stopping ? "마지막 발화 처리 중…" : "한 번 시작하면 대화를 계속 듣습니다")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 20)
            .background(Color(.systemGroupedBackground))
            .navigationTitle("실시간 번역")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { voiceMenu } }
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
            || interpreter.registeringVoice
    }
    private func languageMenu(_ title: String, selection: Binding<String>, allowsAuto: Bool = false) -> some View {
        Menu {
            Picker(title, selection: selection) {
                if allowsAuto { Text("자동 감지").tag("auto") }
                ForEach(SpeechLanguages.supported.filter { allowsAuto || LocalTranslator.supportedLanguageCodes.contains($0.code) }, id: \.code) { language in
                    Text(language.label + (LocalTranslator.supportedLanguageCodes.contains(language.code) ? "" : " · 인식만"))
                        .tag(language.code)
                }
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
    private func name(_ language: String) -> String { SpeechLanguages.label(for: language) }
    private func conversationCard(_ turn: TranslationTurn) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label(speakerName(turn) + " · \(name(turn.source))",
                    systemImage: turn.speakerID != nil && turn.speakerID == interpreter.wearerSpeakerID ? "person.fill" : "person.wave.2")
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                if let speakerID = turn.speakerID {
                    Menu {
                        Button("이 화자를 내 목소리로 지정") { interpreter.selectWearer(speakerID) }
                    } label: { Image(systemName: "ellipsis").padding(4) }
                    .accessibilityLabel("화자 지정")
                }
            }
            Text(turn.original).font(.body).foregroundStyle(.secondary)
            if !turn.translation.isEmpty {
                Text(turn.translation).font(.title2).foregroundStyle(turn.isReply ? Color.primary : Color.blue)
                if turn.isReply {
                    Button { interpreter.playReply(turn) } label: {
                        Label("상대방에게 들려주기", systemImage: "speaker.wave.2")
                    }
                    .font(.subheadline).disabled(interpreter.preparing || interpreter.stopping || interpreter.registeringVoice)
                }
            } else if turn.target == "auto" {
                Text("상대방 언어를 확인하면 번역합니다.").font(.footnote).foregroundStyle(.secondary)
            } else if let error = turn.error {
                Text(error).font(.footnote).foregroundStyle(.orange)
            } else if interpreter.activeTurnID == turn.id && !interpreter.liveTranslation.isEmpty {
                Text(interpreter.liveTranslation).font(.title2).foregroundStyle(.secondary)
            } else {
                ProgressView(interpreter.activeTurnID == turn.id ? "번역 중…" : "번역 대기 중…").font(.caption)
            }
        }
        .textSelection(.enabled).padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 20))
    }
    private func speakerName(_ turn: TranslationTurn) -> String {
        guard let id = turn.speakerID else { return turn.isReply ? "내 언어 발화" : "음성" }
        return id == interpreter.wearerSpeakerID ? "나" : "화자 \(id)"
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
                Text("추가 다운로드 약 \(String(format: "%.2f", Double(interpreter.models.requiredDownloadBytes) / 1_000_000_000)) GB · 설치 후에는 인터넷 없이 사용할 수 있습니다.")
                    .font(.caption).foregroundStyle(.secondary)
                Button(interpreter.models.installed ? "모델 불러오기" : "모델 다운로드") {
                    Task { await interpreter.prepare(download: !interpreter.models.installed) }
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    private var voiceMenu: some View {
        Menu {
            Button(interpreter.wearerSpeakerID == nil ? "내 목소리 등록" : "내 목소리 다시 등록") {
                Task { await interpreter.registerVoice() }
            }.disabled(busy || !interpreter.ready)
            Toggle("내 목소리 제외", isOn: $interpreter.excludeWearer)
                .disabled(interpreter.wearerSpeakerID == nil)
            if interpreter.wearerSpeakerID != nil {
                Button("등록 삭제") { interpreter.forgetWearer() }.disabled(busy)
            }
        } label: { Image(systemName: "person.crop.circle") }
        .accessibilityLabel("내 목소리 설정")
    }
    private var routeInformation: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Menu {
                    Picker("마이크", selection: $interpreter.microphoneUID) {
                        Text("자동").tag("")
                        ForEach(interpreter.inputs, id: \.uid) { input in Text(input.portName).tag(input.uid) }
                        if !interpreter.microphoneUID.isEmpty,
                           !interpreter.inputs.contains(where: { $0.uid == interpreter.microphoneUID }) {
                            Text("연결이 해제된 마이크").tag(interpreter.microphoneUID)
                        }
                    }
                } label: { Label(interpreter.actualInput, systemImage: "mic").lineLimit(1) }
                    .disabled(busy)
                Spacer()
                Label(interpreter.actualOutput, systemImage: interpreter.bluetoothConnected ? "airpods" : "speaker.wave.2")
                    .foregroundStyle(.secondary).lineLimit(1)
            }
            if interpreter.recording {
                Text(interpreter.microphoneMessage).foregroundStyle(.secondary)
            }
        }
        .font(.caption)
    }
}
