import SwiftUI
import UIKit

@main
struct IOSTranslatorApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var interpreter = Interpreter()
    var body: some Scene {
        WindowGroup { TranslatorView(interpreter: interpreter) }
    }
}

@MainActor
final class AppDelegate: NSObject, UIApplicationDelegate {
    static var backgroundDownloadCompletion: (() -> Void)?
    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        Self.backgroundDownloadCompletion = completionHandler
    }
}
