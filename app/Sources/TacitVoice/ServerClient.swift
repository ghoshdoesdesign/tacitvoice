import Foundation

struct Transcript: Decodable {
    let id: String
    let text: String
    let alternatives: [String]
    let mode: String
    let warnings: [String]
}

struct Health: Decodable {
    let ready: Bool
    let error: String?
    let lips: Bool
    let cleanup: Bool
    let problems: [String]
}

/// Talks to the local recognition server (scripts/server.sh) on this Mac.
final class ServerClient {
    private let base = URL(string: "http://127.0.0.1:8765")!
    private let session: URLSession

    init() {
        let config = URLSessionConfiguration.ephemeral  // nothing cached to disk
        config.timeoutIntervalForRequest = 60
        session = URLSession(configuration: config)
    }

    /// The server writes this file on start; it proves a request comes from this user.
    private func token() -> String {
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".tacitvoice/token")
        let text = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func health(completion: @escaping (Health?) -> Void) {
        var request = URLRequest(url: base.appendingPathComponent("health"))
        request.timeoutInterval = 2
        session.dataTask(with: request) { data, _, _ in
            let health = data.flatMap { try? JSONDecoder().decode(Health.self, from: $0) }
            DispatchQueue.main.async { completion(health) }
        }.resume()
    }

    func transcribe(
        _ clip: Clip, app: String, textBeforeCursor: String,
        completion: @escaping (Result<Transcript, ClientError>) -> Void
    ) {
        let meta: [String: Any] = [
            "sampleRate": clip.sampleRate,
            "timestamps": clip.timestamps,
            "app": app,
            "textBeforeCursor": textBeforeCursor,
        ]
        let metaJSON = (try? JSONSerialization.data(withJSONObject: meta)) ?? Data("{}".utf8)

        let boundary = "tacitvoice-\(UUID().uuidString)"
        var body = Data()
        func part(_ name: String, filename: String?, type: String, _ content: Data) {
            var header = "--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\""
            if let filename = filename { header += "; filename=\"\(filename)\"" }
            header += "\r\nContent-Type: \(type)\r\n\r\n"
            body.append(Data(header.utf8))
            body.append(content)
            body.append(Data("\r\n".utf8))
        }
        part("meta", filename: nil, type: "text/plain; charset=utf-8", metaJSON)
        if !clip.audio.isEmpty {
            part("audio", filename: "audio.pcm", type: "application/octet-stream", clip.audio)
        }
        for (index, frame) in clip.frames.enumerated() {
            part("frames", filename: "\(index).jpg", type: "image/jpeg", frame)
        }
        body.append(Data("--\(boundary)--\r\n".utf8))

        var request = URLRequest(url: base.appendingPathComponent("transcribe"))
        request.httpMethod = "POST"
        request.setValue("Bearer \(token())", forHTTPHeaderField: "Authorization")
        request.setValue(
            "multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        session.uploadTask(with: request, from: body) { data, response, error in
            let result: Result<Transcript, ClientError>
            if error != nil {
                result = .failure(.message("Server not running. Start scripts/server.sh"))
            } else if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                switch http.statusCode {
                case 401: result = .failure(.message("Server rejected the app's token."))
                case 503: result = .failure(.message("Models are still loading."))
                default: result = .failure(.message("Server error \(http.statusCode)."))
                }
            } else if let data = data,
                let transcript = try? JSONDecoder().decode(Transcript.self, from: data)
            {
                result = .success(transcript)
            } else {
                result = .failure(.message("Unreadable reply from the server."))
            }
            DispatchQueue.main.async { completion(result) }
        }.resume()
    }
}

enum ClientError: Error {
    case message(String)

    var text: String {
        switch self {
        case .message(let text): return text
        }
    }
}
