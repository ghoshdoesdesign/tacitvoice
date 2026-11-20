import Foundation

struct Transcript: Decodable {
    let id: String
    let text: String
    let alternatives: [String]
    let mode: String
    let warnings: [String]
}

struct ServerProblem: Decodable {
    let detail: String
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
    let base = URL(string: "http://127.0.0.1:8765")!
    let session: URLSession

    init() {
        let config = URLSessionConfiguration.ephemeral  // nothing cached to disk
        config.timeoutIntervalForRequest = 60
        session = URLSession(configuration: config)
    }

    /// The server writes this file on start; it proves a request comes from this user.
    func token() -> String {
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
                default:
                    // The server explains itself in {"detail": "..."}.
                    let problem = data.flatMap {
                        try? JSONDecoder().decode(ServerProblem.self, from: $0)
                    }
                    result = .failure(
                        .message(problem?.detail ?? "Server error \(http.statusCode)."))
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

struct CommandReply: Decodable {
    let reply: String
}

extension ServerClient {
    /// Hand a recognised sentence to the OpenClaw agent on this Mac. Tasks can
    /// take a while, so this waits longer than a transcription does.
    func command(_ text: String, completion: @escaping (Result<String, ClientError>) -> Void) {
        var request = URLRequest(url: base.appendingPathComponent("command"))
        request.httpMethod = "POST"
        request.timeoutInterval = 150
        request.setValue("Bearer \(token())", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["text": text])

        session.dataTask(with: request) { data, response, error in
            let result: Result<String, ClientError>
            if error != nil {
                result = .failure(.message("No answer from the server."))
            } else if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                let problem = data.flatMap { try? JSONDecoder().decode(ServerProblem.self, from: $0) }
                result = .failure(.message(problem?.detail ?? "Server error \(http.statusCode)."))
            } else if let data = data,
                let answer = try? JSONDecoder().decode(CommandReply.self, from: data)
            {
                result = .success(answer.reply)
            } else {
                result = .failure(.message("Unreadable reply from the server."))
            }
            DispatchQueue.main.async { completion(result) }
        }.resume()
    }
}

extension ServerClient {
    /// Ask the server to stop the task the agent is working on. Nothing waits for the answer.
    func cancelCommand() {
        var request = URLRequest(url: base.appendingPathComponent("command/cancel"))
        request.httpMethod = "POST"
        request.timeoutInterval = 5
        request.setValue("Bearer \(token())", forHTTPHeaderField: "Authorization")
        session.dataTask(with: request).resume()
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
