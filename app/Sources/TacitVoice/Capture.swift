import AVFoundation
import CoreImage
import ImageIO

/// One held-key recording, kept in memory only.
struct Clip {
    var frames: [Data] = []        // JPEG images, in order
    var timestamps: [Double] = []  // seconds since the first frame
    var audio = Data()             // 16-bit little-endian mono PCM
    var sampleRate: Double = 16000
}

/// Camera and microphone capture that runs only between start() and stop().
/// The camera's green light is on exactly while this is recording.
final class Capture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate,
    AVCaptureAudioDataOutputSampleBufferDelegate
{
    /// Called on the main thread when the first frame of a recording arrives.
    var onFirstFrame: (() -> Void)?

    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "tacitvoice.capture")
    private let videoOutput = AVCaptureVideoDataOutput()
    private let audioOutput = AVCaptureAudioDataOutput()
    private let ciContext = CIContext()
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private let targetWidth: CGFloat = 960
    private let maxFrames = 30 * 60  // stop collecting after about a minute

    // Touched only on `queue`.
    private var configured = false
    private var recording = false
    private var clip = Clip()
    private var firstFrameTime: Double?
    private var audioFormatOK = true

    private func configure() -> String? {
        if configured { return nil }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1280x720

        guard let camera = AVCaptureDevice.default(for: .video),
            let cameraInput = try? AVCaptureDeviceInput(device: camera),
            session.canAddInput(cameraInput)
        else { return "No camera available." }
        session.addInput(cameraInput)

        guard let mic = AVCaptureDevice.default(for: .audio),
            let micInput = try? AVCaptureDeviceInput(device: mic),
            session.canAddInput(micInput)
        else { return "No microphone available." }
        session.addInput(micInput)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(videoOutput) else { return "Cannot read the camera." }
        session.addOutput(videoOutput)

        // Ask for the format the speech model wants, so no conversion is needed.
        audioOutput.audioSettings = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        audioOutput.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(audioOutput) else { return "Cannot read the microphone." }
        session.addOutput(audioOutput)

        configured = true
        return nil
    }

    /// Turn the camera and microphone on. `failed` is called on the main
    /// thread with a message if they cannot be started.
    func start(failed: @escaping (String) -> Void) {
        queue.async {
            if let problem = self.configure() {
                DispatchQueue.main.async { failed(problem) }
                return
            }
            self.clip = Clip()
            self.firstFrameTime = nil
            self.audioFormatOK = true
            self.recording = true
            self.session.startRunning()
        }
    }

    /// Turn everything off and hand back what was recorded (on the main thread).
    func stop(completion: @escaping (Clip) -> Void) {
        queue.async {
            self.recording = false
            if self.session.isRunning { self.session.stopRunning() }
            let clip = self.clip
            self.clip = Clip()
            DispatchQueue.main.async { completion(clip) }
        }
    }

    func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard recording else { return }
        if output === videoOutput {
            appendVideo(sampleBuffer)
        } else if output === audioOutput {
            appendAudio(sampleBuffer)
        }
    }

    private func appendVideo(_ sampleBuffer: CMSampleBuffer) {
        guard clip.frames.count < maxFrames,
            let pixels = CMSampleBufferGetImageBuffer(sampleBuffer)
        else { return }
        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        var image = CIImage(cvPixelBuffer: pixels)
        let scale = targetWidth / image.extent.width
        if scale < 1 {
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        let quality = CIImageRepresentationOption(
            rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard
            let jpeg = ciContext.jpegRepresentation(
                of: image, colorSpace: colorSpace, options: [quality: 0.8])
        else { return }

        if firstFrameTime == nil {
            firstFrameTime = time
            DispatchQueue.main.async { self.onFirstFrame?() }
        }
        clip.frames.append(jpeg)
        clip.timestamps.append(time - (firstFrameTime ?? time))
    }

    private func appendAudio(_ sampleBuffer: CMSampleBuffer) {
        guard audioFormatOK,
            let format = CMSampleBufferGetFormatDescription(sampleBuffer),
            let description = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee,
            let block = CMSampleBufferGetDataBuffer(sampleBuffer)
        else { return }
        let isFloat = description.mFormatFlags & kAudioFormatFlagIsFloat != 0
        guard description.mBitsPerChannel == 16, description.mChannelsPerFrame == 1, !isFloat else {
            // The server expects 16-bit mono. Sending anything else would be
            // decoded as noise, so send no audio (silent mode still works).
            NSLog("TacitVoice: unexpected microphone format, audio dropped: \(description)")
            audioFormatOK = false
            clip.audio = Data()
            return
        }
        clip.sampleRate = description.mSampleRate
        let length = CMBlockBufferGetDataLength(block)
        guard length > 0 else { return }
        var bytes = Data(count: length)
        let status = bytes.withUnsafeMutableBytes { raw -> OSStatus in
            guard let base = raw.baseAddress else { return -1 }
            return CMBlockBufferCopyDataBytes(
                block, atOffset: 0, dataLength: length, destination: base)
        }
        if status == kCMBlockBufferNoErr { clip.audio.append(bytes) }
    }
}
