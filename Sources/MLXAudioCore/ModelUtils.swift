import Foundation
import HuggingFace

/// Progress info for model downloads, including byte-level tracking.
public struct ModelDownloadProgress: Sendable {
    /// Bytes downloaded so far
    public let bytesDownloaded: Int64
    /// Estimated total bytes (0 if unknown)
    public let bytesTotal: Int64
    /// Fraction completed (0.0 to 1.0)
    public let fractionCompleted: Double
    /// Number of files completed
    public let filesCompleted: Int64
    /// Total number of files
    public let filesTotal: Int64
}

public typealias ModelDownloadProgressHandler = @MainActor @Sendable (ModelDownloadProgress) -> Void

public enum ModelUtils {
    public static func resolveModelType(
        repoID: Repo.ID,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> String? {
        let modelNameComponents = repoID.name.split(separator: "/").last?.split(separator: "-")
        let modelURL = try await resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache
        )
        let configJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: modelURL.appendingPathComponent("config.json")))
        if let config = configJSON as? [String: Any] {
            return (config["model_type"] as? String) ?? (config["architecture"] as? String) ?? modelNameComponents?.first?.lowercased()
        }
        return nil
    }

    /// Resolves a model from cache or downloads it if not cached.
    public static func resolveOrDownloadModel(
        repoID: Repo.ID,
        requiredExtension: String,
        additionalMatchingPatterns: [String] = [],
        hfToken: String? = nil,
        cache: HubCache = .default,
        progressHandler: ModelDownloadProgressHandler? = nil
    ) async throws -> URL {
        let client: HubClient
        if let token = hfToken, !token.isEmpty {
            print("Using HuggingFace token from configuration")
            client = HubClient(host: HubClient.defaultHost, bearerToken: token, cache: cache)
        } else {
            client = HubClient(cache: cache)
        }
        let resolvedCache = client.cache ?? cache
        return try await resolveOrDownloadModel(
            client: client,
            cache: resolvedCache,
            repoID: repoID,
            requiredExtension: requiredExtension,
            additionalMatchingPatterns: additionalMatchingPatterns,
            progressHandler: progressHandler
        )
    }

    /// Resolves a model from cache or downloads it if not cached.
    public static func resolveOrDownloadModel(
        client: HubClient,
        cache: HubCache = .default,
        repoID: Repo.ID,
        requiredExtension: String,
        additionalMatchingPatterns: [String] = [],
        progressHandler: ModelDownloadProgressHandler? = nil
    ) async throws -> URL {
        let normalizedRequiredExtension = requiredExtension.hasPrefix(".")
            ? String(requiredExtension.dropFirst())
            : requiredExtension

        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let modelDir = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)

        // Check if model already exists with required files
        if FileManager.default.fileExists(atPath: modelDir.path) {
            let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: [.fileSizeKey])
            let hasRequiredFile = files?.contains { file in
                guard file.pathExtension == normalizedRequiredExtension else { return false }
                let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                return size > 0
            } ?? false

            if hasRequiredFile {
                let configPath = modelDir.appendingPathComponent("config.json")
                if FileManager.default.fileExists(atPath: configPath.path) {
                    if let configData = try? Data(contentsOf: configPath),
                       let _ = try? JSONSerialization.jsonObject(with: configData) {
                        print("Using cached model at: \(modelDir.path)")
                        return modelDir
                    } else {
                        print("Cached config.json is invalid, clearing cache...")
                        Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
                    }
                }
            } else {
                print("Cached model appears incomplete, clearing cache...")
                Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
            }
        }

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        var allowedExtensions: Set<String> = [
            "*.\(normalizedRequiredExtension)",
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.wav",
        ]
        allowedExtensions.formUnion(additionalMatchingPatterns)

        print("Downloading model \(repoID)...")

        // Start a byte-level progress polling task alongside the download
        let pollingTask: Task<Void, Never>?
        if let progressHandler {
            let dirURL = modelDir
            // Also monitor the HubClient cache directory for in-progress downloads
            let hubCacheDir = cache.cacheDirectory
            pollingTask = Task { @MainActor in
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
                    guard !Task.isCancelled else { break }

                    // Sum up bytes in model dir + hub cache dir (in-progress files)
                    let modelDirBytes = Self.directorySize(dirURL)
                    let hubBytes = Self.directorySize(hubCacheDir)
                    let totalBytes = modelDirBytes + hubBytes

                    let progress = ModelDownloadProgress(
                        bytesDownloaded: totalBytes,
                        bytesTotal: 0, // unknown until download completes
                        fractionCompleted: 0,
                        filesCompleted: 0,
                        filesTotal: 0
                    )
                    progressHandler(progress)
                }
            }
        } else {
            pollingTask = nil
        }

        var lastFilesCompleted: Int64 = 0
        var lastFilesTotal: Int64 = 0

        _ = try await client.downloadSnapshot(
            of: repoID,
            kind: .model,
            to: modelDir,
            revision: "main",
            matching: Array(allowedExtensions),
            progressHandler: { hubProgress in
                lastFilesCompleted = hubProgress.completedUnitCount
                lastFilesTotal = hubProgress.totalUnitCount
            }
        )

        pollingTask?.cancel()

        // Report completion
        if let progressHandler {
            let finalSize = Self.directorySize(modelDir)
            let progress = ModelDownloadProgress(
                bytesDownloaded: finalSize,
                bytesTotal: finalSize,
                fractionCompleted: 1.0,
                filesCompleted: lastFilesCompleted,
                filesTotal: lastFilesTotal
            )
            await progressHandler(progress)
        }

        // Post-download validation
        let downloadedFiles = try? FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: [.fileSizeKey]
        )
        let hasValidFile = downloadedFiles?.contains { file in
            guard file.pathExtension == normalizedRequiredExtension else { return false }
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return size > 0
        } ?? false

        if !hasValidFile {
            Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
            throw ModelUtilsError.incompleteDownload(repoID.description)
        }

        print("Model downloaded to: \(modelDir.path)")
        return modelDir
    }

    /// Calculate total size of all files in a directory (non-recursive for performance).
    private static func directorySize(_ url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize {
                total += Int64(size)
            }
        }
        return total
    }

    private static func clearCaches(modelDir: URL, repoID: Repo.ID, hubCache: HubCache) {
        try? FileManager.default.removeItem(at: modelDir)
        let hubRepoDir = hubCache.repoDirectory(repo: repoID, kind: .model)
        if FileManager.default.fileExists(atPath: hubRepoDir.path) {
            print("Clearing Hub cache at: \(hubRepoDir.path)")
            try? FileManager.default.removeItem(at: hubRepoDir)
        }
    }
}

public enum ModelUtilsError: LocalizedError {
    case incompleteDownload(String)

    public var errorDescription: String? {
        switch self {
        case .incompleteDownload(let repo):
            return "Downloaded model '\(repo)' has missing or zero-byte weight files. "
                + "The cache has been cleared — please try again."
        }
    }
}
