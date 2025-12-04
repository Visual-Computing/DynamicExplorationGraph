#pragma once

/**
 * @file file_io.h
 * @brief File I/O utilities for vector files (fvecs, ivecs format) and filesystem operations.
 */

#include <filesystem>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdlib>

#include <fmt/core.h>

namespace deglib::benchmark
{

// ============================================================================
// Vector File I/O (fvecs, ivecs)
// ============================================================================

/**
 * Write an ivecs file (vectors of uint32_t).
 * Format: Each vector is stored as [dimension (4 bytes)][data (dim * 4 bytes)]
 * 
 * @param fname Output file path
 * @param d Dimension of each vector
 * @param n Number of vectors
 * @param v Pointer to vector data (n * d elements)
 */
inline void ivecs_write(const char* fname, uint32_t d, size_t n, const uint32_t* v) {
    auto out = std::ofstream(fname, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
        fmt::print(stderr, "Error opening file for write: {}\n", fname);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        const auto ptr = v + i * d;
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
        out.write(reinterpret_cast<const char*>(ptr), sizeof(uint32_t) * d);
    }
    out.close();
}

/**
 * Write an fvecs file (vectors of float).
 * Format: Each vector is stored as [dimension (4 bytes)][data (dim * 4 bytes)]
 * 
 * @param fname Output file path
 * @param d Dimension of each vector
 * @param n Number of vectors
 * @param v Pointer to vector data (n * d elements)
 */
inline void fvecs_write(const char* fname, uint32_t d, size_t n, const float* v) {
    auto out = std::ofstream(fname, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
        fmt::print(stderr, "Error opening file for write: {}\n", fname);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        const auto ptr = v + i * d;
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
        out.write(reinterpret_cast<const char*>(ptr), sizeof(float) * d);
    }
    out.close();
}

// ============================================================================
// Filesystem Utilities
// ============================================================================

/**
 * Check if a file or directory exists.
 */
inline bool file_exists(const std::string& path) {
    return std::filesystem::exists(path);
}

inline bool file_exists(const std::filesystem::path& path) {
    return std::filesystem::exists(path);
}

/**
 * Ensure a directory exists, creating it if necessary.
 * @return true if directory exists or was created successfully
 */
inline bool ensure_directory(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) return true;
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) {
        fmt::print(stderr, "Error creating directory '{}': {}\n", path.string(), ec.message());
        return false;
    }
    return true;
}

/**
 * Delete a file if it exists.
 * @return true if file was deleted or didn't exist
 */
inline bool delete_file(const std::string& path) {
    if (!std::filesystem::exists(path)) return true;
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (ec) {
        fmt::print(stderr, "Error deleting file '{}': {}\n", path, ec.message());
        return false;
    }
    fmt::print("Deleted: {}\n", path);
    return true;
}

inline bool delete_file(const std::filesystem::path& path) {
    return delete_file(path.string());
}

/**
 * Rename/move a file.
 * @return true if successful
 */
inline bool rename_file(const std::string& from, const std::string& to) {
    if (!std::filesystem::exists(from)) {
        return false;  // Source doesn't exist, not an error - file may be optional
    }
    std::error_code ec;
    std::filesystem::rename(from, to, ec);
    if (ec) {
        fmt::print(stderr, "Error renaming '{}' to '{}': {}\n", from, to, ec.message());
        return false;
    }
    fmt::print("Renamed: {} -> {}\n", from, to);
    return true;
}

inline bool rename_file(const std::filesystem::path& from, const std::filesystem::path& to) {
    return rename_file(from.string(), to.string());
}

/**
 * Move a file from source to destination, with optional renaming.
 * Creates destination directory if needed.
 * @param src Source file path
 * @param dest Destination file path
 * @return true if file was moved successfully, false if source doesn't exist or error occurred
 */
inline bool move_file(const std::filesystem::path& src, const std::filesystem::path& dest) {
    if (!std::filesystem::exists(src)) {
        return false;  // Source doesn't exist
    }
    
    // Ensure destination directory exists
    ensure_directory(dest.parent_path());
    
    std::error_code ec;
    std::filesystem::rename(src, dest, ec);
    if (ec) {
        fmt::print(stderr, "Error moving '{}' to '{}': {}\n", src.string(), dest.string(), ec.message());
        return false;
    }
    fmt::print("Moved: {} -> {}\n", src.string(), dest.string());
    return true;
}

/**
 * Remove a directory and all its contents.
 * @return true if directory was removed or didn't exist
 */
inline bool remove_directory(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) return true;
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    if (ec) {
        fmt::print(stderr, "Error removing directory '{}': {}\n", path.string(), ec.message());
        return false;
    }
    return true;
}

// ============================================================================
// Download and Extract Utilities
// ============================================================================

/**
 * Download a file from a URL.
 * Uses curl on Windows, wget on other platforms.
 * @param url The URL to download from
 * @param dest_path The destination file path
 * @return true if download was successful
 */
inline bool download_file(const std::string& url, const std::filesystem::path& dest_path) {
    // Ensure destination directory exists
    ensure_directory(dest_path.parent_path());
    
    fmt::print("Downloading: {} -> {}\n", url, dest_path.string());
    
    #ifdef _WIN32
    std::string cmd = fmt::format("curl -L -o \"{}\" \"{}\"", dest_path.string(), url);
    #else
    std::string cmd = fmt::format("wget -O \"{}\" \"{}\"", dest_path.string(), url);
    #endif
    
    int result = std::system(cmd.c_str());
    if (result != 0) {
        fmt::print(stderr, "Download failed with code {}\n", result);
        return false;
    }
    
    fmt::print("Download complete: {}\n", dest_path.string());
    return true;
}

/**
 * Extract a tar.gz archive to a directory.
 * @param archive_path Path to the .tar.gz file
 * @param dest_dir Directory to extract to
 * @return true if extraction was successful
 */
inline bool extract_tar_gz(const std::filesystem::path& archive_path, const std::filesystem::path& dest_dir) {
    if (!std::filesystem::exists(archive_path)) {
        fmt::print(stderr, "Archive not found: {}\n", archive_path.string());
        return false;
    }
    
    // Ensure destination directory exists
    ensure_directory(dest_dir);
    
    fmt::print("Extracting: {} -> {}\n", archive_path.string(), dest_dir.string());
    
    std::string cmd = fmt::format("tar -xzf \"{}\" -C \"{}\"", archive_path.string(), dest_dir.string());
    
    int result = std::system(cmd.c_str());
    if (result != 0) {
        fmt::print(stderr, "Extraction failed with code {}\n", result);
        return false;
    }
    
    fmt::print("Extraction complete\n");
    return true;
}

/**
 * Find a subdirectory containing a specific file.
 * Searches one level deep in the given directory.
 * @param search_dir Directory to search in
 * @param filename File to look for
 * @return Path to the directory containing the file, or empty path if not found
 */
inline std::filesystem::path find_directory_with_file(
    const std::filesystem::path& search_dir,
    const std::string& filename)
{
    // First check if file is directly in search_dir
    if (std::filesystem::exists(search_dir / filename)) {
        return search_dir;
    }
    
    // Search subdirectories
    for (const auto& entry : std::filesystem::directory_iterator(search_dir)) {
        if (entry.is_directory()) {
            if (std::filesystem::exists(entry.path() / filename)) {
                return entry.path();
            }
        }
    }
    
    return {};  // Not found
}

}  // namespace deglib::benchmark
