#pragma once

/**
 * @file logging.h
 * @brief Dual-output logging utilities for benchmark tools.
 * 
 * Provides logging that outputs to both console and file simultaneously.
 */

#include <filesystem>
#include <fstream>
#include <string>

#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/ostream.h>

namespace deglib::benchmark
{

// ============================================================================
// Logging utilities - log to file AND console simultaneously
// ============================================================================

inline std::ostream* log_file_ptr = nullptr;
inline std::ofstream log_file_stream;
inline bool log_to_console = true;

/**
 * Set the log file path. All subsequent log() calls will write to this file.
 * @param path Path to the log file
 * @param append If true, append to existing file; otherwise overwrite
 */
inline void set_log_file(const std::string& path, bool append = false) {
    if (log_file_stream.is_open()) {
        log_file_stream.close();
    }
    auto mode = std::ios::out;
    if (append) mode |= std::ios::app;
    log_file_stream.open(path, mode);
    if (log_file_stream.is_open()) {
        log_file_ptr = &log_file_stream;
    } else {
        log_file_ptr = nullptr;
        fmt::print(stderr, "Warning: Could not open log file '{}'\n", path);
    }
}

/**
 * Close the log file and reset to console-only logging.
 */
inline void reset_log_to_console() {
    if (log_file_stream.is_open()) {
        log_file_stream.close();
    }
    log_file_ptr = nullptr;
}

/**
 * Enable or disable console output.
 * @param enabled If false, log() will only write to file (if set)
 */
inline void set_console_logging(bool enabled) {
    log_to_console = enabled;
}

/**
 * Log a formatted message to both console (if enabled) and file (if set).
 * Uses fmt library formatting syntax.
 */
template<typename... Args>
void log(fmt::format_string<Args...> fmt_str, Args&&... args) {
    std::string msg = fmt::format(fmt_str, std::forward<Args>(args)...);
    if (log_to_console) {
        fmt::print("{}", msg);
    }
    if (log_file_ptr) {
        fmt::print(*log_file_ptr, "{}", msg);
        log_file_ptr->flush();
    }
}

// ============================================================================
// Directory utilities
// ============================================================================

/**
 * Ensure a directory exists, creating it if necessary.
 * @param path Path to the directory
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

}  // namespace deglib::benchmark
