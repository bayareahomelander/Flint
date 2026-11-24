#include "io/file.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>

#include "io/json.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#endif

static_assert(sizeof(size_t) == 8, "Flint is 64-bit only: shapes and offsets are uint64");

namespace flint::io {

using json::clip;
using json::printable;

namespace {

// One read or write call moves at most this much, as Windows' calls take a 32-bit count.
constexpr size_t kChunk = size_t{1} << 30;

// A path as a message quotes it.
constexpr size_t kPathInMessage = 150;

// A path as a message shows it: cut, and made printable, so the message is valid UTF-8.
std::string shown(const std::string& path) {
    const std::string cut = clip(path, kPathInMessage);
    std::string text;
    printable(text, cut.data(), cut.size());
    return text;
}

IoStatus status(IoStatus::Kind kind, std::string message) {
    IoStatus result;
    result.kind = kind;
    result.message = std::move(message);
    return result;
}

IoStatus end_of_file(const std::string& path, uint64_t offset) {
    return status(IoStatus::kEndOfFile,
                  "unexpected end of `" + shown(path) + "` at byte " +
                      std::to_string(offset) + " (did the file shrink after it opened?)");
}

#if defined(_WIN32)

HANDLE as_handle(intptr_t handle) { return reinterpret_cast<HANDLE>(handle); }

IoStatus bad_path(const std::string& path) {
    return status(IoStatus::kBadPath, "the path `" + shown(path) +
                                          "` is not valid UTF-8");
}

// The path as UTF-16, or false if it isn't valid UTF-8.
bool to_wide(const std::string& utf8, std::wstring& wide) {
    wide.clear();
    if (utf8.empty()) {
        return true;
    }
    if (utf8.size() > static_cast<size_t>(INT32_MAX)) {
        return false;
    }
    const int length = static_cast<int>(utf8.size());
    const int count =
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), length, nullptr, 0);
    if (count <= 0) {
        return false;
    }
    wide.resize(static_cast<size_t>(count));
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), length, wide.data(),
                               count) == count;
}

// "call `path`: the OS's message (error code)".
IoStatus os_error(const char* call, const std::string& path, DWORD code) {
    wchar_t wide[512];
    const DWORD length = FormatMessageW(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_MAX_WIDTH_MASK,
        nullptr, code, 0, wide, static_cast<DWORD>(std::size(wide)), nullptr);
    std::string text;
    if (length > 0) {
        char utf8[4 * std::size(wide)];
        const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide, static_cast<int>(length), utf8,
                                              static_cast<int>(sizeof utf8), nullptr, nullptr);
        if (bytes > 0) {
            text.assign(utf8, static_cast<size_t>(bytes));
        }
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\r' || text.back() == '\n')) {
        text.pop_back();
    }
    if (text.empty()) {
        text = "unknown error";
    }
    return status(IoStatus::kOsError, std::string(call) + " `" + shown(path) +
                                          "`: " + text + " (error " + std::to_string(code) +
                                          ")");
}

// Best effort: it runs from destructors, so nothing escapes it, not even an exception.
void remove_path(const std::string& path) noexcept {
    try {
        std::wstring wide;
        if (to_wide(path, wide)) {
            DeleteFileW(wide.c_str());
        }
    } catch (...) {
    }
}

#else

IoStatus os_error(const char* call, const std::string& path, int code) {
    // The text comes in the host's locale, which needn't be UTF-8.
    const char* text = std::strerror(code);
    std::string shown_text;
    printable(shown_text, text, std::strlen(text));
    return status(IoStatus::kOsError, std::string(call) + " `" + shown(path) + "`: " +
                                          shown_text + " (errno " + std::to_string(code) + ")");
}

// Best effort: it runs from destructors, so nothing escapes it.
void remove_path(const std::string& path) noexcept { ::unlink(path.c_str()); }

// The folder that holds the file, for fsync after a rename.
std::string folder_of(const std::string& path) {
    const size_t slash = path.rfind('/');
    if (slash == std::string::npos) {
        return ".";
    }
    return slash == 0 ? "/" : path.substr(0, slash);
}

#endif

}  // namespace

#if defined(_WIN32)

IoStatus File::open_read(const std::string& path) {
    close();
    path_ = path;
    std::wstring wide;
    if (!to_wide(path, wide)) {
        return bad_path(path);
    }
    const HANDLE handle = CreateFileW(wide.c_str(), GENERIC_READ,
                                      FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                      FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return os_error("CreateFileW", path, GetLastError());
    }
    LARGE_INTEGER size;
    if (!GetFileSizeEx(handle, &size)) {
        const DWORD code = GetLastError();
        CloseHandle(handle);
        return os_error("GetFileSizeEx", path, code);
    }
    handle_ = reinterpret_cast<intptr_t>(handle);
    size_ = static_cast<uint64_t>(size.QuadPart);
    return {};
}

IoStatus File::read_at(uint64_t offset, void* dst, size_t bytes) {
    char* out = static_cast<char*>(dst);
    uint64_t position = offset;
    while (bytes > 0) {
        const DWORD want = static_cast<DWORD>(std::min(bytes, kChunk));
        OVERLAPPED where = {};
        where.Offset = static_cast<DWORD>(position & 0xFFFFFFFFu);
        where.OffsetHigh = static_cast<DWORD>(position >> 32);
        DWORD got = 0;
        if (!ReadFile(as_handle(handle_), out, want, &got, &where)) {
            const DWORD code = GetLastError();
            if (code == ERROR_HANDLE_EOF) {
                return end_of_file(path_, position);
            }
            return os_error("ReadFile", path_, code);
        }
        if (got == 0) {
            return end_of_file(path_, position);
        }
        out += got;
        position += got;
        bytes -= got;
    }
    return {};
}

IoStatus File::create(const std::string& path) {
    close();
    path_ = path;
    std::wstring wide;
    if (!to_wide(path, wide)) {
        return bad_path(path);
    }
    const HANDLE handle = CreateFileW(wide.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                      FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return os_error("CreateFileW", path, GetLastError());
    }
    handle_ = reinterpret_cast<intptr_t>(handle);
    size_ = 0;
    return {};
}

IoStatus File::write(const void* src, size_t bytes) {
    const char* in = static_cast<const char*>(src);
    while (bytes > 0) {
        const DWORD want = static_cast<DWORD>(std::min(bytes, kChunk));
        DWORD put = 0;
        if (!WriteFile(as_handle(handle_), in, want, &put, nullptr)) {
            return os_error("WriteFile", path_, GetLastError());
        }
        if (put == 0) {
            return status(IoStatus::kOsError,
                          "WriteFile `" + shown(path_) + "`: wrote no bytes");
        }
        in += put;
        bytes -= put;
    }
    return {};
}

IoStatus File::commit(const std::string& final_path) {
    // Convert both paths first: if that throws, the file is still open, and the writer's
    // destructor deletes it.
    std::wstring from;
    std::wstring to;
    if (!to_wide(path_, from) || !to_wide(final_path, to)) {
        discard();
        return bad_path(final_path);
    }
    if (!FlushFileBuffers(as_handle(handle_))) {
        const DWORD code = GetLastError();
        discard();
        return os_error("FlushFileBuffers", path_, code);
    }
    const BOOL closed = CloseHandle(as_handle(handle_));
    const DWORD close_code = closed ? 0 : GetLastError();
    handle_ = kClosed;
    if (!closed) {
        remove_path(path_);
        return os_error("CloseHandle", path_, close_code);
    }
    if (!MoveFileExW(from.c_str(), to.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD code = GetLastError();
        remove_path(path_);
        return os_error("MoveFileExW", final_path, code);
    }
    return {};
}

void File::close() {
    if (handle_ != kClosed) {
        CloseHandle(as_handle(handle_));
        handle_ = kClosed;
    }
}

#else

IoStatus File::open_read(const std::string& path) {
    close();
    path_ = path;
    int fd;
    do {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        return os_error("open", path, errno);
    }
    struct stat info;
    if (::fstat(fd, &info) != 0) {
        const int code = errno;
        ::close(fd);
        return os_error("fstat", path, code);
    }
    if (S_ISDIR(info.st_mode)) {
        // Linux opens a folder for reading; it would only fail at the first read.
        ::close(fd);
        return os_error("open", path, EISDIR);
    }
    handle_ = fd;
    size_ = info.st_size < 0 ? uint64_t{0} : static_cast<uint64_t>(info.st_size);
    return {};
}

IoStatus File::read_at(uint64_t offset, void* dst, size_t bytes) {
    char* out = static_cast<char*>(dst);
    uint64_t position = offset;
    while (bytes > 0) {
        if (position > static_cast<uint64_t>(INT64_MAX)) {
            return end_of_file(path_, position);
        }
        const size_t want = std::min(bytes, kChunk);
        const ssize_t got =
            ::pread(static_cast<int>(handle_), out, want, static_cast<off_t>(position));
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            return os_error("pread", path_, errno);
        }
        if (got == 0) {
            return end_of_file(path_, position);
        }
        out += got;
        position += static_cast<uint64_t>(got);
        bytes -= static_cast<size_t>(got);
    }
    return {};
}

IoStatus File::create(const std::string& path) {
    close();
    path_ = path;
    int fd;
    do {
        fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        return os_error("open", path, errno);
    }
    handle_ = fd;
    size_ = 0;
    return {};
}

IoStatus File::write(const void* src, size_t bytes) {
    const char* in = static_cast<const char*>(src);
    while (bytes > 0) {
        const size_t want = std::min(bytes, kChunk);
        const ssize_t put = ::write(static_cast<int>(handle_), in, want);
        if (put < 0) {
            if (errno == EINTR) {
                continue;
            }
            return os_error("write", path_, errno);
        }
        if (put == 0) {
            return status(IoStatus::kOsError,
                          "write `" + shown(path_) + "`: wrote no bytes");
        }
        in += put;
        bytes -= static_cast<size_t>(put);
    }
    return {};
}

IoStatus File::commit(const std::string& final_path) {
    const int fd = static_cast<int>(handle_);
    int code = 0;
    while (::fsync(fd) != 0) {
        if (errno != EINTR) {
            code = errno;
            break;
        }
    }
    if (code != 0) {
        discard();
        return os_error("fsync", path_, code);
    }
    // The descriptor is gone after close, even when close reports an error, so it isn't retried.
    const int closed = ::close(fd);
    code = closed == 0 ? 0 : errno;
    handle_ = kClosed;
    if (code != 0) {
        remove_path(path_);
        return os_error("close", path_, code);
    }
    if (::rename(path_.c_str(), final_path.c_str()) != 0) {
        code = errno;
        remove_path(path_);
        return os_error("rename", final_path, code);
    }
    // The rename itself is only durable once its folder is. From here on the file is in place,
    // and a failure says so.
    const std::string folder = folder_of(final_path);
    const auto after_rename = [&](IoStatus failure) {
        failure.message = "`" + shown(final_path) +
                          "` is in place, but syncing its folder failed: " + failure.message;
        return failure;
    };
    int folder_fd;
    do {
        folder_fd = ::open(folder.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    } while (folder_fd < 0 && errno == EINTR);
    if (folder_fd < 0) {
        code = errno;
        return after_rename(os_error("open", folder, code));
    }
    int synced;
    do {
        synced = ::fsync(folder_fd);
    } while (synced != 0 && errno == EINTR);
    code = synced == 0 ? 0 : errno;
    ::close(folder_fd);
    // Some file systems can't sync a folder; the file is in place and synced.
    if (code != 0 && code != EINVAL && code != ENOTSUP) {
        return after_rename(os_error("fsync", folder, code));
    }
    return {};
}

void File::close() {
    if (handle_ != kClosed) {
        ::close(static_cast<int>(handle_));
        handle_ = kClosed;
    }
}

#endif

void File::discard() noexcept {
    close();
    remove_path(path_);
}

}  // namespace flint::io
