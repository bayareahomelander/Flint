// Files for the safetensors reader and writer, on Windows and on POSIX systems. This and file.cpp
// are the only code that touches the operating system's file calls, so the rest stays portable.
// Paths are UTF-8, and every offset is 64 bits: `long` is 32 bits on Windows.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace flint::io {

// How a file operation went.
struct IoStatus {
    enum Kind {
        kOk,
        kBadPath,   // the path isn't valid UTF-8 (Windows converts it to UTF-16)
        kOsError,   // the operating system refused
        kEndOfFile  // a read reached the end of the file first
    };
    Kind kind = kOk;
    // For a failure: what was being done, to which path, and the OS's own words and code.
    std::string message;

    bool ok() const { return kind == kOk; }
};

// One open file, closed by its destructor. A file is used by one thread at a time.
class File {
public:
    File() = default;
    ~File() { close(); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    bool is_open() const { return handle_ != kClosed; }
    // The size of a file opened for reading, taken when it opened.
    uint64_t size() const { return size_; }

    // Opens an existing file for reading, which others may read, rename and delete meanwhile.
    IoStatus open_read(const std::string& path);

    // Reads exactly bytes bytes from the file's offset, with a positional read, so no file
    // pointer is shared. The file ending first is kEndOfFile.
    IoStatus read_at(uint64_t offset, void* dst, size_t bytes);

    // Creates a file for writing, or empties an existing one, which nobody else may open.
    IoStatus create(const std::string& path);

    // Appends bytes bytes to a created file, straight to the OS, with no buffer of our own.
    IoStatus write(const void* src, size_t bytes);

    // Makes a created file durable and moves it over final_path, replacing what is there, and
    // makes the move durable too. The file is closed whatever the outcome. On a failure the
    // file's own path is deleted again, best effort.
    IoStatus commit(const std::string& final_path);

    // A hint about the whole file for the operating system's page cache, best effort: kRandom says
    // reads will be random (no readahead), kDropCache asks it to drop the file's cached pages.
    // Errors are ignored. On Windows it does nothing.
    enum class Advice { kRandom, kDropCache };
    void advise(Advice advice) const noexcept;

    // Closes the file, if it is open.
    void close();

    // Closes a created file and deletes it, best effort, ignoring every error.
    void discard() noexcept;

private:
    static constexpr std::intptr_t kClosed = -1;

    std::intptr_t handle_ = kClosed;  // a HANDLE on Windows, a file descriptor elsewhere
    uint64_t size_ = 0;
    std::string path_;
};

}  // namespace flint::io
