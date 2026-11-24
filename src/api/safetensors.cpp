// The safetensors C API: check the arguments, then hand the work to src/io. Every function
// catches every exception, so none crosses the C boundary: a failed allocation is
// FLINT_ERROR_IO ("out of host memory").
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <vector>

#include "common/error.h"
#include "io/file.h"
#include "io/json.h"
#include "io/safetensors.h"

static_assert(sizeof(size_t) == 8, "Flint is 64-bit only: shapes and offsets are uint64");

struct flint_safetensors_reader {
    flint::io::File file;
    std::string path;
    flint::safetensors::Header header;
    uint64_t data_start = 0;  // the file's offset of the data: 8 + the header's length
};

struct flint_safetensors_writer {
    enum class State { kOpen, kCommitted, kFailed };

    flint_safetensors_writer() = default;
    flint_safetensors_writer(const flint_safetensors_writer&) = delete;
    flint_safetensors_writer& operator=(const flint_safetensors_writer&) = delete;
    // A writer that never committed takes its temporary file with it.
    ~flint_safetensors_writer() {
        if (state != State::kCommitted && file.is_open()) {
            file.discard();
        }
    }

    flint::io::File file;  // the temporary file, while it is open
    std::string path;      // where commit moves it
    uint64_t total = 0;    // the data's declared bytes
    uint64_t written = 0;
    State state = State::kOpen;
};

namespace {

constexpr size_t kPathInMessage = 150;

// Runs body, turning an exception into a status.
template <typename Body>
flint_status guarded(const char* function, Body&& body) {
    try {
        return body();
    } catch (const std::bad_alloc&) {
        return flint::fail(FLINT_ERROR_IO, "%s: out of host memory", function);
    } catch (const std::exception& error) {
        return flint::fail(FLINT_ERROR_IO, "%s: unexpected error: %s", function, error.what());
    } catch (...) {
        return flint::fail(FLINT_ERROR_IO, "%s: unexpected error", function);
    }
}

flint_status invalid(const char* function, const char* message) {
    return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: %s", function, message);
}

flint_status null_argument(const char* function, const char* argument) {
    return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: %s is null", function, argument);
}

// An I/O layer failure as a status: a path that can't be opened is the caller's mistake, the rest
// is the system's.
flint_status io_failure(const char* function, const flint::io::IoStatus& status) {
    const flint_status code = status.kind == flint::io::IoStatus::kBadPath
                                  ? FLINT_ERROR_INVALID_ARGUMENT
                                  : FLINT_ERROR_IO;
    return flint::fail(code, "%s: %s", function, status.message.c_str());
}

flint_status format_failure(const char* function, const std::string& path,
                            const std::string& message) {
    const std::string cut = flint::json::clip(path, kPathInMessage);
    std::string shown;
    flint::json::printable(shown, cut.data(), cut.size());
    return flint::fail(FLINT_ERROR_FORMAT, "%s: `%s`: %s", function, shown.c_str(),
                       message.c_str());
}

// A uint64 from 8 little-endian bytes, whatever the host's order.
uint64_t little_endian_u64(const unsigned char* bytes) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | bytes[i];
    }
    return value;
}

}  // namespace

flint_status flint_safetensors_open(const char* path, flint_safetensors_reader** reader) {
    constexpr const char* name = "flint_safetensors_open";
    if (reader == nullptr) {
        return null_argument(name, "reader");
    }
    *reader = nullptr;
    if (path == nullptr) {
        return null_argument(name, "path");
    }
    if (path[0] == '\0') {
        return invalid(name, "path is empty");
    }
    return guarded(name, [&]() -> flint_status {
        auto opened = std::make_unique<flint_safetensors_reader>();
        opened->path = path;
        flint::io::IoStatus io = opened->file.open_read(opened->path);
        if (!io.ok()) {
            return io_failure(name, io);
        }
        const uint64_t size = opened->file.size();
        if (size < 8) {
            return format_failure(name, opened->path,
                                  "the file is " + std::to_string(size) +
                                      " bytes, too short to hold the header's 8-byte length");
        }
        unsigned char prefix[8];
        io = opened->file.read_at(0, prefix, sizeof prefix);
        if (!io.ok()) {
            return io_failure(name, io);
        }
        const uint64_t length = little_endian_u64(prefix);
        if (length > flint::safetensors::kMaxHeaderBytes) {
            return format_failure(name, opened->path,
                                  "the header's length, " + std::to_string(length) +
                                      " bytes, is over 100,000,000");
        }
        if (length > size - 8) {
            return format_failure(name, opened->path,
                                  "the header's length, " + std::to_string(length) +
                                      " bytes, runs past the end of the " + std::to_string(size) +
                                      "-byte file");
        }
        std::vector<char> text(static_cast<size_t>(length));
        io = opened->file.read_at(8, text.data(), text.size());
        if (!io.ok()) {
            return io_failure(name, io);
        }
        try {
            flint::safetensors::parse_header(text.data(), text.size(), size - 8 - length,
                                             opened->header);
        } catch (const flint::json::FormatError& error) {
            return format_failure(name, opened->path, error.what());
        }
        opened->data_start = 8 + length;
        *reader = opened.release();
        return FLINT_OK;
    });
}

void flint_safetensors_close(flint_safetensors_reader* reader) { delete reader; }

flint_status flint_safetensors_tensor_count(const flint_safetensors_reader* reader,
                                            size_t* count) {
    constexpr const char* name = "flint_safetensors_tensor_count";
    if (reader == nullptr) {
        return null_argument(name, "reader");
    }
    if (count == nullptr) {
        return null_argument(name, "count");
    }
    *count = reader->header.tensors.size();
    return FLINT_OK;
}

flint_status flint_safetensors_tensor_info(const flint_safetensors_reader* reader, size_t index,
                                           flint_safetensors_tensor* info) {
    constexpr const char* name = "flint_safetensors_tensor_info";
    if (reader == nullptr) {
        return null_argument(name, "reader");
    }
    if (info == nullptr) {
        return null_argument(name, "info");
    }
    const flint::safetensors::Header& header = reader->header;
    if (index >= header.tensors.size()) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: index (%zu) must be below the tensor count (%zu)", name, index,
                           header.tensors.size());
    }
    const flint::safetensors::TensorEntry& entry = header.tensors[index];
    info->name = header.name(index);
    info->dtype = entry.dtype->name;
    info->rank = entry.rank;
    info->shape = header.shape(index);
    info->bits = entry.dtype->bits;
    info->offset = static_cast<size_t>(entry.begin);
    info->bytes = static_cast<size_t>(entry.end - entry.begin);
    return FLINT_OK;
}

flint_status flint_safetensors_find(const flint_safetensors_reader* reader, const char* name,
                                    size_t* index) {
    constexpr const char* function = "flint_safetensors_find";
    if (reader == nullptr) {
        return null_argument(function, "reader");
    }
    if (name == nullptr) {
        return null_argument(function, "name");
    }
    if (index == nullptr) {
        return null_argument(function, "index");
    }
    return guarded(function, [&]() -> flint_status {
        const auto& table = reader->header.tensor_by_name;
        const auto found = table.find(std::string_view(name));
        if (found == table.end()) {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: no tensor named %s", function,
                               flint::safetensors::quote(name).c_str());
        }
        *index = found->second;
        return FLINT_OK;
    });
}

flint_status flint_safetensors_metadata_count(const flint_safetensors_reader* reader,
                                              size_t* count) {
    constexpr const char* name = "flint_safetensors_metadata_count";
    if (reader == nullptr) {
        return null_argument(name, "reader");
    }
    if (count == nullptr) {
        return null_argument(name, "count");
    }
    *count = reader->header.metadata.size();
    return FLINT_OK;
}

flint_status flint_safetensors_metadata(const flint_safetensors_reader* reader, size_t index,
                                        const char** key, const char** value) {
    constexpr const char* name = "flint_safetensors_metadata";
    if (reader == nullptr) {
        return null_argument(name, "reader");
    }
    if (key == nullptr) {
        return null_argument(name, "key");
    }
    if (value == nullptr) {
        return null_argument(name, "value");
    }
    const flint::safetensors::Header& header = reader->header;
    if (index >= header.metadata.size()) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: index (%zu) must be below the metadata count (%zu)", name, index,
                           header.metadata.size());
    }
    *key = header.key(index);
    *value = header.value(index);
    return FLINT_OK;
}

flint_status flint_safetensors_metadata_find(const flint_safetensors_reader* reader,
                                             const char* key, const char** value) {
    constexpr const char* function = "flint_safetensors_metadata_find";
    if (reader == nullptr) {
        return null_argument(function, "reader");
    }
    if (key == nullptr) {
        return null_argument(function, "key");
    }
    if (value == nullptr) {
        return null_argument(function, "value");
    }
    return guarded(function, [&]() -> flint_status {
        const auto& table = reader->header.metadata_by_key;
        const auto found = table.find(std::string_view(key));
        if (found == table.end()) {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: no metadata key %s", function,
                               flint::safetensors::quote(key).c_str());
        }
        *value = reader->header.value(found->second);
        return FLINT_OK;
    });
}

flint_status flint_safetensors_read(flint_safetensors_reader* reader, size_t index, size_t offset,
                                    size_t bytes, void* dst) {
    constexpr const char* name = "flint_safetensors_read";
    if (reader == nullptr) {
        return null_argument(name, "reader");
    }
    if (dst == nullptr && bytes > 0) {
        return null_argument(name, "dst");
    }
    return guarded(name, [&]() -> flint_status {
        const flint::safetensors::Header& header = reader->header;
        if (index >= header.tensors.size()) {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                               "%s: index (%zu) must be below the tensor count (%zu)", name, index,
                               header.tensors.size());
        }
        const flint::safetensors::TensorEntry& entry = header.tensors[index];
        const uint64_t tensor_bytes = entry.end - entry.begin;
        if (offset > tensor_bytes || bytes > tensor_bytes - offset) {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                               "%s: %zu bytes from offset %zu pass the end of tensor %s, which "
                               "has %llu bytes",
                               name, bytes, offset,
                               flint::safetensors::quote(header.name(index)).c_str(),
                               static_cast<unsigned long long>(tensor_bytes));
        }
        if (bytes == 0) {
            return FLINT_OK;
        }
        const flint::io::IoStatus io =
            reader->file.read_at(reader->data_start + entry.begin + offset, dst, bytes);
        return io.ok() ? FLINT_OK : io_failure(name, io);
    });
}

flint_status flint_safetensors_create(const char* path, flint_safetensors_tensor* tensors,
                                      size_t tensor_count, const char* const* keys,
                                      const char* const* values, size_t metadata_count,
                                      flint_safetensors_writer** writer) {
    constexpr const char* name = "flint_safetensors_create";
    if (writer == nullptr) {
        return null_argument(name, "writer");
    }
    *writer = nullptr;
    if (path == nullptr) {
        return null_argument(name, "path");
    }
    if (tensors == nullptr && tensor_count > 0) {
        return null_argument(name, "tensors");
    }
    if (keys == nullptr && metadata_count > 0) {
        return null_argument(name, "keys");
    }
    if (values == nullptr && metadata_count > 0) {
        return null_argument(name, "values");
    }
    if (path[0] == '\0') {
        return invalid(name, "path is empty");
    }
    return guarded(name, [&]() -> flint_status {
        std::vector<flint::safetensors::Declaration> declarations;
        declarations.reserve(tensor_count);
        for (size_t i = 0; i < tensor_count; ++i) {
            declarations.push_back({tensors[i].name, tensors[i].dtype, tensors[i].rank,
                                    tensors[i].shape});
        }
        flint::safetensors::Layout layout;
        try {
            flint::safetensors::build_header(declarations.data(), declarations.size(), keys,
                                             values, metadata_count, layout);
        } catch (const flint::safetensors::ArgumentError& error) {
            return invalid(name, error.what());
        }

        auto created = std::make_unique<flint_safetensors_writer>();
        created->path = path;
        flint::io::IoStatus io = created->file.create(created->path + ".tmp");
        if (!io.ok()) {
            return io_failure(name, io);
        }
        // The 8-byte length, then the header, in one write. A failure from here on deletes the
        // temporary file when `created` goes out of scope.
        std::string head(8, '\0');
        for (size_t i = 0; i < 8; ++i) {
            head[i] = static_cast<char>((static_cast<uint64_t>(layout.header.size()) >> (8 * i)) &
                                        0xFF);
        }
        head += layout.header;
        io = created->file.write(head.data(), head.size());
        if (!io.ok()) {
            return io_failure(name, io);
        }
        for (size_t i = 0; i < tensor_count; ++i) {
            tensors[i].bits = layout.placements[i].bits;
            tensors[i].offset = static_cast<size_t>(layout.placements[i].offset);
            tensors[i].bytes = static_cast<size_t>(layout.placements[i].bytes);
        }
        created->total = layout.data_length;
        *writer = created.release();
        return FLINT_OK;
    });
}

flint_status flint_safetensors_write(flint_safetensors_writer* writer, const void* src,
                                     size_t bytes) {
    constexpr const char* name = "flint_safetensors_write";
    if (writer == nullptr) {
        return null_argument(name, "writer");
    }
    if (writer->state == flint_safetensors_writer::State::kCommitted) {
        return invalid(name, "already committed");
    }
    if (writer->state == flint_safetensors_writer::State::kFailed) {
        return flint::fail(FLINT_ERROR_IO, "%s: an earlier write failed", name);
    }
    if (src == nullptr && bytes > 0) {
        return null_argument(name, "src");
    }
    if (bytes > writer->total - writer->written) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: %zu bytes would pass the declared %llu, of which %llu are written",
                           name, bytes, static_cast<unsigned long long>(writer->total),
                           static_cast<unsigned long long>(writer->written));
    }
    if (bytes == 0) {
        return FLINT_OK;
    }
    return guarded(name, [&]() -> flint_status {
        // Failed until the write is known to have worked: an exception inside it, after some
        // bytes reached the disk, must leave the writer failed.
        writer->state = flint_safetensors_writer::State::kFailed;
        const flint::io::IoStatus io = writer->file.write(src, bytes);
        if (!io.ok()) {
            return io_failure(name, io);
        }
        writer->written += bytes;
        writer->state = flint_safetensors_writer::State::kOpen;
        return FLINT_OK;
    });
}

flint_status flint_safetensors_commit(flint_safetensors_writer* writer) {
    constexpr const char* name = "flint_safetensors_commit";
    if (writer == nullptr) {
        return null_argument(name, "writer");
    }
    if (writer->state == flint_safetensors_writer::State::kCommitted) {
        return invalid(name, "already committed");
    }
    if (writer->state == flint_safetensors_writer::State::kFailed) {
        return flint::fail(FLINT_ERROR_IO, "%s: an earlier write failed", name);
    }
    if (writer->written != writer->total) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: %llu of the declared %llu bytes of data are written", name,
                           static_cast<unsigned long long>(writer->written),
                           static_cast<unsigned long long>(writer->total));
    }
    return guarded(name, [&]() -> flint_status {
        writer->state = flint_safetensors_writer::State::kFailed;  // until it is known to work
        const flint::io::IoStatus io = writer->file.commit(writer->path);
        if (!io.ok()) {
            return io_failure(name, io);
        }
        writer->state = flint_safetensors_writer::State::kCommitted;
        return FLINT_OK;
    });
}

void flint_safetensors_destroy(flint_safetensors_writer* writer) { delete writer; }
