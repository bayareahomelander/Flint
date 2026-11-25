#include "data/loader.h"

#include <cstring>
#include <stdexcept>
#include <utility>

#include "data/order.h"
#include "io/json.h"

static_assert(sizeof(size_t) == 8, "Flint is 64-bit only: counts and offsets are uint64");

// The files hold their tokens little-endian, and read() copies them as they are. The library
// builds only for little-endian x86-64: gcc says so, and MSVC's targets are all little-endian.
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "the loader copies little-endian tokens as they are");
#elif !defined(_MSC_VER)
#error "the loader needs a little-endian host, and this compiler doesn't say which this is"
#endif

namespace flint::data {

namespace {

constexpr size_t kPathInMessage = 150;
constexpr uint64_t kHeaderBytes = 64;
constexpr uint64_t kVersion = 1;
constexpr uint64_t kTokenizerId = 1;
constexpr uint64_t kSequenceLimit = uint64_t{1} << 32;  // a file's sequences must stay below it
// A shuffled file may hold this many sequences at most: its permutation takes 4 bytes each, so
// 256 MiB, which keeps the trainer's host memory under its 1 GiB.
constexpr uint64_t kShuffledSequenceLimit = uint64_t{1} << 26;

Status failure(Status::Kind kind, std::string message) {
    Status status;
    status.kind = kind;
    status.message = std::move(message);
    return status;
}

// A path as a message shows it: cut, and made printable, so the message is valid UTF-8.
std::string shown(const std::string& path) {
    const std::string cut = json::clip(path, kPathInMessage);
    std::string text;
    json::printable(text, cut.data(), cut.size());
    return text;
}

// "`path`: message", for a file that breaks a rule.
Status format_failure(const std::string& path, const std::string& message) {
    return failure(Status::kFormat, "`" + shown(path) + "`: " + message);
}

// An I/O layer failure as a status: a path that can't be opened is the caller's mistake, the rest
// is the system's. The I/O layer's message already names the path.
Status io_failure(const io::IoStatus& status) {
    return failure(status.kind == io::IoStatus::kBadPath ? Status::kInvalidArgument : Status::kIo,
                   status.message);
}

// An unsigned integer from its first bytes little-endian, whatever the host's order.
uint64_t little_endian(const unsigned char* bytes, int count) {
    uint64_t value = 0;
    for (int i = count - 1; i >= 0; --i) {
        value = (value << 8) | bytes[i];
    }
    return value;
}

}  // namespace

Loader::~Loader() {
    if (last_file_ != kNone) {
        entries_[last_file_].file->advise(io::File::Advice::kDropCache);
    }
}

Status Loader::open(const std::vector<std::string>& paths, const Config& config) {
    config_ = config;
    entries_.clear();
    entries_.resize(paths.size());
    scratch_.assign(static_cast<size_t>(config.length) + 1, 0);
    epoch_sequences_ = 0;

    for (size_t i = 0; i < paths.size(); ++i) {
        Entry& entry = entries_[i];
        entry.path = paths[i];
        entry.file = std::make_unique<io::File>();
        io::IoStatus io = entry.file->open_read(entry.path);
        if (!io.ok()) {
            return io_failure(io);
        }
        // Advise before the header read: a read at offset 0 would otherwise set off the kernel's
        // initial readahead (Linux cached 4 pages of every file that way), so with the advice
        // first each file leaves only its header's page cached.
        entry.file->advise(io::File::Advice::kRandom);
        const uint64_t size = entry.file->size();
        if (size < kHeaderBytes) {
            return format_failure(entry.path, "the file is " + std::to_string(size) +
                                                  " bytes, shorter than the 64-byte header");
        }
        unsigned char header[kHeaderBytes];
        io = entry.file->read_at(0, header, sizeof header);
        if (!io.ok()) {
            return io_failure(io);
        }
        if (std::memcmp(header, "FLINTTOK", 8) != 0) {
            return format_failure(entry.path, "the magic isn't FLINTTOK");
        }
        const uint64_t version = little_endian(header + 8, 4);
        if (version != kVersion) {
            return format_failure(entry.path,
                                  "the version is " + std::to_string(version) + ", not 1");
        }
        const uint64_t tokenizer = little_endian(header + 12, 4);
        if (tokenizer != kTokenizerId) {
            return format_failure(entry.path,
                                  "the tokenizer id is " + std::to_string(tokenizer) + ", not 1");
        }
        const uint64_t count = little_endian(header + 16, 8);
        for (uint64_t b = 24; b < kHeaderBytes; ++b) {
            if (header[b] != 0) {
                return format_failure(entry.path, "reserved byte " + std::to_string(b) +
                                                      " is " + std::to_string(header[b]) +
                                                      ", not 0");
            }
        }
        // The file's size is 64 + 2 x count, and a count past what 64 bits hold fails this too.
        if (count > (UINT64_MAX - kHeaderBytes) / 2) {
            return format_failure(entry.path,
                                  "the file is " + std::to_string(size) +
                                      " bytes, but the header's count of " +
                                      std::to_string(count) +
                                      " tokens needs more bytes than 64 bits hold");
        }
        if (size != kHeaderBytes + 2 * count) {
            return format_failure(entry.path,
                                  "the file is " + std::to_string(size) +
                                      " bytes, but the header's count of " +
                                      std::to_string(count) + " tokens needs " +
                                      std::to_string(kHeaderBytes + 2 * count) + " bytes");
        }
        entry.usable = config.max_tokens != 0 && config.max_tokens < count ? config.max_tokens
                                                                           : count;
        entry.sequences = entry.usable == 0 ? 0 : (entry.usable - 1) / config.length;
        if (entry.sequences >= kSequenceLimit) {
            return format_failure(entry.path, "the file holds " + std::to_string(entry.sequences) +
                                                  " sequences, 2^32 or more");
        }
        if (config.shuffle && entry.sequences > kShuffledSequenceLimit) {
            return failure(Status::kInvalidArgument,
                           "`" + shown(entry.path) + "`: the file holds " +
                               std::to_string(entry.sequences) +
                               " sequences of length + 1 tokens, more than the 67,108,864 a "
                               "shuffled file may hold (a 256 MiB permutation)");
        }
        epoch_sequences_ += entry.sequences;
    }
    if (epoch_sequences_ == 0) {
        return failure(Status::kInvalidArgument,
                       "no file holds a sequence of length + 1 tokens");
    }
    return {};
}

void Loader::use_files_order(uint64_t epoch) {
    if (files_cached_ && files_epoch_ == epoch) {
        return;
    }
    files_cached_ = false;  // stays false if the rebuild throws
    permutation(files_order_, entries_.size(), stream_start(config_.seed, epoch, kFilesKey));
    files_epoch_ = epoch;
    files_cached_ = true;
}

void Loader::use_sequences_order(uint64_t epoch, uint64_t file) {
    if (sequences_cached_ && sequences_epoch_ == epoch && sequences_file_ == file) {
        return;
    }
    sequences_cached_ = false;
    permutation(sequences_order_, entries_[static_cast<size_t>(file)].sequences,
                stream_start(config_.seed, epoch, file));
    sequences_epoch_ = epoch;
    sequences_file_ = file;
    sequences_cached_ = true;
}

Location Loader::locate(uint64_t index) {
    Location where;
    where.epoch = index / epoch_sequences_;
    uint64_t r = index % epoch_sequences_;
    if (config_.shuffle) {
        use_files_order(where.epoch);
    }
    // Walk the slots of the files' order: a file with no sequences takes a slot and gives nothing.
    for (size_t slot = 0; slot < entries_.size(); ++slot) {
        const uint64_t file = config_.shuffle ? files_order_[slot] : slot;
        const uint64_t sequences = entries_[static_cast<size_t>(file)].sequences;
        if (r < sequences) {
            where.file = file;
            if (config_.shuffle) {
                use_sequences_order(where.epoch, file);
                where.sequence = sequences_order_[static_cast<size_t>(r)];
            } else {
                where.sequence = r;
            }
            return where;
        }
        r -= sequences;
    }
    // r < E, the sum of every file's sequences, so some slot takes it.
    throw std::logic_error("the sequences of the files' slots don't add up to the epoch");
}

Status Loader::read(uint64_t index, uint64_t count, uint16_t* tokens, uint16_t* targets) {
    const size_t length = static_cast<size_t>(config_.length);
    for (uint64_t i = 0; i < count; ++i) {
        const Location where = locate(index + i);
        Entry& entry = entries_[static_cast<size_t>(where.file)];
        // Once the order moves on to another file, its predecessor's pages can go.
        if (last_file_ != kNone && last_file_ != where.file) {
            entries_[last_file_].file->advise(io::File::Advice::kDropCache);
        }
        last_file_ = static_cast<size_t>(where.file);

        const uint64_t first_token = where.sequence * config_.length;
        const io::IoStatus io = entry.file->read_at(kHeaderBytes + 2 * first_token, scratch_.data(),
                                                    2 * scratch_.size());
        if (!io.ok()) {
            return io_failure(io);
        }
        for (size_t t = 0; t <= length; ++t) {
            if (scratch_[t] >= config_.vocab) {
                return format_failure(entry.path, "token " + std::to_string(first_token + t) +
                                                      " is " + std::to_string(scratch_[t]) +
                                                      ", not below vocab " +
                                                      std::to_string(config_.vocab));
            }
        }
        // The inputs are the first length tokens and the targets the last length.
        std::memcpy(tokens + i * length, scratch_.data(), 2 * length);
        std::memcpy(targets + i * length, scratch_.data() + 1, 2 * length);
    }
    return {};
}

}  // namespace flint::data
