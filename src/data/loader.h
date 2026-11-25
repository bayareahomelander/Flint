// The data loader: sequences from Flint's token files (see scripts/fineweb.py for the format) in a
// seeded order that is a pure function of the files, the config and the sequence's index. Host
// memory only. Failures come back as a status kind plus a message, as in io; the C API turns them
// into flint::fail calls.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "io/file.h"

namespace flint::data {

// How a loader call went. The message doesn't start with a function's name: the C API adds it.
struct Status {
    enum Kind {
        kOk,
        kInvalidArgument,  // the caller asked for something that can't work
        kFormat,           // a file isn't valid (a token file header, or a token out of range)
        kIo                // the operating system failed a file call
    };
    Kind kind = kOk;
    std::string message;

    bool ok() const { return kind == kOk; }
};

struct Config {
    uint64_t length = 0;      // tokens per sequence's inputs, and per its targets
    uint64_t max_tokens = 0;  // use only each file's first max_tokens tokens; 0 for all
    uint64_t vocab = 0;       // every token read must be below it
    uint64_t seed = 0;
    bool shuffle = false;
};

// Where an index falls.
struct Location {
    uint64_t epoch = 0;
    uint64_t file = 0;      // its place in the list open took
    uint64_t sequence = 0;  // within the file: its tokens start at token sequence * length
};

// A loader is used by one thread at a time.
class Loader {
public:
    Loader() = default;
    // Advises the page cache to drop the last file read, then closes the files.
    ~Loader();
    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;

    // Opens the files and checks every header and size (the caller has checked the config's ranges
    // and that paths isn't empty). Call it once.
    Status open(const std::vector<std::string>& paths, const Config& config);

    // The sequences in one epoch.
    uint64_t epoch_sequences() const { return epoch_sequences_; }
    uint64_t length() const { return config_.length; }

    // Where sequence number index (counting from the start of the first epoch) falls.
    Location locate(uint64_t index);

    // Reads count sequences from index on: tokens and targets each receive count x length uint16
    // values, which must not overlap and must hold that many.
    Status read(uint64_t index, uint64_t count, uint16_t* tokens, uint16_t* targets);

private:
    struct Entry {
        std::unique_ptr<io::File> file;  // File is neither copyable nor movable
        std::string path;
        uint64_t usable = 0;     // the tokens the loader may use: the count, cut to max_tokens
        uint64_t sequences = 0;  // c_f: (usable - 1) / length, or 0 for no tokens
    };

    // Make files_order_ hold epoch's order of the files, and sequences_order_ hold the order of
    // file's sequences in epoch, rebuilding either only when its key differs. Only for shuffle on:
    // otherwise both orders are the identity and nothing is stored.
    void use_files_order(uint64_t epoch);
    void use_sequences_order(uint64_t epoch, uint64_t file);

    Config config_;
    std::vector<Entry> entries_;
    uint64_t epoch_sequences_ = 0;  // E, the sum of every file's sequences

    // The files' order for the epoch it names, and one file's sequence order for the epoch and
    // file it names. Both are rebuilt when a lookup needs another key.
    bool files_cached_ = false;
    uint64_t files_epoch_ = 0;
    std::vector<uint32_t> files_order_;
    bool sequences_cached_ = false;
    uint64_t sequences_epoch_ = 0;
    uint64_t sequences_file_ = 0;
    std::vector<uint32_t> sequences_order_;

    // The last file read touched, so its pages can be dropped once the order moves on (kNone
    // before the first read).
    static constexpr size_t kNone = static_cast<size_t>(-1);
    size_t last_file_ = kNone;

    std::vector<uint16_t> scratch_;  // one sequence: length + 1 tokens
};

}  // namespace flint::data
