// The data loader's C API: check the arguments, then hand the work to src/data. Every function
// catches every exception, so none crosses the C boundary: a failed allocation is
// FLINT_ERROR_IO ("out of host memory").
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "common/checks.h"
#include "common/error.h"
#include "data/loader.h"

static_assert(sizeof(size_t) == 8, "Flint is 64-bit only: counts and offsets are uint64");

struct flint_loader {
    flint::data::Loader loader;
};

namespace {

constexpr size_t kMaxFiles = 65536;
constexpr size_t kMaxLength = 65536;
constexpr size_t kMaxVocab = 65536;

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

// A loader failure as a status, with the function's name in front of its message.
flint_status loader_failure(const char* function, const flint::data::Status& status) {
    flint_status code = FLINT_ERROR_IO;
    switch (status.kind) {
        case flint::data::Status::kInvalidArgument:
            code = FLINT_ERROR_INVALID_ARGUMENT;
            break;
        case flint::data::Status::kFormat:
            code = FLINT_ERROR_FORMAT;
            break;
        case flint::data::Status::kOk:
        case flint::data::Status::kIo:
            break;
    }
    return flint::fail(code, "%s: %s", function, status.message.c_str());
}

}  // namespace

flint_status flint_loader_open(const char* const* paths, size_t files,
                               const flint_loader_config* config, flint_loader** loader) {
    constexpr const char* name = "flint_loader_open";
    if (loader == nullptr) {
        return null_argument(name, "loader");
    }
    *loader = nullptr;
    if (paths == nullptr) {
        return null_argument(name, "paths");
    }
    if (config == nullptr) {
        return null_argument(name, "config");
    }
    if (files < 1 || files > kMaxFiles) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: files (%zu) must be from 1 to %zu", name, files, kMaxFiles);
    }
    for (size_t i = 0; i < files; ++i) {
        if (paths[i] == nullptr) {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: paths[%zu] is null", name, i);
        }
        if (paths[i][0] == '\0') {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: paths[%zu] is empty", name, i);
        }
    }
    if (config->length < 1 || config->length > kMaxLength) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: length (%zu) must be from 1 to %zu", name, config->length,
                           kMaxLength);
    }
    if (config->vocab < 1 || config->vocab > kMaxVocab) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: vocab (%zu) must be from 1 to %zu", name, config->vocab,
                           kMaxVocab);
    }
    return guarded(name, [&]() -> flint_status {
        std::vector<std::string> list;
        list.reserve(files);
        for (size_t i = 0; i < files; ++i) {
            list.emplace_back(paths[i]);
        }
        flint::data::Config settings;
        settings.length = config->length;
        settings.max_tokens = config->max_tokens;
        settings.vocab = config->vocab;
        settings.seed = config->seed;
        settings.shuffle = config->shuffle != 0;

        auto opened = std::make_unique<flint_loader>();
        const flint::data::Status status = opened->loader.open(list, settings);
        if (!status.ok()) {
            return loader_failure(name, status);
        }
        *loader = opened.release();
        return FLINT_OK;
    });
}

void flint_loader_close(flint_loader* loader) { delete loader; }

flint_status flint_loader_epoch_sequences(const flint_loader* loader, size_t* sequences) {
    constexpr const char* name = "flint_loader_epoch_sequences";
    if (loader == nullptr) {
        return null_argument(name, "loader");
    }
    if (sequences == nullptr) {
        return null_argument(name, "sequences");
    }
    *sequences = static_cast<size_t>(loader->loader.epoch_sequences());
    return FLINT_OK;
}

flint_status flint_loader_locate(flint_loader* loader, size_t index, size_t* epoch, size_t* file,
                                 size_t* sequence) {
    constexpr const char* name = "flint_loader_locate";
    if (loader == nullptr) {
        return null_argument(name, "loader");
    }
    if (epoch == nullptr) {
        return null_argument(name, "epoch");
    }
    if (file == nullptr) {
        return null_argument(name, "file");
    }
    if (sequence == nullptr) {
        return null_argument(name, "sequence");
    }
    return guarded(name, [&]() -> flint_status {
        const flint::data::Location where = loader->loader.locate(index);
        *epoch = static_cast<size_t>(where.epoch);
        *file = static_cast<size_t>(where.file);
        *sequence = static_cast<size_t>(where.sequence);
        return FLINT_OK;
    });
}

flint_status flint_loader_read(flint_loader* loader, size_t index, size_t count, void* tokens,
                               void* targets) {
    constexpr const char* name = "flint_loader_read";
    if (loader == nullptr) {
        return null_argument(name, "loader");
    }
    if (tokens == nullptr) {
        return null_argument(name, "tokens");
    }
    if (targets == nullptr) {
        return null_argument(name, "targets");
    }
    if (count > SIZE_MAX - index) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: index (%zu) + count (%zu) overflows", name, index, count);
    }
    const size_t length = static_cast<size_t>(loader->loader.length());
    // Each output holds count x length uint16 values; the product, in bytes, must fit.
    if (count > SIZE_MAX / 2 / length) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: count (%zu) x length (%zu) x 2 bytes overflows", name, count,
                           length);
    }
    const size_t bytes = count * length * 2;
    if (reinterpret_cast<uintptr_t>(tokens) % 2 != 0) {
        return invalid(name, "tokens is not 2-byte aligned");
    }
    if (!flint::fits(tokens, bytes)) {
        return invalid(name, "tokens runs past the end of the address space");
    }
    if (reinterpret_cast<uintptr_t>(targets) % 2 != 0) {
        return invalid(name, "targets is not 2-byte aligned");
    }
    if (!flint::fits(targets, bytes)) {
        return invalid(name, "targets runs past the end of the address space");
    }
    if (flint::overlap(tokens, bytes, targets, bytes)) {
        return invalid(name, "tokens and targets overlap");
    }
    if (count == 0) {
        return FLINT_OK;
    }
    return guarded(name, [&]() -> flint_status {
        const flint::data::Status status = loader->loader.read(
            index, count, static_cast<uint16_t*>(tokens), static_cast<uint16_t*>(targets));
        return status.ok() ? FLINT_OK : loader_failure(name, status);
    });
}
