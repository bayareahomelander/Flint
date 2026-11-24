#include "common/error.h"

#include <cstdarg>
#include <cstdio>

namespace {

// Long enough for any message Flint writes; longer ones are truncated.
thread_local char last_error[512] = "";

}  // namespace

namespace flint {

flint_status fail(flint_status status, const char* format, ...) {
    va_list args;
    va_start(args, format);
    const int needed = std::vsnprintf(last_error, sizeof last_error, format, args);
    va_end(args);
    if (needed >= static_cast<int>(sizeof last_error)) {
        // The cut may have split a UTF-8 character, and callers decode the message strictly.
        size_t length = sizeof last_error - 1;
        size_t lead = length;
        while (lead > 0 && (static_cast<unsigned char>(last_error[lead - 1]) & 0xC0) == 0x80) {
            --lead;
        }
        if (lead > 0) {
            const unsigned char first = static_cast<unsigned char>(last_error[lead - 1]);
            const size_t sequence = first >= 0xF0 ? 4 : first >= 0xE0 ? 3 : first >= 0xC0 ? 2 : 1;
            if (first >= 0xC0 && (lead - 1) + sequence > length) {
                length = lead - 1;
            }
        }
        last_error[length] = '\0';
    }
    return status;
}

}  // namespace flint

const char* flint_last_error(void) { return last_error; }
