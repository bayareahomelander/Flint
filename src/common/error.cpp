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
    std::vsnprintf(last_error, sizeof last_error, format, args);
    va_end(args);
    return status;
}

}  // namespace flint

const char* flint_last_error(void) { return last_error; }
