// The check that flint_adamw and flint_gpt2_step share.
#pragma once

#include "flint/flint.h"

namespace flint {

// Checks an optimizer step's hyperparameters, recording the first failure as function's error.
flint_status check_adamw_config(const char* function, const flint_adamw_config* config);

}  // namespace flint
