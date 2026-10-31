// flint_bench's command line.
#pragma once

#include <string>
#include <vector>

#include "harness.h"

namespace bench {

// The GEMM layouts: NT is y (m x n) = x (m x k) times w (n x k) transposed, a
// linear layer's forward pass; NN is y = x times w (k x n), its input gradient;
// TN is y (FP32) = x (k x m) transposed times w (k x n), its weight gradient.
enum class GemmLayout { NT, NN, TN };

// "nt", "nn" or "tn".
const char* layout_name(GemmLayout layout);

// What the GEMM does to its outputs before they round to FP16: nothing, add a
// bias to each column, add it and then apply GELU, also storing the sums
// before GELU (these two NT only), multiply by GELU's derivative at those
// stored sums, the backward pass (NN only), or add the bias and then an FP32
// residual stream into an FP32 output that never rounds to FP16 (NT only).
enum class GemmEpilogue { None, Bias, BiasGelu, DGelu, BiasResidual };

// "none", "bias", "bias-gelu", "dgelu" or "bias-residual".
const char* epilogue_name(GemmEpilogue epilogue);

// One GEMM problem of m x n x k; x's and w's layouts depend on the GemmLayout.
struct GemmShape {
    unsigned long long m = 0;
    unsigned long long n = 0;
    unsigned long long k = 0;
};

struct CommandLine {
    bool help = false;                      // --help: print usage() and exit
    std::string benchmark;                  // "copy", "mma" or "gemm"
    std::vector<unsigned long long> sizes;  // copy: bytes
    std::vector<int> chains;                // mma: accumulator chains per warp
    std::vector<GemmShape> shapes;          // gemm: MxNxK
    GemmLayout layout = GemmLayout::NT;     // gemm: how w is laid out
    GemmEpilogue epilogue = GemmEpilogue::None;  // gemm: what happens to the outputs
    Options options;
    std::string out;  // JSON path; empty means default_json_path()
    // argv[1..] for the JSON, minus --out and its value, which is often an
    // absolute path and says nothing about the result.
    std::vector<std::string> args;
};

// Parses argv. Throws std::invalid_argument with a one-line reason on bad input.
CommandLine parse_command_line(int argc, char** argv);

std::string usage();

}  // namespace bench
