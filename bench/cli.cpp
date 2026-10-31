#include "cli.h"

#include <charconv>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace bench {

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

long long parse_integer(const std::string& option, const std::string& text) {
    long long value = 0;
    const char* end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (text.empty() || result.ec != std::errc() || result.ptr != end) {
        fail(option + ": not an integer: '" + text + "'");
    }
    return value;
}

double parse_number(const std::string& option, const std::string& text) {
    double value = 0;
    const char* end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (text.empty() || result.ec != std::errc() || result.ptr != end || !std::isfinite(value)) {
        fail(option + ": not a number: '" + text + "'");
    }
    return value;
}

// A positive integer with an optional binary suffix: 4097, 1M, 16m, 2G.
unsigned long long parse_size(const std::string& item) {
    std::string digits = item;
    unsigned long long scale = 1;
    if (!digits.empty()) {
        switch (digits.back()) {
            case 'k':
            case 'K':
                scale = 1ull << 10;
                break;
            case 'm':
            case 'M':
                scale = 1ull << 20;
                break;
            case 'g':
            case 'G':
                scale = 1ull << 30;
                break;
            default:
                break;
        }
        if (scale != 1) {
            digits.pop_back();
        }
    }
    unsigned long long value = 0;
    const char* end = digits.data() + digits.size();
    const auto result = std::from_chars(digits.data(), end, value);
    if (digits.empty() || result.ec != std::errc() || result.ptr != end) {
        fail("--sizes: not a size: '" + item + "'");
    }
    if (value == 0) {
        fail("--sizes: sizes must be positive: '" + item + "'");
    }
    if (value > (1ull << 62) / scale) {
        fail("--sizes: size too large: '" + item + "'");
    }
    return value * scale;
}

std::vector<unsigned long long> parse_sizes(const std::string& list) {
    std::vector<unsigned long long> sizes;
    std::size_t begin = 0;
    while (true) {
        const std::size_t comma = list.find(',', begin);
        const std::size_t length = comma == std::string::npos ? comma : comma - begin;
        sizes.push_back(parse_size(list.substr(begin, length)));
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return sizes;
}

std::vector<int> parse_chains(const std::string& list) {
    std::vector<int> chains;
    std::size_t begin = 0;
    while (true) {
        const std::size_t comma = list.find(',', begin);
        const std::size_t length = comma == std::string::npos ? comma : comma - begin;
        const std::string item = list.substr(begin, length);
        const long long value = parse_integer("--chains", item);
        if (value != 4 && value != 8 && value != 16) {
            fail("--chains: must be 4, 8 or 16: '" + item + "'");
        }
        chains.push_back(static_cast<int>(value));
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return chains;
}

// The limits flint_gemm_nt, flint_gemm_nn and flint_gemm_tn accept (include/flint/flint.h).
constexpr unsigned long long kMaxGemmM = 8388480;
constexpr unsigned long long kMaxGemmN = 8388480;
constexpr unsigned long long kMaxGemmK = 65536;

// One dimension of a shape: digits only, as a number.
unsigned long long parse_dimension(const std::string& item, const std::string& text) {
    unsigned long long value = 0;
    const char* end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (text.empty() || result.ec != std::errc() || result.ptr != end) {
        fail("--shapes: not a shape MxNxK of positive integers: '" + item + "'");
    }
    return value;
}

GemmShape parse_shape(const std::string& item, GemmLayout layout, GemmEpilogue epilogue) {
    const std::size_t first = item.find('x');
    const std::size_t second = first == std::string::npos ? first : item.find('x', first + 1);
    if (second == std::string::npos || item.find('x', second + 1) != std::string::npos) {
        fail("--shapes: not a shape MxNxK of positive integers: '" + item + "'");
    }
    GemmShape shape;
    shape.m = parse_dimension(item, item.substr(0, first));
    shape.n = parse_dimension(item, item.substr(first + 1, second - first - 1));
    shape.k = parse_dimension(item, item.substr(second + 1));
    if (shape.m == 0 || shape.n == 0 || shape.k == 0) {
        fail("--shapes: dimensions must be positive: '" + item + "'");
    }
    // tn takes any K; nt and nn need K a multiple of 8.
    if (layout != GemmLayout::TN && shape.k % 8 != 0) {
        fail("--shapes: K must be a multiple of 8: '" + item + "'");
    }
    if (layout == GemmLayout::NN && shape.n % 8 != 0) {
        fail("--shapes: N must be a multiple of 8 with --layout nn: '" + item + "'");
    }
    if (layout == GemmLayout::TN && shape.m % 8 != 0) {
        fail("--shapes: M must be a multiple of 8 with --layout tn: '" + item + "'");
    }
    if (layout == GemmLayout::TN && shape.n % 8 != 0) {
        fail("--shapes: N must be a multiple of 8 with --layout tn: '" + item + "'");
    }
    // cuBLASLt's GELU_AUX_BIAS needs its aux output's leading dimension, N, to
    // be a multiple of 8 (Flint's kernel doesn't).
    if (epilogue == GemmEpilogue::BiasGelu && shape.n % 8 != 0) {
        fail("--shapes: N must be a multiple of 8 with --epilogue bias-gelu: '" + item + "'");
    }
    // flint_gemm_nt_bias_residual reads and writes FP32 pairs, so N must be even.
    if (epilogue == GemmEpilogue::BiasResidual && shape.n % 2 != 0) {
        fail("--shapes: N must be even with --epilogue bias-residual: '" + item + "'");
    }
    if (shape.m > kMaxGemmM || shape.n > kMaxGemmN || shape.k > kMaxGemmK) {
        fail("--shapes: past the limits (M up to " + std::to_string(kMaxGemmM) + ", N up to " +
             std::to_string(kMaxGemmN) + ", K up to " + std::to_string(kMaxGemmK) + "): '" + item +
             "'");
    }
    return shape;
}

std::vector<GemmShape> parse_shapes(const std::string& list, GemmLayout layout,
                                    GemmEpilogue epilogue) {
    std::vector<GemmShape> shapes;
    std::size_t begin = 0;
    while (true) {
        const std::size_t comma = list.find(',', begin);
        const std::size_t length = comma == std::string::npos ? comma : comma - begin;
        shapes.push_back(parse_shape(list.substr(begin, length), layout, epilogue));
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return shapes;
}

int parse_count(const std::string& option, const std::string& text) {
    const long long value = parse_integer(option, text);
    if (value < 1 || value > 1000000) {
        fail(option + ": must be between 1 and 1000000: '" + text + "'");
    }
    return static_cast<int>(value);
}

bool starts_with_dashes(const std::string& text) { return text.rfind("--", 0) == 0; }

}  // namespace

const char* layout_name(GemmLayout layout) {
    switch (layout) {
        case GemmLayout::NN:
            return "nn";
        case GemmLayout::TN:
            return "tn";
        default:
            return "nt";
    }
}

const char* epilogue_name(GemmEpilogue epilogue) {
    switch (epilogue) {
        case GemmEpilogue::Bias:
            return "bias";
        case GemmEpilogue::BiasGelu:
            return "bias-gelu";
        case GemmEpilogue::DGelu:
            return "dgelu";
        case GemmEpilogue::BiasResidual:
            return "bias-residual";
        default:
            return "none";
    }
}

CommandLine parse_command_line(int argc, char** argv) {
    CommandLine command;
    std::optional<std::string> sizes;
    std::optional<std::string> chains;
    std::optional<std::string> shapes;
    std::optional<GemmLayout> layout;
    std::optional<GemmEpilogue> epilogue;
    std::optional<LtAlgo> lt_algo;
    std::optional<int> rounds;
    std::optional<int> max_rounds;
    std::optional<double> warmup;
    std::optional<double> sample_ms;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (!starts_with_dashes(arg)) {
            if (!command.benchmark.empty()) {
                fail("unexpected argument: " + arg);
            }
            command.args.push_back(arg);
            command.benchmark = arg;
            continue;
        }
        const std::size_t equals = arg.find('=');
        const std::string name = arg.substr(0, equals);
        std::optional<std::string> inline_value;
        if (equals != std::string::npos) {
            inline_value = arg.substr(equals + 1);
        }

        const bool is_flag = name == "--allow-busy" || name == "--quick" || name == "--help";
        const bool takes_value = name == "--sizes" || name == "--chains" ||
                                 name == "--shapes" || name == "--layout" ||
                                 name == "--epilogue" || name == "--lt-algo" ||
                                 name == "--rounds" || name == "--max-rounds" || name == "--warmup" ||
                                 name == "--sample-ms" || name == "--out";
        if (!is_flag && !takes_value) {
            fail("unknown option: " + name);
        }
        if (is_flag && inline_value) {
            fail(name + " takes no value");
        }

        std::string value;
        bool consumed_next = false;
        if (takes_value) {
            if (inline_value) {
                value = *inline_value;
            } else if (i + 1 < argc && !starts_with_dashes(argv[i + 1])) {
                value = argv[i + 1];
                consumed_next = true;
            } else {
                fail(name + " needs a value");
            }
        }

        if (name != "--out") {
            command.args.push_back(arg);
            if (consumed_next) {
                command.args.push_back(value);
            }
        }
        if (consumed_next) {
            ++i;
        }

        if (name == "--allow-busy") {
            command.options.allow_busy = true;
        } else if (name == "--quick") {
            command.options.quick = true;
        } else if (name == "--help") {
            command.help = true;
        } else if (name == "--sizes") {
            sizes = value;
        } else if (name == "--chains") {
            chains = value;
        } else if (name == "--shapes") {
            shapes = value;
        } else if (name == "--layout") {
            if (value == "nt") {
                layout = GemmLayout::NT;
            } else if (value == "nn") {
                layout = GemmLayout::NN;
            } else if (value == "tn") {
                layout = GemmLayout::TN;
            } else {
                fail("--layout: must be nt, nn or tn: '" + value + "'");
            }
        } else if (name == "--epilogue") {
            if (value == "none") {
                epilogue = GemmEpilogue::None;
            } else if (value == "bias") {
                epilogue = GemmEpilogue::Bias;
            } else if (value == "bias-gelu") {
                epilogue = GemmEpilogue::BiasGelu;
            } else if (value == "dgelu") {
                epilogue = GemmEpilogue::DGelu;
            } else if (value == "bias-residual") {
                epilogue = GemmEpilogue::BiasResidual;
            } else {
                fail("--epilogue: must be none, bias, bias-gelu, bias-residual or dgelu: '" +
                     value + "'");
            }
        } else if (name == "--lt-algo") {
            if (value == "fastest") {
                lt_algo = LtAlgo::Fastest;
            } else if (value == "first") {
                lt_algo = LtAlgo::First;
            } else {
                fail("--lt-algo: must be fastest or first: '" + value + "'");
            }
        } else if (name == "--rounds") {
            rounds = parse_count(name, value);
        } else if (name == "--max-rounds") {
            max_rounds = parse_count(name, value);
        } else if (name == "--warmup") {
            const double seconds = parse_number(name, value);
            if (seconds < 0) {
                fail("--warmup: must not be negative: '" + value + "'");
            }
            warmup = seconds;
        } else if (name == "--sample-ms") {
            const double ms = parse_number(name, value);
            if (ms <= 0) {
                fail("--sample-ms: must be positive: '" + value + "'");
            }
            sample_ms = ms;
        } else {  // --out
            if (value.empty()) {
                fail("--out: empty path");
            }
            command.out = value;
        }
    }

    if (command.benchmark.empty()) {
        if (!command.help) {
            fail("no benchmark given");
        }
    } else if (command.benchmark != "copy" && command.benchmark != "mma" &&
               command.benchmark != "gemm") {
        fail("unknown benchmark: " + command.benchmark);
    }
    if (!command.benchmark.empty()) {
        if (sizes && command.benchmark != "copy") {
            fail("--sizes applies to copy only");
        }
        if (chains && command.benchmark != "mma") {
            fail("--chains applies to mma only");
        }
        if (shapes && command.benchmark != "gemm") {
            fail("--shapes applies to gemm only");
        }
        if (layout && command.benchmark != "gemm") {
            fail("--layout applies to gemm only");
        }
        if (epilogue && command.benchmark != "gemm") {
            fail("--epilogue applies to gemm only");
        }
        if (lt_algo && command.benchmark != "gemm") {
            fail("--lt-algo applies to gemm only");
        }
    }

    // Explicit options win over --quick whatever their order.
    Options& options = command.options;
    const bool quick = options.quick;
    options.warmup_seconds = warmup.value_or(quick ? 0.2 : 1.5);
    options.sample_ms = sample_ms.value_or(quick ? 10 : 100);
    options.rounds = rounds.value_or(quick ? 3 : 10);
    options.lt_algo = lt_algo.value_or(LtAlgo::Fastest);
    if (max_rounds) {
        if (*max_rounds < options.rounds) {
            fail("--max-rounds: " + std::to_string(*max_rounds) + " is below --rounds " +
                 std::to_string(options.rounds));
        }
        options.max_rounds = *max_rounds;
    } else {
        const int base = quick ? 6 : 30;
        options.max_rounds = options.rounds > base ? 3 * options.rounds : base;
    }

    if (command.benchmark == "mma") {
        command.chains = parse_chains(chains.value_or("4,8,16"));
    } else if (command.benchmark == "gemm") {
        command.layout = layout.value_or(GemmLayout::NT);
        command.epilogue = epilogue.value_or(GemmEpilogue::None);
        // The bias epilogues (the residual one too) are the forward pass's, and
        // GELU's backward pass goes with the input gradient.
        if (command.epilogue == GemmEpilogue::DGelu) {
            if (command.layout != GemmLayout::NN) {
                fail("--epilogue dgelu needs --layout nn");
            }
        } else if (command.epilogue != GemmEpilogue::None && command.layout != GemmLayout::NT) {
            fail(std::string("--epilogue ") + epilogue_name(command.epilogue) +
                 " needs --layout nt");
        }
        // The GPT-2 shapes at 8192 tokens: the forward pass for nt (without the
        // logits layer, which has no bias, when nt adds one, and with only the
        // MLP's up projection, the one layer with a GELU, when it adds that, and
        // with only the attention and MLP down projections, the two layers that
        // feed the residual stream, when it adds that), the input gradients for
        // nn (with only the MLP's up projection's, the one layer whose input
        // gradient goes through a GELU's backward pass, when nn adds that), the
        // weight gradients for tn.
        const char* defaults =
            command.epilogue == GemmEpilogue::BiasGelu || command.epilogue == GemmEpilogue::DGelu
                ? "8192x3072x768"
            : command.epilogue == GemmEpilogue::Bias
                ? "8192x2304x768,8192x768x768,8192x3072x768,8192x768x3072"
            : command.epilogue == GemmEpilogue::BiasResidual
                ? "8192x768x768,8192x768x3072"
            : command.layout == GemmLayout::NN
                ? "8192x768x2304,8192x768x768,8192x768x3072,8192x3072x768,8192x768x50304"
            : command.layout == GemmLayout::TN
                ? "2304x768x8192,768x768x8192,3072x768x8192,768x3072x8192,50304x768x8192"
                : "8192x2304x768,8192x768x768,8192x3072x768,8192x768x3072,8192x50304x768";
        command.shapes = parse_shapes(shapes.value_or(defaults), command.layout, command.epilogue);
    } else {
        command.sizes = parse_sizes(sizes.value_or("1M,16M,256M"));
    }
    return command;
}

std::string usage() {
    return "usage: flint_bench <benchmark> [options]\n"
           "\n"
           "benchmarks:\n"
           "  copy                  flint_copy against cudaMemcpyAsync\n"
           "  mma                   tensor-core m16n8k8 FP16 throughput with FP32\n"
           "                        accumulation, timed against the GPU's peak\n"
           "  gemm                  flint_gemm_nt (y = x w^T), flint_gemm_nn (y = x w) or\n"
           "                        flint_gemm_tn (y = x^T w), FP16 with FP32 accumulation,\n"
           "                        against cuBLAS (cuBLASLt with a bias, GELU or its\n"
           "                        derivative)\n"
           "\n"
           "options (--flag value or --flag=value):\n"
           "  --sizes LIST          copy: comma-separated sizes in bytes, with an optional\n"
           "                        K, M or G suffix (KiB, MiB, GiB); default 1M,16M,256M\n"
           "  --chains LIST         mma: independent accumulator chains per warp, from\n"
           "                        4, 8 and 16; default 4,8,16\n"
           "  --layout nt|nn|tn     gemm: nt is y (M x N) = x (M x K) times w (N x K)\n"
           "                        transposed, nn is y = x times w (K x N), tn is y (FP32) =\n"
           "                        x (K x M) transposed times w (K x N), timed accumulating\n"
           "                        into y; default nt\n"
           "  --epilogue none|bias|bias-gelu|bias-residual|dgelu\n"
           "                        gemm: bias adds n FP32 values, one per column of y, to the\n"
           "                        FP32 sums before they round (flint_gemm_nt_bias, against\n"
           "                        cuBLASLt's bias epilogue); bias-gelu then applies GELU,\n"
           "                        also storing the sums before it (flint_gemm_nt_bias_gelu,\n"
           "                        against cuBLASLt's GELU_AUX_BIAS epilogue); bias-residual\n"
           "                        then adds an m x n FP32 residual stream into an FP32 y\n"
           "                        (flint_gemm_nt_bias_residual, against cuBLASLt's bias\n"
           "                        epilogue with the residual as C and beta 1); all three\n"
           "                        need --layout nt. dgelu multiplies the sums by GELU's\n"
           "                        derivative at a stored m x n FP16 pre, the backward pass\n"
           "                        (flint_gemm_nn_dgelu, against cuBLASLt's DGELU\n"
           "                        epilogue); it needs --layout nn. Default none\n"
           "  --lt-algo fastest|first\n"
           "                        gemm: which of cuBLASLt's candidate algorithms the\n"
           "                        epilogue cases (not none) use as the baseline. fastest,\n"
           "                        the default, times the heuristic's first 8 on each\n"
           "                        case's buffers and uses the quickest; first uses the\n"
           "                        heuristic's first choice, as PyTorch does. The JSON\n"
           "                        lists the candidates and their times either way\n"
           "  --shapes LIST         gemm: comma-separated MxNxK; M and N up to 8388480, and K\n"
           "                        up to 65536. nt and nn need K a multiple of 8 (and nn N\n"
           "                        too, as does bias-gelu, for cuBLASLt; bias-residual needs\n"
           "                        N even); tn needs M and N multiples of 8 and takes any K.\n"
           "                        Default the GPT-2 shapes at 8192 tokens. For nt, the\n"
           "                        forward pass: 8192x2304x768,8192x768x768,8192x3072x768,\n"
           "                        8192x768x3072,8192x50304x768, without the last with\n"
           "                        --epilogue bias, only the third with --epilogue bias-gelu,\n"
           "                        and only the second and fourth, the layers that feed the\n"
           "                        residual stream, with --epilogue bias-residual. For nn,\n"
           "                        the input gradients: 8192x768x2304,8192x768x768,\n"
           "                        8192x768x3072,8192x3072x768,8192x768x50304, and only the\n"
           "                        fourth with --epilogue dgelu. For tn, the\n"
           "                        weight gradients: 2304x768x8192,768x768x8192,\n"
           "                        3072x768x8192,768x3072x8192,50304x768x8192\n"
           "  --rounds N            valid rounds required per case (default 10)\n"
           "  --max-rounds N        rounds tried before giving up (default 30, or 3 x rounds\n"
           "                        if --rounds is above 30)\n"
           "  --warmup SECONDS      warm-up time per case (default 1.5)\n"
           "  --sample-ms MS        the shortest timed sample (default 100)\n"
           "  --allow-busy          run even if other processes are on the GPU\n"
           "  --quick               shortened protocol for tests, never a real result:\n"
           "                        warmup 0.2, sample-ms 10, rounds 3, max-rounds 6;\n"
           "                        explicit options override these\n"
           "  --out PATH            JSON output path (default: bench-results/ in the build\n"
           "                        directory, named by benchmark, GPU and time)\n"
           "  --help                show this text\n";
}

}  // namespace bench
