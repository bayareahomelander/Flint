// GPT-2's forward and backward passes: the model's memory, its weights, and the kernels in
// order. This is host code. Every launch goes through the public API, whose checks then
// also guard the assembly: a wrong size here fails rather than running. The one exception is
// a cudaMemsetAsync in the backward pass, which zeroes wpe's gradient rows from length to
// block_size: no public function zeroes memory, and the embedding's kernel doesn't write those
// rows.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iterator>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.h"
#include "common/error.h"

namespace {

// Multiples of the alignment cudaMalloc gives, so one allocation's buffers sit
// as separate allocations' do.
constexpr size_t kAlignment = 256;
constexpr size_t kMaxLayers = 1024;
// A linear layer's k is at most 65,536, and the MLP's second has k = 4 * width.
constexpr size_t kMaxWidth = 16384;
// The attention kernels' limit on a matrix's rows.
constexpr size_t kMaxLength = 1024;
// The limit on the batched GEMMs' batches, which are sequences * heads.
constexpr size_t kMaxHeadBatches = 65535;
// A target is 16 bits, and the loss's kernel has this limit on its columns.
constexpr size_t kMaxPaddedVocab = 65536;
// The plain GEMMs' limit on m.
constexpr size_t kMaxRows = 65535 * 128;
// The backward pass's weight gradients are GEMMs with k = rows, which can be up to this.
constexpr size_t kMaxBackwardRows = 65536;
// The embedding's backward pass sums the position gradient's length * width values per
// sequence as columns, and the column sums have this limit.
constexpr size_t kMaxBackwardPositionValues = 16776960;
// GPT-2's LayerNorm eps, which PyTorch's model also uses.
constexpr float kLayerNormEps = 1e-5f;

// A master weight: FP32 in device memory, rows x cols, and with backward its gradient, the
// same size.
struct Parameter {
    std::string name;
    size_t rows;
    size_t cols;
    void* data;
    void* gradient;
};

struct Norm {
    Parameter* weight;
    Parameter* bias;
};

struct Linear {
    Parameter* weight;
    Parameter* bias;
    // The weight's FP16 compute copy, which the GEMMs read.
    void* compute;
};

// What a block's forward pass leaves for its backward pass: the residual stream (FP32)
// entering the block, after its attention and leaving it; each LayerNorm's FP16 output and
// FP32 statistics; Q, K, V and the probabilities in the heads layout; the attention's
// merged output; the MLP's GELU and the values entering it. A forward-only model has one
// set, which every block's pointers share.
struct Activations {
    void* x_in = nullptr;
    void* x_mid = nullptr;
    void* x_out = nullptr;
    void* h1 = nullptr;
    void* mean1 = nullptr;
    void* rstd1 = nullptr;
    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* p = nullptr;
    void* attn_output = nullptr;
    void* h2 = nullptr;
    void* mean2 = nullptr;
    void* rstd2 = nullptr;
    void* gelu = nullptr;
    void* pre = nullptr;
};

struct Block {
    Norm ln_1;
    Linear attn;
    Linear attn_proj;
    Norm ln_2;
    Linear mlp;
    Linear mlp_proj;
    Activations saved;
};

// A device buffer to allocate: where its pointer goes, and its size.
struct Slot {
    void** pointer;
    size_t bytes;
};

size_t rounded_up(size_t bytes) {
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

}  // namespace

struct flint_gpt2 {
    flint_gpt2_config config{};
    size_t rows = 0;
    size_t head_width = 0;
    size_t head_batches = 0;

    // A deque, so a Block's pointers into it stay valid as it grows.
    std::deque<Parameter> parameters;
    Parameter* wte = nullptr;
    Parameter* wpe = nullptr;
    void* wte_compute = nullptr;
    std::vector<Block> blocks;
    Norm ln_f{};

    // The residual stream leaving the last block, and ln_f's output and statistics, which
    // the output layer reads. Without backward they share the blocks' buffers.
    void* x_final = nullptr;
    void* h_f = nullptr;
    void* mean_f = nullptr;
    void* rstd_f = nullptr;
    // Scratch that every block reuses, in both passes: Q, K and V's rows (and their
    // gradient's), the scores (and the probabilities' gradient), and the heads' output (and
    // its gradient).
    void* qkv = nullptr;
    void* s = nullptr;
    void* heads_output = nullptr;
    void* logits = nullptr;
    void* dlogits = nullptr;
    void* row_loss = nullptr;

    // The backward pass's scratch, only with backward: the residual stream's gradient (FP32)
    // and its FP16 copy, a linear layer's input gradient, GELU's input gradient, and Q, K and
    // V's gradients in the heads layout.
    void* dres = nullptr;
    void* dres_half = nullptr;
    void* dh = nullptr;
    void* dpre = nullptr;
    void* dq = nullptr;
    void* dk = nullptr;
    void* dv = nullptr;
    // One workspace for the bias gradient and the LayerNorm and embedding backward passes,
    // and one for flint_gemm_tn, each the largest any of its calls needs.
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    void* tn_workspace = nullptr;
    size_t tn_workspace_bytes = 0;

    std::vector<Slot> slots;
    std::vector<void*> allocations;
    std::vector<float> host_row_loss;

    flint_gpt2() = default;
    flint_gpt2(const flint_gpt2&) = delete;
    flint_gpt2& operator=(const flint_gpt2&) = delete;
    ~flint_gpt2() {
        for (void* allocation : allocations) {
            cudaFree(allocation);
        }
        // Nothing can report a failed free, and the next launch mustn't take it for its own.
        cudaGetLastError();
    }
};

namespace {

// Returns a failed call's status from the calling function.
#define FLINT_RETURN_IF_FAILED(call)                \
    do {                                            \
        const flint_status flint_status_ = (call);  \
        if (flint_status_ != FLINT_OK) {            \
            return flint_status_;                   \
        }                                           \
    } while (0)

flint_status check_config(const char* name, const flint_gpt2_config* config) {
    if (config == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: config is null", name);
    }
    const flint_gpt2_config& c = *config;
    if (c.layers == 0 || c.layers > kMaxLayers) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: layers (%zu) must be from 1 to %zu",
                           name, c.layers, kMaxLayers);
    }
    if (c.width == 0 || c.width % 8 != 0 || c.width > kMaxWidth) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: width (%zu) must be a multiple of 8 from 8 to %zu", name, c.width,
                           kMaxWidth);
    }
    if (c.heads == 0 || c.width % c.heads != 0 || (c.width / c.heads) % 8 != 0) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: heads (%zu) must split width (%zu) into multiples of 8", name,
                           c.heads, c.width);
    }
    if (c.length == 0 || c.length % 8 != 0 || c.length > kMaxLength) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: length (%zu) must be a multiple of 8 from 8 to %zu", name, c.length,
                           kMaxLength);
    }
    if (c.block_size < c.length || c.block_size > kMaxLength) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: block_size (%zu) must be from length (%zu) to %zu", name,
                           c.block_size, c.length, kMaxLength);
    }
    if (c.padded_vocab == 0 || c.padded_vocab % 8 != 0 || c.padded_vocab > kMaxPaddedVocab) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: padded_vocab (%zu) must be a multiple of 8 from 8 to %zu", name,
                           c.padded_vocab, kMaxPaddedVocab);
    }
    if (c.vocab == 0 || c.vocab > c.padded_vocab) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: vocab (%zu) must be from 1 to padded_vocab (%zu)", name, c.vocab,
                           c.padded_vocab);
    }
    // Divides before it multiplies, so a product that wraps is still too large.
    if (c.sequences == 0 || c.sequences > kMaxHeadBatches / c.heads) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: sequences (%zu) x heads (%zu) must be from 1 to %zu", name,
                           c.sequences, c.heads, kMaxHeadBatches);
    }
    if (c.sequences > kMaxRows / c.length) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: sequences (%zu) x length (%zu) can be up to %zu rows", name,
                           c.sequences, c.length, kMaxRows);
    }
    if (c.backward != 0) {
        if (c.sequences > kMaxBackwardRows / c.length) {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                               "%s: sequences (%zu) x length (%zu) can be up to %zu rows with "
                               "backward",
                               name, c.sequences, c.length, kMaxBackwardRows);
        }
        if (c.length > kMaxBackwardPositionValues / c.width) {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                               "%s: length (%zu) x width (%zu) can be up to %zu with backward",
                               name, c.length, c.width, kMaxBackwardPositionValues);
        }
    }
    return FLINT_OK;
}

Parameter* add_parameter(flint_gpt2& m, std::string name, size_t rows, size_t cols) {
    m.parameters.push_back({std::move(name), rows, cols, nullptr, nullptr});
    Parameter* parameter = &m.parameters.back();
    m.slots.push_back({&parameter->data, rows * cols * 4});
    return parameter;
}

Norm add_norm(flint_gpt2& m, const std::string& name) {
    const size_t width = m.config.width;
    Parameter* weight = add_parameter(m, name + ".weight", 1, width);
    Parameter* bias = add_parameter(m, name + ".bias", 1, width);
    return {weight, bias};
}

Linear add_linear(flint_gpt2& m, const std::string& name, size_t out, size_t in) {
    Parameter* weight = add_parameter(m, name + ".weight", out, in);
    Parameter* bias = add_parameter(m, name + ".bias", 1, out);
    return {weight, bias, nullptr};
}

// The workspaces the backward pass's calls need: the largest of the bias gradient's (at each
// linear layer's width), LayerNorm's and the embedding's, and the largest of flint_gemm_tn's
// over the weight gradients' shapes. The last asks the current GPU for its SM count.
flint_status size_workspaces(flint_gpt2& m) {
    const flint_gpt2_config& c = m.config;
    const size_t rows = m.rows;
    const size_t width = c.width;
    size_t bytes = 0;
    size_t needed = 0;
    for (const size_t cols : {width, 3 * width, 4 * width}) {
        FLINT_RETURN_IF_FAILED(flint_bias_gradient_workspace(rows, cols, &bytes));
        needed = std::max(needed, bytes);
    }
    FLINT_RETURN_IF_FAILED(flint_layernorm_backward_workspace(rows, width, &bytes));
    needed = std::max(needed, bytes);
    FLINT_RETURN_IF_FAILED(flint_embedding_backward_workspace(c.sequences, c.length,
                                                              c.padded_vocab, width, &bytes));
    m.workspace_bytes = std::max(needed, bytes);

    // (m, n) of each weight gradient, all with k = rows: wte, then a block's c_attn, its
    // attention's c_proj, c_fc and the MLP's c_proj.
    const size_t shapes[][2] = {{c.padded_vocab, width},
                                {3 * width, width},
                                {width, width},
                                {4 * width, width},
                                {width, 4 * width}};
    needed = 0;
    for (const auto& shape : shapes) {
        FLINT_RETURN_IF_FAILED(flint_gemm_tn_workspace(shape[0], shape[1], rows, &bytes));
        needed = std::max(needed, bytes);
    }
    m.tn_workspace_bytes = needed;
    return FLINT_OK;
}

// Lays the model out: its parameters in the order of PyTorch's named_parameters, with
// backward their gradients, then the compute copies, the activations, the backward pass's
// scratch and the outputs. It allocates nothing, so flint_gpt2_bytes and flint_gpt2_create
// share it; with backward it asks the GPU for flint_gemm_tn's workspace size. Without
// backward, the activations are one set (the residual stream two buffers, between which
// it ping-pongs), in the order the model has always had. The limits in check_config keep
// every size here far below 2^64.
flint_status lay_out(flint_gpt2& m) {
    const flint_gpt2_config& c = m.config;
    const bool backward = c.backward != 0;
    const size_t width = c.width;
    m.rows = c.sequences * c.length;
    m.head_width = width / c.heads;
    m.head_batches = c.sequences * c.heads;

    m.wte = add_parameter(m, "wte.weight", c.padded_vocab, width);
    m.wpe = add_parameter(m, "wpe.weight", c.block_size, width);
    m.blocks.resize(c.layers);
    for (size_t i = 0; i < c.layers; ++i) {
        const std::string prefix = "blocks." + std::to_string(i) + ".";
        Block& block = m.blocks[i];
        block.ln_1 = add_norm(m, prefix + "ln_1");
        block.attn = add_linear(m, prefix + "attn.c_attn", 3 * width, width);
        block.attn_proj = add_linear(m, prefix + "attn.c_proj", width, width);
        block.ln_2 = add_norm(m, prefix + "ln_2");
        block.mlp = add_linear(m, prefix + "mlp.c_fc", 4 * width, width);
        block.mlp_proj = add_linear(m, prefix + "mlp.c_proj", width, 4 * width);
    }
    m.ln_f = add_norm(m, "ln_f");
    if (backward) {
        for (Parameter& parameter : m.parameters) {
            m.slots.push_back({&parameter.gradient, parameter.rows * parameter.cols * 4});
        }
    }

    m.slots.push_back({&m.wte_compute, c.padded_vocab * width * 2});
    for (Block& block : m.blocks) {
        for (Linear* linear : {&block.attn, &block.attn_proj, &block.mlp, &block.mlp_proj}) {
            m.slots.push_back({&linear->compute, linear->weight->rows * linear->weight->cols * 2});
        }
    }

    const size_t rows = m.rows;
    const size_t residual_bytes = rows * width * 4;
    const size_t rows_bytes = rows * width * 2;
    const size_t heads_bytes = c.sequences * c.heads * c.length * m.head_width * 2;
    const size_t matrix_bytes = m.head_batches * c.length * c.length * 2;
    if (backward) {
        // A block's output is the next block's input, so there are 2 * layers + 1.
        for (Block& block : m.blocks) {
            m.slots.push_back({&block.saved.x_in, residual_bytes});
            m.slots.push_back({&block.saved.x_mid, residual_bytes});
        }
        m.slots.push_back({&m.x_final, residual_bytes});
        for (Block& block : m.blocks) {
            Activations& a = block.saved;
            m.slots.push_back({&a.h1, rows_bytes});
            m.slots.push_back({&a.mean1, rows * 4});
            m.slots.push_back({&a.rstd1, rows * 4});
            m.slots.push_back({&a.q, heads_bytes});
            m.slots.push_back({&a.k, heads_bytes});
            m.slots.push_back({&a.v, heads_bytes});
            m.slots.push_back({&a.p, matrix_bytes});
            m.slots.push_back({&a.attn_output, rows_bytes});
            m.slots.push_back({&a.h2, rows_bytes});
            m.slots.push_back({&a.mean2, rows * 4});
            m.slots.push_back({&a.rstd2, rows * 4});
            m.slots.push_back({&a.gelu, rows * 4 * width * 2});
            m.slots.push_back({&a.pre, rows * 4 * width * 2});
        }
        m.slots.push_back({&m.h_f, rows_bytes});
        m.slots.push_back({&m.mean_f, rows * 4});
        m.slots.push_back({&m.rstd_f, rows * 4});
        m.slots.push_back({&m.qkv, rows * 3 * width * 2});
        m.slots.push_back({&m.s, matrix_bytes});
        m.slots.push_back({&m.heads_output, heads_bytes});

        FLINT_RETURN_IF_FAILED(size_workspaces(m));
        m.slots.push_back({&m.dres, residual_bytes});
        m.slots.push_back({&m.dres_half, rows_bytes});
        m.slots.push_back({&m.dh, rows_bytes});
        m.slots.push_back({&m.dpre, rows * 4 * width * 2});
        m.slots.push_back({&m.dq, heads_bytes});
        m.slots.push_back({&m.dk, heads_bytes});
        m.slots.push_back({&m.dv, heads_bytes});
        m.slots.push_back({&m.workspace, m.workspace_bytes});
        m.slots.push_back({&m.tn_workspace, m.tn_workspace_bytes});
    } else {
        // Block 0's set, which link_activations hands to every block.
        Activations& a = m.blocks[0].saved;
        m.slots.push_back({&a.x_in, residual_bytes});
        m.slots.push_back({&a.x_mid, residual_bytes});
        m.slots.push_back({&a.h1, rows_bytes});
        m.slots.push_back({&a.mean1, rows * 4});
        m.slots.push_back({&a.rstd1, rows * 4});
        m.slots.push_back({&m.qkv, rows * 3 * width * 2});
        m.slots.push_back({&a.q, heads_bytes});
        m.slots.push_back({&a.k, heads_bytes});
        m.slots.push_back({&a.v, heads_bytes});
        m.slots.push_back({&m.s, matrix_bytes});
        m.slots.push_back({&a.p, matrix_bytes});
        m.slots.push_back({&m.heads_output, heads_bytes});
        m.slots.push_back({&a.attn_output, rows_bytes});
        m.slots.push_back({&a.gelu, rows * 4 * width * 2});
        m.slots.push_back({&a.pre, rows * 4 * width * 2});
    }
    m.slots.push_back({&m.logits, rows * c.padded_vocab * 2});
    m.slots.push_back({&m.dlogits, rows * c.padded_vocab * 2});
    m.slots.push_back({&m.row_loss, rows * 4});
    return FLINT_OK;
}

// Fills in the activation pointers that aren't buffers of their own, after allocation. With
// backward, a block's output is the next one's input, and the last block's is x_final.
// Without, every block shares block 0's set: the residual stream leaves each block where it
// entered, and both LayerNorms and ln_f share one output and one set of statistics.
void link_activations(flint_gpt2& m) {
    if (m.config.backward != 0) {
        for (size_t i = 0; i < m.blocks.size(); ++i) {
            m.blocks[i].saved.x_out =
                i + 1 < m.blocks.size() ? m.blocks[i + 1].saved.x_in : m.x_final;
        }
        return;
    }
    Activations& shared = m.blocks[0].saved;
    shared.x_out = shared.x_in;
    shared.h2 = shared.h1;
    shared.mean2 = shared.mean1;
    shared.rstd2 = shared.rstd1;
    for (Block& block : m.blocks) {
        block.saved = shared;
    }
    m.x_final = shared.x_in;
    m.h_f = shared.h1;
    m.mean_f = shared.mean1;
    m.rstd_f = shared.rstd1;
}

size_t total_bytes(const flint_gpt2& m) {
    size_t total = 0;
    for (const Slot& slot : m.slots) {
        total += rounded_up(slot.bytes);
    }
    return total;
}

flint_status allocate(flint_gpt2& m) {
    // Reserved first, so that recording an allocation can't throw and lose it.
    m.allocations.reserve(m.slots.size());
    if (m.config.separate_buffers != 0) {
        for (const Slot& slot : m.slots) {
            void* pointer = nullptr;
            FLINT_CUDA_CHECK(cudaMalloc(&pointer, slot.bytes));
            m.allocations.push_back(pointer);
            *slot.pointer = pointer;
        }
        return FLINT_OK;
    }
    void* base = nullptr;
    FLINT_CUDA_CHECK(cudaMalloc(&base, total_bytes(m)));
    m.allocations.push_back(base);
    char* next = static_cast<char*>(base);
    for (const Slot& slot : m.slots) {
        *slot.pointer = next;
        next += rounded_up(slot.bytes);
    }
    return FLINT_OK;
}

flint_status sync_linear(const Linear& linear, flint_stream stream) {
    return flint_fp32_to_fp16(linear.compute, linear.weight->data,
                              linear.weight->rows * linear.weight->cols, stream);
}

flint_status norm(const flint_gpt2& m, const Norm& layer, const void* x, void* y, void* mean,
                  void* rstd, flint_stream stream) {
    return flint_layernorm_forward(y, mean, rstd, x, layer.weight->data, layer.bias->data, m.rows,
                                   m.config.width, kLayerNormEps, stream);
}

// The default scale of attention, 1 / sqrt(head width), in FP32.
float score_scale(size_t head_width) {
    return 1.0f / std::sqrt(static_cast<float>(head_width));
}

// One block: x + attention(LayerNorm(x)), then that + MLP(LayerNorm(that)), from x_in
// through x_mid to x_out, keeping its activations where block.saved points.
flint_status run_block(const flint_gpt2& m, const Block& block, flint_stream stream) {
    const flint_gpt2_config& c = m.config;
    const Activations& a = block.saved;
    const size_t rows = m.rows;
    const size_t width = c.width;
    const size_t head_width = m.head_width;

    FLINT_RETURN_IF_FAILED(norm(m, block.ln_1, a.x_in, a.h1, a.mean1, a.rstd1, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_nt_bias(m.qkv, a.h1, block.attn.compute,
                                              block.attn.bias->data, rows, 3 * width, width,
                                              stream));
    void* const parts[] = {a.q, a.k, a.v};
    for (size_t i = 0; i < std::size(parts); ++i) {
        FLINT_RETURN_IF_FAILED(flint_heads_split(parts[i], m.qkv, c.sequences, c.length, c.heads,
                                                 head_width, 3 * width, i * width, stream));
    }
    FLINT_RETURN_IF_FAILED(flint_gemm_nt_batched(m.s, a.q, a.k, m.head_batches, c.length, c.length,
                                                 head_width, score_scale(head_width), stream));
    FLINT_RETURN_IF_FAILED(flint_causal_softmax_forward(a.p, m.s, m.head_batches, c.length, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_nn_batched(m.heads_output, a.p, a.v, m.head_batches, c.length,
                                                 head_width, c.length, stream));
    FLINT_RETURN_IF_FAILED(flint_heads_merge(a.attn_output, m.heads_output, c.sequences, c.length,
                                             c.heads, head_width, width, 0, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_nt_bias_residual(a.x_mid, a.attn_output,
                                                       block.attn_proj.compute,
                                                       block.attn_proj.bias->data, a.x_in, rows,
                                                       width, width, stream));

    FLINT_RETURN_IF_FAILED(norm(m, block.ln_2, a.x_mid, a.h2, a.mean2, a.rstd2, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_nt_bias_gelu(a.gelu, a.pre, a.h2, block.mlp.compute,
                                                   block.mlp.bias->data, rows, 4 * width, width,
                                                   stream));
    return flint_gemm_nt_bias_residual(a.x_out, a.gelu, block.mlp_proj.compute,
                                       block.mlp_proj.bias->data, a.x_mid, rows, width, 4 * width,
                                       stream);
}

// A weight's gradient += x^T w, over the rows: written, or added to with accumulate.
flint_status weight_gradient(const flint_gpt2& m, const Parameter& weight, const void* x,
                             const void* w, int accumulate, flint_stream stream) {
    return flint_gemm_tn(weight.gradient, x, w, weight.rows, weight.cols, m.rows, accumulate,
                         m.tn_workspace, m.tn_workspace_bytes, stream);
}

flint_status bias_gradient(const flint_gpt2& m, const Parameter& bias, const void* dy,
                           int accumulate, flint_stream stream) {
    return flint_bias_gradient(bias.gradient, dy, m.rows, bias.cols, accumulate, m.workspace,
                               m.workspace_bytes, stream);
}

// A LayerNorm's backward pass from dh, its output's gradient: its parameters' gradients +=,
// and its input's gradient into dres, added to what dres holds with accumulate_dx.
flint_status norm_backward(const flint_gpt2& m, const Norm& layer, const void* x,
                           const void* mean, const void* rstd, int accumulate_dx, int accumulate,
                           flint_stream stream) {
    return flint_layernorm_backward(m.dres, layer.weight->gradient, layer.bias->gradient, m.dh, x,
                                    mean, rstd, layer.weight->data, m.rows, m.config.width,
                                    accumulate_dx, accumulate, m.workspace, m.workspace_bytes,
                                    stream);
}

// One block's backward pass, MLP then attention. dres holds the gradient of the residual
// stream leaving the block on entry, and of the one entering it on return; each LayerNorm's
// backward pass adds its input's gradient to the identity path's, already in dres.
flint_status run_block_backward(const flint_gpt2& m, const Block& block, int accumulate,
                                flint_stream stream) {
    const flint_gpt2_config& c = m.config;
    const Activations& a = block.saved;
    const size_t rows = m.rows;
    const size_t width = c.width;
    const size_t head_width = m.head_width;
    const size_t batches = m.head_batches;
    const size_t length = c.length;

    // The MLP: the gradient of its output is dres's, rounded to FP16 as autocast's is.
    FLINT_RETURN_IF_FAILED(flint_fp32_to_fp16(m.dres_half, m.dres, rows * width, stream));
    FLINT_RETURN_IF_FAILED(
        weight_gradient(m, *block.mlp_proj.weight, m.dres_half, a.gelu, accumulate, stream));
    FLINT_RETURN_IF_FAILED(bias_gradient(m, *block.mlp_proj.bias, m.dres_half, accumulate, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_nn_dgelu(m.dpre, m.dres_half, block.mlp_proj.compute, a.pre,
                                               rows, 4 * width, width, stream));
    FLINT_RETURN_IF_FAILED(weight_gradient(m, *block.mlp.weight, m.dpre, a.h2, accumulate, stream));
    FLINT_RETURN_IF_FAILED(bias_gradient(m, *block.mlp.bias, m.dpre, accumulate, stream));
    FLINT_RETURN_IF_FAILED(
        flint_gemm_nn(m.dh, m.dpre, block.mlp.compute, rows, width, 4 * width, stream));
    FLINT_RETURN_IF_FAILED(
        norm_backward(m, block.ln_2, a.x_mid, a.mean2, a.rstd2, 1, accumulate, stream));

    // The attention's output projection; its input's gradient goes to dh, then to the heads.
    FLINT_RETURN_IF_FAILED(flint_fp32_to_fp16(m.dres_half, m.dres, rows * width, stream));
    FLINT_RETURN_IF_FAILED(weight_gradient(m, *block.attn_proj.weight, m.dres_half,
                                           a.attn_output, accumulate, stream));
    FLINT_RETURN_IF_FAILED(
        bias_gradient(m, *block.attn_proj.bias, m.dres_half, accumulate, stream));
    FLINT_RETURN_IF_FAILED(
        flint_gemm_nn(m.dh, m.dres_half, block.attn_proj.compute, rows, width, width, stream));
    // dO, in the heads' output buffer, which the forward pass is done with.
    FLINT_RETURN_IF_FAILED(flint_heads_split(m.heads_output, m.dh, c.sequences, length, c.heads,
                                             head_width, width, 0, stream));
    // O = P V: dV = P^T dO, and dP = dO V^T, into the scores' buffer.
    FLINT_RETURN_IF_FAILED(flint_gemm_tn_batched(m.dv, a.p, m.heads_output, batches, length,
                                                 head_width, length, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_nt_batched(m.s, m.heads_output, a.v, batches, length, length,
                                                 head_width, 1.0f, stream));
    // dS in place over dP, carrying the scores' scale, so that dQ and dK need none.
    FLINT_RETURN_IF_FAILED(flint_causal_softmax_backward(m.s, a.p, m.s, batches, length,
                                                         score_scale(head_width), stream));
    // S = scale Q K^T: dQ = dS K, and dK = dS^T Q.
    FLINT_RETURN_IF_FAILED(flint_gemm_nn_batched(m.dq, m.s, a.k, batches, length, head_width,
                                                 length, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_tn_batched(m.dk, m.s, a.q, batches, length, head_width,
                                                 length, stream));
    // The fused qkv rows' gradient, in the qkv buffer.
    const void* const parts[] = {m.dq, m.dk, m.dv};
    for (size_t i = 0; i < std::size(parts); ++i) {
        FLINT_RETURN_IF_FAILED(flint_heads_merge(m.qkv, parts[i], c.sequences, length, c.heads,
                                                 head_width, 3 * width, i * width, stream));
    }
    FLINT_RETURN_IF_FAILED(weight_gradient(m, *block.attn.weight, m.qkv, a.h1, accumulate, stream));
    FLINT_RETURN_IF_FAILED(bias_gradient(m, *block.attn.bias, m.qkv, accumulate, stream));
    FLINT_RETURN_IF_FAILED(
        flint_gemm_nn(m.dh, m.qkv, block.attn.compute, rows, width, 3 * width, stream));
    return norm_backward(m, block.ln_1, a.x_in, a.mean1, a.rstd1, 1, accumulate, stream);
}

}  // namespace

flint_status flint_gpt2_bytes(const flint_gpt2_config* config, size_t* bytes) {
    constexpr const char* name = "flint_gpt2_bytes";
    FLINT_RETURN_IF_FAILED(check_config(name, config));
    if (bytes == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: bytes is null", name);
    }
    try {
        flint_gpt2 m;
        m.config = *config;
        FLINT_RETURN_IF_FAILED(lay_out(m));
        *bytes = total_bytes(m);
    } catch (const std::bad_alloc&) {
        return flint::fail(FLINT_ERROR_CUDA, "%s: out of host memory", name);
    }
    return FLINT_OK;
}

flint_status flint_gpt2_create(const flint_gpt2_config* config, flint_gpt2** model) {
    constexpr const char* name = "flint_gpt2_create";
    if (model == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: model is null", name);
    }
    *model = nullptr;
    FLINT_RETURN_IF_FAILED(check_config(name, config));
    try {
        auto m = std::make_unique<flint_gpt2>();
        m->config = *config;
        FLINT_RETURN_IF_FAILED(lay_out(*m));
        m->host_row_loss.resize(m->rows);
        // The destructor frees whatever a failed allocation left behind.
        FLINT_RETURN_IF_FAILED(allocate(*m));
        link_activations(*m);
        *model = m.release();
    } catch (const std::bad_alloc&) {
        return flint::fail(FLINT_ERROR_CUDA, "%s: out of host memory", name);
    }
    return FLINT_OK;
}

void flint_gpt2_destroy(flint_gpt2* model) {
    delete model;
}

size_t flint_gpt2_parameter_count(const flint_gpt2* model) {
    return model == nullptr ? 0 : model->parameters.size();
}

namespace {

// flint_gpt2_parameter and flint_gpt2_gradient: the index-th parameter's name and shape,
// with its master weight or its gradient.
flint_status parameter_info(const char* name, const flint_gpt2* model, size_t index,
                            flint_gpt2_parameter_info* info, bool gradient) {
    if (model == nullptr || info == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: model and info must not be null",
                           name);
    }
    if (gradient && model->config.backward == 0) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: the model was created without backward, so it has no gradients",
                           name);
    }
    if (index >= model->parameters.size()) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: index (%zu) must be below %zu", name,
                           index, model->parameters.size());
    }
    const Parameter& parameter = model->parameters[index];
    *info = {parameter.name.c_str(), gradient ? parameter.gradient : parameter.data,
             parameter.rows, parameter.cols};
    return FLINT_OK;
}

}  // namespace

flint_status flint_gpt2_parameter(const flint_gpt2* model, size_t index,
                                  flint_gpt2_parameter_info* info) {
    return parameter_info("flint_gpt2_parameter", model, index, info, false);
}

flint_status flint_gpt2_gradient(const flint_gpt2* model, size_t index,
                                 flint_gpt2_parameter_info* info) {
    return parameter_info("flint_gpt2_gradient", model, index, info, true);
}

flint_status flint_gpt2_sync_weights(flint_gpt2* model, flint_stream stream) {
    if (model == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "flint_gpt2_sync_weights: model is null");
    }
    FLINT_RETURN_IF_FAILED(flint_fp32_to_fp16(model->wte_compute, model->wte->data,
                                              model->wte->rows * model->wte->cols, stream));
    for (const Block& block : model->blocks) {
        FLINT_RETURN_IF_FAILED(sync_linear(block.attn, stream));
        FLINT_RETURN_IF_FAILED(sync_linear(block.attn_proj, stream));
        FLINT_RETURN_IF_FAILED(sync_linear(block.mlp, stream));
        FLINT_RETURN_IF_FAILED(sync_linear(block.mlp_proj, stream));
    }
    return FLINT_OK;
}

flint_status flint_gpt2_forward(flint_gpt2* model, const void* tokens, const void* targets,
                                float scale, flint_stream stream) {
    constexpr const char* name = "flint_gpt2_forward";
    if (model == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: model is null", name);
    }
    if (!std::isfinite(scale)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: scale (%g) must be finite", name,
                           static_cast<double>(scale));
    }
    const flint_gpt2& m = *model;
    const flint_gpt2_config& c = m.config;
    const size_t rows = m.rows;
    // Checked here so that a bad pointer fails before anything launches. The model's
    // outputs are the buffers a caller could hand back by mistake.
    const flint::Buffer buffers[] = {
        {"logits", m.logits, rows * c.padded_vocab * 2},
        {"dlogits", m.dlogits, rows * c.padded_vocab * 2},
        {"row_loss", m.row_loss, rows * 4},
        {"tokens", tokens, rows * 2},
        {"targets", targets, rows * 2},
    };
    FLINT_RETURN_IF_FAILED(flint::check_buffers(name, buffers, std::size(buffers), 3));

    FLINT_RETURN_IF_FAILED(flint_embedding_forward(m.blocks[0].saved.x_in, tokens, m.wte->data,
                                                   m.wpe->data, c.sequences, c.length,
                                                   c.padded_vocab, c.width, stream));
    for (const Block& block : m.blocks) {
        FLINT_RETURN_IF_FAILED(run_block(m, block, stream));
    }
    FLINT_RETURN_IF_FAILED(norm(m, m.ln_f, m.x_final, m.h_f, m.mean_f, m.rstd_f, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_nt(m.logits, m.h_f, m.wte_compute, rows, c.padded_vocab,
                                         c.width, stream));
    return flint_softmax_cross_entropy(m.dlogits, m.row_loss, m.logits, targets, scale, rows,
                                       c.vocab, c.padded_vocab, stream);
}

flint_status flint_gpt2_backward(flint_gpt2* model, const void* tokens, int accumulate,
                                 flint_stream stream) {
    constexpr const char* name = "flint_gpt2_backward";
    if (model == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: model is null", name);
    }
    const flint_gpt2& m = *model;
    const flint_gpt2_config& c = m.config;
    if (c.backward == 0) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: the model was created without backward", name);
    }
    const size_t rows = m.rows;
    const size_t width = c.width;
    // As in the forward pass, and the gradients are buffers a caller could hand back too.
    const flint::Buffer buffers[] = {
        {"logits", m.logits, rows * c.padded_vocab * 2},
        {"dlogits", m.dlogits, rows * c.padded_vocab * 2},
        {"row_loss", m.row_loss, rows * 4},
        {"tokens", tokens, rows * 2},
    };
    FLINT_RETURN_IF_FAILED(flint::check_buffers(name, buffers, std::size(buffers), 3));
    for (const Parameter& parameter : m.parameters) {
        if (flint::overlap(tokens, rows * 2, parameter.gradient,
                           parameter.rows * parameter.cols * 4)) {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                               "%s: tokens and the gradient of %s overlap", name,
                               parameter.name.c_str());
        }
    }

    // The output layer, logits = h_f wte^T: wte's gradient gets dlogits^T h_f, and h_f's
    // gradient is dlogits wte.
    FLINT_RETURN_IF_FAILED(weight_gradient(m, *m.wte, m.dlogits, m.h_f, accumulate, stream));
    FLINT_RETURN_IF_FAILED(
        flint_gemm_nn(m.dh, m.dlogits, m.wte_compute, rows, width, c.padded_vocab, stream));
    // ln_f writes dres, which then gathers each block's gradient.
    FLINT_RETURN_IF_FAILED(
        norm_backward(m, m.ln_f, m.x_final, m.mean_f, m.rstd_f, 0, accumulate, stream));
    for (auto block = m.blocks.rbegin(); block != m.blocks.rend(); ++block) {
        FLINT_RETURN_IF_FAILED(run_block_backward(m, *block, accumulate, stream));
    }
    // The embedding always adds to wte's gradient, which holds the output layer's part, so
    // the tied gradient is (old + output layer) + embedding, in that order.
    FLINT_RETURN_IF_FAILED(flint_embedding_backward(m.wte->gradient, m.wpe->gradient, m.dres,
                                                    tokens, c.sequences, c.length, c.padded_vocab,
                                                    width, 1, accumulate, m.workspace,
                                                    m.workspace_bytes, stream));
    // The positions from length to block_size are never seen, so their gradient is 0, which
    // the embedding's kernel doesn't write.
    if (accumulate == 0 && c.block_size > c.length) {
        char* const unseen = static_cast<char*>(m.wpe->gradient) + c.length * width * 4;
        FLINT_CUDA_CHECK(cudaMemsetAsync(unseen, 0, (c.block_size - c.length) * width * 4, stream));
    }
    return FLINT_OK;
}

flint_status flint_gpt2_outputs(const flint_gpt2* model, const void** logits, const void** dlogits,
                                const void** row_loss) {
    if (model == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "flint_gpt2_outputs: model is null");
    }
    if (logits != nullptr) {
        *logits = model->logits;
    }
    if (dlogits != nullptr) {
        *dlogits = model->dlogits;
    }
    if (row_loss != nullptr) {
        *row_loss = model->row_loss;
    }
    return FLINT_OK;
}

flint_status flint_gpt2_mean_loss(flint_gpt2* model, flint_stream stream, double* loss) {
    constexpr const char* name = "flint_gpt2_mean_loss";
    if (model == nullptr || loss == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: model and loss must not be null",
                           name);
    }
    std::vector<float>& host = model->host_row_loss;
    FLINT_CUDA_CHECK(cudaMemcpyAsync(host.data(), model->row_loss, host.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
    FLINT_CUDA_CHECK(cudaStreamSynchronize(stream));
    double sum = 0.0;
    for (const float value : host) {
        sum += static_cast<double>(value);
    }
    *loss = sum / static_cast<double>(host.size());
    return FLINT_OK;
}
