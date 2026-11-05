// GPT-2's forward pass: the model's memory, its weights, and the kernels in order.
// This is host code. Every launch goes through the public API, whose checks then
// also guard the assembly: a wrong size here fails rather than running.
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
// GPT-2's LayerNorm eps, which PyTorch's model also uses.
constexpr float kLayerNormEps = 1e-5f;

// A master weight: FP32 in device memory, rows x cols.
struct Parameter {
    std::string name;
    size_t rows;
    size_t cols;
    void* data;
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

struct Block {
    Norm ln_1;
    Linear attn;
    Linear attn_proj;
    Norm ln_2;
    Linear mlp;
    Linear mlp_proj;
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

    // The residual stream, FP32: each layer reads one of these and writes the other.
    void* residual[2] = {};
    // A LayerNorm's output and its statistics. Layers reuse all of these.
    void* h = nullptr;
    void* mean = nullptr;
    void* rstd = nullptr;
    // Attention: Q, K and V's rows, then their heads, scores, probabilities, the
    // heads' output and that output's rows.
    void* qkv = nullptr;
    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* s = nullptr;
    void* p = nullptr;
    void* heads_output = nullptr;
    void* attn_output = nullptr;
    // The MLP's GELU and the values entering it.
    void* gelu = nullptr;
    void* pre = nullptr;
    void* logits = nullptr;
    void* dlogits = nullptr;
    void* row_loss = nullptr;

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
    return FLINT_OK;
}

Parameter* add_parameter(flint_gpt2& m, std::string name, size_t rows, size_t cols) {
    m.parameters.push_back({std::move(name), rows, cols, nullptr});
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

// Lays the model out: its parameters in the order of PyTorch's named_parameters,
// then the compute copies and the activations. It allocates nothing, so
// flint_gpt2_bytes and flint_gpt2_create share it. The limits in check_config keep
// every size here far below 2^64.
void lay_out(flint_gpt2& m) {
    const flint_gpt2_config& c = m.config;
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

    m.slots.push_back({&m.wte_compute, c.padded_vocab * width * 2});
    for (Block& block : m.blocks) {
        for (Linear* linear : {&block.attn, &block.attn_proj, &block.mlp, &block.mlp_proj}) {
            m.slots.push_back({&linear->compute, linear->weight->rows * linear->weight->cols * 2});
        }
    }

    const size_t rows = m.rows;
    const size_t heads_bytes = c.sequences * c.heads * c.length * m.head_width * 2;
    const size_t matrix_bytes = m.head_batches * c.length * c.length * 2;
    m.slots.push_back({&m.residual[0], rows * width * 4});
    m.slots.push_back({&m.residual[1], rows * width * 4});
    m.slots.push_back({&m.h, rows * width * 2});
    m.slots.push_back({&m.mean, rows * 4});
    m.slots.push_back({&m.rstd, rows * 4});
    m.slots.push_back({&m.qkv, rows * 3 * width * 2});
    m.slots.push_back({&m.q, heads_bytes});
    m.slots.push_back({&m.k, heads_bytes});
    m.slots.push_back({&m.v, heads_bytes});
    m.slots.push_back({&m.s, matrix_bytes});
    m.slots.push_back({&m.p, matrix_bytes});
    m.slots.push_back({&m.heads_output, heads_bytes});
    m.slots.push_back({&m.attn_output, rows * width * 2});
    m.slots.push_back({&m.gelu, rows * 4 * width * 2});
    m.slots.push_back({&m.pre, rows * 4 * width * 2});
    m.slots.push_back({&m.logits, rows * c.padded_vocab * 2});
    m.slots.push_back({&m.dlogits, rows * c.padded_vocab * 2});
    m.slots.push_back({&m.row_loss, rows * 4});
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

// Returns a failed call's status from the calling function.
#define FLINT_RETURN_IF_FAILED(call)                \
    do {                                            \
        const flint_status flint_status_ = (call);  \
        if (flint_status_ != FLINT_OK) {            \
            return flint_status_;                   \
        }                                           \
    } while (0)

flint_status sync_linear(const Linear& linear, flint_stream stream) {
    return flint_fp32_to_fp16(linear.compute, linear.weight->data,
                              linear.weight->rows * linear.weight->cols, stream);
}

flint_status norm(const flint_gpt2& m, const Norm& layer, const void* x, flint_stream stream) {
    return flint_layernorm_forward(m.h, m.mean, m.rstd, x, layer.weight->data, layer.bias->data,
                                   m.rows, m.config.width, kLayerNormEps, stream);
}

// One block: x + attention(LayerNorm(x)), then that + MLP(LayerNorm(that)). The
// residual stream starts and ends in `from`, and `to` holds it in between.
flint_status run_block(const flint_gpt2& m, const Block& block, void* from, void* to,
                       flint_stream stream) {
    const flint_gpt2_config& c = m.config;
    const size_t rows = m.rows;
    const size_t width = c.width;
    const size_t head_width = m.head_width;

    FLINT_RETURN_IF_FAILED(norm(m, block.ln_1, from, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_nt_bias(m.qkv, m.h, block.attn.compute,
                                              block.attn.bias->data, rows, 3 * width, width,
                                              stream));
    void* const parts[] = {m.q, m.k, m.v};
    for (size_t i = 0; i < std::size(parts); ++i) {
        FLINT_RETURN_IF_FAILED(flint_heads_split(parts[i], m.qkv, c.sequences, c.length, c.heads,
                                                 head_width, 3 * width, i * width, stream));
    }
    // The default scale of attention, 1 / sqrt(head width).
    const float score_scale = 1.0f / std::sqrt(static_cast<float>(head_width));
    FLINT_RETURN_IF_FAILED(flint_gemm_nt_batched(m.s, m.q, m.k, m.head_batches, c.length, c.length,
                                                 head_width, score_scale, stream));
    FLINT_RETURN_IF_FAILED(flint_causal_softmax_forward(m.p, m.s, m.head_batches, c.length, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_nn_batched(m.heads_output, m.p, m.v, m.head_batches, c.length,
                                                 head_width, c.length, stream));
    FLINT_RETURN_IF_FAILED(flint_heads_merge(m.attn_output, m.heads_output, c.sequences, c.length,
                                             c.heads, head_width, width, 0, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_nt_bias_residual(to, m.attn_output, block.attn_proj.compute,
                                                       block.attn_proj.bias->data, from, rows,
                                                       width, width, stream));

    FLINT_RETURN_IF_FAILED(norm(m, block.ln_2, to, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_nt_bias_gelu(m.gelu, m.pre, m.h, block.mlp.compute,
                                                   block.mlp.bias->data, rows, 4 * width, width,
                                                   stream));
    return flint_gemm_nt_bias_residual(from, m.gelu, block.mlp_proj.compute,
                                       block.mlp_proj.bias->data, to, rows, width, 4 * width,
                                       stream);
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
        lay_out(m);
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
        lay_out(*m);
        m->host_row_loss.resize(m->rows);
        // The destructor frees whatever a failed allocation left behind.
        FLINT_RETURN_IF_FAILED(allocate(*m));
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

flint_status flint_gpt2_parameter(const flint_gpt2* model, size_t index,
                                  flint_gpt2_parameter_info* info) {
    constexpr const char* name = "flint_gpt2_parameter";
    if (model == nullptr || info == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: model and info must not be null",
                           name);
    }
    if (index >= model->parameters.size()) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: index (%zu) must be below %zu", name,
                           index, model->parameters.size());
    }
    const Parameter& parameter = model->parameters[index];
    *info = {parameter.name.c_str(), parameter.data, parameter.rows, parameter.cols};
    return FLINT_OK;
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

    void* x = m.residual[0];
    void* y = m.residual[1];
    FLINT_RETURN_IF_FAILED(flint_embedding_forward(x, tokens, m.wte->data, m.wpe->data, c.sequences,
                                                   c.length, c.padded_vocab, c.width, stream));
    for (const Block& block : m.blocks) {
        FLINT_RETURN_IF_FAILED(run_block(m, block, x, y, stream));
    }
    FLINT_RETURN_IF_FAILED(norm(m, m.ln_f, x, stream));
    FLINT_RETURN_IF_FAILED(flint_gemm_nt(m.logits, m.h, m.wte_compute, rows, c.padded_vocab, c.width,
                                         stream));
    return flint_softmax_cross_entropy(m.dlogits, m.row_loss, m.logits, targets, scale, rows,
                                       c.vocab, c.padded_vocab, stream);
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
