// The safetensors format, in C++: parsing and checking a file's header, and building one. The
// bytes of the 8-byte length and the data are the callers' business; this is only the JSON header
// and the rules about it. Nothing here touches a file or the GPU.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace flint::safetensors {

// A header longer than this, padding included, is refused.
constexpr uint64_t kMaxHeaderBytes = 100000000;

// One of the 22 dtypes, and the bits one element takes.
struct Dtype {
    const char* name;
    unsigned bits;
};

// The dtype with this exact name, or null.
const Dtype* find_dtype(std::string_view name);

// A writer's declaration breaks a rule. what() says which, without the function's name.
class ArgumentError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// How big a tensor is, or why it can't be.
enum class SizeResult { kOk, kOverflow, kNotWholeBytes };

// The bytes of a tensor of this dtype and shape: the shape's values multiplied left to right
// from 1, each product checked even if a later 0 would make it 0, then times the dtype's bits,
// which must be a whole number of bytes.
SizeResult tensor_bytes(const Dtype& dtype, const size_t* shape, size_t rank, uint64_t& bytes);

struct TensorEntry {
    size_t name_offset;  // into Header::names, where the name sits followed by a NUL
    size_t name_size;
    const Dtype* dtype;
    size_t shape_offset;  // into Header::shapes
    size_t rank;
    uint64_t begin;  // from the data's start
    uint64_t end;
};

struct MetadataEntry {
    size_t key_offset;  // into Header::metadata_text, each string followed by a NUL
    size_t key_size;
    size_t value_offset;
    size_t value_size;
};

// What a valid header says, in the order it says it. The lookup tables point into the strings,
// so a Header stays where it is built.
struct Header {
    Header() : shapes(1, 0) {}
    Header(const Header&) = delete;
    Header& operator=(const Header&) = delete;

    std::string names;
    std::vector<TensorEntry> tensors;
    // Every tensor's shape, one after the other. It starts with an unused value, so a rank 0
    // tensor's shape still points somewhere.
    std::vector<size_t> shapes;
    std::string metadata_text;
    std::vector<MetadataEntry> metadata;
    std::unordered_map<std::string_view, size_t> tensor_by_name;
    std::unordered_map<std::string_view, size_t> metadata_by_key;

    const char* name(size_t index) const { return names.data() + tensors[index].name_offset; }
    const size_t* shape(size_t index) const { return shapes.data() + tensors[index].shape_offset; }
    const char* key(size_t index) const {
        return metadata_text.data() + metadata[index].key_offset;
    }
    const char* value(size_t index) const {
        return metadata_text.data() + metadata[index].value_offset;
    }
};

// Parses the size header bytes at text, which follow the 8-byte length, and checks every rule:
// valid UTF-8, the JSON grammar, the structure, each tensor's size, and that the tensors tile the
// data_length bytes of data exactly. Throws json::FormatError, naming the rule and the tensor.
void parse_header(const char* text, size_t size, uint64_t data_length, Header& header);

// A tensor a writer declares. Every pointer may be null; build_header checks.
struct Declaration {
    const char* name;
    const char* dtype;
    size_t rank;
    const size_t* shape;
};

struct Placement {
    size_t bits;
    uint64_t offset;
    uint64_t bytes;
};

struct Layout {
    std::string header;  // the JSON, padded with spaces to a multiple of 8 bytes
    std::vector<Placement> placements;
    uint64_t data_length = 0;
};

// Checks the declarations and lays the tensors out one after the other in the order given, then
// builds the compact JSON header around them, with the metadata pairs (keys and values, in the
// order given) first when there are any. Throws ArgumentError.
void build_header(const Declaration* tensors, size_t tensor_count, const char* const* keys,
                  const char* const* values, size_t metadata_count, Layout& layout);

// `name`, cut to 128 bytes, as a message quotes it.
std::string quote(std::string_view name);

}  // namespace flint::safetensors
