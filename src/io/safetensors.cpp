#include "io/safetensors.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <iterator>
#include <unordered_set>

#include "io/json.h"

namespace flint::safetensors {

namespace {

using json::Cursor;
using json::FormatError;

// The 22 dtypes of safetensors 0.8.0.
constexpr Dtype kDtypes[] = {
    {"F4", 4},
    {"F6_E2M3", 6},
    {"F6_E3M2", 6},
    {"BOOL", 8},
    {"U8", 8},
    {"I8", 8},
    {"F8_E5M2", 8},
    {"F8_E4M3", 8},
    {"F8_E8M0", 8},
    {"F8_E4M3FNUZ", 8},
    {"F8_E5M2FNUZ", 8},
    {"I16", 16},
    {"U16", 16},
    {"F16", 16},
    {"BF16", 16},
    {"I32", 32},
    {"U32", 32},
    {"F32", 32},
    {"C64", 64},
    {"F64", 64},
    {"I64", 64},
    {"U64", 64},
};
static_assert(std::size(kDtypes) == 22, "safetensors 0.8.0 has 22 dtypes");

constexpr const char* kMetadataKey = "__metadata__";

bool has_nul(const std::string& text) {
    return std::memchr(text.data(), '\0', text.size()) != nullptr;
}

// A name for a message: backquoted, and cut to 128 bytes.
std::string quoted(const std::string& name) { return quote(name); }

// A dtype, which serde reads from a string, or from a map with one key whose value is null (a
// unit variant's other spelling). Either way it must be one of the 22 names.
const Dtype* read_dtype(Cursor& in, const std::string& name, std::string& scratch) {
    scratch.clear();
    if (in.peek() == '"') {
        in.read_string(&scratch);
    } else if (in.peek() == '{') {
        in.advance();
        in.skip_whitespace();
        in.read_string(&scratch);
        in.skip_whitespace();
        in.expect(':');
        in.skip_whitespace();
        if (in.peek() != 'n') {
            throw FormatError("the dtype of tensor " + quoted(name) +
                              " is a map whose value isn't null");
        }
        in.skip_value(3);
        in.skip_whitespace();
        in.expect('}');
    } else {
        throw FormatError("the dtype of tensor " + quoted(name) + " isn't a string or a map");
    }
    const Dtype* dtype = find_dtype(scratch);
    if (dtype == nullptr) {
        throw FormatError("tensor " + quoted(name) + " has the unknown dtype " + quote(scratch));
    }
    return dtype;
}

// A shape: an array of unsigned integers, appended to the header's pool.
void read_shape(Cursor& in, Header& header) {
    in.expect('[');
    in.skip_whitespace();
    if (in.peek() == ']') {
        in.advance();
        return;
    }
    for (;;) {
        in.skip_whitespace();
        header.shapes.push_back(static_cast<size_t>(in.read_unsigned()));
        in.skip_whitespace();
        const int next = in.peek();
        if (next == ',') {
            in.advance();
        } else if (next == ']') {
            in.advance();
            return;
        } else {
            in.fail("expected `,` or `]` in a shape");
        }
    }
}

// data_offsets: an array of exactly two unsigned integers.
void read_offsets(Cursor& in, const std::string& name, uint64_t& begin, uint64_t& end) {
    in.expect('[');
    in.skip_whitespace();
    begin = in.read_unsigned();
    in.skip_whitespace();
    if (in.peek() != ',') {
        throw FormatError("data_offsets of tensor " + quoted(name) +
                          " must hold exactly two integers");
    }
    in.advance();
    in.skip_whitespace();
    end = in.read_unsigned();
    in.skip_whitespace();
    if (in.peek() != ']') {
        throw FormatError("data_offsets of tensor " + quoted(name) +
                          " must hold exactly two integers");
    }
    in.advance();
}

// The tensor's value, whose key is name: an object, where dtype, shape and data_offsets are
// required and any other key's value is skipped, or an array of exactly those three, in that
// order (serde reads the struct from either). Then its size and place are checked.
void parse_tensor(Cursor& in, const std::string& name, Header& header, std::string& scratch) {
    if (has_nul(name)) {
        throw FormatError("the tensor name " + quoted(name) + " holds a NUL");
    }
    const int opening = in.peek();
    if (opening != '{' && opening != '[') {
        throw FormatError("the value of tensor " + quoted(name) + " isn't an object or an array");
    }
    in.advance();
    const Dtype* dtype = nullptr;
    bool have_shape = false;
    bool have_offsets = false;
    const size_t shape_offset = header.shapes.size();
    uint64_t begin = 0;
    uint64_t end = 0;

    if (opening == '[') {
        in.skip_whitespace();
        dtype = read_dtype(in, name, scratch);
        in.skip_whitespace();
        in.expect(',');
        in.skip_whitespace();
        read_shape(in, header);
        have_shape = true;
        in.skip_whitespace();
        in.expect(',');
        in.skip_whitespace();
        read_offsets(in, name, begin, end);
        have_offsets = true;
        in.skip_whitespace();
        in.expect(']');
    } else {
        in.skip_whitespace();
        if (in.peek() == '}') {
            in.advance();
        } else {
            for (;;) {
                in.skip_whitespace();
                scratch.clear();
                in.read_string(&scratch);
                in.skip_whitespace();
                in.expect(':');
                in.skip_whitespace();
                if (scratch == "dtype") {
                    if (dtype != nullptr) {
                        throw FormatError("tensor " + quoted(name) + " has `dtype` twice");
                    }
                    dtype = read_dtype(in, name, scratch);
                } else if (scratch == "shape") {
                    if (have_shape) {
                        throw FormatError("tensor " + quoted(name) + " has `shape` twice");
                    }
                    have_shape = true;
                    read_shape(in, header);
                } else if (scratch == "data_offsets") {
                    if (have_offsets) {
                        throw FormatError("tensor " + quoted(name) + " has `data_offsets` twice");
                    }
                    have_offsets = true;
                    read_offsets(in, name, begin, end);
                } else {
                    in.skip_value(2);
                }
                in.skip_whitespace();
                const int next = in.peek();
                if (next == ',') {
                    in.advance();
                } else if (next == '}') {
                    in.advance();
                    break;
                } else {
                    in.fail("expected `,` or `}` after a tensor's field");
                }
            }
        }
    }
    if (dtype == nullptr) {
        throw FormatError("tensor " + quoted(name) + " has no `dtype`");
    }
    if (!have_shape) {
        throw FormatError("tensor " + quoted(name) + " has no `shape`");
    }
    if (!have_offsets) {
        throw FormatError("tensor " + quoted(name) + " has no `data_offsets`");
    }

    const size_t rank = header.shapes.size() - shape_offset;
    uint64_t bytes = 0;
    switch (tensor_bytes(*dtype, header.shapes.data() + shape_offset, rank, bytes)) {
    case SizeResult::kOk:
        break;
    case SizeResult::kOverflow:
        throw FormatError("the shape of tensor " + quoted(name) + " overflows 64 bits");
    case SizeResult::kNotWholeBytes:
        throw FormatError("tensor " + quoted(name) + " doesn't hold a whole number of bytes");
    }
    if (begin > end) {
        throw FormatError("data_offsets of tensor " + quoted(name) + " end before they begin");
    }
    if (end - begin != bytes) {
        throw FormatError("data_offsets of tensor " + quoted(name) + " span " +
                          std::to_string(end - begin) + " bytes, but its dtype and shape need " +
                          std::to_string(bytes));
    }

    TensorEntry entry;
    entry.name_offset = header.names.size();
    entry.name_size = name.size();
    entry.dtype = dtype;
    entry.shape_offset = shape_offset;
    entry.rank = rank;
    entry.begin = begin;
    entry.end = end;
    header.names.append(name);
    header.names.push_back('\0');
    header.tensors.push_back(entry);
}

// The value of "__metadata__": null, or an object whose values are all strings.
void parse_metadata(Cursor& in, Header& header) {
    if (in.peek() == 'n') {
        in.skip_value(1);
        return;
    }
    if (in.peek() != '{') {
        throw FormatError("`__metadata__` must be an object of strings, or null");
    }
    in.advance();
    in.skip_whitespace();
    if (in.peek() == '}') {
        in.advance();
        return;
    }
    std::string key;
    std::string value;
    for (;;) {
        in.skip_whitespace();
        key.clear();
        in.read_string(&key);
        in.skip_whitespace();
        in.expect(':');
        in.skip_whitespace();
        if (in.peek() != '"') {
            throw FormatError("the value of metadata key " + quoted(key) + " isn't a string");
        }
        value.clear();
        in.read_string(&value);
        if (has_nul(key)) {
            throw FormatError("the metadata key " + quoted(key) + " holds a NUL");
        }
        if (has_nul(value)) {
            throw FormatError("the value of metadata key " + quoted(key) + " holds a NUL");
        }
        MetadataEntry entry;
        entry.key_offset = header.metadata_text.size();
        entry.key_size = key.size();
        header.metadata_text.append(key);
        header.metadata_text.push_back('\0');
        entry.value_offset = header.metadata_text.size();
        entry.value_size = value.size();
        header.metadata_text.append(value);
        header.metadata_text.push_back('\0');
        header.metadata.push_back(entry);
        in.skip_whitespace();
        const int next = in.peek();
        if (next == ',') {
            in.advance();
        } else if (next == '}') {
            in.advance();
            return;
        } else {
            in.fail("expected `,` or `}` in `__metadata__`");
        }
    }
}

// Sorted by (begin, end), the tensors must follow one another from byte 0 to the data's end,
// with no hole, overlap or trailing byte. Zero-size tensors fit wherever the cursor is.
void check_tiling(const Header& header, uint64_t data_length) {
    struct Span {
        uint64_t begin;
        uint64_t end;
        size_t index;
    };
    std::vector<Span> spans;
    spans.reserve(header.tensors.size());
    for (size_t i = 0; i < header.tensors.size(); ++i) {
        spans.push_back({header.tensors[i].begin, header.tensors[i].end, i});
    }
    std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) {
        return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
    });
    uint64_t cursor = 0;
    for (const Span& span : spans) {
        if (span.begin != cursor) {
            const std::string name = quote(header.name(span.index));
            if (span.begin > cursor) {
                throw FormatError("data_offsets of tensor " + name + " leave a hole at byte " +
                                  std::to_string(cursor));
            }
            throw FormatError("data_offsets of tensor " + name +
                              " overlap another tensor at byte " + std::to_string(span.begin));
        }
        cursor = span.end;
    }
    if (cursor > data_length) {
        throw FormatError("data_offsets of tensor " + quote(header.name(spans.back().index)) +
                          " end at byte " + std::to_string(cursor) + ", past the data's " +
                          std::to_string(data_length) + " bytes");
    }
    if (cursor < data_length) {
        throw FormatError("the tensors cover " + std::to_string(cursor) + " of the data's " +
                          std::to_string(data_length) + " bytes: trailing bytes follow them");
    }
}

void append_number(std::string& out, uint64_t value) {
    char digits[24];
    const auto result = std::to_chars(digits, digits + sizeof digits, value);
    out.append(digits, static_cast<size_t>(result.ptr - digits));
}

}  // namespace

const Dtype* find_dtype(std::string_view name) {
    for (const Dtype& dtype : kDtypes) {
        if (name == dtype.name) {
            return &dtype;
        }
    }
    return nullptr;
}

std::string quote(std::string_view name) {
    // Cut first, so no character is split; then made printable, so a NUL, a control character or
    // a byte that isn't UTF-8 can't end or garble the message. A quote or a backslash gets a
    // backslash, as in JSON (they are ASCII, so they never sit inside a longer character).
    const std::string cut = json::clip(std::string(name), 128);
    std::string text = "`";
    size_t run = 0;
    for (size_t i = 0; i < cut.size(); ++i) {
        if (cut[i] == '"' || cut[i] == '\\') {
            json::printable(text, cut.data() + run, i - run);
            text.push_back('\\');
            text.push_back(cut[i]);
            run = i + 1;
        }
    }
    json::printable(text, cut.data() + run, cut.size() - run);
    text.push_back('`');
    return text;
}

SizeResult tensor_bytes(const Dtype& dtype, const size_t* shape, size_t rank, uint64_t& bytes) {
    uint64_t elements = 1;
    for (size_t i = 0; i < rank; ++i) {
        const uint64_t extent = shape[i];
        if (extent != 0 && elements > UINT64_MAX / extent) {
            return SizeResult::kOverflow;
        }
        elements *= extent;
    }
    if (elements > UINT64_MAX / dtype.bits) {
        return SizeResult::kOverflow;
    }
    const uint64_t bits = elements * dtype.bits;
    if (bits % 8 != 0) {
        return SizeResult::kNotWholeBytes;
    }
    bytes = bits / 8;
    return SizeResult::kOk;
}

void parse_header(const char* text, size_t size, uint64_t data_length, Header& header) {
    if (!json::valid_utf8(text, size)) {
        throw FormatError("the header is not valid UTF-8");
    }
    Cursor in(text, text + size);
    std::string key;
    std::string scratch;
    bool have_metadata = false;

    in.skip_whitespace();
    in.expect('{');
    in.skip_whitespace();
    if (in.peek() == '}') {
        in.advance();
    } else {
        for (;;) {
            in.skip_whitespace();
            key.clear();
            in.read_string(&key);
            in.skip_whitespace();
            in.expect(':');
            in.skip_whitespace();
            if (key == kMetadataKey) {
                if (have_metadata) {
                    throw FormatError("the header has `__metadata__` twice");
                }
                have_metadata = true;
                parse_metadata(in, header);
            } else {
                parse_tensor(in, key, header, scratch);
            }
            in.skip_whitespace();
            const int next = in.peek();
            if (next == ',') {
                in.advance();
            } else if (next == '}') {
                in.advance();
                break;
            } else {
                in.fail("expected `,` or `}` in the header's object");
            }
        }
    }
    in.skip_whitespace();
    if (!in.at_end()) {
        in.fail("data after the header's object");
    }

    check_tiling(header, data_length);

    // The names are final now, so the tables can point into them.
    header.tensor_by_name.reserve(header.tensors.size());
    for (size_t i = 0; i < header.tensors.size(); ++i) {
        const TensorEntry& entry = header.tensors[i];
        const std::string_view name(header.names.data() + entry.name_offset, entry.name_size);
        if (!header.tensor_by_name.emplace(name, i).second) {
            throw FormatError("the header has the tensor name " + quote(name) + " twice");
        }
    }
    header.metadata_by_key.reserve(header.metadata.size());
    for (size_t i = 0; i < header.metadata.size(); ++i) {
        const MetadataEntry& entry = header.metadata[i];
        const std::string_view key_view(header.metadata_text.data() + entry.key_offset,
                                        entry.key_size);
        if (!header.metadata_by_key.emplace(key_view, i).second) {
            throw FormatError("the metadata has the key " + quote(key_view) + " twice");
        }
    }
}

void build_header(const Declaration* tensors, size_t tensor_count, const char* const* keys,
                  const char* const* values, size_t metadata_count, Layout& layout) {
    const auto too_long = [] {
        return ArgumentError("the header is longer than " + std::to_string(kMaxHeaderBytes) +
                             " bytes");
    };
    std::string& header = layout.header;
    header.clear();
    layout.placements.clear();
    layout.placements.reserve(tensor_count);
    layout.data_length = 0;
    header.push_back('{');

    if (metadata_count > 0) {
        std::unordered_set<std::string_view> seen;
        seen.reserve(metadata_count);
        header += "\"__metadata__\":{";
        for (size_t i = 0; i < metadata_count; ++i) {
            if (keys[i] == nullptr) {
                throw ArgumentError("keys[" + std::to_string(i) + "] is null");
            }
            if (values[i] == nullptr) {
                throw ArgumentError("values[" + std::to_string(i) + "] is null");
            }
            const size_t key_size = std::strlen(keys[i]);
            const size_t value_size = std::strlen(values[i]);
            if (!json::valid_utf8(keys[i], key_size)) {
                throw ArgumentError("keys[" + std::to_string(i) + "] is not valid UTF-8");
            }
            if (!json::valid_utf8(values[i], value_size)) {
                throw ArgumentError("values[" + std::to_string(i) + "] is not valid UTF-8");
            }
            if (!seen.emplace(keys[i], key_size).second) {
                throw ArgumentError("the metadata key " + quote(keys[i]) + " is given twice");
            }
            if (i > 0) {
                header.push_back(',');
            }
            header.push_back('"');
            json::escape(header, keys[i], key_size);
            header += "\":\"";
            json::escape(header, values[i], value_size);
            header.push_back('"');
            if (header.size() > kMaxHeaderBytes) {
                throw too_long();
            }
        }
        header.push_back('}');
        if (tensor_count > 0) {
            header.push_back(',');
        }
    }

    std::unordered_set<std::string_view> names;
    names.reserve(tensor_count);
    uint64_t offset = 0;
    for (size_t i = 0; i < tensor_count; ++i) {
        const Declaration& tensor = tensors[i];
        const auto where = [i] { return "tensors[" + std::to_string(i) + "]"; };
        if (tensor.name == nullptr) {
            throw ArgumentError(where() + ".name is null");
        }
        if (tensor.dtype == nullptr) {
            throw ArgumentError(where() + ".dtype is null");
        }
        if (tensor.rank > 0 && tensor.shape == nullptr) {
            throw ArgumentError(where() + ".shape is null, with a rank above 0");
        }
        const size_t name_size = std::strlen(tensor.name);
        if (!json::valid_utf8(tensor.name, name_size)) {
            throw ArgumentError(where() + ".name is not valid UTF-8");
        }
        if (std::string_view(tensor.name, name_size) == kMetadataKey) {
            throw ArgumentError(where() + ".name is `__metadata__`, which is reserved");
        }
        if (!names.emplace(tensor.name, name_size).second) {
            throw ArgumentError("the tensor name " + quote(tensor.name) + " is given twice");
        }
        const Dtype* dtype = find_dtype(tensor.dtype);
        if (dtype == nullptr) {
            const std::string dtype_text = tensor.dtype;
            throw ArgumentError(
                "the dtype of tensor " + quote(tensor.name) + " is " +
                (json::valid_utf8(dtype_text.data(), dtype_text.size()) ? quote(dtype_text)
                                                                          : "not valid UTF-8") +
                ", which is none of the 22 dtypes");
        }
        uint64_t bytes = 0;
        switch (tensor_bytes(*dtype, tensor.shape, tensor.rank, bytes)) {
        case SizeResult::kOk:
            break;
        case SizeResult::kOverflow:
            throw ArgumentError("the shape of tensor " + quote(tensor.name) + " overflows 64 bits");
        case SizeResult::kNotWholeBytes:
            throw ArgumentError("tensor " + quote(tensor.name) +
                                " doesn't hold a whole number of bytes");
        }
        if (bytes > UINT64_MAX - offset) {
            throw ArgumentError("the tensors' total size overflows 64 bits");
        }

        if (i > 0) {
            header.push_back(',');
        }
        header.push_back('"');
        json::escape(header, tensor.name, name_size);
        header += "\":{\"dtype\":\"";
        header += dtype->name;
        header += "\",\"shape\":[";
        for (size_t d = 0; d < tensor.rank; ++d) {
            if (d > 0) {
                header.push_back(',');
            }
            append_number(header, tensor.shape[d]);
        }
        header += "],\"data_offsets\":[";
        append_number(header, offset);
        header.push_back(',');
        append_number(header, offset + bytes);
        header += "]}";
        if (header.size() > kMaxHeaderBytes) {
            throw too_long();
        }

        layout.placements.push_back({dtype->bits, offset, bytes});
        offset += bytes;
    }
    header.push_back('}');
    while (header.size() % 8 != 0) {
        header.push_back(' ');
    }
    if (header.size() > kMaxHeaderBytes) {
        throw too_long();
    }
    layout.data_length = offset;
}

}  // namespace flint::safetensors
