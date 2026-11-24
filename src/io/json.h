// The pieces of JSON that a safetensors header needs, written for it: UTF-8 validation, a cursor
// that reads the grammar of RFC 8259 as serde_json applies it, and the escaping the writer uses.
// Nothing here builds a document tree: the reader takes the few things it wants and skips the rest.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace flint::json {

// The header breaks a rule: a byte that isn't valid UTF-8 or JSON, a value of the wrong kind, a
// number out of range. what() says which and where, without the file's path.
class FormatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Opening a value at this nesting level is an error. The top-level object is level 1, a
// tensor's own object is level 2, and every [ or { opens one more.
constexpr int kMaxLevel = 128;

// Whether the bytes are valid UTF-8 by Rust's rule: no stray or missing continuation bytes, no
// overlong forms, no UTF-16 surrogates, nothing above U+10FFFF.
bool valid_utf8(const char* data, size_t size);

// The first limit bytes of text, or fewer when that would cut a UTF-8 character in two, so a
// message that quotes a name stays valid UTF-8.
std::string clip(const std::string& text, size_t limit);

// Appends text to out so that it is valid UTF-8 and has no control characters, for a message:
// control characters are written as escape() writes them, other valid UTF-8 as it is (quotes
// and backslashes too, so a Windows path stays readable), and each byte that doesn't start a
// valid sequence (by valid_utf8's rule) becomes \xNN, in upper case.
void printable(std::string& out, const char* text, size_t size);

// Appends text's bytes to out, escaped as the body of a JSON string (the quotes are the
// caller's): `"` and `\` get a backslash, the control characters \b \f \n \r \t their short
// forms, other bytes below 0x20 become \u00xx in lower case, and everything else stays raw.
void escape(std::string& out, const char* text, size_t size);

// Reads JSON from the bytes [begin, end), which must stay alive and valid UTF-8 as a whole. The
// functions that read a value start at its first byte, so the caller skips whitespace first. Every
// one throws FormatError on bad input, and none recurses deeper than kMaxLevel.
class Cursor {
public:
    Cursor(const char* begin, const char* end);

    // How many bytes have been consumed, for messages.
    size_t position() const { return static_cast<size_t>(current_ - begin_); }
    bool at_end() const { return current_ == end_; }

    // The next byte, or -1 at the end.
    int peek() const { return current_ == end_ ? -1 : static_cast<unsigned char>(*current_); }
    void advance() { ++current_; }

    // Skips space, \t, \n and \r.
    void skip_whitespace();

    // Consumes the byte c, or throws.
    void expect(char c);

    // Consumes a string, quotes included, and appends its decoded bytes to out (\u0000 gives a
    // NUL byte). A null out only checks the string.
    void read_string(std::string* out);

    // Consumes an unsigned integer written as `0` or `[1-9][0-9]*`, at most 2^64 - 1. `-0`,
    // `8.0`, `8e0` and anything at or past 2^64 are errors, as serde reads them as floats.
    uint64_t read_unsigned();

    // Consumes any value whatsoever, checking all of it: its grammar, its strings, its numbers'
    // range and its nesting. depth is how many arrays and objects enclose it.
    void skip_value(int depth);

    // Throws a FormatError saying what is wrong at the current position.
    [[noreturn]] void fail(const char* what) const;

private:
    void skip_number();
    void skip_literal(const char* word);
    unsigned read_hex4();

    const char* begin_;
    const char* current_;
    const char* end_;
};

}  // namespace flint::json
