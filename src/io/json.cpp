#include "io/json.h"

#include <cstring>

namespace flint::json {

namespace {

bool is_continuation(unsigned char byte) { return (byte & 0xC0) == 0x80; }

bool is_digit(int c) { return c >= '0' && c <= '9'; }

// 2^1024 - 2^970: the midpoint between FP64's largest finite value and 2^1024, as 309 decimal
// digits. A number at or above it rounds to infinity (a tie goes to the even neighbor, which is
// 2^1024).
constexpr char kOverflowPoint[] =
    "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977"
    "587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845"
    "817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559"
    "699508093042880177904174497792";
constexpr size_t kOverflowDigits = sizeof(kOverflowPoint) - 1;
static_assert(kOverflowDigits == 309, "2^1024 - 2^970 has 309 digits");

// Exponents saturate here, far past any digit count a header can hold, so they never overflow.
constexpr int64_t kExponentCap = int64_t{1} << 40;

// How many bytes the UTF-8 sequence at p takes, or 0 if it isn't valid (valid_utf8's rule).
size_t sequence_length(const unsigned char* p, size_t remaining) {
    const unsigned char lead = p[0];
    if (lead < 0x80) {
        return 1;
    }
    if (lead < 0xC2 || lead > 0xF4) {
        return 0;
    }
    const size_t length = lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
    if (remaining < length) {
        return 0;
    }
    unsigned char low = 0x80;
    unsigned char high = 0xBF;
    if (lead == 0xE0) {
        low = 0xA0;
    } else if (lead == 0xED) {
        high = 0x9F;
    } else if (lead == 0xF0) {
        low = 0x90;
    } else if (lead == 0xF4) {
        high = 0x8F;
    }
    if (length > 1 && (p[1] < low || p[1] > high)) {
        return 0;
    }
    for (size_t i = 2; i < length; ++i) {
        if (!is_continuation(p[i])) {
            return 0;
        }
    }
    return length;
}

void append_utf8(std::string& out, uint32_t code_point) {
    if (code_point < 0x80) {
        out.push_back(static_cast<char>(code_point));
    } else if (code_point < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else if (code_point < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    }
}

}  // namespace

bool valid_utf8(const char* data, size_t size) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(data);
    const unsigned char* const end = p + size;
    while (p < end) {
        // Plain ASCII, eight bytes at a time.
        while (static_cast<size_t>(end - p) >= 8) {
            uint64_t word;
            std::memcpy(&word, p, 8);
            if ((word & 0x8080808080808080ull) != 0) {
                break;
            }
            p += 8;
        }
        if (p == end) {
            break;
        }
        const unsigned char lead = *p;
        if (lead < 0x80) {
            ++p;
            continue;
        }
        const size_t remaining = static_cast<size_t>(end - p);
        if (lead < 0xC2) {
            return false;  // a stray continuation byte, or an overlong C0 or C1
        }
        if (lead < 0xE0) {
            if (remaining < 2 || !is_continuation(p[1])) {
                return false;
            }
            p += 2;
        } else if (lead < 0xF0) {
            if (remaining < 3) {
                return false;
            }
            unsigned char low = 0x80;
            unsigned char high = 0xBF;
            if (lead == 0xE0) {
                low = 0xA0;  // below it is an overlong form
            } else if (lead == 0xED) {
                high = 0x9F;  // above it is a UTF-16 surrogate
            }
            if (p[1] < low || p[1] > high || !is_continuation(p[2])) {
                return false;
            }
            p += 3;
        } else if (lead < 0xF5) {
            if (remaining < 4) {
                return false;
            }
            unsigned char low = 0x80;
            unsigned char high = 0xBF;
            if (lead == 0xF0) {
                low = 0x90;  // below it is an overlong form
            } else if (lead == 0xF4) {
                high = 0x8F;  // above it is past U+10FFFF
            }
            if (p[1] < low || p[1] > high || !is_continuation(p[2]) || !is_continuation(p[3])) {
                return false;
            }
            p += 4;
        } else {
            return false;  // F5 to FF lead past U+10FFFF
        }
    }
    return true;
}

std::string clip(const std::string& text, size_t limit) {
    if (text.size() <= limit) {
        return text;
    }
    size_t cut = limit;
    while (cut > 0 && is_continuation(static_cast<unsigned char>(text[cut]))) {
        --cut;
    }
    return text.substr(0, cut);
}

void printable(std::string& out, const char* text, size_t size) {
    static const char kHex[] = "0123456789ABCDEF";
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(text);
    size_t run = 0;  // the start of the bytes that stay as they are
    size_t i = 0;
    while (i < size) {
        const unsigned char c = bytes[i];
        if (c >= 0x80) {
            const size_t length = sequence_length(bytes + i, size - i);
            if (length != 0) {
                i += length;
                continue;
            }
            out.append(text + run, i - run);
            out += "\\x";
            out.push_back(kHex[c >> 4]);
            out.push_back(kHex[c & 15]);
            ++i;
            run = i;
            continue;
        }
        if (c >= 0x20) {
            ++i;
            continue;
        }
        // A control character: escape() writes it.
        out.append(text + run, i - run);
        escape(out, text + i, 1);
        ++i;
        run = i;
    }
    out.append(text + run, size - run);
}

void escape(std::string& out, const char* text, size_t size) {
    static const char kHex[] = "0123456789abcdef";
    size_t run = 0;  // the start of the bytes that stay as they are
    for (size_t i = 0; i < size; ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c >= 0x20 && c != '"' && c != '\\') {
            continue;
        }
        out.append(text + run, i - run);
        run = i + 1;
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case 0x08:
            out += "\\b";
            break;
        case 0x0C:
            out += "\\f";
            break;
        case 0x0A:
            out += "\\n";
            break;
        case 0x0D:
            out += "\\r";
            break;
        case 0x09:
            out += "\\t";
            break;
        default:
            out += "\\u00";
            out.push_back(kHex[c >> 4]);
            out.push_back(kHex[c & 15]);
            break;
        }
    }
    out.append(text + run, size - run);
}

Cursor::Cursor(const char* begin, const char* end) : begin_(begin), current_(begin), end_(end) {}

void Cursor::fail(const char* what) const {
    throw FormatError("invalid JSON at header byte " + std::to_string(position()) + ": " + what);
}

void Cursor::skip_whitespace() {
    while (current_ != end_ &&
           (*current_ == ' ' || *current_ == '\t' || *current_ == '\n' || *current_ == '\r')) {
        ++current_;
    }
}

void Cursor::expect(char c) {
    if (current_ == end_ || *current_ != c) {
        const std::string what = std::string("expected `") + c + "`";
        fail(what.c_str());
    }
    ++current_;
}

unsigned Cursor::read_hex4() {
    unsigned value = 0;
    for (int i = 0; i < 4; ++i) {
        if (current_ == end_) {
            fail("a \\u escape needs 4 hex digits");
        }
        const char c = *current_;
        unsigned digit;
        if (c >= '0' && c <= '9') {
            digit = static_cast<unsigned>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            digit = static_cast<unsigned>(c - 'a') + 10;
        } else if (c >= 'A' && c <= 'F') {
            digit = static_cast<unsigned>(c - 'A') + 10;
        } else {
            fail("a \\u escape needs 4 hex digits");
        }
        value = value * 16 + digit;
        ++current_;
    }
    return value;
}

void Cursor::read_string(std::string* out) {
    expect('"');
    for (;;) {
        // Raw bytes: anything from 0x20 up except the quote and the backslash. The whole header
        // is valid UTF-8 already, so they are whole characters.
        const char* run = current_;
        while (current_ != end_ && static_cast<unsigned char>(*current_) >= 0x20 &&
               *current_ != '"' && *current_ != '\\') {
            ++current_;
        }
        if (out != nullptr) {
            out->append(run, static_cast<size_t>(current_ - run));
        }
        if (current_ == end_) {
            fail("the string never ends");
        }
        const char c = *current_;
        if (c == '"') {
            ++current_;
            return;
        }
        if (c != '\\') {
            fail("a control character in a string");
        }
        ++current_;  // the backslash
        if (current_ == end_) {
            fail("the string never ends");
        }
        const char escape_char = *current_++;
        char decoded = 0;
        switch (escape_char) {
        case '"':
        case '\\':
        case '/':
            decoded = escape_char;
            break;
        case 'b':
            decoded = '\b';
            break;
        case 'f':
            decoded = '\f';
            break;
        case 'n':
            decoded = '\n';
            break;
        case 'r':
            decoded = '\r';
            break;
        case 't':
            decoded = '\t';
            break;
        case 'u': {
            uint32_t code_point = read_hex4();
            if (code_point >= 0xD800 && code_point <= 0xDBFF) {
                // A high surrogate must be followed at once by a low one.
                if (end_ - current_ < 2 || current_[0] != '\\' || current_[1] != 'u') {
                    fail("a high surrogate without a low one");
                }
                current_ += 2;
                const uint32_t low = read_hex4();
                if (low < 0xDC00 || low > 0xDFFF) {
                    fail("a high surrogate without a low one");
                }
                code_point = 0x10000 + ((code_point - 0xD800) << 10) + (low - 0xDC00);
            } else if (code_point >= 0xDC00 && code_point <= 0xDFFF) {
                fail("a low surrogate without a high one");
            }
            if (out != nullptr) {
                append_utf8(*out, code_point);
            }
            continue;
        }
        default:
            fail("an unknown escape");
        }
        if (out != nullptr) {
            out->push_back(decoded);
        }
    }
}

uint64_t Cursor::read_unsigned() {
    int c = peek();
    if (!is_digit(c)) {
        fail("expected an unsigned integer");
    }
    uint64_t value = 0;
    if (c == '0') {
        ++current_;
    } else {
        while (is_digit(c = peek())) {
            const uint64_t digit = static_cast<uint64_t>(c - '0');
            if (value > (UINT64_MAX - digit) / 10) {
                fail("an integer past 2^64 - 1");
            }
            value = value * 10 + digit;
            ++current_;
        }
    }
    c = peek();
    if (is_digit(c) || c == '.' || c == 'e' || c == 'E') {
        fail("expected an unsigned integer, not a float or a number with a leading zero");
    }
    return value;
}

void Cursor::skip_literal(const char* word) {
    for (const char* w = word; *w != '\0'; ++w) {
        if (current_ == end_ || *current_ != *w) {
            fail("a misspelled literal");
        }
        ++current_;
    }
}

// Checks -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)? and that the value doesn't round to
// infinity in FP64, exactly: the value is 0.d1d2d3... x 10^q with d1 the first nonzero digit,
// and only the digits and q decide.
void Cursor::skip_number() {
    if (peek() == '-') {
        ++current_;
    }
    char digits[kOverflowDigits];  // the first 309 digits from d1 on
    size_t significant = 0;        // how many digits there are from d1 on
    int64_t leading_zeros = 0;     // zeros before d1, in the integer and fraction parts together
    int64_t integer_digits = 0;
    auto take = [&](int c) {
        if (significant == 0 && c == '0') {
            ++leading_zeros;
        } else {
            if (significant < kOverflowDigits) {
                digits[significant] = static_cast<char>(c);
            }
            ++significant;
        }
    };

    int c = peek();
    if (c == '0') {
        ++current_;
        ++integer_digits;
        take(c);
    } else if (is_digit(c)) {
        while (is_digit(c = peek())) {
            ++current_;
            ++integer_digits;
            take(c);
        }
    } else {
        fail("a number has no digits");
    }
    if (peek() == '.') {
        ++current_;
        if (!is_digit(peek())) {
            fail("a number needs a digit after its point");
        }
        while (is_digit(c = peek())) {
            ++current_;
            take(c);
        }
    }
    int64_t exponent = 0;
    if (peek() == 'e' || peek() == 'E') {
        ++current_;
        bool negative = false;
        if (peek() == '+' || peek() == '-') {
            negative = peek() == '-';
            ++current_;
        }
        if (!is_digit(peek())) {
            fail("a number needs digits in its exponent");
        }
        while (is_digit(c = peek())) {
            ++current_;
            exponent = exponent * 10 + (c - '0');
            if (exponent > kExponentCap) {
                exponent = kExponentCap;
            }
        }
        if (negative) {
            exponent = -exponent;
        }
    }

    if (significant == 0) {
        return;  // zero, whatever its exponent
    }
    const int64_t q = integer_digits - leading_zeros + exponent;
    if (q < 309) {
        return;
    }
    if (q > 309) {
        fail("number out of range");
    }
    // q == 309: out of range if the digits, padded with zeros, are at least the overflow point's.
    for (size_t i = 0; i < kOverflowDigits; ++i) {
        const char digit = i < significant ? digits[i] : '0';
        if (digit != kOverflowPoint[i]) {
            if (digit > kOverflowPoint[i]) {
                fail("number out of range");
            }
            return;
        }
    }
    fail("number out of range");
}

void Cursor::skip_value(int depth) {
    skip_whitespace();
    const int c = peek();
    switch (c) {
    case '{':
    case '[': {
        if (depth + 1 >= kMaxLevel) {
            fail("nested too deeply");
        }
        const bool object = c == '{';
        const char close = object ? '}' : ']';
        ++current_;
        skip_whitespace();
        if (peek() == close) {
            ++current_;
            return;
        }
        for (;;) {
            if (object) {
                skip_whitespace();
                read_string(nullptr);
                skip_whitespace();
                expect(':');
            }
            skip_value(depth + 1);
            skip_whitespace();
            const int next = peek();
            if (next == ',') {
                ++current_;
            } else if (next == close) {
                ++current_;
                return;
            } else {
                fail(object ? "expected `,` or `}`" : "expected `,` or `]`");
            }
        }
    }
    case '"':
        read_string(nullptr);
        return;
    case 't':
        skip_literal("true");
        return;
    case 'f':
        skip_literal("false");
        return;
    case 'n':
        skip_literal("null");
        return;
    default:
        if (c == '-' || is_digit(c)) {
            skip_number();
            return;
        }
        fail("expected a value");
    }
}

}  // namespace flint::json
