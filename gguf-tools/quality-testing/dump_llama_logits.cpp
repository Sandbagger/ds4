#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// Output format (laguna-resident-capture-v2):
//
//   OUT/capture.json
//   OUT/tokenizer.chat_template.jinja
//   OUT/chat-template-think.prompt
//   OUT/chat-template-nothink.prompt
//   OUT/<case-id>.prompt
//   OUT/<case-id>.tokens.i32
//   OUT/<case-id>.logits.f32
//
// .prompt is the exact byte sequence passed to llama_tokenize. Token IDs are
// signed little-endian int32 values. Logits are IEEE-754 little-endian float32
// values in vocabulary-ID order. The yarn continuation case also contains an
// eight-ID .continuation.i32 file and eight .step-NN.logits.f32 files, one full
// vocabulary row for each greedy, teacher-forced step. capture.json names all
// files and records their shapes through token_count, vocab_size, and the fixed
// continuation length. The chat-template file is the exact byte sequence
// exposed by llama_model_chat_template. The two chat-template prompt files
// record the explicit Laguna DS4 think/no-think bytes reconciled against the
// pinned template's generation branch. The pinned Poolside C API has no Jinja
// evaluator, so capture-v2 labels this as source semantics, not runtime render.

[[noreturn]] static void fail(const std::string &message) {
    throw std::runtime_error(message);
}

static std::string read_file(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path.string());
    std::ostringstream contents;
    contents << in.rdbuf();
    if (in.bad()) fail("cannot read " + path.string());
    return contents.str();
}

static void write_file(const fs::path &path, const std::string &contents) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) fail("cannot open " + path.string() + " for writing");
    out.write(contents.data(), (std::streamsize)contents.size());
    if (!out) fail("cannot write " + path.string());
}

struct Json {
    enum class Kind { null_value, boolean, number, string, array, object };
    Kind kind = Kind::null_value;
    bool boolean = false;
    int64_t number = 0;
    std::string string;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;
};

class JsonParser {
public:
    explicit JsonParser(const std::string &input) : input_(input) {}

    Json parse() {
        Json result = parse_value();
        skip_space();
        if (position_ != input_.size()) error("trailing data");
        return result;
    }

private:
    const std::string &input_;
    size_t position_ = 0;

    [[noreturn]] void error(const char *message) const {
        fail("cases JSON at byte " + std::to_string(position_) + ": " + message);
    }

    void skip_space() {
        while (position_ < input_.size() &&
               std::isspace((unsigned char)input_[position_])) {
            position_++;
        }
    }

    char take() {
        if (position_ == input_.size()) error("unexpected end of input");
        return input_[position_++];
    }

    bool consume(char expected) {
        skip_space();
        if (position_ < input_.size() && input_[position_] == expected) {
            position_++;
            return true;
        }
        return false;
    }

    void consume_word(const char *word) {
        while (*word) {
            if (take() != *word++) error("invalid literal");
        }
    }

    static void append_utf8(std::string &out, uint32_t cp) {
        if (cp <= 0x7f) {
            out.push_back((char)cp);
        } else if (cp <= 0x7ff) {
            out.push_back((char)(0xc0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3f)));
        } else if (cp <= 0xffff) {
            out.push_back((char)(0xe0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back((char)(0x80 | (cp & 0x3f)));
        } else {
            out.push_back((char)(0xf0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3f)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back((char)(0x80 | (cp & 0x3f)));
        }
    }

    uint32_t parse_hex4() {
        uint32_t value = 0;
        for (int i = 0; i < 4; i++) {
            char c = take();
            value <<= 4;
            if (c >= '0' && c <= '9') value |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') value |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= (uint32_t)(c - 'A' + 10);
            else error("invalid unicode escape");
        }
        return value;
    }

    std::string parse_string() {
        skip_space();
        if (take() != '"') error("expected string");
        std::string result;
        for (;;) {
            char c = take();
            if (c == '"') return result;
            if ((unsigned char)c < 0x20) error("control character in string");
            if (c != '\\') {
                result.push_back(c);
                continue;
            }
            char escaped = take();
            switch (escaped) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = parse_hex4();
                    if (cp >= 0xd800 && cp <= 0xdbff) {
                        if (take() != '\\' || take() != 'u') {
                            error("missing low unicode surrogate");
                        }
                        uint32_t low = parse_hex4();
                        if (low < 0xdc00 || low > 0xdfff) {
                            error("invalid low unicode surrogate");
                        }
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                    } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                        error("unexpected low unicode surrogate");
                    }
                    append_utf8(result, cp);
                    break;
                }
                default: error("invalid string escape");
            }
        }
    }

    Json parse_number() {
        skip_space();
        size_t start = position_;
        if (position_ < input_.size() && input_[position_] == '-') position_++;
        if (position_ == input_.size() || !std::isdigit((unsigned char)input_[position_])) {
            error("invalid number");
        }
        if (input_[position_] == '0') {
            position_++;
        } else {
            while (position_ < input_.size() &&
                   std::isdigit((unsigned char)input_[position_])) position_++;
        }
        if (position_ < input_.size() &&
            (input_[position_] == '.' || input_[position_] == 'e' ||
             input_[position_] == 'E')) {
            error("only integer numbers are supported");
        }
        std::string text = input_.substr(start, position_ - start);
        size_t used = 0;
        int64_t number;
        try {
            number = std::stoll(text, &used, 10);
        } catch (const std::exception &) {
            error("integer out of range");
        }
        if (used != text.size()) error("invalid integer");
        Json result;
        result.kind = Json::Kind::number;
        result.number = number;
        return result;
    }

    Json parse_array() {
        Json result;
        result.kind = Json::Kind::array;
        if (consume(']')) return result;
        for (;;) {
            result.array.push_back(parse_value());
            if (consume(']')) return result;
            if (!consume(',')) error("expected ',' or ']'");
        }
    }

    Json parse_object() {
        Json result;
        result.kind = Json::Kind::object;
        if (consume('}')) return result;
        for (;;) {
            std::string key = parse_string();
            for (const auto &member : result.object) {
                if (member.first == key) error("duplicate object key");
            }
            if (!consume(':')) error("expected ':'");
            result.object.emplace_back(std::move(key), parse_value());
            if (consume('}')) return result;
            if (!consume(',')) error("expected ',' or '}'");
        }
    }

    Json parse_value() {
        skip_space();
        if (position_ == input_.size()) error("expected value");
        char c = input_[position_];
        if (c == '"') {
            Json result;
            result.kind = Json::Kind::string;
            result.string = parse_string();
            return result;
        }
        if (c == '{') {
            position_++;
            return parse_object();
        }
        if (c == '[') {
            position_++;
            return parse_array();
        }
        if (c == '-' || std::isdigit((unsigned char)c)) return parse_number();
        Json result;
        if (c == 't') {
            consume_word("true");
            result.kind = Json::Kind::boolean;
            result.boolean = true;
            return result;
        }
        if (c == 'f') {
            consume_word("false");
            result.kind = Json::Kind::boolean;
            return result;
        }
        if (c == 'n') {
            consume_word("null");
            return result;
        }
        error("invalid value");
    }
};

static const Json &member(const Json &object, const char *key) {
    if (object.kind != Json::Kind::object) fail("expected JSON object");
    for (const auto &item : object.object) {
        if (item.first == key) return item.second;
    }
    fail(std::string("missing cases JSON key: ") + key);
}

static const Json *optional_member(const Json &object, const char *key) {
    if (object.kind != Json::Kind::object) fail("expected JSON object");
    for (const auto &item : object.object) {
        if (item.first == key) return &item.second;
    }
    return nullptr;
}

static void require_only_keys(
        const Json &object,
        const std::vector<std::string> &allowed,
        const char *name) {
    if (object.kind != Json::Kind::object) fail(std::string(name) + " must be an object");
    for (const auto &item : object.object) {
        if (std::find(allowed.begin(), allowed.end(), item.first) == allowed.end()) {
            fail(std::string(name) + " has unknown key: " + item.first);
        }
    }
}

static std::string json_string(const Json &value, const char *name) {
    if (value.kind != Json::Kind::string) fail(std::string(name) + " must be a string");
    return value.string;
}

static int json_positive_int(const Json &value, const char *name) {
    if (value.kind != Json::Kind::number || value.number <= 0 ||
        value.number > std::numeric_limits<int>::max()) {
        fail(std::string(name) + " must be a positive integer");
    }
    return (int)value.number;
}

struct Case {
    std::string id;
    std::string render;
    std::string prompt_name;
    fs::path prompt;
    int frontier = 0;
    int context = 0;
};

struct CasesFile {
    int vocab_size = 0;
    std::string continuation_case;
    int continuation_tokens = 0;
    std::vector<Case> cases;
};

static bool safe_case_id(const std::string &id) {
    if (id.empty() || id == "." || id == "..") return false;
    return std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '-' || c == '_';
    });
}

static CasesFile parse_cases(const fs::path &path) {
    Json root = JsonParser(read_file(path)).parse();
    require_only_keys(root,
        {"schema", "vocab_size", "continuation_case", "continuation_tokens", "cases"},
        "cases root");
    if (json_string(member(root, "schema"), "schema") != "laguna-resident-oracle-v1") {
        fail("unsupported cases schema");
    }
    CasesFile result;
    result.vocab_size = json_positive_int(member(root, "vocab_size"), "vocab_size");
    result.continuation_case =
        json_string(member(root, "continuation_case"), "continuation_case");
    result.continuation_tokens =
        json_positive_int(member(root, "continuation_tokens"), "continuation_tokens");
    if (result.vocab_size != 100352 ||
        result.continuation_case != "yarn-8193" ||
        result.continuation_tokens != 8) {
        fail("cases provenance must pin vocab=100352 and yarn-8193 continuation=8");
    }

    const Json &cases = member(root, "cases");
    if (cases.kind != Json::Kind::array || cases.array.empty()) {
        fail("cases must be a non-empty array");
    }
    fs::path base = fs::absolute(path).parent_path();
    for (const Json &item : cases.array) {
        require_only_keys(item, {"id", "render", "prompt", "frontier", "ctx"},
                          "case");
        Case c;
        c.id = json_string(member(item, "id"), "case id");
        c.render = json_string(member(item, "render"), "case render");
        if (!safe_case_id(c.id)) fail("unsafe case id: " + c.id);
        if (c.render != "raw" && c.render != "laguna-ds4") {
            fail("case " + c.id + " render must be raw or laguna-ds4");
        }
        c.prompt_name = json_string(member(item, "prompt"), "case prompt");
        fs::path relative = c.prompt_name;
        if (relative.empty() || relative.is_absolute()) {
            fail("case " + c.id + " prompt must be a relative path");
        }
        c.prompt = base / relative;
        c.context = json_positive_int(member(item, "ctx"), "case ctx");
        const Json *frontier = optional_member(item, "frontier");
        if (frontier) c.frontier = json_positive_int(*frontier, "case frontier");
        if ((c.render == "raw") != (c.frontier > 0)) {
            fail("case " + c.id + " raw render requires frontier and only raw cases may use it");
        }
        for (const Case &prior : result.cases) {
            if (prior.id == c.id) fail("duplicate case id: " + c.id);
        }
        result.cases.push_back(std::move(c));
    }
    bool continuation_found = false;
    for (const Case &c : result.cases) {
        if (c.id == result.continuation_case) continuation_found = true;
    }
    if (!continuation_found) fail("continuation_case does not name a case");
    struct ExpectedCase {
        const char *id;
        const char *render;
        const char *prompt;
        int frontier;
        int context;
    };
    static const ExpectedCase expected[] = {
        {"short", "laguna-ds4", "short.txt", 0, 1024},
        {"swa-513", "raw", "swa-513.prompt", 513, 1024},
        {"yarn-8193", "raw", "yarn-8193.prompt", 8193, 8202},
        {"deep-32768", "raw", "deep-32768.prompt", 32768, 32768},
    };
    if (result.cases.size() != sizeof(expected) / sizeof(expected[0])) {
        fail("cases must contain the four canonical entries");
    }
    for (size_t i = 0; i < result.cases.size(); i++) {
        const Case &actual = result.cases[i];
        if (actual.id != expected[i].id || actual.render != expected[i].render ||
            actual.prompt_name != expected[i].prompt ||
            actual.frontier != expected[i].frontier ||
            actual.context != expected[i].context) {
            fail("case " + std::to_string(i) + " does not match canonical schema");
        }
    }
    return result;
}

static std::vector<llama_token> tokenize(
        const llama_vocab *vocab,
        const std::string &text) {
    int32_t n = llama_tokenize(vocab, text.data(), (int32_t)text.size(),
                               nullptr, 0, false, true);
    if (n < 0) n = -n;
    if (n == 0) return {};
    std::vector<llama_token> tokens((size_t)n);
    int32_t got = llama_tokenize(vocab, text.data(), (int32_t)text.size(),
                                 tokens.data(), n, false, true);
    if (got < 0) fail("llama_tokenize failed");
    tokens.resize((size_t)got);
    return tokens;
}

static std::string detokenize(
        const llama_vocab *vocab,
        const std::vector<llama_token> &tokens) {
    int32_t n = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(),
                                 nullptr, 0, false, true);
    if (n < 0) n = -n;
    if (n == 0) return {};
    std::vector<char> text((size_t)n);
    int32_t got = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(),
                                   text.data(), n, false, true);
    if (got < 0) fail("llama_detokenize failed");
    return std::string(text.data(), (size_t)got);
}

static std::string render_laguna_ds4_prompt(
        const std::string &prompt,
        bool think = false,
        const std::string &system = "") {
    std::string rendered("\xE3\x80\x88|EOS|\xE3\x80\x89");
    if (!system.empty()) {
        rendered += "<system>" + system + "</system>\n";
    }
    return rendered + "<user>" + prompt + "</user>\n<assistant>" +
           (think ? "<think>" : "</think>");
}

static uint32_t little_u32(uint32_t value) {
    const uint16_t one = 1;
    if (*(const uint8_t *)&one == 1) return value;
    return ((value & 0x000000ffU) << 24) |
           ((value & 0x0000ff00U) << 8) |
           ((value & 0x00ff0000U) >> 8) |
           ((value & 0xff000000U) >> 24);
}

static void write_tokens(const fs::path &path, const std::vector<llama_token> &tokens) {
    static_assert(sizeof(llama_token) == 4, "llama_token must be 32 bits");
    std::vector<uint32_t> encoded(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        encoded[i] = little_u32((uint32_t)(int32_t)tokens[i]);
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) fail("cannot open " + path.string() + " for writing");
    out.write((const char *)encoded.data(),
              (std::streamsize)(encoded.size() * sizeof(uint32_t)));
    if (!out) fail("cannot write " + path.string());
}

static void write_logits(const fs::path &path, const std::vector<float> &logits) {
    static_assert(sizeof(float) == 4, "float must be 32 bits");
    static_assert(std::numeric_limits<float>::is_iec559, "float must be IEEE-754");
    std::vector<uint32_t> encoded(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        uint32_t bits;
        std::memcpy(&bits, &logits[i], sizeof(bits));
        encoded[i] = little_u32(bits);
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) fail("cannot open " + path.string() + " for writing");
    out.write((const char *)encoded.data(),
              (std::streamsize)(encoded.size() * sizeof(uint32_t)));
    if (!out) fail("cannot write " + path.string());
}

static std::string escape_json(const std::string &input) {
    std::string output;
    for (unsigned char c : input) {
        switch (c) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (c < 0x20) {
                    char escaped[7];
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
                    output += escaped;
                } else {
                    output.push_back((char)c);
                }
        }
    }
    return output;
}

class Sha256 {
public:
    Sha256() = default;

    void update(const void *data, size_t size) {
        const uint8_t *bytes = (const uint8_t *)data;
        total_ += size;
        while (size > 0) {
            size_t n = std::min(size, sizeof(buffer_) - buffered_);
            std::memcpy(buffer_ + buffered_, bytes, n);
            buffered_ += n;
            bytes += n;
            size -= n;
            if (buffered_ == sizeof(buffer_)) {
                transform(buffer_);
                buffered_ = 0;
            }
        }
    }

    std::string finish() {
        uint64_t bit_length = total_ * 8;
        uint8_t padding[128] = {0x80};
        size_t padding_size = buffered_ < 56 ? 56 - buffered_ : 120 - buffered_;
        update(padding, padding_size);
        uint8_t length[8];
        for (int i = 0; i < 8; i++) {
            length[7 - i] = (uint8_t)(bit_length >> (i * 8));
        }
        update(length, sizeof(length));
        static const char hex[] = "0123456789abcdef";
        std::string result(64, '0');
        for (int i = 0; i < 8; i++) {
            for (int byte = 0; byte < 4; byte++) {
                uint8_t value = (uint8_t)(state_[i] >> (24 - 8 * byte));
                result[(size_t)i * 8 + (size_t)byte * 2] = hex[value >> 4];
                result[(size_t)i * 8 + (size_t)byte * 2 + 1] = hex[value & 15];
            }
        }
        return result;
    }

private:
    uint32_t state_[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    uint8_t buffer_[64] = {};
    size_t buffered_ = 0;
    uint64_t total_ = 0;

    static uint32_t rotate_right(uint32_t value, int bits) {
        return (value >> bits) | (value << (32 - bits));
    }

    void transform(const uint8_t block[64]) {
        static constexpr uint32_t k[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)block[i * 4] << 24) |
                   ((uint32_t)block[i * 4 + 1] << 16) |
                   ((uint32_t)block[i * 4 + 2] << 8) |
                   (uint32_t)block[i * 4 + 3];
        }
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotate_right(w[i - 15], 7) ^
                          rotate_right(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotate_right(w[i - 2], 17) ^
                          rotate_right(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; i++) {
            uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                          rotate_right(e, 25);
            uint32_t choose = (e & f) ^ (~e & g);
            uint32_t temp1 = h + s1 + choose + k[i] + w[i];
            uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                          rotate_right(a, 22);
            uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = s0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }
};

static std::string sha256_file(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path.string() + " for hashing");
    Sha256 hash;
    char buffer[65536];
    while (in.read(buffer, sizeof(buffer)) || in.gcount() > 0) {
        hash.update(buffer, (size_t)in.gcount());
    }
    if (in.bad()) fail("cannot hash " + path.string());
    return hash.finish();
}

struct ModelGuard {
    llama_model *value = nullptr;
    ~ModelGuard() { if (value) llama_model_free(value); }
};

struct ContextGuard {
    llama_context *value = nullptr;
    ~ContextGuard() { if (value) llama_free(value); }
};

struct BatchGuard {
    llama_batch value;
    explicit BatchGuard(int n_batch) : value(llama_batch_init(n_batch, 0, 1)) {}
    ~BatchGuard() { llama_batch_free(value); }
};

static bool decode_chunk(
        llama_context *ctx,
        llama_batch &batch,
        const llama_token *tokens,
        int n_tokens,
        int position,
        bool logits_last) {
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = logits_last && i == n_tokens - 1;
    }
    return llama_decode(ctx, batch) == 0;
}

static bool decode_tokens(
        llama_context *ctx,
        llama_batch &batch,
        const std::vector<llama_token> &tokens,
        int n_batch) {
    int offset = 0;
    while (offset < (int)tokens.size()) {
        int n = std::min(n_batch, (int)tokens.size() - offset);
        if (!decode_chunk(ctx, batch, tokens.data() + offset, n, offset,
                          offset + n == (int)tokens.size())) {
            return false;
        }
        offset += n;
    }
    return true;
}

static llama_token greedy_token(const float *logits, int n_vocab) {
    llama_token best = 0;
    for (int token = 1; token < n_vocab; token++) {
        if (logits[token] > logits[best]) best = (llama_token)token;
    }
    return best;
}

static std::vector<float> copy_logits(const float *logits, int n_vocab) {
    if (!logits) fail("llama logits unavailable");
    std::vector<float> result(logits, logits + n_vocab);
    for (int token = 0; token < n_vocab; token++) {
        if (!std::isfinite(result[(size_t)token])) {
            fail("non-finite logit at vocabulary ID " + std::to_string(token));
        }
    }
    return result;
}

static size_t materialize_prompts(
        const llama_vocab *vocab,
        const fs::path &seed_path,
        const fs::path &directory) {
    std::vector<llama_token> seed = tokenize(vocab, read_file(seed_path));
    if (seed.size() < 32768) {
        fail("seed prompt has " + std::to_string(seed.size()) +
             " tokens; at least 32768 required");
    }
    fs::create_directories(directory);
    const std::pair<const char *, size_t> prompts[] = {
        {"swa-513.prompt", 513},
        {"yarn-8193.prompt", 8193},
        {"deep-32768.prompt", 32768},
    };
    for (const auto &prompt : prompts) {
        std::vector<llama_token> expected(seed.begin(), seed.begin() + prompt.second);
        std::string text = detokenize(vocab, expected);
        std::vector<llama_token> actual = tokenize(vocab, text);
        if (actual != expected) {
            fail(std::string("token round trip mismatch for ") + prompt.first);
        }
        write_file(directory / prompt.first, text);
    }
    return seed.size();
}

struct DumpResult {
    Case test_case;
    std::string prompt_file;
    std::string tokens_file;
    std::string logits_file;
    int token_count = 0;
    llama_token argmax = 0;
    std::string continuation_tokens_file;
    std::vector<std::string> continuation_logits_files;
    std::vector<llama_token> continuation_argmax;
    std::vector<std::string> files;
};

static DumpResult dump_case(
        llama_model *model,
        const llama_vocab *vocab,
        int n_vocab,
        const Case &test_case,
        const std::string &continuation_case,
        int continuation_tokens,
        const fs::path &out_root) {
    std::string source = read_file(test_case.prompt);
    std::string prompt = test_case.render == "laguna-ds4"
        ? render_laguna_ds4_prompt(source)
        : source;
    std::vector<llama_token> tokens = tokenize(vocab, prompt);
    if (tokens.empty()) fail("case " + test_case.id + " tokenized to zero tokens");
    if (test_case.frontier > 0 && (int)tokens.size() != test_case.frontier) {
        fail("case " + test_case.id + " expected " +
             std::to_string(test_case.frontier) + " tokens, got " +
             std::to_string(tokens.size()));
    }
    const int steps = test_case.id == continuation_case ? continuation_tokens : 0;
    if (tokens.size() + (size_t)steps > (size_t)test_case.context) {
        fail("case " + test_case.id + " exceeds context " +
             std::to_string(test_case.context));
    }

    llama_context_params params = llama_context_default_params();
    params.n_ctx = (uint32_t)test_case.context;
    params.n_batch = (uint32_t)std::min(test_case.context, 2048);
    params.n_ubatch = (uint32_t)std::min<int>((int)params.n_batch, 512);
    params.n_seq_max = 1;
    params.no_perf = true;
    ContextGuard ctx;
    ctx.value = llama_init_from_model(model, params);
    if (!ctx.value) fail("case " + test_case.id + " failed to create context");
    int n_batch = std::min<int>((int)llama_n_batch(ctx.value), 2048);
    BatchGuard batch(n_batch);
    if (!decode_tokens(ctx.value, batch.value, tokens, n_batch)) {
        fail("case " + test_case.id + " prompt decode failed");
    }

    std::vector<float> final_logits =
        copy_logits(llama_get_logits_ith(ctx.value, -1), n_vocab);
    std::vector<llama_token> greedy;
    std::vector<std::vector<float>> continuation_logits;
    if (steps > 0) {
        continuation_logits.reserve((size_t)steps);
        for (int step = 0; step < steps; step++) {
            const float *row = llama_get_logits_ith(ctx.value, -1);
            if (!row) fail("case " + test_case.id + " continuation logits unavailable");
            continuation_logits.push_back(copy_logits(row, n_vocab));
            llama_token token = greedy_token(continuation_logits.back().data(), n_vocab);
            greedy.push_back(token);
            if (!decode_chunk(ctx.value, batch.value, &token, 1,
                              (int)tokens.size() + step, true)) {
                fail("case " + test_case.id + " continuation decode failed at step " +
                     std::to_string(step));
            }
        }
    }

    DumpResult result;
    result.test_case = test_case;
    result.prompt_file = test_case.id + ".prompt";
    result.tokens_file = test_case.id + ".tokens.i32";
    result.logits_file = test_case.id + ".logits.f32";
    result.token_count = (int)tokens.size();
    result.argmax = greedy_token(final_logits.data(), n_vocab);
    result.files = {result.prompt_file, result.tokens_file, result.logits_file};
    write_file(out_root / result.prompt_file, prompt);
    write_tokens(out_root / result.tokens_file, tokens);
    write_logits(out_root / result.logits_file, final_logits);
    if (steps > 0) {
        result.continuation_tokens_file = test_case.id + ".continuation.i32";
        result.continuation_argmax = greedy;
        write_tokens(out_root / result.continuation_tokens_file, greedy);
        result.files.push_back(result.continuation_tokens_file);
        for (int step = 0; step < steps; step++) {
            char suffix[64];
            std::snprintf(suffix, sizeof(suffix), ".step-%02d.logits.f32", step);
            std::string name = test_case.id + suffix;
            write_logits(out_root / name, continuation_logits[(size_t)step]);
            result.continuation_logits_files.push_back(name);
            result.files.push_back(name);
        }
    }
    std::fprintf(stderr, "%s prompt_bytes=%zu tokens=%zu vocab=%d continuation=%d\n",
                 test_case.id.c_str(), prompt.size(), tokens.size(), n_vocab, steps);
    return result;
}

static void write_capture(
        const fs::path &out_root,
        int n_vocab,
        const std::string &continuation_case,
        int continuation_tokens,
        size_t seed_token_count,
        const std::vector<DumpResult> &results,
        const std::string &chat_template,
        const std::string &template_probe) {
    if (seed_token_count < 32768) {
        fail("benchmark seed has fewer than 32768 Poolside tokens");
    }
    if (chat_template.empty()) fail("model tokenizer.chat_template is empty");
    static const std::string template_file = "tokenizer.chat_template.jinja";
    static const std::string think_file = "chat-template-think.prompt";
    static const std::string nothink_file = "chat-template-nothink.prompt";
    static const std::string reconciliation_system =
        "Laguna template reconciliation probe.";
    const std::string think_prompt = render_laguna_ds4_prompt(
        template_probe, true, reconciliation_system);
    const std::string nothink_prompt = render_laguna_ds4_prompt(
        template_probe, false, reconciliation_system);
    write_file(out_root / template_file, chat_template);
    write_file(out_root / think_file, think_prompt);
    write_file(out_root / nothink_file, nothink_prompt);

    const DumpResult *continuation = nullptr;
    std::vector<std::string> files;
    files.push_back(template_file);
    files.push_back(think_file);
    files.push_back(nothink_file);
    for (const DumpResult &result : results) {
        files.insert(files.end(), result.files.begin(), result.files.end());
        if (result.test_case.id == continuation_case) continuation = &result;
    }
    if (!continuation) fail("continuation result missing");

    std::ostringstream capture;
    capture << "{\n"
            << "  \"schema\": \"laguna-resident-capture-v2\",\n"
            << "  \"oracle\": \"llama\",\n"
            << "  \"runtime_commit\": \"04b2b72cb54048ead292884adbe11f284e3ec950\",\n"
            << "  \"seed_token_count\": " << seed_token_count << ",\n"
            << "  \"vocab_size\": " << n_vocab << ",\n"
            << "  \"model\": {\n"
            << "    \"repository\": \"poolside/Laguna-S-2.1-GGUF\",\n"
            << "    \"revision\": \"e2ccc0579fc18e6ea2362fa25fccbcd470f0e332\",\n"
            << "    \"file\": \"laguna-s-2.1-Q4_K_M.gguf\",\n"
            << "    \"size\": 68248760064,\n"
            << "    \"sha256\": \"a34c74e46688122bef83122f4133031bababbefcf57436dde97048c91e2cc6ff\"\n"
            << "  },\n"
            << "  \"chat_template\": {\n"
            << "    \"file\": \"" << template_file << "\",\n"
            << "    \"bytes\": " << chat_template.size() << ",\n"
            << "    \"sha256\": \"" << sha256_file(out_root / template_file) << "\",\n"
            << "    \"render_contract\": \"pinned-template-semantics-v1\",\n"
            << "    \"reconciliation_system\": \""
            << reconciliation_system << "\",\n"
            << "    \"think_prompt_file\": \"" << think_file << "\",\n"
            << "    \"think_prompt_sha256\": \"" << sha256_file(out_root / think_file) << "\",\n"
            << "    \"nothink_prompt_file\": \"" << nothink_file << "\",\n"
            << "    \"nothink_prompt_sha256\": \"" << sha256_file(out_root / nothink_file) << "\"\n"
            << "  },\n"
            << "  \"cases\": [\n";
    for (size_t i = 0; i < results.size(); i++) {
        const DumpResult &result = results[i];
        const Case &c = result.test_case;
        capture << "    {\"id\": \"" << escape_json(c.id)
                << "\", \"render\": \"" << escape_json(c.render)
                << "\", \"prompt\": \"" << escape_json(c.prompt_name)
                << "\", \"frontier\": " << result.token_count
                << ", \"context\": " << c.context
                << ", \"prompt_file\": \"" << escape_json(result.prompt_file)
                << "\", \"tokens_file\": \"" << escape_json(result.tokens_file)
                << "\", \"logits_file\": \"" << escape_json(result.logits_file)
                << "\", \"token_count\": " << result.token_count
                << ", \"argmax\": " << result.argmax << "}"
                << (i + 1 == results.size() ? "\n" : ",\n");
    }
    capture << "  ],\n"
            << "  \"continuation\": {\"case\": \""
            << escape_json(continuation_case) << "\", \"tokens_file\": \""
            << escape_json(continuation->continuation_tokens_file)
            << "\", \"logits_files\": [";
    for (size_t i = 0; i < continuation->continuation_logits_files.size(); i++) {
        if (i) capture << ", ";
        capture << "\"" << escape_json(continuation->continuation_logits_files[i]) << "\"";
    }
    capture << "], \"argmax\": [";
    for (size_t i = 0; i < continuation->continuation_argmax.size(); i++) {
        if (i) capture << ", ";
        capture << continuation->continuation_argmax[i];
    }
    capture << "]},\n"
            << "  \"files\": {\n";
    for (size_t i = 0; i < files.size(); i++) {
        capture << "    \"" << escape_json(files[i]) << "\": \""
                << sha256_file(out_root / files[i]) << "\""
                << (i + 1 == files.size() ? "\n" : ",\n");
    }
    capture << "  }\n}\n";
    if ((int)continuation->continuation_argmax.size() != continuation_tokens) {
        fail("continuation result has wrong token count");
    }
    write_file(out_root / "capture.json", capture.str());
}

static void usage(const char *program) {
    std::fprintf(stderr,
        "usage: %s --model MODEL --cases cases.json "
        "--seed-prompt benchmark-32768.txt --materialize-prompts DIR --out DIR\n"
        "\n"
        "Writes a laguna-resident-capture-v2 capture.json and flat artifacts.\n"
        "Binary token IDs are int32-le; logits are IEEE-754 float32-le in\n"
        "vocab-ID order. Every emitted artifact is SHA-256 listed in capture.json.\n"
        "See the source-file format comment for the complete file layout.\n",
        program);
}

int main(int argc, char **argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" ||
                      std::string(argv[1]) == "-h")) {
        usage(argv[0]);
        return 0;
    }
    if (argc != 11 || std::string(argv[1]) != "--model" ||
        std::string(argv[3]) != "--cases" ||
        std::string(argv[5]) != "--seed-prompt" ||
        std::string(argv[7]) != "--materialize-prompts" ||
        std::string(argv[9]) != "--out") {
        usage(argv[0]);
        return 2;
    }

    bool backend_initialized = false;
    try {
        Sha256 self_test;
        self_test.update("abc", 3);
        if (self_test.finish() !=
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
            fail("internal SHA-256 self-test failed");
        }
        uintmax_t model_size = fs::file_size(argv[2]);
        if (model_size != UINT64_C(68248760064)) {
            fail("model size " + std::to_string(model_size) +
                 " does not match pinned size 68248760064");
        }
        if (sha256_file(argv[2]) !=
            "a34c74e46688122bef83122f4133031bababbefcf57436dde97048c91e2cc6ff") {
            fail("model SHA-256 does not match the pinned artifact");
        }
        fs::path out_root = argv[10];
        fs::create_directories(out_root);
        if (fs::directory_iterator(out_root) != fs::directory_iterator()) {
            fail("output directory must be empty: " + out_root.string());
        }
        ggml_backend_load_all();
        llama_backend_init();
        backend_initialized = true;

        {
            llama_model_params params = llama_model_default_params();
            params.n_gpu_layers = -1;
            params.use_mmap = true;
            ModelGuard model;
            model.value = llama_model_load_from_file(argv[2], params);
            if (!model.value) fail("failed to open model");
            const char *embedded_template =
                llama_model_chat_template(model.value, nullptr);
            if (!embedded_template || embedded_template[0] == '\0') {
                fail("model tokenizer.chat_template is missing or empty");
            }
            const std::string chat_template(embedded_template);
            const llama_vocab *vocab = llama_model_get_vocab(model.value);
            int n_vocab = llama_vocab_n_tokens(vocab);

            CasesFile cases = parse_cases(argv[4]);
            if (n_vocab != cases.vocab_size) {
                fail("model vocab size " + std::to_string(n_vocab) +
                     " does not match cases vocab size " +
                     std::to_string(cases.vocab_size));
            }
            const size_t seed_token_count =
                materialize_prompts(vocab, argv[6], argv[8]);
            std::vector<DumpResult> results;
            for (const Case &test_case : cases.cases) {
                results.push_back(dump_case(
                    model.value, vocab, n_vocab, test_case,
                    cases.continuation_case, cases.continuation_tokens, argv[10]));
            }
            write_capture(argv[10], n_vocab, cases.continuation_case,
                          cases.continuation_tokens, seed_token_count, results,
                          chat_template, read_file(cases.cases[0].prompt));
        }
        llama_backend_free();
        backend_initialized = false;
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "dump_llama_logits: %s\n", error.what());
        if (backend_initialized) llama_backend_free();
        return 1;
    }
}
