// Produce a minimal greedy-behavior oracle for the frozen Laguna token-513 case.
// Build against Poolside llama.cpp 04b2b72 and its pinned shared libraries.

#include "ggml-backend.h"
#include "llama.h"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr int kPrefixTokens = 512;
static constexpr int32_t kResumeToken = 3612;
static constexpr int kMaxSteps = 32;
static constexpr int kWidth = 3072;
static constexpr int kLayers = 48;
static constexpr int kVocab = 100352;

static constexpr std::array<int32_t, 15> kPrefixPattern = {
    785, 10068, 3612, 8473, 1077, 1857, 606, 330,
    2746, 22910, 1059, 4158, 7799, 83, 268,
};

[[noreturn]] static void fail(const std::string &message) {
    throw std::runtime_error(message);
}

struct Options {
    fs::path model;
    fs::path tokens;
    fs::path out;
    int steps = 0;
};

static void usage(FILE *stream, const char *program) {
    std::fprintf(
        stream,
        "Usage: %s --model MODEL --tokens TOKENS.i32 --out DIR --steps 1..32\n",
        program);
}

static int parse_steps(const std::string &value) {
    errno = 0;
    char *end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' ||
        parsed < 1 || parsed > kMaxSteps) {
        fail("--steps must be an integer from 1 through 32");
    }
    return static_cast<int>(parsed);
}

static Options parse_options(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        usage(stdout, argv[0]);
        std::exit(0);
    }

    Options options;
    bool have_model = false;
    bool have_tokens = false;
    bool have_out = false;
    bool have_steps = false;
    for (int index = 1; index < argc; index++) {
        if (index + 1 >= argc) {
            fail(std::string("missing value for ") + argv[index]);
        }
        const std::string flag = argv[index++];
        const std::string value = argv[index];
        if (flag == "--model" && !have_model) {
            options.model = value;
            have_model = true;
        } else if (flag == "--tokens" && !have_tokens) {
            options.tokens = value;
            have_tokens = true;
        } else if (flag == "--out" && !have_out) {
            options.out = value;
            have_out = true;
        } else if (flag == "--steps" && !have_steps) {
            options.steps = parse_steps(value);
            have_steps = true;
        } else {
            fail("unknown or duplicate argument: " + flag);
        }
    }
    if (!have_model || !have_tokens || !have_out || !have_steps) {
        fail("--model, --tokens, --out, and --steps are required");
    }
    return options;
}

static std::vector<llama_token> read_exact_prefix(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail("cannot open token file: " + path.string());
    }
    const std::vector<unsigned char> bytes(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (input.bad()) {
        fail("cannot read token file: " + path.string());
    }
    if (bytes.size() != kPrefixTokens * sizeof(int32_t)) {
        fail("token file must contain exactly 512 little-endian int32 IDs");
    }

    std::vector<llama_token> tokens;
    tokens.reserve(kPrefixTokens);
    for (int index = 0; index < kPrefixTokens; index++) {
        const size_t offset = static_cast<size_t>(index) * sizeof(int32_t);
        const uint32_t bits =
            static_cast<uint32_t>(bytes[offset + 0]) |
            (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
            (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
            (static_cast<uint32_t>(bytes[offset + 3]) << 24);
        int32_t token = 0;
        std::memcpy(&token, &bits, sizeof(token));
        const int32_t expected =
            kPrefixPattern[static_cast<size_t>(index) % kPrefixPattern.size()];
        if (token != expected) {
            fail("token file does not contain the frozen Laguna prefix at index " +
                 std::to_string(index));
        }
        tokens.push_back(static_cast<llama_token>(token));
    }
    return tokens;
}

static void prepare_output_directory(const fs::path &path) {
    std::error_code error;
    if (fs::exists(path, error)) {
        if (error || !fs::is_directory(path)) {
            fail("output path is not a directory: " + path.string());
        }
    } else if (!fs::create_directories(path, error) || error) {
        fail("cannot create output directory: " + path.string());
    }
    if (fs::directory_iterator(path) != fs::directory_iterator()) {
        fail("output directory must be empty: " + path.string());
    }
}

static void write_bytes_checked(
        const fs::path &path,
        const char *bytes,
        std::streamsize count) {
    if (fs::exists(path)) {
        fail("refusing to overwrite behavior file: " + path.string());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        fail("cannot open behavior file for writing: " + path.string());
    }
    output.write(bytes, count);
    if (!output) {
        fail("cannot write behavior file: " + path.string());
    }
}

static void write_f32_little_endian(
        const fs::path &path,
        const float *values,
        int count) {
    std::vector<char> bytes(static_cast<size_t>(count) * sizeof(float));
    for (int index = 0; index < count; index++) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(values[index]), "float32 is required");
        std::memcpy(&bits, &values[index], sizeof(bits));
        const size_t offset = static_cast<size_t>(index) * sizeof(bits);
        bytes[offset + 0] = static_cast<char>((bits >> 0) & 0xff);
        bytes[offset + 1] = static_cast<char>((bits >> 8) & 0xff);
        bytes[offset + 2] = static_cast<char>((bits >> 16) & 0xff);
        bytes[offset + 3] = static_cast<char>((bits >> 24) & 0xff);
    }
    write_bytes_checked(
        path, bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

static void write_i32_little_endian(
        const fs::path &path,
        const std::vector<int32_t> &values) {
    std::vector<char> bytes(values.size() * sizeof(int32_t));
    for (size_t index = 0; index < values.size(); index++) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(values[index]), "int32 is required");
        std::memcpy(&bits, &values[index], sizeof(bits));
        const size_t offset = index * sizeof(bits);
        bytes[offset + 0] = static_cast<char>((bits >> 0) & 0xff);
        bytes[offset + 1] = static_cast<char>((bits >> 8) & 0xff);
        bytes[offset + 2] = static_cast<char>((bits >> 16) & 0xff);
        bytes[offset + 3] = static_cast<char>((bits >> 24) & 0xff);
    }
    write_bytes_checked(
        path, bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

struct ModelGuard {
    llama_model *value = nullptr;
    ~ModelGuard() {
        if (value != nullptr) {
            llama_model_free(value);
        }
    }
};

struct ContextGuard {
    llama_context *value = nullptr;
    ~ContextGuard() {
        if (value != nullptr) {
            llama_free(value);
        }
    }
};

struct BatchGuard {
    llama_batch value;
    BatchGuard() : value(llama_batch_init(kPrefixTokens, 0, 1)) {}
    ~BatchGuard() { llama_batch_free(value); }
};

static void validate_model(const llama_model *model, int n_vocab) {
    if (llama_model_n_embd(model) != kWidth) {
        fail("model embedding width is not 3072");
    }
    if (llama_model_n_layer(model) != kLayers) {
        fail("model layer count is not 48");
    }
    if (n_vocab != kVocab) {
        fail("model vocabulary size is not 100352");
    }
}

static void require_batch_storage(const llama_batch &batch) {
    if (batch.token == nullptr || batch.pos == nullptr ||
        batch.n_seq_id == nullptr || batch.seq_id == nullptr ||
        batch.logits == nullptr) {
        fail("failed to allocate decode batch");
    }
}

static void decode_prefix(
        llama_context *context,
        llama_batch &batch,
        const std::vector<llama_token> &tokens) {
    batch.n_tokens = kPrefixTokens;
    for (int index = 0; index < kPrefixTokens; index++) {
        batch.token[index] = tokens[static_cast<size_t>(index)];
        batch.pos[index] = index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = false;
    }
    const int32_t result = llama_decode(context, batch);
    if (result != 0) {
        fail("prefix llama_decode failed with status " + std::to_string(result));
    }
}

static void decode_one(
        llama_context *context,
        llama_batch &batch,
        llama_token token,
        llama_pos position,
        const std::string &stage) {
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = position;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;
    const int32_t result = llama_decode(context, batch);
    if (result != 0) {
        fail(stage + " llama_decode failed with status " + std::to_string(result));
    }
}

static void decode_resume(llama_context *context, llama_batch &batch) {
    batch.n_tokens = 1;
    batch.token[0] = kResumeToken;
    batch.pos[0] = kPrefixTokens;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;
    const int32_t result = llama_decode(context, batch);
    if (result != 0) {
        fail("resume llama_decode failed with status " + std::to_string(result));
    }
}

static llama_token greedy_argmax(const float *logits, int n_vocab) {
    if (logits == nullptr) {
        fail("llama_get_logits_ith returned null");
    }
    llama_token best_token = 0;
    if (!std::isfinite(logits[best_token])) {
        fail("non-finite logit at token 0");
    }
    for (llama_token token = 1; token < n_vocab; token++) {
        if (!std::isfinite(logits[token])) {
            fail("non-finite logit at token " + std::to_string(token));
        }
        if (logits[token] > logits[best_token]) {
            best_token = token;
        }
    }
    return best_token;
}

static std::vector<int32_t> produce_behavior(
        llama_context *context,
        llama_batch &batch,
        const fs::path &out,
        int n_vocab,
        int steps) {
    std::vector<int32_t> continuation;
    continuation.reserve(static_cast<size_t>(steps));
    for (int step = 0; step < steps; step++) {
        const float *logits = llama_get_logits_ith(context, 0);
        char filename[48];
        std::snprintf(
            filename, sizeof(filename), "behavior-step-%02d.logits.f32", step);
        write_f32_little_endian(out / filename, logits, n_vocab);
        const llama_token next = greedy_argmax(logits, n_vocab);
        continuation.push_back(static_cast<int32_t>(next));
        decode_one(
            context, batch, next,
            static_cast<llama_pos>(kPrefixTokens + 1 + step),
            "continuation step " + std::to_string(step));
    }
    return continuation;
}

int main(int argc, char **argv) {
    bool backend_initialized = false;
    try {
        const Options options = parse_options(argc, argv);
        const std::vector<llama_token> prefix = read_exact_prefix(options.tokens);
        prepare_output_directory(options.out);

        ggml_backend_load_all();
        llama_backend_init();
        backend_initialized = true;

        std::vector<int32_t> continuation;
        {
            llama_model_params model_params = llama_model_default_params();
            model_params.n_gpu_layers = -1;
            model_params.use_mmap = true;
            ModelGuard model;
            model.value = llama_model_load_from_file(options.model.c_str(), model_params);
            if (model.value == nullptr) {
                fail("failed to load model");
            }
            const llama_vocab *vocab = llama_model_get_vocab(model.value);
            if (vocab == nullptr) {
                fail("model has no vocabulary");
            }
            const int n_vocab = llama_vocab_n_tokens(vocab);
            validate_model(model.value, n_vocab);

            llama_context_params context_params = llama_context_default_params();
            context_params.n_ctx = 1024;
            context_params.n_batch = 1024;
            context_params.n_ubatch = 512;
            context_params.n_seq_max = 1;
            context_params.no_perf = true;
            context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
            ContextGuard context;
            context.value = llama_init_from_model(model.value, context_params);
            if (context.value == nullptr) {
                fail("failed to create context");
            }

            BatchGuard batch;
            require_batch_storage(batch.value);
            decode_prefix(context.value, batch.value, prefix);
            decode_resume(context.value, batch.value);

            continuation = produce_behavior(
                context.value, batch.value, options.out, n_vocab, options.steps);
        }

        write_i32_little_endian(
            options.out / "behavior-continuation.i32", continuation);
        llama_backend_free();
        backend_initialized = false;
        std::printf(
            "prefix_tokens=%d\nresume_token=%d\nsteps=%d\n"
            "continuation=behavior-continuation.i32\nout=%s\n",
            kPrefixTokens, kResumeToken, options.steps, options.out.c_str());
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "probe_poolside_laguna_behavior: %s\n", error.what());
        if (backend_initialized) {
            llama_backend_free();
        }
        return 1;
    }
}
