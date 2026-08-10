// Capture the exact Poolside Laguna short-prompt residual stream.
// Build against Poolside llama.cpp 04b2b72 and its build-c7-diag output.

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <array>
#include <cerrno>
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

static constexpr int kWidth = 3072;
static constexpr int kTokens = 22;
static constexpr int kLayers = 48;
static constexpr int kVocab = 100352;

static constexpr std::array<int32_t, kTokens> kExpectedTokens = {
    2, 97, 1437, 99, 53225, 3203, 330, 10068, 3612, 31063, 81,
    365, 1161, 15631, 83, 268, 532, 1437, 99, 268, 23, 19,
};

[[noreturn]] static void fail(const std::string &message) {
    throw std::runtime_error(message);
}

struct Options {
    fs::path model;
    fs::path tokens;
    fs::path out;
    enum llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
};

static void usage(FILE *stream, const char *program) {
    std::fprintf(
        stream,
        "Usage: %s --model MODEL --tokens TOKENS.i32 --out DIR "
        "[--flash-attn auto|disabled]\n",
        program);
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
    bool have_flash_attn = false;
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) {
            fail(std::string("missing value for ") + argv[i]);
        }
        const std::string flag = argv[i++];
        const std::string value = argv[i];
        if (flag == "--model" && !have_model) {
            options.model = value;
            have_model = true;
        } else if (flag == "--tokens" && !have_tokens) {
            options.tokens = value;
            have_tokens = true;
        } else if (flag == "--out" && !have_out) {
            options.out = value;
            have_out = true;
        } else if (flag == "--flash-attn" && !have_flash_attn) {
            if (value == "auto") {
                options.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
            } else if (value == "disabled") {
                options.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            } else {
                fail("--flash-attn must be auto or disabled");
            }
            have_flash_attn = true;
        } else {
            fail("unknown or duplicate argument: " + flag);
        }
    }
    if (!have_model || !have_tokens || !have_out) {
        fail("--model, --tokens, and --out are required");
    }
    return options;
}

static std::array<llama_token, kTokens> read_exact_tokens(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail("cannot open token file: " + path.string());
    }
    std::vector<unsigned char> bytes(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (input.bad()) {
        fail("cannot read token file: " + path.string());
    }
    if (bytes.size() != kTokens * sizeof(int32_t)) {
        fail("token file must contain exactly 22 little-endian int32 IDs (88 bytes)");
    }

    std::array<llama_token, kTokens> tokens{};
    for (int index = 0; index < kTokens; index++) {
        const size_t offset = static_cast<size_t>(index) * sizeof(int32_t);
        const uint32_t bits =
            static_cast<uint32_t>(bytes[offset + 0]) |
            (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
            (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
            (static_cast<uint32_t>(bytes[offset + 3]) << 24);
        int32_t token = 0;
        std::memcpy(&token, &bits, sizeof(token));
        if (token != kExpectedTokens[index]) {
            fail("token file does not contain the exact Laguna short prompt at index " +
                 std::to_string(index));
        }
        tokens[index] = static_cast<llama_token>(token);
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

static void write_f32_little_endian(const fs::path &path, const std::vector<float> &values) {
    if (fs::exists(path)) {
        fail("refusing to overwrite diagnostic file: " + path.string());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        fail("cannot open diagnostic file for writing: " + path.string());
    }
    for (float value : values) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "float32 is required");
        std::memcpy(&bits, &value, sizeof(bits));
        const std::array<char, 4> bytes = {
            static_cast<char>((bits >> 0) & 0xff),
            static_cast<char>((bits >> 8) & 0xff),
            static_cast<char>((bits >> 16) & 0xff),
            static_cast<char>((bits >> 24) & 0xff),
        };
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    if (!output) {
        fail("cannot write diagnostic file: " + path.string());
    }
}

enum class TargetKind { none, embedding, layer, logits };

struct Target {
    TargetKind kind = TargetKind::none;
    int layer = -1;
};

static Target classify_target(const char *name) {
    if (std::strcmp(name, "embd") == 0) {
        return {TargetKind::embedding, -1};
    }
    // The final Laguna residual is the same tensor first named l_out-47 and
    // then renamed h_nextn when the graph exposes it to speculative drafters.
    if (std::strcmp(name, "h_nextn") == 0) {
        return {TargetKind::layer, kLayers - 1};
    }
    if (std::strcmp(name, "result_output") == 0) {
        return {TargetKind::logits, -1};
    }

    static constexpr const char *prefix = "l_out-";
    const size_t prefix_length = std::strlen(prefix);
    if (std::strncmp(name, prefix, prefix_length) != 0) {
        return {};
    }

    const char *suffix = name + prefix_length;
    if (*suffix == '\0') {
        return {};
    }
    errno = 0;
    char *end = nullptr;
    const long parsed = std::strtol(suffix, &end, 10);
    if (errno != 0 || *end != '\0' || parsed < 0 || parsed >= kLayers) {
        return {};
    }
    if (std::string(prefix) + std::to_string(parsed) != name) {
        return {};
    }
    return {TargetKind::layer, static_cast<int>(parsed)};
}

struct ProbeState {
    fs::path out;
    int n_vocab = 0;
    bool embedding_seen = false;
    std::array<bool, kLayers> layer_seen{};
    bool logits_seen = false;
    std::string error;
};

static void validate_f32_contiguous(const ggml_tensor *tensor, const std::string &name) {
    if (tensor->type != GGML_TYPE_F32) {
        fail(name + " must be F32, got " + ggml_type_name(tensor->type));
    }
    if (!ggml_is_contiguous(tensor)) {
        fail(name + " must be contiguous");
    }
}

static std::vector<float> copy_exact_tensor(
        const ggml_tensor *tensor,
        const std::string &name,
        int64_t ne0,
        int64_t ne1) {
    validate_f32_contiguous(tensor, name);
    if (tensor->ne[0] != ne0 || tensor->ne[1] != ne1 ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        fail(name + " must have shape [" + std::to_string(ne0) + "," +
             std::to_string(ne1) + ",1,1]");
    }
    const size_t count = static_cast<size_t>(ne0) * static_cast<size_t>(ne1);
    if (ggml_nbytes(tensor) != count * sizeof(float)) {
        fail(name + " has a noncanonical byte count");
    }

    std::vector<float> values(count);
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    return values;
}

static void capture_target(ProbeState &state, ggml_tensor *tensor, Target target) {
    switch (target.kind) {
        case TargetKind::embedding: {
            if (state.embedding_seen) {
                fail("duplicate embd callback");
            }
            state.embedding_seen = true;
            const std::vector<float> values =
                copy_exact_tensor(tensor, "embd", kWidth, kTokens);
            // GGML ne[0] is the contiguous feature axis, so this is token-major.
            write_f32_little_endian(state.out / "embd.f32", values);
            break;
        }
        case TargetKind::layer: {
            if (state.layer_seen[target.layer]) {
                fail("duplicate l_out callback for layer " + std::to_string(target.layer));
            }
            state.layer_seen[target.layer] = true;
            const std::string name = "l_out-" + std::to_string(target.layer);
            const std::vector<float> values =
                copy_exact_tensor(tensor, name, kWidth, kTokens);
            char filename[32];
            std::snprintf(filename, sizeof(filename), "layer-%02d.f32", target.layer);
            write_f32_little_endian(state.out / filename, values);
            break;
        }
        case TargetKind::logits: {
            if (state.logits_seen) {
                fail("duplicate result_output callback");
            }
            state.logits_seen = true;
            const std::vector<float> values =
                copy_exact_tensor(tensor, "result_output", state.n_vocab, 1);
            write_f32_little_endian(state.out / "logits.f32", values);
            break;
        }
        case TargetKind::none:
            break;
    }
}

static bool probe_callback(ggml_tensor *tensor, bool ask, void *user_data) {
    auto &state = *static_cast<ProbeState *>(user_data);
    const Target target = classify_target(tensor->name);
    if (ask) {
        return target.kind != TargetKind::none && state.error.empty();
    }
    if (target.kind == TargetKind::none) {
        return true;
    }
    if (!state.error.empty()) {
        return false;
    }

    try {
        capture_target(state, tensor, target);
        return true;
    } catch (const std::exception &error) {
        state.error = error.what();
        return false;
    }
}

static void require_complete_capture(const ProbeState &state) {
    if (!state.embedding_seen) {
        fail("embd callback was not observed");
    }
    for (int layer = 0; layer < kLayers; layer++) {
        if (!state.layer_seen[layer]) {
            fail("l_out callback was not observed for layer " + std::to_string(layer));
        }
    }
    if (!state.logits_seen) {
        fail("result_output callback was not observed");
    }
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
    BatchGuard() : value(llama_batch_init(kTokens, 0, 1)) {}
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

static void decode_short_prompt(
        llama_context *context,
        llama_batch &batch,
        const std::array<llama_token, kTokens> &tokens,
        const ProbeState &state) {
    if (batch.token == nullptr || batch.pos == nullptr || batch.n_seq_id == nullptr ||
        batch.seq_id == nullptr || batch.logits == nullptr) {
        fail("failed to allocate decode batch");
    }
    batch.n_tokens = kTokens;
    for (int index = 0; index < kTokens; index++) {
        batch.token[index] = tokens[index];
        batch.pos[index] = index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = index == kTokens - 1;
    }
    const int32_t result = llama_decode(context, batch);
    if (!state.error.empty()) {
        fail(state.error);
    }
    if (result != 0) {
        fail("llama_decode failed with status " + std::to_string(result));
    }
}

int main(int argc, char **argv) {
    bool backend_initialized = false;
    try {
        const Options options = parse_options(argc, argv);
        const std::array<llama_token, kTokens> tokens = read_exact_tokens(options.tokens);
        prepare_output_directory(options.out);

        ggml_backend_load_all();
        llama_backend_init();
        backend_initialized = true;

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

            ProbeState state;
            state.out = options.out;
            state.n_vocab = n_vocab;

            llama_context_params context_params = llama_context_default_params();
            context_params.n_ctx = 1024;
            context_params.n_batch = 1024;
            context_params.n_ubatch = 512;
            context_params.n_seq_max = 1;
            context_params.no_perf = true;
            context_params.flash_attn_type = options.flash_attn_type;
            context_params.cb_eval = probe_callback;
            context_params.cb_eval_user_data = &state;

            ContextGuard context;
            context.value = llama_init_from_model(model.value, context_params);
            if (context.value == nullptr) {
                fail("failed to create context");
            }

            BatchGuard batch;
            decode_short_prompt(context.value, batch.value, tokens, state);
            require_complete_capture(state);
        }

        llama_backend_free();
        backend_initialized = false;
        std::printf("embedding=embd.f32\nlayers=48\nlogits=logits.f32\nout=%s\n",
                    options.out.c_str());
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "probe_poolside_laguna_layers: %s\n", error.what());
        if (backend_initialized) {
            llama_backend_free();
        }
        return 1;
    }
}
