#include "ds4.h"
#include "ds4_build_info.h"
#include "ds4_distributed.h"
#include "ds4_gpu_args.h"
#include "ds4_gpu.h"
#include "ds4_help.h"
#include "ds4_plan_io.h"

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then snapshots the live session in memory when the payload is small enough,
 * performs a fixed greedy decode run without allowing EOS, restores the
 * snapshot or replays the prefix, and continues to the next frontier.  Snapshot
 * save/restore time is intentionally outside both timing windows.
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES (UINT64_C(1) << 30)

typedef struct {
    const char *model_path;
    const char *qualification_plan_path;
    const char *qualification_sequence_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    const char *expert_profile_path;
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
    ds4_backend backend;
    int qualification_control_fd;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int power_percent;
    uint32_t prefill_chunk;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_full_layers;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    double step_mul;
    const char *dump_frontier_logits_dir;
    ds4_dist_options dist;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_cache_experts_set;
    bool ssd_streaming_cache_bytes_set;
    bool ssd_streaming_full_layers_set;
    bool qualification_plan_path_set;
    bool qualification_sequence_path_set;
    bool qualification_control_fd_set;
    bool cuda_tensor_parallel;
    bool show_output;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static uint64_t bench_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static uint64_t bench_snapshot_max_bytes(void) {
    const char *env = getenv("DS4_BENCH_SNAPSHOT_MAX_BYTES");
    if (!env || env[0] == '\0') return DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES;
    if (!strcmp(env, "unlimited") || !strcmp(env, "UNLIMITED") ||
        !strcmp(env, "inf") || !strcmp(env, "INF")) {
        return UINT64_MAX;
    }
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (env[0] == '\0' || !end || *end != '\0') {
        fprintf(stderr,
                "ds4-bench: invalid DS4_BENCH_SNAPSHOT_MAX_BYTES=%s; using default %llu\n",
                env,
                (unsigned long long)DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES);
        return DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES;
    }
    return (uint64_t)v;
}

static double bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_BENCH, topic);
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4-bench: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4-bench: valid backends are: metal, cuda, cpu\n");
#endif
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "ds4-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ds4-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "ds4-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "ds4-bench: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = "ds4flash.gguf",
        .system = "You are a helpful assistant.",
        .backend = default_backend(),
        .ctx_start = 2048,
        .ctx_max = 32768,
        .step_incr = 2048,
        .gen_tokens = 128,
        .step_mul = 1.0,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.dist,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-bench: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--qualification-plan")) {
            if (c.qualification_plan_path_set) {
                fprintf(stderr,
                        "ds4-bench: --qualification-plan may only be specified once\n");
                exit(2);
            }
            const char *path = need_arg(&i, argc, argv, arg);
            if (path[0] == '\0') {
                fprintf(stderr,
                        "ds4-bench: --qualification-plan requires a non-empty path\n");
                exit(2);
            }
            c.qualification_plan_path = path;
            c.qualification_plan_path_set = true;
        } else if (!strcmp(arg, "--qualification-control-fd")) {
            if (c.qualification_control_fd_set) {
                fprintf(stderr,
                        "ds4-bench: --qualification-control-fd may only be specified once\n");
                exit(2);
            }
            c.qualification_control_fd =
                parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.qualification_control_fd_set = true;
        } else if (!strcmp(arg, "--qualification-sequence")) {
            if (c.qualification_sequence_path_set) {
                fprintf(stderr,
                        "ds4-bench: --qualification-sequence may only be specified once\n");
                exit(2);
            }
            c.qualification_sequence_path = need_arg(&i, argc, argv, arg);
            c.qualification_sequence_path_set = true;
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-prompt-file")) {
            c.chat_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-mul")) {
            c.step_mul = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "--tokens") || !strcmp(arg, "-n")) {
            c.gen_tokens = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-frontier-logits-dir")) {
            c.dump_frontier_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-profile")) {
            c.expert_profile_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--gpu-vram")) {
            c.gpu_vram_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu-devices")) {
            c.gpu_devices_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--cuda-tensor-parallel")) {
            c.cuda_tensor_parallel = true;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-bytes")) {
            uint64_t bytes = 0;
            if (!ds4_parse_positive_u64_decimal(
                    need_arg(&i, argc, argv, arg), &bytes)) {
                fprintf(stderr,
                        "ds4-bench: --ssd-streaming-cache-bytes must be canonical positive decimal bytes\n");
                exit(2);
            }
            if (c.ssd_streaming_cache_experts_set) {
                fprintf(stderr,
                        "ds4-bench: --ssd-streaming-cache-bytes cannot be combined with --ssd-streaming-cache-experts\n");
                exit(2);
            }
            if (c.ssd_streaming_cache_bytes_set &&
                c.ssd_streaming_cache_bytes != bytes) {
                fprintf(stderr,
                        "ds4-bench: conflicting --ssd-streaming-cache-bytes values\n");
                exit(2);
            }
            c.ssd_streaming_cache_bytes = bytes;
            c.ssd_streaming_cache_bytes_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            if (c.ssd_streaming_cache_bytes_set) {
                fprintf(stderr,
                        "ds4-bench: --ssd-streaming-cache-bytes cannot be combined with --ssd-streaming-cache-experts\n");
                exit(2);
            }
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4-bench: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.ssd_streaming_cache_experts = experts;
            c.ssd_streaming_cache_bytes = bytes;
            c.ssd_streaming_cache_experts_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            int v = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.ssd_streaming_full_layers = (uint32_t)v;
            c.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-bench: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4-bench: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            c.prefill_chunk = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.power_percent < 1 || c.power_percent > 100) {
                fprintf(stderr, "ds4-bench: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--show-output")) {
            c.show_output = true;
        } else {
            fprintf(stderr, "ds4-bench: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (c.qualification_plan_path_set) return c;

    if (c.qualification_sequence_path_set) {
        if (c.prompt_path || c.chat_prompt_path || c.csv_path ||
            c.dump_frontier_logits_dir) {
            fprintf(stderr,
                    "ds4-bench: --qualification-sequence owns prompt and output selection\n");
            exit(2);
        }
        return c;
    }

    if (!!c.prompt_path == !!c.chat_prompt_path) {
        fprintf(stderr, "ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "ds4-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_mul < 1.0) {
        fprintf(stderr, "ds4-bench: --step-mul must be >= 1\n");
        exit(2);
    }
    if (c.step_mul == 1.0 && c.step_incr <= 0) {
        fprintf(stderr, "ds4-bench: --step-incr must be positive when --step-mul is 1\n");
        exit(2);
    }
    if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
        fprintf(stderr, "ds4-bench: requested context is too large\n");
        exit(2);
    }
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "ds4-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
        exit(2);
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.dist, NULL, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        exit(2);
    }
    if (c.dist.role == DS4_DISTRIBUTED_WORKER) {
        fprintf(stderr, "ds4-bench: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }
    return c;
}

typedef struct {
    char prompt_path[PATH_MAX];
    char prompt_sha256[DS4_PLAN_IO_SHA256_HEX_SIZE];
    char manifest_sha256[DS4_PLAN_IO_SHA256_HEX_SIZE];
    uint32_t prompt_tokens;
    uint32_t requested_output_tokens;
    uint32_t repetitions;
    bool resident_mode;
} bench_qualification_sequence;

typedef struct {
    bool ready;
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    ds4_gpu_nvml_inventory_snapshot pre_child;
    uint8_t build_identity[DS4_RUNTIME_BUILD_IDENTITY_BYTES];
#endif
} bench_qualification_external;

static bool json_flat_string(const char *json, const char *key,
                             char *out, size_t outcap) {
    char needle[96];
    if (snprintf(needle, sizeof(needle), "\"%s\":\"", key) <= 0) return false;
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    size_t n = 0;
    while (*p && *p != '"') {
        if (*p == '\\' || (unsigned char)*p < 0x20 || n + 1 >= outcap) return false;
        out[n++] = *p++;
    }
    if (*p != '"') return false;
    out[n] = '\0';
    return n > 0;
}

static bool json_flat_u32(const char *json, const char *key, uint32_t *out) {
    char needle[96];
    if (snprintf(needle, sizeof(needle), "\"%s\":", key) <= 0) return false;
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    if (*p < '1' || *p > '9') return false;
    uint64_t value = 0;
    do {
        value = value * 10u + (uint64_t)(*p++ - '0');
        if (value > UINT32_MAX) return false;
    } while (*p >= '0' && *p <= '9');
    if (*p != ',' && *p != '}') return false;
    *out = (uint32_t)value;
    return true;
}

static bool sha256_hex_valid(const char *value) {
    if (!value || strlen(value) != 64u) return false;
    for (size_t i = 0; i < 64u; i++) {
        if (!((value[i] >= '0' && value[i] <= '9') ||
              (value[i] >= 'a' && value[i] <= 'f'))) return false;
    }
    return true;
}

static bool sha256_file_path(const char *path,
                             char digest[DS4_PLAN_IO_SHA256_HEX_SIZE]) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    char err[256];
    const bool ok = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0 &&
        ds4_plan_io_sha256_fd(fd, (uint64_t)st.st_size, digest, err, sizeof(err));
    close(fd);
    return ok;
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
static int hexadecimal_nibble(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
}

static bool capture_running_build_identity(
        uint8_t out[DS4_RUNTIME_BUILD_IDENTITY_BYTES]) {
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    if (!sha256_file_path("/proc/self/exe", digest)) return false;
    for (size_t i = 0; i < DS4_RUNTIME_BUILD_IDENTITY_BYTES; i++) {
        const int high = hexadecimal_nibble(digest[2u * i]);
        const int low = hexadecimal_nibble(digest[2u * i + 1u]);
        if (high < 0 || low < 0) return false;
        out[i] = (uint8_t)((unsigned)high * 16u + (unsigned)low);
    }
    return true;
}
#endif

static bool validate_qualification_sequence(
        const char *path, bench_qualification_sequence *sequence,
        char *err, size_t errcap) {
    char *json = read_file(path);
    char schema[64];
    char mode[16];
    memset(sequence, 0, sizeof(*sequence));
    const bool parsed =
        json_flat_string(json, "schema", schema, sizeof(schema)) &&
        !strcmp(schema, "ds4.qualification-sequence/v1") &&
        json_flat_string(json, "prompt_path", sequence->prompt_path,
                         sizeof(sequence->prompt_path)) &&
        json_flat_string(json, "prompt_sha256", sequence->prompt_sha256,
                         sizeof(sequence->prompt_sha256)) &&
        json_flat_string(json, "manifest_sha256", sequence->manifest_sha256,
                         sizeof(sequence->manifest_sha256)) &&
        json_flat_string(json, "mode", mode, sizeof(mode)) &&
        json_flat_u32(json, "prompt_tokens", &sequence->prompt_tokens) &&
        json_flat_u32(json, "requested_output_tokens",
                      &sequence->requested_output_tokens) &&
        json_flat_u32(json, "repetitions", &sequence->repetitions);
    size_t field_count = 0;
    for (const char *cursor = json; (cursor = strstr(cursor, "\":")) != NULL;
         cursor += 2) {
        field_count++;
    }
    free(json);
    if (!parsed || field_count != 8u ||
        !sha256_hex_valid(sequence->prompt_sha256) ||
        !sha256_hex_valid(sequence->manifest_sha256) ||
        (strcmp(mode, "resident") && strcmp(mode, "streamed"))) {
        snprintf(err, errcap, "invalid closed qualification sequence");
        return false;
    }
    if (sequence->repetitions != 4) {
        snprintf(err, errcap, "qualification sequence repetitions != 4");
        return false;
    }
    if (sequence->prompt_tokens == 0 || sequence->requested_output_tokens == 0) {
        snprintf(err, errcap, "qualification sequence token counts must be non-zero");
        return false;
    }
    sequence->resident_mode = !strcmp(mode, "resident");
    char observed[DS4_PLAN_IO_SHA256_HEX_SIZE];
    if (!sha256_file_path(sequence->prompt_path, observed) ||
        strcmp(observed, sequence->prompt_sha256)) {
        snprintf(err, errcap, "qualification prompt digest mismatch");
        return false;
    }
    return true;
}

static bool bench_runtime_json(ds4_engine *engine,
                               ds4_runtime_wire_snapshot *snapshot,
                               char *json, size_t jsoncap) {
    size_t length = 0;
    return ds4_engine_runtime_snapshot(engine, snapshot) &&
        ds4_runtime_wire_snapshot_json(snapshot, json, jsoncap, &length);
}

static void qualification_milestone(FILE *out, const char *milestone,
                                    uint32_t repetition_index,
                                    const char *request_id,
                                    uint64_t accepted_monotonic_ns) {
    fprintf(out,
            "{\"schema\":\"ds4.bench.qualification-milestone/v1\","
            "\"milestone\":\"%s\",\"repetition_index\":%u,"
            "\"request_id\":\"%s\",\"accepted_monotonic_ns\":%" PRIu64
            ",\"monotonic_ns\":%" PRIu64 "}\n",
            milestone, repetition_index, request_id, accepted_monotonic_ns,
            bench_now_ns());
    fflush(out);
}

static int run_qualification_sequence(const bench_config *cfg,
                                      ds4_engine *engine,
                                      const bench_qualification_sequence *sequence,
                                      const bench_qualification_external *external) {
    char *rendered = read_file(sequence->prompt_path);
    ds4_tokens prompt = {0};
    ds4_tokenize_text(engine, rendered, &prompt);
    free(rendered);
    if (prompt.len != (int)sequence->prompt_tokens) {
        fprintf(stderr,
                "ds4-bench: qualification prompt token count mismatch: got %d expected %u\n",
                prompt.len, sequence->prompt_tokens);
        ds4_tokens_free(&prompt);
        return 1;
    }
    const uint32_t context = sequence->prompt_tokens +
                             sequence->requested_output_tokens + 1u;
    const int eos = ds4_token_eos(engine);
    int rc = 0;
    for (uint32_t repetition = 0; repetition < sequence->repetitions; repetition++) {
        ds4_session *session = NULL;
        if (ds4_session_create(&session, engine, (int)context) != 0) {
            rc = 1;
            break;
        }
        ds4_runtime_request_context request = {0};
        ds4_runtime_request_metrics metrics = {0};
        const uint64_t accepted = bench_now_ns();
        char err[256] = {0};
        if (!ds4_runtime_request_begin(&request, accepted) ||
            !ds4_runtime_request_set_prompt_tokens(&request, prompt.len) ||
            !ds4_runtime_request_mark_prefill_started(&request, bench_now_ns())) {
            fprintf(stderr, "ds4-bench: cannot initialize qualification request\n");
            ds4_session_free(session);
            rc = 1;
            break;
        }
        qualification_milestone(stdout, "request_accepted", repetition,
                                request.request_id, accepted);
        if (ds4_session_sync_attributed(session, &prompt, &request,
                                        err, sizeof(err)) != 0 ||
            !ds4_runtime_request_mark_prefill_complete(&request, bench_now_ns())) {
            fprintf(stderr, "ds4-bench: qualification prefill failed: %s\n", err);
            ds4_session_free(session);
            rc = 1;
            break;
        }
        uint32_t generated = 0;
        for (; generated < sequence->requested_output_tokens; generated++) {
            const int token = ds4_session_argmax_excluding(session, eos);
            if (token < 0 ||
                ds4_session_eval_attributed(session, token, &request,
                                            err, sizeof(err)) != 0 ||
                !ds4_runtime_request_add_generated_tokens(&request, 1) ||
                !ds4_runtime_request_record_visible_decoded(
                    &request, 1, bench_now_ns())) {
                fprintf(stderr, "ds4-bench: qualification decode failed: %s\n", err);
                rc = 1;
                break;
            }
            if (generated == 0) {
                if (!ds4_runtime_request_mark_first_visible_emitted(
                        &request, bench_now_ns())) {
                    rc = 1;
                    break;
                }
                qualification_milestone(stdout, "first_token", repetition,
                                        request.request_id, accepted);
            }
        }
        if (rc == 0 &&
            ds4_session_request_barrier(session, &request, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: qualification request barrier failed: %s\n", err);
            rc = 1;
        }
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
        ds4_engine_laguna_external_checkpoint_observation observation;
        if (rc == 0 && !sequence->resident_mode &&
            (!external || !external->ready ||
             ds4_engine_laguna_external_checkpoint(
                 engine, &external->pre_child, external->build_identity,
                 &observation) != DS4_RUNTIME_STATUS_OK)) {
            fprintf(stderr, "ds4-bench: qualification external checkpoint failed\n");
            rc = 1;
        }
#else
        (void)external;
#endif
        if (rc == 0 &&
            !ds4_runtime_request_finish(&request, DS4_RUNTIME_REQUEST_COMPLETED,
                                        bench_now_ns(), &metrics)) {
            fprintf(stderr, "ds4-bench: qualification request finalization failed\n");
            rc = 1;
        }
        if (rc == 0) {
            char request_json[16384];
            char runtime_json[65536];
            size_t request_length = 0;
            ds4_runtime_wire_snapshot snapshot;
            if (!ds4_runtime_request_metrics_json(
                    &metrics, request_json, sizeof(request_json), &request_length) ||
                !bench_runtime_json(engine, &snapshot,
                                    runtime_json, sizeof(runtime_json))) {
                fprintf(stderr, "ds4-bench: qualification evidence serialization failed\n");
                rc = 1;
            } else {
                const ds4_runtime_snapshot *a = &snapshot.allocations;
                qualification_milestone(stdout, "request_complete", repetition,
                                        request.request_id, accepted);
                fprintf(stdout,
                        "{\"schema\":\"ds4.bench.qualification-sample/v1\","
                        "\"milestone\":\"request_complete\","
                        "\"repetition_index\":%u,\"request_id\":\"%s\","
                        "\"accepted_monotonic_ns\":%" PRIu64 ","
                        "\"monotonic_ns\":%" PRIu64 ","
                        "\"manifest_sha256\":\"%s\","
                        "\"resident_mode\":%s,"
                        "\"session_payload_bytes\":%" PRIu64 ","
                        "\"kv_allocated_bytes\":%" PRIu64 ","
                        "\"configured_prefill_rows\":%u,"
                        "\"allocated_prefill_rows\":%u,"
                        "\"expert_cache_bound_bytes\":%" PRIu64 ","
                        "\"expert_cache_current_bytes\":%" PRIu64 ","
                        "\"expert_cache_peak_bytes\":%" PRIu64 ","
                        "\"qualification_total_current_bytes\":%" PRIu64 ","
                        "\"qualification_total_peak_bytes\":%" PRIu64 ","
                        "\"model_source_resident_bytes\":%" PRIu64 ","
                        "\"external_attribution\":{"
                        "\"valid\":%s,\"generation\":%" PRIu64 ","
                        "\"checkpoint_sequence\":%" PRIu64 ","
                        "\"model_device_major\":%u,\"model_device_minor\":%u,"
                        "\"model_inode\":%" PRIu64 ","
                        "\"model_pss_bytes\":%" PRIu64 ","
                        "\"host_library_unattributed_bytes\":%" PRIu64 ","
                        "\"nvml_process_bytes\":%" PRIu64 ","
                        "\"tracked_cuda_physical_bytes\":%" PRIu64 ","
                        "\"cuda_library_unattributed_bytes\":%" PRIu64 ","
                        "\"unrelated_process_inventory_stable\":%s},"
                        "\"request_metrics\":%s,\"runtime_snapshot\":%s}\n",
                        repetition, metrics.request_id, accepted, bench_now_ns(),
                        sequence->manifest_sha256,
                        sequence->resident_mode ? "true" : "false",
                        ds4_session_payload_bytes(session),
                        a->category_current[DS4_RUNTIME_CATEGORY_KV_STATE],
                        snapshot.configured_prefill_rows,
                        snapshot.allocated_prefill_rows,
                        snapshot.expert_cache_limit_bytes,
                        a->category_current[DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD],
                        a->category_peak[DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD],
                        a->qualification_total_current,
                        a->qualification_total_peak,
                        a->report_current[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT],
                        a->external_sample.attributed_valid ? "true" : "false",
                        a->external_sample.attributed_generation,
                        a->external_sample.checkpoint_sequence,
                        a->external_sample.smaps_model_device_major,
                        a->external_sample.smaps_model_device_minor,
                        a->external_sample.smaps_model_inode,
                        a->external_sample.smaps_model_pss_bytes,
                        a->external_sample.host_library_unattributed_bytes,
                        a->external_sample.nvml_process_bytes,
                        a->external_sample.tracked_cuda_physical_bytes,
                        a->external_sample.cuda_library_unattributed_bytes,
                        a->external_sample.unrelated_process_inventory_stable
                            ? "true" : "false",
                        request_json, runtime_json);
                fflush(stdout);
            }
        }
        ds4_session_free(session);
        if (rc != 0) break;
    }
    ds4_tokens_free(&prompt);
    (void)cfg;
    return rc;
}

static void json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

static int write_frontier_logits_json(
        const bench_config *cfg,
        ds4_engine         *engine,
        ds4_session        *session,
        int                 frontier,
        int                 previous) {
    if (!cfg->dump_frontier_logits_dir) return 0;

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        fprintf(stderr, "ds4-bench: out of memory copying frontier logits\n");
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4-bench: failed to copy frontier logits at %d\n", frontier);
        free(logits);
        return 1;
    }

    char path[PATH_MAX];
    const int n = snprintf(path,
                           sizeof(path),
                           "%s/frontier_%06d.logits.json",
                           cfg->dump_frontier_logits_dir,
                           frontier);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "ds4-bench: frontier logits path is too long\n");
        free(logits);
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        free(logits);
        return 1;
    }

    const int argmax = ds4_session_argmax(session);
    fprintf(fp, "{\n  \"source\":\"ds4-bench\",\n  \"model\":");
    json_write_string(fp, cfg->model_path);
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quality\":%s,\n"
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n"
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n"
            "  \"ctx\":%d,\n  \"vocab\":%d,\n"
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
            ds4_backend_name(cfg->backend),
            cfg->quality ? "true" : "false",
            ds4_engine_routed_quant_bits(engine),
            frontier,
            frontier,
            frontier - previous,
            cfg->ctx_alloc,
            vocab,
            argmax,
            logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4-bench: failed to close %s\n", path);
        free(logits);
        return 1;
    }
    free(logits);
    return 0;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next;
    if (c->step_mul == 1.0) {
        if (cur > INT_MAX - c->step_incr) next = c->ctx_max;
        else next = cur + c->step_incr;
    } else {
        const double v = ceil((double)cur * c->step_mul);
        next = v > (double)INT_MAX ? c->ctx_max : (int)v;
        if (next <= cur) next = cur + 1;
    }
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void log_context_memory(ds4_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk,
                               bool        ssd_streaming) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill_mode(backend,
                                                      ctx_size,
                                                      prefill_chunk,
                                                      ssd_streaming);
    fprintf(stderr,
            "ds4-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static int wait_distributed_route(ds4_session *session) {
    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4-bench: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4-bench: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }
        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4-bench: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static void maybe_warn_distributed_step_shape(const bench_config *cfg, ds4_session *session) {
    if (!cfg || !session || cfg->dist.role != DS4_DISTRIBUTED_COORDINATOR) return;
    uint32_t chunk = cfg->dist.prefill_chunk;
    if (chunk == 0) {
        const int cap = ds4_session_prefill_cap(session);
        if (cap > 0) chunk = (uint32_t)cap;
    }
    if (chunk == 0) return;
    if (cfg->step_mul == 1.0 &&
        cfg->step_incr > 0 &&
        (uint32_t)cfg->step_incr < chunk &&
        cfg->ctx_start < cfg->ctx_max)
    {
        fprintf(stderr,
                "ds4-bench: note: --step-incr=%d is smaller than distributed prefill chunk %u; "
                "suffix rows will not show multi-chunk pipeline overlap\n",
                cfg->step_incr,
                chunk);
    }
}

int main(int argc, char **argv) {
    int version_handled = 0;
    const int version_rc = ds4_build_info_maybe_print_version(
        argc, argv, "--version-json", &version_handled);
    if (version_handled || version_rc != 0) return version_rc;
    char qualification_argv_err[256];
    const int qualification_argv_rc = ds4_qualification_args_preflight(
        argc, argv, DS4_QUALIFICATION_FRONTEND_BENCH,
        qualification_argv_err, sizeof(qualification_argv_err));
    if (qualification_argv_rc != 0) {
        fprintf(stderr, "ds4-bench: %s\n", qualification_argv_err);
        return qualification_argv_rc;
    }
    bench_config cfg = parse_options(argc, argv);
    bench_qualification_sequence qualification_sequence = {0};
    bench_qualification_external qualification_external = {0};
    if (cfg.qualification_sequence_path_set) {
        char sequence_err[256];
        if (!validate_qualification_sequence(cfg.qualification_sequence_path,
                                             &qualification_sequence,
                                             sequence_err,
                                             sizeof(sequence_err))) {
            fprintf(stderr, "ds4-bench: %s\n", sequence_err);
            return 2;
        }
        if (qualification_sequence.resident_mode == cfg.ssd_streaming) {
            fprintf(stderr,
                    "ds4-bench: qualification sequence mode does not match runtime mode\n");
            return 2;
        }
        const uint64_t required_context =
            (uint64_t)qualification_sequence.prompt_tokens +
            (uint64_t)qualification_sequence.requested_output_tokens + 1u;
        if (required_context > INT_MAX) {
            fprintf(stderr, "ds4-bench: qualification sequence context is too large\n");
            return 2;
        }
        cfg.ctx_start = (int)qualification_sequence.prompt_tokens;
        cfg.ctx_max = cfg.ctx_start;
        cfg.ctx_alloc = (int)required_context;
        cfg.gen_tokens = (int)qualification_sequence.requested_output_tokens;
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
        if (!qualification_sequence.resident_mode) {
            qualification_external.ready =
                ds4_gpu_nvml_inventory_capture(
                    &qualification_external.pre_child) != 0 &&
                capture_running_build_identity(
                    qualification_external.build_identity);
            if (!qualification_external.ready) {
                fprintf(stderr,
                        "ds4-bench: cannot capture pre-allocation qualification identity\n");
                return 2;
            }
        }
#else
        if (!qualification_sequence.resident_mode) {
            fprintf(stderr,
                    "ds4-bench: streamed qualification evidence requires CUDA on Linux\n");
            return 2;
        }
#endif
    }

    /* Hint the packer at the largest ctx this bench run will exercise
     * so per-layer KV bytes are priced for the real session size, not
     * a stale 4096 default. Single-tier and CPU paths ignore this. */
    int placement_ctx_hint = cfg.ctx_max;
    if (cfg.ctx_alloc > placement_ctx_hint) placement_ctx_hint = cfg.ctx_alloc;

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .runtime_build_info = ds4_build_info_get(),
        .qualification_plan_path = cfg.qualification_plan_path,
        .qualification_control_fd = cfg.qualification_control_fd,
        .qualification_control_fd_set = cfg.qualification_control_fd_set,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .context_size = cfg.ctx_alloc,
        .prefill_chunk = cfg.prefill_chunk,
        .ssd_streaming_cache_experts = cfg.ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = cfg.ssd_streaming_cache_bytes,
        .ssd_streaming_full_layers = cfg.ssd_streaming_full_layers,
        .ssd_streaming_preload_experts = cfg.ssd_streaming_preload_experts,
        .simulate_used_memory_bytes = cfg.simulate_used_memory_bytes,
        .power_percent = cfg.power_percent,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .cuda_tensor_parallel = cfg.cuda_tensor_parallel,
        .ssd_streaming = cfg.ssd_streaming,
        .ssd_streaming_cold = cfg.ssd_streaming_cold,
        .ssd_streaming_cache_experts_set =
            cfg.ssd_streaming_cache_experts_set,
        .ssd_streaming_cache_bytes_set = cfg.ssd_streaming_cache_bytes_set,
        .ssd_streaming_full_layers_set = cfg.ssd_streaming_full_layers_set,
        .qualification_plan_path_set = cfg.qualification_plan_path_set,
        .expert_profile_path = cfg.expert_profile_path,
        .distributed = cfg.dist,
    };

    ds4_gpu_config gpu_cfg = {0};
    bool skip_cuda = false;
    const bool have_gpu_config = cfg.gpu_vram_arg || cfg.gpu_devices_arg;
    if (cfg.qualification_plan_path_set) {
        if (have_gpu_config) {
            fprintf(stderr,
                    "ds4-bench: --qualification-plan cannot be combined "
                    "with --gpu-vram or --gpu-devices\n");
            return 2;
        }
        char plan_err[512];
        const int plan_rc = ds4_engine_write_qualification_plan(
            &opt, plan_err, sizeof(plan_err));
        if (plan_rc != 0) {
            fprintf(stderr, "ds4-bench: %s\n", plan_err);
        }
        return plan_rc;
    }
    if (have_gpu_config) {
        char gpu_err[256];
        if (parse_gpu_vram_arg(cfg.gpu_vram_arg, cfg.gpu_devices_arg,
                               &gpu_cfg, &skip_cuda,
                               gpu_err, sizeof(gpu_err)) != 0) {
            fprintf(stderr, "ds4-bench: %s\n", gpu_err);
            return 2;
        }
        cfg.backend = skip_cuda ? DS4_BACKEND_CPU : DS4_BACKEND_CUDA;
        opt.backend = cfg.backend;
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&cfg.dist, &opt, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        return 2;
    }
    ds4_engine *engine = NULL;
    if (have_gpu_config && !skip_cuda) {
        const bool was_auto =
            (cfg.gpu_vram_arg && !strcmp(cfg.gpu_vram_arg, "auto")) ||
            (!cfg.gpu_vram_arg && cfg.gpu_devices_arg);
        char layout[256];
        if (format_gpu_layout_line(&gpu_cfg, was_auto,
                                   layout, sizeof(layout)) > 0) {
            fprintf(stdout, "%s\n", layout);
            fflush(stdout);
        }
        const int open_rc = ds4_engine_create_with_gpu_config(
                &engine, &opt, &gpu_cfg);
        if (open_rc != 0) return open_rc;
    } else {
        const int open_rc = ds4_engine_open(&engine, &opt);
        if (open_rc != 0) return open_rc;
    }
    log_context_memory(opt.backend,
                       cfg.ctx_alloc,
                       ds4_engine_prefill_chunk(engine),
                       cfg.ssd_streaming);

    if (cfg.qualification_sequence_path_set) {
        const int qualification_rc =
            run_qualification_sequence(&cfg, engine, &qualification_sequence,
                                       &qualification_external);
        ds4_engine_close(engine);
        return qualification_rc;
    }

    char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
    ds4_tokens prompt = {0};
    if (cfg.chat_prompt_path) {
        ds4_encode_chat_prompt(engine, cfg.system, text, DS4_THINK_NONE, &prompt);
    } else {
        ds4_tokenize_text(engine, text, &prompt);
    }
    free(text);

    if (prompt.len < cfg.ctx_max) {
        fprintf(stderr,
                "ds4-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                prompt.len,
                cfg.ctx_max);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "ds4-bench: failed to create session\n");
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    if (cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR &&
        wait_distributed_route(session) != 0)
    {
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    maybe_warn_distributed_step_shape(&cfg, session);

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "ds4-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
    }
    fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,session_payload_bytes,kv_allocated_bytes\n");
    fflush(out);

    const int eos = ds4_token_eos(engine);
    const bool distributed = cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR;
    ds4_session_snapshot snap = {0};
    const uint64_t snapshot_max_bytes = bench_snapshot_max_bytes();
    bool warned_large_snapshot = false;
    char err[256];
    int previous = 0;
    int rc = 0;

    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        ds4_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };

        const double prefill_t0 = bench_now_sec();
        if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: prefill to %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }
        const double prefill_t1 = bench_now_sec();
        const double prefill_sec = prefill_t1 - prefill_t0;
        const int prefill_tokens = frontier - previous;

        if (write_frontier_logits_json(&cfg, engine, session, frontier, previous) != 0) {
            rc = 1;
            break;
        }

        const bool need_restore_after_generation =
            cfg.gen_tokens > 0 && frontier < cfg.ctx_max;
        bool have_snapshot = false;
        if (need_restore_after_generation && !distributed &&
            getenv("DS4_BENCH_DISABLE_SNAPSHOT") == NULL) {
            const uint64_t payload_bytes = ds4_session_payload_bytes(session);
            const bool large_snapshot_forced =
                getenv("DS4_BENCH_FORCE_SNAPSHOT") != NULL;
            if (payload_bytes > snapshot_max_bytes && !large_snapshot_forced) {
                if (!warned_large_snapshot) {
                    fprintf(stderr,
                            "ds4-bench: session payload snapshot is %.2f GiB, above the %.2f GiB benchmark limit; "
                            "replaying prefixes instead (set DS4_BENCH_FORCE_SNAPSHOT=1 to force snapshots)\n",
                            bytes_to_gib(payload_bytes),
                            bytes_to_gib(snapshot_max_bytes));
                    warned_large_snapshot = true;
                }
            } else if (payload_bytes > 0) {
                if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-bench: snapshot at %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
                have_snapshot = true;
            }
        }

        const double gen_t0 = bench_now_sec();
        double gen_first_sec = 0.0;
        double gen_steady_sec = 0.0;
        int gen_done = 0;
        int *gen_token_buf = cfg.show_output && cfg.gen_tokens > 0
            ? malloc((size_t)cfg.gen_tokens * sizeof(gen_token_buf[0]))
            : NULL;
        int gen_token_count = 0;
        for (int i = 0; i < cfg.gen_tokens; i++) {
            if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
                fprintf(stderr, "ds4-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const int token = ds4_session_argmax_excluding(session, eos);
            if (token < 0) {
                fprintf(stderr, "ds4-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const double token_t0 = bench_now_sec();
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
            const double token_t1 = bench_now_sec();
            if (i == 0) gen_first_sec = token_t1 - token_t0;
            else gen_steady_sec += token_t1 - token_t0;
            if (gen_token_buf) gen_token_buf[gen_token_count++] = token;
            gen_done++;
        }
        const double gen_t1 = bench_now_sec();
        if (cfg.show_output && gen_token_buf && gen_token_count > 0) {
            fprintf(stderr, "ds4-bench: gen[ctx=%d] decoded text: \"", frontier);
            for (int i = 0; i < gen_token_count; i++) {
                size_t tlen = 0;
                char *txt = ds4_token_text(engine, gen_token_buf[i], &tlen);
                if (txt) {
                    fwrite(txt, 1, tlen, stderr);
                    free(txt);
                }
            }
            fprintf(stderr, "\"\n");
            fflush(stderr);
        }
        free(gen_token_buf);
        if (rc != 0) break;

        if (!need_restore_after_generation) {
            /* Nothing later depends on the frontier state. */
        } else if (distributed || !have_snapshot) {
            if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: replay restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        } else {
            if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        const double gen_sec = gen_t1 - gen_t0;
        const int gen_steady_tokens = gen_done > 1 ? gen_done - 1 : 0;
        ds4_runtime_wire_snapshot allocation_snapshot;
        char allocation_json[4096];
        uint64_t kv_allocated_bytes = 0;
        if (bench_runtime_json(engine, &allocation_snapshot,
                               allocation_json, sizeof(allocation_json))) {
            kv_allocated_bytes = allocation_snapshot.allocations.category_current[
                DS4_RUNTIME_CATEGORY_KV_STATE];
        }
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%.3f,%d,%.2f,%llu,%llu\n",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                gen_done,
                gen_sec > 0.0 ? (double)gen_done / gen_sec : 0.0,
                gen_first_sec * 1000.0,
                gen_steady_tokens,
                gen_steady_sec > 0.0 ? (double)gen_steady_tokens / gen_steady_sec : 0.0,
                (unsigned long long)ds4_session_payload_bytes(session),
                (unsigned long long)kv_allocated_bytes);
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

    if (out != stdout) fclose(out);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return rc;
}
