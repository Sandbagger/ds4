#include "ds4.h"
#include "ds4_build_info.h"
#include "ds4_distributed.h"
#include "ds4_gpu.h"
#include "ds4_gpu_args.h"
#include "ds4_help.h"
#include "ds4_bench_sequence.h"
#include "ds4_bench_qualification.h"
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

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#endif

#define DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES (UINT64_C(1) << 30)

typedef struct {
    const char *model_path;
    const char *qualification_plan_path;
    const char *qualification_sequence_path;
    const char *qualification_manifest_sha256;
    const char *qualification_sequence_sha256;
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
    bool qualification_manifest_sha256_set;
    bool qualification_sequence_sha256_set;
    bool qualification_control_fd_set;
    bool cuda_tensor_parallel;
    bool show_output;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
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
        } else if (!strcmp(arg, "--qualification-sequence")) {
            if (c.qualification_sequence_path_set) {
                fprintf(stderr,
                        "ds4-bench: --qualification-sequence may only be specified once\n");
                exit(2);
            }
            const char *path = need_arg(&i, argc, argv, arg);
            if (path[0] == '\0') {
                fprintf(stderr,
                        "ds4-bench: --qualification-sequence requires a non-empty path\n");
                exit(2);
            }
            c.qualification_sequence_path = path;
            c.qualification_sequence_path_set = true;
        } else if (!strcmp(arg, "--qualification-manifest-sha256")) {
            if (c.qualification_manifest_sha256_set) {
                fprintf(stderr,
                        "ds4-bench: --qualification-manifest-sha256 may only be specified once\n");
                exit(2);
            }
            const char *digest = need_arg(&i, argc, argv, arg);
            if (digest[0] == '\0') {
                fprintf(stderr,
                        "ds4-bench: --qualification-manifest-sha256 requires a non-empty digest\n");
                exit(2);
            }
            c.qualification_manifest_sha256 = digest;
            c.qualification_manifest_sha256_set = true;
        } else if (!strcmp(arg, "--qualification-sequence-sha256")) {
            if (c.qualification_sequence_sha256_set) {
                fprintf(stderr,
                        "ds4-bench: --qualification-sequence-sha256 may only be specified once\n");
                exit(2);
            }
            const char *digest = need_arg(&i, argc, argv, arg);
            if (digest[0] == '\0') {
                fprintf(stderr,
                        "ds4-bench: --qualification-sequence-sha256 requires a non-empty digest\n");
                exit(2);
            }
            c.qualification_sequence_sha256 = digest;
            c.qualification_sequence_sha256_set = true;
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

    const bool qualification_sequence_requested =
        c.qualification_sequence_path_set ||
        c.qualification_manifest_sha256_set ||
        c.qualification_sequence_sha256_set;
    if (qualification_sequence_requested) {
        if (!c.qualification_sequence_path_set ||
            !c.qualification_manifest_sha256_set ||
            !c.qualification_sequence_sha256_set) {
            fprintf(stderr,
                    "ds4-bench: qualification sequence requires exactly one complete "
                    "triplet; missing one or more required options (each must occur "
                    "once): --qualification-sequence, "
                    "--qualification-manifest-sha256, and "
                    "--qualification-sequence-sha256\n");
            exit(2);
        }
        if (c.qualification_plan_path_set) {
            fprintf(stderr,
                    "ds4-bench: qualification sequence cannot be combined with "
                    "--qualification-plan\n");
            exit(2);
        }
        if (c.prompt_path || c.chat_prompt_path) {
            fprintf(stderr,
                    "ds4-bench: qualification sequence cannot be combined with "
                    "--prompt-file or --chat-prompt-file\n");
            exit(2);
        }
        /* The sequence is authenticated before backend availability is
         * decided.  Unsupported builds fail closed at the post-parse
         * boundary, before any model or engine access. */
        return c;
    }

    if (c.qualification_plan_path_set) return c;

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

#if defined(DS4_BENCH_QUALIFICATION_TEST_BACKEND) || \
    (!defined(DS4_NO_GPU) && !defined(__APPLE__) && \
     !defined(DS4_ROCM_BUILD))

#if !defined(__linux__)
static bool qualification_executable_path(
        char path[PATH_MAX], char *error, size_t error_size) {
    if (!path || PATH_MAX < 2) {
        if (error && error_size != 0u) {
            (void)snprintf(error, error_size,
                           "executable path buffer is unavailable");
        }
        return false;
    }
#if defined(__APPLE__)
    uint32_t path_size = (uint32_t)PATH_MAX;
    if (_NSGetExecutablePath(path, &path_size) != 0) {
        if (error && error_size != 0u) {
            (void)snprintf(error, error_size,
                           "failed to resolve the current executable path");
        }
        return false;
    }
    path[PATH_MAX - 1] = '\0';
#elif defined(_WIN32)
    const DWORD path_size = GetModuleFileNameA(
        NULL, path, (DWORD)PATH_MAX);
    if (path_size == 0u || path_size >= (DWORD)PATH_MAX) {
        if (error && error_size != 0u) {
            (void)snprintf(error, error_size,
                           "failed to resolve the current executable path");
        }
        return false;
    }
    path[path_size] = '\0';
#else
    if (error && error_size != 0u) {
        (void)snprintf(error, error_size,
                       "current executable path is unsupported on this platform");
    }
    return false;
#endif
    if (path[0] == '\0') {
        if (error && error_size != 0u) {
            (void)snprintf(error, error_size,
                           "current executable path is empty");
        }
        return false;
    }
    return true;
}
#endif

static int qualification_hex_value(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
}

static bool qualification_build_identity(
        uint8_t identity[DS4_RUNTIME_BUILD_IDENTITY_BYTES],
        char *error, size_t error_size) {
    if (!identity) {
        if (error && error_size != 0u) {
            (void)snprintf(error, error_size,
                           "executable build identity output is null");
        }
        return false;
    }
    memset(identity, 0, DS4_RUNTIME_BUILD_IDENTITY_BYTES);

    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_BINARY
    flags |= O_BINARY;
#endif
#if defined(__linux__)
    const int fd = open("/proc/self/exe", flags);
#else
    char path[PATH_MAX];
    if (!qualification_executable_path(path, error, error_size)) {
        return false;
    }
    const int fd = open(path, flags);
#endif
    if (fd < 0) {
        if (error && error_size != 0u) {
            (void)snprintf(error, error_size,
                           "open current executable: %s", strerror(errno));
        }
        return false;
    }

    bool ok = false;
    struct stat status;
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE] = {0};
    if (fstat(fd, &status) != 0) {
        if (error && error_size != 0u) {
            (void)snprintf(error, error_size,
                           "stat current executable: %s", strerror(errno));
        }
    } else if (!S_ISREG(status.st_mode) || status.st_size <= 0) {
        if (error && error_size != 0u) {
            (void)snprintf(error, error_size,
                           "current executable is not a non-empty regular file");
        }
    } else if (!ds4_plan_io_sha256_fd(
                   fd, (uint64_t)status.st_size, digest,
                   error, error_size)) {
        /* ds4_plan_io_sha256_fd supplies the bounded diagnostic. */
    } else {
        ok = true;
        for (size_t i = 0u;
             i < DS4_RUNTIME_BUILD_IDENTITY_BYTES; i++) {
            const int high = qualification_hex_value(digest[i * 2u]);
            const int low = qualification_hex_value(digest[i * 2u + 1u]);
            if (high < 0 || low < 0) {
                ok = false;
                if (error && error_size != 0u) {
                    (void)snprintf(error, error_size,
                                   "current executable SHA-256 is not hexadecimal");
                }
                break;
            }
            identity[i] = (uint8_t)((high << 4) | low);
        }
    }
    if (close(fd) != 0 && ok) {
        ok = false;
        if (error && error_size != 0u) {
            (void)snprintf(error, error_size,
                           "close current executable: %s", strerror(errno));
        }
    }
    if (!ok) memset(identity, 0, DS4_RUNTIME_BUILD_IDENTITY_BYTES);
    return ok;
}

static bool qualification_next_timestamp(
        uint64_t *last_timestamp, uint64_t *timestamp_out) {
    if (!last_timestamp || !timestamp_out) return false;
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 ||
        now.tv_sec < 0 || now.tv_nsec < 0 || now.tv_nsec >= 1000000000L) {
        return false;
    }
    const uint64_t seconds = (uint64_t)now.tv_sec;
    const uint64_t nanos = (uint64_t)now.tv_nsec;
    if (seconds > (UINT64_MAX - nanos) / UINT64_C(1000000000)) {
        return false;
    }
    uint64_t timestamp = seconds * UINT64_C(1000000000) + nanos;
    if (timestamp == 0u) timestamp = 1u;
    if (timestamp <= *last_timestamp) {
        if (*last_timestamp == UINT64_MAX) return false;
        timestamp = *last_timestamp + 1u;
    }
    *last_timestamp = timestamp;
    *timestamp_out = timestamp;
    return true;
}

static int qualification_lifecycle_failure(
        const char *operation, const char *detail) {
    fprintf(stderr, "ds4-bench: qualification %s failed: %s\n",
            operation ? operation : "operation",
            detail && detail[0] ? detail : "backend returned failure");
    return 1;
}

static bool qualification_copy_rendered_prompt(
        const ds4_bench_sequence *sequence,
        char **rendered_out,
        char *error, size_t error_size) {
    if (!sequence || !rendered_out || !sequence->input_bytes ||
        sequence->input_size == 0u || sequence->input_size_bytes == 0u ||
        sequence->input_size_bytes > (uint64_t)(SIZE_MAX - 1u) ||
        sequence->input_size != (size_t)sequence->input_size_bytes) {
        if (error && error_size != 0u) {
            (void)snprintf(error, error_size,
                           "qualification sequence input span is invalid");
        }
        return false;
    }
    char *rendered = (char *)malloc(sequence->input_size + 1u);
    if (!rendered) {
        if (error && error_size != 0u) {
            (void)snprintf(error, error_size,
                           "allocate rendered qualification prompt: %s",
                           strerror(errno));
        }
        return false;
    }
    memcpy(rendered, sequence->input_bytes, sequence->input_size);
    rendered[sequence->input_size] = '\0';
    *rendered_out = rendered;
    return true;
}

/* Execute one authenticated, fixed-shape benchmark sequence.  The caller
 * owns engine, prompt, rendered, and sequence; this runner owns each session
 * only until its repetition reaches a terminal milestone or aborts. */
static int run_qualification_lifecycle(
        ds4_engine *engine,
        const ds4_bench_sequence *sequence,
        const ds4_tokens *prompt,
        const ds4_gpu_nvml_inventory_snapshot *pre_child,
        const uint8_t expected_build_identity[
            DS4_RUNTIME_BUILD_IDENTITY_BYTES],
        FILE *stream) {
    if (!engine || !sequence || !prompt || !pre_child ||
        !expected_build_identity || !stream) {
        return qualification_lifecycle_failure(
            "lifecycle", "invalid runner input");
    }

    uint64_t last_timestamp = 0u;
    char error[256] = {0};
    for (uint32_t repetition_index = 0;
         repetition_index < 4;
         repetition_index++) {
        ds4_session *session = NULL;
        ds4_runtime_request_context request;
        ds4_runtime_request_metrics metrics;
        ds4_runtime_wire_snapshot runtime_snapshot;
        ds4_engine_laguna_external_checkpoint_observation observation;
        uint64_t timestamp = 0u;
        int rc = 0;
        memset(&request, 0, sizeof(request));
        memset(&metrics, 0, sizeof(metrics));
        memset(&runtime_snapshot, 0, sizeof(runtime_snapshot));
        memset(&observation, 0, sizeof(observation));
        error[0] = '\0';

        do {
            if (ds4_session_create(&session, engine, 32768) != 0) {
                rc = qualification_lifecycle_failure(
                    "session create", "backend returned failure");
                break;
            }
            if (!qualification_next_timestamp(&last_timestamp, &timestamp)) {
                rc = qualification_lifecycle_failure(
                    "request timestamp", "monotonic clock failed");
                break;
            }
            if (!ds4_runtime_request_begin(&request, timestamp)) {
                rc = qualification_lifecycle_failure(
                    "request begin", "request accounting rejected the request");
                break;
            }
            if (!ds4_runtime_request_set_prompt_tokens(
                    &request, (uint64_t)sequence->prompt_tokens)) {
                rc = qualification_lifecycle_failure(
                    "request prompt", "request accounting rejected prompt tokens");
                break;
            }

            memset(&observation, 0, sizeof(observation));
            if (ds4_engine_laguna_external_checkpoint(
                    engine, pre_child, expected_build_identity,
                    &observation) != DS4_RUNTIME_STATUS_OK) {
                rc = qualification_lifecycle_failure(
                    "accepted checkpoint", "external attribution rejected the checkpoint");
                break;
            }
            memset(&runtime_snapshot, 0, sizeof(runtime_snapshot));
            if (!ds4_engine_runtime_snapshot(engine, &runtime_snapshot)) {
                rc = qualification_lifecycle_failure(
                    "accepted snapshot", "runtime snapshot failed");
                break;
            }
            if (!qualification_next_timestamp(&last_timestamp, &timestamp)) {
                rc = qualification_lifecycle_failure(
                    "accepted timestamp", "monotonic clock failed");
                break;
            }
            ds4_bench_qualification_record record = {
                .sequence = sequence,
                .event = DS4_BENCH_QUALIFICATION_EVENT_REQUEST_ACCEPTED,
                .request_id = request.request_id,
                .repetition_index = repetition_index,
                .monotonic_ns = timestamp,
                .session_payload_bytes = ds4_session_payload_bytes(session),
                .runtime_snapshot = &runtime_snapshot,
                .request_metrics = NULL,
            };
            if (!ds4_bench_qualification_emit_record(
                    stream, &record, error, sizeof(error))) {
                rc = qualification_lifecycle_failure(
                    "request_accepted", error);
                break;
            }

            if (!qualification_next_timestamp(&last_timestamp, &timestamp) ||
                !ds4_runtime_request_mark_prefill_started(
                    &request, timestamp)) {
                rc = qualification_lifecycle_failure(
                    "prefill start", "request accounting rejected prefill start");
                break;
            }
            if (ds4_session_sync_attributed(
                    session, prompt, &request, error, sizeof(error)) != 0) {
                rc = qualification_lifecycle_failure("attributed prefill", error);
                break;
            }
            if (!qualification_next_timestamp(&last_timestamp, &timestamp) ||
                !ds4_runtime_request_mark_prefill_complete(
                    &request, timestamp)) {
                rc = qualification_lifecycle_failure(
                    "prefill complete", "request accounting rejected prefill completion");
                break;
            }

            const int token = ds4_session_argmax_excluding(
                session, ds4_token_eos(engine));
            if (token < 0) {
                rc = qualification_lifecycle_failure(
                    "token selection", "backend returned no non-EOS token");
                break;
            }
            if (ds4_session_eval_attributed(
                    session, token, &request, error, sizeof(error)) != 0) {
                rc = qualification_lifecycle_failure("attributed decode", error);
                break;
            }
            if (!ds4_runtime_request_add_generated_tokens(&request, 1u)) {
                rc = qualification_lifecycle_failure(
                    "generated accounting", "request accounting rejected generated token");
                break;
            }
            if (!qualification_next_timestamp(&last_timestamp, &timestamp) ||
                !ds4_runtime_request_record_visible_decoded(
                    &request, 1u, timestamp)) {
                rc = qualification_lifecycle_failure(
                    "visible accounting", "request accounting rejected visible token");
                break;
            }
            if (!qualification_next_timestamp(&last_timestamp, &timestamp) ||
                !ds4_runtime_request_mark_first_visible_emitted(
                    &request, timestamp)) {
                rc = qualification_lifecycle_failure(
                    "first-visible accounting", "request accounting rejected first visible token");
                break;
            }

            memset(&observation, 0, sizeof(observation));
            if (ds4_engine_laguna_external_checkpoint(
                    engine, pre_child, expected_build_identity,
                    &observation) != DS4_RUNTIME_STATUS_OK) {
                rc = qualification_lifecycle_failure(
                    "first-token checkpoint", "external attribution rejected the checkpoint");
                break;
            }
            memset(&runtime_snapshot, 0, sizeof(runtime_snapshot));
            if (!ds4_engine_runtime_snapshot(engine, &runtime_snapshot)) {
                rc = qualification_lifecycle_failure(
                    "first-token snapshot", "runtime snapshot failed");
                break;
            }
            if (!qualification_next_timestamp(&last_timestamp, &timestamp)) {
                rc = qualification_lifecycle_failure(
                    "first-token timestamp", "monotonic clock failed");
                break;
            }
            record = (ds4_bench_qualification_record){
                .sequence = sequence,
                .event = DS4_BENCH_QUALIFICATION_EVENT_FIRST_TOKEN,
                .request_id = request.request_id,
                .repetition_index = repetition_index,
                .monotonic_ns = timestamp,
                .session_payload_bytes = ds4_session_payload_bytes(session),
                .runtime_snapshot = &runtime_snapshot,
                .request_metrics = NULL,
            };
            if (!ds4_bench_qualification_emit_record(
                    stream, &record, error, sizeof(error))) {
                rc = qualification_lifecycle_failure("first_token", error);
                break;
            }

            if (ds4_session_request_barrier(
                    session, &request, error, sizeof(error)) != 0) {
                rc = qualification_lifecycle_failure("request barrier", error);
                break;
            }
            if (!qualification_next_timestamp(&last_timestamp, &timestamp) ||
                !ds4_runtime_request_finish(
                    &request, DS4_RUNTIME_REQUEST_COMPLETED,
                    timestamp, &metrics)) {
                rc = qualification_lifecycle_failure(
                    "request finish", "request accounting rejected completion");
                break;
            }

            memset(&observation, 0, sizeof(observation));
            if (ds4_engine_laguna_external_checkpoint(
                    engine, pre_child, expected_build_identity,
                    &observation) != DS4_RUNTIME_STATUS_OK) {
                rc = qualification_lifecycle_failure(
                    "completion checkpoint", "external attribution rejected the checkpoint");
                break;
            }
            memset(&runtime_snapshot, 0, sizeof(runtime_snapshot));
            if (!ds4_engine_runtime_snapshot(engine, &runtime_snapshot)) {
                rc = qualification_lifecycle_failure(
                    "completion snapshot", "runtime snapshot failed");
                break;
            }
            if (!qualification_next_timestamp(&last_timestamp, &timestamp)) {
                rc = qualification_lifecycle_failure(
                    "completion timestamp", "monotonic clock failed");
                break;
            }
            record = (ds4_bench_qualification_record){
                .sequence = sequence,
                .event = DS4_BENCH_QUALIFICATION_EVENT_REQUEST_COMPLETE,
                .request_id = request.request_id,
                .repetition_index = repetition_index,
                .monotonic_ns = timestamp,
                .session_payload_bytes = ds4_session_payload_bytes(session),
                .runtime_snapshot = &runtime_snapshot,
                .request_metrics = &metrics,
            };
            if (!ds4_bench_qualification_emit_record(
                    stream, &record, error, sizeof(error))) {
                rc = qualification_lifecycle_failure("request_complete", error);
                break;
            }
        } while (false);

        if (session) ds4_session_free(session);
        if (rc != 0) return rc;
    }
    return 0;
}

#endif

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

    if (cfg.qualification_sequence_path_set) {
        ds4_bench_sequence sequence = {0};
        char sequence_error[256] = {0};
        if (!ds4_bench_sequence_parse_file_trusted(
                cfg.qualification_sequence_path,
                cfg.qualification_manifest_sha256,
                cfg.qualification_sequence_sha256,
                &sequence,
                sequence_error,
                sizeof(sequence_error))) {
            ds4_bench_sequence_free(&sequence);
            fprintf(stderr,
                    "ds4-bench: qualification sequence rejected: %s\n",
                    sequence_error[0] ? sequence_error : "invalid sequence");
            return 2;
        }

        ds4_engine *qualification_engine = NULL;
        ds4_tokens qualification_prompt = {0};
        char *qualification_rendered = NULL;
        int qualification_rc = 2;
#if defined(DS4_BENCH_QUALIFICATION_TEST_BACKEND) || \
    (!defined(DS4_NO_GPU) && !defined(__APPLE__) && \
     !defined(DS4_ROCM_BUILD))
        if (cfg.backend != DS4_BACKEND_CUDA ||
            (cfg.gpu_vram_arg && !strcmp(cfg.gpu_vram_arg, "0"))) {
            fprintf(stderr,
                    "ds4-bench: qualification sequence requires the CUDA backend\n");
            goto qualification_cleanup;
        }
        if (!cfg.qualification_control_fd_set) {
            fprintf(stderr,
                    "ds4-bench: qualification sequence requires an inherited "
                    "--qualification-control-fd\n");
            goto qualification_cleanup;
        }
        if (cfg.gpu_vram_arg || cfg.gpu_devices_arg) {
            fprintf(stderr,
                    "ds4-bench: qualification sequence cannot use a GPU layout; "
                    "the canonical engine configuration is fixed\n");
            goto qualification_cleanup;
        }

        uint8_t expected_build_identity[
            DS4_RUNTIME_BUILD_IDENTITY_BYTES] = {0};
        if (!qualification_build_identity(
                expected_build_identity,
                sequence_error, sizeof(sequence_error))) {
            fprintf(stderr,
                    "ds4-bench: qualification executable identity failed: %s\n",
                    sequence_error[0] ? sequence_error : "unknown error");
            goto qualification_cleanup;
        }
        ds4_gpu_nvml_inventory_snapshot pre_child = {0};
        if (ds4_gpu_nvml_inventory_capture(&pre_child) != 0) {
            fprintf(stderr,
                    "ds4-bench: qualification pre-child NVML inventory failed\n");
            goto qualification_cleanup;
        }

        const ds4_engine_options qualification_options = {
            .model_path = cfg.model_path,
            .runtime_build_info = ds4_build_info_get(),
            .backend = DS4_BACKEND_CUDA,
            .context_size = 32768,
            .prefill_chunk = 4096u,
            .ssd_streaming_cache_bytes = sequence.cache_bytes,
            .ssd_streaming_cache_bytes_set = true,
            .ssd_streaming = true,
            .placement_ctx_hint = 32768,
            .session_slots = 1u,
            .qualification_control_fd = cfg.qualification_control_fd,
            .qualification_control_fd_set = true,
        };
        if (ds4_engine_open(&qualification_engine,
                            &qualification_options) != 0) {
            fprintf(stderr,
                    "ds4-bench: qualification engine open failed\n");
            goto qualification_cleanup;
        }
        if (!qualification_copy_rendered_prompt(
                &sequence, &qualification_rendered,
                sequence_error, sizeof(sequence_error))) {
            fprintf(stderr,
                    "ds4-bench: qualification prompt rejected: %s\n",
                    sequence_error[0] ? sequence_error : "invalid input");
            goto qualification_cleanup;
        }
        ds4_tokenize_rendered_chat(
            qualification_engine, qualification_rendered,
            &qualification_prompt);
        if (sequence.prompt_tokens > (uint32_t)INT_MAX ||
            qualification_prompt.len != (int)sequence.prompt_tokens) {
            fprintf(stderr,
                    "ds4-bench: qualification prompt token count %d does not "
                    "match sequence prompt_tokens=%u\n",
                    qualification_prompt.len,
                    sequence.prompt_tokens);
            goto qualification_cleanup;
        }
        qualification_rc = run_qualification_lifecycle(
            qualification_engine, &sequence, &qualification_prompt,
            &pre_child, expected_build_identity, stdout);
#else
        fprintf(stderr,
                "ds4-bench: qualification sequence unsupported on this build; "
                "authenticated execution requires CUDA\n");
        goto qualification_cleanup;
#endif

qualification_cleanup:
        if (qualification_engine) {
            ds4_engine_close(qualification_engine);
        }
        ds4_tokens_free(&qualification_prompt);
        free(qualification_rendered);
        ds4_bench_sequence_free(&sequence);
        return qualification_rc;
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
    fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes\n");
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
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%.3f,%d,%.2f,%llu\n",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                gen_done,
                gen_sec > 0.0 ? (double)gen_done / gen_sec : 0.0,
                gen_first_sec * 1000.0,
                gen_steady_tokens,
                gen_steady_sec > 0.0 ? (double)gen_steady_tokens / gen_steady_sec : 0.0,
                (unsigned long long)(have_snapshot ? snap.len : 0));
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
