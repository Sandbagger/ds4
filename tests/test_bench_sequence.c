#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L
#define DS4_BENCH_SEQUENCE_TEST_HOOKS 1

#include "../ds4_bench_sequence.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static int g_failed;
static int g_total;

#define CHECK(condition, message) do {                                      \
    g_total++;                                                             \
    if (!(condition)) {                                                    \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);    \
        g_failed++;                                                        \
    }                                                                      \
} while (0)

static const char *const profile_ids[] = {
    "cache-8gib", "cache-12gib", "cache-16gib"
};
static const uint64_t cache_bytes[] = {
    UINT64_C(8589934592), UINT64_C(12884901888), UINT64_C(17179869184)
};
static const uint32_t prompt_order[][4] = {
    {512, 2048, 28672, 8192},
    {2048, 8192, 512, 28672},
    {8192, 28672, 2048, 512},
};
static const char *const prompt_ids[] = {
    "native-512", "native-2048", "native-8192", "native-28672"
};
static const char *const manifest_sha256 =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
static const char *const input_sha256 =
    "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

static size_t make_sequence(char *buffer,
                            size_t capacity,
                            size_t profile_index,
                            size_t order_index) {
    return (size_t)snprintf(
        buffer,
        capacity,
        "schema=ds4.qualification-sequence/v1\n"
        "manifest_sha256=%s\n"
        "profile_id=%s\n"
        "cache_bytes=%llu\n"
        "prompt_order_index=%zu\n"
        "prompt_id=native-%u\n"
        "prompt_tokens=%u\n"
        "mode=streamed\n"
        "input_size_bytes=5\n"
        "input_sha256=%s\n"
        "input_base64=aGVsbG8=\n"
        "max_generated_tokens=512\n"
        "temperature=0\n"
        "top_k=0\n"
        "top_p=1\n"
        "min_p=0.05\n"
        "seed=1\n"
        "stop_sequences_count=0\n"
        "stop_token_policy=model-native\n"
        "repetition_count=4\n"
        "repetition=0:cold\n"
        "repetition=1:warm-1\n"
        "repetition=2:warm-2\n"
        "repetition=3:warm-3\n",
        manifest_sha256,
        profile_ids[profile_index],
        (unsigned long long)cache_bytes[profile_index],
        order_index,
        prompt_order[profile_index][order_index],
        prompt_order[profile_index][order_index],
        input_sha256);
}

static bool write_bytes(const char *path, const void *data, size_t size) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return false;
    const unsigned char *bytes = (const unsigned char *)data;
    size_t offset = 0;
    while (offset < size) {
        ssize_t written = write(fd, bytes + offset, size - offset);
        if (written > 0) {
            offset += (size_t)written;
        } else if (written < 0 && errno == EINTR) {
            continue;
        } else {
            (void)close(fd);
            return false;
        }
    }
    return close(fd) == 0;
}

static bool replace_line(char *buffer,
                         size_t *size,
                         size_t capacity,
                         const char *prefix,
                         const char *replacement) {
    char *start = strstr(buffer, prefix);
    if (!start) return false;
    char *end = strchr(start, '\n');
    if (!end) return false;
    const size_t old_size = (size_t)(end - start);
    const size_t new_size = strlen(replacement);
    if (new_size > old_size && *size > capacity - (new_size - old_size)) return false;
    if (new_size < old_size) {
        memmove(start + new_size,
                start + old_size,
                *size - (size_t)(start - buffer) - old_size);
    } else if (new_size > old_size) {
        memmove(start + new_size,
                start + old_size,
                *size - (size_t)(start - buffer) - old_size);
    }
    memcpy(start, replacement, new_size);
    *size += new_size - old_size;
    buffer[*size] = '\0';
    return true;
}

static bool remove_line(char *buffer, size_t *size, const char *prefix) {
    char *start = strstr(buffer, prefix);
    if (!start) return false;
    char *end = strchr(start, '\n');
    if (!end) return false;
    end++;
    memmove(start, end, *size - (size_t)(end - buffer));
    *size -= (size_t)(end - start);
    buffer[*size] = '\0';
    return true;
}

static bool parse_reject(const char *path,
                         const void *data,
                         size_t size,
                         const char *label) {
    char error[256] = {0};
    CHECK(write_bytes(path, data, size), "write rejection fixture");
    ds4_bench_sequence parsed = {0};
    const bool accepted = ds4_bench_sequence_parse_file(
        path, &parsed, error, sizeof(error));
    CHECK(!accepted, label);
    CHECK(error[0] != '\0', "rejection includes a diagnostic");
    ds4_bench_sequence_free(&parsed);
    return !accepted;
}

static void mutate_after_first_read(int fd, void *context) {
    (void)fd;
    const char *path = (const char *)context;
    int writer = open(path, O_WRONLY);
    if (writer >= 0) {
        (void)ftruncate(writer, 1);
        (void)close(writer);
    }
}

int main(void) {
    char directory[] = "/tmp/ds4-bench-sequence-XXXXXX";
    CHECK(mkdtemp(directory) != NULL, "create temporary directory");
    char path[512];
    (void)snprintf(path, sizeof(path), "%s/sequence", directory);

    char sequence[4096];
    for (size_t profile = 0; profile < 3; profile++) {
        for (size_t order = 0; order < 4; order++) {
            size_t size = make_sequence(sequence, sizeof(sequence), profile, order);
            CHECK(size < sizeof(sequence), "valid fixture fits bounded buffer");
            CHECK(write_bytes(path, sequence, size), "write valid fixture");
            char error[256] = {0};
            ds4_bench_sequence parsed = {0};
            CHECK(ds4_bench_sequence_parse_file(path, &parsed, error, sizeof(error)),
                  "accept every profile/order mapping");
            CHECK(parsed.cache_bytes == cache_bytes[profile], "cache bytes map exactly");
            CHECK(strcmp(parsed.profile_id, profile_ids[profile]) == 0,
                  "profile id is owned by parsed result");
            CHECK(parsed.prompt_order_index == order, "prompt order index is parsed");
            CHECK(parsed.prompt_tokens == prompt_order[profile][order],
                  "prompt tokens map exactly");
            CHECK(strcmp(parsed.prompt_id, prompt_ids[prompt_order[profile][order] == 512 ? 0 :
                                      prompt_order[profile][order] == 2048 ? 1 :
                                      prompt_order[profile][order] == 8192 ? 2 : 3]) == 0,
                  "prompt id maps exactly");
            CHECK(parsed.input_size_bytes == 5 && parsed.input_size == 5,
                  "decoded input size is exact");
            CHECK(parsed.input_bytes != NULL &&
                  memcmp(parsed.input_bytes, "hello", 5) == 0,
                  "decoded input is owned and exact");
            CHECK(strcmp(parsed.manifest_sha256, manifest_sha256) == 0,
                  "manifest digest is parsed");
            CHECK(strcmp(parsed.input_sha256, input_sha256) == 0,
                  "input digest is parsed");
            ds4_bench_sequence_free(&parsed);
            CHECK(parsed.input_bytes == NULL && parsed.input_size == 0,
                  "cleanup releases owned input");
            CHECK(ds4_bench_sequence_parse_file(path, &parsed, error, sizeof(error)),
                  "parsed result can be reused after cleanup");
            ds4_bench_sequence_free(&parsed);
        }
    }

    size_t size = make_sequence(sequence, sizeof(sequence), 0, 0);
    CHECK(remove_line(sequence, &size, "repetition=3:warm-3"), "remove a line");
    parse_reject(path, sequence, size, "missing line is rejected");
    size = make_sequence(sequence, sizeof(sequence), 0, 0);
    CHECK(replace_line(sequence, &size, sizeof(sequence),
                       "manifest_sha256=", "schema=ds4.qualification-sequence/v1"),
          "make duplicate line");
    parse_reject(path, sequence, size, "duplicate line is rejected");
    size = make_sequence(sequence, sizeof(sequence), 0, 0);
    CHECK(replace_line(sequence, &size, sizeof(sequence),
                       "schema=", "manifest_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
          "make reordered line");
    parse_reject(path, sequence, size, "reordered line is rejected");
    size = make_sequence(sequence, sizeof(sequence), 0, 0);
    CHECK(replace_line(sequence, &size, sizeof(sequence),
                       "schema=", "unknown=x"), "make unknown line");
    parse_reject(path, sequence, size, "unknown line is rejected");
    size = make_sequence(sequence, sizeof(sequence), 0, 0);
    sequence[size++] = '\n';
    sequence[size] = '\0';
    parse_reject(path, sequence, size, "extra line is rejected");
    size = make_sequence(sequence, sizeof(sequence), 0, 0);
    CHECK(replace_line(sequence, &size, sizeof(sequence),
                       "repetition=2:warm-2", ""), "make blank line");
    parse_reject(path, sequence, size, "blank line is rejected");

    size = make_sequence(sequence, sizeof(sequence), 0, 0);
    char crlf[4096];
    size_t crlf_size = 0;
    for (size_t i = 0; i < size; i++) {
        if (sequence[i] == '\n') crlf[crlf_size++] = '\r';
        crlf[crlf_size++] = sequence[i];
    }
    parse_reject(path, crlf, crlf_size, "CRLF is rejected");
    size = make_sequence(sequence, sizeof(sequence), 0, 0);
    sequence[10] = '\0';
    parse_reject(path, sequence, size, "embedded NUL is rejected");

    const struct {
        const char *prefix;
        const char *replacement;
        const char *label;
    } bad_lines[] = {
        {"manifest_sha256=", "manifest_sha256=0000000000000000000000000000000000000000000000000000000000000000", "zero manifest digest is rejected"},
        {"manifest_sha256=", "manifest_sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "uppercase manifest digest is rejected"},
        {"profile_id=", "profile_id=cache-12gib", "wrong profile is rejected"},
        {"cache_bytes=", "cache_bytes=12884901888", "wrong cache ceiling is rejected"},
        {"prompt_order_index=", "prompt_order_index=01", "noncanonical order index is rejected"},
        {"prompt_order_index=", "prompt_order_index=4", "out-of-range order index is rejected"},
        {"prompt_id=", "prompt_id=native-2048", "wrong prompt id is rejected"},
        {"prompt_tokens=", "prompt_tokens=2048", "wrong prompt token count is rejected"},
        {"mode=", "mode=resident", "wrong mode is rejected"},
        {"input_size_bytes=", "input_size_bytes=01", "noncanonical input size is rejected"},
        {"input_size_bytes=", "input_size_bytes=bad", "bad input size is rejected"},
        {"input_size_bytes=", "input_size_bytes=16777217", "oversized input is rejected"},
        {"input_size_bytes=", "input_size_bytes=18446744073709551616", "overflow input size is rejected"},
        {"input_sha256=", "input_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "input hash mismatch is rejected"},
        {"input_base64=", "input_base64=aGVsbG8", "unpadded base64 is rejected"},
        {"input_base64=", "input_base64=!!!!", "invalid base64 is rejected"},
        {"max_generated_tokens=", "max_generated_tokens=511", "wrong max tokens is rejected"},
        {"temperature=", "temperature=1", "wrong temperature is rejected"},
        {"top_k=", "top_k=1", "wrong top-k is rejected"},
        {"top_p=", "top_p=0", "wrong top-p is rejected"},
        {"min_p=", "min_p=0", "wrong min-p is rejected"},
        {"seed=", "seed=2", "wrong seed is rejected"},
        {"stop_sequences_count=", "stop_sequences_count=1", "wrong stop count is rejected"},
        {"stop_token_policy=", "stop_token_policy=custom", "wrong stop policy is rejected"},
        {"repetition_count=", "repetition_count=3", "wrong repetition count is rejected"},
        {"repetition=0:cold", "repetition=0:warm-0", "wrong cold repetition is rejected"},
        {"repetition=1:warm-1", "repetition=1:cold", "wrong warm repetition is rejected"},
    };
    for (size_t i = 0; i < sizeof(bad_lines) / sizeof(bad_lines[0]); i++) {
        size = make_sequence(sequence, sizeof(sequence), 0, 0);
        CHECK(replace_line(sequence, &size, sizeof(sequence),
                           bad_lines[i].prefix, bad_lines[i].replacement),
              "make malformed line");
        parse_reject(path, sequence, size, bad_lines[i].label);
    }

    size = make_sequence(sequence, sizeof(sequence), 0, 0);
    CHECK(replace_line(sequence, &size, sizeof(sequence),
                       "input_size_bytes=", "input_size_bytes=6"),
          "make decoded size mismatch");
    parse_reject(path, sequence, size, "decoded size mismatch is rejected");

    char symlink_path[512];
    (void)snprintf(symlink_path, sizeof(symlink_path), "%s/symlink", directory);
    CHECK(symlink(path, symlink_path) == 0, "create sequence symlink");
    char error[256] = {0};
    ds4_bench_sequence parsed = {0};
    CHECK(!ds4_bench_sequence_parse_file(symlink_path, &parsed, error, sizeof(error)),
          "symlink sequence is rejected");
    ds4_bench_sequence_free(&parsed);
    CHECK(unlink(symlink_path) == 0, "remove sequence symlink");

    size = make_sequence(sequence, sizeof(sequence), 0, 0);
    CHECK(write_bytes(path, sequence, size), "write mutation fixture");
    ds4_bench_sequence_test_set_after_first_read_hook(
        mutate_after_first_read, path);
    memset(error, 0, sizeof(error));
    parsed = (ds4_bench_sequence){0};
    CHECK(!ds4_bench_sequence_parse_file(path, &parsed, error, sizeof(error)),
          "identity mutation during read is rejected");
    CHECK(error[0] != '\0', "mutation includes a diagnostic");
    ds4_bench_sequence_free(&parsed);
    ds4_bench_sequence_test_set_after_first_read_hook(NULL, NULL);

    char directory_path[512];
    (void)snprintf(directory_path, sizeof(directory_path), "%s/directory", directory);
    CHECK(mkdir(directory_path, 0700) == 0, "create non-regular fixture");
    memset(error, 0, sizeof(error));
    parsed = (ds4_bench_sequence){0};
    CHECK(!ds4_bench_sequence_parse_file(directory_path, &parsed, error, sizeof(error)),
          "non-regular sequence is rejected");
    ds4_bench_sequence_free(&parsed);
    CHECK(rmdir(directory_path) == 0, "remove non-regular fixture");

    int oversized = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(oversized >= 0, "create oversized fixture");
    if (oversized >= 0) {
        CHECK(ftruncate(oversized, (off_t)DS4_BENCH_SEQUENCE_MAX_FILE_BYTES + 1) == 0,
              "size oversized fixture");
        CHECK(close(oversized) == 0, "close oversized fixture");
        memset(error, 0, sizeof(error));
        parsed = (ds4_bench_sequence){0};
        CHECK(!ds4_bench_sequence_parse_file(path, &parsed, error, sizeof(error)),
              "oversized sequence is rejected");
        ds4_bench_sequence_free(&parsed);
    }

    CHECK(unlink(path) == 0, "remove sequence fixture");
    CHECK(rmdir(directory) == 0, "remove temporary directory");
    fprintf(stdout, "bench-sequence: %d checks, %d failures\n", g_total, g_failed);
    return g_failed == 0 ? 0 : 1;
}
