#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L

#include "../ds4_bench_sequence.h"
#include "../ds4_plan_io.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static int failures;
static int total;

#define CHECK(condition, message) do {                                      \
    total++;                                                               \
    if (!(condition)) {                                                    \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);    \
        failures++;                                                        \
    }                                                                       \
} while (0)

static const char *const manifest_sha256 =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
static const char *const input_sha256 =
    "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
static size_t make_sequence(char *buffer, size_t capacity) {
    return (size_t)snprintf(
        buffer,
        capacity,
        "schema=ds4.qualification-sequence/v1\n"
        "manifest_sha256=%s\n"
        "profile_id=cache-8gib\n"
        "cache_bytes=8589934592\n"
        "prompt_order_index=0\n"
        "prompt_id=native-512\n"
        "prompt_tokens=512\n"
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

static bool replace_text(char *buffer,
                         size_t size,
                         const char *old_text,
                         const char *new_text) {
    const size_t old_size = strlen(old_text);
    const size_t new_size = strlen(new_text);
    if (old_size != new_size || old_size > size) return false;
    char *found = strstr(buffer, old_text);
    if (found == NULL) return false;
    memcpy(found, new_text, new_size);
    return true;
}

static bool sequence_is_clean(const ds4_bench_sequence *sequence) {
    return sequence != NULL &&
           sequence->manifest_sha256[0] == '\0' &&
           sequence->profile_id[0] == '\0' &&
           sequence->cache_bytes == 0 &&
           sequence->prompt_order_index == 0 &&
           sequence->prompt_id[0] == '\0' &&
           sequence->prompt_tokens == 0 &&
           sequence->input_size_bytes == 0 &&
           sequence->input_size == 0 &&
           sequence->input_bytes == NULL &&
           sequence->input_sha256[0] == '\0' &&
           sequence->sequence_sha256[0] == '\0';
}

static void seed_owned_output(ds4_bench_sequence *sequence) {
    ds4_bench_sequence_init(sequence);
    sequence->input_bytes = (unsigned char *)malloc(4u);
    if (sequence->input_bytes != NULL) {
        memcpy(sequence->input_bytes, "old!", 4u);
        sequence->input_size = 4u;
        sequence->input_size_bytes = 4u;
    }
    memcpy(sequence->manifest_sha256, "old", sizeof("old"));
    memcpy(sequence->sequence_sha256, "old", sizeof("old"));
}

static void expect_rejected(const char *path,
                            const char *expected_manifest,
                            const char *expected_sequence,
                            const char *label) {
    ds4_bench_sequence sequence;
    seed_owned_output(&sequence);
    char error[256] = {0};
    const bool accepted = ds4_bench_sequence_parse_file_trusted(
        path,
        expected_manifest,
        expected_sequence,
        &sequence,
        error,
        sizeof(error));
    CHECK(!accepted, label);
    CHECK(error[0] != '\0', "trusted rejection includes a diagnostic");
    CHECK(sequence_is_clean(&sequence), "trusted failure leaves output clean");
    ds4_bench_sequence_free(&sequence);
}

static bool mtime_equal(const struct stat *left, const struct stat *right) {
#if defined(__APPLE__)
    return left->st_mtimespec.tv_sec == right->st_mtimespec.tv_sec &&
           left->st_mtimespec.tv_nsec == right->st_mtimespec.tv_nsec;
#else
    return left->st_mtim.tv_sec == right->st_mtim.tv_sec &&
           left->st_mtim.tv_nsec == right->st_mtim.tv_nsec;
#endif
}

static bool restore_stat_times(const char *path, const struct stat *before) {
    struct timespec times[2];
#if defined(__APPLE__)
    times[0] = before->st_atimespec;
    times[1] = before->st_mtimespec;
#else
    times[0] = before->st_atim;
    times[1] = before->st_mtim;
#endif
    return utimensat(AT_FDCWD, path, times, 0) == 0;
}

int main(void) {
    char path[] = "/tmp/ds4-bench-sequence-trusted-XXXXXX";
    const int initial_fd = mkstemp(path);
    CHECK(initial_fd >= 0, "create trusted sequence fixture");
    if (initial_fd < 0) return 1;
    CHECK(close(initial_fd) == 0, "close trusted sequence fixture");

    char sequence_bytes[4096];
    const size_t sequence_size = make_sequence(
        sequence_bytes, sizeof(sequence_bytes));
    CHECK(sequence_size < sizeof(sequence_bytes),
          "trusted fixture fits bounded buffer");
    CHECK(sequence_size > 0, "trusted fixture is nonempty");
    CHECK(write_bytes(path, sequence_bytes, sequence_size),
          "write trusted sequence fixture");

    char sequence_sha256[DS4_PLAN_IO_SHA256_HEX_SIZE];
    char error[256] = {0};
    CHECK(ds4_plan_io_sha256(sequence_bytes,
                             sequence_size,
                             sequence_sha256,
                             error,
                             sizeof(error)),
          "hash complete trusted sequence bytes");

    ds4_bench_sequence parsed = {0};
    CHECK(ds4_bench_sequence_parse_file_trusted(
              path,
              manifest_sha256,
              sequence_sha256,
              &parsed,
              error,
              sizeof(error)),
          "valid trusted sequence is accepted");
    CHECK(strcmp(parsed.manifest_sha256, manifest_sha256) == 0,
          "trusted parser preserves manifest digest");
    CHECK(strcmp(parsed.sequence_sha256, sequence_sha256) == 0,
          "trusted parser populates exact sequence digest");
    CHECK(parsed.input_bytes != NULL && parsed.input_size == 5u &&
              memcmp(parsed.input_bytes, "hello", 5u) == 0,
          "trusted parser preserves decoded input ownership");
    ds4_bench_sequence_free(&parsed);

    char wrong_manifest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    char wrong_sequence[DS4_PLAN_IO_SHA256_HEX_SIZE];
    memcpy(wrong_manifest, manifest_sha256, sizeof(wrong_manifest));
    memcpy(wrong_sequence, sequence_sha256, sizeof(wrong_sequence));
    wrong_manifest[0] = wrong_manifest[0] == 'a' ? 'b' : 'a';
    wrong_sequence[0] = wrong_sequence[0] == 'a' ? 'b' : 'a';
    expect_rejected(path,
                    wrong_manifest,
                    sequence_sha256,
                    "trusted manifest mismatch is rejected");
    expect_rejected(path,
                    manifest_sha256,
                    wrong_sequence,
                    "trusted sequence mismatch is rejected");

    char invalid[DS4_PLAN_IO_SHA256_HEX_SIZE];
    memset(invalid, '0', DS4_PLAN_IO_SHA256_HEX_LENGTH);
    invalid[DS4_PLAN_IO_SHA256_HEX_LENGTH] = '\0';
    expect_rejected(path, invalid, sequence_sha256,
                    "all-zero trusted manifest digest is rejected");
    expect_rejected(path, manifest_sha256, invalid,
                    "all-zero trusted sequence digest is rejected");
    memset(invalid, 'A', DS4_PLAN_IO_SHA256_HEX_LENGTH);
    expect_rejected(path, invalid, sequence_sha256,
                    "uppercase trusted manifest digest is rejected");
    expect_rejected(path, manifest_sha256, invalid,
                    "uppercase trusted sequence digest is rejected");
    memset(invalid, 'g', DS4_PLAN_IO_SHA256_HEX_LENGTH);
    expect_rejected(path, invalid, sequence_sha256,
                    "non-hex trusted manifest digest is rejected");
    expect_rejected(path, manifest_sha256, invalid,
                    "non-hex trusted sequence digest is rejected");
    invalid[1] = '\0';
    expect_rejected(path, invalid, sequence_sha256,
                    "short trusted manifest digest is rejected");
    expect_rejected(path, manifest_sha256, invalid,
                    "short trusted sequence digest is rejected");

    struct stat before;
    CHECK(stat(path, &before) == 0, "stat original trusted sequence");
    char mutated[4096];
    memcpy(mutated, sequence_bytes, sequence_size);
    mutated[sequence_size] = '\0';
    CHECK(replace_text(
              mutated,
              sequence_size,
              "input_sha256=2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
              "input_sha256=187c9bceeb919e1b3e6d20fa50ecabf7d9d50b5343e8f9a3d912abb13929102e"),
          "mutate same-size input digest");
    CHECK(replace_text(mutated,
                       sequence_size,
                       "input_base64=aGVsbG8=",
                       "input_base64=amVsbG8="),
          "mutate same-size decoded input");
    CHECK(write_bytes(path, mutated, sequence_size),
          "write same-size mutation");
    CHECK(restore_stat_times(path, &before),
          "restore sequence mtime after same-size mutation");
    struct stat restored;
    CHECK(stat(path, &restored) == 0, "stat restored mutation");
    CHECK(restored.st_size == before.st_size &&
              restored.st_dev == before.st_dev &&
              restored.st_ino == before.st_ino &&
              mtime_equal(&before, &restored),
          "same-size mutation restores file identity fields");
    expect_rejected(path,
                    manifest_sha256,
                    sequence_sha256,
                    "same-size restored-mtime mutation is rejected");

    int oversized = open(path, O_WRONLY | O_TRUNC);
    CHECK(oversized >= 0, "open oversized trusted fixture");
    if (oversized >= 0) {
        CHECK(ftruncate(
                  oversized,
                  (off_t)DS4_BENCH_SEQUENCE_MAX_FILE_BYTES + 1) == 0,
              "size oversized trusted fixture sparsely");
        CHECK(close(oversized) == 0, "close oversized trusted fixture");
        expect_rejected(path,
                        manifest_sha256,
                        sequence_sha256,
                        "oversized trusted sequence is rejected boundedly");
    }

    CHECK(unlink(path) == 0, "remove trusted sequence fixture");
    fprintf(stdout, "trusted-bench-sequence: %d checks, %d failures\n",
            total, failures);
    return failures == 0 ? 0 : 1;
}
