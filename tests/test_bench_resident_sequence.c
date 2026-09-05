#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L

#include "../ds4_bench_sequence.h"
#include "../ds4_plan_io.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
static const uint32_t prompt_targets[] = {512u, 2048u, 8192u, 28672u};

static size_t make_sequence(char *buffer,
                            size_t capacity,
                            const char *schema,
                            const char *manifest,
                            const char *profile,
                            const char *cache_bytes,
                            uint32_t order_index,
                            uint32_t prompt_tokens,
                            const char *mode,
                            const char *input_size,
                            const char *input_digest,
                            const char *input_base64,
                            const char *last_repetition) {
    return (size_t)snprintf(
        buffer,
        capacity,
        "schema=%s\n"
        "manifest_sha256=%s\n"
        "profile_id=%s\n"
        "cache_bytes=%s\n"
        "prompt_order_index=%u\n"
        "prompt_id=native-%u\n"
        "prompt_tokens=%u\n"
        "mode=%s\n"
        "input_size_bytes=%s\n"
        "input_sha256=%s\n"
        "input_base64=%s\n"
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
        "%s\n",
        schema,
        manifest,
        profile,
        cache_bytes,
        order_index,
        prompt_tokens,
        prompt_tokens,
        mode,
        input_size,
        input_digest,
        input_base64,
        last_repetition);
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

static bool hash_bytes(const void *data, size_t size, char digest[65]) {
    char error[256] = {0};
    return ds4_plan_io_sha256(data, size, digest, error, sizeof(error));
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

static bool resident_is_clean(const ds4_bench_resident_sequence *resident) {
    return resident != NULL && sequence_is_clean(&resident->sequence);
}

static void seed_resident_output(ds4_bench_resident_sequence *resident) {
    ds4_bench_resident_sequence_init(resident);
    resident->sequence.input_bytes = (unsigned char *)malloc(4u);
    if (resident->sequence.input_bytes != NULL) {
        memcpy(resident->sequence.input_bytes, "old!", 4u);
        resident->sequence.input_size = 4u;
        resident->sequence.input_size_bytes = 4u;
    }
    memcpy(resident->sequence.manifest_sha256, "old", sizeof("old"));
    memcpy(resident->sequence.sequence_sha256, "old", sizeof("old"));
}

static void seed_streamed_output(ds4_bench_sequence *sequence) {
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

static void expect_resident_rejected(const char *path,
                                     const char *expected_manifest,
                                     const char *expected_sequence,
                                     const char *label) {
    ds4_bench_resident_sequence resident;
    seed_resident_output(&resident);
    char error[256] = {0};
    const bool accepted = ds4_bench_resident_sequence_parse_file_trusted(
        path,
        expected_manifest,
        expected_sequence,
        &resident,
        error,
        sizeof(error));
    CHECK(!accepted, label);
    CHECK(error[0] != '\0', "resident rejection includes a diagnostic");
    CHECK(resident_is_clean(&resident),
          "resident failure leaves nested output clean");
    ds4_bench_resident_sequence_free(&resident);
}

static void expect_streamed_rejected(const char *path,
                                     const char *expected_manifest,
                                     const char *expected_sequence,
                                     const char *label) {
    ds4_bench_sequence sequence;
    seed_streamed_output(&sequence);
    char error[256] = {0};
    const bool accepted = ds4_bench_sequence_parse_file_trusted(
        path,
        expected_manifest,
        expected_sequence,
        &sequence,
        error,
        sizeof(error));
    CHECK(!accepted, label);
    CHECK(error[0] != '\0', "streamed rejection includes a diagnostic");
    CHECK(sequence_is_clean(&sequence),
          "streamed failure leaves output clean");
    ds4_bench_sequence_free(&sequence);
}

int main(void) {
    CHECK(strcmp(DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA,
                 "ds4.resident-qualification-sequence/v1") == 0,
          "resident schema is distinct and exact");
    CHECK(DS4_BENCH_SEQUENCE_LINE_COUNT == 24u,
          "resident sequence keeps the exact 24-line contract");

    char path[] = "/tmp/ds4-bench-resident-sequence-XXXXXX";
    const int initial_fd = mkstemp(path);
    CHECK(initial_fd >= 0, "create resident sequence fixture");
    if (initial_fd < 0) return 1;
    CHECK(close(initial_fd) == 0, "close resident sequence fixture");

    char sequence_bytes[4096];
    char sequence_sha256[DS4_BENCH_SEQUENCE_SHA256_HEX_SIZE];
    char error[256] = {0};
    for (uint32_t order_index = 0; order_index < 4u; order_index++) {
        const size_t sequence_size = make_sequence(
            sequence_bytes,
            sizeof(sequence_bytes),
            DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA,
            manifest_sha256,
            "resident",
            "0",
            order_index,
            prompt_targets[order_index],
            "resident",
            "5",
            input_sha256,
            "aGVsbG8=",
            "repetition=3:warm-3");
        CHECK(sequence_size < sizeof(sequence_bytes),
              "resident fixture fits bounded buffer");
        CHECK(sequence_size > 0, "resident fixture is nonempty");
        CHECK(write_bytes(path, sequence_bytes, sequence_size),
              "write resident sequence fixture");
        CHECK(hash_bytes(sequence_bytes, sequence_size, sequence_sha256),
              "hash resident sequence bytes");

        ds4_bench_resident_sequence parsed;
        ds4_bench_resident_sequence_init(&parsed);
        memset(error, 0, sizeof(error));
        CHECK(ds4_bench_resident_sequence_parse_file_trusted(
                  path,
                  manifest_sha256,
                  sequence_sha256,
                  &parsed,
                  error,
                  sizeof(error)),
              "valid resident sequence is accepted");
        CHECK(strcmp(parsed.sequence.manifest_sha256, manifest_sha256) == 0,
              "resident parser preserves manifest digest under sequence");
        CHECK(strcmp(parsed.sequence.sequence_sha256, sequence_sha256) == 0,
              "resident parser preserves serialized digest under sequence");
        CHECK(strcmp(parsed.sequence.profile_id, "resident") == 0 &&
                  parsed.sequence.cache_bytes == 0,
              "resident profile has zero cache bytes");
        CHECK(parsed.sequence.prompt_order_index == order_index,
              "resident parser preserves every prompt order index");
        CHECK(strcmp(parsed.sequence.prompt_id,
                     order_index == 0u ? "native-512" :
                     order_index == 1u ? "native-2048" :
                     order_index == 2u ? "native-8192" : "native-28672") == 0,
              "resident parser preserves canonical prompt id");
        CHECK(parsed.sequence.prompt_tokens == prompt_targets[order_index],
              "resident parser preserves canonical prompt token count");
        CHECK(parsed.sequence.input_size_bytes == 5u &&
                  parsed.sequence.input_size == 5u &&
                  parsed.sequence.input_bytes != NULL &&
                  memcmp(parsed.sequence.input_bytes, "hello", 5u) == 0,
              "resident parser owns and decodes the input bytes");
        CHECK(strcmp(parsed.sequence.input_sha256, input_sha256) == 0,
              "resident parser preserves input digest");
        ds4_bench_resident_sequence_free(&parsed);
        CHECK(resident_is_clean(&parsed),
              "resident free clears nested owned output");
    }

    /* Both trusted parsers must reject the other mode even with valid bytes
     * and the matching serialized digest. */
    size_t sequence_size = make_sequence(
        sequence_bytes,
        sizeof(sequence_bytes),
        DS4_BENCH_SEQUENCE_SCHEMA,
        manifest_sha256,
        "cache-8gib",
        "8589934592",
        0u,
        512u,
        "streamed",
        "5",
        input_sha256,
        "aGVsbG8=",
        "repetition=3:warm-3");
    CHECK(write_bytes(path, sequence_bytes, sequence_size),
          "write streamed cross-mode fixture");
    CHECK(hash_bytes(sequence_bytes, sequence_size, sequence_sha256),
          "hash streamed cross-mode fixture");
    expect_resident_rejected(path, manifest_sha256, sequence_sha256,
                             "resident parser rejects streamed input");

    sequence_size = make_sequence(
        sequence_bytes,
        sizeof(sequence_bytes),
        DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA,
        manifest_sha256,
        "resident",
        "0",
        0u,
        512u,
        "resident",
        "5",
        input_sha256,
        "aGVsbG8=",
        "repetition=3:warm-3");
    CHECK(write_bytes(path, sequence_bytes, sequence_size),
          "write resident cross-mode fixture");
    CHECK(hash_bytes(sequence_bytes, sequence_size, sequence_sha256),
          "hash resident cross-mode fixture");
    expect_streamed_rejected(path, manifest_sha256, sequence_sha256,
                             "streamed parser rejects resident input");

    /* Expected digest authentication and canonical-digest validation. */
    char wrong_digest[DS4_BENCH_SEQUENCE_SHA256_HEX_SIZE];
    memcpy(wrong_digest, manifest_sha256, sizeof(wrong_digest));
    wrong_digest[0] = 'b';
    expect_resident_rejected(path, wrong_digest, sequence_sha256,
                             "resident manifest digest mismatch is rejected");
    memcpy(wrong_digest, sequence_sha256, sizeof(wrong_digest));
    wrong_digest[0] = wrong_digest[0] == 'a' ? 'b' : 'a';
    expect_resident_rejected(path, manifest_sha256, wrong_digest,
                             "resident sequence digest mismatch is rejected");
    memset(wrong_digest, '0', DS4_BENCH_SEQUENCE_SHA256_HEX_LENGTH);
    wrong_digest[DS4_BENCH_SEQUENCE_SHA256_HEX_LENGTH] = '\0';
    expect_resident_rejected(path, wrong_digest, sequence_sha256,
                             "all-zero resident manifest digest is rejected");
    expect_resident_rejected(path, manifest_sha256, wrong_digest,
                             "all-zero resident sequence digest is rejected");
    memset(wrong_digest, 'A', DS4_BENCH_SEQUENCE_SHA256_HEX_LENGTH);
    expect_resident_rejected(path, wrong_digest, sequence_sha256,
                             "uppercase resident manifest digest is rejected");
    expect_resident_rejected(path, manifest_sha256, wrong_digest,
                             "uppercase resident sequence digest is rejected");
    memset(wrong_digest, 'g', DS4_BENCH_SEQUENCE_SHA256_HEX_LENGTH);
    expect_resident_rejected(path, wrong_digest, sequence_sha256,
                             "non-hex resident manifest digest is rejected");
    expect_resident_rejected(path, manifest_sha256, wrong_digest,
                             "non-hex resident sequence digest is rejected");
    wrong_digest[1] = '\0';
    expect_resident_rejected(path, wrong_digest, sequence_sha256,
                             "short resident manifest digest is rejected");
    expect_resident_rejected(path, manifest_sha256, wrong_digest,
                             "short resident sequence digest is rejected");

    /* Authentication of the file must not select a parser kind.  Independently
     * mix each mode discriminator and authenticate those exact mutated bytes. */
    const struct {
        const char *schema, *profile, *cache, *mode, *input_size;
    } mixed[] = {
        {DS4_BENCH_SEQUENCE_SCHEMA, "resident", "0", "resident", "5"},
        {DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA, "cache-8gib", "0", "resident", "5"},
        {DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA, "resident", "0", "streamed", "5"},
        {DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA, "resident", "00", "resident", "5"},
        {DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA, "resident", "0", "resident", "4"},
        {DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA, "resident", "0", "resident", "16777217"},
    };
    for (size_t i = 0; i < sizeof(mixed) / sizeof(mixed[0]); i++) {
        sequence_size = make_sequence(
            sequence_bytes, sizeof(sequence_bytes), mixed[i].schema,
            manifest_sha256, mixed[i].profile, mixed[i].cache, 0u, 512u,
            mixed[i].mode, mixed[i].input_size, input_sha256, "aGVsbG8=",
            "repetition=3:warm-3");
        CHECK(sequence_size < sizeof(sequence_bytes), "mixed fixture fits");
        CHECK(write_bytes(path, sequence_bytes, sequence_size), "write mixed fixture");
        CHECK(hash_bytes(sequence_bytes, sequence_size, sequence_sha256),
              "hash exact mixed fixture");
        expect_resident_rejected(path, manifest_sha256, sequence_sha256,
                                 "resident parser rejects mixed contract");
        expect_streamed_rejected(path, manifest_sha256, sequence_sha256,
                                 "streamed parser rejects mixed contract");
    }

    /* Semantic lines are authenticated with their mutated serialized bytes so
     * each rejection reaches the resident schema/order contract. */
    sequence_size = make_sequence(
        sequence_bytes, sizeof(sequence_bytes),
        DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA, manifest_sha256, "resident", "0",
        1u, 512u, "resident", "5", input_sha256, "aGVsbG8=",
        "repetition=3:warm-3");
    CHECK(write_bytes(path, sequence_bytes, sequence_size),
          "write wrong-order resident fixture");
    CHECK(hash_bytes(sequence_bytes, sequence_size, sequence_sha256),
          "hash wrong-order resident fixture");
    expect_resident_rejected(path, manifest_sha256, sequence_sha256,
                             "resident wrong prompt order is rejected");

    sequence_size = make_sequence(
        sequence_bytes, sizeof(sequence_bytes),
        DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA, manifest_sha256, "resident", "1",
        0u, 512u, "resident", "5", input_sha256, "aGVsbG8=",
        "repetition=3:warm-3");
    CHECK(write_bytes(path, sequence_bytes, sequence_size),
          "write wrong-cache resident fixture");
    CHECK(hash_bytes(sequence_bytes, sequence_size, sequence_sha256),
          "hash wrong-cache resident fixture");
    expect_resident_rejected(path, manifest_sha256, sequence_sha256,
                             "resident wrong cache bytes are rejected");

    sequence_size = make_sequence(
        sequence_bytes, sizeof(sequence_bytes),
        DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA, manifest_sha256, "resident", "0",
        0u, 512u, "resident", "5", input_sha256, "aGVsbG8=",
        "repetition=3:warm-2");
    CHECK(write_bytes(path, sequence_bytes, sequence_size),
          "write wrong-repetition resident fixture");
    CHECK(hash_bytes(sequence_bytes, sequence_size, sequence_sha256),
          "hash wrong-repetition resident fixture");
    expect_resident_rejected(path, manifest_sha256, sequence_sha256,
                             "resident wrong repetition is rejected");

    sequence_size = make_sequence(
        sequence_bytes, sizeof(sequence_bytes),
        DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA, manifest_sha256, "resident", "0",
        0u, 512u, "resident", "5", input_sha256, "aGVsbG8=",
        "repetition=3:warm-3");
    CHECK(replace_text(sequence_bytes, sequence_size,
                       "input_sha256="
                       "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
                       "input_sha256="
                       "187c9bceeb919e1b3e6d20fa50ecabf7d9d50b5343e8f9a3d912abb13929102e"),
          "mutate resident input digest in place");
    CHECK(write_bytes(path, sequence_bytes, sequence_size),
          "write resident input-digest mutation");
    CHECK(hash_bytes(sequence_bytes, sequence_size, sequence_sha256),
          "hash resident input-digest mutation");
    expect_resident_rejected(path, manifest_sha256, sequence_sha256,
                             "resident input digest mutation is rejected");

    sequence_size = make_sequence(
        sequence_bytes, sizeof(sequence_bytes),
        DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA, manifest_sha256, "resident", "0",
        0u, 512u, "resident", "5", input_sha256, "aGVsbG8=",
        "repetition=3:warm-3");
    CHECK(replace_text(sequence_bytes, sequence_size, "input_base64=aGVsbG8=",
                       "input_base64=aGVsbGk="),
          "mutate resident base64 in place");
    CHECK(write_bytes(path, sequence_bytes, sequence_size),
          "write resident base64 mutation");
    CHECK(hash_bytes(sequence_bytes, sequence_size, sequence_sha256),
          "hash resident base64 mutation");
    expect_resident_rejected(path, manifest_sha256, sequence_sha256,
                             "resident base64 mutation is rejected");

    sequence_size = make_sequence(
        sequence_bytes, sizeof(sequence_bytes),
        DS4_BENCH_RESIDENT_SEQUENCE_SCHEMA, manifest_sha256, "resident", "0",
        0u, 512u, "resident", "5", input_sha256, "aGVsbG8=",
        "repetition=3:warm-3");
    CHECK(write_bytes(path, sequence_bytes, sequence_size - 1u),
          "write resident EOF mutation");
    CHECK(hash_bytes(sequence_bytes, sequence_size - 1u, sequence_sha256),
          "hash resident EOF mutation");
    expect_resident_rejected(path, manifest_sha256, sequence_sha256,
                             "resident missing final LF is rejected");

    int oversized = open(path, O_WRONLY | O_TRUNC);
    CHECK(oversized >= 0, "open oversized resident fixture");
    if (oversized >= 0) {
        CHECK(ftruncate(oversized,
                        (off_t)DS4_BENCH_SEQUENCE_MAX_FILE_BYTES + 1) == 0,
              "size resident fixture beyond parser bound");
        CHECK(close(oversized) == 0, "close oversized resident fixture");
        expect_resident_rejected(path, manifest_sha256, sequence_sha256,
                                 "oversized resident sequence is rejected");
    }

    CHECK(unlink(path) == 0, "remove resident sequence fixture");
    fprintf(stdout, "resident-bench-sequence: %d checks, %d failures\n",
            total, failures);
    return failures == 0 ? 0 : 1;
}
