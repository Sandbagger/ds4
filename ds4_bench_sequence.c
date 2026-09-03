#if defined(__linux__)
#define _GNU_SOURCE
#elif defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L

#include "ds4_bench_sequence.h"
#include "ds4_plan_io.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define DS4_BENCH_SEQUENCE_MAX_LINE_BYTES \
    DS4_BENCH_SEQUENCE_MAX_FILE_BYTES
#define DS4_BENCH_SEQUENCE_READ_CHUNK (64u * 1024u)

typedef struct {
    const unsigned char *bytes;
    size_t size;
    size_t offset;
    size_t line_number;
} ds4_bench_sequence_cursor;

#ifdef DS4_BENCH_SEQUENCE_TEST_HOOKS
static ds4_bench_sequence_test_after_first_read_hook
    ds4_bench_sequence_after_first_read_hook;
static void *ds4_bench_sequence_after_first_read_context;

void ds4_bench_sequence_test_set_after_first_read_hook(
        ds4_bench_sequence_test_after_first_read_hook hook,
        void *context) {
    ds4_bench_sequence_after_first_read_hook = hook;
    ds4_bench_sequence_after_first_read_context = context;
}
#endif

static void clear_error(char *error, size_t error_size) {
    if (error != NULL && error_size != 0) error[0] = '\0';
}

static void set_error(char *error,
                      size_t error_size,
                      const char *format,
                      ...) {
    if (error == NULL || error_size == 0) return;
    va_list args;
    va_start(args, format);
    (void)vsnprintf(error, error_size, format, args);
    va_end(args);
    error[error_size - 1] = '\0';
}

static bool sequence_stat_identity_matches(const struct stat *before,
                                           const struct stat *after) {
#if defined(__APPLE__)
    const bool mtime_matches =
        before->st_mtimespec.tv_sec == after->st_mtimespec.tv_sec &&
        before->st_mtimespec.tv_nsec == after->st_mtimespec.tv_nsec;
#else
    const bool mtime_matches =
        before->st_mtim.tv_sec == after->st_mtim.tv_sec &&
        before->st_mtim.tv_nsec == after->st_mtim.tv_nsec;
#endif
    return before->st_dev == after->st_dev &&
           before->st_ino == after->st_ino &&
           before->st_size == after->st_size &&
           mtime_matches;
}

void ds4_bench_sequence_init(ds4_bench_sequence *sequence) {
    if (sequence != NULL) memset(sequence, 0, sizeof(*sequence));
}

void ds4_bench_sequence_free(ds4_bench_sequence *sequence) {
    if (sequence == NULL) return;
    if (sequence->input_bytes != NULL) {
        memset(sequence->input_bytes, 0, sequence->input_size);
        free(sequence->input_bytes);
    }
    memset(sequence, 0, sizeof(*sequence));
}

static bool read_sequence_file(const char *path,
                               unsigned char **bytes_out,
                               size_t *size_out,
                               char *error,
                               size_t error_size) {
    if (path == NULL || path[0] == '\0') {
        set_error(error, error_size,
                  "qualification sequence path must be non-empty");
        return false;
    }
#if !defined(O_NOFOLLOW)
    set_error(error, error_size,
              "qualification sequence cannot enforce no-symlink opening");
    return false;
#else
    int flags = O_RDONLY | O_NOFOLLOW;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NONBLOCK
    flags |= O_NONBLOCK;
#endif
    int fd = open(path, flags);
    if (fd < 0) {
        const int saved_errno = errno;
        set_error(error, error_size,
                  "open qualification sequence: %s", strerror(saved_errno));
        return false;
    }

    unsigned char *bytes = NULL;
    bool ok = false;
    struct stat before;
    if (fstat(fd, &before) != 0) {
        const int saved_errno = errno;
        set_error(error, error_size,
                  "stat qualification sequence: %s", strerror(saved_errno));
        goto cleanup;
    }
    if (!S_ISREG(before.st_mode)) {
        set_error(error, error_size,
                  "qualification sequence is not a regular file");
        goto cleanup;
    }
    if (before.st_size <= 0 ||
        (uintmax_t)before.st_size > DS4_BENCH_SEQUENCE_MAX_FILE_BYTES) {
        set_error(error, error_size,
                  "qualification sequence size is outside the %u-byte bound",
                  DS4_BENCH_SEQUENCE_MAX_FILE_BYTES);
        goto cleanup;
    }
    if ((uintmax_t)before.st_size > SIZE_MAX - 1u) {
        set_error(error, error_size,
                  "qualification sequence size overflows memory bounds");
        goto cleanup;
    }
    const size_t expected_size = (size_t)before.st_size;
    bytes = (unsigned char *)malloc(expected_size + 1u);
    if (bytes == NULL) {
        set_error(error, error_size,
                  "allocate qualification sequence: %s", strerror(errno));
        goto cleanup;
    }

    size_t offset = 0;
#ifdef DS4_BENCH_SEQUENCE_TEST_HOOKS
    bool first_read_hook_ran = false;
#endif
    while (offset < expected_size) {
        size_t chunk = expected_size - offset;
        if (chunk > DS4_BENCH_SEQUENCE_READ_CHUNK) {
            chunk = DS4_BENCH_SEQUENCE_READ_CHUNK;
        }
        const off_t read_offset = (off_t)offset;
        if (read_offset < 0 || (size_t)read_offset != offset) {
            set_error(error, error_size,
                      "qualification sequence offset exceeds platform bounds");
            goto cleanup;
        }
        ssize_t count;
        do {
            count = pread(fd, bytes + offset, chunk, read_offset);
        } while (count < 0 && errno == EINTR);
        if (count < 0) {
            const int saved_errno = errno;
            set_error(error, error_size,
                      "read qualification sequence: %s", strerror(saved_errno));
            goto cleanup;
        }
        if (count == 0) {
            set_error(error, error_size,
                      "qualification sequence was truncated while reading");
            goto cleanup;
        }
        if ((size_t)count > chunk) {
            set_error(error, error_size,
                      "qualification sequence read exceeded its bound");
            goto cleanup;
        }
        offset += (size_t)count;
#ifdef DS4_BENCH_SEQUENCE_TEST_HOOKS
        if (!first_read_hook_ran) {
            first_read_hook_ran = true;
            if (ds4_bench_sequence_after_first_read_hook != NULL) {
                ds4_bench_sequence_after_first_read_hook(
                    fd, ds4_bench_sequence_after_first_read_context);
            }
        }
#endif
    }
    bytes[expected_size] = '\0';

    /* A second read at the recorded end makes growth fail even before the
     * identity comparison, while fstat catches replacement, truncation, and
     * metadata mutation. */
    unsigned char extra;
    const off_t expected_offset = (off_t)expected_size;
    if (expected_offset < 0 || (size_t)expected_offset != expected_size) {
        set_error(error, error_size,
                  "qualification sequence size exceeds platform bounds");
        goto cleanup;
    }
    ssize_t extra_count;
    do {
        extra_count = pread(fd, &extra, 1u, expected_offset);
    } while (extra_count < 0 && errno == EINTR);
    if (extra_count < 0) {
        const int saved_errno = errno;
        set_error(error, error_size,
                  "verify qualification sequence length: %s",
                  strerror(saved_errno));
        goto cleanup;
    }
    if (extra_count != 0) {
        set_error(error, error_size,
                  "qualification sequence grew while reading");
        goto cleanup;
    }

    struct stat after;
    if (fstat(fd, &after) != 0) {
        const int saved_errno = errno;
        set_error(error, error_size,
                  "restat qualification sequence: %s", strerror(saved_errno));
        goto cleanup;
    }
    if (!sequence_stat_identity_matches(&before, &after)) {
        set_error(error, error_size,
                  "qualification sequence identity changed while reading");
        goto cleanup;
    }
    *bytes_out = bytes;
    *size_out = expected_size;
    bytes = NULL;
    ok = true;

cleanup:
    free(bytes);
    if (close(fd) != 0 && ok) {
        const int saved_errno = errno;
        set_error(error, error_size,
                  "close qualification sequence: %s", strerror(saved_errno));
        free(*bytes_out);
        *bytes_out = NULL;
        *size_out = 0;
        ok = false;
    }
    return ok;
#endif
}

static bool next_line(ds4_bench_sequence_cursor *cursor,
                      const unsigned char **line_out,
                      size_t *length_out,
                      char *error,
                      size_t error_size) {
    if (cursor->offset >= cursor->size) {
        set_error(error, error_size,
                  "qualification sequence is missing line %zu",
                  cursor->line_number + 1u);
        return false;
    }
    const size_t start = cursor->offset;
    const unsigned char *newline = memchr(
        cursor->bytes + start, '\n', cursor->size - start);
    if (newline == NULL) {
        set_error(error, error_size,
                  "qualification sequence line %zu is not LF-terminated",
                  cursor->line_number + 1u);
        return false;
    }
    const size_t length = (size_t)(newline - (cursor->bytes + start));
    if (length == 0) {
        set_error(error, error_size,
                  "qualification sequence line %zu is blank",
                  cursor->line_number + 1u);
        return false;
    }
    if (memchr(cursor->bytes + start, '\r', length) != NULL) {
        set_error(error, error_size,
                  "qualification sequence line %zu contains CR",
                  cursor->line_number + 1u);
        return false;
    }
    if (memchr(cursor->bytes + start, '\0', length) != NULL) {
        set_error(error, error_size,
                  "qualification sequence line %zu contains NUL",
                  cursor->line_number + 1u);
        return false;
    }
    if (length > DS4_BENCH_SEQUENCE_MAX_LINE_BYTES) {
        set_error(error, error_size,
                  "qualification sequence line %zu exceeds its bound",
                  cursor->line_number + 1u);
        return false;
    }
    *line_out = cursor->bytes + start;
    *length_out = length;
    cursor->offset += length + 1u;
    cursor->line_number++;
    return true;
}

static bool line_literal(const unsigned char *line,
                         size_t length,
                         const char *expected,
                         size_t line_number,
                         char *error,
                         size_t error_size) {
    const size_t expected_length = strlen(expected);
    if (length != expected_length ||
        memcmp(line, expected, expected_length) != 0) {
        set_error(error, error_size,
                  "qualification sequence line %zu is not '%s'",
                  line_number, expected);
        return false;
    }
    return true;
}

static bool line_value(const unsigned char *line,
                       size_t length,
                       const char *prefix,
                       const unsigned char **value_out,
                       size_t *value_length_out,
                       size_t line_number,
                       char *error,
                       size_t error_size) {
    const size_t prefix_length = strlen(prefix);
    if (length <= prefix_length ||
        memcmp(line, prefix, prefix_length) != 0) {
        set_error(error, error_size,
                  "qualification sequence line %zu has an unexpected key",
                  line_number);
        return false;
    }
    *value_out = line + prefix_length;
    *value_length_out = length - prefix_length;
    if (*value_length_out == 0) {
        set_error(error, error_size,
                  "qualification sequence line %zu has an empty value",
                  line_number);
        return false;
    }
    return true;
}

static bool copy_string_value(const unsigned char *value,
                              size_t value_length,
                              char *destination,
                              size_t destination_size,
                              size_t line_number,
                              char *error,
                              size_t error_size) {
    if (value_length >= destination_size) {
        set_error(error, error_size,
                  "qualification sequence line %zu value is too long",
                  line_number);
        return false;
    }
    memcpy(destination, value, value_length);
    destination[value_length] = '\0';
    return true;
}

static bool parse_uint64_decimal(const unsigned char *value,
                                 size_t value_length,
                                 bool positive,
                                 uint64_t *result,
                                 size_t line_number,
                                 char *error,
                                 size_t error_size) {
    if (value_length == 0 ||
        (value_length > 1u && value[0] == '0')) {
        set_error(error, error_size,
                  "qualification sequence line %zu is not canonical decimal",
                  line_number);
        return false;
    }
    uint64_t parsed = 0;
    for (size_t i = 0; i < value_length; i++) {
        if (value[i] < '0' || value[i] > '9') {
            set_error(error, error_size,
                      "qualification sequence line %zu is not canonical decimal",
                      line_number);
            return false;
        }
        const uint64_t digit = (uint64_t)(value[i] - '0');
        if (parsed > (UINT64_MAX - digit) / UINT64_C(10)) {
            set_error(error, error_size,
                      "qualification sequence line %zu decimal overflows uint64",
                      line_number);
            return false;
        }
        parsed = parsed * UINT64_C(10) + digit;
    }
    if (positive && parsed == 0) {
        set_error(error, error_size,
                  "qualification sequence line %zu must be positive",
                  line_number);
        return false;
    }
    *result = parsed;
    return true;
}

static bool validate_expected_sha256(const char *value,
                                    const char *label,
                                    char *error,
                                    size_t error_size) {
    if (value == NULL) {
        set_error(error, error_size,
                  "trusted %s SHA-256 must be non-null", label);
        return false;
    }
    size_t length = 0;
    while (length < DS4_BENCH_SEQUENCE_SHA256_HEX_SIZE &&
           value[length] != '\0') {
        length++;
    }
    if (length != DS4_BENCH_SEQUENCE_SHA256_HEX_LENGTH) {
        set_error(error, error_size,
                  "trusted %s SHA-256 must be a canonical 64-character digest",
                  label);
        return false;
    }
    bool all_zero = true;
    for (size_t i = 0; i < length; i++) {
        const unsigned char c = (unsigned char)value[i];
        const bool hex = (c >= '0' && c <= '9') ||
                         (c >= 'a' && c <= 'f');
        if (!hex) {
            set_error(error, error_size,
                      "trusted %s SHA-256 must be lowercase hexadecimal",
                      label);
            return false;
        }
        if (c != '0') all_zero = false;
    }
    if (all_zero) {
        set_error(error, error_size,
                  "trusted %s SHA-256 must not be all zero", label);
        return false;
    }
    return true;
}

static bool parse_sha256(const unsigned char *value,
                         size_t value_length,
                         char output[DS4_BENCH_SEQUENCE_SHA256_HEX_SIZE],
                         size_t line_number,
                         char *error,
                         size_t error_size) {
    if (value_length != DS4_BENCH_SEQUENCE_SHA256_HEX_LENGTH) {
        set_error(error, error_size,
                  "qualification sequence line %zu is not a SHA-256",
                  line_number);
        return false;
    }
    bool all_zero = true;
    for (size_t i = 0; i < value_length; i++) {
        const unsigned char c = value[i];
        const bool hex = (c >= '0' && c <= '9') ||
                         (c >= 'a' && c <= 'f');
        if (!hex) {
            set_error(error, error_size,
                      "qualification sequence line %zu is not lowercase SHA-256",
                      line_number);
            return false;
        }
        if (c != '0') all_zero = false;
    }
    if (all_zero) {
        set_error(error, error_size,
                  "qualification sequence line %zu uses an all-zero SHA-256",
                  line_number);
        return false;
    }
    memcpy(output, value, value_length);
    output[value_length] = '\0';
    return true;
}

static int base64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return (int)(c - 'A');
    if (c >= 'a' && c <= 'z') return (int)(c - 'a' + 26u);
    if (c >= '0' && c <= '9') return (int)(c - '0' + 52u);
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static bool decode_base64(const unsigned char *value,
                          size_t value_length,
                          uint64_t expected_size,
                          unsigned char **bytes_out,
                          size_t *size_out,
                          size_t line_number,
                          char *error,
                          size_t error_size) {
    if (value_length == 0 || (value_length & 3u) != 0) {
        set_error(error, error_size,
                  "qualification sequence line %zu is not padded base64",
                  line_number);
        return false;
    }
    size_t padding = 0;
    if (value[value_length - 1u] == '=') padding++;
    if (value_length >= 2u && value[value_length - 2u] == '=') padding++;
    if (padding > 2u) {
        set_error(error, error_size,
                  "qualification sequence line %zu has invalid base64 padding",
                  line_number);
        return false;
    }
    const size_t data_length = value_length - padding;
    for (size_t i = 0; i < data_length; i++) {
        if (base64_value(value[i]) < 0) {
            set_error(error, error_size,
                      "qualification sequence line %zu has invalid base64 alphabet",
                      line_number);
            return false;
        }
    }
    for (size_t i = data_length; i < value_length; i++) {
        if (value[i] != '=') {
            set_error(error, error_size,
                      "qualification sequence line %zu has invalid base64 padding",
                      line_number);
            return false;
        }
    }
    const size_t groups = value_length / 4u;
    if (groups > SIZE_MAX / 3u) {
        set_error(error, error_size,
                  "qualification sequence base64 size overflows memory bounds");
        return false;
    }
    const size_t decoded_size = groups * 3u - padding;
    if (decoded_size == 0 ||
        decoded_size > DS4_BENCH_SEQUENCE_MAX_INPUT_BYTES ||
        (uintmax_t)decoded_size != (uintmax_t)expected_size) {
        set_error(error, error_size,
                  "qualification sequence decoded input size does not match its bound");
        return false;
    }
    if (padding == 1u) {
        const int last = base64_value(value[value_length - 2u]);
        if (last < 0 || (last & 0x03) != 0) {
            set_error(error, error_size,
                      "qualification sequence line %zu is noncanonical base64",
                      line_number);
            return false;
        }
    } else if (padding == 2u) {
        const int last = base64_value(value[value_length - 3u]);
        if (last < 0 || (last & 0x0f) != 0) {
            set_error(error, error_size,
                      "qualification sequence line %zu is noncanonical base64",
                      line_number);
            return false;
        }
    }

    unsigned char *decoded = (unsigned char *)malloc(decoded_size);
    if (decoded == NULL) {
        set_error(error, error_size,
                  "allocate qualification sequence input: %s", strerror(errno));
        return false;
    }
    size_t output = 0;
    for (size_t group = 0; group < groups; group++) {
        const unsigned char *quad = value + group * 4u;
        const int a = base64_value(quad[0]);
        const int b = base64_value(quad[1]);
        const int c = quad[2] == '=' ? 0 : base64_value(quad[2]);
        const int d = quad[3] == '=' ? 0 : base64_value(quad[3]);
        if (a < 0 || b < 0 || c < 0 || d < 0) {
            free(decoded);
            set_error(error, error_size,
                      "qualification sequence line %zu has invalid base64 alphabet",
                      line_number);
            return false;
        }
        decoded[output++] = (unsigned char)((a << 2) | (b >> 4));
        if (output < decoded_size) {
            decoded[output++] = (unsigned char)((b << 4) | (c >> 2));
        }
        if (output < decoded_size) {
            decoded[output++] = (unsigned char)((c << 6) | d);
        }
    }
    if (output != decoded_size) {
        free(decoded);
        set_error(error, error_size,
                  "qualification sequence base64 decoder size mismatch");
        return false;
    }
    if (memchr(decoded, '\0', decoded_size) != NULL) {
        free(decoded);
        set_error(error, error_size,
                  "qualification sequence decoded input contains NUL");
        return false;
    }
    *bytes_out = decoded;
    *size_out = decoded_size;
    return true;
}

static bool parse_payload(const unsigned char *bytes,
                          size_t size,
                          const char *expected_manifest_sha256,
                          ds4_bench_sequence *result,
                          char *error,
                          size_t error_size) {
    ds4_bench_sequence_cursor cursor = {
        .bytes = bytes,
        .size = size,
        .offset = 0,
        .line_number = 0,
    };
    const unsigned char *line;
    size_t line_length;
    const unsigned char *value;
    size_t value_length;
    uint64_t parsed;
    const size_t expected_lines = DS4_BENCH_SEQUENCE_LINE_COUNT;

    if (!next_line(&cursor, &line, &line_length, error, error_size) ||
        !line_literal(line, line_length,
                      "schema=" DS4_BENCH_SEQUENCE_SCHEMA,
                      1u, error, error_size)) return false;

    if (!next_line(&cursor, &line, &line_length, error, error_size) ||
        !line_value(line, line_length, "manifest_sha256=", &value,
                    &value_length, 2u, error, error_size) ||
        !parse_sha256(value, value_length, result->manifest_sha256, 2u,
                      error, error_size)) return false;
    if (expected_manifest_sha256 != NULL &&
        memcmp(result->manifest_sha256,
               expected_manifest_sha256,
               DS4_BENCH_SEQUENCE_SHA256_HEX_LENGTH) != 0) {
        set_error(error, error_size,
                  "qualification sequence manifest SHA-256 does not match trusted digest");
        return false;
    }

    if (!next_line(&cursor, &line, &line_length, error, error_size) ||
        !line_value(line, line_length, "profile_id=", &value,
                    &value_length, 3u, error, error_size) ||
        !copy_string_value(value, value_length, result->profile_id,
                           sizeof(result->profile_id), 3u, error, error_size)) return false;

    static const char *const profiles[] = {
        "cache-8gib", "cache-12gib", "cache-16gib"
    };
    static const uint64_t caches[] = {
        UINT64_C(8589934592), UINT64_C(12884901888), UINT64_C(17179869184)
    };
    size_t profile_index = 0;
    while (profile_index < sizeof(profiles) / sizeof(profiles[0]) &&
           strcmp(result->profile_id, profiles[profile_index]) != 0) {
        profile_index++;
    }
    if (profile_index == sizeof(profiles) / sizeof(profiles[0])) {
        set_error(error, error_size,
                  "qualification sequence has an unknown profile_id");
        return false;
    }

    if (!next_line(&cursor, &line, &line_length, error, error_size) ||
        !line_value(line, line_length, "cache_bytes=", &value,
                    &value_length, 4u, error, error_size) ||
        !parse_uint64_decimal(value, value_length, true, &parsed, 4u,
                              error, error_size)) return false;
    if (parsed != caches[profile_index]) {
        set_error(error, error_size,
                  "qualification sequence cache_bytes does not match profile");
        return false;
    }
    result->cache_bytes = parsed;

    if (!next_line(&cursor, &line, &line_length, error, error_size) ||
        !line_value(line, line_length, "prompt_order_index=", &value,
                    &value_length, 5u, error, error_size) ||
        !parse_uint64_decimal(value, value_length, false, &parsed, 5u,
                              error, error_size) || parsed > 3u) {
        if (error == NULL || error_size == 0 || error[0] == '\0') {
            set_error(error, error_size,
                      "qualification sequence prompt_order_index is outside 0..3");
        }
        return false;
    }
    result->prompt_order_index = (uint32_t)parsed;

    static const uint32_t orders[][4] = {
        {512, 2048, 28672, 8192},
        {2048, 8192, 512, 28672},
        {8192, 28672, 2048, 512},
    };
    const uint32_t expected_tokens = orders[profile_index][result->prompt_order_index];
    char expected_prompt[32];
    (void)snprintf(expected_prompt, sizeof(expected_prompt),
                   "native-%u", expected_tokens);

    if (!next_line(&cursor, &line, &line_length, error, error_size) ||
        !line_value(line, line_length, "prompt_id=", &value,
                    &value_length, 6u, error, error_size) ||
        !copy_string_value(value, value_length, result->prompt_id,
                           sizeof(result->prompt_id), 6u, error, error_size)) return false;
    if (strcmp(result->prompt_id, expected_prompt) != 0) {
        set_error(error, error_size,
                  "qualification sequence prompt_id does not match profile order");
        return false;
    }

    if (!next_line(&cursor, &line, &line_length, error, error_size) ||
        !line_value(line, line_length, "prompt_tokens=", &value,
                    &value_length, 7u, error, error_size) ||
        !parse_uint64_decimal(value, value_length, true, &parsed, 7u,
                              error, error_size)) return false;
    if (parsed != expected_tokens) {
        set_error(error, error_size,
                  "qualification sequence prompt_tokens does not match profile order");
        return false;
    }
    result->prompt_tokens = (uint32_t)parsed;

    if (!next_line(&cursor, &line, &line_length, error, error_size) ||
        !line_literal(line, line_length, "mode=streamed", 8u,
                      error, error_size)) return false;

    if (!next_line(&cursor, &line, &line_length, error, error_size) ||
        !line_value(line, line_length, "input_size_bytes=", &value,
                    &value_length, 9u, error, error_size) ||
        !parse_uint64_decimal(value, value_length, true, &parsed, 9u,
                              error, error_size)) return false;
    if (parsed > DS4_BENCH_SEQUENCE_MAX_INPUT_BYTES) {
        set_error(error, error_size,
                  "qualification sequence input exceeds the %u-byte bound",
                  DS4_BENCH_SEQUENCE_MAX_INPUT_BYTES);
        return false;
    }
    result->input_size_bytes = parsed;

    if (!next_line(&cursor, &line, &line_length, error, error_size) ||
        !line_value(line, line_length, "input_sha256=", &value,
                    &value_length, 10u, error, error_size) ||
        !parse_sha256(value, value_length, result->input_sha256, 10u,
                      error, error_size)) return false;

    const unsigned char *base64_value_text;
    size_t base64_value_length;
    if (!next_line(&cursor, &line, &line_length, error, error_size) ||
        !line_value(line, line_length, "input_base64=", &base64_value_text,
                    &base64_value_length, 11u, error, error_size) ||
        !decode_base64(base64_value_text, base64_value_length,
                       result->input_size_bytes, &result->input_bytes,
                       &result->input_size, 11u, error, error_size)) return false;

    static const char *const fixed_lines[] = {
        "max_generated_tokens=512",
        "temperature=0",
        "top_k=0",
        "top_p=1",
        "min_p=0.05",
        "seed=1",
        "stop_sequences_count=0",
        "stop_token_policy=model-native",
        "repetition_count=4",
        "repetition=0:cold",
        "repetition=1:warm-1",
        "repetition=2:warm-2",
        "repetition=3:warm-3",
    };
    for (size_t index = 0; index < sizeof(fixed_lines) / sizeof(fixed_lines[0]); index++) {
        if (!next_line(&cursor, &line, &line_length, error, error_size) ||
            !line_literal(line, line_length, fixed_lines[index],
                          12u + index, error, error_size)) {
            ds4_bench_sequence_free(result);
            return false;
        }
    }
    if (cursor.line_number != expected_lines || cursor.offset != size) {
        ds4_bench_sequence_free(result);
        set_error(error, error_size,
                  "qualification sequence must contain exactly %u LF-terminated lines",
                  DS4_BENCH_SEQUENCE_LINE_COUNT);
        return false;
    }

    char observed_hash[DS4_BENCH_SEQUENCE_SHA256_HEX_SIZE];
    if (!ds4_plan_io_sha256(result->input_bytes,
                            result->input_size,
                            observed_hash,
                            error,
                            error_size)) {
        ds4_bench_sequence_free(result);
        return false;
    }
    if (strcmp(observed_hash, result->input_sha256) != 0) {
        ds4_bench_sequence_free(result);
        set_error(error, error_size,
                  "qualification sequence input SHA-256 does not match decoded bytes");
        return false;
    }
    return true;
}

static bool parse_file_impl(const char *path,
                            const char *expected_manifest_sha256,
                            const char *expected_sequence_sha256,
                            ds4_bench_sequence *sequence,
                            char *error,
                            size_t error_size) {
    unsigned char *bytes = NULL;
    size_t size = 0;
    if (!read_sequence_file(path, &bytes, &size, error, error_size)) {
        return false;
    }

    char sequence_sha256[DS4_BENCH_SEQUENCE_SHA256_HEX_SIZE];
    if (!ds4_plan_io_sha256(bytes,
                            size,
                            sequence_sha256,
                            error,
                            error_size)) {
        free(bytes);
        return false;
    }
    if (expected_sequence_sha256 != NULL &&
        memcmp(sequence_sha256,
               expected_sequence_sha256,
               DS4_BENCH_SEQUENCE_SHA256_HEX_LENGTH) != 0) {
        free(bytes);
        set_error(error, error_size,
                  "qualification sequence SHA-256 does not match trusted digest");
        return false;
    }

    ds4_bench_sequence parsed = {0};
    const bool ok = parse_payload(bytes,
                                  size,
                                  expected_manifest_sha256,
                                  &parsed,
                                  error,
                                  error_size);
    free(bytes);
    if (!ok) {
        ds4_bench_sequence_free(&parsed);
        return false;
    }
    memcpy(parsed.sequence_sha256,
           sequence_sha256,
           sizeof(parsed.sequence_sha256));
    *sequence = parsed;
    return true;
}

bool ds4_bench_sequence_parse_file(const char *path,
                                   ds4_bench_sequence *sequence,
                                   char *error,
                                   size_t error_size) {
    clear_error(error, error_size);
    if (sequence == NULL) {
        set_error(error, error_size,
                  "qualification sequence output is null");
        return false;
    }
    ds4_bench_sequence_free(sequence);
    return parse_file_impl(path,
                           NULL,
                           NULL,
                           sequence,
                           error,
                           error_size);
}

bool ds4_bench_sequence_parse_file_trusted(
        const char *path,
        const char *expected_manifest_sha256,
        const char *expected_sequence_sha256,
        ds4_bench_sequence *sequence,
        char *error,
        size_t error_size) {
    clear_error(error, error_size);
    if (sequence == NULL) {
        set_error(error, error_size,
                  "qualification sequence output is null");
        return false;
    }
    ds4_bench_sequence_free(sequence);
    if (!validate_expected_sha256(expected_manifest_sha256,
                                  "manifest",
                                  error,
                                  error_size) ||
        !validate_expected_sha256(expected_sequence_sha256,
                                  "sequence",
                                  error,
                                  error_size)) {
        return false;
    }
    return parse_file_impl(path,
                           expected_manifest_sha256,
                           expected_sequence_sha256,
                           sequence,
                           error,
                           error_size);
}
