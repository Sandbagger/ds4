#if defined(__linux__)
#define _GNU_SOURCE
#elif defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L

#include "ds4_plan_io.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define DS4_PLAN_IO_SIDECAR_SUFFIX ".sha256"
#define DS4_PLAN_IO_TEMP_SUFFIX ".tmp.XXXXXX"

typedef struct {
    uint32_t state[8];
    uint64_t total_bytes;
    unsigned char block[64];
    size_t block_size;
} ds4_plan_sha256_context;

#ifdef DS4_PLAN_IO_TEST_HOOKS
static ds4_plan_io_test_after_first_pread_hook
    ds4_plan_io_after_first_pread_hook;
static void *ds4_plan_io_after_first_pread_context;

void ds4_plan_io_test_set_after_first_pread_hook(
        ds4_plan_io_test_after_first_pread_hook hook,
        void *context) {
    ds4_plan_io_after_first_pread_hook = hook;
    ds4_plan_io_after_first_pread_context = context;
}
#endif

static const uint32_t ds4_plan_sha256_constants[64] = {
    UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf),
    UINT32_C(0xe9b5dba5), UINT32_C(0x3956c25b), UINT32_C(0x59f111f1),
    UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5), UINT32_C(0xd807aa98),
    UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
    UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7),
    UINT32_C(0xc19bf174), UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786),
    UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc), UINT32_C(0x2de92c6f),
    UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
    UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8),
    UINT32_C(0xbf597fc7), UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
    UINT32_C(0x06ca6351), UINT32_C(0x14292967), UINT32_C(0x27b70a85),
    UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
    UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e),
    UINT32_C(0x92722c85), UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b),
    UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3), UINT32_C(0xd192e819),
    UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
    UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c),
    UINT32_C(0x34b0bcb5), UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a),
    UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3), UINT32_C(0x748f82ee),
    UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
    UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7),
    UINT32_C(0xc67178f2),
};

static void ds4_plan_clear_error(char *error, size_t error_size) {
    if (error != NULL && error_size != 0) error[0] = '\0';
}

static void ds4_plan_set_error(char *error,
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

static uint32_t ds4_plan_rotr32(uint32_t value, unsigned int count) {
    return (value >> count) | (value << (32u - count));
}

static uint32_t ds4_plan_load_be32(const unsigned char *bytes) {
    return ((uint32_t)bytes[0] << 24u) |
           ((uint32_t)bytes[1] << 16u) |
           ((uint32_t)bytes[2] << 8u) |
           (uint32_t)bytes[3];
}

static void ds4_plan_sha256_transform(ds4_plan_sha256_context *context,
                                      const unsigned char block[64]) {
    uint32_t words[64];
    for (size_t i = 0; i < 16; i++) {
        words[i] = ds4_plan_load_be32(block + i * 4u);
    }
    for (size_t i = 16; i < 64; i++) {
        const uint32_t x = words[i - 15];
        const uint32_t y = words[i - 2];
        const uint32_t sigma0 = ds4_plan_rotr32(x, 7u) ^
                                ds4_plan_rotr32(x, 18u) ^ (x >> 3u);
        const uint32_t sigma1 = ds4_plan_rotr32(y, 17u) ^
                                ds4_plan_rotr32(y, 19u) ^ (y >> 10u);
        words[i] = words[i - 16] + sigma0 + words[i - 7] + sigma1;
    }

    uint32_t a = context->state[0];
    uint32_t b = context->state[1];
    uint32_t c = context->state[2];
    uint32_t d = context->state[3];
    uint32_t e = context->state[4];
    uint32_t f = context->state[5];
    uint32_t g = context->state[6];
    uint32_t h = context->state[7];
    for (size_t i = 0; i < 64; i++) {
        const uint32_t big_sigma1 = ds4_plan_rotr32(e, 6u) ^
                                    ds4_plan_rotr32(e, 11u) ^
                                    ds4_plan_rotr32(e, 25u);
        const uint32_t choose = (e & f) ^ ((~e) & g);
        const uint32_t temporary1 = h + big_sigma1 + choose +
                                    ds4_plan_sha256_constants[i] + words[i];
        const uint32_t big_sigma0 = ds4_plan_rotr32(a, 2u) ^
                                    ds4_plan_rotr32(a, 13u) ^
                                    ds4_plan_rotr32(a, 22u);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temporary2 = big_sigma0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }

    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void ds4_plan_sha256_init(ds4_plan_sha256_context *context) {
    static const uint32_t initial_state[8] = {
        UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
        UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
        UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
        UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19),
    };
    memcpy(context->state, initial_state, sizeof(initial_state));
    context->total_bytes = 0;
    context->block_size = 0;
}

static void ds4_plan_sha256_update(ds4_plan_sha256_context *context,
                                   const unsigned char *bytes,
                                   size_t size) {
    context->total_bytes += (uint64_t)size;
    while (size != 0) {
        size_t available = sizeof(context->block) - context->block_size;
        if (available > size) available = size;
        memcpy(context->block + context->block_size, bytes, available);
        context->block_size += available;
        bytes += available;
        size -= available;
        if (context->block_size == sizeof(context->block)) {
            ds4_plan_sha256_transform(context, context->block);
            context->block_size = 0;
        }
    }
}

static void ds4_plan_sha256_final(ds4_plan_sha256_context *context,
                                  unsigned char digest[32]) {
    const uint64_t bit_length = context->total_bytes * UINT64_C(8);
    context->block[context->block_size++] = 0x80u;
    if (context->block_size > 56u) {
        memset(context->block + context->block_size,
               0,
               sizeof(context->block) - context->block_size);
        ds4_plan_sha256_transform(context, context->block);
        context->block_size = 0;
    }
    memset(context->block + context->block_size, 0, 56u - context->block_size);
    for (size_t i = 0; i < 8; i++) {
        context->block[63u - i] = (unsigned char)(bit_length >> (i * 8u));
    }
    ds4_plan_sha256_transform(context, context->block);
    for (size_t i = 0; i < 8; i++) {
        digest[i * 4u] = (unsigned char)(context->state[i] >> 24u);
        digest[i * 4u + 1u] = (unsigned char)(context->state[i] >> 16u);
        digest[i * 4u + 2u] = (unsigned char)(context->state[i] >> 8u);
        digest[i * 4u + 3u] = (unsigned char)context->state[i];
    }
}

bool ds4_plan_io_sha256(const void *data,
                        size_t size,
                        char digest_hex[DS4_PLAN_IO_SHA256_HEX_SIZE],
                        char *error,
                        size_t error_size) {
    static const char hex[] = "0123456789abcdef";
    ds4_plan_clear_error(error, error_size);
    if (digest_hex == NULL) {
        ds4_plan_set_error(error,
                           error_size,
                           "invalid argument: digest output is null");
        return false;
    }
    digest_hex[0] = '\0';
    if (data == NULL && size != 0) {
        ds4_plan_set_error(error,
                           error_size,
                           "invalid argument: data is null with nonzero size");
        return false;
    }
    if ((uintmax_t)size > UINT64_MAX / UINT64_C(8)) {
        ds4_plan_set_error(error,
                           error_size,
                           "invalid argument: data length exceeds SHA-256 limit");
        return false;
    }

    ds4_plan_sha256_context context;
    unsigned char digest[32];
    ds4_plan_sha256_init(&context);
    if (size != 0) {
        ds4_plan_sha256_update(&context, (const unsigned char *)data, size);
    }
    ds4_plan_sha256_final(&context, digest);
    for (size_t i = 0; i < sizeof(digest); i++) {
        digest_hex[i * 2u] = hex[digest[i] >> 4u];
        digest_hex[i * 2u + 1u] = hex[digest[i] & 0x0fu];
    }
    digest_hex[DS4_PLAN_IO_SHA256_HEX_LENGTH] = '\0';
    return true;
}

static bool ds4_plan_stat_identity_matches(const struct stat *before,
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

bool ds4_plan_io_sha256_fd(
        int fd,
        uint64_t expected_size,
        char out[DS4_PLAN_IO_SHA256_HEX_SIZE],
        char *err,
        size_t errcap) {
    static const char hex[] = "0123456789abcdef";
    ds4_plan_clear_error(err, errcap);
    if (out == NULL) {
        ds4_plan_set_error(err, errcap,
                           "invalid argument: digest output is null");
        return false;
    }
    out[0] = '\0';
    if (fd < 0) {
        ds4_plan_set_error(err, errcap,
                           "invalid argument: file descriptor is invalid");
        return false;
    }
    if (expected_size > UINT64_MAX / UINT64_C(8)) {
        ds4_plan_set_error(err, errcap,
                           "invalid argument: file exceeds SHA-256 limit");
        return false;
    }
    const off_t expected_offset = (off_t)expected_size;
    if (expected_offset < 0 || (uint64_t)expected_offset != expected_size) {
        ds4_plan_set_error(err, errcap,
                           "invalid argument: file size exceeds pread range");
        return false;
    }

    struct stat before;
    if (fstat(fd, &before) != 0) {
        const int saved_errno = errno;
        ds4_plan_set_error(err, errcap, "stat opened file: %s",
                           strerror(saved_errno));
        return false;
    }
    if (!S_ISREG(before.st_mode)) {
        ds4_plan_set_error(err, errcap,
                           "opened descriptor is not a regular file");
        return false;
    }
    if (before.st_size < 0 ||
        (uint64_t)before.st_size != expected_size) {
        ds4_plan_set_error(err, errcap,
                           "opened file size does not match expected size");
        return false;
    }

    ds4_plan_sha256_context context;
    ds4_plan_sha256_init(&context);
    unsigned char buffer[64u * 1024u];
    uint64_t offset = 0;
#ifdef DS4_PLAN_IO_TEST_HOOKS
    bool first_pread_hook_ran = false;
#endif
    while (offset != expected_size) {
        size_t chunk = sizeof(buffer);
        const uint64_t remaining = expected_size - offset;
        if (remaining < (uint64_t)chunk) chunk = (size_t)remaining;
        const off_t read_offset = (off_t)offset;
        ssize_t count;
        do {
            count = pread(fd, buffer, chunk, read_offset);
        } while (count < 0 && errno == EINTR);
        if (count < 0) {
            const int saved_errno = errno;
            ds4_plan_set_error(err, errcap, "read opened file: %s",
                               strerror(saved_errno));
            return false;
        }
        if (count == 0) {
            ds4_plan_set_error(err, errcap,
                               "opened file was truncated while hashing");
            return false;
        }
        ds4_plan_sha256_update(&context, buffer, (size_t)count);
        offset += (uint64_t)count;
#ifdef DS4_PLAN_IO_TEST_HOOKS
        if (!first_pread_hook_ran) {
            first_pread_hook_ran = true;
            if (ds4_plan_io_after_first_pread_hook != NULL) {
                ds4_plan_io_after_first_pread_hook(
                    fd, ds4_plan_io_after_first_pread_context);
            }
        }
#endif
    }

    unsigned char extra;
    ssize_t extra_count;
    do {
        extra_count = pread(fd, &extra, 1u, expected_offset);
    } while (extra_count < 0 && errno == EINTR);
    if (extra_count < 0) {
        const int saved_errno = errno;
        ds4_plan_set_error(err, errcap, "verify opened file length: %s",
                           strerror(saved_errno));
        return false;
    }
    if (extra_count != 0) {
        ds4_plan_set_error(err, errcap,
                           "opened file grew while hashing");
        return false;
    }

    struct stat after;
    if (fstat(fd, &after) != 0) {
        const int saved_errno = errno;
        ds4_plan_set_error(err, errcap, "restat opened file: %s",
                           strerror(saved_errno));
        return false;
    }
    if (!ds4_plan_stat_identity_matches(&before, &after)) {
        ds4_plan_set_error(err, errcap,
                           "opened file identity changed while hashing");
        return false;
    }

    unsigned char digest[32];
    ds4_plan_sha256_final(&context, digest);
    for (size_t i = 0; i < sizeof(digest); i++) {
        out[i * 2u] = hex[digest[i] >> 4u];
        out[i * 2u + 1u] = hex[digest[i] & 0x0fu];
    }
    out[DS4_PLAN_IO_SHA256_HEX_LENGTH] = '\0';
    return true;
}

static bool ds4_plan_path_length(const char *path,
                                 size_t *length_out,
                                 char *error,
                                 size_t error_size) {
    if (path == NULL) {
        ds4_plan_set_error(error,
                           error_size,
                           "invalid argument: plan path is null");
        return false;
    }
    size_t length = 0;
    while (length <= DS4_PLAN_IO_PATH_LIMIT && path[length] != '\0') length++;
    if (length == 0) {
        ds4_plan_set_error(error,
                           error_size,
                           "invalid argument: plan path is empty");
        return false;
    }
    const size_t appended = sizeof(DS4_PLAN_IO_SIDECAR_SUFFIX) - 1u +
                            sizeof(DS4_PLAN_IO_TEMP_SUFFIX) - 1u;
    if (length > DS4_PLAN_IO_PATH_LIMIT ||
        length > SIZE_MAX - appended - 1u ||
        length + appended > DS4_PLAN_IO_PATH_LIMIT) {
        ds4_plan_set_error(error,
                           error_size,
                           "invalid argument: plan path is too long");
        return false;
    }
    *length_out = length;
    return true;
}

static char *ds4_plan_append(const char *path,
                             size_t path_length,
                             const char *suffix,
                             size_t suffix_length,
                             char *error,
                             size_t error_size) {
    if (path_length > SIZE_MAX - suffix_length - 1u) {
        ds4_plan_set_error(error,
                           error_size,
                           "invalid argument: plan path length overflow");
        return NULL;
    }
    char *result = malloc(path_length + suffix_length + 1u);
    if (result == NULL) {
        ds4_plan_set_error(error,
                           error_size,
                           "allocate publication path: %s",
                           strerror(errno));
        return NULL;
    }
    memcpy(result, path, path_length);
    memcpy(result + path_length, suffix, suffix_length);
    result[path_length + suffix_length] = '\0';
    return result;
}

static bool ds4_plan_target_absent(const char *path,
                                   const char *kind,
                                   char *error,
                                   size_t error_size) {
    struct stat status;
    if (lstat(path, &status) == 0) {
        ds4_plan_set_error(error,
                           error_size,
                           "refusing to replace existing %s '%s'",
                           kind,
                           path);
        return false;
    }
    if (errno != ENOENT) {
        const int saved_errno = errno;
        ds4_plan_set_error(error,
                           error_size,
                           "inspect %s target '%s': %s",
                           kind,
                           path,
                           strerror(saved_errno));
        return false;
    }
    return true;
}

static bool ds4_plan_write_all(int fd,
                               const unsigned char *bytes,
                               size_t size) {
    size_t offset = 0;
    while (offset != size) {
        size_t chunk = size - offset;
        if (chunk > (size_t)UINT32_MAX) chunk = (size_t)UINT32_MAX;
        const ssize_t written = write(fd, bytes + offset, chunk);
        if (written > 0) {
            offset += (size_t)written;
            continue;
        }
        if (written < 0 && errno == EINTR) continue;
        if (written == 0) errno = EIO;
        return false;
    }
    return true;
}

static bool ds4_plan_fsync(int fd) {
    for (;;) {
        if (fsync(fd) == 0) return true;
        if (errno != EINTR) return false;
    }
}

static bool ds4_plan_unlink(const char *path) {
    for (;;) {
        if (unlink(path) == 0) return true;
        if (errno == ENOENT) return true;
        if (errno != EINTR) return false;
    }
}

static char *ds4_plan_parent_directory(const char *path,
                                       char *error,
                                       size_t error_size) {
    const char *slash = strrchr(path, '/');
    if (slash == NULL) {
        char *parent = malloc(2u);
        if (parent != NULL) memcpy(parent, ".", 2u);
        if (parent == NULL) {
            ds4_plan_set_error(error,
                               error_size,
                               "allocate parent-directory path: %s",
                               strerror(errno));
        }
        return parent;
    }
    const size_t length = slash == path ? 1u : (size_t)(slash - path);
    char *parent = malloc(length + 1u);
    if (parent == NULL) {
        ds4_plan_set_error(error,
                           error_size,
                           "allocate parent-directory path: %s",
                           strerror(errno));
        return NULL;
    }
    memcpy(parent, path, length);
    parent[length] = '\0';
    return parent;
}

static bool ds4_plan_probe_temporary(const char *target,
                                     char *temporary,
                                     const char *kind,
                                     char *error,
                                     size_t error_size) {
    const int fd = mkstemp(temporary);
    if (fd < 0) {
        const int saved_errno = errno;
        ds4_plan_set_error(error,
                           error_size,
                           "create temporary %s for '%s': %s",
                           kind,
                           target,
                           strerror(saved_errno));
        return false;
    }
    if (close(fd) != 0) {
        const int saved_errno = errno;
        (void)ds4_plan_unlink(temporary);
        ds4_plan_set_error(error,
                           error_size,
                           "close temporary %s probe for '%s': %s",
                           kind,
                           target,
                           strerror(saved_errno));
        return false;
    }
    if (!ds4_plan_unlink(temporary)) {
        const int saved_errno = errno;
        ds4_plan_set_error(error,
                           error_size,
                           "remove temporary %s probe for '%s': %s",
                           kind,
                           target,
                           strerror(saved_errno));
        return false;
    }
    return true;
}

bool ds4_plan_io_preflight_target(const char *path,
                                  char *error,
                                  size_t error_size) {
    ds4_plan_clear_error(error, error_size);

    size_t path_length = 0;
    if (!ds4_plan_path_length(path, &path_length, error, error_size)) {
        return false;
    }
    char *sidecar = ds4_plan_append(
        path,
        path_length,
        DS4_PLAN_IO_SIDECAR_SUFFIX,
        sizeof(DS4_PLAN_IO_SIDECAR_SUFFIX) - 1u,
        error,
        error_size);
    if (sidecar == NULL) return false;

    const size_t temporary_suffix_length =
        sizeof(DS4_PLAN_IO_TEMP_SUFFIX) - 1u;
    char *plan_temporary = ds4_plan_append(
        path,
        path_length,
        DS4_PLAN_IO_TEMP_SUFFIX,
        temporary_suffix_length,
        error,
        error_size);
    char *sidecar_temporary = NULL;
    if (plan_temporary != NULL) {
        sidecar_temporary = ds4_plan_append(
            sidecar,
            path_length + sizeof(DS4_PLAN_IO_SIDECAR_SUFFIX) - 1u,
            DS4_PLAN_IO_TEMP_SUFFIX,
            temporary_suffix_length,
            error,
            error_size);
    }
    if (plan_temporary == NULL || sidecar_temporary == NULL) {
        free(sidecar_temporary);
        free(plan_temporary);
        free(sidecar);
        return false;
    }

    bool ok = ds4_plan_target_absent(path, "plan", error, error_size) &&
              ds4_plan_target_absent(
                  sidecar, "digest sidecar", error, error_size);
    if (!ok) goto cleanup;

    char *parent = ds4_plan_parent_directory(path, error, error_size);
    if (parent == NULL) {
        ok = false;
        goto cleanup;
    }
    struct stat status;
    if (stat(parent, &status) != 0) {
        const int saved_errno = errno;
        ds4_plan_set_error(error,
                           error_size,
                           "inspect parent directory '%s': %s",
                           parent,
                           strerror(saved_errno));
        free(parent);
        ok = false;
        goto cleanup;
    }
    if (!S_ISDIR(status.st_mode)) {
        ds4_plan_set_error(error,
                           error_size,
                           "qualification-plan parent '%s' is not a directory",
                           parent);
        free(parent);
        ok = false;
        goto cleanup;
    }
    free(parent);

    ok = ds4_plan_probe_temporary(
             path, plan_temporary, "plan", error, error_size) &&
         ds4_plan_probe_temporary(sidecar,
                                  sidecar_temporary,
                                  "digest sidecar",
                                  error,
                                  error_size);

cleanup:
    free(sidecar_temporary);
    free(plan_temporary);
    free(sidecar);
    return ok;
}

static bool ds4_plan_sync_parent(const char *target,
                                 const char *kind,
                                 char *error,
                                 size_t error_size) {
    char *parent = ds4_plan_parent_directory(target, error, error_size);
    if (parent == NULL) return false;
    const int fd = open(parent, O_RDONLY);
    if (fd < 0) {
        const int saved_errno = errno;
        ds4_plan_set_error(error,
                           error_size,
                           "open parent directory for %s '%s': %s",
                           kind,
                           target,
                           strerror(saved_errno));
        free(parent);
        return false;
    }
    if (!ds4_plan_fsync(fd)) {
        const int saved_errno = errno;
        (void)close(fd);
        ds4_plan_set_error(error,
                           error_size,
                           "synchronize parent directory for %s '%s': %s",
                           kind,
                           target,
                           strerror(saved_errno));
        free(parent);
        return false;
    }
    if (close(fd) != 0) {
        const int saved_errno = errno;
        ds4_plan_set_error(error,
                           error_size,
                           "close parent directory for %s '%s': %s",
                           kind,
                           target,
                           strerror(saved_errno));
        free(parent);
        return false;
    }
    free(parent);
    return true;
}

static bool ds4_plan_stage_one(const char *target,
                               char *temporary,
                               const unsigned char *bytes,
                               size_t size,
                               const char *kind,
                               struct stat *identity,
                               char *error,
                               size_t error_size) {
    int fd = mkstemp(temporary);
    if (fd < 0) {
        const int saved_errno = errno;
        ds4_plan_set_error(error,
                           error_size,
                           "create temporary %s for '%s': %s",
                           kind,
                           target,
                           strerror(saved_errno));
        return false;
    }

    if (!ds4_plan_write_all(fd, bytes, size)) {
        const int saved_errno = errno;
        (void)close(fd);
        (void)ds4_plan_unlink(temporary);
        ds4_plan_set_error(error,
                           error_size,
                           "write temporary %s for '%s': %s",
                           kind,
                           target,
                           strerror(saved_errno));
        return false;
    }
    if (!ds4_plan_fsync(fd)) {
        const int saved_errno = errno;
        (void)close(fd);
        (void)ds4_plan_unlink(temporary);
        ds4_plan_set_error(error,
                           error_size,
                           "synchronize temporary %s for '%s': %s",
                           kind,
                           target,
                           strerror(saved_errno));
        return false;
    }
    if (fstat(fd, identity) != 0) {
        const int saved_errno = errno;
        (void)close(fd);
        (void)ds4_plan_unlink(temporary);
        ds4_plan_set_error(error,
                           error_size,
                           "inspect temporary %s for '%s': %s",
                           kind,
                           target,
                           strerror(saved_errno));
        return false;
    }
    if (close(fd) != 0) {
        const int saved_errno = errno;
        (void)ds4_plan_unlink(temporary);
        ds4_plan_set_error(error,
                           error_size,
                           "close temporary %s for '%s': %s",
                           kind,
                           target,
                           strerror(saved_errno));
        return false;
    }
    return true;
}

static int ds4_plan_rename_noreplace(const char *temporary,
                                     const char *target) {
#if defined(__APPLE__)
    if (renamex_np(temporary, target, RENAME_EXCL) == 0) return 0;
    if (errno != ENOTSUP && errno != EINVAL) return -1;
#elif defined(__linux__)
    if (renameat2(AT_FDCWD,
                  temporary,
                  AT_FDCWD,
                  target,
                  RENAME_NOREPLACE) == 0) {
        return 0;
    }
    if (errno != ENOSYS && errno != ENOTSUP && errno != EOPNOTSUPP &&
        errno != EINVAL) {
        return -1;
    }
#endif

    /* The mkstemp file and target are in the same directory, so hard-linking
     * gives the fallback the same atomic no-clobber property.  The common
     * cleanup path removes the now-redundant temporary link. */
    return link(temporary, target);
}

static bool ds4_plan_commit_one(const char *target,
                                const char *temporary,
                                const char *kind,
                                char *error,
                                size_t error_size) {
    if (ds4_plan_rename_noreplace(temporary, target) != 0) {
        const int saved_errno = errno;
        if (saved_errno == EEXIST) {
            ds4_plan_set_error(error,
                               error_size,
                               "refusing to replace existing %s '%s'",
                               kind,
                               target);
        } else {
            ds4_plan_set_error(error,
                               error_size,
                               "publish %s '%s': %s",
                               kind,
                               target,
                               strerror(saved_errno));
        }
        return false;
    }
    return true;
}

static bool ds4_plan_remove_if_identity(const char *path,
                                        const struct stat *identity) {
    struct stat current;
    if (lstat(path, &current) != 0) return errno == ENOENT;
    if (current.st_dev != identity->st_dev ||
        current.st_ino != identity->st_ino) {
        return false;
    }
    return ds4_plan_unlink(path);
}

bool ds4_plan_io_publish_checked(
        const char *path,
        const void *bytes,
        size_t size,
        ds4_plan_io_validation_callback validate,
        void *validation_context,
        char digest_hex[DS4_PLAN_IO_SHA256_HEX_SIZE],
        char *error,
        size_t error_size) {
    ds4_plan_clear_error(error, error_size);
    if (digest_hex != NULL) digest_hex[0] = '\0';

    size_t path_length = 0;
    if (!ds4_plan_path_length(path, &path_length, error, error_size)) {
        return false;
    }
    if (digest_hex == NULL) {
        ds4_plan_set_error(error,
                           error_size,
                           "invalid argument: digest output is null");
        return false;
    }
    if (bytes == NULL && size != 0) {
        ds4_plan_set_error(error,
                           error_size,
                           "invalid argument: data is null with nonzero size");
        return false;
    }
    if ((uintmax_t)size > UINT64_MAX / UINT64_C(8)) {
        ds4_plan_set_error(error,
                           error_size,
                           "invalid argument: data length exceeds SHA-256 limit");
        return false;
    }

    const size_t sidecar_suffix_length =
        sizeof(DS4_PLAN_IO_SIDECAR_SUFFIX) - 1u;
    const size_t temporary_suffix_length =
        sizeof(DS4_PLAN_IO_TEMP_SUFFIX) - 1u;
    char *sidecar = ds4_plan_append(path,
                                    path_length,
                                    DS4_PLAN_IO_SIDECAR_SUFFIX,
                                    sidecar_suffix_length,
                                    error,
                                    error_size);
    char *plan_temporary = NULL;
    char *sidecar_temporary = NULL;
    bool success = false;
    if (sidecar == NULL) goto cleanup;
    plan_temporary = ds4_plan_append(path,
                                     path_length,
                                     DS4_PLAN_IO_TEMP_SUFFIX,
                                     temporary_suffix_length,
                                     error,
                                     error_size);
    if (plan_temporary == NULL) goto cleanup;
    sidecar_temporary = ds4_plan_append(
        sidecar,
        path_length + sidecar_suffix_length,
        DS4_PLAN_IO_TEMP_SUFFIX,
        temporary_suffix_length,
        error,
        error_size);
    if (sidecar_temporary == NULL) goto cleanup;

    if (!ds4_plan_target_absent(path, "plan", error, error_size) ||
        !ds4_plan_target_absent(sidecar,
                                "digest sidecar",
                                error,
                                error_size)) {
        goto cleanup;
    }

    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    if (!ds4_plan_io_sha256(bytes, size, digest, error, error_size)) goto cleanup;
    unsigned char sidecar_bytes[DS4_PLAN_IO_SHA256_HEX_LENGTH + 1u];
    memcpy(sidecar_bytes, digest, DS4_PLAN_IO_SHA256_HEX_LENGTH);
    sidecar_bytes[DS4_PLAN_IO_SHA256_HEX_LENGTH] = '\n';

    struct stat plan_identity;
    struct stat sidecar_identity;
    bool plan_committed = false;
    bool sidecar_committed = false;
    if (!ds4_plan_stage_one(path,
                            plan_temporary,
                            (const unsigned char *)bytes,
                            size,
                            "plan",
                            &plan_identity,
                            error,
                            error_size)) {
        goto cleanup;
    }
    if (!ds4_plan_stage_one(sidecar,
                            sidecar_temporary,
                            sidecar_bytes,
                            sizeof(sidecar_bytes),
                            "digest sidecar",
                            &sidecar_identity,
                            error,
                            error_size)) {
        goto rollback;
    }
    if (validate != NULL) {
        if (!validate(validation_context, error, error_size)) {
            if (error != NULL && error_size != 0 && error[0] == '\0') {
                ds4_plan_set_error(error,
                                   error_size,
                                   "qualification-plan validation rejected publication");
            }
            goto rollback;
        }
        ds4_plan_clear_error(error, error_size);
    }
    if (!ds4_plan_commit_one(path,
                             plan_temporary,
                             "plan",
                             error,
                             error_size)) {
        goto rollback;
    }
    plan_committed = true;
    if (!ds4_plan_commit_one(sidecar,
                             sidecar_temporary,
                             "digest sidecar",
                             error,
                             error_size)) {
        goto rollback;
    }
    sidecar_committed = true;

    if (!ds4_plan_unlink(plan_temporary) ||
        !ds4_plan_unlink(sidecar_temporary)) {
        const int saved_errno = errno;
        ds4_plan_set_error(error,
                           error_size,
                           "remove qualification-plan temporary link: %s",
                           strerror(saved_errno));
        goto rollback;
    }
    if (!ds4_plan_sync_parent(path,
                              "qualification plan pair",
                              error,
                              error_size)) {
        goto rollback;
    }

    memcpy(digest_hex, digest, sizeof(digest));
    success = true;
    goto cleanup;

rollback: {
        char ignored_error[1];
        if (sidecar_temporary != NULL) {
            (void)ds4_plan_unlink(sidecar_temporary);
        }
        if (plan_temporary != NULL) {
            (void)ds4_plan_unlink(plan_temporary);
        }
        if (sidecar_committed) {
            (void)ds4_plan_remove_if_identity(sidecar, &sidecar_identity);
        }
        if (plan_committed) {
            (void)ds4_plan_remove_if_identity(path, &plan_identity);
        }
        if (sidecar_committed || plan_committed) {
            (void)ds4_plan_sync_parent(path,
                                       "qualification plan rollback",
                                       ignored_error,
                                       sizeof(ignored_error));
        }
    }

cleanup:
    if (!success) {
        if (sidecar_temporary != NULL) {
            (void)ds4_plan_unlink(sidecar_temporary);
        }
        if (plan_temporary != NULL) {
            (void)ds4_plan_unlink(plan_temporary);
        }
    }
    free(sidecar_temporary);
    free(plan_temporary);
    free(sidecar);
    return success;
}

bool ds4_plan_io_publish(const char *path,
                         const void *bytes,
                         size_t size,
                         char digest_hex[DS4_PLAN_IO_SHA256_HEX_SIZE],
                         char *error,
                         size_t error_size) {
    return ds4_plan_io_publish_checked(path,
                                       bytes,
                                       size,
                                       NULL,
                                       NULL,
                                       digest_hex,
                                       error,
                                       error_size);
}
