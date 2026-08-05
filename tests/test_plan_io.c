#define _POSIX_C_SOURCE 200809L

#include "../ds4_plan_io.h"

#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
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

#define CHECK(cond, msg) do {                                                  \
    g_total++;                                                                 \
    if (!(cond)) {                                                             \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);             \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

static bool make_path(char *out,
                      size_t out_size,
                      const char *dir,
                      const char *name) {
    const int n = snprintf(out, out_size, "%s/%s", dir, name);
    return n >= 0 && (size_t)n < out_size;
}

static bool write_fixture(const char *path, const void *bytes, size_t size) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) return false;
    const bool wrote = size == 0 || fwrite(bytes, 1, size, file) == size;
    return fclose(file) == 0 && wrote;
}

static unsigned char *read_fixture(const char *path, size_t *size_out) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) return NULL;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    const long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    const size_t size = (size_t)length;
    unsigned char *bytes = malloc(size == 0 ? 1 : size);
    if (bytes == NULL) {
        fclose(file);
        return NULL;
    }
    if (size != 0 && fread(bytes, 1, size, file) != size) {
        free(bytes);
        fclose(file);
        return NULL;
    }
    if (fclose(file) != 0) {
        free(bytes);
        return NULL;
    }
    *size_out = size;
    return bytes;
}

static int count_temporary_files(const char *dir) {
    DIR *stream = opendir(dir);
    if (stream == NULL) return -1;
    int count = 0;
    for (;;) {
        errno = 0;
        struct dirent *entry = readdir(stream);
        if (entry == NULL) {
            if (errno != 0) count = -1;
            break;
        }
        if (strstr(entry->d_name, ".tmp.") != NULL) count++;
    }
    if (closedir(stream) != 0) return -1;
    return count;
}

static bool make_temporary_directory(char *path_template) {
    const int fd = mkstemp(path_template);
    if (fd < 0) return false;
    if (close(fd) != 0) {
        (void)unlink(path_template);
        return false;
    }
    if (unlink(path_template) != 0) return false;
    return mkdir(path_template, 0700) == 0;
}

static void test_sha256_vectors(void) {
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    char error[256] = "stale";

    CHECK(ds4_plan_io_sha256(NULL, 0, digest, error, sizeof(error)),
          "SHA-256 accepts an empty byte sequence");
    CHECK(strcmp(digest,
                 "e3b0c44298fc1c149afbf4c8996fb924"
                 "27ae41e4649b934ca495991b7852b855") == 0,
          "SHA-256 empty vector is exact");
    CHECK(error[0] == '\0', "successful SHA-256 clears stale error text");

    CHECK(ds4_plan_io_sha256("abc", 3, digest, error, sizeof(error)),
          "SHA-256 accepts ordinary bytes");
    CHECK(strcmp(digest,
                 "ba7816bf8f01cfea414140de5dae2223"
                 "b00361a396177a9cb410ff61f20015ad") == 0,
          "SHA-256 abc vector is exact");

    memset(digest, 'x', sizeof(digest));
    memset(error, 0, sizeof(error));
    CHECK(!ds4_plan_io_sha256(NULL, 1, digest, error, sizeof(error)),
          "SHA-256 rejects null non-empty input");
    CHECK(digest[0] == '\0', "failed SHA-256 clears digest output");
    CHECK(strcmp(error,
                 "invalid argument: data is null with nonzero size") == 0,
          "null-data SHA-256 error is exact");

    memset(error, 'x', sizeof(error));
    CHECK(!ds4_plan_io_sha256("abc", 3, NULL, error, sizeof(error)),
          "SHA-256 rejects a null digest output");
    CHECK(strcmp(error, "invalid argument: digest output is null") == 0,
          "null-output SHA-256 error is exact");

    if ((uintmax_t)SIZE_MAX > UINT64_MAX / UINT64_C(8)) {
        const unsigned char one = 1;
        memset(digest, 'x', sizeof(digest));
        CHECK(!ds4_plan_io_sha256(
                  &one,
                  (size_t)(UINT64_MAX / UINT64_C(8) + UINT64_C(1)),
                  digest,
                  error,
                  sizeof(error)),
              "SHA-256 rejects bit-length overflow before reading input");
        CHECK(strcmp(error,
                     "invalid argument: data length exceeds SHA-256 limit") == 0,
              "SHA-256 overflow error is exact");
        CHECK(digest[0] == '\0',
              "SHA-256 overflow clears digest output");
    }
}

static void test_publish_and_immutability(const char *root) {
    static const unsigned char bytes_a[] = {
        '{', '"', 'x', '"', ':', 0x00, ',', 0xff, '}', '\n'
    };
    static const unsigned char bytes_b[] = "second immutable plan\n";
    char plan_a[512];
    char plan_b[512];
    char sidecar_a[520];
    char digest_a[DS4_PLAN_IO_SHA256_HEX_SIZE];
    char digest_b[DS4_PLAN_IO_SHA256_HEX_SIZE];
    char error[512] = "stale";
    CHECK(make_path(plan_a, sizeof(plan_a), root, "plan-a.json"),
          "first plan path fits test buffer");
    CHECK(make_path(plan_b, sizeof(plan_b), root, "plan-b.json"),
          "second plan path fits test buffer");
    CHECK(snprintf(sidecar_a, sizeof(sidecar_a), "%s.sha256", plan_a) > 0,
          "sidecar path fits test buffer");

    CHECK(ds4_plan_io_publish(plan_a,
                              bytes_a,
                              sizeof(bytes_a),
                              digest_a,
                              error,
                              sizeof(error)),
          "raw plan bytes publish successfully");
    CHECK(error[0] == '\0', "successful publication clears stale error text");

    size_t file_size = 0;
    unsigned char *file_bytes = read_fixture(plan_a, &file_size);
    CHECK(file_bytes != NULL && file_size == sizeof(bytes_a) &&
              memcmp(file_bytes, bytes_a, sizeof(bytes_a)) == 0,
          "published plan preserves arbitrary bytes exactly");
    free(file_bytes);

    size_t sidecar_size = 0;
    unsigned char *sidecar_bytes = read_fixture(sidecar_a, &sidecar_size);
    char expected_sidecar[DS4_PLAN_IO_SHA256_HEX_SIZE + 1];
    const int expected_size = snprintf(expected_sidecar,
                                       sizeof(expected_sidecar),
                                       "%s\n",
                                       digest_a);
    CHECK(sidecar_bytes != NULL && expected_size == 65 &&
              sidecar_size == (size_t)expected_size &&
              memcmp(sidecar_bytes, expected_sidecar, sidecar_size) == 0,
          "external digest sidecar contains exactly lowercase hex plus newline");
    free(sidecar_bytes);

    CHECK(ds4_plan_io_publish(plan_b,
                              bytes_b,
                              sizeof(bytes_b) - 1,
                              digest_b,
                              error,
                              sizeof(error)),
          "second immutable content publishes to a distinct target");
    CHECK(strcmp(digest_a, digest_b) != 0,
          "changing exact plan bytes changes the digest");
    CHECK(count_temporary_files(root) == 0,
          "successful publication leaves no same-directory temporary files");

    memset(error, 0, sizeof(error));
    CHECK(!ds4_plan_io_publish(plan_a,
                               bytes_b,
                               sizeof(bytes_b) - 1,
                               digest_b,
                               error,
                               sizeof(error)),
          "publisher refuses to overwrite an existing plan");
    char expected_error[640];
    CHECK(snprintf(expected_error,
                   sizeof(expected_error),
                   "refusing to replace existing plan '%s'",
                   plan_a) > 0,
          "existing-plan expected error fits");
    CHECK(strcmp(error, expected_error) == 0,
          "existing-plan error is exact and identifies the target");
    file_bytes = read_fixture(plan_a, &file_size);
    CHECK(file_bytes != NULL && file_size == sizeof(bytes_a) &&
              memcmp(file_bytes, bytes_a, sizeof(bytes_a)) == 0,
          "refused overwrite leaves original plan unchanged");
    free(file_bytes);
    CHECK(count_temporary_files(root) == 0,
          "refused overwrite leaves no temporary files");
}

static void test_nested_directory(const char *root) {
    char nested[512];
    char deeper[512];
    char plan[512];
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    char error[512];
    CHECK(make_path(nested, sizeof(nested), root, "nested"),
          "nested directory path fits");
    CHECK(make_path(deeper, sizeof(deeper), nested, "deeper"),
          "deeper directory path fits");
    CHECK(mkdir(nested, 0700) == 0, "nested directory created");
    CHECK(mkdir(deeper, 0700) == 0, "deeper directory created");
    CHECK(make_path(plan, sizeof(plan), deeper, "plan.json"),
          "nested plan path fits");
    CHECK(ds4_plan_io_publish(plan,
                              "{}\n",
                              3,
                              digest,
                              error,
                              sizeof(error)),
          "plan and sidecar publish inside a nested directory");
    CHECK(access(plan, F_OK) == 0, "nested plan exists");
    CHECK(count_temporary_files(deeper) == 0,
          "nested publication leaves no temporary files");
}

static void test_preexisting_sidecar(const char *root) {
    static const char sentinel[] = "do not replace\n";
    char plan[512];
    char sidecar[520];
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    char error[640];
    CHECK(make_path(plan, sizeof(plan), root, "blocked.json"),
          "blocked plan path fits");
    CHECK(snprintf(sidecar, sizeof(sidecar), "%s.sha256", plan) > 0,
          "blocked sidecar path fits");
    CHECK(write_fixture(sidecar, sentinel, sizeof(sentinel) - 1),
          "pre-existing sidecar fixture created");
    CHECK(!ds4_plan_io_publish(plan,
                               "{}",
                               2,
                               digest,
                               error,
                               sizeof(error)),
          "publisher refuses a pre-existing sidecar before writing the plan");
    char expected_error[700];
    CHECK(snprintf(expected_error,
                   sizeof(expected_error),
                   "refusing to replace existing digest sidecar '%s'",
                   sidecar) > 0,
          "existing-sidecar expected error fits");
    CHECK(strcmp(error, expected_error) == 0,
          "existing-sidecar error is exact and identifies the target");
    CHECK(access(plan, F_OK) != 0 && errno == ENOENT,
          "pre-existing sidecar failure does not publish a plan");
    size_t size = 0;
    unsigned char *bytes = read_fixture(sidecar, &size);
    CHECK(bytes != NULL && size == sizeof(sentinel) - 1 &&
              memcmp(bytes, sentinel, size) == 0,
          "pre-existing sidecar remains unchanged");
    free(bytes);
    CHECK(count_temporary_files(root) == 0,
          "pre-existing sidecar failure leaves no temporary files");
}

static void test_post_plan_sidecar_failure(const char *root) {
    errno = 0;
    const long name_max = pathconf(root, _PC_NAME_MAX);
    if (name_max < 32 ||
        (uintmax_t)name_max > DS4_PLAN_IO_PATH_LIMIT / 2u) {
        return;
    }

    const size_t temporary_suffix_size = sizeof(".tmp.XXXXXX") - 1u;
    const size_t basename_size = (size_t)name_max - temporary_suffix_size;
    char *basename = malloc(basename_size + 1u);
    const size_t root_size = strlen(root);
    char *plan = malloc(root_size + 1u + basename_size + 1u);
    char *sidecar = malloc(root_size + 1u + basename_size +
                           sizeof(".sha256"));
    CHECK(basename != NULL && plan != NULL && sidecar != NULL,
          "sidecar-failure paths allocated");
    if (basename == NULL || plan == NULL || sidecar == NULL) {
        free(sidecar);
        free(plan);
        free(basename);
        return;
    }
    memset(basename, 's', basename_size);
    basename[basename_size] = '\0';
    CHECK(snprintf(plan,
                   root_size + 1u + basename_size + 1u,
                   "%s/%s",
                   root,
                   basename) > 0,
          "sidecar-failure plan path built");
    CHECK(snprintf(sidecar,
                   root_size + 1u + basename_size + sizeof(".sha256"),
                   "%s.sha256",
                   plan) > 0,
          "sidecar-failure digest path built");

    static const char bytes[] = "complete but unauthenticated\n";
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE] = "stale";
    char error[1024];
    CHECK(!ds4_plan_io_publish(plan,
                               bytes,
                               sizeof(bytes) - 1u,
                               digest,
                               error,
                               sizeof(error)),
          "sidecar-stage failure is returned to the caller");
    CHECK(strstr(error, "create temporary digest sidecar for '") == error &&
              strstr(error, sidecar) != NULL,
          "sidecar-stage error names the failed artifact");
    CHECK(digest[0] == '\0',
          "failed pair publication does not return an authenticated digest");

    size_t published_size = 0;
    unsigned char *published = read_fixture(plan, &published_size);
    CHECK(published != NULL && published_size == sizeof(bytes) - 1u &&
              memcmp(published, bytes, published_size) == 0,
          "sidecar-stage failure leaves the complete published plan intact");
    free(published);
    CHECK(access(sidecar, F_OK) != 0 && errno == ENOENT,
          "sidecar-stage failure leaves no digest sidecar");
    CHECK(count_temporary_files(root) == 0,
          "sidecar-stage failure cleans its same-directory temporary file");

    (void)unlink(plan);
    free(sidecar);
    free(plan);
    free(basename);
}

static void test_invalid_paths(const char *root) {
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE];
    char error[512];
    CHECK(!ds4_plan_io_publish(NULL,
                               "x",
                               1,
                               digest,
                               error,
                               sizeof(error)),
          "null plan path is rejected");
    CHECK(strcmp(error, "invalid argument: plan path is null") == 0,
          "null-path error is exact");

    CHECK(!ds4_plan_io_publish("",
                               "x",
                               1,
                               digest,
                               error,
                               sizeof(error)),
          "empty plan path is rejected");
    CHECK(strcmp(error, "invalid argument: plan path is empty") == 0,
          "empty-path error is exact");

    char *too_long = malloc(DS4_PLAN_IO_PATH_LIMIT + 2u);
    CHECK(too_long != NULL, "too-long path buffer allocated");
    if (too_long != NULL) {
        memset(too_long, 'p', DS4_PLAN_IO_PATH_LIMIT + 1u);
        too_long[DS4_PLAN_IO_PATH_LIMIT + 1u] = '\0';
        CHECK(!ds4_plan_io_publish(too_long,
                                   "x",
                                   1,
                                   digest,
                                   error,
                                   sizeof(error)),
              "too-long plan path is rejected before filesystem access");
        CHECK(strcmp(error,
                     "invalid argument: plan path is too long") == 0,
              "too-long-path error is exact");
        free(too_long);
    }

    char missing_parent[512];
    CHECK(make_path(missing_parent,
                    sizeof(missing_parent),
                    root,
                    "missing/plan.json"),
          "missing-parent path fits");
    CHECK(!ds4_plan_io_publish(missing_parent,
                               "x",
                               1,
                               digest,
                               error,
                               sizeof(error)),
          "nonexistent parent fails publication");
    CHECK(strstr(error, "create temporary plan for '") == error &&
              strstr(error, missing_parent) != NULL &&
              strstr(error, "No such file or directory") != NULL,
          "nonexistent-parent error names operation, path, and OS failure");
    CHECK(count_temporary_files(root) == 0,
          "nonexistent-parent failure leaves no temporary files");

    char tiny_error[1] = {'x'};
    CHECK(!ds4_plan_io_publish("",
                               "x",
                               1,
                               digest,
                               tiny_error,
                               sizeof(tiny_error)),
          "errors still return false with a one-byte error buffer");
    CHECK(tiny_error[0] == '\0',
          "truncated error output remains nul-terminated");
}

static void cleanup_tree(const char *root) {
    char path[512];
    char sidecar[520];
    static const char *const files[] = {
        "plan-a.json", "plan-b.json", "blocked.json"
    };
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); i++) {
        if (!make_path(path, sizeof(path), root, files[i])) continue;
        (void)unlink(path);
        const int n = snprintf(sidecar, sizeof(sidecar), "%s.sha256", path);
        if (n >= 0 && (size_t)n < sizeof(sidecar)) (void)unlink(sidecar);
    }
    if (make_path(path, sizeof(path), root, "nested/deeper/plan.json")) {
        (void)unlink(path);
        const int n = snprintf(sidecar, sizeof(sidecar), "%s.sha256", path);
        if (n >= 0 && (size_t)n < sizeof(sidecar)) (void)unlink(sidecar);
    }
    if (make_path(path, sizeof(path), root, "nested/deeper")) (void)rmdir(path);
    if (make_path(path, sizeof(path), root, "nested")) (void)rmdir(path);
    (void)rmdir(root);
}

int main(void) {
    char root[] = "/tmp/ds4-plan-io-test.XXXXXX";
    CHECK(make_temporary_directory(root), "temporary test directory created");
    if (g_failed != 0) return 1;

    test_sha256_vectors();
    test_publish_and_immutability(root);
    test_nested_directory(root);
    test_preexisting_sidecar(root);
    test_post_plan_sidecar_failure(root);
    test_invalid_paths(root);

    cleanup_tree(root);
    if (g_failed != 0) {
        fprintf(stderr,
                "test_plan_io: %d/%d assertion(s) failed\n",
                g_failed,
                g_total);
        return 1;
    }
    fprintf(stdout, "test_plan_io: %d assertions passed\n", g_total);
    return 0;
}
