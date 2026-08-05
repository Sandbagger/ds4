#define _POSIX_C_SOURCE 200809L

#include "../ds4_plan_io.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
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

static bool write_exclusive_fixture(const char *path,
                                    const void *bytes,
                                    size_t size) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return false;
    size_t offset = 0;
    while (offset != size) {
        const ssize_t wrote = write(
            fd, (const unsigned char *)bytes + offset, size - offset);
        if (wrote > 0) {
            offset += (size_t)wrote;
        } else if (wrote < 0 && errno == EINTR) {
            continue;
        } else {
            (void)close(fd);
            (void)unlink(path);
            return false;
        }
    }
    const bool synced = fsync(fd) == 0;
    const bool closed = close(fd) == 0;
    const bool ok = synced && closed;
    if (!ok) (void)unlink(path);
    return ok;
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

static bool directory_has_prefix(const char *dir, const char *prefix) {
    DIR *stream = opendir(dir);
    if (stream == NULL) return false;
    const size_t prefix_size = strlen(prefix);
    bool found = false;
    for (;;) {
        struct dirent *entry = readdir(stream);
        if (entry == NULL) break;
        if (strncmp(entry->d_name, prefix, prefix_size) == 0) {
            found = true;
            break;
        }
    }
    (void)closedir(stream);
    return found;
}

static pid_t spawn_target_racer(const char *dir,
                                const char *temporary_prefix,
                                const char *target,
                                const char *sentinel) {
    const pid_t publisher = getpid();
    const pid_t child = fork();
    if (child != 0) return child;

    struct timespec started;
    if (clock_gettime(CLOCK_MONOTONIC, &started) != 0) _exit(20);
    for (;;) {
        if (directory_has_prefix(dir, temporary_prefix)) {
            if (kill(publisher, SIGSTOP) != 0) _exit(21);
            const bool wrote = write_exclusive_fixture(
                target, sentinel, strlen(sentinel));
            const int resume_rc = kill(publisher, SIGCONT);
            _exit(wrote && resume_rc == 0 ? 0 : 22);
        }
        struct timespec now;
        if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) _exit(23);
        if (now.tv_sec - started.tv_sec >= 5) _exit(24);
    }
}

static bool wait_for_racer(pid_t child) {
    int status = 0;
    return child > 0 && waitpid(child, &status, 0) == child &&
           WIFEXITED(status) && WEXITSTATUS(status) == 0;
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

static void test_target_preflight(const char *root) {
    char valid[512];
    char existing[512];
    char missing_parent[512];
    char error[512] = "stale";

    CHECK(make_path(valid, sizeof(valid), root, "preflight.json"),
          "preflight target path fits");
    CHECK(ds4_plan_io_preflight_target(valid, error, sizeof(error)),
          "preflight accepts an absent target in an existing parent");
    CHECK(error[0] == '\0', "successful target preflight clears stale error");
    CHECK(access(valid, F_OK) != 0 && errno == ENOENT,
          "target preflight does not create the plan");

    CHECK(make_path(existing, sizeof(existing), root, "preflight-existing.json"),
          "existing preflight target path fits");
    CHECK(write_fixture(existing, "occupied", 8),
          "existing preflight target fixture written");
    CHECK(!ds4_plan_io_preflight_target(existing, error, sizeof(error)),
          "preflight rejects an existing immutable target");
    CHECK(strstr(error, "refusing to replace existing plan") == error,
          "existing target preflight explains immutability");
    CHECK(unlink(existing) == 0, "existing preflight fixture removed");

    CHECK(make_path(missing_parent,
                    sizeof(missing_parent),
                    root,
                    "missing/preflight.json"),
          "missing-parent preflight path fits");
    CHECK(!ds4_plan_io_preflight_target(
              missing_parent, error, sizeof(error)),
          "preflight rejects a missing parent before model access");
    CHECK(strstr(error, "parent directory") != NULL &&
              strstr(error, "No such file or directory") != NULL,
          "missing-parent preflight names the directory failure");
}

static char *make_name_max_plan(const char *root,
                                size_t basename_size,
                                char fill) {
    const size_t root_size = strlen(root);
    char *path = malloc(root_size + 1u + basename_size + 1u);
    if (path == NULL) return NULL;
    memcpy(path, root, root_size);
    path[root_size] = '/';
    memset(path + root_size + 1u, fill, basename_size);
    path[root_size + 1u + basename_size] = '\0';
    return path;
}

static void test_preflight_exact_temporary_names(const char *root) {
    errno = 0;
    const long name_max = pathconf(root, _PC_NAME_MAX);
    if (name_max < 32 ||
        (uintmax_t)name_max > DS4_PLAN_IO_PATH_LIMIT / 2u) {
        return;
    }

    const size_t plan_temp_suffix = sizeof(".tmp.XXXXXX") - 1u;
    const size_t sidecar_suffix = sizeof(".sha256") - 1u;
    char *plan_temp_too_long = make_name_max_plan(
        root, (size_t)name_max - sidecar_suffix, 'p');
    char *sidecar_temp_too_long = make_name_max_plan(
        root, (size_t)name_max - plan_temp_suffix, 's');
    CHECK(plan_temp_too_long != NULL && sidecar_temp_too_long != NULL,
          "exact-temp preflight paths allocated");
    if (plan_temp_too_long == NULL || sidecar_temp_too_long == NULL) {
        free(sidecar_temp_too_long);
        free(plan_temp_too_long);
        return;
    }

    char error[1024];
    CHECK(!ds4_plan_io_preflight_target(
              plan_temp_too_long, error, sizeof(error)),
          "preflight rejects an uncreatable exact plan temp name");
    CHECK(strstr(error, "temporary plan") != NULL,
          "plan-temp preflight error identifies the failed artifact");
    CHECK(!ds4_plan_io_preflight_target(
              sidecar_temp_too_long, error, sizeof(error)),
          "preflight rejects an uncreatable exact sidecar temp name");
    CHECK(strstr(error, "temporary digest sidecar") != NULL,
          "sidecar-temp preflight error identifies the failed artifact");
    CHECK(access(plan_temp_too_long, F_OK) != 0 && errno == ENOENT,
          "plan-temp preflight creates no final plan");
    CHECK(access(sidecar_temp_too_long, F_OK) != 0 && errno == ENOENT,
          "sidecar-temp preflight creates no final plan");
    CHECK(count_temporary_files(root) == 0,
          "exact-temp preflight cleans every probe file");

    free(sidecar_temp_too_long);
    free(plan_temp_too_long);
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

static void test_racing_plan_target_is_not_replaced(const char *root) {
    static const char sentinel[] = "racing creator owns this plan\n";
    char plan[512];
    char sidecar[520];
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE] = "stale";
    char error[1024];
    CHECK(make_path(plan, sizeof(plan), root, "race.json"),
          "race target path fits");
    CHECK(snprintf(sidecar, sizeof(sidecar), "%s.sha256", plan) > 0,
          "race sidecar path fits");

    const size_t payload_size = 32u * 1024u * 1024u;
    unsigned char *payload = malloc(payload_size);
    CHECK(payload != NULL, "race payload allocated");
    if (payload == NULL) return;
    memset(payload, 'r', payload_size);

    const pid_t racer = spawn_target_racer(
        root, "race.json.tmp.", plan, sentinel);
    CHECK(racer > 0, "plan-target racer started");
    const bool published = ds4_plan_io_publish(
        plan, payload, payload_size, digest, error, sizeof(error));
    CHECK(wait_for_racer(racer), "plan-target racer completed");
    CHECK(!published, "racing plan target makes publication fail");
    CHECK(strstr(error, "existing plan") != NULL,
          "racing plan error reports immutable target conflict");
    CHECK(digest[0] == '\0',
          "racing plan failure returns no authenticated digest");

    size_t size = 0;
    unsigned char *bytes = read_fixture(plan, &size);
    CHECK(bytes != NULL && size == sizeof(sentinel) - 1u &&
              memcmp(bytes, sentinel, size) == 0,
          "publisher never replaces a racing plan target");
    free(bytes);
    CHECK(access(sidecar, F_OK) != 0 && errno == ENOENT,
          "plan-target race publishes no sidecar");
    CHECK(count_temporary_files(root) == 0,
          "plan-target race leaves no temporary files");

    free(payload);
}

static void test_second_commit_failure_rolls_back_plan(const char *root) {
    static const char sentinel[] = "racing creator owns this sidecar\n";
    char plan[512];
    char sidecar[520];
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE] = "stale";
    char error[1024];
    CHECK(make_path(plan, sizeof(plan), root, "rollback.json"),
          "rollback plan path fits");
    CHECK(snprintf(sidecar, sizeof(sidecar), "%s.sha256", plan) > 0,
          "rollback sidecar path fits");

    const size_t payload_size = 32u * 1024u * 1024u;
    unsigned char *payload = malloc(payload_size);
    CHECK(payload != NULL, "rollback race payload allocated");
    if (payload == NULL) return;
    memset(payload, 'q', payload_size);

    const pid_t racer = spawn_target_racer(
        root, "rollback.json.tmp.", sidecar, sentinel);
    CHECK(racer > 0, "sidecar-target racer started");
    const bool published = ds4_plan_io_publish(
        plan, payload, payload_size, digest, error, sizeof(error));
    CHECK(wait_for_racer(racer), "sidecar-target racer completed");
    CHECK(!published, "racing sidecar target makes publication fail");
    CHECK(strstr(error, "existing digest sidecar") != NULL,
          "sidecar race reports immutable target conflict");
    CHECK(digest[0] == '\0',
          "second-commit failure returns no authenticated digest");
    CHECK(access(plan, F_OK) != 0 && errno == ENOENT,
          "second-commit failure rolls back this call's plan");

    size_t size = 0;
    unsigned char *bytes = read_fixture(sidecar, &size);
    CHECK(bytes != NULL && size == sizeof(sentinel) - 1u &&
              memcmp(bytes, sentinel, size) == 0,
          "publisher never replaces a racing sidecar target");
    free(bytes);
    CHECK(count_temporary_files(root) == 0,
          "second-commit rollback leaves no temporary files");
    free(payload);
}

typedef struct {
    const char *directory;
    const char *plan;
    const char *sidecar;
    const char *rejection;
    int calls;
    bool saw_staged_pair_before_visibility;
} checked_publication_fixture;

static bool validate_checked_publication(void *context,
                                         char *error,
                                         size_t error_size) {
    checked_publication_fixture *fixture = context;
    fixture->calls++;
    errno = 0;
    const bool plan_absent = access(fixture->plan, F_OK) != 0 && errno == ENOENT;
    errno = 0;
    const bool sidecar_absent =
        access(fixture->sidecar, F_OK) != 0 && errno == ENOENT;
    fixture->saw_staged_pair_before_visibility =
        plan_absent && sidecar_absent &&
        count_temporary_files(fixture->directory) == 2;
    if (fixture->rejection == NULL) return true;
    (void)snprintf(error, error_size, "%s", fixture->rejection);
    return false;
}

static void test_checked_publication_validation(const char *root) {
    static const char payload[] = "identity-bound qualification plan\n";
    static const char rejection[] = "opened model identity changed before commit";
    char rejected_plan[512];
    char rejected_sidecar[520];
    char accepted_plan[512];
    char accepted_sidecar[520];
    char digest[DS4_PLAN_IO_SHA256_HEX_SIZE] = "stale";
    char error[512] = "stale";
    CHECK(make_path(rejected_plan, sizeof(rejected_plan), root, "checked-reject.json"),
          "checked rejection plan path fits");
    CHECK(snprintf(rejected_sidecar,
                   sizeof(rejected_sidecar),
                   "%s.sha256",
                   rejected_plan) > 0,
          "checked rejection sidecar path fits");
    CHECK(make_path(accepted_plan, sizeof(accepted_plan), root, "checked-accept.json"),
          "checked acceptance plan path fits");
    CHECK(snprintf(accepted_sidecar,
                   sizeof(accepted_sidecar),
                   "%s.sha256",
                   accepted_plan) > 0,
          "checked acceptance sidecar path fits");

    checked_publication_fixture rejected = {
        .directory = root,
        .plan = rejected_plan,
        .sidecar = rejected_sidecar,
        .rejection = rejection,
    };
    CHECK(!ds4_plan_io_publish_checked(rejected_plan,
                                       payload,
                                       sizeof(payload) - 1u,
                                       validate_checked_publication,
                                       &rejected,
                                       digest,
                                       error,
                                       sizeof(error)),
          "checked publisher rejects a changed caller identity");
    CHECK(rejected.calls == 1,
          "checked publisher invokes validation exactly once");
    CHECK(rejected.saw_staged_pair_before_visibility,
          "validation runs after both stages and before final visibility");
    CHECK(strcmp(error, rejection) == 0,
          "checked publisher preserves the callback error exactly");
    CHECK(digest[0] == '\0',
          "checked rejection returns no authenticated digest");
    CHECK(access(rejected_plan, F_OK) != 0 && errno == ENOENT,
          "checked rejection exposes no plan");
    CHECK(access(rejected_sidecar, F_OK) != 0 && errno == ENOENT,
          "checked rejection exposes no sidecar");
    CHECK(count_temporary_files(root) == 0,
          "checked rejection removes both staged temporary files");

    checked_publication_fixture accepted = {
        .directory = root,
        .plan = accepted_plan,
        .sidecar = accepted_sidecar,
    };
    CHECK(ds4_plan_io_publish_checked(accepted_plan,
                                      payload,
                                      sizeof(payload) - 1u,
                                      validate_checked_publication,
                                      &accepted,
                                      digest,
                                      error,
                                      sizeof(error)),
          "checked publisher commits after callback acceptance");
    CHECK(accepted.calls == 1,
          "accepted checked publisher invokes validation exactly once");
    CHECK(accepted.saw_staged_pair_before_visibility,
          "accepted validation sees both durable stages before visibility");
    CHECK(error[0] == '\0',
          "successful checked publication leaves no error text");
    CHECK(access(accepted_plan, F_OK) == 0,
          "accepted checked publication exposes the plan");
    CHECK(access(accepted_sidecar, F_OK) == 0,
          "accepted checked publication exposes the sidecar");
    CHECK(count_temporary_files(root) == 0,
          "accepted checked publication leaves no temporary files");
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

    static const char bytes[] = "must remain invisible without its digest\n";
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

    CHECK(access(plan, F_OK) != 0 && errno == ENOENT,
          "sidecar-stage failure publishes no unauthenticated plan");
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
        "plan-a.json", "plan-b.json", "blocked.json", "race.json",
        "rollback.json", "checked-accept.json"
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
    test_target_preflight(root);
    test_preflight_exact_temporary_names(root);
    test_publish_and_immutability(root);
    test_racing_plan_target_is_not_replaced(root);
    test_second_commit_failure_rolls_back_plan(root);
    test_checked_publication_validation(root);
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
