#define _DEFAULT_SOURCE 1
#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__APPLE__)
typedef char model_source_mincore_byte;
#else
typedef unsigned char model_source_mincore_byte;
#endif

static int model_source_test_mincore(
    void *address, size_t length, model_source_mincore_byte *vector);
static long model_source_test_sysconf(int name);

#define mincore model_source_test_mincore
#define sysconf model_source_test_sysconf
#include "../ds4_runtime.c"
#undef mincore
#undef sysconf

static int g_assertions;
static int g_failures;

#define CHECK(condition, message) do {                                      \
    g_assertions++;                                                         \
    if (!(condition)) {                                                     \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);    \
        g_failures++;                                                       \
    }                                                                         \
} while (0)

enum {
    MODEL_SOURCE_MINCORE_REAL = 0,
    MODEL_SOURCE_MINCORE_ALTERNATING = 1,
    MODEL_SOURCE_MINCORE_ALL = 2,
    MODEL_SOURCE_MINCORE_NONE = 3,
};

#if defined(_SC_PAGESIZE)
#define MODEL_SOURCE_SC_PAGE _SC_PAGESIZE
#else
#define MODEL_SOURCE_SC_PAGE _SC_PAGE_SIZE
#endif

typedef struct {
    uintptr_t address;
    size_t length;
    size_t pages;
} model_source_mincore_call;

enum { MODEL_SOURCE_CALL_CAPACITY = 8 };
static model_source_mincore_call g_calls[MODEL_SOURCE_CALL_CAPACITY];
static size_t g_call_count;
static size_t g_page_cursor;
static uint64_t g_page_size;
static int g_mincore_mode;
static int g_fail_call = -1;
static bool g_bad_geometry;
static long g_sysconf_result;
static size_t g_sysconf_calls;

static void reset_interposers(uint64_t page_size, long sysconf_result,
                              int mincore_mode, int fail_call) {
    memset(g_calls, 0, sizeof(g_calls));
    g_call_count = 0;
    g_page_cursor = 0;
    g_page_size = page_size;
    g_sysconf_result = sysconf_result;
    g_sysconf_calls = 0;
    g_mincore_mode = mincore_mode;
    g_fail_call = fail_call;
    g_bad_geometry = false;
}

static int model_source_test_mincore(
        void *address, size_t length, model_source_mincore_byte *vector) {
    const size_t call = g_call_count++;
    const size_t pages = g_page_size != 0
        ? length / (size_t)g_page_size : 0;
    if (call < MODEL_SOURCE_CALL_CAPACITY) {
        g_calls[call] = (model_source_mincore_call){
            (uintptr_t)address, length, pages,
        };
    }
    if (g_page_size == 0 || length % (size_t)g_page_size != 0 ||
        pages > 65536u || call >= MODEL_SOURCE_CALL_CAPACITY) {
        g_bad_geometry = true;
        errno = EINVAL;
        return -1;  /* Never overflow the production vector in a bad fixture. */
    }
    if (g_mincore_mode == MODEL_SOURCE_MINCORE_REAL) {
        return mincore(address, length, vector);
    }
    for (size_t i = 0; i < pages; i++) {
        const bool resident = g_mincore_mode == MODEL_SOURCE_MINCORE_ALL ||
            (g_mincore_mode == MODEL_SOURCE_MINCORE_ALTERNATING &&
             ((g_page_cursor + i) & 1u) == 0u);
        /* Only bit zero is residency; other status bits must not count. */
        vector[i] = resident ? (model_source_mincore_byte)0x81u :
                               (model_source_mincore_byte)0x80u;
    }
    g_page_cursor += pages;
    if (call == (size_t)g_fail_call) {
        errno = EIO;
        return -1;
    }
    return 0;
}

static long model_source_test_sysconf(int name) {
    g_sysconf_calls++;
    if (name == MODEL_SOURCE_SC_PAGE) return g_sysconf_result;
    return sysconf(name);
}

static void test_real_tiny_file(uint64_t page_size) {
    char path[] = "/tmp/ds4-model-source-XXXXXX";
    const uint64_t file_size = page_size + 1u;
    const uint64_t mapped_bytes = page_size * 2u;
    const size_t mapped_length = (size_t)mapped_bytes;
    int fd = -1;
    void *mapping = MAP_FAILED;
    bool path_owned = false;
    fd = mkstemp(path);
    CHECK(fd >= 0, "create tiny model source file");
    if (fd < 0) return;
    path_owned = true;
    if (unlink(path) != 0) {
        CHECK(false, "unlink owned source file before mapping");
        goto cleanup;
    }
    path_owned = false;
    if (ftruncate(fd, (off_t)file_size) != 0) {
        CHECK(false, "size the owned source file");
        goto cleanup;
    }
    {
        const unsigned char first = 0x31u;
        const unsigned char tail = 0x73u;
        if (pwrite(fd, &first, 1u, 0) != 1 ||
            pwrite(fd, &tail, 1u, (off_t)page_size) != 1) {
            CHECK(false, "populate both owned source pages");
            goto cleanup;
        }
    }
    mapping = mmap(NULL, mapped_length, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, 0);
    CHECK(mapping != MAP_FAILED, "map the unlinked tiny model source");
    if (mapping == MAP_FAILED) goto cleanup;
    volatile unsigned char *bytes = (volatile unsigned char *)mapping;
    CHECK(bytes[0] == 0x31u && bytes[page_size] == 0x73u,
          "mapping exposes both file contents");
    bytes[0] = 0x31u;
    bytes[page_size] = 0x73u;

    model_source_mincore_byte independent[2] = {0, 0};
    CHECK(mincore(mapping, mapped_length, independent) == 0,
          "independently sample the touched mapping");
    uint64_t expected = 0;
    for (size_t i = 0; i < 2u; i++) {
        if (((unsigned char)independent[i] & 1u) != 0u) expected += page_size;
    }

    CHECK(expected == mapped_bytes,
          "independent sample sees both touched pages, including the tail");
    uint64_t resident = UINT64_C(0xfeedface);
    reset_interposers(page_size, (long)page_size,
                      MODEL_SOURCE_MINCORE_REAL, -1);
    const bool sampled = ds4_runtime_model_source_resident_bytes(
        mapping, file_size, page_size, &resident);
    CHECK(sampled && resident == expected && resident % page_size == 0u,
          "resident bytes match independent full-page mincore charge");
    CHECK(g_call_count == 1u && !g_bad_geometry &&
              g_calls[0].address == (uintptr_t)mapping &&
              g_calls[0].length == mapped_length && g_calls[0].pages == 2u,
          "real sample uses the rounded two-page mapping geometry");
    CHECK(bytes[0] == 0x31u && bytes[page_size] == 0x73u &&
              fcntl(fd, F_GETFD) >= 0,
          "sampling leaves mapping contents and caller FD usable");
    {
        const unsigned char replacement = 0x5au;
        unsigned char readback = 0;
        CHECK(pwrite(fd, &replacement, 1u, 0) == 1 &&
                  pread(fd, &readback, 1u, 0) == 1 &&
                  readback == replacement && bytes[0] == replacement,
              "caller can still use the FD and mapping after sampling");
    }
cleanup:
    if (mapping != MAP_FAILED) CHECK(munmap(mapping, mapped_length) == 0,
                                      "unmap owned tiny model mapping");
    if (fd >= 0) CHECK(close(fd) == 0, "close owned tiny model FD");
    if (path_owned) CHECK(unlink(path) == 0, "remove owned temporary path");
}

static void expect_reject(void *mapping, uint64_t model_size,
                          uint64_t supplied_page_size, long system_page_size,
                          const char *message) {
    uint64_t resident = UINT64_C(0x9a5a5a5a);
    reset_interposers(supplied_page_size, system_page_size,
                      MODEL_SOURCE_MINCORE_NONE, -1);
    const bool sampled = ds4_runtime_model_source_resident_bytes(
        mapping, model_size, supplied_page_size, &resident);
    CHECK(!sampled && resident == UINT64_C(0x9a5a5a5a) &&
              g_call_count == 0u, message);
}

static void expect_null_output(void *mapping, uint64_t model_size,
                               uint64_t supplied_page_size,
                               long system_page_size) {
    reset_interposers(supplied_page_size, system_page_size,
                      MODEL_SOURCE_MINCORE_NONE, -1);
    CHECK(!ds4_runtime_model_source_resident_bytes(
              mapping, model_size, supplied_page_size, NULL) &&
              g_call_count == 0u, "reject a NULL output pointer");
}

static void test_invalid_inputs(uint64_t page_size) {
    const uintptr_t aligned = (uintptr_t)0x10000000u &
                              ~(uintptr_t)(page_size - 1u);
    void *mapping = (void *)aligned;
    uint64_t control = UINT64_C(0xabcdef);
    reset_interposers(page_size, (long)page_size,
                      MODEL_SOURCE_MINCORE_NONE, -1);
    CHECK(ds4_runtime_model_source_resident_bytes(
              mapping, page_size + 1u, page_size, &control) &&
              control == 0u && g_call_count == 1u && !g_bad_geometry,
          "invalid-input cases start from an accepted synthetic control");
    expect_reject(NULL, page_size + 1u, page_size, (long)page_size,
                  "reject a NULL model mapping");
    expect_null_output(mapping, page_size + 1u, page_size, (long)page_size);
    expect_reject(mapping, 0u, page_size, (long)page_size,
                  "reject a zero model size");
    expect_reject(mapping, page_size + 1u, 0u, (long)page_size,
                  "reject a zero supplied page size");
    expect_reject(mapping, page_size + 1u, 3u, (long)page_size,
                  "reject a non-power-of-two supplied page size");
    expect_reject((void *)(aligned + 1u), page_size + 1u, page_size,
                  (long)page_size, "reject an unaligned model mapping");
    expect_reject(mapping,
                  UINT64_MAX - (page_size - 1u) + 1u, page_size,
                  (long)page_size, "reject rounded-size overflow");
    const uintptr_t end_aligned = UINTPTR_MAX &
                                  ~(uintptr_t)(page_size - 1u);
    expect_reject((void *)end_aligned, page_size, page_size, (long)page_size,
                  "reject model pointer interval overflow");
    expect_reject(mapping, page_size + 1u, page_size / 2u,
                  (long)page_size, "reject a smaller power-of-two page size");
    expect_reject(mapping, page_size + 1u, page_size * 2u,
                  (long)page_size, "reject a larger power-of-two page size");

    const long invalid_system_sizes[] = {-1L, 0L, 3L, LONG_MAX};
    for (size_t i = 0; i < sizeof(invalid_system_sizes) /
                              sizeof(invalid_system_sizes[0]); i++) {
        expect_reject(mapping, page_size + 1u, page_size,
                      invalid_system_sizes[i],
                      "reject an error or invalid system page size");
    }
}

static void test_interposed_batches(uint64_t page_size) {
    const uintptr_t aligned = (uintptr_t)0x20000000u &
                              ~(uintptr_t)(page_size - 1u);
    void *mapping = (void *)aligned;
    uint64_t resident = UINT64_C(0x11111111);
    reset_interposers(page_size, (long)page_size,
                      MODEL_SOURCE_MINCORE_ALL, -1);
    CHECK(ds4_runtime_model_source_resident_bytes(
              mapping, page_size + 1u, page_size, &resident) &&
              resident == page_size * 2u && g_call_count == 1u &&
              g_calls[0].pages == 2u,
          "successful all-resident sample charges a full tail page");

    resident = UINT64_C(0x22222222);
    reset_interposers(page_size, (long)page_size,
                      MODEL_SOURCE_MINCORE_NONE, -1);
    CHECK(ds4_runtime_model_source_resident_bytes(
              mapping, page_size + 1u, page_size, &resident) &&
              resident == 0u && g_call_count == 1u,
          "successful no-resident sample commits zero");

    const uint64_t pages = UINT64_C(65536) + 3u;
    const uint64_t rounded = pages * page_size;
    const uint64_t model_size = rounded - page_size + 17u;
    resident = UINT64_C(0x33333333);
    reset_interposers(page_size, (long)page_size,
                      MODEL_SOURCE_MINCORE_ALTERNATING, -1);
    CHECK(ds4_runtime_model_source_resident_bytes(
              mapping, model_size, page_size, &resident) &&
              resident == (UINT64_C(32768) + 2u) * page_size &&
              g_call_count == 2u && !g_bad_geometry,
          "alternating simulated residency is exact across bounded batches");
    CHECK(g_calls[0].address == aligned &&
              g_calls[0].length == UINT64_C(65536) * page_size &&
              g_calls[0].pages == 65536u &&
              g_calls[1].address == aligned + g_calls[0].length &&
              g_calls[1].length == UINT64_C(3) * page_size &&
              g_calls[1].pages == 3u,
          "bulk simulation observes exact addresses and partial-tail rounding");

    resident = UINT64_C(0x44444444);
    reset_interposers(page_size, (long)page_size,
                      MODEL_SOURCE_MINCORE_ALTERNATING, 1);
    CHECK(!ds4_runtime_model_source_resident_bytes(
              mapping, model_size, page_size, &resident) &&
              resident == UINT64_C(0x44444444) && g_call_count == 2u &&
              !g_bad_geometry,
          "a later mincore failure leaves output unchanged");
}

int main(void) {
    alarm(15);
    const long system_page_size = sysconf(MODEL_SOURCE_SC_PAGE);
    CHECK(system_page_size > 0 &&
              (system_page_size & (system_page_size - 1L)) == 0,
          "host reports a positive power-of-two page size");
    if (system_page_size > 0 &&
        (system_page_size & (system_page_size - 1L)) == 0) {
        const uint64_t page_size = (uint64_t)system_page_size;
        test_real_tiny_file(page_size);
        test_invalid_inputs(page_size);
        test_interposed_batches(page_size);
    }
    if (g_failures != 0) {
        fprintf(stderr, "model-source-sampler: %d/%d checks failed\n",
                g_failures, g_assertions);
        return 1;
    }
    printf("model-source-sampler: %d checks passed\n", g_assertions);
    return 0;
}
