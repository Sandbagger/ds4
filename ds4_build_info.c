#include "ds4_build_info.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifndef DS4_BUILD_REVISION
#error "DS4_BUILD_REVISION must be stamped by the build"
#endif
#ifndef DS4_BUILD_DIRTY
#error "DS4_BUILD_DIRTY must be stamped by the build"
#endif
#ifndef DS4_BUILD_BACKEND
#error "DS4_BUILD_BACKEND must be stamped by the build"
#endif
#ifndef DS4_BUILD_FEATURES
#error "DS4_BUILD_FEATURES must be stamped by the build"
#endif

static ds4_runtime_build_info build_info;
static bool build_info_initialized;
static bool build_info_valid;

static int feature_compare(const void *left, const void *right) {
    return strcmp((const char *)left, (const char *)right);
}

static bool copy_bounded(char *destination, size_t capacity,
                         const char *source) {
    const size_t length = source ? strlen(source) : 0u;
    if (length == 0u || length >= capacity) return false;
    memcpy(destination, source, length + 1u);
    return true;
}

static bool revision_valid(const char *revision) {
    if (!revision || strlen(revision) != 40u) return false;
    bool nonzero = false;
    for (size_t i = 0; i < 40u; i++) {
        const char byte = revision[i];
        if (!((byte >= '0' && byte <= '9') ||
              (byte >= 'a' && byte <= 'f'))) {
            return false;
        }
        if (byte != '0') nonzero = true;
    }
    return nonzero;
}

static bool backend_valid(const char *backend) {
    return backend &&
        (strcmp(backend, "cpu") == 0 || strcmp(backend, "metal") == 0 ||
         strcmp(backend, "cuda") == 0 || strcmp(backend, "rocm") == 0);
}

static bool feature_valid(const char *feature) {
    if (!feature || feature[0] < 'a' || feature[0] > 'z') return false;
    for (size_t i = 1u; feature[i] != '\0'; i++) {
        const char byte = feature[i];
        if (!((byte >= 'a' && byte <= 'z') ||
              (byte >= '0' && byte <= '9') || byte == '_')) {
            return false;
        }
    }
    return true;
}

static void build_info_initialize(void) {
    if (build_info_initialized) return;
    build_info_initialized = true;
    memset(&build_info, 0, sizeof(build_info));

    if (!revision_valid(DS4_BUILD_REVISION) ||
        !backend_valid(DS4_BUILD_BACKEND) ||
        (DS4_BUILD_DIRTY != 0 && DS4_BUILD_DIRTY != 1) ||
        !copy_bounded(build_info.revision, sizeof(build_info.revision),
                      DS4_BUILD_REVISION) ||
        !copy_bounded(build_info.backend, sizeof(build_info.backend),
                      DS4_BUILD_BACKEND)) {
        return;
    }
    build_info.dirty = DS4_BUILD_DIRTY != 0;

    const char *cursor = DS4_BUILD_FEATURES;
    while (*cursor != '\0') {
        const char *comma = strchr(cursor, ',');
        const size_t length = comma ? (size_t)(comma - cursor) : strlen(cursor);
        if (length == 0u || length >= DS4_RUNTIME_FEATURE_CAPACITY ||
            build_info.feature_count >= DS4_RUNTIME_FEATURE_COUNT) {
            return;
        }
        char *feature = build_info.features[build_info.feature_count];
        memcpy(feature, cursor, length);
        feature[length] = '\0';
        if (!feature_valid(feature)) return;
        build_info.feature_count++;
        if (!comma) break;
        cursor = comma + 1u;
    }
    if (build_info.feature_count == 0u) return;

    qsort(build_info.features, build_info.feature_count,
          sizeof(build_info.features[0]), feature_compare);
    size_t unique_count = 0u;
    for (size_t i = 0u; i < build_info.feature_count; i++) {
        if (unique_count != 0u &&
            strcmp(build_info.features[unique_count - 1u],
                   build_info.features[i]) == 0) {
            continue;
        }
        if (unique_count != i) {
            memcpy(build_info.features[unique_count], build_info.features[i],
                   sizeof(build_info.features[unique_count]));
        }
        unique_count++;
    }
    for (size_t i = unique_count; i < build_info.feature_count; i++) {
        memset(build_info.features[i], 0, sizeof(build_info.features[i]));
    }
    build_info.feature_count = unique_count;
    build_info_valid = true;
}

const ds4_runtime_build_info *ds4_build_info_get(void) {
    build_info_initialize();
    return build_info_valid ? &build_info : NULL;
}

int ds4_build_info_write_json(FILE *stream) {
    const ds4_runtime_build_info *info = ds4_build_info_get();
    if (!stream || !info) return 1;
    if (fprintf(stream,
                "{\"schema\":\"ds4.version/v1\",\"revision\":\"%s\","
                "\"dirty\":%s,\"backend\":\"%s\",\"features\":[",
                info->revision, info->dirty ? "true" : "false",
                info->backend) < 0) {
        return 1;
    }
    for (size_t i = 0u; i < info->feature_count; i++) {
        if ((i != 0u && fputc(',', stream) == EOF) ||
            fprintf(stream, "\"%s\"", info->features[i]) < 0) {
            return 1;
        }
    }
    return fputs("]}\n", stream) == EOF || fflush(stream) != 0 ? 1 : 0;
}

int ds4_build_info_maybe_print_version(int argc, char **argv,
                                       const char *option, int *handled) {
    if (!option || !handled || strcmp(option, "--version-json") != 0) return 1;
    *handled = 0;
    if (argc < 2 || !argv || !argv[1] || strcmp(argv[1], option) != 0) {
        return 0;
    }
    *handled = 1;
    return ds4_build_info_write_json(stdout);
}
