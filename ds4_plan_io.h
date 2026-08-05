#ifndef DS4_PLAN_IO_H
#define DS4_PLAN_IO_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_PLAN_IO_SHA256_HEX_LENGTH 64u
#define DS4_PLAN_IO_SHA256_HEX_SIZE (DS4_PLAN_IO_SHA256_HEX_LENGTH + 1u)

/* A bounded scan avoids unbounded strlen() on an invalid path.  The actual
 * accepted length is smaller because publication appends both ".sha256" and
 * a same-directory temporary-file suffix. */
#define DS4_PLAN_IO_PATH_LIMIT 4096u

/* Hash an arbitrary byte sequence.  data may be NULL only when size is zero.
 * On success digest_hex contains 64 lowercase hexadecimal characters and a
 * trailing nul.  On failure digest_hex is cleared when it is non-NULL. */
bool ds4_plan_io_sha256(const void *data,
                        size_t size,
                        char digest_hex[DS4_PLAN_IO_SHA256_HEX_SIZE],
                        char *error,
                        size_t error_size);

/* Validate immutable output identity and probe both exact same-directory
 * temporary names without leaving either final artifact or a probe file.
 * Plan-only frontends call this before opening the model so an invalid
 * destination cannot trigger a 68-GiB metadata walk. */
bool ds4_plan_io_preflight_target(const char *path,
                                  char *error,
                                  size_t error_size);

/* Durably publish exact plan bytes and an external FILE.sha256 containing
 * "<digest>\n".  Both artifacts are completely staged with same-directory
 * mkstemp, write-all, file fsync, and close before either becomes visible.
 * Publication uses atomic no-replace rename on Darwin/Linux and a same-device
 * hard-link fallback; ordinary replacement rename is never used.  One parent
 * directory fsync follows the pair.
 *
 * Existing or racing plan and sidecar paths are never replaced.  A returned
 * failure removes temporary files and rolls back final artifacts whose inode
 * identity proves that this call installed them.
 *
 * No Darwin/Linux filesystem primitive makes two separate names crash-atomic.
 * Power loss before the parent-directory fsync can therefore leave an
 * incomplete pair.  Consumers must require both files and verify that the
 * sidecar matches the exact plan bytes; an orphan is never qualification
 * evidence. */
bool ds4_plan_io_publish(const char *path,
                         const void *bytes,
                         size_t size,
                         char digest_hex[DS4_PLAN_IO_SHA256_HEX_SIZE],
                         char *error,
                         size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
