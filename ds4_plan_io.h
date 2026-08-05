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

/* Validate immutable output identity and its parent directory without
 * creating either artifact.  Plan-only frontends call this before opening the
 * model so an invalid destination cannot trigger a 68-GiB metadata walk. */
bool ds4_plan_io_preflight_target(const char *path,
                                  char *error,
                                  size_t error_size);

/* Durably publish exact plan bytes, followed by an external FILE.sha256 that
 * contains "<digest>\n".  Each artifact uses a same-directory mkstemp file,
 * write-all, file fsync, close, rename, and parent-directory fsync.
 *
 * Existing plan and sidecar paths are rejected.  Portable C/POSIX has no
 * rename-without-replacement operation, so this is an explicit precheck and
 * assumes the publication directory is controlled by the caller.  A racing
 * creator could still be replaced between the precheck and rename.
 *
 * The two artifacts are intentionally not claimed to be pair-atomic.  If the
 * plan rename succeeds and sidecar publication later fails, the complete plan
 * remains at FILE as unauthenticated evidence and this function returns false.
 * It never removes or rewrites that published plan during failure cleanup. */
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
