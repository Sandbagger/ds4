#ifndef DS4_BUILD_INFO_H
#define DS4_BUILD_INFO_H

#include <stdio.h>

#include "ds4_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Return the immutable identity stamped into this backend's frontend object. */
const ds4_runtime_build_info *ds4_build_info_get(void);

/* Write one closed ds4.version/v1 JSON value followed by a newline. */
int ds4_build_info_write_json(FILE *stream);

/* Handle the terminal --version-json frontend mode before other parsing. */
int ds4_build_info_maybe_print_version(int argc, char **argv,
                                       const char *option, int *handled);

#ifdef __cplusplus
}
#endif

#endif
