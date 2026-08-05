#include "ds4_gpu.h"
#include "ds4_laguna_stream.h"
#include "ds4_runtime.h"

#include <type_traits>

using laguna_compact_create_fn = int (*)(
    ds4_gpu_laguna_compact **,
    int,
    const void *,
    uint64_t,
    const ds4_laguna_ledger *,
    const ds4_laguna_allocation_plan *,
    ds4_runtime_tracker *);
using laguna_compact_destroy_fn = void (*)(ds4_gpu_laguna_compact *);

static_assert(std::is_same<
              decltype(&ds4_gpu_laguna_compact_create),
              laguna_compact_create_fn>::value,
              "compact create C ABI drifted");
static_assert(std::is_same<
              decltype(&ds4_gpu_laguna_compact_destroy),
              laguna_compact_destroy_fn>::value,
              "compact destroy C ABI drifted");

int main() {
    const uint64_t gib = 1024ull * 1024ull * 1024ull;
    uint64_t reduction = 0;
    if (!ds4_runtime_reduction_qualified(80ull * gib, 44ull * gib,
                                         &reduction) ||
        reduction != 36ull * gib) {
        return 1;
    }

    ds4_laguna_page_range output[1];
    size_t output_count = 1;
    uint64_t output_bytes = 1;
    if (!ds4_laguna_full_page_union(nullptr, 0, 4096u,
                                     output, 1,
                                     &output_count, &output_bytes) ||
        output_count != 0 || output_bytes != 0) {
        return 2;
    }
    return 0;
}
