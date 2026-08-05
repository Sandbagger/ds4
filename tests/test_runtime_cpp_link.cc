#include "ds4_laguna_stream.h"
#include "ds4_runtime.h"

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
