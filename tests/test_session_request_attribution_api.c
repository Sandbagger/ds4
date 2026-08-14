#include "ds4.h"

#include <string.h>

/* Public compile/link contract for explicit, operation-scoped request
 * attribution.  Existing session APIs remain source/ABI-compatible wrappers
 * with NULL attribution; request identity never becomes ambient session state.
 * Callers own each context and keep it alive until request_barrier returns. */
int main(void) {
    ds4_session *session = NULL;
    ds4_tokens prompt = {0};
    ds4_runtime_request_context request_a = {0};
    ds4_runtime_request_context request_b = {0};
    char error[128] = {0};

    ds4_attributed_decode_item decode[] = {
        { .session = session, .token = 1, .request = &request_a },
        { .session = session, .token = 2, .request = &request_b },
    };

    (void)ds4_session_sync_attributed(
        session, &prompt, &request_a, error, sizeof(error));
    (void)ds4_session_eval_attributed(
        session, 1, &request_a, error, sizeof(error));
    (void)ds4_sessions_eval_batch_attributed(
        decode, 2, error, sizeof(error));
    (void)ds4_sessions_eval_batch_with_prefill_attributed(
        decode, 2, session, &prompt, &request_b, error, sizeof(error));
    (void)ds4_session_request_barrier(
        session, &request_a, error, sizeof(error));

    return decode[0].session != session ||
        decode[0].token != 1 || decode[0].request != &request_a;
}
