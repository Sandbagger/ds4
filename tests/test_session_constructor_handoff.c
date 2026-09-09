/* Native constructor-handoff fixture: actual ds4.c, no production seam. */
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#define DS4_NO_GPU 1
#define DS4_TEST_HOOKS 1
#include "../ds4.h"
#include "../ds4_distributed.h"
#include "../ds4_laguna_plan.h"
#include "../ds4_laguna_stream.h"
#include "../ds4_tp.h"
#include "../ds4_layer_pack.h"
#include "../ds4_gpu_mgpu.h"

/* Forward declarations keep the macro boundary local to the included core. */
typedef struct fixture_state fixture_state;
static fixture_state *g_fixture;
static void *fixture_calloc(size_t count, size_t size);
static void *fixture_malloc(size_t size);
static void *fixture_realloc(void *ptr, size_t size);
static void fixture_free(void *ptr);
static int fixture_mutex_lock(pthread_mutex_t *mutex);
static int fixture_mutex_unlock(pthread_mutex_t *mutex);

#define calloc fixture_calloc
#define malloc fixture_malloc
#define realloc fixture_realloc
#define free fixture_free
#define pthread_mutex_lock fixture_mutex_lock
#define pthread_mutex_unlock fixture_mutex_unlock
#include "../ds4.c"
#undef calloc
#undef malloc
#undef realloc
#undef free
#undef pthread_mutex_lock
#undef pthread_mutex_unlock

struct fixture_state {
    ds4_engine *engine;
    ds4_session **public_slot;
    ds4_session **cleanup_slot;
    void *live_payload;
    size_t live_bytes;
    size_t current_bytes;
    size_t peak_bytes;
    size_t calloc_calls;
    size_t calloc_bytes;
    size_t malloc_calls;
    size_t realloc_calls;
    size_t physical_free_calls;
    unsigned unexpected_allocations;
    unsigned unknown_free_calls;
    unsigned unknown_mutex_calls;
    unsigned real_lock_failures;
    unsigned real_unlock_failures;
    unsigned physical_runtime_locks;
    unsigned physical_runtime_unlocks;
    unsigned physical_exact_locks;
    unsigned physical_exact_unlocks;
    unsigned synthetic_runtime_unlock_failures;
    unsigned synthetic_cleanup_lock_refusals;
    unsigned order_violations;
    unsigned failed_expectations;
    unsigned sticky_errors;
    bool live_known;
    bool fail_runtime_unlock;
    bool arm_cleanup_lock_refusal;
    bool refuse_next_exact_lock;
    bool lost_handoff_recorded;
};

static ds4_engine g_engine;
typedef char fixture_engine_size_guard[
    sizeof(ds4_engine) <= 4u * 1024u * 1024u ? 1 : -1];
typedef char fixture_session_size_guard[
    sizeof(ds4_session) <= 4u * 1024u * 1024u ? 1 : -1];

enum { FIXTURE_MAX_BYTES = 4u * 1024u * 1024u };

static void fixture_error(void) {
    if (g_fixture) g_fixture->sticky_errors++;
}

static bool fixture_live_pointer(const void *ptr) {
    return g_fixture && g_fixture->live_known &&
           ptr != NULL && ptr == g_fixture->live_payload;
}

static void *fixture_calloc(size_t count, size_t size) {
    fixture_state *f = g_fixture;
    if (!f) return NULL;
    f->calloc_calls++;
    /* The budget is cumulative: every second attempt is refused before libc. */
    if (f->calloc_calls != 1) {
        f->unexpected_allocations++;
        fixture_error();
        return NULL;
    }
    if (size != 0 && count > SIZE_MAX / size) {
        f->unexpected_allocations++;
        fixture_error();
        return NULL;
    }
    const size_t total = count * size;
    if (count != 1 || size != sizeof(ds4_session) ||
        total > FIXTURE_MAX_BYTES || f->live_known) {
        f->unexpected_allocations++;
        fixture_error();
        return NULL;
    }
    void *ptr = calloc(count, size);
    if (!ptr) {
        f->unexpected_allocations++;
        fixture_error();
        return NULL;
    }
    f->live_payload = ptr;
    f->live_bytes = total;
    f->current_bytes = total;
    f->peak_bytes = total;
    f->calloc_bytes = total;
    f->live_known = true;
    const unsigned char *bytes = (const unsigned char *)ptr;
    for (size_t i = 0; i < total; i++) {
        if (bytes[i] != 0) {
            fixture_error();
            break;
        }
    }
    return ptr;
}

static void *fixture_malloc(size_t size) {
    if (g_fixture) {
        g_fixture->malloc_calls++;
        g_fixture->unexpected_allocations++;
        fixture_error();
    }
    (void)size;
    return NULL;
}

static void *fixture_realloc(void *ptr, size_t size) {
    if (g_fixture) {
        g_fixture->realloc_calls++;
        g_fixture->unexpected_allocations++;
        fixture_error();
    }
    (void)ptr;
    (void)size;
    return NULL;
}

static void fixture_free(void *ptr) {
    fixture_state *f = g_fixture;
    if (!ptr) return;
    if (!fixture_live_pointer(ptr)) {
        if (f) {
            f->unknown_free_calls++;
            fixture_error();
        }
        return;
    }

    /* All payload reads happen only after known-live membership succeeds. */
    ds4_session *session = (ds4_session *)ptr;
    bool published = (f->public_slot && *f->public_slot == session) ||
                     (f->cleanup_slot && *f->cleanup_slot == session);
    if (!published || session->engine != f->engine ||
        session->exact_cache_session_reserved ||
        f->engine->exact_cache_active_sessions != 0) {
        f->order_violations++;
        fixture_error();
    }

    /* Physical free precedes retirement of the witness and its byte total. */
    free(ptr);
    f->physical_free_calls++;
    f->live_known = false;
    f->live_payload = NULL;
    f->live_bytes = 0;
    f->current_bytes = 0;
}

static int fixture_mutex_kind(pthread_mutex_t *mutex) {
    fixture_state *f = g_fixture;
    if (!f || !f->engine) return 0;
    if (mutex == &f->engine->runtime_snapshot_mutex) return 1;
    if (mutex == &f->engine->exact_cache_session_mutex) return 2;
    return 0;
}

static int fixture_mutex_lock(pthread_mutex_t *mutex) {
    fixture_state *f = g_fixture;
    const int kind = fixture_mutex_kind(mutex);
    if (!kind) {
        if (f) {
            f->unknown_mutex_calls++;
            fixture_error();
        }
        return EINVAL;
    }
    if (kind == 2 && f->refuse_next_exact_lock) {
        /* Synthetic control-flow refusal: no pthread lock effect occurs. */
        f->refuse_next_exact_lock = false;
        f->synthetic_cleanup_lock_refusals++;
        return EBUSY;
    }
    const int rc = pthread_mutex_lock(mutex);
    if (rc != 0) {
        f->real_lock_failures++;
        fixture_error();
    } else if (kind == 1) {
        f->physical_runtime_locks++;
    } else {
        f->physical_exact_locks++;
    }
    return rc;
}

static int fixture_mutex_unlock(pthread_mutex_t *mutex) {
    fixture_state *f = g_fixture;
    const int kind = fixture_mutex_kind(mutex);
    if (!kind) {
        if (f) {
            f->unknown_mutex_calls++;
            fixture_error();
        }
        return EINVAL;
    }
    const int rc = pthread_mutex_unlock(mutex);
    if (rc != 0) {
        f->real_unlock_failures++;
        fixture_error();
        return rc;
    }
    if (kind == 1) {
        f->physical_runtime_unlocks++;
        if (f->fail_runtime_unlock) {
            /* Physical unlock succeeded; only the observed result is synthetic. */
            f->fail_runtime_unlock = false;
            f->synthetic_runtime_unlock_failures++;
            if (f->arm_cleanup_lock_refusal) f->refuse_next_exact_lock = true;
            return EIO;
        }
    } else {
        f->physical_exact_unlocks++;
    }
    return 0;
}

static bool fixture_init(fixture_state *f, bool inject_cleanup_refusal) {
    memset(&g_engine, 0, sizeof(g_engine));
    memset(f, 0, sizeof(*f));
    f->engine = &g_engine;
    f->fail_runtime_unlock = inject_cleanup_refusal;
    f->arm_cleanup_lock_refusal = inject_cleanup_refusal;
    g_fixture = f;

    g_engine.backend = DS4_BACKEND_CUDA;
    g_engine.ssd_streaming_cache_bytes_set = true;
    g_engine.ssd_streaming_cache_bytes = 1;
    g_engine.exact_cache_context_limit = 32768;
    g_engine.exact_cache_session_limit = 1u;
    g_engine.laguna_compact_runtime = true;
    g_engine.laguna_allocation_plan.context_tokens = 32768u;
    g_engine.test_session_create_no_alloc = true;

    int rc = pthread_mutex_init(&g_engine.runtime_snapshot_mutex, NULL);
    if (rc != 0) {
        fixture_error();
        return false;
    }
    g_engine.runtime_snapshot_mutex_initialized = true;
    rc = pthread_mutex_init(&g_engine.exact_cache_session_mutex, NULL);
    if (rc != 0) {
        fixture_error();
        if (pthread_mutex_destroy(&g_engine.runtime_snapshot_mutex) == 0)
            g_engine.runtime_snapshot_mutex_initialized = false;
        return false;
    }
    g_engine.exact_cache_session_mutex_initialized = true;
    return true;
}

static bool fixture_destroy(fixture_state *f) {
    /* Latch refusal before either destroy while any owner/current/count remains. */
    if (!f || !f->engine || f->live_known || f->live_payload != NULL ||
        f->live_bytes != 0 || f->current_bytes != 0 ||
        f->engine->exact_cache_active_sessions != 0) {
        if (f) {
            f->order_violations++;
            fixture_error();
        }
        return false;
    }
    bool ok = true;
    if (g_engine.exact_cache_session_mutex_initialized) {
        const int rc = pthread_mutex_destroy(&g_engine.exact_cache_session_mutex);
        if (rc != 0) ok = false;
        else g_engine.exact_cache_session_mutex_initialized = false;
    }
    if (g_engine.runtime_snapshot_mutex_initialized) {
        const int rc = pthread_mutex_destroy(&g_engine.runtime_snapshot_mutex);
        if (rc != 0) ok = false;
        else g_engine.runtime_snapshot_mutex_initialized = false;
    }
    if (!ok) {
        f->order_violations++;
        fixture_error();
    }
    return ok;
}

static bool fixture_heap_clean(const fixture_state *f) {
    return f->calloc_calls == 1 && f->calloc_bytes == sizeof(ds4_session) &&
           f->malloc_calls == 0 && f->realloc_calls == 0 &&
           f->unexpected_allocations == 0 && f->physical_free_calls == 1 &&
           f->peak_bytes == sizeof(ds4_session) && f->live_bytes == 0 &&
           f->current_bytes == 0 && f->order_violations == 0 &&
           sizeof(ds4_engine) <= FIXTURE_MAX_BYTES &&
           sizeof(ds4_session) <= FIXTURE_MAX_BYTES;
}

static void fixture_failed_expectation(fixture_state *f, const char *label) {
    f->failed_expectations++;
    fprintf(stderr, "fixture expectation failed before cleanup: %s\n", label);
}

static int fixture_attempt_checked_cleanup(fixture_state *f,
                                           ds4_session **public_slot,
                                           ds4_session **witness_slot) {
    ds4_session **slot = NULL;
    if (public_slot && fixture_live_pointer(*public_slot)) {
        slot = public_slot;
    } else if (f->live_known && f->live_payload != NULL) {
        *witness_slot = (ds4_session *)f->live_payload;
        slot = witness_slot;
    }
    if (!slot) return 2;
    f->cleanup_slot = slot;
    return ds4_session_free_checked(slot);
}

static bool fixture_all_clean(const fixture_state *f) {
    return fixture_heap_clean(f) && f->unknown_free_calls == 0 &&
           f->unknown_mutex_calls == 0 && f->real_lock_failures == 0 &&
           f->real_unlock_failures == 0 && f->failed_expectations == 0 &&
           f->sticky_errors == 0 && !f->live_known &&
           f->live_payload == NULL && f->live_bytes == 0 &&
           f->current_bytes == 0;
}

static int run_success(void) {
    fixture_state f;
    if (!fixture_init(&f, false)) return 2;
    ds4_session *owner = NULL;
    ds4_session *witness_slot = NULL;
    f.public_slot = &owner;
    const int create_rc = ds4_session_create(&owner, &g_engine, 32768);
    const bool control_ok = create_rc == 0 && fixture_live_pointer(owner) &&
        owner->engine == &g_engine && owner->exact_cache_session_reserved &&
        g_engine.exact_cache_active_sessions == 1u &&
        f.synthetic_runtime_unlock_failures == 0 &&
        f.synthetic_cleanup_lock_refusals == 0 && f.sticky_errors == 0;
    if (!control_ok)
        fixture_failed_expectation(&f, "success returned live owner/reservation");
    const int release_rc = control_ok
        ? ds4_session_free_checked(&owner)
        : fixture_attempt_checked_cleanup(&f, &owner, &witness_slot);
    const bool cleanup_consumed = release_rc == 1 && owner == NULL &&
        witness_slot == NULL && !f.live_known && f.live_payload == NULL &&
        f.current_bytes == 0 && g_engine.exact_cache_active_sessions == 0 &&
        f.physical_free_calls == 1;
    const bool cleanup_effects = f.physical_runtime_locks == 1 &&
        f.physical_runtime_unlocks == 1 && f.physical_exact_locks == 2 &&
        f.physical_exact_unlocks == 2 && f.real_lock_failures == 0 &&
        f.real_unlock_failures == 0 && f.synthetic_runtime_unlock_failures == 0 &&
        f.synthetic_cleanup_lock_refusals == 0 && f.unknown_mutex_calls == 0;
    const bool destroyed = fixture_destroy(&f);
    const bool all_cleanliness = fixture_all_clean(&f);
    printf("case=success create=%d control_ok=%d cleanup_consumed=%d "
           "cleanup_effects=%d destroyed=%d all_cleanliness=%d "
           "release_rc=%d failed_expectations=%u\n",
           create_rc, control_ok, cleanup_consumed, cleanup_effects, destroyed,
           all_cleanliness, release_rc, f.failed_expectations);
    return control_ok && cleanup_consumed && cleanup_effects && destroyed &&
        all_cleanliness ? 0 : 2;
}

static int run_unlock_cleanup_consumed(void) {
    fixture_state f;
    if (!fixture_init(&f, false)) return 2;
    f.fail_runtime_unlock = true;
    ds4_session *owner = NULL;
    ds4_session *witness_slot = NULL;
    f.public_slot = &owner;
    const int create_rc = ds4_session_create(&owner, &g_engine, 32768);
    const bool control_ok = create_rc == 2 && owner == NULL && !f.live_known &&
        f.live_payload == NULL && f.current_bytes == 0 &&
        g_engine.exact_cache_active_sessions == 0 && f.physical_free_calls == 1 &&
        f.synthetic_runtime_unlock_failures == 1 &&
        f.synthetic_cleanup_lock_refusals == 0 &&
        f.physical_runtime_locks == 1 && f.physical_runtime_unlocks == 1 &&
        f.physical_exact_locks == 2 && f.physical_exact_unlocks == 2 &&
        f.real_lock_failures == 0 && f.real_unlock_failures == 0 &&
        f.unknown_mutex_calls == 0 && f.sticky_errors == 0;
    if (!control_ok)
        fixture_failed_expectation(&f, "unlock-cleanup-consumed consumed owner");
    const int cleanup_rc = control_ok
        ? 1 : fixture_attempt_checked_cleanup(&f, &owner, &witness_slot);
    const bool cleanup_consumed = cleanup_rc == 1 && owner == NULL &&
        witness_slot == NULL && !f.live_known && f.live_payload == NULL &&
        f.current_bytes == 0 && g_engine.exact_cache_active_sessions == 0 &&
        f.physical_free_calls == 1;
    const bool cleanup_effects = f.synthetic_runtime_unlock_failures == 1 &&
        f.synthetic_cleanup_lock_refusals == 0 && f.physical_runtime_locks == 1 &&
        f.physical_runtime_unlocks == 1 && f.physical_exact_locks == 2 &&
        f.physical_exact_unlocks == 2 && f.real_lock_failures == 0 &&
        f.real_unlock_failures == 0;
    const bool destroyed = fixture_destroy(&f);
    const bool all_cleanliness = fixture_all_clean(&f);
    printf("case=unlock-cleanup-consumed create=%d control_ok=%d "
           "cleanup_consumed=%d cleanup_effects=%d destroyed=%d "
           "all_cleanliness=%d cleanup_rc=%d failed_expectations=%u\n",
           create_rc, control_ok, cleanup_consumed, cleanup_effects, destroyed,
           all_cleanliness, cleanup_rc, f.failed_expectations);
    return control_ok && cleanup_consumed && cleanup_effects && destroyed &&
        all_cleanliness ? 0 : 2;
}

static int run_unlock_cleanup_retained(void) {
    fixture_state f;
    if (!fixture_init(&f, true)) return 2;
    ds4_session *owner = NULL;
    ds4_session *witness_slot = NULL;
    f.public_slot = &owner;
    const int create_rc = ds4_session_create(&owner, &g_engine, 32768);
    const bool witness_live = f.live_known && f.live_payload != NULL;
    ds4_session *witness = witness_live
        ? (ds4_session *)f.live_payload : NULL;
    const bool output_retained = create_rc == 2 && owner == witness &&
        witness_live && witness->engine == &g_engine &&
        witness->exact_cache_session_reserved &&
        g_engine.exact_cache_active_sessions == 1u;
    const bool output_lost = create_rc == 2 && owner == NULL && witness_live &&
        witness->engine == &g_engine && witness->exact_cache_session_reserved &&
        g_engine.exact_cache_active_sessions == 1u;
    const bool control_ok = (output_retained || output_lost) &&
        f.synthetic_runtime_unlock_failures == 1 &&
        f.synthetic_cleanup_lock_refusals == 1 &&
        f.physical_runtime_locks == 1 && f.physical_runtime_unlocks == 1 &&
        f.physical_exact_locks == 1 && f.physical_exact_unlocks == 1 &&
        f.real_lock_failures == 0 && f.real_unlock_failures == 0 &&
        f.unknown_mutex_calls == 0 && f.unexpected_allocations == 0 &&
        f.sticky_errors == 0;

    /* Record the feature assertion before any witness-based cleanup. */
    if (output_lost) {
        f.lost_handoff_recorded = true;
        fprintf(stderr, "RED candidate: post-success unlock lost live output handoff\n");
    }
    if (!control_ok)
        fixture_failed_expectation(&f, "retained control and handoff state");

    const int retry_rc = witness_live
        ? fixture_attempt_checked_cleanup(&f, &owner, &witness_slot) : 2;
    const bool retry_consumed = retry_rc == 1 && owner == NULL &&
        witness_slot == NULL && !f.live_known && f.live_payload == NULL &&
        f.current_bytes == 0 && g_engine.exact_cache_active_sessions == 0 &&
        f.physical_free_calls == 1;
    const bool cleanup_effects = f.synthetic_runtime_unlock_failures == 1 &&
        f.synthetic_cleanup_lock_refusals == 1 && f.physical_runtime_locks == 1 &&
        f.physical_runtime_unlocks == 1 && f.physical_exact_locks == 2 &&
        f.physical_exact_unlocks == 2 && f.real_lock_failures == 0 &&
        f.real_unlock_failures == 0;
    const bool destroyed = fixture_destroy(&f);
    const bool all_cleanliness = fixture_all_clean(&f);
    printf("case=unlock-cleanup-retained create=%d control_ok=%d retained=%d "
           "lost=%d recorded=%d retry_consumed=%d cleanup_effects=%d "
           "destroyed=%d all_cleanliness=%d retry_rc=%d "
           "failed_expectations=%u\n",
           create_rc, control_ok, output_retained, output_lost,
           f.lost_handoff_recorded, retry_consumed, cleanup_effects, destroyed,
           all_cleanliness, retry_rc, f.failed_expectations);
    if (output_lost && f.lost_handoff_recorded && control_ok && retry_consumed &&
        cleanup_effects && destroyed && all_cleanliness)
        return 1;
    if (output_retained && control_ok && retry_consumed && cleanup_effects &&
        destroyed && all_cleanliness)
        return 0;
    return 2;
}

int main(int argc, char **argv) {
    alarm(15);
    const struct rlimit core_limit = {0, 0};
    if (setrlimit(RLIMIT_CORE, &core_limit) != 0 || argc != 2) return 2;
    if (strcmp(argv[1], "success") == 0) return run_success();
    if (strcmp(argv[1], "unlock-cleanup-consumed") == 0)
        return run_unlock_cleanup_consumed();
    if (strcmp(argv[1], "unlock-cleanup-retained") == 0)
        return run_unlock_cleanup_retained();
    return 2;
}
