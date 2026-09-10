#include <errno.h>
#include <inttypes.h>
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
typedef struct fixture_state fixture_state;
static fixture_state *g_fixture;
static void *fixture_calloc(size_t,size_t);
static void *fixture_malloc(size_t);
static void *fixture_realloc(void *,size_t);
static void fixture_free(void *);
static int fixture_mutex_lock(pthread_mutex_t *);
static int fixture_mutex_unlock(pthread_mutex_t *);
static int fixture_mutex_destroy(pthread_mutex_t *);
#define calloc fixture_calloc
#define malloc fixture_malloc
#define realloc fixture_realloc
#define free fixture_free
#define pthread_mutex_lock fixture_mutex_lock
#define pthread_mutex_unlock fixture_mutex_unlock
#define pthread_mutex_destroy fixture_mutex_destroy
#include "../ds4.c"
#undef calloc
#undef malloc
#undef realloc
#undef free
#undef pthread_mutex_lock
#undef pthread_mutex_unlock
#undef pthread_mutex_destroy
enum { FX_CTX=32768,FX_NS=2,FX_NC=3,FX_NM=512,
 FX_OUT=32768,FX_OWNER=4u*1024u*1024u,
   FX_HEAP=12u*1024u*1024u };
typedef char fx_engine_size_guard[sizeof(ds4_engine)<=FX_OWNER?1:-1];
typedef char fx_session_size_guard[sizeof(ds4_session)<=FX_OWNER?1:-1];
struct fx_rel {
 bool call,attempted,rc_avail,be,bs,baa,bra,ae_avail,ae,as_avail,as,aaa,ara,ar,aom_avail,aom,semantic_mismatch; int rc; uint32_t ba,bra_v,aaa_v,ara_v;
};
struct fx_close {
 bool recorded,call,called,skipped,bn_avail,be_avail,be,baa,bra,brm,bem,ae_avail,ae_storage_avail,ae,aaa,ara,an_avail,arm,are,source_probe,source_free_marker;
 uint32_t bn,ba,bra_v,ae_v,ara_v,an;
};
struct fx_snap {
 bool saved,eng,ses,aa,rr,rm,em; uint32_t active; bool reserved; int release_rc,fault_real_rc,fault_return_rc; unsigned arms,selection,synthetic,real_successes,
 exact_locks,exact_unlocks,session_frees,engine_frees;
};
struct fx_output {
 char text[FX_OUT]; char keys[FX_NM][64]; size_t len,n; bool overflow;
};
struct fixture_state {
 ds4_engine*engine; uintptr_t engine_addr; size_t engine_bytes,session_bytes_total,core_calloc_calls,core_calloc_bytes,physical_free_calls; bool engine_acquired,engine_live;
 unsigned engine_source_free,engine_rescue_free; ds4_session*owner[FX_NS],*session[FX_NS]; uintptr_t session_addr[FX_NS],borrow_addr[FX_NS]; size_t session_bytes[FX_NS];
 bool acquired[FX_NS],session_live[FX_NS],test_no_alloc[FX_NS],payload_empty[FX_NS],borrow_avail[FX_NS],reserved_avail[FX_NS],
 reserved[FX_NS],create_attempted[FX_NS],create_success[FX_NS],active_after_create_avail[FX_NS]; uint32_t active_after_create[FX_NS]; int create_rc[FX_NS];
 unsigned session_source_free[FX_NS],session_rescue_free[FX_NS];
  struct fx_rel release[FX_NS],first_release,retry_release;
 unsigned checked_releases,semantic_mismatches; uintptr_t runtime_mutex_addr,exact_mutex_addr; bool runtime_init_attempted,exact_init_attempted,runtime_initialized,
 exact_initialized,runtime_mutex_live,exact_mutex_live,runtime_destroy_attempted,exact_destroy_attempted;
 int runtime_init_rc,exact_init_rc,runtime_destroy_rc,exact_destroy_rc; unsigned runtime_source_destroy,exact_source_destroy,runtime_rescue_destroy,exact_rescue_destroy;
 unsigned runtime_locks,runtime_unlocks,exact_locks,exact_unlocks,real_lock_failures,real_unlock_failures,real_destroy_failures,unknown_mutex; bool exact_fault_armed;
 unsigned fault_arms,fault_selections,fault_synthetic,fault_real_successes; int fault_real_rc,fault_return_rc;
  struct fx_close close[FX_NC];
 bool close_in_progress,close_source_probe,close_source_free_marker; uint32_t close_before_sessions; unsigned product_source_probe_count,product_source_marker_count;
 bool product_premature_free;
  struct fx_snap first_snapshot,first_snapshot_copy;
 bool first_snapshot_copy_valid,first_snapshot_equal_before_available,first_snapshot_equal_before,first_snapshot_equal_after_available,first_snapshot_equal_after;
 bool post_close_saved,post_close_eng_avail,post_close_eng_storage_avail,post_close_eng,post_close_ses,post_close_aa,post_close_rr,
 post_close_rm,post_close_em,post_close_reserved; uint32_t post_close_active; bool post_retry_saved,post_retry_eng_avail,post_retry_eng_storage_avail,
 post_retry_eng,post_retry_ses,post_retry_aa,post_retry_rr,post_retry_reserved; uint32_t post_retry_active; bool pre_rescue_saved,pre_rescue_eng,
 pre_rescue_session_live[FX_NS],pre_rescue_runtime_mutex_live,pre_rescue_exact_mutex_live; uint32_t pre_rescue_sessions;
 unsigned pre_rescue_engine_source_free,pre_rescue_engine_rescue_free,pre_rescue_session_source_free[FX_NS],pre_rescue_session_rescue_free[FX_NS],
 pre_rescue_runtime_source_destroy,pre_rescue_exact_source_destroy,pre_rescue_runtime_rescue_destroy,pre_rescue_exact_rescue_destroy;
 bool rescue_proof_available,rescue_proof_ok,rescue_final_residual_zero,rescue_source_effects_unchanged; unsigned rescue_engine_delta,rescue_engine_expected,
 rescue_runtime_delta,rescue_runtime_expected,rescue_exact_delta,rescue_exact_expected,rescue_session_delta[FX_NS],rescue_session_expected[FX_NS];
 unsigned unexpected_alloc,unknown_free,order_errors,sticky; bool setup_profile,setup_pool,setup_lock,setup_engine,setup_tracker,setup_ok; uint32_t configured_limit;
};
static void fx_error(fixture_state *f) {
  if (f != NULL) {
 f->sticky++;
  }
}
static void fx_order(fixture_state *f) {
  if (f != NULL) {
 f->order_errors++; fx_error(f);
  }
}
static bool fx_zero(const void *p,size_t n) {
  if (p == NULL) {
    return false;
  }
 const unsigned char*b=(const unsigned char*)p;
  for (size_t i = 0; i < n; i++) {
    if (b[i] != 0) {
      return false;
    }
  }
  return true;
}
static bool fx_model_empty(const ds4_model *m) {
  return m != NULL && m->fd == -1 && m->map == NULL && m->size == 0 &&
 m->n_kv==0&&m->n_tensors==0&&m->alignment==0&& m->header_end==0&&m->metadata_end==0&& m->tensor_directory_end==0&&m->tensor_data_pos==0&&
 m->max_tensor_bytes==0&&m->kv==NULL&&m->tensors==NULL&& fx_zero(&m->identity,sizeof(m->identity));
}
static bool fx_engine_empty(const ds4_engine *e) {
  return e != NULL && fx_model_empty(&e->model) && fx_model_empty(&e->mtp_model) &&
 fx_zero(&e->vocab,sizeof(e->vocab))&& fx_zero(&e->weights,sizeof(e->weights))&& fx_zero(&e->laguna_ledger,sizeof(e->laguna_ledger))&&
 fx_zero(&e->laguna_allocation_plan,sizeof(e->laguna_allocation_plan))&& fx_zero(&e->laguna_runtime_tracker,sizeof(e->laguna_runtime_tracker))&&
 fx_zero(&e->runtime_snapshot_context,sizeof(e->runtime_snapshot_context))&& fx_zero(&e->runtime_snapshot_input,sizeof(e->runtime_snapshot_input))&&
!e->runtime_snapshot_mutex_initialized&&!e->runtime_snapshot_ready&& e->qualification_control==NULL&& e->qualification_checkpoint_sequence==0&&
 e->directional_steering_file==NULL&& e->directional_steering_dirs==NULL&& fx_zero(e->laguna_ledger_record_ids,sizeof(e->laguna_ledger_record_ids))&&
 fx_zero(e->laguna_ledger_records_live,sizeof(e->laguna_ledger_records_live))&& fx_zero(e->laguna_inventory_record_ids,sizeof(e->laguna_inventory_record_ids))&&
 fx_zero(e->laguna_inventory_records_live,sizeof(e->laguna_inventory_records_live))&& !e->laguna_compact_runtime&&!e->laguna_compact_create_attempted&&
!e->laguna_runtime_tracker_ready&& fx_zero(&e->mtp_weights,sizeof(e->mtp_weights))&& fx_zero(&e->dspark_weights,sizeof(e->dspark_weights))&& !e->metal_ready&&!e->mtp_ready&&
 e->distributed.role==DS4_DISTRIBUTED_NONE&& fx_zero(&e->distributed,sizeof(e->distributed))&& fx_zero(&e->tp,sizeof(e->tp))&&
 fx_zero(&e->simulated_memory,sizeof(e->simulated_memory))&& fx_zero(&e->gpu_cfg,sizeof(e->gpu_cfg))&& fx_zero(&e->placement,sizeof(e->placement))&&
 e->n_placement_entries==0&&e->multi_tier==0&& e->placement_ctx_hint==0;
}
static unsigned fx_live_sessions(const fixture_state *f) {
  if (f == NULL) {
    return 0;
  }
 unsigned n=0;
  for (size_t i = 0; i < FX_NS; i++) {
    if (f->session_live[i]) {
 n++;
    }
  }
  return n;
}
static unsigned fx_reserved_count(const fixture_state *f) {
  if (f == NULL) {
    return 0;
  }
 unsigned n=0;
  for (size_t i = 0; i < FX_NS; i++) {
    if (f->session_live[i] && f->session[i] != NULL &&
      f->session[i]->exact_cache_session_reserved) {
 n++;
    }
  }
  return n;
}
static bool fx_payload_empty(const ds4_session *s) {
  return s != NULL && s->test_no_alloc && s->distributed == NULL &&
 s->checkpoint.v==NULL&&s->greedy_splitkv_segment.v==NULL&& s->logits==NULL&&s->sample_probs==NULL&& s->mtp_logits==NULL&&fx_zero(&s->cpu_cache,sizeof(s->cpu_cache))&&
 fx_zero(&s->cpu_scratch,sizeof(s->cpu_scratch));
}
static bool fx_snap_equal(const struct fx_snap *a,const struct fx_snap *b) {
  if (a == NULL || b == NULL) {
    return false;
  }
  return memcmp(a,b,sizeof(*a)) == 0;
}
static int fx_mutex_kind(pthread_mutex_t *m) {
 fixture_state*f=g_fixture;
  if (f == NULL || !f->engine_live || f->engine == NULL || m == NULL) {
    return 0;
  }
  if (m == &f->engine->runtime_snapshot_mutex) {
    return 1;
  }
  if (m == &f->engine->exact_cache_session_mutex) {
    return 2;
  }
  return 0;
}
static bool fx_mutex_live(fixture_state *f,int k) {
  return f != NULL && ((k == 1 && f->runtime_mutex_live) ||
 (k==2&&f->exact_mutex_live));
}
static void *fixture_calloc(size_t count,size_t size) {
 fixture_state*f=g_fixture;
  if (f == NULL) {
    return NULL;
  }
 f->core_calloc_calls++;
  if (size != 0 && count > SIZE_MAX / size) {
 f->unexpected_alloc++; fx_error(f);
    return NULL;
  }
 size_t total=count*size;
  if (total > SIZE_MAX - f->core_calloc_bytes) {
 f->unexpected_alloc++; fx_error(f);
    return NULL;
  }
 f->core_calloc_bytes+=total;
  if (!f->engine_live || count != 1 || size != sizeof(ds4_session) ||
 total==0||total>FX_OWNER||f->engine_bytes>FX_HEAP|| f->session_bytes_total>FX_HEAP-f->engine_bytes||
    total > FX_HEAP - f->engine_bytes - f->session_bytes_total) {
 f->unexpected_alloc++; fx_error(f);
    return NULL;
  }
 size_t ix=FX_NS;
  for (size_t i = 0; i < FX_NS; i++) {
    if (!f->acquired[i]) {
 ix=i; break;
    }
  }
  if (ix == FX_NS) {
 f->unexpected_alloc++; fx_error(f);
    return NULL;
  }
 void*p=calloc(count,size);
  if (p == NULL) {
 f->unexpected_alloc++; fx_error(f);
    return NULL;
  }
 f->acquired[ix]=true; f->session_live[ix]=true; f->session[ix]=(ds4_session*)p; f->session_addr[ix]=(uintptr_t)p; f->session_bytes[ix]=total; f->session_bytes_total+=total;
 const unsigned char*b=(const unsigned char*)p;
  for (size_t i = 0; i < total; i++) {
    if (b[i] != 0) {
 f->unexpected_alloc++; fx_error(f); break;
    }
  }
  return p;
}
static void *fixture_malloc(size_t n) {
 fixture_state*f=g_fixture; (void)n;
  if (f != NULL) {
 f->unexpected_alloc++; fx_error(f);
  }
  return NULL;
}
static void *fixture_realloc(void *p,size_t n) {
 fixture_state*f=g_fixture; (void)p; (void)n;
  if (f != NULL) {
 f->unexpected_alloc++; fx_error(f);
  }
  return NULL;
}
static void fixture_free(void *p) {
 fixture_state*f=g_fixture;
  if (p == NULL) {
 return;
  }
  if (f == NULL) {
    fputs("fixture: non-NULL free before owned context\n",stderr);
    _Exit(125);
  }
  if (!f->engine_live || f->engine == NULL) {
 f->unknown_free++; fx_error(f); return;
  }
  if (p == f->engine) {
    if (f->runtime_mutex_live || f->exact_mutex_live) {
 fx_order(f);
    }
    if (f->close_in_progress && f->close_source_probe) {
 f->product_premature_free=true; f->close_source_free_marker=true; f->product_source_marker_count++;
    }
 free(p); f->physical_free_calls++; f->engine_source_free++; f->engine_live=false; f->engine=NULL; return;
  }
 size_t ix=FX_NS;
  for (size_t i = 0; i < FX_NS; i++) {
    if (f->session_live[i] && p == f->session[i]) {
 ix=i; break;
    }
  }
  if (ix == FX_NS) {
 f->unknown_free++; fx_error(f); return;
  }
 ds4_session*s=f->session[ix];
  if (f->owner[ix] != s || s->engine != f->engine ||
    !f->payload_empty[ix]) {
 fx_order(f);
  }
 free(p); f->physical_free_calls++; f->session_source_free[ix]++; f->session[ix]=NULL; f->session_live[ix]=false;
}
static int fixture_mutex_lock(pthread_mutex_t *m) {
 fixture_state*f=g_fixture; int k=fx_mutex_kind(m);
  if (!fx_mutex_live(f,k)) {
    if (f != NULL) {
 f->unknown_mutex++; fx_error(f);
    }
    return EINVAL;
  }
 int rc=pthread_mutex_lock(m);
  if (rc != 0) {
 f->real_lock_failures++; fx_error(f);
  } else if (k == 1) {
 f->runtime_locks++;
  } else {
 f->exact_locks++;
  }
  return rc;
}
static int fixture_mutex_unlock(pthread_mutex_t *m) {
 fixture_state*f=g_fixture; int k=fx_mutex_kind(m);
  if (!fx_mutex_live(f,k)) {
    if (f != NULL) {
 f->unknown_mutex++; fx_error(f);
    }
    return EINVAL;
  }
 bool selected=k==2&&f->exact_fault_armed;
  if (selected) {
 f->exact_fault_armed=false; f->fault_selections++;
  }
 int rc=pthread_mutex_unlock(m);
  if (rc != 0) {
 f->real_unlock_failures++; fx_error(f);
    if (selected) {
 f->fault_real_rc=rc; f->fault_return_rc=rc;
    }
    return rc;
  }
  if (k == 1) {
 f->runtime_unlocks++;
  } else {
 f->exact_unlocks++;
  }
  if (!selected) {
    return 0;
  }
 f->fault_real_rc=0; f->fault_return_rc=EPERM; f->fault_real_successes++; f->fault_synthetic++;
  return EPERM;
}
static int fixture_mutex_destroy(pthread_mutex_t *m) {
 fixture_state*f=g_fixture; int k=fx_mutex_kind(m);
  if (!fx_mutex_live(f,k)) {
    if (f != NULL) {
 f->unknown_mutex++; fx_error(f);
    }
    return EINVAL;
  }
 int rc=pthread_mutex_destroy(m);
  if (k == 1) {
 f->runtime_destroy_attempted=true; f->runtime_destroy_rc=rc;
    if (rc == 0) {
 f->runtime_source_destroy++; f->runtime_mutex_live=false;
    }
  } else {
 f->exact_destroy_attempted=true; f->exact_destroy_rc=rc;
    if (rc == 0) {
 f->exact_source_destroy++; f->exact_mutex_live=false;
    }
  }
  if (rc != 0) {
 f->real_destroy_failures++; fx_error(f);
  }
  return rc;
}
static bool fx_rescue_session(fixture_state *f,size_t i) {
  if (f == NULL || i >= FX_NS || !f->session_live[i] ||
    f->session[i] == NULL || !f->payload_empty[i]) {
    if (f != NULL) {
 fx_order(f);
    }
    return false;
  }
 ds4_session*s=f->session[i]; free(s); f->physical_free_calls++; f->session_rescue_free[i]++; f->session[i]=NULL; f->session_live[i]=false; f->owner[i]=NULL;
  return true;
}
static bool fx_rescue_engine(fixture_state *f) {
  if (f == NULL) {
    return false;
  }
  if (!f->engine_live) {
    return true;
  }
  if (f->engine == NULL || fx_live_sessions(f) != 0) {
 fx_order(f);
    return false;
  }
  if (f->exact_mutex_live) {
 int rc=pthread_mutex_destroy(&f->engine->exact_cache_session_mutex); f->exact_destroy_attempted=true; f->exact_destroy_rc=rc;
    if (rc != 0) {
 f->real_destroy_failures++; fx_error(f);
      return false;
    }
 f->exact_mutex_live=false; f->exact_rescue_destroy++;
  }
  if (f->runtime_mutex_live) {
 int rc=pthread_mutex_destroy(&f->engine->runtime_snapshot_mutex); f->runtime_destroy_attempted=true; f->runtime_destroy_rc=rc;
    if (rc != 0) {
 f->real_destroy_failures++; fx_error(f);
      return false;
    }
 f->runtime_mutex_live=false; f->runtime_rescue_destroy++;
  }
  if (f->exact_mutex_live || f->runtime_mutex_live) {
 fx_order(f);
    return false;
  }
 free(f->engine); f->physical_free_calls++; f->engine_rescue_free++; f->engine_live=false; f->engine=NULL;
  return true;
}
static bool fx_init(fixture_state *f,uint32_t limit) {
  if (f == NULL || (limit != 1u && limit != 2u)) {
    return false;
  }
 memset(f,0,sizeof(*f)); f->runtime_init_rc=-999; f->exact_init_rc=-999; f->runtime_destroy_rc=-999; f->exact_destroy_rc=-999; f->configured_limit=limit; g_fixture=f;
 f->setup_profile=!g_expert_profile.active; f->setup_pool=!g_pool.initialized; f->setup_lock=g_ds4_lock_fd==-1;
  if (!f->setup_profile || !f->setup_pool || !f->setup_lock) {
 fx_error(f);
    return false;
  }
 ds4_engine*e=(ds4_engine*)calloc(1,sizeof(*e));
  if (e == NULL) {
 fx_error(f);
    return false;
  }
 f->engine=e; f->engine_addr=(uintptr_t)e; f->engine_bytes=sizeof(*e); f->engine_acquired=true; f->engine_live=true; e->model.fd=-1; e->mtp_model.fd=-1;
 f->setup_engine=fx_engine_empty(e);
  if (!f->setup_engine) {
 fx_error(f);
    return false;
  }
 e->backend=DS4_BACKEND_CUDA; e->ssd_streaming_cache_bytes_set=true; e->ssd_streaming_cache_bytes=1; e->exact_cache_context_limit=FX_CTX; e->exact_cache_session_limit=limit;
 e->laguna_compact_runtime=true; e->laguna_allocation_plan.context_tokens=FX_CTX; e->test_session_create_no_alloc=true; f->setup_tracker=!e->laguna_runtime_tracker_ready;
 f->runtime_mutex_addr=(uintptr_t)&e->runtime_snapshot_mutex; f->runtime_init_attempted=true; int rc=pthread_mutex_init(&e->runtime_snapshot_mutex,NULL); f->runtime_init_rc=rc;
  if (rc != 0) {
 fx_error(f);
    return false;
  }
 e->runtime_snapshot_mutex_initialized=true; f->runtime_initialized=true; f->runtime_mutex_live=true; f->exact_mutex_addr=(uintptr_t)&e->exact_cache_session_mutex;
 f->exact_init_attempted=true; rc=pthread_mutex_init(&e->exact_cache_session_mutex,NULL); f->exact_init_rc=rc;
  if (rc != 0) {
 fx_error(f);
    return false;
  }
 e->exact_cache_session_mutex_initialized=true; f->exact_initialized=true; f->exact_mutex_live=true; f->setup_ok=f->setup_profile&&f->setup_pool&&f->setup_lock&&
 f->setup_engine&&f->setup_tracker;
  return f->setup_ok;
}
static bool fx_create(fixture_state *f,size_t i) {
  if (f == NULL || i >= FX_NS || !f->engine_live ||
    f->create_attempted[i]) {
    if (f != NULL) {
 fx_order(f);
    }
    return false;
  }
 f->create_attempted[i]=true; f->owner[i]=NULL; int rc=ds4_session_create(&f->owner[i],f->engine,FX_CTX); f->create_rc[i]=rc;
  if (rc != 0) {
    return false;
  }
  if (!f->session_live[i] || f->session[i] == NULL ||
    f->owner[i] != f->session[i]) {
 fx_order(f);
    return false;
  }
 ds4_session*s=f->session[i]; f->create_success[i]=true; f->test_no_alloc[i]=s->test_no_alloc; f->payload_empty[i]=fx_payload_empty(s); f->borrow_avail[i]=true;
 f->borrow_addr[i]=(uintptr_t)s->engine; f->reserved_avail[i]=true; f->reserved[i]=s->exact_cache_session_reserved;
  if (f->engine_live && f->engine != NULL) {
 f->active_after_create_avail[i]=true; f->active_after_create[i]=f->engine->exact_cache_active_sessions;
  }
  if (s->engine != f->engine || !f->test_no_alloc[i] ||
    !f->payload_empty[i]) {
 fx_order(f);
  }
  return true;
}
static void fx_rel_after(fixture_state *f,size_t i,struct fx_rel *r) {
 r->ae_avail=true; r->ae=f->engine_live; r->as_avail=true; r->as=f->session_live[i];
  if (f->session_live[i] && f->session[i] != NULL) {
 r->ara=true; r->ara_v=f->session[i]->exact_cache_session_reserved; r->aom_avail=true; r->aom=f->owner[i]==f->session[i];
  }
  if (f->engine_live && f->engine != NULL) {
 r->aaa=true; r->aaa_v=f->engine->exact_cache_active_sessions;
  }
}
static bool fx_release(fixture_state *f,size_t i,struct fx_rel *r) {
  if (f == NULL || i >= FX_NS || r == NULL) {
    return false;
  }
 memset(r,0,sizeof(*r)); r->call=f->engine_live&&f->engine!=NULL&&f->session_live[i]&& f->session[i]!=NULL&&f->owner[i]==f->session[i];
  if (!r->call) {
    return false;
  }
 r->attempted=true; r->be=true; r->bs=true; r->baa=true; r->ba=f->engine->exact_cache_active_sessions; r->bra=true; r->bra_v=f->session[i]->exact_cache_session_reserved;
 r->rc=ds4_session_free_checked(&f->owner[i]); r->rc_avail=true; f->checked_releases++; fx_rel_after(f,i,r); bool status_valid=r->rc==0||r->rc==1;
 bool consumed=!f->session_live[i];
  if (!status_valid || (r->rc == 1 && !consumed) ||
    (r->rc == 0 && consumed)) {
 r->semantic_mismatch=true; f->semantic_mismatches++;
  }
  return true;
}
static bool fx_release_consumed(const fixture_state *f,size_t i,
                const struct fx_rel *r) {
  return f != NULL && i < FX_NS && r != NULL && r->rc_avail && r->rc == 1 &&
!f->session_live[i];
}
static bool fx_arm_fault(fixture_state *f) {
  if (f == NULL || !f->engine_live || f->engine == NULL ||
!f->exact_mutex_live||!f->session_live[0]||f->fault_arms!=0||
    f->exact_fault_armed) {
    if (f != NULL) {
 fx_order(f);
    }
    return false;
  }
 f->exact_fault_armed=true; f->fault_arms++;
  return true;
}
static void fx_close(fixture_state *f,size_t i) {
  if (f == NULL || i >= FX_NC) {
 return;
  }
  struct fx_close *c = &f->close[i];
  if (c->recorded) {
 fx_order(f); return;
  }
 memset(c,0,sizeof(*c)); c->recorded=true; c->bn_avail=true; c->bn=fx_live_sessions(f); c->be_avail=true; c->be=f->engine_live; c->bra=true; c->bra_v=fx_reserved_count(f);
  c->brm=f->runtime_mutex_live; c->bem=f->exact_mutex_live;
  if (!f->engine_live || f->engine == NULL) {
    c->skipped=true; c->ae_avail=true; c->ae=f->engine_live;
    c->an_avail=true; c->an=fx_live_sessions(f);
    c->ara=true; c->ara_v=fx_reserved_count(f);
    c->arm=f->runtime_mutex_live; c->are=f->exact_mutex_live;
    return;
  }
 c->call=true; c->called=true; c->baa=true; c->ba=f->engine->exact_cache_active_sessions; c->brm=f->runtime_mutex_live; c->bem=f->exact_mutex_live; c->source_probe=c->bn!=0;
  if (c->source_probe) {
 f->product_source_probe_count++;
  }
 f->close_in_progress=true; f->close_source_probe=c->source_probe; f->close_source_free_marker=false; f->close_before_sessions=c->bn; ds4_engine_close(f->engine);
 f->close_in_progress=false; c->source_free_marker=f->close_source_free_marker; f->close_source_probe=false; f->close_source_free_marker=false; c->ae_avail=true;
 c->ae=f->engine_live; c->ae_storage_avail=f->engine_live&&f->engine!=NULL; c->an_avail=true; c->an=fx_live_sessions(f); c->arm=f->runtime_mutex_live; c->are=f->exact_mutex_live;
 c->ara=true; c->ara_v=fx_reserved_count(f);
  if (c->ae_storage_avail) {
 c->aaa=true; c->ae_v=f->engine->exact_cache_active_sessions;
  }
}
static void fx_first_snapshot(fixture_state *f) {
  if (f == NULL || f->first_snapshot.saved ||
    !f->first_release.rc_avail) {
    if (f != NULL) {
 fx_order(f);
    }
 return;
  }
  struct fx_snap *s = &f->first_snapshot;
 memset(s,0,sizeof(*s)); s->saved=true; s->eng=f->engine_live; s->ses=f->session_live[0]; s->release_rc=f->first_release.rc; s->arms=f->fault_arms;
 s->selection=f->fault_selections; s->synthetic=f->fault_synthetic; s->real_successes=f->fault_real_successes; s->fault_real_rc=f->fault_real_rc;
 s->fault_return_rc=f->fault_return_rc; s->exact_locks=f->exact_locks; s->exact_unlocks=f->exact_unlocks; s->session_frees=f->session_source_free[0];
 s->engine_frees=f->engine_source_free; s->rm=f->runtime_mutex_live; s->em=f->exact_mutex_live;
  if (f->session_live[0] && f->session[0] != NULL) {
 s->rr=true; s->reserved=f->session[0]->exact_cache_session_reserved;
  }
  if (f->engine_live && f->engine != NULL) {
 s->aa=true; s->active=f->engine->exact_cache_active_sessions;
  }
 memcpy(&f->first_snapshot_copy,s,sizeof(*s)); f->first_snapshot_copy_valid=true;
}
static void fx_post_close(fixture_state *f,bool attempted) {
  if (f == NULL) {
 return;
  }
 f->post_close_saved=attempted;
  if (!attempted) {
 return;
  }
 f->post_close_eng_avail=true; f->post_close_eng=f->engine_live; f->post_close_eng_storage_avail=f->engine_live&&f->engine!=NULL; f->post_close_ses=f->session_live[0];
 f->post_close_rm=f->runtime_mutex_live; f->post_close_em=f->exact_mutex_live;
  if (f->session_live[0] && f->session[0] != NULL) {
 f->post_close_rr=true; f->post_close_reserved= f->session[0]->exact_cache_session_reserved;
  }
  if (f->post_close_eng_storage_avail) {
 f->post_close_aa=true; f->post_close_active=f->engine->exact_cache_active_sessions;
  }
}
static void fx_post_retry(fixture_state *f) {
  if (f == NULL) {
 return;
  }
 f->post_retry_saved=true; f->post_retry_eng_avail=true; f->post_retry_eng=f->engine_live; f->post_retry_eng_storage_avail=f->engine_live&&f->engine!=NULL;
 f->post_retry_ses=f->session_live[0];
  if (f->session_live[0] && f->session[0] != NULL) {
 f->post_retry_rr=true; f->post_retry_reserved= f->session[0]->exact_cache_session_reserved;
  }
  if (f->post_retry_eng_storage_avail) {
 f->post_retry_aa=true; f->post_retry_active=f->engine->exact_cache_active_sessions;
  }
}
static void fx_pre_rescue(fixture_state *f) {
  if (f == NULL || f->pre_rescue_saved) {
 return;
  }
 f->pre_rescue_saved=true; f->pre_rescue_eng=f->engine_live; f->pre_rescue_sessions=fx_live_sessions(f); f->pre_rescue_runtime_mutex_live=f->runtime_mutex_live;
 f->pre_rescue_exact_mutex_live=f->exact_mutex_live; f->pre_rescue_engine_source_free=f->engine_source_free; f->pre_rescue_engine_rescue_free=f->engine_rescue_free;
 f->pre_rescue_runtime_source_destroy=f->runtime_source_destroy; f->pre_rescue_exact_source_destroy=f->exact_source_destroy;
 f->pre_rescue_runtime_rescue_destroy=f->runtime_rescue_destroy; f->pre_rescue_exact_rescue_destroy=f->exact_rescue_destroy;
  for (size_t i = 0; i < FX_NS; i++) {
 f->pre_rescue_session_live[i]=f->session_live[i]; f->pre_rescue_session_source_free[i]=f->session_source_free[i]; f->pre_rescue_session_rescue_free[i]=f->session_rescue_free[i];
  }
  if (f->first_snapshot_copy_valid) {
 f->first_snapshot_equal_before_available=true; f->first_snapshot_equal_before= fx_snap_equal(&f->first_snapshot,&f->first_snapshot_copy);
  }
}
static void fx_finalize_rescue_proof(fixture_state *f) {
  if (f == NULL || !f->pre_rescue_saved) {
 return;
  }
 f->rescue_proof_available=true; bool ok=true; f->rescue_engine_expected=f->pre_rescue_eng?1u:0u; f->rescue_runtime_expected= f->pre_rescue_runtime_mutex_live?1u:0u;
 f->rescue_exact_expected=f->pre_rescue_exact_mutex_live?1u:0u;
  if (f->engine_rescue_free < f->pre_rescue_engine_rescue_free ||
 f->runtime_rescue_destroy<f->pre_rescue_runtime_rescue_destroy||
    f->exact_rescue_destroy < f->pre_rescue_exact_rescue_destroy) {
 ok=false;
  } else {
 f->rescue_engine_delta= f->engine_rescue_free-f->pre_rescue_engine_rescue_free; f->rescue_runtime_delta= f->runtime_rescue_destroy-f->pre_rescue_runtime_rescue_destroy;
 f->rescue_exact_delta= f->exact_rescue_destroy-f->pre_rescue_exact_rescue_destroy;
  }
  for (size_t i = 0; i < FX_NS; i++) {
 f->rescue_session_expected[i]= f->pre_rescue_session_live[i]?1u:0u;
    if (f->session_rescue_free[i] < f->pre_rescue_session_rescue_free[i]) {
 ok=false;
    } else {
 f->rescue_session_delta[i]= f->session_rescue_free[i]-f->pre_rescue_session_rescue_free[i];
    }
  }
 f->rescue_source_effects_unchanged= f->engine_source_free==f->pre_rescue_engine_source_free&& f->runtime_source_destroy==f->pre_rescue_runtime_source_destroy&&
 f->exact_source_destroy==f->pre_rescue_exact_source_destroy;
  for (size_t i = 0; i < FX_NS; i++) {
    if (f->session_source_free[i] !=
      f->pre_rescue_session_source_free[i]) {
 f->rescue_source_effects_unchanged=false;
    }
  }
 f->rescue_final_residual_zero=!f->engine_live&& !f->runtime_mutex_live&&!f->exact_mutex_live&& fx_live_sessions(f)==0;
  if (f->rescue_engine_delta != f->rescue_engine_expected ||
 f->rescue_runtime_delta!=f->rescue_runtime_expected|| f->rescue_exact_delta!=f->rescue_exact_expected|| !f->rescue_source_effects_unchanged||
    !f->rescue_final_residual_zero) {
 ok=false;
  }
  for (size_t i = 0; i < FX_NS; i++) {
    if (f->rescue_session_delta[i] != f->rescue_session_expected[i]) {
 ok=false;
    }
  }
  if (f->first_snapshot_copy_valid) {
 f->first_snapshot_equal_after_available=true; f->first_snapshot_equal_after= fx_snap_equal(&f->first_snapshot,&f->first_snapshot_copy);
    if (!f->first_snapshot_equal_before ||
      !f->first_snapshot_equal_after) {
 ok=false;
    }
  }
 f->rescue_proof_ok=ok;
}
static bool fx_cleanup(fixture_state *f) {
  if (f == NULL) {
    return false;
  }
 bool ok=true;
  for (size_t i = 0; i < FX_NS; i++) {
    if (f->session_live[i]) {
      if (!fx_rescue_session(f,i)) {
 ok=false;
      }
    }
  }
  if (f->engine_live) {
    if (!fx_rescue_engine(f)) {
 ok=false;
    }
  }
  return ok && !f->engine_live && !f->runtime_mutex_live &&
!f->exact_mutex_live&&fx_live_sessions(f)==0;
}
static bool fx_clean(const fixture_state *f) {
  if (f == NULL || !f->setup_ok || !f->engine_acquired || f->engine_live ||
 f->runtime_mutex_live||f->exact_mutex_live|| fx_live_sessions(f)!=0||f->unknown_free||f->unknown_mutex|| f->unexpected_alloc||f->real_lock_failures||
 f->real_unlock_failures||f->real_destroy_failures||
    f->order_errors || f->sticky || !f->rescue_proof_ok) {
    return false;
  }
 unsigned n=0;
  for (size_t i = 0; i < FX_NS; i++) {
 n+=f->acquired[i]?1u:0u;
  }
  if (f->engine_source_free + f->engine_rescue_free != 1u ||
 f->core_calloc_calls!=n||f->core_calloc_bytes!=f->session_bytes_total|| f->physical_free_calls!=1u+n|| f->runtime_locks!=f->runtime_unlocks||
    f->exact_locks != f->exact_unlocks) {
    return false;
  }
  for (size_t i = 0; i < FX_NS; i++) {
    if (f->session_source_free[i] + f->session_rescue_free[i] !=
      (f->acquired[i] ? 1u : 0u)) {
      return false;
    }
  }
  return f->fault_arms == f->fault_selections &&
 f->fault_selections==f->fault_synthetic&& f->fault_selections==f->fault_real_successes&& !f->exact_fault_armed;
}
static bool fx_emit(struct fx_output *o,const char *key,const char *value) {
  if (o == NULL || key == NULL || value == NULL || o->overflow) {
    return false;
  }
 size_t kl=strlen(key);
  if (kl == 0 || kl >= sizeof(o->keys[0]) || o->n >= FX_NM) {
 o->overflow=true;
    return false;
  }
  for (size_t i = 0; i < o->n; i++) {
    if (strcmp(o->keys[i],key) == 0) {
 o->overflow=true;
      return false;
    }
  }
 size_t rem=sizeof(o->text)-o->len; int n=snprintf(o->text+o->len,rem,"%s=%s\n",key,value);
  if (n < 0 || (size_t)n >= rem) {
 o->overflow=true;
    return false;
  }
 memcpy(o->keys[o->n],key,kl+1); o->n++; o->len+=(size_t)n;
  return true;
}
static void fx_bool(struct fx_output *o,const char *k,bool v) {
 fx_emit(o,k,v?"1":"0");
}
static void fx_int(struct fx_output *o,const char *k,int v) {
 char b[32]; snprintf(b,sizeof(b),"%d",v); fx_emit(o,k,b);
}
static void fx_uint(struct fx_output *o,const char *k,uint64_t v) {
 char b[48]; snprintf(b,sizeof(b),"%" PRIu64,v); fx_emit(o,k,b);
}
static void fx_addr(struct fx_output *o,const char *k,uintptr_t v) {
 char b[48]; snprintf(b,sizeof(b),"0x%" PRIxPTR,v); fx_emit(o,k,b);
}
static void fx_avail_b(struct fx_output *o,const char *k,bool a,bool v) {
 char b[96]; snprintf(b,sizeof(b),"%s_available",k); fx_bool(o,b,a); snprintf(b,sizeof(b),"%s_value",k); fx_bool(o,b,a&&v);
}
static void fx_avail_i(struct fx_output *o,const char *k,bool a,int v) {
 char b[96]; snprintf(b,sizeof(b),"%s_available",k); fx_bool(o,b,a); snprintf(b,sizeof(b),"%s_value",k); fx_int(o,b,a?v:0);
}
static void fx_avail_u(struct fx_output *o,const char *k,bool a,
           uint64_t v) {
 char b[96]; snprintf(b,sizeof(b),"%s_available",k); fx_bool(o,b,a); snprintf(b,sizeof(b),"%s_value",k); fx_uint(o,b,a?v:0);
}
static void fx_avail_a(struct fx_output *o,const char *k,bool a,
           uintptr_t v) {
 char b[96]; snprintf(b,sizeof(b),"%s_available",k); fx_bool(o,b,a); snprintf(b,sizeof(b),"%s_value",k); fx_addr(o,b,a?v:0);
}
static void fx_emit_rel(struct fx_output *o,const char *p,
            const struct fx_rel *r) {
 char k[96];
#define RB(s,v) do { snprintf(k,sizeof(k),"%s_%s",p,s); fx_bool(o,k,(v)); } while (0)
#define RU(s,a,v) do { snprintf(k,sizeof(k),"%s_%s",p,s); fx_avail_u(o,k,(a),(v)); } while (0)
#define RI(s,a,v) do { snprintf(k,sizeof(k),"%s_%s",p,s); fx_avail_i(o,k,(a),(v)); } while (0)
 RB("call_available",r->call); RB("attempted",r->attempted); RI("rc",r->rc_avail,r->rc); RB("before_engine_live",r->be); RB("before_session_live",r->bs);
 RU("before_active",r->baa,r->ba); RU("before_reserved",r->bra,r->bra_v); RB("after_engine_live_available",r->ae_avail); RB("after_engine_live",r->ae);
 RB("after_session_live",r->as); RB("after_session_live_available",r->as_avail); RU("after_active",r->aaa,r->aaa_v); RU("after_reserved",r->ara,r->ara_v);
 RB("after_owner_match",r->aom); RB("after_owner_match_available",r->aom_avail); RB("semantic_mismatch",r->semantic_mismatch);
#undef RB
#undef RU
#undef RI
}
static void fx_emit_close(struct fx_output *o,size_t i,
             const struct fx_close *c) {
 char p[32],k[96]; snprintf(p,sizeof(p),"close%zu",i+1u);
#define CB(s,v) do { snprintf(k,sizeof(k),"%s_%s",p,s); fx_bool(o,k,(v)); } while (0)
#define CU(s,a,v) do { snprintf(k,sizeof(k),"%s_%s",p,s); fx_avail_u(o,k,(a),(v)); } while (0)
 CB("recorded",c->recorded); CB("call_available",c->call); CB("called",c->called); CB("skipped",c->skipped); CU("before_sessions_live",c->bn_avail,c->bn);
 CB("before_engine_live_available",c->be_avail); CB("before_engine_live",c->be); CU("before_active",c->baa,c->ba); CU("before_reserved",c->bra,c->bra_v);
 CB("before_runtime_mutex_live",c->brm); CB("before_exact_mutex_live",c->bem); CB("source_probe",c->source_probe); CB("source_free_marker",c->source_free_marker);
 CB("after_engine_live_available",c->ae_avail); CB("after_engine_storage_available",c->ae_storage_avail); CB("after_engine_live",c->ae); CU("after_active",c->aaa,c->ae_v);
 CU("after_reserved",c->ara,c->ara_v); CU("after_sessions_live",c->an_avail,c->an); CB("after_runtime_mutex_live",c->arm); CB("after_exact_mutex_live",c->are);
#undef CB
#undef CU
}
static void fx_emit_session(struct fx_output *o,const fixture_state *f,
              size_t i) {
 char k[96]; snprintf(k,sizeof(k),"session%zu_create_attempted",i); fx_bool(o,k,f->create_attempted[i]); snprintf(k,sizeof(k),"session%zu_create_success",i);
 fx_bool(o,k,f->create_success[i]); snprintf(k,sizeof(k),"session%zu_create_rc",i); fx_avail_i(o,k,f->create_attempted[i],f->create_rc[i]);
 snprintf(k,sizeof(k),"session%zu_address",i); fx_avail_a(o,k,f->acquired[i],f->session_addr[i]); snprintf(k,sizeof(k),"session%zu_bytes",i);
 fx_avail_u(o,k,f->acquired[i],f->session_bytes[i]); snprintf(k,sizeof(k),"session%zu_engine_borrow",i); fx_avail_a(o,k,f->borrow_avail[i],f->borrow_addr[i]);
 snprintf(k,sizeof(k),"session%zu_test_no_alloc",i); fx_bool(o,k,f->test_no_alloc[i]); snprintf(k,sizeof(k),"session%zu_backend_payload_empty",i);
 fx_bool(o,k,f->payload_empty[i]); snprintf(k,sizeof(k),"session%zu_reserved",i); fx_avail_u(o,k,f->reserved_avail[i],f->reserved[i]);
 snprintf(k,sizeof(k),"session%zu_active_after_create",i); fx_avail_u(o,k,f->active_after_create_avail[i],f->active_after_create[i]);
 snprintf(k,sizeof(k),"session%zu_live_final",i); fx_bool(o,k,f->session_live[i]); snprintf(k,sizeof(k),"session%zu_source_free_calls",i);
 fx_uint(o,k,f->session_source_free[i]); snprintf(k,sizeof(k),"session%zu_rescue_free_calls",i); fx_uint(o,k,f->session_rescue_free[i]);
 snprintf(k,sizeof(k),"session%zu_rescue_delta",i); fx_avail_u(o,k,f->rescue_proof_available,f->rescue_session_delta[i]);
 snprintf(k,sizeof(k),"session%zu_rescue_expected",i); fx_uint(o,k,f->rescue_session_expected[i]); snprintf(k,sizeof(k),"session%zu_pre_rescue_live",i);
 fx_bool(o,k,f->pre_rescue_saved&&f->pre_rescue_session_live[i]); snprintf(k,sizeof(k),"session%zu_pre_rescue_source_free_calls",i);
 fx_uint(o,k,f->pre_rescue_saved?f->pre_rescue_session_source_free[i]:0); snprintf(k,sizeof(k),"session%zu_pre_rescue_rescue_free_calls",i);
 fx_uint(o,k,f->pre_rescue_saved?f->pre_rescue_session_rescue_free[i]:0);
}
static void fx_emit_all(struct fx_output *o,const char *name,
 const fixture_state*f,bool done,bool clean,
            int rc) {
 fx_emit(o,"case",name); fx_bool(o,"setup_ok",f->setup_ok); fx_bool(o,"native_cleanup_complete",done); fx_bool(o,"native_observation_clean",clean);
 fx_int(o,"native_return_code",rc); fx_bool(o,"void_engine_close_status_available",false); fx_int(o,"void_engine_close_status",0);
 fx_bool(o,"product_premature_engine_free",f->product_premature_free); fx_uint(o,"product_source_probe_count",f->product_source_probe_count);
 fx_uint(o,"product_source_free_marker_count",f->product_source_marker_count); fx_uint(o,"semantic_product_mismatch_count",f->semantic_mismatches);
 fx_bool(o,"setup_profile_inactive",f->setup_profile); fx_bool(o,"setup_thread_pool_inactive",f->setup_pool); fx_bool(o,"setup_instance_fd_clear",f->setup_lock);
 fx_bool(o,"setup_engine_empty",f->setup_engine); fx_bool(o,"setup_runtime_tracker_unready",f->setup_tracker); fx_uint(o,"configured_context_tokens",FX_CTX);
 fx_uint(o,"configured_session_limit",f->configured_limit); fx_bool(o,"configured_backend_cuda",true); fx_bool(o,"configured_cache_bytes_set",true);
 fx_bool(o,"configured_compact_runtime",true); fx_int(o,"configured_model_fd",-1); fx_int(o,"configured_mtp_model_fd",-1); fx_bool(o,"engine_allocated",f->engine_acquired);
 fx_avail_a(o,"engine_address",f->engine_acquired,f->engine_addr); fx_avail_u(o,"engine_bytes",f->engine_acquired,f->engine_bytes);
 fx_bool(o,"engine_live_final",f->engine_live); fx_uint(o,"engine_source_free_calls",f->engine_source_free); fx_uint(o,"engine_rescue_free_calls",f->engine_rescue_free);
 fx_bool(o,"engine_size_limit_ok",f->engine_bytes<=FX_OWNER); fx_bool(o,"session_size_limit_ok",sizeof(ds4_session)<=FX_OWNER); fx_bool(o,"cumulative_heap_limit_ok",
 f->engine_bytes<=FX_HEAP&& f->session_bytes_total<=FX_HEAP-f->engine_bytes); unsigned ns=0;
  for (size_t i = 0; i < FX_NS; i++) {
 ns+=f->acquired[i]?1u:0u;
  }
 fx_uint(o,"session_alloc_count",ns); fx_uint(o,"checked_release_calls",f->checked_releases); fx_uint(o,"core_calloc_calls",f->core_calloc_calls);
 fx_uint(o,"core_calloc_bytes",f->core_calloc_bytes); fx_uint(o,"session_bytes_total",f->session_bytes_total); fx_uint(o,"physical_free_calls",f->physical_free_calls);
 fx_uint(o,"unknown_free_calls",f->unknown_free); fx_uint(o,"unknown_mutex_calls",f->unknown_mutex); fx_uint(o,"unexpected_allocations",f->unexpected_alloc);
 fx_uint(o,"order_violations",f->order_errors); fx_uint(o,"sticky_errors",f->sticky); fx_avail_a(o,"runtime_mutex_address",f->runtime_init_attempted,f->runtime_mutex_addr);
 fx_avail_a(o,"exact_mutex_address",f->exact_init_attempted,f->exact_mutex_addr); fx_bool(o,"runtime_mutex_initialized",f->runtime_initialized);
 fx_avail_i(o,"runtime_mutex_init_result",f->runtime_init_attempted,f->runtime_init_rc); fx_bool(o,"exact_mutex_initialized",f->exact_initialized);
 fx_avail_i(o,"exact_mutex_init_result",f->exact_init_attempted,f->exact_init_rc); fx_bool(o,"runtime_mutex_live_final",f->runtime_mutex_live);
 fx_bool(o,"exact_mutex_live_final",f->exact_mutex_live); fx_avail_i(o,"runtime_mutex_destroy_result",f->runtime_destroy_attempted,f->runtime_destroy_rc);
 fx_avail_i(o,"exact_mutex_destroy_result",f->exact_destroy_attempted,f->exact_destroy_rc); fx_uint(o,"runtime_source_destroy_calls",f->runtime_source_destroy);
 fx_uint(o,"exact_source_destroy_calls",f->exact_source_destroy); fx_uint(o,"runtime_rescue_destroy_calls",f->runtime_rescue_destroy);
 fx_uint(o,"exact_rescue_destroy_calls",f->exact_rescue_destroy); fx_uint(o,"physical_runtime_locks",f->runtime_locks);
 fx_uint(o,"physical_runtime_unlocks",f->runtime_unlocks); fx_uint(o,"physical_exact_locks",f->exact_locks); fx_uint(o,"physical_exact_unlocks",f->exact_unlocks);
 fx_uint(o,"real_lock_failures",f->real_lock_failures); fx_uint(o,"real_unlock_failures",f->real_unlock_failures);
 fx_uint(o,"real_destroy_failures",f->real_destroy_failures); fx_uint(o,"fault_arm_count",f->fault_arms); fx_bool(o,"fault_armed_final",f->exact_fault_armed);
 fx_uint(o,"fault_selection_count",f->fault_selections); fx_uint(o,"fault_synthetic_refusal_count",f->fault_synthetic);
 fx_uint(o,"fault_selected_real_successes",f->fault_real_successes); fx_avail_i(o,"fault_selected_real_result",f->fault_selections,f->fault_real_rc);
 fx_avail_i(o,"fault_selected_return_result",f->fault_selections,f->fault_return_rc);
  for (size_t i = 0; i < FX_NS; i++) {
 fx_emit_session(o,f,i); char p[32]; snprintf(p,sizeof(p),"release%zu",i); fx_emit_rel(o,p,&f->release[i]);
  }
 fx_emit_rel(o,"first_release",&f->first_release); fx_emit_rel(o,"retry_release",&f->retry_release); const struct fx_snap*s=&f->first_snapshot;
 fx_bool(o,"first_snapshot_saved",s->saved); fx_avail_i(o,"first_snapshot_release_rc",s->saved,s->release_rc);
 fx_uint(o,"first_snapshot_fault_arm_count",s->saved?s->arms:0); fx_uint(o,"first_snapshot_fault_selection_count",s->saved?s->selection:0);
 fx_uint(o,"first_snapshot_fault_synthetic_refusal_count",s->saved?s->synthetic:0); fx_uint(o,"first_snapshot_fault_real_successes",s->saved?s->real_successes:0);
 fx_avail_i(o,"first_snapshot_fault_real_result",s->saved&&s->selection!=0,s->fault_real_rc); fx_avail_i(o,"first_snapshot_fault_return_result",
 s->saved&&s->selection!=0,s->fault_return_rc); fx_bool(o,"first_snapshot_engine_live",s->saved&&s->eng); fx_bool(o,"first_snapshot_session_live",s->saved&&s->ses);
 fx_avail_u(o,"first_snapshot_active",s->saved&&s->aa,s->active); fx_bool(o,"first_snapshot_reserved",s->saved&&s->rr&&s->reserved);
 fx_bool(o,"first_snapshot_reserved_available",s->saved&&s->rr); fx_bool(o,"first_snapshot_runtime_mutex_live",s->saved&&s->rm);
 fx_bool(o,"first_snapshot_exact_mutex_live",s->saved&&s->em); fx_uint(o,"first_snapshot_physical_exact_locks",s->saved?s->exact_locks:0);
 fx_uint(o,"first_snapshot_physical_exact_unlocks",s->saved?s->exact_unlocks:0); fx_uint(o,"first_snapshot_source_session_free_calls",s->saved?s->session_frees:0);
 fx_uint(o,"first_snapshot_source_engine_free_calls",s->saved?s->engine_frees:0); fx_bool(o,"first_snapshot_copy_available",f->first_snapshot_copy_valid);
 fx_uint(o,"first_snapshot_copy_bytes",f->first_snapshot_copy_valid?sizeof(f->first_snapshot_copy):0); fx_avail_b(o,"first_snapshot_equal_before_rescue",
 f->first_snapshot_equal_before_available,f->first_snapshot_equal_before); fx_avail_b(o,"first_snapshot_equal_after_rescue",f->first_snapshot_equal_after_available,
 f->first_snapshot_equal_after); fx_bool(o,"first_snapshot_preserved",f->first_snapshot_equal_after_available&& f->first_snapshot_equal_after);
 fx_bool(o,"post_close_saved",f->post_close_saved); fx_bool(o,"post_close_engine_live_available",f->post_close_saved&&f->post_close_eng_avail);
 fx_bool(o,"post_close_engine_storage_available",f->post_close_saved&&f->post_close_eng_storage_avail);
 fx_bool(o,"post_close_engine_live",f->post_close_saved&&f->post_close_eng); fx_bool(o,"post_close_session_live",f->post_close_saved&&f->post_close_ses);
 fx_avail_u(o,"post_close_active",f->post_close_saved&&f->post_close_aa,f->post_close_active); fx_bool(o,"post_close_reserved_available",
 f->post_close_saved&&f->post_close_rr); fx_bool(o,"post_close_reserved",f->post_close_saved&&f->post_close_rr&&f->post_close_reserved);
 fx_bool(o,"post_close_runtime_mutex_live",f->post_close_saved&&f->post_close_rm); fx_bool(o,"post_close_exact_mutex_live",f->post_close_saved&&f->post_close_em);
 fx_bool(o,"post_retry_saved",f->post_retry_saved); fx_bool(o,"post_retry_engine_live_available",f->post_retry_saved&&f->post_retry_eng_avail);
 fx_bool(o,"post_retry_engine_storage_available",f->post_retry_saved&&f->post_retry_eng_storage_avail);
 fx_bool(o,"post_retry_engine_live",f->post_retry_saved&&f->post_retry_eng); fx_bool(o,"post_retry_session_live",f->post_retry_saved&&f->post_retry_ses);
 fx_avail_u(o,"post_retry_active",f->post_retry_saved&&f->post_retry_aa,f->post_retry_active); fx_bool(o,"post_retry_reserved_available",
 f->post_retry_saved&&f->post_retry_rr); fx_bool(o,"post_retry_reserved",f->post_retry_saved&&f->post_retry_rr&&f->post_retry_reserved);
 fx_bool(o,"pre_rescue_saved",f->pre_rescue_saved); fx_bool(o,"pre_rescue_engine_live",f->pre_rescue_saved&&f->pre_rescue_eng); fx_uint(o,"pre_rescue_sessions_live",
 f->pre_rescue_saved?f->pre_rescue_sessions:0); fx_bool(o,"pre_rescue_runtime_mutex_live",f->pre_rescue_saved&&f->pre_rescue_runtime_mutex_live);
 fx_bool(o,"pre_rescue_exact_mutex_live",f->pre_rescue_saved&&f->pre_rescue_exact_mutex_live); fx_uint(o,"pre_rescue_engine_source_free_calls",
 f->pre_rescue_saved?f->pre_rescue_engine_source_free:0); fx_uint(o,"pre_rescue_engine_rescue_free_calls",f->pre_rescue_saved?f->pre_rescue_engine_rescue_free:0);
 fx_uint(o,"pre_rescue_runtime_source_destroy_calls",f->pre_rescue_saved?f->pre_rescue_runtime_source_destroy:0); fx_uint(o,"pre_rescue_exact_source_destroy_calls",
 f->pre_rescue_saved?f->pre_rescue_exact_source_destroy:0); fx_uint(o,"pre_rescue_runtime_rescue_destroy_calls",f->pre_rescue_saved?f->pre_rescue_runtime_rescue_destroy:0);
 fx_uint(o,"pre_rescue_exact_rescue_destroy_calls",f->pre_rescue_saved?f->pre_rescue_exact_rescue_destroy:0); fx_bool(o,"rescue_proof_available",f->rescue_proof_available);
 fx_bool(o,"rescue_proof_ok",f->rescue_proof_ok); fx_bool(o,"rescue_final_residual_zero",f->rescue_final_residual_zero); fx_bool(o,"rescue_source_effects_unchanged",
 f->rescue_source_effects_unchanged); fx_uint(o,"rescue_engine_delta",f->rescue_engine_delta); fx_uint(o,"rescue_engine_expected",f->rescue_engine_expected);
 fx_uint(o,"rescue_runtime_delta",f->rescue_runtime_delta); fx_uint(o,"rescue_runtime_expected",f->rescue_runtime_expected);
 fx_uint(o,"rescue_exact_delta",f->rescue_exact_delta); fx_uint(o,"rescue_exact_expected",f->rescue_exact_expected);
  for (size_t i = 0; i < FX_NC; i++) {
 fx_emit_close(o,i,&f->close[i]);
  }
}
static int fx_finish(const char *name,fixture_state *f,
          struct fx_output *o) {
  if (f == NULL || o == NULL) {
    return 125;
  }
 fx_pre_rescue(f); bool done=fx_cleanup(f); fx_finalize_rescue_proof(f); bool clean=done&&fx_clean(f); int rc=clean?0:125; fx_emit_all(o,name,f,done,clean,rc);
  if (o->overflow || fwrite(o->text,1,o->len,stdout) != o->len ||
    fflush(stdout) != 0) {
    return 125;
  }
  return rc;
}
static int run_empty(void) {
 fixture_state f;
  struct fx_output o = {0};
  if (!fx_init(&f,1)) {
    return fx_finish("empty",&f,&o);
  }
 fx_close(&f,0);
  return fx_finish("empty",&f,&o);
}
static int run_live_one(void) {
 fixture_state f;
  struct fx_output o = {0};
  if (!fx_init(&f,1)) {
    return fx_finish("live-one",&f,&o);
  }
 fx_create(&f,0); fx_close(&f,0);
  if (f.engine_live && f.session_live[0]) {
 fx_release(&f,0,&f.release[0]);
    if (fx_release_consumed(&f,0,&f.release[0]) && f.engine_live) {
 fx_close(&f,1);
    }
  }
  return fx_finish("live-one",&f,&o);
}
static int run_live_two(void) {
 fixture_state f;
  struct fx_output o = {0};
  if (!fx_init(&f,2)) {
    return fx_finish("live-two",&f,&o);
  }
 fx_create(&f,0); fx_create(&f,1); fx_close(&f,0);
  if (f.engine_live && f.session_live[0]) {
 fx_release(&f,0,&f.release[0]);
  }
  if (fx_release_consumed(&f,0,&f.release[0]) && f.engine_live &&
    f.session_live[1]) {
 fx_close(&f,1);
    if (f.engine_live && f.session_live[1]) {
 fx_release(&f,1,&f.release[1]);
    }
    if (fx_release_consumed(&f,1,&f.release[1]) && f.engine_live) {
 fx_close(&f,2);
    }
  }
  return fx_finish("live-two",&f,&o);
}
static int run_retained(void) {
 fixture_state f;
  struct fx_output o = {0};
  if (!fx_init(&f,1)) {
    return fx_finish("retained-after-unlock",&f,&o);
  }
 fx_create(&f,0); fx_arm_fault(&f); fx_release(&f,0,&f.first_release); fx_first_snapshot(&f); bool first_consumed=fx_release_consumed(&f,0,&f.first_release);
  if (!first_consumed && f.first_release.rc_avail &&
      f.first_release.rc == 0 && !f.first_release.semantic_mismatch &&
      f.session_live[0] && f.engine_live) {
 fx_close(&f,0); fx_post_close(&f,true);
    if (f.engine_live && f.session_live[0]) {
 fx_release(&f,0,&f.retry_release); fx_post_retry(&f);
      if (fx_release_consumed(&f,0,&f.retry_release) && f.engine_live) {
 fx_close(&f,1);
      }
    } else {
 fx_post_retry(&f);
    }
  } else {
 fx_post_close(&f,false); fx_post_retry(&f);
  }
  return fx_finish("retained-after-unlock",&f,&o);
}
int main(int argc,char **argv) {
 alarm(15);
  const struct rlimit lim = {0,0};
  if (setrlimit(RLIMIT_CORE,&lim) != 0 || argc != 2) {
    return 125;
  }
  if (strcmp(argv[1],"empty") == 0) {
    return run_empty();
  }
  if (strcmp(argv[1],"live-one") == 0) {
    return run_live_one();
  }
  if (strcmp(argv[1],"live-two") == 0) {
    return run_live_two();
  }
  if (strcmp(argv[1],"retained-after-unlock") == 0) {
    return run_retained();
  }
  return 125;
}
