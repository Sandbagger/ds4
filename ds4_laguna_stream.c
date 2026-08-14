#include "ds4_laguna_stream.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const ds4_laguna_tensor_desc *tensor;
} tensor_ref;

static bool set_error(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen != 0) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return false;
}

static bool add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool is_power_of_two(uint64_t value) {
    return value != 0 && (value & (value - 1u)) == 0;
}

static bool align_u64(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (!is_power_of_two(alignment)) return false;
    const uint64_t mask = alignment - 1u;
    if (value > UINT64_MAX - mask) return false;
    *out = (value + mask) & ~mask;
    return true;
}

static int compare_tensor_ref_source(const void *lhs, const void *rhs) {
    const tensor_ref *a = lhs;
    const tensor_ref *b = rhs;
    if (a->tensor->source_offset < b->tensor->source_offset) return -1;
    if (a->tensor->source_offset > b->tensor->source_offset) return 1;
    if (a->tensor->stable_index < b->tensor->stable_index) return -1;
    if (a->tensor->stable_index > b->tensor->stable_index) return 1;
    return 0;
}

static bool append_source_range(ds4_laguna_ledger *ledger,
                                size_t source_capacity,
                                ds4_laguna_source_range_kind kind,
                                uint64_t offset,
                                uint64_t bytes,
                                char *err,
                                size_t errlen) {
    if (bytes == 0) return true;
    if (ledger->source_range_count >= source_capacity) {
        return set_error(err, errlen, "source range allocation overflow");
    }
    ds4_laguna_source_range *range =
        &ledger->source_ranges[ledger->source_range_count++];
    range->kind = kind;
    range->source_offset = offset;
    range->source_bytes = bytes;
    if (!add_u64(ledger->non_tensor_source_bytes, bytes,
                 &ledger->non_tensor_source_bytes)) {
        return set_error(err, errlen, "non-tensor source byte overflow");
    }
    return true;
}

static bool tensor_names_equal(const ds4_laguna_tensor_desc *a,
                               const ds4_laguna_tensor_desc *b) {
    return a->name_len == b->name_len &&
           memcmp(a->name, b->name, a->name_len) == 0;
}

static bool tensor_shape_bytes(const ds4_laguna_tensor_desc *tensor,
                               uint64_t *elements,
                               uint64_t *bytes,
                               char *err,
                               size_t errlen) {
    if (tensor->ndim == 0 || tensor->ndim > DS4_LAGUNA_MAX_DIMS) {
        return set_error(err, errlen,
                         "tensor %llu has invalid dimension count %u",
                         (unsigned long long)tensor->stable_index,
                         tensor->ndim);
    }
    if (tensor->block_elems == 0 || tensor->block_bytes == 0) {
        return set_error(err, errlen,
                         "tensor %llu has invalid block geometry",
                         (unsigned long long)tensor->stable_index);
    }

    uint64_t count = 1;
    for (uint32_t d = 0; d < tensor->ndim; d++) {
        if (tensor->dim[d] == 0) {
            return set_error(err, errlen,
                             "tensor %llu has a zero dimension",
                             (unsigned long long)tensor->stable_index);
        }
        if (!mul_u64(count, tensor->dim[d], &count)) {
            return set_error(err, errlen,
                             "tensor %llu dimension multiplication overflow",
                             (unsigned long long)tensor->stable_index);
        }
    }

    uint64_t blocks = count / tensor->block_elems;
    if (count % tensor->block_elems != 0) {
        if (!add_u64(blocks, 1u, &blocks)) {
            return set_error(err, errlen,
                             "tensor %llu block count overflow",
                             (unsigned long long)tensor->stable_index);
        }
    }
    uint64_t packed = 0;
    if (!mul_u64(blocks, tensor->block_bytes, &packed)) {
        return set_error(err, errlen,
                         "tensor %llu packed size overflow",
                         (unsigned long long)tensor->stable_index);
    }
    *elements = count;
    *bytes = packed;
    return true;
}

void ds4_laguna_ledger_free(ds4_laguna_ledger *ledger) {
    if (!ledger) return;
    free(ledger->tensor_ranges);
    free(ledger->source_ranges);
    free(ledger->expert_entries);
    memset(ledger, 0, sizeof(*ledger));
}

static int cache_key_compare(ds4_laguna_expert_key a,
                             ds4_laguna_expert_key b) {
    if (a.layer_id < b.layer_id) return -1;
    if (a.layer_id > b.layer_id) return 1;
    if (a.expert_id < b.expert_id) return -1;
    if (a.expert_id > b.expert_id) return 1;
    return 0;
}

static bool cache_key_equal(ds4_laguna_expert_key a,
                            ds4_laguna_expert_key b) {
    return cache_key_compare(a, b) == 0;
}

static ds4_laguna_expert_key cache_entry_key(
        const ds4_laguna_expert_entry *entry) {
    const ds4_laguna_expert_key key = {
        .layer_id = entry->layer,
        .expert_id = entry->expert,
    };
    return key;
}

static ds4_laguna_expert_key cache_slot_key(
        const ds4_laguna_cache_slot *slot) {
    const ds4_laguna_expert_key key = {
        .layer_id = slot->layer,
        .expert_id = slot->expert,
    };
    return key;
}

static bool cache_entry_index(const ds4_laguna_cache_policy *policy,
                              ds4_laguna_expert_key key,
                              size_t *entry_index) {
    size_t lo = 0;
    size_t hi = policy->entry_count;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2u;
        const int order = cache_key_compare(
            cache_entry_key(&policy->entries[mid]), key);
        if (order < 0) lo = mid + 1u;
        else hi = mid;
    }
    if (lo >= policy->entry_count ||
        !cache_key_equal(cache_entry_key(&policy->entries[lo]), key)) {
        return false;
    }
    if (entry_index) *entry_index = lo;
    return true;
}

static bool cache_policy_core_valid(
        const ds4_laguna_cache_policy *policy) {
    if (!policy || !policy->entries || policy->entry_count == 0 ||
        !policy->slots || policy->slot_count == 0 ||
        policy->slot_count > UINT32_MAX ||
        !policy->route_hotness || !policy->entry_to_slot ||
        policy->max_selected_per_token == 0 ||
        policy->max_selected_per_token > policy->slot_count) {
        return false;
    }
    return true;
}

static bool cache_entries_ordered(
        const ds4_laguna_cache_policy *policy) {
    if (!cache_policy_core_valid(policy)) return false;
    for (size_t i = 1; i < policy->entry_count; i++) {
        if (cache_key_compare(cache_entry_key(&policy->entries[i - 1u]),
                              cache_entry_key(&policy->entries[i])) >= 0) {
            return false;
        }
    }
    return true;
}

static bool cache_slot_state_valid(
        const ds4_laguna_cache_policy *policy,
        size_t slot_index,
        size_t *entry_index) {
    if (!cache_policy_core_valid(policy) ||
        slot_index >= policy->slot_count) {
        return false;
    }
    const ds4_laguna_cache_slot *slot = &policy->slots[slot_index];
    if (slot->last_used > policy->sequence) return false;
    if (slot->state == DS4_LAGUNA_CACHE_SLOT_EMPTY) {
        return slot->refs == 0 && slot->layer == UINT32_MAX &&
               slot->expert == UINT32_MAX;
    }
    if (slot->generation == 0) return false;
    size_t found_entry = 0;
    if (!cache_entry_index(policy, cache_slot_key(slot), &found_entry) ||
        policy->entry_to_slot[found_entry] != (uint32_t)slot_index) {
        return false;
    }
    if (slot->state == DS4_LAGUNA_CACHE_SLOT_LOADING ||
        slot->state == DS4_LAGUNA_CACHE_SLOT_READY) {
        if (slot->refs != 0) return false;
    } else if (slot->state == DS4_LAGUNA_CACHE_SLOT_IN_USE) {
        if (slot->refs == 0) return false;
    } else {
        return false;
    }
    if (entry_index) *entry_index = found_entry;
    return true;
}

ds4_laguna_cache_status ds4_laguna_cache_policy_audit(
        const ds4_laguna_cache_policy *policy) {
    if (!cache_entries_ordered(policy)) return DS4_LAGUNA_CACHE_UNSAFE;
    for (size_t i = 0; i < policy->entry_count; i++) {
        const uint32_t slot_index = policy->entry_to_slot[i];
        if (slot_index == DS4_LAGUNA_CACHE_SLOT_NONE) continue;
        if ((size_t)slot_index >= policy->slot_count ||
            !cache_key_equal(cache_slot_key(&policy->slots[slot_index]),
                             cache_entry_key(&policy->entries[i]))) {
            return DS4_LAGUNA_CACHE_UNSAFE;
        }
    }
    for (size_t i = 0; i < policy->slot_count; i++) {
        if (!cache_slot_state_valid(policy, i, NULL)) {
            return DS4_LAGUNA_CACHE_UNSAFE;
        }
    }
    return DS4_LAGUNA_CACHE_OK;
}

static void cache_slot_clear(ds4_laguna_cache_slot *slot) {
    slot->last_used = 0;
    slot->layer = UINT32_MAX;
    slot->expert = UINT32_MAX;
    slot->refs = 0;
    slot->state = DS4_LAGUNA_CACHE_SLOT_EMPTY;
}

static ds4_laguna_cache_handle cache_handle_for_slot(
        const ds4_laguna_cache_policy *policy,
        size_t entry_index,
        size_t slot_index) {
    const ds4_laguna_cache_slot *slot = &policy->slots[slot_index];
    const ds4_laguna_cache_handle handle = {
        .slot_index = (uint32_t)slot_index,
        .generation = slot->generation,
        .entry_index = entry_index,
        .key = cache_entry_key(&policy->entries[entry_index]),
    };
    return handle;
}

static void cache_handle_clear(ds4_laguna_cache_handle *handle) {
    if (!handle) return;
    handle->slot_index = DS4_LAGUNA_CACHE_SLOT_NONE;
    handle->generation = 0;
    handle->entry_index = SIZE_MAX;
    handle->key.layer_id = UINT32_MAX;
    handle->key.expert_id = UINT32_MAX;
    handle->lifecycle_epoch = 0;
}

static ds4_laguna_cache_status cache_handle_resolve(
        ds4_laguna_cache_policy *policy,
        ds4_laguna_cache_handle handle,
        ds4_laguna_cache_slot **slot_out,
        size_t *entry_index_out) {
    if (!cache_policy_core_valid(policy) ||
        handle.slot_index == DS4_LAGUNA_CACHE_SLOT_NONE ||
        (size_t)handle.slot_index >= policy->slot_count ||
        handle.entry_index >= policy->entry_count ||
        !cache_key_equal(handle.key,
            cache_entry_key(&policy->entries[handle.entry_index]))) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    ds4_laguna_cache_slot *slot = &policy->slots[handle.slot_index];
    if (slot->generation != handle.generation) {
        return DS4_LAGUNA_CACHE_RECOVERABLE;
    }
    if (slot->state == DS4_LAGUNA_CACHE_SLOT_EMPTY) {
        if (slot->refs != 0 || slot->layer != UINT32_MAX ||
            slot->expert != UINT32_MAX ||
            policy->entry_to_slot[handle.entry_index] !=
                DS4_LAGUNA_CACHE_SLOT_NONE) {
            return DS4_LAGUNA_CACHE_UNSAFE;
        }
    } else {
        size_t slot_entry = 0;
        if (!cache_key_equal(cache_slot_key(slot), handle.key) ||
            !cache_slot_state_valid(policy, handle.slot_index,
                                    &slot_entry) ||
            slot_entry != handle.entry_index) {
            return DS4_LAGUNA_CACHE_UNSAFE;
        }
    }
    if (slot_out) *slot_out = slot;
    if (entry_index_out) *entry_index_out = handle.entry_index;
    return DS4_LAGUNA_CACHE_OK;
}

ds4_laguna_cache_status ds4_laguna_cache_policy_init(
        ds4_laguna_cache_policy *policy,
        const ds4_laguna_expert_entry *entries,
        size_t entry_count,
        ds4_laguna_cache_slot *slots,
        size_t slot_count,
        uint64_t *route_hotness,
        uint32_t *entry_to_slot,
        size_t max_selected_per_token) {
    if (!policy) return DS4_LAGUNA_CACHE_UNSAFE;
    memset(policy, 0, sizeof(*policy));
    if (!entries || entry_count == 0 || !slots || slot_count == 0 ||
        slot_count > UINT32_MAX || !route_hotness || !entry_to_slot ||
        max_selected_per_token == 0 ||
        max_selected_per_token > slot_count) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    for (size_t i = 1; i < entry_count; i++) {
        if (cache_key_compare(cache_entry_key(&entries[i - 1u]),
                              cache_entry_key(&entries[i])) >= 0) {
            return DS4_LAGUNA_CACHE_UNSAFE;
        }
    }

    policy->entries = entries;
    policy->entry_count = entry_count;
    policy->slots = slots;
    policy->slot_count = slot_count;
    policy->route_hotness = route_hotness;
    policy->entry_to_slot = entry_to_slot;
    policy->max_selected_per_token = max_selected_per_token;
    for (size_t i = 0; i < entry_count; i++) {
        route_hotness[i] = 0;
        entry_to_slot[i] = DS4_LAGUNA_CACHE_SLOT_NONE;
    }
    for (size_t i = 0; i < slot_count; i++) {
        memset(&slots[i], 0, sizeof(slots[i]));
        slots[i].layer = UINT32_MAX;
        slots[i].expert = UINT32_MAX;
        slots[i].state = DS4_LAGUNA_CACHE_SLOT_EMPTY;
    }
    return ds4_laguna_cache_policy_audit(policy);
}

ds4_laguna_cache_status ds4_laguna_cache_policy_note_routes(
        ds4_laguna_cache_policy *policy,
        const ds4_laguna_expert_key *keys,
        size_t key_count) {
    if (!cache_policy_core_valid(policy) ||
        (key_count != 0 && !keys)) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    for (size_t i = 0; i < key_count; i++) {
        if (!cache_entry_index(policy, keys[i], NULL)) {
            return DS4_LAGUNA_CACHE_UNSAFE;
        }
    }
    for (size_t i = 0; i < key_count; i++) {
        size_t entry_index = 0;
        (void)cache_entry_index(policy, keys[i], &entry_index);
        if (policy->route_hotness[entry_index] != UINT64_MAX) {
            policy->route_hotness[entry_index]++;
        }
    }
    return DS4_LAGUNA_CACHE_OK;
}

ds4_laguna_cache_status ds4_laguna_cache_policy_note_route(
        ds4_laguna_cache_policy *policy,
        ds4_laguna_expert_key key) {
    return ds4_laguna_cache_policy_note_routes(policy, &key, 1u);
}

static bool cache_victim_precedes(const ds4_laguna_cache_policy *policy,
                                  size_t candidate,
                                  size_t candidate_entry,
                                  size_t current,
                                  size_t current_entry) {
    const ds4_laguna_cache_slot *a = &policy->slots[candidate];
    const ds4_laguna_cache_slot *b = &policy->slots[current];
    if (policy->route_hotness[candidate_entry] !=
        policy->route_hotness[current_entry]) {
        return policy->route_hotness[candidate_entry] <
               policy->route_hotness[current_entry];
    }
    if (a->last_used != b->last_used) {
        return a->last_used < b->last_used;
    }
    return cache_key_compare(cache_slot_key(a), cache_slot_key(b)) < 0;
}

ds4_laguna_cache_status ds4_laguna_cache_policy_acquire(
        ds4_laguna_cache_policy *policy,
        ds4_laguna_expert_key key,
        ds4_laguna_cache_handle *handle,
        ds4_laguna_cache_acquire_outcome *outcome) {
    cache_handle_clear(handle);
    if (outcome) *outcome = DS4_LAGUNA_CACHE_ACQUIRE_NONE;
    if (!handle || !outcome || !cache_policy_core_valid(policy)) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }

    size_t entry_index = 0;
    if (!cache_entry_index(policy, key, &entry_index)) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    const uint32_t published = policy->entry_to_slot[entry_index];
    if (published != DS4_LAGUNA_CACHE_SLOT_NONE) {
        if ((size_t)published >= policy->slot_count ||
            !cache_slot_state_valid(policy, published, NULL)) {
            return DS4_LAGUNA_CACHE_UNSAFE;
        }
        ds4_laguna_cache_slot *slot = &policy->slots[published];
        if (!cache_key_equal(cache_slot_key(slot), key)) {
            return DS4_LAGUNA_CACHE_UNSAFE;
        }
        if (slot->state == DS4_LAGUNA_CACHE_SLOT_LOADING) {
            *outcome = DS4_LAGUNA_CACHE_ACQUIRE_BUSY_LOADING;
            return DS4_LAGUNA_CACHE_RECOVERABLE;
        }
        if (slot->state == DS4_LAGUNA_CACHE_SLOT_IN_USE) {
            *outcome = DS4_LAGUNA_CACHE_ACQUIRE_BUSY_IN_USE;
            return DS4_LAGUNA_CACHE_RECOVERABLE;
        }
        if (slot->state != DS4_LAGUNA_CACHE_SLOT_READY ||
            policy->sequence == UINT64_MAX) {
            return DS4_LAGUNA_CACHE_UNSAFE;
        }
        policy->sequence++;
        slot->last_used = policy->sequence;
        slot->refs = 1u;
        slot->state = DS4_LAGUNA_CACHE_SLOT_IN_USE;
        *handle = cache_handle_for_slot(policy, entry_index, published);
        *outcome = DS4_LAGUNA_CACHE_ACQUIRE_HIT_RESERVED;
        return DS4_LAGUNA_CACHE_OK;
    }

    size_t selected_slot = SIZE_MAX;
    size_t selected_entry = SIZE_MAX;
    for (size_t i = 0; i < policy->slot_count; i++) {
        size_t slot_entry = SIZE_MAX;
        if (!cache_slot_state_valid(policy, i, &slot_entry)) {
            return DS4_LAGUNA_CACHE_UNSAFE;
        }
        const ds4_laguna_cache_slot *slot = &policy->slots[i];
        if (slot->state == DS4_LAGUNA_CACHE_SLOT_EMPTY) {
            if (selected_slot == SIZE_MAX ||
                policy->slots[selected_slot].state !=
                    DS4_LAGUNA_CACHE_SLOT_EMPTY) {
                selected_slot = i;
                selected_entry = SIZE_MAX;
            }
        } else if (slot->state == DS4_LAGUNA_CACHE_SLOT_READY &&
                   (selected_slot == SIZE_MAX ||
                    (policy->slots[selected_slot].state !=
                         DS4_LAGUNA_CACHE_SLOT_EMPTY &&
                     cache_victim_precedes(policy, i, slot_entry,
                                           selected_slot,
                                           selected_entry)))) {
            selected_slot = i;
            selected_entry = slot_entry;
        }
    }
    if (selected_slot == SIZE_MAX) {
        *outcome = DS4_LAGUNA_CACHE_ACQUIRE_PRESSURE;
        return DS4_LAGUNA_CACHE_RECOVERABLE;
    }

    ds4_laguna_cache_slot *slot = &policy->slots[selected_slot];
    if (slot->generation == UINT64_MAX ||
        policy->sequence == UINT64_MAX) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    if (slot->state == DS4_LAGUNA_CACHE_SLOT_READY) {
        policy->entry_to_slot[selected_entry] = DS4_LAGUNA_CACHE_SLOT_NONE;
    }

    policy->sequence++;
    slot->generation++;
    slot->last_used = policy->sequence;
    slot->layer = key.layer_id;
    slot->expert = key.expert_id;
    slot->refs = 0;
    slot->state = DS4_LAGUNA_CACHE_SLOT_LOADING;
    policy->entry_to_slot[entry_index] = (uint32_t)selected_slot;
    *handle = cache_handle_for_slot(policy, entry_index, selected_slot);
    *outcome = DS4_LAGUNA_CACHE_ACQUIRE_LOAD_OWNER;
    return DS4_LAGUNA_CACHE_OK;
}

ds4_laguna_cache_status ds4_laguna_cache_policy_publish(
        ds4_laguna_cache_policy *policy,
        ds4_laguna_cache_handle handle) {
    ds4_laguna_cache_slot *slot = NULL;
    const ds4_laguna_cache_status resolved =
        cache_handle_resolve(policy, handle, &slot, NULL);
    if (resolved != DS4_LAGUNA_CACHE_OK) return resolved;
    if (slot->state == DS4_LAGUNA_CACHE_SLOT_EMPTY) {
        return DS4_LAGUNA_CACHE_RECOVERABLE;
    }
    if (slot->state != DS4_LAGUNA_CACHE_SLOT_LOADING) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    slot->refs = 1u;
    slot->state = DS4_LAGUNA_CACHE_SLOT_IN_USE;
    return DS4_LAGUNA_CACHE_OK;
}

ds4_laguna_cache_status ds4_laguna_cache_policy_pin(
        ds4_laguna_cache_policy *policy,
        ds4_laguna_cache_handle handle) {
    ds4_laguna_cache_slot *slot = NULL;
    const ds4_laguna_cache_status resolved =
        cache_handle_resolve(policy, handle, &slot, NULL);
    if (resolved != DS4_LAGUNA_CACHE_OK) return resolved;
    if ((slot->state != DS4_LAGUNA_CACHE_SLOT_READY &&
         slot->state != DS4_LAGUNA_CACHE_SLOT_IN_USE) ||
        slot->refs == UINT32_MAX) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    slot->refs++;
    slot->state = DS4_LAGUNA_CACHE_SLOT_IN_USE;
    return DS4_LAGUNA_CACHE_OK;
}

ds4_laguna_cache_status ds4_laguna_cache_policy_unpin(
        ds4_laguna_cache_policy *policy,
        ds4_laguna_cache_handle handle) {
    ds4_laguna_cache_slot *slot = NULL;
    const ds4_laguna_cache_status resolved =
        cache_handle_resolve(policy, handle, &slot, NULL);
    if (resolved != DS4_LAGUNA_CACHE_OK) return resolved;
    if (slot->state != DS4_LAGUNA_CACHE_SLOT_IN_USE || slot->refs == 0) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    if (slot->refs == 1u && policy->sequence == UINT64_MAX) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    slot->refs--;
    if (slot->refs == 0) {
        policy->sequence++;
        slot->last_used = policy->sequence;
        slot->state = DS4_LAGUNA_CACHE_SLOT_READY;
    }
    return DS4_LAGUNA_CACHE_OK;
}

ds4_laguna_cache_status ds4_laguna_cache_policy_fail(
        ds4_laguna_cache_policy *policy,
        ds4_laguna_cache_handle handle) {
    if (ds4_laguna_cache_policy_audit(policy) != DS4_LAGUNA_CACHE_OK) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    ds4_laguna_cache_slot *slot = NULL;
    size_t entry_index = SIZE_MAX;
    const ds4_laguna_cache_status resolved =
        cache_handle_resolve(policy, handle, &slot, &entry_index);
    if (resolved != DS4_LAGUNA_CACHE_OK) return resolved;
    if (slot->state == DS4_LAGUNA_CACHE_SLOT_EMPTY) {
        return DS4_LAGUNA_CACHE_RECOVERABLE;
    }
    if (slot->state != DS4_LAGUNA_CACHE_SLOT_LOADING) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    policy->entry_to_slot[entry_index] = DS4_LAGUNA_CACHE_SLOT_NONE;
    cache_slot_clear(slot);
    return ds4_laguna_cache_policy_audit(policy) == DS4_LAGUNA_CACHE_OK
               ? DS4_LAGUNA_CACHE_RECOVERABLE
               : DS4_LAGUNA_CACHE_UNSAFE;
}

ds4_laguna_cache_status ds4_laguna_cache_policy_cancel(
        ds4_laguna_cache_policy *policy,
        ds4_laguna_cache_handle handle) {
    if (ds4_laguna_cache_policy_audit(policy) != DS4_LAGUNA_CACHE_OK) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    if (handle.slot_index == DS4_LAGUNA_CACHE_SLOT_NONE ||
        (size_t)handle.slot_index >= policy->slot_count) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    ds4_laguna_cache_slot *slot = &policy->slots[handle.slot_index];
    if (slot->generation != handle.generation) {
        return DS4_LAGUNA_CACHE_RECOVERABLE;
    }
    if (slot->state == DS4_LAGUNA_CACHE_SLOT_EMPTY) {
        return cache_slot_state_valid(policy, handle.slot_index, NULL)
                   ? DS4_LAGUNA_CACHE_OK
                   : DS4_LAGUNA_CACHE_UNSAFE;
    }
    size_t entry_index = SIZE_MAX;
    const ds4_laguna_cache_status resolved =
        cache_handle_resolve(policy, handle, &slot, &entry_index);
    if (resolved != DS4_LAGUNA_CACHE_OK) return resolved;
    if (slot->state == DS4_LAGUNA_CACHE_SLOT_READY) {
        return DS4_LAGUNA_CACHE_OK;
    }
    if (slot->state == DS4_LAGUNA_CACHE_SLOT_IN_USE) {
        return DS4_LAGUNA_CACHE_RECOVERABLE;
    }
    if (slot->state != DS4_LAGUNA_CACHE_SLOT_LOADING) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    policy->entry_to_slot[entry_index] = DS4_LAGUNA_CACHE_SLOT_NONE;
    cache_slot_clear(slot);
    return ds4_laguna_cache_policy_audit(policy) == DS4_LAGUNA_CACHE_OK
               ? DS4_LAGUNA_CACHE_RECOVERABLE
               : DS4_LAGUNA_CACHE_UNSAFE;
}

ds4_laguna_cache_status ds4_laguna_cache_policy_drain(
        ds4_laguna_cache_policy *policy) {
    if (ds4_laguna_cache_policy_audit(policy) != DS4_LAGUNA_CACHE_OK) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    for (size_t i = 0; i < policy->slot_count; i++) {
        if (!cache_slot_state_valid(policy, i, NULL)) {
            return DS4_LAGUNA_CACHE_UNSAFE;
        }
        if (policy->slots[i].state == DS4_LAGUNA_CACHE_SLOT_LOADING ||
            policy->slots[i].state == DS4_LAGUNA_CACHE_SLOT_IN_USE) {
            return DS4_LAGUNA_CACHE_RECOVERABLE;
        }
    }
    for (size_t i = 0; i < policy->slot_count; i++) {
        ds4_laguna_cache_slot *slot = &policy->slots[i];
        if (slot->state == DS4_LAGUNA_CACHE_SLOT_READY) {
            size_t entry_index = 0;
            (void)cache_entry_index(policy, cache_slot_key(slot),
                                    &entry_index);
            policy->entry_to_slot[entry_index] =
                DS4_LAGUNA_CACHE_SLOT_NONE;
            cache_slot_clear(slot);
        }
    }
    return ds4_laguna_cache_policy_audit(policy);
}

static void cache_group_outputs_clear(
        ds4_laguna_expert_key *grouped_keys,
        size_t grouped_key_capacity,
        ds4_laguna_expert_group *groups,
        size_t group_capacity,
        size_t *grouped_key_count,
        size_t *group_count) {
    if (grouped_key_count) *grouped_key_count = 0;
    if (group_count) *group_count = 0;
    if (grouped_keys && grouped_key_capacity != 0) {
        memset(grouped_keys, 0,
               grouped_key_capacity * sizeof(grouped_keys[0]));
    }
    if (groups && group_capacity != 0) {
        memset(groups, 0, group_capacity * sizeof(groups[0]));
    }
}

static bool cache_range_bounds(const void *base,
                               size_t bytes,
                               uintptr_t *start,
                               uintptr_t *end) {
    if (bytes != 0 && !base) return false;
    const uintptr_t address = (uintptr_t)base;
    if (address > UINTPTR_MAX - bytes) return false;
    *start = address;
    *end = address + bytes;
    return true;
}

static bool cache_ranges_overlap(const void *a,
                                 size_t a_bytes,
                                 const void *b,
                                 size_t b_bytes) {
    if (a_bytes == 0 || b_bytes == 0) return false;
    uintptr_t a_start = 0;
    uintptr_t a_end = 0;
    uintptr_t b_start = 0;
    uintptr_t b_end = 0;
    if (!cache_range_bounds(a, a_bytes, &a_start, &a_end) ||
        !cache_range_bounds(b, b_bytes, &b_start, &b_end)) {
        return true;
    }
    return a_start < b_end && b_start < a_end;
}

ds4_laguna_cache_status ds4_laguna_cache_policy_group(
        const ds4_laguna_cache_policy *policy,
        const ds4_laguna_expert_key *selected,
        size_t token_count,
        size_t selected_per_token,
        ds4_laguna_expert_key *grouped_keys,
        size_t grouped_key_capacity,
        ds4_laguna_expert_group *groups,
        size_t group_capacity,
        size_t *grouped_key_count,
        size_t *group_count) {
    if (grouped_key_count) *grouped_key_count = 0;
    if (group_count) *group_count = 0;
    if (!grouped_key_count || !group_count ||
        grouped_key_capacity > SIZE_MAX / sizeof(grouped_keys[0]) ||
        group_capacity > SIZE_MAX / sizeof(groups[0]) ||
        (grouped_key_capacity != 0 && !grouped_keys) ||
        (group_capacity != 0 && !groups)) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    if (!cache_policy_core_valid(policy) || selected_per_token == 0 ||
        token_count > SIZE_MAX / selected_per_token) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    const size_t selected_count = token_count * selected_per_token;
    if (selected_count > SIZE_MAX / sizeof(selected[0]) ||
        (selected_count != 0 && !selected)) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    const size_t selected_bytes = selected_count * sizeof(selected[0]);
    const size_t grouped_bytes =
        grouped_key_capacity * sizeof(grouped_keys[0]);
    const size_t group_bytes = group_capacity * sizeof(groups[0]);
    const bool exact_in_place = grouped_keys == selected;
    if ((!exact_in_place &&
         cache_ranges_overlap(selected, selected_bytes,
                              grouped_keys, grouped_bytes)) ||
        cache_ranges_overlap(selected, selected_bytes,
                             groups, group_bytes) ||
        cache_ranges_overlap(grouped_keys, grouped_bytes,
                             groups, group_bytes)) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    if (selected_count == 0) {
        cache_group_outputs_clear(grouped_keys, grouped_key_capacity,
                                  groups, group_capacity,
                                  grouped_key_count, group_count);
        return DS4_LAGUNA_CACHE_OK;
    }

    const uint32_t layer_id = selected[0].layer_id;
    for (size_t row = 0; row < token_count; row++) {
        size_t row_unique = 0;
        const size_t row_start = row * selected_per_token;
        for (size_t column = 0; column < selected_per_token; column++) {
            const ds4_laguna_expert_key key =
                selected[row_start + column];
            if (key.layer_id != layer_id ||
                !cache_entry_index(policy, key, NULL)) {
                return DS4_LAGUNA_CACHE_UNSAFE;
            }

            bool seen_in_row = false;
            for (size_t earlier = 0; earlier < column; earlier++) {
                if (cache_key_equal(key,
                        selected[row_start + earlier])) {
                    seen_in_row = true;
                    break;
                }
            }
            if (!seen_in_row) row_unique++;
            if (row_unique > policy->slot_count ||
                row_unique > policy->max_selected_per_token) {
                return DS4_LAGUNA_CACHE_UNSAFE;
            }
        }
    }

    size_t unique_count = 0;
    if (exact_in_place) {
        /* Capacity must be known before compaction mutates the input. Without
         * caller-owned scratch, exact overlap therefore needs an O(N^2)
         * first-occurrence preflight; the normal disjoint path below remains
         * O(N * U). */
        for (size_t i = 0; i < selected_count; i++) {
            bool seen = false;
            for (size_t earlier = 0; earlier < i; earlier++) {
                if (cache_key_equal(selected[i], selected[earlier])) {
                    seen = true;
                    break;
                }
            }
            if (!seen) unique_count++;
        }
        if (unique_count > grouped_key_capacity) {
            return DS4_LAGUNA_CACHE_RECOVERABLE;
        }
    } else {
        /* Allocation-free stable deduplication is intentionally O(N * U)
         * until measured usage warrants caller-owned bitmap/epoch scratch. */
        cache_group_outputs_clear(grouped_keys, grouped_key_capacity,
                                  groups, group_capacity,
                                  grouped_key_count, group_count);
        for (size_t i = 0; i < selected_count; i++) {
            bool seen = false;
            for (size_t earlier = 0; earlier < unique_count; earlier++) {
                if (cache_key_equal(selected[i], grouped_keys[earlier])) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;
            if (unique_count >= grouped_key_capacity) {
                cache_group_outputs_clear(
                    grouped_keys, grouped_key_capacity,
                    groups, group_capacity,
                    grouped_key_count, group_count);
                return DS4_LAGUNA_CACHE_RECOVERABLE;
            }
            grouped_keys[unique_count++] = selected[i];
        }
    }

    if (unique_count > UINT32_MAX) {
        return DS4_LAGUNA_CACHE_UNSAFE;
    }
    const size_t required_groups =
        unique_count / policy->slot_count +
        (unique_count % policy->slot_count != 0 ? 1u : 0u);
    if (required_groups > group_capacity) {
        if (!exact_in_place) {
            cache_group_outputs_clear(
                grouped_keys, grouped_key_capacity,
                groups, group_capacity,
                grouped_key_count, group_count);
        }
        return DS4_LAGUNA_CACHE_RECOVERABLE;
    }

    if (exact_in_place) {
        size_t written = 0;
        for (size_t i = 0; i < selected_count; i++) {
            bool seen = false;
            for (size_t earlier = 0; earlier < written; earlier++) {
                if (cache_key_equal(selected[i], grouped_keys[earlier])) {
                    seen = true;
                    break;
                }
            }
            if (!seen) grouped_keys[written++] = selected[i];
        }
    }
    if (unique_count < grouped_key_capacity) {
        memset(&grouped_keys[unique_count], 0,
               (grouped_key_capacity - unique_count) *
                   sizeof(grouped_keys[0]));
    }
    if (groups && group_capacity != 0) {
        memset(groups, 0, group_bytes);
    }
    for (size_t i = 0; i < required_groups; i++) {
        const size_t first = i * policy->slot_count;
        size_t count = unique_count - first;
        if (count > policy->slot_count) count = policy->slot_count;
        groups[i].first_key = (uint32_t)first;
        groups[i].key_count = (uint32_t)count;
    }
    *grouped_key_count = unique_count;
    *group_count = required_groups;
    return DS4_LAGUNA_CACHE_OK;
}

bool ds4_laguna_ledger_build(ds4_laguna_ledger *out,
                             const ds4_laguna_ledger_spec *spec,
                             const ds4_laguna_tensor_desc *tensors,
                             size_t n_tensors,
                             char *err,
                             size_t errlen) {
    if (err && errlen != 0) err[0] = '\0';
    if (!out) return set_error(err, errlen, "ledger output is null");
    if (out->tensor_ranges || out->source_ranges || out->expert_entries) {
        return set_error(err, errlen,
                         "ledger output must be empty before build");
    }
    memset(out, 0, sizeof(*out));
    if (!spec) return set_error(err, errlen, "ledger spec is null");
    if (!tensors || n_tensors == 0) {
        return set_error(err, errlen, "ledger tensor table is null or empty");
    }
    if (!is_power_of_two(spec->gguf_alignment)) {
        return set_error(err, errlen,
                         "GGUF alignment must be a nonzero power of two");
    }
    if (spec->device_alignment != 256u) {
        return set_error(err, errlen,
                         "device alignment must be exactly 256 bytes");
    }
    if (spec->header_end == 0) {
        return set_error(err, errlen, "GGUF header range is empty");
    }
    if (spec->header_end > spec->metadata_end ||
        spec->metadata_end > spec->tensor_directory_end ||
        spec->tensor_directory_end > spec->tensor_data_start ||
        spec->tensor_data_start > spec->file_size) {
        return set_error(err, errlen, "malformed GGUF structural boundaries");
    }
    if (spec->tensor_directory_end == spec->metadata_end) {
        return set_error(err, errlen, "GGUF tensor directory range is empty");
    }
    uint64_t expected_tensor_data_start = 0;
    if (!align_u64(spec->tensor_directory_end, spec->gguf_alignment,
                   &expected_tensor_data_start) ||
        spec->tensor_data_start != expected_tensor_data_start) {
        return set_error(err, errlen,
                         "tensor data start violates exact alignment");
    }
    if (spec->first_routed_layer >= spec->layer_count ||
        spec->expert_count == 0) {
        return set_error(err, errlen, "invalid routed layer or expert bounds");
    }
    if (n_tensors > SIZE_MAX / sizeof(out->tensor_ranges[0]) ||
        n_tensors > SIZE_MAX / sizeof(tensor_ref)) {
        return set_error(err, errlen, "tensor table allocation overflow");
    }

    const uint64_t routed_layer_count =
        (uint64_t)spec->layer_count - spec->first_routed_layer;
    uint64_t expert_entry_count = 0;
    if (!mul_u64(routed_layer_count, spec->expert_count,
                 &expert_entry_count) ||
        expert_entry_count > SIZE_MAX /
            sizeof(out->expert_entries[0])) {
        return set_error(err, errlen, "expert entry allocation overflow");
    }
    uint64_t role_count_u64 = 0;
    if (!mul_u64(routed_layer_count, 3u, &role_count_u64) ||
        role_count_u64 > SIZE_MAX / sizeof(ds4_laguna_tensor_desc *)) {
        return set_error(err, errlen, "routed role allocation overflow");
    }
    if (n_tensors > (SIZE_MAX - 5u) / 2u) {
        return set_error(err, errlen, "source range allocation overflow");
    }
    const size_t source_capacity = n_tensors * 2u + 5u;
    if (source_capacity > SIZE_MAX / sizeof(out->source_ranges[0])) {
        return set_error(err, errlen, "source range allocation overflow");
    }

    out->tensor_ranges = calloc(n_tensors, sizeof(out->tensor_ranges[0]));
    out->source_ranges = calloc(source_capacity,
                                sizeof(out->source_ranges[0]));
    out->expert_entries = calloc((size_t)expert_entry_count,
                                 sizeof(out->expert_entries[0]));
    tensor_ref *source_order = calloc(n_tensors, sizeof(source_order[0]));
    const ds4_laguna_tensor_desc **roles =
        calloc((size_t)role_count_u64, sizeof(roles[0]));
    if (!out->tensor_ranges || !out->source_ranges ||
        !out->expert_entries || !source_order || !roles) {
        free(source_order);
        free(roles);
        ds4_laguna_ledger_free(out);
        return set_error(err, errlen, "ledger allocation failed");
    }

    out->file_size = spec->file_size;
    out->tensor_count = n_tensors;
    out->tensor_range_count = n_tensors;
    out->expert_entry_count = expert_entry_count;

    bool ok = true;
    uint64_t common_expert_bytes = 0;
    for (size_t i = 0; ok && i < n_tensors; i++) {
        const ds4_laguna_tensor_desc *tensor = &tensors[i];
        if (!tensor->name || tensor->name_len == 0) {
            ok = set_error(err, errlen,
                           "tensor %llu has an invalid name",
                           (unsigned long long)tensor->stable_index);
            break;
        }
        for (size_t j = 0; j < i; j++) {
            if (tensor->stable_index == tensors[j].stable_index) {
                ok = set_error(err, errlen,
                               "duplicate tensor stable index %llu",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor_names_equal(tensor, &tensors[j])) {
                ok = set_error(err, errlen,
                               "duplicate tensor name at stable index %llu",
                               (unsigned long long)tensor->stable_index);
                break;
            }
        }
        if (!ok) break;
        if (tensor->tensor_class == DS4_LAGUNA_TENSOR_UNCLASSIFIED) {
            ok = set_error(err, errlen,
                           "tensor %llu is unclassified",
                           (unsigned long long)tensor->stable_index);
            break;
        }
        if (tensor->tensor_class != DS4_LAGUNA_TENSOR_STATIC &&
            tensor->tensor_class != DS4_LAGUNA_TENSOR_ROUTED_EXPERT) {
            ok = set_error(err, errlen,
                           "tensor %llu has an invalid class",
                           (unsigned long long)tensor->stable_index);
            break;
        }
        if (tensor->source_offset < spec->tensor_data_start) {
            ok = set_error(err, errlen,
                           "tensor %llu starts before tensor data",
                           (unsigned long long)tensor->stable_index);
            break;
        }
        if ((tensor->source_offset - spec->tensor_data_start) %
                spec->gguf_alignment != 0) {
            ok = set_error(err, errlen,
                           "tensor %llu source offset violates GGUF alignment",
                           (unsigned long long)tensor->stable_index);
            break;
        }
        if (tensor->source_offset > spec->file_size ||
            tensor->source_bytes > spec->file_size - tensor->source_offset) {
            ok = set_error(err, errlen,
                           "tensor %llu points outside the file",
                           (unsigned long long)tensor->stable_index);
            break;
        }

        uint64_t elements = 0;
        uint64_t expected_bytes = 0;
        if (!tensor_shape_bytes(tensor, &elements, &expected_bytes,
                                err, errlen)) {
            ok = false;
            break;
        }
        (void)elements;

        uint64_t expert_bytes = 0;
        if (tensor->tensor_class == DS4_LAGUNA_TENSOR_STATIC) {
            if (tensor->routed_layer != UINT32_MAX ||
                tensor->routed_projection !=
                    DS4_LAGUNA_ROUTED_PROJECTION_NONE) {
                ok = set_error(err, errlen,
                               "static tensor %llu claims a routed role",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor->source_bytes != expected_bytes) {
                ok = set_error(err, errlen,
                               "static tensor %llu source size mismatch",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            uint64_t aligned_bytes = 0;
            if (!align_u64(tensor->source_bytes,
                           spec->device_alignment,
                           &aligned_bytes) ||
                !add_u64(out->static_aligned_device_bytes,
                         aligned_bytes,
                         &out->static_aligned_device_bytes)) {
                ok = set_error(err, errlen,
                               "static tensor device alignment overflow");
                break;
            }
            if (!add_u64(out->static_source_bytes,
                         tensor->source_bytes,
                         &out->static_source_bytes)) {
                ok = set_error(err, errlen, "static source byte overflow");
                break;
            }
            out->static_parent_count++;
        } else {
            if (tensor->ndim != 3) {
                ok = set_error(err, errlen,
                               "routed tensor %llu must have three dimensions",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor->dim[0] % tensor->block_elems != 0) {
                ok = set_error(err, errlen,
                               "routed tensor %llu dim0 is not block-exact",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor->dim[2] != spec->expert_count) {
                ok = set_error(err, errlen,
                               "routed tensor %llu dim2 does not match expert count",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            uint64_t row_bytes = 0;
            uint64_t parent_bytes = 0;
            if (!mul_u64(tensor->dim[0] / tensor->block_elems,
                         tensor->block_bytes,
                         &row_bytes) ||
                !mul_u64(tensor->dim[1], row_bytes, &expert_bytes) ||
                !mul_u64(tensor->dim[2], expert_bytes, &parent_bytes)) {
                ok = set_error(err, errlen,
                               "routed tensor %llu size overflow",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor->source_bytes != parent_bytes ||
                tensor->source_bytes != expected_bytes) {
                ok = set_error(err, errlen,
                               "routed tensor %llu source size mismatch",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor->routed_layer < spec->first_routed_layer ||
                tensor->routed_layer >= spec->layer_count) {
                ok = set_error(err, errlen,
                               "routed tensor %llu has an out-of-range layer",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            if (tensor->routed_projection <
                    DS4_LAGUNA_ROUTED_PROJECTION_GATE ||
                tensor->routed_projection >
                    DS4_LAGUNA_ROUTED_PROJECTION_DOWN) {
                ok = set_error(err, errlen,
                               "routed tensor %llu has an invalid projection",
                               (unsigned long long)tensor->stable_index);
                break;
            }
            const size_t role =
                ((size_t)(tensor->routed_layer - spec->first_routed_layer) *
                 3u) +
                ((size_t)tensor->routed_projection - 1u);
            if (roles[role]) {
                ok = set_error(err, errlen,
                               "duplicate routed projection for layer %u",
                               tensor->routed_layer);
                break;
            }
            roles[role] = tensor;
            if (common_expert_bytes == 0) {
                common_expert_bytes = expert_bytes;
            } else if (expert_bytes != common_expert_bytes) {
                ok = set_error(err, errlen,
                               "inconsistent routed projection size");
                break;
            }
            if (!add_u64(out->routed_source_bytes,
                         tensor->source_bytes,
                         &out->routed_source_bytes)) {
                ok = set_error(err, errlen, "routed source byte overflow");
                break;
            }
            out->routed_parent_count++;
        }

        out->tensor_ranges[i].stable_index = tensor->stable_index;
        out->tensor_ranges[i].tensor_class = tensor->tensor_class;
        out->tensor_ranges[i].routed_layer =
            tensor->tensor_class == DS4_LAGUNA_TENSOR_ROUTED_EXPERT ?
                tensor->routed_layer : UINT32_MAX;
        out->tensor_ranges[i].routed_projection =
            tensor->tensor_class == DS4_LAGUNA_TENSOR_ROUTED_EXPERT ?
                tensor->routed_projection :
                DS4_LAGUNA_ROUTED_PROJECTION_NONE;
        out->tensor_ranges[i].source_offset = tensor->source_offset;
        out->tensor_ranges[i].source_bytes = tensor->source_bytes;
        source_order[i].tensor = tensor;
    }

    for (size_t role = 0; ok && role < (size_t)role_count_u64; role++) {
        if (!roles[role]) {
            ok = set_error(err, errlen,
                           "missing routed projection for layer %u",
                           spec->first_routed_layer + (uint32_t)(role / 3u));
        }
    }

    if (ok) {
        out->routed_projection_expert_bytes = common_expert_bytes;
        size_t entry_index = 0;
        for (uint32_t layer = spec->first_routed_layer;
             ok && layer < spec->layer_count;
             layer++) {
            const size_t role_base =
                (size_t)(layer - spec->first_routed_layer) * 3u;
            const ds4_laguna_tensor_desc *gate = roles[role_base];
            const ds4_laguna_tensor_desc *up = roles[role_base + 1u];
            const ds4_laguna_tensor_desc *down = roles[role_base + 2u];
            for (uint32_t expert = 0;
                 ok && expert < spec->expert_count;
                 expert++, entry_index++) {
                ds4_laguna_expert_entry *entry =
                    &out->expert_entries[entry_index];
                entry->layer = layer;
                entry->expert = expert;
                const ds4_laguna_tensor_desc *parents[3] = {
                    gate, up, down,
                };
                ds4_laguna_expert_view *views[3] = {
                    &entry->gate, &entry->up, &entry->down,
                };
                uint64_t device_cursor = 0;
                for (size_t projection = 0; projection < 3; projection++) {
                    uint64_t expert_delta = 0;
                    if (!mul_u64(expert, common_expert_bytes,
                                 &expert_delta) ||
                        !add_u64(parents[projection]->source_offset,
                                 expert_delta,
                                 &views[projection]->source_offset)) {
                        ok = set_error(err, errlen,
                                       "expert source view overflow");
                        break;
                    }
                    views[projection]->parent_stable_index =
                        parents[projection]->stable_index;
                    views[projection]->source_bytes = common_expert_bytes;
                    views[projection]->device_offset = device_cursor;
                    if (!add_u64(device_cursor, common_expert_bytes,
                                 &device_cursor) ||
                        !align_u64(device_cursor, spec->device_alignment,
                                   &device_cursor)) {
                        ok = set_error(err, errlen,
                                       "expert slot alignment overflow");
                        break;
                    }
                }
                entry->used_bytes = device_cursor;
                if (entry->used_bytes > out->slot_stride_bytes) {
                    out->slot_stride_bytes = entry->used_bytes;
                }
            }
        }
    }

    if (ok) {
        qsort(source_order, n_tensors, sizeof(source_order[0]),
              compare_tensor_ref_source);
        ok = append_source_range(out, source_capacity,
                                 DS4_LAGUNA_SOURCE_HEADER,
                                 0, spec->header_end,
                                 err, errlen) &&
             append_source_range(out, source_capacity,
                                 DS4_LAGUNA_SOURCE_METADATA,
                                 spec->header_end,
                                 spec->metadata_end - spec->header_end,
                                 err, errlen) &&
             append_source_range(out, source_capacity,
                                 DS4_LAGUNA_SOURCE_TENSOR_DIRECTORY,
                                 spec->metadata_end,
                                 spec->tensor_directory_end - spec->metadata_end,
                                 err, errlen) &&
             append_source_range(out, source_capacity,
                                 DS4_LAGUNA_SOURCE_ALIGNMENT_PADDING,
                                 spec->tensor_directory_end,
                                 spec->tensor_data_start -
                                     spec->tensor_directory_end,
                                 err, errlen);
        uint64_t cursor = spec->tensor_data_start;
        for (size_t i = 0; ok && i < n_tensors; i++) {
            const ds4_laguna_tensor_desc *tensor = source_order[i].tensor;
            if (tensor->source_offset < cursor) {
                ok = set_error(err, errlen,
                               "tensor parent source ranges overlap");
                break;
            }
            ok = append_source_range(out, source_capacity,
                                     DS4_LAGUNA_SOURCE_TENSOR_PADDING,
                                     cursor,
                                     tensor->source_offset - cursor,
                                     err, errlen);
            if (!ok || !add_u64(tensor->source_offset,
                                tensor->source_bytes, &cursor)) {
                if (ok) ok = set_error(err, errlen,
                                       "tensor parent end overflow");
                break;
            }
        }
        if (ok) {
            ok = append_source_range(out, source_capacity,
                                     DS4_LAGUNA_SOURCE_TENSOR_PADDING,
                                     cursor, spec->file_size - cursor,
                                     err, errlen);
        }
    }

    if (ok) {
        uint64_t tensor_source_bytes = 0;
        uint64_t reconciled = 0;
        ok = add_u64(out->static_source_bytes,
                     out->routed_source_bytes,
                     &tensor_source_bytes) &&
             add_u64(tensor_source_bytes,
                     out->non_tensor_source_bytes,
                     &reconciled);
        if (!ok || reconciled != spec->file_size) {
            ok = set_error(err, errlen,
                           "full-file source range reconciliation failed");
        }
    }

    free(source_order);
    free(roles);
    if (!ok) ds4_laguna_ledger_free(out);
    return ok;
}

static void allocation_callsite(ds4_laguna_allocation_plan *plan,
                                size_t index,
                                uint32_t id,
                                const char *name,
                                ds4_runtime_category category,
                                ds4_runtime_physical_domain domain,
                                uint64_t bound_bytes) {
    ds4_runtime_callsite *site = &plan->callsites[index];
    site->id = id;
    site->name = name;
    site->category = category;
    site->domain = domain;
    site->bound_bytes = bound_bytes;
}

bool ds4_laguna_allocation_plan_make(
        ds4_laguna_allocation_plan *out,
        const ds4_laguna_ledger *ledger,
        const ds4_laguna_allocation_plan_spec *spec,
        char *err,
        size_t errlen) {
    if (err && errlen != 0) err[0] = '\0';
    if (!out || !ledger || !spec) {
        return set_error(err, errlen, "allocation plan argument is null");
    }
    memset(out, 0, sizeof(*out));

    const uint64_t gib = UINT64_C(1024) * 1024u * 1024u;
    const uint64_t mib = UINT64_C(1024) * 1024u;
    const uint64_t expected_file_size = UINT64_C(68248759648);
    const uint64_t expected_static_bytes = UINT64_C(4374164480);
    const uint64_t expected_slot_stride = UINT64_C(5308416);
    const uint64_t expected_expert_entries = UINT64_C(12032);
    if (spec->context_tokens != 32768u || spec->prefill_rows != 4096u ||
        (spec->session_count != 1u && spec->session_count != 2u)) {
        return set_error(err, errlen,
                         "allocation plan requires a 32K/4K one- or two-session profile");
    }
    if (spec->configured_cache_bytes != 8u * gib &&
        spec->configured_cache_bytes != 12u * gib &&
        spec->configured_cache_bytes != 16u * gib) {
        return set_error(err, errlen,
                         "allocation plan cache must be exactly 8, 12, or 16 GiB");
    }
    if (ledger->slot_stride_bytes == 0 ||
        spec->configured_cache_bytes / ledger->slot_stride_bytes == 0) {
        return set_error(err, errlen,
                         "allocation plan cache contains no complete slot");
    }
    if (ledger->file_size != expected_file_size ||
        ledger->tensor_count != UINT64_C(814) ||
        ledger->static_parent_count != UINT64_C(673) ||
        ledger->routed_parent_count != UINT64_C(141) ||
        ledger->static_aligned_device_bytes != expected_static_bytes ||
        ledger->expert_entry_count != expected_expert_entries ||
        ledger->slot_stride_bytes != expected_slot_stride) {
        return set_error(err, errlen,
                         "allocation plan requires the exact Laguna ledger");
    }

    const uint64_t slots = spec->configured_cache_bytes /
                           ledger->slot_stride_bytes;
    if (slots == 0 || slots > UINT32_MAX) {
        return set_error(err, errlen,
                         "allocation plan cache contains no valid slot count");
    }
    uint64_t cache_payload = 0;
    if (!mul_u64(slots, ledger->slot_stride_bytes, &cache_payload) ||
        cache_payload > spec->configured_cache_bytes) {
        return set_error(err, errlen, "allocation plan cache payload overflow");
    }

    uint64_t source_capacity = 0;
    uint64_t tensor_range_bytes = 0;
    uint64_t source_range_bytes = 0;
    uint64_t expert_entry_bytes = 0;
    uint64_t ledger_arrays = 0;
    if (!mul_u64(ledger->tensor_count, UINT64_C(2), &source_capacity) ||
        !add_u64(source_capacity, UINT64_C(5), &source_capacity) ||
        !mul_u64(ledger->tensor_count,
                 sizeof(ledger->tensor_ranges[0]), &tensor_range_bytes) ||
        !mul_u64(source_capacity,
                 sizeof(ledger->source_ranges[0]), &source_range_bytes) ||
        !mul_u64(ledger->expert_entry_count,
                 sizeof(ledger->expert_entries[0]), &expert_entry_bytes) ||
        !add_u64(tensor_range_bytes, source_range_bytes, &ledger_arrays) ||
        !add_u64(ledger_arrays, expert_entry_bytes, &ledger_arrays)) {
        return set_error(err, errlen,
                         "allocation plan ledger array arithmetic overflow");
    }
    const uint64_t route_hotness = UINT64_C(96256);
    const uint64_t host_entry_to_slot = UINT64_C(48128);
    const uint64_t device_entry_to_slot = UINT64_C(48128);
    const uint64_t static_offsets = UINT64_C(6512);
    uint64_t slot_state = 0;
    uint64_t metadata = 0;
    if (!mul_u64(slots, UINT64_C(32), &slot_state) ||
        !add_u64(ledger_arrays, route_hotness, &metadata) ||
        !add_u64(metadata, host_entry_to_slot, &metadata) ||
        !add_u64(metadata, device_entry_to_slot, &metadata) ||
        !add_u64(metadata, static_offsets, &metadata) ||
        !add_u64(metadata, slot_state, &metadata)) {
        return set_error(err, errlen,
                         "allocation plan metadata arithmetic overflow");
    }

    uint64_t kv_tokens = 0;
    uint64_t kv_bytes_per_session = 0;
    uint64_t kv_bytes = 0;
    if (!mul_u64(UINT64_C(12), spec->context_tokens, &kv_tokens) ||
        !add_u64(kv_tokens, UINT64_C(36) * 512u, &kv_tokens) ||
        !mul_u64(kv_tokens, UINT64_C(4096), &kv_bytes_per_session) ||
        !mul_u64(kv_bytes_per_session, spec->session_count, &kv_bytes)) {
        return set_error(err, errlen, "allocation plan KV arithmetic overflow");
    }
    uint64_t graph_bytes_per_session = 0;
    uint64_t graph_bytes = 0;
    if (!ds4_runtime_checked_affine_bytes(
            spec->prefill_rows, UINT64_C(375156), UINT64_C(413704),
            &graph_bytes_per_session) ||
        !mul_u64(graph_bytes_per_session, spec->session_count,
                 &graph_bytes)) {
        return set_error(err, errlen,
                         "allocation plan graph arithmetic overflow");
    }
    uint64_t staging_bytes = 0;
    if (!mul_u64(UINT64_C(4), ledger->slot_stride_bytes, &staging_bytes)) {
        return set_error(err, errlen,
                         "allocation plan staging arithmetic overflow");
    }

    if (spec->session_count == 1u) {
        out->profile_id = spec->configured_cache_bytes == 8u * gib ?
            "cache-8gib" : spec->configured_cache_bytes == 12u * gib ?
            "cache-12gib" : "cache-16gib";
    } else {
        out->profile_id = spec->configured_cache_bytes == 8u * gib ?
            "cache-8gib-sessions-2" :
            spec->configured_cache_bytes == 12u * gib ?
            "cache-12gib-sessions-2" : "cache-16gib-sessions-2";
    }
    out->context_tokens = spec->context_tokens;
    out->prefill_rows = spec->prefill_rows;
    out->session_count = spec->session_count;
    out->configured_cache_bytes = spec->configured_cache_bytes;
    out->effective_cache_limit_bytes = spec->configured_cache_bytes;
    out->slot_stride_bytes = ledger->slot_stride_bytes;
    out->cache_payload_bytes = cache_payload;
    out->cache_tail_uncharged_bytes =
        spec->configured_cache_bytes - cache_payload;
    out->slot_count = (uint32_t)slots;
    out->staging_buffer_count = 4u;
    out->staging_buffer_bytes = ledger->slot_stride_bytes;
    out->owned_category_bounds[DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS] =
        ledger->static_aligned_device_bytes;
    out->owned_category_bounds[
        DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD] = cache_payload;
    out->owned_category_bounds[
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES] = metadata;
    out->owned_category_bounds[DS4_RUNTIME_CATEGORY_KV_STATE] = kv_bytes;
    out->owned_category_bounds[DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH] =
        graph_bytes;
    out->owned_category_bounds[DS4_RUNTIME_CATEGORY_PINNED_STAGING] =
        staging_bytes;
    uint64_t session_host_bytes = 0;
    uint64_t other_host_bytes = 0;
    if (!mul_u64(128u * mib, spec->session_count, &session_host_bytes) ||
        !add_u64(896u * mib, session_host_bytes, &other_host_bytes)) {
        return set_error(err, errlen,
                         "allocation plan session host arithmetic overflow");
    }
    out->owned_category_bounds[DS4_RUNTIME_CATEGORY_OTHER_HOST] =
        other_host_bytes;
    out->owned_category_bounds[DS4_RUNTIME_CATEGORY_OTHER_CUDA] = 2u * gib;

    out->report_bounds[DS4_RUNTIME_REPORT_MODEL_MAPPED_VIRTUAL] =
        ledger->file_size;
    out->report_bounds[
        DS4_RUNTIME_REPORT_MODEL_MAPPING_REGISTERED] = 0;
    out->report_bounds[DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT] = 2u * gib;
    out->report_bounds[
        DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED] = gib / 2u;
    out->report_bounds[
        DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED] = gib / 2u;

    uint64_t owned_total = 0;
    uint64_t owned_non_cache = 0;
    for (size_t i = 0; i < DS4_RUNTIME_OWNED_CATEGORY_COUNT; i++) {
        if (!add_u64(owned_total, out->owned_category_bounds[i],
                     &owned_total)) {
            return set_error(err, errlen,
                             "allocation plan owned total overflow");
        }
        if (i != DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD &&
            !add_u64(owned_non_cache, out->owned_category_bounds[i],
                     &owned_non_cache)) {
            return set_error(err, errlen,
                             "allocation plan non-cache total overflow");
        }
    }
    uint64_t external = 0;
    if (!add_u64(out->report_bounds[
                     DS4_RUNTIME_REPORT_MODEL_SOURCE_RESIDENT],
                 out->report_bounds[
                     DS4_RUNTIME_REPORT_HOST_LIBRARY_UNATTRIBUTED],
                 &external) ||
        !add_u64(external,
                 out->report_bounds[
                     DS4_RUNTIME_REPORT_CUDA_LIBRARY_UNATTRIBUTED],
                 &external) ||
        !add_u64(owned_non_cache, external,
                 &out->qualification_non_cache_bound_bytes) ||
        !add_u64(owned_total, external,
                 &out->planned_qualification_bytes) ||
        !add_u64(spec->configured_cache_bytes,
                 (spec->session_count == 1u ? 16u : 20u) * gib,
                 &out->qualification_total_bound_bytes)) {
        return set_error(err, errlen,
                         "allocation plan qualification arithmetic overflow");
    }
    out->owned_non_cache_bound_bytes = owned_non_cache;
    out->owned_total_bound_bytes = owned_total;
    if (out->qualification_non_cache_bound_bytes >
            (spec->session_count == 1u ? 16u : 20u) * gib ||
        out->planned_qualification_bytes >
            out->qualification_total_bound_bytes) {
        return set_error(err, errlen,
                         "allocation plan exceeds qualification total bound");
    }

    allocation_callsite(out, 0, DS4_LAGUNA_CALLSITE_STATIC_SLAB,
        "laguna.static_slab", DS4_RUNTIME_CATEGORY_STATIC_WEIGHTS,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, expected_static_bytes);
    allocation_callsite(out, 1, DS4_LAGUNA_CALLSITE_EXPERT_CACHE_PAYLOAD,
        "laguna.expert_cache_payload",
        DS4_RUNTIME_CATEGORY_EXPERT_CACHE_PAYLOAD,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, cache_payload);
    allocation_callsite(out, 2, DS4_LAGUNA_CALLSITE_LEDGER_ARRAYS,
        "laguna.ledger_arrays",
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_HOST, ledger_arrays);
    allocation_callsite(out, 3, DS4_LAGUNA_CALLSITE_ROUTE_HOTNESS,
        "laguna.route_hotness",
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_HOST, route_hotness);
    allocation_callsite(out, 4, DS4_LAGUNA_CALLSITE_HOST_ENTRY_TO_SLOT,
        "laguna.host_entry_to_slot",
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_HOST, host_entry_to_slot);
    allocation_callsite(out, 5, DS4_LAGUNA_CALLSITE_DEVICE_ENTRY_TO_SLOT,
        "laguna.device_entry_to_slot",
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, device_entry_to_slot);
    allocation_callsite(out, 6, DS4_LAGUNA_CALLSITE_STATIC_OFFSETS,
        "laguna.static_offsets",
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_HOST, static_offsets);
    allocation_callsite(out, 7, DS4_LAGUNA_CALLSITE_SLOT_STATE,
        "laguna.slot_state",
        DS4_RUNTIME_CATEGORY_CACHE_METADATA_ADDRESS_TABLES,
        DS4_RUNTIME_DOMAIN_HOST, slot_state);
    allocation_callsite(out, 8, DS4_LAGUNA_CALLSITE_KV_STATE,
        "laguna.kv_state", DS4_RUNTIME_CATEGORY_KV_STATE,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, kv_bytes);
    allocation_callsite(out, 9, DS4_LAGUNA_CALLSITE_GRAPH_SCRATCH,
        "laguna.graph_scratch", DS4_RUNTIME_CATEGORY_GRAPH_SCRATCH,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, graph_bytes);
    allocation_callsite(out, 10, DS4_LAGUNA_CALLSITE_PINNED_STAGING_0,
        "laguna.pinned_staging.0", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
        DS4_RUNTIME_DOMAIN_HOST, ledger->slot_stride_bytes);
    allocation_callsite(out, 11, DS4_LAGUNA_CALLSITE_PINNED_STAGING_1,
        "laguna.pinned_staging.1", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
        DS4_RUNTIME_DOMAIN_HOST, ledger->slot_stride_bytes);
    allocation_callsite(out, 12, DS4_LAGUNA_CALLSITE_PINNED_STAGING_2,
        "laguna.pinned_staging.2", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
        DS4_RUNTIME_DOMAIN_HOST, ledger->slot_stride_bytes);
    allocation_callsite(out, 13, DS4_LAGUNA_CALLSITE_PINNED_STAGING_3,
        "laguna.pinned_staging.3", DS4_RUNTIME_CATEGORY_PINNED_STAGING,
        DS4_RUNTIME_DOMAIN_HOST, ledger->slot_stride_bytes);
    allocation_callsite(out, 14, DS4_LAGUNA_CALLSITE_OTHER_HOST_ENGINE,
        "laguna.other_host.engine", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 64u * mib);
    allocation_callsite(out, 15, DS4_LAGUNA_CALLSITE_OTHER_HOST_MODEL,
        "laguna.other_host.model", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 192u * mib);
    allocation_callsite(out, 16, DS4_LAGUNA_CALLSITE_OTHER_HOST_BOOTSTRAP,
        "laguna.other_host.bootstrap", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 128u * mib);
    allocation_callsite(out, 17, DS4_LAGUNA_CALLSITE_OTHER_HOST_VOCAB,
        "laguna.other_host.vocab", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 256u * mib);
    allocation_callsite(out, 18, DS4_LAGUNA_CALLSITE_OTHER_HOST_SESSION,
        "laguna.other_host.session", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, session_host_bytes);
    allocation_callsite(out, 19, DS4_LAGUNA_CALLSITE_OTHER_HOST_TRACKER,
        "laguna.other_host.tracker", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 128u * mib);
    allocation_callsite(out, 20, DS4_LAGUNA_CALLSITE_OTHER_HOST_SERIALIZER,
        "laguna.other_host.serializer", DS4_RUNTIME_CATEGORY_OTHER_HOST,
        DS4_RUNTIME_DOMAIN_HOST, 128u * mib);
    allocation_callsite(out, 21,
        DS4_LAGUNA_CALLSITE_OTHER_CUDA_KERNEL_TMP,
        "laguna.other_cuda.kernel_tmp", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 512u * mib);
    allocation_callsite(out, 22,
        DS4_LAGUNA_CALLSITE_OTHER_CUDA_ROUTED_WORKSPACE,
        "laguna.other_cuda.routed_workspace",
        DS4_RUNTIME_CATEGORY_OTHER_CUDA,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 768u * mib);
    allocation_callsite(out, 23,
        DS4_LAGUNA_CALLSITE_OTHER_CUDA_DESCRIPTOR_UPLOAD,
        "laguna.other_cuda.descriptor_upload",
        DS4_RUNTIME_CATEGORY_OTHER_CUDA,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 256u * mib);
    allocation_callsite(out, 24,
        DS4_LAGUNA_CALLSITE_OTHER_CUDA_TRANSIENT,
        "laguna.other_cuda.transient", DS4_RUNTIME_CATEGORY_OTHER_CUDA,
        DS4_RUNTIME_DOMAIN_CUDA_DEVICE, 512u * mib);
    out->callsite_count = DS4_LAGUNA_ALLOCATION_CALLSITE_COUNT;
    return true;
}

static int compare_page_range(const void *lhs, const void *rhs) {
    const ds4_laguna_page_range *a = lhs;
    const ds4_laguna_page_range *b = rhs;
    if (a->offset < b->offset) return -1;
    if (a->offset > b->offset) return 1;
    if (a->bytes < b->bytes) return -1;
    if (a->bytes > b->bytes) return 1;
    return 0;
}

bool ds4_laguna_full_page_union(
        const ds4_laguna_page_range *input,
        size_t input_count,
        uint64_t page_size,
        ds4_laguna_page_range *output,
        size_t output_capacity,
        size_t *output_count,
        uint64_t *output_bytes) {
    if (output_count) *output_count = 0;
    if (output_bytes) *output_bytes = 0;
    if (!output_count || !output_bytes || !is_power_of_two(page_size) ||
        (input_count != 0 && !input) ||
        (output_capacity != 0 && !output)) {
        return false;
    }

    size_t rounded_count = 0;
    for (size_t i = 0; i < input_count; i++) {
        if (input[i].bytes == 0) continue;
        uint64_t raw_end = 0;
        if (!add_u64(input[i].offset, input[i].bytes, &raw_end)) {
            return false;
        }
        uint64_t start_pages = input[i].offset / page_size;
        if (input[i].offset % page_size != 0) {
            if (start_pages == UINT64_MAX) return false;
            start_pages++;
        }
        if (start_pages > UINT64_MAX / page_size) return false;
        const uint64_t safe_start = start_pages * page_size;
        const uint64_t safe_end = (raw_end / page_size) * page_size;
        if (safe_start >= safe_end) continue;
        if (rounded_count >= output_capacity) {
            *output_count = 0;
            *output_bytes = 0;
            return false;
        }
        output[rounded_count].offset = safe_start;
        output[rounded_count].bytes = safe_end - safe_start;
        rounded_count++;
    }
    if (rounded_count == 0) return true;

    qsort(output, rounded_count, sizeof(output[0]), compare_page_range);
    size_t merged_count = 1;
    for (size_t i = 1; i < rounded_count; i++) {
        ds4_laguna_page_range *previous = &output[merged_count - 1u];
        uint64_t previous_end = 0;
        uint64_t current_end = 0;
        if (!add_u64(previous->offset, previous->bytes, &previous_end) ||
            !add_u64(output[i].offset, output[i].bytes, &current_end)) {
            *output_count = 0;
            *output_bytes = 0;
            return false;
        }
        if (output[i].offset <= previous_end) {
            if (current_end > previous_end) {
                previous->bytes = current_end - previous->offset;
            }
        } else {
            output[merged_count++] = output[i];
        }
    }

    uint64_t total = 0;
    for (size_t i = 0; i < merged_count; i++) {
        if (!add_u64(total, output[i].bytes, &total)) {
            *output_count = 0;
            *output_bytes = 0;
            return false;
        }
    }
    *output_count = merged_count;
    *output_bytes = total;
    return true;
}

typedef struct {
    size_t first;
    size_t after;
    size_t resulting_count;
    uint64_t merged_start;
    uint64_t merged_bytes;
    uint64_t newly_unique_bytes;
    bool has_eligible_interval;
} ds4_laguna_page_range_insert_plan;

static bool page_range_union_insert_plan(
        const ds4_laguna_page_range *ranges,
        size_t range_count,
        size_t range_capacity,
        uint64_t page_size,
        uint64_t raw_offset,
        uint64_t raw_bytes,
        ds4_laguna_page_range_insert_plan *out) {
    if (!out || !is_power_of_two(page_size) ||
        range_count > range_capacity ||
        (range_capacity != 0 && !ranges)) {
        return false;
    }

    uint64_t previous_end = 0;
    for (size_t i = 0; i < range_count; i++) {
        uint64_t current_end = 0;
        if (ranges[i].bytes == 0 || ranges[i].offset % page_size != 0 ||
            ranges[i].bytes % page_size != 0 ||
            !add_u64(ranges[i].offset, ranges[i].bytes, &current_end) ||
            (i != 0 && previous_end >= ranges[i].offset)) {
            return false;
        }
        previous_end = current_end;
    }

    ds4_laguna_page_range_insert_plan plan;
    memset(&plan, 0, sizeof(plan));
    plan.resulting_count = range_count;
    if (raw_bytes == 0) {
        *out = plan;
        return true;
    }

    uint64_t raw_end = 0;
    if (!add_u64(raw_offset, raw_bytes, &raw_end)) return false;

    uint64_t safe_start = raw_offset;
    const uint64_t start_remainder = safe_start % page_size;
    if (start_remainder != 0 &&
        !add_u64(safe_start, page_size - start_remainder, &safe_start)) {
        return false;
    }
    const uint64_t safe_end = raw_end - raw_end % page_size;
    if (safe_start >= safe_end) {
        *out = plan;
        return true;
    }

    size_t first = 0;
    while (first < range_count) {
        uint64_t range_end = 0;
        if (!add_u64(ranges[first].offset, ranges[first].bytes,
                     &range_end)) {
            return false;
        }
        if (range_end >= safe_start) break;
        first++;
    }

    uint64_t merged_start = safe_start;
    uint64_t merged_end = safe_end;
    uint64_t replaced_bytes = 0;
    size_t after = first;
    while (after < range_count && ranges[after].offset <= merged_end) {
        uint64_t range_end = 0;
        if (!add_u64(ranges[after].offset, ranges[after].bytes,
                     &range_end) ||
            !add_u64(replaced_bytes, ranges[after].bytes,
                     &replaced_bytes)) {
            return false;
        }
        if (ranges[after].offset < merged_start) {
            merged_start = ranges[after].offset;
        }
        if (range_end > merged_end) merged_end = range_end;
        after++;
    }

    const size_t replaced_count = after - first;
    size_t resulting_count = 0;
    if (replaced_count == 0) {
        if (range_count >= range_capacity) return false;
        resulting_count = range_count + 1u;
    } else {
        resulting_count = range_count - replaced_count + 1u;
    }
    if (resulting_count > range_capacity) return false;

    const uint64_t merged_bytes = merged_end - merged_start;
    if (replaced_bytes > merged_bytes) return false;
    const size_t suffix_count = range_count - after;
    if (suffix_count > SIZE_MAX / sizeof(ranges[0])) return false;

    plan.first = first;
    plan.after = after;
    plan.resulting_count = resulting_count;
    plan.merged_start = merged_start;
    plan.merged_bytes = merged_bytes;
    plan.newly_unique_bytes = merged_bytes - replaced_bytes;
    plan.has_eligible_interval = true;
    *out = plan;
    return true;
}

bool ds4_laguna_page_range_union_preview(
        const ds4_laguna_page_range *ranges,
        size_t range_count,
        size_t range_capacity,
        uint64_t page_size,
        uint64_t raw_offset,
        uint64_t raw_bytes,
        size_t *resulting_range_count,
        uint64_t *newly_unique_bytes,
        uint64_t *newly_unique_pages) {
    if (!resulting_range_count || !newly_unique_bytes ||
        !newly_unique_pages) {
        return false;
    }
    ds4_laguna_page_range_insert_plan plan;
    if (!page_range_union_insert_plan(ranges, range_count, range_capacity,
                                      page_size, raw_offset, raw_bytes,
                                      &plan)) {
        return false;
    }
    *resulting_range_count = plan.resulting_count;
    *newly_unique_bytes = plan.newly_unique_bytes;
    *newly_unique_pages = plan.newly_unique_bytes / page_size;
    return true;
}

bool ds4_laguna_page_range_union_insert(
        ds4_laguna_page_range *ranges,
        size_t *range_count,
        size_t range_capacity,
        uint64_t page_size,
        uint64_t raw_offset,
        uint64_t raw_bytes,
        uint64_t *newly_unique_bytes,
        uint64_t *newly_unique_pages) {
    if (!range_count || !newly_unique_bytes || !newly_unique_pages) {
        return false;
    }
    ds4_laguna_page_range_insert_plan plan;
    if (!page_range_union_insert_plan(ranges, *range_count, range_capacity,
                                      page_size, raw_offset, raw_bytes,
                                      &plan)) {
        return false;
    }

    const size_t suffix_count = *range_count - plan.after;
    if (plan.has_eligible_interval && suffix_count != 0) {
        memmove(&ranges[plan.first + 1u], &ranges[plan.after],
                suffix_count * sizeof(ranges[0]));
    }
    if (plan.has_eligible_interval) {
        ranges[plan.first].offset = plan.merged_start;
        ranges[plan.first].bytes = plan.merged_bytes;
    }
    *range_count = plan.resulting_count;
    *newly_unique_bytes = plan.newly_unique_bytes;
    *newly_unique_pages = plan.newly_unique_bytes / page_size;
    return true;
}

static uint64_t saturating_add_u64(uint64_t a, uint64_t b) {
    return b > UINT64_MAX - a ? UINT64_MAX : a + b;
}

bool ds4_laguna_page_advice_note_touched(
        ds4_laguna_page_advice_counters *counters,
        uint64_t newly_touched_pages) {
    if (!counters || counters->errno_bucket_count >
                         DS4_LAGUNA_PAGE_ADVICE_ERRNO_BUCKET_CAPACITY) {
        return false;
    }
    counters->touched_eligible_unique_pages = saturating_add_u64(
        counters->touched_eligible_unique_pages, newly_touched_pages);
    return true;
}

bool ds4_laguna_page_advice_note_result(
        ds4_laguna_page_advice_counters *counters,
        uint64_t attempted_bytes,
        uint64_t newly_advised_pages,
        int error_number) {
    if (!counters || error_number < 0 ||
        (error_number != 0 && newly_advised_pages != 0) ||
        counters->errno_bucket_count >
            DS4_LAGUNA_PAGE_ADVICE_ERRNO_BUCKET_CAPACITY) {
        return false;
    }

    ds4_laguna_page_advice_counters next = *counters;
    size_t bucket_index = next.errno_bucket_count;
    if (error_number != 0) {
        for (size_t i = 0; i < next.errno_bucket_count; i++) {
            if (next.errno_buckets[i].error_number == error_number) {
                bucket_index = i;
                break;
            }
        }
        if (bucket_index == next.errno_bucket_count) {
            if (next.errno_bucket_count >=
                DS4_LAGUNA_PAGE_ADVICE_ERRNO_BUCKET_CAPACITY) {
                return false;
            }
            next.errno_buckets[bucket_index].error_number = error_number;
            next.errno_buckets[bucket_index].calls = 0;
            next.errno_buckets[bucket_index].bytes = 0;
            next.errno_bucket_count++;
        }
    }

    next.attempted_calls = saturating_add_u64(next.attempted_calls, 1u);
    next.attempted_bytes = saturating_add_u64(
        next.attempted_bytes, attempted_bytes);
    if (error_number == 0) {
        next.successful_calls = saturating_add_u64(
            next.successful_calls, 1u);
        next.successful_bytes = saturating_add_u64(
            next.successful_bytes, attempted_bytes);
        next.advised_unique_pages = saturating_add_u64(
            next.advised_unique_pages, newly_advised_pages);
    } else {
        next.failed_calls = saturating_add_u64(next.failed_calls, 1u);
        next.failed_bytes = saturating_add_u64(
            next.failed_bytes, attempted_bytes);
        ds4_laguna_page_advice_errno_bucket *bucket =
            &next.errno_buckets[bucket_index];
        bucket->calls = saturating_add_u64(bucket->calls, 1u);
        bucket->bytes = saturating_add_u64(bucket->bytes, attempted_bytes);
    }
    *counters = next;
    return true;
}

bool ds4_laguna_page_conservative_source_charge(
        uint64_t model_size_bytes,
        uint64_t prior_post_advice_resident_bytes,
        uint64_t touched_since_sample_unique_bytes,
        bool exact_sample_available,
        uint64_t exact_sample_bytes,
        uint64_t *charge_out) {
    if (!charge_out) return false;
    uint64_t charge = saturating_add_u64(
        prior_post_advice_resident_bytes,
        touched_since_sample_unique_bytes);
    if (exact_sample_available && exact_sample_bytes > charge) {
        charge = exact_sample_bytes;
    }
    if (charge > model_size_bytes) charge = model_size_bytes;
    *charge_out = charge;
    return true;
}
