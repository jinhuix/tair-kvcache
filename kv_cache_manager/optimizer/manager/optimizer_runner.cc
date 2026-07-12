#include "kv_cache_manager/optimizer/manager/optimizer_runner.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/eviction_policy/checkpoint_lru.h"
#include "kv_cache_manager/optimizer/manager/optimizer_loader.h"

namespace kv_cache_manager {
namespace {
int64_t TtlUsToNs(int64_t ttl_us) { return ttl_us > 0 ? ttl_us * 1000 : ttl_us; }

void ValidateSupportedQueryTypeOrThrow(const std::string &query_type) {
    if (!IsSupportedQueryType(query_type)) {
        throw std::runtime_error("Unsupported optimizer query_type: " + query_type);
    }
}

void MergeEvictedBlocks(OptIndexerManager::EvictedBlocks *dst, const OptIndexerManager::EvictedBlocks &src) {
    if (dst == nullptr) {
        return;
    }
    for (const auto &[instance_id, blocks] : src) {
        auto &merged = (*dst)[instance_id];
        merged.insert(merged.end(), blocks.begin(), blocks.end());
    }
}

uint64_t MixUint64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

size_t ContiguousHitPrefixLength(const QueryHit &query_hit, size_t key_count) {
    if (key_count == 0) {
        return 0;
    }
    std::vector<bool> hit_mask(key_count, false);
    for (const size_t idx : query_hit.local_hit_indices) {
        if (idx < hit_mask.size()) {
            hit_mask[idx] = true;
        }
    }
    for (const size_t idx : query_hit.remote_hit_indices) {
        if (idx < hit_mask.size()) {
            hit_mask[idx] = true;
        }
    }

    size_t prefix = 0;
    while (prefix < hit_mask.size() && hit_mask[prefix]) {
        ++prefix;
    }
    return prefix;
}

void KeepHitIndicesBefore(std::vector<size_t> *indices, size_t prefix_len) {
    if (indices == nullptr) {
        return;
    }
    indices->erase(
        std::remove_if(indices->begin(), indices->end(), [prefix_len](size_t idx) { return idx >= prefix_len; }),
        indices->end());
}

size_t ValidateFullBlockTrace(const GetLocationSchemaTrace &trace, size_t block_size) {
    if (block_size == 0) {
        throw std::runtime_error("GetCacheLocation requires positive instance block_size");
    }

    const size_t input_tokens = trace.input_token_count();
    const size_t max_full_blocks = input_tokens / block_size;
    if (trace.keys().size() <= max_full_blocks) {
        return input_tokens;
    }

    throw std::runtime_error(
        "GetCacheLocation trace contains partial tail block keys: instance_id=" + trace.instance_id() +
        ", trace_id=" + trace.trace_id() + ", keys=" + std::to_string(trace.keys().size()) +
        ", input_len=" + std::to_string(input_tokens) + ", block_size=" + std::to_string(block_size) +
        ", max_full_blocks=" + std::to_string(max_full_blocks) +
        ". Standard optimizer traces must drop incomplete tail blocks before replay.");
}
} // namespace

void OptimizerRunner::Run(OptimizerConfig &config, bool reset_runner_state) {
    write_delay_ns_ = config.trace_replay_config().write_delay_ns();
    mamba_state_config_ = config.mamba_state_config();
    if (write_delay_ns_ <= 0) {
        throw std::runtime_error("trace_replay.write_delay_ns must be positive");
    }

    auto starting_time = std::chrono::high_resolution_clock::now();
    auto traces = OptimizerLoader::LoadTrace(config);
    auto ending_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(ending_time - starting_time).count();
    KVCM_LOG_INFO(
        "Loaded %zu traces from file: %s in %ld ms", traces.size(), config.trace_file_path().c_str(), duration);

    starting_time = std::chrono::high_resolution_clock::now();
    RunTraces(traces, reset_runner_state);
    ending_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(ending_time - starting_time).count();
    KVCM_LOG_INFO("Playback traces in %ld ms", duration);
}

void OptimizerRunner::RunTraces(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces,
                                bool reset_runner_state) {
    ResetReplayState(reset_runner_state);
    for (const auto &trace : traces) {
        if (trace) {
            FlushPendingWritesThrough(trace->timestamp_ns());
        }
        RunTrace(trace);
    }
    FlushAllPendingWrites();
}

void OptimizerRunner::ResetReplayState(bool clear_mamba_state) {
    pending_writes_ = {};
    next_pending_write_sequence_ = 0;
    if (clear_mamba_state) {
        ClearAllMambaStates();
        next_mamba_state_sequence_ = 0;
        next_mamba_checkpoint_id_ = 1;
    }
}

void OptimizerRunner::RunTrace(std::shared_ptr<OptimizerSchemaTrace> trace) {
    if (!trace) {
        return;
    }

    if (auto request_trace = std::dynamic_pointer_cast<RequestSchemaTrace>(trace)) {
        HandleRequest(*request_trace);
    } else if (auto get_trace = std::dynamic_pointer_cast<GetLocationSchemaTrace>(trace)) {
        ValidateSupportedQueryTypeOrThrow(get_trace->query_type());
        HandleGetLocation(*get_trace);
        stats_collector_->UpdateTimestamp(get_trace->instance_id(), get_trace->timestamp_ns());
    } else if (auto write_trace = std::dynamic_pointer_cast<WriteCacheSchemaTrace>(trace)) {
        HandleWriteCache(*write_trace);
        stats_collector_->UpdateTimestamp(write_trace->instance_id(), write_trace->timestamp_ns());
    } else {
        KVCM_LOG_WARN("Unknown trace type, skipping");
    }
}

std::shared_ptr<RadixTreeIndex> OptimizerRunner::GetIndexer(const std::string &instance_id) {
    auto indexer = indexer_manager_->GetOptIndexer(instance_id);
    if (!indexer) {
        KVCM_LOG_ERROR("Optimizer indexer not found for instance_id: %s", instance_id.c_str());
    }
    return indexer;
}

void OptimizerRunner::HandleRequest(const RequestSchemaTrace &trace) {
    ValidateSupportedQueryTypeOrThrow(trace.query_type());
    ReadRecord read_record = HandleGetLocation(trace);
    stats_collector_->UpdateTimestamp(trace.instance_id(), trace.timestamp_ns());
    const size_t mamba_hit_blocks = mamba_state_config_.enabled() ? read_record.mamba_state_hit_blocks : 0;
    ScheduleRequestWrite(trace, mamba_hit_blocks);
}

void OptimizerRunner::ScheduleRequestWrite(const RequestSchemaTrace &trace, size_t mamba_hit_blocks) {
    if (trace.timestamp_ns() > std::numeric_limits<int64_t>::max() - write_delay_ns_) {
        throw std::runtime_error("request write timestamp overflows int64: instance_id=" + trace.instance_id() +
                                 ", trace_id=" + trace.trace_id());
    }

    WriteCacheSchemaTrace write_trace;
    write_trace.set_instance_id(trace.instance_id());
    write_trace.set_trace_id(trace.trace_id() + ":write");
    write_trace.set_timestamp_ns(trace.timestamp_ns() + write_delay_ns_);
    write_trace.set_keys(trace.keys());
    write_trace.set_ttl_us(trace.ttl_us());
    pending_writes_.push(PendingWrite{write_trace.timestamp_ns(),
                                      next_pending_write_sequence_++,
                                      std::move(write_trace),
                                      mamba_hit_blocks});
}

void OptimizerRunner::FlushPendingWritesThrough(int64_t timestamp_ns) {
    while (!pending_writes_.empty() && pending_writes_.top().timestamp_ns <= timestamp_ns) {
        auto pending = pending_writes_.top();
        pending_writes_.pop();
        RunPendingWrite(pending);
    }
}

void OptimizerRunner::FlushAllPendingWrites() {
    while (!pending_writes_.empty()) {
        auto pending = pending_writes_.top();
        pending_writes_.pop();
        RunPendingWrite(pending);
    }
}

void OptimizerRunner::RunPendingWrite(const PendingWrite &pending) {
    HandleCacheInsert(pending.trace, true, nullptr, pending.mamba_hit_blocks);
    stats_collector_->UpdateTimestamp(pending.trace.instance_id(), pending.trace.timestamp_ns());
}

ReadRecord OptimizerRunner::SubmitReadRecord(const std::string &instance_id,
                                             const std::string &trace_id,
                                             const std::vector<int64_t> &keys,
                                             int64_t timestamp_ns,
                                             const QueryHit &query_hit,
                                             const std::shared_ptr<RadixTreeIndex> &indexer,
                                             size_t local_read_block_num,
                                             size_t remote_read_block_num,
                                             size_t input_tokens,
                                             size_t block_size_tokens,
                                             size_t mamba_state_candidate_blocks,
                                             size_t mamba_state_hit_blocks) {
    ReadRecord record{};
    record.timestamp_ns = timestamp_ns;
    record.trace_id = trace_id;
    record.keys_ptr = &keys;
    record.current_cache_blocks = eviction_manager_->GetCurrentInstanceUsage(instance_id);

    auto indexer_map = indexer_manager_->GetAllOptIndexers();
    record.blocks_per_instance.resize(indexer_map.size(), 0);
    size_t idx = 0;
    for (const auto &pair : indexer_map) {
        record.blocks_per_instance[idx] = eviction_manager_->GetCurrentInstanceUsage(pair.first);
        idx++;
    }

    record.remote_hit_blocks = query_hit.remote_hit_block_num;
    record.local_hit_blocks = query_hit.local_hit_block_num;
    record.remote_hit_indices = query_hit.remote_hit_indices;
    record.local_hit_indices = query_hit.local_hit_indices;
    record.per_tier_hit_blocks = query_hit.per_tier_hit_block_num;
    record.input_tokens = input_tokens;
    record.block_size_tokens = block_size_tokens;
    record.tier_names = indexer->GetTierNames();
    record.per_tier_blocks = eviction_manager_->GetCurrentInstanceUsagePerTier(instance_id);
    record.local_read_blocks = local_read_block_num;
    record.remote_read_blocks = remote_read_block_num;
    record.mamba_state_candidate_blocks = mamba_state_candidate_blocks;
    record.mamba_state_hit_blocks = mamba_state_hit_blocks;

    stats_collector_->OnReadComplete(instance_id, record);
    return record;
}

ReadRecord OptimizerRunner::HandleGetLocation(const GetLocationSchemaTrace &trace,
                                              bool touch_local_hits,
                                              bool local_hits_are_reads) {
    ReadRecord record{};
    std::string instance_id = trace.instance_id();
    auto indexer = GetIndexer(instance_id);
    if (!indexer) {
        return record;
    }

    const size_t block_size = indexer_manager_->GetInstanceBlockSize(instance_id);
    const size_t input_tokens = ValidateFullBlockTrace(trace, block_size);

    auto pending_evicted_blocks = indexer_manager_->EvictExpiredBeforeAccess(instance_id, trace.timestamp_ns());
    HandleMambaStateEvictions(&pending_evicted_blocks);

    bool refresh_ttl_on_read = true;
    auto it = instance_ttl_refresh_on_read_.find(instance_id);
    if (it != instance_ttl_refresh_on_read_.end()) {
        refresh_ttl_on_read = it->second;
    }

    const bool defer_full_touch_until_mamba_hit = UsesSharedMambaCapacity();
    QueryHit query_hit;
    if (IsPrefixMatchQueryType(trace.query_type())) {
        indexer->PrefixQuery(trace.keys(),
                             trace.block_mask(),
                             trace.timestamp_ns(),
                             &query_hit,
                             refresh_ttl_on_read,
                             touch_local_hits,
                             local_hits_are_reads,
                             !defer_full_touch_until_mamba_hit);
    } else if (IsBatchGetQueryType(trace.query_type())) {
        indexer->BatchQuery(trace.keys(),
                            trace.block_mask(),
                            trace.timestamp_ns(),
                            &query_hit,
                            refresh_ttl_on_read,
                            touch_local_hits,
                            local_hits_are_reads,
                            !defer_full_touch_until_mamba_hit);
    } else {
        ValidateSupportedQueryTypeOrThrow(trace.query_type());
    }
    indexer->ConsumeTierFlow();
    if (indexer->ConsumeReadTriggeredTierWrite()) {
        auto capacity_eviction = indexer_manager_->CheckAndEvict(instance_id, trace.timestamp_ns());
        HandleMambaStateEvictions(&capacity_eviction.evicted_blocks);
        MergeEvictedBlocks(&pending_evicted_blocks, capacity_eviction.evicted_blocks);
    }

    size_t local_read_block_num = 0;
    size_t remote_read_block_num = trace.keys().size();
    size_t local_mask_block_num = 0;
    if (std::holds_alternative<BlockMaskVector>(trace.block_mask())) {
        const auto &mask_vector = std::get<BlockMaskVector>(trace.block_mask());
        const size_t n = std::min(mask_vector.size(), trace.keys().size());
        local_mask_block_num = std::count(mask_vector.begin(), mask_vector.begin() + n, true);
    } else if (std::holds_alternative<BlockMaskOffset>(trace.block_mask())) {
        local_mask_block_num = std::min(std::get<BlockMaskOffset>(trace.block_mask()), trace.keys().size());
    }
    local_read_block_num = local_hits_are_reads ? local_mask_block_num : 0;
    remote_read_block_num = trace.keys().size() - local_mask_block_num;
    const auto [mamba_state_candidate_blocks, mamba_state_hit_blocks] =
        ApplyMambaStateRead(instance_id, trace.keys(), trace.timestamp_ns(), &query_hit);
    if (defer_full_touch_until_mamba_hit && mamba_state_hit_blocks > 0) {
        std::vector<int64_t> touched_keys(trace.keys().begin(), trace.keys().begin() + mamba_state_hit_blocks);
        indexer->TouchKeysAtTier(touched_keys, "shared", trace.timestamp_ns(), refresh_ttl_on_read);
        indexer->ConsumeTierFlow();
    }

    record = SubmitReadRecord(instance_id,
                              trace.trace_id(),
                              trace.keys(),
                              trace.timestamp_ns(),
                              query_hit,
                              indexer,
                              local_read_block_num,
                              remote_read_block_num,
                              input_tokens,
                              block_size,
                              mamba_state_candidate_blocks,
                              mamba_state_hit_blocks);
    indexer_manager_->CleanEvictedBlocks(pending_evicted_blocks, trace.timestamp_ns(), true);
    return record;
}

WriteRecord OptimizerRunner::HandleWriteCache(const WriteCacheSchemaTrace &trace) {
    return HandleCacheInsert(trace, true, nullptr);
}

WriteRecord OptimizerRunner::HandleFillCachePath(const WriteCacheSchemaTrace &trace,
                                                 const std::vector<size_t> &materialized_indices) {
    return HandleCacheInsert(trace, false, &materialized_indices);
}

void OptimizerRunner::ClearMambaState(const std::string &instance_id) {
    auto cache_it = mamba_state_checkpoints_.find(instance_id);
    if (cache_it != mamba_state_checkpoints_.end()) {
        for (auto &[_, record] : cache_it->second) {
            for (const auto &object : record.objects) {
                if (object != nullptr) {
                    mamba_state_object_index_.erase(object.get());
                    eviction_manager_->UnregisterExternalBlock(instance_id, object.get());
                }
            }
        }
    }
    mamba_state_checkpoints_.erase(instance_id);
    mamba_branch_prefix_history_.erase(instance_id);
}

void OptimizerRunner::ClearAllMambaStates() {
    std::vector<std::string> instance_ids;
    instance_ids.reserve(mamba_state_checkpoints_.size());
    for (const auto &[instance_id, _] : mamba_state_checkpoints_) {
        instance_ids.push_back(instance_id);
    }
    for (const auto &instance_id : instance_ids) {
        ClearMambaState(instance_id);
    }
    mamba_branch_prefix_history_.clear();
    mamba_state_object_index_.clear();
}

std::vector<OptimizerRunner::PrefixSignature>
OptimizerRunner::BuildPrefixSignatures(const std::vector<int64_t> &keys) const {
    std::vector<PrefixSignature> signatures(keys.size() + 1);
    uint64_t hash1 = 1469598103934665603ULL;
    uint64_t hash2 = 1099511628211ULL;
    for (size_t idx = 0; idx < keys.size(); ++idx) {
        const uint64_t mixed_key = MixUint64(static_cast<uint64_t>(keys[idx]));
        hash1 ^= mixed_key;
        hash1 *= 1099511628211ULL;
        hash2 ^= mixed_key + 0x9e3779b97f4a7c15ULL + (hash2 << 6) + (hash2 >> 2);
        hash2 = MixUint64(hash2);
        signatures[idx + 1] = PrefixSignature{idx + 1, hash1, hash2};
    }
    return signatures;
}

std::vector<size_t> OptimizerRunner::ChunkMambaCheckpointIndices(size_t key_count) const {
    std::vector<size_t> indices;
    if (!mamba_state_config_.enabled() || key_count == 0 || mamba_state_config_.chunk_size_blocks() == 0) {
        return indices;
    }

    for (size_t next = mamba_state_config_.chunk_size_blocks(); next <= key_count;
         next += mamba_state_config_.chunk_size_blocks()) {
        indices.push_back(next - 1);
    }
    const size_t request_end = key_count - 1;
    if (indices.empty() || indices.back() != request_end) {
        indices.push_back(request_end);
    }
    return indices;
}

std::vector<size_t> OptimizerRunner::BranchMambaCheckpointIndices(
    const std::string &instance_id,
    const std::vector<PrefixSignature> &prefix_signatures) const {
    std::vector<size_t> indices;
    if (!mamba_state_config_.enabled() || prefix_signatures.size() <= 1) {
        return indices;
    }
    const size_t request_end = prefix_signatures.size() - 2;
    const auto history_it = mamba_branch_prefix_history_.find(instance_id);
    if (history_it == mamba_branch_prefix_history_.end() || history_it->second.empty()) {
        if (mamba_state_config_.branch_save_request_end_checkpoint()) {
            indices.push_back(request_end);
        }
        return indices;
    }

    const auto &history = history_it->second;
    for (size_t prefix_len = prefix_signatures.size() - 1; prefix_len > 0; --prefix_len) {
        if (history.find(prefix_signatures[prefix_len]) != history.end()) {
            indices.push_back(prefix_len - 1);
            break;
        }
    }
    if (mamba_state_config_.branch_save_request_end_checkpoint() &&
        (indices.empty() || indices.back() != request_end)) {
        indices.push_back(request_end);
    }
    return indices;
}

std::vector<size_t> OptimizerRunner::SelectMambaCheckpointIndices(
    const std::string &instance_id,
    size_t key_count,
    const std::vector<PrefixSignature> &prefix_signatures) const {
    if (mamba_state_config_.checkpoint_strategy() == MambaCheckpointStrategy::BRANCH) {
        return BranchMambaCheckpointIndices(instance_id, prefix_signatures);
    }
    return ChunkMambaCheckpointIndices(key_count);
}

bool OptimizerRunner::UsesSharedMambaCapacity() const {
    return mamba_state_config_.enabled();
}

size_t OptimizerRunner::MambaCheckpointObjectCount(const MambaCheckpointRecord &record) const {
    size_t count = 0;
    for (const auto &object : record.objects) {
        if (object != nullptr) {
            ++count;
        }
    }
    return count;
}

bool OptimizerRunner::MambaCheckpointIsResident(const MambaCheckpointRecord &record) const {
    if (!UsesSharedMambaCapacity()) {
        return true;
    }
    if (record.objects.size() < mamba_state_config_.group_count()) {
        return false;
    }
    for (size_t slot = 0; slot < mamba_state_config_.group_count(); ++slot) {
        if (record.objects[slot] == nullptr) {
            return false;
        }
    }
    return true;
}

CheckpointLruEvictionPolicy *OptimizerRunner::GetCheckpointLruPolicy(const std::string &instance_id) const {
    auto policy = eviction_manager_->GetSharedPolicy(instance_id);
    return dynamic_cast<CheckpointLruEvictionPolicy *>(policy.get());
}

bool OptimizerRunner::RegisterCheckpointWithEvictionPolicy(const std::string &instance_id,
                                                           const std::vector<int64_t> &keys,
                                                           size_t checkpoint_index,
                                                           MambaCheckpointRecord *record,
                                                           int64_t timestamp_ns) {
    auto *policy = GetCheckpointLruPolicy(instance_id);
    if (policy == nullptr) {
        return true;
    }
    if (record == nullptr || checkpoint_index >= keys.size()) {
        return false;
    }
    if (record->eviction_id == 0) {
        record->eviction_id = next_mamba_checkpoint_id_++;
    }
    if (policy->HasCheckpoint(record->eviction_id)) {
        return true;
    }

    auto indexer = GetIndexer(instance_id);
    if (!indexer) {
        return false;
    }
    BlockEntry *boundary = indexer->FindPathBlock(keys, checkpoint_index);
    std::vector<BlockEntry *> objects;
    objects.reserve(record->objects.size());
    for (const auto &object : record->objects) {
        if (object != nullptr) {
            objects.push_back(object.get());
        }
    }
    return policy->RegisterCheckpoint(record->eviction_id, boundary, objects, timestamp_ns);
}

bool OptimizerRunner::RegisterMambaStateObject(const std::string &instance_id,
                                               const PrefixSignature &signature,
                                               size_t group_slot,
                                               MambaCheckpointRecord *record,
                                               int64_t timestamp_ns) {
    if (record == nullptr) {
        return false;
    }
    const size_t group_count = mamba_state_config_.group_count();
    if (group_slot >= group_count) {
        return false;
    }
    if (record->objects.size() < group_count) {
        record->objects.resize(group_count);
    }
    if (record->objects[group_slot] != nullptr) {
        auto *existing = record->objects[group_slot].get();
        if (existing->location_map.find("shared") != existing->location_map.end()) {
            return false;
        }
        mamba_state_object_index_.erase(existing);
        record->objects[group_slot].reset();
    }
    const size_t group_id = group_slot + 1; // full-attention uses group id 0; Mamba groups start at 1.
    auto object = std::make_unique<BlockEntry>();
    const uint64_t mixed = MixUint64(signature.hash1) ^ MixUint64(signature.hash2) ^
                           MixUint64(static_cast<uint64_t>(signature.length)) ^ MixUint64(group_id);
    object->key = static_cast<int64_t>(mixed);
    object->writing_time = timestamp_ns;
    object->last_access_time = -1;
    object->ttl_anchor_time = timestamp_ns;
    object->location_map["shared"] = TierStat{0, -1, timestamp_ns, 1};
    BlockEntry *ptr = object.get();
    record->objects[group_slot] = std::move(object);
    mamba_state_object_index_[ptr] = MambaObjectRef{instance_id, signature, group_id};
    eviction_manager_->RegisterExternalBlock(instance_id, ptr);
    return true;
}

void OptimizerRunner::TouchMambaCheckpointObjects(const std::string &instance_id,
                                                  MambaCheckpointRecord *record,
                                                  int64_t timestamp_ns) {
    if (record == nullptr || !UsesSharedMambaCapacity()) {
        return;
    }
    for (const auto &object : record->objects) {
        if (object != nullptr) {
            eviction_manager_->TouchExternalBlock(instance_id, object.get(), timestamp_ns, true);
        }
    }
}

bool OptimizerRunner::InstanceTtlRefreshOnRead(const std::string &instance_id) const {
    auto it = instance_ttl_refresh_on_read_.find(instance_id);
    if (it != instance_ttl_refresh_on_read_.end()) {
        return it->second;
    }
    return true;
}

void OptimizerRunner::TouchMambaBranchPrefixOnAdmission(const std::string &instance_id,
                                                        const std::vector<int64_t> &keys,
                                                        size_t prefix_blocks,
                                                        int64_t timestamp_ns) {
    if (prefix_blocks == 0 || keys.empty()) {
        return;
    }
    auto indexer = GetIndexer(instance_id);
    if (!indexer) {
        return;
    }
    const auto &tier_names = indexer->GetTierNames();
    if (std::find(tier_names.begin(), tier_names.end(), "shared") == tier_names.end()) {
        return;
    }
    const size_t touch_count = std::min(prefix_blocks, keys.size());
    std::vector<int64_t> touched_keys(keys.begin(), keys.begin() + touch_count);
    indexer->TouchKeysAtTier(touched_keys, "shared", timestamp_ns, InstanceTtlRefreshOnRead(instance_id));
    indexer->ConsumeTierFlow();
}

size_t OptimizerRunner::CountMaterializedBlocks(size_t key_count,
                                                const std::vector<size_t> *materialized_indices) const {
    if (materialized_indices == nullptr) {
        return key_count;
    }
    std::vector<bool> selected(key_count, false);
    for (const size_t idx : *materialized_indices) {
        if (idx < selected.size()) {
            selected[idx] = true;
        }
    }
    return static_cast<size_t>(std::count(selected.begin(), selected.end(), true));
}

size_t OptimizerRunner::EstimateMambaStateAdmissionObjects(
    const std::string &instance_id,
    const std::vector<int64_t> &keys,
    const std::vector<size_t> *materialized_indices,
    size_t min_checkpoint_prefix_blocks) const {
    if (!mamba_state_config_.enabled() || !UsesSharedMambaCapacity() || keys.empty()) {
        return 0;
    }

    const auto signatures = BuildPrefixSignatures(keys);
    const auto checkpoint_indices = SelectMambaCheckpointIndices(instance_id, keys.size(), signatures);
    if (checkpoint_indices.empty()) {
        return 0;
    }

    std::vector<bool> allowed(keys.size(), true);
    if (materialized_indices != nullptr) {
        std::fill(allowed.begin(), allowed.end(), false);
        for (const size_t idx : *materialized_indices) {
            if (idx < allowed.size()) {
                allowed[idx] = true;
            }
        }
    }

    size_t object_count = 0;
    const auto cache_it = mamba_state_checkpoints_.find(instance_id);
    for (const size_t idx : checkpoint_indices) {
        if (idx >= allowed.size() || !allowed[idx]) {
            continue;
        }
        const size_t checkpoint_prefix_blocks = idx + 1;
        if (checkpoint_prefix_blocks <= min_checkpoint_prefix_blocks) {
            continue;
        }
        size_t resident_objects = 0;
        if (cache_it != mamba_state_checkpoints_.end()) {
            const auto checkpoint_it = cache_it->second.find(signatures[idx + 1]);
            if (checkpoint_it != cache_it->second.end()) {
                resident_objects = MambaCheckpointObjectCount(checkpoint_it->second);
            }
        }
        const size_t group_count = mamba_state_config_.group_count();
        if (resident_objects < group_count) {
            object_count += group_count - resident_objects;
        }
    }
    return object_count;
}

size_t OptimizerRunner::MambaBranchPrefixAdmissionTouchBlocks(
    const std::string &instance_id,
    const std::vector<int64_t> &keys,
    const std::vector<size_t> *materialized_indices,
    size_t min_checkpoint_prefix_blocks) const {
    if (!mamba_state_config_.enabled() || !UsesSharedMambaCapacity() || keys.empty() ||
        mamba_state_config_.checkpoint_strategy() != MambaCheckpointStrategy::BRANCH) {
        return 0;
    }

    const auto signatures = BuildPrefixSignatures(keys);
    const auto checkpoint_indices = SelectMambaCheckpointIndices(instance_id, keys.size(), signatures);
    const auto history_it = mamba_branch_prefix_history_.find(instance_id);
    if (checkpoint_indices.empty() || history_it == mamba_branch_prefix_history_.end()) {
        return 0;
    }

    std::vector<bool> allowed(keys.size(), true);
    if (materialized_indices != nullptr) {
        std::fill(allowed.begin(), allowed.end(), false);
        for (const size_t idx : *materialized_indices) {
            if (idx < allowed.size()) {
                allowed[idx] = true;
            }
        }
    }

    size_t touch_blocks = 0;
    const auto cache_it = mamba_state_checkpoints_.find(instance_id);
    for (const size_t idx : checkpoint_indices) {
        if (idx >= allowed.size() || !allowed[idx]) {
            continue;
        }
        const size_t checkpoint_prefix_blocks = idx + 1;
        if (checkpoint_prefix_blocks <= min_checkpoint_prefix_blocks ||
            history_it->second.count(signatures[idx + 1]) == 0) {
            continue;
        }
        size_t resident_objects = 0;
        if (cache_it != mamba_state_checkpoints_.end()) {
            const auto checkpoint_it = cache_it->second.find(signatures[idx + 1]);
            if (checkpoint_it != cache_it->second.end()) {
                resident_objects = MambaCheckpointObjectCount(checkpoint_it->second);
            }
        }
        if (resident_objects < mamba_state_config_.group_count()) {
            touch_blocks = std::max(touch_blocks, checkpoint_prefix_blocks);
        }
    }
    return touch_blocks;
}

size_t OptimizerRunner::HandleMambaStateEvictions(OptIndexerManager::EvictedBlocks *evicted_blocks) {
    size_t evicted = 0;
    if (evicted_blocks == nullptr) {
        return evicted;
    }
    for (auto &[_, blocks] : *evicted_blocks) {
        auto out_it = blocks.begin();
        for (auto *block : blocks) {
            auto object_it = mamba_state_object_index_.find(block);
            if (object_it == mamba_state_object_index_.end()) {
                *out_it++ = block;
                continue;
            }
            const auto ref = object_it->second;
            mamba_state_object_index_.erase(object_it);
            auto cache_it = mamba_state_checkpoints_.find(ref.instance_id);
            if (cache_it == mamba_state_checkpoints_.end()) {
                continue;
            }
            auto record_it = cache_it->second.find(ref.signature);
            if (record_it == cache_it->second.end()) {
                continue;
            }
            auto &objects = record_it->second.objects;
            bool cleared = false;
            if (ref.group_id > 0) {
                const size_t slot = ref.group_id - 1;
                if (slot < objects.size() && objects[slot].get() == block) {
                    objects[slot].reset();
                    cleared = true;
                }
            }
            if (!cleared) {
                for (auto &object : objects) {
                    if (object.get() == block) {
                        object.reset();
                        break;
                    }
                }
            }
            if (MambaCheckpointObjectCount(record_it->second) == 0) {
                cache_it->second.erase(record_it);
            }
            ++evicted;
        }
        blocks.erase(out_it, blocks.end());
    }
    return evicted;
}

void OptimizerRunner::ObserveMambaBranchPrefixes(const std::string &instance_id,
                                                 const std::vector<PrefixSignature> &prefix_signatures) {
    if (mamba_state_config_.checkpoint_strategy() != MambaCheckpointStrategy::BRANCH || prefix_signatures.size() <= 1) {
        return;
    }

    auto &history = mamba_branch_prefix_history_[instance_id];
    for (size_t prefix_len = 1; prefix_len < prefix_signatures.size(); ++prefix_len) {
        history.insert(prefix_signatures[prefix_len]);
    }
}

std::pair<size_t, size_t> OptimizerRunner::ApplyMambaStateRead(const std::string &instance_id,
                                                               const std::vector<int64_t> &keys,
                                                               int64_t timestamp_ns,
                                                               QueryHit *query_hit) {
    if (!mamba_state_config_.enabled() || query_hit == nullptr) {
        return {0, 0};
    }

    auto &checkpoints = mamba_state_checkpoints_[instance_id];

    const size_t candidate_prefix = ContiguousHitPrefixLength(*query_hit, keys.size());
    if (candidate_prefix == 0 || checkpoints.empty()) {
        query_hit->remote_hit_block_num = 0;
        query_hit->local_hit_block_num = 0;
        query_hit->remote_hit_indices.clear();
        query_hit->local_hit_indices.clear();
        std::fill(query_hit->per_tier_hit_block_num.begin(), query_hit->per_tier_hit_block_num.end(), 0);
        return {candidate_prefix, 0};
    }

    const auto signatures = BuildPrefixSignatures(keys);
    size_t hit_blocks = 0;
    for (size_t prefix_len = candidate_prefix; prefix_len > 0; --prefix_len) {
        auto checkpoint_it = checkpoints.find(signatures[prefix_len]);
        if (checkpoint_it != checkpoints.end() && MambaCheckpointIsResident(checkpoint_it->second)) {
            hit_blocks = prefix_len;
            checkpoint_it->second.last_access_ns = timestamp_ns;
            checkpoint_it->second.sequence = next_mamba_state_sequence_++;
            if (auto *policy = GetCheckpointLruPolicy(instance_id); policy != nullptr) {
                policy->TouchCheckpoint(checkpoint_it->second.eviction_id, timestamp_ns);
            }
            TouchMambaCheckpointObjects(instance_id, &checkpoint_it->second, timestamp_ns);
            break;
        }
    }

    KeepHitIndicesBefore(&query_hit->local_hit_indices, hit_blocks);
    KeepHitIndicesBefore(&query_hit->remote_hit_indices, hit_blocks);
    query_hit->local_hit_block_num = query_hit->local_hit_indices.size();
    query_hit->remote_hit_block_num = query_hit->remote_hit_indices.size();

    size_t remaining_tier_hits = query_hit->local_hit_block_num + query_hit->remote_hit_block_num;
    for (auto &tier_hits : query_hit->per_tier_hit_block_num) {
        const size_t kept = std::min(tier_hits, remaining_tier_hits);
        tier_hits = kept;
        remaining_tier_hits -= kept;
    }
    return {candidate_prefix, hit_blocks};
}

void OptimizerRunner::ApplyMambaStateWrite(const std::string &instance_id,
                                           const std::vector<int64_t> &keys,
                                           int64_t timestamp_ns,
                                           const std::vector<size_t> *materialized_indices,
                                           size_t min_checkpoint_prefix_blocks) {
    if (!mamba_state_config_.enabled() || keys.empty()) {
        return;
    }

    const auto signatures = BuildPrefixSignatures(keys);
    const auto checkpoint_indices = SelectMambaCheckpointIndices(instance_id, keys.size(), signatures);
    if (checkpoint_indices.empty()) {
        ObserveMambaBranchPrefixes(instance_id, signatures);
        return;
    }
    const auto history_it = mamba_branch_prefix_history_.find(instance_id);
    const auto is_historical_branch_prefix = [&](const PrefixSignature &signature) {
        return mamba_state_config_.checkpoint_strategy() == MambaCheckpointStrategy::BRANCH &&
               history_it != mamba_branch_prefix_history_.end() && history_it->second.count(signature) > 0;
    };

    std::vector<bool> allowed(keys.size(), true);
    if (materialized_indices != nullptr) {
        std::fill(allowed.begin(), allowed.end(), false);
        for (const size_t idx : *materialized_indices) {
            if (idx < allowed.size()) {
                allowed[idx] = true;
            }
        }
    }

    auto &checkpoints = mamba_state_checkpoints_[instance_id];
    for (const size_t idx : checkpoint_indices) {
        if (idx >= allowed.size() || !allowed[idx]) {
            continue;
        }
        const size_t checkpoint_prefix_blocks = idx + 1;
        if (checkpoint_prefix_blocks <= min_checkpoint_prefix_blocks) {
            continue;
        }
        auto checkpoint_it = checkpoints.find(signatures[idx + 1]);
        if (checkpoint_it == checkpoints.end()) {
            checkpoint_it =
                checkpoints.emplace(signatures[idx + 1], MambaCheckpointRecord{timestamp_ns, next_mamba_state_sequence_++})
                    .first;
            checkpoint_it->second.eviction_id = next_mamba_checkpoint_id_++;
        }

        if (UsesSharedMambaCapacity()) {
            bool registered_new_object = false;
            for (size_t slot = 0; slot < mamba_state_config_.group_count(); ++slot) {
                registered_new_object |=
                    RegisterMambaStateObject(instance_id, signatures[idx + 1], slot, &checkpoint_it->second, timestamp_ns);
            }
            checkpoint_it->second.last_access_ns = timestamp_ns;
            checkpoint_it->second.sequence = next_mamba_state_sequence_++;
            if (!RegisterCheckpointWithEvictionPolicy(
                    instance_id, keys, idx, &checkpoint_it->second, timestamp_ns)) {
                throw std::runtime_error("failed to register complete Mamba checkpoint with checkpoint_lru");
            }
            if (registered_new_object && is_historical_branch_prefix(signatures[idx + 1])) {
                TouchMambaCheckpointObjects(instance_id, &checkpoint_it->second, timestamp_ns);
                TouchMambaBranchPrefixOnAdmission(instance_id, keys, checkpoint_prefix_blocks, timestamp_ns);
            }
        } else {
            checkpoint_it->second.last_access_ns = timestamp_ns;
            checkpoint_it->second.sequence = next_mamba_state_sequence_++;
        }
    }
    ObserveMambaBranchPrefixes(instance_id, signatures);
    EvictMambaStateIfNeeded(instance_id);
}

size_t OptimizerRunner::EvictMambaStateIfNeeded(const std::string &instance_id) {
    const size_t limit = mamba_state_config_.max_resident_checkpoints();
    if (limit == 0) {
        return 0;
    }
    auto cache_it = mamba_state_checkpoints_.find(instance_id);
    if (cache_it == mamba_state_checkpoints_.end()) {
        return 0;
    }
    auto &checkpoints = cache_it->second;
    size_t evicted = 0;
    while (checkpoints.size() > limit) {
        auto oldest = checkpoints.end();
        for (auto it = checkpoints.begin(); it != checkpoints.end(); ++it) {
            if (oldest == checkpoints.end() ||
                std::tie(it->second.last_access_ns, it->second.sequence) <
                    std::tie(oldest->second.last_access_ns, oldest->second.sequence)) {
                oldest = it;
            }
        }
        if (oldest == checkpoints.end()) {
            break;
        }
        if (UsesSharedMambaCapacity()) {
            for (const auto &object : oldest->second.objects) {
                if (object != nullptr) {
                    mamba_state_object_index_.erase(object.get());
                    eviction_manager_->UnregisterExternalBlock(instance_id, object.get());
                }
            }
        }
        checkpoints.erase(oldest);
        ++evicted;
    }
    return evicted;
}

WriteRecord OptimizerRunner::HandleCacheInsert(const WriteCacheSchemaTrace &trace,
                                               bool count_new_tier_write_touch,
                                               const std::vector<size_t> *materialized_indices,
                                               size_t mamba_hit_blocks) {
    WriteRecord record;
    record.timestamp_ns = trace.timestamp_ns();
    record.trace_id = trace.trace_id();

    std::string instance_id = trace.instance_id();
    auto indexer = GetIndexer(instance_id);
    if (!indexer) {
        return record;
    }

    auto pending_evicted_blocks = indexer_manager_->EvictExpiredBeforeAccess(instance_id, trace.timestamp_ns());
    HandleMambaStateEvictions(&pending_evicted_blocks);

    int64_t effective_ttl_ns = TtlUsToNs(trace.ttl_us());
    auto ttl_disabled_it = instance_group_ttl_disabled_.find(instance_id);
    if (ttl_disabled_it != instance_group_ttl_disabled_.end() && ttl_disabled_it->second) {
        effective_ttl_ns = -1;
    }

    std::vector<size_t> mamba_checkpoint_admission_indices;
    const std::vector<size_t> *effective_materialized_indices = materialized_indices;
    if (count_new_tier_write_touch && materialized_indices == nullptr && mamba_state_config_.enabled()) {
        const size_t first_materialized_block = mamba_hit_blocks;
        for (size_t idx = first_materialized_block; idx < trace.keys().size(); ++idx) {
            mamba_checkpoint_admission_indices.push_back(idx);
        }
        effective_materialized_indices = &mamba_checkpoint_admission_indices;
    }
    const std::vector<size_t> *requested_materialized_indices = effective_materialized_indices;

    const size_t branch_prefix_touch_blocks =
        MambaBranchPrefixAdmissionTouchBlocks(instance_id, trace.keys(), effective_materialized_indices, mamba_hit_blocks);
    if (branch_prefix_touch_blocks > 0) {
        TouchMambaBranchPrefixOnAdmission(instance_id, trace.keys(), branch_prefix_touch_blocks, trace.timestamp_ns());
    }

    auto *checkpoint_policy = GetCheckpointLruPolicy(instance_id);
    std::vector<size_t> checkpoint_lru_admission_indices;
    const auto refresh_checkpoint_lru_admission_indices = [&]() -> size_t {
        if (checkpoint_policy == nullptr) {
            return 0;
        }

        std::vector<bool> selected(trace.keys().size(), requested_materialized_indices == nullptr);
        size_t requested_count = requested_materialized_indices == nullptr ? trace.keys().size() : 0;
        if (requested_materialized_indices != nullptr) {
            for (const size_t idx : *requested_materialized_indices) {
                if (idx < selected.size() && !selected[idx]) {
                    selected[idx] = true;
                    ++requested_count;
                }
            }
        }

        size_t added_missing_path_blocks = 0;
        for (size_t idx = 0; idx < trace.keys().size(); ++idx) {
            BlockEntry *block = indexer->FindPathBlock(trace.keys(), idx);
            if (block == nullptr || block->location_map.find("shared") == block->location_map.end()) {
                if (!selected[idx]) {
                    selected[idx] = true;
                    ++added_missing_path_blocks;
                }
            }
        }

        checkpoint_lru_admission_indices.clear();
        checkpoint_lru_admission_indices.reserve(trace.keys().size());
        for (size_t idx = 0; idx < selected.size(); ++idx) {
            if (selected[idx]) {
                checkpoint_lru_admission_indices.push_back(idx);
            }
        }
        effective_materialized_indices = &checkpoint_lru_admission_indices;

        if (added_missing_path_blocks > 0) {
            KVCM_LOG_WARN("checkpoint_lru admission added missing path blocks instance_id=%s trace_id=%s "
                          "requested_blocks=%zu added_missing_path_blocks=%zu final_materialized_blocks=%zu "
                          "mamba_hit_blocks=%zu total_blocks=%zu",
                          instance_id.c_str(), trace.trace_id().c_str(), requested_count, added_missing_path_blocks,
                          checkpoint_lru_admission_indices.size(), mamba_hit_blocks, trace.keys().size());
        }
        return added_missing_path_blocks;
    };
    refresh_checkpoint_lru_admission_indices();
    const auto compute_admission_blocks = [&]() {
        const size_t full_blocks = checkpoint_policy == nullptr
                                       ? CountMaterializedBlocks(trace.keys().size(), effective_materialized_indices)
                                       : indexer->CountMissingPathBlocks(
                                             trace.keys(), effective_materialized_indices, "shared");
        return full_blocks + EstimateMambaStateAdmissionObjects(
                                 instance_id, trace.keys(), effective_materialized_indices, mamba_hit_blocks);
    };

    size_t admission_reserved_blocks = compute_admission_blocks();
    bool admission_fits = false;
    while (true) {
        const size_t size_before = checkpoint_policy == nullptr ? 0 : checkpoint_policy->size();
        auto admission_eviction =
            indexer_manager_->CheckAndEvictForAdmission(instance_id, admission_reserved_blocks, trace.timestamp_ns());
        HandleMambaStateEvictions(&admission_eviction.evicted_blocks);
        MergeEvictedBlocks(&pending_evicted_blocks, admission_eviction.evicted_blocks);
        indexer_manager_->CleanEvictedBlocks(pending_evicted_blocks, trace.timestamp_ns(), true);
        pending_evicted_blocks.clear();
        refresh_checkpoint_lru_admission_indices();

        if (checkpoint_policy == nullptr) {
            admission_fits = true; // Preserve the behavior of existing policies.
            break;
        }
        // Atomic eviction may remove a checkpoint that supplied blocks on the
        // incoming path. Recompute the exact missing footprint and repeat until
        // either it fits or no further resident object can be reclaimed.
        admission_reserved_blocks = compute_admission_blocks();
        if (indexer_manager_->CanFitAdmission(instance_id, admission_reserved_blocks)) {
            admission_fits = true;
            break;
        }
        if (checkpoint_policy->size() >= size_before) {
            break;
        }
    }

    size_t write_blocks = trace.keys().size();
    if (effective_materialized_indices != nullptr) {
        std::vector<bool> selected(trace.keys().size(), false);
        for (const size_t idx : *effective_materialized_indices) {
            if (idx < selected.size()) {
                selected[idx] = true;
            }
        }
        write_blocks = std::count(selected.begin(), selected.end(), true);
    }
    record.write_blocks = write_blocks;

    if (!admission_fits) {
        // A checkpoint is indivisible. If its complete incremental footprint
        // cannot fit even after reclaiming everything, reject this cache write
        // instead of creating an over-capacity or partially resident checkpoint.
        ObserveMambaBranchPrefixes(instance_id, BuildPrefixSignatures(trace.keys()));
        if (count_new_tier_write_touch) {
            stats_collector_->OnWriteComplete(instance_id, record);
        }
        return record;
    }

    RadixTreeIndex::InsertResult result;
    if (count_new_tier_write_touch && effective_materialized_indices == nullptr) {
        result = indexer->InsertOnly(trace.keys(), trace.timestamp_ns(), effective_ttl_ns);
    } else if (effective_materialized_indices != nullptr) {
        result =
            indexer->FillPathOnly(trace.keys(), *effective_materialized_indices, trace.timestamp_ns(), effective_ttl_ns);
    } else {
        throw std::runtime_error("HandleCacheInsert fill requires materialized indices");
    }
    record.newly_inserted_blocks = result.inserted_keys.size();
    ApplyMambaStateWrite(instance_id, trace.keys(), trace.timestamp_ns(), nullptr, mamba_hit_blocks);
    if (count_new_tier_write_touch) {
        stats_collector_->OnWriteComplete(instance_id, record);
    }
    indexer_manager_->CleanEvictedBlocks(pending_evicted_blocks, trace.timestamp_ns(), true);
    return record;
}
} // namespace kv_cache_manager
