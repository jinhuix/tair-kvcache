#include "kv_cache_manager/optimizer/eviction_policy/checkpoint_lru.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

CheckpointLruEvictionPolicy::CheckpointLruEvictionPolicy(const std::string &name,
                                                         const CheckpointLruParams &params)
    : EvictionPolicy(name)
    , evict_unreferenced_full_blocks_first_(params.evict_unreferenced_full_blocks_first)
    , score_alpha_(std::clamp(params.score_alpha, 0.0, 1.0))
    , score_min_interval_seconds_(std::max(params.score_min_interval_seconds, 1e-9))
    , score_first_hit_interval_seconds_(std::max(params.score_first_hit_interval_seconds, 1e-9))
    , score_initial_hotness_(std::max(params.score_initial_hotness, 0.0))
    , score_max_hotness_(std::max(params.score_max_hotness, score_initial_hotness_)) {}

void CheckpointLruEvictionPolicy::AddUnreferencedFullBlock(BlockEntry *block) {
    if (block == nullptr || block->owner_node == nullptr || block->checkpoint_ref_count != 0 ||
        unreferenced_full_index_.count(block) != 0) {
        return;
    }
    unreferenced_full_lru_.push_front(block);
    unreferenced_full_index_[block] = unreferenced_full_lru_.begin();
}

void CheckpointLruEvictionPolicy::RemoveUnreferencedFullBlock(BlockEntry *block) {
    auto it = unreferenced_full_index_.find(block);
    if (it == unreferenced_full_index_.end()) {
        return;
    }
    unreferenced_full_lru_.erase(it->second);
    unreferenced_full_index_.erase(it);
}

void CheckpointLruEvictionPolicy::OnBlockWritten(BlockEntry *block) {
    if (block == nullptr) {
        return;
    }
    const auto [_, inserted] = resident_blocks_.insert(block);
    if (!inserted) {
        return;
    }
    AddUnreferencedFullBlock(block);
}

void CheckpointLruEvictionPolicy::OnNodeWritten(std::vector<BlockEntry *> &blocks) {
    for (auto *block : blocks) {
        OnBlockWritten(block);
    }
}

bool CheckpointLruEvictionPolicy::RegisterCheckpoint(uint64_t checkpoint_id,
                                                     BlockEntry *full_boundary,
                                                     const std::vector<BlockEntry *> &mamba_objects,
                                                     int64_t timestamp) {
    if (full_boundary == nullptr) {
        KVCM_LOG_ERROR("checkpoint_lru register failed checkpoint_id=%llu reason=full_boundary_null "
                       "mamba_objects=%zu resident_blocks=%zu checkpoints=%zu",
                       static_cast<unsigned long long>(checkpoint_id), mamba_objects.size(), resident_blocks_.size(),
                       checkpoints_.size());
        return false;
    }
    if (full_boundary->owner_node == nullptr) {
        KVCM_LOG_ERROR("checkpoint_lru register failed checkpoint_id=%llu reason=full_boundary_detached "
                       "boundary=%p key=%lld mamba_objects=%zu resident_blocks=%zu checkpoints=%zu",
                       static_cast<unsigned long long>(checkpoint_id), static_cast<void *>(full_boundary),
                       static_cast<long long>(full_boundary->key), mamba_objects.size(), resident_blocks_.size(),
                       checkpoints_.size());
        return false;
    }
    if (mamba_objects.empty()) {
        KVCM_LOG_ERROR("checkpoint_lru register failed checkpoint_id=%llu reason=no_mamba_objects "
                       "boundary=%p key=%lld resident_blocks=%zu checkpoints=%zu",
                       static_cast<unsigned long long>(checkpoint_id), static_cast<void *>(full_boundary),
                       static_cast<long long>(full_boundary->key), resident_blocks_.size(), checkpoints_.size());
        return false;
    }
    if (checkpoints_.count(checkpoint_id) != 0) {
        KVCM_LOG_ERROR("checkpoint_lru register failed checkpoint_id=%llu reason=duplicate_checkpoint "
                       "boundary=%p key=%lld mamba_objects=%zu resident_blocks=%zu checkpoints=%zu",
                       static_cast<unsigned long long>(checkpoint_id), static_cast<void *>(full_boundary),
                       static_cast<long long>(full_boundary->key), mamba_objects.size(), resident_blocks_.size(),
                       checkpoints_.size());
        return false;
    }

    // Validate the complete prefix and all Mamba groups before changing any
    // reference count, so failed admission is side-effect free.
    std::vector<BlockEntry *> prefix;
    std::unordered_set<BlockEntry *> seen;
    size_t prefix_depth = 0;
    for (BlockEntry *block = full_boundary; block != nullptr; block = block->prefix_parent) {
        const bool has_owner = block->owner_node != nullptr;
        const bool resident = resident_blocks_.count(block) != 0;
        const bool has_location = block->location_map.find(name()) != block->location_map.end();
        const bool acyclic = seen.insert(block).second;
        if (!has_owner || !resident || !has_location || !acyclic) {
            KVCM_LOG_ERROR("checkpoint_lru register failed checkpoint_id=%llu reason=bad_prefix_block "
                           "depth=%zu block=%p key=%lld has_owner=%d resident=%d has_location=%d acyclic=%d "
                           "ref_count=%zu resident_blocks=%zu checkpoints=%zu",
                           static_cast<unsigned long long>(checkpoint_id), prefix_depth, static_cast<void *>(block),
                           static_cast<long long>(block->key), has_owner ? 1 : 0, resident ? 1 : 0,
                           has_location ? 1 : 0, acyclic ? 1 : 0, block->checkpoint_ref_count,
                           resident_blocks_.size(), checkpoints_.size());
            return false;
        }
        prefix.push_back(block);
        ++prefix_depth;
    }
    size_t object_idx = 0;
    for (auto *object : mamba_objects) {
        if (object == nullptr) {
            KVCM_LOG_ERROR("checkpoint_lru register failed checkpoint_id=%llu reason=null_mamba_object "
                           "object_idx=%zu prefix_blocks=%zu resident_blocks=%zu checkpoints=%zu",
                           static_cast<unsigned long long>(checkpoint_id), object_idx, prefix.size(),
                           resident_blocks_.size(), checkpoints_.size());
            return false;
        }
        const bool external_object = object->owner_node == nullptr;
        const bool resident = resident_blocks_.count(object) != 0;
        const bool has_location = object->location_map.find(name()) != object->location_map.end();
        const bool unregistered = mamba_to_checkpoint_.count(object) == 0;
        if (!external_object || !resident || !has_location || !unregistered) {
            KVCM_LOG_ERROR("checkpoint_lru register failed checkpoint_id=%llu reason=bad_mamba_object "
                           "object_idx=%zu object=%p key=%lld external_object=%d resident=%d has_location=%d "
                           "unregistered=%d prefix_blocks=%zu resident_blocks=%zu checkpoints=%zu",
                           static_cast<unsigned long long>(checkpoint_id), object_idx, static_cast<void *>(object),
                           static_cast<long long>(object->key), external_object ? 1 : 0, resident ? 1 : 0,
                           has_location ? 1 : 0, unregistered ? 1 : 0, prefix.size(), resident_blocks_.size(),
                           checkpoints_.size());
            return false;
        }
        ++object_idx;
    }

    checkpoint_lru_.push_front(checkpoint_id);
    CheckpointRecord record;
    record.full_boundary = full_boundary;
    record.mamba_objects = mamba_objects;
    record.prefix_blocks = prefix.size();
    record.hotness = score_initial_hotness_;
    record.last_access_time = timestamp;
    record.lru_it = checkpoint_lru_.begin();
    checkpoints_.emplace(checkpoint_id, std::move(record));
    full_boundary_to_checkpoint_[full_boundary] = checkpoint_id;

    for (auto *block : prefix) {
        if (block->checkpoint_ref_count == 0) {
            RemoveUnreferencedFullBlock(block);
        }
        ++block->checkpoint_ref_count;
    }
    for (auto *object : mamba_objects) {
        mamba_to_checkpoint_[object] = checkpoint_id;
    }
    return true;
}

bool CheckpointLruEvictionPolicy::TouchCheckpoint(uint64_t checkpoint_id, int64_t timestamp) {
    auto it = checkpoints_.find(checkpoint_id);
    if (it == checkpoints_.end()) {
        return false;
    }
    auto &record = it->second;
    double interval_seconds = score_first_hit_interval_seconds_;
    if (record.last_hit_time >= 0 && timestamp > record.last_hit_time) {
        interval_seconds = static_cast<double>(timestamp - record.last_hit_time) / 1e9;
    }
    const double interval_minutes = std::max(interval_seconds, score_min_interval_seconds_) / 60.0;
    const double instant_hotness = 1.0 / interval_minutes;
    record.hotness = std::min(score_max_hotness_,
                              (1.0 - score_alpha_) * record.hotness + score_alpha_ * instant_hotness);
    ++record.hit_count;
    record.last_hit_time = timestamp;
    record.last_access_time = timestamp;
    checkpoint_lru_.splice(checkpoint_lru_.begin(), checkpoint_lru_, record.lru_it);
    record.lru_it = checkpoint_lru_.begin();
    return true;
}

void CheckpointLruEvictionPolicy::SetProtectedCheckpointsForAdmission(
    const std::vector<uint64_t> &checkpoint_ids) {
    protected_checkpoints_for_admission_.clear();
    for (const uint64_t checkpoint_id : checkpoint_ids) {
        if (checkpoint_id != 0 && checkpoints_.count(checkpoint_id) != 0) {
            protected_checkpoints_for_admission_.insert(checkpoint_id);
        }
    }
}

void CheckpointLruEvictionPolicy::ClearProtectedCheckpointsForAdmission() {
    protected_checkpoints_for_admission_.clear();
}

void CheckpointLruEvictionPolicy::OnBlockAccessed(BlockEntry *block, int64_t timestamp) {
    auto it = mamba_to_checkpoint_.find(block);
    if (it != mamba_to_checkpoint_.end()) {
        TouchCheckpoint(it->second, timestamp);
    }
}

bool CheckpointLruEvictionPolicy::DetachPhysicalBlock(BlockEntry *block,
                                                      std::vector<BlockEntry *> *evicted) {
    if (block == nullptr || resident_blocks_.erase(block) == 0) {
        return false;
    }
    RemoveUnreferencedFullBlock(block);
    ClearBlockLocation(block);
    if (evicted != nullptr) {
        evicted->push_back(block);
    }
    return true;
}

size_t CheckpointLruEvictionPolicy::DetachCheckpoint(uint64_t checkpoint_id,
                                                     std::vector<BlockEntry *> *evicted) {
    auto checkpoint_it = checkpoints_.find(checkpoint_id);
    if (checkpoint_it == checkpoints_.end()) {
        return 0;
    }

    // Copy raw members before erasing the record and its list iterator.
    BlockEntry *boundary = checkpoint_it->second.full_boundary;
    const auto mamba_objects = checkpoint_it->second.mamba_objects;
    checkpoint_lru_.erase(checkpoint_it->second.lru_it);
    full_boundary_to_checkpoint_.erase(boundary);
    checkpoints_.erase(checkpoint_it);

    const size_t before = evicted == nullptr ? 0 : evicted->size();
    size_t detached_without_output = 0;
    for (auto *object : mamba_objects) {
        mamba_to_checkpoint_.erase(object);
        if (DetachPhysicalBlock(object, evicted) && evicted == nullptr) {
            ++detached_without_output;
        }
    }

    std::unordered_set<BlockEntry *> seen;
    for (BlockEntry *block = boundary; block != nullptr; block = block->prefix_parent) {
        if (!seen.insert(block).second) {
            throw std::runtime_error("cycle detected in checkpoint prefix chain");
        }
        if (block->checkpoint_ref_count == 0) {
            throw std::runtime_error("checkpoint prefix reference count underflow");
        }
        --block->checkpoint_ref_count;
        if (block->checkpoint_ref_count == 0) {
            if (DetachPhysicalBlock(block, evicted) && evicted == nullptr) {
                ++detached_without_output;
            }
        }
    }
    return evicted == nullptr ? detached_without_output : evicted->size() - before;
}

size_t CheckpointLruEvictionPolicy::EvictUnreferencedFullBlocks(size_t count,
                                                               std::vector<BlockEntry *> *evicted) {
    size_t detached = 0;
    while (detached < count && !unreferenced_full_lru_.empty()) {
        BlockEntry *block = unreferenced_full_lru_.back();
        if (block == nullptr || block->checkpoint_ref_count != 0) {
            RemoveUnreferencedFullBlock(block);
            continue;
        }
        if (DetachPhysicalBlock(block, evicted)) {
            ++detached;
        }
    }
    return detached;
}

size_t CheckpointLruEvictionPolicy::EstimateExclusiveFullBlocks(const CheckpointRecord &record) const {
    size_t exclusive_blocks = 0;
    std::unordered_set<BlockEntry *> seen;
    for (BlockEntry *block = record.full_boundary; block != nullptr; block = block->prefix_parent) {
        if (!seen.insert(block).second) {
            throw std::runtime_error("cycle detected in checkpoint prefix chain");
        }
        if (block->checkpoint_ref_count == 1) {
            ++exclusive_blocks;
        }
    }
    return exclusive_blocks;
}

size_t CheckpointLruEvictionPolicy::EstimateFallbackPrefixBlocks(uint64_t checkpoint_id,
                                                                 const CheckpointRecord &record) const {
    std::unordered_set<BlockEntry *> seen;
    for (BlockEntry *block = record.full_boundary; block != nullptr; block = block->prefix_parent) {
        if (!seen.insert(block).second) {
            throw std::runtime_error("cycle detected in checkpoint prefix chain");
        }
        auto it = full_boundary_to_checkpoint_.find(block);
        if (it == full_boundary_to_checkpoint_.end() || it->second == checkpoint_id) {
            continue;
        }
        auto checkpoint_it = checkpoints_.find(it->second);
        if (checkpoint_it != checkpoints_.end()) {
            return checkpoint_it->second.prefix_blocks;
        }
    }
    return 0;
}

double CheckpointLruEvictionPolicy::CheckpointScore(uint64_t checkpoint_id,
                                                    const CheckpointRecord &record) const {
    const size_t fallback_prefix_blocks = EstimateFallbackPrefixBlocks(checkpoint_id, record);
    const size_t marginal_blocks =
        std::max<size_t>(1, record.prefix_blocks > fallback_prefix_blocks ? record.prefix_blocks - fallback_prefix_blocks
                                                                          : 1);
    const size_t evict_cost_blocks =
        std::max<size_t>(1, record.mamba_objects.size() + EstimateExclusiveFullBlocks(record));
    return std::log1p(static_cast<double>(marginal_blocks) / static_cast<double>(evict_cost_blocks)) *
           record.hotness;
}

uint64_t CheckpointLruEvictionPolicy::SelectLowestScoreCheckpoint() const {
    uint64_t selected = 0;
    double selected_score = std::numeric_limits<double>::infinity();
    int64_t selected_last_access = std::numeric_limits<int64_t>::max();
    for (const auto &[checkpoint_id, record] : checkpoints_) {
        if (protected_checkpoints_for_admission_.count(checkpoint_id) != 0) {
            continue;
        }
        const double score = CheckpointScore(checkpoint_id, record);
        if (selected == 0 || score < selected_score ||
            (score == selected_score &&
             std::tie(record.last_access_time, checkpoint_id) < std::tie(selected_last_access, selected))) {
            selected = checkpoint_id;
            selected_score = score;
            selected_last_access = record.last_access_time;
        }
    }
    return selected;
}

std::vector<BlockEntry *> CheckpointLruEvictionPolicy::EvictBlocks(size_t count) {
    std::vector<BlockEntry *> evicted;
    if (count == 0) {
        return evicted;
    }

    if (evict_unreferenced_full_blocks_first_) {
        EvictUnreferencedFullBlocks(count, &evicted);
    }
    while (evicted.size() < count && !checkpoint_lru_.empty()) {
        const uint64_t victim = SelectLowestScoreCheckpoint();
        if (victim == 0) {
            if (!protected_checkpoints_for_admission_.empty()) {
                KVCM_LOG_DEBUG("checkpoint_lru admission eviction stopped because all remaining checkpoints are "
                               "protected count=%zu protected=%zu requested=%zu evicted=%zu",
                               checkpoints_.size(), protected_checkpoints_for_admission_.size(), count, evicted.size());
            }
            break;
        }
        DetachCheckpoint(victim, &evicted);
    }
    if (evicted.size() < count) {
        EvictUnreferencedFullBlocks(count - evicted.size(), &evicted);
    }
    return evicted;
}

bool CheckpointLruEvictionPolicy::RemoveBlock(BlockEntry *block) {
    if (block == nullptr) {
        return false;
    }
    auto mamba_it = mamba_to_checkpoint_.find(block);
    if (mamba_it != mamba_to_checkpoint_.end()) {
        return DetachCheckpoint(mamba_it->second, nullptr) > 0;
    }
    // A referenced full block cannot be removed independently without
    // invalidating one or more resident checkpoints.
    if (block->checkpoint_ref_count != 0) {
        return false;
    }
    return DetachPhysicalBlock(block, nullptr);
}

void CheckpointLruEvictionPolicy::Clear() {
    for (auto *block : resident_blocks_) {
        if (block != nullptr) {
            block->checkpoint_ref_count = 0;
            ClearBlockLocation(block);
        }
    }
    resident_blocks_.clear();
    unreferenced_full_lru_.clear();
    unreferenced_full_index_.clear();
    checkpoint_lru_.clear();
    checkpoints_.clear();
    mamba_to_checkpoint_.clear();
    full_boundary_to_checkpoint_.clear();
    protected_checkpoints_for_admission_.clear();
}

} // namespace kv_cache_manager
