#include "kv_cache_manager/optimizer/eviction_policy/checkpoint_lru.h"

#include <algorithm>
#include <stdexcept>
#include <tuple>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

CheckpointLruEvictionPolicy::CheckpointLruEvictionPolicy(const std::string &name,
                                                         const CheckpointLruParams &params)
    : EvictionPolicy(name)
    , evict_unreferenced_full_blocks_first_(params.evict_unreferenced_full_blocks_first)
    , score_value_bonus_seconds_(std::max(params.score_value_bonus_seconds, 0.0)) {}

bool CheckpointLruEvictionPolicy::ScoreHeapGreater::operator()(const ScoreHeapEntry &lhs,
                                                               const ScoreHeapEntry &rhs) const {
    return std::tie(lhs.score, lhs.last_access_time, lhs.checkpoint_id) >
           std::tie(rhs.score, rhs.last_access_time, rhs.checkpoint_id);
}

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
    if (checkpoint_id == 0) {
        KVCM_LOG_ERROR("checkpoint_lru register failed reason=checkpoint_id_zero");
        return false;
    }
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
    if (full_boundary_to_checkpoint_.count(full_boundary) != 0) {
        KVCM_LOG_ERROR("checkpoint_lru register failed checkpoint_id=%llu reason=duplicate_full_boundary "
                       "boundary=%p key=%lld checkpoints=%zu",
                       static_cast<unsigned long long>(checkpoint_id), static_cast<void *>(full_boundary),
                       static_cast<long long>(full_boundary->key), checkpoints_.size());
        return false;
    }

    // Validate the complete prefix and all Mamba groups before changing any
    // reference count, so failed admission is side-effect free.
    std::vector<BlockEntry *> prefix;
    std::unordered_set<BlockEntry *> seen;
    size_t prefix_depth = 0;
    uint64_t parent_checkpoint_id = 0;
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
        if (parent_checkpoint_id == 0) {
            auto parent_it = full_boundary_to_checkpoint_.find(block);
            if (parent_it != full_boundary_to_checkpoint_.end()) {
                parent_checkpoint_id = parent_it->second;
            }
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
    record.parent_checkpoint_id = parent_checkpoint_id;
    const size_t parent_prefix_blocks =
        parent_checkpoint_id == 0 ? 0 : checkpoints_.at(parent_checkpoint_id).prefix_blocks;
    record.marginal_blocks = std::max<size_t>(1, record.prefix_blocks - parent_prefix_blocks);
    record.last_access_time = timestamp;
    record.lru_it = checkpoint_lru_.begin();
    auto [checkpoint_it, inserted] = checkpoints_.emplace(checkpoint_id, std::move(record));
    if (!inserted) {
        throw std::runtime_error("checkpoint map insertion failed after duplicate validation");
    }
    full_boundary_to_checkpoint_[full_boundary] = checkpoint_id;

    // A checkpoint can be registered between an existing parent and one or
    // more of its direct children. Reparent only those direct children whose
    // prefix contains the new boundary; their marginal value changes. A
    // previously unreferenced boundary cannot have resident descendants, so
    // the common append-only admission path avoids scanning siblings/roots.
    std::vector<uint64_t> candidate_children;
    if (full_boundary->checkpoint_ref_count != 0) {
        if (parent_checkpoint_id == 0) {
            candidate_children.assign(root_checkpoints_.begin(), root_checkpoints_.end());
        } else {
            const auto &siblings = checkpoints_.at(parent_checkpoint_id).children;
            candidate_children.assign(siblings.begin(), siblings.end());
        }
    }
    std::unordered_set<uint64_t> scores_to_update;
    for (uint64_t child_id : candidate_children) {
        auto child_it = checkpoints_.find(child_id);
        if (child_it == checkpoints_.end() ||
            !IsCheckpointDescendantOf(child_it->second, full_boundary, prefix.size())) {
            continue;
        }
        if (parent_checkpoint_id == 0) {
            root_checkpoints_.erase(child_id);
        } else {
            checkpoints_.at(parent_checkpoint_id).children.erase(child_id);
        }
        child_it->second.parent_checkpoint_id = checkpoint_id;
        child_it->second.marginal_blocks =
            std::max<size_t>(1, child_it->second.prefix_blocks - prefix.size());
        checkpoint_it->second.children.insert(child_id);
        scores_to_update.insert(child_id);
    }
    if (parent_checkpoint_id == 0) {
        root_checkpoints_.insert(checkpoint_id);
    } else {
        checkpoints_.at(parent_checkpoint_id).children.insert(checkpoint_id);
    }

    for (auto *block : prefix) {
        const size_t old_ref_count = block->checkpoint_ref_count;
        const uint64_t old_checkpoint_xor = full_block_checkpoint_xor_[block];
        full_block_checkpoint_xor_[block] = old_checkpoint_xor ^ checkpoint_id;
        if (block->checkpoint_ref_count == 0) {
            RemoveUnreferencedFullBlock(block);
            ++checkpoint_it->second.exclusive_full_blocks;
        } else if (old_ref_count == 1) {
            auto sole_owner_it = checkpoints_.find(old_checkpoint_xor);
            if (sole_owner_it == checkpoints_.end() || sole_owner_it->second.exclusive_full_blocks == 0) {
                throw std::runtime_error("checkpoint exclusive full block accounting is inconsistent");
            }
            --sole_owner_it->second.exclusive_full_blocks;
            scores_to_update.insert(old_checkpoint_xor);
        }
        ++block->checkpoint_ref_count;
    }
    for (auto *object : mamba_objects) {
        mamba_to_checkpoint_[object] = checkpoint_id;
    }
    if (score_base_timestamp_ < 0) {
        score_base_timestamp_ = timestamp;
    }
    scores_to_update.insert(checkpoint_id);
    UpdateCheckpointScores(scores_to_update);
    return true;
}

bool CheckpointLruEvictionPolicy::TouchCheckpoint(uint64_t checkpoint_id, int64_t timestamp) {
    auto it = checkpoints_.find(checkpoint_id);
    if (it == checkpoints_.end()) {
        return false;
    }
    auto &record = it->second;
    if (timestamp <= record.last_access_time) {
        return true;
    }
    record.last_access_time = timestamp;
    ++record.hit_count;
    checkpoint_lru_.splice(checkpoint_lru_.begin(), checkpoint_lru_, record.lru_it);
    record.lru_it = checkpoint_lru_.begin();
    UpdateCheckpointScore(checkpoint_id);
    return true;
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
    const uint64_t parent_checkpoint_id = checkpoint_it->second.parent_checkpoint_id;
    const auto children = checkpoint_it->second.children;
    std::unordered_set<uint64_t> scores_to_update;

    if (parent_checkpoint_id == 0) {
        root_checkpoints_.erase(checkpoint_id);
    } else {
        checkpoints_.at(parent_checkpoint_id).children.erase(checkpoint_id);
    }
    for (uint64_t child_id : children) {
        auto child_it = checkpoints_.find(child_id);
        if (child_it == checkpoints_.end()) {
            continue;
        }
        child_it->second.parent_checkpoint_id = parent_checkpoint_id;
        const size_t parent_prefix_blocks =
            parent_checkpoint_id == 0 ? 0 : checkpoints_.at(parent_checkpoint_id).prefix_blocks;
        child_it->second.marginal_blocks =
            std::max<size_t>(1, child_it->second.prefix_blocks - parent_prefix_blocks);
        if (parent_checkpoint_id == 0) {
            root_checkpoints_.insert(child_id);
        } else {
            checkpoints_.at(parent_checkpoint_id).children.insert(child_id);
        }
        scores_to_update.insert(child_id);
    }
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
        const size_t old_ref_count = block->checkpoint_ref_count;
        auto xor_it = full_block_checkpoint_xor_.find(block);
        if (xor_it == full_block_checkpoint_xor_.end()) {
            throw std::runtime_error("checkpoint prefix xor accounting is missing");
        }
        xor_it->second ^= checkpoint_id;
        --block->checkpoint_ref_count;
        if (block->checkpoint_ref_count == 0) {
            full_block_checkpoint_xor_.erase(xor_it);
            if (DetachPhysicalBlock(block, evicted) && evicted == nullptr) {
                ++detached_without_output;
            }
        } else if (old_ref_count == 2) {
            const uint64_t sole_owner_id = xor_it->second;
            auto sole_owner_it = checkpoints_.find(sole_owner_id);
            if (sole_owner_it == checkpoints_.end()) {
                throw std::runtime_error("remaining sole checkpoint is missing");
            }
            ++sole_owner_it->second.exclusive_full_blocks;
            scores_to_update.insert(sole_owner_id);
        }
    }
    UpdateCheckpointScores(scores_to_update);
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

bool CheckpointLruEvictionPolicy::IsCheckpointDescendantOf(const CheckpointRecord &record,
                                                           BlockEntry *ancestor_boundary,
                                                           size_t ancestor_prefix_blocks) const {
    if (record.prefix_blocks < ancestor_prefix_blocks) {
        return false;
    }
    BlockEntry *block = record.full_boundary;
    for (size_t depth = record.prefix_blocks; depth > ancestor_prefix_blocks && block != nullptr; --depth) {
        block = block->prefix_parent;
    }
    return block == ancestor_boundary;
}

double CheckpointLruEvictionPolicy::CheckpointScore(const CheckpointRecord &record) const {
    const size_t evict_cost_blocks =
        std::max<size_t>(1, record.mamba_objects.size() + record.exclusive_full_blocks);
    const int64_t elapsed_ns = score_base_timestamp_ < 0 || record.last_access_time <= score_base_timestamp_
                                   ? 0
                                   : record.last_access_time - score_base_timestamp_;
    const double last_access_seconds = static_cast<double>(elapsed_ns) / 1e9;
    const double value_density = static_cast<double>(record.marginal_blocks) /
                                 static_cast<double>(record.marginal_blocks + evict_cost_blocks);
    const double has_hit = record.hit_count == 0 ? 0.0 : 1.0;
    return last_access_seconds + score_value_bonus_seconds_ * has_hit * value_density;
}

void CheckpointLruEvictionPolicy::UpdateCheckpointScore(uint64_t checkpoint_id) {
    auto it = checkpoints_.find(checkpoint_id);
    if (it == checkpoints_.end()) {
        return;
    }
    auto &record = it->second;
    record.cached_score = CheckpointScore(record);
    ++record.score_version;
    score_heap_.push(
        ScoreHeapEntry{record.cached_score, record.last_access_time, checkpoint_id, record.score_version});
    MaybeRebuildScoreHeap();
}

void CheckpointLruEvictionPolicy::UpdateCheckpointScores(
    const std::unordered_set<uint64_t> &checkpoint_ids) {
    for (uint64_t checkpoint_id : checkpoint_ids) {
        UpdateCheckpointScore(checkpoint_id);
    }
}

void CheckpointLruEvictionPolicy::MaybeRebuildScoreHeap() {
    const size_t max_stale_entries = std::max<size_t>(1024, checkpoints_.size() * 4);
    if (score_heap_.size() > checkpoints_.size() + max_stale_entries) {
        RebuildScoreHeap();
    }
}

void CheckpointLruEvictionPolicy::RebuildScoreHeap() {
    decltype(score_heap_) fresh;
    for (const auto &[checkpoint_id, record] : checkpoints_) {
        fresh.push(ScoreHeapEntry{record.cached_score, record.last_access_time, checkpoint_id, record.score_version});
    }
    score_heap_.swap(fresh);
}

uint64_t CheckpointLruEvictionPolicy::SelectLowestScoreCheckpoint() {
    while (!score_heap_.empty()) {
        const ScoreHeapEntry candidate = score_heap_.top();
        score_heap_.pop();
        auto it = checkpoints_.find(candidate.checkpoint_id);
        if (it == checkpoints_.end() || it->second.score_version != candidate.version) {
            continue;
        }
        return candidate.checkpoint_id;
    }
    return 0;
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
    full_block_checkpoint_xor_.clear();
    root_checkpoints_.clear();
    score_heap_ = decltype(score_heap_){};
    score_base_timestamp_ = -1;
}

} // namespace kv_cache_manager
