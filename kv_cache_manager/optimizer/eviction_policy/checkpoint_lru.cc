#include "kv_cache_manager/optimizer/eviction_policy/checkpoint_lru.h"

#include <stdexcept>

namespace kv_cache_manager {

CheckpointLruEvictionPolicy::CheckpointLruEvictionPolicy(const std::string &name,
                                                         const CheckpointLruParams &params)
    : EvictionPolicy(name)
    , evict_unreferenced_full_blocks_first_(params.evict_unreferenced_full_blocks_first) {}

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
    if (full_boundary == nullptr || full_boundary->owner_node == nullptr || mamba_objects.empty() ||
        checkpoints_.count(checkpoint_id) != 0) {
        return false;
    }

    // Validate the complete prefix and all Mamba groups before changing any
    // reference count, so failed admission is side-effect free.
    std::vector<BlockEntry *> prefix;
    std::unordered_set<BlockEntry *> seen;
    for (BlockEntry *block = full_boundary; block != nullptr; block = block->prefix_parent) {
        if (block->owner_node == nullptr || resident_blocks_.count(block) == 0 ||
            block->location_map.find(name()) == block->location_map.end() || !seen.insert(block).second) {
            return false;
        }
        prefix.push_back(block);
    }
    for (auto *object : mamba_objects) {
        if (object == nullptr || object->owner_node != nullptr || resident_blocks_.count(object) == 0 ||
            object->location_map.find(name()) == object->location_map.end() ||
            mamba_to_checkpoint_.count(object) != 0) {
            return false;
        }
    }

    checkpoint_lru_.push_front(checkpoint_id);
    CheckpointRecord record;
    record.full_boundary = full_boundary;
    record.mamba_objects = mamba_objects;
    record.last_access_time = timestamp;
    record.lru_it = checkpoint_lru_.begin();
    checkpoints_.emplace(checkpoint_id, std::move(record));

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
    it->second.last_access_time = timestamp;
    checkpoint_lru_.splice(checkpoint_lru_.begin(), checkpoint_lru_, it->second.lru_it);
    it->second.lru_it = checkpoint_lru_.begin();
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
    checkpoint_lru_.erase(checkpoint_it->second.lru_it);
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

std::vector<BlockEntry *> CheckpointLruEvictionPolicy::EvictBlocks(size_t count) {
    std::vector<BlockEntry *> evicted;
    if (count == 0) {
        return evicted;
    }

    if (evict_unreferenced_full_blocks_first_) {
        EvictUnreferencedFullBlocks(count, &evicted);
    }
    while (evicted.size() < count && !checkpoint_lru_.empty()) {
        DetachCheckpoint(checkpoint_lru_.back(), &evicted);
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
}

} // namespace kv_cache_manager
