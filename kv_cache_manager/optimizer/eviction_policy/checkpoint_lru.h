#pragma once

#include <cstdint>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"

namespace kv_cache_manager {

// Exact LRU over complete Mamba checkpoints. Full-attention blocks are retained
// by checkpoint reference counts; a Mamba checkpoint is always evicted together
// with all of its state groups and every prefix block whose last reference
// disappears. Newly written full blocks are allowed to remain unreferenced so
// the simulator matches the inference engine's write-all behavior. Under
// pressure those currently unmatchable blocks are reclaimed first.
class CheckpointLruEvictionPolicy : public EvictionPolicy {
public:
    explicit CheckpointLruEvictionPolicy(const std::string &name, const CheckpointLruParams &params);
    ~CheckpointLruEvictionPolicy() override = default;

    void OnBlockWritten(BlockEntry *block) override;
    void OnBlockCopied(BlockEntry *block) override { OnBlockWritten(block); }
    void OnNodeWritten(std::vector<BlockEntry *> &blocks) override;
    std::vector<BlockEntry *> EvictBlocks(size_t count) override;
    bool RemoveBlock(BlockEntry *block) override;
    void Clear() override;
    size_t size() const override { return resident_blocks_.size(); }

    bool RegisterCheckpoint(uint64_t checkpoint_id,
                            BlockEntry *full_boundary,
                            const std::vector<BlockEntry *> &mamba_objects,
                            int64_t timestamp);
    bool TouchCheckpoint(uint64_t checkpoint_id, int64_t timestamp);
    bool HasCheckpoint(uint64_t checkpoint_id) const { return checkpoints_.count(checkpoint_id) != 0; }
    size_t checkpoint_count() const { return checkpoints_.size(); }
    size_t unreferenced_full_block_count() const { return unreferenced_full_lru_.size(); }

private:
    struct CheckpointRecord {
        BlockEntry *full_boundary = nullptr;
        std::vector<BlockEntry *> mamba_objects;
        int64_t last_access_time = -1;
        std::list<uint64_t>::iterator lru_it;
    };

    void OnBlockAccessed(BlockEntry *block, int64_t timestamp) override;
    void AddUnreferencedFullBlock(BlockEntry *block);
    void RemoveUnreferencedFullBlock(BlockEntry *block);
    bool DetachPhysicalBlock(BlockEntry *block, std::vector<BlockEntry *> *evicted);
    size_t DetachCheckpoint(uint64_t checkpoint_id, std::vector<BlockEntry *> *evicted);
    size_t EvictUnreferencedFullBlocks(size_t count, std::vector<BlockEntry *> *evicted);

    bool evict_unreferenced_full_blocks_first_ = true;
    std::unordered_set<BlockEntry *> resident_blocks_;
    std::list<BlockEntry *> unreferenced_full_lru_;
    std::unordered_map<BlockEntry *, std::list<BlockEntry *>::iterator> unreferenced_full_index_;
    std::list<uint64_t> checkpoint_lru_;
    std::unordered_map<uint64_t, CheckpointRecord> checkpoints_;
    std::unordered_map<BlockEntry *, uint64_t> mamba_to_checkpoint_;
};

} // namespace kv_cache_manager
