#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"
#include "kv_cache_manager/optimizer/eviction_policy/common_structure.h"

namespace kv_cache_manager {

class PromoteLruEvictionPolicy : public EvictionPolicy {
public:
    explicit PromoteLruEvictionPolicy(const std::string &name, const PromoteLruParams &params);
    ~PromoteLruEvictionPolicy() override;

    size_t size() const override { return node_map_.size(); }
    void OnBlockWritten(BlockEntry *block) override;
    void OnNodeWritten(std::vector<BlockEntry *> &blocks) override;
    void OnBlockCopied(BlockEntry *block) override;
    void OnBlockAccessedWithOptions(BlockEntry *block, int64_t timestamp, bool refresh_ttl_on_read) override;
    void OnBlockTouched(BlockEntry *block, int64_t timestamp) override;
    std::vector<BlockEntry *> EvictBlocks(size_t count) override;
    bool RemoveBlock(BlockEntry *block) override;
    void Clear() override;

private:
    int32_t shard_count_ = 1;
    int32_t sample_times_ = 1;
    double amplification_factor_ = 1.0;

    struct PromoteListNode : public LinkedListNode {
        BlockEntry *payload_ = nullptr;
        bool protected_queue = false;
    };

    struct CandidateEntry {
        BlockEntry *block = nullptr;
        PromoteListNode *node = nullptr;
        int32_t shard_index = 0;
        bool protected_queue = false;
    };

    void OnBlockAccessed(BlockEntry *block, int64_t timestamp) override;
    bool PromoteEnabledForTier(const PromoteLruParams &params) const;
    int32_t GetShardIndex(BlockEntry *block) const;
    int64_t GetShardTailTime(const std::vector<LinkedList> &lists, int32_t shard_index) const;
    std::vector<LinkedList> &ListsForQueue(bool protected_queue);
    void InsertNewBlock(BlockEntry *block);
    void RefreshInCurrentQueue(PromoteListNode *node);
    void PromoteOrRefresh(PromoteListNode *node);
    void SampleFromShard(std::vector<LinkedList> &lists,
                         int32_t shard_index,
                         size_t count,
                         bool protected_queue,
                         std::vector<CandidateEntry> &candidates);
    void ReturnCandidates(const std::vector<CandidateEntry> &candidates);
    void CommitEviction(const std::vector<CandidateEntry> &candidates, std::vector<BlockEntry *> &evicted_blocks);
    size_t EvictFromQueue(size_t count, bool protected_queue, std::vector<BlockEntry *> &evicted_blocks);
    void ClearListLocations();

    bool promote_enabled_ = true;
    std::vector<LinkedList> probation_shard_lists_;
    std::vector<LinkedList> protected_shard_lists_;
    std::unordered_map<BlockEntry *, PromoteListNode *> node_map_;
};

} // namespace kv_cache_manager
