#include "kv_cache_manager/optimizer/eviction_policy/promote_lru.h"

#include <algorithm>
#include <climits>

namespace kv_cache_manager {

PromoteLruEvictionPolicy::PromoteLruEvictionPolicy(const std::string &name, const PromoteLruParams &params)
    : EvictionPolicy(name)
    , shard_count_(params.shard_count > 0 ? params.shard_count : 1)
    , sample_times_(params.sample_times > 0 ? params.sample_times : 1)
    , amplification_factor_(params.eviction_amplification_factor > 1.0 ? params.eviction_amplification_factor : 1.0)
    , promote_enabled_(PromoteEnabledForTier(params))
    , probation_shard_lists_(shard_count_)
    , protected_shard_lists_(shard_count_) {}

PromoteLruEvictionPolicy::~PromoteLruEvictionPolicy() {
    for (auto &shard_list : probation_shard_lists_) {
        shard_list.clear();
    }
    for (auto &shard_list : protected_shard_lists_) {
        shard_list.clear();
    }
    node_map_.clear();
}

bool PromoteLruEvictionPolicy::PromoteEnabledForTier(const PromoteLruParams &params) const {
    if (params.enabled_tiers.empty()) {
        return true;
    }
    return std::find(params.enabled_tiers.begin(), params.enabled_tiers.end(), name()) != params.enabled_tiers.end();
}

int32_t PromoteLruEvictionPolicy::GetShardIndex(BlockEntry *block) const {
    return static_cast<int32_t>(static_cast<uint64_t>(block->key) % shard_count_);
}

int64_t PromoteLruEvictionPolicy::GetShardTailTime(const std::vector<LinkedList> &lists,
                                                   int32_t shard_index) const {
    LinkedListNode *tail = lists[shard_index].getTail();
    if (tail == nullptr) {
        return INT64_MAX;
    }
    auto *lru_node = static_cast<const PromoteListNode *>(tail);
    return lru_node->payload_ ? GetTierAccessTime(lru_node->payload_) : INT64_MAX;
}

std::vector<LinkedList> &PromoteLruEvictionPolicy::ListsForQueue(bool protected_queue) {
    return protected_queue ? protected_shard_lists_ : probation_shard_lists_;
}

void PromoteLruEvictionPolicy::InsertNewBlock(BlockEntry *block) {
    if (block == nullptr) {
        return;
    }
    auto it = node_map_.find(block);
    if (it != node_map_.end()) {
        RefreshInCurrentQueue(it->second);
        return;
    }

    auto *node = new PromoteListNode();
    node->payload_ = block;
    node->protected_queue = !promote_enabled_;
    auto &lists = ListsForQueue(node->protected_queue);
    const int32_t shard_index = GetShardIndex(block);
    if (node->protected_queue) {
        lists[shard_index].push_front(node);
    } else {
        lists[shard_index].push_front(node);
    }
    node_map_[block] = node;
}

void PromoteLruEvictionPolicy::OnBlockWritten(BlockEntry *block) { InsertNewBlock(block); }

void PromoteLruEvictionPolicy::OnNodeWritten(std::vector<BlockEntry *> &blocks) {
    for (auto *block : blocks) {
        OnBlockWritten(block);
    }
}

void PromoteLruEvictionPolicy::OnBlockCopied(BlockEntry *block) { InsertNewBlock(block); }

void PromoteLruEvictionPolicy::RefreshInCurrentQueue(PromoteListNode *node) {
    if (node == nullptr) {
        return;
    }
    auto &lists = ListsForQueue(node->protected_queue);
    lists[GetShardIndex(node->payload_)].move_to_front(node);
}

void PromoteLruEvictionPolicy::PromoteOrRefresh(PromoteListNode *node) {
    if (node == nullptr) {
        return;
    }
    if (!promote_enabled_ || node->protected_queue) {
        RefreshInCurrentQueue(node);
        return;
    }

    probation_shard_lists_[GetShardIndex(node->payload_)].unlink(node);
    node->protected_queue = true;
    protected_shard_lists_[GetShardIndex(node->payload_)].push_front(node);
}

void PromoteLruEvictionPolicy::OnBlockAccessed(BlockEntry *block, int64_t timestamp) {
    OnBlockAccessedWithOptions(block, timestamp, true);
}

void PromoteLruEvictionPolicy::OnBlockAccessedWithOptions(BlockEntry *block,
                                                          int64_t timestamp,
                                                          bool refresh_ttl_on_read) {
    (void)timestamp;
    (void)refresh_ttl_on_read;
    auto it = node_map_.find(block);
    if (it == node_map_.end()) {
        return;
    }
    PromoteOrRefresh(it->second);
}

void PromoteLruEvictionPolicy::OnBlockTouched(BlockEntry *block, int64_t timestamp) {
    (void)timestamp;
    auto it = node_map_.find(block);
    if (it == node_map_.end()) {
        return;
    }
    RefreshInCurrentQueue(it->second);
}

void PromoteLruEvictionPolicy::SampleFromShard(std::vector<LinkedList> &lists,
                                               int32_t shard_index,
                                               size_t count,
                                               bool protected_queue,
                                               std::vector<CandidateEntry> &candidates) {
    LinkedList &shard_list = lists[shard_index];
    for (size_t i = 0; i < count; ++i) {
        if (shard_list.empty()) {
            break;
        }
        LinkedListNode *tail_node = shard_list.getTail();
        if (tail_node == nullptr) {
            break;
        }
        auto *lru_node = static_cast<PromoteListNode *>(tail_node);
        if (lru_node->payload_ == nullptr) {
            shard_list.remove(tail_node);
            continue;
        }
        shard_list.unlink(tail_node);
        candidates.push_back({lru_node->payload_, lru_node, shard_index, protected_queue});
    }
}

void PromoteLruEvictionPolicy::ReturnCandidates(const std::vector<CandidateEntry> &candidates) {
    for (const auto &entry : candidates) {
        auto &lists = ListsForQueue(entry.protected_queue);
        lists[entry.shard_index].push_back(entry.node);
    }
}

void PromoteLruEvictionPolicy::CommitEviction(const std::vector<CandidateEntry> &candidates,
                                              std::vector<BlockEntry *> &evicted_blocks) {
    for (const auto &entry : candidates) {
        BlockEntry *block = entry.block;
        evicted_blocks.push_back(block);
        node_map_.erase(block);
        ClearBlockLocation(block);
        delete entry.node;
    }
}

size_t PromoteLruEvictionPolicy::EvictFromQueue(size_t count,
                                                bool protected_queue,
                                                std::vector<BlockEntry *> &evicted_blocks) {
    if (count == 0) {
        return 0;
    }

    auto &lists = ListsForQueue(protected_queue);
    size_t queue_size = 0;
    for (const auto &list : lists) {
        queue_size += list.size();
    }
    if (queue_size == 0) {
        return 0;
    }

    const size_t before = evicted_blocks.size();
    size_t target_count = std::min(count, queue_size);
    size_t amplified_count = static_cast<size_t>(amplification_factor_ * target_count);
    amplified_count = std::max(amplified_count, target_count);
    amplified_count = std::min(amplified_count, queue_size);

    int32_t num_rounds = std::min(sample_times_, shard_count_);
    size_t per_shard_count = (amplified_count + num_rounds - 1) / num_rounds;

    std::vector<CandidateEntry> candidates;
    for (int32_t round = 0; round < num_rounds; ++round) {
        int32_t oldest_shard = -1;
        int64_t oldest_time = INT64_MAX;
        for (int32_t s = 0; s < shard_count_; ++s) {
            if (lists[s].empty()) {
                continue;
            }
            int64_t tail_time = GetShardTailTime(lists, s);
            if (tail_time < oldest_time) {
                oldest_time = tail_time;
                oldest_shard = s;
            }
        }
        if (oldest_shard < 0) {
            break;
        }
        SampleFromShard(lists, oldest_shard, per_shard_count, protected_queue, candidates);
        if (candidates.size() >= amplified_count) {
            break;
        }
    }

    if (candidates.empty()) {
        return 0;
    }

    if (candidates.size() <= target_count) {
        CommitEviction(candidates, evicted_blocks);
    } else {
        std::partial_sort(candidates.begin(),
                          candidates.begin() + target_count,
                          candidates.end(),
                          [this](const CandidateEntry &a, const CandidateEntry &b) {
                              return GetTierAccessTime(a.block) < GetTierAccessTime(b.block);
                          });

        std::vector<CandidateEntry> to_evict(candidates.begin(), candidates.begin() + target_count);
        std::vector<CandidateEntry> to_return(candidates.begin() + target_count, candidates.end());

        CommitEviction(to_evict, evicted_blocks);
        ReturnCandidates(to_return);
    }

    return evicted_blocks.size() - before;
}

std::vector<BlockEntry *> PromoteLruEvictionPolicy::EvictBlocks(size_t count) {
    std::vector<BlockEntry *> evicted;
    evicted.reserve(std::min(count, node_map_.size()));
    const size_t probation_evicted = EvictFromQueue(count, false, evicted);
    if (probation_evicted < count) {
        EvictFromQueue(count - probation_evicted, true, evicted);
    }
    return evicted;
}

bool PromoteLruEvictionPolicy::RemoveBlock(BlockEntry *block) {
    auto it = node_map_.find(block);
    if (it == node_map_.end()) {
        return false;
    }
    auto *node = it->second;
    auto &lists = ListsForQueue(node->protected_queue);
    lists[GetShardIndex(block)].remove(node);
    node_map_.erase(it);
    ClearBlockLocation(block);
    return true;
}

void PromoteLruEvictionPolicy::ClearListLocations() {
    for (auto &[block, node] : node_map_) {
        (void)node;
        ClearBlockLocation(block);
    }
}

void PromoteLruEvictionPolicy::Clear() {
    ClearListLocations();
    for (auto &shard_list : probation_shard_lists_) {
        shard_list.clear();
    }
    for (auto &shard_list : protected_shard_lists_) {
        shard_list.clear();
    }
    node_map_.clear();
}

} // namespace kv_cache_manager
