#include "kv_cache_manager/optimizer/eviction_policy/promote_lru.h"

#include <algorithm>
#include <climits>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

PromoteLruEvictionPolicy::PromoteLruEvictionPolicy(const std::string &name, const PromoteLruParams &params)
    : EvictionPolicy(name)
    , shard_count_(params.shard_count > 0 ? params.shard_count : 1)
    , sample_times_(params.sample_times > 0 ? params.sample_times : 1)
    , amplification_factor_(params.eviction_amplification_factor > 1.0 ? params.eviction_amplification_factor : 1.0)
    , promote_enabled_(PromoteEnabledForTier(params))
    , protected_queue_capacity_ratio_(std::max(0.0, std::min(params.protected_queue_capacity_ratio, 1.0)))
    , ttl_ns_(params.ttl_seconds > 0 ? params.ttl_seconds * 1000000000LL : 0)
    , queue_monitor_enabled_(params.queue_monitor_enabled)
    , queue_monitor_interval_(params.queue_monitor_interval > 0 ? static_cast<size_t>(params.queue_monitor_interval)
                                                                 : 1000)
    , probation_shard_lists_(shard_count_)
    , protected_shard_lists_(shard_count_) {
    KVCM_LOG_INFO("Promote LRU params tier=%s promote_enabled=%d protected_queue_capacity_ratio=%.6f "
                  "ttl_seconds=%ld queue_monitor_enabled=%d queue_monitor_interval=%zu",
                  EvictionPolicy::name().c_str(),
                  promote_enabled_,
                  protected_queue_capacity_ratio_,
                  params.ttl_seconds,
                  queue_monitor_enabled_,
                  queue_monitor_interval_);
}

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

bool PromoteLruEvictionPolicy::TtlEnabled() const { return ttl_ns_ > 0; }

void PromoteLruEvictionPolicy::RefreshTtl(BlockEntry *block, int64_t timestamp) {
    if (!TtlEnabled() || block == nullptr) {
        return;
    }
    int64_t anchor_time = timestamp;
    if (anchor_time < 0) {
        anchor_time = GetTierAccessTime(block);
    }
    if (anchor_time < 0 || anchor_time == INT64_MAX) {
        return;
    }
    if (anchor_time > last_known_timestamp_) {
        last_known_timestamp_ = anchor_time;
    }
    block->ttl_ns = ttl_ns_;
    block->ttl_anchor_time = anchor_time;
}

void PromoteLruEvictionPolicy::InsertNewBlock(BlockEntry *block) {
    if (block == nullptr) {
        return;
    }
    auto it = node_map_.find(block);
    if (it != node_map_.end()) {
        RefreshInCurrentQueue(it->second);
        RefreshTtl(block, GetTierAccessTime(block));
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
    RefreshTtl(block, GetTierAccessTime(block));
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
    (void)refresh_ttl_on_read;
    AdvanceClock(timestamp);
    auto it = node_map_.find(block);
    if (it == node_map_.end()) {
        return;
    }
    RefreshTtl(block, timestamp);
    PromoteOrRefresh(it->second);
}

void PromoteLruEvictionPolicy::OnBlockTouched(BlockEntry *block, int64_t timestamp) {
    AdvanceClock(timestamp);
    auto it = node_map_.find(block);
    if (it == node_map_.end()) {
        return;
    }
    RefreshTtl(block, timestamp);
    RefreshInCurrentQueue(it->second);
}

void PromoteLruEvictionPolicy::AdvanceClock(int64_t timestamp) {
    if (timestamp > last_known_timestamp_) {
        last_known_timestamp_ = timestamp;
    }
}

size_t PromoteLruEvictionPolicy::QueueSize(const std::vector<LinkedList> &lists) const {
    size_t total = 0;
    for (const auto &list : lists) {
        total += list.size();
    }
    return total;
}

size_t PromoteLruEvictionPolicy::ExternalQueueSize(const std::vector<LinkedList> &lists) const {
    size_t total = 0;
    for (const auto &list : lists) {
        auto *node = list.getHead();
        for (size_t i = 0; i < list.size() && node != nullptr; ++i, node = node->next) {
            auto *promote_node = static_cast<const PromoteListNode *>(node);
            if (promote_node->payload_ != nullptr && promote_node->payload_->owner_node == nullptr) {
                ++total;
            }
        }
    }
    return total;
}

void PromoteLruEvictionPolicy::MaybeLogQueueMonitor(const char *reason,
                                                    size_t max_resident_blocks,
                                                    size_t protected_limit,
                                                    size_t demoted_blocks,
                                                    size_t evicted_probation_blocks,
                                                    size_t evicted_protected_blocks) const {
    if (!queue_monitor_enabled_) {
        return;
    }
    const bool periodic = queue_monitor_interval_ > 0 && prepare_calls_ % queue_monitor_interval_ == 0;
    const bool early_demote = demoted_blocks > 0 && demote_calls_ <= 20;
    const bool early_eviction =
        (evicted_probation_blocks + evicted_protected_blocks) > 0 && evict_calls_ <= 20;
    const bool protected_eviction = evicted_protected_blocks > 0;
    if (!periodic && !early_demote && !early_eviction && !protected_eviction) {
        return;
    }

    const size_t probation_size = QueueSize(probation_shard_lists_);
    const size_t protected_size = QueueSize(protected_shard_lists_);
    const size_t probation_external = ExternalQueueSize(probation_shard_lists_);
    const size_t protected_external = ExternalQueueSize(protected_shard_lists_);
    KVCM_LOG_INFO("PROMOTE_LRU_MONITOR tier=%s reason=%s prepare_calls=%llu evict_calls=%llu "
                  "max_resident_blocks=%zu protected_limit=%zu probation=%zu protected=%zu "
                  "probation_external=%zu protected_external=%zu demoted_last=%zu "
                  "evicted_probation_last=%zu evicted_protected_last=%zu demote_calls=%llu "
                  "demoted_total=%llu evicted_probation_total=%llu evicted_protected_total=%llu "
                  "evicted_probation_external_total=%llu evicted_protected_external_total=%llu",
                  name().c_str(),
                  reason,
                  static_cast<unsigned long long>(prepare_calls_),
                  static_cast<unsigned long long>(evict_calls_),
                  max_resident_blocks,
                  protected_limit,
                  probation_size,
                  protected_size,
                  probation_external,
                  protected_external,
                  demoted_blocks,
                  evicted_probation_blocks,
                  evicted_protected_blocks,
                  static_cast<unsigned long long>(demote_calls_),
                  static_cast<unsigned long long>(demoted_blocks_),
                  static_cast<unsigned long long>(probation_evicted_blocks_),
                  static_cast<unsigned long long>(protected_evicted_blocks_),
                  static_cast<unsigned long long>(probation_external_evicted_blocks_),
                  static_cast<unsigned long long>(protected_external_evicted_blocks_));
}

void PromoteLruEvictionPolicy::DemoteOldestProtectedBlocks(size_t count) {
    size_t demoted = 0;
    for (size_t i = 0; i < count; ++i) {
        int32_t oldest_shard = -1;
        int64_t oldest_time = INT64_MAX;
        for (int32_t s = 0; s < shard_count_; ++s) {
            if (protected_shard_lists_[s].empty()) {
                continue;
            }
            int64_t tail_time = GetShardTailTime(protected_shard_lists_, s);
            if (tail_time < oldest_time) {
                oldest_time = tail_time;
                oldest_shard = s;
            }
        }
        if (oldest_shard < 0) {
            break;
        }

        LinkedListNode *tail_node = protected_shard_lists_[oldest_shard].getTail();
        if (tail_node == nullptr) {
            break;
        }
        auto *node = static_cast<PromoteListNode *>(tail_node);
        protected_shard_lists_[oldest_shard].unlink(node);
        node->protected_queue = false;
        probation_shard_lists_[GetShardIndex(node->payload_)].push_back(node);
        ++demoted;
    }
    if (demoted > 0) {
        ++demote_calls_;
        demoted_blocks_ += demoted;
    }
}

void PromoteLruEvictionPolicy::PrepareCapacityEviction(size_t max_resident_blocks) {
    ++prepare_calls_;
    if (!promote_enabled_ || protected_queue_capacity_ratio_ >= 1.0 || max_resident_blocks == 0) {
        MaybeLogQueueMonitor("prepare_skip", max_resident_blocks, 0, 0, 0, 0);
        return;
    }

    const size_t protected_limit = static_cast<size_t>(max_resident_blocks * protected_queue_capacity_ratio_);
    const size_t protected_size = QueueSize(protected_shard_lists_);
    if (protected_size <= protected_limit) {
        MaybeLogQueueMonitor("prepare_no_demote", max_resident_blocks, protected_limit, 0, 0, 0);
        return;
    }
    const size_t demote_count = protected_size - protected_limit;
    DemoteOldestProtectedBlocks(demote_count);
    MaybeLogQueueMonitor("prepare_demote", max_resident_blocks, protected_limit, demote_count, 0, 0);
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
        if (entry.protected_queue) {
            ++protected_evicted_blocks_;
            if (block != nullptr && block->owner_node == nullptr) {
                ++protected_external_evicted_blocks_;
            }
        } else {
            ++probation_evicted_blocks_;
            if (block != nullptr && block->owner_node == nullptr) {
                ++probation_external_evicted_blocks_;
            }
        }
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

size_t PromoteLruEvictionPolicy::EvictExpiredFromQueue(size_t count,
                                                       bool protected_queue,
                                                       std::vector<BlockEntry *> &evicted_blocks) {
    if (!TtlEnabled() || count == 0) {
        return 0;
    }

    auto &lists = ListsForQueue(protected_queue);
    const size_t before = evicted_blocks.size();
    while (evicted_blocks.size() - before < count) {
        int32_t oldest_expired_shard = -1;
        int64_t oldest_expire_time = INT64_MAX;
        for (int32_t s = 0; s < shard_count_; ++s) {
            if (lists[s].empty()) {
                continue;
            }
            auto *tail_node = static_cast<PromoteListNode *>(lists[s].getTail());
            if (tail_node == nullptr || tail_node->payload_ == nullptr) {
                continue;
            }
            BlockEntry *block = tail_node->payload_;
            if (!block->IsExpired(last_known_timestamp_)) {
                continue;
            }
            const int64_t expire_time = block->ttl_anchor_time + block->ttl_ns;
            if (expire_time < oldest_expire_time) {
                oldest_expire_time = expire_time;
                oldest_expired_shard = s;
            }
        }

        if (oldest_expired_shard < 0) {
            break;
        }

        auto *node = static_cast<PromoteListNode *>(lists[oldest_expired_shard].getTail());
        lists[oldest_expired_shard].unlink(node);
        std::vector<CandidateEntry> candidate{
            {node->payload_, node, oldest_expired_shard, protected_queue},
        };
        CommitEviction(candidate, evicted_blocks);
    }
    return evicted_blocks.size() - before;
}

std::vector<BlockEntry *> PromoteLruEvictionPolicy::EvictBlocks(size_t count) {
    ++evict_calls_;
    std::vector<BlockEntry *> evicted;
    evicted.reserve(std::min(count, node_map_.size()));
    const uint64_t before_probation_evicted = probation_evicted_blocks_;
    const uint64_t before_protected_evicted = protected_evicted_blocks_;
    if (evicted.size() < count) {
        EvictExpiredFromQueue(count - evicted.size(), false, evicted);
    }
    if (evicted.size() < count) {
        EvictExpiredFromQueue(count - evicted.size(), true, evicted);
    }
    if (evicted.size() < count) {
        EvictFromQueue(count - evicted.size(), false, evicted);
    }
    if (evicted.size() < count) {
        EvictFromQueue(count - evicted.size(), true, evicted);
    }
    MaybeLogQueueMonitor("evict",
                         0,
                         0,
                         0,
                         static_cast<size_t>(probation_evicted_blocks_ - before_probation_evicted),
                         static_cast<size_t>(protected_evicted_blocks_ - before_protected_evicted));
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
