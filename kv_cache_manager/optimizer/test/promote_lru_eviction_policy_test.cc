#include <memory>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/policy_factory.h"
#include "kv_cache_manager/optimizer/eviction_policy/promote_lru.h"

using namespace kv_cache_manager;

class PromoteLruEvictionPolicyTest : public TESTBASE {
protected:
    PromoteLruParams DefaultParams() {
        PromoteLruParams params;
        params.shard_count = 1;
        params.sample_times = 1;
        return params;
    }

    BlockEntry MakeBlock(int64_t key, const std::string &tier_name = "shared", int64_t timestamp = 0) {
        BlockEntry block;
        block.key = key;
        block.last_access_time = timestamp;
        block.writing_time = timestamp;
        block.location_map[tier_name] = TierStat{0, timestamp, timestamp};
        return block;
    }
};

TEST_F(PromoteLruEvictionPolicyTest, EvictsProbationBeforeProtected) {
    PromoteLruEvictionPolicy policy("shared", DefaultParams());
    auto block1 = MakeBlock(1);
    auto block2 = MakeBlock(2);
    auto block3 = MakeBlock(3);

    policy.OnBlockWritten(&block1);
    policy.OnBlockWritten(&block2);
    policy.OnBlockAccessedWithOptions(&block1, 100, true);
    policy.OnBlockWritten(&block3);

    auto evicted = policy.EvictBlocks(2);
    ASSERT_EQ(evicted.size(), 2);
    EXPECT_EQ(evicted[0]->key, 2);
    EXPECT_EQ(evicted[1]->key, 3);
    EXPECT_EQ(policy.size(), 1);
    EXPECT_FALSE(block1.location_map.empty());
}

TEST_F(PromoteLruEvictionPolicyTest, TouchDoesNotPromoteProbationBlock) {
    PromoteLruEvictionPolicy policy("shared", DefaultParams());
    auto block1 = MakeBlock(1);
    auto block2 = MakeBlock(2);
    auto block3 = MakeBlock(3);

    policy.OnBlockWritten(&block1);
    policy.OnBlockWritten(&block2);
    policy.OnBlockWritten(&block3);
    policy.OnBlockAccessedWithOptions(&block1, 100, true);
    policy.OnBlockTouched(&block2, 200);

    auto evicted = policy.EvictBlocks(2);
    ASSERT_EQ(evicted.size(), 2);
    EXPECT_EQ(evicted[0]->key, 3);
    EXPECT_EQ(evicted[1]->key, 2);
    EXPECT_FALSE(block1.location_map.empty());
}

TEST_F(PromoteLruEvictionPolicyTest, ReinsertAfterEvictionStartsInProbationAgain) {
    PromoteLruEvictionPolicy policy("shared", DefaultParams());
    auto block1 = MakeBlock(1);
    auto block2 = MakeBlock(2);

    policy.OnBlockWritten(&block1);
    policy.OnBlockAccessedWithOptions(&block1, 100, true);
    auto first_evicted = policy.EvictBlocks(1);
    ASSERT_EQ(first_evicted.size(), 1);
    EXPECT_EQ(first_evicted[0]->key, 1);
    EXPECT_TRUE(block1.location_map.empty());

    block1.location_map["shared"] = TierStat{0, 200, 200};
    policy.OnBlockWritten(&block1);
    policy.OnBlockWritten(&block2);
    policy.OnBlockAccessedWithOptions(&block2, 300, true);

    auto second_evicted = policy.EvictBlocks(1);
    ASSERT_EQ(second_evicted.size(), 1);
    EXPECT_EQ(second_evicted[0]->key, 1);
}

TEST_F(PromoteLruEvictionPolicyTest, DisabledTierFallsBackToPlainLru) {
    PromoteLruParams params;
    params.enabled_tiers = {"l2"};
    PromoteLruEvictionPolicy policy("l1", params);
    auto block1 = MakeBlock(1, "l1");
    auto block2 = MakeBlock(2, "l1");

    policy.OnBlockWritten(&block1);
    policy.OnBlockAccessedWithOptions(&block1, 100, true);
    policy.OnBlockWritten(&block2);

    auto evicted = policy.EvictBlocks(1);
    ASSERT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0]->key, 1);
    EXPECT_TRUE(block1.location_map.empty());
    EXPECT_FALSE(block2.location_map.empty());
}

TEST_F(PromoteLruEvictionPolicyTest, FactoryCreatesPromoteLruPolicy) {
    PromoteLruParams params;
    params.enabled_tiers = {"shared"};
    auto policy =
        EvictionPolicyFactory::CreatePolicy(EvictionPolicyType::POLICY_PROMOTE_LRU, "shared", 10, params);
    ASSERT_NE(policy, nullptr);
    EXPECT_EQ(policy->name(), "shared");
}
