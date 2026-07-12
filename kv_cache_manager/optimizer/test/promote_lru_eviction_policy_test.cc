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
    static constexpr int64_t kSecondNs = 1000000000LL;

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

TEST_F(PromoteLruEvictionPolicyTest, TtlEvictsExpiredProbationBeforeExpiredProtected) {
    auto params = DefaultParams();
    params.ttl_seconds = 10;
    PromoteLruEvictionPolicy policy("shared", params);
    auto probation_expired = MakeBlock(1, "shared", 0);
    auto protected_expired = MakeBlock(2, "shared", 0);
    auto probation_live = MakeBlock(3, "shared", 15 * kSecondNs);

    policy.OnBlockWritten(&probation_expired);
    policy.OnBlockWritten(&protected_expired);
    policy.OnBlockAccessedWithOptions(&protected_expired, 0, true);
    policy.OnBlockWritten(&probation_live);
    policy.AdvanceClock(15 * kSecondNs);

    auto evicted = policy.EvictBlocks(2);
    ASSERT_EQ(evicted.size(), 2);
    EXPECT_EQ(evicted[0]->key, 1);
    EXPECT_EQ(evicted[1]->key, 2);
    EXPECT_TRUE(probation_expired.location_map.empty());
    EXPECT_TRUE(protected_expired.location_map.empty());
    EXPECT_FALSE(probation_live.location_map.empty());
}

TEST_F(PromoteLruEvictionPolicyTest, TtlFallsBackToProbationThenProtectedLru) {
    auto params = DefaultParams();
    params.ttl_seconds = 10;
    PromoteLruEvictionPolicy policy("shared", params);
    auto probation_expired = MakeBlock(1, "shared", 0);
    auto probation_live = MakeBlock(2, "shared", 15 * kSecondNs);
    auto protected_live = MakeBlock(3, "shared", 16 * kSecondNs);

    policy.OnBlockWritten(&probation_expired);
    policy.OnBlockWritten(&probation_live);
    policy.OnBlockWritten(&protected_live);
    policy.OnBlockAccessedWithOptions(&protected_live, 16 * kSecondNs, true);
    policy.AdvanceClock(17 * kSecondNs);

    auto evicted = policy.EvictBlocks(3);
    ASSERT_EQ(evicted.size(), 3);
    EXPECT_EQ(evicted[0]->key, 1);
    EXPECT_EQ(evicted[1]->key, 2);
    EXPECT_EQ(evicted[2]->key, 3);
}

TEST_F(PromoteLruEvictionPolicyTest, TtlRefreshKeepsAccessedProtectedBlockAlive) {
    auto params = DefaultParams();
    params.ttl_seconds = 10;
    PromoteLruEvictionPolicy policy("shared", params);
    auto protected_refreshed = MakeBlock(1, "shared", 0);
    auto probation_expired = MakeBlock(2, "shared", 0);

    policy.OnBlockWritten(&protected_refreshed);
    policy.OnBlockAccessedWithOptions(&protected_refreshed, 8 * kSecondNs, true);
    policy.OnBlockWritten(&probation_expired);
    policy.AdvanceClock(11 * kSecondNs);

    auto evicted = policy.EvictBlocks(1);
    ASSERT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0]->key, 2);
    EXPECT_FALSE(protected_refreshed.location_map.empty());
}

TEST_F(PromoteLruEvictionPolicyTest, FactoryCreatesPromoteLruPolicy) {
    PromoteLruParams params;
    params.enabled_tiers = {"shared"};
    auto policy =
        EvictionPolicyFactory::CreatePolicy(EvictionPolicyType::POLICY_PROMOTE_LRU, "shared", 10, params);
    ASSERT_NE(policy, nullptr);
    EXPECT_EQ(policy->name(), "shared");
}
