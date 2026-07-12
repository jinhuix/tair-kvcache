#include <algorithm>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/eviction_policy/checkpoint_lru.h"

using namespace kv_cache_manager;

class CheckpointLruEvictionPolicyTest : public TESTBASE {
protected:
    static BlockEntry FullBlock(int64_t key, BlockEntry *parent = nullptr) {
        BlockEntry block;
        block.key = key;
        block.owner_node = reinterpret_cast<RadixTreeNode *>(1);
        block.prefix_parent = parent;
        block.location_map["shared"] = TierStat{};
        return block;
    }

    static BlockEntry MambaBlock(int64_t key) {
        BlockEntry block;
        block.key = key;
        block.location_map["shared"] = TierStat{};
        return block;
    }

    static bool Contains(const std::vector<BlockEntry *> &blocks, const BlockEntry *target) {
        return std::find(blocks.begin(), blocks.end(), target) != blocks.end();
    }
};

TEST_F(CheckpointLruEvictionPolicyTest, EvictsCompleteCheckpointAtomically) {
    CheckpointLruEvictionPolicy policy("shared", CheckpointLruParams{});
    auto full1 = FullBlock(1);
    auto full2 = FullBlock(2, &full1);
    auto state1 = MambaBlock(101);
    auto state2 = MambaBlock(102);
    for (auto *block : std::vector<BlockEntry *>{&full1, &full2, &state1, &state2}) {
        policy.OnBlockWritten(block);
    }

    ASSERT_TRUE(policy.RegisterCheckpoint(1, &full2, {&state1, &state2}, 10));
    EXPECT_EQ(full1.checkpoint_ref_count, 1);
    EXPECT_EQ(full2.checkpoint_ref_count, 1);

    const auto evicted = policy.EvictBlocks(1);
    ASSERT_EQ(evicted.size(), 4);
    EXPECT_TRUE(Contains(evicted, &state1));
    EXPECT_TRUE(Contains(evicted, &state2));
    EXPECT_TRUE(Contains(evicted, &full1));
    EXPECT_TRUE(Contains(evicted, &full2));
    EXPECT_EQ(policy.size(), 0);
    EXPECT_EQ(policy.checkpoint_count(), 0);
}
TEST_F(CheckpointLruEvictionPolicyTest, SharedPrefixSurvivesUntilLastCheckpointIsEvicted) {
    CheckpointLruEvictionPolicy policy("shared", CheckpointLruParams{});
    auto full1 = FullBlock(1);
    auto full2 = FullBlock(2, &full1);
    auto full3 = FullBlock(3, &full2);
    auto full4 = FullBlock(4, &full3);
    auto shallow_state = MambaBlock(101);
    auto deep_state = MambaBlock(201);
    for (auto *block :
         std::vector<BlockEntry *>{&full1, &full2, &full3, &full4, &shallow_state, &deep_state}) {
        policy.OnBlockWritten(block);
    }
    ASSERT_TRUE(policy.RegisterCheckpoint(1, &full2, {&shallow_state}, 10));
    ASSERT_TRUE(policy.RegisterCheckpoint(2, &full4, {&deep_state}, 20));

    auto first = policy.EvictBlocks(1);
    ASSERT_EQ(first.size(), 3);
    EXPECT_TRUE(Contains(first, &deep_state));
    EXPECT_TRUE(Contains(first, &full3));
    EXPECT_TRUE(Contains(first, &full4));
    EXPECT_FALSE(full1.location_map.empty());
    EXPECT_FALSE(full2.location_map.empty());
    EXPECT_EQ(full1.checkpoint_ref_count, 1);
    EXPECT_EQ(full2.checkpoint_ref_count, 1);

    auto second = policy.EvictBlocks(1);
    ASSERT_EQ(second.size(), 3);
    EXPECT_TRUE(Contains(second, &shallow_state));
    EXPECT_TRUE(Contains(second, &full1));
    EXPECT_TRUE(Contains(second, &full2));
    EXPECT_EQ(policy.size(), 0);
}

TEST_F(CheckpointLruEvictionPolicyTest, HotCheckpointCanOutrankLowerCostCheckpoint) {
    CheckpointLruEvictionPolicy policy("shared", CheckpointLruParams{});
    auto full1 = FullBlock(1);
    auto full2 = FullBlock(2, &full1);
    auto full3 = FullBlock(3, &full2);
    auto full4 = FullBlock(4, &full3);
    auto shallow_state = MambaBlock(101);
    auto deep_state = MambaBlock(201);
    for (auto *block :
         std::vector<BlockEntry *>{&full1, &full2, &full3, &full4, &shallow_state, &deep_state}) {
        policy.OnBlockWritten(block);
    }
    ASSERT_TRUE(policy.RegisterCheckpoint(1, &full2, {&shallow_state}, 10));
    ASSERT_TRUE(policy.RegisterCheckpoint(2, &full4, {&deep_state}, 20));
    ASSERT_TRUE(policy.TouchCheckpoint(2, 30));

    const auto evicted = policy.EvictBlocks(1);
    ASSERT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0], &shallow_state);
    EXPECT_TRUE(policy.HasCheckpoint(2));
}

TEST_F(CheckpointLruEvictionPolicyTest, WrittenTailIsKeptThenReclaimedFirstUnderPressure) {
    CheckpointLruEvictionPolicy policy("shared", CheckpointLruParams{});
    auto full1 = FullBlock(1);
    auto full2 = FullBlock(2, &full1);
    auto tail = FullBlock(3, &full2);
    auto state = MambaBlock(101);
    for (auto *block : std::vector<BlockEntry *>{&full1, &full2, &tail, &state}) {
        policy.OnBlockWritten(block);
    }
    ASSERT_TRUE(policy.RegisterCheckpoint(1, &full2, {&state}, 10));
    ASSERT_EQ(policy.unreferenced_full_block_count(), 1);
    EXPECT_FALSE(tail.location_map.empty());

    const auto evicted = policy.EvictBlocks(1);
    ASSERT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0], &tail);
    EXPECT_TRUE(policy.HasCheckpoint(1));
    EXPECT_FALSE(full1.location_map.empty());
    EXPECT_FALSE(state.location_map.empty());
}
