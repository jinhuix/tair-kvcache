#include <fstream>
#include <memory>
#include <stdexcept>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/instance_config.h"
#include "kv_cache_manager/optimizer/config/instance_group_config.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/manager/optimizer_manager.h"

using namespace kv_cache_manager;

class OptimizerManagerTest : public TESTBASE {
public:
    void SetUp() override { config_ = CreateTestOptimizerConfig(); }

protected:
    OptimizerConfig CreateTestOptimizerConfig();
    OptimizerConfig config_;
};

OptimizerConfig OptimizerManagerTest::CreateTestOptimizerConfig() {
    OptimizerConfig config;
    config.set_trace_file_path("/tmp/test_trace.json");
    config.set_output_result_path("/tmp/test_result.json");

    EvictionConfig eviction_config;
    eviction_config.set_eviction_mode(EvictionMode::EVICTION_MODE_INSTANCE_PRECISE);
    eviction_config.set_eviction_batch_size_per_instance(10);
    config.set_eviction_params(eviction_config);
    // 创建实例组配置
    OptInstanceGroupConfig instance_group;
    instance_group.set_group_name("test_group");
    instance_group.set_quota_capacity(1024 * 1024 * 100); // 100MB
    instance_group.set_used_percentage(0.0);
    instance_group.set_hierarchical_eviction_enabled(false);

    OptTierConfig tier1;
    tier1.set_unique_name("tier1");
    tier1.set_capacity(1024 * 1024 * 10);
    tier1.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    tier1.set_band_width_mbps(1000);
    instance_group.set_storages({tier1});

    // 添加实例配置到实例组
    OptInstanceConfig instance1;
    instance1.set_instance_id("instance1");
    instance1.set_instance_group_name("test_group");
    instance1.set_block_size(1024);
    LruParams params;
    params.sample_rate = 1.0;
    EvictionPolicyParam policy_param;
    policy_param = params;
    instance1.set_eviction_policy_param(policy_param);
    instance1.set_eviction_policy_type(EvictionPolicyType::POLICY_LRU);

    instance_group.set_instances({instance1});

    config.set_instance_groups({instance_group});

    return config;
}

TEST_F(OptimizerManagerTest, BasicInitialization) {
    OptimizerManager manager(config_);
    EXPECT_TRUE(manager.Init());
}

TEST_F(OptimizerManagerTest, WriteCacheTtlSecondsUsesNanosecondTimestamps) {
    auto config = CreateTestOptimizerConfig();
    auto instance_groups = config.instance_groups();
    ASSERT_EQ(instance_groups.size(), 1);

    auto group = instance_groups[0];
    group.set_default_block_ttl_seconds(3600);
    auto instances = group.instances();
    ASSERT_EQ(instances.size(), 1);

    TtlParams ttl_params;
    ttl_params.fallback_on_pressure = false;
    instances[0].set_eviction_policy_type(EvictionPolicyType::POLICY_TTL);
    instances[0].set_eviction_policy_param(ttl_params);
    group.set_instances(instances);
    config.set_instance_groups({group});

    OptimizerManager manager(config);
    ASSERT_TRUE(manager.Init());

    const int64_t write_ts_ns = 1'000'000'000;
    manager.WriteCache("instance1", "write", write_ts_ns, {1}, 1);

    BlockMask remote_read_mask = std::vector<bool>{false};
    auto hit_before_expire = manager.GetCacheLocation(
        "instance1", "read_before_expire", write_ts_ns + 1'000'000, {1}, remote_read_mask, 1024);
    EXPECT_EQ(hit_before_expire.kvcm_hit_length, 1);

    auto hit_after_expire = manager.GetCacheLocation(
        "instance1", "read_after_expire", write_ts_ns + 2'000'000'000, {1}, remote_read_mask, 1024);
    EXPECT_EQ(hit_after_expire.kvcm_hit_length, 0);
}

TEST_F(OptimizerManagerTest, BatchGetMatchesIndexedBlocksWithoutPrefix) {
    OptimizerManager manager(config_);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> path = {21, 22};
    manager.WriteCache("instance1", "write_path", 1000, path);

    BlockMask remote_read_mask = std::vector<bool>{false};
    auto prefix_miss = manager.GetCacheLocation("instance1", "prefix_miss", 2000, {22}, remote_read_mask, 1024);
    EXPECT_EQ(prefix_miss.kvcm_hit_length, 0);

    auto batch_hit =
        manager.GetCacheLocation("instance1", "batch_hit", 3000, {22}, remote_read_mask, 1024, true, true, "batch_get");
    EXPECT_EQ(batch_hit.kvcm_hit_length, 1);
    ASSERT_EQ(batch_hit.hit_indices.size(), 1);
    EXPECT_EQ(batch_hit.hit_indices[0], 0);
}

TEST_F(OptimizerManagerTest, TemplateAnalysisReadRecordKeepsTraceIdAndKeys) {
    OptimizerManager manager(config_, false, true);
    ASSERT_TRUE(manager.Init());
    ASSERT_NE(manager.template_prefix_tracker_, nullptr);

    const std::vector<int64_t> keys = {1, 2, 3};
    manager.WriteCache("instance1", "write_trace", 1000, keys);

    BlockMask remote_read_mask = std::vector<bool>{false, false, false};
    manager.GetCacheLocation("instance1", "read_trace", 2000, keys, remote_read_mask, 3 * 1024);

    auto data_it = manager.template_prefix_tracker_->instance_data_.find("instance1");
    ASSERT_NE(data_it, manager.template_prefix_tracker_->instance_data_.end());
    ASSERT_EQ(data_it->second.trace_reads.size(), 1);

    EXPECT_EQ(data_it->second.trace_reads[0].trace_id, "read_trace");
    EXPECT_EQ(data_it->second.trace_reads[0].keys, keys);
}

TEST_F(OptimizerManagerTest, ReadUsesExplicitInputLen) {
    OptimizerManager manager(config_);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {1};
    manager.WriteCache("instance1", "write_trace", 1000, keys);

    BlockMask remote_read_mask = std::vector<bool>{false};
    auto res = manager.GetCacheLocation("instance1", "read_trace", 2000, keys, remote_read_mask, 1537);
    EXPECT_EQ(res.kvcm_hit_length, 1);

    const auto *last_read = manager.hit_rate_tracker_->LastReadRecord("instance1");
    ASSERT_NE(last_read, nullptr);
    EXPECT_EQ(last_read->input_tokens, 1537);
    EXPECT_EQ(last_read->remote_hit_blocks, 1);
}

TEST_F(OptimizerManagerTest, MambaStateCheckpointsGateOptimizerRunHits) {
    auto config = CreateTestOptimizerConfig();
    OptMambaStateConfig mamba_state;
    mamba_state.set_enabled(true);
    mamba_state.set_chunk_size_blocks(2);
    mamba_state.set_bytes_per_state(128);
    config.set_mamba_state_config(mamba_state);

    auto groups = config.instance_groups();
    ASSERT_EQ(groups.size(), 1);
    auto group = groups[0];
    group.set_quota_capacity(-1);
    group.set_used_percentage(1.0);
    auto instances = group.instances();
    ASSERT_EQ(instances.size(), 1);
    instances[0].set_block_size(16);
    instances[0].set_bytes_per_token(1);
    group.set_instances(instances);
    config.set_instance_groups({group});

    OptimizerManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> full_request = {1, 2, 3, 4, 5};
    manager.WriteCache("instance1", "write_full", 1000, full_request);

    BlockMask remote_read_mask = std::vector<bool>{false, false, false};
    auto partial_hit = manager.GetCacheLocation("instance1", "read_three", 2000, {1, 2, 3}, remote_read_mask, 48);
    EXPECT_EQ(partial_hit.kvcm_hit_length, 2);

    const auto *partial_record = manager.hit_rate_tracker_->LastReadRecord("instance1");
    ASSERT_NE(partial_record, nullptr);
    EXPECT_EQ(partial_record->remote_hit_blocks, 2);
    EXPECT_EQ(partial_record->mamba_state_candidate_blocks, 3);
    EXPECT_EQ(partial_record->mamba_state_hit_blocks, 2);

    BlockMask full_remote_read_mask = std::vector<bool>{false, false, false, false, false};
    auto full_hit = manager.GetCacheLocation("instance1", "read_full", 3000, full_request, full_remote_read_mask, 80);
    EXPECT_EQ(full_hit.kvcm_hit_length, 5);

    const auto *full_record = manager.hit_rate_tracker_->LastReadRecord("instance1");
    ASSERT_NE(full_record, nullptr);
    EXPECT_EQ(full_record->remote_hit_blocks, 5);
    EXPECT_EQ(full_record->mamba_state_candidate_blocks, 5);
    EXPECT_EQ(full_record->mamba_state_hit_blocks, 5);
}

TEST_F(OptimizerManagerTest, MambaStateBranchCheckpointUsesDeepestHistoricalPrefix) {
    auto config = CreateTestOptimizerConfig();
    OptMambaStateConfig mamba_state;
    mamba_state.set_enabled(true);
    mamba_state.set_checkpoint_strategy(MambaCheckpointStrategy::BRANCH);
    mamba_state.set_bytes_per_state(128);
    mamba_state.set_group_count(2);
    config.set_mamba_state_config(mamba_state);

    auto groups = config.instance_groups();
    ASSERT_EQ(groups.size(), 1);
    auto group = groups[0];
    group.set_quota_capacity(-1);
    group.set_used_percentage(1.0);
    auto instances = group.instances();
    ASSERT_EQ(instances.size(), 1);
    instances[0].set_block_size(16);
    instances[0].set_bytes_per_token(1);
    group.set_instances(instances);
    config.set_instance_groups({group});

    OptimizerManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> first_request = {1, 2, 3, 4};
    manager.WriteCache("instance1", "write_first", 1000, first_request);

    const std::vector<int64_t> branch_request = {1, 2, 9};
    manager.WriteCache("instance1", "write_branch", 2000, branch_request);

    BlockMask branch_mask = std::vector<bool>{false, false, false};
    auto branch_hit =
        manager.GetCacheLocation("instance1", "read_branch", 3000, branch_request, branch_mask, 48);
    EXPECT_EQ(branch_hit.kvcm_hit_length, 2);

    const auto *branch_read = manager.hit_rate_tracker_->LastReadRecord("instance1");
    ASSERT_NE(branch_read, nullptr);
    EXPECT_EQ(branch_read->mamba_state_candidate_blocks, 3);
    EXPECT_EQ(branch_read->mamba_state_hit_blocks, 2);

    manager.WriteCache("instance1", "write_repeat", 4000, first_request);

    BlockMask full_mask = std::vector<bool>{false, false, false, false};
    auto full_hit =
        manager.GetCacheLocation("instance1", "read_repeat", 5000, first_request, full_mask, 64);
    EXPECT_EQ(full_hit.kvcm_hit_length, 4);

    const auto *full_read = manager.hit_rate_tracker_->LastReadRecord("instance1");
    ASSERT_NE(full_read, nullptr);
    EXPECT_EQ(full_read->mamba_state_candidate_blocks, 4);
    EXPECT_EQ(full_read->mamba_state_hit_blocks, 4);
}

TEST_F(OptimizerManagerTest, MambaStateBranchCanAlsoSaveRequestEndCheckpoint) {
    auto config = CreateTestOptimizerConfig();
    OptMambaStateConfig mamba_state;
    mamba_state.set_enabled(true);
    mamba_state.set_checkpoint_strategy(MambaCheckpointStrategy::BRANCH);
    mamba_state.set_branch_save_request_end_checkpoint(true);
    mamba_state.set_bytes_per_state(128);
    mamba_state.set_group_count(2);
    config.set_mamba_state_config(mamba_state);

    auto groups = config.instance_groups();
    ASSERT_EQ(groups.size(), 1);
    auto group = groups[0];
    group.set_quota_capacity(-1);
    group.set_used_percentage(1.0);
    auto instances = group.instances();
    ASSERT_EQ(instances.size(), 1);
    instances[0].set_block_size(16);
    instances[0].set_bytes_per_token(1);
    group.set_instances(instances);
    config.set_instance_groups({group});

    OptimizerManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> first_request = {1, 2, 3};
    manager.WriteCache("instance1", "write_first", 1000, first_request);

    BlockMask remote_mask = std::vector<bool>{false, false, false};
    auto full_hit =
        manager.GetCacheLocation("instance1", "read_first", 2000, first_request, remote_mask, 48);
    EXPECT_EQ(full_hit.kvcm_hit_length, 3);

    const auto *read_record = manager.hit_rate_tracker_->LastReadRecord("instance1");
    ASSERT_NE(read_record, nullptr);
    EXPECT_EQ(read_record->mamba_state_candidate_blocks, 3);
    EXPECT_EQ(read_record->mamba_state_hit_blocks, 3);
}

TEST_F(OptimizerManagerTest, MambaStateResidentCheckpointsUseLruEviction) {
    auto config = CreateTestOptimizerConfig();
    OptMambaStateConfig mamba_state;
    mamba_state.set_enabled(true);
    mamba_state.set_chunk_size_blocks(2);
    mamba_state.set_bytes_per_state(128);
    mamba_state.set_group_count(3);
    mamba_state.set_max_resident_checkpoints(2);
    config.set_mamba_state_config(mamba_state);

    auto groups = config.instance_groups();
    ASSERT_EQ(groups.size(), 1);
    auto group = groups[0];
    group.set_quota_capacity(-1);
    group.set_used_percentage(1.0);
    auto instances = group.instances();
    ASSERT_EQ(instances.size(), 1);
    instances[0].set_block_size(16);
    instances[0].set_bytes_per_token(1);
    group.set_instances(instances);
    config.set_instance_groups({group});

    OptimizerManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> first_request = {1, 2, 3, 4};
    manager.WriteCache("instance1", "write_first", 1000, first_request);

    BlockMask first_two_mask = std::vector<bool>{false, false};
    auto hot_checkpoint =
        manager.GetCacheLocation("instance1", "touch_first_checkpoint", 2000, {1, 2}, first_two_mask, 32);
    EXPECT_EQ(hot_checkpoint.kvcm_hit_length, 2);

    manager.WriteCache("instance1", "write_second", 3000, {10, 11});

    BlockMask full_mask = std::vector<bool>{false, false, false, false};
    auto after_eviction =
        manager.GetCacheLocation("instance1", "read_first_after_eviction", 4000, first_request, full_mask, 64);
    EXPECT_EQ(after_eviction.kvcm_hit_length, 2);

    const auto *last_read = manager.hit_rate_tracker_->LastReadRecord("instance1");
    ASSERT_NE(last_read, nullptr);
    EXPECT_EQ(last_read->mamba_state_candidate_blocks, 4);
    EXPECT_EQ(last_read->mamba_state_hit_blocks, 2);
}

TEST_F(OptimizerManagerTest, CapacityEvictionReservesSpaceBeforeAdmission) {
    auto config = CreateTestOptimizerConfig();
    OptMambaStateConfig mamba_state;
    mamba_state.set_enabled(true);
    mamba_state.set_chunk_size_blocks(2);
    mamba_state.set_bytes_per_state(1);
    mamba_state.set_group_count(1);
    config.set_mamba_state_config(mamba_state);

    auto groups = config.instance_groups();
    ASSERT_EQ(groups.size(), 1);
    auto group = groups[0];
    group.set_quota_capacity(4);
    group.set_used_percentage(1.0);
    auto instances = group.instances();
    ASSERT_EQ(instances.size(), 1);
    instances[0].set_block_size(1);
    instances[0].set_bytes_per_token(1);
    PromoteLruParams promote_lru_params;
    instances[0].set_eviction_policy_type(EvictionPolicyType::POLICY_PROMOTE_LRU);
    instances[0].set_eviction_policy_param(promote_lru_params);
    group.set_instances(instances);
    config.set_instance_groups({group});

    OptimizerManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> hot_prefix = {1, 2};
    manager.WriteCache("instance1", "write_hot", 1000, hot_prefix);

    BlockMask hot_mask = std::vector<bool>{false, false};
    auto hot_hit = manager.GetCacheLocation("instance1", "read_hot", 2000, hot_prefix, hot_mask, 2);
    EXPECT_EQ(hot_hit.kvcm_hit_length, 2);

    manager.WriteCache("instance1", "write_cold", 3000, {10, 11});

    auto cold_after_admission =
        manager.GetCacheLocation("instance1", "read_cold_after_admission", 4000, {10, 11}, hot_mask, 2);
    EXPECT_EQ(cold_after_admission.kvcm_hit_length, 2);
}

TEST_F(OptimizerManagerTest, PromoteLruHotAdmitsBranchCheckpointAndPrefix) {
    auto config = CreateTestOptimizerConfig();
    OptMambaStateConfig mamba_state;
    mamba_state.set_enabled(true);
    mamba_state.set_checkpoint_strategy(MambaCheckpointStrategy::BRANCH);
    mamba_state.set_bytes_per_state(1);
    mamba_state.set_group_count(1);
    config.set_mamba_state_config(mamba_state);

    auto groups = config.instance_groups();
    ASSERT_EQ(groups.size(), 1);
    auto group = groups[0];
    group.set_quota_capacity(4);
    group.set_used_percentage(1.0);
    auto instances = group.instances();
    ASSERT_EQ(instances.size(), 1);
    instances[0].set_block_size(1);
    instances[0].set_bytes_per_token(1);
    PromoteLruParams promote_lru_params;
    promote_lru_params.shard_count = 1;
    promote_lru_params.sample_times = 1;
    instances[0].set_eviction_policy_type(EvictionPolicyType::POLICY_PROMOTE_LRU);
    instances[0].set_eviction_policy_param(promote_lru_params);
    group.set_instances(instances);
    config.set_instance_groups({group});

    OptimizerManager manager(config);
    ASSERT_TRUE(manager.Init());

    manager.WriteCache("instance1", "write_first", 1000, {1, 2, 3, 4});

    // The second request shares prefix {1, 2}. Its branch checkpoint is created during write.
    // Capacity pressure in the same write would evict {1, 2} under cold admission.
    manager.WriteCache("instance1", "write_branch", 2000, {1, 2, 9});

    BlockMask branch_mask = std::vector<bool>{false, false, false};
    auto branch_after_pressure =
        manager.GetCacheLocation("instance1", "read_branch_after_pressure", 3000, {1, 2, 9}, branch_mask, 3);
    EXPECT_EQ(branch_after_pressure.kvcm_hit_length, 2);

    const auto *last_read = manager.hit_rate_tracker_->LastReadRecord("instance1");
    ASSERT_NE(last_read, nullptr);
    EXPECT_EQ(last_read->mamba_state_candidate_blocks, 3);
    EXPECT_EQ(last_read->mamba_state_hit_blocks, 2);
}

TEST_F(OptimizerManagerTest, DirectRunTraceFilePreservesMambaStateAcrossWarmupReset) {
    auto config = CreateTestOptimizerConfig();
    const std::string warmup_trace = GetTestTempRootPath() + "/warmup_mamba_trace.jsonl";
    const std::string measure_trace = GetTestTempRootPath() + "/measure_mamba_trace.jsonl";
    config.set_trace_file_path(warmup_trace);

    OptMambaStateConfig mamba_state;
    mamba_state.set_enabled(true);
    mamba_state.set_chunk_size_blocks(2);
    mamba_state.set_bytes_per_state(1);
    mamba_state.set_group_count(1);
    config.set_mamba_state_config(mamba_state);

    auto groups = config.instance_groups();
    ASSERT_EQ(groups.size(), 1);
    auto group = groups[0];
    group.set_quota_capacity(6);
    group.set_used_percentage(1.0);
    auto instances = group.instances();
    ASSERT_EQ(instances.size(), 1);
    instances[0].set_block_size(1);
    instances[0].set_bytes_per_token(1);
    PromoteLruParams promote_lru_params;
    instances[0].set_eviction_policy_type(EvictionPolicyType::POLICY_PROMOTE_LRU);
    instances[0].set_eviction_policy_param(promote_lru_params);
    group.set_instances(instances);
    config.set_instance_groups({group});

    {
        std::ofstream out(warmup_trace);
        out << R"({"type":"write","instance_id":"instance1","trace_id":"write_hot","timestamp_ns":1000,"keys":[1,2]})"
            << "\n";
        out << R"({"type":"get","instance_id":"instance1","trace_id":"read_hot","timestamp_ns":2000,"keys":[1,2],"input_len":2,"block_mask":[false,false]})"
            << "\n";
    }
    {
        std::ofstream out(measure_trace);
        out << R"({"type":"write","instance_id":"instance1","trace_id":"write_cold","timestamp_ns":3000,"keys":[10,11]})"
            << "\n";
        out << R"({"type":"get","instance_id":"instance1","trace_id":"read_hot_after_reset","timestamp_ns":4000,"keys":[1,2],"input_len":2,"block_mask":[false,false]})"
            << "\n";
    }

    OptimizerManager manager(config);
    ASSERT_TRUE(manager.Init());
    manager.DirectRun();
    manager.ResetStats();
    manager.DirectRunTraceFile(measure_trace);

    const auto *last_read = manager.hit_rate_tracker_->LastReadRecord("instance1");
    ASSERT_NE(last_read, nullptr);
    EXPECT_EQ(last_read->trace_id, "read_hot_after_reset");
    EXPECT_EQ(last_read->remote_hit_blocks, 2);
    EXPECT_EQ(last_read->mamba_state_candidate_blocks, 2);
    EXPECT_EQ(last_read->mamba_state_hit_blocks, 2);
}

TEST_F(OptimizerManagerTest, ReadRejectsPartialTailBlockKeys) {
    OptimizerManager manager(config_);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {1, 2};
    manager.WriteCache("instance1", "write_trace", 1000, keys);

    BlockMask remote_read_mask = std::vector<bool>{false, false};
    EXPECT_THROW(manager.GetCacheLocation("instance1", "read_trace", 2000, keys, remote_read_mask, 1537),
                 std::runtime_error);
}

TEST_F(OptimizerManagerTest, ReadWithoutFullBlocksCountsInputTokens) {
    OptimizerManager manager(config_);
    ASSERT_TRUE(manager.Init());

    BlockMask remote_read_mask = std::vector<bool>{};
    auto res = manager.GetCacheLocation("instance1", "short_read", 1000, {}, remote_read_mask, 128);
    EXPECT_EQ(res.kvcm_hit_length, 0);

    const auto *last_read = manager.hit_rate_tracker_->LastReadRecord("instance1");
    ASSERT_NE(last_read, nullptr);
    EXPECT_EQ(last_read->input_tokens, 128);
    EXPECT_EQ(last_read->local_read_blocks, 0);
    EXPECT_EQ(last_read->remote_read_blocks, 0);
    EXPECT_EQ(last_read->local_hit_blocks, 0);
    EXPECT_EQ(last_read->remote_hit_blocks, 0);
}

TEST_F(OptimizerManagerTest, RequestTraceSchedulesDelayedWrite) {
    auto config = CreateTestOptimizerConfig();
    config.set_trace_file_path(GetTestTempRootPath() + "/request_trace.jsonl");
    config.set_output_result_path(GetTestTempRootPath() + "/request_trace_result");

    OptTraceReplayConfig trace_replay_config;
    trace_replay_config.set_mode(TraceReplayMode::REQUEST);
    trace_replay_config.set_write_delay_ns(1000);
    config.set_trace_replay_config(trace_replay_config);

    auto groups = config.instance_groups();
    ASSERT_EQ(groups.size(), 1);
    auto group = groups[0];
    group.set_quota_capacity(-1);
    group.set_used_percentage(1.0);
    auto instances = group.instances();
    ASSERT_EQ(instances.size(), 1);
    instances[0].set_block_size(16);
    instances[0].set_bytes_per_token(1);
    group.set_instances(instances);
    config.set_instance_groups({group});

    std::ofstream out(config.trace_file_path());
    out << R"({"type":"request","instance_id":"instance1","trace_id":"r1","timestamp_ns":1000,"keys":[1],"input_len":16,"block_mask":[]})"
        << "\n";
    out << R"({"type":"request","instance_id":"instance1","trace_id":"r2","timestamp_ns":1500,"keys":[1],"input_len":16,"block_mask":[]})"
        << "\n";
    out << R"({"type":"request","instance_id":"instance1","trace_id":"r3","timestamp_ns":2000,"keys":[1],"input_len":16,"block_mask":[]})"
        << "\n";
    out.close();

    OptimizerManager manager(config);
    ASSERT_TRUE(manager.Init());
    manager.DirectRun();

    auto data_it = manager.hit_rate_tracker_->instance_data_.find("instance1");
    ASSERT_NE(data_it, manager.hit_rate_tracker_->instance_data_.end());
    ASSERT_EQ(data_it->second.read_records.size(), 3);
    ASSERT_EQ(data_it->second.write_records.size(), 3);

    EXPECT_EQ(data_it->second.read_records[0].remote_hit_blocks, 0);
    EXPECT_EQ(data_it->second.read_records[1].remote_hit_blocks, 0);
    EXPECT_EQ(data_it->second.read_records[2].remote_hit_blocks, 1);
    EXPECT_EQ(data_it->second.write_records[0].timestamp_ns, 2000);
    EXPECT_EQ(data_it->second.write_records[1].timestamp_ns, 2500);
    EXPECT_EQ(data_it->second.write_records[2].timestamp_ns, 3000);
}
