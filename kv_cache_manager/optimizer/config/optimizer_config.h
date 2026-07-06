#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/instance_group_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
namespace kv_cache_manager {

enum class TraceReplayMode {
    READ_WRITE = 0,
    REQUEST = 1,
};

TraceReplayMode ToTraceReplayMode(const std::string &str);
bool IsValidTraceReplayMode(const std::string &str);
std::string ToString(const TraceReplayMode &mode);

enum class MambaCheckpointStrategy {
    CHUNK = 0,
    BRANCH = 1,
};

MambaCheckpointStrategy ToMambaCheckpointStrategy(const std::string &str);
bool IsValidMambaCheckpointStrategy(const std::string &str);
std::string ToString(const MambaCheckpointStrategy &strategy);

class OptTraceReplayConfig : public Jsonizable {
public:
    OptTraceReplayConfig() = default;
    ~OptTraceReplayConfig() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] TraceReplayMode mode() const { return mode_; }
    [[nodiscard]] int64_t write_delay_ns() const { return write_delay_ns_; }
    void set_mode(TraceReplayMode mode) { mode_ = mode; }
    void set_write_delay_ns(int64_t delay_ns) { write_delay_ns_ = delay_ns; }

private:
    TraceReplayMode mode_ = TraceReplayMode::READ_WRITE;
    int64_t write_delay_ns_ = 1;
};

class OptMambaStateConfig : public Jsonizable {
public:
    OptMambaStateConfig() = default;
    ~OptMambaStateConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] bool enabled() const { return enabled_; }
    [[nodiscard]] MambaCheckpointStrategy checkpoint_strategy() const { return checkpoint_strategy_; }
    [[nodiscard]] bool branch_save_request_end_checkpoint() const { return branch_save_request_end_checkpoint_; }
    [[nodiscard]] size_t chunk_size_blocks() const { return chunk_size_blocks_; }
    [[nodiscard]] size_t bytes_per_state() const { return bytes_per_state_; }
    [[nodiscard]] size_t group_count() const { return group_count_; }
    [[nodiscard]] size_t max_resident_checkpoints() const { return max_resident_checkpoints_; }

    void set_enabled(bool enabled) { enabled_ = enabled; }
    void set_checkpoint_strategy(MambaCheckpointStrategy strategy) { checkpoint_strategy_ = strategy; }
    void set_branch_save_request_end_checkpoint(bool enabled) { branch_save_request_end_checkpoint_ = enabled; }
    void set_chunk_size_blocks(size_t chunk_size_blocks) { chunk_size_blocks_ = chunk_size_blocks; }
    void set_bytes_per_state(size_t bytes_per_state) { bytes_per_state_ = bytes_per_state; }
    void set_group_count(size_t group_count) { group_count_ = group_count; }
    void set_max_resident_checkpoints(size_t max_resident_checkpoints) {
        max_resident_checkpoints_ = max_resident_checkpoints;
    }

private:
    bool enabled_ = false;
    MambaCheckpointStrategy checkpoint_strategy_ = MambaCheckpointStrategy::CHUNK;
    bool branch_save_request_end_checkpoint_ = false;
    size_t chunk_size_blocks_ = 0;
    size_t bytes_per_state_ = 0;
    size_t group_count_ = 1;
    size_t max_resident_checkpoints_ = 0;
};

class OptimizerConfig : public Jsonizable {
public:
    OptimizerConfig() = default;
    ~OptimizerConfig() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

public:
    [[nodiscard]] const std::string &trace_file_path() const { return trace_file_path_; }
    [[nodiscard]] const std::string &output_result_path() const { return output_result_path_; }
    [[nodiscard]] const EvictionConfig &eviction_config() const { return eviction_config_; }
    [[nodiscard]] const OptTraceReplayConfig &trace_replay_config() const { return trace_replay_config_; }
    [[nodiscard]] const OptMambaStateConfig &mamba_state_config() const { return mamba_state_config_; }
    [[nodiscard]] const std::vector<OptInstanceGroupConfig> &instance_groups() const { return instance_groups_; }
    [[nodiscard]] std::vector<OptInstanceGroupConfig> &mutable_instance_groups() { return instance_groups_; }

    void set_trace_file_path(const std::string &path) { trace_file_path_ = path; }
    void set_output_result_path(const std::string &path) { output_result_path_ = path; }
    void set_eviction_params(const EvictionConfig &config) { eviction_config_ = config; }
    void set_trace_replay_config(const OptTraceReplayConfig &config) { trace_replay_config_ = config; }
    void set_mamba_state_config(const OptMambaStateConfig &config) { mamba_state_config_ = config; }
    void set_instance_groups(const std::vector<OptInstanceGroupConfig> &groups) { instance_groups_ = groups; }

private:
    std::string trace_file_path_;
    std::string output_result_path_;
    EvictionConfig eviction_config_;
    OptTraceReplayConfig trace_replay_config_;
    OptMambaStateConfig mamba_state_config_;
    std::vector<OptInstanceGroupConfig> instance_groups_;
};

} // namespace kv_cache_manager
