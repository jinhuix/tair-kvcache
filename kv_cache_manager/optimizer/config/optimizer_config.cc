#include "kv_cache_manager/optimizer/config/optimizer_config.h"

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
TraceReplayMode ToTraceReplayMode(const std::string &str) {
    if (str == "request") {
        return TraceReplayMode::REQUEST;
    }
    return TraceReplayMode::READ_WRITE;
}

bool IsValidTraceReplayMode(const std::string &str) { return str == "read_write" || str == "request"; }

std::string ToString(const TraceReplayMode &mode) {
    switch (mode) {
    case TraceReplayMode::REQUEST:
        return "request";
    case TraceReplayMode::READ_WRITE:
    default:
        return "read_write";
    }
}

MambaCheckpointStrategy ToMambaCheckpointStrategy(const std::string &str) {
    if (str == "branch") {
        return MambaCheckpointStrategy::BRANCH;
    }
    return MambaCheckpointStrategy::CHUNK;
}

bool IsValidMambaCheckpointStrategy(const std::string &str) { return str == "chunk" || str == "branch"; }

std::string ToString(const MambaCheckpointStrategy &strategy) {
    switch (strategy) {
    case MambaCheckpointStrategy::BRANCH:
        return "branch";
    case MambaCheckpointStrategy::CHUNK:
    default:
        return "chunk";
    }
}

bool OptTraceReplayConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    std::string mode_str = "read_write";
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "mode", mode_str, std::string("read_write"));
    if (!IsValidTraceReplayMode(mode_str)) {
        return false;
    }
    mode_ = ToTraceReplayMode(mode_str);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "write_delay_ns", write_delay_ns_, int64_t(1));
    if (write_delay_ns_ <= 0) {
        KVCM_LOG_ERROR("trace_replay.write_delay_ns must be positive, got %ld", write_delay_ns_);
        return false;
    }
    return true;
}

void OptTraceReplayConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "mode", ToString(mode_));
    Put(writer, "write_delay_ns", write_delay_ns_);
}

bool OptMambaStateConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (!rapid_value.IsObject()) {
        KVCM_LOG_ERROR("mamba_state must be an object");
        return false;
    }

    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "enabled", enabled_, true);
    if (!enabled_) {
        return true;
    }

    std::string checkpoint_strategy = "chunk";
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "checkpoint_strategy", checkpoint_strategy, std::string("chunk"));
    if (!IsValidMambaCheckpointStrategy(checkpoint_strategy)) {
        KVCM_LOG_ERROR("mamba_state.checkpoint_strategy must be chunk or branch");
        return false;
    }
    checkpoint_strategy_ = ToMambaCheckpointStrategy(checkpoint_strategy);
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "branch_save_request_end_checkpoint", branch_save_request_end_checkpoint_, false);

    uint64_t chunk_size_blocks = 0;
    if (rapid_value.HasMember("chunk_size_blocks")) {
        KVCM_JSON_GET_MACRO(rapid_value, "chunk_size_blocks", chunk_size_blocks);
    } else if (rapid_value.HasMember("chunk_size")) {
        KVCM_JSON_GET_MACRO(rapid_value, "chunk_size", chunk_size_blocks);
    } else if (checkpoint_strategy_ == MambaCheckpointStrategy::CHUNK) {
        KVCM_LOG_ERROR("mamba_state requires positive chunk_size_blocks");
        return false;
    }
    if (checkpoint_strategy_ == MambaCheckpointStrategy::CHUNK && chunk_size_blocks == 0) {
        KVCM_LOG_ERROR("mamba_state.chunk_size_blocks must be positive");
        return false;
    }
    chunk_size_blocks_ = static_cast<size_t>(chunk_size_blocks);

    uint64_t bytes_per_state = 0;
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "bytes_per_state", bytes_per_state, uint64_t(0));
    bytes_per_state_ = static_cast<size_t>(bytes_per_state);

    uint64_t group_count = 1;
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "group_count", group_count, uint64_t(1));
    if (group_count == 0) {
        KVCM_LOG_ERROR("mamba_state.group_count must be positive");
        return false;
    }
    group_count_ = static_cast<size_t>(group_count);

    uint64_t max_resident_checkpoints = 0;
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "max_resident_checkpoints", max_resident_checkpoints, uint64_t(0));
    max_resident_checkpoints_ = static_cast<size_t>(max_resident_checkpoints);
    return true;
}

void OptMambaStateConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "enabled", enabled_);
    if (!enabled_) {
        return;
    }
    Put(writer, "checkpoint_strategy", ToString(checkpoint_strategy_));
    Put(writer, "branch_save_request_end_checkpoint", branch_save_request_end_checkpoint_);
    Put(writer, "chunk_size_blocks", static_cast<uint64_t>(chunk_size_blocks_));
    Put(writer, "bytes_per_state", static_cast<uint64_t>(bytes_per_state_));
    Put(writer, "group_count", static_cast<uint64_t>(group_count_));
    Put(writer, "max_resident_checkpoints", static_cast<uint64_t>(max_resident_checkpoints_));
}

bool OptimizerConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "trace_file_path", trace_file_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "output_result_path", output_result_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "eviction_params", eviction_config_);
    trace_replay_config_ = OptTraceReplayConfig();
    if (rapid_value.HasMember("trace_replay")) {
        if (!rapid_value["trace_replay"].IsObject()) {
            KVCM_LOG_ERROR("trace_replay must be an object");
            return false;
        }
        if (!trace_replay_config_.FromRapidValue(rapid_value["trace_replay"])) {
            return false;
        }
    }
    mamba_state_config_ = OptMambaStateConfig();
    if (rapid_value.HasMember("mamba_state")) {
        if (!mamba_state_config_.FromRapidValue(rapid_value["mamba_state"])) {
            return false;
        }
    }
    KVCM_JSON_GET_MACRO(rapid_value, "instance_groups", instance_groups_);
    return true;
};

void OptimizerConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "trace_file_path", trace_file_path_);
    Put(writer, "output_result_path", output_result_path_);
    Put(writer, "eviction_params", eviction_config_);
    Put(writer, "trace_replay", trace_replay_config_);
    if (mamba_state_config_.enabled()) {
        Put(writer, "mamba_state", mamba_state_config_);
    }
    Put(writer, "instance_groups", instance_groups_);
}
} // namespace kv_cache_manager
