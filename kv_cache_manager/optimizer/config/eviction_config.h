#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/config/types.h"
namespace kv_cache_manager {
struct LruParams : public Jsonizable {
    double sample_rate = 0.0;
    int32_t shard_count = 1024;
    int32_t sample_times = 32;
    double eviction_amplification_factor = 1.0;
    bool FromRapidValue(const rapidjson::Value &v) override {
        KVCM_JSON_GET_MACRO(v, "sample_rate", sample_rate);
        KVCM_JSON_GET_MACRO(v, "shard_count", shard_count);
        KVCM_JSON_GET_MACRO(v, "sample_times", sample_times);
        KVCM_JSON_GET_MACRO(v, "eviction_amplification_factor", eviction_amplification_factor);
        return true;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "sample_rate", sample_rate);
        Put(writer, "shard_count", shard_count);
        Put(writer, "sample_times", sample_times);
        Put(writer, "eviction_amplification_factor", eviction_amplification_factor);
    }
};
struct RandomLruParams : public Jsonizable {
    double sample_rate = 0.0;
    bool FromRapidValue(const rapidjson::Value &v) override {
        KVCM_JSON_GET_MACRO(v, "sample_rate", sample_rate);
        return true;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "sample_rate", sample_rate);
    }
};

struct TtlParams : public Jsonizable {
    bool fallback_on_pressure = true;
    bool FromRapidValue(const rapidjson::Value &v) override {
        KVCM_JSON_GET_MACRO(v, "fallback_on_pressure", fallback_on_pressure);
        return true;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "fallback_on_pressure", fallback_on_pressure);
    }
};

struct PromoteLruParams : public Jsonizable {
    double sample_rate = 0.0;
    int32_t shard_count = 1024;
    int32_t sample_times = 32;
    double eviction_amplification_factor = 1.0;
    double protected_queue_capacity_ratio = 1.0;
    int64_t ttl_seconds = 0;
    bool queue_monitor_enabled = false;
    int32_t queue_monitor_interval = 1000;
    std::vector<std::string> enabled_tiers;
    bool FromRapidValue(const rapidjson::Value &v) override {
        KVCM_JSON_GET_MACRO(v, "sample_rate", sample_rate);
        KVCM_JSON_GET_MACRO(v, "shard_count", shard_count);
        KVCM_JSON_GET_MACRO(v, "sample_times", sample_times);
        KVCM_JSON_GET_MACRO(v, "eviction_amplification_factor", eviction_amplification_factor);
        KVCM_JSON_GET_DEFAULT_MACRO(v, "protected_queue_capacity_ratio", protected_queue_capacity_ratio, 1.0);
        KVCM_JSON_GET_DEFAULT_MACRO(v, "ttl_seconds", ttl_seconds, static_cast<int64_t>(0));
        KVCM_JSON_GET_DEFAULT_MACRO(v, "queue_monitor_enabled", queue_monitor_enabled, false);
        KVCM_JSON_GET_DEFAULT_MACRO(v, "queue_monitor_interval", queue_monitor_interval, 1000);
        KVCM_JSON_GET_DEFAULT_MACRO(v, "enabled_tiers", enabled_tiers, std::vector<std::string>{});
        return true;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "sample_rate", sample_rate);
        Put(writer, "shard_count", shard_count);
        Put(writer, "sample_times", sample_times);
        Put(writer, "eviction_amplification_factor", eviction_amplification_factor);
        Put(writer, "protected_queue_capacity_ratio", protected_queue_capacity_ratio);
        Put(writer, "ttl_seconds", ttl_seconds);
        Put(writer, "queue_monitor_enabled", queue_monitor_enabled);
        Put(writer, "queue_monitor_interval", queue_monitor_interval);
        Put(writer, "enabled_tiers", enabled_tiers);
    }
};

struct CheckpointLruParams : public Jsonizable {
    bool evict_unreferenced_full_blocks_first = true;
    double score_value_bonus_seconds = 600.0;
    bool FromRapidValue(const rapidjson::Value &v) override {
        KVCM_JSON_GET_DEFAULT_MACRO(
            v, "evict_unreferenced_full_blocks_first", evict_unreferenced_full_blocks_first, true);
        KVCM_JSON_GET_DEFAULT_MACRO(v, "score_value_bonus_seconds", score_value_bonus_seconds, 600.0);
        return true;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "evict_unreferenced_full_blocks_first", evict_unreferenced_full_blocks_first);
        Put(writer, "score_value_bonus_seconds", score_value_bonus_seconds);
    }
};

using EvictionPolicyParam =
    std::variant<LruParams, RandomLruParams, TtlParams, PromoteLruParams, CheckpointLruParams>;

class EvictionConfig : public Jsonizable {
public:
    EvictionConfig() = default;
    ~EvictionConfig() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] EvictionMode eviction_mode() const { return eviction_mode_; }
    [[nodiscard]] int32_t eviction_batch_size_per_instance() const { return eviction_batch_size_per_instance_; }

    void set_eviction_mode(EvictionMode mode) { eviction_mode_ = mode; }
    void set_eviction_batch_size_per_instance(int32_t size) { eviction_batch_size_per_instance_ = size; }

private:
    EvictionMode eviction_mode_ = EvictionMode::EVICTION_MODE_UNSPECIFIED;
    int32_t eviction_batch_size_per_instance_ = 100;
};
} // namespace kv_cache_manager
