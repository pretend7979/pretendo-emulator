// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <string_view>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::Nx2WriteProfile {

// Metadata and requested dirty bytes only. No guest reads, readbacks or range changes.
inline const bool g_enabled = [] {
    const char* value = std::getenv("EDEN_NX2_WRITE_PROFILE");
    return std::getenv("EDEN_NX2_GPU_PROFILE") && value && std::string_view{value} == "1";
}();
inline bool Enabled() { return g_enabled; }

enum class Kind : u8 {
    Unknown, GraphicsStorage, ComputeStorage, GraphicsSegmented, ComputeSegmented,
    GraphicsTexture, ComputeTexture, TransformFeedback, DmaMirror,
};
enum class Proof : u8 { None, Fixed, UniformUnvalidated, UniformSnapshot };
inline constexpr std::array kind_names{"unknown", "graphics SSBO", "compute SSBO",
    "graphics segmented", "compute segmented", "graphics texture buffer",
    "compute texture buffer", "transform feedback", "DMA mirror"};
inline constexpr std::array proof_names{"none", "fixed metadata", "uniform unvalidated",
    "uniform snapshot"};

struct Key {
    u64 canonical{}, auxiliary{}, native{};
    u32 stage{~0U}, index{~0U}, cbuf{~0U}, offset{~0U}, child{~0U};
    u16 end{};
    u8 proof_count{};
    Kind kind{Kind::Unknown};
    Proof proof{Proof::None};
    bool global_fallback{};
    bool metadata_available{};
    bool operator==(const Key&) const = default;
};

struct Stage {
    Key identity;
    std::array<Key, 64> storage{};
    size_t count{};
};

// Info is reflected shader metadata; this helper never evaluates a proof or reads a cbuf.
template <typename Info>
Stage MakeStage(const Info& info, u64 canonical, u64 native, u32 stage, u64 auxiliary = 0,
                const u16* validated_ends = nullptr, size_t validated_count = 0) {
    Stage result;
    result.identity.canonical = canonical;
    result.identity.auxiliary = auxiliary;
    result.identity.native = native;
    result.identity.stage = stage;
    result.identity.global_fallback = info.stores_global_memory;
    result.identity.metadata_available = true;
    for (const auto& desc : info.storage_buffers_descriptors) {
        if (result.count == result.storage.size()) break;
        const size_t index = result.count++;
        auto& key = result.storage[index];
        key = result.identity;
        key.index = static_cast<u32>(index);
        key.cbuf = desc.cbuf_index;
        key.offset = desc.cbuf_offset;
        key.child = desc.pointer_offset;
        for (const auto& proof : info.uniform_storage_write_proofs) {
            if (proof.root_cbuf_index == desc.cbuf_index && proof.root_cbuf_offset == desc.cbuf_offset &&
                key.proof_count != (std::numeric_limits<u8>::max)()) ++key.proof_count;
        }
        if (!desc.is_written || desc.count != 1 || desc.pointer_offset != ~0U ||
            info.stores_global_memory) continue;
        if (desc.has_uniform_write_proofs) {
            key.proof = Proof::UniformUnvalidated;
            if (validated_ends && index < validated_count && validated_ends[index] && key.proof_count) {
                key.proof = Proof::UniformSnapshot;
                key.end = validated_ends[index];
            }
        } else if (desc.fixed_write_end) {
            key.proof = Proof::Fixed;
            key.end = desc.fixed_write_end;
        }
    }
    return result;
}

inline Key OtherKey(const Stage* stage, Kind kind, u32 index = ~0U) {
    Key key = stage ? stage->identity : Key{};
    key.kind = kind;
    key.index = index;
    return key;
}

inline Key StorageKey(const Stage* stage, u32 index, Kind kind, bool snapshot_valid = false) {
    Key key = stage && index < stage->count && index < stage->storage.size()
        ? stage->storage[index] : Key{};
    key.kind = kind;
    key.index = index;
    if (key.proof == Proof::UniformSnapshot && !snapshot_valid) {
        key.proof = Proof::UniformUnvalidated;
        key.end = 0;
    }
    return key;
}

inline void AddSaturated(u64& value, u64 amount) {
    value = amount > (std::numeric_limits<u64>::max)() - value
        ? (std::numeric_limits<u64>::max)() : value + amount;
}

struct Totals {
    u64 calls{}, requested{}, max_mark{}, max_binding{}, max_owner{}, proof_calls{}, proof_prefix{};
    void Add(u64 bytes, u64 binding, u64 owner, u16 end) {
        AddSaturated(calls, 1);
        AddSaturated(requested, bytes);
        max_mark = (std::max)(max_mark, bytes);
        max_binding = (std::max)(max_binding, binding);
        max_owner = (std::max)(max_owner, owner);
        if (end && end <= binding && end <= bytes) {
            AddSaturated(proof_calls, 1);
            AddSaturated(proof_prefix, end);
        }
    }
};
struct Row { Key key; Totals totals; bool occupied{}; };
struct Table {
    std::array<Row, 256> rows{};
    Totals total, overflow;
    size_t used{};
    static size_t Hash(const Key& key) {
        u64 value = 1469598103934665603ULL;
        const auto add = [&](u64 word) { value = (value ^ word) * 1099511628211ULL; };
        add(key.canonical); add(key.auxiliary); add(key.native); add(key.stage); add(key.index);
        add(key.cbuf); add(key.offset); add(key.child); add(key.end); add(key.proof_count);
        add(static_cast<u8>(key.kind)); add(static_cast<u8>(key.proof)); add(key.global_fallback);
        add(key.metadata_available);
        return static_cast<size_t>(value & 255);
    }
    void Record(Key key, u64 bytes, u64 binding, u64 owner) {
        if (static_cast<size_t>(key.kind) >= kind_names.size()) key.kind = Kind::Unknown;
        if (static_cast<size_t>(key.proof) >= proof_names.size()) key.proof = Proof::None;
        if (key.proof != Proof::Fixed && key.proof != Proof::UniformSnapshot) key.end = 0;
        if (key.end > binding || key.end > bytes) { key.end = 0; key.proof = Proof::None; }
        total.Add(bytes, binding, owner, key.end);
        const size_t first = Hash(key);
        for (size_t probe = 0; probe < rows.size(); ++probe) {
            auto& row = rows[(first + probe) & 255];
            if (!row.occupied) { row.key = key; row.occupied = true; ++used; }
            if (row.key == key) { row.totals.Add(bytes, binding, owner, key.end); return; }
        }
        overflow.Add(bytes, binding, owner, key.end);
    }
    std::array<const Row*, 16> Top() const {
        std::array<const Row*, 16> top{};
        for (const auto& row : rows) {
            if (!row.occupied) continue;
            for (size_t i = 0; i < top.size(); ++i) {
                if (!top[i] || row.totals.requested > top[i]->totals.requested) {
                    for (size_t j = top.size() - 1; j > i; --j) top[j] = top[j - 1];
                    top[i] = &row;
                    break;
                }
            }
        }
        return top;
    }
};
inline thread_local Table t_table;

inline void Record(Key key, u64 bytes, u64 binding, u64 owner) {
    if (Enabled()) t_table.Record(key, bytes, binding, owner);
}

// Existing profile Report calls this outside active scopes, on its own >=10s window.
inline void Report(u64 window_ns, const char* thread) {
    if (!Enabled() || !t_table.total.calls) return;
    LOG_INFO(HW_GPU, "{} writer marks window_ns={} calls={} requested_bytes={} keys={} "
             "overflow_calls={} overflow_bytes={} (requested marks, not unique writes/downloads)",
             thread, window_ns, t_table.total.calls, t_table.total.requested, t_table.used,
             t_table.overflow.calls, t_table.overflow.requested);
    for (const Row* row : t_table.Top()) {
        if (!row) break;
        const auto& k = row->key;
        const auto& t = row->totals;
        LOG_INFO(HW_GPU, "Writer kind={} canonical={:016x} auxiliary={:016x} native={:016x} "
                 "stage={} index={} cbuf={} offset={} child={} metadata={} global_fallback={} proof={} "
                 "proof_count={} end={} calls={} requested_bytes={} max_mark={} max_binding={} "
                 "max_owner={} proof_calls={} proof_prefix_bytes={}",
                 kind_names[static_cast<size_t>(k.kind)], k.canonical, k.auxiliary, k.native,
                 k.stage, k.index, k.cbuf, k.offset, k.child, k.metadata_available, k.global_fallback,
                 proof_names[static_cast<size_t>(k.proof)], k.proof_count, k.end, t.calls,
                 t.requested, t.max_mark, t.max_binding, t.max_owner, t.proof_calls, t.proof_prefix);
    }
    t_table = {};
}

} // namespace VideoCommon::Nx2WriteProfile
