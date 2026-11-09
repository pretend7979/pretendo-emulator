// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "common/copyback_watch.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string_view>
#include "common/settings.h"
#include "core/core.h"
#include "video_core/dma_pusher.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"

namespace Tegra {

struct BlitPointerIdentity {
    u64 time_ns{}, gmmu_id{}, device_first{}, device_last{};
    u64 physical_first{}, physical_last{}, host_first{}, host_last{};
    bool range_valid{}, device_first_valid{}, device_last_valid{};
};

struct BlitSyncpointObservation {
    u32 id{}, guest{}, host{};
    bool valid{};
};

struct BlitSubmitObservation {
    Common::CopybackWatch::Attachment copyback{};
    u64 serial{}, publish_begin_ns{};
    // The GPU thread can reach the boundary before the submitting call returns.
    std::atomic<u64> publish_return_ns{}, ioctl_tail_queued_ns{};
    BlitPointerIdentity before{}, after{};
    BlitSyncpointObservation at_snapshot{}, before_publish{}, input_dependency{};
    u32 input_flags{}, input_fence_id{}, input_fence_value{}, output_fence_id{}, output_fence_value{};
    u32 max_before{}, min_cached_before{}, increment{};
    std::array<u64, 5> headers{};
    std::array<u32, 64> first_entry_words{};
    u32 first_entry_total{}, first_entry_captured{};
};

inline u64 BlitProbeTime() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline bool BlitSubmitProbeEnabled() {
    static const bool enabled = [] {
        const char* identity = std::getenv("EDEN_NX2_PB_SUBMIT_IDENTITY");
        const char* boundary = std::getenv("EDEN_NX2_PB_BOUNDARY_TRACE");
        const char* fault = std::getenv("EDEN_NX2_PB_FIRST_FAULT");
        return identity && std::string_view{identity} == "1" &&
               boundary && std::string_view{boundary} == "1" && fault && *fault;
    }();
    return enabled && Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL;
}

inline bool BlitSubmitProbeStructure(const CommandList& list) {
    return list.command_lists.size() == 5 && list.prefetch_command_list.empty() &&
           list.command_lists[1].sync && list.command_lists[1].size == 1 &&
           !list.command_lists[2].sync && list.command_lists[2].size == 25;
}

inline bool BlitSubmitProbeShape(const CommandList& list) {
    constexpr std::array<u32, 25> shape{
        0x91020002, 0x200181ca, 0x1040, 0x200181cb, 0x1e00,
        0x200181cc, 0x438, 0x200181cd, 1, 0x200181cf,
        0, 0x20018105, 0x1e00, 0x20028100, 0x22,
        0, 0x20028102, 0x22, 0, 0x20028106,
        0x1e00, 0x438, 0x200180c0, 0x306, 0x91000002,
    };
    if (!BlitSubmitProbeStructure(list) || list.submitted_offsets.size() != 5 ||
        list.snapshot_origin != CommandSnapshotOrigin::Submit) return false;
    const size_t boundary = list.submitted_offsets[1];
    const size_t offset = list.submitted_offsets[2];
    if (boundary >= list.submitted_words.size() || list.submitted_words[boundary].argument != 0 ||
        offset > list.submitted_words.size() || shape.size() > list.submitted_words.size() - offset)
        return false;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 15 && i != 18 && list.submitted_words[offset + i].argument != shape[i]) return false;
    }
    return true;
}

inline BlitPointerIdentity ReadBlitPointerIdentity(Core::System& system, MemoryManager& memory,
                                                  GPUVAddr address) {
    BlitPointerIdentity result{.time_ns = BlitProbeTime(), .gmmu_id = memory.GetID()};
    constexpr size_t bytes = 25 * sizeof(u32);
    if (address > UINT64_MAX - (bytes - 1) || (address & 0xfff) > 0x1000 - bytes ||
        !memory.IsWithinGPUAddressRange(address) ||
        !memory.IsWithinGPUAddressRange(address + bytes - 1) ||
        !memory.IsFullyMappedRange(address, bytes)) return result;
    const auto first = memory.GpuToCpuAddress(address);
    const auto last = memory.GpuToCpuAddress(address + bytes - 1);
    result.device_first_valid = first.has_value();
    result.device_last_valid = last.has_value();
    result.device_first = first.value_or(0);
    result.device_last = last.value_or(0);
    if (!first || !last) return result;
    auto& smmu = system.Host1x().MemoryManager();
    result.physical_first = smmu.GetPhysicalRawAddressFromDAddr(*first);
    result.physical_last = smmu.GetPhysicalRawAddressFromDAddr(*last);
    result.host_first = reinterpret_cast<uintptr_t>(memory.GetPointer(address));
    result.host_last = reinterpret_cast<uintptr_t>(memory.GetPointer(address + bytes - 1));
    result.range_valid = result.host_first != 0 && result.host_last != 0;
    return result;
}

inline Common::CopybackWatch::Attachment ArmBlitCopybackWatch(
    const BlitPointerIdentity& before, const BlitPointerIdentity& after) {
    using namespace Common::CopybackWatch;
    if (!Enabled() || !BlitSubmitProbeEnabled()) return {};
    const auto valid = [](const BlitPointerIdentity& value) {
        return value.range_valid && value.device_first_valid && value.device_last_valid &&
            value.device_first <= UINT64_MAX - (SpanBytes - 1) &&
            value.physical_first <= UINT64_MAX - (SpanBytes - 1) &&
            value.host_first <= UINT64_MAX - (SpanBytes - 1) &&
            value.device_last == value.device_first + SpanBytes - 1 &&
            value.physical_last == value.physical_first + SpanBytes - 1 &&
            value.host_last == value.host_first + SpanBytes - 1 &&
            (value.device_first & 0xfff) <= 0x1000 - SpanBytes &&
            (value.physical_first & 0xfff) <= 0x1000 - SpanBytes &&
            (value.host_first & 0xfff) <= 0x1000 - SpanBytes;
    };
    if (!valid(before) || !valid(after) || before.gmmu_id != after.gmmu_id ||
        before.device_first != after.device_first || before.physical_first != after.physical_first ||
        before.host_first != after.host_first) return {.reason = ArmReason::InvalidIdentity};
    return Arm(Identity{.session = Session(), .gmmu = after.gmmu_id,
        .device = after.device_first, .physical = after.physical_first, .host = after.host_first});
}

inline BlitSyncpointObservation ReadBlitSyncpoint(Core::System& system, u32 id) {
    BlitSyncpointObservation result{.id = id};
    if (id >= Host1x::SyncpointManager::NUM_MAX_SYNCPOINTS) return result;
    auto& syncpoints = system.Host1x().GetSyncpointManager();
    result.guest = syncpoints.GetGuestSyncpointValue(id);
    result.host = syncpoints.GetHostSyncpointValue(id);
    result.valid = true;
    return result;
}

} // namespace Tegra
