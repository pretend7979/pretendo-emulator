// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <memory>
#include <limits>
#include <vector>

#include "common/assert.h"
#include "common/logging.h"
#include "core/core.h"
#include "video_core/control/channel_state.h"
#include "video_core/control/scheduler.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"

namespace Tegra::Control {

void Scheduler::BindChannel(GPU& gpu, s32 bind_id) {
    // With the GPU thread split the rasterizer's channel switches with the methods around it, on
    // the back thread (deferred outside the lock: the back takes it to bind).
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.Defer([this, gpu_ptr = &gpu, bind_id] {
            std::scoped_lock lk{scheduling_guard};
            gpu_ptr->BindChannel(bind_id);
        });
        return;
    }
    std::scoped_lock lk{scheduling_guard};
    gpu.BindChannel(bind_id);
}

void Scheduler::Push(GPU& gpu, s32 channel, CommandList&& entries) {
    std::shared_ptr<ChannelState> channel_state;
    {
        std::unique_lock lk(scheduling_guard);
        auto it = channels.find(channel);
        ASSERT(it != channels.end());
        channel_state = it->second;
    }
    BindChannel(gpu, channel_state->bind_id);
    // Process commands outside the lock to reduce contention.
    // Multiple channels can prepare their commands in parallel.
    channel_state->payload->dma_pusher.Push(std::move(entries));
    channel_state->payload->dma_pusher.DispatchCalls();
    if (channel_state->payload->dma_pusher.HasPendingAcquire() &&
        std::find(pending_channels.begin(), pending_channels.end(), channel_state) ==
            pending_channels.end()) {
        pending_channels.push_back(channel_state);
        any_pending.store(true, std::memory_order_release);
    }
    if (!pending_channels.empty()) {
        ResumePending(gpu);
    }
}

bool Scheduler::ResumePending(GPU& gpu) {
    bool any_progress = false;
    // Running one channel can release the semaphore another waits for: retry until none moves.
    for (bool progress = true; progress;) {
        progress = false;
        for (const auto& state : pending_channels) {
            auto& dma_pusher = state->payload->dma_pusher;
            // Powered off (shutting down), DispatchCalls returns at once and the acquire stays
            // pending: resuming it again counted as progress forever and the GPU thread never
            // reached its stop request (closing the game hung).
            if (!dma_pusher.system.IsPoweredOn()) {
                return any_progress;
            }
            if (!dma_pusher.HasPendingAcquire() || !dma_pusher.CanResumeAcquire()) {
                continue;
            }
            BindChannel(gpu, state->bind_id);
            dma_pusher.DispatchCalls();
            progress = true;
            any_progress = true;
        }
    }
    std::erase_if(pending_channels, [](const auto& state) {
        return !state->payload->dma_pusher.HasPendingAcquire();
    });
    any_pending.store(!pending_channels.empty(), std::memory_order_release);
    return any_progress;
}

bool Scheduler::HasDeferredAcquire() const {
    return std::ranges::any_of(pending_channels, [](const auto& state) {
        const auto* acquire = state->payload->dma_pusher.puller.PendingAcquireForDiagnostics();
        return acquire && acquire->retirement;
    });
}

u64 Scheduler::PendingChannelFingerprint() const {
    u64 fingerprint = 0xcbf29ce484222325ULL;
    const auto add = [&](u64 value) {
        fingerprint = (fingerprint ^ value) * 0x100000001b3ULL;
    };
    add(pending_channels.size());
    for (size_t index = 0; index < std::min<size_t>(pending_channels.size(), 16); ++index) {
        const auto& state = *pending_channels[index];
        const auto& dma = state.payload->dma_pusher;
        add(static_cast<u32>(state.bind_id));
        add(dma.suspensions.load(std::memory_order_relaxed));
        add(dma.dma_pushbuffer.size());
        add(dma.dma_pushbuffer_subindex);
        add(dma.CommandResumeOffsetForDiagnostics());
        if (const auto* acquire = dma.puller.PendingAcquireForDiagnostics()) {
            add(acquire->address);
            add(acquire->value);
            add(acquire->operation);
            const bool mapped = acquire->address <= std::numeric_limits<GPUVAddr>::max() - sizeof(u32) &&
                                dma.memory_manager.IsWithinGPUAddressRange(acquire->address) &&
                                dma.memory_manager.IsWithinGPUAddressRange(acquire->address + sizeof(u32) - 1) &&
                                dma.memory_manager.IsFullyMappedRange(acquire->address, sizeof(u32));
            u32 current{};
            if (mapped) dma.memory_manager.ReadBlockUnsafe(acquire->address, &current, sizeof(current));
            add(mapped);
            add(current);
        }
    }
    return fingerprint;
}

void Scheduler::LogPendingChannels() const {
    constexpr size_t limit = 16;
    for (size_t index = 0; index < std::min(pending_channels.size(), limit); ++index) {
        const auto& state = *pending_channels[index];
        const auto& dma = state.payload->dma_pusher;
        const auto* acquire = dma.puller.PendingAcquireForDiagnostics();
        const GPUVAddr address = acquire ? acquire->address : 0;
        const bool mapped = acquire && address <= std::numeric_limits<GPUVAddr>::max() - sizeof(u32) &&
                            dma.memory_manager.IsWithinGPUAddressRange(address) &&
                            dma.memory_manager.IsWithinGPUAddressRange(address + sizeof(u32) - 1) &&
                            dma.memory_manager.IsFullyMappedRange(address, sizeof(u32));
        u32 current{};
        if (mapped) dma.memory_manager.ReadBlockUnsafe(address, &current, sizeof(current));
        GPUVAddr entry_address{};
        u32 entry_words{};
        size_t entries{}, prefetch_words{};
        u64 snapshot_suspensions{};
        if (!dma.dma_pushbuffer.empty()) {
            const auto& list = dma.dma_pushbuffer.front();
            entries = list.command_lists.size();
            prefetch_words = list.prefetch_command_list.size();
            snapshot_suspensions = list.snapshot_suspensions;
            if (dma.dma_pushbuffer_subindex < entries) {
                entry_address = list.command_lists[dma.dma_pushbuffer_subindex].addr;
                entry_words = static_cast<u32>(list.command_lists[dma.dma_pushbuffer_subindex].size);
            }
        }
        LOG_WARNING(HW_GPU,
            "Pending snapshot channel={} powered_on={} acquire={} address={:#x} expected={:#x} current={:#x} mapped={} op={} device={:#x} latched={} lists={} entry={}/{} entry_address={:#x} entry_words={} prefetch_words={} resume_word={} suspensions={} snapshot_suspensions={} method={:#x} subchannel={} method_count={} dma_get={:#x} dma_word={}",
            state.bind_id, dma.system.IsPoweredOn(), acquire != nullptr, address,
            acquire ? acquire->value : 0, current, mapped, acquire ? acquire->operation : 0,
            mapped ? dma.memory_manager.GpuToCpuAddress(address).value_or(0) : 0,
            Nx2SemaphoreWaiters::Get().Latched(&dma.puller), dma.dma_pushbuffer.size(),
            dma.dma_pushbuffer_subindex, entries, entry_address, entry_words, prefetch_words,
            dma.CommandResumeOffsetForDiagnostics(), dma.suspensions.load(std::memory_order_relaxed),
            snapshot_suspensions, dma.dma_state.method, dma.dma_state.subchannel,
            dma.dma_state.method_count, dma.dma_state.dma_get, dma.dma_state.dma_word_offset);
    }
    // Error severity flushes the file backend, making preceding rows visible during a stall.
    LOG_ERROR(HW_GPU, "Pending snapshot complete: {} channels, {} omitted; CPU memory observed without flushing GPU work",
              pending_channels.size(), pending_channels.size() > limit ? pending_channels.size() - limit : 0);
}

void Scheduler::DeclareChannel(std::shared_ptr<ChannelState> new_channel) {
    s32 channel = new_channel->bind_id;
    std::unique_lock lk(scheduling_guard);
    channels.emplace(channel, new_channel);
}

} // namespace Tegra::Control
