// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/copyback_watch.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <fmt/ranges.h>

#include "common/range_sets.inc"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/uniform_stream_reuse.h"
#include "video_core/guest_memory.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_profile.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

using Core::DEVICE_PAGESIZE;

template <class P>
BufferCache<P>::BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_)
    : runtime{runtime_}, device_memory{device_memory_}, memory_tracker{device_memory} {
    // A new runtime has a new scheduler (its ticks restart) and a new stream buffer: whatever the
    // stream uniform reuse remembers refers to the old ones. Before the early return below.
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    // Ensure the first slot is used for the null buffer
    void(slot_buffers.insert(runtime, NullBufferParams{}));
    gpu_modified_ranges.Clear();
    inline_buffer_id = NULL_BUFFER_ID;
#ifdef YUZU_LEGACY
    immediately_free = (Settings::values.vram_usage_mode.GetValue() == Settings::VramUsageMode::Aggressive);
#endif
    if (!runtime.CanReportMemoryUsage()) {
        minimum_memory = DEFAULT_EXPECTED_MEMORY;
        critical_memory = DEFAULT_CRITICAL_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
    const s64 min_spacing_expected = device_local_memory - 1_GiB;
    const s64 min_spacing_critical = device_local_memory - 512_MiB;
    const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    minimum_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                 DEFAULT_EXPECTED_MEMORY));
    critical_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                 DEFAULT_CRITICAL_MEMORY));
}

template <class P>
BufferCache<P>::~BufferCache() = default;

template <class P>
void BufferCache<P>::RunGarbageCollector(bool aggressive_gc) {
    const u64 ticks_to_destroy = aggressive_gc ? 60 : 120;
    int num_iterations = aggressive_gc ? 64 : 32;
    // Their GPU-written data is downloaded first, all with one wait for the GPU: one wait per
    // buffer stalled the GPU thread every frame on title 1 (an 8 GB GPU near its budget, no
    // per-fence downloads keeping the buffers clean), ~2.5% of its time.
    boost::container::small_vector<BufferId, 64> evicted;
    lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, [&](BufferId buffer_id) {
        if (num_iterations == 0) {
            return true;
        }
        --num_iterations;
        // Below the critical threshold, buffers holding GPU-written data stay: evicting one
        // means downloading it first. The thresholds compare the whole device's use (mostly
        // textures) with fixed headroom, so on an 8 GB GPU the collector ran every frame once a
        // title 1 course was loaded, and the downloads were ~2.7% of the GPU thread.
        const Buffer& buffer = slot_buffers[buffer_id];
        if (!aggressive_gc && gpu_modified_ranges.Intersects(buffer.CpuAddr(), buffer.SizeBytes())) {
            return false;
        }
        evicted.push_back(buffer_id);
        return false;
    });
    if constexpr (USE_MEMORY_MAPS) {
        DownloadBuffersMemory(evicted);
    } else {
        for (const BufferId buffer_id : evicted) {
            DownloadBufferMemory(slot_buffers[buffer_id]);
        }
    }
    for (const BufferId buffer_id : evicted) {
        DeleteBuffer(buffer_id);
    }
}

template <class P>
void BufferCache<P>::TickFrame() {
    // Homebrew console apps don't create or bind any channels, so this will be nullptr.
    if (!channel_state) {
        return;
    }
    runtime.TickFrame(slot_buffers);
    if (frame_tick % 600 == 0 && !uploaded_buffer_ranges.empty()) {
        for (const auto& [address, size] : uploaded_buffer_ranges) {
            device_memory.HintCold(address, size);
        }
        uploaded_buffer_ranges.clear();
    }
    if (static u64 frames{}; ++frames % 600 == 0) {
        u64 total{};
        size_t count{};
        std::vector<std::pair<u64, VAddr>> largest;
        for (const auto [buffer_id, buffer] : slot_buffers) {
            total += buffer->SizeBytes();
            ++count;
            largest.emplace_back(buffer->SizeBytes(), buffer->CpuAddr());
        }
        std::partial_sort(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()),
                          largest.end(), std::greater<>{});
        LOG_INFO(HW_GPU, "Buffer cache: {} buffers {} MiB; largest {}", count, total >> 20,
                 fmt::join(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()), ", "));
    }

    // Calculate hits and shots and move hit bits to the right
    const u32 hits = std::reduce(channel_state->uniform_cache_hits.begin(),
                                 channel_state->uniform_cache_hits.end());
    const u32 shots = std::reduce(channel_state->uniform_cache_shots.begin(),
                                  channel_state->uniform_cache_shots.end());
    std::copy_n(channel_state->uniform_cache_hits.begin(),
                channel_state->uniform_cache_hits.size() - 1,
                channel_state->uniform_cache_hits.begin() + 1);
    std::copy_n(channel_state->uniform_cache_shots.begin(),
                channel_state->uniform_cache_shots.size() - 1,
                channel_state->uniform_cache_shots.begin() + 1);
    channel_state->uniform_cache_hits[0] = 0;
    channel_state->uniform_cache_shots[0] = 0;

    const bool skip_preferred = hits * 256 < shots * 251;
    channel_state->uniform_buffer_skip_cache_size = skip_preferred ? DEFAULT_SKIP_CACHE_SIZE : 0;

    // If we can obtain the memory info, use it instead of the estimate.
    if (runtime.CanReportMemoryUsage()) {
        total_used_memory = runtime.GetDeviceMemoryUsage();
    }
    if (gpu_memory != nullptr && gpu_memory->UsesNativeTiling()) {
        // Switch 2 titles fill the device with textures (their collector watches the budget);
        // title 1's buffers were ~200 MiB of an 8 GB GPU's ~6.8 GB in use, yet this collector
        // ran aggressively every frame on the device-wide figure, evicting up to 64 buffers (and
        // downloading the GPU-written ones, ~5% of the GPU thread) that were recreated later.
        // Collect when the buffers themselves are large. At 1 GiB it held geometry of scenes
        // long gone: title 1's title screen grew to ~0.9 GB of buffers in ten minutes.
        if (buffer_bytes >= 512_MiB) {
            RunGarbageCollector(buffer_bytes >= 1_GiB);
        }
    } else if (total_used_memory >= minimum_memory) {
        RunGarbageCollector(total_used_memory >= critical_memory);
    }
    ++frame_tick;
    delayed_destruction_ring.Tick();

    for (auto& buffer : async_buffers_death_ring) {
        runtime.FreeDeferredStagingBuffer(buffer);
    }
    async_buffers_death_ring.clear();
}

template <class P>
void BufferCache<P>::UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        virtual_ranges.Unmap(as_id, gpu_addr, size);
    }
}

template <class P>
void BufferCache<P>::WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data) {
    Nx2NoteListHostWrite(device_addr, size);
    if (IsRegionGpuModified(device_addr, size)) {
        ClearDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
    }
    // The tracker marks whole pages CPU-modified, which drops their GPU-modified state: GPU-written
    // bytes elsewhere on these pages would never be downloaded, and the page's next upload would
    // send the stale guest copy over them. Hyrule Warriors: Age of Imprisonment writes small
    // structures next to the draw arguments its compute dispatches fill in; the draws then read
    // leftover bytes as vertex counts. Download those bytes first (the CPU's own were subtracted
    // above, so they aren't overwritten).
    // Downloading them here waits for the host GPU, and OoT's CPU writes land next to GPU data
    // ~1000 times a second. Uploads already skip gpu_modified_ranges, so the bytes only have to
    // reach guest memory before the CPU reads them: copy them at the next fence (a later CPU write
    // drops them through ClearDownload) and trap reads until then.
    // EDEN_NX2_CPU_WRITE_PAGE_SYNC=1 downloads them here instead.
    static const bool sync_page_download = [] {
        const char* value = std::getenv("EDEN_NX2_CPU_WRITE_PAGE_SYNC");
        return value != nullptr && value[0] == '1';
    }();
    const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (keep_page_gpu_data && gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
        if (sync_page_download) {
            VideoCommon::Nx2Profile::WaitTag wait_tag{VideoCommon::Nx2Profile::WaitSource::CpuWritePage};
            DownloadMemory(page_begin, page_end - page_begin);
        } else {
            gpu_modified_ranges.ForEachInRange(page_begin, page_end - page_begin,
                                               [&](DAddr begin, DAddr end) {
                uncommitted_gpu_modified_ranges.Add(begin, end - begin);
                ArmReadTraps(begin, end - begin);
            });
        }
    }
    memory_tracker.MarkRegionAsCpuModified(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
}

template <class P>
void BufferCache<P>::CachedWriteMemory(DAddr device_addr, u64 size) {
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return;
    }
    DAddr aligned_start = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        WriteMemory(device_addr, size);
        return;
    }

    Nx2NoteListHostWrite(device_addr, size);
    tmp_buffer.resize_destructive(size);
    device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);

    InlineMemoryImplementation(device_addr, size, tmp_buffer);
}

template <class P>
bool BufferCache<P>::OnCPUWrite(DAddr device_addr, u64 size) {
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return false;
    }
    if (memory_tracker.IsRegionGpuModified(device_addr, size)) {
        return true;
    }
    WriteMemory(device_addr, size);
    return false;
}

template <class P>
std::optional<VideoCore::RasterizerDownloadArea> BufferCache<P>::GetFlushArea(DAddr device_addr,
                                                                              u64 size) {
    std::optional<VideoCore::RasterizerDownloadArea> area{};
    area.emplace();
    DAddr device_addr_start_aligned = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
    DAddr device_addr_end_aligned = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
    area->start_address = device_addr_start_aligned;
    area->end_address = device_addr_end_aligned;
    // Preflushable selects later fence downloads; it does not prove that an
    // earlier reactive read has finished copying these bytes to guest memory.
    // Check the whole range, including any pages not marked Preflushable.
    area->preemtive = !IsRegionGpuModified(device_addr_start_aligned,
                                           device_addr_end_aligned - device_addr_start_aligned);
    // Mark only data that the GPU wrote. A mark stays for the session and sends later GPU writes
    // in the range to a download at each fence.
    if (!area->preemtive) {
        memory_tracker.MarkRegionAsPreflushable(device_addr_start_aligned,
                                                device_addr_end_aligned - device_addr_start_aligned);
    } else {
        // The core keeps this area and has no mark. The next GPU write here must arm the range
        // again, which ends the area and sets the trap.
        read_trapped_ranges.Subtract(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned);
    }
    return area;
}

template <class P>
void BufferCache<P>::DownloadMemory(DAddr device_addr, u64 size) {
    ForEachBufferInRange(device_addr, size, [&](BufferId, Buffer& buffer) {
        DownloadBufferMemory(buffer, device_addr, size);
    });
}

template <class P>
void BufferCache<P>::ClearDownload(DAddr device_addr, u64 size) {
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    uncommitted_gpu_modified_ranges.Subtract(device_addr, size);
    for (auto& interval_set : committed_gpu_modified_ranges) {
        interval_set.Subtract(device_addr, size);
    }
}

template <class P>
void BufferCache<P>::WriteQueryPayload(DAddr device_addr, u32 value) {
    Common::CopybackWatch::ScopedWriter copyback_scope{Common::CopybackWatch::Kind::BufferQuery};
    ClearDownload(device_addr, sizeof(value));
    gpu_modified_ranges.Subtract(device_addr, sizeof(value));
    device_memory.WriteBlockUnsafe(device_addr, &value, sizeof(value));
    // Command-list copies become stale only after the new CPU bytes are available.
    Nx2NoteListHostWrite(device_addr, sizeof(value));
}

template <class P>
bool BufferCache<P>::DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    if (amount == 0) {
        return true;
    }

    const std::optional<DAddr> cpu_src_address = gpu_memory->GpuToCpuAddress(src_address);
    const std::optional<DAddr> cpu_dest_address = gpu_memory->GpuToCpuAddress(dest_address);
    const GPUVAddr src_end = src_address + amount - 1;
    const GPUVAddr dst_end = dest_address + amount - 1;
    if (!cpu_src_address || !cpu_dest_address || src_end < src_address ||
        dst_end < dest_address || !gpu_memory->IsWithinGPUAddressRange(src_end) ||
        !gpu_memory->IsWithinGPUAddressRange(dst_end) ||
        !gpu_memory->IsContinuousRange(src_address, amount) ||
        !gpu_memory->IsContinuousRange(dest_address, amount)) {
        return false;
    }
    const bool source_dirty = IsRegionRegistered(*cpu_src_address, amount);
    const bool dest_dirty = IsRegionRegistered(*cpu_dest_address, amount);
    if (!source_dirty && !dest_dirty) {
        return false;
    }

    Nx2NoteListHostWrite(*cpu_dest_address, amount);
    ClearDownload(*cpu_dest_address, amount);
    MarkVertexBuffersStale(*cpu_dest_address, amount);

    BufferId buffer_a;
    BufferId buffer_b;
    do {
        channel_state->has_deleted_buffers = false;
        buffer_a = FindBuffer(*cpu_src_address, static_cast<u32>(amount), false);
        buffer_b = FindBuffer(*cpu_dest_address, static_cast<u32>(amount), false);
    } while (channel_state->has_deleted_buffers);
    auto& src_buffer = slot_buffers[buffer_a];
    auto& dest_buffer = slot_buffers[buffer_b];
    SynchronizeBuffer(src_buffer, *cpu_src_address, static_cast<u32>(amount));
    SynchronizeBuffer(dest_buffer, *cpu_dest_address, static_cast<u32>(amount));
    std::array copies{BufferCopy{
        .src_offset = src_buffer.Offset(*cpu_src_address),
        .dst_offset = dest_buffer.Offset(*cpu_dest_address),
        .size = amount,
    }};

    boost::container::small_vector<std::pair<DAddr, size_t>, 4> tmp_intervals;
    auto mirror = [&](DAddr base_address, DAddr base_address_end) {
        const u64 size = base_address_end - base_address;
        const DAddr diff = base_address - *cpu_src_address;
        const DAddr new_base_address = *cpu_dest_address + diff;
        tmp_intervals.push_back({new_base_address, size});
        if (Nx2WriteProfile::Enabled()) {
            Nx2WriteProfile::Record(
                Nx2WriteProfile::OtherKey(nullptr, Nx2WriteProfile::Kind::DmaMirror),
                size, amount, dest_buffer.SizeBytes());
        }
        TrackBufferWriteDownloads(new_base_address, size);
    };
    gpu_modified_ranges.ForEachInRange(*cpu_src_address, amount, mirror);
    // This subtraction in this order is important for overlapping copies.
    gpu_modified_ranges.Subtract(*cpu_dest_address, amount);
    const bool has_new_downloads = tmp_intervals.size() != 0;
    for (const auto& pair : tmp_intervals) {
        gpu_modified_ranges.Add(pair.first, pair.second);
        if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
            Tegra::Nx2MarkGpuWritten(pair.first, pair.second);
            Nx2NoteListWrite(buffer_b, pair.first, pair.second);
        }
    }
    const auto& copy = copies[0];
    src_buffer.MarkUsage(copy.src_offset, copy.size);
    dest_buffer.MarkUsage(copy.dst_offset, copy.size);
    runtime.CopyBuffer(dest_buffer, src_buffer, copies, true);
    if (has_new_downloads) {
        memory_tracker.MarkRegionAsGpuModified(*cpu_dest_address, amount);
        const bool should_sync = Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
        if (should_sync) {
            runtime.Finish();
        }
    }

    Tegra::Memory::DeviceGuestMemoryScoped<u8, Tegra::Memory::GuestMemoryFlags::UnsafeReadWrite>
        tmp(device_memory, *cpu_src_address, amount, &tmp_buffer);
    tmp.SetAddressAndSize(*cpu_dest_address, amount);
    return true;
}

template <class P>
bool BufferCache<P>::DMAClear(GPUVAddr dst_address, u64 amount, u32 value) {
    if (amount == 0) {
        return true;
    }
    const size_t size = amount * sizeof(u32);

    const std::optional<DAddr> cpu_dst_address = gpu_memory->GpuToCpuAddress(dst_address);
    const GPUVAddr dst_end = dst_address + size - 1;
    if (!cpu_dst_address || dst_end < dst_address ||
        !gpu_memory->IsWithinGPUAddressRange(dst_end) ||
        !gpu_memory->IsContinuousRange(dst_address, size)) {
        return false;
    }
    const bool dest_dirty = IsRegionRegistered(*cpu_dst_address, size);
    if (!dest_dirty) {
        return false;
    }

    Nx2NoteListHostWrite(*cpu_dst_address, size);
    ClearDownload(*cpu_dst_address, size);
    gpu_modified_ranges.Subtract(*cpu_dst_address, size);

    const BufferId buffer = FindBuffer(*cpu_dst_address, static_cast<u32>(size), false);
    Buffer& dest_buffer = slot_buffers[buffer];
    const u32 offset = dest_buffer.Offset(*cpu_dst_address);
    runtime.ClearBuffer(dest_buffer, offset, size, value);
    dest_buffer.MarkUsage(offset, size);
    return true;
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                                 ObtainBufferSynchronize sync_info,
                                                                 ObtainBufferOperation post_op) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        return {&slot_buffers[NULL_BUFFER_ID], 0};
    }
    return ObtainCPUBuffer(*device_addr, size, sync_info, post_op);
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainCPUBuffer(
    DAddr device_addr, u32 size, ObtainBufferSynchronize sync_info, ObtainBufferOperation post_op) {
    const BufferId buffer_id = FindBuffer(device_addr, size, false);
    Buffer& buffer = slot_buffers[buffer_id];

    // synchronize op
    switch (sync_info) {
    case ObtainBufferSynchronize::FullSynchronize:
        SynchronizeBuffer(buffer, device_addr, size);
        break;
    default:
        break;
    }

    switch (post_op) {
    case ObtainBufferOperation::MarkAsWritten:
        MarkWrittenBuffer(buffer_id, device_addr, size);
        break;
    case ObtainBufferOperation::DiscardWrite: {
        ClearDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
        break;
    }
    default:
        break;
    }

    return {&buffer, buffer.Offset(device_addr)};
}

template <class P>
void BufferCache<P>::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                               u32 size) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        channel_state->uniform_buffers[stage][index] = NULL_BINDING;
        return;
    }
    const Binding binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = BufferId{},
    };
    channel_state->uniform_buffers[stage][index] = binding;
}

template <class P>
void BufferCache<P>::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    auto& current = channel_state->uniform_buffers[stage][index];
    if (current.device_addr == 0 && current.size == 0 && current.buffer_id == NULL_BUFFER_ID)
        [[likely]] {
        return;
    }
    current = NULL_BINDING;
}

template <class P>
void BufferCache<P>::UpdateGraphicsBuffers(bool is_indexed) {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateGraphicsBuffers(is_indexed);
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::UpdateComputeBuffers(std::span<NativeStorageDomainResult> domains) {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateComputeBuffers(domains);
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::BindHostGeometryBuffers(bool is_indexed) {
    if (is_indexed) {
        BindHostIndexBuffer();
    } else if constexpr (!HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        if (draw_state.topology == Maxwell::PrimitiveTopology::Quads ||
            draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
            runtime.BindQuadIndexBuffer(draw_state.topology, draw_state.vertex_buffer.first,
                                        draw_state.vertex_buffer.count);
        }
    }
    BindHostVertexBuffers();
    BindHostTransformFeedbackBuffers();
    if (current_draw_indirect) {
        BindHostDrawIndirectBuffers();
    }
}

template <class P>
void BufferCache<P>::BindHostStageBuffers(size_t stage, const Nx2WriteProfile::Stage* writer) {
    BindHostGraphicsUniformBuffers(stage);
    if (channel_state->enabled_storage_buffers[stage] != 0) {
        BindHostGraphicsStorageBuffers(stage, writer);
    }
    if (channel_state->enabled_texture_buffers[stage] != 0) {
        BindHostGraphicsTextureBuffers(stage, writer);
    }
}

template <class P>
void BufferCache<P>::BindHostComputeBuffers(const Nx2WriteProfile::Stage* writer,
                                            std::span<const u32> dirty_limits) {
    BindHostComputeUniformBuffers();
    if (channel_state->enabled_compute_storage_buffers != 0) {
        BindHostComputeStorageBuffers(writer, dirty_limits);
    }
    if (channel_state->enabled_compute_texture_buffers != 0) {
        BindHostComputeTextureBuffers(writer);
    }
    if (any_buffer_uploaded) {
        runtime.PostCopyBarrier();
        any_buffer_uploaded = false;
    }
}

template <class P>
auto BufferCache<P>::GetComputeUniformBindingSnapshot(u32 index) const
    -> ComputeUniformBindingSnapshot {
    ComputeUniformBindingSnapshot result;
    if (!channel_state || index >= channel_state->compute_uniform_buffers.size() ||
        !channel_state->compute_uniform_buffer_sizes) {
        result.skip_reason = "binding_index";
        return result;
    }
    if ((channel_state->enabled_compute_uniform_buffer_mask & (u32{1} << index)) == 0) {
        result.skip_reason = "binding_disabled";
        return result;
    }
    const Binding& binding = channel_state->compute_uniform_buffers[index];
    result.buffer_id = binding.buffer_id;
    result.size = (std::min)(binding.size, (*channel_state->compute_uniform_buffer_sizes)[index]);
    if (binding.buffer_id == NULL_BUFFER_ID || result.size == 0) {
        result.skip_reason = "null_or_empty_binding";
        return result;
    }
    const Buffer& buffer = slot_buffers[binding.buffer_id];
    if (binding.device_addr < buffer.CpuAddr()) {
        result.skip_reason = "binding_before_buffer";
        return result;
    }
    const u64 offset = binding.device_addr - buffer.CpuAddr();
    if (offset > (std::numeric_limits<u32>::max)() || offset > buffer.SizeBytes() ||
        result.size > buffer.SizeBytes() - offset) {
        result.skip_reason = "binding_outside_buffer";
        return result;
    }
    result.buffer = &buffer;
    result.offset = static_cast<u32>(offset);
    result.skip_reason = nullptr;
    return result;
}

template <class P>
auto BufferCache<P>::GetGraphicsUniformBindingSnapshot(size_t stage, u32 index) const
    -> ComputeUniformBindingSnapshot {
    ComputeUniformBindingSnapshot result;
    if (!channel_state || stage >= channel_state->uniform_buffers.size() ||
        index >= channel_state->uniform_buffers[stage].size() ||
        !channel_state->uniform_buffer_sizes) {
        result.skip_reason = "binding_index";
        return result;
    }
    if ((channel_state->enabled_uniform_buffer_masks[stage] & (u32{1} << index)) == 0) {
        result.skip_reason = "binding_disabled";
        return result;
    }
    const Binding& binding = channel_state->uniform_buffers[stage][index];
    result.buffer_id = binding.buffer_id;
    result.size = (std::min)(binding.size, (*channel_state->uniform_buffer_sizes)[stage][index]);
    if (binding.buffer_id == NULL_BUFFER_ID || result.size == 0) {
        result.skip_reason = "null_or_empty_binding";
        return result;
    }
    const Buffer& buffer = slot_buffers[binding.buffer_id];
    if (binding.device_addr < buffer.CpuAddr()) {
        result.skip_reason = "binding_before_buffer";
        return result;
    }
    const u64 offset = binding.device_addr - buffer.CpuAddr();
    if (offset > (std::numeric_limits<u32>::max)() || offset > buffer.SizeBytes() ||
        result.size > buffer.SizeBytes() - offset) {
        result.skip_reason = "binding_outside_buffer";
        return result;
    }
    result.buffer = &buffer;
    result.offset = static_cast<u32>(offset);
    result.skip_reason = nullptr;
    return result;
}

template <class P>
auto BufferCache<P>::GetComputeStorageBindingSnapshot(u32 index) const
    -> ComputeStorageBindingSnapshot {
    ComputeStorageBindingSnapshot result;
    if (!channel_state || index >= channel_state->compute_storage_buffers.size()) {
        result.skip_reason = "binding_index";
        return result;
    }
    if ((channel_state->enabled_compute_storage_buffers & (u64{1} << index)) == 0) {
        result.skip_reason = "binding_disabled";
        return result;
    }
    const Binding& binding = channel_state->compute_storage_buffers[index];
    result.gpu_addr = binding.gpu_addr;
    result.device_addr = binding.device_addr;
    result.size = binding.size;
    result.segment_count = binding.segment_count;
    if (binding.segment_count >= 2) {
        result.skip_reason = "multi_range_binding";
        return result;
    }
    if (binding.buffer_id == NULL_BUFFER_ID || binding.size == 0) {
        result.skip_reason = "null_or_empty_binding";
        return result;
    }
    const Buffer& buffer = slot_buffers[binding.buffer_id];
    if (binding.device_addr < buffer.CpuAddr()) {
        result.skip_reason = "binding_before_buffer";
        return result;
    }
    const u64 offset = binding.device_addr - buffer.CpuAddr();
    if (offset > (std::numeric_limits<u32>::max)() || offset > buffer.SizeBytes() ||
        binding.size > buffer.SizeBytes() - offset) {
        result.skip_reason = "binding_outside_buffer";
        return result;
    }
    if (binding.gpu_addr > (std::numeric_limits<GPUVAddr>::max)() - (binding.size - 1)) {
        result.skip_reason = "gpu_range_overflow";
        return result;
    }
    result.buffer = &buffer;
    result.offset = static_cast<u32>(offset);
    result.skip_reason = nullptr;
    return result;
}

template <class P>
auto BufferCache<P>::GetNativeStorageRangeWitness(
    GPUVAddr pointer, const ComputeStorageBindingSnapshot& snapshot) const
    -> NativeStorageRangeWitness {
    NativeStorageRangeWitness result;
    result.pointer = pointer;
    if (!gpu_memory || !pointer || !gpu_memory->IsWithinGPUAddressRange(pointer)) return result;
    result.alignment = runtime.GetStorageBufferAlignment();
    result.reason = "invalid alignment";
    if (!result.alignment || (result.alignment & (result.alignment - 1)) != 0) return result;
    result.aligned_pointer = Common::AlignDown(pointer, result.alignment);
    result.alignment_prefix = pointer - result.aligned_pointer;
    result.address_space_limit = gpu_memory->GetAddressSpaceSize() - result.aligned_pointer;
    if constexpr (requires { runtime.GetMaxStorageBufferRange(); }) {
        result.backend_limit = runtime.GetMaxStorageBufferRange();
    }
    result.pointer_kind_suffix = gpu_memory->GetMemoryLayoutSize(pointer);
    result.aligned_kind_suffix = gpu_memory->GetMemoryLayoutSize(result.aligned_pointer);
    result.candidate_span = (std::min)({result.aligned_kind_suffix, result.backend_limit,
        u64{(std::numeric_limits<u32>::max)()}, result.address_space_limit});
    result.reason = "empty bounded kind run";
    if (result.candidate_span <= result.alignment_prefix) return result;
    const auto aligned_device = gpu_memory->GpuToCpuAddress(result.aligned_pointer);
    const auto device = gpu_memory->GpuToCpuAddress(pointer);
    result.reason = "alignment translation";
    if (!aligned_device || !device || *device < *aligned_device ||
        *device - *aligned_device != result.alignment_prefix) return result;
    constexpr u64 ScanLimit = u64{256} << 20;
    result.scan_request = (std::min)(result.candidate_span, ScanLimit);
    result.scan_capped = result.scan_request < result.candidate_span;
    result.reason = "physical address overflow";
    if (*aligned_device > (std::numeric_limits<DAddr>::max)() - (result.scan_request - 1))
        return result;
    result.physical_span = gpu_memory->MaxContinuousRange(result.aligned_pointer, result.scan_request);
    result.accessible_span = (std::min)(result.physical_span, result.scan_request);
    result.reason = "physical continuity ends before pointer";
    if (result.accessible_span <= result.alignment_prefix) return result;
    result.host_matches = snapshot.buffer && snapshot.segment_count < 2 &&
        snapshot.gpu_addr == result.aligned_pointer && snapshot.device_addr == *aligned_device &&
        snapshot.offset <= snapshot.buffer->SizeBytes() &&
        snapshot.size <= snapshot.buffer->SizeBytes() - snapshot.offset;
    if (result.host_matches) {
        result.host_suffix = snapshot.buffer->SizeBytes() - snapshot.offset;
        result.uncovered_bytes = result.accessible_span > result.host_suffix
            ? result.accessible_span - result.host_suffix : 0;
    }
    result.valid = true;
    result.reason = result.scan_capped && result.physical_span == result.scan_request
        ? "scan cap reached; accessible span is a lower bound" : "bounded accessible domain";
    return result;
}

template <class P>
auto BufferCache<P>::PrepareComputeReadOnlyDomain(u32 index, GPUVAddr pointer)
    -> NativeStorageDomainResult {
    NativeStorageDomainResult result;
    if (!channel_state || index >= channel_state->compute_storage_buffers.size() || index >= 64 ||
        (channel_state->enabled_compute_storage_buffers & (u64{1} << index)) == 0 ||
        (channel_state->written_compute_storage_buffers & (u64{1} << index)) != 0 ||
        !kepler_compute || !kepler_compute->IsNativeMode()) return result;
    Binding& binding = channel_state->compute_storage_buffers[index];
    result.previous_size = result.bound_size = binding.size;
    if (binding.buffer_id == NULL_BUFFER_ID || binding.segment_count >= 2 || !binding.size ||
        !gpu_memory || !pointer || !gpu_memory->IsWithinGPUAddressRange(pointer)) return result;
    const u64 alignment = runtime.GetStorageBufferAlignment();
    result.reason = "invalid alignment";
    if (!alignment || (alignment & (alignment - 1)) != 0) return result;
    const GPUVAddr aligned = Common::AlignDown(pointer, alignment);
    const u64 prefix = pointer - aligned;
    const auto aligned_device = gpu_memory->GpuToCpuAddress(aligned);
    const auto device = gpu_memory->GpuToCpuAddress(pointer);
    result.reason = "binding identity or alignment mismatch";
    if (!aligned_device || !device || *device < *aligned_device ||
        *device - *aligned_device != prefix || binding.gpu_addr != aligned ||
        binding.device_addr != *aligned_device) return result;
    u64 backend_limit{};
    if constexpr (requires { runtime.GetMaxStorageBufferRange(); })
        backend_limit = runtime.GetMaxStorageBufferRange();
    // This is a rejection budget for new resources, not an inferred guest allocation size.
    constexpr u64 resource_budget = u64{256} << 20;
    const u64 limit = (std::min)({resource_budget, backend_limit,
                                 u64{(std::numeric_limits<u32>::max)()}});
    result.candidate_span = (std::min)(u64{gpu_memory->GetMemoryLayoutSize(aligned)},
                                     gpu_memory->GetAddressSpaceSize() - aligned);
    result.reason = "bounded domain ends before pointer";
    if (limit <= prefix || result.candidate_span <= prefix) return result;
    // The extra byte proves an oversized physical domain; never truncate it to the budget.
    result.scan_span = (std::min)(result.candidate_span, limit + 1);
    result.reason = "physical address overflow";
    constexpr u64 device_limit = u64{1} << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
    if (*aligned_device >= device_limit || result.scan_span > device_limit - *aligned_device)
        return result;
    result.contiguous_span = gpu_memory->MaxContinuousRange(aligned, result.scan_span);
    result.reason = "invalid physical span";
    if (result.contiguous_span > result.scan_span) return result;
    result.reason = "physical domain exceeds resource or descriptor limit";
    if (result.contiguous_span > limit) return result;
    result.reason = "physical domain ends before pointer";
    if (result.contiguous_span <= prefix) return result;
    const u64 page = *aligned_device >> CACHING_PAGEBITS;
    if (page < page_table.size()) {
        const BufferId owner_id = page_table[page];
        if (owner_id && owner_id != NULL_BUFFER_ID) {
            const Buffer& owner = slot_buffers[owner_id];
            if (*aligned_device >= owner.CpuAddr() &&
                *aligned_device - owner.CpuAddr() <= owner.SizeBytes()) {
                result.owner_suffix = owner.SizeBytes() - (*aligned_device - owner.CpuAddr());
            }
        }
    }
    binding.size = static_cast<u32>(result.contiguous_span);
    result.bound_size = binding.size;
    result.allocation_limit = static_cast<u32>(limit);
    result.applied = true;
    result.reason = "validated physical domain; owner resolution pending";
    return result;
}

template <class P>
auto BufferCache<P>::ExpandGraphicsReadOnlyDomain(size_t stage, u32 index)
    -> NativeStorageDomainResult {
    NativeStorageDomainResult result;
    if (!channel_state || stage >= channel_state->storage_buffers.size() || index >= 32 ||
        index >= channel_state->storage_buffers[stage].size() ||
        (channel_state->enabled_storage_buffers[stage] & (u32{1} << index)) == 0 ||
        (channel_state->written_storage_buffers[stage] & (u32{1} << index)) != 0) return result;
    Binding& binding = channel_state->storage_buffers[stage][index];
    result.previous_size = result.bound_size = binding.size;
    if (!binding.buffer_id || binding.buffer_id == NULL_BUFFER_ID || !binding.size ||
        !gpu_memory || !binding.gpu_addr ||
        !gpu_memory->IsWithinGPUAddressRange(binding.gpu_addr)) return result;
    const u64 alignment = runtime.GetStorageBufferAlignment();
    result.reason = "invalid alignment";
    if (!alignment || (alignment & (alignment - 1)) != 0 ||
        (binding.gpu_addr & (alignment - 1)) != 0) return result;
    const auto device = gpu_memory->GpuToCpuAddress(binding.gpu_addr);
    result.reason = "binding translation changed";
    if (!device || *device != binding.device_addr) return result;
    const Buffer& owner = slot_buffers[binding.buffer_id];
    result.reason = "binding outside settled owner";
    if (*device < owner.CpuAddr() || *device - owner.CpuAddr() >= owner.SizeBytes())
        return result;
    const u64 offset = *device - owner.CpuAddr();
    if (offset > (std::numeric_limits<u32>::max)()) return result;
    result.owner_suffix = owner.SizeBytes() - offset;
    if (binding.size > result.owner_suffix) return result;
    u64 backend_limit{};
    if constexpr (requires { runtime.GetMaxStorageBufferRange(); })
        backend_limit = runtime.GetMaxStorageBufferRange();
    constexpr u64 validation_budget = u64{256} << 20;
    const u64 limit = (std::min)({validation_budget, backend_limit, result.owner_suffix,
                                 u64{(std::numeric_limits<u32>::max)()}});
    result.candidate_span = (std::min)(u64{gpu_memory->GetMemoryLayoutSize(binding.gpu_addr)},
        gpu_memory->GetAddressSpaceSize() - binding.gpu_addr);
    result.reason = "domain or limit smaller than current binding";
    if (limit < binding.size || result.candidate_span < binding.size) return result;
    // One extra byte distinguishes a complete domain from truncation at the owner or limit.
    result.scan_span = (std::min)(result.candidate_span, limit + 1);
    constexpr u64 device_limit = u64{1} << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
    result.reason = "physical address overflow";
    if (*device >= device_limit || result.scan_span > device_limit - *device) return result;
    result.contiguous_span = gpu_memory->MaxContinuousRange(binding.gpu_addr, result.scan_span);
    result.reason = "physical domain outside settled owner or limit";
    if (result.contiguous_span > result.scan_span || result.contiguous_span > limit ||
        result.contiguous_span < binding.size) return result;
    binding.size = static_cast<u32>(result.contiguous_span);
    // Fresh physical validation supersedes cached virtual segment metadata.
    binding.segment_first = 0;
    binding.segment_count = 0;
    result.bound_size = binding.size;
    result.applied = true;
    result.reason = "validated physical domain in settled owner";
    return result;
}

template <class P>
auto BufferCache<P>::GetGraphicsStorageBindingSnapshot(size_t stage, u32 index) const
    -> ComputeStorageBindingSnapshot {
    ComputeStorageBindingSnapshot result;
    if (!channel_state || stage >= channel_state->storage_buffers.size() ||
        index >= channel_state->storage_buffers[stage].size()) {
        result.skip_reason = "binding_index";
        return result;
    }
    if ((channel_state->enabled_storage_buffers[stage] & (u32{1} << index)) == 0) {
        result.skip_reason = "binding_disabled";
        return result;
    }
    const Binding& binding = channel_state->storage_buffers[stage][index];
    result.gpu_addr = binding.gpu_addr;
    result.device_addr = binding.device_addr;
    result.size = binding.size;
    result.segment_count = binding.segment_count;
    if (binding.segment_count >= 2) {
        result.skip_reason = "multi_range_binding";
        return result;
    }
    if (binding.buffer_id == NULL_BUFFER_ID || binding.size == 0) {
        result.skip_reason = "null_or_empty_binding";
        return result;
    }
    const Buffer& buffer = slot_buffers[binding.buffer_id];
    if (binding.device_addr < buffer.CpuAddr()) {
        result.skip_reason = "binding_before_buffer";
        return result;
    }
    const u64 offset = binding.device_addr - buffer.CpuAddr();
    if (offset > (std::numeric_limits<u32>::max)() || offset > buffer.SizeBytes() ||
        binding.size > buffer.SizeBytes() - offset) {
        result.skip_reason = "binding_outside_buffer";
        return result;
    }
    if (binding.gpu_addr > (std::numeric_limits<GPUVAddr>::max)() - (binding.size - 1)) {
        result.skip_reason = "gpu_range_overflow";
        return result;
    }
    result.buffer = &buffer;
    result.offset = static_cast<u32>(offset);
    result.skip_reason = nullptr;
    return result;
}

template <class P>
void BufferCache<P>::SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                            const UniformBufferSizes* sizes) {
    if (channel_state->enabled_uniform_buffer_masks == mask &&
        channel_state->uniform_buffer_sizes == sizes) [[likely]] {
        return;
    }
    if (channel_state->enabled_uniform_buffer_masks != mask) {
        channel_state->fast_bound_uniform_buffers.fill(0);
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers.fill(~u32{0});
            channel_state->uniform_buffer_binding_sizes.fill({});
        }
        channel_state->enabled_uniform_buffer_masks = mask;
    }
    channel_state->uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::SetComputeUniformBufferState(u32 mask,
                                                  const ComputeUniformBufferSizes* sizes) {
    channel_state->enabled_compute_uniform_buffer_mask = mask;
    channel_state->compute_uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::UnbindGraphicsStorageBuffers(size_t stage) {
    if (channel_state->enabled_storage_buffers[stage] == 0) {
        return;
    }
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_graphics_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_storage_buffers[stage]));
        }
    }
    channel_state->enabled_storage_buffers[stage] = 0;
    channel_state->written_storage_buffers[stage] = 0;
}

template <class P>
bool BufferCache<P>::BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index,
                                               u32 cbuf_offset, bool is_written, u32 pointer_offset,
                                               bool native_pointer, bool indirect_range_trial,
                                               u32 diagnostic_minimum,
                                               StorageRangeDiagnostic* diagnostic,
                                               GraphicsStoragePointerGuard* write_guard) {
    if (write_guard) *write_guard = {};
    const bool already_enabled =
        ((channel_state->enabled_storage_buffers[stage] >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_graphics_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping graphics storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return false;
            }
        }
    }
    channel_state->enabled_storage_buffers[stage] |= 1U << ssbo_index;
    channel_state->written_storage_buffers[stage] |= (is_written ? 1U : 0U) << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_graphics_storage_buffers;
        }
    }

    if (pointer_offset != ~0U) {
        const auto pointer = is_written ? std::nullopt
            : ResolveGraphicsStorageBufferPointer(stage, cbuf_index, cbuf_offset, pointer_offset, write_guard);
        if (diagnostic) *diagnostic = StorageRangeDiagnostic{
            .requested = diagnostic_minimum, .reason = "unresolved or writable derived pointer"};
        Binding binding = pointer
            ? StorageBufferBinding(0, cbuf_index, false, true, *pointer,
                                   diagnostic_minimum, diagnostic) : NULL_BINDING;
        if (diagnostic) diagnostic->resolved_pointer = pointer.value_or(0);
        if (indirect_range_trial && diagnostic_minimum == 0 && !is_written && pointer &&
            binding.buffer_id != NULL_BUFFER_ID) {
            if (const auto expanded = GraphicsIndirectRangeBinding(*pointer);
                expanded && expanded->size > binding.size) {
                static std::atomic<u32> reports{};
                if (reports.load(std::memory_order_relaxed) < 16 &&
                    reports.fetch_add(1, std::memory_order_relaxed) < 16) {
                    LOG_INFO(HW_GPU,
                        "Native graphics read-only pool stage={} cbuf={} offset={} child={} "
                        "pointer={:#x} bytes={} -> {}",
                        stage, cbuf_index, cbuf_offset, pointer_offset, *pointer,
                        binding.size, expanded->size);
                }
                binding = *expanded;
            }
        }
        if (diagnostic) {
            diagnostic->bound_gpu_addr = binding.gpu_addr;
            diagnostic->bound_size = binding.size;
        }
        channel_state->storage_buffers[stage][ssbo_index] = binding;
    } else {
        const auto& cbufs = maxwell3d->state.shader_stages[stage];
        const GPUVAddr ssbo_addr = cbufs.const_buffers[cbuf_index].address + cbuf_offset;
        channel_state->storage_buffers[stage][ssbo_index] =
            StorageBufferBinding(ssbo_addr, cbuf_index, is_written, native_pointer);
    }
    return (channel_state->storage_buffers[stage][ssbo_index].buffer_id != NULL_BUFFER_ID);
}

template <class P>
void BufferCache<P>::UnbindGraphicsTextureBuffers(size_t stage) {
    if (channel_state->enabled_texture_buffers[stage] == 0) {
        return;
    }
    channel_state->enabled_texture_buffers[stage] = 0;
    channel_state->written_texture_buffers[stage] = 0;
    channel_state->image_texture_buffers[stage] = 0;
}

template <class P>
void BufferCache<P>::BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr,
                                               u32 size, PixelFormat format, bool is_written,
                                               bool is_image) {
    channel_state->enabled_texture_buffers[stage] |= 1U << tbo_index;
    channel_state->written_texture_buffers[stage] |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_texture_buffers[stage] |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->texture_buffers[stage][tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::UnbindComputeStorageBuffers() {
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_compute_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_compute_storage_buffers));
        }
    }
    channel_state->enabled_compute_storage_buffers = 0;
    channel_state->written_compute_storage_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                              bool is_written, u32 pointer_offset,
                                              std::optional<GPUVAddr> resolved_pointer,
                                              bool pointer_prevalidated, u32 diagnostic_minimum,
                                              StorageRangeDiagnostic* diagnostic) {
    if (diagnostic) {
        *diagnostic = StorageRangeDiagnostic{.requested = diagnostic_minimum,
                                            .reason = "descriptor gate"};
    }
    if (is_written || cbuf_index != 0 || cbuf_offset != 96 ||
        (pointer_offset != 24 && pointer_offset != 32)) {
        diagnostic_minimum = 0;
    }
    if (ssbo_index >= channel_state->compute_storage_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Storage buffer index {} exceeds maximum storage buffer count",
                  ssbo_index);
        return;
    }
    const bool already_enabled =
        ((channel_state->enabled_compute_storage_buffers >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_compute_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping compute storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return;
            }
        }
    }
    channel_state->enabled_compute_storage_buffers |= u64{1} << ssbo_index;
    channel_state->written_compute_storage_buffers |= (is_written ? u64{1} : u64{0}) << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_compute_storage_buffers;
        }
    }

    const auto& launch_desc = kepler_compute->launch_description;
    if (pointer_offset != ~0U || pointer_prevalidated) {
        // Never synchronize here: another binding may already have uploaded this pointer table.
        // The prepass resolved every root/entry before any upload, and passes that exact value.
        const bool valid_bank = cbuf_index < launch_desc.const_buffer_config.size() &&
            ((launch_desc.const_buffer_enable_mask.Value() >> cbuf_index) & 1U) != 0;
        channel_state->compute_storage_buffers[ssbo_index] =
            (pointer_offset == ~0U || !is_written) && kepler_compute->IsNativeMode() &&
                valid_bank && resolved_pointer
                ? StorageBufferBinding(0, cbuf_index, is_written, true, *resolved_pointer,
                                       diagnostic_minimum, diagnostic)
                : NULL_BINDING;
        return;
    }
    if (((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) == 0) {
        LOG_WARNING(HW_GPU, "Skipped binding SSBO: cbuf index {} is not enabled", cbuf_index);
        return;
    }
    ASSERT(((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) != 0);

    const auto& cbufs = launch_desc.const_buffer_config;
    const GPUVAddr ssbo_addr = cbufs[cbuf_index].Address() + cbuf_offset;
    channel_state->compute_storage_buffers[ssbo_index] =
        StorageBufferBinding(ssbo_addr, cbuf_index, is_written, kepler_compute->IsNativeMode());
}

template <class P>
void BufferCache<P>::UnbindComputeTextureBuffers() {
    channel_state->enabled_compute_texture_buffers = 0;
    channel_state->written_compute_texture_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                              PixelFormat format, bool is_written, bool is_image) {
    if (tbo_index >= channel_state->compute_texture_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Texture buffer index {} exceeds maximum texture buffer count",
                  tbo_index);
        return;
    }
    channel_state->enabled_compute_texture_buffers |= 1U << tbo_index;
    channel_state->written_compute_texture_buffers |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_compute_texture_buffers |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->compute_texture_buffers[tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::FlushCachedWrites() {
    memory_tracker.FlushCachedWrites();
}

template <class P>
bool BufferCache<P>::HasUncommittedFlushes() const noexcept {
    return !uncommitted_gpu_modified_ranges.Empty() || !committed_gpu_modified_ranges.empty();
}

template <class P>
void BufferCache<P>::AccumulateFlushes() {
    if (uncommitted_gpu_modified_ranges.Empty()) {
        return;
    }
    committed_gpu_modified_ranges.emplace_back(std::move(uncommitted_gpu_modified_ranges));
}

template <class P>
bool BufferCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return (!async_buffers.empty() && async_buffers.front().has_value());
}

template <class P>
void BufferCache<P>::CommitAsyncFlushesHigh() {
    Nx2Profile::Scope profile_scope{Nx2Profile::Zone::DownloadPrepare};
    AccumulateFlushes();

    if (committed_gpu_modified_ranges.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }

    auto it = committed_gpu_modified_ranges.begin();
    while (it != committed_gpu_modified_ranges.end()) {
        auto& current_intervals = *it;
        auto next_it = std::next(it);
        while (next_it != committed_gpu_modified_ranges.end()) {
            next_it->ForEach([&current_intervals](DAddr start, DAddr end) {
                current_intervals.Subtract(start, end - start);
            });
            next_it++;
        }
        it++;
    }

    boost::container::small_vector<std::pair<BufferCopy, BufferId>, 16> downloads;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    for (const Common::RangeSet<DAddr>& range_set : committed_gpu_modified_ranges) {
        range_set.ForEach([&](DAddr interval_lower, DAddr interval_upper) {
            const std::size_t size = interval_upper - interval_lower;
            const DAddr device_addr = interval_lower;
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_start = buffer.CpuAddr();
                const DAddr buffer_end = buffer_start + buffer.SizeBytes();
                const DAddr new_start = (std::max)(buffer_start, device_addr);
                const DAddr new_end = (std::min)(buffer_end, device_addr + size);
                memory_tracker.ForEachDownloadRange(
                    new_start, new_end - new_start, false,
                    [&](u64 device_addr_out, u64 range_size) {
                        const DAddr buffer_addr = buffer.CpuAddr();
                        const auto add_download = [&](DAddr start, DAddr end) {
                            const u64 new_offset = start - buffer_addr;
                            const u64 new_size = end - start;
                            downloads.push_back({
                                BufferCopy{
                                    .src_offset = new_offset,
                                    .dst_offset = total_size_bytes,
                                    .size = new_size,
                                },
                                buffer_id,
                            });
                            // Align up to avoid cache conflicts
                            constexpr u64 align = 64ULL;
                            constexpr u64 mask = ~(align - 1ULL);
                            total_size_bytes += (new_size + align - 1) & mask;
                            largest_copy = (std::max)(largest_copy, new_size);
                        };

                        gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                           add_download);
                    });
            });
        });
    }
    committed_gpu_modified_ranges.clear();
    if (downloads.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }
    Nx2Profile::Add(Nx2Profile::Counter::DownloadBytes, total_size_bytes);
    Nx2Profile::Add(Nx2Profile::Counter::DownloadCopies, downloads.size());
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes, true);
    boost::container::small_vector<BufferCopy, 4> normalized_copies;
    Common::RangeSet<DAddr> download_ranges;
    static const bool batch_download_copies = [] {
        const char* value = std::getenv("EDEN_NX2_BATCH_DOWNLOAD_COPIES");
        return value != nullptr && value[0] == '1';
    }();
    struct DownloadBatch {
        BufferId buffer_id{};
        boost::container::small_vector<BufferCopy, 4> copies;
    };
    boost::container::small_vector<DownloadBatch, 16> download_batches;
    runtime.PreCopyBarrier();
    for (auto& [copy, buffer_id] : downloads) {
        copy.dst_offset += download_staging.offset;
        const std::array copies{copy};
        BufferCopy second_copy{copy};
        Buffer& buffer = slot_buffers[buffer_id];
        second_copy.src_offset = static_cast<size_t>(buffer.CpuAddr()) + copy.src_offset;
        const DAddr orig_device_addr = static_cast<DAddr>(second_copy.src_offset);
        if (Nx2Profile::Enabled()) {
            Nx2Profile::Scope dma_scan_scope{Nx2Profile::Zone::ProfileDmaScan};
            u64 on_dma_pages = 0;
            const auto& pages = Nx2Profile::DmaReadPages();
            for (DAddr page = orig_device_addr >> 12; page <= (orig_device_addr + copy.size - 1) >> 12;
                 ++page) {
                if (pages.contains(page)) {
                    const DAddr begin = (std::max)(orig_device_addr, page << 12);
                    const DAddr end = (std::min)(orig_device_addr + copy.size, (page + 1) << 12);
                    on_dma_pages += end - begin;
                }
            }
            Nx2Profile::Add(Nx2Profile::Counter::DownloadBytesOnDmaPages, on_dma_pages);
        }
        download_ranges.Add(orig_device_addr, copy.size);
        buffer.MarkUsage(copy.src_offset, copy.size);
        if (batch_download_copies) {
            auto batch = std::ranges::find(download_batches, buffer_id, &DownloadBatch::buffer_id);
            if (batch == download_batches.end()) {
                download_batches.push_back(DownloadBatch{.buffer_id = buffer_id, .copies = {copy}});
            } else {
                batch->copies.push_back(copy);
            }
        } else {
            runtime.CopyBuffer(download_staging.buffer, buffer, copies, false);
        }
        normalized_copies.push_back(second_copy);
    }
    if (batch_download_copies) {
        for (const auto& batch : download_batches) {
            runtime.CopyBuffer(download_staging.buffer, slot_buffers[batch.buffer_id], batch.copies,
                               false);
        }
    }
    runtime.PostCopyBarrier();
    pending_downloads.emplace_back(std::move(normalized_copies));
    pending_download_ranges.emplace_back(std::move(download_ranges));
    async_buffers.emplace_back(download_staging);
}

template <class P>
void BufferCache<P>::CommitAsyncFlushes() {
    CommitAsyncFlushesHigh();
}

template <class P>
void BufferCache<P>::PopAsyncFlushes() {
    PopAsyncBuffers();
}

template <class P>
void BufferCache<P>::PopAsyncBuffers() {
    Common::CopybackWatch::ScopedWriter copyback_scope{Common::CopybackWatch::Kind::BufferAsync};
    Nx2Profile::Scope profile_scope{Nx2Profile::Zone::DownloadRetire};
    if (async_buffers.empty()) {
        return;
    }
    if (!async_buffers.front().has_value()) {
        async_buffers.pop_front();
        return;
    }
    auto& downloads = pending_downloads.front();
    auto& ranges = pending_download_ranges.front();
    auto& async_buffer = async_buffers.front();
    u8* base = async_buffer->mapped_span.data();
    const size_t base_offset = async_buffer->offset;
    for (const auto& copy : downloads) {
        const DAddr device_addr = static_cast<DAddr>(copy.src_offset);
        const u64 dst_offset = copy.dst_offset - base_offset;
        const u8* read_mapped_memory = base + dst_offset;
        // Only what no write replaced since the copy was taken: the rest is older than guest
        // memory (CPU writes) or than the GPU's copy (GPU writes, still to be downloaded).
        ranges.ForEachInRange(device_addr, copy.size, [&](DAddr start, DAddr end) {
            {
                Nx2Profile::Scope copy_scope{Nx2Profile::Zone::DownloadCopy};
                device_memory.WriteBlockUnsafe(start, &read_mapped_memory[start - device_addr],
                                               end - start);
            }
            Nx2Profile::Add(Nx2Profile::Counter::DownloadCpuCopyBytes, end - start);
            Nx2Profile::Add(Nx2Profile::Counter::DownloadCpuCopyRanges, 1);
            gpu_modified_ranges.Subtract(start, end - start);
        });
    }
    async_buffers_death_ring.emplace_back(*async_buffer);
    async_buffers.pop_front();
    pending_downloads.pop_front();
    pending_download_ranges.pop_front();
}

template <class P>
bool BufferCache<P>::IsRegionGpuModified(DAddr addr, size_t size) {
    // One lookup instead of visiting every range inside (~4% of the GPU thread for title 1's
    // inline uploads once GPU-written ranges are no longer cleared by downloads).
    return gpu_modified_ranges.Intersects(addr, size);
}

template <class P>
bool BufferCache<P>::IsRegionRegistered(DAddr addr, size_t size) {
    const DAddr end_addr = addr + size;
    const u64 page_end = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    for (u64 page = addr >> CACHING_PAGEBITS; page < page_end;) {
        const BufferId buffer_id = page_table[page];
        if (!buffer_id) {
            ++page;
            continue;
        }
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buf_start_addr = buffer.CpuAddr();
        const DAddr buf_end_addr = buf_start_addr + buffer.SizeBytes();
        if (buf_start_addr < end_addr && addr < buf_end_addr) {
            return true;
        }
        page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    }
    return false;
}

template <class P>
bool BufferCache<P>::IsRegionCpuModified(DAddr addr, size_t size) {
    return memory_tracker.IsRegionCpuModified(addr, size);
}

template <class P>
void BufferCache<P>::BindHostIndexBuffer() {
    Buffer& buffer = slot_buffers[channel_state->index_buffer.buffer_id];
    TouchBuffer(buffer, channel_state->index_buffer.buffer_id);
    const u32 offset = buffer.Offset(channel_state->index_buffer.device_addr);
    const u32 size = channel_state->index_buffer.size;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (draw_state.inline_index_draw_indexes.empty()) {
        SynchronizeBuffer(buffer, channel_state->index_buffer.device_addr, size);
    } else {
        if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
            auto upload_staging = runtime.UploadStagingBuffer(size);
            std::array<BufferCopy, 1> copies{{BufferCopy{.src_offset = upload_staging.offset, .dst_offset = 0, .size = size}}};
            std::memcpy(upload_staging.mapped_span.data(), draw_state.inline_index_draw_indexes.data(), size);
            runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true);
        } else {
            buffer.ImmediateUpload(0, draw_state.inline_index_draw_indexes);
        }
    }
    if constexpr (HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const u32 new_offset = offset + draw_state.index_buffer.first * u32(draw_state.index_buffer.FormatSizeInBytes());
        runtime.BindIndexBuffer(buffer, new_offset, size);
    } else {
        buffer.MarkUsage(offset, size);
        runtime.BindIndexBuffer(draw_state.topology, draw_state.index_buffer.format, draw_state.index_buffer.first, draw_state.index_buffer.count, buffer, offset, size);
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size,
                                          u32 stride) {
    if constexpr (IS_OPENGL) {
        runtime.BindVertexBuffer(index, buffer, offset, size, stride);
    } else {
        runtime.BindVertexBuffer(index, buffer.Handle(), offset, size, stride);
    }
}

template <class P>
Binding& BufferCache<P>::VertexBufferSlot(u32 index) {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
const Binding& BufferCache<P>::VertexBufferSlot(u32 index) const {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
void BufferCache<P>::UpdateVertexBufferSlot(u32 index, const Binding& binding) {
    Binding& slot = VertexBufferSlot(index);
    if (slot.device_addr != binding.device_addr || slot.size != binding.size) {
        ++vertex_buffers_serial;
    }
    slot = binding;
    if (binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
        enabled_vertex_buffers_mask |= (1u << index);
    } else {
        enabled_vertex_buffers_mask &= ~(1u << index);
    }
}

template <class P>
void BufferCache<P>::MarkVertexBuffersStale(DAddr device_addr, u64 size) {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = enabled_vertex_buffers_mask;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            const Binding& binding = v_buffer[index];
            if (device_addr < binding.device_addr + binding.size &&
                binding.device_addr < device_addr + size) {
                stale_vertex_buffers_mask |= 1U << index;
            }
        }
    }
}

template <class P>
void BufferCache<P>::RefreshRepackedVertexBuffers() {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = std::exchange(stale_vertex_buffers_mask, 0) & enabled_vertex_buffers_mask;
        auto& flags = maxwell3d->dirty.flags;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            if (runtime.IsVertexRepacked(index)) {
                flags[Dirty::VertexBuffer0 + index] = true;
            }
        }
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffers() {
    RefreshRepackedVertexBuffers();

#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        auto& flags = maxwell3d->dirty.flags;
        u32 enabled_mask = enabled_vertex_buffers_mask;
        HostBindings<Buffer> bindings{};
        u32 last_index = (std::numeric_limits<u32>::max)();
        const auto flush_bindings = [&]() {
            if (bindings.buffers.empty()) {
                return;
            }
            bindings.max_index = bindings.min_index + static_cast<u32>(bindings.buffers.size());
            runtime.BindVertexBuffers(bindings);
            bindings = HostBindings<Buffer>{};
            last_index = (std::numeric_limits<u32>::max)();
        };
        while (enabled_mask != 0) {
            const u32 index = std::countr_zero(enabled_mask);
            enabled_mask &= (enabled_mask - 1);
            const Binding& binding = VertexBufferSlot(index);
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                flush_bindings();
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;
            const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
            const u32 offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, binding.size);
            if (!bindings.buffers.empty() && index != last_index + 1) {
                flush_bindings();
            }
            if (bindings.buffers.empty()) {
                bindings.min_index = index;
            }
            bindings.buffers.push_back(&buffer);
            bindings.offsets.push_back(offset);
            bindings.sizes.push_back(binding.size);
            bindings.strides.push_back(stride);
            last_index = index;
        }
        flush_bindings();
    } else {
        HostBindings<typename P::Buffer> host_bindings;
        bool any_valid{false};
        auto& flags = maxwell3d->dirty.flags;
        for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
            const Binding& binding = channel_state->vertex_buffers[index];
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;

            host_bindings.min_index = (std::min)(host_bindings.min_index, index);
            host_bindings.max_index = (std::max)(host_bindings.max_index, index);
            any_valid = true;
        }

        if (any_valid) {
            host_bindings.max_index++;
            for (u32 index = host_bindings.min_index; index < host_bindings.max_index; index++) {
                flags[Dirty::VertexBuffer0 + index] = false;

                const Binding& binding = channel_state->vertex_buffers[index];
                Buffer& buffer = slot_buffers[binding.buffer_id];

                const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
                const u32 offset = buffer.Offset(binding.device_addr);
                buffer.MarkUsage(offset, binding.size);

                host_bindings.buffers.push_back(&buffer);
                host_bindings.offsets.push_back(offset);
                host_bindings.sizes.push_back(binding.size);
                host_bindings.strides.push_back(stride);
            }
            runtime.BindVertexBuffers(host_bindings);
        }
    }
}

template <class P>
void BufferCache<P>::BindHostDrawIndirectBuffers() {
    const auto bind_buffer = [this](const Binding& binding) {
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        SynchronizeBuffer(buffer, binding.device_addr, binding.size);
    };
    if (current_draw_indirect->include_count) {
        bind_buffer(channel_state->count_buffer_binding);
    }
    bind_buffer(channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffers(size_t stage) {
    u32 dirty = ~0U;
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        dirty = std::exchange(channel_state->dirty_uniform_buffers[stage], 0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        const bool needs_bind = ((dirty >> index) & 1) != 0;
        BindHostGraphicsUniformBuffer(stage, index, binding_index, needs_bind);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            ++binding_index;
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind) {
    if (BindGraphicsUniformWriteSnapshot(stage, index, binding_index)) return;
    ++channel_state->uniform_cache_shots[0];
    // Bank 24 (root table 0, NVN2's push constants) is read from bank 17, which such games do not
    // bind: stream the table's current words as that buffer.
    if (index == 17 && maxwell3d->root_tables.loaded && channel_state->uniform_buffers[stage][index].size == 0) {
        const auto& words = maxwell3d->root_tables.tables[0];
        const u32 size = (*channel_state->uniform_buffer_sizes)[stage][index];
        const u32 copied = (std::min)(size, static_cast<u32>(sizeof(words)));
        if constexpr (IS_OPENGL) {
            std::array<u8, sizeof(words)> data{};
            std::memcpy(data.data(), words.data(), copied);
            runtime.BindFastUniformBuffer(stage, binding_index, copied);
            runtime.PushFastUniformBuffer(stage, binding_index, std::span<const u8>(data.data(), copied));
        } else {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            std::memset(span.data(), 0, span.size());
            std::memcpy(span.data(), words.data(), copied);
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        return;
    }
    const Binding& binding = channel_state->uniform_buffers[stage][index];
    const DAddr device_addr = binding.device_addr;
    const u32 size = (std::min)(binding.size, (*channel_state->uniform_buffer_sizes)[stage][index]);
    Buffer& buffer = slot_buffers[binding.buffer_id];
    TouchBuffer(buffer, binding.buffer_id);
    const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
    const u32 offset = has_host_buffer ? buffer.Offset(device_addr) : 0;
    const bool needs_alignment_stream = [&]() {
        if constexpr (IS_OPENGL) {
            return false;
        } else {
            if (!has_host_buffer) {
                return false;
            }
            const u32 alignment = runtime.GetUniformBufferAlignment();
            return alignment > 1 && (offset % alignment) != 0;
        }
    }();
    const bool use_fast_buffer = needs_alignment_stream
        || (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size
            && !memory_tracker.IsRegionGpuModified(device_addr, size));
    if (use_fast_buffer) {
        if constexpr (IS_OPENGL) {
            if (runtime.HasFastBufferSubData()) {
                // Fast path for Nvidia
                const bool should_fast_bind =
                    !HasFastUniformBufferBound(stage, binding_index) ||
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
                if (should_fast_bind) {
                    // We only have to bind when the currently bound buffer is not the fast version
                    channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
                    runtime.BindFastUniformBuffer(stage, binding_index, size);
                }
                const auto span = ImmediateBufferWithData(device_addr, size);
                runtime.PushFastUniformBuffer(stage, binding_index, span);
                return;
            }
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        // Stream buffer path to avoid stalling on non-Nvidia drivers or Vulkan
        bool streamed = false;
        if constexpr (!IS_OPENGL && VideoCommon::StreamUniformReusable<Runtime>) {
            // The same bytes as this slot's previous bind, in the same command buffer: bind that
            // stream allocation again instead of allocating and copying (uniform_stream_reuse.h).
            // EDEN_NX2_UBOREUSE=0 turns it off.
            if (VideoCommon::UniformStreamReuseEnabled()) {
                streamed = VideoCommon::BindSmallUniformReusing(
                    runtime, device_memory, stage, index, binding_index, device_addr, size);
            }
        }
        if (!streamed) {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        }
        return;
    }
    // Classic cached path
    if (SynchronizeBuffer(buffer, device_addr, size)) {
        ++channel_state->uniform_cache_hits[0];
    }
    // Skip binding if it's not needed and if the bound buffer is not the fast version
    // This exists to avoid instances where the fast buffer is bound and a GPU write happens
    needs_bind |= HasFastUniformBufferBound(stage, binding_index);
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        needs_bind |= channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
    }
    if (!needs_bind) {
        return;
    }
    if constexpr (IS_OPENGL) {
        // Mark the index as dirty if offset doesn't match
        const bool is_copy_bind = offset != 0 && !runtime.SupportsNonZeroUniformOffset();
        channel_state->dirty_uniform_buffers[stage] |= (is_copy_bind ? 1U : 0U) << index;
    }
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
    }
    buffer.MarkUsage(offset, size);
    if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
        runtime.BindUniformBuffer(stage, binding_index, buffer, offset, size);
    } else {
        runtime.BindUniformBuffer(buffer, offset, size);
    }
    channel_state->fast_bound_uniform_buffers[stage] &= ~(1u << binding_index);
}

template <class P>
void BufferCache<P>::ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                              std::vector<MultiRangeSegment>& pool) {
    binding.segment_first = 0;
    binding.segment_count = 0;
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.gpu_addr == 0 || binding.size == 0) {
            return;
        }
        if (is_written && !runtime.PrefersSparseSources()) {
            return;
        }
        const VirtualSegments* found =
            virtual_ranges.Query(*gpu_memory, binding.gpu_addr, binding.size);
        if (!found || found->size() < 2) {
            return;
        }
        const VirtualSegments segments = *found;
        const u32 first = static_cast<u32>(pool.size());
        const bool prefer_sparse = runtime.PrefersSparseSources();
        for (const VirtualSegment& segment : segments) {
            const BufferId buffer_id =
                FindBuffer(segment.device_addr, segment.size, prefer_sparse);
            if (!buffer_id) {
                pool.resize(first);
                return;
            }
            pool.push_back(MultiRangeSegment{
                .buffer_id = buffer_id,
                .device_addr = segment.device_addr,
                .size = segment.size,
            });
        }
        binding.segment_first = first;
        binding.segment_count = static_cast<u32>(segments.size());
    }
}

template <class P>
bool BufferCache<P>::BindMultiRangeStorage(const Binding& binding, bool is_written,
                                           std::span<const MultiRangeSegment> pool,
                                           const Nx2WriteProfile::Key* writer) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.segment_count < 2) {
            return false;
        }
        if (binding.segment_first + binding.segment_count > pool.size()) {
            return false;
        }
        const u64 key = (static_cast<u64>(gpu_memory->GetID()) << 48) ^ binding.gpu_addr;
        runtime.ResetMultiRange();
        for (u32 index = 0; index < binding.segment_count; ++index) {
            const MultiRangeSegment& segment = pool[binding.segment_first + index];
            Buffer& buffer = slot_buffers[segment.buffer_id];
            TouchBuffer(buffer, segment.buffer_id);
            if (SynchronizeBuffer(buffer, segment.device_addr, segment.size)) {
                runtime.InvalidateMultiRange(key);
            }
            const u32 offset = buffer.Offset(segment.device_addr);
            buffer.MarkUsage(offset, segment.size);
            if (is_written) {
                std::optional<Nx2WriteProfile::Key> segment_writer;
                if (Nx2WriteProfile::Enabled()) {
                    segment_writer = writer ? *writer : Nx2WriteProfile::Key{};
                    segment_writer->kind = writer && writer->kind == Nx2WriteProfile::Kind::ComputeStorage
                        ? Nx2WriteProfile::Kind::ComputeSegmented : Nx2WriteProfile::Kind::GraphicsSegmented;
                    segment_writer->proof = Nx2WriteProfile::Proof::None;
                    segment_writer->end = 0;
                }
                MarkWrittenBuffer(segment.buffer_id, segment.device_addr, segment.size, false,
                                  segment_writer ? &*segment_writer : nullptr, binding.size);
            }
            runtime.PushMultiRangeSource(buffer, offset, segment.size);
        }
        return runtime.BindMultiRangeStorageBuffer(key, is_written);
    } else {
        return false;
    }
}

template <class P>
void BufferCache<P>::BindHostGraphicsStorageBuffers(size_t stage, const Nx2WriteProfile::Stage* writer) {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        const Binding& binding = channel_state->storage_buffers[stage][index];
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        std::optional<Nx2WriteProfile::Key> attribution;
        if (Nx2WriteProfile::Enabled() && is_written)
            attribution = Nx2WriteProfile::StorageKey(writer, index,
                Nx2WriteProfile::Kind::GraphicsStorage, GraphicsUniformWriteSnapshotsValid());
        if (BindMultiRangeStorage(binding, is_written, graphics_segments,
                                  attribution ? &*attribution : nullptr)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size, false,
                              attribution ? &*attribution : nullptr);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindStorageBuffer(stage, binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsTextureBuffers(size_t stage, const Nx2WriteProfile::Stage* writer) {
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written = ((channel_state->written_texture_buffers[stage] >> index) & 1) != 0;
        if (is_written) {
            std::optional<Nx2WriteProfile::Key> attribution;
            if (Nx2WriteProfile::Enabled())
                attribution = Nx2WriteProfile::OtherKey(writer, Nx2WriteProfile::Kind::GraphicsTexture, index);
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size, false,
                              attribution ? &*attribution : nullptr);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_texture_buffers[stage] >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    HostBindings<typename P::Buffer> host_bindings;
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        const Binding& binding = channel_state->transform_feedback_buffers[index];
        const auto& control = maxwell3d->regs.transform_feedback.controls[index];
        const bool has_layout = control.varying_count != 0 || control.stride != 0;

        Buffer* host_buffer = &slot_buffers[NULL_BUFFER_ID];
        u32 offset = 0;
        u32 size = 0;

        if (has_layout && binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            size = binding.size;
            SynchronizeBuffer(buffer, binding.device_addr, size);
            std::optional<Nx2WriteProfile::Key> attribution;
            if (Nx2WriteProfile::Enabled())
                attribution = Nx2WriteProfile::OtherKey(nullptr, Nx2WriteProfile::Kind::TransformFeedback, index);
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size, false,
                              attribution ? &*attribution : nullptr);
            offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, size);
            host_buffer = &buffer;
        }

        host_bindings.buffers.push_back(host_buffer);
        host_bindings.offsets.push_back(offset);
        host_bindings.sizes.push_back(size);
    }
    runtime.BindTransformFeedbackBuffers(host_bindings);
}

template <class P>
void BufferCache<P>::BindHostComputeUniformBuffers() {
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        // Mark all uniform buffers as dirty
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->fast_bound_uniform_buffers.fill(0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        const Binding& binding = channel_state->compute_uniform_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size =
            (std::min)(binding.size, (*channel_state->compute_uniform_buffer_sizes)[index]);
        const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
        const u32 offset = has_host_buffer ? buffer.Offset(binding.device_addr) : 0;
        const bool needs_alignment_stream = [&]() {
            if constexpr (IS_OPENGL) {
                return false;
            } else {
                if (!has_host_buffer) {
                    return false;
                }
                const u32 alignment = runtime.GetUniformBufferAlignment();
                return alignment > 1 && (offset % alignment) != 0;
            }
        }();
        // Small buffers the GPU has not written are streamed, as graphics ones are: synchronizing
        // them made every dispatch re-protect pages an inline upload had just unprotected (two
        // VirtualProtect calls per dispatch, ~4,000 dispatches a second in title 1).
        const bool use_fast_buffer =
            needs_alignment_stream ||
            (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size &&
             !memory_tracker.IsRegionGpuModified(binding.device_addr, size));
        if constexpr (!IS_OPENGL) {
            if (use_fast_buffer) {
                const std::span<u8> span =
                    runtime.BindMappedUniformBuffer(0, binding_index, size);
                device_memory.ReadBlockUnsafe(binding.device_addr, span.data(), size);
                return;
            }
        }

        SynchronizeBuffer(buffer, binding.device_addr, size);

        buffer.MarkUsage(offset, size);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            runtime.BindComputeUniformBuffer(binding_index, buffer, offset, size);
            ++binding_index;
        } else {
            runtime.BindUniformBuffer(buffer, offset, size);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeStorageBuffers(const Nx2WriteProfile::Stage* writer,
                                                   std::span<const u32> dirty_limits) {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        const Binding& binding = channel_state->compute_storage_buffers[index];
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        std::optional<Nx2WriteProfile::Key> attribution;
        if (Nx2WriteProfile::Enabled() && is_written)
            attribution = Nx2WriteProfile::StorageKey(writer, index,
                Nx2WriteProfile::Kind::ComputeStorage);
        if (BindMultiRangeStorage(binding, is_written, compute_segments,
                                  attribution ? &*attribution : nullptr)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        if (is_written) {
            const u32 limit = index < dirty_limits.size() ? dirty_limits[index] : 0;
            const u32 dirty_size = limit && limit < size ? limit : size;
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, dirty_size, true,
                              attribution ? &*attribution : nullptr, size);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindComputeStorageBuffer(binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeTextureBuffers(const Nx2WriteProfile::Stage* writer) {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->compute_texture_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written =
            ((channel_state->written_compute_texture_buffers >> index) & 1) != 0;
        if (is_written) {
            std::optional<Nx2WriteProfile::Key> attribution;
            if (Nx2WriteProfile::Enabled())
                attribution = Nx2WriteProfile::OtherKey(writer, Nx2WriteProfile::Kind::ComputeTexture, index);
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size, false,
                              attribution ? &*attribution : nullptr);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_compute_texture_buffers >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::DoUpdateGraphicsBuffers(bool is_indexed) {
    graphics_segments.clear();
    BufferOperations([&]() {
        if (is_indexed) {
            UpdateIndexBuffer();
        }
        UpdateVertexBuffers();
        UpdateTransformFeedbackBuffers();
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            UpdateUniformBuffers(stage);
            UpdateStorageBuffers(stage);
            UpdateTextureBuffers(stage);
        }
        if (current_draw_indirect) {
            UpdateDrawIndirect();
        }
    });
}

template <class P>
void BufferCache<P>::DoUpdateComputeBuffers(std::span<NativeStorageDomainResult> domains) {
    compute_segments.clear();
    BufferOperations([&]() {
        UpdateComputeUniformBuffers();
        UpdateComputeStorageBuffers(domains);
        UpdateComputeTextureBuffers();
    });
}

template <class P>
void BufferCache<P>::UpdateIndexBuffer() {
    // We have to check for the dirty flags and index count
    // The index count is currently changed without updating the dirty flags
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto& index_buffer_ref = draw_state.index_buffer;
    auto& flags = maxwell3d->dirty.flags;
    if (!flags[Dirty::IndexBuffer]) {
        return;
    }
    flags[Dirty::IndexBuffer] = false;
    if (!draw_state.inline_index_draw_indexes.empty()) [[unlikely]] {
        auto inline_index_size = static_cast<u32>(draw_state.inline_index_draw_indexes.size());
        u32 buffer_size = Common::AlignUp(inline_index_size, CACHING_PAGESIZE);
        if (inline_buffer_id == NULL_BUFFER_ID) [[unlikely]] {
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        if (slot_buffers[inline_buffer_id].SizeBytes() < buffer_size) [[unlikely]] {
            slot_buffers.erase(inline_buffer_id);
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        channel_state->index_buffer = Binding{
            .device_addr = 0,
            .size = inline_index_size,
            .buffer_id = inline_buffer_id,
        };
        return;
    }

    const GPUVAddr gpu_addr_begin = index_buffer_ref.StartAddress();
    const GPUVAddr gpu_addr_end = index_buffer_ref.EndAddress();
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr_begin);
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    const u32 draw_size = (index_buffer_ref.count + index_buffer_ref.first) * u32(index_buffer_ref.FormatSizeInBytes());
    const u32 size = (std::min)(address_size, draw_size);
    if (size == 0 || !device_addr) {
        channel_state->index_buffer = NULL_BINDING;
        return;
    }
    channel_state->index_buffer = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = FindBuffer(*device_addr, size, false),
    };
}

template <class P>
void BufferCache<P>::UpdateVertexBuffers() {
    auto& flags = maxwell3d->dirty.flags;
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffers]) {
        return;
    }
    flags[Dirty::VertexBuffers] = false;

    for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
        UpdateVertexBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateVertexBuffer(u32 index) {
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffer0 + index]) {
        return;
    }
    const auto& array = maxwell3d->regs.vertex_streams[index];
    const auto& limit = maxwell3d->regs.vertex_stream_limits[index];
    const GPUVAddr gpu_addr_begin = array.Address();
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    const u64 native_size = native ? maxwell3d->regs.NativeVertexStreamSize(index) : 0;
    if (native && array.enable != 0 && native_size > (std::numeric_limits<u32>::max)()) {
        throw std::runtime_error("Native vertex stream exceeds supported 32-bit buffer size");
    }
    const GPUVAddr gpu_addr_end = native ? gpu_addr_begin + native_size : limit.Address() + 1;
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr_begin);
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    u32 size = address_size; // TODO: Analyze stride and number of vertices
    if (array.enable == 0 || size == 0 || !device_addr) {
        channel_state->vertex_buffers[index] = NULL_BINDING;
        UpdateVertexBufferSlot(index, NULL_BINDING);
        return;
    }
    if (!gpu_memory->IsWithinGPUAddressRange(gpu_addr_end) || size >= 64_MiB) {
        size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr_begin, size));
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    const Binding binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
    channel_state->vertex_buffers[index] = binding;
    UpdateVertexBufferSlot(index, binding);
}

template <class P>
void BufferCache<P>::UpdateDrawIndirect() {
    const auto update = [this](GPUVAddr gpu_addr, size_t size, Binding& binding) {
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        if (!device_addr) {
            binding = NULL_BINDING;
            return;
        }
        binding = Binding{
            .device_addr = *device_addr,
            .size = static_cast<u32>(size),
            .buffer_id = FindBuffer(*device_addr, static_cast<u32>(size), false),
        };
    };
    if (current_draw_indirect->include_count) {
        update(current_draw_indirect->count_start_address, sizeof(u32),
               channel_state->count_buffer_binding);
    }
    update(current_draw_indirect->indirect_start_address, current_draw_indirect->buffer_size,
           channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::UpdateUniformBuffers(size_t stage) {
    if (channel_state->enabled_uniform_buffer_masks[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        Binding& binding = channel_state->uniform_buffers[stage][index];
        if (binding.buffer_id) {
            // Already updated
            return;
        }
        // Mark as dirty
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers[stage] |= 1U << index;
        }
        // Resolve buffer
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateStorageBuffers(size_t stage) {
    if (channel_state->enabled_storage_buffers[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->storage_buffers[stage][index];
        const BufferId buffer_id = FindBuffer(binding.device_addr, binding.size, false);
        binding.buffer_id = buffer_id;
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        ResolveMultiRangeStorage(binding, is_written, graphics_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateTextureBuffers(size_t stage) {
    if (channel_state->enabled_texture_buffers[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        Binding& binding = channel_state->texture_buffers[stage][index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        UpdateTransformFeedbackBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffer(u32 index) {
    const auto& binding = maxwell3d->regs.transform_feedback.buffers[index];
    const GPUVAddr gpu_addr = binding.Address() + binding.start_offset;
    const u32 size = binding.size;
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (binding.enable == 0 || size == 0 || !device_addr) {
        channel_state->transform_feedback_buffers[index] = NULL_BINDING;
        return;
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    channel_state->transform_feedback_buffers[index] = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
}

template <class P>
void BufferCache<P>::UpdateComputeUniformBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        Binding& binding = channel_state->compute_uniform_buffers[index];
        binding = NULL_BINDING;
        const auto& launch_desc = kepler_compute->launch_description;
        if (((launch_desc.const_buffer_enable_mask >> index) & 1) != 0) {
            const auto& cbuf = launch_desc.const_buffer_config[index];
            const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(cbuf.Address());
            if (device_addr) {
                binding.device_addr = *device_addr;
                binding.size = cbuf.size;
            }
        }
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateComputeStorageBuffers(std::span<NativeStorageDomainResult> domains) {
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        Binding& binding = channel_state->compute_storage_buffers[index];
        auto* domain = index < domains.size() ? &domains[index] : nullptr;
        const u32 allocation_limit = domain ? domain->allocation_limit : 0;
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false, allocation_limit);
        if (allocation_limit && binding.buffer_id == NULL_BUFFER_ID) {
            binding.size = domain->previous_size;
            domain->bound_size = binding.size;
            domain->allocation_limit = 0;
            domain->applied = false;
            domain->reason = "merged owner exceeds resource or address limit";
            binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
        } else if (allocation_limit) {
            const Buffer& owner = slot_buffers[binding.buffer_id];
            domain->owner_suffix = owner.SizeBytes() - (binding.device_addr - owner.CpuAddr());
            domain->reason = "resolved exact accessible domain";
        }
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        if (domain && domain->applied && domain->allocation_limit) {
            // A fresh physical-domain proof must not be replaced by cached virtual segments.
            binding.segment_first = 0;
            binding.segment_count = 0;
        } else {
            ResolveMultiRangeStorage(binding, is_written, compute_segments);
        }
    });
}

template <class P>
void BufferCache<P>::UpdateComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        Binding& binding = channel_state->compute_texture_buffers[index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::TrackBufferWriteDownloads(DAddr device_addr, u64 size) {
    if (size == 0) {
        return;
    }
    Nx2Profile::Add(Nx2Profile::Counter::BufferWriteRequestedBytes, size);
    static const bool use_intersections = [] {
        const char* value = std::getenv("EDEN_NX2_DOWNLOAD_INTERSECTIONS");
        return value != nullptr && value[0] == '1';
    }();
    const auto queue_download = [&](DAddr begin, u64 length) {
        uncommitted_gpu_modified_ranges.Add(begin, length);
        Nx2Profile::Add(Nx2Profile::Counter::FenceDownloadQueuedBytes, length);
    };
    if (!use_intersections) {
        if (DownloadsAtFence(device_addr, size)) {
            queue_download(device_addr, size);
        } else {
            ArmReadTraps(device_addr, size);
        }
        return;
    }
    if (AsyncBufferDownloads()) {
        queue_download(device_addr, size);
        return;
    }

    // A conservative native binding may cover MiB while the CPU reads one page.
    // Keep fence downloads on the previously read pages and trap every other page.
    const DAddr write_end = device_addr + size;
    DAddr cursor = device_addr;
    memory_tracker.ForEachPreflushableRange(device_addr, size, [&](DAddr begin, u64 length) {
        const DAddr end = (std::min)(write_end, begin + length);
        begin = (std::max)(device_addr, begin);
        if (begin >= end) {
            return;
        }
        if (cursor < begin) {
            ArmReadTraps(cursor, begin - cursor);
        }
        queue_download(begin, end - begin);
        cursor = end;
    });
    if (cursor < write_end) {
        ArmReadTraps(cursor, write_end - cursor);
    }
}

template <class P>
void BufferCache<P>::MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync,
                                      const Nx2WriteProfile::Key* writer, u32 bound_size) {
    if (Nx2WriteProfile::Enabled()) {
        Nx2WriteProfile::Record(writer ? *writer : Nx2WriteProfile::Key{}, size,
                               bound_size ? bound_size : size, slot_buffers[buffer_id].SizeBytes());
    }
    Nx2Profile::Scope profile_scope{Nx2Profile::Zone::BufferWriteTrack};
    if constexpr (!IS_OPENGL) {
        if (needs_sync) {
            Buffer& buffer = slot_buffers[buffer_id];
            buffer.setWriteTick(runtime.CurrentTick());
        }
    }
    // The downloads already taken over this range hold older bytes: when their fences pop
    // they must not write them over this write's result in guest memory, nor drop the range
    // from gpu_modified_ranges (the next fence downloads it instead).
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    memory_tracker.MarkRegionAsGpuModified(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
    gpu_modified_ranges.Add(device_addr, size);
    TrackBufferWriteDownloads(device_addr, size);
    if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        Tegra::Nx2MarkGpuWritten(device_addr, size);
        Nx2NoteListWrite(buffer_id, device_addr, size);
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last;) {
            // Storage buffers can be large: skip 64 pages at a time where none is a list page.
            const u64 word =
                Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                (page % 64);
            if (word == 0) {
                page = (page / 64 + 1) * 64;
                continue;
            }
            if (word & 1) {
                const DAddr page_addr = page << page_bits;
                {
                    std::scoped_lock lock{nx2_list_mutex};
                    Nx2ListPage& entry = nx2_list_pages[page_addr];
                    ++entry.writes;
                    if (!nx2_collect_list_copies && !entry.draw_written) {
                        entry.draw_written = true;
                        LOG_INFO(HW_GPU, "Command list page {:#x} is written by draws too: its "
                                         "lists drain the split",
                                 page_addr);
                    }
                }
                if (nx2_collect_list_copies) {
                    nx2_list_copy_requests.emplace_back(buffer_id, page_addr);
                }
            }
            ++page;
        }
    }
}

template <class P>
bool BufferCache<P>::Nx2CopyListPages(u64 tick) {
    if constexpr (IS_OPENGL) {
        return false;
    } else {
        if (nx2_list_copy_requests.empty()) {
            return false;
        }
        std::ranges::sort(nx2_list_copy_requests, {}, &std::pair<BufferId, DAddr>::second);
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        bool recorded = false;
        for (size_t i = 0; i < nx2_list_copy_requests.size(); ++i) {
            const auto [buffer_id, page] = nx2_list_copy_requests[i];
            if (i + 1 < nx2_list_copy_requests.size() &&
                nx2_list_copy_requests[i + 1].second == page) {
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            const DAddr begin = (std::max<DAddr>)(page, buffer.CpuAddr());
            const DAddr end =
                (std::min<DAddr>)(page + page_size, buffer.CpuAddr() + buffer.SizeBytes());
            if (begin >= end) {
                continue;
            }
            Nx2Staging staging = runtime.DownloadStagingBuffer(page_size, true);
            const BufferCopy copy{
                .src_offset = buffer.Offset(begin),
                .dst_offset = staging.offset + (begin - page),
                .size = end - begin,
            };
            buffer.MarkUsage(copy.src_offset, copy.size);
            runtime.CopyBuffer(staging.buffer, buffer, std::span{&copy, 1}, true);
            recorded = true;
            std::scoped_lock lock{nx2_list_mutex};
            Nx2ListPage& entry = nx2_list_pages[page];
            if (entry.copy) {
                nx2_retired_list_copies.emplace_back(entry.copy_tick, std::move(*entry.copy));
            }
            entry.copy = staging;
            entry.copy_writes = entry.writes;
            entry.copy_tick = tick;
            entry.copy_begin = static_cast<u32>(begin - page);
            entry.copy_end = static_cast<u32>(end - page);
        }
        nx2_list_copy_requests.clear();
        std::erase_if(nx2_retired_list_copies, [this](auto& retired) {
            if (!runtime.IsFree(retired.first)) {
                return false;
            }
            runtime.FreeDeferredStagingBuffer(retired.second);
            return true;
        });
        return recorded;
    }
}

template <class P>
auto BufferCache<P>::Nx2ReadListCopy(DAddr device_addr, std::span<u8> out) -> Nx2ListRead {
    if constexpr (IS_OPENGL) {
        return Nx2ListRead::Unknown;
    } else {
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        const DAddr page = device_addr & ~(page_size - 1);
        const u64 offset = device_addr - page;
        if (offset + out.size() > page_size) {
            return Nx2ListRead::Unknown;
        }
        const auto current = [&](const Nx2ListPage& entry) {
            return entry.copy && !entry.draw_written && entry.copy_writes == entry.writes &&
                   offset >= entry.copy_begin && offset + out.size() <= entry.copy_end;
        };
        u64 tick{};
        u64 writes{};
        {
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page);
            if (it == nx2_list_pages.end()) {
                return Nx2ListRead::Unknown;
            }
            const Nx2ListPage& entry = it->second;
            // Pages the CPU writes the lists of (GPU-written once, long before) stay synced.
            if (!entry.draw_written && entry.synced_writes == entry.writes) {
                return Nx2ListRead::Synced;
            }
            if (!current(entry)) {
                return Nx2ListRead::Unknown;
            }
            // The lists after the first of a batch find the copy done: read it at once.
            if (runtime.IsFree(entry.copy_tick)) {
                std::memcpy(out.data(), entry.copy->mapped_span.data() + offset, out.size());
                return Nx2ListRead::Copied;
            }
            tick = entry.copy_tick;
            writes = entry.copy_writes;
        }
        // Submitted with the work that wrote the page (Nx2CopyListPages's caller flushes).
        {
            VideoCommon::Nx2Profile::WaitTag wait_tag{VideoCommon::Nx2Profile::WaitSource::ListCopy};
            runtime.WaitSubmittedTick(tick);
        }
        std::scoped_lock lock{nx2_list_mutex};
        const auto it = nx2_list_pages.find(page);
        if (it == nx2_list_pages.end() || !current(it->second) || it->second.copy_tick != tick ||
            it->second.copy_writes != writes) {
            return Nx2ListRead::Unknown;
        }
        std::memcpy(out.data(), it->second.copy->mapped_span.data() + offset, out.size());
        return Nx2ListRead::Copied;
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListSynced(DAddr page) {
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        Nx2ListPage& entry = nx2_list_pages[page];
        entry.synced_writes = entry.writes;
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListHostWrite(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last; ++page) {
            if (!((Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                   (page % 64)) &
                  1)) {
                continue;
            }
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page << page_bits);
            if (it == nx2_list_pages.end()) {
                continue;
            }
            Nx2ListPage& entry = it->second;
            const bool synced = entry.synced_writes == entry.writes;
            ++entry.writes;
            if (synced) {
                entry.synced_writes = entry.writes;
            }
        }
    }
}

template <class P>
BufferId BufferCache<P>::FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible,
                                   u32 allocation_limit) {
    if (allocation_limit) {
        constexpr u64 device_limit = u64{1} << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
        if (!size || size > allocation_limit || device_addr >= device_limit ||
            size > device_limit - device_addr) return NULL_BUFFER_ID;
    }
    const u64 page = device_addr >> CACHING_PAGEBITS;
    if (device_addr == 0 || page >= page_table.size()) {
        return NULL_BUFFER_ID;
    }
    const BufferId buffer_id = page_table[page];
    if (buffer_id) {
        Buffer& buffer = slot_buffers[buffer_id];
        if (buffer.IsInBounds(device_addr, size)) {
            const bool should_sync = Settings::IsGPUFenceBehaviorAccurate();
            if (should_sync) {
                SynchronizeBufferWrites(buffer);
            }
            bool usable = true;
            if constexpr (requires { buffer.IsSparseCompatible(); }) {
                if (sparse_compatible && !buffer.IsSparseCompatible()) {
                    usable = false;
                }
            }
            if (usable) {
                return buffer_id;
            }
        }
    }
    return CreateBuffer(device_addr, size, sparse_compatible, allocation_limit);
}

template <class P>
void BufferCache<P>::SynchronizeBufferWrites(Buffer& buffer) {
    if constexpr (!IS_OPENGL) {
        const u64 buffer_tick = buffer.getWriteTick();
        if (!runtime.IsFree(buffer_tick)) {
            VideoCommon::Nx2Profile::WaitTag wait_tag{VideoCommon::Nx2Profile::WaitSource::BufferWrite};
            runtime.Wait(buffer_tick);
        }
    }
}

template <class P>
typename BufferCache<P>::OverlapResult BufferCache<P>::ResolveOverlaps(DAddr device_addr,
                                                                       u32 wanted_size,
                                                                       u32 allocation_limit) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    DAddr begin = device_addr;
    DAddr end = device_addr + wanted_size;
    s64 stream_score = 0;
    bool has_stream_leap = false;
    bool committed = false;
    SCOPE_EXIT {
        if (allocation_limit && !committed) {
            for (const BufferId id : overlap_ids) slot_buffers[id].Unpick();
        }
    };
    const auto within_limit = [&] {
        constexpr u64 device_limit = u64{1} << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
        return begin <= end && end <= device_limit && end - begin <= allocation_limit &&
               Common::DivCeil(end, CACHING_PAGESIZE) <= page_table.size();
    };
    if (allocation_limit && (!begin || !wanted_size || !within_limit()))
        return OverlapResult{.valid = false};
    auto expand_begin = [&](DAddr add_value) {
        static constexpr DAddr min_page = CACHING_PAGESIZE + Core::DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = Core::DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    auto expand_end = [&](DAddr add_value) {
        static constexpr DAddr max_page = 1ULL << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        if (allocation_limit && device_addr >> CACHING_PAGEBITS >= page_table.size())
            return OverlapResult{.valid = false};
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS];
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.IsPicked()) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.Pick();
        const DAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        if (allocation_limit && overlap.SizeBytes() >
                (std::numeric_limits<DAddr>::max)() - overlap_device_addr)
            return OverlapResult{.valid = false};
        const DAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        if (allocation_limit && !within_limit()) return OverlapResult{.valid = false};
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (allocation_limit) {
                constexpr DAddr min_page = CACHING_PAGESIZE + Core::DEVICE_PAGESIZE;
                constexpr DAddr max_page = u64{1} << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
                constexpr DAddr leap = CACHING_PAGESIZE * 128;
                if ((expands_right && (begin < min_page || leap > begin - min_page)) ||
                    (expands_left && leap > max_page - end))
                    return OverlapResult{.valid = false};
            }
            if (expands_right) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_end(CACHING_PAGESIZE * 128);
            }
            if (allocation_limit && !within_limit()) return OverlapResult{.valid = false};
        }
    }
    committed = true;
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

template <class P>
void BufferCache<P>::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                                 bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    boost::container::small_vector<BufferCopy, 10> copies;
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    copies.push_back(BufferCopy{
        .src_offset = 0,
        .dst_offset = dst_base_offset,
        .size = overlap.SizeBytes(),
    });
    new_buffer.MarkUsage(copies[0].dst_offset, copies[0].size);
    runtime.CopyBuffer(new_buffer, overlap, copies, true);
#ifdef YUZU_LEGACY
    if (immediately_free)
        runtime.Finish();
#endif
    DeleteBuffer(overlap_id, true);
}

template <class P>
BufferId BufferCache<P>::CreateBuffer(DAddr device_addr, u32 wanted_size,
                                      bool sparse_compatible, u32 allocation_limit) {
    if (allocation_limit) {
        constexpr u64 device_limit = u64{1} << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
        if (!wanted_size || device_addr >= device_limit ||
            wanted_size > device_limit - device_addr) return NULL_BUFFER_ID;
        const DAddr rounded_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
        const DAddr rounded_begin = Common::AlignDown(device_addr, CACHING_PAGESIZE);
        if (!rounded_begin || rounded_end > device_limit ||
            rounded_end - rounded_begin > allocation_limit)
            return NULL_BUFFER_ID;
    }
    DAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size, allocation_limit);
    if (!overlap.valid) return NULL_BUFFER_ID;
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(runtime, overlap.begin, size, sparse_compatible);
    auto& new_buffer = slot_buffers[new_buffer_id];
    const size_t size_bytes = new_buffer.SizeBytes();
    runtime.ClearBuffer(new_buffer, 0, size_bytes, 0);
    new_buffer.MarkUsage(0, size_bytes);
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
    }
    Register(new_buffer_id);
    TouchBuffer(new_buffer, new_buffer_id);
    return new_buffer_id;
}

template <class P>
void BufferCache<P>::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

template <class P>
void BufferCache<P>::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <class P>
template <bool insert>
void BufferCache<P>::ChangeRegister(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    if (insert) {
        buffer_bytes += Common::AlignUp(size, 1024);
        total_used_memory += Common::AlignUp(size, 1024);
        buffer.setLRUID(lru_cache.Insert(buffer_id, frame_tick));
    } else {
        buffer_bytes -= Common::AlignUp(size, 1024);
        total_used_memory -= Common::AlignUp(size, 1024);
        lru_cache.Free(buffer.getLRUID());
    }
    const DAddr device_addr_begin = buffer.CpuAddr();
    const DAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page] = buffer_id;
        } else {
            page_table[page] = BufferId{};
        }
    }
}

template <class P>
void BufferCache<P>::TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept {
    if (buffer_id != NULL_BUFFER_ID) {
        lru_cache.Touch(buffer.getLRUID(), frame_tick);
    }
}

template <class P>
bool BufferCache<P>::SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size) {
#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        upload_copies.clear();
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        const DAddr buffer_start = buffer.cpu_addr_cached;
        const auto add_upload = [&](DAddr start, DAddr end) {
            if (start == end) return;
            const u64 range_size = end - start;
            upload_copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = start - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        };
        memory_tracker.ForEachUploadRange(device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            DAddr upload_start = device_addr_out;
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, [&](DAddr gpu_start, DAddr gpu_end) {
                add_upload(upload_start, gpu_start);
                upload_start = gpu_end;
            });
            add_upload(upload_start, device_addr_out + range_size);
        });
        if (total_size_bytes == 0) {
            return true;
        }
        const std::span<BufferCopy> copies_span(upload_copies.data(), upload_copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        // Mapped uploads carry their own barriers (or run in the upload command buffer, which
        // ends with one); another barrier here only ended the render pass (~50 times a frame in
        // Pokopia).
        if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
            any_buffer_uploaded = true;
        }
        return false;
    } else {
        boost::container::small_vector<BufferCopy, 4> copies;
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        DAddr buffer_start = buffer.CpuAddr();
        memory_tracker.ForEachUploadRange(device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = device_addr_out - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        });
        if (total_size_bytes == 0) {
            return true;
        }
        const std::span<BufferCopy> copies_span(copies.data(), copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        return false;
    }
}

template <class P>
void BufferCache<P>::UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                                  std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        MappedUploadMemory(buffer, total_size_bytes, copies);
    } else {
        ImmediateUploadMemory(buffer, largest_copy, copies);
    }
}

template <class P>
void BufferCache<P>::ImmediateUploadMemory([[maybe_unused]] Buffer& buffer,
                                           [[maybe_unused]] u64 largest_copy,
                                           [[maybe_unused]] std::span<const BufferCopy> copies) {
    if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
        std::span<u8> immediate_buffer;
        for (const BufferCopy& copy : copies) {
            std::span<const u8> upload_span;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            if (IsRangeGranular(device_addr, copy.size)) {
                auto* const ptr = device_memory.GetPointer<u8>(device_addr);
                if (ptr != nullptr) {
                    upload_span = std::span(ptr, copy.size);
                }
            } else {
                if (immediate_buffer.empty()) {
                    immediate_buffer = ImmediateBuffer(largest_copy);
                }
                device_memory.ReadBlockUnsafe(device_addr, immediate_buffer.data(), copy.size);
                upload_span = immediate_buffer.subspan(0, copy.size);
            }
            buffer.ImmediateUpload(copy.dst_offset, upload_span);
        }
    }
}

template <class P>
void BufferCache<P>::MappedUploadMemory([[maybe_unused]] Buffer& buffer,
                                        [[maybe_unused]] u64 total_size_bytes,
                                        [[maybe_unused]] std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS) {
        auto upload_staging = runtime.UploadStagingBuffer(total_size_bytes);
        const std::span<u8> staging_pointer = upload_staging.mapped_span;
        for (BufferCopy& copy : copies) {
            u8* const src_pointer = staging_pointer.data() + copy.src_offset;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            device_memory.ReadBlockUnsafe(device_addr, src_pointer, copy.size);
            // Large uploads are mostly geometry the host GPU now has its own copy of: TickFrame
            // marks their guest memory cold (see TextureCache::RefreshContents).
            if (copy.size >= 256_KiB) {
                uploaded_buffer_ranges.emplace_back(device_addr, copy.size);
            }
            // Apply the staging offset
            copy.src_offset += upload_staging.offset;
        }
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    }
}

template <class P>
bool BufferCache<P>::InlineMemory(DAddr dest_address, size_t copy_size,
                                  std::span<const u8> inlined_buffer) {
    const bool is_dirty = IsRegionRegistered(dest_address, copy_size);
    if (!is_dirty) {
        return false;
    }
    DAddr aligned_start = Common::AlignDown(dest_address, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(dest_address + copy_size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        return false;
    }

    InlineMemoryImplementation(dest_address, copy_size, inlined_buffer);

    return true;
}

template <class P>
void BufferCache<P>::InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                                std::span<const u8> inlined_buffer) {
    ClearDownload(dest_address, copy_size);
    gpu_modified_ranges.Subtract(dest_address, copy_size);
    MarkVertexBuffersStale(dest_address, copy_size);

    BufferId buffer_id = FindBuffer(dest_address, static_cast<u32>(copy_size), false);
    auto& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, dest_address, static_cast<u32>(copy_size));

    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        auto upload_staging = runtime.UploadStagingBuffer(copy_size);
        std::array copies{BufferCopy{
            .src_offset = upload_staging.offset,
            .dst_offset = buffer.Offset(dest_address),
            .size = copy_size,
        }};
        u8* const src_pointer = upload_staging.mapped_span.data();
        std::memcpy(src_pointer, inlined_buffer.data(), copy_size);
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    } else {
        buffer.ImmediateUpload(buffer.Offset(dest_address), inlined_buffer.first(copy_size));
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer) {
    DownloadBufferMemory(buffer, buffer.CpuAddr(), buffer.SizeBytes());
}

template <class P>
void BufferCache<P>::DownloadBuffersMemory(std::span<const BufferId> buffer_ids) {
    Common::CopybackWatch::ScopedWriter copyback_scope{Common::CopybackWatch::Kind::BufferReactive};
    Nx2Profile::Scope profile_scope{Nx2Profile::Zone::DownloadReactive};
    struct Download {
        BufferId buffer_id;
        BufferCopy copy;
    };
    std::vector<Download> downloads;
    u64 total_size_bytes = 0;
    for (const BufferId buffer_id : buffer_ids) {
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buffer_addr = buffer.CpuAddr();
        memory_tracker.ForEachDownloadRangeAndClear(
            buffer_addr, buffer.SizeBytes(), [&](u64 device_addr_out, u64 range_size) {
                gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                   [&](DAddr start, DAddr end) {
                    downloads.push_back({buffer_id, BufferCopy{
                                                        .src_offset = start - buffer_addr,
                                                        .dst_offset = total_size_bytes,
                                                        .size = end - start,
                                                    }});
                    total_size_bytes += Common::AlignUp(end - start, 64);
                });
                ClearDownload(device_addr_out, range_size);
                gpu_modified_ranges.Subtract(device_addr_out, range_size);
            });
    }
    if (total_size_bytes == 0) {
        return;
    }
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
    for (Download& download : downloads) {
        download.copy.dst_offset += download_staging.offset;
        Buffer& buffer = slot_buffers[download.buffer_id];
        buffer.MarkUsage(download.copy.src_offset, download.copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, std::span{&download.copy, 1}, true);
    }
    {
        VideoCommon::Nx2Profile::WaitTag wait_tag{VideoCommon::Nx2Profile::WaitSource::BufferDownload, true};
        runtime.Finish();
    }
    const u8* const mapped_memory = download_staging.mapped_span.data();
    for (const Download& download : downloads) {
        const DAddr device_addr = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        {
            Nx2Profile::Scope copy_scope{Nx2Profile::Zone::DownloadCopy};
            device_memory.WriteBlockUnsafe(
                device_addr, mapped_memory + (download.copy.dst_offset - download_staging.offset),
                download.copy.size);
        }
        Nx2Profile::Add(Nx2Profile::Counter::DownloadCpuCopyBytes, download.copy.size);
        Nx2Profile::Add(Nx2Profile::Counter::DownloadCpuCopyRanges, 1);
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer, DAddr device_addr, u64 size) {
    Common::CopybackWatch::ScopedWriter copyback_scope{Common::CopybackWatch::Kind::BufferReactive};
    Nx2Profile::Scope profile_scope{Nx2Profile::Zone::DownloadReactive};
    boost::container::small_vector<BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    memory_tracker.ForEachDownloadRangeAndClear(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const DAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](DAddr start, DAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(BufferCopy{
                    .src_offset = new_offset,
                    .dst_offset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
                largest_copy = (std::max)(largest_copy, new_size);
            };

            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            ClearDownload(device_addr_out, range_size);
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }

    if constexpr (USE_MEMORY_MAPS) {
        auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
        const u8* const mapped_memory = download_staging.mapped_span.data();
        const std::span<BufferCopy> copies_span(copies.data(), copies.data() + copies.size());
        for (BufferCopy& copy : copies) {
            // Modify copies to have the staging offset in mind
            copy.dst_offset += download_staging.offset;
            buffer.MarkUsage(copy.src_offset, copy.size);
        }
        runtime.CopyBuffer(download_staging.buffer, buffer, copies_span, true);
        {
            VideoCommon::Nx2Profile::WaitTag wait_tag{VideoCommon::Nx2Profile::WaitSource::BufferDownload, true};
            runtime.Finish();
        }
        for (const BufferCopy& copy : copies) {
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            // Undo the modified offset
            const u64 dst_offset = copy.dst_offset - download_staging.offset;
            const u8* copy_mapped_memory = mapped_memory + dst_offset;
            {
                Nx2Profile::Scope copy_scope{Nx2Profile::Zone::DownloadCopy};
                device_memory.WriteBlockUnsafe(copy_device_addr, copy_mapped_memory, copy.size);
            }
            Nx2Profile::Add(Nx2Profile::Counter::DownloadCpuCopyBytes, copy.size);
            Nx2Profile::Add(Nx2Profile::Counter::DownloadCpuCopyRanges, 1);
        }
    } else {
        const std::span<u8> immediate_buffer = ImmediateBuffer(largest_copy);
        for (const BufferCopy& copy : copies) {
            buffer.ImmediateDownload(copy.src_offset, immediate_buffer.subspan(0, copy.size));
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            {
                Nx2Profile::Scope copy_scope{Nx2Profile::Zone::DownloadCopy};
                device_memory.WriteBlockUnsafe(copy_device_addr, immediate_buffer.data(), copy.size);
            }
            Nx2Profile::Add(Nx2Profile::Counter::DownloadCpuCopyBytes, copy.size);
            Nx2Profile::Add(Nx2Profile::Counter::DownloadCpuCopyRanges, 1);
        }
    }
}

template <class P>
void BufferCache<P>::DeleteBuffer(BufferId buffer_id, bool do_not_mark) {
    if constexpr (requires { runtime.OnBufferDeleted(slot_buffers[buffer_id]); }) {
        runtime.OnBufferDeleted(slot_buffers[buffer_id]);
    }
    read_trapped_ranges.Subtract(slot_buffers[buffer_id].CpuAddr(),
                                 slot_buffers[buffer_id].SizeBytes());
    bool dirty_index{false};
    boost::container::small_vector<u64, NUM_VERTEX_BUFFERS> dirty_vertex_buffers;
    const auto scalar_replace = [buffer_id](Binding& binding) {
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
        }
    };
    const auto replace = [scalar_replace](std::span<Binding> bindings) {
        std::ranges::for_each(bindings, scalar_replace);
    };

    if (channel_state->index_buffer.buffer_id == buffer_id) {
        channel_state->index_buffer.buffer_id = BufferId{};
        dirty_index = true;
    }

    for (u32 index = 0; index < channel_state->vertex_buffers.size(); index++) {
        auto& binding = channel_state->vertex_buffers[index];
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
            UpdateVertexBufferSlot(index, binding);
            dirty_vertex_buffers.push_back(index);
        }
    }
    std::ranges::for_each(channel_state->uniform_buffers, replace);
    std::ranges::for_each(channel_state->storage_buffers, replace);
    replace(channel_state->transform_feedback_buffers);
    replace(channel_state->compute_uniform_buffers);
    replace(channel_state->compute_storage_buffers);

    // Mark the whole buffer as CPU written to stop tracking CPU writes
    if (!do_not_mark) {
        Buffer& buffer = slot_buffers[buffer_id];
        memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
    }

    Unregister(buffer_id);

#ifdef YUZU_LEGACY
    if (!do_not_mark || !immediately_free)
#endif
        delayed_destruction_ring.Push(std::move(slot_buffers[buffer_id]));

    slot_buffers.erase(buffer_id);

    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->uniform_buffer_binding_sizes.fill({});
    }

    auto& flags = maxwell3d->dirty.flags;
    if (dirty_index) {
        flags[Dirty::IndexBuffer] = true;
    }

    if (dirty_vertex_buffers.size() > 0) {
        flags[Dirty::VertexBuffers] = true;
        for (auto index : dirty_vertex_buffers) {
            flags[Dirty::VertexBuffer0 + index] = true;
        }
    }
    channel_state->has_deleted_buffers = true;
}

template <class P>
std::optional<GPUVAddr> BufferCache<P>::ReadStoragePointer(GPUVAddr address) {
    constexpr u64 Bytes = sizeof(GPUVAddr);
    if ((address & 3) != 0 ||
        address > (std::numeric_limits<GPUVAddr>::max)() - (Bytes - 1) ||
        !gpu_memory->IsWithinGPUAddressRange(address) ||
        !gpu_memory->IsWithinGPUAddressRange(address + Bytes - 1) ||
        !gpu_memory->IsFullyMappedRange(address, Bytes)) {
        return std::nullopt;
    }
    // MemoryManager::Read does not flush GPU-written buffer bytes. Calling its
    // flushing ReadBlock here would reacquire our mutex through the rasterizer.
    // Resolve each device page directly, using the existing synchronous download
    // only when dirty; clean CPU-authored pointer tables do not wait for the GPU.
    for (u64 offset = 0; offset < Bytes;) {
        const GPUVAddr cursor = address + offset;
        const u64 bytes = (std::min)(Bytes - offset, u64{DEVICE_PAGESIZE} -
                                                     (cursor & (DEVICE_PAGESIZE - 1)));
        const auto device_address = gpu_memory->GpuToCpuAddress(cursor);
        if (!device_address) return std::nullopt;
        if (IsRegionGpuModified(*device_address, bytes)) {
            VideoCommon::Nx2Profile::WaitTag wait_tag{VideoCommon::Nx2Profile::WaitSource::StoragePointer};
            DownloadMemory(*device_address, bytes);
        }
        offset += bytes;
    }
    GPUVAddr value{};
    gpu_memory->ReadBlockUnsafe(address, &value, Bytes);
    return value;
}

template <class P>
std::optional<GPUVAddr> BufferCache<P>::ResolveGraphicsStorageBufferPointer(
    size_t stage, u32 cbuf_index, u32 cbuf_offset, u32 pointer_offset,
    GraphicsStoragePointerGuard* guard) {
    if (guard) *guard = {};
    if (stage >= maxwell3d->state.shader_stages.size() || pointer_offset == ~0U) {
        return std::nullopt;
    }
    const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
    if (cbuf_index >= cbufs.size()) return std::nullopt;
    const auto& cbuf = cbufs[cbuf_index];
    if (!cbuf.enabled || cbuf_offset > cbuf.size ||
        sizeof(GPUVAddr) > cbuf.size - cbuf_offset ||
        cbuf.address > (std::numeric_limits<GPUVAddr>::max)() - cbuf_offset) {
        return std::nullopt;
    }
    const auto root = ReadStoragePointer(cbuf.address + cbuf_offset);
    if (!root || *root == 0 ||
        *root > (std::numeric_limits<GPUVAddr>::max)() - pointer_offset ||
        !gpu_memory->IsWithinGPUAddressRange(*root) ||
        !gpu_memory->IsFullyMappedRange(*root, 1)) {
        return std::nullopt;
    }
    // Direct descriptors which supply a derived pointer must use native sizing
    // too. A mapped word outside that actual parent extent cannot be read by the
    // shader's parent SSBO, even when it is accessible to this host resolver.
    const size_t mapped_size =
        (std::min)(gpu_memory->GetMemoryLayoutSize(*root), static_cast<size_t>(8_MiB));
    const size_t parent_size = gpu_memory->MaxContinuousRange(*root, mapped_size);
    if (pointer_offset > parent_size || sizeof(GPUVAddr) > parent_size - pointer_offset) {
        return std::nullopt;
    }
    const auto target = ReadStoragePointer(*root + pointer_offset);
    if (!target || *target == 0 || !gpu_memory->IsWithinGPUAddressRange(*target) ||
        !gpu_memory->IsFullyMappedRange(*target, 1)) {
        return std::nullopt;
    }
    if (guard) {
        const GPUVAddr root_address = cbuf.address + cbuf_offset;
        const GPUVAddr entry_address = *root + pointer_offset;
        const auto root_device = gpu_memory->GpuToCpuAddress(root_address);
        const auto entry_device = gpu_memory->GpuToCpuAddress(entry_address);
        if (root_device && entry_device &&
            gpu_memory->MaxContinuousRange(root_address, sizeof(GPUVAddr)) == sizeof(GPUVAddr) &&
            gpu_memory->MaxContinuousRange(entry_address, sizeof(GPUVAddr)) == sizeof(GPUVAddr)) {
            *guard = {*root_device, *entry_device, true};
        }
    }
    return target;
}

template <class P>
void BufferCache<P>::ResetGraphicsUniformWriteSnapshots() {
    graphics_uniform_write_snapshots.clear();
    graphics_uniform_write_snapshots_valid = true;
}

template <class P>
auto BufferCache<P>::ResolveGraphicsUniformWriteSpan(
    size_t stage, u32 bank, u32 offset, s32 displacement, u32 width)
    -> std::optional<GraphicsUniformWriteSpan> {
    if constexpr (IS_OPENGL) return std::nullopt;
    if (stage >= NUM_STAGES || bank >= NUM_GRAPHICS_UNIFORM_BUFFERS || width != 4 ||
        (displacement != 0 && displacement != -16) || (offset & 7) != 0 ||
        !channel_state->uniform_buffer_sizes ||
        (channel_state->enabled_uniform_buffer_masks[stage] & (1U << bank)) == 0)
        return std::nullopt;
    const auto& cbuf = maxwell3d->state.shader_stages[stage].const_buffers[bank];
    const u32 size = (std::min)(cbuf.size, (*channel_state->uniform_buffer_sizes)[stage][bank]);
    if (!cbuf.enabled || size == 0 || size > GRAPHICS_UNIFORM_PROOF_BYTES ||
        offset > size || sizeof(u64) > size - offset ||
        cbuf.address > (std::numeric_limits<GPUVAddr>::max)() - (size - 1) ||
        !gpu_memory->IsWithinGPUAddressRange(cbuf.address) ||
        !gpu_memory->IsWithinGPUAddressRange(cbuf.address + size - 1) ||
        !gpu_memory->IsFullyMappedRange(cbuf.address, size) ||
        gpu_memory->MaxContinuousRange(cbuf.address, size) != size)
        return std::nullopt;
    const auto device_addr = gpu_memory->GpuToCpuAddress(cbuf.address);
    if (!device_addr || *device_addr > (std::numeric_limits<DAddr>::max)() - (size - 1) ||
        IsRegionGpuModified(*device_addr, size)) return std::nullopt;
    auto snapshot = std::ranges::find_if(graphics_uniform_write_snapshots,
        [&](const auto& item) { return item.stage == stage && item.bank == bank; });
    if (snapshot == graphics_uniform_write_snapshots.end()) {
        if (graphics_uniform_write_snapshots.size() == GRAPHICS_UNIFORM_PROOF_COUNT)
            return std::nullopt;
        auto& added = graphics_uniform_write_snapshots.emplace_back();
        added.stage = stage;
        added.bank = bank;
        added.gpu_addr = cbuf.address;
        added.device_addr = *device_addr;
        added.size = size;
        device_memory.ReadBlockUnsafe(*device_addr, added.bytes.data(), size);
        snapshot = std::prev(graphics_uniform_write_snapshots.end());
    } else if (snapshot->gpu_addr != cbuf.address || snapshot->device_addr != *device_addr ||
               snapshot->size != size) {
        return std::nullopt;
    }
    u64 value{};
    std::memcpy(&value, snapshot->bytes.data() + offset, sizeof(value));
    const u64 subtract = displacement < 0 ? static_cast<u64>(-s64{displacement}) : 0;
    if (value < subtract) return std::nullopt;
    value -= subtract;
    const u64 alignment = runtime.GetStorageBufferAlignment();
    if (alignment == 0 || !std::has_single_bit(alignment) || alignment > 0xffff ||
        value > 0xffff - (alignment - 1) || width > 0xffff - value - (alignment - 1))
        return std::nullopt;
    return GraphicsUniformWriteSpan{static_cast<u16>(value + alignment - 1 + width),
                                    {*device_addr, size}};
}

template <class P>
bool BufferCache<P>::BindGraphicsUniformWriteSnapshot(size_t stage, u32 bank, u32 binding_index) {
    if constexpr (IS_OPENGL) {
        return false;
    } else {
        if (graphics_uniform_write_snapshots.empty()) return false;
        const auto snapshot = std::ranges::find_if(graphics_uniform_write_snapshots,
            [&](const auto& item) { return item.stage == stage && item.bank == bank; });
        if (snapshot == graphics_uniform_write_snapshots.end()) return false;
        const auto& cbuf = maxwell3d->state.shader_stages[stage].const_buffers[bank];
        const auto& binding = channel_state->uniform_buffers[stage][bank];
        const u32 size = (std::min)(binding.size,
            (*channel_state->uniform_buffer_sizes)[stage][bank]);
        const auto device_addr = gpu_memory->GpuToCpuAddress(cbuf.address);
        if (!cbuf.enabled || cbuf.address != snapshot->gpu_addr ||
            (std::min)(cbuf.size, (*channel_state->uniform_buffer_sizes)[stage][bank]) != snapshot->size ||
            !device_addr || *device_addr != snapshot->device_addr ||
            binding.device_addr != snapshot->device_addr || size != snapshot->size ||
            binding.buffer_id == NULL_BUFFER_ID ||
            !gpu_memory->IsFullyMappedRange(cbuf.address, size) ||
            gpu_memory->MaxContinuousRange(cbuf.address, size) != size ||
            IsRegionGpuModified(snapshot->device_addr, size)) {
            // Configure rejects the draw after descriptor preparation; never use a fresh
            // uniform upload together with a proof based on different bytes.
            graphics_uniform_write_snapshots_valid = false;
            return false;
        }
        const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
        std::memcpy(span.data(), snapshot->bytes.data(), size);
        channel_state->fast_bound_uniform_buffers[stage] |= 1U << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        return true;
    }
}

template <class P>
void BufferCache<P>::GuardGraphicsStoragePointerWrites(
    size_t stage, size_t ssbo_index, const GraphicsStoragePointerGuard& guard, u32 active_stages,
    const GraphicsStorageWriteBounds& write_bounds,
    std::span<const GraphicsUniformWriteSource> uniform_sources) {
    if (stage >= NUM_STAGES || ssbo_index >= NUM_STORAGE_BUFFERS) return;
    const auto overlaps = [](DAddr protected_address, u64 protected_size, const Binding& written) {
        if (written.size == 0 || written.buffer_id == NULL_BUFFER_ID) return false;
        if (protected_size == 0 ||
            protected_address > (std::numeric_limits<DAddr>::max)() - (protected_size - 1) ||
            written.device_addr > (std::numeric_limits<DAddr>::max)() - (written.size - 1))
            return true;
        // Subtraction avoids overflowing either interval's end address.
        return protected_address >= written.device_addr
            ? protected_address - written.device_addr < written.size
            : written.device_addr - protected_address < protected_size;
    };
    bool safe = guard.valid;
    for (size_t other_stage = 0; safe && other_stage < NUM_STAGES; ++other_stage) {
        if ((active_stages & (1U << other_stage)) == 0) continue;
        u32 written = channel_state->enabled_storage_buffers[other_stage] &
                      channel_state->written_storage_buffers[other_stage];
        while (written != 0) {
            const u32 index = std::countr_zero(written);
            written &= written - 1;
            const Binding& binding = channel_state->storage_buffers[other_stage][index];
            if (binding.size == 0 || binding.buffer_id == NULL_BUFFER_ID) continue;
            const u16 end = write_bounds[other_stage][index];
            const auto device_address = gpu_memory->GpuToCpuAddress(binding.gpu_addr);
            if (end == 0 || end > binding.size || !device_address ||
                *device_address != binding.device_addr ||
                !gpu_memory->IsFullyMappedRange(binding.gpu_addr, binding.size) ||
                gpu_memory->MaxContinuousRange(binding.gpu_addr, binding.size) != binding.size) {
                safe = false;
                break;
            }
            if (overlaps(guard.root, sizeof(GPUVAddr), binding) ||
                overlaps(guard.entry, sizeof(GPUVAddr), binding) ||
                std::ranges::any_of(uniform_sources, [&](const auto& source) {
                    return overlaps(source.device_addr, source.size, binding);
                })) {
                safe = false;
                break;
            }
        }
    }
    if (!safe) channel_state->storage_buffers[stage][ssbo_index] = NULL_BINDING;
}

template <class P>
std::optional<GPUVAddr> BufferCache<P>::ResolveComputeStorageBufferPointer(
    u32 cbuf_index, u32 cbuf_offset, u32 pointer_offset) {
    if (!kepler_compute || !kepler_compute->IsNativeMode()) {
        return std::nullopt;
    }
    const auto& qmd = kepler_compute->launch_description;
    const auto& cbufs = qmd.const_buffer_config;
    // Validate the bank before indexing it or shifting the enable mask.
    if (cbuf_index >= cbufs.size() ||
        ((qmd.const_buffer_enable_mask.Value() >> cbuf_index) & 1U) == 0) {
        return std::nullopt;
    }
    const auto& cbuf = cbufs[cbuf_index];
    const u32 size = cbuf.size.Value();
    const GPUVAddr address = cbuf.Address();
    if (cbuf_offset > size || sizeof(GPUVAddr) > size - cbuf_offset ||
        address > (std::numeric_limits<GPUVAddr>::max)() - cbuf_offset) {
        return std::nullopt;
    }
    const auto root = ReadStoragePointer(address + cbuf_offset);
    if (!root || *root == 0 ||
        !gpu_memory->IsWithinGPUAddressRange(*root) ||
        !gpu_memory->IsFullyMappedRange(*root, 1)) {
        return std::nullopt;
    }
    if (pointer_offset == ~0U) return root;
    if (*root > (std::numeric_limits<GPUVAddr>::max)() - pointer_offset) return std::nullopt;
    // The runtime backend reads through the retained direct parent SSBO. The entry must fit
    // its actual native-sized extent, including any shorter mapping/physical-contiguity limit.
    const size_t mapped_size =
        (std::min)(gpu_memory->GetMemoryLayoutSize(*root), static_cast<size_t>(8_MiB));
    const size_t parent_size = gpu_memory->MaxContinuousRange(*root, mapped_size);
    if (pointer_offset > parent_size || sizeof(GPUVAddr) > parent_size - pointer_offset) {
        return std::nullopt;
    }
    const auto target = ReadStoragePointer(*root + pointer_offset);
    if (!target || *target == 0 || !gpu_memory->IsWithinGPUAddressRange(*target) ||
        !gpu_memory->IsFullyMappedRange(*target, 1)) {
        return std::nullopt;
    }
    return target;
}

template <class P>
std::optional<Binding> BufferCache<P>::GraphicsIndirectRangeBinding(GPUVAddr gpu_addr) const {
    if (!gpu_addr || !gpu_memory->IsWithinGPUAddressRange(gpu_addr)) return std::nullopt;
    const u32 alignment = runtime.GetStorageBufferAlignment();
    if (!alignment || (alignment & (alignment - 1)) != 0) return std::nullopt;
    const GPUVAddr aligned = Common::AlignDown(gpu_addr, alignment);
    const u64 prefix = gpu_addr - aligned;
    u64 limit = u64{64} << 20;
    if constexpr (requires { runtime.GetMaxStorageBufferRange(); }) {
        limit = (std::min)(limit, u64{runtime.GetMaxStorageBufferRange()});
    } else {
        return std::nullopt;
    }
    limit = (std::min)(limit, u64{(std::numeric_limits<u32>::max)()});
    if (prefix >= limit) return std::nullopt;
    const u64 payload = (std::min)(u64{gpu_memory->GetMemoryLayoutSize(gpu_addr)}, limit - prefix);
    if (!payload || gpu_addr > (std::numeric_limits<GPUVAddr>::max)() - (payload - 1) ||
        !gpu_memory->IsWithinGPUAddressRange(gpu_addr + payload - 1)) return std::nullopt;
    const auto aligned_device = gpu_memory->GpuToCpuAddress(aligned);
    const auto device = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!aligned_device || !device || *device < *aligned_device ||
        *device - *aligned_device != prefix) return std::nullopt;
    const u64 contiguous = gpu_memory->MaxContinuousRange(aligned, prefix + payload);
    const u64 size = (std::min)(prefix + payload, contiguous);
    if (size <= prefix) return std::nullopt;
    return Binding{.device_addr = *aligned_device, .gpu_addr = aligned,
                   .size = static_cast<u32>(size), .buffer_id = BufferId{}};
}

template <class P>
std::optional<Binding> BufferCache<P>::DiagnosticStorageBufferBinding(
    GPUVAddr gpu_addr, u32 minimum, StorageRangeDiagnostic& diagnostic) const {
    diagnostic = StorageRangeDiagnostic{.requested = minimum, .reason = "invalid request"};
    if (minimum == 0 || gpu_addr == 0 ||
        gpu_addr > (std::numeric_limits<GPUVAddr>::max)() - (minimum - 1) ||
        !gpu_memory->IsWithinGPUAddressRange(gpu_addr) ||
        !gpu_memory->IsWithinGPUAddressRange(gpu_addr + minimum - 1)) return std::nullopt;
    const u32 alignment = runtime.GetStorageBufferAlignment();
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return std::nullopt;
    const GPUVAddr aligned = Common::AlignDown(gpu_addr, alignment);
    const u64 prefix = gpu_addr - aligned;
    diagnostic.alignment_prefix = prefix;
    if constexpr (requires { runtime.GetMaxStorageBufferRange(); }) {
        diagnostic.max_descriptor_range = runtime.GetMaxStorageBufferRange();
    }
    diagnostic.reason = "descriptor limit";
    if (prefix > diagnostic.max_descriptor_range ||
        minimum > diagnostic.max_descriptor_range - prefix ||
        prefix > (std::numeric_limits<u32>::max)() ||
        minimum > (std::numeric_limits<u32>::max)() - prefix) return std::nullopt;
    diagnostic.mapped_size = gpu_memory->GetMemoryLayoutSize(gpu_addr);
    diagnostic.reason = "mapping extent";
    if (minimum > diagnostic.mapped_size) return std::nullopt;
    const u64 exact_size = prefix + minimum;
    const auto aligned_device = gpu_memory->GpuToCpuAddress(aligned);
    const auto device = gpu_memory->GpuToCpuAddress(gpu_addr);
    diagnostic.reason = "alignment translation";
    if (!aligned_device || !device || *device < *aligned_device ||
        *device - *aligned_device != prefix) return std::nullopt;
    // Include the alignment prefix: no holes or physical discontinuities are accepted.
    diagnostic.contiguous_size = gpu_memory->MaxContinuousRange(aligned, exact_size);
    diagnostic.reason = "physical continuity";
    if (diagnostic.contiguous_size < exact_size) return std::nullopt;
    diagnostic.applied = true;
    diagnostic.reason = "applied exact diagnostic extent";
    // Buffer allocation still uses cache pages, but the descriptor itself is exact. Do not
    // round its end beyond the verified mapping or maxStorageBufferRange.
    return Binding{.device_addr = *aligned_device, .gpu_addr = aligned,
                   .size = static_cast<u32>(exact_size), .buffer_id = BufferId{}};
}

/// Rate-limited report of storage bindings grown past their default size by shader feedback.
inline void LogStorageGrowth(GPUVAddr gpu_addr, size_t default_size, size_t grown_size) {
    static std::atomic<u64> grown{};
    static std::atomic<size_t> largest{};
    static std::atomic<s64> last_log{};
    grown.fetch_add(1, std::memory_order_relaxed);
    if (grown_size > largest.load(std::memory_order_relaxed)) {
        largest.store(grown_size, std::memory_order_relaxed);
    }
    const s64 now = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    if (now - last_log.load(std::memory_order_relaxed) >= 5) {
        last_log.store(now, std::memory_order_relaxed);
        LOG_INFO(HW_GPU, "Storage feedback grew {} bindings (largest {:#x}, latest {:#x}: {:#x} -> {:#x})",
                 grown.load(), largest.load(), gpu_addr, default_size, grown_size);
    }
}

template <class P>
Binding BufferCache<P>::StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                             bool native_pointer,
                                             std::optional<GPUVAddr> resolved_pointer,
                                             u32 diagnostic_minimum,
                                             StorageRangeDiagnostic* diagnostic) const {
    const GPUVAddr gpu_addr = resolved_pointer ? *resolved_pointer : gpu_memory->Read<u64>(ssbo_addr);

    if (gpu_addr == 0) {
        return NULL_BINDING;
    }

    if (diagnostic_minimum && native_pointer && !is_written && diagnostic) {
        if (const auto candidate =
                DiagnosticStorageBufferBinding(gpu_addr, diagnostic_minimum, *diagnostic)) {
            return *candidate;
        }
        // A partial extent is not a successful diagnostic. Retain the ordinary binding.
    }

    // Reads the shader reported past an earlier binding of this slot (see StorageFeedback),
    // measured from the pointer, rounded up to a power of two and capped at 512 MiB. Ends at or
    // past 2 GiB come from constant buffer words that are not pointers and are ignored.
    const auto feedback_size = [&]() -> size_t {
        const u32 alignment = runtime.GetStorageBufferAlignment();
        const u64 prefix = gpu_addr - Common::AlignDown(gpu_addr, alignment);
        if (storage_feedback_end <= prefix || storage_feedback_end >= 0x8000'0000U) {
            return 0;
        }
        return (std::min<size_t>)(std::bit_ceil(static_cast<size_t>(storage_feedback_end - prefix)),
                                  512_MiB);
    };
    const auto size = [&]() {
        if (native_pointer) {
            // Native constant buffers carry bare pointers followed by unrelated parameters.
            // EDEN_NX2_NATIVE_SSBO_MB: the most a bare pointer may reach before feedback
            // (default 8 MiB).
            static const size_t native_limit = [] {
                const char* value = std::getenv("EDEN_NX2_NATIVE_SSBO_MB");
                const long megabytes = value ? std::strtol(value, nullptr, 10) : 0;
                return megabytes > 0 && megabytes <= 2048 ? static_cast<size_t>(megabytes) * 1_MiB
                                                          : static_cast<size_t>(8_MiB);
            }();
            const size_t wanted = feedback_size();
            const size_t layout_size = gpu_memory->GetMemoryLayoutSize(gpu_addr);
            const size_t mapped_size = (std::min)(layout_size, (std::max)(native_limit, wanted));
            if (wanted > native_limit) [[unlikely]] {
                LogStorageGrowth(gpu_addr, native_limit, mapped_size);
            }
            // A written binding is one device range, even across adjacent GPU mappings.
            return static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr, mapped_size));
        }
        const u32 memory_layout_size = static_cast<u32>(gpu_memory->GetMemoryLayoutSize(gpu_addr));
        const u64 next_qword = gpu_memory->Read<u64>(ssbo_addr + 8);
        const u32 packed_size = static_cast<u32>(next_qword);
        const bool next_qword_is_size = static_cast<u32>(next_qword >> 32) == 0 &&
                                        packed_size != 0 &&
                                        packed_size <= memory_layout_size;
        if (next_qword_is_size) {
            return packed_size;
        }
        const size_t wanted = feedback_size();
        if (wanted > 8_MiB) [[unlikely]] {
            LogStorageGrowth(gpu_addr, 8_MiB, (std::min<size_t>)(memory_layout_size, wanted));
        }
        return (std::min)(memory_layout_size,
                          static_cast<u32>((std::max<size_t>)(8_MiB, wanted)));
    }();

    // Alignment only applies to the offset of the buffer
    const u32 alignment = runtime.GetStorageBufferAlignment();
    const GPUVAddr aligned_gpu_addr = Common::AlignDown(gpu_addr, alignment);
    const u32 aligned_size = static_cast<u32>(gpu_addr - aligned_gpu_addr) + size;

    const std::optional<DAddr> aligned_device_addr = gpu_memory->GpuToCpuAddress(aligned_gpu_addr);
    if (!aligned_device_addr || size == 0) {
        LOG_DEBUG(HW_GPU, "Failed to find storage buffer for cbuf index {}", cbuf_index);
        return NULL_BINDING;
    }
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    ASSERT_MSG(device_addr, "Unaligned storage buffer address not found for cbuf index {}",
               cbuf_index);
    // The end address used for size calculation does not need to be aligned
    const DAddr cpu_end = Common::AlignUp(*device_addr + size, Core::DEVICE_PAGESIZE);

    u32 binding_size = static_cast<u32>(cpu_end - *aligned_device_addr);
    if (is_written) {
        binding_size = aligned_size;
    }
    const Binding binding{
        .device_addr = *aligned_device_addr,
        .gpu_addr = aligned_gpu_addr,
        .size = binding_size,
        .buffer_id = BufferId{},
    };
    return binding;
}

template <class P>
TextureBufferBinding BufferCache<P>::GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                             PixelFormat format) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    TextureBufferBinding binding;
    if (!device_addr || size == 0) {
        binding.device_addr = 0;
        binding.size = 0;
        binding.buffer_id = NULL_BUFFER_ID;
        binding.format = PixelFormat::Invalid;
    } else {
        binding.device_addr = *device_addr;
        binding.size = size;
        binding.buffer_id = BufferId{};
        binding.format = format;
    }
    return binding;
}

template <class P>
std::span<const u8> BufferCache<P>::ImmediateBufferWithData(DAddr device_addr, size_t size) {
    u8* const base_pointer = device_memory.GetPointer<u8>(device_addr);
    if (IsRangeGranular(device_addr, size) ||
        base_pointer + size == device_memory.GetPointer<u8>(device_addr + size)) {
        return std::span(base_pointer, size);
    } else {
        const std::span<u8> span = ImmediateBuffer(size);
        device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        return span;
    }
}

template <class P>
std::span<u8> BufferCache<P>::ImmediateBuffer(size_t wanted_capacity) {
    immediate_buffer_alloc.resize_destructive(wanted_capacity);
    return std::span<u8>(immediate_buffer_alloc.data(), wanted_capacity);
}

template <class P>
bool BufferCache<P>::HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept {
    return ((channel_state->fast_bound_uniform_buffers[stage] >> binding_index) & 1u) != 0;
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectCount() {
    auto& buffer = slot_buffers[channel_state->count_buffer_binding.buffer_id];
    return std::make_pair(&buffer, buffer.Offset(channel_state->count_buffer_binding.device_addr));
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectBuffer() {
    auto& buffer = slot_buffers[channel_state->indirect_buffer_binding.buffer_id];
    return std::make_pair(&buffer,
                          buffer.Offset(channel_state->indirect_buffer_binding.device_addr));
}

} // namespace VideoCommon
