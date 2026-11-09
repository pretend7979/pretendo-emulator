// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <string_view>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/vk_query_cache.h"

#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/renderer_vulkan/vk_command_pool.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

struct Scheduler::GpuTimestampState {
    static constexpr u32 SlotCount = 128;
    static constexpr u32 QueriesPerSlot = 4;
    static constexpr u32 HarvestLimit = 8;
    static constexpr u64 ReportIntervalNs = 5'000'000'000;

    enum class Phase : u8 { Free, Recording, Submitted };

    struct Slot {
        Phase phase{};
        u64 tick{};
        u64 started_ns{};
        bool external_wait{};
    };

    struct Result {
        u64 timestamp{};
        u64 available{};
    };

    struct Distribution {
        u64 count{};
        long double total_ns{};
        u64 max_ns{};
        std::array<u64, 65> histogram{};

        void Add(u64 ns) {
            ++count;
            total_ns += ns;
            max_ns = std::max(max_ns, ns);
            ++histogram[static_cast<std::size_t>(std::bit_width(ns))];
        }

        double MeanMs() const {
            return count ? static_cast<double>(total_ns / count / 1'000'000.0L) : 0.0;
        }

        double P95UpperMs() const {
            const u64 target = count - count / 20;
            u64 cumulative{};
            for (u32 index = 0; index < histogram.size(); ++index) {
                cumulative += histogram[index];
                if (cumulative >= target) {
                    const u64 upper = index == 64 ? std::numeric_limits<u64>::max()
                                                  : (u64{1} << index) - 1;
                    return static_cast<double>(upper) / 1'000'000.0;
                }
            }
            return 0.0;
        }
    };

    static u64 NowNs() {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
    }

    static std::unique_ptr<GpuTimestampState> Create(const Device& device) {
        const char* const value = std::getenv("EDEN_NX2_GPU_TIMESTAMPS");
        if (!value || std::string_view{value} != "1") {
            return {};
        }
        const auto properties = device.GetPhysical().GetProperties();
        const auto families = device.GetPhysical().GetQueueFamilyProperties();
        const auto family = device.GetGraphicsFamily();
        if (family >= families.size() || !properties.limits.timestampComputeAndGraphics ||
            families[family].timestampValidBits == 0 ||
            families[family].timestampValidBits > 64 ||
            !std::isfinite(properties.limits.timestampPeriod) ||
            properties.limits.timestampPeriod <= 0.0f ||
            !device.GetDispatchLoader().vkCmdWriteTimestamp) {
            LOG_WARNING(Render_Vulkan, "GPU timestamp diagnostic unavailable on graphics queue");
            return {};
        }
        try {
            auto result = std::make_unique<GpuTimestampState>(
                device, families[family].timestampValidBits, properties.limits.timestampPeriod);
            LOG_INFO(Render_Vulkan,
                     "GPU timestamp diagnostic enabled: {} slots, {} bits, {} ns/tick; "
                     "intervals include dependencies and overlap, not GPU busy utilization",
                     SlotCount, result->valid_bits, result->period_ns);
            return result;
        } catch (const vk::Exception& exception) {
            if (exception.GetResult() == VK_ERROR_DEVICE_LOST) {
                throw;
            }
            LOG_WARNING(Render_Vulkan, "GPU timestamp pool creation failed: {}", exception.what());
            return {};
        }
    }

    GpuTimestampState(const Device& device_, u32 valid_bits_, double period_ns_)
        : device{device_}, valid_bits{valid_bits_}, period_ns{period_ns_},
          pool{device.GetLogical().CreateQueryPool({
              .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
              .pNext = nullptr,
              .flags = 0,
              .queryType = VK_QUERY_TYPE_TIMESTAMP,
              .queryCount = SlotCount * QueriesPerSlot,
              .pipelineStatistics = 0,
          })}, report_ns{NowNs()} {}

    ~GpuTimestampState() {
        u32 uncollected{};
        u32 unsubmitted{};
        for (const auto& slot : slots) {
            uncollected += slot.phase == Phase::Submitted;
            unsubmitted += slot.phase == Phase::Recording;
        }
        Report(NowNs(), true);
        LOG_INFO(Render_Vulkan,
                 "GPU timestamp diagnostic ended: {} submitted samples uncollected, "
                 "{} recording samples unsubmitted; no final diagnostic wait",
                 uncollected, unsubmitted);
    }

    bool Duration(u64 begin, u64 end, u64 age_ns, u64& duration_ns) const {
        const u64 mask = valid_bits == 64 ? std::numeric_limits<u64>::max()
                                          : (u64{1} << valid_bits) - 1;
        const long double wrap_ns = std::ldexp(static_cast<long double>(period_ns),
                                               static_cast<int>(valid_bits));
        if (static_cast<long double>(age_ns) >= wrap_ns) {
            return false;
        }
        const u64 ticks = (end - begin) & mask;
        const long double ns = static_cast<long double>(ticks) * period_ns;
        if (!std::isfinite(ns) || ns > static_cast<long double>(age_ns) ||
            ns >= static_cast<long double>(std::numeric_limits<u64>::max())) {
            return false;
        }
        duration_ns = static_cast<u64>(ns);
        return true;
    }

    void Harvest(u64 known_tick, u64 now_ns) {
        u32 reads{};
        for (u32 inspected = 0; inspected < SlotCount && reads < HarvestLimit; ++inspected) {
            const u32 index = harvest_cursor;
            harvest_cursor = (harvest_cursor + 1) % SlotCount;
            auto& slot = slots[index];
            if (slot.phase != Phase::Submitted || slot.tick > known_tick) {
                continue;
            }
            ++reads;
            std::array<Result, QueriesPerSlot> values{};
            const VkResult result = device.GetLogical().GetQueryResults(
                *pool, index * QueriesPerSlot, QueriesPerSlot, sizeof(values), values.data(),
                sizeof(Result), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            if (result == VK_NOT_READY ||
                (result == VK_SUCCESS && std::ranges::any_of(
                    values, [](const Result& value) { return value.available == 0; }))) {
                ++not_ready;
                continue;
            }
            if (result != VK_SUCCESS) {
                ++query_errors;
                enabled = false;
                if (result == VK_ERROR_DEVICE_LOST) {
                    device.ReportLoss();
                    vk::Check(result);
                }
                LOG_WARNING(Render_Vulkan, "GPU timestamp result read failed: {}; sampling disabled",
                            static_cast<int>(result));
                return;
            }
            const u64 age_ns = now_ns >= slot.started_ns ? now_ns - slot.started_ns : 0;
            u64 upload_ns{};
            u64 main_ns{};
            if (!Duration(values[0].timestamp, values[1].timestamp, age_ns, upload_ns) ||
                !Duration(values[2].timestamp, values[3].timestamp, age_ns, main_ns)) {
                ++invalid_results;
            } else {
                upload.Add(upload_ns);
                main.Add(main_ns);
                sampled_external_waits += slot.external_wait;
                if (main_ns >= longest_main_ns) {
                    longest_main_ns = main_ns;
                    longest_tick = slot.tick;
                    longest_external_wait = slot.external_wait;
                }
            }
            slot = {};
        }
    }

    void Begin(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf, u64 known_tick) {
        current = SlotCount;
        if (!enabled) {
            return;
        }
        const u64 now_ns = NowNs();
        Harvest(known_tick, now_ns);
        if (!enabled) {
            return;
        }
        for (u32 checked = 0; checked < SlotCount; ++checked) {
            const u32 index = reserve_cursor;
            reserve_cursor = (reserve_cursor + 1) % SlotCount;
            if (slots[index].phase != Phase::Free) {
                continue;
            }
            current = index;
            slots[index] = {.phase = Phase::Recording, .tick = 0, .started_ns = now_ns,
                            .external_wait = false};
            const u32 first = index * QueriesPerSlot;
            upload_cmdbuf.ResetQueryPool(*pool, first, 2);
            upload_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *pool, first);
            cmdbuf.ResetQueryPool(*pool, first + 2, 2);
            cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *pool, first + 2);
            return;
        }
        ++ring_full;
    }

    void End(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf, u64 tick,
             bool external_wait) {
        if (current == SlotCount) {
            return;
        }
        auto& slot = slots[current];
        slot.tick = tick;
        slot.external_wait = external_wait;
        const u32 first = current * QueriesPerSlot;
        upload_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pool, first + 1);
        cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pool, first + 3);
    }

    void Submitted(u64 tick, VkResult result, u64 ready_ns, u64 enter_ns, u64 locked_ns,
                   u64 call_ns) {
        const u64 now_ns = NowNs();
        ++submissions;
        if (enter_ns >= ready_ns && locked_ns >= enter_ns && now_ns >= call_ns) {
            worker_delay.Add(enter_ns - ready_ns);
            submit_lock.Add(locked_ns - enter_ns);
            submit_api.Add(now_ns - call_ns);
        }
        if (current != SlotCount) {
            auto& slot = slots[current];
            if (result == VK_SUCCESS && slot.phase == Phase::Recording && slot.tick == tick) {
                slot.phase = Phase::Submitted;
            } else {
                ++submit_errors;
                enabled = false;
            }
            current = SlotCount;
        }
        Report(now_ns, false);
    }

    void Report(u64 now_ns, bool final) {
        if (!final && now_ns - report_ns < ReportIntervalNs) {
            return;
        }
        LOG_INFO(Render_Vulkan,
                 "GPU timestamp intervals: window {:.3f} s, submissions {}, samples {}, "
                 "ring drops {}, not ready {}, invalid {}, query errors {}, submit errors {}; "
                 "upload mean/p95-upper/max {:.3f}/{:.3f}/{:.3f} ms, "
                 "main mean/p95-upper/max {:.3f}/{:.3f}/{:.3f} ms; "
                 "host worker/submit-lock/queue-call mean {:.3f}/{:.3f}/{:.3f} ms; "
                 "longest main tick {} external_wait {}, sampled external waits {}; "
                 "GPU intervals overlap and include dependencies",
                 static_cast<double>(now_ns - report_ns) / 1'000'000'000.0, submissions,
                 main.count, ring_full, not_ready, invalid_results, query_errors, submit_errors,
                 upload.MeanMs(), upload.P95UpperMs(), static_cast<double>(upload.max_ns) / 1'000'000.0,
                 main.MeanMs(), main.P95UpperMs(), static_cast<double>(main.max_ns) / 1'000'000.0,
                 worker_delay.MeanMs(), submit_lock.MeanMs(), submit_api.MeanMs(), longest_tick,
                 longest_external_wait, sampled_external_waits);
        report_ns = now_ns;
        submissions = ring_full = not_ready = invalid_results = query_errors = submit_errors = 0;
        longest_main_ns = longest_tick = sampled_external_waits = 0;
        longest_external_wait = false;
        upload = {};
        main = {};
        worker_delay = {};
        submit_lock = {};
        submit_api = {};
    }

    const Device& device;
    const u32 valid_bits;
    const double period_ns;
    vk::QueryPool pool;
    std::array<Slot, SlotCount> slots{};
    u32 current{SlotCount};
    u32 reserve_cursor{};
    u32 harvest_cursor{};
    bool enabled{true};
    u64 report_ns{};
    u64 submissions{};
    u64 ring_full{};
    u64 not_ready{};
    u64 invalid_results{};
    u64 query_errors{};
    u64 submit_errors{};
    u64 longest_main_ns{};
    u64 longest_tick{};
    u64 sampled_external_waits{};
    bool longest_external_wait{};
    Distribution upload;
    Distribution main;
    Distribution worker_delay;
    Distribution submit_lock;
    Distribution submit_api;
};


void Scheduler::CommandChunk::ExecuteAll(vk::CommandBuffer cmdbuf,
                                         vk::CommandBuffer upload_cmdbuf) {
    auto command = first;
    while (command != nullptr) {
        auto next = command->GetNext();
        command->Execute(cmdbuf, upload_cmdbuf);
        command->~Command();
        command = next;
    }
    submit = false;
    command_offset = 0;
    first = nullptr;
    last = nullptr;
}

Scheduler::Scheduler(const Device& device_, StateTracker& state_tracker_)
    : device{device_}, state_tracker{state_tracker_},
      master_semaphore{std::make_unique<MasterSemaphore>(device)},
      command_pool{std::make_unique<CommandPool>(*master_semaphore, device)} {

    const auto enabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && std::string_view{value} == "1";
    };
    fence_flush_dedup = enabled("EDEN_NX2_FENCE_FLUSH_DEDUP");
    fence_flush_dedup_log = enabled("EDEN_NX2_FENCE_FLUSH_DEDUP_LOG");
    gpu_timestamps = GpuTimestampState::Create(device);
    AcquireNewChunk();
    AllocateWorkerCommandBuffer();
    worker_thread = std::jthread([this](std::stop_token token) { WorkerThread(token); });
}

Scheduler::~Scheduler() = default;

u64 Scheduler::Flush(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Flush");
    }
    // When flushing, we only send data to the worker thread; no waiting is necessary.
    const u64 signal_value = SubmitExecution(signal_semaphore, wait_semaphore);
    AllocateNewContext();
    last_flush = std::chrono::steady_clock::now();
    flush_pending = false;
    return signal_value;
}

void Scheduler::NoteFenceSubmission() {
    if (fence_flush_dedup) {
        paired_fence_flush_tick = CurrentTick();
    }
}

bool Scheduler::TrySkipFenceFlush(bool eligible) {
    const u64 paired_tick = std::exchange(paired_fence_flush_tick, 0);
    if (!fence_flush_dedup || !eligible) {
        return false;
    }
    ++fence_flush_stats.attempts;
    bool skip = false;
    if (paired_tick == 0) {
        ++fence_flush_stats.unpaired;
    } else if (CurrentTick() != paired_tick) {
        ++fence_flush_stats.changed_tick;
    } else if (on_submit || flush_pending) {
        ++fence_flush_stats.required_submit;
    } else if (pending_commands) {
        ++fence_flush_stats.recorded_work;
    } else {
        // Closing queries or realizing a deferred clear can record more work. Check again
        // after exactly the same finalization that a real submission would perform.
        EndPendingOperations();
        if (pending_commands || deferred_clear.framebuffer != nullptr || state.renderpass ||
            CurrentTick() != paired_tick || on_submit || flush_pending) {
            ++fence_flush_stats.finalized_work;
        } else {
            InvalidateState();
            ++fence_flush_stats.skipped;
            skip = true;
        }
    }
    if (fence_flush_dedup_log && fence_flush_stats.attempts % 600 == 0) {
        LOG_INFO(Render_Vulkan,
                 "Fence flush dedup: attempts={} skipped={} unpaired={} changed_tick={} "
                 "required_submit={} recorded_work={} finalized_work={}",
                 fence_flush_stats.attempts, fence_flush_stats.skipped,
                 fence_flush_stats.unpaired, fence_flush_stats.changed_tick,
                 fence_flush_stats.required_submit, fence_flush_stats.recorded_work,
                 fence_flush_stats.finalized_work);
    }
    return skip;
}

bool Scheduler::DeferFlush() {
    // Fences come one per compute release: ~11,500 a second in Hyrule Warriors: Age of
    // Imprisonment fights, each a queue submission (~22,000 a second with the others), which
    // kept the Vulkan worker busy submitting. Batched within 500 us: ~10-15% more fps there.
    // Command list copies after dispatches were ~9,000 more a second, of which the front waited
    // for about 5%.
    static const s64 interval_us = [] {
        const char* value = std::getenv("EDEN_NX2_FENCE_FLUSH_US");
        return value == nullptr ? s64{500} : static_cast<s64>(std::atoll(value));
    }();
    if (interval_us <= 0 || !VideoCommon::GpuSplit::OnBackThread() ||
        std::chrono::steady_clock::now() - last_flush >= std::chrono::microseconds{interval_us}) {
        return false;
    }
    flush_pending = true;
    return true;
}

void Scheduler::FlushDeferred() {
    if (flush_pending) {
        Flush();
    }
}

bool Scheduler::WaitSubmitted(u64 tick) {
    if (IsSubmitted(tick)) {
        return true;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    split.RequestIdleHook();
    // No time limit: the back can spend seconds on one entry (a large shader's translation),
    // and a fence reported as submitted before it is would release its downloads too early.
    const auto start = std::chrono::steady_clock::now();
    auto next_log = start + std::chrono::seconds{2};
    for (u32 spins = 0; !IsSubmitted(tick); ++spins) {
        if (!split.BackRunning()) {
            return IsSubmitted(tick);
        }
        if (const auto now = std::chrono::steady_clock::now(); now >= next_log) {
            next_log = now + std::chrono::seconds{2};
            LOG_WARNING(Render_Vulkan, "Waited {} s for the GPU thread to submit tick {}",
                        std::chrono::duration_cast<std::chrono::seconds>(now - start).count(),
                        tick);
            split.RequestIdleHook();
        }
        if (spins < 64) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds{50});
        }
    }
    return true;
}

void Scheduler::Finish(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Finish");
    }
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    VideoCommon::CommandTrace::Get().Mark("finish.begin", presubmit_tick,
                                        master_semaphore->KnownGpuTick());
    SubmitExecution(signal_semaphore, wait_semaphore);
    Wait(presubmit_tick);
    VideoCommon::CommandTrace::Get().Mark("finish.end", presubmit_tick,
                                        master_semaphore->KnownGpuTick());
    AllocateNewContext();
}

void Scheduler::WaitWorker() {
    DispatchWork();
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::WorkerWait};

    // Ensure the queue is drained.
    {
        std::unique_lock ql{queue_mutex};
        event_cv.wait(ql, [this] { return work_queue.empty(); });
    }

    // Now wait for execution to finish.
    std::scoped_lock el{execution_mutex};
}

void Scheduler::DispatchWork() {
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::DispatchWork");
    }
    if (chunk && !chunk->Empty()) {
        // Waking the worker is a system call (~2% of the GPU thread at one dispatch per 8
        // draws): skip it while the worker is still busy, it pops the queue before waiting.
        bool wake;
        {
            std::scoped_lock ql{queue_mutex};
            work_queue.push(std::move(chunk));
            wake = worker_waiting;
        }
        if (wake) {
            event_cv.notify_all();
        }
        AcquireNewChunk();
    }
}

void Scheduler::BeginRenderPassImpl(const Framebuffer* framebuffer, VkRenderPass renderpass,
                                    const VkClearValue* clear_values, u32 clear_value_count) {
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    state.renderpass = renderpass;
    state.framebuffer = framebuffer_handle;
    state.render_area = render_area;

    if (GPU::Logging::IsActive() && Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string render_pass_info =
            fmt::format("renderArea={}x{}, numImages={}", render_area.width, render_area.height,
                        framebuffer->NumImages());
        GPU::Logging::GPULogger::GetInstance().LogRenderPassBegin(render_pass_info);
    }

    std::array<VkClearValue, 9> values{};
    for (u32 i = 0; i < clear_value_count && i < values.size(); ++i) {
        values[i] = clear_values[i];
    }
    Record([renderpass, framebuffer_handle, render_area, values, clear_value_count](
               vk::CommandBuffer cmdbuf) {
        const VkRenderPassBeginInfo renderpass_bi{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = renderpass,
            .framebuffer = framebuffer_handle,
            .renderArea =
                {
                    .offset = {.x = 0, .y = 0},
                    .extent = render_area,
                },
            .clearValueCount = clear_value_count,
            .pClearValues = clear_value_count != 0 ? values.data() : nullptr,
        };
        cmdbuf.BeginRenderPass(renderpass_bi, VK_SUBPASS_CONTENTS_INLINE);
    });
    num_renderpass_images = framebuffer->NumImages();
    renderpass_images = framebuffer->Images();
    renderpass_image_ranges = framebuffer->ImageRanges();
    framebuffer->MarkResolveShadowsUpToDate();
}

void Scheduler::RealizeDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    const DeferredClear dc = deferred_clear;
    deferred_clear = {};

    std::array<VkClearValue, 9> clear_values{};
    u32 count = 0;
    const RenderPassKey& base = dc.framebuffer->RenderPassKeyBase();
    for (u32 slot = 0; slot < 8; ++slot) {
        if (base.color_formats[slot] == VideoCore::Surface::PixelFormat::Invalid) {
            continue;
        }
        clear_values[count++] = dc.color_values[slot];
    }
    if (base.depth_format != VideoCore::Surface::PixelFormat::Invalid) {
        clear_values[count++] = dc.depth_stencil_value;
    }
    const u32 color_discard_mask =
        dc.framebuffer->DiscardsMsaaColor() ? dc.color_clear_mask : 0u;
    const bool depth_stencil_discard =
        dc.depth_stencil && dc.framebuffer->DiscardsMsaaDepthStencil();
    const VkRenderPass renderpass = dc.framebuffer->RenderPassVariant(
        dc.color_clear_mask, dc.depth_stencil, color_discard_mask, depth_stencil_discard);
    EndRenderPass();
    BeginRenderPassImpl(dc.framebuffer, renderpass, clear_values.data(), count);
}

bool Scheduler::DeferColorClear(const Framebuffer* framebuffer, u32 rt_slot,
                                const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.color_clear_mask |= 1u << rt_slot;
    deferred_clear.color_values[rt_slot] = value;
    if (fence_flush_dedup) {
        pending_commands = true;
    }
    return true;
}

bool Scheduler::DeferDepthStencilClear(const Framebuffer* framebuffer, const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.depth_stencil = true;
    deferred_clear.depth_stencil_value = value;
    if (fence_flush_dedup) {
        pending_commands = true;
    }
    return true;
}

void Scheduler::FlushDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    RealizeDeferredClear();
    EndRenderPass();
}

void Scheduler::RequestRenderpass(const Framebuffer* framebuffer, bool may_store) {
    if (fragment_barrier_pending) {
        EndRenderPass();
    }
    RequestRenderpassImpl(framebuffer);
    render_pass_has_stores |= may_store;
}

void Scheduler::RequestRenderpassImpl(const Framebuffer* framebuffer) {
    if (deferred_clear.framebuffer == framebuffer) {
        RealizeDeferredClear();
        return;
    }
    const VkRenderPass renderpass = framebuffer->RenderPass();
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    // A draw continues the active pass whenever the same framebuffer and area are bound. The
    // render-pass *variant* (a deferred clear's loadOp=CLEAR form) is only meaningful at begin -
    // loadOps are applied once, and the base/variant passes are render-pass-compatible (only
    // loadOp/storeOp differ). Comparing the exact handle here made every draw after a deferred
    // clear restart the pass and re-LOAD what was just CLEARed (~118 MB of attachment traffic per
    // world-target turn). The framebuffer handle already fixes the attachment structure and key,
    // so it is the correct identity to compare.
    if (state.renderpass && framebuffer_handle == state.framebuffer &&
        render_area.width == state.render_area.width &&
        render_area.height == state.render_area.height) {
        return;
    }
    // Ends any active pass and realizes a deferred clear
    EndRenderPass();
    BeginRenderPassImpl(framebuffer, renderpass, nullptr, 0);
}

void Scheduler::RequestOutsideRenderPassOperationContext() {
    EndRenderPass();
}

bool Scheduler::UpdateGraphicsPipeline(GraphicsPipeline* pipeline) {
    if (state.graphics_pipeline == pipeline) {
        if (pipeline && pipeline->UsesExtendedDynamicState() &&
            state.needs_state_enable_refresh) {
            state_tracker.InvalidateStateEnableFlag();
            state.needs_state_enable_refresh = false;
        }
        return false;
    }

    state.graphics_pipeline = pipeline;

    if (!pipeline) {
        return true;
    }

    if (!pipeline->UsesExtendedDynamicState()) {
        state.needs_state_enable_refresh = true;
    } else if (state.needs_state_enable_refresh) {
        state_tracker.InvalidateStateEnableFlag();
        state.needs_state_enable_refresh = false;
    }

    return true;
}

bool Scheduler::UpdateRescaling(bool is_rescaling) {
    if (state.rescaling_defined && is_rescaling == state.is_rescaling) {
        return false;
    }
    state.rescaling_defined = true;
    state.is_rescaling = is_rescaling;
    return true;
}

bool Scheduler::UpdateDescriptorBufferChunk(u32 descriptor_chunk) {
    if (state.descriptor_buffer_bound && descriptor_chunk == state.descriptor_buffer_chunk) {
        return false;
    }
    state.descriptor_buffer_bound = true;
    state.descriptor_buffer_chunk = descriptor_chunk;
    return true;
}

void Scheduler::WorkerThread(std::stop_token stop_token) {
    Common::SetCurrentThreadName("VulkanWorker");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();

    const auto TryPopQueue{[this](auto& work) -> bool {
        if (work_queue.empty()) {
            return false;
        }

        work = std::move(work_queue.front());
        work_queue.pop();
        event_cv.notify_all();
        return true;
    }};

    while (!stop_token.stop_requested()) {
        std::unique_ptr<CommandChunk> work;

        {
            std::unique_lock lk{queue_mutex};

            // Wait for work.
            event_cv.wait(lk, stop_token, [&] {
                worker_waiting = !TryPopQueue(work);
                return !worker_waiting;
            });
            worker_waiting = false;

            // If we've been asked to stop, we're done.
            if (stop_token.stop_requested()) {
                return;
            }

            // Exchange lock ownership so that we take the execution lock before
            // the queue lock goes out of scope. This allows us to force execution
            // to complete in the next step.
            void(std::exchange(lk, std::unique_lock{execution_mutex}));

            // Perform the work, tracking whether the chunk was a submission
            // before executing.
            const bool has_submit = work->HasSubmit();
            work->ExecuteAll(current_cmdbuf, current_upload_cmdbuf);

            // If the chunk was a submission, reallocate the command buffer.
            if (has_submit) {
                AllocateWorkerCommandBuffer();
            }
        }

        {
            std::scoped_lock rl{reserve_mutex};

            // Recycle the chunk back to the reserve.
            chunk_reserve.emplace_back(std::move(work));
        }
    }
}

void Scheduler::AllocateWorkerCommandBuffer() {
    const VideoCommon::CommandTraceScope trace{"allocate_context", 0, CurrentTick()};
    current_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    current_upload_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_upload_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    if (gpu_timestamps) {
        gpu_timestamps->Begin(current_cmdbuf, current_upload_cmdbuf,
                              master_semaphore->KnownGpuTick());
    }
}

u64 Scheduler::SubmitExecution(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    paired_fence_flush_tick = 0;
    VideoCommon::CommandTrace::Get().Mark("submit.record_begin", CurrentTick(),
                                        master_semaphore->KnownGpuTick());
    EndPendingOperations();
    InvalidateState();

    const u64 signal_value = master_semaphore->NextTick();
    const u64 timestamp_ready = gpu_timestamps ? GpuTimestampState::NowNs() : 0;
    RecordWithUploadBuffer([signal_semaphore, wait_semaphore, signal_value, timestamp_ready,
                            this](vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) {
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        upload_cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WRITE_BARRIER);
        if (gpu_timestamps) {
            gpu_timestamps->End(cmdbuf, upload_cmdbuf, signal_value,
                                wait_semaphore != VK_NULL_HANDLE);
        }
        upload_cmdbuf.End();
        cmdbuf.End();

        if (on_submit) {
            on_submit();
        }

        const u64 timestamp_enter = gpu_timestamps ? GpuTimestampState::NowNs() : 0;
        std::scoped_lock lock{submit_mutex};
        const u64 timestamp_locked = gpu_timestamps ? GpuTimestampState::NowNs() : 0;
        VideoCommon::CommandTrace::Get().Mark("queue_submit.begin", signal_value,
                                            master_semaphore->KnownGpuTick());
        const u64 timestamp_call = gpu_timestamps ? GpuTimestampState::NowNs() : 0;
        const VkResult result = master_semaphore->SubmitQueue(
            cmdbuf, upload_cmdbuf, signal_semaphore, wait_semaphore, signal_value);
        if (gpu_timestamps) {
            gpu_timestamps->Submitted(signal_value, result, timestamp_ready,
                                      timestamp_enter, timestamp_locked, timestamp_call);
        }
        VideoCommon::CommandTrace::Get().Mark("queue_submit.end", signal_value,
                                            master_semaphore->KnownGpuTick(),
                                            static_cast<u64>(static_cast<s64>(result)));
        switch (result) {
        case VK_SUCCESS:
            // Log successful queue submission
            if (GPU::Logging::IsActive() &&
                Settings::values.gpu_log_vulkan_calls.GetValue()) {
                GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                    "vkQueueSubmit", "", VK_SUCCESS);
            }
            break;
        case VK_ERROR_DEVICE_LOST:
            device.ReportLoss();
            [[fallthrough]];
        default:
            vk::Check(result);
            break;
        }
    });
    chunk->MarkSubmit();
    DispatchWork();
    pending_commands = false;
    VideoCommon::CommandTrace::Get().Mark("submit.record_end", signal_value,
                                        master_semaphore->KnownGpuTick());
    return signal_value;
}

void Scheduler::AllocateNewContext() {
    // Enable counters once again. These are disabled when a command buffer is finished.
}

void Scheduler::InvalidateState() {
    state.graphics_pipeline = nullptr;
    state.rescaling_defined = false;
    state.descriptor_buffer_bound = false;
    state_tracker.InvalidateCommandBufferState();
}

void Scheduler::EndPendingOperations() {
    // Pause, don't reset: a submission can land between a guest's counter reset and its report,
    // and resetting here would drop the samples counted so far (the report then reads too low).
    query_cache->CounterClose(VideoCommon::QueryType::ZPassPixelCount64);
    EndRenderPass();
}

void Scheduler::EndRenderPass()
    {
        RealizeDeferredClear();
        fragment_barrier_pending = false;
        if (!state.renderpass) {
            return;
        }
        render_pass_has_stores = false;

        query_cache->CounterClose(VideoCommon::QueryType::StreamingByteCount);

        // Log render pass end
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            GPU::Logging::GPULogger::GetInstance().LogRenderPassEnd();
        }

        query_cache->CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, false);
        query_cache->NotifySegment(false);

        Record([num_images = num_renderpass_images,
                       images = renderpass_images,
                       ranges = renderpass_image_ranges,
                       has_transform_feedback = device.IsExtTransformFeedbackSupported()](
                          vk::CommandBuffer cmdbuf) {
            std::array<VkImageMemoryBarrier, 9> barriers;
            for (size_t i = 0; i < num_images; ++i) {
                const VkImageSubresourceRange& range = ranges[i];
                const bool is_color = (range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
                const bool is_depth_stencil = (range.aspectMask
                                              & (VK_IMAGE_ASPECT_DEPTH_BIT
                                                 | VK_IMAGE_ASPECT_STENCIL_BIT)) !=0;

                VkAccessFlags src_access = 0;

                if (is_color)
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                else if (is_depth_stencil)
                    src_access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                else
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                  | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

                barriers[i] = VkImageMemoryBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = src_access,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = images[i],
                        .subresourceRange = range,
                };
            }
            cmdbuf.EndRenderPass();
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                                   0, nullptr, nullptr, vk::Span(barriers.data(), num_images));
            if (has_transform_feedback) {
                static constexpr VkMemoryBarrier XFB_OUTPUT_BARRIER{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
                    .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                       VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       0, XFB_OUTPUT_BARRIER);
            }
        });

        state.renderpass = VkRenderPass{};
        num_renderpass_images = 0;
    }


void Scheduler::AcquireNewChunk() {
    std::scoped_lock rl{reserve_mutex};

    if (chunk_reserve.empty()) {
        // If we don't have anything reserved, we need to make a new chunk.
        chunk = std::make_unique<CommandChunk>();
    } else {
        // Otherwise, we can just take from the reserve.
        chunk = std::move(chunk_reserve.back());
        chunk_reserve.pop_back();
    }
}

} // namespace Vulkan
