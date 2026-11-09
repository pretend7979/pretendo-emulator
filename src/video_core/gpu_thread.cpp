// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstdlib>
#include <string_view>
#include <thread>

#ifdef __linux__
#include <sys/prctl.h>
#elif defined(_WIN32)
#include "common/windows/timer_resolution.h"
#endif

#include "common/assert.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "common/thread_sampler.h"
#include "core/core.h"
#include "core/frontend/graphics_context.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/gpu.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_profile.h"
#include "video_core/host1x/host1x.h"
#include "video_core/renderer_base.h"

namespace VideoCommon::GPUThread {

namespace {

// Opt-in pending-channel diagnostics only; the normal polling/sleep timer is separate.
struct PendingDiagnosticTiming {
    using Clock = std::chrono::steady_clock;
    Clock::time_point next_sample{};
    Clock::time_point last_progress{};
    Clock::time_point next_report{};
    u64 last_fingerprint{};
    u32 reports{};
    bool observed{};

    void Reset() {
        observed = false;
        next_sample = {};
    }

    bool SampleDue(Clock::time_point now) {
        if (now < next_sample) return false;
        next_sample = now + std::chrono::seconds{1};
        return true;
    }

    bool Observe(u64 fingerprint, Clock::time_point now) {
        if (!observed || fingerprint != last_fingerprint) {
            observed = true;
            last_fingerprint = fingerprint;
            last_progress = now;
            next_report = now + std::chrono::seconds{5};
            return false;
        }
        if (now < next_report || reports >= 4) return false;
        next_report = now + std::chrono::seconds{30};
        ++reports;
        return true;
    }
};

// Tells a CPU thread waiting in PushCommand that its command ran.
void SignalCommand(SynchState& state, u64 fence, bool block) {
    state.signaled_fence.store(fence);
    if (block) {
        // We have to lock the write_lock to ensure that the condition_variable wait not get a
        // race between the check and the lock itself.
        std::scoped_lock lk{state.write_lock};
        state.cv.notify_all();
    }
}

} // Anonymous namespace

ThreadManager::ThreadManager(Core::System& system_)
    : system{system_}
{}

ThreadManager::~ThreadManager() = default;

void ThreadManager::StartThread(VideoCore::RendererBase& renderer, Core::Frontend::GraphicsContext& context, Tegra::Control::Scheduler& scheduler) {
    rasterizer = renderer.ReadRasterizer();
    thread = std::jthread([&](std::stop_token stop_token) {
        Common::SetCurrentThreadName("GPU");
        Common::ThreadSampler::RegisterCurrentThread("GPU");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
        Common::SetCurrentThreadToPerformanceCores();
        // EDEN_NX2_PIN_GPU=10 (Windows): logical CPU 10 for this thread, its core for nothing else.
        Common::ReserveCoresForCurrentThread("EDEN_NX2_PIN_GPU");
        system.RegisterHostThread();

        auto current_context = context.Acquire();
        CommandDataContainer next;
        // When a pending channel last moved or a command arrived (see the polling below).
        auto last_progress = std::chrono::steady_clock::now();
        namespace Profile = VideoCommon::Nx2Profile;
        const char* pending_trace = std::getenv("EDEN_NX2_PENDING_SNAPSHOT");
        const bool pending_diagnostics = pending_trace && std::string_view{pending_trace} == "1";
        PendingDiagnosticTiming diagnostic_timing;
        auto& split = VideoCommon::GpuSplit::Get();
        while (!stop_token.stop_requested()) {
            Profile::Report();
            split.MaybeStart(system);
            if (scheduler.HasPendingChannels()) {
                // Suspended channels wait on semaphores the CPU releases: poll them meanwhile.
                bool resumed = false;
                {
                    Profile::Scope scope{Profile::Zone::ResumePending};
                    resumed = scheduler.ResumePending(system.GPU());
                }
                if (resumed) {
                    last_progress = std::chrono::steady_clock::now();
                }
                if (pending_diagnostics && !scheduler.HasPendingChannels()) {
                    diagnostic_timing.Reset();
                }
                if (pending_diagnostics && scheduler.HasPendingChannels()) {
                    const auto now = std::chrono::steady_clock::now();
                    if (diagnostic_timing.SampleDue(now) &&
                        diagnostic_timing.Observe(scheduler.PendingChannelFingerprint(), now)) {
                        u64 submitted{};
                        bool submitted_available{};
                        {
                            std::unique_lock lock{state.write_lock, std::try_to_lock};
                            if (lock.owns_lock()) {
                                submitted = state.last_fence;
                                submitted_available = true;
                            }
                        }
                        LOG_WARNING(HW_GPU, "Pending snapshot after {} ms without observed channel progress: queue submitted={} available={} signaled={}",
                            std::chrono::duration_cast<std::chrono::milliseconds>(now - diagnostic_timing.last_progress).count(),
                            submitted, submitted_available, state.signaled_fence.load(std::memory_order_relaxed));
                        scheduler.LogPendingChannels();
                    }
                }
                if (!state.queue.TryPop(next)) {
                    if (!scheduler.HasPendingChannels()) {
                        continue;
                    }
                    // The CPU usually releases the semaphore soon after: poll for half a
                    // millisecond before sleeping, or every wait also waits out a host sleep
                    // (title 1 waits ~25 times a second).
                    Profile::Scope scope{Profile::Zone::PendingSleep};
                    const auto waited = std::chrono::steady_clock::now() - last_progress;
                    if (waited < std::chrono::microseconds{500}) {
                        std::this_thread::yield();
                    } else if (waited < std::chrono::seconds{2}) {
#ifdef _WIN32
                        // sleep_for(50us) rounds up to Sleep(1) (>= 1 ms) on Windows;
                        // SleepForOneTick uses NtDelayExecution(-1) for one high-res timer tick.
                        Common::Windows::SleepForOneTick();
#else
                        std::this_thread::sleep_for(std::chrono::microseconds{50});
#endif
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds{1});
                    }
                    continue;
                }
            } else {
                if (pending_diagnostics) diagnostic_timing.Reset();
                Profile::Scope scope{Profile::Zone::Idle};
                state.queue.PopWait(next, stop_token);
            }
            if (stop_token.stop_requested()) {
                break;
            }
            if (!std::holds_alternative<SubmitListCommand>(next.data) &&
                scheduler.HasDeferredAcquire()) {
                // Keep mapping and other queued work behind the same retirement boundary.
                // Hold this item in place; only earlier channel submissions may overlap.
                const auto wait_start = std::chrono::steady_clock::now();
                for (;;) {
                    if (stop_token.stop_requested() || !system.IsPoweredOn()) {
                        return;
                    }
                    {
                        Profile::Scope scope{Profile::Zone::ResumePending};
                        scheduler.ResumePending(system.GPU());
                    }
                    if (!scheduler.HasDeferredAcquire()) {
                        break;
                    }
                    Profile::Scope scope{Profile::Zone::PendingSleep};
                    if (std::chrono::steady_clock::now() - wait_start <
                        std::chrono::microseconds{500}) {
                        std::this_thread::yield();
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds{1});
                    }
                }
            }
            if (stop_token.stop_requested()) {
                break;
            }
            if (split.Active()) {
                // Only command lists are parsed here; everything else runs on the back thread
                // in order, and so does the signal that the command ran.
                auto* const rasterizer_ptr = renderer.ReadRasterizer();
                if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                    Profile::Scope scope{Profile::Zone::Push};
                    scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
                } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                    split.Defer([gpu = &system.GPU()] { gpu->TickWork(); });
                } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                    if (run->before_later_reads) {
                        split.Drain();
                        run->func();
                    } else {
                        split.DeferFunction(std::move(run->func));
                    }
                } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                    split.Defer([rasterizer_ptr, addr = flush->addr, size = flush->size] {
                        VideoCommon::Nx2Profile::WaitTag wait_tag{VideoCommon::Nx2Profile::WaitSource::GuestCpuRead};
                        rasterizer_ptr->FlushRegion(addr, size);
                    });
                } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                    split.Defer([rasterizer_ptr, addr = invalidate->addr, size = invalidate->size] {
                        rasterizer_ptr->OnCacheInvalidation(addr, size);
                    });
                } else {
                    ASSERT(false);
                }
                split.Defer([state_ptr = &state, fence = next.fence, block = next.block] {
                    SignalCommand(*state_ptr, fence, block);
                });
                split.Publish();
                last_progress = std::chrono::steady_clock::now();
                continue;
            }
            if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::Push};
                scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
            } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                Profile::Scope scope{Profile::Zone::Tick};
                system.GPU().TickWork();
            } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::Run};
                run->func();
            } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                Profile::WaitTag wait_tag{Profile::WaitSource::GuestCpuRead};
                renderer.ReadRasterizer()->FlushRegion(flush->addr, flush->size);
            } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->OnCacheInvalidation(invalidate->addr, invalidate->size);
            } else {
                ASSERT(false);
            }
            SignalCommand(state, next.fence, next.block);
            last_progress = std::chrono::steady_clock::now();
        }
    });
}

void ThreadManager::SubmitList(s32 channel, Tegra::CommandList&& entries, bool is_async) {
    PushCommand(SubmitListCommand(channel, std::move(entries)), false, is_async);
}

void ThreadManager::FlushRegion(DAddr addr, u64 size, bool is_async) {
    if (!is_async) {
        // Always flush with synchronous GPU mode
        PushCommand(FlushRegionCommand(addr, size), false, is_async);
    }
}

void ThreadManager::TickGPU(bool is_async) {
    PushCommand(GPUTickCommand(), false, is_async);
}

void ThreadManager::Run(std::function<void()>&& func, bool is_async, bool before_later_reads) {
    PushCommand(RunCommand{std::move(func), before_later_reads}, false, is_async);
}

void ThreadManager::InvalidateRegion(DAddr addr, u64 size) {
    rasterizer->OnCacheInvalidation(addr, size);
}

void ThreadManager::FlushAndInvalidateRegion(DAddr addr, u64 size, bool is_async) {
    if (Settings::IsGPULevelHigh()) {
        if (!is_async) {
            PushCommand(FlushRegionCommand(addr, size), false, is_async);
        } else {
            auto& gpu = system.GPU();
            const u64 fence = gpu.RequestFlush(addr, size);
            TickGPU(is_async);
            gpu.WaitForSyncOperation(fence);
        }
    }
    rasterizer->OnCacheInvalidation(addr, size);
}

u64 ThreadManager::PushCommand(CommandData&& command_data, bool block, bool is_async) {
    if (!is_async) {
        // In synchronous GPU mode, block the caller until the command has executed
        block = true;
    }

    std::unique_lock lk(state.write_lock);
    const u64 fence{++state.last_fence};
    state.queue.EmplaceWait(std::move(command_data), fence, block);

    if (block) {
        state.cv.wait(lk, thread.get_stop_token(), [this, fence] {
            return fence <= state.signaled_fence.load(std::memory_order_relaxed);
        });
    }

    return fence;
}

void ThreadManager::NotifyShutdown() {
    if (thread.joinable()) {
        thread.request_stop();
        thread.join();
    }
    VideoCommon::GpuSplit::Get().Stop();
}

} // namespace VideoCommon::GPUThread
