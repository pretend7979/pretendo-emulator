// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_set>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/cpu_features.h"
#include "video_core/nx2_write_profile.h"

// EDEN_NX2_GPU_PROFILE: time where the GPU thread spends its time. Each zone counts its own
// (exclusive) time, nested zones excluded; every 10 s the thread logs each zone's share.
namespace VideoCommon::Nx2Profile {

enum class Zone : u32 {
    Other, ///< GPU thread time outside every zone below (register writes, command parsing)
    Idle,
    PendingSleep,
    ResumePending,
    Push,
    Tick,
    Run,
    FlushInvalidate,
    Macro,
    Draw,
    Clear,
    Compute,
    FlushWork,
    PipelineLookup,
    Configure,
    SyncDescriptors,
    ReadHandles,
    FillImageViews,
    TextureBuffers,
    UpdateBuffers,
    DescriptorAcquire,
    BindStageBuffers,
    PushDescriptors,
    RenderTargets,
    ConfigureDraw,
    DynamicState,
    DrawRecord,
    TextureUpload,
    Fence,        ///< Fences made (SignalFence)
    FenceSkipped, ///< Empty fences not made
    Method,       ///< Engine method calls (what a split GPU thread's back half would run)
    SplitDrain,   ///< Split GPU thread: the front waits for the back (gpu_split.h)
    SplitWait,    ///< Split GPU thread: the back waits for work
    CommandRead,
    CommandFlush,
    CommandRefresh,
    CommandHash,
    CommandEnd,
    CommandDiagnostic,
    GpuCompletionWait,
    WorkerWait,
    DownloadPrepare,
    DownloadRetire,
    DownloadCopy,
    DownloadReactive,
    BufferWriteTrack,
    ProfileDmaScan,
    FenceRetire,
    CommandSyncWait,
    FenceRetirementWait,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Zone::Count)> zone_names{
    "other",         "idle",          "pending sleep",  "resume pending", "push",
    "tick",          "run",           "flush/inval",    "macro",          "draw",
    "clear",         "compute",       "flush work",     "pipeline lookup", "configure",
    "sync descriptors", "read handles", "fill image views", "texture buffers", "update buffers",
    "descriptor acquire", "bind stage buffers", "push descriptors", "render targets",
    "configure draw", "dynamic state", "draw record", "texture upload", "fence",
    "fence skipped", "method", "split drain", "split wait",
    "command read",
    "command flush",
    "command refresh",
    "command hash",
    "command end",
    "command diagnostic",
    "gpu completion wait",
    "worker wait",
    "download prepare",
    "download retire",
    "download CPU copy",
    "reactive download",
    "buffer write track",
    "profile DMA scan",
    "fence retire",
    "command sync wait",
    "fence retirement wait",
};

// Read once at startup: Enabled() runs several times per method write (no guard to check).
inline const bool g_enabled = std::getenv("EDEN_NX2_GPU_PROFILE") != nullptr;

inline bool Enabled() {
    return g_enabled;
}

// Nanoseconds; from the host's own counter where the wall clock reads it directly (the zones
// run ~150,000 times a second, and QueryPerformanceCounter cost ~5% of the GPU thread).
inline u64 Now() {
    return static_cast<u64>(Common::g_wall_clock.GetTimeNS().count());
}

/// Who asked for a host GPU completion wait (see WaitTag), to split "gpu completion wait".
enum class WaitSource : u32 {
    Other,
    Fence,
    BufferWrite,
    BufferDownload,
    ListCopy,
    StagingRing,
    DescriptorRing,
    TextureDownload,
    // Flush requests whose buffer downloads keep these labels (see WaitTag's keep_outer)
    SemaphoreRead,
    MmeDmaRead,
    IndirectList,
    CommandPages,
    GuestCpuRead,
    SafeGpuRead, ///< MemoryManager::ReadBlock without a more specific label
    CommandRead,
    DmaEngine,
    InlineUpload,
    MacroParams,
    StoragePointer,
    CpuWritePage,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(WaitSource::Count)>
    wait_source_names{
        "other",          "fence",          "buffer write",    "buffer download",
        "list copy",      "staging ring",   "descriptor ring", "texture download",
        "semaphore read", "MME DMA read",   "indirect list",   "command pages",
        "guest CPU read", "safe GPU read",  "command read",    "DMA engine",
        "inline upload",  "macro params",   "storage pointer", "CPU write page",
    };

struct ThreadData {
    std::array<u64, static_cast<size_t>(Zone::Count)> exclusive_ns{};
    std::array<u64, static_cast<size_t>(Zone::Count)> count{};
    std::array<u64, 16> counters{}; ///< See Counter
    std::array<u64, static_cast<size_t>(WaitSource::Count)> wait_ns{};
    std::array<u64, static_cast<size_t>(WaitSource::Count)> wait_count{};
    u64 child_ns{};    ///< Time of the zones nested in the current one
    u64 window_start{};
};

inline thread_local ThreadData t_data;
inline thread_local WaitSource t_wait_source{WaitSource::Other};

/// Attributes the GPU completion waits made while it lives to a source.
class WaitTag {
public:
    /// keep_outer: a more specific source set by a caller further up wins.
    explicit WaitTag(WaitSource source, bool keep_outer = false) : previous{t_wait_source} {
        if (!keep_outer || previous == WaitSource::Other) {
            t_wait_source = source;
        }
    }
    ~WaitTag() {
        t_wait_source = previous;
    }

    WaitTag(const WaitTag&) = delete;
    WaitTag& operator=(const WaitTag&) = delete;

private:
    WaitSource previous;
};

/// Times one blocking GPU completion wait against the current WaitTag's source.
class WaitTimer {
public:
    WaitTimer() : enabled{Enabled()} {
        if (enabled) {
            start = Now();
        }
    }
    ~WaitTimer() {
        if (!enabled) {
            return;
        }
        const auto index = static_cast<size_t>(t_wait_source);
        t_data.wait_ns[index] += Now() - start;
        ++t_data.wait_count[index];
    }

    WaitTimer(const WaitTimer&) = delete;
    WaitTimer& operator=(const WaitTimer&) = delete;

private:
    bool enabled;
    u64 start{};
};

class Scope {
public:
    explicit Scope(Zone zone_) : zone{zone_}, enabled{Enabled()} {
        if (!enabled) {
            return;
        }
        parent_child_ns = t_data.child_ns;
        t_data.child_ns = 0;
        start = Now();
    }

    ~Scope() {
        if (!enabled) {
            return;
        }
        const u64 elapsed = Now() - start;
        auto& data = t_data;
        const auto index = static_cast<size_t>(zone);
        data.exclusive_ns[index] += elapsed - data.child_ns;
        ++data.count[index];
        data.child_ns = parent_child_ns + elapsed;
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Zone zone;
    bool enabled;
    u64 start{};
    u64 parent_child_ns{};
};

/// Amounts counted on the GPU thread, logged per second with the zones.
enum class Counter : u32 {
    DownloadBytes,  ///< Buffer data copied back from the host GPU at fences
    DownloadCopies, ///< Buffer ranges copied back
    DmaReadBytes,   ///< Guest memory the native MME read with DMA
    DownloadBytesOnDmaPages, ///< Of DownloadBytes: on pages the MME has DMA-read
    MethodWrites,   ///< 3D register writes (Maxwell3D::CallMethod below the macro range)
    SameValueWrites, ///< Of MethodWrites: the register already held the value
    AcquireChecks,  ///< Semaphore acquires evaluated
    AcquireWaits,   ///< Of AcquireChecks: not met yet (the channel waits)
    BufferWriteRequestedBytes, ///< Conservative byte ranges marked GPU-written
    FenceDownloadQueuedBytes, ///< Bytes selected for fence downloads before coalescing
    DownloadCpuCopyBytes, ///< Bytes actually copied from staging into guest memory
    DownloadCpuCopyRanges, ///< Surviving ranges copied into guest memory
    FenceActionAcquireCalls,
    SemaphoreTriggerFenceCalls,
    SemaphoreExecuteFenceCalls,
    DeferredSemaphoreRetirements,
};

/// Device pages (4 KiB) the native MME has DMA-read since start (GPU thread only; filled only
/// while profiling): which buffer downloads anything on the GPU thread reads back.
inline std::unordered_set<u64>& DmaReadPages() {
    static std::unordered_set<u64> pages;
    return pages;
}

inline void Add(Counter counter, u64 amount) {
    if (Enabled()) {
        t_data.counters[static_cast<size_t>(counter)] += amount;
    }
}

/// Called between commands or fence batches, with no Scope active: logs and resets
/// the totals every 10 s.
inline void Report(const char* thread_name = "GPU thread") {
    if (!Enabled()) {
        return;
    }
    auto& data = t_data;
    const u64 now = Now();
    if (data.window_start == 0) {
        data.window_start = now;
        data.child_ns = 0;
        return;
    }
    const u64 window = now - data.window_start;
    if (window < 10'000'000'000ULL) {
        return;
    }
    // Time in no zone at all is the wall time minus the zones' totals.
    u64 zoned{};
    for (size_t i = 1; i < data.exclusive_ns.size(); ++i) {
        zoned += data.exclusive_ns[i];
    }
    data.exclusive_ns[0] = window > zoned ? window - zoned : 0;
    std::string line;
    for (size_t i = 0; i < data.exclusive_ns.size(); ++i) {
        if (data.exclusive_ns[i] == 0 && data.count[i] == 0) {
            continue;
        }
        // Per second of wall time: milliseconds spent and calls made.
        const double ms = static_cast<double>(data.exclusive_ns[i]) * 1000.0 / window;
        const double calls = static_cast<double>(data.count[i]) * 1e9 / window;
        line += fmt::format(" | {} {:.1f} ms ({:.0f}/s)", zone_names[i], ms, calls);
    }
    line += fmt::format(" | buffer downloads {:.0f} KiB in {:.0f} copies ({:.0f} KiB on pages "
                        "MME DMA reads) | MME DMA reads {:.0f} KiB",
                        static_cast<double>(data.counters[0]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[1]) * 1e9 / window,
                        static_cast<double>(data.counters[3]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[2]) * 1e9 / 1024.0 / window);
    line += fmt::format(" | 3D register writes {:.0f} ({:.0f} same value)",
                        static_cast<double>(data.counters[4]) * 1e9 / window,
                        static_cast<double>(data.counters[5]) * 1e9 / window);
    line += fmt::format(" | acquires {:.0f} ({:.0f} waited)",
                        static_cast<double>(data.counters[6]) * 1e9 / window,
                        static_cast<double>(data.counters[7]) * 1e9 / window);
    const double kib_per_second = 1e9 / 1024.0 / static_cast<double>(window);
    line += fmt::format(" | buffer writes {:.0f} KiB requested, {:.0f} KiB queued at fences",
                        static_cast<double>(data.counters[8]) * kib_per_second,
                        static_cast<double>(data.counters[9]) * kib_per_second);
    line += fmt::format(" | puller fence calls: acquire {:.0f}, trigger {:.0f}, execute {:.0f}",
                        static_cast<double>(data.counters[12]) * 1e9 / static_cast<double>(window),
                        static_cast<double>(data.counters[13]) * 1e9 / static_cast<double>(window),
                        static_cast<double>(data.counters[14]) * 1e9 / static_cast<double>(window));
    line += fmt::format(" | deferred semaphore retirements {:.0f}",
                        static_cast<double>(data.counters[15]) * 1e9 / static_cast<double>(window));
    line += fmt::format(" | buffer CPU copies {:.0f} KiB in {:.0f} ranges",
                        static_cast<double>(data.counters[10]) * kib_per_second,
                        static_cast<double>(data.counters[11]) * 1e9 / static_cast<double>(window));
    line += " | gpu waits by source:";
    for (size_t i = 0; i < data.wait_ns.size(); ++i) {
        if (data.wait_count[i] == 0) {
            continue;
        }
        line += fmt::format(" {} {:.1f} ms ({:.0f}/s)", wait_source_names[i],
                            static_cast<double>(data.wait_ns[i]) * 1000.0 / window,
                            static_cast<double>(data.wait_count[i]) * 1e9 / window);
    }
    LOG_INFO(HW_GPU, "{} per second:{}", thread_name, line);
    Nx2WriteProfile::Report(window, thread_name);
    data = {};
    data.window_start = now;
}

} // namespace VideoCommon::Nx2Profile
