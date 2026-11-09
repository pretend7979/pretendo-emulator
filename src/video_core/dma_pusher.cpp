// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <optional>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/nx2_profile.h"
#include "video_core/dma_pusher.h"
#include "video_core/pushbuffer_submit_probe.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/guest_memory.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra {

constexpr u32 MacroRegistersStart = 0xE00;
constexpr u32 ComputeInline = 0x6D;

namespace {
const char* PushbufferFaultPath();
void NoteBlitFetchBoundary(DmaPusher& pusher, const CommandList& list, size_t entry,
                          u64 fetch_before, u64 fetch_after);
}


DmaPusher::DmaPusher(Core::System& system_, MemoryManager& memory_manager_, Control::ChannelState& channel_state_)
    : system{system_}
    , memory_manager{memory_manager_}
    , channel_state{channel_state_}
    , signal_sync{false}
    , synced{false}
{}

DmaPusher::~DmaPusher() = default;

void DmaPusher::DispatchCalls() {
    // The sub-index is kept: a channel suspended on a semaphore resumes inside its command list.
    dma_state.is_last_call = true;
    while (system.IsPoweredOn()) {
        if (!Step()) {
            break;
        }
    }
    if (const auto* acquire = puller.PendingAcquireForDiagnostics();
        acquire && acquire->retirement && !acquire->semaphore_suspended) {
        return;
    }
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.Defer([gpu = &system.GPU()] {
            VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::CommandEnd};
            gpu->FlushCommands();
            gpu->OnCommandListEnd();
        });
        split.Publish();
        return;
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::CommandEnd};
    system.GPU().FlushCommands();
    system.GPU().OnCommandListEnd();
}

bool DmaPusher::Step() {
    if (!ib_enable || dma_pushbuffer.empty()) {
        return false;
    }
    const auto* acquire = puller.PendingAcquireForDiagnostics();
    const bool resumed_retirement =
        acquire && acquire->retirement && !acquire->semaphore_suspended;
    const u64 suspensions_before = suspensions.load(std::memory_order_relaxed);
    if (!puller.ResumeAcquire(*this)) {
        const auto* pending = puller.PendingAcquireForDiagnostics();
        if (!pending || !pending->retirement) {
            // A genuinely unmet semaphore resumes through the original snapshot/refetch path.
            retained_command_source.reset();
        }
        return false;
    }
    // Lists queued while the channel waited were read from memory, not as submitted: the game
    // writes the commands a list runs before it releases the acquire, so the words at submit time
    // could be older. Reading them as late as the parse is too late in Hyrule Warriors: Age of
    // Imprisonment fights: a command ring the game had already refilled for a later frame, read
    // as this frame's commands (methods on unbound subchannels, garbage QMDs, then a channel
    // waiting forever on a release it never got). The acquire is released now, so the commands
    // are written and the memory not yet reused: copy them here. EDEN_NX2_RESNAPSHOT=0: off.
    if (ResnapshotEnabled() && suspensions.load(std::memory_order_relaxed) != suspensions_before) {
        const u64 now = suspensions.load(std::memory_order_acquire);
        for (CommandList& list : dma_pushbuffer) {
            if (!list.submitted_offsets.empty()) {
                RefreshSnapshot(list, now, CommandSnapshotOrigin::AcquireRefresh);
            }
        }
    }

    CommandList& command_list = dma_pushbuffer.front();

    const size_t prefetch_size = command_list.prefetch_command_list.size();
    const size_t command_list_size = command_list.command_lists.size();

    if (prefetch_size == 0 && command_list_size == 0) {
        dma_pushbuffer.pop_front();
        dma_pushbuffer_subindex = 0;
        return true;
    }

    if (prefetch_size > 0) {
        if (PushbufferFaultPath() != nullptr) command_read_diagnostic = {};
        const std::span<const CommandHeader> commands{command_list.prefetch_command_list};
        command_resume_offset +=
            ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset, "prefetch");
        if (puller.HasPendingAcquire()) {
            return false;
        }
        command_resume_offset = 0;
        dma_pushbuffer.pop_front();
        return true;
    }

    auto& current_command = command_list.command_lists[dma_pushbuffer_subindex];
    const CommandListHeader& header = current_command;
    dma_state.dma_get = header.addr;

    if (!resumed_retirement && signal_sync && !synced) {
        VideoCommon::Nx2Profile::Scope wait_scope{
            VideoCommon::Nx2Profile::Zone::CommandSyncWait};
        std::unique_lock lk(sync_mutex);
        sync_cv.wait(lk, [this]() { return synced.load(); });
        signal_sync = false;
        synced = false;
    }

    if (!resumed_retirement && g_nx2_ampere && command_resume_offset == 0 && header.sync) {
        // Submission copies taken before this fetch barrier cannot supply later segments.
        const u64 previous_fetch = fetch_generation.fetch_add(1, std::memory_order_release);
        NoteBlitFetchBoundary(*this, command_list, dma_pushbuffer_subindex,
                              previous_fetch, previous_fetch + 1);
    }

    if (resumed_retirement) {
        if (!retained_command_source || !CommandSourceMatches(*retained_command_source) ||
            command_resume_offset > retained_command_source->words.size()) {
            UNREACHABLE_MSG("Retired command source changed before continuation");
        }
        const CommandSource source = *retained_command_source;
        retained_command_source.reset();
        if (command_resume_offset < source.words.size()) {
            command_resume_offset += ProcessCommandSource(
                source.words, command_resume_offset, source.label, source.storage, true);
        }
    } else if (header.size > 0) {
        if (subchannels[dma_state.subchannel] && dma_state.method_count) {
            const auto engine = subchannel_type[dma_state.subchannel];
            const bool kepler_payload = engine == Engines::EngineTypes::KeplerCompute && dma_state.method == ComputeInline && dma_state.non_incrementing;
            const bool macro_payload = engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart;
            if (kepler_payload || macro_payload) {
                const size_t words = std::min<size_t>(dma_state.method_count, header.size);
                auto* const engine_state = subchannels[dma_state.subchannel];
                if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
                    // The caches behind IsMemoryDirty belong to the split's back thread.
                    split.Defer([engine_state, memory = &memory_manager, address = GPUVAddr{dma_state.dma_get},
                                 bytes = words * sizeof(u32)] {
                        engine_state->current_dirty = memory->IsMemoryDirty(address, bytes);
                    });
                } else {
                    engine_state->current_dirty = memory_manager.IsMemoryDirty(dma_state.dma_get, words * sizeof(u32));
                }
            }
        }
        if (command_resume_offset == 0) {
            // A method's data continues across GP entries, whatever their level: title 4 ends a
            // main entry after a macro's first word and supplies the remaining parameters in a
            // 5-word subroutine entry. EDEN_NX2_PARK_SPLIT_METHODS restores the old guess (an
            // entry elsewhere is a call that starts clean; resume where the split one ended).
            static const bool park_split_methods = std::getenv("EDEN_NX2_PARK_SPLIT_METHODS") != nullptr;
            if (g_nx2_ampere) {
                if (static u32 split_logs{}; dma_state.method_count != 0 && split_logs < 64) {
                    ++split_logs;
                    LOG_DEBUG(HW_GPU, "Split method {:#x} ({} left) on ch {}: next entry {:#x} size {} raw {:#x} (level bit {}, priv bit {}), expected {:#x}, parked {}",
                             dma_state.method, dma_state.method_count, channel_state.bind_id,
                             u64{header.addr}, u64{header.size}, header.raw, u64{header.is_push_buffer},
                             u64{header.allow_flush}, split_method_end, parked_methods.size());
                    std::array<u32, 8> next_words{};
                    std::array<u32, 4> prev_words{};
                    memory_manager.ReadBlockUnsafe(header.addr, next_words.data(),
                                             std::min<size_t>(header.size, 8) * sizeof(u32));
                    memory_manager.ReadBlockUnsafe(split_method_end - sizeof(prev_words), prev_words.data(),
                                             sizeof(prev_words));
                    LOG_DEBUG(HW_GPU, "  previous entry ends {:08x}; next entry starts {:08x}",
                              fmt::join(prev_words, " "), fmt::join(next_words, " "));
                }
                if (park_split_methods) {
                    ResolveSplitMethod(header.addr);
                }
            }
            if (dma_pushbuffer_subindex < command_list.submit_hashes.size()) {
                CheckSubmitHash(header, command_list.submit_hashes[dma_pushbuffer_subindex]);
            }
        }
        // A resumed entry is read again: the words after the acquire may be new by now.
        // A safe read flushes the caches, which belong to the split's back thread while it runs
        // (the split only starts with unsafe reads; the settings can change mid-game).
        const bool use_safe = !VideoCommon::GpuSplit::Get().Active() &&
                              (Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh() : Settings::IsDMALevelSafe());
        // The split's back marks the pages GPU work writes as it runs that work: a page first
        // written by a dispatch or copy still queued for it reads as unwritten, and its words as
        // stale (Fast Fusion's culling fills in instanced draws' arguments; stale ones asked for
        // billions of instances and lost the device loading a race).
        const size_t entry_bytes = size_t{header.size} * sizeof(u32);
        bool gpu_written = IsGpuWrittenCommands(dma_state.dma_get, entry_bytes);
        if (!gpu_written && VideoCommon::GpuSplit::Get().Active() &&
            !IsTrustedCpuCommands(dma_state.dma_get, entry_bytes)) {
            if (VideoCommon::GpuSplit::Get().WaitForWriters()) {
                gpu_written = IsGpuWrittenCommands(dma_state.dma_get, entry_bytes);
            }
            if (!gpu_written) {
                NoteCpuCommands(dma_state.dma_get, entry_bytes);
            }
        }
        // The words as submitted, unless this is a resumed entry or the GPU has newer ones the
        // safe read below would flush (with the split, the front can't tell: it reads memory
        // unflushed anyway). Entries on pages the GPU writes (culling) always take the flushing
        // read: their words at submit time are an older frame's.
        CommandReadDiagnostic* diagnostic = nullptr;
        if (PushbufferFaultPath() != nullptr) {
            command_read_diagnostic = {
                .valid = true, .use_safe = use_safe, .gpu_written = gpu_written,
                .sync_wait = static_cast<bool>(header.sync),
                .resume_offset = command_resume_offset, .entry_index = dma_pushbuffer_subindex,
                .snapshot_fetch = command_list.snapshot_fetch_generation,
                .compared_fetch = fetch_generation.load(std::memory_order_acquire),
                .snapshot_suspensions = command_list.snapshot_suspensions,
                .original_fetch = command_list.snapshot_original_fetch_generation,
                .original_suspensions = command_list.snapshot_original_suspensions,
                .refresh_count = command_list.snapshot_refresh_count,
                .origin = command_list.snapshot_origin,
            };
            diagnostic = &command_read_diagnostic;
        }
        // The wrappers retain the original evaluation order and number of memory/atomic queries.
        const auto check = [&](u32 bit, bool value) {
            if (diagnostic) {
                diagnostic->checked |= bit;
                if (!value) diagnostic->rejected |= bit;
            }
            return value;
        };
        const auto current_suspensions = [&] {
            const u64 value = suspensions.load(std::memory_order_acquire);
            if (diagnostic) diagnostic->compared_suspensions = value;
            return value;
        };
        const auto dirty = [&] {
            const bool value = memory_manager.IsMemoryDirty(dma_state.dma_get,
                                                            header.size * sizeof(u32));
            if (diagnostic) {
                diagnostic->dirty_checked = true;
                diagnostic->dirty = value;
            }
            return value;
        };
        // A SYNC_WAIT barrier processed since the submission does not reject the snapshot: when
        // the GPU thread runs behind, nearly every queued list is past one, and OoT has by then
        // refilled that command memory for a later frame (bit 2 rejections, then garbage render
        // targets loading Kakariko). GPU-written pages and acquires still reject it below.
        const bool submitted = check(1, command_resume_offset == 0) &&
                               check(4, dma_pushbuffer_subindex < command_list.submitted_offsets.size()) &&
                               check(8, (command_list.snapshot_suspensions & 1) == 0) &&
                               check(16, command_list.snapshot_suspensions == current_suspensions()) &&
                               check(32, !gpu_written) && check(64, !(use_safe && dirty()));
        if (diagnostic) diagnostic->selected = submitted;
        if (submitted) {
            const std::span<const CommandHeader> commands{
                command_list.submitted_words.data() +
                    command_list.submitted_offsets[dma_pushbuffer_subindex],
                header.size};
            RecordParsedEntry(header, commands);
            command_resume_offset += ProcessCommandSource(
                commands, 0, "submitted", CommandSourceStorage::Submitted);
        } else if (use_safe) {
            VideoCommon::Nx2Profile::WaitTag wait_tag{VideoCommon::Nx2Profile::WaitSource::CommandRead};
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::SafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            RecordParsedEntry(header, commands);
            command_resume_offset +=
                ProcessCommandSource(commands, command_resume_offset, "safe",
                    commands.data() == command_headers.data() ? CommandSourceStorage::Scratch
                                                            : CommandSourceStorage::Physical);
        } else if (ReadGpuWrittenCommands(dma_state.dma_get, static_cast<u32>(header.size))) {
            const std::span<const CommandHeader> commands{command_headers.data(), header.size};
            RecordParsedEntry(header, commands);
            command_resume_offset +=
                ProcessCommandSource(commands, command_resume_offset, "gpu-page-copy",
                                     CommandSourceStorage::Scratch);
        } else {
            FlushGpuWrittenCommands(dma_state.dma_get, size_t{header.size} * sizeof(u32));
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::UnsafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            RecordParsedEntry(header, commands);
            command_resume_offset +=
                ProcessCommandSource(commands, command_resume_offset, "unsafe-after-flush",
                    commands.data() == command_headers.data() ? CommandSourceStorage::Scratch
                                                            : CommandSourceStorage::Physical);
        }
    }
    if (header.size > 0) {
        if (puller.HasPendingAcquire()) {
            return false;
        }
        command_resume_offset = 0;
        if (dma_state.method_count != 0) {
            split_method_end = header.addr + header.size * sizeof(u32);
        }
    }

    if (++dma_pushbuffer_subindex >= command_list_size) {
        // A split method resumes within the submission that split it; ring addresses repeat
        // every frame, so a stale one could otherwise swallow the headers of a later entry.
        for (const auto& parked : parked_methods) {
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1) < 16) {
                LOG_WARNING(HW_GPU,
                            "Submission ended with parked method {:#x} (sub {}, {} words left, "
                            "resume {:#x})",
                            parked.method, parked.subchannel, parked.method_count,
                            parked.resume_address);
            }
        }
        parked_methods.clear();
        dma_pushbuffer.pop_front();
        dma_pushbuffer_subindex = 0;
    } else {
        signal_sync = command_list.command_lists[dma_pushbuffer_subindex].sync && Settings::values.sync_memory_operations.GetValue();
    }

    if (signal_sync) {
        const auto signal = [this] {
            rasterizer->SignalFence([this]() {
                std::scoped_lock lk(sync_mutex);
                synced = true;
                sync_cv.notify_all();
            });
        };
        // The fence is queued on the Vulkan scheduler, which the split's back owns even while
        // idle (see FlushGpuWrittenCommands).
        if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
            split.Defer(signal);
            split.Drain();
        } else {
            signal();
        }
    }

    return true;
}

// EDEN_NX2_SEM_WATCH=<gpu address>: log every method whose argument is the low word of an
// address in the 16 bytes there (finds the engine and method that release a semaphore).
static const u32 g_semaphore_watch = [] {
    const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
    return value ? static_cast<u32>(std::strtoull(value, nullptr, 16)) : 0U;
}();

// EDEN_NX2_PB_CHECK also keeps the last parsed entries, for DumpCurrentEntry.
static const bool g_record_parsed_entries = std::getenv("EDEN_NX2_PB_CHECK") != nullptr;

namespace {

constexpr size_t PushbufferFaultWordLimit = 0x4000;

const char* PushbufferFaultPath() {
    static const char* const path = [] {
        const char* value = std::getenv("EDEN_NX2_PB_FIRST_FAULT");
        return value != nullptr && value[0] != '\0' ? value : nullptr;
    }();
    return path;
}

struct PushbufferFaultContext;
thread_local PushbufferFaultContext* current_pushbuffer_fault = nullptr;

struct PushbufferSegmentEndEvent {
    int channel{};
    GPUVAddr address{};
    size_t entry{}, word{};
    u32 raw{};
};
struct PushbufferSegmentEndHistory {
    std::array<PushbufferSegmentEndEvent, 32> events;
    u64 total{};
};
thread_local PushbufferSegmentEndHistory pushbuffer_segment_end_history;

struct CompletedPushbufferEntry {
    int channel{};
    GPUVAddr address{};
    size_t entry{}, word_offset{}, total_words{}, processed_words{};
    const char* source{};
    DmaPusher::DmaState initial_state{}, final_state{};
    bool initial_increment_once{}, final_increment_once{};
    CommandReadDiagnostic read_diagnostic{};
    std::vector<u32> before_words, submitted_words;
};
struct ParsedPushbufferHeader {
    int channel{};
    GPUVAddr address{};
    size_t entry{}, word_offset{}, parser_index{}, span_words{};
    u64 completed_entry_serial{}, fetch_generation{}, suspensions{};
    const char* source{};
    u32 parsed{}, before{}, submitted{};
    bool before_valid{}, submitted_valid{};
};
// One classified difference survives the rolling completed-entry history for this owner slot.
struct SubmittedPushbufferMismatch {
    int channel{};
    GPUVAddr address{};
    size_t entry{}, word_offset{}, total_words{}, first_different_word{};
    u64 completed_entry_serial{}, observation_steady_ns{};
    const char* source{};
    DmaPusher::DmaState initial_state{};
    bool initial_increment_once{};
    CommandReadDiagnostic read_diagnostic{};
    size_t first_neighbor{}, neighbor_count{}, list_entries{};
    std::array<u64, 3> neighbor_headers{};
    std::vector<u32> before_words, submitted_words;
};

// Default-off CPU-only boundary diagnostics for the recurrent OOT blit-ring fault.
// No saved words from this record participate in command selection or execution.
struct BlitBoundaryObservation {
    u64 serial{}, fetch_before{}, fetch_after{}, suspensions{}, completed_serial{};
    u64 snapshot_fetch{}, snapshot_suspensions{}, original_fetch{}, original_suspensions{};
    u64 refresh_count{}, boundary_header{}, command_header{}, observation_steady_ns{};
    CommandSnapshotOrigin origin{};
    GPUVAddr address{};
    int channel{};
    bool cpu_copy_valid{}, parse_valid{};
    std::array<u32, 25> submitted{}, at_boundary{}, at_parse{};
    CommandReadDiagnostic parse_decision{};
    const char* parse_source{};
    std::shared_ptr<const BlitSubmitObservation> submit_identity;
    BlitPointerIdentity boundary_before{}, boundary_after{}, parse_identity{};
    BlitSyncpointObservation boundary_syncpoint{}, parse_syncpoint{}, input_boundary{}, input_parse{};
    u64 publish_return_at_boundary{}, tail_queued_at_boundary{};
};

struct CompletedPushbufferHistory {
    DmaPusher* owner{};
    std::array<CompletedPushbufferEntry, 8> entries;
    u64 total{};
    std::array<ParsedPushbufferHeader, 64> headers;
    u64 header_total{};
    std::optional<SubmittedPushbufferMismatch> first_submitted_mismatch;
    std::array<BlitBoundaryObservation, 16> blit_boundaries{};
    u64 blit_boundary_total{};
    std::optional<BlitBoundaryObservation> first_blit_boundary_difference;
};
thread_local std::array<CompletedPushbufferHistory, 8> completed_pushbuffer_histories;
thread_local size_t completed_pushbuffer_next_owner{};

CompletedPushbufferHistory& CompletedHistoryFor(DmaPusher& pusher) {
    for (auto& history : completed_pushbuffer_histories) {
        if (history.owner == &pusher) {
            return history;
        }
    }
    auto& history = completed_pushbuffer_histories[
        completed_pushbuffer_next_owner++ % completed_pushbuffer_histories.size()];
    history = {};
    history.owner = &pusher;
    return history;
}


bool BlitBoundaryTraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PB_BOUNDARY_TRACE");
        return value != nullptr && std::string_view{value} == "1";
    }();
    return enabled && PushbufferFaultPath() != nullptr &&
           Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL;
}

bool IsBlitBoundaryCandidate(const CommandList& list, size_t entry) {
    // This filter matches the captured five-entry blit submission, not arbitrary
    // startup traffic. Its two low address words may relocate between runs.
    constexpr std::array<u32, 25> shape{
        0x91020002, 0x200181ca, 0x00001040, 0x200181cb, 0x00001e00,
        0x200181cc, 0x00000438, 0x200181cd, 0x00000001, 0x200181cf,
        0x00000000, 0x20018105, 0x00001e00, 0x20028100, 0x00000022,
        0, 0x20028102, 0x00000022, 0, 0x20028106,
        0x00001e00, 0x00000438, 0x200180c0, 0x00000306, 0x91000002,
    };
    if (entry != 1 || list.command_lists.size() != 5 ||
        list.submitted_offsets.size() != 5 || !list.prefetch_command_list.empty() ||
        list.snapshot_origin != CommandSnapshotOrigin::Submit) return false;
    const auto& boundary = list.command_lists[1];
    const auto& next = list.command_lists[2];
    if (!boundary.sync || boundary.size != 1 || next.sync || next.size != shape.size())
        return false;
    const size_t boundary_offset = list.submitted_offsets[1];
    const size_t offset = list.submitted_offsets[2];
    if (boundary_offset >= list.submitted_words.size() ||
        list.submitted_words[boundary_offset].argument != 0 ||
        offset > list.submitted_words.size() ||
        shape.size() > list.submitted_words.size() - offset) return false;
    for (size_t word = 0; word < shape.size(); ++word) {
        if (word != 15 && word != 18 &&
            list.submitted_words[offset + word].argument != shape[word]) return false;
    }
    return true;
}

void NoteBlitFetchBoundary(DmaPusher& pusher, const CommandList& list, size_t entry,
                          u64 fetch_before, u64 fetch_after) {
    if (!BlitBoundaryTraceEnabled() || !IsBlitBoundaryCandidate(list, entry)) return;
    auto& history = CompletedHistoryFor(pusher);
    // A ring, not a first-N cap: later failures retain their nearby boundaries.
    const u64 serial = history.blit_boundary_total++;
    auto& record = history.blit_boundaries[serial % history.blit_boundaries.size()];
    record = {};
    record.serial = serial;
    record.fetch_before = fetch_before;
    record.fetch_after = fetch_after;
    record.suspensions = pusher.suspensions.load(std::memory_order_relaxed);
    record.completed_serial = history.total;
    record.channel = pusher.channel_state.bind_id;
    record.snapshot_fetch = list.snapshot_fetch_generation;
    record.snapshot_suspensions = list.snapshot_suspensions;
    record.original_fetch = list.snapshot_original_fetch_generation;
    record.original_suspensions = list.snapshot_original_suspensions;
    record.refresh_count = list.snapshot_refresh_count;
    record.origin = list.snapshot_origin;
    record.boundary_header = list.command_lists[1].raw;
    record.command_header = list.command_lists[2].raw;
    record.address = list.command_lists[2].addr;
    if (BlitSubmitProbeEnabled() && list.blit_submit_observation) {
        record.submit_identity = list.blit_submit_observation;
        record.boundary_before = ReadBlitPointerIdentity(pusher.system, pusher.memory_manager, record.address);
        record.boundary_syncpoint = ReadBlitSyncpoint(pusher.system, record.submit_identity->output_fence_id);
        if (record.submit_identity->input_flags & 1U)
            record.input_boundary = ReadBlitSyncpoint(pusher.system, record.submit_identity->input_fence_id);
        record.publish_return_at_boundary = record.submit_identity->publish_return_ns.load(std::memory_order_acquire);
        record.tail_queued_at_boundary = record.submit_identity->ioctl_tail_queued_ns.load(std::memory_order_acquire);
    }
    record.observation_steady_ns = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    for (size_t word = 0; word < record.submitted.size(); ++word)
        record.submitted[word] = list.submitted_words[list.submitted_offsets[2] + word].argument;
    constexpr size_t bytes = sizeof(record.at_boundary);
    // A missing mapping/backing is reported as missing, never as fabricated zero data.
    // Restrict to one mapped page so GetPointer covers the complete100-byte copy.
    auto& memory = pusher.memory_manager;
    if (record.address > UINT64_MAX - (bytes - 1) ||
        (record.address & 0xfff) > 0x1000 - bytes ||
        !memory.IsWithinGPUAddressRange(record.address) ||
        !memory.IsWithinGPUAddressRange(record.address + bytes - 1) ||
        !memory.IsFullyMappedRange(record.address, bytes) ||
        memory.GetPointer(record.address) == nullptr) return;
    memory.ReadBlockUnsafe(record.address, record.at_boundary.data(), bytes);
    if (record.submit_identity)
        record.boundary_after = ReadBlitPointerIdentity(pusher.system, memory, record.address);
    record.cpu_copy_valid = true;
}

void NoteBlitBoundaryParse(DmaPusher& pusher, std::span<const u32> words,
                           size_t word_offset, const char* source,
                           const CommandReadDiagnostic& decision) {
    if (!BlitBoundaryTraceEnabled() || word_offset != 0 || words.size() != 25 ||
        !decision.valid || decision.resume_offset != 0 || decision.entry_index != 2 ||
        (decision.checked & 2) == 0 || std::string_view{source} == "prefetch" ||
        pusher.dma_pushbuffer.empty()) return;
    const auto& list = pusher.dma_pushbuffer.front();
    if (!IsBlitBoundaryCandidate(list, 1)) return;
    auto& history = CompletedHistoryFor(pusher);
    const u64 first = history.blit_boundary_total > history.blit_boundaries.size()
                          ? history.blit_boundary_total - history.blit_boundaries.size() : 0;
    for (u64 i = history.blit_boundary_total; i > first; --i) {
        auto& record = history.blit_boundaries[(i - 1) % history.blit_boundaries.size()];
        if (record.parse_valid || record.fetch_after != decision.compared_fetch ||
            record.command_header != list.command_lists[2].raw ||
            record.address != pusher.dma_state.dma_get ||
            record.snapshot_fetch != decision.snapshot_fetch ||
            record.snapshot_suspensions != decision.snapshot_suspensions) continue;
        std::copy(words.begin(), words.end(), record.at_parse.begin());
        record.parse_valid = true;
        record.parse_decision = decision;
        record.parse_source = source;
        if (record.submit_identity) {
            record.parse_identity = ReadBlitPointerIdentity(pusher.system, pusher.memory_manager, record.address);
            record.parse_syncpoint = ReadBlitSyncpoint(pusher.system, record.submit_identity->output_fence_id);
            if (record.submit_identity->input_flags & 1U)
                record.input_parse = ReadBlitSyncpoint(pusher.system, record.submit_identity->input_fence_id);
        }
        // Preserve the first matching classified difference even after ring wrap.
        if (!history.first_blit_boundary_difference && !decision.selected &&
            decision.rejected != 0 && !decision.use_safe && !decision.gpu_written &&
            std::string_view{source} == "unsafe-after-flush" &&
            record.at_parse != record.submitted) {
            history.first_blit_boundary_difference = record;
        }
        break;
    }
}

struct PushbufferFaultContext {
    DmaPusher& pusher;
    std::span<const CommandHeader> live_words;
    std::vector<u32> before_words;
    std::vector<u32> submitted_words;
    size_t word_offset;
    size_t parser_index{};
    const char* source;
    PushbufferFaultContext* previous;
    DmaPusher::DmaState initial_state;
    CommandReadDiagnostic read_diagnostic;
    bool initial_increment_once;
    size_t initial_entry;
    size_t processed_words{};
    bool returned{};
    bool final_increment_once{};

    PushbufferFaultContext(DmaPusher& pusher_, std::span<const CommandHeader> commands,
                          size_t word_offset_, const char* source_, bool initial_increment_once_ = false)
        : pusher{pusher_}, live_words{commands}, word_offset{word_offset_}, source{source_},
          previous{current_pushbuffer_fault}, initial_state{pusher_.dma_state},
          read_diagnostic{pusher_.command_read_diagnostic},
          initial_increment_once{initial_increment_once_},
          initial_entry{pusher_.dma_pushbuffer_subindex} {
        const size_t count = std::min(commands.size(), PushbufferFaultWordLimit);
        before_words.reserve(count);
        for (const CommandHeader& word : commands.first(count)) {
            before_words.push_back(word.argument);
        }
        if (std::string_view{source} != "prefetch" && !pusher.dma_pushbuffer.empty()) {
            const auto& list = pusher.dma_pushbuffer.front();
            const size_t entry = pusher.dma_pushbuffer_subindex;
            if (entry < list.submitted_offsets.size()) {
                const size_t begin = list.submitted_offsets[entry];
                if (begin <= list.submitted_words.size() &&
                    word_offset <= list.submitted_words.size() - begin) {
                    const size_t offset = begin + word_offset;
                    const size_t available = list.submitted_words.size() - offset;
                    // Only the current entry, never the following entry's submitted words.
                    if (entry < list.command_lists.size() &&
                        word_offset <= list.command_lists[entry].size) {
                        const size_t entry_left = list.command_lists[entry].size - word_offset;
                        const size_t submitted_count = std::min({count, available, entry_left});
                        for (const auto& word : std::span{list.submitted_words}.subspan(
                                 offset, submitted_count)) {
                            submitted_words.push_back(word.argument);
                        }
                    }
                }
            }
        }
        current_pushbuffer_fault = this;
        NoteBlitBoundaryParse(pusher, before_words, word_offset, source, read_diagnostic);
        NoteSubmittedMismatch();
    }

    ~PushbufferFaultContext() {
        VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::CommandDiagnostic};
        current_pushbuffer_fault = previous;
        if (returned) {
            auto& history = CompletedHistoryFor(pusher);
            history.entries[history.total++ % history.entries.size()] = {
                .channel = pusher.channel_state.bind_id,
                .address = initial_state.dma_get,
                .entry = initial_entry,
                .word_offset = word_offset,
                .total_words = live_words.size(),
                .processed_words = processed_words,
                .source = source,
                .initial_state = initial_state,
                .final_state = pusher.dma_state,
                .initial_increment_once = initial_increment_once,
                .final_increment_once = final_increment_once,
                .read_diagnostic = read_diagnostic,
                .before_words = std::move(before_words),
                .submitted_words = std::move(submitted_words),
            };
        }
    }

    void NoteSubmittedMismatch() {
        // Safe/GPU-written/resumed entries can legitimately differ from their submission copy.
        if (!read_diagnostic.valid || read_diagnostic.selected || read_diagnostic.rejected == 0 ||
            read_diagnostic.use_safe || read_diagnostic.gpu_written ||
            read_diagnostic.resume_offset != 0 || word_offset != 0 ||
            std::string_view{source} != "unsafe-after-flush" || before_words.empty() ||
            before_words.size() != submitted_words.size()) {
            return;
        }
        auto& history = CompletedHistoryFor(pusher);
        if (history.first_submitted_mismatch) {
            return;
        }
        const auto different = std::mismatch(before_words.begin(), before_words.end(),
                                             submitted_words.begin()).first;
        if (different == before_words.end()) {
            return;
        }
        auto& mismatch = history.first_submitted_mismatch.emplace();
        mismatch.channel = pusher.channel_state.bind_id;
        mismatch.address = initial_state.dma_get;
        mismatch.entry = initial_entry;
        mismatch.word_offset = word_offset;
        mismatch.total_words = live_words.size();
        mismatch.first_different_word = static_cast<size_t>(different - before_words.begin());
        mismatch.completed_entry_serial = history.total;
        // This clock's epoch is implementation-defined; the one-time log supplies log-time context.
        mismatch.observation_steady_ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        mismatch.source = source;
        mismatch.initial_state = initial_state;
        mismatch.initial_increment_once = initial_increment_once;
        mismatch.read_diagnostic = read_diagnostic;
        mismatch.before_words = before_words;
        mismatch.submitted_words = submitted_words;
        if (!pusher.dma_pushbuffer.empty()) {
            const auto& list = pusher.dma_pushbuffer.front();
            mismatch.list_entries = list.command_lists.size();
            mismatch.first_neighbor = initial_entry > 0 ? initial_entry - 1 : 0;
            for (size_t i = mismatch.first_neighbor;
                 i < list.command_lists.size() && mismatch.neighbor_count < mismatch.neighbor_headers.size(); ++i) {
                mismatch.neighbor_headers[mismatch.neighbor_count++] = list.command_lists[i].raw;
            }
        }
        LOG_INFO(HW_GPU, "Pushbuffer submitted/live difference: class=unsafe_non_gpu_unresumed "
                        "channel {} entry {} serial {} get={:#x} first_word={} "
                        "compared_words={} total_words={} observation_steady_ns={} rejected={:#x}",
                 mismatch.channel, mismatch.entry, mismatch.completed_entry_serial,
                 mismatch.address, mismatch.first_different_word, mismatch.before_words.size(),
                 mismatch.total_words, mismatch.observation_steady_ns, read_diagnostic.rejected);
    }

    void NoteHeader(size_t index, u32 raw) {
        auto& history = CompletedHistoryFor(pusher);
        history.headers[history.header_total++ % history.headers.size()] = {
            .channel = pusher.channel_state.bind_id,
            .address = initial_state.dma_get,
            .entry = initial_entry,
            .word_offset = word_offset,
            .parser_index = index,
            .span_words = live_words.size(),
            .completed_entry_serial = history.total,
            .fetch_generation = pusher.fetch_generation.load(std::memory_order_relaxed),
            .suspensions = pusher.suspensions.load(std::memory_order_relaxed),
            .source = source,
            .parsed = raw,
            .before = index < before_words.size() ? before_words[index] : 0,
            .submitted = index < submitted_words.size() ? submitted_words[index] : 0,
            .before_valid = index < before_words.size(),
            .submitted_valid = index < submitted_words.size(),
        };
    }

    void NoteSegmentEndCandidate(size_t index, u32 raw) {
        auto& history = pushbuffer_segment_end_history;
        const u64 event_number = history.total++;
        history.events[event_number % history.events.size()] = {
            pusher.channel_state.bind_id, pusher.dma_state.dma_get,
            pusher.dma_pushbuffer_subindex, word_offset + index, raw,
        };
        if (event_number < 8) {
            LOG_INFO(HW_GPU, "Eligible END_PB_SEGMENT: channel {} entry {} get={:#x} word={} "
                            "raw={:08x} source={}", pusher.channel_state.bind_id,
                     pusher.dma_pushbuffer_subindex, pusher.dma_state.dma_get,
                     word_offset + index, raw, source);
        }
    }
};

void WriteFaultWords(std::FILE* file, const char* name, std::span<const u32> words) {
    const size_t count = std::min(words.size(), PushbufferFaultWordLimit);
    std::fprintf(file, "%s total=%zu captured=%zu hash=%016llx\n", name, words.size(), count,
                 static_cast<unsigned long long>(HashPushbufferWords(words.first(count))));
    for (size_t word = 0; word < count; ++word) {
        if ((word % 8) == 0) {
            std::fprintf(file, "%06zx:", word * sizeof(u32));
        }
        std::fprintf(file, " %08x", words[word]);
        if ((word % 8) == 7 || word + 1 == count) {
            std::fputc('\n', file);
        }
    }
}


void WriteReadDecision(std::FILE* file, const char* prefix, const CommandReadDiagnostic& d) {
    std::fprintf(file, "%s valid=%u selected=%u checked=%x rejected=%x resume=%llu entry=%llu "
                       "snapshot_fetch=%llu compared_fetch=%llu snapshot_suspensions=%llu "
                       "compared_suspensions=%llu gpu_written=%u use_safe=%u dirty_checked=%u "
                       "dirty=%u sync_wait=%u origin=%u original_fetch=%llu original_suspensions=%llu refreshes=%llu\n",
                 prefix, unsigned(d.valid), unsigned(d.selected), d.checked, d.rejected,
                 static_cast<unsigned long long>(d.resume_offset),
                 static_cast<unsigned long long>(d.entry_index),
                 static_cast<unsigned long long>(d.snapshot_fetch),
                 static_cast<unsigned long long>(d.compared_fetch),
                 static_cast<unsigned long long>(d.snapshot_suspensions),
                 static_cast<unsigned long long>(d.compared_suspensions),
                 unsigned(d.gpu_written), unsigned(d.use_safe), unsigned(d.dirty_checked),
                 unsigned(d.dirty), unsigned(d.sync_wait), static_cast<u32>(d.origin),
                 static_cast<unsigned long long>(d.original_fetch),
                 static_cast<unsigned long long>(d.original_suspensions),
                 static_cast<unsigned long long>(d.refresh_count));
}


void WriteBlitPointerIdentity(std::FILE* file, const char* prefix, const char* stage,
                              const BlitPointerIdentity& value) {
    std::fprintf(file, "%s_identity stage=%s time_ns=%llu gmmu=%llu range_valid=%u "
                       "device_first_valid=%u device_last_valid=%u device_first=%016llx "
                       "device_last=%016llx physical_first=%016llx physical_last=%016llx "
                       "host_first=%016llx host_last=%016llx\n",
                 prefix, stage, static_cast<unsigned long long>(value.time_ns),
                 static_cast<unsigned long long>(value.gmmu_id), unsigned(value.range_valid),
                 unsigned(value.device_first_valid), unsigned(value.device_last_valid),
                 static_cast<unsigned long long>(value.device_first),
                 static_cast<unsigned long long>(value.device_last),
                 static_cast<unsigned long long>(value.physical_first),
                 static_cast<unsigned long long>(value.physical_last),
                 static_cast<unsigned long long>(value.host_first),
                 static_cast<unsigned long long>(value.host_last));
}

void WriteBlitSyncpoint(std::FILE* file, const char* prefix, const char* stage,
                        const BlitSyncpointObservation& value) {
    std::fprintf(file, "%s_syncpoint stage=%s valid=%u id=%u guest=%u host=%u\n", prefix,
                 stage, unsigned(value.valid), value.id, value.guest, value.host);
}

void WriteCopybackWatch(std::FILE* file, const char* prefix,
                        const Common::CopybackWatch::Attachment& attachment) {
    using namespace Common::CopybackWatch;
    if (!Enabled()) return;
    const auto ull = [](u64 value) { return static_cast<unsigned long long>(value); };
    auto& state = State();
    std::fprintf(file, "%s_copyback reason=%u submit_ticket=%llu registry_busy=%llu "
        "capacity_rejected=%llu identity_rejected=%llu allocation_rejected=%llu "
        "payload=source_sample_before_copy unknown_writers_not_excluded=1\n",
        prefix, static_cast<u32>(attachment.reason), ull(attachment.ticket),
        ull(state.dropped_busy.load(std::memory_order_relaxed)),
        ull(state.rejected_capacity.load(std::memory_order_relaxed)),
        ull(state.rejected_identity.load(std::memory_order_relaxed)),
        ull(state.rejected_allocation.load(std::memory_order_relaxed)));
    if (!attachment.watch) return;
    const auto& identity = attachment.watch->identity;
    const auto snapshot = ReadSnapshot(*attachment.watch);
    std::fprintf(file, "%s_copyback_watch session=%llu gmmu=%llu device=%016llx "
        "physical=%016llx host=%016llx bytes=100 tickets=%llu events=%zu "
        "ring_capacity=%zu busy_slots=%llu dropped_busy=%llu dropped_overflow=%llu "
        "dropped_ticket_race=%llu overwritten_possible=%u\n", prefix,
        ull(identity.session), ull(identity.gmmu), ull(identity.device), ull(identity.physical),
        ull(identity.host), ull(snapshot.tickets), snapshot.count, EventCapacity,
        ull(snapshot.busy_slots), ull(snapshot.dropped_busy), ull(snapshot.dropped_overflow),
        ull(snapshot.dropped_ticket_race), unsigned(snapshot.tickets > EventCapacity));
    for (size_t index = 0; index < snapshot.count; ++index) {
        const auto& event = snapshot.events[index];
        std::fprintf(file, "%s_copyback_event ticket=%llu start_ns=%llu completion_ns=%llu "
            "thread=%llu kind=%u seam=%u writer_session=%llu writer_address=%016llx dst=%016llx src=%016llx "
            "copy_bytes=%llu overlap_offset=%u overlap_bytes=%u sampled_source=", prefix,
            ull(event.ticket), ull(event.start_ns), ull(event.completion_ns), ull(event.thread),
            static_cast<u32>(event.kind), static_cast<u32>(event.seam), ull(event.writer_session), ull(event.writer_address),
            ull(event.destination), ull(event.source), ull(event.copy_size),
            event.overlap_offset, event.overlap_size);
        for (size_t i = 0; i < event.overlap_size; ++i) std::fprintf(file, "%02x", event.bytes[i]);
        std::fprintf(file, "\n");
    }
}

void WriteBlitSubmitIdentity(std::FILE* file, const char* prefix,
                             const BlitBoundaryObservation& record) {
    const auto& value = *record.submit_identity;
    WriteCopybackWatch(file, prefix, value.copyback);
    std::fprintf(file, "%s_submit serial=%llu input_flags=%08x input_fence=(%u,%u) "
                       "output_fence=(%u,%u) increment=%u max_before=%u min_cached_before=%u "
                       "publish_begin_ns=%llu publish_return_seen_at_boundary_ns=%llu "
                       "ioctl_tail_queued_seen_at_boundary_ns=%llu first_entry_total=%u "
                       "first_entry_captured=%u\n",
                 prefix, static_cast<unsigned long long>(value.serial), value.input_flags,
                 value.input_fence_id, value.input_fence_value,
                 value.output_fence_id, value.output_fence_value, value.increment,
                 value.max_before, value.min_cached_before,
                 static_cast<unsigned long long>(value.publish_begin_ns),
                 static_cast<unsigned long long>(record.publish_return_at_boundary),
                 static_cast<unsigned long long>(record.tail_queued_at_boundary),
                 value.first_entry_total, value.first_entry_captured);
    std::fprintf(file, "%s_submit_headers", prefix);
    for (const u64 header : value.headers)
        std::fprintf(file, " %016llx", static_cast<unsigned long long>(header));
    std::fprintf(file, "\n");
    WriteBlitPointerIdentity(file, prefix, "submit_before_copy", value.before);
    WriteBlitPointerIdentity(file, prefix, "submit_after_copy", value.after);
    WriteBlitPointerIdentity(file, prefix, "boundary_before_copy", record.boundary_before);
    WriteBlitPointerIdentity(file, prefix, "boundary_after_copy", record.boundary_after);
    if (record.parse_valid) WriteBlitPointerIdentity(file, prefix, "parse", record.parse_identity);
    WriteBlitSyncpoint(file, prefix, "submit_snapshot", value.at_snapshot);
    WriteBlitSyncpoint(file, prefix, "submit_before_publish", value.before_publish);
    WriteBlitSyncpoint(file, prefix, "boundary", record.boundary_syncpoint);
    if (record.parse_valid) WriteBlitSyncpoint(file, prefix, "parse", record.parse_syncpoint);
    if (value.input_flags & 1U) {
        WriteBlitSyncpoint(file, prefix, "input_dependency_submit", value.input_dependency);
        WriteBlitSyncpoint(file, prefix, "input_dependency_boundary", record.input_boundary);
        if (record.parse_valid) WriteBlitSyncpoint(file, prefix, "input_dependency_parse", record.input_parse);
    }
    WriteFaultWords(file, (std::string{prefix} + "_first_entry_submitted").c_str(),
                    std::span{value.first_entry_words}.first(value.first_entry_captured));
}

void WriteBlitBoundaryObservation(std::FILE* file, const char* prefix,
                                  const BlitBoundaryObservation& record) {
    std::fprintf(file, "%s serial=%llu channel=%d boundary_entry=1 command_entry=2 "
                       "address=%016llx boundary_header=%016llx command_header=%016llx "
                       "fetch_before=%llu fetch_after=%llu snapshot_fetch=%llu "
                       "suspensions=%llu snapshot_suspensions=%llu original_fetch=%llu "
                       "original_suspensions=%llu origin=%u refreshes=%llu completed_serial=%llu "
                       "observation_steady_ns=%llu cpu_copy_valid=%u parse_valid=%u "
                       "source=cpu_unsafe_no_flush boundary_gpu_dirty_not_queried=1\n",
                 prefix, static_cast<unsigned long long>(record.serial), record.channel,
                 static_cast<unsigned long long>(record.address),
                 static_cast<unsigned long long>(record.boundary_header),
                 static_cast<unsigned long long>(record.command_header),
                 static_cast<unsigned long long>(record.fetch_before),
                 static_cast<unsigned long long>(record.fetch_after),
                 static_cast<unsigned long long>(record.snapshot_fetch),
                 static_cast<unsigned long long>(record.suspensions),
                 static_cast<unsigned long long>(record.snapshot_suspensions),
                 static_cast<unsigned long long>(record.original_fetch),
                 static_cast<unsigned long long>(record.original_suspensions),
                 static_cast<unsigned>(record.origin),
                 static_cast<unsigned long long>(record.refresh_count),
                 static_cast<unsigned long long>(record.completed_serial),
                 static_cast<unsigned long long>(record.observation_steady_ns),
                 unsigned(record.cpu_copy_valid), unsigned(record.parse_valid));
    if (record.submit_identity) WriteBlitSubmitIdentity(file, prefix, record);
    WriteFaultWords(file, (std::string{prefix} + "_submitted").c_str(), record.submitted);
    if (record.cpu_copy_valid)
        WriteFaultWords(file, (std::string{prefix} + "_boundary_cpu").c_str(), record.at_boundary);
    if (record.parse_valid) {
        std::fprintf(file, "%s_parse_source=%s\n", prefix, record.parse_source);
        WriteReadDecision(file, (std::string{prefix} + "_parse_decision").c_str(), record.parse_decision);
        WriteFaultWords(file, (std::string{prefix} + "_parse").c_str(), record.at_parse);
    }
}

} // namespace

void DmaPusher::TraceSemaphoreRetirementGate(bool eligible, const char* reason,
                                               bool configured, u32 argument) const {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_SEM_RETIREMENT_GATE_TRACE");
        const char* split = std::getenv("EDEN_NX2_SPLIT");
        return value && std::string_view{value} == "1" && split && std::string_view{split} == "0";
    }();
    const auto& split = VideoCommon::GpuSplit::Get();
    if (!enabled || Settings::GetCurrentProgramID() != 0x0400486024DEC000ULL ||
        split.Active() || split.BackRunning() || VideoCommon::GpuSplit::OnBackThread()) return;
    static thread_local u32 reports{};
    static thread_local std::chrono::steady_clock::time_point next_report{};
    if (reports >= 64) return;
    const auto now = std::chrono::steady_clock::now();
    if (reports >= 8 && now < next_report) return;
    ++reports;
    next_report = now + std::chrono::seconds{5};

    const auto* context = current_pushbuffer_fault;
    const bool context_valid = context && &context->pusher == this;
    const size_t word_offset = context_valid ? context->word_offset : 0;
    const size_t parser_index = context_valid ? context->parser_index : 0;
    const size_t span_words = context_valid ? context->live_words.size() : 0;
    const bool position_valid = context_valid && parser_index < span_words &&
                                word_offset <= std::numeric_limits<size_t>::max() - parser_index;
    const size_t actual_word = position_valid ? word_offset + parser_index : 0;
    const bool span_end_valid = context_valid &&
                               word_offset <= std::numeric_limits<size_t>::max() - span_words;
    const size_t span_end = span_end_valid ? word_offset + span_words : 0;
    const bool before_valid = context_valid && parser_index < context->before_words.size();
    const u32 before = before_valid ? context->before_words[parser_index] : 0;
    const bool submitted_valid = context_valid && parser_index < context->submitted_words.size();
    const u32 submitted = submitted_valid ? context->submitted_words[parser_index] : 0;
    const char* source = context_valid ? context->source : "unavailable";

    bool entry_valid{};
    u64 entry_address{}, entry_words{}, prefetch_words{};
    if (!dma_pushbuffer.empty()) {
        const auto& list = dma_pushbuffer.front();
        prefetch_words = list.prefetch_command_list.size();
        if (prefetch_words == 0 && dma_pushbuffer_subindex < list.command_lists.size()) {
            const auto& header = list.command_lists[dma_pushbuffer_subindex];
            entry_address = header.addr;
            entry_words = header.size;
            entry_valid = true;
        }
    }
    const bool entry_position_valid = position_valid && span_end_valid && entry_valid &&
        std::string_view{source} != "prefetch" && context->initial_entry == dma_pushbuffer_subindex &&
        context->initial_state.dma_get == dma_state.dma_get && dma_state.dma_get == entry_address &&
        span_end == entry_words && actual_word < entry_words;
    const u64 trailing_words = entry_position_valid ? entry_words - actual_word - 1 : 0;
    bool header_valid{};
    u32 parsed_header{};
    size_t header_word{};
    if (context_valid) {
        const auto history = std::ranges::find_if(completed_pushbuffer_histories,
            [this](const auto& item) { return item.owner == this; });
        if (history != completed_pushbuffer_histories.end() && history->header_total != 0) {
            const auto& header = history->headers[(history->header_total - 1) % history->headers.size()];
            header_valid = header.channel == channel_state.bind_id &&
                header.entry == context->initial_entry && header.address == context->initial_state.dma_get &&
                header.word_offset == word_offset && header.span_words == span_words &&
                header.completed_entry_serial == history->total && header.parser_index <= parser_index;
            if (header_valid) {
                parsed_header = header.parsed;
                header_word = header.parser_index;
            }
        }
    }
    LOG_INFO(HW_GPU,
        "Semaphore retirement gate sample={} eligible={} reason={} configured={} title={:016x} "
        "backend={} async_gpu={} channel={} method={:#x} method_count={} argument={:08x} "
        "get={:#x} dma_byte_offset={:#x} entry={} entry_valid={} entry_address={:#x} "
        "entry_words={} prefetch_words={} context={} source={} word_offset={} parser_index={} "
        "span_words={} actual_word={} entry_position_valid={} trailing_words={} "
        "header_valid={} parsed_header={:08x} header_index={} before_valid={} before={:08x} "
        "submitted_valid={} submitted={:08x}",
        reports, eligible, reason, configured, Settings::GetCurrentProgramID(),
        static_cast<u32>(Settings::values.renderer_backend.GetValue()),
        Settings::values.use_asynchronous_gpu_emulation.GetValue(), channel_state.bind_id,
        dma_state.method, dma_state.method_count, argument, dma_state.dma_get, dma_state.dma_word_offset,
        dma_pushbuffer_subindex, entry_valid, entry_address, entry_words, prefetch_words,
        context_valid, source, word_offset, parser_index, span_words, actual_word,
        entry_position_valid, trailing_words, header_valid, parsed_header, header_word,
        before_valid, before, submitted_valid, submitted);
}

void DmaPusher::CaptureCurrentPushbufferFault(const char* reason,
                                            const MacroFaultDiagnostic* macro) {
    const char* path = PushbufferFaultPath();
    auto* context = current_pushbuffer_fault;
    if (path == nullptr || context == nullptr) {
        return;
    }
    static std::atomic<bool> captured{};
    if (captured.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    // Opt-in diagnostics must stop before later methods consume a known-invalid stream.
    const bool stop_on_first_fault = [] {
        const char* value = std::getenv("EDEN_NX2_PB_STOP_ON_FIRST_FAULT");
        return value != nullptr && std::string_view{value} == "1";
    }();
    auto& pusher = context->pusher;
    std::FILE* file = std::fopen(path, "wb");
    if (file == nullptr) {
        LOG_ERROR(HW_GPU, "First pushbuffer fault: {} (cannot open diagnostic {})", reason, path);
        if (stop_on_first_fault) std::_Exit(86);
        return;
    }
    const auto& state = pusher.dma_state;
    std::fprintf(file,
                 "EDEN_PB_FIRST_FAULT_V1\nreason=%s source=%s channel=%d\n"
                 "get=%016llx parser_word_offset=%zu parser_index=%zu span_words=%zu "
                 "copied_words=%zu\nmethod=%x subchannel=%u remaining=%u non_incrementing=%u "
                 "dma_byte_offset=%016llx resume_word_offset=%zu suspensions=%llu\n",
                 reason, context->source, pusher.channel_state.bind_id,
                 static_cast<unsigned long long>(state.dma_get), context->word_offset,
                 context->parser_index, context->live_words.size(), context->before_words.size(),
                 state.method, state.subchannel, state.method_count,
                 static_cast<unsigned>(state.non_incrementing),
                 static_cast<unsigned long long>(state.dma_word_offset),
                 pusher.CommandResumeOffsetForDiagnostics(),
                 static_cast<unsigned long long>(pusher.suspensions.load(std::memory_order_acquire)));
    WriteReadDecision(file, "read_decision", context->read_diagnostic);
    const auto& history = pushbuffer_segment_end_history;
    std::fprintf(file, "eligible_mode7_total=%llu retained=%zu\n",
                 static_cast<unsigned long long>(history.total),
                 static_cast<size_t>(std::min<u64>(history.total, history.events.size())));
    const u64 first_event = history.total > history.events.size() ?
                                history.total - history.events.size() : 0;
    for (u64 i = first_event; i < history.total; ++i) {
        const auto& event = history.events[i % history.events.size()];
        std::fprintf(file, "eligible_mode7[%llu] channel=%d entry=%zu address=%016llx "
                          "word=%zu raw=%08x\n", static_cast<unsigned long long>(i),
                     event.channel, event.entry, static_cast<unsigned long long>(event.address),
                     event.word, event.raw);
    }
    std::fprintf(file, "start_method=%x start_subchannel=%u start_remaining=%u "
                      "start_non_incrementing=%u\n", context->initial_state.method,
                 context->initial_state.subchannel, context->initial_state.method_count,
                 static_cast<unsigned>(context->initial_state.non_incrementing));
    std::fprintf(file, "parked_methods=%zu split_method_end=%016llx increment_once=%u\n",
                 pusher.parked_methods.size(),
                 static_cast<unsigned long long>(pusher.split_method_end),
                 static_cast<unsigned>(pusher.dma_increment_once));
    for (size_t i = 0; i < pusher.parked_methods.size() && i < 16; ++i) {
        const auto& parked = pusher.parked_methods[i];
        std::fprintf(file, "parked[%zu] method=%x subchannel=%u remaining=%u non_incrementing=%u "
                          "increment_once=%u resume=%016llx\n", i, parked.method,
                     parked.subchannel, parked.method_count,
                     static_cast<unsigned>(parked.non_incrementing),
                     static_cast<unsigned>(parked.increment_once),
                     static_cast<unsigned long long>(parked.resume_address));
    }
    if (!pusher.dma_pushbuffer.empty()) {
        const auto& list = pusher.dma_pushbuffer.front();
        std::fprintf(file, "entry_index=%zu entries=%zu prefetch_words=%zu snapshot_suspensions=%llu\n",
                     pusher.dma_pushbuffer_subindex, list.command_lists.size(),
                     list.prefetch_command_list.size(),
                     static_cast<unsigned long long>(list.snapshot_suspensions));
        const size_t first = pusher.dma_pushbuffer_subindex > 1 ?
                                 pusher.dma_pushbuffer_subindex - 1 : 0;
        for (size_t i = first; i < list.command_lists.size() && i < first + 3; ++i) {
            const auto& header = list.command_lists[i];
            std::fprintf(file, "entry[%zu] raw=%016llx address=%016llx words=%llu\n", i,
                         static_cast<unsigned long long>(header.raw),
                         static_cast<unsigned long long>(header.addr.Value()),
                         static_cast<unsigned long long>(header.size.Value()));
        }
    }
    for (const auto& completed_history : completed_pushbuffer_histories) {
        if (completed_history.owner != &pusher) {
            continue;
        }
        if (BlitBoundaryTraceEnabled()) {
            const u64 first_boundary = completed_history.blit_boundary_total >
                    completed_history.blit_boundaries.size()
                ? completed_history.blit_boundary_total - completed_history.blit_boundaries.size() : 0;
            std::fprintf(file, "blit_boundary_total=%llu retained=%llu first_difference=%u\n",
                         static_cast<unsigned long long>(completed_history.blit_boundary_total),
                         static_cast<unsigned long long>(completed_history.blit_boundary_total - first_boundary),
                         unsigned(completed_history.first_blit_boundary_difference.has_value()));
            for (u64 i = first_boundary; i < completed_history.blit_boundary_total; ++i)
                WriteBlitBoundaryObservation(file, ("blit_boundary_" + std::to_string(i)).c_str(),
                    completed_history.blit_boundaries[i % completed_history.blit_boundaries.size()]);
            if (completed_history.first_blit_boundary_difference)
                WriteBlitBoundaryObservation(file, "first_blit_boundary_difference",
                    *completed_history.first_blit_boundary_difference);
        }
        std::fprintf(file, "first_submitted_mismatch present=%u\n",
                     static_cast<unsigned>(completed_history.first_submitted_mismatch.has_value()));
        if (completed_history.first_submitted_mismatch) {
            const auto& mismatch = *completed_history.first_submitted_mismatch;
            std::fprintf(file, "mismatch_class=unsafe_non_gpu_unresumed channel=%d entry=%zu "
                              "address=%016llx word_offset=%zu total_words=%zu compared_words=%zu "
                              "first_different_word=%zu prefix_complete=%u completed_serial=%llu "
                              "observation_steady_ns=%llu clock=steady_clock epoch=implementation_defined "
                              "source=%s list_entries=%zu\n",
                         mismatch.channel, mismatch.entry,
                         static_cast<unsigned long long>(mismatch.address), mismatch.word_offset,
                         mismatch.total_words, mismatch.before_words.size(), mismatch.first_different_word,
                         static_cast<unsigned>(mismatch.before_words.size() == mismatch.total_words),
                         static_cast<unsigned long long>(mismatch.completed_entry_serial),
                         static_cast<unsigned long long>(mismatch.observation_steady_ns),
                         mismatch.source, mismatch.list_entries);
            std::fprintf(file, "mismatch_initial method=%x subchannel=%u remaining=%u "
                              "non_incrementing=%u increment_once=%u\n",
                         mismatch.initial_state.method, mismatch.initial_state.subchannel,
                         mismatch.initial_state.method_count,
                         static_cast<unsigned>(mismatch.initial_state.non_incrementing),
                         static_cast<unsigned>(mismatch.initial_increment_once));
            for (size_t i = 0; i < mismatch.neighbor_count; ++i) {
                std::fprintf(file, "mismatch_entry[%zu] raw=%016llx\n", mismatch.first_neighbor + i,
                             static_cast<unsigned long long>(mismatch.neighbor_headers[i]));
            }
            WriteReadDecision(file, "mismatch_read_decision", mismatch.read_diagnostic);
            WriteFaultWords(file, "mismatch_preparse_words", mismatch.before_words);
            WriteFaultWords(file, "mismatch_submitted_words", mismatch.submitted_words);
        }
        const u64 first_header = completed_history.header_total > completed_history.headers.size() ?
                                     completed_history.header_total - completed_history.headers.size() : 0;
        std::fprintf(file, "parsed_header_total=%llu retained=%llu\n",
                     static_cast<unsigned long long>(completed_history.header_total),
                     static_cast<unsigned long long>(completed_history.header_total - first_header));
        for (u64 i = first_header; i < completed_history.header_total; ++i) {
            const auto& header = completed_history.headers[i % completed_history.headers.size()];
            std::fprintf(file, "parsed_header[%llu] channel=%d entry=%zu address=%016llx "
                              "word_offset=%zu parser_index=%zu span_words=%zu completed_serial=%llu "
                              "fetch_generation=%llu suspensions=%llu "
                              "source=%s parsed=%08x before=%08x before_valid=%u "
                              "submitted=%08x submitted_valid=%u\n",
                         static_cast<unsigned long long>(i), header.channel, header.entry,
                         static_cast<unsigned long long>(header.address), header.word_offset,
                         header.parser_index, header.span_words,
                         static_cast<unsigned long long>(header.completed_entry_serial),
                         static_cast<unsigned long long>(header.fetch_generation),
                         static_cast<unsigned long long>(header.suspensions), header.source,
                         header.parsed, header.before, static_cast<unsigned>(header.before_valid),
                         header.submitted, static_cast<unsigned>(header.submitted_valid));
        }
        const u64 first = completed_history.total > completed_history.entries.size() ?
                              completed_history.total - completed_history.entries.size() : 0;
        for (u64 i = first; i < completed_history.total; ++i) {
            const auto& entry = completed_history.entries[i % completed_history.entries.size()];
            std::fprintf(file, "completed[%llu] channel=%d entry=%zu address=%016llx "
                              "word_offset=%zu total_words=%zu processed_words=%zu source=%s\n",
                         static_cast<unsigned long long>(i), entry.channel, entry.entry,
                         static_cast<unsigned long long>(entry.address), entry.word_offset,
                         entry.total_words, entry.processed_words, entry.source);
            std::fprintf(file, "completed_initial method=%x subchannel=%u remaining=%u "
                              "non_incrementing=%u increment_once=%u\n",
                         entry.initial_state.method, entry.initial_state.subchannel,
                         entry.initial_state.method_count,
                         static_cast<unsigned>(entry.initial_state.non_incrementing),
                         static_cast<unsigned>(entry.initial_increment_once));
            std::fprintf(file, "completed_final method=%x subchannel=%u remaining=%u "
                              "non_incrementing=%u increment_once=%u\n",
                         entry.final_state.method, entry.final_state.subchannel,
                         entry.final_state.method_count,
                         static_cast<unsigned>(entry.final_state.non_incrementing),
                         static_cast<unsigned>(entry.final_increment_once));
            const std::string prefix = "completed_" + std::to_string(i);
            WriteReadDecision(file, (prefix + "_read_decision").c_str(), entry.read_diagnostic);
            WriteFaultWords(file, (prefix + "_preparse_words").c_str(), entry.before_words);
            WriteFaultWords(file, (prefix + "_submitted_words").c_str(), entry.submitted_words);
            const size_t bytes = entry.before_words.size() * sizeof(u32);
            const GPUVAddr offset = entry.word_offset * sizeof(u32);
            const bool address_ok = entry.word_offset <= UINT64_MAX / sizeof(u32) &&
                                    offset <= UINT64_MAX - entry.address;
            const GPUVAddr address = address_ok ? entry.address + offset : 0;
            auto& memory = pusher.memory_manager;
            const bool mapped = bytes != 0 && address_ok &&
                                std::string_view{entry.source} != "prefetch" &&
                                bytes - 1 <= UINT64_MAX - address &&
                                memory.IsWithinGPUAddressRange(address) &&
                                memory.IsWithinGPUAddressRange(address + bytes - 1) &&
                                memory.IsFullyMappedRange(address, bytes);
            std::fprintf(file, "completed_raw_cpu_snapshot address=%016llx bytes=%zu mapped=%u "
                              "flushed=0\n", static_cast<unsigned long long>(address), bytes,
                         static_cast<unsigned>(mapped));
            if (mapped) {
                std::vector<u32> words(entry.before_words.size());
                memory.ReadBlockUnsafe(address, words.data(), bytes);
                WriteFaultWords(file, (prefix + "_raw_cpu_words").c_str(), words);
            }
        }
    }

    // The pre-parse copy and fault-time view are separate: UnsafeRead can be a live guest span.
    WriteFaultWords(file, "preparse_words", context->before_words);
    std::vector<u32> live_words;
    for (const auto& word : context->live_words.first(context->before_words.size())) {
        live_words.push_back(word.argument);
    }
    WriteFaultWords(file, "fault_span_words", live_words);
    WriteFaultWords(file, "submitted_words", context->submitted_words);

    const size_t bytes = context->before_words.size() * sizeof(u32);
    const GPUVAddr offset = context->word_offset * sizeof(u32);
    const bool address_ok = context->word_offset <= UINT64_MAX / sizeof(u32) &&
                            offset <= UINT64_MAX - state.dma_get;
    const GPUVAddr address = address_ok ? state.dma_get + offset : 0;
    auto& memory = pusher.memory_manager;
    const bool raw_mapped = bytes != 0 && address_ok &&
                            std::string_view{context->source} != "prefetch" &&
                            bytes - 1 <= UINT64_MAX - address &&
                            memory.IsWithinGPUAddressRange(address) &&
                            memory.IsWithinGPUAddressRange(address + bytes - 1) &&
                            memory.IsFullyMappedRange(address, bytes);
    std::fprintf(file, "raw_cpu_snapshot address=%016llx bytes=%zu mapped=%u flushed=0\n",
                 static_cast<unsigned long long>(address), bytes, static_cast<unsigned>(raw_mapped));
    if (raw_mapped) {
        std::vector<u32> raw_words(context->before_words.size());
        memory.ReadBlockUnsafe(address, raw_words.data(), bytes);
        WriteFaultWords(file, "raw_cpu_words", raw_words);
    }
    if (macro != nullptr) {
        std::fprintf(file,
                     "macro_start=%x pc=%x consumed=%zu code_generation=%llu "
                     "mem_address_a=%08x mem_address_b=%08x data_ram_address=%08x\n",
                     macro->start, macro->pc, macro->consumed,
                     static_cast<unsigned long long>(macro->code_generation),
                     macro->memory_registers[0], macro->memory_registers[1],
                     macro->memory_registers[2]);
        WriteFaultWords(file, "macro_parameters", macro->parameters);
        WriteFaultWords(file, "macro_fifo", macro->fifo);
        WriteFaultWords(file, "macro_code", macro->code);
        WriteFaultWords(file, "macro_data_ram", macro->data_ram);
    }
    const bool write_failed = std::ferror(file) != 0;
    const bool close_failed = std::fclose(file) != 0;
    LOG_ERROR(HW_GPU, "First pushbuffer fault: {} channel {} entry {} word {} captured to {} "
                      "(write_failed={})", reason, pusher.channel_state.bind_id,
              pusher.dma_pushbuffer_subindex, context->word_offset + context->parser_index,
              path, write_failed || close_failed);
    // fclose above flushes and closes the artifact before this deliberate diagnostic exit.
    if (stop_on_first_fault) std::_Exit(86);
}

bool DmaPusher::CommandSourceMatches(const CommandSource& source) const {
    if (dma_pushbuffer.empty() || source.list != &dma_pushbuffer.front() ||
        source.entry != dma_pushbuffer_subindex) {
        return false;
    }
    const auto& list = dma_pushbuffer.front();
    if (!list.prefetch_command_list.empty() || source.entry >= list.command_lists.size()) {
        return false;
    }
    const auto& header = list.command_lists[source.entry];
    if (header.raw != source.header || header.size == 0 ||
        dma_state.dma_get != header.addr || source.words.size() != header.size ||
        source.words.data() == nullptr) {
        return false;
    }
    switch (source.storage) {
    case CommandSourceStorage::Submitted: {
        if (source.entry >= list.submitted_offsets.size()) {
            return false;
        }
        const size_t offset = list.submitted_offsets[source.entry];
        return offset <= list.submitted_words.size() &&
               source.words.size() <= list.submitted_words.size() - offset &&
               source.words.data() == list.submitted_words.data() + offset;
    }
    case CommandSourceStorage::Scratch:
        return source.words.data() == command_headers.data() &&
               source.words.size() <= command_headers.size();
    case CommandSourceStorage::Physical:
        // GuestMemory borrows the permanent physical DRAM allocation. GVA mappings may
        // change meanwhile; translating again would select a different command source.
        return true;
    }
    return false;
}

bool DmaPusher::CanRetainCommandSourceForRetirement() const {
    if (!active_command_source || !CommandSourceMatches(*active_command_source) ||
        dma_state.dma_word_offset % sizeof(u32) != 0) {
        return false;
    }
    const u64 word = dma_state.dma_word_offset / sizeof(u32);
    return word >= active_command_source->first_word &&
           word < active_command_source->words.size();
}

size_t DmaPusher::ProcessCommandSource(std::span<const CommandHeader> commands, size_t word_offset,
                                      const char* source, CommandSourceStorage storage,
                                      bool retirement_resume) {
    const CommandSource selected{
        commands, &dma_pushbuffer.front(), dma_pushbuffer_subindex,
        dma_pushbuffer.front().command_lists[dma_pushbuffer_subindex].raw,
        word_offset, source, storage,
    };
    const auto* previous = active_command_source;
    active_command_source = &selected;
    bool retain = false;
    SCOPE_EXIT {
        active_command_source = previous;
        if (!retain) {
            retained_command_source.reset();
        }
    };
    const char* parser_source = source;
    if (retirement_resume) {
        command_read_diagnostic.resume_offset = word_offset;
        if (std::string_view{source} == "submitted") {
            parser_source = "retained-submitted";
        } else if (std::string_view{source} == "safe") {
            parser_source = "retained-safe";
        } else if (std::string_view{source} == "gpu-page-copy") {
            parser_source = "retained-gpu-page-copy";
        } else {
            parser_source = "retained-unsafe-after-flush";
        }
    }
    const size_t processed = ProcessCommands(commands.subspan(word_offset), word_offset, parser_source);
    if (const auto* pending = puller.PendingAcquireForDiagnostics();
        pending && pending->retirement && !pending->semaphore_suspended) {
        retained_command_source = selected;
        retain = true;
    }
    return processed;
}

size_t DmaPusher::ProcessCommands(std::span<const CommandHeader> commands, size_t word_offset,
                                  const char* source) {
    std::unique_ptr<PushbufferFaultContext> fault;
    if (PushbufferFaultPath() != nullptr) [[unlikely]] {
        VideoCommon::Nx2Profile::Scope diagnostic_scope{VideoCommon::Nx2Profile::Zone::CommandDiagnostic};
        fault = std::make_unique<PushbufferFaultContext>(*this, commands, word_offset, source, static_cast<bool>(dma_increment_once));
    }
    for (size_t index = 0; index < commands.size();) {
        if (fault) [[unlikely]] {
            fault->parser_index = index;
        }
        // Data word of methods command
        if (dma_state.method_count && dma_state.non_incrementing) {
            auto const& command_header = commands[index]; //must ref (MUltiMethod re)
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            u32 max_write = u32(std::min<std::size_t>(index + dma_state.method_count, commands.size()) - index);
            if (g_nx2_ampere && dma_state.method < non_puller_methods) {
                max_write = 1; // a semaphore acquire may suspend the channel after any word
            }
            CallMultiMethod(&command_header.argument, max_write);
            dma_state.method_count -= max_write;
            dma_state.is_last_call = true;
            index += max_write;
        } else if (dma_state.method_count) {
            // Fast-path batching for consecutive non-execution methods in increasing mode
            if (!dma_state.non_incrementing && !dma_increment_once &&
                dma_state.method >= non_puller_methods &&
                !VideoCommon::GpuSplit::Get().Active() &&
                g_semaphore_watch == 0) {
                auto subchannel = subchannels[dma_state.subchannel];
                if (subchannel != nullptr) {
                    const size_t available = std::min<size_t>(dma_state.method_count, commands.size() - index);
                    u32 batch_count = 0;
                    while (batch_count < available &&
                           (dma_state.method + batch_count) < subchannel->execution_mask.size() &&
                           !subchannel->execution_mask[dma_state.method + batch_count]) {
                        ++batch_count;
                    }
                    if (batch_count > 1) {
                        for (u32 i = 0; i < batch_count; ++i) {
                            subchannel->method_sink.emplace_back(dma_state.method + i, commands[index + i].argument);
                        }
                        dma_state.method += batch_count;
                        dma_state.method_count -= batch_count;
                        index += batch_count;
                        continue;
                    }
                }
            }

            auto const command_header = commands[index]; //can copy
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            dma_state.is_last_call = dma_state.method_count <= 1;
            CallMethod(command_header.argument);
            dma_state.method += !dma_state.non_incrementing ? 1 : 0;
            dma_state.non_incrementing |= dma_increment_once;
            dma_state.method_count--;
            index++;
        } else {
            auto const command_header = commands[index]; //can copy
            // No command active - this is the first word of a new one
            if (fault) {
                fault->NoteHeader(index, command_header.argument);
            }
            if (fault && (command_header.argument >> 29) == 7) [[unlikely]] {
                fault->NoteSegmentEndCandidate(index, command_header.argument);
            }
            switch (command_header.mode) {
            case SubmissionMode::Increasing:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = false;
                break;
            case SubmissionMode::NonIncreasing:
                SetState(command_header);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::Inline:
                dma_state.method = command_header.method;
                dma_state.subchannel = command_header.subchannel;
                dma_state.dma_word_offset = u64(-s64(dma_state.dma_get)); // negate to set address as 0
                CallMethod(command_header.arg_count);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::IncreaseOnce:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = true;
                break;
            default:
                break;
            }
            index++;
        }
        if (puller.HasPendingAcquire()) {
            if (fault) {
                fault->processed_words = index;
                fault->final_increment_once = dma_increment_once;
                fault->returned = true;
            }
            return index;
        }
    }
    if (fault) {
        fault->processed_words = commands.size();
        fault->final_increment_once = dma_increment_once;
        fault->returned = true;
    }
    return commands.size();
}

// NVN on Switch 2 ends GPFIFO entries in the middle of a method's data and calls another command
// buffer (e.g. a compute dispatch) before the entry that carries the rest, which starts right
// where the first one ended. An entry starting anywhere else is such a call: run it with a clean
// method state and resume the unfinished method when the stream continues at that address.
void DmaPusher::ResolveSplitMethod(GPUVAddr entry_address) {
    const auto continued = std::ranges::find(parked_methods, entry_address,
                                             &ParkedMethod::resume_address);
    std::optional<ParkedMethod> resume;
    if (continued != parked_methods.end()) {
        resume = *continued;
        parked_methods.erase(continued);
    }
    if (dma_state.method_count != 0 && entry_address != split_method_end) {
        if (parked_methods.size() == MaxParkedMethods) {
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1) < 8) {
                const auto& oldest = parked_methods.front();
                LOG_WARNING(HW_GPU,
                            "Dropping parked method {:#x} (sub {}, {} words left, resume {:#x})",
                            oldest.method, oldest.subchannel, oldest.method_count,
                            oldest.resume_address);
            }
            parked_methods.erase(parked_methods.begin());
        }
        parked_methods.push_back(ParkedMethod{
            .method = dma_state.method,
            .subchannel = dma_state.subchannel,
            .method_count = dma_state.method_count,
            .non_incrementing = dma_state.non_incrementing,
            .increment_once = dma_increment_once,
            .resume_address = split_method_end,
        });
        dma_state.method_count = 0;
        dma_increment_once = false;
    }
    if (resume && dma_state.method_count == 0) {
        dma_state.method = resume->method;
        dma_state.subchannel = resume->subchannel;
        dma_state.method_count = resume->method_count;
        dma_state.non_incrementing = resume->non_incrementing;
        dma_increment_once = resume->increment_once;
    }
}

void DmaPusher::SetState(const CommandHeader& command_header) {
    dma_state.method = command_header.method;
    dma_state.subchannel = command_header.subchannel;
    dma_state.method_count = command_header.method_count;
}


void DmaPusher::WatchArgument(u32 method, u32 argument) const {
    const u32 watch = g_semaphore_watch;
    if ((argument & ~0xfU) == watch) {
        LOG_INFO(HW_GPU, "Semaphore watch: channel {} subchannel {} method {:#x} argument {:#x}",
                 channel_state.bind_id, dma_state.subchannel, method, argument);
    }
}

bool DmaPusher::WantsSegment() const {
    // Only macro parameters and compute uploads read the engine's current_dma_segment.
    const auto engine = subchannel_type[dma_state.subchannel];
    return engine == Engines::EngineTypes::KeplerCompute ||
           (engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart);
}

void DmaPusher::CallMethod(u32 argument) {
    if (g_semaphore_watch != 0) [[unlikely]] {
        WatchArgument(dma_state.method, argument);
    }
    auto& split = VideoCommon::GpuSplit::Get();
    if (dma_state.method < non_puller_methods) {
        puller.CallPullerMethod(*this, Engines::Puller::MethodCall{
            dma_state.method,
            argument,
            dma_state.subchannel,
            dma_state.method_count,
        });
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        if (subchannel == nullptr) {
            ReportUnboundSubchannel();
            return;
        }
        // Compute dispatches and DMA copies write the command lists ReadGpuWrittenCommands reads.
        const auto engine_type = subchannel_type[dma_state.subchannel];
        const bool writer = engine_type == Engines::EngineTypes::KeplerCompute ||
                            engine_type == Engines::EngineTypes::MaxwellDMA;
        SCOPE_EXIT {
            if (writer) {
                split.NoteWriter();
            }
        };
        if (split.Active()) {
            const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
            if (split.FrontMacros()) {
                const auto type = subchannel_type[dma_state.subchannel];
                if (type == Engines::EngineTypes::Maxwell3D) {
                    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannel);
                    if (maxwell3d->GetNativeMacroEngine().native_turing) {
                        maxwell3d->FrontCallMethod(system, split, dma_state.method, argument,
                                                   dma_state.is_last_call);
                        return;
                    }
                } else if (type == Engines::EngineTypes::KeplerCompute) {
                    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannel);
                    if (compute->FrontRunsMacros()) {
                        compute->FrontCallMethod(system, split, dma_state.method, argument,
                                                 dma_state.is_last_call, &segment);
                        return;
                    }
                }
            }
            split.Method(subchannel, dma_state.method, argument, dma_state.is_last_call,
                         WantsSegment() ? &segment : nullptr);
            return;
        }
        if (!subchannel->execution_mask[dma_state.method]) {
            subchannel->method_sink.emplace_back(dma_state.method, argument);
        } else {
            VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
            subchannel->ConsumeSink(system);
            subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
            subchannel->CallMethod(system, dma_state.method, argument, dma_state.is_last_call);
        }
    }
}

void DmaPusher::CallMultiMethod(const u32* base_start, u32 num_methods) {
    if (g_semaphore_watch != 0) [[unlikely]] {
        for (u32 i = 0; i < num_methods; ++i) {
            WatchArgument(dma_state.method + (dma_state.non_incrementing ? 0 : i), base_start[i]);
        }
    }
    auto& split = VideoCommon::GpuSplit::Get();
    if (dma_state.method < non_puller_methods) {
        puller.CallMultiMethod(*this, dma_state.method, dma_state.subchannel, base_start, num_methods, dma_state.method_count);
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        if (subchannel == nullptr) {
            ReportUnboundSubchannel();
            return;
        }
        // Compute dispatches and DMA copies write the command lists ReadGpuWrittenCommands reads.
        const auto engine_type = subchannel_type[dma_state.subchannel];
        const bool writer = engine_type == Engines::EngineTypes::KeplerCompute ||
                            engine_type == Engines::EngineTypes::MaxwellDMA;
        SCOPE_EXIT {
            if (writer) {
                split.NoteWriter();
            }
        };
        if (split.Active()) {
            const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
            if (split.FrontMacros()) {
                const auto type = subchannel_type[dma_state.subchannel];
                if (type == Engines::EngineTypes::Maxwell3D) {
                    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannel);
                    if (maxwell3d->GetNativeMacroEngine().native_turing) {
                        maxwell3d->FrontCallMultiMethod(system, split, dma_state.method, base_start,
                                                        num_methods, dma_state.method_count);
                        return;
                    }
                } else if (type == Engines::EngineTypes::KeplerCompute) {
                    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannel);
                    if (compute->FrontRunsMacros()) {
                        compute->FrontCallMultiMethod(system, split, dma_state.method, base_start,
                                                      num_methods, dma_state.method_count, &segment);
                        return;
                    }
                }
            }
            split.MultiMethod(subchannel, dma_state.method, base_start, num_methods,
                              dma_state.method_count, WantsSegment() ? &segment : nullptr);
            return;
        }
        VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
        subchannel->ConsumeSink(system);
        subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
        subchannel->CallMultiMethod(system, dma_state.method, base_start, num_methods, dma_state.method_count);
    }
}

void DmaPusher::RefreshSnapshot(CommandList& list, u64 now, CommandSnapshotOrigin origin) {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::CommandRefresh};
    list.snapshot_origin = origin;
    ++list.snapshot_refresh_count;
    list.snapshot_suspensions = now;
    list.snapshot_fetch_generation = fetch_generation.load(std::memory_order_acquire);
    for (size_t i = 0; i < list.command_lists.size() && i < list.submitted_offsets.size(); ++i) {
        const CommandListHeader& entry = list.command_lists[i];
        const u32 offset = list.submitted_offsets[i];
        if (offset + entry.size > list.submitted_words.size()) {
            continue;
        }
        memory_manager.ReadBlockUnsafe(entry.addr, list.submitted_words.data() + offset,
                                       entry.size * sizeof(u32));
    }
}

bool DmaPusher::IsGpuWrittenCommands(GPUVAddr address, size_t size) {
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || size == 0) {
        return false;
    }
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < address + size; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsGpuWritten(*device)) {
            return true;
        }
    }
    return false;
}

namespace {
// Waiting for queued writers before every entry only matters the first time GPU work writes a
// page (its bit in Nx2IsGpuWritten never clears afterwards). OoT waited ~1,800 times a second,
// ~0.3 s of the front's time, for its CPU-written command ring. Pages checked this often with
// no GPU write skip the wait; a page GPU work writes later is still caught by its bit.
// EDEN_NX2_TRUST_CPU_COMMAND_PAGES=1 trusts after 128 checks; a larger number sets the count.
u32 TrustedCpuCommandChecks() {
    static const u32 checks = [] {
        const char* value = std::getenv("EDEN_NX2_TRUST_CPU_COMMAND_PAGES");
        const u32 parsed = value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0U;
        return parsed == 1 ? 128U : parsed;
    }();
    return checks;
}

bool TrustCpuCommandPages() {
    return TrustedCpuCommandChecks() != 0;
}
} // Anonymous namespace

bool DmaPusher::IsTrustedCpuCommands(GPUVAddr address, size_t size) {
    if (!TrustCpuCommandPages() || size == 0) {
        return false;
    }
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < address + size; page += 0x1000) {
        const auto device = memory_manager.GpuToCpuAddress(page);
        if (!device) {
            return false;
        }
        const auto it = cpu_command_page_checks.find(*device & ~DAddr{0xfff});
        if (it == cpu_command_page_checks.end() || it->second < TrustedCpuCommandChecks()) {
            return false;
        }
    }
    return true;
}

void DmaPusher::NoteCpuCommands(GPUVAddr address, size_t size) {
    if (!TrustCpuCommandPages()) {
        return;
    }
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < address + size; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page)) {
            u32& checks = cpu_command_page_checks[*device & ~DAddr{0xfff}];
            if (checks < TrustedCpuCommandChecks()) {
                ++checks;
            }
        }
    }
}

bool DmaPusher::ReadGpuWrittenCommands(GPUVAddr address, u32 count) {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::CommandRead};
    // The pages are marked as command list pages: from the next write on, the buffer cache copies
    // them to host memory right after the dispatch or copy that writes them and submits. The
    // front then waits only for the back to run that work and for the GPU to finish the copy, not
    // for everything queued and a cache flush (FlushGpuWrittenCommands: ~400 GPU round trips a
    // second in a Mario Kart World race, during which the back had nothing to do). Pages
    // FlushGpuWrittenCommands flushed whole and no GPU work wrote since are read from guest
    // memory.
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || count == 0 ||
        rasterizer == nullptr) {
        return false;
    }
    const GPUVAddr end = address + u64{count} * sizeof(u32);
    bool written = false;
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < end; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsGpuWritten(*device)) {
            Tegra::Nx2MarkCommandListPage(*device);
            written = true;
        }
    }
    if (!written) {
        return false;
    }
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.WaitForWriters();
    }
    using GpuWrittenList = VideoCore::RasterizerInterface::GpuWrittenList;
    command_headers.resize_destructive(count);
    u8* const out = reinterpret_cast<u8*>(command_headers.data());
    for (GPUVAddr chunk = address; chunk < end;) {
        const GPUVAddr chunk_end = (std::min)(end, (chunk & ~GPUVAddr{0xfff}) + 0x1000);
        const std::span<u8> part{out + (chunk - address), static_cast<size_t>(chunk_end - chunk)};
        const auto device = memory_manager.GpuToCpuAddress(chunk);
        if (!device) {
            return false;
        }
        GpuWrittenList where = GpuWrittenList::Synced;
        if (Tegra::Nx2IsGpuWritten(*device)) {
            where = rasterizer->ReadGpuWrittenCommandList(*device, part);
        }
        if (where == GpuWrittenList::Unknown) {
            return false;
        }
        if (where == GpuWrittenList::Synced) {
            memory_manager.ReadBlockUnsafe(chunk, part.data(), part.size());
        }
        chunk = chunk_end;
    }
    return true;
}

void DmaPusher::FlushGpuWrittenCommands(GPUVAddr address, size_t size) {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::CommandFlush};
    // Title 1 runs small command lists its culling dispatches fill in each frame. Accurate GPU
    // mode reads every command list through the caches (safe reads); here only the ones on pages
    // the GPU has written are, or their draws run with an older frame's words (trees vanish).
    // These are the lists ReadGpuWrittenCommands can't take from a copy or guest memory (a page's
    // first list, or pages draws write too).
    if (!IsGpuWrittenCommands(address, size)) {
        return;
    }
    // The caches belong to the split's back: it first runs what the front queued (the dispatch
    // that writes these words among it). The whole pages are flushed: their next lists read
    // guest memory until GPU work writes them again.
    const GPUVAddr first_page = address & ~GPUVAddr{0xfff};
    const GPUVAddr end_page = (address + size + 0xfff) & ~GPUVAddr{0xfff};
    const auto flush = [memory = &memory_manager, first_page, end_page] {
        if (memory->IsMemoryDirty(first_page, end_page - first_page,
                                  VideoCommon::CacheType::BufferCache)) {
            VideoCommon::Nx2Profile::WaitTag wait_tag{VideoCommon::Nx2Profile::WaitSource::CommandPages};
            memory->FlushRegion(first_page, end_page - first_page,
                                VideoCommon::CacheType::BufferCache);
        }
    };
    // Run the flush on the back too: a drained back still runs its idle hook (a deferred
    // Scheduler::Flush), which swapped the command chunk under the front's download (OoT
    // crashed in Scheduler::SubmitExecution).
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.Defer(flush);
        split.Drain();
    } else {
        flush();
    }
    if (rasterizer == nullptr) {
        return;
    }
    for (GPUVAddr page = first_page; page < end_page; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsCommandListPage(*device)) {
            rasterizer->NoteGpuWrittenCommandListSynced(*device);
        }
    }
}

void DmaPusher::ReportUnboundSubchannel() {
    CaptureCurrentPushbufferFault("unbound-subchannel");
    // Methods sent to a subchannel with no engine (no SetObject yet) are dropped.
    static std::atomic<u32> reported{};
    if (reported.fetch_add(1) < 8) {
        LOG_WARNING(HW_GPU, "Dropping method {:#x} on unbound subchannel {}", dma_state.method,
                    dma_state.subchannel);
    }
}

void DmaPusher::CheckSubmitHash(const CommandListHeader& header, u64 submit_hash) {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::CommandHash};
    std::vector<u32> words(header.size);
    memory_manager.ReadBlockUnsafe(header.addr, words.data(), words.size() * sizeof(u32));
    static std::atomic<u64> checked{};
    static std::atomic<u64> changed{};
    ++checked;
    if (HashPushbufferWords(words) != submit_hash) {
        if (changed.fetch_add(1) < 16) {
            LOG_WARNING(HW_GPU, "pushbuffer entry {:#x}+{:#x} changed between submit and execution",
                        header.addr.Value(), header.size.Value());
        }
    }
    if ((checked % 20000) == 0) {
        LOG_WARNING(HW_GPU, "pushbuffer check: {} of {} entries changed after submit",
                    changed.load(), checked.load());
    }
}

namespace {
// EDEN_NX2_LIST_REPEAT=1: how much of the command stream repeats, as a whole entry (the same
// words seen before) or by layout only (the same methods, other values), logged every 10 s.
class ListRepeatStats {
public:
    void Note(std::span<const CommandHeader> commands) {
        u64 content = 0xcbf29ce484222325ULL;
        u64 shape = content;
        for (size_t i = 0; i < commands.size();) {
            const CommandHeader command = commands[i];
            shape = (shape ^ command.argument) * 0x100000001b3ULL;
            size_t args = 0;
            switch (command.mode) {
            case SubmissionMode::Increasing:
            case SubmissionMode::NonIncreasing:
            case SubmissionMode::IncreaseOnce:
                args = command.method_count;
                break;
            default:
                break;
            }
            i += 1 + args;
        }
        for (const CommandHeader& command : commands) {
            content = (content ^ command.argument) * 0x100000001b3ULL;
        }
        content ^= commands.size();
        shape ^= commands.size() << 32;

        std::scoped_lock lock{mutex};
        const u64 words = commands.size();
        total_entries++;
        total_words += words;
        if (seen_content.size() > (1U << 20) || last_by_shape.size() > (1U << 16)) {
            seen_content.clear();
            last_by_shape.clear();
        }
        if (!seen_content.insert(content).second) {
            exact_words += words;
        } else if (auto it = last_by_shape.find(shape);
                   it != last_by_shape.end() && it->second.size() == words) {
            shape_words += words;
            for (size_t i = 0; i < words; ++i) {
                changed_words += it->second[i] != commands[i].argument;
            }
        } else {
            new_words += words;
        }
        auto& last = last_by_shape[shape];
        last.resize(words);
        for (size_t i = 0; i < words; ++i) {
            last[i] = commands[i].argument;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_report < std::chrono::seconds{10}) {
            return;
        }
        if (last_report.time_since_epoch().count() != 0 && total_words != 0) {
            const double scale = 100.0 / static_cast<double>(total_words);
            LOG_WARNING(HW_GPU,
                        "list repeat: {} entries, {} words; exact repeat {:.1f}%, same layout "
                        "{:.1f}% ({:.1f}% of those words changed), new {:.1f}%",
                        total_entries, total_words, exact_words * scale, shape_words * scale,
                        shape_words ? 100.0 * changed_words / shape_words : 0.0,
                        new_words * scale);
        }
        last_report = now;
        total_entries = total_words = exact_words = shape_words = changed_words = new_words = 0;
    }

private:
    std::mutex mutex;
    std::unordered_set<u64> seen_content;
    std::unordered_map<u64, std::vector<u32>> last_by_shape;
    std::chrono::steady_clock::time_point last_report{};
    u64 total_entries{}, total_words{}, exact_words{}, shape_words{}, changed_words{}, new_words{};
};
} // namespace

void DmaPusher::RecordParsedEntry(const CommandListHeader& header,
                                  std::span<const CommandHeader> commands) {
    static const bool list_repeat = std::getenv("EDEN_NX2_LIST_REPEAT") != nullptr;
    if (list_repeat) {
        static ListRepeatStats stats;
        stats.Note(commands);
    }
    if (!g_record_parsed_entries) {
        return;
    }
    u64 hash = 0xcbf29ce484222325ULL;
    for (const CommandHeader& command : commands) {
        hash = (hash ^ command.argument) * 0x100000001b3ULL;
    }
    parsed_entries[parsed_entry_count++ % parsed_entries.size()] = {header.addr, static_cast<u32>(header.size), hash};
}

void DmaPusher::DumpCurrentEntry(const char* reason) {
    // Entries whose memory changed after they were parsed (the CPU rewrote them meanwhile).
    for (size_t i = 0; i < std::min(parsed_entry_count, parsed_entries.size()); ++i) {
        const auto& entry = parsed_entries[(parsed_entry_count - 1 - i) % parsed_entries.size()];
        std::vector<u32> words(entry.size);
        memory_manager.ReadBlockUnsafe(entry.address, words.data(), words.size() * sizeof(u32));
        u64 hash = 0xcbf29ce484222325ULL;
        for (const u32 word : words) {
            hash = (hash ^ word) * 0x100000001b3ULL;
        }
        LOG_WARNING(HW_GPU, "  parsed -{}: {:#x}+{:#x} {}", i, entry.address, entry.size,
                    hash == entry.hash ? "unchanged" : "CHANGED since parsed");
    }
    // EDEN_NX2_PB_DUMP=<file>: also write the whole command list (u64 address, u32 size and the
    // current words of every entry, then the current entry index) for offline replay.
    if (static const char* const path = std::getenv("EDEN_NX2_PB_DUMP");
        path != nullptr && !dma_pushbuffer.empty()) {
        if (std::FILE* file = std::fopen(path, "wb")) {
            const auto& lists = dma_pushbuffer.front().command_lists;
            const u64 current = dma_pushbuffer_subindex;
            const u64 count = lists.size();
            std::fwrite(&count, sizeof(count), 1, file);
            std::fwrite(&current, sizeof(current), 1, file);
            for (const CommandListHeader& header : lists) {
                const u64 address = header.addr;
                const u32 size = static_cast<u32>(header.size);
                std::vector<u32> words(size);
                memory_manager.ReadBlockUnsafe(address, words.data(), words.size() * sizeof(u32));
                std::fwrite(&address, sizeof(address), 1, file);
                std::fwrite(&size, sizeof(size), 1, file);
                std::fwrite(words.data(), sizeof(u32), words.size(), file);
            }
            std::fclose(file);
        }
    }
    LOG_WARNING(HW_GPU, "pushbuffer dump ({}): method={:#x} sub={} count={} non_inc={} get={:#x}",
                reason, dma_state.method, dma_state.subchannel, dma_state.method_count,
                dma_state.non_incrementing, dma_state.dma_get);
    if (dma_pushbuffer.empty()) {
        return;
    }
    const auto& lists = dma_pushbuffer.front().command_lists;
    const size_t first = dma_pushbuffer_subindex >= 2 ? dma_pushbuffer_subindex - 2 : 0;
    for (size_t i = first; i <= dma_pushbuffer_subindex && i < lists.size(); ++i) {
        const CommandListHeader& header = lists[i];
        LOG_WARNING(HW_GPU, "  entry {} of {}: addr={:#x} size={:#x}", i, lists.size(),
                    header.addr.Value(), header.size.Value());
        for (u64 word = 0; word < std::min<u64>(header.size, 0x80); word += 8) {
            std::string gpu_line;
            std::string cpu_line;
            for (u64 j = word; j < std::min<u64>(header.size, word + 8); ++j) {
                const GPUVAddr address = header.addr + j * sizeof(u32);
                const u32 gpu_value = memory_manager.Read<u32>(address);
                const u32 cpu_value = system.ApplicationMemory().Read32(address);
                gpu_line += fmt::format(" {:08x}", gpu_value);
                cpu_line += gpu_value == cpu_value ? std::string(" ........")
                                                   : fmt::format(" {:08x}", cpu_value);
            }
            LOG_WARNING(HW_GPU, "    +{:03x}:{}", word * 4, gpu_line);
            LOG_WARNING(HW_GPU, "    cpu :{}", cpu_line);
        }
    }
}

void DmaPusher::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

} // namespace Tegra
