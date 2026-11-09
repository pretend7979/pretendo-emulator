// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <optional>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <deque>
#include <queue>

#include "common/bit_field.h"
#include "common/common_types.h"
#include "common/scratch_buffer.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/engines/puller.h"

namespace Core {
class System;
}

namespace Tegra {

namespace Control {
struct ChannelState;
}

class GPU;
class MemoryManager;

enum class SubmissionMode : u32 {
    IncreasingOld = 0,
    Increasing = 1,
    NonIncreasingOld = 2,
    NonIncreasing = 3,
    Inline = 4,
    IncreaseOnce = 5
};

// Note that, traditionally, methods are treated as 4-byte addressable locations, and hence
// their numbers are written down multiplied by 4 in Docs. Here we are not multiply by 4.
// So the values you see in docs might be multiplied by 4.
// Register documentation:
// https://github.com/NVIDIA/open-gpu-doc/blob/ab27fc22db5de0d02a4cabe08e555663b62db4d4/classes/host/cla26f.h
//
// Register Description (approx):
// https://github.com/NVIDIA/open-gpu-doc/blob/ab27fc22db5de0d02a4cabe08e555663b62db4d4/manuals/volta/gv100/dev_pbdma.ref.txt
enum class BufferMethods : u32 {
    BindObject = 0x0,
    Illegal = 0x1,
    Nop = 0x2,
    SemaphoreAddressHigh = 0x4,
    SemaphoreAddressLow = 0x5,
    SemaphoreSequencePayload = 0x6,
    SemaphoreOperation = 0x7,
    NonStallInterrupt = 0x8,
    WrcacheFlush = 0x9,
    MemOpA = 0xA,
    MemOpB = 0xB,
    MemOpC = 0xC,
    MemOpD = 0xD,
    RefCnt = 0x14,
    SemaphoreAcquire = 0x1A,
    SemaphoreRelease = 0x1B,
    SyncpointPayload = 0x1C,
    SyncpointOperation = 0x1D,
    WaitForIdle = 0x1E,
    CRCCheck = 0x1F,
    Yield = 0x20,
    NonPullerMethods = 0x40,
};

struct CommandListHeader {
    union {
        u64 raw;
        BitField<0, 40, GPUVAddr> addr;
        BitField<40, 1, u64> allow_flush;
        BitField<41, 1, u64> is_push_buffer;
        BitField<42, 21, u64> size;
        BitField<63, 1, u64> sync;
    };
};
static_assert(sizeof(CommandListHeader) == sizeof(u64), "CommandListHeader is incorrect size");

union CommandHeader {
    u32 argument;
    BitField<0, 13, u32> method;
    BitField<0, 24, u32> method_count_;
    BitField<13, 3, u32> subchannel;
    BitField<16, 13, u32> arg_count;
    BitField<16, 13, u32> method_count;
    BitField<29, 3, SubmissionMode> mode;
};
static_assert(std::is_standard_layout_v<CommandHeader>, "CommandHeader is not standard layout");
static_assert(sizeof(CommandHeader) == sizeof(u32), "CommandHeader has incorrect size!");

inline CommandHeader BuildCommandHeader(BufferMethods method, u32 arg_count, SubmissionMode mode) {
    CommandHeader result{};
    result.method.Assign(static_cast<u32>(method));
    result.arg_count.Assign(arg_count);
    result.mode.Assign(mode);
    return result;
}

enum class CommandSnapshotOrigin : u32 { Unknown, Submit, PushRefresh, AcquireRefresh };

// Immutable inputs to the short-circuit snapshot-selection decision, for opt-in PB diagnostics.
struct CommandReadDiagnostic {
    bool valid{}, selected{}, use_safe{}, gpu_written{}, dirty_checked{}, dirty{}, sync_wait{};
    u32 checked{}, rejected{};
    u64 resume_offset{}, entry_index{}, snapshot_fetch{}, compared_fetch{};
    u64 snapshot_suspensions{}, compared_suspensions{};
    u64 original_fetch{}, original_suspensions{}, refresh_count{};
    CommandSnapshotOrigin origin{};
};

struct BlitSubmitObservation;

struct CommandList final {
    CommandList() = default;
    explicit CommandList(std::size_t size) : command_lists(size) {}
    explicit CommandList(boost::container::small_vector<CommandHeader, 512>&& prefetch_command_list_)
        : prefetch_command_list{std::move(prefetch_command_list_)} {}

    boost::container::small_vector<CommandListHeader, 512> command_lists;
    boost::container::small_vector<CommandHeader, 512> prefetch_command_list;
    /// EDEN_NX2_PB_CHECK: hash of each entry's words when it was submitted.
    std::vector<u64> submit_hashes;
    /// The entries' words when they were submitted (NX2), each entry's at submitted_offsets[i]:
    /// executed instead of the memory, which the game may already be refilling for a later frame.
    std::vector<CommandHeader> submitted_words;
    std::vector<u32> submitted_offsets;
    /// DmaPusher::suspensions when the words were copied. A list copied during a suspension, or
    /// that waited behind one, reads memory instead: games write the commands it runs before they
    /// release the acquire.
    u64 snapshot_suspensions = 0;
    u64 snapshot_fetch_generation = 0;
    CommandSnapshotOrigin snapshot_origin = CommandSnapshotOrigin::Unknown;
    u64 snapshot_original_fetch_generation{}, snapshot_original_suspensions{};
    u64 snapshot_refresh_count{};
    std::shared_ptr<const BlitSubmitObservation> blit_submit_observation;
};

/// FNV-1a over pushbuffer words (EDEN_NX2_PB_CHECK).
inline u64 HashPushbufferWords(std::span<const u32> words) {
    u64 hash = 0xcbf29ce484222325ULL;
    for (const u32 word : words) {
        hash = (hash ^ word) * 0x100000001b3ULL;
    }
    return hash;
}

/// @brief The DmaPusher class implements DMA submission to FIFOs, providing an area of memory that the
/// emulated app fills with commands and tells PFIFO to process. The pushbuffers are then assembled
/// into a "command stream" consisting of 32-bit words that make up "commands".
/// See https://envytools.readthedocs.io/en/latest/hw/fifo/dma-pusher.html#fifo-dma-pusher for
/// details on this implementation.
class DmaPusher final {
public:
    /// Counts suspensions on a semaphore acquire and the resumes after them, so it is odd while
    /// the channel is suspended (see CommandList::snapshot_suspensions).
    std::atomic<u64> suspensions{0};
    /// Advances when a GP SYNC_WAIT barrier permits fetching the next segment.
    std::atomic<u64> fetch_generation{0};

    explicit DmaPusher(Core::System& system_, MemoryManager& memory_manager_, Control::ChannelState& channel_state_);
    ~DmaPusher();

    void Push(CommandList&& entries) {
        // Copied during a suspension, or behind one: copy again now, unless still suspended.
        const u64 now = suspensions.load(std::memory_order_acquire);
        if (ResnapshotEnabled() && !entries.submitted_offsets.empty() && (now & 1) == 0 &&
            entries.snapshot_suspensions != now) {
            RefreshSnapshot(entries, now, CommandSnapshotOrigin::PushRefresh);
        }
        dma_pushbuffer.push_back(std::move(entries));
    }

    void DispatchCalls();

    /// Switch 2: true while the channel is suspended on a semaphore acquire.
    [[nodiscard]] bool HasPendingAcquire() const {
        return puller.HasPendingAcquire();
    }
    [[nodiscard]] bool CanResumeAcquire() {
        return puller.CanResumeAcquire(*this);
    }

    [[nodiscard]] size_t CommandResumeOffsetForDiagnostics() const {
        return command_resume_offset;
    }

    void BindSubchannel(Engines::EngineInterface* engine, u32 subchannel_id, Engines::EngineTypes engine_type) {
        subchannels[subchannel_id] = engine;
        subchannel_type[subchannel_id] = engine_type;
    }

    void BindRasterizer(VideoCore::RasterizerInterface* rasterizer);

private:
    static constexpr u32 non_puller_methods = 0x40;
    static constexpr u32 max_subchannels = 8;
    bool Step();
    /// Returns how many words were processed: fewer than given when a semaphore acquire
    /// suspended the channel.
    size_t ProcessCommands(std::span<const CommandHeader> commands, size_t word_offset = 0,
                           const char* source = "prefetch");
    enum class CommandSourceStorage { Submitted, Scratch, Physical };
    struct CommandSource {
        std::span<const CommandHeader> words;
        const CommandList* list;
        size_t entry;
        u64 header;
        size_t first_word;
        const char* label;
        CommandSourceStorage storage;
    };
    size_t ProcessCommandSource(std::span<const CommandHeader> commands, size_t word_offset,
                                const char* source, CommandSourceStorage storage,
                                bool retirement_resume = false);
    [[nodiscard]] bool CommandSourceMatches(const CommandSource& source) const;
    // The original selected storage outlives a pure retirement pause. Do not copy or refetch
    // a live physical span: the old blocking acquire continued reading this same storage.
    const CommandSource* active_command_source{};
    std::optional<CommandSource> retained_command_source;
    size_t command_resume_offset{}; ///< Words of the current entry done before a suspension

    void SetState(const CommandHeader& command_header);

    void CallMethod(u32 argument);
    void CallMultiMethod(const u32* base_start, u32 num_methods);
    [[nodiscard]] bool WantsSegment() const;
    void ReportUnboundSubchannel();
    void WatchArgument(u32 method, u32 argument) const;

public:
    /// Logs the GPFIFO entry being processed (and the one before it), read through the GPU MMU
    /// and straight from guest memory, to find pushbuffer desyncs. Debug aid for Switch 2.
    void DumpCurrentEntry(const char* reason);
    // Private, default-off diagnostic for the parser currently executing on this thread.
    struct MacroFaultDiagnostic {
        u32 start{};
        u32 pc{};
        size_t consumed{};
        u64 code_generation{};
        std::array<u32, 3> memory_registers{};
        std::span<const u32> parameters;
        std::span<const u32> fifo;
        std::span<const u32> code;
        std::span<const u32> data_ram;
    };
    CommandReadDiagnostic command_read_diagnostic{};
    static void CaptureCurrentPushbufferFault(const char* reason,
                                             const MacroFaultDiagnostic* macro = nullptr);
    [[nodiscard]] bool CanRetainCommandSourceForRetirement() const;
    void TraceSemaphoreRetirementGate(bool eligible, const char* reason,
                                      bool configured, u32 argument) const;
    void CheckSubmitHash(const CommandListHeader& header, u64 submit_hash);
    void RecordParsedEntry(const CommandListHeader& header, std::span<const CommandHeader> commands);
    /// Whether the entry lies on device pages the GPU has written (NX2).
    bool IsGpuWrittenCommands(GPUVAddr address, size_t size);
    /// EDEN_NX2_TRUST_CPU_COMMAND_PAGES: whether every page of the entry has been checked often
    /// enough, with no GPU write, to skip waiting for queued writers before checking it again.
    bool IsTrustedCpuCommands(GPUVAddr address, size_t size);
    void NoteCpuCommands(GPUVAddr address, size_t size);
    std::unordered_map<GPUVAddr, u32> cpu_command_page_checks;
    /// Copies the list's words from memory again, as of suspension count `now`.
    void RefreshSnapshot(CommandList& list, u64 now, CommandSnapshotOrigin origin);

    /// EDEN_NX2_RESNAPSHOT=0 turns snapshot refreshes off.
    static bool ResnapshotEnabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("EDEN_NX2_RESNAPSHOT");
            return value == nullptr || value[0] != '0';
        }();
        return enabled;
    }
    /// Reads a command list on pages the GPU wrote into command_headers, from the copies the
    /// buffer cache made after the work that wrote them or from guest memory for pages flushed
    /// since. False when not all are current (or the list isn't GPU-written): the caller drains,
    /// flushes and reads guest memory.
    bool ReadGpuWrittenCommands(GPUVAddr address, u32 count);
    /// Brings command list words the GPU wrote (still only in host buffers) to guest memory.
    void FlushGpuWrittenCommands(GPUVAddr address, size_t size);

private:

public:
    Common::ScratchBuffer<CommandHeader> command_headers; ///< Buffer for list of commands fetched at once

    std::deque<CommandList> dma_pushbuffer; ///< Queue of command lists to be processed
    std::size_t dma_pushbuffer_subindex{};  ///< Index within a command list within the pushbuffer

    struct DmaState {
        u32 method;            ///< Current method
        u32 subchannel;        ///< Current subchannel
        u32 method_count;      ///< Current method count
        u32 length_pending;    ///< Large NI command length pending
        GPUVAddr dma_get;      ///< Currently read segment
        u64 dma_word_offset;   ///< Current word offset from address
        bool non_incrementing; ///< Current command's NI flag
        bool is_last_call;
    };

    Core::System& system;
    MemoryManager& memory_manager;
    Control::ChannelState& channel_state;

    DmaState dma_state{};

    // Switch 2: methods whose data continues in a later GPFIFO entry, set aside while other
    // command buffers run (NVN interleaves several, so more than one can be pending).
    struct ParkedMethod {
        u32 method;
        u32 subchannel;
        u32 method_count;
        bool non_incrementing;
        bool increment_once;
        GPUVAddr resume_address; ///< Where the entry that continues the method starts
    };
    void ResolveSplitMethod(GPUVAddr entry_address);
    static constexpr size_t MaxParkedMethods = 16;

    // Debug aid: hashes of the words of the last entries as they were parsed.
    struct ParsedEntry {
        GPUVAddr address;
        u32 size;
        u64 hash;
    };
    std::array<ParsedEntry, 64> parsed_entries{};
    size_t parsed_entry_count{};
    std::vector<ParkedMethod> parked_methods;
    GPUVAddr split_method_end{}; ///< End of the last entry that ended inside a method

    std::array<Engines::EngineInterface*, max_subchannels> subchannels{};
    std::array<Engines::EngineTypes, max_subchannels> subchannel_type;

    Engines::Puller puller;
    std::mutex sync_mutex;
    std::condition_variable sync_cv;

    VideoCore::RasterizerInterface* rasterizer = nullptr;

    // Not bitfields: the fence thread sets synced, and a bitfield write would rewrite the byte it
    // shares with dma_increment_once and signal_sync (#479).
    const bool ib_enable = true; ///< IB mode enabled
    bool dma_increment_once = false;
    bool signal_sync = false;
    std::atomic<bool> synced{false};
};

} // namespace Tegra
