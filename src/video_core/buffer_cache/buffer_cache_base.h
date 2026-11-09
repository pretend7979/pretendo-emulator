// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include <array>
#include <bit>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/container/unordered_map.h"
#include <boost/container/static_vector.hpp>
#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/spin_recursive_mutex.h"
#include "common/div_ceil.h"
#include "common/literals.h"
#include "common/lru_cache.h"
#include "common/range_sets.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/slot_vector.h"
#include "video_core/buffer_cache/buffer_base.h"
#include "video_core/buffer_cache/virtual_range_cache.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_write_profile.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

using BufferId = Common::SlotId;

using VideoCore::Surface::PixelFormat;
using namespace Common::Literals;

#ifdef __APPLE__
constexpr u32 NUM_VERTEX_BUFFERS = 16;
#else
constexpr u32 NUM_VERTEX_BUFFERS = 32;
#endif
constexpr u32 NUM_TRANSFORM_FEEDBACK_BUFFERS = 4;
constexpr u32 NUM_GRAPHICS_UNIFORM_BUFFERS = 18;
constexpr u32 NUM_COMPUTE_UNIFORM_BUFFERS = 8;
// Graphics retain 32 storage slots. Native compute expansion is gated by host descriptor limits.
constexpr u32 NUM_STORAGE_BUFFERS = 32;
constexpr u32 NUM_COMPUTE_STORAGE_BUFFERS = 64;
constexpr u32 NUM_TEXTURE_BUFFERS = 32;
constexpr u32 NUM_STAGES = 5;

static_assert(NUM_GRAPHICS_UNIFORM_BUFFERS <= 32, "fast bitmask must fit u32");

using UniformBufferSizes = std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>;
using ComputeUniformBufferSizes = std::array<u32, NUM_COMPUTE_UNIFORM_BUFFERS>;

enum class ObtainBufferSynchronize : u32 {
    NoSynchronize = 0,
    FullSynchronize = 1,
    SynchronizeNoDirty = 2,
};

enum class ObtainBufferOperation : u32 {
    DoNothing = 0,
    MarkAsWritten = 1,
    DiscardWrite = 2,
    MarkQuery = 3,
};

static constexpr BufferId NULL_BUFFER_ID{0};
static constexpr u32 DEFAULT_SKIP_CACHE_SIZE = static_cast<u32>(4_KiB);

struct Binding {
    DAddr device_addr{};
    GPUVAddr gpu_addr{};
    u32 size{};
    BufferId buffer_id;
    u32 segment_first{};
    u32 segment_count{};
};

struct MultiRangeSegment {
    BufferId buffer_id;
    DAddr device_addr{};
    u32 size{};
};

struct TextureBufferBinding : Binding {
    PixelFormat format;
};

static constexpr Binding NULL_BINDING{
    .device_addr = 0,
    .size = 0,
    .buffer_id = NULL_BUFFER_ID,
};

template <typename Buffer>
struct HostBindings {
    boost::container::static_vector<Buffer*, NUM_VERTEX_BUFFERS> buffers;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> offsets;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> sizes;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> strides;
    u32 min_index{NUM_VERTEX_BUFFERS};
    u32 max_index{0};
};

class BufferCacheChannelInfo : public ChannelInfo {
public:
    BufferCacheChannelInfo() = delete;
    BufferCacheChannelInfo(Tegra::Control::ChannelState& state) noexcept : ChannelInfo(state) {}
    BufferCacheChannelInfo(const BufferCacheChannelInfo& state) = delete;
    BufferCacheChannelInfo& operator=(const BufferCacheChannelInfo&) = delete;

    Binding index_buffer;
    std::array<Binding, NUM_VERTEX_BUFFERS> vertex_buffers;
    std::array<std::array<Binding, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES> uniform_buffers;
    std::array<std::array<Binding, NUM_STORAGE_BUFFERS>, NUM_STAGES> storage_buffers;
    std::array<std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS>, NUM_STAGES> texture_buffers;
    std::array<Binding, NUM_TRANSFORM_FEEDBACK_BUFFERS> transform_feedback_buffers;
    Binding count_buffer_binding;
    Binding indirect_buffer_binding;

    std::array<Binding, NUM_COMPUTE_UNIFORM_BUFFERS> compute_uniform_buffers;
    std::array<Binding, NUM_COMPUTE_STORAGE_BUFFERS> compute_storage_buffers;
    std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS> compute_texture_buffers;

    std::array<u32, NUM_STAGES> enabled_uniform_buffer_masks{};
    u32 enabled_compute_uniform_buffer_mask = 0;

    const UniformBufferSizes* uniform_buffer_sizes{};
    const ComputeUniformBufferSizes* compute_uniform_buffer_sizes{};

    std::array<u32, NUM_STAGES> enabled_storage_buffers{};
    std::array<u32, NUM_STAGES> written_storage_buffers{};
    u64 enabled_compute_storage_buffers = 0;
    u64 written_compute_storage_buffers = 0;

    u32 total_graphics_storage_buffers = 0;
    u32 total_compute_storage_buffers = 0;

    std::array<u32, NUM_STAGES> enabled_texture_buffers{};
    std::array<u32, NUM_STAGES> written_texture_buffers{};
    std::array<u32, NUM_STAGES> image_texture_buffers{};
    u32 enabled_compute_texture_buffers = 0;
    u32 written_compute_texture_buffers = 0;
    u32 image_compute_texture_buffers = 0;

    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_hits{};
    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_shots{};

    u32 uniform_buffer_skip_cache_size = DEFAULT_SKIP_CACHE_SIZE;

    bool has_deleted_buffers = false;

    std::array<u32, NUM_STAGES> dirty_uniform_buffers{};
    std::array<u32, NUM_STAGES> fast_bound_uniform_buffers{};
    std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
        uniform_buffer_binding_sizes{};
};

template <class P>
class BufferCache : public VideoCommon::ChannelSetupCaches<BufferCacheChannelInfo> {
    // Page size for caching purposes.
    // This is unrelated to the CPU page size and it can be changed as it seems optimal.
#ifdef YUZU_LEGACY
    static constexpr u32 CACHING_PAGEBITS = 12;
#else
    static constexpr u32 CACHING_PAGEBITS = 16;
#endif
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;

    static constexpr bool IS_OPENGL = P::IS_OPENGL;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS =
        P::HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT =
        P::HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = P::NEEDS_BIND_UNIFORM_INDEX;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = P::NEEDS_BIND_STORAGE_INDEX;
    static constexpr bool USE_MEMORY_MAPS = P::USE_MEMORY_MAPS;
    static constexpr bool SEPARATE_IMAGE_BUFFERS_BINDINGS = P::SEPARATE_IMAGE_BUFFER_BINDINGS;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = P::USE_MEMORY_MAPS_FOR_UPLOADS;

#ifdef YUZU_LEGACY
    static constexpr s64 TARGET_THRESHOLD = 3_GiB;
#else
    static constexpr s64 TARGET_THRESHOLD = 4_GiB;
#endif

    static constexpr s64 DEFAULT_EXPECTED_MEMORY = 512_MiB;
    static constexpr s64 DEFAULT_CRITICAL_MEMORY = 1_GiB;

    // Debug Flags.

    static constexpr bool DISABLE_DOWNLOADS = true;

    using Maxwell = Tegra::Engines::Maxwell3D::Regs;

    using Runtime = typename P::Runtime;
    using Buffer = typename P::Buffer;
    using Async_Buffer = typename P::Async_Buffer;
    using MemoryTracker = typename P::MemoryTracker;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        DAddr begin;
        DAddr end;
        bool has_stream_leap = false;
        bool valid = true;
    };

public:
    explicit BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_);

    ~BufferCache();

    void TickFrame();

    bool BindMultiRangeStorage(const Binding& binding, bool is_written,
                               std::span<const MultiRangeSegment> pool,
                               const Nx2WriteProfile::Key* writer = nullptr);

    void ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                  std::vector<MultiRangeSegment>& pool);

    void UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size);

    /// `keep_page_gpu_data`: download GPU-written bytes sharing the written pages first (false
    /// when the memory is going away).
    void WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data = true);

    /// Publishes a synchronous short query payload without allowing older buffer
    /// downloads to replace it. The caller holds mutex on the GPU owner thread.
    void WriteQueryPayload(DAddr device_addr, u32 value);

    void CachedWriteMemory(DAddr device_addr, u64 size);

    bool OnCPUWrite(DAddr device_addr, u64 size);

    void DownloadMemory(DAddr device_addr, u64 size);

    std::optional<VideoCore::RasterizerDownloadArea> GetFlushArea(DAddr device_addr, u64 size);

    /// The core consumed the read traps of these pages for a read the texture cache answered:
    /// the next GPU write here must arm them again (GetFlushArea forgets them otherwise).
    void ForgetReadTraps(DAddr device_addr, u64 size) {
        read_trapped_ranges.Subtract(
            Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE),
            Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE) -
                Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE));
    }

    /// Without downloads at every fence, a guest CPU read is the only way GPU-written buffer
    /// data reaches guest memory, so reads must ask the buffer cache (GetFlushArea).
    bool FlushesOnCpuRead() const {
        return !AsyncBufferDownloads();
    }

    bool InlineMemory(DAddr dest_address, size_t copy_size, std::span<const u8> inlined_buffer);

    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size);

    void DisableGraphicsUniformBuffer(size_t stage, u32 index);

    void UpdateGraphicsBuffers(bool is_indexed);

    struct NativeStorageDomainResult;
    void UpdateComputeBuffers(std::span<NativeStorageDomainResult> domains = {});

    void BindHostGeometryBuffers(bool is_indexed);

    void BindHostStageBuffers(size_t stage, const Nx2WriteProfile::Stage* writer = nullptr);

    void BindHostComputeBuffers(const Nx2WriteProfile::Stage* writer = nullptr,
                                std::span<const u32> dirty_limits = {});

    struct ComputeUniformBindingSnapshot {
        const Buffer* buffer{};
        BufferId buffer_id{};
        u32 offset{};
        u32 size{};
        const char* skip_reason{"unavailable"};
    };
    /// Capture-only cache ownership lookup after binding; caller holds mutex.
    /// The actual descriptor must still match because small UBOs may be streamed.
    [[nodiscard]] ComputeUniformBindingSnapshot GetComputeUniformBindingSnapshot(u32 index) const;
    [[nodiscard]] ComputeUniformBindingSnapshot GetGraphicsUniformBindingSnapshot(size_t stage, u32 index) const;

    /// Private scene-bound diagnostic; zero request leaves ordinary binding unchanged.
    struct StorageRangeDiagnostic {
        GPUVAddr resolved_pointer{};
        GPUVAddr bound_gpu_addr{};
        u32 bound_size{};
        u32 requested{};
        u64 mapped_size{};
        u64 contiguous_size{};
        u64 max_descriptor_range{};
        u64 alignment_prefix{};
        bool applied{};
        const char* reason{"not requested"};
    };

    struct ComputeStorageBindingSnapshot {
        const Buffer* buffer{};
        GPUVAddr gpu_addr{};
        DAddr device_addr{};
        u32 offset{};
        u32 size{};
        u32 segment_count{};
        const char* skip_reason{"unavailable"};
    };
    /// Capture-only observation after BindHostComputeBuffers; caller holds mutex.
    /// Does not synchronize, allocate or change resource usage. Multi-range views are skipped.
    [[nodiscard]] ComputeStorageBindingSnapshot GetComputeStorageBindingSnapshot(u32 index) const;

    struct NativeStorageRangeWitness {
        GPUVAddr pointer{}, aligned_pointer{};
        u64 alignment{}, alignment_prefix{};
        u64 pointer_kind_suffix{}, aligned_kind_suffix{};
        u64 backend_limit{}, address_space_limit{}, candidate_span{};
        u64 scan_request{}, physical_span{}, accessible_span{};
        u64 host_suffix{}, uncovered_bytes{};
        bool scan_capped{}, host_matches{}, valid{};
        const char* reason{"invalid pointer"};
    };
    /// Capture-only accessible-domain witness, not logical allocation ownership.
    [[nodiscard]] NativeStorageRangeWitness GetNativeStorageRangeWitness(
        GPUVAddr pointer, const ComputeStorageBindingSnapshot& snapshot) const;

    struct NativeStorageDomainResult {
        u64 candidate_span{}, scan_span{}, contiguous_span{}, owner_suffix{};
        u32 previous_size{}, bound_size{}, allocation_limit{};
        bool applied{};
        const char* reason{"invalid binding"};
    };
    /// Stage a validated readonly domain before normal owner allocation and binding.
    [[nodiscard]] NativeStorageDomainResult PrepareComputeReadOnlyDomain(u32 index, GPUVAddr pointer);
    /// Expand a settled readonly graphics binding only within its existing validated owner.
    [[nodiscard]] NativeStorageDomainResult ExpandGraphicsReadOnlyDomain(size_t stage, u32 index);

    /// Capture-only observation after BindHostStageBuffers; caller holds mutex.
    /// No synchronization or mutation; unsupported multi-range bindings are explicit skips.
    [[nodiscard]] ComputeStorageBindingSnapshot GetGraphicsStorageBindingSnapshot(size_t stage, u32 index) const;

    void SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                const UniformBufferSizes* sizes);

    void SetComputeUniformBufferState(u32 mask, const ComputeUniformBufferSizes* sizes);

    void UnbindGraphicsStorageBuffers(size_t stage);

    struct GraphicsStoragePointerGuard {
        DAddr root{};
        DAddr entry{};
        bool valid{};
    };

    struct GraphicsUniformWriteSource {
        DAddr device_addr{};
        u32 size{};
    };
    struct GraphicsUniformWriteSpan {
        u16 end{};
        GraphicsUniformWriteSource source;
    };
    /// Retain clean CPU-authored constants and bind those exact bytes for this draw.
    void ResetGraphicsUniformWriteSnapshots();
    [[nodiscard]] std::optional<GraphicsUniformWriteSpan> ResolveGraphicsUniformWriteSpan(
        size_t stage, u32 bank, u32 offset, s32 displacement, u32 width);
    [[nodiscard]] bool GraphicsUniformWriteSnapshotsValid() const {
        return graphics_uniform_write_snapshots_valid;
    }

    /// Validate before host binding; only raw logical binding ranges are inspected.
    using GraphicsStorageWriteBounds = std::array<std::array<u16, NUM_STORAGE_BUFFERS>, NUM_STAGES>;
    void GuardGraphicsStoragePointerWrites(size_t stage, size_t ssbo_index,
        const GraphicsStoragePointerGuard& guard, u32 active_stages,
        const GraphicsStorageWriteBounds& write_bounds,
        std::span<const GraphicsUniformWriteSource> uniform_sources = {});

    bool BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                   bool is_written, u32 pointer_offset = ~0U,
                                   bool native_pointer = false,
                                   bool indirect_range_trial = false,
                                   u32 diagnostic_minimum = 0,
                                   StorageRangeDiagnostic* diagnostic = nullptr,
                                   GraphicsStoragePointerGuard* write_guard = nullptr);

    /// Resolve a fixed one-level pointer table; caller holds the buffer-cache mutex.
    /// Call for every derived descriptor before configuring any stage bindings.
    [[nodiscard]] std::optional<GPUVAddr> ResolveGraphicsStorageBufferPointer(
        size_t stage, u32 cbuf_index, u32 cbuf_offset, u32 pointer_offset,
        GraphicsStoragePointerGuard* guard = nullptr);

    void UnbindGraphicsTextureBuffers(size_t stage);

    void BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                   PixelFormat format, bool is_written, bool is_image);

    void UnbindComputeStorageBuffers();

    /// Set by the pipeline before binding a storage buffer: the byte end, from the binding's
    /// aligned start, of the furthest read its shader reported past an earlier binding (0: none).
    u32 storage_feedback_end{};

    /// Derived descriptors require the result of the prebinding resolver below. Nullopt is a
    /// failed resolution, not a request to reread the pointer after other bindings are uploaded.
    void BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                  bool is_written, u32 pointer_offset = ~0U,
                                  std::optional<GPUVAddr> resolved_pointer = std::nullopt,
                                  bool pointer_prevalidated = false,
                                  u32 diagnostic_minimum = 0,
                                  StorageRangeDiagnostic* diagnostic = nullptr);

    /// Coherent native root (sentinel offset) or one-level table read, with mutex held. Resolve every
    /// derived descriptor before configuring ANY compute uniform or storage binding.
    [[nodiscard]] std::optional<GPUVAddr> ResolveComputeStorageBufferPointer(
        u32 cbuf_index, u32 cbuf_offset, u32 pointer_offset);

    void UnbindComputeTextureBuffers();

    void BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size, PixelFormat format,
                                  bool is_written, bool is_image);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                       ObtainBufferSynchronize sync_info,
                                                       ObtainBufferOperation post_op);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainCPUBuffer(DAddr gpu_addr, u32 size,
                                                          ObtainBufferSynchronize sync_info,
                                                          ObtainBufferOperation post_op);
    void FlushCachedWrites();

    /// Return true when there are uncommitted buffers to be downloaded
    [[nodiscard]] bool HasUncommittedFlushes() const noexcept;

    void AccumulateFlushes();

    /// Return true when the caller should wait for async downloads
    [[nodiscard]] bool ShouldWaitAsyncFlushes() const noexcept;

    /// Commit asynchronous downloads
    void CommitAsyncFlushes();
    void CommitAsyncFlushesHigh();

    /// Pop asynchronous downloads
    void PopAsyncFlushes();
    void PopAsyncBuffers();

    bool DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount);

    bool DMAClear(GPUVAddr src_address, u64 amount, u32 value);

    /// Switch 2 GPU-written command lists (nx2.h, g_nx2_command_list_pages). While collecting (a
    /// compute dispatch or a DMA copy being recorded), writes to command list pages are noted;
    /// Nx2CopyListPages then records copies of those pages into host memory after the work.
    void Nx2CollectListCopies(bool collect) {
        nx2_collect_list_copies = collect;
        if (!collect) {
            nx2_list_copy_requests.clear();
        }
    }
    /// True when it recorded copies, which the GPU completes at `tick` (the caller submits).
    bool Nx2CopyListPages(u64 tick);
    enum class Nx2ListRead { Unknown, Copied, Synced };
    /// Front thread: Synced when guest memory holds the page `device_addr` lies in (flushed and
    /// written by no GPU work since), else Copied with `out` filled from the copy of its last
    /// write once the GPU is done with it. Unknown otherwise: the caller drains and flushes.
    Nx2ListRead Nx2ReadListCopy(DAddr device_addr, std::span<u8> out);
    /// Front thread, the split drained: the command list page was flushed to guest memory.
    void Nx2NoteListSynced(DAddr page);
    /// CPU and inline writes: they reach guest memory, so a synced page stays synced, but its copy
    /// no longer holds the page.
    void Nx2NoteListHostWrite(DAddr device_addr, u64 size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(DAddr addr, size_t size);

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(DAddr addr, size_t size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(DAddr addr, size_t size);

    void SetDrawIndirect(
        const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect_) {
        current_draw_indirect = current_draw_indirect_;
    }

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectCount();

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectBuffer();

    template <typename Func>
    void BufferOperations(Func&& func) {
        do {
            channel_state->has_deleted_buffers = false;
            func();
        } while (channel_state->has_deleted_buffers);
    }

    Common::SpinRecursiveMutex mutex;
    Runtime& runtime;
    bool any_buffer_uploaded = false;

private:
    static constexpr size_t GRAPHICS_UNIFORM_PROOF_BYTES = 256;
    static constexpr size_t GRAPHICS_UNIFORM_PROOF_COUNT = 32;
    struct GraphicsUniformWriteSnapshot {
        size_t stage{};
        u32 bank{};
        GPUVAddr gpu_addr{};
        DAddr device_addr{};
        u32 size{};
        std::array<u8, GRAPHICS_UNIFORM_PROOF_BYTES> bytes{};
    };
    boost::container::static_vector<GraphicsUniformWriteSnapshot,
                                   GRAPHICS_UNIFORM_PROOF_COUNT> graphics_uniform_write_snapshots;
    bool graphics_uniform_write_snapshots_valid{true};
    bool BindGraphicsUniformWriteSnapshot(size_t stage, u32 bank, u32 binding_index);

    template <typename Mask, typename Func>
    static void ForEachEnabledBit(Mask enabled_mask, Func&& func) {
        for (u32 index = 0; enabled_mask != 0; ++index, enabled_mask >>= 1) {
            const int disabled_bits = std::countr_zero(enabled_mask);
            index += disabled_bits;
            enabled_mask >>= disabled_bits;
            func(index);
        }
    }

    template <typename Func>
    void ForEachBufferInRange(DAddr device_addr, u64 size, Func&& func) {
        const u64 page_end = Common::DivCeil(device_addr + size, CACHING_PAGESIZE);
        for (u64 page = device_addr >> CACHING_PAGEBITS; page < page_end;) {
            const BufferId buffer_id = page_table[page];
            if (!buffer_id) {
                ++page;
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            func(buffer_id, buffer);

            const DAddr end_addr = buffer.CpuAddr() + buffer.SizeBytes();
            page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
        }
    }

    static bool IsRangeGranular(DAddr device_addr, size_t size) {
        return (device_addr & ~Core::DEVICE_PAGEMASK) ==
               ((device_addr + size) & ~Core::DEVICE_PAGEMASK);
    }

    void RunGarbageCollector(bool aggressive_gc);

    void BindHostIndexBuffer();

    void BindHostVertexBuffers();

    void BindHostDrawIndirectBuffers();

    void BindHostGraphicsUniformBuffers(size_t stage);

    void BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind);

    void BindHostGraphicsStorageBuffers(size_t stage, const Nx2WriteProfile::Stage* writer = nullptr);

    void BindHostGraphicsTextureBuffers(size_t stage, const Nx2WriteProfile::Stage* writer = nullptr);

    void BindHostTransformFeedbackBuffers();

    void BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size, u32 stride);

    void BindHostComputeUniformBuffers();

    void BindHostComputeStorageBuffers(const Nx2WriteProfile::Stage* writer = nullptr,
                                       std::span<const u32> dirty_limits = {});

    void BindHostComputeTextureBuffers(const Nx2WriteProfile::Stage* writer = nullptr);

    void DoUpdateGraphicsBuffers(bool is_indexed);

    void DoUpdateComputeBuffers(std::span<NativeStorageDomainResult> domains);

    void UpdateIndexBuffer();

    void UpdateVertexBuffers();

    void UpdateVertexBuffer(u32 index);

    void UpdateDrawIndirect();

    void UpdateUniformBuffers(size_t stage);

    void UpdateStorageBuffers(size_t stage);

    void UpdateTextureBuffers(size_t stage);

    void UpdateTransformFeedbackBuffers();

    void UpdateTransformFeedbackBuffer(u32 index);

    void UpdateComputeUniformBuffers();

    void UpdateComputeStorageBuffers(std::span<NativeStorageDomainResult> domains);

    void UpdateComputeTextureBuffers();

    void MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync = false,
                           const Nx2WriteProfile::Key* writer = nullptr, u32 bound_size = 0);

    /// Counts a GPU write to the command list pages in the range (invalidating their copies) and,
    /// while collecting, asks for new copies.
    void Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size);

    [[nodiscard]] BufferId FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible,
                                      u32 allocation_limit = 0);

    void SynchronizeBufferWrites(Buffer& buffer);

    [[nodiscard]] OverlapResult ResolveOverlaps(DAddr device_addr, u32 wanted_size,
                                               u32 allocation_limit = 0);

    void JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);

    [[nodiscard]] BufferId CreateBuffer(DAddr device_addr, u32 wanted_size,
                                        bool sparse_compatible, u32 allocation_limit = 0);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    void TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept;

    bool SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size);

    void UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                      std::span<BufferCopy> copies);

    void ImmediateUploadMemory(Buffer& buffer, u64 largest_copy,
                               std::span<const BufferCopy> copies);

    void MappedUploadMemory(Buffer& buffer, u64 total_size_bytes, std::span<BufferCopy> copies);

    void DownloadBufferMemory(Buffer& buffer_id);

    void DownloadBufferMemory(Buffer& buffer_id, DAddr device_addr, u64 size);

    /// Downloads the GPU-written data of several buffers with one wait for the GPU.
    void DownloadBuffersMemory(std::span<const BufferId> buffer_ids);

    void DeleteBuffer(BufferId buffer_id, bool do_not_mark = false);

    [[nodiscard]] Binding StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                               bool native_pointer = false,
                                               std::optional<GPUVAddr> resolved_pointer = std::nullopt,
                                               u32 diagnostic_minimum = 0,
                                               StorageRangeDiagnostic* diagnostic = nullptr) const;
    /// Default-off graphics experiment: exact, mapped and contiguous extent, at most 64 MiB.
    [[nodiscard]] std::optional<Binding> GraphicsIndirectRangeBinding(GPUVAddr gpu_addr) const;
    [[nodiscard]] std::optional<Binding> DiagnosticStorageBufferBinding(
        GPUVAddr gpu_addr, u32 minimum, StorageRangeDiagnostic& diagnostic) const;

    [[nodiscard]] std::optional<GPUVAddr> ReadStoragePointer(GPUVAddr address);

    [[nodiscard]] TextureBufferBinding GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                               PixelFormat format);

    [[nodiscard]] std::span<const u8> ImmediateBufferWithData(DAddr device_addr, size_t size);

    [[nodiscard]] std::span<u8> ImmediateBuffer(size_t wanted_capacity);

    [[nodiscard]] bool HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept;

    [[nodiscard]] Binding& VertexBufferSlot(u32 index);

    [[nodiscard]] const Binding& VertexBufferSlot(u32 index) const;

    void UpdateVertexBufferSlot(u32 index, const Binding& binding);

    /// Flags the vertex buffers overlapping a written range so host copies derived from them are
    /// rebuilt.
    void MarkVertexBuffersStale(DAddr device_addr, u64 size);

    /// Re-dirties repacked vertex buffers whose source data was written since they were bound.
    void RefreshRepackedVertexBuffers();

    void ClearDownload(DAddr base_addr, u64 size);

    void InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                    std::span<const u8> inlined_buffer);

    Tegra::MaxwellDeviceMemoryManager& device_memory;

    Common::SlotVector<Buffer> slot_buffers;
#ifdef YUZU_LEGACY
    static constexpr size_t TICKS_TO_DESTROY = 6;
#else
    static constexpr size_t TICKS_TO_DESTROY = 8;
#endif
    DelayedDestructionRing<Buffer, TICKS_TO_DESTROY> delayed_destruction_ring;

    const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect{};

    u32 last_index_count = 0;

    u32 enabled_vertex_buffers_mask = 0;
    u32 stale_vertex_buffers_mask = 0;
    u64 vertex_buffers_serial = 0;
    std::array<Binding, 32> v_buffer{};

    boost::container::small_vector<BufferCopy, 4> upload_copies;

    MemoryTracker memory_tracker;
    Common::RangeSet<DAddr> uncommitted_gpu_modified_ranges;

    /// Whether GPU-written buffer ranges are copied back to guest memory at the next fence;
    /// otherwise only when something flushes them (as upstream yuzu below High GPU accuracy).
    /// Not for Switch 2 titles: title 1 had 250-330 MB/s copied back (a fence per compute
    /// dispatch, the fencing thread holding the buffer cache while it wrote them), while its CPU
    /// read GPU-written memory at most 64 times in 200 s (counted with reactive flushing) and
    /// the native MME's DMA reads never touched those pages. Without them a fence costs ~1 us
    /// instead of ~5 and draw setup ~7% less. EDEN_NX2_BUFFER_DOWNLOADS=0/1 forces it.
    bool AsyncBufferDownloads() const {
        static const int forced = [] {
            const char* value = std::getenv("EDEN_NX2_BUFFER_DOWNLOADS");
            return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
        }();
        if (forced >= 0) {
            return forced == 1;
        }
        return gpu_memory == nullptr || !gpu_memory->UsesNativeTiling();
    }

    /// Selects fence downloads and reactive traps for a buffer write. The default
    /// retains the whole-range policy; EDEN_NX2_DOWNLOAD_INTERSECTIONS=1 restricts
    /// native selective downloads to CPU-read page intersections for A/B testing.
    void TrackBufferWriteDownloads(DAddr device_addr, u64 size);

    /// Whether a GPU write to this range is copied back at the next fence. Ranges the guest CPU
    /// has read (Preflushable, set by the first reactive flush) always are: later reads skip the
    /// flush and rely on these downloads, e.g. collision meshes skinned by compute shaders.
    bool DownloadsAtFence(DAddr device_addr, u64 size) noexcept {
        return AsyncBufferDownloads() || memory_tracker.IsRegionPreflushable(device_addr, size);
    }

    /// GPU writes that are not downloaded at fences: trap the guest CPU's next read of their
    /// pages, which flushes them once and marks them Preflushable (then DownloadsAtFence).
    void ArmReadTraps(DAddr device_addr, u64 size) {
        const DAddr begin = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
        const DAddr end = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
        // Only the parts not armed yet: the core visits each page it is given, and a buffer the
        // GPU rewrites every frame is mostly still armed (the guest read a few of its pages).
        DAddr cursor = begin;
        read_trapped_ranges.ForEachInRange(begin, end - begin, [&](DAddr lower, DAddr upper) {
            if (lower > cursor) {
                device_memory.ArmReadTraps(cursor, lower - cursor);
            }
            cursor = (std::max)(cursor, upper);
        });
        if (cursor < end) {
            device_memory.ArmReadTraps(cursor, end - cursor);
        }
        read_trapped_ranges.Add(begin, end - begin);
    }

    Common::RangeSet<DAddr> read_trapped_ranges;
    Common::RangeSet<DAddr> gpu_modified_ranges;
    std::deque<Common::RangeSet<DAddr>> committed_gpu_modified_ranges;

    // Async Buffers
    std::deque<std::optional<Async_Buffer>> async_buffers;
    std::deque<boost::container::small_vector<BufferCopy, 4>> pending_downloads;
    /// For each entry of pending_downloads: the ranges it still writes to guest memory when its
    /// fence pops. A write to a range after a download was taken removes it from that download
    /// (and every other pending one): its bytes are older than the write.
    std::deque<Common::RangeSet<DAddr>> pending_download_ranges;
    std::optional<Async_Buffer> current_buffer;

    std::deque<Async_Buffer> async_buffers_death_ring;

    size_t immediate_buffer_capacity = 0;
    Common::ScratchBuffer<u8> immediate_buffer_alloc;
    /// Guest memory large uploads came from since the last cold hint (device address, size).
    std::vector<std::pair<DAddr, u64>> uploaded_buffer_ranges;

    struct LRUItemParams {
        using ObjectType = BufferId;
        using TickType = u64;
    };
    Common::LeastRecentlyUsedCache<LRUItemParams> lru_cache;
    VirtualRangeCache virtual_ranges;
    std::vector<MultiRangeSegment> graphics_segments;
    std::vector<MultiRangeSegment> compute_segments;
    u64 frame_tick = 0;
    u64 total_used_memory = 0;
    u64 buffer_bytes = 0; ///< The buffers' own size (total_used_memory may be the device's)
    u64 minimum_memory = 0;
    u64 critical_memory = 0;
    BufferId inline_buffer_id;
#ifdef YUZU_LEGACY
    bool immediately_free = false;
#endif

    // Covers the device address space (MaxwellDeviceTraits::device_virtual_bits).
    std::array<BufferId, ((1ULL << 36) >> CACHING_PAGEBITS)> page_table;
    Common::ScratchBuffer<u8> tmp_buffer;

    /// A command list page (Nx2CopyListPages): the GPU writes noted to it and the last copy.
    using Nx2Staging = decltype(std::declval<Runtime&>().DownloadStagingBuffer(size_t{}, true));
    struct Nx2ListPage {
        u64 writes{};      ///< GPU writes noted to the page
        u64 copy_writes{}; ///< `writes` when the copy was made: the copy is current while equal
        u64 copy_tick{};   ///< GPU tick that completes the copy
        /// `writes` when the front last flushed the page: guest memory holds it while equal
        u64 synced_writes{~u64{0}};
        u32 copy_begin{};  ///< Bytes of the page the copy holds (a buffer can end inside it)
        u32 copy_end{};
        /// Written outside a compute dispatch or a DMA copy (a draw's storage writes) at least
        /// once: the front can't tell from what it sent whether such a write is still queued, so
        /// the page's lists always drain the split instead of reading a copy.
        bool draw_written{};
        std::optional<Nx2Staging> copy;
    };
    std::mutex nx2_list_mutex; ///< nx2_list_pages: the split's front reads the copies
    std::unordered_map<DAddr, Nx2ListPage> nx2_list_pages;
    /// Copies replaced while the GPU may still write them: freed once it is past their tick.
    std::vector<std::pair<u64, Nx2Staging>> nx2_retired_list_copies;
    boost::container::small_vector<std::pair<BufferId, DAddr>, 8> nx2_list_copy_requests;
    bool nx2_collect_list_copies{};
};

} // namespace VideoCommon
