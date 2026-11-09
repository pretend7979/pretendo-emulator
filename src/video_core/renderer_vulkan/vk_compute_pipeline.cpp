// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <fmt/format.h>

#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_frame_dump.h"
#include "video_core/renderer_vulkan/oot_native_domain.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_storage_feedback.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_handle_program.h"
#include "video_core/shader_notify.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "common/logging.h"
#include "common/settings.h"

namespace Vulkan {

using Shader::ImageBufferDescriptor;
using Shader::Backend::SPIRV::RESCALING_LAYOUT_WORDS_OFFSET;
using Tegra::Texture::TexturePair;

namespace {

// Private opt-in driver-call probe, paired with the dispatch synchronization trace.
u32 BeginComputePipelineTrace(u64 shader_hash, u32 storage_buffers, u32 subgroup_size) noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_COMPUTE_SYNC_TRACE");
        return value && *value && std::strlen(value) <= 4096;
    }();
    if (!enabled) return 0;
    static std::atomic<u32> count{};
    u32 previous = count.load(std::memory_order_relaxed);
    do {
        if (previous >= 128) return 0;
    } while (!count.compare_exchange_weak(previous, previous + 1, std::memory_order_relaxed));
    const u32 serial = previous + 1;
    std::fprintf(stderr,
        "{\"kind\":\"compute_pipeline_build\",\"serial\":%u,\"phase\":\"begin\",\"shader_hash\":\"%016llx\",\"storage_buffers\":%u,\"subgroup_size\":%u}\n",
        serial, static_cast<unsigned long long>(shader_hash), storage_buffers, subgroup_size);
    std::fflush(stderr);
    return serial;
}

void EndComputePipelineTrace(u32 serial, u64 shader_hash, VkResult result,
                             bool exception, u64 microseconds) noexcept {
    if (!serial) return;
    std::fprintf(stderr,
        "{\"kind\":\"compute_pipeline_build\",\"serial\":%u,\"phase\":\"end\",\"shader_hash\":\"%016llx\",\"result\":%d,\"exception\":%s,\"microseconds\":%llu}\n",
        serial, static_cast<unsigned long long>(shader_hash), static_cast<int>(result),
        exception ? "true" : "false", static_cast<unsigned long long>(microseconds));
    std::fflush(stderr);
}

} // namespace

ComputePipeline::ComputePipeline(const Device& device_, Scheduler& scheduler, vk::PipelineCache& pipeline_cache_,
                                 DescriptorPool& descriptor_pool,
                                 GuestDescriptorQueue& guest_descriptor_queue_,
                                 DescriptorBufferRing& descriptor_buffer_ring_,
                                 Common::ThreadWorker* thread_worker,
                                 PipelineStatistics* pipeline_statistics,
                                 VideoCore::ShaderNotify* shader_notify, const Shader::Info& info_,
                                 vk::ShaderModule spv_module_, u64 shader_hash_,
                                 Shader::ComputeWriteBounds write_bounds_)
    : device{device_},
      pipeline_cache(pipeline_cache_), guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_}, info{info_},
      write_bounds{std::move(write_bounds_)},
      shader_hash{shader_hash_}, spv_module(std::move(spv_module_)) {
    if (shader_notify) {
        shader_notify->MarkShaderBuilding();
    }
    std::copy_n(info.constant_buffer_used_sizes.begin(), uniform_buffer_sizes.size(),
                uniform_buffer_sizes.begin());
    num_descriptor_entries = NumDescriptorEntries(info);

    DescriptorLayoutBuilder builder{device};
    builder.Add(info, VK_SHADER_STAGE_COMPUTE_BIT);

    uses_push_descriptor = builder.CanUsePushDescriptor();
    uses_descriptor_buffer = builder.CanUseDescriptorBuffer() && descriptor_buffer_ring.IsValid();
    descriptor_set_layout =
        builder.CreateDescriptorSetLayout(uses_push_descriptor, uses_descriptor_buffer);
    if (uses_descriptor_buffer) {
        descriptor_buffer_layout = builder.MakeDescriptorBufferLayout(*descriptor_set_layout);
        if (!descriptor_buffer_ring.CanAllocate(descriptor_buffer_layout.size)) {
            LOG_DEBUG(Render_Vulkan,
                        "Compute shader {:016X} needs {} descriptor bytes per dispatch, falling "
                        "back to sets",
                        shader_hash, descriptor_buffer_layout.size);
            uses_descriptor_buffer = false;
            descriptor_buffer_layout = {};
            descriptor_set_layout = builder.CreateDescriptorSetLayout(false);
        }
    }
    pipeline_layout = builder.CreatePipelineLayout(*descriptor_set_layout);
    if (!uses_descriptor_buffer) {
        descriptor_update_template =
            builder.CreateTemplate(*descriptor_set_layout, *pipeline_layout, uses_push_descriptor);
        if (!uses_push_descriptor) {
            descriptor_allocator =
                descriptor_pool.Allocator(device, scheduler, *descriptor_set_layout, info);
        }
    }

    auto func{[this, shader_notify, pipeline_statistics] {
        const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size_ci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
            .pNext = nullptr,
            .requiredSubgroupSize = GuestWarpSize,
        };
        // Only warp-level ops observe the subgroup size. Forcing 32 on other shaders makes Intel
        // compile SIMD32, which can spill a register-heavy kernel for minutes (Mario Kart World).
        const bool uses_warp_ops = info.uses_subgroup_vote || info.uses_subgroup_invocation_id ||
                                   info.uses_subgroup_shuffles || info.uses_subgroup_mask ||
                                   info.uses_quad_shuffles || info.uses_fswzadd ||
                                   info.uses_cooperative_matrix;
        VkPipelineCreateFlags flags{};
        if (device.IsKhrPipelineExecutablePropertiesEnabled() && Settings::values.renderer_debug.GetValue()) {
            flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
        }
        if (uses_descriptor_buffer) {
            flags |= VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        }
        const VkComputePipelineCreateInfo compute_ci{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = flags,
            .stage{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = device.IsExtSubgroupSizeControlSupported() && uses_warp_ops &&
                                 device.IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT)
                             ? &subgroup_size_ci
                             : nullptr,
                // Cooperative matrices need every subgroup fully populated.
                .flags = info.uses_cooperative_matrix
                             ? VkPipelineShaderStageCreateFlags{
                                   VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT}
                             : VkPipelineShaderStageCreateFlags{},
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = *spv_module,
                .pName = "main",
                .pSpecializationInfo = nullptr,
            },
            .layout = *pipeline_layout,
            .basePipelineHandle = 0,
            .basePipelineIndex = 0,
        };
        const u32 trace_serial = BeginComputePipelineTrace(
            shader_hash, static_cast<u32>(info.storage_buffers_descriptors.size()),
            compute_ci.stage.pNext ? GuestWarpSize : 0);
        const auto trace_started = trace_serial ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
        const auto trace_end = [&](VkResult result, bool exception) noexcept {
            if (trace_serial) {
                EndComputePipelineTrace(trace_serial, shader_hash, result, exception,
                    static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - trace_started).count()));
            }
        };
        try {
            pipeline = device.GetLogical().CreateComputePipeline(compute_ci, *pipeline_cache);
            trace_end(VK_SUCCESS, false);
        } catch (const vk::Exception& exception) {
            trace_end(exception.GetResult(), true);
            LOG_CRITICAL(Render_Vulkan, "Adreno rejected compute shader {:016X}: {}", shader_hash,
                         exception.what());
            std::scoped_lock lock{build_mutex};
            is_built = true;
            build_condvar.notify_one();
            if (shader_notify) {
                shader_notify->MarkShaderComplete();
            }
            return;
        } catch (...) {
            trace_end(VK_ERROR_UNKNOWN, true);
            throw;
        }

        // Log compute pipeline creation
        if (GPU::Logging::IsActive()) {
            GPU::Logging::GPULogger::GetInstance().LogPipelineStateChange(
                "ComputePipeline created"
            );
        }

        if (pipeline_statistics) {
            pipeline_statistics->Collect(device, *pipeline);
        }
        // Nothing reads the module once the pipeline exists; the driver keeps its SPIR-V while it
        // lives.
        spv_module.reset();
        std::scoped_lock lock{build_mutex};
        is_built = true;
        build_condvar.notify_one();
        if (shader_notify) {
            shader_notify->MarkShaderComplete();
        }
    }};
    if (thread_worker) {
        thread_worker->QueueWork(std::move(func));
    } else {
        func();
    }
}

bool ComputePipeline::Configure(Tegra::Engines::KeplerCompute& kepler_compute,
                                Tegra::MemoryManager& gpu_memory, Scheduler& scheduler,
                                BufferCache& buffer_cache, TextureCache& texture_cache,
                                FrameDump* frame_dump, u64 native_key,
                                const FrameDump* range_control, u64 range_native_key) {
    // TEMP diagnostic (EDEN_NX2_FOG_SCALE_LOG): OoT's fog volume fractions, c3 +0x200..+0x208.
    if (static const bool fog_log = std::getenv("EDEN_NX2_FOG_SCALE_LOG") != nullptr;
        fog_log && shader_hash == 0x780c0536b89fc89eULL) {
        static u32 count{};
        if (count++ % 30 == 0) {
            const auto& cb = kepler_compute.launch_description.const_buffer_config[3];
            const GPUVAddr address = (GPUVAddr{cb.address_high} << 32) | cb.address_low;
            const f32 x = std::bit_cast<f32>(gpu_memory.Read<u32>(address + 0x200));
            const f32 y = std::bit_cast<f32>(gpu_memory.Read<u32>(address + 0x204));
            const f32 z = std::bit_cast<f32>(gpu_memory.Read<u32>(address + 0x208));
            LOG_WARNING(Render_Vulkan, "fog volume fractions {:.4f} {:.4f} {:.4f} (cbuf {:#x})", x,
                        y, z, address);
        }
    }
    // Metadata capture is bounded; retain an independent guard for this known pipeline.
    if (FrameDump::ComputeCaptureSkyEnabled() && !FrameDump::DelayedSkyCaptureEnabled() &&
        shader_hash == 0x45384f519a8eacd3ULL)
        throw std::runtime_error("DIAGNOSTIC: sky target was not captured; native9694 remains blocked");
    const u64 capture_hash = FrameDump::ComputeCaptureSkyEnabled()
        ? 0xb76d7b3e9e93093dULL : 0x45384f519a8eacd3ULL;
    if (FrameDump::ComputeCaptureStopEnabled() &&
        ((shader_hash == capture_hash && native_key != FrameDump::ComputeCaptureStopNativeKey()) ||
         (native_key == FrameDump::ComputeCaptureStopNativeKey() && shader_hash != capture_hash))) {
        throw std::runtime_error("DIAGNOSTIC capture-stop target identity unavailable; kernel not dispatched");
    }
    // Dirty GPU-written roots or table entries can require downloads. Finish ALL such reads
    // before any compute binding/upload, so a later pointer read cannot invalidate an earlier UBO.
    std::array<std::optional<GPUVAddr>, Shader::Info::MAX_SSBOS> resolved_pointers{};
    ASSERT(info.storage_buffers_descriptors.size() <= resolved_pointers.size());
    size_t pointer_index{};
    for (const auto& desc : info.storage_buffers_descriptors) {
        ASSERT(desc.count == 1);
        if (kepler_compute.IsNativeMode() && (desc.pointer_offset == ~0U || !desc.is_written)) {
            resolved_pointers[pointer_index] = buffer_cache.ResolveComputeStorageBufferPointer(
                desc.cbuf_index, desc.cbuf_offset, desc.pointer_offset);
        }
        ++pointer_index;
    }
    guest_descriptor_queue.Acquire(scheduler, num_descriptor_entries, uses_descriptor_buffer);

    buffer_cache.SetComputeUniformBufferState(info.constant_buffer_mask, &uniform_buffer_sizes);
    buffer_cache.UnbindComputeStorageBuffers();
    std::optional<std::array<BufferCache::StorageRangeDiagnostic, Shader::Info::MAX_SSBOS>> range_diagnostics;
    // The persistent range control is independent of the one-dispatch readback pointer.
    const bool range_identity_matches = native_key == 0 || native_key == range_native_key;
    const auto domain_override = range_control ? range_control->NativeDomainRangeOverride()
                                              : std::optional<bool>{};
    const bool native_domain_mode = range_identity_matches && kepler_compute.IsNativeMode() &&
        OotNativeDomainPolicyEnabled(domain_override, OotNativeDomainBaselineRequested(),
                                    FrameDump::ComputeCaptureStopEnabled()) &&
        std::ranges::any_of(info.storage_buffers_descriptors, [&](const auto& desc) {
            return desc.count == 1 && IsOotNativeDomainTarget(Settings::GetCurrentProgramID(),
                range_native_key, shader_hash, desc.cbuf_index, desc.cbuf_offset,
                desc.pointer_offset, desc.is_written);
        });
    std::optional<std::array<BufferCache::NativeStorageDomainResult, Shader::Info::MAX_SSBOS>> domain_results;
    if (native_domain_mode) domain_results.emplace();
    if (!native_domain_mode && range_control && range_identity_matches && kepler_compute.IsNativeMode() &&
        range_control->LightingRangeMinimum(range_native_key, shader_hash, 0, 96, 24, false)) {
        range_diagnostics.emplace();
    }
    size_t ssbo_index{};
    for (const auto& desc : info.storage_buffers_descriptors) {
        ASSERT(desc.count == 1);
        const u32 diagnostic_minimum = !native_domain_mode && range_control && range_identity_matches && kepler_compute.IsNativeMode()
            ? range_control->LightingRangeMinimum(range_native_key, shader_hash, desc.cbuf_index,
                desc.cbuf_offset, desc.pointer_offset, desc.is_written) : 0;
        buffer_cache.storage_feedback_end = info.uses_storage_feedback && !desc.is_written
            ? StorageFeedback::Get()->Read(info.storage_feedback_offset, ssbo_index) : 0;
        buffer_cache.BindComputeStorageBuffer(ssbo_index, desc.cbuf_index, desc.cbuf_offset,
                                              desc.is_written, desc.pointer_offset,
                                              resolved_pointers[ssbo_index], kepler_compute.IsNativeMode(),
                                              diagnostic_minimum,
                                              diagnostic_minimum ? &(*range_diagnostics)[ssbo_index] : nullptr);
        ++ssbo_index;
    }
    buffer_cache.storage_feedback_end = 0;

    texture_cache.SynchronizeDescriptors(true);

    boost::container::small_vector<VideoCommon::ImageViewInOut, 64> views;
    boost::container::small_vector<VideoCommon::SamplerId, 64> samplers;

    const auto& qmd{kepler_compute.launch_description};
    const auto& cbufs{qmd.const_buffer_config};
    const bool via_header_index{qmd.linked_tsc != 0};
    const auto read_handle{[&](const auto& desc, u32 index) {
        if (desc.handle_program != ~0U) {
            // Read by the shader from a storage buffer: computed for this dispatch.
            return TexturePair(VideoCommon::EvaluateTextureHandle(
                                   info, desc.handle_program, gpu_memory,
                                   [&](u32 cbuf) -> GPUVAddr {
                                       return cbuf < cbufs.size() &&
                                                      ((qmd.const_buffer_enable_mask >> cbuf) &
                                                       1) != 0
                                                  ? cbufs[cbuf].Address()
                                                  : GPUVAddr{};
                                   }),
                               via_header_index);
        }
        ASSERT(((qmd.const_buffer_enable_mask >> desc.cbuf_index) & 1) != 0);
        const u32 index_offset{index << desc.size_shift};
        const u32 offset{desc.cbuf_offset + index_offset};
        const GPUVAddr addr{cbufs[desc.cbuf_index].Address() + offset};
        if constexpr (std::is_same_v<decltype(desc), const Shader::TextureDescriptor&> ||
                      std::is_same_v<decltype(desc), const Shader::TextureBufferDescriptor&>) {
            if (desc.has_secondary) {
                ASSERT(((qmd.const_buffer_enable_mask >> desc.secondary_cbuf_index) & 1) != 0);
                const u32 secondary_offset{desc.secondary_cbuf_offset + index_offset};
                const GPUVAddr separate_addr{cbufs[desc.secondary_cbuf_index].Address() +
                                             secondary_offset};
                const u32 lhs_raw{gpu_memory.Read<u32>(addr) << desc.shift_left};
                const u32 rhs_raw{gpu_memory.Read<u32>(separate_addr) << desc.secondary_shift_left};
                return TexturePair(lhs_raw | rhs_raw, via_header_index);
            }
        }
        return TexturePair(gpu_memory.Read<u32>(addr), via_header_index);
    }};
    const auto add_image{[&](const auto& desc, bool blacklist) {
        for (u32 index = 0; index < desc.count; ++index) {
            const auto handle{read_handle(desc, index)};
            views.push_back({
                .index = handle.first,
                .blacklist = blacklist,
                .id = {},
            });
        }
    }};
    for (const auto& desc : info.texture_buffer_descriptors) {
        add_image(desc, false);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        add_image(desc, false);
    }
    for (const auto& desc : info.texture_descriptors) {
        for (u32 index = 0; index < desc.count; ++index) {
            const auto handle{VideoCommon::HandleIsStoragePointer(info, desc, index)
                                  ? VideoCommon::NULL_TEXTURE_HANDLE
                                  : read_handle(desc, index)};
            views.push_back({handle.first});

            VideoCommon::SamplerId sampler = texture_cache.GetSamplerId(handle.second, true);
            samplers.push_back(sampler);
        }
    }
    for (const auto& desc : info.image_descriptors) {
        add_image(desc, desc.is_written);
    }
    texture_cache.FillImageViews(std::span(views.data(), views.size()), true);

    buffer_cache.UnbindComputeTextureBuffers();
    size_t index{};
    const auto add_buffer{[&](const auto& desc) {
        constexpr bool is_image = std::is_same_v<decltype(desc), const ImageBufferDescriptor&>;
        for (u32 i = 0; i < desc.count; ++i) {
            bool is_written{false};
            if constexpr (is_image) {
                is_written = desc.is_written;
            }
            ImageView& image_view = texture_cache.GetImageView(views[index].id);
            PixelFormat format{image_view.format};
            if constexpr (is_image) {
                if (const auto explicit_format{PixelFormatFromImageFormat(desc.format)}) {
                    format = *explicit_format;
                }
            }
            buffer_cache.BindComputeTextureBuffer(index, image_view.GpuAddr(),
                                                  image_view.BufferSize(), format,
                                                  is_written, is_image);
            ++index;
        }
    }};
    std::ranges::for_each(info.texture_buffer_descriptors, add_buffer);
    std::ranges::for_each(info.image_buffer_descriptors, add_buffer);

    if (native_domain_mode) {
        u32 domain_index{};
        for (const auto& desc : info.storage_buffers_descriptors) {
            if (desc.count == 1 && IsOotNativeDomainTarget(Settings::GetCurrentProgramID(),
                    range_native_key, shader_hash, desc.cbuf_index, desc.cbuf_offset,
                    desc.pointer_offset, desc.is_written)) {
                (*domain_results)[domain_index] = buffer_cache.PrepareComputeReadOnlyDomain(
                    domain_index, resolved_pointers[domain_index].value_or(0));
            }
            ++domain_index;
        }
    }
    buffer_cache.UpdateComputeBuffers(domain_results
        ? std::span{*domain_results} : std::span<BufferCache::NativeStorageDomainResult>{});
    if (native_domain_mode) {
        u32 domain_index{};
        for (const auto& desc : info.storage_buffers_descriptors) {
            if (desc.count == 1 && IsOotNativeDomainTarget(Settings::GetCurrentProgramID(),
                    range_native_key, shader_hash, desc.cbuf_index, desc.cbuf_offset,
                    desc.pointer_offset, desc.is_written)) {
                const auto& domain = (*domain_results)[domain_index];
                static std::atomic<u32> success_reports{}, failure_reports{};
                auto& reports = domain.applied ? success_reports : failure_reports;
                if (reports.load(std::memory_order_relaxed) < 16 &&
                    reports.fetch_add(1, std::memory_order_relaxed) < 16) {
                    const auto message = fmt::format(
                        "Native readonly domain shader {:016x} child {} applied={} reason={} "
                        "old={} bound={} candidate={} scan={} contiguous={} owner={}",
                        range_native_key, desc.pointer_offset, domain.applied, domain.reason,
                        domain.previous_size, domain.bound_size, domain.candidate_span, domain.scan_span,
                        domain.contiguous_span, domain.owner_suffix);
                    if (domain.applied) {
                        LOG_INFO(Render_Vulkan, "{}", message);
                    } else {
                        LOG_WARNING(Render_Vulkan, "{}", message);
                    }
                }
            }
            ++domain_index;
        }
    }
    // Proofs limit only dirty marking. Keep every read binding, upload and usage extent intact.
    std::array<u32, Shader::Info::MAX_SSBOS> dirty_limits{};
    if (Shader::ComputeWriteBoundsEnabled() && kepler_compute.IsNativeMode() &&
        Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL) {
        const std::array<u32, 3> local{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z};
        const std::array<u32, 3> groups{qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z};
        const bool indirect = kepler_compute.GetIndirectComputeAddress().has_value();
        for (const auto& proof : write_bounds.storage) {
            if (proof.index >= info.storage_buffers_descriptors.size() ||
                proof.index >= dirty_limits.size()) continue;
            const auto& desc = info.storage_buffers_descriptors[proof.index];
            if (!desc.is_written || desc.count != 1 || desc.pointer_offset != ~0U ||
                desc.cbuf_index != proof.cbuf_index || desc.cbuf_offset != proof.cbuf_offset)
                continue;
            dirty_limits[proof.index] = Shader::ComputeWriteBoundEnd(proof,
                write_bounds.workgroup_size, local, groups, indirect,
                buffer_cache.runtime.GetStorageBufferAlignment()).value_or(0);
        }
    }
    // This path synchronizes and binds the current (possibly expanded) readonly size.
    if (VideoCommon::Nx2WriteProfile::Enabled()) {
        const auto writer = VideoCommon::Nx2WriteProfile::MakeStage(
            info, shader_hash, range_native_key, 5);
        buffer_cache.BindHostComputeBuffers(&writer, dirty_limits);
    } else {
        buffer_cache.BindHostComputeBuffers(nullptr, dirty_limits);
    }
    if (buffer_cache.any_buffer_uploaded) {
        buffer_cache.runtime.PostCopyBarrier();
        buffer_cache.any_buffer_uploaded = false;
    }

    RescalingPushConstant rescaling;
    const VideoCommon::SamplerId* samplers_it{samplers.data()};
    const VideoCommon::ImageViewInOut* views_it{views.data()};
    PushImageDescriptors(texture_cache, guest_descriptor_queue, info, rescaling, samplers_it,
                         views_it);
    // range_control is the frame dump itself, passed whatever the capture mode.
    if (FrameDump* const dump = const_cast<FrameDump*>(range_control); dump && dump->Capturing()) {
        size_t view_index = Shader::NumDescriptors(info.texture_buffer_descriptors) +
            Shader::NumDescriptors(info.image_buffer_descriptors) +
            Shader::NumDescriptors(info.texture_descriptors);
        u32 slot{};
        for (const auto& desc : info.image_descriptors) {
            for (u32 element = 0; element < desc.count; ++element, ++view_index, ++slot) {
                if (desc.is_written && view_index < views.size()) {
                    dump->NoteWrittenImage(shader_hash,
                                                 texture_cache.GetImageView(views[view_index].id),
                                                 slot);
                }
            }
        }
    }

    if (!is_built.load(std::memory_order::relaxed)) {
        // Wait for the pipeline to be built
        scheduler.Record([this](vk::CommandBuffer) {
            std::unique_lock lock{build_mutex};
            build_condvar.wait(lock, [this] { return is_built.load(std::memory_order::relaxed); });
        });
    }

    // Log compute pipeline binding
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogPipelineBind(true, "compute pipeline");
    }

    const DescriptorUpdateEntry* const descriptor_data{guest_descriptor_queue.UpdateData()};
    if (frame_dump && frame_dump->CapturingLightingInputs() && native_key == 0xd9b1657f3a0b4bc2ULL) {
        size_t view_index = Shader::NumDescriptors(info.texture_buffer_descriptors) +
            Shader::NumDescriptors(info.image_buffer_descriptors) + Shader::NumDescriptors(info.texture_descriptors);
        size_t entry_index = Shader::NumDescriptors(info.constant_buffer_descriptors) +
            Shader::NumDescriptors(info.storage_buffers_descriptors) + view_index;
        for (const auto& desc : info.image_descriptors) {
            for (u32 array_index = 0; array_index < desc.count; ++array_index, ++view_index, ++entry_index) {
                if (!desc.is_written || desc.count != 1 || desc.cbuf_index != 0 ||
                    (desc.cbuf_offset != 0x98 && desc.cbuf_offset != 0xa0) ||
                    view_index >= views.size() || entry_index >= num_descriptor_entries) continue;
                auto& view = texture_cache.GetImageView(views[view_index].id);
                const auto expected = view.StorageView(desc.type, desc.format);
                const bool matches = expected != VK_NULL_HANDLE &&
                    descriptor_data[entry_index].image.imageView == expected;
                frame_dump->NoteLightingOutput({
                    .texture_index = desc.cbuf_offset == 0x98 ? 0U : 1U,
                    .tic = views[view_index].index, .cbuf_index = desc.cbuf_index,
                    .cbuf_offset = desc.cbuf_offset, .view = &view}, matches);
            }
        }
    }
    const bool particle_inputs = frame_dump && frame_dump->CapturingParticleInputs() &&
        native_key == FrameDump::ProducerNativeKey() &&
        FrameDump::IsParticleProducerPair(native_key, shader_hash);
    if (frame_dump && (frame_dump->CapturingComputeCaptureStop() || particle_inputs ||
                       (frame_dump->CapturingLightingInputs() && native_key == 0xd9b1657f3a0b4bc2ULL) ||
                       (frame_dump->CapturingShInputs() && native_key == 0x203053e20b8c989aULL))) {
        u32 entry_index{};
        u32 shader_binding{};
        for (const auto& desc : info.constant_buffer_descriptors) {
            for (u32 array_index = 0; array_index < desc.count; ++array_index, ++entry_index) {
                if (entry_index >= num_descriptor_entries) {
                    if (frame_dump->CapturingComputeCaptureStop())
                        throw std::runtime_error("DIAGNOSTIC UBO descriptor index exceeds payload; kernel not dispatched");
                    FrameDump::ComputeBufferInput missing;
                    missing.binding_index = shader_binding;
                    missing.cbuf_index = desc.index;
                    missing.skip_reason = "UBO descriptor index exceeds payload";
                    frame_dump->NoteCaptureStopUniformBuffer(missing);
                    continue;
                }
                const auto& entry = descriptor_data[entry_index];
                const auto stream = buffer_cache.runtime.UniformStreamSnapshot();
                FrameDump::ComputeBufferInput uniform;
                uniform.binding_index = shader_binding;
                uniform.cbuf_index = desc.index;
                uniform.buffer = stream.buffer;
                uniform.source_capacity = stream.capacity;
                uniform.descriptor_checked = true;
                uniform.descriptor_entry = entry_index;
                uniform.shader_binding = shader_binding;
                uniform.descriptor_address_mode = uses_descriptor_buffer;
                if (uses_descriptor_buffer) {
                    uniform.descriptor_address = entry.address.address;
                    uniform.descriptor_range = entry.address.range;
                    uniform.descriptor_matches = stream.buffer != VK_NULL_HANDLE &&
                        stream.device_address != 0 && entry.address.address >= stream.device_address;
                    if (uniform.descriptor_matches)
                        uniform.offset = entry.address.address - stream.device_address;
                } else {
                    uniform.descriptor_buffer = entry.buffer.buffer;
                    uniform.descriptor_offset = entry.buffer.offset;
                    uniform.descriptor_range = entry.buffer.range;
                    uniform.offset = entry.buffer.offset;
                    uniform.descriptor_matches = stream.buffer != VK_NULL_HANDLE &&
                        entry.buffer.buffer == stream.buffer;
                }
                uniform.descriptor_matches = uniform.descriptor_matches &&
                    uniform.offset <= stream.capacity && uniform.descriptor_range != 0 &&
                    uniform.descriptor_range <= stream.capacity - uniform.offset;
                if (!uniform.descriptor_matches && desc.count == 1) {
                    const auto cached = buffer_cache.GetComputeUniformBindingSnapshot(desc.index);
                    bool matches = cached.buffer && cached.buffer->Handle() != VK_NULL_HANDLE &&
                        cached.size == uniform.descriptor_range;
                    if (matches && uses_descriptor_buffer) {
                        const VkDeviceAddress base = cached.buffer->DeviceAddress();
                        matches = base != 0 && uniform.descriptor_address >= base &&
                            uniform.descriptor_address - base == cached.offset;
                    } else if (matches) {
                        matches = uniform.descriptor_buffer == cached.buffer->Handle() &&
                            uniform.descriptor_offset == cached.offset;
                    }
                    if (matches) {
                        uniform.buffer = cached.buffer->Handle();
                        uniform.source_capacity = cached.buffer->SizeBytes();
                        uniform.offset = cached.offset;
                        uniform.descriptor_matches = true;
                    }
                    frame_dump->NoteCaptureStopMetadata(fmt::format(
                        R"("descriptor_class":"uniform_buffer_cache_validation","cbuf_index":{},"shader_binding":{},"cache_id":{},"cache_available":{},"matches_descriptor":{},"cache_offset":{},"cache_range":{},"cache_capacity":{},"skip_reason":"{}")",
                        desc.index, shader_binding, cached.buffer_id.Value(), cached.buffer != nullptr,
                        matches, cached.offset, cached.size, cached.buffer ? cached.buffer->SizeBytes() : 0,
                        cached.skip_reason ? cached.skip_reason : ""));
                }
                if (uniform.descriptor_range <= 65536)
                    uniform.size = static_cast<u32>(uniform.descriptor_range);
                else uniform.skip_reason = "64 KiB uniform buffer byte cap";
                frame_dump->NoteCaptureStopUniformBuffer(uniform);
                if (uses_descriptor_buffer) {
                    frame_dump->NoteCaptureStopMetadata(fmt::format(
                        R"("descriptor_class":"uniform_buffer","cbuf_index":{},"shader_binding":{},"array_index":{},"descriptor_entry":{},"address_mode":true,"address":"{:#x}","range":{},"ubo_payload_record_follows":true)",
                        desc.index, shader_binding, array_index, entry_index, entry.address.address, entry.address.range));
                } else {
                    frame_dump->NoteCaptureStopMetadata(fmt::format(
                        R"("descriptor_class":"uniform_buffer","cbuf_index":{},"shader_binding":{},"array_index":{},"descriptor_entry":{},"address_mode":false,"buffer":"{}","offset":{},"range":{},"ubo_payload_record_follows":true)",
                        desc.index, shader_binding, array_index, entry_index, fmt::ptr(entry.buffer.buffer), entry.buffer.offset, entry.buffer.range));
                }
            }
            ++shader_binding;
        }
        frame_dump->NoteCaptureStopMetadata(fmt::format(
            R"("shader_hash":"{:016x}","descriptor_entries":{},"storage_bindings":{},"uniform_entries":{},"uses_descriptor_buffer":{},"uses_push_descriptor":{},"robust_buffer_access":{})",
            shader_hash, num_descriptor_entries, info.storage_buffers_descriptors.size(), entry_index,
            uses_descriptor_buffer, uses_push_descriptor, device.IsRobustBufferAccessEnabled()));
    }
    VkDeviceSize descriptor_buffer_offset{};
    u32 descriptor_buffer_chunk{};
    if (uses_descriptor_buffer) {
        const DescriptorBufferRing::Allocation alloc{
            descriptor_buffer_ring.Allocate(scheduler, descriptor_buffer_layout.size)};
        if (!alloc.host) {
            LOG_DEBUG(Render_Vulkan, "Failed to reserve descriptor memory, skipping dispatch");
            return false;
        }
        WriteDescriptorBuffer(device, descriptor_buffer_layout, descriptor_data, alloc.host);
        descriptor_buffer_offset = alloc.offset;
        descriptor_buffer_chunk = alloc.chunk;
    }

    if (frame_dump) {
        // views begins with texture/image buffers. Only texture_descriptors are sampled inputs;
        // image_descriptors following them include outputs and are deliberately excluded.
        size_t sampled_index{};
        for (const auto& desc : info.texture_buffer_descriptors) sampled_index += desc.count;
        for (const auto& desc : info.image_buffer_descriptors) sampled_index += desc.count;
        std::array<FrameDump::ComputeInput, 16> inputs{};
        const size_t input_limit = native_key == FrameDump::ProducerNativeKey() ? inputs.size() : 3;
        size_t input_count{};
        size_t sampled_entry = Shader::NumDescriptors(info.constant_buffer_descriptors) +
            Shader::NumDescriptors(info.storage_buffers_descriptors) +
            Shader::NumDescriptors(info.texture_buffer_descriptors) +
            Shader::NumDescriptors(info.image_buffer_descriptors);
        size_t sampled_binding = info.constant_buffer_descriptors.size() +
            info.storage_buffers_descriptors.size() + info.texture_buffer_descriptors.size() +
            info.image_buffer_descriptors.size();
        for (const auto& desc : info.texture_descriptors) {
            for (u32 array_index = 0; array_index < desc.count && input_count < input_limit; ++array_index) {
                if (sampled_index >= views.size()) break;
                const auto& sampled = views[sampled_index++];
                ImageView& sampled_view = texture_cache.GetImageView(sampled.id);
                bool capture_view_matches = true;
                if (frame_dump->CapturingComputeCaptureStop()) {
                    if (sampled_entry >= num_descriptor_entries) {
                        throw std::runtime_error("DIAGNOSTIC sampled descriptor index exceeds payload; kernel not dispatched");
                    }
                    // Image entries use the image union member in both descriptor modes.
                    const auto& entry = descriptor_data[sampled_entry].image;
                    const VkImageView expected_view = sampled_view.Handle(desc.type);
                    capture_view_matches = expected_view != VK_NULL_HANDLE && entry.imageView == expected_view;
                    const auto& range = sampled_view.range;
                    frame_dump->NoteCaptureStopMetadata(fmt::format(
                        R"("descriptor_class":"sampled_texture","texture_index":{},"descriptor_entry":{},"shader_binding":{},"array_index":{},"tic":{},"cbuf_index":{},"cbuf_offset":{},"texture_type":{},"actual_image_view":"{}","actual_sampler":"{}","actual_layout":{},"snapshot_image_view":"{}","matches_snapshot":{},"view_image":"{}","view_pixel_format":{},"view_type":{},"view_size":[{},{},{}],"view_base_level":{},"view_levels":{},"view_base_layer":{},"view_layers":{})",
                        input_count, sampled_entry, sampled_binding, array_index, sampled.index,
                        desc.cbuf_index, desc.cbuf_offset + (array_index << desc.size_shift),
                        static_cast<u32>(desc.type), fmt::ptr(entry.imageView), fmt::ptr(entry.sampler),
                        static_cast<u32>(entry.imageLayout), fmt::ptr(expected_view), capture_view_matches,
                        fmt::ptr(sampled_view.ImageHandle()), static_cast<u32>(sampled_view.format),
                        static_cast<u32>(sampled_view.type), sampled_view.size.width, sampled_view.size.height,
                        sampled_view.size.depth, range.base.level, range.extent.levels,
                        range.base.layer, range.extent.layers));
                    if (const Image* image = sampled_view.BackingImage()) {
                        const auto& image_info = image->info;
                        frame_dump->NoteCaptureStopMetadata(fmt::format(
                            R"("descriptor_class":"sampled_backing","texture_index":{},"image":"{}","pixel_format":{},"image_type":{},"size":[{},{},{}],"levels":{},"layers":{},"host_level_offset":{},"samples":{},"sparse":{},"rescaled":{},"usage":{},"flags":{})",
                            input_count, fmt::ptr(image->Handle()), static_cast<u32>(image_info.format),
                            static_cast<u32>(image_info.type), image_info.size.width, image_info.size.height,
                            image_info.size.depth, image_info.resources.levels, image_info.resources.layers,
                            image->host_level_offset, image_info.num_samples, image_info.is_sparse,
                            image->IsRescaled(), image->UsageFlags(), static_cast<u32>(image->flags)));
                    } else {
                        frame_dump->NoteCaptureStopMetadata(fmt::format(
                            R"("descriptor_class":"sampled_backing","texture_index":{},"skipped":"no backing image")", input_count));
                    }
                }
                ++sampled_entry;
                inputs[input_count] = {
                    .texture_index = static_cast<u32>(input_count),
                    .tic = sampled.index,
                    .cbuf_index = desc.cbuf_index,
                    .cbuf_offset = desc.cbuf_offset + (array_index << desc.size_shift),
                    .view = capture_view_matches ? &sampled_view : nullptr,
                };
                ++input_count;
            }
            ++sampled_binding;
            if (input_count == input_limit) break;
        }
        if (native_key == FrameDump::ProducerNativeKey()) {
            const bool expanded_producer = native_key != 0xe883414e8c9be33bULL;
            std::array<FrameDump::ComputeBufferInput, 64> storage_inputs{};
            const size_t storage_limit = frame_dump->CapturingComputeCaptureStop() ? 8 :
                expanded_producer ? storage_inputs.size() : 2;
            size_t storage_input_count{};
            // Keep the first proven missing vegetation pointer and its child in the bounded
            // capture even when newly discovered descriptors push them past the first eight.
            std::array<u32, 64> selected_bindings{};
            size_t selected_count{};
            const bool vegetation = native_key == 0xd6dcff9e09b573e4ULL;
            for (u32 pass = 0; pass < 2; ++pass) {
                u32 binding_cursor{};
                for (const auto& desc : info.storage_buffers_descriptors) {
                    const u32 binding = binding_cursor++;
                    if (!expanded_producer && (desc.count != 1 || desc.cbuf_index != 0 ||
                        (desc.cbuf_offset != 0x40 && desc.cbuf_offset != 0x50))) continue;
                    const bool priority = vegetation && desc.cbuf_index == 4 &&
                        desc.cbuf_offset == 0x20;
                    if (priority != (pass == 0) || selected_count == storage_limit) continue;
                    selected_bindings[selected_count++] = binding;
                }
            }
            u32 range_witness_mask{};
            for (const u32 binding_index : std::span{selected_bindings.data(), selected_count}) {
                const auto& desc = info.storage_buffers_descriptors[binding_index];
                if (!expanded_producer && (desc.count != 1 || desc.cbuf_index != 0 ||
                    (desc.cbuf_offset != 0x40 && desc.cbuf_offset != 0x50))) continue;
                const auto snapshot = buffer_cache.GetComputeStorageBindingSnapshot(binding_index);
                const bool range_witness = kepler_compute.IsNativeMode() &&
                    Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL &&
                    !frame_dump->ProducerInputsRecorded() &&
                    ((frame_dump->CapturingLightingInputs() && native_key == 0xd9b1657f3a0b4bc2ULL &&
                      shader_hash == 0x75d4369f3b1f5868ULL) ||
                     (frame_dump->CapturingShInputs() && native_key == 0x203053e20b8c989aULL &&
                      shader_hash == 0xc73574a9766531cbULL)) &&
                    desc.count == 1 && !desc.is_written && desc.cbuf_index == 0 &&
                    desc.cbuf_offset == 96 && (desc.pointer_offset == 24 || desc.pointer_offset == 32);
                const u32 range_witness_bit = desc.pointer_offset == 24 ? 1U : 2U;
                if (range_witness && (range_witness_mask & range_witness_bit) == 0) {
                    range_witness_mask |= range_witness_bit;
                    if (domain_results) {
                        const auto& applied = (*domain_results)[binding_index];
                        frame_dump->NoteCaptureStopMetadata(fmt::format(
                            R"("descriptor_class":"native_domain_binding","native_key":"{:016x}","shader_hash":"{:016x}","binding_index":{},"pointer_table_offset":{},"applied":{},"reason":"{}","previous_size":{},"bound_size":{},"candidate_span":{},"scan_span":{},"contiguous_span":{},"owner_suffix":{})",
                            native_key, shader_hash, binding_index, desc.pointer_offset, applied.applied,
                            applied.reason, applied.previous_size, applied.bound_size, applied.candidate_span,
                            applied.scan_span, applied.contiguous_span, applied.owner_suffix));
                    }
                    const auto domain = buffer_cache.GetNativeStorageRangeWitness(
                        resolved_pointers[binding_index].value_or(0), snapshot);
                    frame_dump->NoteCaptureStopMetadata(fmt::format(
                        R"("descriptor_class":"native_range_domain_witness","native_key":"{:016x}","shader_hash":"{:016x}","binding_index":{},"pointer_table_offset":{},"pointer":{},"aligned_pointer":{},"alignment":{},"alignment_prefix":{},"pointer_kind_run_suffix":{},"aligned_kind_run_suffix":{},"backend_limit":{},"u32_limit":{},"address_space_limit":{},"candidate_span":{},"scan_request":{},"scan_capped":{},"physical_span":{},"accessible_span":{},"accessible_span_is_lower_bound":{},"host_matches":{},"current_bound_size":{},"current_buffer_offset":{},"current_buffer_capacity":{},"host_suffix":{},"uncovered_bytes":{},"valid":{},"reason":"{}")",
                        native_key, shader_hash, binding_index, desc.pointer_offset, domain.pointer, domain.aligned_pointer,
                        domain.alignment, domain.alignment_prefix, domain.pointer_kind_suffix,
                        domain.aligned_kind_suffix, domain.backend_limit, (std::numeric_limits<u32>::max)(),
                        domain.address_space_limit, domain.candidate_span, domain.scan_request,
                        domain.scan_capped, domain.physical_span, domain.accessible_span,
                        domain.valid && domain.scan_capped && domain.physical_span == domain.scan_request,
                        domain.host_matches, snapshot.size, snapshot.offset,
                        snapshot.buffer ? snapshot.buffer->SizeBytes() : 0,
                        domain.host_suffix, domain.uncovered_bytes, domain.valid, domain.reason));
                }
                if (frame_dump->CapturingLightingInputs() && range_diagnostics &&
                    (*range_diagnostics)[binding_index].requested != 0) {
                    const auto& range = (*range_diagnostics)[binding_index];
                    frame_dump->NoteCaptureStopMetadata(fmt::format(
                        R"("descriptor_class":"lighting_range_diagnostic","binding_index":{},"pointer_table_offset":{},"requested_minimum":{},"applied":{},"reason":"{}","mapped_extent":{},"physical_contiguous_extent":{},"max_storage_buffer_range":{},"alignment_prefix":{},"actual_bound_size":{},"actual_buffer_capacity":{})",
                        binding_index, desc.pointer_offset, range.requested, range.applied,
                        range.reason, range.mapped_size, range.contiguous_size,
                        range.max_descriptor_range, range.alignment_prefix, snapshot.size,
                        snapshot.buffer ? snapshot.buffer->SizeBytes() : 0));
                }
                auto& input = storage_inputs[storage_input_count++];
                input.binding_index = binding_index;
                input.is_written = desc.is_written;
                input.cbuf_index = desc.cbuf_index;
                input.cbuf_offset = desc.cbuf_offset;
                input.pointer_table_offset = desc.pointer_offset;
                input.device_addr = snapshot.device_addr;
                input.total_bindings = static_cast<u32>(info.storage_buffers_descriptors.size());
                input.binding_selection_truncated = selected_count < info.storage_buffers_descriptors.size();
                const size_t copy_limit = particle_inputs || (frame_dump->CapturingShInputs() &&
                    native_key == 0x203053e20b8c989aULL) ? 10 : 8;
                input.capture_requested = !expanded_producer || storage_input_count <= copy_limit;
                input.buffer = snapshot.buffer ? snapshot.buffer->Handle() : VK_NULL_HANDLE;
                input.source_capacity = snapshot.buffer ? snapshot.buffer->SizeBytes() : 0;
                input.offset = snapshot.offset;
                input.size = snapshot.size;
                input.gpu_addr = snapshot.gpu_addr;
                input.segment_count = snapshot.segment_count;
                input.skip_reason = snapshot.skip_reason;
                if (desc.count != 1) input.skip_reason = "descriptor count is not one";
                if (frame_dump->CapturingComputeCaptureStop() || frame_dump->CapturingLightingInputs() ||
                    frame_dump->CapturingShInputs() || particle_inputs) {
                    size_t entry_index = Shader::NumDescriptors(info.constant_buffer_descriptors);
                    for (u32 prior = 0; prior < binding_index; ++prior)
                        entry_index += info.storage_buffers_descriptors[prior].count;
                    input.descriptor_checked = true;
                    input.descriptor_entry = static_cast<u32>(entry_index);
                    input.shader_binding = static_cast<u32>(info.constant_buffer_descriptors.size()) + binding_index;
                    input.descriptor_address_mode = uses_descriptor_buffer;
                    if (entry_index < num_descriptor_entries) {
                        const auto& entry = descriptor_data[entry_index];
                        if (uses_descriptor_buffer) {
                            input.descriptor_address = entry.address.address;
                            input.descriptor_range = entry.address.range;
                            const u64 base = snapshot.buffer ? snapshot.buffer->DeviceAddress() : 0;
                            input.descriptor_matches = base != 0 && base <= std::numeric_limits<u64>::max() - snapshot.offset &&
                                entry.address.address == base + snapshot.offset && entry.address.range == snapshot.size;
                        } else {
                            input.descriptor_buffer = entry.buffer.buffer;
                            input.descriptor_offset = entry.buffer.offset;
                            input.descriptor_range = entry.buffer.range;
                            input.descriptor_matches = snapshot.buffer && entry.buffer.buffer == snapshot.buffer->Handle() &&
                                entry.buffer.offset == snapshot.offset && entry.buffer.range == snapshot.size;
                        }
                    }
                }
                // Advisory original-pointer snapshot; the copy always uses the actual bound range.
                if (desc.cbuf_index >= cbufs.size()) continue;
                const auto& cb = cbufs[desc.cbuf_index];
                const bool offset_valid = desc.cbuf_offset <= cb.size &&
                    sizeof(u64) <= cb.size - desc.cbuf_offset;
                const GPUVAddr base = cb.Address();
                const bool address_valid = base <= (std::numeric_limits<GPUVAddr>::max)() - desc.cbuf_offset;
                const GPUVAddr address = address_valid ? base + desc.cbuf_offset : 0;
                input.pointer_valid = ((qmd.const_buffer_enable_mask >> desc.cbuf_index) & 1) != 0 &&
                    offset_valid && address_valid &&
                    address <= (std::numeric_limits<GPUVAddr>::max)() - (sizeof(u64) - 1) &&
                    gpu_memory.IsWithinGPUAddressRange(address) &&
                    gpu_memory.IsWithinGPUAddressRange(address + sizeof(u64) - 1) &&
                    gpu_memory.IsFullyMappedRange(address, sizeof(u64));
                if (input.pointer_valid)
                    gpu_memory.ReadBlockUnsafe(address, &input.pointer, sizeof(input.pointer));
            }
            frame_dump->NoteProducerInputs(native_key, std::span{inputs.data(), input_count},
                                           std::span{storage_inputs.data(), storage_input_count});
        } else {
            frame_dump->NoteComputeInputs(native_key, std::span{inputs.data(), input_count});
        }
    }

    const bool bind_descriptor_buffer{
        uses_descriptor_buffer && scheduler.UpdateDescriptorBufferChunk(descriptor_buffer_chunk)};

    const bool is_rescaling = !info.texture_descriptors.empty() || !info.image_descriptors.empty();
    scheduler.Record([this, descriptor_data, is_rescaling, descriptor_buffer_offset,
                      descriptor_buffer_chunk, bind_descriptor_buffer,
                      rescaling_data = rescaling.Data()](vk::CommandBuffer cmdbuf) {
        if (bind_descriptor_buffer) {
            const VkDescriptorBufferBindingInfoEXT binding_info{
                descriptor_buffer_ring.BindingInfo(descriptor_buffer_chunk)};
            cmdbuf.BindDescriptorBuffersEXT(binding_info);
        }
        if (!pipeline) {
            return;
        }
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
        if (!descriptor_set_layout) {
            return;
        }
        if (is_rescaling) {
            cmdbuf.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 RESCALING_LAYOUT_WORDS_OFFSET, sizeof(rescaling_data),
                                 rescaling_data.data());
        }
        if (uses_descriptor_buffer) {
            const u32 buffer_index{};
            cmdbuf.SetDescriptorBufferOffsetsEXT(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout,
                                                 0, buffer_index, descriptor_buffer_offset);
        } else if (uses_push_descriptor) {
            cmdbuf.PushDescriptorSetWithTemplateKHR(*descriptor_update_template, *pipeline_layout,
                                                    0, descriptor_data);
        } else {
            const VkDescriptorSet descriptor_set{descriptor_allocator.Commit()};
            const vk::Device& dev{device.GetLogical()};
            dev.UpdateDescriptorSet(descriptor_set, *descriptor_update_template, descriptor_data);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0,
                                      descriptor_set, nullptr);
        }
    });
    return true;
}

} // namespace Vulkan
