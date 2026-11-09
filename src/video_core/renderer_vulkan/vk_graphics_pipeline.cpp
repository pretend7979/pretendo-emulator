// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <charconv>
#include <optional>
#include <string_view>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <span>

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <fmt/format.h>

#include "video_core/renderer_vulkan/pipeline_helper.h"

#include "common/bit_field.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_frame_dump.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_storage_feedback.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_handle_program.h"
#include "video_core/shader_notify.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/nx2_profile.h"

#if defined(__clang__) && defined(NDEBUG)
#define LAMBDA_FORCEINLINE __attribute__((always_inline))
#elif defined(_MSC_VER) && defined(NDEBUG)
#define LAMBDA_FORCEINLINE [[msvc::forceinline]]
#else
#define LAMBDA_FORCEINLINE
#endif

namespace Vulkan {
namespace {

// Diagnostic only; inspect bounded metadata for one explicitly selected shader.
std::optional<u64> BindlessPoolProbeShader() {
    static const std::optional<u64> selected = []() -> std::optional<u64> {
        const char* request = std::getenv("EDEN_NX2_BINDLESS_POOL_PROBE");
        if (!request) request = std::getenv("EDEN_NX2_BINDLESS_DREF_PROBE");
        if (!request) return {};
        const std::string_view text{request};
        if (text.size() != 16 &&
            !(text.size() == 18 && text[16] == ':' && (text[17] == '0' || text[17] == '1'))) {
            return {};
        }
        u64 hash{};
        const auto result = std::from_chars(text.data(), text.data() + 16, hash, 16);
        if (result.ec != std::errc{} || result.ptr != text.data() + 16) return {};
        return hash;
    }();
    return selected;
}

std::atomic<u32> bindless_pool_probe_reports{};

using boost::container::small_vector;
using boost::container::static_vector;
using Shader::ImageBufferDescriptor;
using Shader::Backend::SPIRV::RENDERAREA_LAYOUT_OFFSET;
using Shader::Backend::SPIRV::RESCALING_LAYOUT_DOWN_FACTOR_OFFSET;
using Shader::Backend::SPIRV::RESCALING_LAYOUT_WORDS_OFFSET;
using Tegra::Texture::TexturePair;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::PixelFormatFromDepthFormat;
using VideoCore::Surface::PixelFormatFromRenderTargetFormat;

constexpr size_t NUM_STAGES = Maxwell::MaxShaderStage;

// The stage of a program: task and mesh programs sit in the vertex and tessellation slots.
VkShaderStageFlagBits StageFlag(const Shader::Info& info, VkShaderStageFlagBits slot_stage) {
    if (info.is_task) {
        return VK_SHADER_STAGE_TASK_BIT_EXT;
    }
    if (info.is_mesh) {
        return VK_SHADER_STAGE_MESH_BIT_EXT;
    }
    return slot_stage;
}

DescriptorLayoutBuilder MakeBuilder(const Device& device, std::span<const Shader::Info> infos) {
    DescriptorLayoutBuilder builder{device};
    for (size_t index = 0; index < infos.size(); ++index) {
        static constexpr std::array stages{
            VK_SHADER_STAGE_VERTEX_BIT,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            VK_SHADER_STAGE_GEOMETRY_BIT,
            VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        builder.Add(infos[index], StageFlag(infos[index], stages.at(index)));
    }
    return builder;
}

template <class StencilFace>
VkStencilOpState GetStencilFaceState(const StencilFace& face) {
    return {
        .failOp = MaxwellToVK::StencilOp(face.ActionStencilFail()),
        .passOp = MaxwellToVK::StencilOp(face.ActionDepthPass()),
        .depthFailOp = MaxwellToVK::StencilOp(face.ActionDepthFail()),
        .compareOp = MaxwellToVK::ComparisonOp(face.TestFunc()),
        .compareMask = 0,
        .writeMask = 0,
        .reference = 0,
    };
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    static constexpr std::array unsupported_topologies{
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
        VK_PRIMITIVE_TOPOLOGY_PATCH_LIST,
        // VK_PRIMITIVE_TOPOLOGY_QUAD_LIST_EXT,
    };
    return std::ranges::find(unsupported_topologies, topology) == unsupported_topologies.end();
}

bool IsLine(VkPrimitiveTopology topology) {
    static constexpr std::array line_topologies{
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
        // VK_PRIMITIVE_TOPOLOGY_LINE_LOOP_EXT,
    };
    return std::ranges::find(line_topologies, topology) != line_topologies.end();
}

VkViewportSwizzleNV UnpackViewportSwizzle(u16 swizzle) {
    union Swizzle {
        u32 raw;
        BitField<0, 3, Maxwell::ViewportSwizzle> x;
        BitField<4, 3, Maxwell::ViewportSwizzle> y;
        BitField<8, 3, Maxwell::ViewportSwizzle> z;
        BitField<12, 3, Maxwell::ViewportSwizzle> w;
    };
    const Swizzle unpacked{swizzle};
    return VkViewportSwizzleNV{
        .x = MaxwellToVK::ViewportSwizzle(unpacked.x),
        .y = MaxwellToVK::ViewportSwizzle(unpacked.y),
        .z = MaxwellToVK::ViewportSwizzle(unpacked.z),
        .w = MaxwellToVK::ViewportSwizzle(unpacked.w),
    };
}

PixelFormat DecodeFormat(u8 encoded_format) {
    const auto format{static_cast<Tegra::RenderTargetFormat>(encoded_format)};
    if (format == Tegra::RenderTargetFormat::NONE) {
        return PixelFormat::Invalid;
    }
    return PixelFormatFromRenderTargetFormat(format);
}

RenderPassKey MakeRenderPassKey(const FixedPipelineState& state, const Device& device) {
    RenderPassKey key{};
    std::ranges::transform(state.color_formats, key.color_formats.begin(), DecodeFormat);
    if (state.depth_enabled != 0) {
        const auto depth_format{static_cast<Tegra::DepthFormat>(state.depth_format.Value())};
        key.depth_format = PixelFormatFromDepthFormat(depth_format);
    } else {
        key.depth_format = PixelFormat::Invalid;
    }
    key.samples = MaxwellToVK::MsaaMode(state.msaa_mode);
    const bool has_color = std::ranges::any_of(key.color_formats, [](PixelFormat format) {
        return format != PixelFormat::Invalid;
    });
    key.resolve_color =
        key.samples != VK_SAMPLE_COUNT_1_BIT && has_color && device.IsTiler();
    return key;
}

size_t NumAttachments(const FixedPipelineState& state) {
    size_t num{};
    for (size_t index = 0; index < Maxwell::NumRenderTargets; ++index) {
        const auto format{static_cast<Tegra::RenderTargetFormat>(state.color_formats[index])};
        if (format != Tegra::RenderTargetFormat::NONE) {
            num = index + 1;
        }
    }
    return num;
}

template <typename Spec>
bool Passes(const std::array<vk::ShaderModule, NUM_STAGES>& modules,
            const std::array<Shader::Info, NUM_STAGES>& stage_infos) {
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!Spec::enabled_stages[stage] && modules[stage]) {
            return false;
        }
        const auto& info{stage_infos[stage]};
        if constexpr (!Spec::has_storage_buffers) {
            if (!info.storage_buffers_descriptors.empty()) {
                return false;
            }
        }
        if constexpr (!Spec::has_texture_buffers) {
            if (!info.texture_buffer_descriptors.empty()) {
                return false;
            }
        }
        if constexpr (!Spec::has_image_buffers) {
            if (!info.image_buffer_descriptors.empty()) {
                return false;
            }
        }
        if constexpr (!Spec::has_images) {
            if (!info.image_descriptors.empty()) {
                return false;
            }
        }
    }
    return true;
}

using ConfigureFuncPtr = bool (*)(GraphicsPipeline*, bool, bool, bool);

template <typename Spec, typename... Specs>
ConfigureFuncPtr FindSpec(const std::array<vk::ShaderModule, NUM_STAGES>& modules,
                          const std::array<Shader::Info, NUM_STAGES>& stage_infos) {
    if constexpr (sizeof...(Specs) > 0) {
        if (!Passes<Spec>(modules, stage_infos)) {
            return FindSpec<Specs...>(modules, stage_infos);
        }
    }
    return GraphicsPipeline::MakeConfigureSpecFunc<Spec>();
}

struct SimpleVertexFragmentSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, true};
    static constexpr bool has_storage_buffers = false;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = false;
};

struct SimpleVertexSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, false};
    static constexpr bool has_storage_buffers = false;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = false;
};

struct SimpleStorageSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, true};
    static constexpr bool has_storage_buffers = true;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = false;
};

struct SimpleImageSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, true};
    static constexpr bool has_storage_buffers = false;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = true;
};

struct DefaultSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, true, true, true, true};
    static constexpr bool has_storage_buffers = true;
    static constexpr bool has_texture_buffers = true;
    static constexpr bool has_image_buffers = true;
    static constexpr bool has_images = true;
};

ConfigureFuncPtr ConfigureFunc(const std::array<vk::ShaderModule, NUM_STAGES>& modules,
                               const std::array<Shader::Info, NUM_STAGES>& infos) {
    return FindSpec<SimpleVertexSpec, SimpleVertexFragmentSpec, SimpleStorageSpec, SimpleImageSpec,
                    DefaultSpec>(modules, infos);
}
} // Anonymous namespace

// TODO(crueter): This is the worst-formatted code I have EVER seen
GraphicsPipeline::GraphicsPipeline(
    Scheduler& scheduler_, BufferCache& buffer_cache_, TextureCache& texture_cache_,
    vk::PipelineCache& pipeline_cache_, VideoCore::ShaderNotify* shader_notify,
    const Device& device_, DescriptorPool& descriptor_pool,
    GuestDescriptorQueue& guest_descriptor_queue_, DescriptorBufferRing& descriptor_buffer_ring_,
    Common::ThreadWorker* worker_thread,
    PipelineStatistics* pipeline_statistics, RenderPassCache& render_pass_cache,
    const GraphicsPipelineCacheKey& key_, std::array<vk::ShaderModule, NUM_STAGES> stages,
    const std::array<const Shader::Info*, NUM_STAGES>& infos,
    const std::array<bool, NUM_STAGES>& native_stages_,
    const std::array<std::array<u32, 3>, NUM_STAGES>& workgroup_sizes_)
    : key{key_}, device{device_}, texture_cache{texture_cache_}, buffer_cache{buffer_cache_},
      pipeline_cache(pipeline_cache_), scheduler{scheduler_},
      guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_}, spv_modules{std::move(stages)},
      native_stages{native_stages_}, workgroup_sizes{workgroup_sizes_} {
    if (shader_notify) {
        shader_notify->MarkShaderBuilding();
    }
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        const Shader::Info* const info{infos[stage]};
        if (!info) {
            continue;
        }
        stage_infos[stage] = *info;
        enabled_uniform_buffer_masks[stage] = info->constant_buffer_mask;
        std::ranges::copy(info->constant_buffer_used_sizes, uniform_buffer_sizes[stage].begin());
        num_image_elements += Shader::NumDescriptors(info->texture_buffer_descriptors);
        num_image_elements += Shader::NumDescriptors(info->image_buffer_descriptors);
        num_textures += Shader::NumDescriptors(info->texture_descriptors);
        num_image_elements += Shader::NumDescriptors(info->texture_descriptors);
        num_image_elements += Shader::NumDescriptors(info->image_descriptors);
        num_descriptor_entries += NumDescriptorEntries(*info);
    }
    fragment_has_color0_output = stage_infos[NUM_STAGES - 1].stores_frag_color[0];
    // The guest leaves render target components the fragment shader doesn't output untouched,
    // while Vulkan writes undefined values to them (AMD writes garbage), so they are masked out.
    // (From Lichii.)
    for (size_t rt = 0; rt < Maxwell::NumRenderTargets; ++rt) {
        color_output_mask |= u32{stage_infos[NUM_STAGES - 1].frag_color_components[rt]}
                             << (rt * 4);
    }

    const auto any_written = [](const auto& descriptors) {
        return std::ranges::any_of(descriptors, [](const auto& desc) { return desc.is_written; });
    };
    writes_storage = std::ranges::any_of(stage_infos, [&](const Shader::Info& info) {
        return info.stores_global_memory || any_written(info.storage_buffers_descriptors) ||
               any_written(info.image_descriptors) || any_written(info.image_buffer_descriptors);
    });

    DescriptorLayoutBuilder builder{MakeBuilder(device, stage_infos)};
    uses_push_descriptor = builder.CanUsePushDescriptor();
    uses_descriptor_buffer = builder.CanUseDescriptorBuffer() && descriptor_buffer_ring.IsValid();
    descriptor_set_layout =
        builder.CreateDescriptorSetLayout(uses_push_descriptor, uses_descriptor_buffer);
    if (uses_descriptor_buffer) {
        descriptor_buffer_layout = builder.MakeDescriptorBufferLayout(*descriptor_set_layout);
        if (!descriptor_buffer_ring.CanAllocate(descriptor_buffer_layout.size)) {
            LOG_WARNING(Render_Vulkan,
                        "Graphics pipeline {:016X} needs {} descriptor bytes per draw, falling back "
                        "to sets",
                        key.Hash(), descriptor_buffer_layout.size);
            uses_descriptor_buffer = false;
            descriptor_buffer_layout = {};
            descriptor_set_layout = builder.CreateDescriptorSetLayout(uses_push_descriptor);
        }
    }

    const VkDescriptorSetLayout set_layout{*descriptor_set_layout};
    pipeline_layout = builder.CreatePipelineLayout(set_layout);
    if (!uses_descriptor_buffer) {
        descriptor_update_template =
            builder.CreateTemplate(set_layout, *pipeline_layout, uses_push_descriptor);
        if (!uses_push_descriptor) {
            descriptor_allocator =
                descriptor_pool.Allocator(device, scheduler, set_layout, stage_infos);
        }
    }

    auto func{[this, shader_notify, &render_pass_cache, pipeline_statistics] {
        const VkRenderPass render_pass{render_pass_cache.Get(MakeRenderPassKey(key.state, device))};
        Validate();
        try {
            MakePipeline(render_pass);
        } catch (const vk::Exception& exception) {
            LOG_CRITICAL(Render_Vulkan, "Graphics pipeline build failed: {}", exception.what());
            std::scoped_lock lock{build_mutex};
            is_built = true;
            build_condvar.notify_one();
            if (shader_notify) {
                shader_notify->MarkShaderComplete();
            }
            return;
        }
        if (pipeline_statistics) {
            pipeline_statistics->Collect(device, *pipeline);
        }
        // Nothing reads the shader modules once the pipeline exists, and the driver keeps a copy
        // of each one's SPIR-V while it lives (title 1: ~1,500 pipelines, 2-5 modules each).
        for (vk::ShaderModule& module : spv_modules) {
            module.reset();
        }

        std::scoped_lock lock{build_mutex};
        is_built = true;
        build_condvar.notify_one();
        if (shader_notify) {
            shader_notify->MarkShaderComplete();
        }
    }};
    // Before the build: it frees the modules this looks at.
    configure_func = ConfigureFunc(spv_modules, stage_infos);
    if (worker_thread) {
        worker_thread->QueueWork(std::move(func));
    } else {
        func();
    }
}

void GraphicsPipeline::AddTransition(GraphicsPipeline* transition) {
    transition_keys.push_back(transition->key);
    transitions.push_back(transition);
}

template <typename Spec>
bool GraphicsPipeline::ConfigureImpl(bool is_indexed, bool,
                                     bool native_pointer_trial) {
    std::optional<BufferCache::GraphicsStorageWriteBounds> writer_bounds;
    buffer_cache.ResetGraphicsUniformWriteSnapshots();
    SCOPE_EXIT { buffer_cache.ResetGraphicsUniformWriteSnapshots(); };
    // Metadata only: GPU copies remain gated by the selected capture pass in the rasterizer.
    static const bool record_sampled = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_SAMPLED");
        return value && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    static const bool task_range_probe = [] {
        const char* value = std::getenv("EDEN_NX2_FOLIAGE_TASK_RANGE");
        return value && std::strcmp(value, "1") == 0;
    }();
    const bool selected_task_range = task_range_probe &&
        Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL &&
        key.unique_hashes[1] == 0xf378ff9ca77352f0ULL &&
        key.unique_hashes[3] == 0x277d8ef0f4003ac4ULL &&
        key.unique_hashes[5] == 0x4b0ca12cb55286d7ULL;
    if (record_sampled) sampled_images.clear();
    struct GuardedPointer {
        size_t stage;
        size_t index;
        BufferCache::GraphicsStoragePointerGuard guard;
    };
    boost::container::small_vector<GuardedPointer, 4> guarded_pointers;
    if constexpr (Spec::has_storage_buffers) {
        // Refresh pointer tables before any stage binds their direct or derived buffers.
        for (size_t stage = 0; stage < stage_infos.size(); ++stage) {
            const auto& info = stage_infos[stage];
            if (!info.is_task && !info.is_mesh) continue;
            const bool has_writes = std::ranges::any_of(info.storage_buffers_descriptors,
                [](const auto& desc) { return desc.is_written; });
            size_t index{};
            for (const auto& desc : info.storage_buffers_descriptors) {
                if (desc.pointer_offset != ~0U) {
                    BufferCache::GraphicsStoragePointerGuard guard;
                    (void)buffer_cache.ResolveGraphicsStorageBufferPointer(
                        stage, desc.cbuf_index, desc.cbuf_offset, desc.pointer_offset);
                    if (has_writes) guarded_pointers.push_back({stage, index, guard});
                }
                ++index;
            }
        }
    }
    boost::container::small_vector<VideoCommon::ImageViewInOut, 160> views;
    boost::container::small_vector<VideoCommon::SamplerId, 96> samplers;
    views.reserve(num_image_elements);
    samplers.reserve(num_textures);

    namespace Profile = VideoCommon::Nx2Profile;
    {
        Profile::Scope scope{Profile::Zone::SyncDescriptors};
        texture_cache.SynchronizeDescriptors(false);
    }

    buffer_cache.SetUniformBuffersState(enabled_uniform_buffer_masks, &uniform_buffer_sizes);

    const auto& regs{maxwell3d->regs};
    const bool via_header_index{regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding};

    if (const auto selected = BindlessPoolProbeShader();
        selected && *selected == key.unique_hashes[static_cast<size_t>(Maxwell::ShaderType::Pixel)] &&
        bindless_pool_probe_reports.load(std::memory_order_relaxed) < 16 &&
        bindless_pool_probe_reports.fetch_add(1, std::memory_order_relaxed) < 16) {
        const Shader::Info& fragment = stage_infos[static_cast<size_t>(Maxwell::ShaderType::Pixel) - 1];
        u32 graphics_entries{};
        u32 graphics_sampled{};
        u32 graphics_samplers{};
        for (const Shader::Info& info : stage_infos) {
            graphics_entries += NumDescriptorEntries(info);
            graphics_sampled += Shader::NumDescriptors(info.texture_descriptors) +
                                Shader::NumDescriptors(info.texture_buffer_descriptors);
            graphics_samplers += Shader::NumDescriptors(info.texture_descriptors);
        }
        LOG_WARNING(Render_Vulkan,
                    "DIAGNOSTIC bindless pool facts shader={:016x} tic_base={:#x} tic_last={} "
                    "tsc_base={:#x} tsc_raw_last={} via_header={} ps_entries={} ps_sampled={} "
                    "ps_samplers={} graphics_entries={} graphics_sampled={} graphics_samplers={} "
                    "limit_stage_images={} limit_stage_samplers={} limit_stage_resources={} "
                    "limit_set_images={} limit_set_samplers={} nonuniform_sampled={}",
                    *selected, regs.tex_header.Address(), regs.tex_header.limit,
                    regs.tex_sampler.Address(), regs.tex_sampler.limit,
                    via_header_index, NumDescriptorEntries(fragment),
                    Shader::NumDescriptors(fragment.texture_descriptors) +
                        Shader::NumDescriptors(fragment.texture_buffer_descriptors),
                    Shader::NumDescriptors(fragment.texture_descriptors), graphics_entries,
                    graphics_sampled, graphics_samplers, device.GetMaxPerStageDescriptorSampledImages(),
                    device.GetMaxPerStageDescriptorSamplers(), device.GetMaxPerStageResources(),
                    device.GetMaxDescriptorSetSampledImages(), device.GetMaxDescriptorSetSamplers(),
                    device.IsSampledImageArrayNonUniformIndexingSupported());
    }

    const auto config_stage{[&](size_t stage) LAMBDA_FORCEINLINE {
        const Shader::Info& info{stage_infos[stage]};
        buffer_cache.UnbindGraphicsStorageBuffers(stage);
        if constexpr (Spec::has_storage_buffers) {
            size_t ssbo_index{};
            for (const auto& desc : info.storage_buffers_descriptors) {
                ASSERT(desc.count == 1);
                const bool native_pointer_table = desc.pointer_offset == ~0U &&
                    (info.is_task || info.is_mesh) &&
                    std::ranges::any_of(info.storage_buffers_descriptors, [&](const auto& child) {
                        return child.pointer_offset != ~0U && child.cbuf_index == desc.cbuf_index &&
                               child.cbuf_offset == desc.cbuf_offset;
                    });
                const u32 diagnostic_minimum = selected_task_range && stage == 0 &&
                    desc.cbuf_index == 0 && desc.cbuf_offset == 0x670 &&
                    desc.pointer_offset == 0 && !desc.is_written ? 16'108'224U : 0U;
                BufferCache::GraphicsStoragePointerGuard* write_guard{};
                if (!guarded_pointers.empty()) {
                    const auto it = std::ranges::find_if(guarded_pointers, [&](const auto& pointer) {
                        return pointer.stage == stage && pointer.index == ssbo_index;
                    });
                    if (it != guarded_pointers.end()) write_guard = &it->guard;
                }
                BufferCache::StorageRangeDiagnostic range_diagnostic;
                buffer_cache.storage_feedback_end = info.uses_storage_feedback && !desc.is_written
                    ? StorageFeedback::Get()->Read(info.storage_feedback_offset, ssbo_index) : 0;
                buffer_cache.BindGraphicsStorageBuffer(stage, ssbo_index, desc.cbuf_index,
                                                       desc.cbuf_offset, desc.is_written,
                                                       desc.pointer_offset,
                                                       native_pointer_table ||
                                                           (native_pointer_trial && native_stages[stage]),
                                                       native_stages[stage],
                                                       diagnostic_minimum,
                                                       diagnostic_minimum ? &range_diagnostic : nullptr, write_guard);
                buffer_cache.storage_feedback_end = 0;
                if (diagnostic_minimum) {
                    static std::atomic<u64> last_pointer{~u64{0}};
                    static std::atomic<u32> reports{};
                    if (reports.load(std::memory_order_relaxed) < 16 &&
                        last_pointer.exchange(range_diagnostic.resolved_pointer,
                                              std::memory_order_relaxed) !=
                            range_diagnostic.resolved_pointer &&
                        reports.fetch_add(1, std::memory_order_relaxed) < 16) {
                        LOG_INFO(HW_GPU,
                            "DIAGNOSTIC foliage task range pointer={:#x} requested={} mapped={} "
                            "contiguous={} max_descriptor={} alignment_prefix={} applied={} "
                            "binding_gpu={:#x} binding_bytes={} reason={}",
                            range_diagnostic.resolved_pointer, range_diagnostic.requested,
                            range_diagnostic.mapped_size, range_diagnostic.contiguous_size,
                            range_diagnostic.max_descriptor_range, range_diagnostic.alignment_prefix,
                            range_diagnostic.applied, range_diagnostic.bound_gpu_addr,
                            range_diagnostic.bound_size,
                            range_diagnostic.reason);
                    }
                }
                ++ssbo_index;
            }
        }
        const auto& cbufs{maxwell3d->state.shader_stages[stage].const_buffers};
        const auto read_handle{[&](const auto& desc, u32 index) {
            if (desc.handle_program != ~0U) {
                // Read by the shader from a storage buffer: computed for this draw.
                return TexturePair(VideoCommon::EvaluateTextureHandle(
                                       info, desc.handle_program, *gpu_memory,
                                       [&](u32 cbuf) -> GPUVAddr {
                                           return cbuf < cbufs.size() && cbufs[cbuf].enabled
                                                      ? cbufs[cbuf].address
                                                      : GPUVAddr{};
                                       }),
                                   via_header_index);
            }
            ASSERT(cbufs[desc.cbuf_index].enabled);
            const u32 index_offset{index << desc.size_shift};
            const u32 offset{desc.cbuf_offset + index_offset};
            const GPUVAddr addr{cbufs[desc.cbuf_index].address + offset};
            if constexpr (std::is_same_v<decltype(desc), const Shader::TextureDescriptor&> ||
                          std::is_same_v<decltype(desc), const Shader::TextureBufferDescriptor&>) {
                if (desc.has_secondary) {
                    ASSERT(cbufs[desc.secondary_cbuf_index].enabled);
                    const u32 second_offset{desc.secondary_cbuf_offset + index_offset};
                    const GPUVAddr separate_addr{cbufs[desc.secondary_cbuf_index].address +
                                                 second_offset};
                    const u32 lhs_raw{gpu_memory->Read<u32>(addr) << desc.shift_left};
                    const u32 rhs_raw{gpu_memory->Read<u32>(separate_addr)
                                      << desc.secondary_shift_left};
                    const u32 raw{lhs_raw | rhs_raw};
                    return TexturePair(raw, via_header_index);
                }
            }
            return TexturePair(gpu_memory->Read<u32>(addr), via_header_index);
        }};
        const auto add_image{[&](const auto& desc, bool blacklist) LAMBDA_FORCEINLINE {
            for (u32 index = 0; index < desc.count; ++index) {
                const auto handle{read_handle(desc, index)};
                views.push_back({
                    .index = handle.first,
                    .blacklist = blacklist,
                    .id = {}
                });
            }
        }};
        if constexpr (Spec::has_texture_buffers) {
            for (const auto& desc : info.texture_buffer_descriptors) {
                add_image(desc, false);
            }
        }
        if constexpr (Spec::has_image_buffers) {
            for (const auto& desc : info.image_buffer_descriptors) {
                add_image(desc, false);
            }
        }
        for (const auto& desc : info.texture_descriptors) {
            for (u32 index = 0; index < desc.count; ++index) {
                const auto handle{VideoCommon::HandleIsStoragePointer(info, desc, index)
                                      ? VideoCommon::NULL_TEXTURE_HANDLE
                                      : read_handle(desc, index)};
                if (record_sampled) {
                    sampled_images.push_back({static_cast<u32>(stage), handle.first, handle.second,
                                              views.size(), {}});
                }
                views.push_back({handle.first});

                VideoCommon::SamplerId sampler{texture_cache.GetSamplerId(handle.second, false)};
                samplers.push_back(sampler);
            }
        }
        if constexpr (Spec::has_images) {
            for (const auto& desc : info.image_descriptors) {
                add_image(desc, desc.is_written);
            }
        }

        return true;
    }};
    {
        Profile::Scope scope{Profile::Zone::ReadHandles};
        if constexpr (Spec::enabled_stages[0]) {
            config_stage(0);
        }
        if constexpr (Spec::enabled_stages[1]) {
            config_stage(1);
        }
        if constexpr (Spec::enabled_stages[2]) {
            config_stage(2);
        }
        if constexpr (Spec::enabled_stages[3]) {
            config_stage(3);
        }
        if constexpr (Spec::enabled_stages[4]) {
            config_stage(4);
        }
    }
    if (!guarded_pointers.empty()) {
        const bool other_writes = regs.transform_feedback_enabled != 0 ||
            std::ranges::any_of(stage_infos, [](const Shader::Info& info) {
                return info.stores_global_memory ||
                       std::ranges::any_of(info.image_descriptors,
                           [](const auto& desc) { return desc.is_written; }) ||
                       std::ranges::any_of(info.image_buffer_descriptors,
                           [](const auto& desc) { return desc.is_written; });
            });
        BufferCache::GraphicsStorageWriteBounds write_bounds{};
        boost::container::small_vector<BufferCache::GraphicsUniformWriteSource, 4> uniform_sources;
        bool uniform_proofs_valid = true;
        for (size_t stage = 0; stage < stage_infos.size(); ++stage) {
            if (!Spec::enabled_stages[stage]) continue;
            const auto& descriptors = stage_infos[stage].storage_buffers_descriptors;
            const auto& proofs = stage_infos[stage].uniform_storage_write_proofs;
            std::array<bool, 32> proof_used{};
            if (proofs.size() > proof_used.size()) { uniform_proofs_valid = false; continue; }
            for (size_t index = 0; index < descriptors.size(); ++index) {
                const auto& desc = descriptors[index];
                auto& end = write_bounds[stage][index];
                end = desc.fixed_write_end;
                if (!desc.has_uniform_write_proofs) continue;
                bool found = false;
                if (!desc.is_written || desc.pointer_offset != ~0U || desc.count != 1) {
                    uniform_proofs_valid = false;
                    continue;
                }
                for (size_t p = 0; p < proofs.size(); ++p) {
                    const auto& proof = proofs[p];
                    if (proof.root_cbuf_index != desc.cbuf_index ||
                        proof.root_cbuf_offset != desc.cbuf_offset) continue;
                    found = true;
                    proof_used[p] = true;
                    const auto span = buffer_cache.ResolveGraphicsUniformWriteSpan(stage,
                        proof.offset_cbuf_index, proof.offset_cbuf_offset,
                        proof.displacement, proof.width);
                    if (!span) { uniform_proofs_valid = false; continue; }
                    end = (std::max)(end, span->end);
                    uniform_sources.push_back(span->source);
                }
                if (!found) uniform_proofs_valid = false;
            }
            for (size_t p = 0; p < proofs.size(); ++p)
                if (!proof_used[p]) uniform_proofs_valid = false;
        }
        if (VideoCommon::Nx2WriteProfile::Enabled() && !other_writes && uniform_proofs_valid)
            writer_bounds = write_bounds;
        for (auto& pointer : guarded_pointers) {
            if (other_writes || !uniform_proofs_valid) pointer.guard.valid = false;
            constexpr u32 active_stages = (Spec::enabled_stages[0] ? 1U : 0U) |
                (Spec::enabled_stages[1] ? 2U : 0U) | (Spec::enabled_stages[2] ? 4U : 0U) |
                (Spec::enabled_stages[3] ? 8U : 0U) | (Spec::enabled_stages[4] ? 16U : 0U);
            buffer_cache.GuardGraphicsStoragePointerWrites(pointer.stage, pointer.index,
                                                           pointer.guard, active_stages, write_bounds, uniform_sources);
        }
    }
    ASSERT(views.size() == num_image_elements);
    ASSERT(samplers.size() == num_textures);
    {
        Profile::Scope scope{Profile::Zone::FillImageViews};
        texture_cache.FillImageViews(std::span(views.data(), views.size()), false,
                                     Spec::has_images);
    }
    if (record_sampled) {
        for (auto& sampled : sampled_images) sampled.id = views[sampled.view_index].id;
    }

    if constexpr (Spec::has_texture_buffers || Spec::has_image_buffers) {
        VideoCommon::ImageViewInOut* texture_buffer_it{views.data()};
        const auto bind_stage_info{[&](size_t stage) LAMBDA_FORCEINLINE {
            size_t index{};
            const auto add_buffer{[&](const auto& desc) {
                constexpr bool is_image = std::is_same_v<decltype(desc), const ImageBufferDescriptor&>;
                for (u32 i = 0; i < desc.count; ++i) {
                    bool is_written{false};
                    if constexpr (is_image) {
                        is_written = desc.is_written;
                    }
                    ImageView& image_view{texture_cache.GetImageView(texture_buffer_it->id)};
                    PixelFormat format{image_view.format};
                    if constexpr (is_image) {
                        if (const auto explicit_format{PixelFormatFromImageFormat(desc.format)}) {
                            format = *explicit_format;
                        }
                    }
                    buffer_cache.BindGraphicsTextureBuffer(stage, index, image_view.GpuAddr(),
                                                           image_view.BufferSize(), format,
                                                           is_written, is_image);
                    ++index;
                    ++texture_buffer_it;
                }
            }};
            buffer_cache.UnbindGraphicsTextureBuffers(stage);

            const Shader::Info& info{stage_infos[stage]};
            if constexpr (Spec::has_texture_buffers) {
                for (const auto& desc : info.texture_buffer_descriptors) {
                    add_buffer(desc);
                }
            }
            if constexpr (Spec::has_image_buffers) {
                for (const auto& desc : info.image_buffer_descriptors) {
                    add_buffer(desc);
                }
            }
            texture_buffer_it += Shader::NumDescriptors(info.texture_descriptors);
            if constexpr (Spec::has_images) {
                texture_buffer_it += Shader::NumDescriptors(info.image_descriptors);
            }
        }};
        Profile::Scope scope{Profile::Zone::TextureBuffers};
        if constexpr (Spec::enabled_stages[0]) {
            bind_stage_info(0);
        }
        if constexpr (Spec::enabled_stages[1]) {
            bind_stage_info(1);
        }
        if constexpr (Spec::enabled_stages[2]) {
            bind_stage_info(2);
        }
        if constexpr (Spec::enabled_stages[3]) {
            bind_stage_info(3);
        }
        if constexpr (Spec::enabled_stages[4]) {
            bind_stage_info(4);
        }
    } else {
        if constexpr (Spec::enabled_stages[0]) {
            buffer_cache.UnbindGraphicsTextureBuffers(0);
        }
        if constexpr (Spec::enabled_stages[1]) {
            buffer_cache.UnbindGraphicsTextureBuffers(1);
        }
        if constexpr (Spec::enabled_stages[2]) {
            buffer_cache.UnbindGraphicsTextureBuffers(2);
        }
        if constexpr (Spec::enabled_stages[3]) {
            buffer_cache.UnbindGraphicsTextureBuffers(3);
        }
        if constexpr (Spec::enabled_stages[4]) {
            buffer_cache.UnbindGraphicsTextureBuffers(4);
        }
    }

    if (regs.transform_feedback_enabled != 0) {
        scheduler.RequestOutsideRenderPassOperationContext();
    }

    {
        Profile::Scope scope{Profile::Zone::UpdateBuffers};
        buffer_cache.UpdateGraphicsBuffers(is_indexed);
        const bool canopy_lighting_pool =
            Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL &&
            (key.unique_hashes == std::array<u64, 6>{0, 0x4bcce0ad3a25fab8ULL, 0, 0, 0,
                                                   0xf16638b51d8e93acULL} ||
             key.unique_hashes == std::array<u64, 6>{0, 0xeb82ba5dfc9276f5ULL, 0, 0, 0,
                                                   0xf16638b51d8e93acULL}) &&
            IsMeshPipeline() && native_stages[0] && stage_infos[0].is_mesh &&
            !stage_infos[0].is_task;
        if (canopy_lighting_pool) {
            u32 index{};
            for (const auto& desc : stage_infos[0].storage_buffers_descriptors) {
                if (desc.count == 1 && !desc.is_written && desc.cbuf_index == 0 &&
                    desc.cbuf_offset == 0x780 && desc.pointer_offset == 32) {
                    const auto domain = buffer_cache.ExpandGraphicsReadOnlyDomain(0, index);
                    static std::atomic<u32> reports{};
                    if (reports.load(std::memory_order_relaxed) < 8 &&
                        reports.fetch_add(1, std::memory_order_relaxed) < 8) {
                        LOG_INFO(HW_GPU,
                            "OOT canopy lighting domain previous={} bytes={} physical={} "
                            "owner_suffix={} applied={} reason={}",
                            domain.previous_size, domain.bound_size, domain.contiguous_span,
                            domain.owner_suffix, domain.applied, domain.reason);
                    }
                }
                ++index;
            }
        }
        buffer_cache.BindHostGeometryBuffers(is_indexed);
    }

    {
        Profile::Scope scope{Profile::Zone::DescriptorAcquire};
        guest_descriptor_queue.Acquire(scheduler, num_descriptor_entries, uses_descriptor_buffer);
    }

    RescalingPushConstant rescaling;
    RenderAreaPushConstant render_area;
    const VideoCommon::SamplerId* samplers_it{samplers.data()};
    const VideoCommon::ImageViewInOut* views_it{views.data()};
    const auto prepare_stage{[&](size_t stage) LAMBDA_FORCEINLINE {
        {
            Profile::Scope scope{Profile::Zone::BindStageBuffers};
            if (VideoCommon::Nx2WriteProfile::Enabled()) {
                const auto writer = VideoCommon::Nx2WriteProfile::MakeStage(
                    stage_infos[stage], key.unique_hashes[stage + 1], 0,
                    static_cast<u32>(stage), stage == 0 ? key.unique_hashes[0] : 0,
                    writer_bounds ? (*writer_bounds)[stage].data() : nullptr,
                    writer_bounds ? (*writer_bounds)[stage].size() : 0);
                buffer_cache.BindHostStageBuffers(stage, &writer);
            } else {
                buffer_cache.BindHostStageBuffers(stage);
            }
        }
        {
            Profile::Scope scope{Profile::Zone::PushDescriptors};
            PushImageDescriptors(texture_cache, guest_descriptor_queue, stage_infos[stage],
                                 rescaling, samplers_it, views_it);
        }
        const auto& info{stage_infos[stage]};
        if (info.uses_render_area) {
            render_area.uses_render_area = true;
            // Switch 2 titles may never program the surface clip; use colour target 0 then.
            const u32 clip_width = regs.surface_clip.width, clip_height = regs.surface_clip.height;
            const bool clip_set = clip_width != 0 && clip_height != 0;
            render_area.words = {static_cast<float>(clip_set ? clip_width : u32{regs.rt[0].width}),
                                 static_cast<float>(clip_set ? clip_height : u32{regs.rt[0].height})};
        }
    }};
    if constexpr (Spec::enabled_stages[0]) {
        prepare_stage(0);
    }
    if constexpr (Spec::enabled_stages[1]) {
        prepare_stage(1);
    }
    if constexpr (Spec::enabled_stages[2]) {
        prepare_stage(2);
    }
    if constexpr (Spec::enabled_stages[3]) {
        prepare_stage(3);
    }
    if constexpr (Spec::enabled_stages[4]) {
        prepare_stage(4);
    }
    if (!buffer_cache.GraphicsUniformWriteSnapshotsValid()) return false;
    if (buffer_cache.any_buffer_uploaded) {
        buffer_cache.runtime.PostCopyBarrier();
        buffer_cache.any_buffer_uploaded = false;
    }
    {
        Profile::Scope scope{Profile::Zone::RenderTargets};
        texture_cache.UpdateRenderTargets(false);
        const std::span<const VideoCommon::ImageViewInOut> view_span{views.data(), views.size()};
        texture_cache.CheckFeedbackLoop(view_span);
        // A fragment barrier only has to end the pass if this draw reads what it wrote (Lichii).
        if (scheduler.HasFragmentBarrier() && !writes_storage &&
            !scheduler.RenderPassHasStores() && !texture_cache.SamplesRenderTarget(view_span)) {
            scheduler.KeepRenderPassAcrossBarrier();
        }
    }
    if (IsBuilt() && !pipeline) {
        return false;
    }
    Profile::Scope scope{Profile::Zone::ConfigureDraw};
    return ConfigureDraw(rescaling, render_area);
}

bool GraphicsPipeline::CaptureFoliageInputs(FrameDump& frame_dump, std::string_view fields) {
    const bool material = frame_dump.CapturingMaterialInputs();
    const bool canopy = frame_dump.CapturingCanopyInputs();
    const bool selected_pair = material || canopy;
    const auto canopy_pair = frame_dump.CanopyInputShaders();
    const bool canopy_mesh = canopy && frame_dump.CanopyInputIsMesh();
    const u64 TaskHash = canopy ? canopy_pair[0] :
                         material ? 0x00e4eb4772d5076bULL : 0xf378ff9ca77352f0ULL;
    const bool colored_foliage = !selected_pair && key.unique_hashes[3] == 0x87183e2c33d87f2aULL &&
                                 key.unique_hashes[5] == 0xc4a9495a97b2ae03ULL;
    const u64 MeshHash = canopy ? canopy_pair[1] : material ? 0x6cb075ad84a5c605ULL :
                        colored_foliage ? 0x87183e2c33d87f2aULL : 0x277d8ef0f4003ac4ULL;
    const u64 FragmentHash = selected_pair ? MeshHash :
                            colored_foliage ? 0xc4a9495a97b2ae03ULL : 0x4b0ca12cb55286d7ULL;
    if (!frame_dump.CapturingTaskInputs() || key.unique_hashes[1] != TaskHash ||
        key.unique_hashes[5] != FragmentHash ||
        (selected_pair ? IsMeshPipeline() != canopy_mesh || key.unique_hashes[0] != 0 ||
                        key.unique_hashes[2] != 0 || key.unique_hashes[3] != 0 ||
                        key.unique_hashes[4] != 0 ||
                        (canopy && (stage_infos[0].is_mesh != canopy_mesh || stage_infos[0].is_task)) :
                    key.unique_hashes[3] != MeshHash)) return false;
    if (frame_dump.TaskInputsRecorded())
        return frame_dump.NoteTaskInputs(TaskHash, MeshHash, fields, {});

    constexpr size_t InputCap = 32;
    std::array<FrameDump::TaskBufferInput, InputCap> inputs{};
    size_t count{}, required{};
    u32 stage_entry{}, shader_binding{};
    const auto* entries = guest_descriptor_queue.UpdateData();
    const auto stream = buffer_cache.runtime.UniformStreamSnapshot();
    const auto descriptor = [&](FrameDump::ComputeBufferInput& input, u32 entry) {
        input.descriptor_checked = true;
        input.descriptor_entry = entry;
        input.shader_binding = shader_binding;
        input.descriptor_address_mode = uses_descriptor_buffer;
        if (!entries || entry >= num_descriptor_entries) {
            input.skip_reason = "descriptor index exceeds payload";
            return false;
        }
        if (uses_descriptor_buffer) {
            input.descriptor_address = entries[entry].address.address;
            input.descriptor_range = entries[entry].address.range;
        } else {
            input.descriptor_buffer = entries[entry].buffer.buffer;
            input.descriptor_offset = entries[entry].buffer.offset;
            input.descriptor_range = entries[entry].buffer.range;
        }
        return true;
    };
    const auto matches = [&](FrameDump::ComputeBufferInput& input, VkDeviceAddress base) {
        const bool owner = input.buffer != VK_NULL_HANDLE &&
            (uses_descriptor_buffer ? base != 0 && input.descriptor_address >= base &&
                input.descriptor_address - base == input.offset :
                input.descriptor_buffer == input.buffer && input.descriptor_offset == input.offset);
        input.descriptor_matches = owner && input.size != 0 &&
            input.descriptor_range == input.size && input.offset <= input.source_capacity &&
            input.size <= input.source_capacity - input.offset;
        return input.descriptor_matches;
    };
    const u32 capture_passes = canopy ? 2U : 1U;
    for (u32 capture_pass = 0; capture_pass < capture_passes; ++capture_pass) {
        stage_entry = shader_binding = 0;
        for (u32 stage = 0; stage < stage_infos.size(); ++stage) {
            const auto& info = stage_infos[stage];
            u32 entry = stage_entry;
            const bool selected = stage == 0 || stage == 4 || (!selected_pair && stage == 2);
            u32 uniform_index{};
            for (const auto& desc : info.constant_buffer_descriptors) {
                for (u32 array_index = 0; array_index < desc.count; ++array_index, ++entry) {
                    if (!selected || (canopy && capture_pass != 0)) continue;
                    ++required;
                    if (count == inputs.size()) continue;
                    auto& captured = inputs[count++];
                    captured.stage = stage;
                    captured.shader_hash = stage == 0 ? TaskHash : stage == 2 ? MeshHash : FragmentHash;
                    auto& input = captured.binding;
                    input.uniform_buffer = true;
                    input.binding_index = uniform_index;
                    input.cbuf_index = desc.index;
                    if (!descriptor(input, entry)) continue;
                    if (desc.count != 1) {
                        input.skip_reason = "uniform descriptor array is unsupported";
                        continue;
                    }
                    if (input.descriptor_range == 0 || input.descriptor_range > 65536) {
                        input.skip_reason = "empty or over 64 KiB uniform descriptor";
                        continue;
                    }
                    input.size = static_cast<u32>(input.descriptor_range);
                    input.buffer = stream.buffer;
                    input.source_capacity = stream.capacity;
                    input.offset = uses_descriptor_buffer ?
                        (input.descriptor_address >= stream.device_address ?
                            input.descriptor_address - stream.device_address : stream.capacity) :
                        input.descriptor_offset;
                    if (!matches(input, stream.device_address)) {
                        const auto cached = buffer_cache.GetGraphicsUniformBindingSnapshot(stage, desc.index);
                        if (cached.buffer && cached.size == input.size) {
                            input.buffer = cached.buffer->Handle();
                            input.source_capacity = cached.buffer->SizeBytes();
                            input.offset = cached.offset;
                            matches(input, cached.buffer->DeviceAddress());
                        }
                    }
                    if (!input.descriptor_matches)
                        input.skip_reason = "uniform descriptor has no validated stream or cache owner";
                }
                ++uniform_index;
                ++shader_binding;
            }
            u32 storage_index{};
            for (const auto& desc : info.storage_buffers_descriptors) {
                for (u32 array_index = 0; array_index < desc.count; ++array_index, ++entry) {
                    if (!selected || (canopy && capture_pass != 1)) continue;
                    ++required;
                    if (count == inputs.size()) continue;
                    auto& captured = inputs[count++];
                    captured.stage = stage;
                    captured.shader_hash = stage == 0 ? TaskHash : stage == 2 ? MeshHash : FragmentHash;
                    captured.pointer_offset = desc.pointer_offset;
                    auto& input = captured.binding;
                    input.binding_index = storage_index;
                    input.cbuf_index = desc.cbuf_index;
                    input.cbuf_offset = desc.cbuf_offset;
                    input.is_written = desc.is_written;
                    if (!descriptor(input, entry)) continue;
                    if (desc.count != 1 || desc.is_written) {
                        input.skip_reason = "storage descriptor is not scalar read-only";
                        continue;
                    }
                    const auto snapshot = buffer_cache.GetGraphicsStorageBindingSnapshot(stage, storage_index);
                    input.buffer = snapshot.buffer ? snapshot.buffer->Handle() : VK_NULL_HANDLE;
                    input.source_capacity = snapshot.buffer ? snapshot.buffer->SizeBytes() : 0;
                    input.offset = snapshot.offset;
                    input.size = snapshot.size;
                    input.gpu_addr = snapshot.gpu_addr;
                    input.segment_count = snapshot.segment_count;
                    input.skip_reason = snapshot.skip_reason;
                    if (!matches(input, snapshot.buffer ? snapshot.buffer->DeviceAddress() : 0) &&
                        !input.skip_reason)
                        input.skip_reason = "storage descriptor differs from current cache binding";
                }
                ++storage_index;
                ++shader_binding;
            }
            stage_entry += NumDescriptorEntries(info);
            shader_binding += static_cast<u32>(info.texture_buffer_descriptors.size() +
                info.image_buffer_descriptors.size() + info.texture_descriptors.size() +
                info.image_descriptors.size());
        }
    }
    if (canopy) {
        // Prioritize uniforms and small pointer tables within the bounded copy budget.
        std::stable_sort(inputs.begin(), inputs.begin() + count, [](const auto& a, const auto& b) {
            if (a.binding.uniform_buffer != b.binding.uniform_buffer)
                return a.binding.uniform_buffer;
            return a.binding.size < b.binding.size;
        });
    }
    const std::string metadata = fmt::format(
        R"({}{}"fragment_hash":"{:016x}","required_buffer_descriptors":{},"collected_buffer_descriptors":{},"buffer_descriptor_selection_complete":{},"actual_descriptor_entries":{},"alpha_texture_payload":"not_captured_compressed","generated_task_mesh_outputs":"not_captured")",
        fields, fields.empty() ? "" : ",", FragmentHash, required, count,
        required == count, num_descriptor_entries);
    return frame_dump.NoteTaskInputs(TaskHash, MeshHash, metadata,
                                    std::span{inputs.data(), count});
}


bool GraphicsPipeline::ConfigureDraw(const RescalingPushConstant& rescaling,
                                     const RenderAreaPushConstant& render_area) {
    const void* const descriptor_data{guest_descriptor_queue.UpdateData()};

    VkDeviceSize descriptor_buffer_offset{};
    u32 descriptor_buffer_chunk{};
    if (descriptor_set_layout && uses_descriptor_buffer) {
        const auto* const entries = static_cast<const DescriptorUpdateEntry*>(descriptor_data);
        const bool reuse_allocation =
            last_descriptor_buffer_generation == descriptor_buffer_ring.CurrentGeneration() &&
            last_descriptor_payload.size() == num_descriptor_entries &&
            std::memcmp(last_descriptor_payload.data(), entries,
                        num_descriptor_entries * sizeof(DescriptorUpdateEntry)) == 0;
        if (reuse_allocation) {
            descriptor_buffer_offset = last_descriptor_buffer_offset;
            descriptor_buffer_chunk = last_descriptor_buffer_chunk;
            descriptor_buffer_ring.TouchFrame(scheduler);
        } else {
            const DescriptorBufferRing::Allocation alloc{
                descriptor_buffer_ring.Allocate(scheduler, descriptor_buffer_layout.size)};
            if (!alloc.host) {
                LOG_DEBUG(Render_Vulkan, "Failed to reserve descriptor memory, skipping draw");
                return false;
            }
            WriteDescriptorBuffer(device, descriptor_buffer_layout, entries, alloc.host);
            descriptor_buffer_offset = alloc.offset;
            descriptor_buffer_chunk = alloc.chunk;
            last_descriptor_buffer_offset = alloc.offset;
            last_descriptor_buffer_chunk = alloc.chunk;
            last_descriptor_buffer_generation = alloc.generation;
            last_descriptor_payload.assign(entries, entries + num_descriptor_entries);
        }
    }

    scheduler.RequestRenderpass(texture_cache.GetFramebuffer(), writes_storage);
    if (!is_built.load(std::memory_order::relaxed)) {
        // Wait for the pipeline to be built
        scheduler.Record([this](vk::CommandBuffer) {
            std::unique_lock lock{build_mutex};
            build_condvar.wait(lock, [this] { return is_built.load(std::memory_order::relaxed); });
        });
    }
    const bool is_rescaling{texture_cache.IsRescaling()};
    const bool update_rescaling{scheduler.UpdateRescaling(is_rescaling)};
    const bool bind_pipeline{scheduler.UpdateGraphicsPipeline(this)};
    const bool bind_descriptor_buffer{
        descriptor_set_layout && uses_descriptor_buffer &&
        scheduler.UpdateDescriptorBufferChunk(descriptor_buffer_chunk)};

    // Log graphics pipeline binding
    if (bind_pipeline && GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string pipeline_info = fmt::format("hash={:#016x}", key.Hash());
        GPU::Logging::GPULogger::GetInstance().LogPipelineBind(false, pipeline_info);
    }

    bool update_descriptors = true;
    if (descriptor_set_layout && !uses_push_descriptor && !uses_descriptor_buffer) {
        const auto* const entries = static_cast<const DescriptorUpdateEntry*>(descriptor_data);
        update_descriptors =
            bind_pipeline || last_descriptor_payload.size() != num_descriptor_entries ||
            std::memcmp(last_descriptor_payload.data(), entries,
                        num_descriptor_entries * sizeof(DescriptorUpdateEntry)) != 0;
        if (update_descriptors) {
            last_descriptor_payload.assign(entries, entries + num_descriptor_entries);
        }
    }
    scheduler.Record([this, descriptor_data, bind_pipeline, update_descriptors,
                      descriptor_buffer_offset, descriptor_buffer_chunk, bind_descriptor_buffer,
                      rescaling_data = rescaling.Data(), is_rescaling, update_rescaling,
                      uses_render_area = render_area.uses_render_area,
                      render_area_data = render_area.words](vk::CommandBuffer cmdbuf) {
        if (bind_descriptor_buffer) {
            const VkDescriptorBufferBindingInfoEXT binding_info{
                descriptor_buffer_ring.BindingInfo(descriptor_buffer_chunk)};
            cmdbuf.BindDescriptorBuffersEXT(binding_info);
        }
        if (bind_pipeline) {
            if (!pipeline) {
                return;
            }
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline);
        }
        cmdbuf.PushConstants(*pipeline_layout, VK_SHADER_STAGE_ALL_GRAPHICS,
                             RESCALING_LAYOUT_WORDS_OFFSET, sizeof(rescaling_data),
                             rescaling_data.data());
        if (update_rescaling) {
            const f32 config_down_factor{Settings::values.resolution_info.down_factor};
            const f32 scale_down_factor{is_rescaling ? config_down_factor : 1.0f};
            cmdbuf.PushConstants(*pipeline_layout, VK_SHADER_STAGE_ALL_GRAPHICS,
                                 RESCALING_LAYOUT_DOWN_FACTOR_OFFSET, sizeof(scale_down_factor),
                                 &scale_down_factor);
        }
        if (uses_render_area) {
            cmdbuf.PushConstants(*pipeline_layout, VK_SHADER_STAGE_ALL_GRAPHICS,
                                 RENDERAREA_LAYOUT_OFFSET, sizeof(render_area_data),
                                 &render_area_data);
        }
        if (!descriptor_set_layout) {
            return;
        }
        if (uses_descriptor_buffer) {
            const u32 buffer_index{};
            cmdbuf.SetDescriptorBufferOffsetsEXT(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout,
                                                 0, buffer_index, descriptor_buffer_offset);
        } else if (uses_push_descriptor) {
            cmdbuf.PushDescriptorSetWithTemplateKHR(*descriptor_update_template, *pipeline_layout,
                                                    0, descriptor_data);
        } else if (update_descriptors) {
            const VkDescriptorSet descriptor_set{descriptor_allocator.Commit()};
            const vk::Device& dev{device.GetLogical()};
            dev.UpdateDescriptorSet(descriptor_set, *descriptor_update_template, descriptor_data);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout, 0,
                                      descriptor_set, nullptr);
        }
    });
    return true;
}

void GraphicsPipeline::MakePipeline(VkRenderPass render_pass) {
    FixedPipelineState::DynamicState dynamic{};
    if (!key.state.extended_dynamic_state) {
        dynamic = key.state.dynamic_state;
    } else {
        dynamic.raw1 = key.state.dynamic_state.raw1;
    }
    static_vector<VkVertexInputBindingDescription, 32> vertex_bindings;
    static_vector<VkVertexInputBindingDivisorDescriptionEXT, 32> vertex_binding_divisors;
    static_vector<VkVertexInputAttributeDescription, 32> vertex_attributes;
    if (!key.state.dynamic_vertex_input) {
        const size_t num_vertex_arrays = (std::min)(
            Maxwell::NumVertexArrays, static_cast<size_t>(device.GetMaxVertexInputBindings()));
        for (size_t index = 0; index < num_vertex_arrays; ++index) {
            const bool instanced = key.state.binding_divisors[index] != 0;
            const auto rate =
                instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX;
            vertex_bindings.push_back({
                .binding = static_cast<u32>(index),
                .stride = key.state.vertex_strides[index],
                .inputRate = rate,
            });
            if (instanced) {
                vertex_binding_divisors.push_back({
                    .binding = static_cast<u32>(index),
                    .divisor = key.state.binding_divisors[index],
                });
            }
        }
        for (size_t index = 0; index < key.state.attributes.size(); ++index) {
            const auto& attribute = key.state.attributes[index];
            if (!attribute.enabled || !stage_infos[0].loads.Generic(index)) {
                continue;
            }
            vertex_attributes.push_back({
                .location = static_cast<u32>(index),
                .binding = attribute.buffer,
                .format = MaxwellToVK::VertexFormat(device, attribute.Type(), attribute.Size()),
                .offset = attribute.offset,
            });
        }
    }
    ASSERT(vertex_attributes.size() <= device.GetMaxVertexInputAttributes());

    VkPipelineVertexInputStateCreateInfo vertex_input_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = static_cast<u32>(vertex_bindings.size()),
        .pVertexBindingDescriptions = vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(vertex_attributes.size()),
        .pVertexAttributeDescriptions = vertex_attributes.data(),
    };
    const VkPipelineVertexInputDivisorStateCreateInfoEXT input_divisor_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .vertexBindingDivisorCount = static_cast<u32>(vertex_binding_divisors.size()),
        .pVertexBindingDivisors = vertex_binding_divisors.data(),
    };
    if (!vertex_binding_divisors.empty()) {
        vertex_input_ci.pNext = &input_divisor_ci;
    }
    const bool has_tess_stages = spv_modules[1] || spv_modules[2];
    auto input_assembly_topology = MaxwellToVK::PrimitiveTopology(device, key.state.topology);
    if (input_assembly_topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST) {
        if (!has_tess_stages) {
            LOG_WARNING(Render_Vulkan, "Patch topology used without tessellation, using points");
            input_assembly_topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        }
    } else {
        if (has_tess_stages) {
            // The Vulkan spec requires patch list IA topology be used with tessellation
            // shader stages. Forcing it fixes a crash on some drivers
            LOG_WARNING(Render_Vulkan,
                        "Patch topology not used with tessellation, using patch list");
            input_assembly_topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        }
    }
    const VkPipelineInputAssemblyStateCreateInfo input_assembly_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = input_assembly_topology,
        .primitiveRestartEnable =
        // MoltenVK/Metal always has primitive restart enabled and cannot disable it
        device.IsMoltenVK() ? VK_TRUE :
        (dynamic.primitive_restart_enable != 0 &&
                ((input_assembly_topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
                  device.IsTopologyListPrimitiveRestartSupported()) ||
                 SupportsPrimitiveRestart(input_assembly_topology) ||
                 (input_assembly_topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
                  device.IsPatchListPrimitiveRestartSupported()))
            ? VK_TRUE
            : VK_FALSE),
    };
    const VkPipelineTessellationStateCreateInfo tessellation_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .patchControlPoints = key.state.patch_control_points_minus_one.Value() + 1,
    };
    std::array<VkViewportSwizzleNV, Maxwell::NumViewports> swizzles;
    std::ranges::transform(key.state.viewport_swizzles, swizzles.begin(), UnpackViewportSwizzle);
    VkPipelineViewportSwizzleStateCreateInfoNV swizzle_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = Maxwell::NumViewports,
        .pViewportSwizzles = swizzles.data(),
    };
    VkPipelineViewportDepthClipControlCreateInfoEXT ndc_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
        .pNext = nullptr,
        .negativeOneToOne = key.state.ndc_minus_one_to_one.Value() != 0 ? VK_TRUE : VK_FALSE,
    };
    const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
    VkPipelineViewportStateCreateInfo viewport_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = num_viewports,
        .pViewports = nullptr,
        .scissorCount = num_viewports,
        .pScissors = nullptr,
    };
    if (device.IsNvViewportSwizzleSupported()) {
        swizzle_ci.pNext = std::exchange(viewport_ci.pNext, &swizzle_ci);
    }
    if (device.IsExtDepthClipControlSupported()) {
        ndc_info.pNext = std::exchange(viewport_ci.pNext, &ndc_info);
    }
    VkPipelineRasterizationStateCreateInfo rasterization_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable =
        static_cast<VkBool32>(dynamic.depth_clamp_disabled == 0 ? VK_TRUE : VK_FALSE),
        .rasterizerDiscardEnable =
        static_cast<VkBool32>(dynamic.rasterize_enable == 0 ? VK_TRUE : VK_FALSE),
        .polygonMode =
        MaxwellToVK::PolygonMode(FixedPipelineState::UnpackPolygonMode(key.state.polygon_mode)),
        .cullMode = static_cast<VkCullModeFlags>(
            dynamic.cull_enable ? MaxwellToVK::CullFace(dynamic.CullFace()) : VK_CULL_MODE_NONE),
        .frontFace = MaxwellToVK::FrontFace(dynamic.FrontFace()),
        .depthBiasEnable = (dynamic.depth_bias_enable != 0 ? VK_TRUE : VK_FALSE),
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
        // TODO(alekpop): Transfer from regs
    };
    const bool smooth_lines_supported =
        device.IsExtLineRasterizationSupported() && device.SupportsSmoothLines();
    const bool stippled_lines_supported =
        device.IsExtLineRasterizationSupported() && device.SupportsStippledRectangularLines();
    VkPipelineRasterizationLineStateCreateInfoEXT line_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .lineRasterizationMode = key.state.smooth_lines != 0 && smooth_lines_supported
                                     ? VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT
                                     : VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT,
        .stippledLineEnable =
            (dynamic.line_stipple_enable && stippled_lines_supported) ? VK_TRUE : VK_FALSE,
        .lineStippleFactor = key.state.line_stipple_factor,
        .lineStipplePattern = static_cast<uint16_t>(key.state.line_stipple_pattern),
    };
    VkPipelineRasterizationConservativeStateCreateInfoEXT conservative_raster{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .conservativeRasterizationMode = key.state.conservative_raster_enable != 0
                                             ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                                             : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT,
        .extraPrimitiveOverestimationSize = 0.0f,
    };
    const bool supports_provoking_first_mode = device.SupportsProvokingVertexFirstMode();
    const bool supports_provoking_last_mode = device.SupportsProvokingVertexLastMode();
    const bool requested_provoking_last_mode = key.state.provoking_vertex_last != 0;
    const VkProvokingVertexModeEXT provoking_vertex_mode =
        requested_provoking_last_mode
            ? (supports_provoking_last_mode ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
                                            : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT)
            : (supports_provoking_first_mode ? VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT
                                             : VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT);
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .provokingVertexMode = provoking_vertex_mode,
    };

    if (IsLine(input_assembly_topology) && device.IsExtLineRasterizationSupported()) {
        line_state.pNext = std::exchange(rasterization_ci.pNext, &line_state);
    }
    if (device.IsExtConservativeRasterizationSupported()) {
        conservative_raster.pNext = std::exchange(rasterization_ci.pNext, &conservative_raster);
    }
    if (device.IsExtProvokingVertexSupported()) {
        provoking_vertex.pNext = std::exchange(rasterization_ci.pNext, &provoking_vertex);
    }

    const bool supports_alpha_output = fragment_has_color0_output;
    const bool alpha_to_one_supported = device.SupportsAlphaToOne();
    const VkPipelineMultisampleStateCreateInfo multisample_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = MaxwellToVK::MsaaMode(key.state.msaa_mode),
        .sampleShadingEnable = Settings::values.sample_shading.GetValue() > 0 ? VK_TRUE : VK_FALSE,
        .minSampleShading = f32(Settings::values.sample_shading.GetValue()) / 100.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable =
            supports_alpha_output && key.state.alpha_to_coverage_enabled != 0 ? VK_TRUE : VK_FALSE,
        .alphaToOneEnable = supports_alpha_output && alpha_to_one_supported &&
                           key.state.alpha_to_one_enabled != 0 ? VK_TRUE : VK_FALSE,
    };
    const VkPipelineDepthStencilStateCreateInfo depth_stencil_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthTestEnable = dynamic.depth_test_enable,
        .depthWriteEnable = dynamic.depth_write_enable,
        .depthCompareOp = dynamic.depth_test_enable
                              ? MaxwellToVK::ComparisonOp(dynamic.DepthTestFunc())
                              : VK_COMPARE_OP_ALWAYS,
        .depthBoundsTestEnable = dynamic.depth_bounds_enable && device.IsDepthBoundsSupported(),
        .stencilTestEnable = dynamic.stencil_enable,
        .front = GetStencilFaceState(dynamic.front),
        .back = GetStencilFaceState(dynamic.back),
        .minDepthBounds = f32(key.state.depth_bounds_min),
        .maxDepthBounds = f32(key.state.depth_bounds_max),
    };
    if (dynamic.depth_bounds_enable && !device.IsDepthBoundsSupported()) {
        LOG_WARNING(Render_Vulkan, "Depth bounds is enabled but not supported");
    }
    static_vector<VkPipelineColorBlendAttachmentState, Maxwell::NumRenderTargets> cb_attachments;
    const size_t num_attachments{NumAttachments(key.state)};
    for (size_t index = 0; index < num_attachments; ++index) {
        static constexpr std::array mask_table{
            VK_COLOR_COMPONENT_R_BIT,
            VK_COLOR_COMPONENT_G_BIT,
            VK_COLOR_COMPONENT_B_BIT,
            VK_COLOR_COMPONENT_A_BIT,
        };
        const auto& blend{key.state.attachments[index]};
        const std::array mask{blend.Mask()};
        VkColorComponentFlags write_mask{};
        for (size_t i = 0; i < mask_table.size(); ++i) {
            write_mask |= mask[i] ? mask_table[i] : 0;
        }
        write_mask &= ColorOutputMask(index);
        cb_attachments.push_back({
            .blendEnable = blend.enable != 0,
            .srcColorBlendFactor = MaxwellToVK::BlendFactor(blend.SourceRGBFactor()),
            .dstColorBlendFactor = MaxwellToVK::BlendFactor(blend.DestRGBFactor()),
            .colorBlendOp = MaxwellToVK::BlendEquation(blend.EquationRGB()),
            .srcAlphaBlendFactor = MaxwellToVK::BlendFactor(blend.SourceAlphaFactor()),
            .dstAlphaBlendFactor = MaxwellToVK::BlendFactor(blend.DestAlphaFactor()),
            .alphaBlendOp = MaxwellToVK::BlendEquation(blend.EquationAlpha()),
            .colorWriteMask = write_mask,
        });
    }
    const VkPipelineColorBlendStateCreateInfo color_blend_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = dynamic.logic_op_enable != 0,
        .logicOp = static_cast<VkLogicOp>(dynamic.logic_op.Value()),
        .attachmentCount = static_cast<u32>(cb_attachments.size()),
        .pAttachments = cb_attachments.data(),
        .blendConstants = {}
    };
    static_vector<VkDynamicState, 34> dynamic_states{
        VK_DYNAMIC_STATE_VIEWPORT,           VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,         VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS,       VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_LINE_WIDTH,
    };
    if (key.state.extended_dynamic_state) {
        static constexpr std::array extended{
            VK_DYNAMIC_STATE_CULL_MODE_EXT,
            VK_DYNAMIC_STATE_FRONT_FACE_EXT,
            VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
            VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_STENCIL_OP_EXT,
            VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended.begin(), extended.end());

        // VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT
        if (!key.state.dynamic_vertex_input) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT);
        }
    }

    // VK_DYNAMIC_STATE_VERTEX_INPUT_EXT
    if (key.state.dynamic_vertex_input) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_EXT);
    }

    // EDS2 - Core (3 states)
    if (key.state.extended_dynamic_state_2) {
        static constexpr std::array extended2{
            VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE_EXT,
            VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT,
            VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended2.begin(), extended2.end());
    }

    // EDS2 - LogicOp (granular)
    if (key.state.extended_dynamic_state_2_logic_op) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_LOGIC_OP_EXT);
    }

    // EDS3 - Blending (composite: 3 states)
    if (key.state.extended_dynamic_state_3_blend) {
        static constexpr std::array extended3{
            VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
            VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
            VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended3.begin(), extended3.end());
    }

    // VK_EXT_color_write_enable fallback for fully on/off render targets when EDS3 blending is not available.
    if (!key.state.extended_dynamic_state_3_blend && key.state.color_write_enable_dynamic) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT);
    }

    // EDS3 - Enables (composite: per-feature)
    if (key.state.extended_dynamic_state_3_enables) {
        if (device.SupportsDynamicState3DepthClampEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3LogicOpEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3LineRasterizationMode()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT);
        }
        if (device.SupportsDynamicState3ConservativeRasterizationMode()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_CONSERVATIVE_RASTERIZATION_MODE_EXT);
        }
        if (device.SupportsDynamicState3LineStippleEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LINE_STIPPLE_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3AlphaToCoverageEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3AlphaToOneEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_ALPHA_TO_ONE_ENABLE_EXT);
        }
    }

    const bool is_mesh{IsMeshPipeline()};
    if (is_mesh) {
        // Mesh pipelines have no vertex input or input assembly.
        dynamic_states.erase(std::remove_if(dynamic_states.begin(), dynamic_states.end(),
                                            [](VkDynamicState state) {
                                                return state == VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT ||
                                                       state == VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT ||
                                                       state == VK_DYNAMIC_STATE_VERTEX_INPUT_EXT ||
                                                       state == VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT ||
                                                       state == VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT;
                                            }),
                             dynamic_states.end());
    }
    const VkPipelineDynamicStateCreateInfo dynamic_state_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .requiredSubgroupSize = GuestWarpSize,
    };
    static_vector<VkPipelineShaderStageCreateInfo, 5> shader_stages;
    for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
        if (!spv_modules[stage]) {
            continue;
        }
        const Shader::Info& info{stage_infos[stage]};
        const VkShaderStageFlagBits stage_flag{
            StageFlag(info, MaxwellToVK::ShaderStage(Shader::StageFromIndex(stage)))};
        // Votes, ballots and shuffles span the guest's 32-thread warp. A driver free to pick a
        // narrower subgroup can split the warp and change the shader's result.
        const bool uses_warp_ops = info.uses_subgroup_vote || info.uses_subgroup_invocation_id ||
                                   info.uses_subgroup_shuffles || info.uses_subgroup_mask;
        const bool workgroup_fits =
            (stage_flag != VK_SHADER_STAGE_TASK_BIT_EXT &&
             stage_flag != VK_SHADER_STAGE_MESH_BIT_EXT) ||
            device.IsGuestWarpWorkgroupSizeSupported(workgroup_sizes[stage]);
        const bool force_warp_size = uses_warp_ops && workgroup_fits &&
                                     device.IsExtSubgroupSizeControlSupported() &&
                                     device.IsGuestWarpSizeSupported(stage_flag);
        shader_stages.emplace_back(VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = force_warp_size ? &subgroup_size_ci : nullptr,
            .flags = 0,
            .stage = stage_flag,
            .module = *spv_modules[stage],
            .pName = "main",
            .pSpecializationInfo = nullptr,
        });
    }
    VkPipelineCreateFlags flags{};
    if (device.IsKhrPipelineExecutablePropertiesEnabled() && Settings::values.renderer_debug.GetValue()) {
        flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    }
    if (uses_descriptor_buffer) {
        flags |= VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    }

    pipeline = device.GetLogical().CreateGraphicsPipeline({
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
        .stageCount = static_cast<u32>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = is_mesh ? nullptr : &vertex_input_ci,
        .pInputAssemblyState = is_mesh ? nullptr : &input_assembly_ci,
        .pTessellationState = is_mesh ? nullptr : &tessellation_ci,
        .pViewportState = &viewport_ci,
        .pRasterizationState = &rasterization_ci,
        .pMultisampleState = &multisample_ci,
        .pDepthStencilState = &depth_stencil_ci,
        .pColorBlendState = &color_blend_ci,
        .pDynamicState = &dynamic_state_ci,
        .layout = *pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = 0,
    }, *pipeline_cache);

    // Log graphics pipeline creation
    if (GPU::Logging::IsActive()) {
        const std::string pipeline_info = fmt::format(
            "GraphicsPipeline created: stages={}, attachments={}",
            shader_stages.size(),
            color_blend_ci.attachmentCount
        );
        GPU::Logging::GPULogger::GetInstance().LogPipelineStateChange(pipeline_info);
    }
}

void GraphicsPipeline::Validate() {
    size_t num_images{};
    for (const auto& info : stage_infos) {
        num_images += Shader::NumDescriptors(info.texture_buffer_descriptors);
        num_images += Shader::NumDescriptors(info.image_buffer_descriptors);
        num_images += Shader::NumDescriptors(info.texture_descriptors);
        num_images += Shader::NumDescriptors(info.image_descriptors);
    }
    ASSERT(num_images == num_image_elements);
}

} // namespace Vulkan
