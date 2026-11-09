// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <limits>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <exception>
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ampere/program_override.h"
#include <tuple>
#include <set>
#include <fstream>
#include <mutex>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/renderer_vulkan.h"

#include "common/assert.h"
#include "shader_recompiler/exception.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/native_query_trace.h"
#include "video_core/indirect_reset.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_storage_feedback.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/nx2_profile.h"

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewType;


namespace {
struct DrawParams {
    u32 base_instance;
    u32 num_instances;
    u32 base_vertex;
    u32 num_vertices;
    u32 first_index;
    bool is_indexed;
};

// CPU state only; these fields do not report GPU predicate results or generated mesh output.
std::string DrawSubmissionGuestState(
    const Maxwell& regs, const Tegra::Engines::Maxwell3D::DrawManager::State& draw,
    bool host_execute) {
    const auto& vp = regs.viewport_transform[0];
    const auto& sc = regs.scissor_test[0];
    const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) |
                   regs.pipelines[1].reserved[1];
    const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) |
                   regs.pipelines[5].reserved[1];
    const auto blend_json = [](const auto& blend) {
        return fmt::format(
            R"({{"color_op":{},"color_src":{},"color_dst":{},"alpha_op":{},"alpha_src":{},"alpha_dst":{}}})",
            static_cast<u32>(blend.color_op), static_cast<u32>(blend.color_source),
            static_cast<u32>(blend.color_dest), static_cast<u32>(blend.alpha_op),
            static_cast<u32>(blend.alpha_source), static_cast<u32>(blend.alpha_dest));
    };
    std::string fields = fmt::format(
        R"("guest_vs":"{:#x}","guest_ps":"{:#x}","guest_rt0":"{:#x}","guest_rt_count":{},"guest_depth":"{:#x}","guest_depth_format":{},"zeta_enable":{},"color_mask_common":{},"color_mask0":{},"blend0":{},"depth_test":{},"depth_write":{},"depth_func":{},"rasterize_enable":{},"cull_enable":{},"cull_face":{},"front_face":{},"scissor0":[{},{},{},{},{}],"viewport0_bits":[{},{},{},{},{},{}],"host_execute":{},"render_enable_mode":{},"render_enable_override":{},"predicate_address":"{:#x}","predicate_words":"not_captured")",
        vs, ps, regs.rt[0].Address(), regs.rt_control.count.Value(), regs.zeta.Address(),
        static_cast<u32>(regs.zeta.format), regs.zeta_enable, regs.color_mask_common,
        regs.color_mask[0].raw, regs.blend.enable[0], regs.depth_test_enable,
        regs.depth_write_enabled, static_cast<u32>(regs.depth_test_func), regs.rasterize_enable,
        regs.gl_cull_test_enabled, static_cast<u32>(regs.gl_cull_face),
        static_cast<u32>(regs.gl_front_face), sc.enable, sc.min_x.Value(), sc.max_x.Value(),
        sc.min_y.Value(), sc.max_y.Value(), std::bit_cast<u32>(vp.scale_x),
        std::bit_cast<u32>(vp.scale_y), std::bit_cast<u32>(vp.scale_z),
        std::bit_cast<u32>(vp.translate_x), std::bit_cast<u32>(vp.translate_y),
        std::bit_cast<u32>(vp.translate_z), host_execute,
        static_cast<u32>(regs.render_enable.mode), static_cast<u32>(regs.render_enable_override),
        regs.render_enable.Address());
    fields += fmt::format(
        R"(,"guest_rt0_format":{},"guest_rt0_extent":[{},{}],"blend_per_target":{},"blend_global":{},"blend_target0":{},"iterated_blend":{},"blend_constant_bits":[{},{},{},{}],"guest_draw":{{"topology":{},"vertex_first":{},"vertex_count":{},"index_first":{},"index_count":{},"base_index":{},"base_instance":{},"instances":{}}})",
        static_cast<u32>(regs.rt[0].format), regs.rt[0].width, regs.rt[0].height,
        regs.blend_per_target_enabled, blend_json(regs.blend), blend_json(regs.blend_per_target[0]),
        regs.iterated_blend.enable, std::bit_cast<u32>(regs.blend_color.r),
        std::bit_cast<u32>(regs.blend_color.g), std::bit_cast<u32>(regs.blend_color.b),
        std::bit_cast<u32>(regs.blend_color.a), static_cast<u32>(draw.topology),
        draw.vertex_buffer.first, draw.vertex_buffer.count, draw.index_buffer.first,
        draw.index_buffer.count, draw.base_index, draw.base_instance, draw.instance_count);
    return fields;
}

VkViewport GetViewportState(const Device& device, const Maxwell& regs, size_t index, float scale) {
    const auto& src = regs.viewport_transform[index];
    const auto conv = [scale](float value) {
        float new_value = value * scale;
        if (scale < 1.0f) {
            const bool sign = std::signbit(value);
            new_value = std::round(std::abs(new_value));
            new_value = sign ? -new_value : new_value;
        }
        return new_value;
    };
    const float x = conv(src.translate_x - src.scale_x);
    const float width = conv(src.scale_x * 2.0f);
    float y = conv(src.translate_y - src.scale_y);
    float height = conv(src.scale_y * 2.0f);

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const bool y_negate = !device.IsNvViewportSwizzleSupported() &&
                          src.swizzle.y == Maxwell::ViewportSwizzle::NegativeY;

    if (lower_left) {
        // Flip by surface clip height
        y += conv(static_cast<f32>(regs.surface_clip.height));
        height = -height;
    }

    if (y_negate) {
        // Flip by viewport height
        y += height;
        height = -height;
    }

    const float reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
    VkViewport viewport{
        .x = x,
        .y = y,
        .width = width != 0.0f ? width : 1.0f,
        .height = height != 0.0f ? height : 1.0f,
        .minDepth = src.translate_z - src.scale_z * reduce_z,
        .maxDepth = src.translate_z + src.scale_z,
    };
    if (!device.IsExtDepthRangeUnrestrictedSupported()) {
        viewport.minDepth = std::clamp(viewport.minDepth, 0.0f, 1.0f);
        viewport.maxDepth = std::clamp(viewport.maxDepth, 0.0f, 1.0f);
    }
    return viewport;
}

VkRect2D GetScissorState(const Maxwell& regs, size_t index, u32 up_scale = 1, u32 down_shift = 0) {
    const auto& src = regs.scissor_test[index];
    VkRect2D scissor{};
    const auto scale_up = [&](s32 value) -> s32 {
        if (value == 0) {
            return 0U;
        }
        const s32 upset = value * up_scale;
        s32 acumm = 0;
        if ((up_scale >> down_shift) == 0) {
            acumm = upset % 2;
        }
        const s32 converted_value = (value * up_scale) >> down_shift;
        return value < 0 ? std::min<s32>(converted_value - acumm, -1)
                         : std::max<s32>(converted_value + acumm, 1);
    };

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const s32 clip_height = regs.surface_clip.height;

    // Flip coordinates if lower left
    s32 min_y = lower_left ? (clip_height - src.max_y) : src.min_y.Value();
    s32 max_y = lower_left ? (clip_height - src.min_y) : src.max_y.Value();

    // Bound to render area
    min_y = (std::max)(min_y, 0);
    max_y = (std::max)(max_y, 0);

    if (src.enable) {
        scissor.offset.x = scale_up(src.min_x);
        scissor.offset.y = scale_up(min_y);
        scissor.extent.width = scale_up(src.max_x - src.min_x);
        scissor.extent.height = scale_up(max_y - min_y);
    } else {
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent.width = (std::numeric_limits<s32>::max)();
        scissor.extent.height = (std::numeric_limits<s32>::max)();
    }
    return scissor;
}

DrawParams MakeDrawParams(const Tegra::Engines::Maxwell3D::DrawManager::State& draw_state, u32 num_instances, bool is_indexed) {
    DrawParams params{
        .base_instance = draw_state.base_instance,
        .num_instances = num_instances,
        .base_vertex = is_indexed ? draw_state.base_index : draw_state.vertex_buffer.first,
        .num_vertices = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
        .first_index = is_indexed ? draw_state.index_buffer.first : 0,
        .is_indexed = is_indexed,
    };
    // 6 triangle vertices per quad, base vertex is part of the index
    // See BindQuadIndexBuffer for more details
    if (draw_state.topology == Maxwell::PrimitiveTopology::Quads) {
        params.num_vertices = (params.num_vertices / 4) * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    } else if (draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
        params.num_vertices = (params.num_vertices - 2) / 2 * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    }
    return params;
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
        return false;
    default:
        return true;
    }
}

bool IsPrimitiveRestartSupported(const Device& device, VkPrimitiveTopology topology) {
    return ((topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsTopologyListPrimitiveRestartSupported()) ||
            SupportsPrimitiveRestart(topology) ||
            (topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsPatchListPrimitiveRestartSupported()));
}



// Private task/mesh replay diagnostic; caller holds the existing buffer-cache lock.
std::string TaskInputConstants(Tegra::Engines::Maxwell3D& maxwell,
                              Tegra::MemoryManager& memory, BufferCache& buffer_cache,
                              const GraphicsPipeline& pipeline) {
    constexpr u32 BankByteCap = 64 * 1024;
    constexpr u32 TotalByteCap = 512 * 1024;
    u32 captured_bytes{};
    std::string banks;
    for (const u32 stage : {0U, 2U}) {
        const auto* info = pipeline.CaptureStageInfo(stage);
        if (!info) continue;
        const auto& cbufs = maxwell.state.shader_stages[stage].const_buffers;
        for (u32 bank = 0; bank < cbufs.size() && bank < 32; ++bank) {
            if ((info->constant_buffer_mask & (1U << bank)) == 0) continue;
            const auto& cb = cbufs[bank];
            const u32 bytes = cb.enabled
                ? (std::min)({cb.size, BankByteCap, TotalByteCap - captured_bytes}) & ~3U : 0;
            captured_bytes += bytes;
            std::string words, invalid, untranslated, dirty;
            const auto append = [](std::string& output, u32 value) {
                if (!output.empty()) output += ',';
                output += std::to_string(value);
            };
            for (u32 offset = 0; offset < bytes; offset += sizeof(u32)) {
                const bool address_valid = cb.address <= (std::numeric_limits<GPUVAddr>::max)() - offset;
                const GPUVAddr address = address_valid ? cb.address + offset : 0;
                const bool valid = address_valid &&
                    address <= (std::numeric_limits<GPUVAddr>::max)() - (sizeof(u32) - 1) &&
                    memory.IsWithinGPUAddressRange(address) &&
                    memory.IsWithinGPUAddressRange(address + sizeof(u32) - 1) &&
                    memory.IsFullyMappedRange(address, sizeof(u32));
                if (!words.empty()) words += ',';
                if (!valid) {
                    words += "null";
                    append(invalid, offset);
                    continue;
                }
                u32 word{};
                memory.ReadBlockUnsafe(address, &word, sizeof(word));
                words += std::to_string(word);
                const auto device_address = memory.GpuToCpuAddress(address);
                if (!device_address) append(untranslated, offset);
                else if (buffer_cache.IsRegionGpuModified(*device_address, sizeof(word)))
                    append(dirty, offset);
            }
            banks += fmt::format(
                R"({}{{"stage":{},"bank":{},"address":"{:#x}","enabled":{},"size":{},"captured_bytes":{},"truncated":{},"words":[{}],"invalid_offsets":[{}],"untranslated_offsets":[{}],"gpu_modified_offsets":[{}]}})",
                banks.empty() ? "" : ",", stage, bank, cb.address, cb.enabled, cb.size, bytes,
                bytes != cb.size, words, invalid, untranslated, dirty);
        }
    }
    std::string headers;
    const auto& regs = maxwell.regs;
    for (const u32 slot : {1U, 3U, 4U}) {
        const GPUVAddr address = (u64{regs.pipelines[slot].reserved[0] & 0xff} << 32) |
                                 regs.pipelines[slot].reserved[1];
        std::array<u32, 20> words{};
        constexpr size_t HeaderBytes = sizeof(words);
        const bool valid = regs.IsShaderConfigEnabled(slot) &&
            address <= (std::numeric_limits<GPUVAddr>::max)() - (HeaderBytes - 1) &&
            memory.IsWithinGPUAddressRange(address) &&
            memory.IsWithinGPUAddressRange(address + HeaderBytes - 1) &&
            memory.IsFullyMappedRange(address, HeaderBytes);
        std::string values;
        bool translated = false;
        bool dirty = false;
        if (valid) {
            memory.ReadBlockUnsafe(address, words.data(), HeaderBytes);
            const auto device_address = memory.GpuToCpuAddress(address);
            translated = device_address.has_value();
            if (translated) dirty = buffer_cache.IsRegionGpuModified(*device_address, HeaderBytes);
            for (const u32 word : words) {
                if (!values.empty()) values += ',';
                values += std::to_string(word);
            }
        }
        headers += fmt::format(
            R"({}{{"slot":{},"address":"{:#x}","enabled":{},"valid":{},"translated":{},"gpu_modified":{},"words":[{}]}})",
            headers.empty() ? "" : ",", slot, address, regs.IsShaderConfigEnabled(slot), valid,
            translated, dirty, values);
    }
    return fmt::format(
        R"("constant_buffers_source":"guest_cpu_snapshot","constant_buffer_byte_cap":{},"constant_buffer_captured_bytes":{},"constant_buffers":[{}],"mesh_enabled":{},"mesh_registers":[{},{},{}],"program_headers_source":"guest_cpu_snapshot","program_headers":[{}])",
        TotalByteCap, captured_bytes, banks, regs.IsMeshEnabled(),
        regs.reg_array[Maxwell::MeshShaderA],
        regs.reg_array[Maxwell::MeshShaderB],
        regs.reg_array[Maxwell::MeshInitShader], headers);
}

// Private producer diagnostic: CPU snapshots retain dirty/invalid word locations.
std::string ComputeProducerConstants(Tegra::Engines::KeplerCompute& compute,
                                     Tegra::MemoryManager& memory, BufferCache& buffer_cache,
                                     u64 native_key, u32 used_mask, bool live_fog_inputs = false) {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_PRODUCER");
        return value && std::strcmp(value, "1") == 0;
    }();
    if ((!enabled && !FrameDump::ComputeCaptureStopEnabled() && !live_fog_inputs) ||
        native_key != FrameDump::ProducerNativeKey()) return {};
    constexpr u32 BankByteCap = 64 * 1024;
    constexpr u32 TotalByteCap = 512 * 1024;
    constexpr u32 TableByteCap = 2048;
    const bool table_advisory = native_key == 0xd6dcff9e09b573e4ULL;
    const bool expanded = native_key != 0xe883414e8c9be33bULL;
    const u32 bank_budget = TotalByteCap - (table_advisory ? TableByteCap : 0);
    u32 captured_bytes{};
    std::array<std::optional<u32>, 1024> table_words{};
    std::array<bool, 1024> table_dirty{};
    std::array<bool, 1024> table_translated{};
    const auto& qmd = compute.launch_description;
    std::scoped_lock lock{buffer_cache.mutex};
    std::string banks;
    constexpr std::array<u32, 8> TableFirst{4, 0, 1, 2, 3, 5, 6, 7};
    for (u32 ordinal = 0; ordinal < qmd.const_buffer_config.size(); ++ordinal) {
        const u32 bank = table_advisory ? TableFirst[ordinal] : ordinal;
        const u32 enabled_or_used = used_mask | qmd.const_buffer_enable_mask.Value();
        if (expanded ? (enabled_or_used & (1U << bank)) == 0 : (bank < 3 || bank > 5)) continue;
        const auto& cb = qmd.const_buffer_config[bank];
        const GPUVAddr base = cb.Address();
        const bool bank_enabled = (qmd.const_buffer_enable_mask & (1U << bank)) != 0;
        const u32 limit = expanded ? bank_budget - captured_bytes : BankByteCap;
        const u32 bytes = expanded && !bank_enabled ? 0 : (std::min)({cb.size.Value(), BankByteCap, limit}) & ~3U;
        captured_bytes += bytes;
        std::string words, invalid, untranslated, dirty;
        const auto append = [](std::string& output, u32 value) {
            if (!output.empty()) output += ',';
            output += std::to_string(value);
        };
        for (u32 offset = 0; offset < bytes; offset += sizeof(u32)) {
            const bool address_valid = base <= (std::numeric_limits<GPUVAddr>::max)() - offset;
            const GPUVAddr address = address_valid ? base + offset : 0;
            const bool valid = bank_enabled && address_valid &&
                address <= (std::numeric_limits<GPUVAddr>::max)() - (sizeof(u32) - 1) &&
                memory.IsWithinGPUAddressRange(address) &&
                memory.IsWithinGPUAddressRange(address + sizeof(u32) - 1) &&
                memory.IsFullyMappedRange(address, sizeof(u32));
            if (!words.empty()) words += ',';
            if (!valid) {
                words += "null";
                append(invalid, offset);
                continue;
            }
            u32 word{};
            memory.ReadBlockUnsafe(address, &word, sizeof(word));
            words += std::to_string(word);
            const auto device_address = memory.GpuToCpuAddress(address);
            if (!device_address) {
                append(untranslated, offset);
            } else if (buffer_cache.IsRegionGpuModified(*device_address, sizeof(word))) {
                append(dirty, offset);
            }
            if (table_advisory && bank == 4 && offset / 4 < table_words.size()) {
                table_words[offset / 4] = word;
                table_translated[offset / 4] = device_address.has_value();
                table_dirty[offset / 4] = device_address &&
                    buffer_cache.IsRegionGpuModified(*device_address, sizeof(word));
            }
        }
        banks += fmt::format(
            R"({}{{"stage":"compute","bank":{},"address":"{:#x}","enabled":{},"size":{},"captured_bytes":{},"truncated":{},"words":[{}],"invalid_offsets":[{}],"untranslated_offsets":[{}],"gpu_modified_offsets":[{}]}})",
            banks.empty() ? "" : ",", bank, base, bank_enabled, cb.size.Value(), bytes,
            bytes != cb.size.Value(), words, invalid, untranslated, dirty);
    }
    std::string advisory;
    if (table_advisory) {
        const auto& cb = qmd.const_buffer_config[4];
        const bool enabled_used = ((qmd.const_buffer_enable_mask.Value() & used_mask) & (1U << 4)) != 0;
        const u32 slots = enabled_used ? (std::min)(cb.size.Value() / 16, 256U) : 0;
        std::string rows;
        u32 entry_bytes{};
        for (u32 slot = 0; slot < slots; ++slot) {
            const auto& low = table_words[slot * 4];
            const auto& high = table_words[slot * 4 + 1];
            const bool root_valid = low.has_value() && high.has_value();
            const u64 root = root_valid ? u64{*low} | (u64{*high} << 32) : 0;
            const bool root_dirty = table_dirty[slot * 4] || table_dirty[slot * 4 + 1];
            const bool root_translated = table_translated[slot * 4] && table_translated[slot * 4 + 1];
            const bool entry_valid = root_valid && root != 0 && (root & 3) == 0 &&
                root <= (std::numeric_limits<GPUVAddr>::max)() - 7 &&
                memory.IsWithinGPUAddressRange(root) && memory.IsWithinGPUAddressRange(root + 7) &&
                memory.IsFullyMappedRange(root, 8);
            const auto device = entry_valid ? memory.GpuToCpuAddress(root) : std::nullopt;
            const auto device_high = entry_valid ? memory.GpuToCpuAddress(root + 4) : std::nullopt;
            bool entry_dirty{};
            u64 entry{};
            if (entry_valid) {
                memory.ReadBlockUnsafe(root, &entry, 8);
                entry_bytes += 8;
                for (u32 word = 0; word < 2; ++word) {
                    if (const auto d = memory.GpuToCpuAddress(root + word * 4))
                        entry_dirty |= buffer_cache.IsRegionGpuModified(*d, 4);
                }
            }
            rows += fmt::format(
                R"({}{{"slot":{},"cbuf_offset":{},"root_valid":{},"root_translated":{},"root":"{:#x}","root_gpu_modified":{},"entry_offset":0,"entry_valid":{},"entry_translated":{},"entry_gpu_modified":{},"entry":{}}})",
                rows.empty() ? "" : ",", slot, slot * 16, root_valid, root_translated, root, root_dirty,
                entry_valid, device.has_value() && device_high.has_value(), entry_dirty,
                entry_valid ? fmt::format("\"{:#x}\"", entry) : "null");
        }
        captured_bytes += entry_bytes;
        advisory = fmt::format(
            R"(,"producer_table_advisory":{{"source":"guest_cpu_no_flush","bank":4,"enabled_and_used":{},"slot_stride":16,"slot_cap":256,"slots":{},"truncated":{},"entry_byte_cap":2048,"entry_bytes":{},"rows":[{}]}})",
            enabled_used, slots, enabled_used && cb.size.Value() / 16 > slots, entry_bytes, rows);
    }
    return fmt::format(R"(,"producer_constant_buffers_source":"guest_cpu_snapshot","producer_constant_buffers_byte_cap":{},"producer_constant_buffers_captured_bytes":{},"producer_constant_buffers_used_mask":{},"producer_constant_buffers":[{}])",
                       TotalByteCap, captured_bytes, used_mask, banks) + advisory;
}

// Private capture-only diagnostic: fixed lighting-kernel uniforms, no GPU flush/readback.
std::string ComputeLightingConstants(Tegra::Engines::KeplerCompute& compute,
                                     Tegra::MemoryManager& memory, BufferCache& buffer_cache,
                                     u64 native_key, bool live_fog_inputs = false) {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_COMPUTE_UNIFORMS");
        return value && std::strcmp(value, "1") == 0;
    }();
    if ((!enabled && !live_fog_inputs) || (native_key != 0xe883414e8c9be33bULL &&
                     native_key != 0xec68e40eb783ff32ULL)) return {};
    const auto& qmd = compute.launch_description;
    std::scoped_lock lock{buffer_cache.mutex};
    const auto constants = [&](u32 bank, std::span<const u32> offsets) {
        const auto& cb = qmd.const_buffer_config[bank];
        const GPUVAddr base = cb.Address();
        const bool cb_enabled = (qmd.const_buffer_enable_mask & (1U << bank)) != 0;
        std::string words;
        for (const u32 offset : offsets) {
            u32 word{};
            const bool address_valid = base <= (std::numeric_limits<GPUVAddr>::max)() - offset;
            const GPUVAddr address = address_valid ? base + offset : 0;
            const bool valid = cb_enabled && offset <= cb.size && sizeof(word) <= cb.size - offset &&
                address_valid && address <= (std::numeric_limits<GPUVAddr>::max)() - (sizeof(word) - 1) &&
                memory.IsWithinGPUAddressRange(address) &&
                memory.IsWithinGPUAddressRange(address + sizeof(word) - 1) &&
                memory.IsFullyMappedRange(address, sizeof(word));
            if (valid) memory.ReadBlockUnsafe(address, &word, sizeof(word));
            const auto device_address = valid ? memory.GpuToCpuAddress(address) : std::nullopt;
            const bool gpu_modified = device_address &&
                buffer_cache.IsRegionGpuModified(*device_address, sizeof(word));
            words += fmt::format(
                R"({}{{"offset":{},"address":"{:#x}","address_valid":{},"valid":{},"device_address_valid":{},"gpu_modified":{},"word":{}}})",
                words.empty() ? "" : ",", offset, address, address_valid, valid,
                device_address.has_value(), device_address ? (gpu_modified ? "true" : "false") : "null",
                valid ? std::to_string(word) : "null");
        }
        return fmt::format(
            R"({{"stage":"compute","bank":{},"address":"{:#x}","enabled":{},"size":{},"words":[{}]}})",
            bank, base, cb_enabled, cb.size.Value(), words);
    };
    constexpr std::array e883_c3{
        0x230U, 0x234U, 0x238U, 0x23cU, 0x240U, 0x244U, 0x248U, 0x24cU, 0x250U, 0x254U,
        0x258U, 0x25cU, 0x260U, 0x264U, 0x268U, 0x26cU, 0x270U, 0x274U, 0x278U, 0x27cU,
        0x280U, 0x284U, 0x288U, 0x28cU, 0x290U, 0x294U, 0x940U, 0x944U, 0x948U, 0x94cU,
        0x950U, 0x954U, 0x958U, 0x96cU, 0x970U, 0x3150U, 0x3154U, 0x3158U, 0x3170U, 0x3174U,
        0x3178U, 0x317cU, 0x3180U, 0x3184U, 0x3188U, 0x318cU, 0x3190U, 0x3194U, 0x3198U, 0x319cU,
        0x31a0U, 0x31a4U, 0x31a8U, 0x31acU
    };
    constexpr std::array e883_c4{
        0x3d0U, 0x3d4U
    };
    constexpr std::array e883_c5{
        0x80U, 0x84U, 0x88U, 0x8cU, 0x90U, 0x94U, 0x98U, 0x9cU, 0xa0U, 0xa4U,
        0xa8U, 0xacU, 0xc0U, 0xc4U, 0xc8U, 0xccU, 0xd0U, 0xd4U, 0xd8U, 0xdcU,
        0xe0U, 0xe4U, 0xe8U, 0xecU, 0x100U, 0x104U, 0x108U, 0x10cU, 0x110U, 0x114U,
        0x118U, 0x11cU, 0x120U, 0x124U, 0x128U, 0x12cU, 0x130U, 0x134U, 0x138U, 0x13cU,
        0x140U, 0x144U, 0x148U, 0x14cU, 0x150U, 0x154U, 0x158U, 0x15cU, 0x160U, 0x164U,
        0x168U, 0x16cU, 0x170U, 0x174U, 0x178U, 0x17cU, 0x180U, 0x184U, 0x188U, 0x18cU,
        0x190U, 0x194U, 0x198U, 0x19cU, 0x1a0U, 0x1a4U, 0x1a8U, 0x1acU, 0x1bcU, 0x1c8U,
        0x1d0U, 0x1d4U, 0x1d8U, 0x1e0U, 0x1e4U, 0x1e8U, 0x1ecU, 0x1f0U, 0x1f4U, 0x1f8U,
        0x1fcU, 0x200U, 0x204U, 0x208U, 0x210U, 0x288U, 0x290U, 0x294U, 0x298U, 0x29cU,
        0x2a0U, 0x2a4U, 0x2a8U, 0x2acU, 0x2b0U, 0x2b4U, 0x2b8U, 0x2bcU, 0x2c0U, 0x2c4U,
        0x2c8U, 0x2ccU, 0x2d0U, 0x2e0U, 0x2e4U, 0x2e8U, 0x2ecU, 0x2f0U, 0x2f4U, 0x2f8U,
        0x300U, 0x304U, 0x308U, 0x310U, 0x314U, 0x318U, 0x320U, 0x324U, 0x328U, 0x32cU,
        0x330U, 0x334U, 0x338U, 0x33cU, 0x340U, 0x350U, 0x354U, 0x358U, 0x35cU, 0x360U,
        0x364U, 0x368U, 0x370U, 0x374U, 0x378U, 0x380U, 0x384U, 0x388U, 0x390U, 0x394U,
        0x398U, 0x39cU, 0x3a0U, 0x3a4U, 0x3a8U, 0x3acU
    };
    constexpr std::array ec68_c3{
        0x180U, 0x184U, 0x18cU, 0x200U, 0x204U, 0x208U
    };
    const std::string banks = native_key == 0xec68e40eb783ff32ULL ? constants(3, ec68_c3)
        : constants(3, e883_c3) + "," + constants(4, e883_c4) + "," + constants(5, e883_c5);
    return fmt::format(R"(,"lighting_constants_source":"guest_cpu_snapshot","lighting_constants_complete":{},"lighting_constants":[{}])",
                       native_key == 0xec68e40eb783ff32ULL, banks);
}

// Private opt-in diagnostic: a bounded CPU snapshot, without flushing GPU resources.
std::string ComputeMetadata(Tegra::Engines::KeplerCompute& compute, Tegra::MemoryManager& memory,
                            BufferCache& buffer_cache, u64* native_key_out = nullptr,
                            bool live_fog_inputs = false) {
    using Shader::Ampere::Instruction;
    const auto mapped = [&](GPUVAddr address, size_t size) {
        return size && address <= (std::numeric_limits<u64>::max)() - size &&
               memory.IsWithinGPUAddressRange(address) &&
               memory.IsWithinGPUAddressRange(address + size - 1) &&
               memory.IsFullyMappedRange(address, size);
    };
    const auto words_json = [](auto words) {
        std::string text;
        for (const u32 word : words) text += fmt::format("{}{}", text.empty() ? "" : ",", word);
        return text;
    };
    const auto& qmd = compute.launch_description;
    const GPUVAddr program = compute.regs.code_loc.Address();
    std::vector<Instruction> code;
    Shader::Ampere::ProgramCaptureState capture;
    bool complete = false, invalid_target = false;
    // 4096 instructions / 64 KiB per row; larger programs retain their prefix and no exact key.
    for (u32 offset = 0; offset < 65536 && !complete && !invalid_target;) {
        const GPUVAddr address = program + offset;
        if (address < program) break;
        const u32 bytes = std::min(4096U - static_cast<u32>(address & 4095), 65536U - offset) & ~15U;
        if (!bytes || !mapped(address, bytes)) break;
        const u32 begin = static_cast<u32>(code.size());
        code.resize(begin + bytes / sizeof(Instruction));
        memory.ReadBlockUnsafe(address, code.data() + begin, bytes);
        for (u32 index = begin; index < code.size(); ++index) {
            const auto result = capture.Observe(code[index], index);
            if (result == Shader::Ampere::ProgramCaptureState::Result::Complete ||
                result == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) {
                complete = result == Shader::Ampere::ProgramCaptureState::Result::Complete;
                invalid_target = !complete;
                code.resize(index + 1);
                break;
            }
        }
        offset += bytes;
    }
    const auto cbuf_word = [&](u32 bank, u32 offset) -> std::optional<u32> {
        if (bank >= qmd.const_buffer_config.size() ||
            !(qmd.const_buffer_enable_mask & (1U << bank))) return {};
        const auto& buffer = qmd.const_buffer_config[bank];
        const GPUVAddr address = buffer.Address() + offset;
        if (offset > buffer.size || buffer.size - offset < sizeof(u32) ||
            address < buffer.Address() || !mapped(address, sizeof(u32))) return {};
        u32 value{};
        memory.ReadBlockUnsafe(address, &value, sizeof(value));
        return value;
    };
    std::string constants;
    for (u32 offset = 0; offset < 0x100; offset += 4) {
        const auto value = cbuf_word(0, offset);
        if (!value) break;
        constants += fmt::format("{}{}", constants.empty() ? "" : ",", *value);
    }
    std::string descriptors;
    std::set<std::tuple<u32, u32, bool>> seen;
    for (const auto& insn : code) {
        const u32 opcode = insn.Bits(0, 12);
        const bool write = opcode == 0xb9b;
        if (opcode != 0xb60 && opcode != 0xb63 && opcode != 0xb66 &&
            opcode != 0xb97 && !write) continue;
        const u32 bank = insn.Bits(54, 5), offset = insn.Bits(40, 14) * 4;
        if (!seen.emplace(bank, offset, write).second) continue;
        if (seen.size() > 64) break;
        const auto value = cbuf_word(bank, offset);
        std::string fields = fmt::format(
            R"("cb":{},"offset":{},"opcode":{},"write":{},"handle":{})",
            bank, offset, opcode, write, value ? std::to_string(*value) : "null");
        if (value) {
            const auto [index, sampler] = Tegra::Texture::TexturePair(*value, qmd.linked_tsc != 0);
            const u32 tic_index = Tegra::Texture::TextureHeaderIndex(index, compute.regs.tic.limit, true);
            const GPUVAddr tic_address = compute.regs.tic.Address() +
                                        u64{tic_index} * sizeof(Tegra::Texture::TICEntry);
            const bool valid = tic_index <= compute.regs.tic.limit &&
                               tic_address >= compute.regs.tic.Address() &&
                               mapped(tic_address, sizeof(Tegra::Texture::TICEntry));
            fields += fmt::format(R"(,"tic":{},"resolved_tic":{},"tsc":{},"tic_valid":{})",
                                  index, tic_index, sampler, valid);
            // Same bounded CPU snapshot as TIC metadata; never flush or download the sampler.
            const u32 tsc_limit = qmd.linked_tsc ? compute.regs.tic.limit : compute.regs.tsc.limit;
            const GPUVAddr tsc_base = compute.regs.tsc.Address();
            const u64 tsc_offset = u64{sampler} * sizeof(Tegra::Texture::TSCEntry);
            const bool tsc_no_wrap =
                tsc_base <= (std::numeric_limits<GPUVAddr>::max)() - tsc_offset;
            const GPUVAddr tsc_address = tsc_no_wrap ? tsc_base + tsc_offset : 0;
            const bool tsc_valid = sampler <= tsc_limit && tsc_no_wrap &&
                                   mapped(tsc_address, sizeof(Tegra::Texture::TSCEntry));
            fields += fmt::format(
                R"(,"linked_tsc":{},"tsc_address":"{:#x}","tsc_limit":{},"tsc_valid":{})",
                qmd.linked_tsc != 0, tsc_address, tsc_limit, tsc_valid);
            if (tsc_valid) {
                std::array<u32, 8> tsc_words{};
                memory.ReadBlockUnsafe(tsc_address, tsc_words.data(), sizeof(Tegra::Texture::TSCEntry));
                fields += fmt::format(R"(,"tsc_words":[{}])", words_json(tsc_words));
            }
            if (valid) {
                Tegra::Texture::TICEntry tic{};
                memory.ReadBlockUnsafe(tic_address, &tic, sizeof(tic));
                std::array<u32, 8> words{};
                std::memcpy(words.data(), &tic, sizeof(tic));
                fields += fmt::format(R"(,"tic_words":[{}],"gpu_addr":"{:#x}","format":{},"width":{},"height":{},"depth":{},"base_layer":{})",
                                      words_json(words), tic.Address(), static_cast<u32>(tic.format.Value()),
                                      tic.Width(), tic.Height(), tic.Depth(), tic.BaseLayer());
            }
        }
        descriptors += fmt::format("{}{{{}}}", descriptors.empty() ? "" : ",", fields);
    }
    const u64 key = complete ? Shader::Ampere::NativeProgramKey(code) : 0;
    if (native_key_out) *native_key_out = key;
    return fmt::format(
        R"("source":"guest_cpu_snapshot","program":"{:#x}","native_key":"{:016x}","program_complete":{},"invalid_target":{},"instructions":{},"grid":[{},{},{}],"block":[{},{},{}],"cbuf0_address":"{:#x}","cbuf0_size":{},"cbuf0_words":[{}],"descriptors":[{}])",
        program, key, complete, invalid_target, code.size(), qmd.grid_dim_x.Value(),
        qmd.grid_dim_y.Value(), qmd.grid_dim_z.Value(), qmd.block_dim_x.Value(),
        qmd.block_dim_y.Value(), qmd.block_dim_z.Value(), qmd.const_buffer_config[0].Address(),
        qmd.const_buffer_config[0].size.Value(), constants, descriptors) +
        ComputeLightingConstants(compute, memory, buffer_cache, key, live_fog_inputs);
}

} // Anonymous namespace

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,
                                   Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                   const Device& device_, MemoryAllocator& memory_allocator_,
                                   StateTracker& state_tracker_, Scheduler& scheduler_)
    : gpu{gpu_}, device_memory{device_memory_}, device{device_},
      memory_allocator{memory_allocator_}, state_tracker{state_tracker_}, scheduler{scheduler_},
      staging_pool(device, memory_allocator, scheduler), descriptor_pool(device, scheduler),
      guest_descriptor_queue(device, UpdateDescriptorQueue::GUEST_FRAME_PAYLOAD_SIZE,
                             device.IsExtDescriptorBufferSupported()),
      compute_pass_descriptor_queue(device, UpdateDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE),
      descriptor_buffer_ring(device, memory_allocator),
      blit_image(device, scheduler, state_tracker, descriptor_pool), render_pass_cache(device),
      texture_cache_runtime{
          device,     scheduler,         memory_allocator, staging_pool,
          blit_image, render_pass_cache, descriptor_pool,  compute_pass_descriptor_queue},
      texture_cache(texture_cache_runtime, device_memory),
      buffer_cache_runtime(device, memory_allocator, scheduler, staging_pool,
                           guest_descriptor_queue, compute_pass_descriptor_queue, descriptor_pool),
      buffer_cache(device_memory, buffer_cache_runtime),
      query_cache_runtime(this, device_memory, buffer_cache, device, memory_allocator, scheduler,
                          staging_pool, compute_pass_descriptor_queue, descriptor_pool, texture_cache),
      query_cache(gpu, *this, device_memory, query_cache_runtime),
      pipeline_cache(device_memory, device, scheduler, descriptor_pool, guest_descriptor_queue,
                     descriptor_buffer_ring, render_pass_cache, buffer_cache, texture_cache,
                     gpu.ShaderNotify()),
      accelerate_dma(buffer_cache, texture_cache, scheduler),
      fence_manager(*this, gpu, texture_cache, buffer_cache, query_cache, device, scheduler),
      frame_dump(device, memory_allocator, scheduler), wfi_event(device.GetLogical().CreateEvent()) {
    scheduler.SetQueryCache(query_cache);
    StorageFeedback::Initialize(device, memory_allocator, scheduler);
    VideoCommon::GpuSplit::Get().SetIdleHook(
        [](void* argument) { static_cast<Scheduler*>(argument)->FlushDeferred(); },
        &scheduler);
}

RasterizerVulkan::~RasterizerVulkan() {
    VideoCommon::GpuSplit::Get().SetIdleHook(nullptr, nullptr);
    scheduler.WaitWorker();
    scheduler.Finish();
    StorageFeedback::Shutdown();
}

// Experimental adaptation of PokeDinner commit 76df4591a747: execute the reset shader's
// indirect record writes on the host until general GPU-address translation is available.
bool RasterizerVulkan::TryNx2IndirectReset() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_RESETFIX");
        return !value || std::string_view{value} != "0";
    }();
    if (!enabled || !maxwell3d->GetNativeMacroEngine().native_turing ||
        !pipeline_cache.IsIndirectResetShader()) {
        return false;
    }
    const auto reject = [](std::string_view reason) {
        static u32 reported_rejections{};
        if (reported_rejections++ < 8) {
            LOG_WARNING(HW_GPU, "Experimental indirect reset skipped: {}", reason);
        }
        return false;
    };
    const auto in_gpu_range = [this](u64 address, size_t size) {
        return size != 0 && address <= (std::numeric_limits<u64>::max)() - size &&
               gpu_memory->IsWithinGPUAddressRange(address) &&
               gpu_memory->IsWithinGPUAddressRange(address + size - 1);
    };
    const auto& cbuf = maxwell3d->state.shader_stages[0].const_buffers[0];
    if (!cbuf.enabled || cbuf.size < 0x120 || !in_gpu_range(cbuf.address, 0x120) ||
        !gpu_memory->IsFullyMappedRange(cbuf.address + 0x110, 16)) {
        return reject("descriptor unavailable");
    }
    std::array<u64, 2> descriptor{};
    gpu_memory->ReadBlock(cbuf.address + 0x110, descriptor.data(), sizeof(descriptor));
    const u64 list = descriptor[0];
    const auto& draw = maxwell3d->draw_manager.draw_state;
    const u64 first = draw.vertex_buffer.first;
    const u64 count = draw.vertex_buffer.count;
    const auto range = VideoCommon::IndirectResetRange(list, static_cast<u32>(descriptor[1]),
                                                        first, count);
    if (!range || !in_gpu_range(range->begin, range->size) ||
        !gpu_memory->IsFullyMappedRange(range->begin, range->size)) {
        return reject("list outside mapped descriptor");
    }
    if (gpu_memory->IsMemoryDirty(range->begin, range->size, VideoCommon::CacheType::BufferCache)) {
        VideoCommon::Nx2Profile::WaitTag wait_tag{VideoCommon::Nx2Profile::WaitSource::IndirectList};
        gpu_memory->FlushRegion(range->begin, range->size, VideoCommon::CacheType::BufferCache);
    }
    std::vector<u64> pointers(count);
    gpu_memory->ReadBlockUnsafe(range->begin, pointers.data(), range->size);
    static constexpr std::array<u32, 5> record{0, 1, 0, 0, 0};
    for (const u64 pointer : pointers) {
        if (!pointer) {
            continue;
        }
        if (!in_gpu_range(pointer, sizeof(record)) ||
            !gpu_memory->IsFullyMappedRange(pointer, sizeof(record))) {
            return reject("record not mapped");
        }
        if (pointer < range->begin + range->size && range->begin < pointer + sizeof(record)) {
            return reject("record overlaps pointer list");
        }
        const auto segments = gpu_memory->GetSubmappedRange(pointer, sizeof(record));
        if (segments.size() != 1 || segments[0].first != pointer ||
            segments[0].second != sizeof(record)) {
            return reject("record crosses a mapping boundary");
        }
    }
    const std::span<const u8> bytes{reinterpret_cast<const u8*>(record.data()), sizeof(record)};
    for (const u64 pointer : pointers) {
        if (pointer) {
            AccelerateInlineToMemory(pointer, bytes.size(), bytes);
        }
    }
    static u32 reported{};
    if (reported++ < 5) {
        LOG_INFO(HW_GPU, "Experimental indirect reset: {} records", count);
    }
    return true;
}

void RasterizerVulkan::NoteComputeForGraphicsBarrier() {
    // Diagnostic only: keep GPU readback and normal/default synchronization unchanged.
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_COMPUTE_GRAPHICS_BARRIER");
        return value && std::string_view{value} == "1";
    }();
    if (enabled) {
        pending_compute_graphics_barrier = true;
    }
}

void RasterizerVulkan::ApplyComputeForGraphicsBarrier() {
    if (!pending_compute_graphics_barrier) {
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                             VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT |
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_INPUT_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT |
                                   VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                                   VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                               0, barrier);
    });
    pending_compute_graphics_barrier = false;
    static u32 reported{};
    if (reported < 8) {
        ++reported;
        LOG_INFO(Render_Vulkan, "Diagnostic compute-to-graphics memory barrier recorded ({}/8)",
                 reported);
    }
}

template <typename Func>
void RasterizerVulkan::PrepareDraw(bool is_indexed, Func&& draw_func) {
    const VideoCommon::CommandTraceScope draw_trace{"prepare_draw", 0, scheduler.CurrentTick(),
        is_indexed, maxwell3d->regs.rt[0].Address(), maxwell3d->regs.rt[0].width};
    if (frame_dump.Capturing() && frame_dump.BeforeDraw()) {
        scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
    }

    SCOPE_EXIT {
        gpu.TickWork();
    };
    bool submitted = false;
    SCOPE_EXIT {
        auto& native = maxwell3d->GetNativeMacroEngine();
        if (native.native_turing && !submitted) {
            native.native_execution_failed = true;
            LOG_DEBUG(HW_GPU, "Native graphics draw failed; refusing submission completion");
        }
    };
    namespace Profile = VideoCommon::Nx2Profile;
    {
        Profile::Scope scope{Profile::Zone::FlushWork};
        FlushWork();
        gpu_memory->FlushCaching();
    }

    if (frame_dump.Capturing()) frame_dump.DrainIfFull();

    const auto report_failure = [&](std::string_view reason) {
        if (frame_dump.CapturingDrawSubmissions()) {
            frame_dump.NoteDrawSubmission(
                fmt::format(R"("status":"skipped","command":"prepare","reason":"{}","is_indexed":{},)",
                            reason, is_indexed) + DrawSubmissionGuestState(maxwell3d->regs, maxwell3d->draw_manager.draw_state,
                                             maxwell3d->ShouldExecute()), false);
        }
        static std::set<std::tuple<u64, u64, std::string_view>> seen;
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        if (seen.size() < 64 && seen.emplace(vs, ps, reason).second) {
            LOG_WARNING(HW_GPU, "Draw skipped ({}): vs={:#x} ps={:#x} rt0={:#x} {}x{}", reason, vs, ps,
                        regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height);
        }
    };
    GraphicsPipeline* const pipeline{[&] {
        const VideoCommon::CommandTraceScope trace{"graphics_lookup", 0, scheduler.CurrentTick()};
        Profile::Scope scope{Profile::Zone::PipelineLookup};
        return pipeline_cache.CurrentGraphicsPipeline();
    }()};
    if (!pipeline) {
        report_failure("no pipeline");
        return;
    }
    if (!is_indexed && TryNx2IndirectReset()) {
        submitted = true;
        return;
    }
    // Task/mesh pipelines: the draw's vertex count is the number of task workgroups.
    mesh_draw = pipeline->IsMeshPipeline();
    // EDEN_NX2_NO_MESH=1: skip task/mesh draws as before they were supported (A/B comparisons).
    // EDEN_NX2_NO_MESH_FILE=<path>: skip them while that file exists (checked every 1024 draws),
    // to compare the same scene with and without them in one run.
    static const bool no_mesh_always{std::getenv("EDEN_NX2_NO_MESH") != nullptr};
    static const char* const no_mesh_file{std::getenv("EDEN_NX2_NO_MESH_FILE")};
    static bool no_mesh{no_mesh_always};
    static u32 no_mesh_check{};
    if (no_mesh_file && (no_mesh_check++ % 1024) == 0) {
        no_mesh = std::filesystem::exists(no_mesh_file);
    }
    if (mesh_draw && (no_mesh || !device.IsExtMeshShaderSupported())) {
        report_failure(no_mesh ? "EDEN_NX2_NO_MESH" : "no VK_EXT_mesh_shader");
        return;
    }
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    // update engine as channel may be different.
    pipeline->SetEngine(maxwell3d, gpu_memory);
    const u32 repack_dirty = buffer_cache_runtime.SetVertexRepackLayout(
        maxwell3d->regs, pipeline->HasDynamicVertexInput());
    if (repack_dirty) {
        auto& flags = maxwell3d->dirty.flags;
        flags[VideoCommon::Dirty::VertexBuffers] = true;
        flags[Dirty::VertexInput] = true;
        for (u32 binding = 0; binding < 32; ++binding) {
            if (repack_dirty & (1U << binding)) {
                flags[VideoCommon::Dirty::VertexBuffer0 + binding] = true;
            }
        }
    }
    VideoCommon::CommandTrace::Get().Mark("draw_configure.begin", scheduler.CurrentTick(), 0,
        reinterpret_cast<std::uintptr_t>(pipeline), pipeline->IsMeshPipeline(), is_indexed);
    if (Profile::Scope scope{Profile::Zone::Configure};
        !pipeline->Configure(is_indexed, frame_dump.GraphicsIndirectRangeTrial(),
                             frame_dump.GraphicsNativePointerTrial())) {
        VideoCommon::CommandTrace::Get().Mark("draw_configure.failed", scheduler.CurrentTick());
        report_failure("configure failed");
        return;
    }
    VideoCommon::CommandTrace::Get().Mark("draw_configure.end", scheduler.CurrentTick(), 0,
        reinterpret_cast<std::uintptr_t>(pipeline), pipeline->IsMeshPipeline(), is_indexed);

    if (pending_compute_graphics_barrier) {
        ApplyComputeForGraphicsBarrier();
        scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
    }

    {
        Profile::Scope scope{Profile::Zone::DynamicState};
        UpdateDynamicStates();
    }

    // TEMP diagnostic (EDEN_NX2_FOG_SCALE_LOG): the fog composite's UV scale and target.
    if (static const bool fog_log = std::getenv("EDEN_NX2_FOG_SCALE_LOG") != nullptr;
        fog_log && std::ranges::find(pipeline->ShaderHashes(), 0x621c4755238870a8ULL) !=
                       pipeline->ShaderHashes().end()) {
        static u32 count{};
        if (count++ % 30 == 0) {
            const auto& cb = maxwell3d->state.shader_stages[0].const_buffers[3];
            const auto& ps4 = maxwell3d->state.shader_stages[4].const_buffers[4];
            LOG_WARNING(Render_Vulkan, "fog composite volume scale {:.4f} {:.4f} {:.4f}",
                        std::bit_cast<f32>(gpu_memory->Read<u32>(ps4.address + 0x600)),
                        std::bit_cast<f32>(gpu_memory->Read<u32>(ps4.address + 0x604)),
                        std::bit_cast<f32>(gpu_memory->Read<u32>(ps4.address + 0x608)));
            const auto& rt = maxwell3d->regs.rt[0];
            const auto& viewport = maxwell3d->regs.viewport_transform[0];
            LOG_WARNING(Render_Vulkan,
                        "fog composite uv scale {:.4f} {:.4f}, rt0 {}x{}, viewport scale "
                        "{:.1f} {:.1f} translate {:.1f} {:.1f}, clip {}x{}",
                        std::bit_cast<f32>(gpu_memory->Read<u32>(cb.address + 0x120)),
                        std::bit_cast<f32>(gpu_memory->Read<u32>(cb.address + 0x124)), u32{rt.width},
                        u32{rt.height}, viewport.scale_x, viewport.scale_y,
                        viewport.translate_x, viewport.translate_y,
                        u32{maxwell3d->regs.surface_clip.width},
                        u32{maxwell3d->regs.surface_clip.height});
        }
    }
    if (frame_dump.Capturing()) {
        FrameDumpPass(texture_cache.GetFramebuffer());
        const auto& regs = maxwell3d->regs;
        const u64 guest_vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) |
                             regs.pipelines[1].reserved[1];
        const u64 guest_ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) |
                             regs.pipelines[5].reserved[1];
        frame_dump.NoteDraw(pipeline->ShaderHashes(), guest_vs, guest_ps);

        if (((!mesh_draw && frame_dump.CapturingMaterialInputs()) ||
             (frame_dump.CapturingCanopyInputs() && mesh_draw == frame_dump.CanopyInputIsMesh())) &&
            pipeline->CaptureFoliageInputs(frame_dump, {})) {
            scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
        }
        if (frame_dump.CapturingTaskInputs() && mesh_draw) {
            const auto hashes = pipeline->ShaderHashes();
            const u64 task_hash = hashes.size() > 1 ? hashes[1] : 0;
            const u64 mesh_hash = hashes.size() > 3 ? hashes[3] : 0;
            if (task_hash == 0xf378ff9ca77352f0ULL && hashes.size() > 5 &&
                ((mesh_hash == 0x277d8ef0f4003ac4ULL && hashes[5] == 0x4b0ca12cb55286d7ULL) ||
                 (mesh_hash == 0x87183e2c33d87f2aULL && hashes[5] == 0xc4a9495a97b2ae03ULL))) {
                const auto fields = frame_dump.TaskInputsRecorded() ? std::string{} :
                    TaskInputConstants(*maxwell3d, *gpu_memory, buffer_cache, *pipeline);
                if (pipeline->CaptureFoliageInputs(frame_dump, fields))
                    scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
            } else if (task_hash == 0x46cc6aaba42f700fULL && mesh_hash != 0) {
                std::array<FrameDump::TaskBufferInput, 17> inputs{};
                size_t input_count{};
                size_t required_count{};
                std::string fields;
                if (!frame_dump.TaskInputsRecorded()) {
                    fields = TaskInputConstants(*maxwell3d, *gpu_memory, buffer_cache, *pipeline);
                    for (const u32 stage : {0U, 2U}) {
                        const auto* info = pipeline->CaptureStageInfo(stage);
                        if (!info) continue;
                        u32 binding_index{};
                        for (const auto& desc : info->storage_buffers_descriptors) {
                            const u32 current_index = binding_index++;
                            ++required_count;
                            if (input_count == inputs.size()) continue;
                            auto& captured = inputs[input_count++];
                            captured.stage = stage;
                            captured.shader_hash = stage == 0 ? task_hash : mesh_hash;
                            captured.pointer_offset = desc.pointer_offset;
                            auto& input = captured.binding;
                            input.binding_index = current_index;
                            input.cbuf_index = desc.cbuf_index;
                            input.cbuf_offset = desc.cbuf_offset;
                            if (desc.count != 1 || desc.is_written) {
                                input.skip_reason = "descriptor is not a scalar read-only buffer";
                                continue;
                            }
                            const auto snapshot = buffer_cache.GetGraphicsStorageBindingSnapshot(stage, current_index);
                            input.buffer = snapshot.buffer ? snapshot.buffer->Handle() : VK_NULL_HANDLE;
                            input.source_capacity = snapshot.buffer ? snapshot.buffer->SizeBytes() : 0;
                            input.offset = snapshot.offset;
                            input.size = snapshot.size;
                            input.gpu_addr = snapshot.gpu_addr;
                            input.segment_count = snapshot.segment_count;
                            input.skip_reason = snapshot.skip_reason;
                            const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
                            if (desc.cbuf_index >= cbufs.size()) continue;
                            const auto& cb = cbufs[desc.cbuf_index];
                            const bool offset_valid = desc.cbuf_offset <= cb.size &&
                                sizeof(GPUVAddr) <= cb.size - desc.cbuf_offset;
                            const bool address_valid = cb.address <=
                                (std::numeric_limits<GPUVAddr>::max)() - desc.cbuf_offset;
                            const GPUVAddr address = address_valid ? cb.address + desc.cbuf_offset : 0;
                            input.pointer_valid = cb.enabled && offset_valid && address_valid &&
                                address <= (std::numeric_limits<GPUVAddr>::max)() - (sizeof(GPUVAddr) - 1) &&
                                gpu_memory->IsWithinGPUAddressRange(address) &&
                                gpu_memory->IsWithinGPUAddressRange(address + sizeof(GPUVAddr) - 1) &&
                                gpu_memory->IsFullyMappedRange(address, sizeof(GPUVAddr));
                            if (input.pointer_valid) {
                                gpu_memory->ReadBlockUnsafe(address, &input.pointer, sizeof(input.pointer));
                                if (desc.pointer_offset != ~0U) {
                                    const bool table_offset_valid = input.pointer != 0 && input.pointer <=
                                        (std::numeric_limits<GPUVAddr>::max)() - desc.pointer_offset;
                                    const GPUVAddr table_address = table_offset_valid
                                        ? input.pointer + desc.pointer_offset : 0;
                                    input.pointer_valid = table_offset_valid &&
                                        table_address <= (std::numeric_limits<GPUVAddr>::max)() - (sizeof(GPUVAddr) - 1) &&
                                        gpu_memory->IsWithinGPUAddressRange(table_address) &&
                                        gpu_memory->IsWithinGPUAddressRange(table_address + sizeof(GPUVAddr) - 1) &&
                                        gpu_memory->IsFullyMappedRange(table_address, sizeof(GPUVAddr));
                                    if (input.pointer_valid)
                                        gpu_memory->ReadBlockUnsafe(table_address, &input.pointer, sizeof(input.pointer));
                                }
                            }
                        }
                    }
                    fields += fmt::format(R"(,"required_storage_descriptors":{},"collected_storage_descriptors":{})",
                                          required_count, input_count);
                }
                if (frame_dump.NoteTaskInputs(task_hash, mesh_hash, fields,
                                               std::span{inputs.data(), input_count})) {
                    scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
                }
            }
        }
        if (frame_dump.CapturingMetadata() &&
            std::ranges::find(pipeline->ShaderHashes(), 0x3ddfb1d7a854c849ULL) !=
                pipeline->ShaderHashes().end()) {
            // Used fog PS/fullscreen VS words only; CPU snapshot, no GPU download.
            const auto constants = [&](u32 stage, u32 bank, std::span<const u32> offsets) {
                const auto& cb = maxwell3d->state.shader_stages[stage].const_buffers[bank];
                std::string words;
                for (const u32 offset : offsets) {
                    u32 word{};
                    const bool valid = cb.enabled && offset <= cb.size && sizeof(word) <= cb.size - offset &&
                        cb.address <= std::numeric_limits<GPUVAddr>::max() - offset - sizeof(word) &&
                        gpu_memory->IsFullyMappedRange(cb.address + offset, sizeof(word));
                    if (valid) gpu_memory->ReadBlockUnsafe(cb.address + offset, &word, sizeof(word));
                    const auto cpu_address = valid ? gpu_memory->GpuToCpuAddress(cb.address + offset)
                                                   : std::nullopt;
                    const bool gpu_modified = cpu_address &&
                        buffer_cache.IsRegionGpuModified(*cpu_address, sizeof(word));
                    words += fmt::format(R"({}{{"offset":{},"valid":{},"gpu_modified":{},"word":{}}})",
                                         words.empty() ? "" : ",", offset, valid, gpu_modified, word);
                }
                return fmt::format(R"({{"stage":{},"bank":{},"address":"{:#x}","enabled":{},"size":{},"words":[{}]}})",
                                   stage, bank, cb.address, cb.enabled, cb.size, words);
            };
            constexpr std::array c3{0x0U, 0x4U, 0x8U, 0x10U, 0x14U, 0x18U, 0x20U, 0x24U, 0x28U, 0xc0U, 0xc4U, 0xc8U, 0x110U, 0x114U, 0x128U, 0x12cU, 0x9d0U, 0x9e8U, 0x9ecU, 0xa04U, 0xbd0U, 0xbd4U, 0xbf0U};
            constexpr std::array c4{0x22cU, 0x5e0U, 0x5e4U, 0x5e8U, 0x5f0U, 0x5f4U, 0x5f8U, 0x600U, 0x604U, 0x608U, 0x640U, 0x644U, 0x648U, 0x64cU, 0x650U, 0x654U, 0x658U, 0x660U, 0x664U, 0x668U, 0x66cU, 0x670U, 0x678U, 0x67cU, 0x680U, 0x684U, 0x688U, 0x68cU};
            std::string fog_constants = constants(4, 3, c3) + "," + constants(4, 4, c4);
            if (std::ranges::find(pipeline->ShaderHashes(), 0x621c4755238870a8ULL) !=
                pipeline->ShaderHashes().end()) {
                constexpr std::array vs_c3{0x120U, 0x124U};
                fog_constants += "," + constants(0, 3, vs_c3);
            }
            frame_dump.NoteDrawMetadata(fmt::format(
                R"("memory_source":"guest_cpu_snapshot","ps_hash":"3ddfb1d7a854c849","fog_constants":[{}])",
                fog_constants));
        }

        if (frame_dump.CapturingMetadata() &&
            std::ranges::find(pipeline->ShaderHashes(), 0x91f539a6c54c97dbULL) !=
                pipeline->ShaderHashes().end()) {
            const auto words_json = [](const auto& words) {
                std::string text;
                for (const u32 word : words)
                    text += fmt::format("{}{}", text.empty() ? "" : ",", word);
                return text;
            };
            const auto read_words = [&](GPUVAddr address, auto& words) {
                const size_t bytes = sizeof(words);
                if (address > std::numeric_limits<GPUVAddr>::max() - bytes ||
                    !gpu_memory->IsFullyMappedRange(address, bytes)) return false;
                gpu_memory->ReadBlockUnsafe(address, words.data(), bytes);
                return true;
            };
            std::array<GPUVAddr, 6> addresses{};
            std::string slots;
            for (size_t slot = 0; slot < addresses.size(); ++slot) {
                addresses[slot] = (u64{regs.pipelines[slot].reserved[0] & 0xff} << 32) |
                                  regs.pipelines[slot].reserved[1];
                slots += fmt::format(
                    R"({}{{"slot":{},"enabled":{},"address":"{:#x}"}})",
                    slots.empty() ? "" : ",", slot, regs.IsShaderConfigEnabled(slot), addresses[slot]);
            }
            std::string headers;
            for (const u32 slot : {1U, 3U, 4U}) {
                std::array<u32, 20> words{};
                const bool valid = regs.IsShaderConfigEnabled(slot) &&
                                   read_words(addresses[slot], words);
                headers += fmt::format(
                    R"({}{{"slot":{},"valid":{},"words":[{}]}})",
                    headers.empty() ? "" : ",", slot, valid, words_json(words));
            }
            const auto constants_json = [&](u32 bank, u32 offset, auto& words) {
                const auto& cb = maxwell3d->state.shader_stages[4].const_buffers[bank];
                const size_t bytes = sizeof(words);
                const bool valid = cb.enabled && offset <= cb.size && bytes <= cb.size - offset &&
                    cb.address <= std::numeric_limits<GPUVAddr>::max() - offset &&
                    read_words(cb.address + offset, words);
                return fmt::format(
                    R"({{"bank":{},"offset":{},"enabled":{},"size":{},"valid":{},"words":[{}]}})",
                    bank, offset, cb.enabled, cb.size, valid, words_json(words));
            };
            std::array<u32, 1> cube_lod{};
            std::array<u32, 11> cube_matrix{};
            std::array<u32, 4> projection{};
            const std::string constants = constants_json(6, 0x610, cube_lod) + "," +
                constants_json(3, 0xb0, cube_matrix) + "," +
                constants_json(3, 0x250, projection);
            frame_dump.NoteDrawMetadata(fmt::format(
                R"("memory_source":"guest_cpu_snapshot","ps_hash":"91f539a6c54c97db","program_slots":[{}],"mesh_enabled":{},"mesh_control":{},"mesh_registers":[{},{},{}],"program_headers":[{}],"constants":[{}])",
                slots, regs.IsMeshEnabled(), regs.reg_array[Maxwell::MeshControl],
                regs.reg_array[Maxwell::MeshShaderA], regs.reg_array[Maxwell::MeshShaderB],
                regs.reg_array[Maxwell::MeshInitShader], headers, constants));
        }


        if (frame_dump.CapturingMetadata()) {
            const auto hashes = pipeline->ShaderHashes();
            const auto compose = std::ranges::find_if(hashes, [](u64 hash) {
                return hash == 0xbded5ce917183214ULL || hash == 0x6cb06edf14be1d79ULL;
            });
            if (compose != hashes.end()) {
                std::string constants;
                for (const u32 bank : {3U, 4U}) {
                    const auto& cb = maxwell3d->state.shader_stages[4].const_buffers[bank];
                    std::array<u32, 15> words{};
                    const bool valid = cb.enabled && sizeof(words) <= cb.size &&
                        cb.address <= std::numeric_limits<GPUVAddr>::max() - sizeof(words) &&
                        gpu_memory->IsFullyMappedRange(cb.address, sizeof(words));
                    if (valid) gpu_memory->ReadBlockUnsafe(cb.address, words.data(), sizeof(words));
                    std::string values;
                    for (const u32 word : words)
                        values += fmt::format("{}{}", values.empty() ? "" : ",", word);
                    const auto device_addr = valid ? gpu_memory->GpuToCpuAddress(cb.address)
                                                   : std::nullopt;
                    const bool gpu_modified = device_addr &&
                        buffer_cache.IsRegionGpuModified(*device_addr, sizeof(words));
                    constants += fmt::format(
                        R"({}{{"bank":{},"address":"{:#x}","offset":0,"enabled":{},"size":{},"valid":{},"device_addr":"{:#x}","gpu_modified":{},"ubo_alignment":{},"words":[{}]}})",
                        constants.empty() ? "" : ",", bank, cb.address, cb.enabled, cb.size,
                        valid, device_addr.value_or(0), gpu_modified,
                        device.GetUniformBufferAlignment(), values);
                }
                frame_dump.NoteDrawMetadata(fmt::format(
                    R"("memory_source":"guest_cpu_snapshot","ps_hash":"{:016x}","compose_constants":[{}])",
                    *compose, constants));
            }
        }


        if (frame_dump.CapturingMetadata() &&
            std::ranges::find(pipeline->ShaderHashes(), 0x96dd01c5b0361fcbULL) !=
                pipeline->ShaderHashes().end()) {
            const auto constants_json = [&](u32 stage, u32 offset, auto& words) {
                const auto& cb = maxwell3d->state.shader_stages[stage].const_buffers[3];
                const bool valid = cb.enabled && offset <= cb.size &&
                    sizeof(words) <= cb.size - offset &&
                    cb.address <= std::numeric_limits<GPUVAddr>::max() - offset - sizeof(words) &&
                    gpu_memory->IsFullyMappedRange(cb.address + offset, sizeof(words));
                if (valid) gpu_memory->ReadBlockUnsafe(cb.address + offset, words.data(), sizeof(words));
                std::string values;
                for (const u32 word : words)
                    values += fmt::format("{}{}", values.empty() ? "" : ",", word);
                return fmt::format(
                    R"({{"stage":{},"bank":3,"address":"{:#x}","offset":{},"enabled":{},"size":{},"valid":{},"words":[{}]}})",
                    stage, cb.address, offset, cb.enabled, cb.size, valid, values);
            };
            std::array<u32, 3> weights{};
            std::array<u32, 1> exponent{};
            std::array<u32, 4> blend{};
            static_assert(sizeof(blend) == sizeof(regs.blend_color));
            std::memcpy(blend.data(), &regs.blend_color, sizeof(blend));
            frame_dump.NoteDrawMetadata(fmt::format(
                R"("memory_source":"guest_cpu_snapshot","ps_hash":"96dd01c5b0361fcb","bloom_weights":[{},{}],"blend_constant_words":[{},{},{},{}])",
                constants_json(4, 0x60, weights), constants_json(0, 0x80, exponent),
                blend[0], blend[1], blend[2], blend[3]));
        }

        if (frame_dump.CapturingMetadata() && maxwell3d->GetNativeMacroEngine().native_turing) {
            // Private CPU-only diagnostic. These are ABI candidates, not a claim every pair is an SSBO.
            std::string storage_candidates;
            for (u32 stage = 0; stage < 5; ++stage) {
                if (!regs.IsShaderConfigEnabled(stage + 1)) continue;
                const auto& cb = maxwell3d->state.shader_stages[stage].const_buffers[0];
                if (!cb.enabled) continue;
                for (u32 offset = 0x500; offset < 0x800; offset += 16) {
                    std::array<u64, 2> raw{};
                    const GPUVAddr address = cb.address + offset;
                    if (offset > cb.size || sizeof(raw) > cb.size - offset || address < cb.address ||
                        address > std::numeric_limits<GPUVAddr>::max() - sizeof(raw) ||
                        !gpu_memory->IsFullyMappedRange(address, sizeof(raw))) continue;
                    gpu_memory->ReadBlockUnsafe(address, raw.data(), sizeof(raw));
                    const GPUVAddr pointer = raw[0];
                    if (!pointer || !gpu_memory->IsWithinGPUAddressRange(pointer) ||
                        !gpu_memory->IsFullyMappedRange(pointer, 1)) continue;
                    const u64 layout = gpu_memory->GetMemoryLayoutSize(pointer);
                    const u64 candidate_size = std::min<u64>(layout, 8ULL * 1024 * 1024);
                    if (!candidate_size || pointer > std::numeric_limits<GPUVAddr>::max() - candidate_size ||
                        !gpu_memory->IsWithinGPUAddressRange(pointer + candidate_size - 1)) continue;
                    const u64 continuous = gpu_memory->MaxContinuousRange(pointer, candidate_size);
                    const auto ranges = gpu_memory->GetSubmappedRange(pointer, candidate_size);
                    std::string segments;
                    GPUVAddr expected = pointer;
                    bool fully_mapped = true;
                    u64 mapped_bytes{};
                    for (size_t index = 0; index < ranges.size(); ++index) {
                        const auto& [begin, size] = ranges[index];
                        fully_mapped &= begin == expected;
                        expected = begin + size;
                        mapped_bytes += size;
                        if (index >= 16) continue;
                        const auto device_address = gpu_memory->GpuToCpuAddress(begin);
                        segments += fmt::format(
                            R"({}{{"gpu_addr":"{:#x}","device_addr":"{:#x}","bytes":{}}})",
                            segments.empty() ? "" : ",", begin, device_address.value_or(0), size);
                    }
                    fully_mapped &= expected == pointer + candidate_size;
                    const u32 legacy_layout32 = static_cast<u32>(layout);
                    const u32 next_low = static_cast<u32>(raw[1]);
                    const bool legacy_size_word = (raw[1] >> 32) == 0 && next_low && next_low <= legacy_layout32;
                    const u32 legacy_size = legacy_size_word ? next_low :
                        std::min<u32>(legacy_layout32, 8U * 1024 * 1024);
                    const u64 alignment = device.GetStorageBufferAlignment();
                    const GPUVAddr aligned_pointer = Common::AlignDown(pointer, alignment);
                    const auto device_address = gpu_memory->GpuToCpuAddress(pointer);
                    const auto aligned_device = gpu_memory->GpuToCpuAddress(aligned_pointer);
                    const auto readonly_bound = [&](u64 size) -> u64 {
                        if (!size || !device_address || !aligned_device ||
                            *device_address > std::numeric_limits<u64>::max() - size) return 0;
                        return Common::AlignUp(*device_address + size, Core::DEVICE_PAGESIZE) - *aligned_device;
                    };
                    const auto descriptor_device = gpu_memory->GpuToCpuAddress(address);
                    const bool descriptor_gpu_modified = descriptor_device &&
                        buffer_cache.IsRegionGpuModified(*descriptor_device, sizeof(raw));
                    storage_candidates += fmt::format(
                        R"({}{{"stage":{},"cb":0,"offset":{},"descriptor":"{:#x}","descriptor_gpu_modified":{},"pointer":"{:#x}","next_qword":"{:#x}","layout_bytes":{},"candidate_bytes":{},"continuous_bytes":{},"mapped_bytes":{},"fully_mapped":{},"segment_count":{},"segments":[{}],"legacy_size_word":{},"legacy_logical_bytes":{},"legacy_readonly_bound_bytes":{},"native_readonly_bound_bytes":{},"storage_alignment":{}}})",
                        storage_candidates.empty() ? "" : ",", stage, offset, address,
                        descriptor_gpu_modified, pointer, raw[1], layout, candidate_size,
                        continuous, mapped_bytes, fully_mapped, ranges.size(), segments,
                        legacy_size_word, legacy_size, readonly_bound(legacy_size),
                        readonly_bound(continuous), alignment);
                }
            }
            frame_dump.NoteDrawMetadata(fmt::format(
                R"("memory_source":"guest_cpu_snapshot","native_storage_candidate_window":"cb0[0x500..0x7ff]","native_storage_candidates":[{}])",
                storage_candidates));
        }

        if (frame_dump.CapturingSampled()) {
            boost::container::small_vector<FrameDump::Sampled, 16> sampled;
            for (const auto& texture : pipeline->SampledImages()) {
                auto& input = sampled.emplace_back(FrameDump::Sampled{
                    .stage = texture.stage, .tic = texture.tic,
                    .resolved_tic = Tegra::Texture::TextureHeaderIndex(
                        texture.tic, regs.tex_header.limit, gpu_memory->UsesNativeTiling()),
                    .tsc = texture.tsc,
                    .view = &texture_cache.GetImageView(texture.id),
                });
                const auto read_descriptor = [&](GPUVAddr base, u32 limit, u32 index,
                                                 std::array<u32, 8>& words) {
                    const u64 offset = u64{index} * sizeof(words);
                    if (index > limit || offset > std::numeric_limits<GPUVAddr>::max() - sizeof(words) ||
                        base > std::numeric_limits<GPUVAddr>::max() - offset - sizeof(words))
                        return false;
                    const GPUVAddr address = base + offset;
                    if (!gpu_memory->IsFullyMappedRange(address, sizeof(words))) return false;
                    gpu_memory->ReadBlockUnsafe(address, words.data(), sizeof(words));
                    return true;
                };
                input.tic_valid = read_descriptor(regs.tex_header.Address(), regs.tex_header.limit,
                                                 input.resolved_tic, input.tic_words);
                input.tsc_valid = read_descriptor(regs.tex_sampler.Address(), regs.tex_sampler.limit,
                                                 texture.tsc, input.tsc_words);
            }
            if (frame_dump.NoteSampled(sampled)) {
                scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
            }
        }
    }
    query_cache.NotifySegment(true);
    HandleTransformFeedback();
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    {
        Profile::Scope scope{Profile::Zone::DrawRecord};
        draw_func();
    }
    submitted = true;
    static u32 native_capture_count{};
    if (maxwell3d->GetNativeMacroEngine().native_turing &&
        native_capture_count < 16 && [] {
            static const bool capture = std::getenv("EDEN_NATIVE_DRAW_CAPTURE") != nullptr;
            return capture;
        }()) {
        const u32 capture = native_capture_count++;
        const auto prefix = fmt::format("native-draw-{}", capture);
        const auto dump = [&](const std::string& name, GPUVAddr address, size_t size) {
            if (!size || !gpu_memory->IsFullyMappedRange(address, size)) return;
            std::vector<u8> bytes(size);
            gpu_memory->ReadBlockUnsafe(address, bytes.data(), bytes.size());
            std::ofstream file(prefix + name + ".bin", std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            LOG_INFO(HW_GPU, "Native draw capture {}{}: address={:#x}, bytes={}, nonzero={}",
                     capture, name, address, size,
                     std::count_if(bytes.begin(), bytes.end(), [](u8 byte) { return byte != 0; }));
        };
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const size_t size = static_cast<size_t>(rt.array_pitch) * 4;
        if (size && size <= 16 * 1024 * 1024) {
            if (const auto address = gpu_memory->GpuToCpuAddress(rt.Address(), size)) {
                LOG_INFO(HW_GPU, "Native draw {} render target physical address={:#x}", capture, *address);
                texture_cache.DownloadMemory(*address, size);
                dump("-rt0", rt.Address(), size);
            }
        }
        std::ofstream registers(prefix + "-registers.bin", std::ios::binary);
        registers.write(reinterpret_cast<const char*>(&regs), sizeof(regs));
        for (size_t stream = 0; stream < 4; ++stream) {
            if (regs.vertex_streams[stream].IsEnabled())
                dump(fmt::format("-vertex-{}", stream), regs.vertex_streams[stream].Address(), 128);
        }
        for (size_t stage : {0U, 4U}) {
            for (size_t bank : {2U, 3U}) {
                const auto& cb = maxwell3d->state.shader_stages[stage].const_buffers[bank];
                if (cb.enabled) dump(fmt::format("-cbuf-{}-{}", stage, bank), cb.address, std::min<size_t>(cb.size, 128));
            }
        }
    }
}


namespace {
// EDEN_NX2_WINDOW_TRACE: describe clears and draws into screen-sized colour targets (debug aid).
void TraceWindowTarget(const Tegra::Engines::Maxwell3D& maxwell3d, Tegra::MemoryManager& memory,
                       const char* kind, u32 count) {
    static const bool trace = std::getenv("EDEN_NX2_WINDOW_TRACE") != nullptr;
    static u32 traced{};
    const auto& regs = maxwell3d.regs;
    static const u64 skip = [] {
        const char* value = std::getenv("EDEN_NX2_WINDOW_TRACE_SKIP");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ULL;
    }();
    static u64 seen_events{};
    static const u64 only_ps = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_PS");
        return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    const u64 pixel_shader =
        (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
    if (!trace || regs.rt[0].width != 3840 || traced >= 400 || seen_events++ < skip ||
        (only_ps != 0 && pixel_shader != only_ps)) {
        return;
    }
    ++traced;
    const auto& vp = regs.viewport_transform[0];
    const auto& sc = regs.scissor_test[0];
    LOG_INFO(HW_GPU,
             "nx2 window {} rt0={:#x} count={} clear={:#x} color=({},{},{},{}) mask_common={} "
             "mask0={:#x} rt_count={} vp scale=({},{}) translate=({},{}) scissor en={} x={}..{} "
             "y={}..{} blend0={} vs={:#x} ps={:#x} rasterize={}",
             kind, regs.rt[0].Address(), count, regs.clear_surface.raw, regs.clear_color[0],
             regs.clear_color[1], regs.clear_color[2], regs.clear_color[3], regs.color_mask_common,
             regs.color_mask[0].raw, regs.rt_control.count.Value(), vp.scale_x, vp.scale_y,
             vp.translate_x, vp.translate_y, sc.enable, sc.min_x.Value(), sc.max_x.Value(),
             sc.min_y.Value(), sc.max_y.Value(), regs.blend.enable[0],
             (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1],
             (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1],
             regs.rasterize_enable);
    if (kind[0] != 'd') {
        return;
    }
    // Pixel-stage constant buffer 3: colour constants of the loading-screen passes.
    if (const auto& cb3 = maxwell3d.state.shader_stages[4].const_buffers[3]; cb3.enabled) {
        std::array<f32, 8> values{};
        memory.ReadBlockUnsafe(cb3.address, values.data(), sizeof(values));
        LOG_INFO(HW_GPU, "nx2 window ps c[3]@{:#x} size={}: {}", cb3.address, cb3.size,
                 fmt::join(values, " "));
    }
    static u32 texture_traces{};
    if (texture_traces++ > 200) {
        return;
    }
    // Texture used by the pixel shader's first bound TEX (handle in c[2][0x20]) and its TIC.
    const auto& cb = maxwell3d.state.shader_stages[4].const_buffers[2];
    const u32 handle = cb.enabled ? memory.Read<u32>(cb.address + 0x20) : 0;
    const u32 tic_index = handle & 0xfffff;
    const GPUVAddr tic_address = regs.tex_header.Address() + u64{tic_index} * 32;
    std::array<u32, 8> tic{};
    memory.ReadBlockUnsafe(tic_address, tic.data(), sizeof(tic));
    LOG_INFO(HW_GPU,
             "nx2 window texture: cb2={:#x} en={} handle={:#x} tic pool={:#x} limit={} "
             "sampler pool={:#x} tic[{}]={:08x}",
             cb.address, cb.enabled, handle, regs.tex_header.Address(), regs.tex_header.limit,
             regs.tex_sampler.Address(), tic_index, fmt::join(tic, " "));
    // First bytes of the texture's memory (block-linear, so just a sample).
    const GPUVAddr texture = (GPUVAddr{tic[2] & 0xffff} << 32) | tic[1];
    std::array<u32, 16> texels{};
    memory.ReadBlockUnsafe(texture, texels.data(), sizeof(texels));
    LOG_INFO(HW_GPU, "nx2 window texture {:#x} {}x{}: {:08x}", texture, (tic[4] & 0xffff) + 1,
             (tic[5] & 0xffff) + 1, fmt::join(texels, " "));
    // EDEN_NX2_TEXTURE_DUMP=<file>: save the first 4 MiB of the last traced texture's memory.
    if (const char* path = std::getenv("EDEN_NX2_TEXTURE_DUMP"); path != nullptr) {
        std::vector<u8> bytes(2048u * 10240u);
        memory.ReadBlockUnsafe(texture, bytes.data(), bytes.size());
        if (std::FILE* file = std::fopen(path, "wb")) {
            std::fwrite(bytes.data(), 1, bytes.size(), file);
            std::fclose(file);
        }
    }
}
} // Anonymous namespace

void RasterizerVulkan::Draw(bool is_indexed, u32 instance_count) {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Draw};
    {
        // A draw's first and count can come from memory a skipped (rejected) compute program
        // should have written. Past 16M vertices they are garbage: building the quad index
        // buffer for one threw out of the GPU thread and ended the process (Hyrule Warriors:
        // Age of Imprisonment, entering a new area), and indexed ones read far out of bounds.
        const auto& state = maxwell3d->draw_manager.draw_state;
        const u64 first = is_indexed ? state.index_buffer.first : state.vertex_buffer.first;
        const u64 count = is_indexed ? state.index_buffer.count : state.vertex_buffer.count;
        constexpr u64 MaxVertices = u64{1} << 24;
        if (first + count > MaxVertices) {
            Tegra::DmaPusher::CaptureCurrentPushbufferFault("invalid-draw-range");
            if (frame_dump.CapturingDrawSubmissions()) {
                frame_dump.NoteDrawSubmission(
                    fmt::format(R"("status":"skipped","command":"direct","reason":"invalid_draw_range","is_indexed":{},"first":{},"count":{},)",
                                is_indexed, first, count) + DrawSubmissionGuestState(maxwell3d->regs, maxwell3d->draw_manager.draw_state,
                                             maxwell3d->ShouldExecute()), false);
            }
            static std::atomic<u32> skipped{};
            if (skipped.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(HW_GPU, "Draw skipped: {} {} from {} (vertex/index count past {})",
                            count, is_indexed ? "indices" : "vertices", first, MaxVertices);
            }
            return;
        }
    }
    TraceWindowTarget(*maxwell3d, *gpu_memory, is_indexed ? "draw-indexed" : "draw",
                      is_indexed ? maxwell3d->draw_manager.draw_state.index_buffer.count
                                 : maxwell3d->draw_manager.draw_state.vertex_buffer.count);
    // EDEN_NX2_CB24_TRACE: for draws whose vertex shader reads bank 24, show the inline
    // constant buffer upload target (CB_ADDRESS) and its first words.
    if (static const bool trace = std::getenv("EDEN_NX2_CB24_TRACE") != nullptr; trace) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 logged{};
        if ((vs == 0x11471c9a00 || vs == 0x11471ca600 || vs == 0x11471cd300) && logged++ < 3) {
            std::array<u32, 24> words{};
            gpu_memory->ReadBlockUnsafe(regs.const_buffer.Address(), words.data(), sizeof(words));
            LOG_INFO(HW_GPU, "bank24 draw vs={:#x}: CB_ADDRESS {:#x} size {:#x} offset {:#x}: {:08x}", vs,
                     regs.const_buffer.Address(), regs.const_buffer.size, regs.const_buffer.offset,
                     fmt::join(words, " "));
            // Search every bound vertex-stage buffer for render-target-size values.
            const auto& stage = maxwell3d->state.shader_stages[0];
            for (u32 slot = 0; slot < Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers; ++slot) {
                const auto& buffer = stage.const_buffers[slot];
                if (!buffer.enabled || buffer.size == 0) continue;
                std::vector<u32> data(std::min<u32>(buffer.size, 0x10000) / 4);
                gpu_memory->ReadBlockUnsafe(buffer.address, data.data(), data.size() * 4);
                for (size_t i = 0; i < data.size(); ++i) {
                    const u32 w = data[i];
                    if (w == 0x45200000 || w == 0x44b40000 || w == 0x39cccccd || w == 0x3a360b61 ||
                        w == 0x44a00000 || w == 0x44340000 || w == 2559 || w == 1439) {
                        LOG_INFO(HW_GPU, "  slot {} +{:#x} = {:#x}", slot, i * 4, w);
                    }
                }
            }
        }
    }
    // EDEN_NX2_TRACE_VS=<address>: describe the next few draws using that vertex shader like the
    // after-upscale trace does (render target, blend, fragment textures).
    static const u64 trace_vs = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_VS");
        return value ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    if (trace_vs != 0) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 traced{};
        if (vs == trace_vs && traced++ % 200 == 0 && traced < 1000) {
            Tegra::Engines::g_nx2_trace_draws_after_upscale = 1;
        }
    }
    if (Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        --Tegra::Engines::g_nx2_trace_draws_after_upscale;
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        LOG_INFO(HW_GPU, "after upscale: draw rt0 {:#x} {}x{} fmt {:#x} block {}/{}/{} pitch_linear {} depth {} array_pitch {:#x} ctrl {:#x} vs {:#x} ps {:#x}",
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.rt[0].tile_mode.block_width.Value(),
                 regs.rt[0].tile_mode.block_height.Value(), regs.rt[0].tile_mode.block_depth.Value(),
                 regs.rt[0].tile_mode.is_pitch_linear.Value(), regs.rt[0].depth.Value(),
                 regs.rt[0].array_pitch, regs.rt_control.count.Value(), vs, ps);
        // Fragment-stage texture handles (bound buffer 2) and the images they name.
        const auto& handles = maxwell3d->state.shader_stages[4].const_buffers[2];
        if (handles.enabled) {
            for (u32 i = 0; i < 24; ++i) {
                const u32 handle = gpu_memory->Read<u32>(handles.address + i * 4);
                if (handle == 0 || handle == 0xffffffff) continue;
                Tegra::Texture::TICEntry tic{};
                gpu_memory->ReadBlockUnsafe(regs.tex_header.Address() + u64{handle & 0xfffff} * sizeof(tic),
                                            &tic, sizeof(tic));
                LOG_INFO(HW_GPU, "  fs handle[{}] = {:#x}: {:#x} {}x{} format {:#x}", i, handle,
                         tic.Address(), tic.Width(), tic.Height(), static_cast<u32>(tic.format.Value()));
            }
        }
        const auto& bt = regs.blend_per_target[0];
        LOG_INFO(HW_GPU, "  blend enable {} per-target {} global op {} src {:#x} dst {:#x} | rt0 op {} src {:#x} dst {:#x} alpha {:#x}/{:#x} mask {:#x}",
                 regs.blend.enable[0], regs.blend_per_target_enabled,
                 static_cast<u32>(regs.blend.color_op), static_cast<u32>(regs.blend.color_source),
                 static_cast<u32>(regs.blend.color_dest), static_cast<u32>(bt.color_op),
                 static_cast<u32>(bt.color_source), static_cast<u32>(bt.color_dest),
                 static_cast<u32>(bt.alpha_source), static_cast<u32>(bt.alpha_dest),
                 regs.color_mask[0].raw);
    }
    // EDEN_NX2_SKIP_VS=<address>[,<address>...]: drop draws using those vertex shaders (debug aid).
    static const std::vector<u64> skip_list = [] {
        std::vector<u64> list;
        if (const char* value = std::getenv("EDEN_NX2_SKIP_VS")) {
            for (char* end = nullptr; *value; value = *end ? end + 1 : end) {
                list.push_back(std::strtoull(value, &end, 16));
                if (end == value) break;
            }
        }
        return list;
    }();
    if (!skip_list.empty()) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        if (std::find(skip_list.begin(), skip_list.end(), vs) != skip_list.end()) return;
    }
    // EDEN_NX2_RT_TRACE: log the first draw to every distinct colour target (debug aid).
    if (static const bool trace = std::getenv("EDEN_NX2_RT_TRACE") != nullptr; trace) {
        static std::set<std::tuple<GPUVAddr, u32, u32, u64, u64>> seen;
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        const auto key = std::make_tuple(rt.Address(), rt.width, static_cast<u32>(rt.format), vs, ps);
        if (rt.width >= 640 && seen.size() < 600 && seen.insert(key).second) {
            LOG_INFO(HW_GPU, "nx2 draw target {:#x} {}x{} format={:#x} vs={:#x} ps={:#x} blend={} (pairs so far {})",
                     rt.Address(), rt.width, rt.height, static_cast<u32>(rt.format), vs, ps,
                     regs.blend.enable[0], seen.size());
        }
    }
    PrepareDraw(is_indexed, [this, is_indexed, instance_count] {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        const u32 num_instances{instance_count};
        const DrawParams draw_params{MakeDrawParams(draw_state, num_instances, is_indexed)};
        VideoCommon::CommandTrace::Get().Mark(mesh_draw ? "draw_mesh.record" : "draw.record",
            scheduler.CurrentTick(), 0, draw_params.num_vertices, draw_params.num_instances,
            draw_params.first_index);
        const auto note_recorded = [&] {
            if (!frame_dump.CapturingDrawSubmissions()) return;
            const char* command = mesh_draw ? "vkCmdDrawMeshTasksEXT" :
                                  draw_params.is_indexed ? "vkCmdDrawIndexed" : "vkCmdDraw";
            frame_dump.NoteDrawSubmission(
                fmt::format(R"("status":"recorded","command":"{}","argument_source":"cpu_draw_params","mesh_pipeline":{},"is_indexed":{},"count":{},"instances":{},"first_index":{},"base_vertex_bits":{},"vertex_offset_i32":{},"base_instance":{},"mesh_groups":[{},{},{}],)",
                            command, mesh_draw, draw_params.is_indexed, draw_params.num_vertices,
                            draw_params.num_instances, draw_params.first_index, draw_params.base_vertex,
                            static_cast<s32>(draw_params.base_vertex), draw_params.base_instance,
                            mesh_draw ? draw_params.num_vertices : 0U, mesh_draw ? 1U : 0U,
                            mesh_draw ? 1U : 0U) + DrawSubmissionGuestState(maxwell3d->regs, maxwell3d->draw_manager.draw_state,
                                             maxwell3d->ShouldExecute()), true);
        };

        if (mesh_draw) {
            scheduler.Record([groups = draw_params.num_vertices](vk::CommandBuffer cmdbuf) {
                cmdbuf.DrawMeshTasksEXT(groups, 1, 1);
            });
            note_recorded();
            return;
        }
        scheduler.Record([draw_params](vk::CommandBuffer cmdbuf) {
            if (draw_params.is_indexed) {
                cmdbuf.DrawIndexed(draw_params.num_vertices, draw_params.num_instances,
                                   draw_params.first_index, draw_params.base_vertex,
                                   draw_params.base_instance);
            } else {
                cmdbuf.Draw(draw_params.num_vertices, draw_params.num_instances,
                            draw_params.base_vertex, draw_params.base_instance);
            }
        });

        note_recorded();

        // Log draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string params = is_indexed ?
                fmt::format("vertices={}, instances={}, firstIndex={}, baseVertex={}, baseInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.first_index, draw_params.base_vertex, draw_params.base_instance) :
                fmt::format("vertices={}, instances={}, firstVertex={}, firstInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.base_vertex, draw_params.base_instance);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                is_indexed ? "vkCmdDrawIndexed" : "vkCmdDraw", params, VK_SUCCESS);
        }
    });
}

void RasterizerVulkan::DrawIndirect() {
    const VideoCommon::CommandTraceScope trace{"draw_indirect", 0, scheduler.CurrentTick()};
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Draw};
    const auto& params = maxwell3d->draw_manager.indirect_state;
    buffer_cache.SetDrawIndirect(&params);
    PrepareDraw(params.is_indexed, [this, &params] {
        const auto note_indirect = [&](const char* command, bool recorded, u64 argument_offset = 0,
                                       u64 count_offset = 0) {
            if (!frame_dump.CapturingDrawSubmissions()) return;
            frame_dump.NoteDrawSubmission(
                fmt::format(R"("status":"{}","command":"{}","argument_source":"gpu_words_not_captured","mesh_pipeline":{},"is_indexed":{},"is_byte_count":{},"include_count":{},"indirect_address":"{:#x}","count_address":"{:#x}","buffer_bytes":{},"max_draw_count":{},"stride":{},"host_buffer_offsets_known":{},"argument_vk_offset":{},"count_vk_offset":{},)",
                            recorded ? "recorded" : "skipped", command, mesh_draw, params.is_indexed,
                            params.is_byte_count, params.include_count, params.indirect_start_address,
                            params.count_start_address, params.buffer_size, params.max_draw_counts,
                            params.stride, recorded, argument_offset, count_offset) +
                    DrawSubmissionGuestState(maxwell3d->regs, maxwell3d->draw_manager.draw_state,
                                             maxwell3d->ShouldExecute()), true);
        };
        if (mesh_draw) {
            // Indirect task/mesh draws would need vkCmdDrawMeshTasksIndirectEXT's layout.
            static bool logged{};
            if (!std::exchange(logged, true)) {
                LOG_WARNING(Render_Vulkan, "Indirect task/mesh draw skipped");
            }
            note_indirect("mesh_indirect_unsupported", false);
            return;
        }
        const auto indirect_buffer = buffer_cache.GetDrawIndirectBuffer();
        const auto& buffer = indirect_buffer.first;
        const auto& offset = indirect_buffer.second;
        if (params.is_byte_count) {
            scheduler.Record([buffer_obj = buffer->Handle(), offset,
                              stride = params.stride](vk::CommandBuffer cmdbuf) {
                cmdbuf.DrawIndirectByteCountEXT(1, 0, buffer_obj, offset, 0,
                                                static_cast<u32>(stride));
            });
            note_indirect("vkCmdDrawIndirectByteCountEXT", true, offset);
            return;
        }
        if (params.include_count) {
            const auto count = buffer_cache.GetDrawIndirectCount();
            const auto& draw_buffer = count.first;
            const auto& offset_base = count.second;
            scheduler.Record([draw_buffer_obj = draw_buffer->Handle(),
                              buffer_obj = buffer->Handle(), offset_base, offset,
                              params](vk::CommandBuffer cmdbuf) {
                if (params.is_indexed) {
                    cmdbuf.DrawIndexedIndirectCount(
                        buffer_obj, offset, draw_buffer_obj, offset_base,
                        static_cast<u32>(params.max_draw_counts), static_cast<u32>(params.stride));
                } else {
                    cmdbuf.DrawIndirectCount(buffer_obj, offset, draw_buffer_obj, offset_base,
                                             static_cast<u32>(params.max_draw_counts),
                                             static_cast<u32>(params.stride));
                }
            });
            note_indirect(params.is_indexed ? "vkCmdDrawIndexedIndirectCount" : "vkCmdDrawIndirectCount",
                          true, offset, offset_base);
            return;
        }
        scheduler.Record([buffer_obj = buffer->Handle(), offset, params](vk::CommandBuffer cmdbuf) {
            if (params.is_indexed) {
                cmdbuf.DrawIndexedIndirect(buffer_obj, offset,
                                           static_cast<u32>(params.max_draw_counts),
                                           static_cast<u32>(params.stride));
            } else {
                cmdbuf.DrawIndirect(buffer_obj, offset, static_cast<u32>(params.max_draw_counts),
                                    static_cast<u32>(params.stride));
            }
        });

        note_indirect(params.is_indexed ? "vkCmdDrawIndexedIndirect" : "vkCmdDrawIndirect", true, offset);

        // Log indirect draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string log_params = fmt::format("drawCount={}, stride={}",
                params.max_draw_counts, params.stride);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                params.is_indexed ? "vkCmdDrawIndexedIndirect" : "vkCmdDrawIndirect",
                log_params, VK_SUCCESS);
        }
    });
    buffer_cache.SetDrawIndirect(nullptr);
}

void RasterizerVulkan::DrawTexture() {
    if (frame_dump.Capturing()) {
        if (frame_dump.BeforeDraw()) {
            scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
        }
        frame_dump.NoteOperation("blit");
    }


    SCOPE_EXIT {
        gpu.TickWork();
    };
    FlushWork();

    std::scoped_lock l{texture_cache.mutex};
    texture_cache.SynchronizeDescriptors(false);
    texture_cache.UpdateRenderTargets(false);

    ApplyComputeForGraphicsBarrier();
    UpdateDynamicStates();

    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    const auto& draw_texture_state = maxwell3d->draw_manager.draw_texture_state;
    const auto& sampler = texture_cache.GetSampler(draw_texture_state.src_sampler, false);
    const auto& texture = texture_cache.GetImageView(draw_texture_state.src_texture);
    const auto* framebuffer = texture_cache.GetFramebuffer();
    if (frame_dump.Capturing()) FrameDumpPass(framebuffer);

    const bool src_rescaling = texture_cache.IsRescaling() && texture.IsRescaled();
    const bool dst_rescaling = texture_cache.IsRescaling() && framebuffer->IsRescaled();

    const auto ScaleSrc = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return src_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    const auto ScaleDst = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return dst_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    Region2D dst_region = {Offset2D{.x = ScaleDst(draw_texture_state.dst_x0),
                                    .y = ScaleDst(draw_texture_state.dst_y0)},
                           Offset2D{.x = ScaleDst(draw_texture_state.dst_x1),
                                    .y = ScaleDst(draw_texture_state.dst_y1)}};
    Region2D src_region = {Offset2D{.x = ScaleSrc(draw_texture_state.src_x0),
                                    .y = ScaleSrc(draw_texture_state.src_y0)},
                           Offset2D{.x = ScaleSrc(draw_texture_state.src_x1),
                                    .y = ScaleSrc(draw_texture_state.src_y1)}};
    Extent3D src_size = {static_cast<u32>(ScaleSrc(texture.size.width)),
                         static_cast<u32>(ScaleSrc(texture.size.height)), texture.size.depth};
    blit_image.BlitColor(framebuffer, texture.RenderTarget(), texture.ImageHandle(),
                         sampler->Handle(), dst_region, src_region, src_size);
}

void RasterizerVulkan::Clear(u32 layer_count) {
    if (frame_dump.Capturing()) {
        if (frame_dump.BeforeDraw()) {
            scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
        }
        frame_dump.NoteOperation("clear");
    }

    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Clear};
    if (Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        const auto& regs = maxwell3d->regs;
        LOG_INFO(HW_GPU, "after upscale: clear rt0 {:#x} {}x{} fmt {:#x} surface {:#x} colour {} {} {} {}",
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.clear_surface.raw, regs.clear_color[0],
                 regs.clear_color[1], regs.clear_color[2], regs.clear_color[3]);
    }
    TraceWindowTarget(*maxwell3d, *gpu_memory, "clear", layer_count);
    FlushWork();
    gpu_memory->FlushCaching();

    auto& regs = maxwell3d->regs;
    const bool use_color = regs.clear_surface.R || regs.clear_surface.G || regs.clear_surface.B ||
                           regs.clear_surface.A;
    const bool use_depth = regs.clear_surface.Z;
    const bool use_stencil = regs.clear_surface.S;
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }

    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(true);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    if (frame_dump.Capturing()) FrameDumpPass(framebuffer);
    const VkExtent2D render_area = framebuffer->RenderArea();

    constexpr bool ENABLE_DEFERRED_CLEAR = true;
    const bool color_full_channels = regs.clear_surface.R && regs.clear_surface.G &&
                                     regs.clear_surface.B && regs.clear_surface.A;
    const bool stencil_partial = use_stencil && framebuffer->HasAspectStencilBit() &&
                                 regs.stencil_front_mask != 0xFF && regs.stencil_front_mask != 0;
    const bool ds_used = use_depth || use_stencil;
    const bool ds_deferrable =
        !ds_used || ((!framebuffer->HasAspectDepthBit() || use_depth) &&
                     (!framebuffer->HasAspectStencilBit() || use_stencil) && !stencil_partial);
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    VkRect2D default_scissor{};
    default_scissor.offset.x = 0;
    default_scissor.offset.y = 0;
    default_scissor.extent.width = (std::numeric_limits<s32>::max)();
    default_scissor.extent.height = (std::numeric_limits<s32>::max)();

    VkClearRect clear_rect{
        .rect = regs.clear_control.use_scissor ? GetScissorState(regs, 0, up_scale, down_shift)
                                               : default_scissor,
        .baseArrayLayer = regs.clear_surface.layer,
        .layerCount = layer_count,
    };
    const auto clamp_rect_to_render_area = [render_area](VkRect2D& rect) -> bool {
        const auto clamp_axis = [](s32& offset, u32& extent, u32 limit) {
            auto clamp_offset = [&offset, limit]() {
                if (limit == 0) {
                    offset = 0;
                    return;
                }
                offset = std::clamp(offset, 0, static_cast<s32>(limit));
            };

            if (extent == 0) {
                clamp_offset();
                return;
            }
            if (offset < 0) {
                const u32 shrink = (std::min)(extent, static_cast<u32>(-offset));
                extent -= shrink;
                offset = 0;
            }
            if (limit == 0) {
                extent = 0;
                offset = 0;
                return;
            }
            if (offset >= s32(limit)) {
                offset = s32(limit);
                extent = 0;
                return;
            }
            const u64 end_coord = u64(offset) + extent;
            if (end_coord > limit) {
                extent = limit - u32(offset);
            }
        };

        clamp_axis(rect.offset.x, rect.extent.width, render_area.width);
        clamp_axis(rect.offset.y, rect.extent.height, render_area.height);
        return rect.extent.width != 0 && rect.extent.height != 0;
    };
    if (!clamp_rect_to_render_area(clear_rect.rect)) {
        return;
    }

    const bool clear_covers_render_area =
        clear_rect.rect.offset.x == 0 && clear_rect.rect.offset.y == 0 &&
        clear_rect.rect.extent.width >= render_area.width &&
        clear_rect.rect.extent.height >= render_area.height;
    const bool can_defer_clear = ENABLE_DEFERRED_CLEAR && (!regs.clear_control.use_scissor || clear_covers_render_area) &&
                                 regs.clear_surface.layer == 0 &&
                                 !scheduler.IsRenderPassActive() &&
                                 (!use_color || color_full_channels) && ds_deferrable;
    if (!can_defer_clear) {
        scheduler.RequestRenderpass(framebuffer);
    }

    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    UpdateViewportsState(regs);

    const u32 color_attachment = regs.clear_surface.RT;
    if (use_color && framebuffer->HasAspectColorBit(color_attachment)) {
        const auto format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[color_attachment].format);
        bool is_integer = IsPixelFormatInteger(format);
        bool is_signed = IsPixelFormatSignedInteger(format);
        size_t int_size = PixelComponentSizeBitsInteger(format);
        VkClearValue clear_value{};
        if (!is_integer) {
            std::memcpy(clear_value.color.float32, regs.clear_color.data(), regs.clear_color.size() * sizeof(f32));
        } else if (!is_signed) {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.uint32[i] = u32(f32(u64(int_size) << 1U) * regs.clear_color[i]);
        } else {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.int32[i] = s32(f32(s64(int_size - 1) << 1) * (regs.clear_color[i] - 0.5f));
        }

        if (color_full_channels) {
            if (can_defer_clear) {
                scheduler.DeferColorClear(framebuffer, color_attachment, clear_value);
            } else {
                scheduler.Record([color_attachment, clear_value, clear_rect](vk::CommandBuffer cmdbuf) {
                    const VkClearAttachment attachment{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .colorAttachment = color_attachment,
                        .clearValue = clear_value,
                    };
                    cmdbuf.ClearAttachments(attachment, clear_rect);
                });
            }
        } else {
            u8 color_mask = u8(regs.clear_surface.R | regs.clear_surface.G << 1 | regs.clear_surface.B << 2 | regs.clear_surface.A << 3);
            Region2D dst_region = {
                Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
                Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                         .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
            blit_image.ClearColor(framebuffer, color_mask, regs.clear_color, dst_region);
        }
    }

    if (!use_depth && !use_stencil) {
        return;
    }
    VkImageAspectFlags aspect_flags = 0;
    if (use_depth && framebuffer->HasAspectDepthBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (use_stencil && framebuffer->HasAspectStencilBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    if (aspect_flags == 0) {
        return;
    }

    if (use_stencil && framebuffer->HasAspectStencilBit() && regs.stencil_front_mask != 0xFF &&
        regs.stencil_front_mask != 0) {
        Region2D dst_region = {
            Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
            Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                     .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
        blit_image.ClearDepthStencil(framebuffer, use_depth, regs.clear_depth,
                                     u8(regs.stencil_front_mask), regs.clear_stencil,
                                     regs.stencil_front_func_mask, dst_region);
    } else if (can_defer_clear) {
        VkClearValue ds_value{};
        ds_value.depthStencil.depth = regs.clear_depth;
        ds_value.depthStencil.stencil = regs.clear_stencil;
        scheduler.DeferDepthStencilClear(framebuffer, ds_value);
    } else {
        scheduler.Record([clear_depth = regs.clear_depth, clear_stencil = regs.clear_stencil,
                          clear_rect, aspect_flags](vk::CommandBuffer cmdbuf) {
            VkClearAttachment attachment;
            attachment.aspectMask = aspect_flags;
            attachment.colorAttachment = 0;
            attachment.clearValue.depthStencil.depth = clear_depth;
            attachment.clearValue.depthStencil.stencil = clear_stencil;
            cmdbuf.ClearAttachments(attachment, clear_rect);
        });
    }
}

// Private, opt-in attribution probe. Each file is bounded and every phase is flushed before
// waiting. Synchronization changes timing; this records the failing bracket, not its cause.
class ComputeSyncTrace {
public:
    static ComputeSyncTrace& Get() {
        static ComputeSyncTrace trace;
        return trace;
    }

    bool Enabled() const noexcept { return file != nullptr; }

    void Begin(u64 program, u64 native_key) noexcept {
        ++serial;
        current_program = program;
        current_native_key = native_key;
    }

    void Record(const char* phase, std::string_view metadata, bool exception = false) noexcept {
        if (!file) return;
        constexpr size_t MAX_METADATA = 128 * 1024;
        constexpr size_t MAX_FILE_BYTES = 8 * 1024 * 1024;
        if (metadata.size() > MAX_METADATA) {
            metadata = R"("metadata_oversized":true)";
        }
        // Includes ample space for the fixed-format prefix/suffix. Rotation keeps two files.
        const size_t record_bound = metadata.size() + 256;
        if (bytes > MAX_FILE_BYTES - record_bound) {
            std::fclose(file);
            slot ^= 1;
            file = std::fopen(paths[slot].c_str(), "wb");
            bytes = 0;
            if (!file) return;
        }
        const int prefix = std::fprintf(file,
            R"({"serial":%llu,"phase":"%s","exception":%s,"dispatch_program":"0x%llx","dispatch_native_key":"%016llx",)",
            static_cast<unsigned long long>(serial), phase, exception ? "true" : "false",
            static_cast<unsigned long long>(current_program),
            static_cast<unsigned long long>(current_native_key));
        const size_t written = std::fwrite(metadata.data(), 1, metadata.size(), file);
        const int suffix = std::fprintf(file, "}\n");
        if (prefix < 0 || written != metadata.size() || suffix < 0 || std::fflush(file) != 0) {
            std::fclose(file);
            file = nullptr;
            return;
        }
        bytes += static_cast<size_t>(prefix) + written + static_cast<size_t>(suffix);
    }

private:
    ComputeSyncTrace() {
        const char* value = std::getenv("EDEN_NX2_COMPUTE_SYNC_TRACE");
        if (!value || !*value || std::strlen(value) > 4096) return;
        paths[0] = std::string{value} + ".0.jsonl";
        paths[1] = std::string{value} + ".1.jsonl";
        file = std::fopen(paths[0].c_str(), "wb");
    }
    ~ComputeSyncTrace() { if (file) std::fclose(file); }
    std::array<std::string, 2> paths;
    std::FILE* file{};
    size_t bytes{};
    u64 serial{};
    u64 current_program{};
    u64 current_native_key{};
    size_t slot{};
};

void RasterizerVulkan::DispatchCompute() {
    auto& sync_trace = ComputeSyncTrace::Get();
    const bool sync_compute = sync_trace.Enabled() && kepler_compute->IsNativeMode();
    const bool live_fog_inputs = frame_dump.CapturingFogInputs() &&
                                 program_id == 0x0400486024DEC000ULL;
    const bool trace_compute = frame_dump.CapturingComputeMetadata() &&
        (!frame_dump.CapturingFogInputs() || live_fog_inputs) && kepler_compute->IsNativeMode();
    const bool capture_stop_armed = FrameDump::ComputeCaptureStopEnabled() && kepler_compute->IsNativeMode();
    u64 native_key{};
    std::string compute_metadata = (trace_compute || sync_compute || capture_stop_armed)
        ? ComputeMetadata(*kepler_compute, *gpu_memory, buffer_cache, &native_key, live_fog_inputs) : "";
    const bool capture_stop = capture_stop_armed && native_key == FrameDump::ComputeCaptureStopNativeKey();
    const bool capture_sky = capture_stop && FrameDump::ComputeCaptureSkyEnabled();
    if (capture_stop_armed && FrameDump::ComputeCaptureSkyEnabled() &&
        !FrameDump::DelayedSkyCaptureEnabled() && native_key == 0x9694a462ae7e186cULL)
        throw std::runtime_error("DIAGNOSTIC: sky target was not captured; native9694 remains blocked");
    FrameDump* input_dump = trace_compute &&
        ((native_key == 0xec68e40eb783ff32ULL && frame_dump.CapturingComputeInputs()) ||
         frame_dump.SelectProducerInputDispatch(native_key))
        ? &frame_dump : nullptr;
    const int exceptions_before = trace_compute ? std::uncaught_exceptions() : 0;
    const char* compute_phase = "pre_drain_begin";
    const int sync_exceptions_before = std::uncaught_exceptions();
    if (sync_compute) {
        sync_trace.Begin(kepler_compute->regs.code_loc.Address(), native_key);
        sync_trace.Record(compute_phase, compute_metadata);
    }
    SCOPE_EXIT {
        if (sync_compute) {
            // Logging only: never wait or throw while unwinding a Vulkan exception.
            sync_trace.Record(compute_phase, compute_metadata,
                              std::uncaught_exceptions() > sync_exceptions_before);
        }
    };
    SCOPE_EXIT {
        if (trace_compute) {
            frame_dump.NoteComputeMetadata(fmt::format(
                R"({},"phase":"{}","exception":{})", compute_metadata, compute_phase,
                std::uncaught_exceptions() > exceptions_before));
        }
    };
    if (frame_dump.Capturing()) {
        frame_dump.EndPass();
        frame_dump.DrainIfFull();
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Compute};
    FlushWork();
    gpu_memory->FlushCaching();
    if (sync_compute) {
        scheduler.Finish();
        compute_phase = "pre_drain_complete";
        sync_trace.Record(compute_phase, compute_metadata);
    }

    if (capture_stop) {
        frame_dump.BeginComputeCaptureStop(native_key);
        input_dump = &frame_dump;
        frame_dump.NoteCaptureStopMetadata(compute_metadata);
    }
    try {
    compute_phase = "pipeline_lookup";
    if (sync_compute) sync_trace.Record(compute_phase, compute_metadata);
    u64 range_native_key{};
    ComputePipeline* const pipeline{pipeline_cache.CurrentComputePipeline(&range_native_key)};
    if (!pipeline) {
        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute pipeline unavailable");
        return;
    }
    if (input_dump && native_key == FrameDump::ProducerNativeKey()) {
        compute_metadata += ComputeProducerConstants(*kepler_compute, *gpu_memory, buffer_cache,
                                                     native_key, pipeline->CaptureConstantBufferMask(),
                                                     live_fog_inputs);
    }
    if (capture_stop) frame_dump.NoteCaptureStopMetadata(compute_metadata);
    std::scoped_lock lock{texture_cache.mutex, buffer_cache.mutex};
    // Command list pages the dispatch writes are copied for the split's front right after it.
    buffer_cache.Nx2CollectListCopies(true);
    SCOPE_EXIT {
        buffer_cache.Nx2CollectListCopies(false);
    };
    compute_phase = "resource_configuration";
    if (sync_compute) sync_trace.Record(compute_phase, compute_metadata);
    if (!pipeline->Configure(*kepler_compute, *gpu_memory, scheduler, buffer_cache,
                             texture_cache, input_dump, native_key, &frame_dump, range_native_key)) {
        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute resources unavailable");
        return;
    }

    if (capture_stop) {
        const auto& capture_qmd = kepler_compute->launch_description;
        const bool bounded_sky = capture_sky && frame_dump.ReadyForSkyCaptureDispatch() &&
            !kepler_compute->GetIndirectComputeAddress() &&
            capture_qmd.grid_dim_x == 1 && capture_qmd.grid_dim_y == 1 &&
            capture_qmd.grid_dim_z == 1 && capture_qmd.block_dim_x == 32 &&
            capture_qmd.block_dim_y == 32 && capture_qmd.block_dim_z == 1;
        if (!bounded_sky) {
            compute_phase = "diagnostic_capture_stop";
            frame_dump.FinishComputeCaptureStop();
            throw std::runtime_error("DIAGNOSTIC: captured resources and stopped before dispatch; this is intentional");
        }
    }
    // Configure has recorded pipeline/descriptors. Do not Finish between here and dispatch:
    // Finish allocates a fresh command context and would discard those bindings.
    if (sync_compute) sync_trace.Record("resources_configured", compute_metadata);
    const auto& qmd{kepler_compute->launch_description};
    compute_phase = "dispatch_validation";
    auto indirect_address = kepler_compute->GetIndirectComputeAddress();
    if (indirect_address) {
        // DispatchIndirect
        static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
        const auto post_op = VideoCommon::ObtainBufferOperation::DiscardWrite;
        const auto [buffer, offset] =
            buffer_cache.ObtainBuffer(*indirect_address, 12, sync_info, post_op);
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([pipeline, indirect_buffer = buffer->Handle(),
                          indirect_offset = offset](vk::CommandBuffer cmdbuf) {
            if (!pipeline->IsBound()) {
                return;
            }
            cmdbuf.DispatchIndirect(indirect_buffer, indirect_offset);
        });
        if (input_dump && native_key == 0xd9b1657f3a0b4bc2ULL && frame_dump.CapturingLightingInputs())
            frame_dump.NoteLightingOutputsAfter(native_key);
        frame_dump.NoteWrittenImagesAfter();
        NoteComputeForGraphicsBarrier();
        SubmitNx2ListCopies();
        compute_phase = "indirect_post_drain_begin";
        if (sync_compute) {
            sync_trace.Record(compute_phase, compute_metadata);
            scheduler.Finish();
            sync_trace.Record("indirect_post_drain_complete", compute_metadata);
        }
        compute_phase = "indirect_dispatched";
        return;
    }
    const std::array<u32, 3> dim{qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z};
    const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
    if (dim[0] > max_dim[0] || dim[1] > max_dim[1] || dim[2] > max_dim[2]) {
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    static constexpr VkMemoryBarrier READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
    };
    scheduler.Record([](vk::CommandBuffer cmdbuf) { cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               0, READ_BARRIER); });
    scheduler.Record([pipeline, dim, capture_sky](vk::CommandBuffer cmdbuf) {
        if (!pipeline->IsBound()) {
            if (capture_sky)
                throw std::runtime_error("DIAGNOSTIC sky pipeline was not bound; dispatch not recorded");
            return;
        }
        cmdbuf.Dispatch(dim[0], dim[1], dim[2]);
    });
    if (input_dump && native_key == 0xd9b1657f3a0b4bc2ULL && frame_dump.CapturingLightingInputs())
        frame_dump.NoteLightingOutputsAfter(native_key);
    frame_dump.NoteWrittenImagesAfter();
    NoteComputeForGraphicsBarrier();
    SubmitNx2ListCopies();

    if (capture_sky) {
        compute_phase = "diagnostic_sky_capture_stop";
        frame_dump.FinishComputeCaptureStop(true);
        if (FrameDump::DelayedSkyCaptureEnabled()) {
            if (!frame_dump.CompleteDelayedSkyCapture()) return;
            throw std::runtime_error("DIAGNOSTIC: captured two delayed sky dispatches before and after; this is intentional");
        }
        throw std::runtime_error("DIAGNOSTIC: captured one sky dispatch before and after; this is intentional");
    }
    compute_phase = "post_drain_begin";
    if (sync_compute) {
        sync_trace.Record(compute_phase, compute_metadata);
        scheduler.Finish();
        sync_trace.Record("post_drain_complete", compute_metadata);
    }
    compute_phase = "dispatched";

    // Log compute dispatch
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string params = fmt::format("groupCountX={}, groupCountY={}, groupCountZ={}",
            dim[0], dim[1], dim[2]);
        GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
            "vkCmdDispatch", params, VK_SUCCESS);
    }
    } catch (const Shader::Exception& error) {
        if (capture_stop) {
            throw std::runtime_error(fmt::format("DIAGNOSTIC capture-stop setup failed; kernel not dispatched: {}", error.what()));
        }
        throw;
    }
}

void RasterizerVulkan::ResetCounter(VideoCommon::QueryType type) {
    switch (type) {
    case VideoCommon::QueryType::ZPassPixelCount64:
    case VideoCommon::QueryType::StreamingByteCount:
    case VideoCommon::QueryType::StreamingPrimitivesSucceeded:
    case VideoCommon::QueryType::VtgPrimitivesOut:
        query_cache.CounterReset(type);
        return;
    default:
        LOG_DEBUG(Render_Vulkan, "Unimplemented counter reset={}", type);
        return;
    }
}

void RasterizerVulkan::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                             VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    query_cache.CounterReport(gpu_addr, type, flags, payload, subreport);
}

void RasterizerVulkan::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                 u32 size) {
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void Vulkan::RasterizerVulkan::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerVulkan::FlushAll() {}

void RasterizerVulkan::SubmitNx2ListCopies() {
    if (buffer_cache.Nx2CopyListPages(scheduler.CurrentTick()) && !scheduler.DeferFlush()) {
        scheduler.Flush();
    }
}

VideoCore::RasterizerInterface::GpuWrittenList RasterizerVulkan::ReadGpuWrittenCommandList(
    DAddr addr, std::span<u8> out) {
    // No cache lock: the front only reads the copies, under their own lock.
    switch (buffer_cache.Nx2ReadListCopy(addr, out)) {
    case BufferCache::Nx2ListRead::Copied:
        return GpuWrittenList::Copied;
    case BufferCache::Nx2ListRead::Synced:
        return GpuWrittenList::Synced;
    default:
        return GpuWrittenList::Unknown;
    }
}

void RasterizerVulkan::NoteGpuWrittenCommandListSynced(DAddr page) {
    buffer_cache.Nx2NoteListSynced(page);
}

void RasterizerVulkan::FlushQueryResults(DAddr addr, u64 size) {
    if (addr == 0 || size == 0 || !query_cache.SemiFlushRegion(addr, size)) {
        return;
    }
    // Some reports still wait for host results: wait for an operation queued behind them. The
    // fence is forced, since an empty one would run its callback at once and leave the operation
    // queued. (From Lichii #215.)
    std::mutex wait_mutex;
    std::condition_variable wait_cv;
    bool written{};
    fence_manager.SyncOperation([&] {
        std::scoped_lock lock{wait_mutex};
        written = true;
        wait_cv.notify_all();
    });
    fence_manager.SignalFence([] {}, true);
    scheduler.FlushDeferred();
    std::unique_lock lock{wait_mutex};
    wait_cv.wait(lock, [&written] { return written; });
}

void RasterizerVulkan::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.FlushRegion(addr, size);
    }
}

bool RasterizerVulkan::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (True(which & VideoCommon::CacheType::QueryCache) &&
        query_cache.IsRegionGpuModified(addr, size)) {
        return true;
    }
    if (!Settings::IsGPULevelHigh()) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushArea(DAddr addr, u64 size) {
    std::optional<VideoCore::RasterizerDownloadArea> texture_area;
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_area = texture_cache.GetFlushArea(addr, size);
    }
    if (texture_area) {
        // The core disarmed the read's pages before asking, and skips later reads in this area,
        // even through another device alias of those pages. The buffer cache arms only pages it
        // doesn't count as armed, so it must forget the whole area.
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.ForgetReadTraps(texture_area->start_address,
                                     texture_area->end_address - texture_area->start_address);
        return *texture_area;
    }
    if (buffer_cache.FlushesOnCpuRead()) {
        std::scoped_lock lock{buffer_cache.mutex};
        auto area = buffer_cache.GetFlushArea(addr, size);
        if (area) {
            return *area;
        }
    }
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    return new_area;
}

void RasterizerVulkan::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.InvalidateRegion(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::ShaderCache))) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
}

void RasterizerVulkan::InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) {
    // Even with skip_cpu_inner_invalidation: engines writing images in software (DMA pitch to
    // block linear copies of decoded video frames) reach the texture cache only through here.
    {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            texture_cache.WriteMemory(addr, size);
        }
    }
    if (Settings::values.skip_cpu_inner_invalidation.GetValue()) {
        return;
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            buffer_cache.WriteMemory(addr, size);
        }
    }
    for (const auto& [addr, size] : sequences) {
        query_cache.InvalidateRegion(addr, size);
    }
    pipeline_cache.InvalidateRegions(sequences);
}

bool RasterizerVulkan::OnCPUWrite(DAddr addr, u64 size) {
    DEBUG_ASSERT(addr != 0 || size != 0);
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.OnCPUWrite(addr, size)) {
            return true;
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return false;
}

void RasterizerVulkan::OnCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }

    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
}

bool RasterizerVulkan::TryCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) return true;
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        // WriteMemory downloads GPU-written bytes elsewhere on the written pages, which only the
        // GPU thread may do: leave those pages to the fallback.
        const DAddr page_begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        const DAddr page_end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
        if (buffer_cache.IsRegionGpuModified(page_begin, page_end - page_begin)) return false;
        texture_cache.WriteMemory(addr, size);
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return true;
}

void RasterizerVulkan::InvalidateGPUCache() {
    gpu.InvalidateGPUCache();
}

void RasterizerVulkan::UnmapMemory(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size, false);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
}

void RasterizerVulkan::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapGPUMemory(as_id, addr, size);
    }
    buffer_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerVulkan::SignalFence(std::function<void()>&& func, bool force_delay) {
    fence_manager.SignalFence(std::move(func), force_delay);
}

void RasterizerVulkan::SyncOperation(std::function<void()>&& func) {
    fence_manager.SyncOperation(std::move(func));
}

void RasterizerVulkan::SignalSyncPoint(u32 value) {
    fence_manager.SignalSyncPoint(value);
}

void RasterizerVulkan::SignalReference() {
    fence_manager.SignalReference();
}

void RasterizerVulkan::ReleaseFences(bool force) {
    fence_manager.WaitPendingFences(force, [this] {
        // The owner must submit before blocking: the fencing thread cannot run its idle hook.
        if (VideoCommon::GpuSplit::OnBackThread()) {
            scheduler.FlushDeferred();
        }
    });
}

void RasterizerVulkan::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                                VideoCommon::CacheType which) {
    if (Settings::IsGPULevelHigh()) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

void RasterizerVulkan::WaitForIdle() {
    // Everything but wait pixel operations. This intentionally includes FRAGMENT_SHADER_BIT because
    // fragment shaders can still write storage buffers.
    VkPipelineStageFlags flags =
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
        VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (device.IsExtTransformFeedbackSupported()) {
        flags |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
    }

    query_cache.NotifyWFI();

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([event = *wfi_event, flags](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetEvent(event, flags);
        cmdbuf.WaitEvents(event, flags, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, {}, {}, {});
    });
    fence_manager.SignalOrdering();
}

void RasterizerVulkan::FragmentBarrier() {
    // Ending the render pass records the barrier. The next draw ends it only if it reads what
    // the pass wrote (Lichii #161). EDEN_NX2_FRAGMENT_BARRIER=end always ends it.
    static const bool always_end = [] {
        const char* value = std::getenv("EDEN_NX2_FRAGMENT_BARRIER");
        return value != nullptr && std::strcmp(value, "end") == 0;
    }();
    if (always_end) {
        scheduler.RequestOutsideRenderPassOperationContext();
        return;
    }
    scheduler.DeferFragmentBarrier();
}

void RasterizerVulkan::TiledCacheBarrier() {
    // TODO: Implementing tiled barriers requires rewriting a good chunk of the Vulkan backend
}

void RasterizerVulkan::FlushCommands() {
    const bool skip_paired_flush = scheduler.TrySkipFenceFlush(
        program_id == 0x0400486024DEC000ULL && draw_counter != 0 &&
        !VideoCommon::GpuSplit::Get().BackRunning() && !VideoCommon::GpuSplit::OnBackThread());
    if (draw_counter == 0) {
        return;
    }
    draw_counter = 0;
    if (skip_paired_flush) {
        return;
    }
    // Fences with downloads or queries come here right after queueing: ~9,400 a second in
    // Hyrule Warriors: Age of Imprisonment fights. Batched like the fences themselves.
    if (!scheduler.DeferFlush()) {
        scheduler.Flush();
    }
}

void RasterizerVulkan::FrameDumpPass(const Framebuffer* framebuffer) {
    std::array<ImageView*, VideoCommon::NUM_RT> colors{};
    const auto& ids = texture_cache.GetRenderTargets().color_buffer_ids;
    for (size_t slot = 0; slot < ids.size(); ++slot) {
        if (ids[slot]) colors[slot] = &texture_cache.GetImageView(ids[slot]);
    }
    ImageView* depth = nullptr;
    const auto depth_id = texture_cache.GetRenderTargets().depth_buffer_id;
    if (frame_dump.CapturingFogDepth() && depth_id) {
        depth = &texture_cache.GetImageView(depth_id);
    }
    if (frame_dump.BeginPass(framebuffer, colors, depth)) scheduler.RequestRenderpass(framebuffer);
}

void RasterizerVulkan::TickFrame() {
    frame_dump.TickFrame();
    draw_counter = 0;
    pipeline_cache.TickFrame();
    guest_descriptor_queue.TickFrame();
    compute_pass_descriptor_queue.TickFrame();
    descriptor_buffer_ring.TickFrame();
    if (StorageFeedback* const feedback = StorageFeedback::Get()) {
        feedback->Tick(scheduler);
    }
    fence_manager.TickFrame();
    staging_pool.TickFrame();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
    }
}

bool RasterizerVulkan::AccelerateConditionalRendering() {
    gpu_memory->FlushCaching();
    return query_cache.AccelerateHostConditionalRendering();
}

bool RasterizerVulkan::HasDrawTransformFeedback() {
    return device.IsTransformFeedbackDrawSupported();
}

bool RasterizerVulkan::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                             const Tegra::Engines::Fermi2D::Surface& dst,
                                             const Tegra::Engines::Fermi2D::Config& copy_config) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

bool RasterizerVulkan::AccelerateTextureCopy(const Tegra::Texture::TICEntry& src,
                                            const Tegra::Texture::TICEntry& dst,
                                            const Tegra::Engines::Fermi2D::Config& copy_config) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitTexture(dst, src, copy_config);
}

bool RasterizerVulkan::UsesNvidiaDlss() const {
    return texture_cache_runtime.UsesNvidiaDlss();
}

bool RasterizerVulkan::SupportsGameDlssNetwork() const {
    // Compute pipelines are created 32 threads wide where the device can be told to.
    return device.IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT);
}

std::optional<VideoCore::GameDlssProbe> RasterizerVulkan::ProbeGameDlssOutput(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.ProbeGameDlssOutput(dst, src);
}

bool RasterizerVulkan::AccelerateTemporalUpscale(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
    const Tegra::Engines::Fermi2D::Surface& motion_vectors, const Tegra::Texture::TICEntry* depth,
    std::array<f32, 2> jitter, std::array<f32, 2> motion_scale) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.TemporalUpscale(dst, src, motion_vectors, depth, jitter, motion_scale);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerVulkan::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerVulkan::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                                std::span<const u8> memory) {
    auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (gpu_memory->UsesNativeTiling()) {
        u32 first_word{};
        if (memory.size() >= sizeof(first_word)) std::memcpy(&first_word, memory.data(), sizeof(first_word));
        VideoCommon::TraceNativeQuery(11, gpu_memory->GetID(), address, cpu_addr.value_or(0),
            0, first_word, static_cast<u32>(copy_size));
        static const GPUVAddr watch = [] {
            const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
            return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
        }();
        if (watch != 0 && address < watch + 16 && address + copy_size > watch) {
            LOG_INFO(Render_Vulkan, "Semaphore watch: inline write {:#x} size {} first word {:#x}",
                     address, copy_size, first_word);
        }
    }
    if (!cpu_addr) [[unlikely]] {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    // EDEN_NX2_INLINE_DEFER=0: invalidate every cache at once (for comparisons).
    static const bool defer = [] {
        const char* value = std::getenv("EDEN_NX2_INLINE_DEFER");
        return value == nullptr || value[0] != '0';
    }();
    // One cache at a time: holding both while waiting for either (std::lock) waited longer, the
    // fencing thread and CPU write traps holding one of them most of the time.
    bool uploaded;
    {
        std::scoped_lock lock{buffer_cache.mutex};
        // Guest memory has the words now: a command list copy of the page no longer does.
        buffer_cache.Nx2NoteListHostWrite(*cpu_addr, copy_size);
        uploaded = buffer_cache.InlineMemory(*cpu_addr, copy_size, memory);
        if (!uploaded && !defer) {
            buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
    }
    // Pending query results must not land over the new words later: drop them now.
    query_cache.InvalidateRegion(*cpu_addr, copy_size);
    if (!uploaded && defer) {
        // Title 1 makes thousands of small inline writes between draws (~10% of the GPU
        // thread invalidating every cache for each). Guest memory is written; the buffer,
        // texture and shader caches see the merged ranges at the next FlushCaching, which runs
        // before anything of theirs is used. Writes into GPU-written buffer data were uploaded
        // in place above, as before.
        gpu_memory->AccumulateInvalidation(address, copy_size);
        return;
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
}

std::optional<FramebufferTextureInfo> RasterizerVulkan::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr, u32 pixel_stride) {
    if (!framebuffer_addr) {
        return {};
    }
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    static u32 display_trace_count{};
    static const bool frame_capture = std::getenv("EDEN_NATIVE_FRAME_CAPTURE") != nullptr;
    if (frame_capture && display_trace_count++ < 64) {
        LOG_INFO(HW_GPU, "Native display source: address={:#x}, offset={:#x}, dimensions={}x{}, stride={}, accelerated={}, mask={:#x}",
                 framebuffer_addr, config.offset, config.width, config.height, pixel_stride,
                 image_view != nullptr, config.layer_stack_mask);
    }
    if (!image_view) {
        return {};
    }

    if (frame_dump.Capturing()) {
        frame_dump.EndPass();
        frame_dump.CapturePresent(*image_view);
    }
    query_cache.NotifySegment(false);

    const auto& resolution = Settings::values.resolution_info;

    FramebufferTextureInfo info{};
    info.image = image_view->ImageHandle();
    info.image_view = image_view->Handle(Shader::TextureType::Color2D);
    info.width = image_view->size.width;
    info.height = image_view->size.height;
    info.scaled_width = scaled ? resolution.ScaleUp(info.width) : info.width;
    info.scaled_height = scaled ? resolution.ScaleUp(info.height) : info.height;
    return info;
}

void RasterizerVulkan::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                         const VideoCore::DiskResourceLoadCallback& callback) {
    pipeline_cache.LoadDiskResources(title_id, stop_loading, callback);
}

void RasterizerVulkan::FlushWork() {
#ifdef __ANDROID__
    static constexpr u32 DRAWS_TO_DISPATCH = 512;
    static constexpr u32 CHECK_MASK = 3;
#else
    // Every 32 draws (was 8): each dispatch woke the worker thread, and Switch 2 titles make
    // ~55,000 draws a second. A full chunk is dispatched on its own (Scheduler::Record).
    static constexpr u32 DRAWS_TO_DISPATCH = 4096;
    static constexpr u32 CHECK_MASK = 31;
#endif // __ANDROID__

    static_assert(DRAWS_TO_DISPATCH % (CHECK_MASK + 1) == 0);
    if ((++draw_counter & CHECK_MASK) != 0) {
        return;
    }
    // Keep the host GPU working while native commands are still being recorded.
    static const u32 native_flush_draws = [] {
        constexpr u32 fallback = 128;
        const char* value = std::getenv("EDEN_NX2_FLUSH_DRAWS");
        if (value == nullptr || value[0] < '0' || value[0] > '9') {
            return fallback;
        }
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (*end != '\0' || parsed == 0) {
            return fallback;
        }
        return static_cast<u32>(std::clamp(parsed, 32UL, 1'048'576UL));
    }();
    const u32 draws_to_dispatch = Tegra::g_nx2_ampere.load(std::memory_order_relaxed)
                                     ? native_flush_draws
                                     : DRAWS_TO_DISPATCH;
    if (draw_counter < draws_to_dispatch) {
        scheduler.DispatchWork();
        return;
    }
    scheduler.Flush();
    draw_counter = 0;
}

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_,
                             Scheduler& scheduler_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, scheduler{scheduler_} {}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(src_address, amount, value);
}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.Nx2CollectListCopies(true);
    const bool copied = buffer_cache.DMACopy(src_address, dest_address, amount);
    if (buffer_cache.Nx2CopyListPages(scheduler.CurrentTick())) {
        scheduler.Flush();
    }
    buffer_cache.Nx2CollectListCopies(false);
    return copied;
}

template <bool IS_IMAGE_UPLOAD>
bool AccelerateDMA::DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                                       const Tegra::DMA::BufferOperand& buffer_operand,
                                       const Tegra::DMA::ImageOperand& image_operand) {
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    const auto image_id = texture_cache.DmaImageId(image_operand, IS_IMAGE_UPLOAD);
    if (image_id == VideoCommon::NULL_IMAGE_ID) {
        return false;
    }
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
    const auto post_op = IS_IMAGE_UPLOAD ? VideoCommon::ObtainBufferOperation::DoNothing
                                         : VideoCommon::ObtainBufferOperation::MarkAsWritten;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(buffer_operand.address, buffer_size, sync_info, post_op);

    const auto [image, copy] = texture_cache.DmaBufferImageCopy(
        copy_info, buffer_operand, image_operand, image_id, IS_IMAGE_UPLOAD);
    const std::span copy_span{&copy, 1};

    if constexpr (IS_IMAGE_UPLOAD) {
        texture_cache.PrepareImage(image_id, true, false);
        image->UploadMemory(buffer->Handle(), offset, copy_span);
    } else {
        if (offset % BytesPerBlock(image->info.format)) {
            return false;
        }
        texture_cache.DownloadImageIntoBuffer(image, buffer->Handle(), offset, copy_span,
                                              buffer_operand.address, buffer_size);
    }
    return true;
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& image_operand,
                                  const Tegra::DMA::BufferOperand& buffer_operand) {
    return DmaBufferImageCopy<false>(copy_info, buffer_operand, image_operand);
}

bool AccelerateDMA::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::BufferOperand& buffer_operand,
                                  const Tegra::DMA::ImageOperand& image_operand) {
    return DmaBufferImageCopy<true>(copy_info, buffer_operand, image_operand);
}

void RasterizerVulkan::UpdateDynamicStates() {
    auto& regs = maxwell3d->regs;
    auto& flags = maxwell3d->dirty.flags;
    const auto topology = maxwell3d->draw_manager.draw_state.topology;
    const bool topology_changed = state_tracker.ChangePrimitiveTopology(topology);
    if (topology_changed) {
        flags[Dirty::DepthBiasEnable] = true;
        flags[Dirty::PrimitiveRestartEnable] = true;
    }

    UpdateViewportsState(regs);
    UpdateScissorsState(regs);
    UpdateDepthBias(regs);
    UpdateBlendConstants(regs);
    UpdateDepthBounds(regs);
    UpdateStencilFaces(regs);
    UpdateLineWidth(regs);

    if (device.IsExtExtendedDynamicStateSupported()) {
        UpdateCullMode(regs);
        UpdateDepthCompareOp(regs);
        UpdateFrontFace(regs);
        UpdateStencilOp(regs);
        if (state_tracker.TouchStateEnable()) {
            UpdateDepthBoundsTestEnable(regs);
            UpdateDepthTestEnable(regs);
            UpdateDepthWriteEnable(regs);
            UpdateStencilTestEnable(regs);
        }
        if (topology_changed) {
            scheduler.Record([topology_vk = MaxwellToVK::PrimitiveTopology(device, topology)](
                                 vk::CommandBuffer cmdbuf) {
                cmdbuf.SetPrimitiveTopologyEXT(topology_vk);
            });
        }
    }

    if (device.IsExtExtendedDynamicState2Supported()) {
        UpdatePrimitiveRestartEnable(regs);
        UpdateRasterizerDiscardEnable(regs);
        UpdateDepthBiasEnable(regs);
    }

    if (device.IsExtExtendedDynamicState2ExtrasSupported()) {
        UpdateLogicOp(regs);
    }

    if (device.IsExtExtendedDynamicState3EnablesSupported()) {
        using namespace Tegra::Engines;
        // AMD Workaround: LogicOp incompatible with float render targets
        if (device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_OPEN_SOURCE ||
            device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_PROPRIETARY) {
            const auto has_float = std::any_of(
                regs.vertex_attrib_format.begin(), regs.vertex_attrib_format.end(),
                [](const auto& attrib) {
                    return attrib.type == Maxwell3D::Regs::VertexAttribute::Type::Float;
                }
            );
            if (regs.logic_op.enable) {
                regs.logic_op.enable = static_cast<u32>(!has_float);
            }
        }
        UpdateLogicOpEnable(regs);
        UpdateDepthClampEnable(regs);
        UpdateLineRasterizationMode(regs);
        UpdateLineStippleEnable(regs);
        UpdateConservativeRasterizationMode(regs);
        UpdateAlphaToCoverageEnable(regs);
        UpdateAlphaToOneEnable(regs);
    }

    if (device.IsExtExtendedDynamicState3BlendingSupported()) {
        UpdateBlending(regs);
    } else if (device.IsExtColorWriteEnableSupported()) {
        UpdateColorWriteEnable(regs);
    }

    if (device.IsExtVertexInputDynamicStateSupported()) {
        if (auto* gp = pipeline_cache.CurrentGraphicsPipeline(); gp && gp->HasDynamicVertexInput()) {
            UpdateVertexInput(regs);
        }
    }
}

void RasterizerVulkan::HandleTransformFeedback() {
    static std::once_flag warn_unsupported;

    const auto& regs = maxwell3d->regs;
    if (!device.IsExtTransformFeedbackSupported()) {
        if (regs.transform_feedback_enabled != 0) {
            std::call_once(warn_unsupported, [&] {
                LOG_WARNING(Render_Vulkan, "Transform feedback requested by guest but VK_EXT_transform_feedback is unavailable; queries disabled");
            });
        } else {
            std::call_once(warn_unsupported, [&] {
                LOG_INFO(Render_Vulkan, "VK_EXT_transform_feedback not available on device");
            });
        }
        return;
    }
    query_cache.CounterEnable(VideoCommon::QueryType::StreamingByteCount,
                              regs.transform_feedback_enabled);
    if (regs.transform_feedback_enabled != 0) {
        // Log extension usage for transform feedback
        if (GPU::Logging::IsActive()) {
            GPU::Logging::GPULogger::GetInstance().LogExtensionUsage(
                "VK_EXT_transform_feedback", "HandleTransformFeedback");
        }
        UNIMPLEMENTED_IF(regs.IsShaderConfigEnabled(Maxwell::ShaderType::TessellationInit) ||
                         regs.IsShaderConfigEnabled(Maxwell::ShaderType::Tessellation));
    }
}

void RasterizerVulkan::UpdateViewportsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchViewports()) {
        return;
    }

    maxwell3d->dirty.flags[Dirty::Scissors] = true;

    if (!regs.viewport_scale_offset_enabled) {
        float x = static_cast<float>(regs.surface_clip.x);
        float y = static_cast<float>(regs.surface_clip.y);
        float width = (std::max)(1.0f, static_cast<float>(regs.surface_clip.width));
        float height = (std::max)(1.0f, static_cast<float>(regs.surface_clip.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            y += height;
            height = -height;
        }
        VkViewport viewport{
            .x = x,
            .y = y,
            .width = width,
            .height = height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        scheduler.Record([this, viewport](vk::CommandBuffer cmdbuf) {
            const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkViewport, Maxwell::NumViewports> viewport_list{};
            viewport_list.fill(viewport);
            const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
            cmdbuf.SetViewport(0, viewports);
        });
        return;
    }
    const bool is_rescaling{texture_cache.IsRescaling()};
    const float scale = is_rescaling ? Settings::values.resolution_info.up_factor : 1.0f;
    const std::array viewport_list{
        GetViewportState(device, regs, 0, scale),  GetViewportState(device, regs, 1, scale),
        GetViewportState(device, regs, 2, scale),  GetViewportState(device, regs, 3, scale),
        GetViewportState(device, regs, 4, scale),  GetViewportState(device, regs, 5, scale),
        GetViewportState(device, regs, 6, scale),  GetViewportState(device, regs, 7, scale),
        GetViewportState(device, regs, 8, scale),  GetViewportState(device, regs, 9, scale),
        GetViewportState(device, regs, 10, scale), GetViewportState(device, regs, 11, scale),
        GetViewportState(device, regs, 12, scale), GetViewportState(device, regs, 13, scale),
        GetViewportState(device, regs, 14, scale), GetViewportState(device, regs, 15, scale),
    };
    scheduler.Record([this, viewport_list](vk::CommandBuffer cmdbuf) {
        const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
        cmdbuf.SetViewport(0, viewports);
    });
}

void RasterizerVulkan::UpdateScissorsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchScissors()) {
        return;
    }
    if (!regs.viewport_scale_offset_enabled) {
        u32 x = regs.surface_clip.x;
        u32 y = regs.surface_clip.y;
        u32 width = (std::max)(1u, static_cast<u32>(regs.surface_clip.width));
        u32 height = (std::max)(1u, static_cast<u32>(regs.surface_clip.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            y = regs.surface_clip.height - (y + height);
        }
        VkRect2D scissor{};
        scissor.offset.x = static_cast<int32_t>(x);
        scissor.offset.y = static_cast<int32_t>(y);
        scissor.extent.width  = width;
        scissor.extent.height = height;
        scheduler.Record([this, scissor](vk::CommandBuffer cmdbuf) {
            const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkRect2D, Maxwell::NumViewports> scissor_list{};
            scissor_list.fill(scissor);
            const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
            cmdbuf.SetScissor(0, scissors);
        });
        return;
    }
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    const std::array scissor_list{
        GetScissorState(regs, 0, up_scale, down_shift),
        GetScissorState(regs, 1, up_scale, down_shift),
        GetScissorState(regs, 2, up_scale, down_shift),
        GetScissorState(regs, 3, up_scale, down_shift),
        GetScissorState(regs, 4, up_scale, down_shift),
        GetScissorState(regs, 5, up_scale, down_shift),
        GetScissorState(regs, 6, up_scale, down_shift),
        GetScissorState(regs, 7, up_scale, down_shift),
        GetScissorState(regs, 8, up_scale, down_shift),
        GetScissorState(regs, 9, up_scale, down_shift),
        GetScissorState(regs, 10, up_scale, down_shift),
        GetScissorState(regs, 11, up_scale, down_shift),
        GetScissorState(regs, 12, up_scale, down_shift),
        GetScissorState(regs, 13, up_scale, down_shift),
        GetScissorState(regs, 14, up_scale, down_shift),
        GetScissorState(regs, 15, up_scale, down_shift),
    };
    scheduler.Record([this, scissor_list](vk::CommandBuffer cmdbuf) {
        const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
        cmdbuf.SetScissor(0, scissors);
    });
}

void RasterizerVulkan::UpdateDepthBias(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBias()) {
        return;
    }
    float units = regs.depth_bias / 2.0f;
    const bool is_d24 = regs.zeta.format == Tegra::DepthFormat::Z24_UNORM_S8_UINT ||
                        regs.zeta.format == Tegra::DepthFormat::X8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::S8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::V8Z24_UNORM;

    if (is_d24 && !device.SupportsD24DepthBuffer()) {
        static constexpr const size_t length = sizeof(NEEDS_D24) / sizeof(NEEDS_D24[0]);

        static constexpr const u64* start = NEEDS_D24;
        static constexpr const u64* end = NEEDS_D24 + length;

        const u64* it = std::find(start, end, program_id);

        if (it != end) {
            // the base formulas can be obtained from here:
            //   https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-output-merger-stage-depth-bias
            const double rescale_factor =
                static_cast<double>(1ULL << (32 - 24)) / (static_cast<double>(0x1.ep+127));
            units = static_cast<float>(static_cast<double>(units) * rescale_factor);
        }
    }

    scheduler.Record([constant = units, clamp = regs.depth_bias_clamp,
                      factor = regs.slope_scale_depth_bias, this](vk::CommandBuffer cmdbuf) {
        if (device.IsExtDepthBiasControlSupported()) {
            static VkDepthBiasRepresentationInfoEXT bias_info{
                .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
                .pNext = nullptr,
                .depthBiasRepresentation =
                    VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT,
                .depthBiasExact = VK_FALSE,
            };

            cmdbuf.SetDepthBias(constant, clamp, factor, &bias_info);
        } else {
            cmdbuf.SetDepthBias(constant, clamp, factor);
        }
    });
}

void RasterizerVulkan::UpdateBlendConstants(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchBlendConstants()) {
        return;
    }
    const std::array blend_color = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                                    regs.blend_color.a};
    scheduler.Record(
        [blend_color](vk::CommandBuffer cmdbuf) { cmdbuf.SetBlendConstants(blend_color.data()); });
}

void RasterizerVulkan::UpdateDepthBounds(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBounds()) {
        return;
    }
    scheduler.Record([min = regs.depth_bounds[0], max = regs.depth_bounds[1]](
                         vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBounds(min, max); });
}

void RasterizerVulkan::UpdateStencilFaces(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilProperties()) {
        return;
    }
    bool update_references = state_tracker.TouchStencilReference();
    bool update_write_mask = state_tracker.TouchStencilWriteMask();
    bool update_compare_masks = state_tracker.TouchStencilCompare();
    if (state_tracker.TouchStencilSide(regs.stencil_two_side_enable != 0)) {
        update_references = true;
        update_write_mask = true;
        update_compare_masks = true;
    }
    if (update_references) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_back_ref);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_front_ref);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_ref = regs.stencil_front_ref, back_ref = regs.stencil_back_ref,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_ref != back_ref;
                // Front face
                cmdbuf.SetStencilReference(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_ref);
                if (set_back) {
                    cmdbuf.SetStencilReference(VK_STENCIL_FACE_BACK_BIT, back_ref);
                }
            });
        }();
    }
    if (update_write_mask) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_back_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_front_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_write_mask = regs.stencil_front_mask,
                              back_write_mask = regs.stencil_back_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_write_mask != back_write_mask;
                // Front face
                cmdbuf.SetStencilWriteMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_write_mask);
                if (set_back) {
                    cmdbuf.SetStencilWriteMask(VK_STENCIL_FACE_BACK_BIT, back_write_mask);
                }
            });
        }();
    }
    if (update_compare_masks) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_back_func_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_front_func_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_test_mask = regs.stencil_front_func_mask,
                              back_test_mask = regs.stencil_back_func_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_test_mask != back_test_mask;
                // Front face
                cmdbuf.SetStencilCompareMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                      : VK_STENCIL_FACE_FRONT_AND_BACK,
                                             front_test_mask);
                if (set_back) {
                    cmdbuf.SetStencilCompareMask(VK_STENCIL_FACE_BACK_BIT, back_test_mask);
                }
            });
        }();
    }
    state_tracker.ClearStencilReset();
}

void RasterizerVulkan::UpdateLineWidth(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineWidth()) {
        return;
    }
    const float width =
        regs.line_anti_alias_enable ? regs.line_width_smooth : regs.line_width_aliased;
    scheduler.Record([width](vk::CommandBuffer cmdbuf) { cmdbuf.SetLineWidth(width); });
}

void RasterizerVulkan::UpdateCullMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchCullMode()) {
        return;
    }
    scheduler.Record([enabled = regs.gl_cull_test_enabled,
                      cull_face = regs.gl_cull_face](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetCullModeEXT(enabled ? MaxwellToVK::CullFace(cull_face) : VK_CULL_MODE_NONE);
    });
}

void RasterizerVulkan::UpdateDepthBoundsTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBoundsTestEnable()) {
        return;
    }
    bool enabled = regs.depth_bounds_enable;
    if (enabled && !device.IsDepthBoundsSupported()) {
        LOG_WARNING(Render_Vulkan, "Depth bounds is enabled but not supported");
        enabled = false;
    }
    scheduler.Record([enable = enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthBoundsTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthTestEnable()) {
        return;
    }
    scheduler.Record([enable = regs.depth_test_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthWriteEnable()) {
        return;
    }
    scheduler.Record([enable = regs.depth_write_enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthWriteEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdatePrimitiveRestartEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchPrimitiveRestartEnable()) {
        return;
    }

    bool enable = regs.primitive_restart.enabled != 0;
    if (device.IsMoltenVK()) {
        enable = true;
    } else if (enable) {
        const auto topology = MaxwellToVK::PrimitiveTopology(device, maxwell3d->draw_manager.draw_state.topology);
        enable = IsPrimitiveRestartSupported(device, topology);
    }

    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetPrimitiveRestartEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateRasterizerDiscardEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchRasterizerDiscardEnable()) {
        return;
    }
    scheduler.Record([disable = regs.rasterize_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetRasterizerDiscardEnableEXT(disable == 0);
    });
}

void RasterizerVulkan::UpdateConservativeRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchConservativeRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3ConservativeRasterizationMode()) {
        return;
    }

    scheduler.Record([enable = regs.conservative_raster_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetConservativeRasterizationModeEXT(
            enable ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                   : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
    });
}

void RasterizerVulkan::UpdateLineStippleEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineStippleEnable()) {
        return;
    }

    if (!device.SupportsDynamicState3LineStippleEnable()) {
        return;
    }

    scheduler.Record([enable = regs.line_stipple_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineStippleEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateLineRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!device.IsExtLineRasterizationSupported()) {
        return;
    }
    if (!state_tracker.TouchLineRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3LineRasterizationMode()) {
        static std::once_flag warn_missing_rect;
        std::call_once(warn_missing_rect, [] {
            LOG_WARNING(Render_Vulkan,
                        "Driver lacks rectangular line rasterization support; skipping dynamic "
                        "line state updates");
        });
        return;
    }

    const bool wants_smooth = regs.line_anti_alias_enable != 0;
    VkLineRasterizationModeEXT mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT;
    if (wants_smooth) {
        if (device.SupportsSmoothLines()) {
            mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT;
        } else {
            static std::once_flag warn_missing_smooth;
            std::call_once(warn_missing_smooth, [] {
                LOG_WARNING(Render_Vulkan,
                            "Line anti-aliasing requested but smoothLines feature unavailable; "
                            "using rectangular rasterization");
            });
        }
    }
    scheduler.Record([mode](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineRasterizationModeEXT(mode);
    });
}

void RasterizerVulkan::UpdateDepthBiasEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBiasEnable()) {
        return;
    }
    constexpr size_t POINT = 0;
    constexpr size_t LINE = 1;
    constexpr size_t POLYGON = 2;
    static constexpr std::array POLYGON_OFFSET_ENABLE_LUT = {
        POINT,   // Points
        LINE,    // Lines
        LINE,    // LineLoop
        LINE,    // LineStrip
        POLYGON, // Triangles
        POLYGON, // TriangleStrip
        POLYGON, // TriangleFan
        POLYGON, // Quads
        POLYGON, // QuadStrip
        POLYGON, // Polygon
        LINE,    // LinesAdjacency
        LINE,    // LineStripAdjacency
        POLYGON, // TrianglesAdjacency
        POLYGON, // TriangleStripAdjacency
        POLYGON, // Patches
    };
    const std::array enabled_lut{
        regs.polygon_offset_point_enable,
        regs.polygon_offset_line_enable,
        regs.polygon_offset_fill_enable,
    };
    const u32 topology_index = u32(maxwell3d->draw_manager.draw_state.topology);
    const u32 enable = enabled_lut[POLYGON_OFFSET_ENABLE_LUT[topology_index]];
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBiasEnableEXT(enable != 0); });
}

void RasterizerVulkan::UpdateLogicOpEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOpEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3LogicOpEnable()) {
        return;
    }
    scheduler.Record([enable = regs.logic_op.enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLogicOpEnableEXT(enable != 0);
    });
}

void RasterizerVulkan::UpdateDepthClampEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthClampEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3DepthClampEnable()) {
        return;
    }
    bool is_enabled = !(regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::Passthrough ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumXYZ ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumZ);
    scheduler.Record(
        [is_enabled](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthClampEnableEXT(is_enabled); });
}

void RasterizerVulkan::UpdateAlphaToCoverageEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchAlphaToCoverageEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3AlphaToCoverageEnable()) {
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    const bool enable = pipeline != nullptr && pipeline->SupportsAlphaToCoverage() &&
                        regs.anti_alias_alpha_control.alpha_to_coverage != 0;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToCoverageEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateAlphaToOneEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchAlphaToOneEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3AlphaToOneEnable()) {
        static std::once_flag warn_alpha_to_one;
        std::call_once(warn_alpha_to_one, [] {
            LOG_WARNING(Render_Vulkan,
                        "Alpha-to-one is not supported on this device; forcing it disabled");
        });
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    const bool enable = pipeline != nullptr && pipeline->SupportsAlphaToOne() &&
                        regs.anti_alias_alpha_control.alpha_to_one != 0;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToOneEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateDepthCompareOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthCompareOp()) {
        return;
    }
    scheduler.Record([func = regs.depth_test_func](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthCompareOpEXT(MaxwellToVK::ComparisonOp(func));
    });
}

void RasterizerVulkan::UpdateFrontFace(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchFrontFace()) {
        return;
    }

    VkFrontFace front_face = MaxwellToVK::FrontFace(regs.gl_front_face);
    if (regs.window_origin.flip_y != 0) {
        front_face = front_face == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                           : VK_FRONT_FACE_CLOCKWISE;
    }
    scheduler.Record(
        [front_face](vk::CommandBuffer cmdbuf) { cmdbuf.SetFrontFaceEXT(front_face); });
}

void RasterizerVulkan::UpdateStencilOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilOp()) {
        return;
    }
    const Maxwell::StencilOp::Op fail = regs.stencil_front_op.fail;
    const Maxwell::StencilOp::Op zfail = regs.stencil_front_op.zfail;
    const Maxwell::StencilOp::Op zpass = regs.stencil_front_op.zpass;
    const Maxwell::ComparisonOp compare = regs.stencil_front_op.func;
    if (regs.stencil_two_side_enable) {
        // Separate stencil op per face
        const Maxwell::StencilOp::Op back_fail = regs.stencil_back_op.fail;
        const Maxwell::StencilOp::Op back_zfail = regs.stencil_back_op.zfail;
        const Maxwell::StencilOp::Op back_zpass = regs.stencil_back_op.zpass;
        const Maxwell::ComparisonOp back_compare = regs.stencil_back_op.func;
        scheduler.Record([fail, zfail, zpass, compare, back_fail, back_zfail, back_zpass,
                          back_compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_BIT, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_BACK_BIT, MaxwellToVK::StencilOp(back_fail),
                                   MaxwellToVK::StencilOp(back_zpass),
                                   MaxwellToVK::StencilOp(back_zfail),
                                   MaxwellToVK::ComparisonOp(back_compare));
        });
    } else {
        // Front face defines the stencil op of both faces
        scheduler.Record([fail, zfail, zpass, compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_AND_BACK, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
        });
    }
}

void RasterizerVulkan::UpdateLogicOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOp()) {
        return;
    }
    const auto op_value = static_cast<u32>(regs.logic_op.op);
    auto op = op_value >= 0x1500 && op_value < 0x1510 ? static_cast<VkLogicOp>(op_value - 0x1500)
                                                      : VK_LOGIC_OP_NO_OP;
    scheduler.Record([op](vk::CommandBuffer cmdbuf) { cmdbuf.SetLogicOpEXT(op); });
}

void RasterizerVulkan::UpdateBlending(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchBlending()) {
        return;
    }

    if (state_tracker.TouchColorMask()) {
        std::array<VkColorComponentFlags, Maxwell::NumRenderTargets> setup_masks{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
            auto& current = setup_masks[index];
            if (mask.R) {
                current |= VK_COLOR_COMPONENT_R_BIT;
            }
            if (mask.G) {
                current |= VK_COLOR_COMPONENT_G_BIT;
            }
            if (mask.B) {
                current |= VK_COLOR_COMPONENT_B_BIT;
            }
            if (mask.A) {
                current |= VK_COLOR_COMPONENT_A_BIT;
            }
        }
        scheduler.Record([setup_masks](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorWriteMaskEXT(0, setup_masks);
        });
    }

    if (state_tracker.TouchBlendEnable()) {
        std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            bool is_integer = false;
            if (regs.rt[index].format != Tegra::RenderTargetFormat::NONE) {
                const auto format =
                    VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[index].format);
                is_integer = IsPixelFormatInteger(format);
            }
            setup_enables[index] =
                (!is_integer && regs.blend.enable[index] != 0) ? VK_TRUE : VK_FALSE;
        }
        scheduler.Record([setup_enables](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEnableEXT(0, setup_enables);
        });
    }

    if (state_tracker.TouchBlendEquations()) {
        std::array<VkColorBlendEquationEXT, Maxwell::NumRenderTargets> setup_blends{};

        const auto blend_setup = [&](auto& host_blend, const auto& guest_blend) {
            host_blend.srcColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_source);
            host_blend.dstColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_dest);
            host_blend.colorBlendOp = MaxwellToVK::BlendEquation(guest_blend.color_op);
            host_blend.srcAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_source);
            host_blend.dstAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_dest);
            host_blend.alphaBlendOp = MaxwellToVK::BlendEquation(guest_blend.alpha_op);
        };

        // Single blend equation for all targets
        if (!regs.blend_per_target_enabled) {
            // Temporary workaround for games that use iterated blending
            if (regs.iterated_blend.enable && Settings::values.use_squashed_iterated_blend) {
                setup_blends[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].colorBlendOp = VK_BLEND_OP_ADD;
                setup_blends[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
                setup_blends[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                setup_blends[0].alphaBlendOp = VK_BLEND_OP_ADD;
            } else {
                blend_setup(setup_blends[0], regs.blend);
            }

            // Copy first blend state to all other targets
            for (size_t index = 1; index < Maxwell::NumRenderTargets; index++) {
                setup_blends[index] = setup_blends[0];
            }
        } else {
            // Per-target blending
            for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
                blend_setup(setup_blends[index], regs.blend_per_target[index]);
            }
        }

        scheduler.Record([setup_blends](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEquationEXT(0, setup_blends);
        });
    }
}

void RasterizerVulkan::UpdateColorWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchColorMask()) {
        return;
    }
    std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
    for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
        const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
        setup_enables[index] = (mask.R || mask.G || mask.B || mask.A) ? VK_TRUE : VK_FALSE;
    }
    scheduler.Record([setup_enables](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetColorWriteEnableEXT(setup_enables);
    });
}

void RasterizerVulkan::UpdateStencilTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilTestEnable()) {
        return;
    }
    scheduler.Record([enable = regs.stencil_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetStencilTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateVertexInput(Tegra::Engines::Maxwell3D::Regs& regs) {
    auto& dirty{maxwell3d->dirty.flags};
    const bool vertex_input_dirty = dirty[Dirty::VertexInput];
    const bool vertex_buffers_dirty = dirty[VideoCommon::Dirty::VertexBuffers];
    if (!vertex_input_dirty && !vertex_buffers_dirty) {
        return;
    }
    dirty[Dirty::VertexInput] = false;

    boost::container::static_vector<VkVertexInputBindingDescription2EXT, 32> bindings;
    boost::container::static_vector<VkVertexInputAttributeDescription2EXT, 32> attributes;

    const u32 max_attributes =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexAttributes,
                                          device.GetMaxVertexInputAttributes()));
    const u32 max_bindings =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexArrays,
                                          device.GetMaxVertexInputBindings()));


    for (u32 index = 0; index < max_attributes; ++index) {
        const Maxwell::VertexAttribute attribute{regs.vertex_attrib_format[index]};
        const u32 binding{attribute.buffer};
        if (attribute.constant || binding >= max_bindings) {
            continue;
        }
        attributes.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .location = index,
            .binding = binding,
            .format = MaxwellToVK::VertexFormat(device, attribute.type, attribute.size),
            .offset = attribute.offset +
                      (buffer_cache_runtime.IsVertexRepacked(binding) && attribute.offset >= 65 ? 3 : 0),
        });
    }

    for (u32 binding = 0; binding < max_bindings; ++binding) {
        const auto& input_binding{regs.vertex_streams[binding]};
        const bool is_instanced{regs.vertex_stream_instances.IsInstancingEnabled(binding)};
        bindings.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .binding = binding,
            .stride = buffer_cache_runtime.IsVertexRepacked(binding) ? 72U : input_binding.stride.Value(),
            .inputRate = is_instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX,
            .divisor = is_instanced ? input_binding.frequency : 1,
        });
    }

    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        dirty[Dirty::VertexAttribute0 + index] = false;
    }
    for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
        dirty[Dirty::VertexBinding0 + index] = false;
    }

    scheduler.Record([bindings, attributes](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetVertexInputEXT(bindings, attributes);
    });
}

void RasterizerVulkan::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.CreateChannel(channel);
        buffer_cache.CreateChannel(channel);
    }
    pipeline_cache.CreateChannel(channel);
    query_cache.CreateChannel(channel);
    state_tracker.SetupTables(channel);
}

void RasterizerVulkan::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.BindToChannel(channel_id);
        buffer_cache.BindToChannel(channel_id);
    }
    pipeline_cache.BindToChannel(channel_id);
    query_cache.BindToChannel(channel_id);
    state_tracker.ChangeChannel(channel);
    state_tracker.InvalidateState();
}

void RasterizerVulkan::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.EraseChannel(channel_id);
        buffer_cache.EraseChannel(channel_id);
    }
    pipeline_cache.EraseChannel(channel_id);
    query_cache.EraseChannel(channel_id);
}

} // namespace Vulkan
