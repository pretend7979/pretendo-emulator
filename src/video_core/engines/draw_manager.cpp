// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>

#include <fmt/format.h>

#include "common/settings.h"
#include "core/core.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/mesh_count_provenance.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra::Engines {

namespace {

// Private, CPU-only diagnostic. No polling/formatting unless explicitly enabled and armed.
class OotDrawAttemptTrace {
public:
    static constexpr u32 MaxRecords = 65536;
    static constexpr u64 MaxFrames = 16;
    static constexpr u32 MaxCaptures = 8;
    static constexpr u32 MaxProgramSnapshots = 32;

    explicit OotDrawAttemptTrace(const char* path) : prefix{path ? path : ""} {}

    bool Enabled() const noexcept { return !prefix.empty(); }

    bool CapturingProgramSnapshots() const noexcept { return active; }

    bool WantProgramSnapshots(GPUVAddr vertex, GPUVAddr mesh, bool zero_count) {
        if (!active) return false;
        if (zero_count) return true;
        const std::array<GPUVAddr, 2> pair{vertex, mesh};
        for (u32 index = 0; index < positive_reference_pairs; ++index) {
            if (reference_pairs[index] == pair) return true;
        }
        if (positive_reference_pairs == reference_pairs.size()) return false;
        reference_pairs[positive_reference_pairs++] = pair;
        return true;
    }

    const char* ProgramSnapshotSkipReason(GPUVAddr address) const noexcept {
        if (address == 0) return "zero_address";
        for (u32 index = 0; index < program_snapshots; ++index) {
            if (snapshot_addresses[index] == address) return "duplicate";
        }
        return program_snapshots == MaxProgramSnapshots ? "program_cap" : "inactive";
    }

    bool ClaimProgramSnapshot(GPUVAddr address) {
        if (!active || address == 0 || program_snapshots == MaxProgramSnapshots) return false;
        for (u32 index = 0; index < program_snapshots; ++index) {
            if (snapshot_addresses[index] == address) return false;
        }
        snapshot_addresses[program_snapshots++] = address;
        return true;
    }

    bool Begin(u64 present) {
        if (!Enabled()) return false;
        if (active && (present < first_present || present - first_present >= MaxFrames)) {
            Finish(present, "frame_limit");
        }
        if (!active && present != last_poll && capture_index < MaxCaptures) {
            last_poll = present;
            const std::filesystem::path marker{prefix + ".now"};
            const std::filesystem::path output{
                fmt::format("{}-{:02}.jsonl", prefix, capture_index)};
            std::error_code error;
            if (std::filesystem::exists(marker, error) &&
                !std::filesystem::exists(output, error)) {
                file.clear();
                file.open(output, std::ios::out | std::ios::binary);
                if (file) {
                    std::filesystem::remove(marker, error);
                    active = true;
                    first_present = present;
                    records = 0;
                    program_snapshots = positive_reference_pairs = 0;
                    ++capture_index;
                    file << fmt::format(
                        R"({{"kind":"draw_attempt_trace","guest_present":{},"max_frames":{},"max_records":{},"source":"final_draw_gate_cpu_state","gpu_readbacks":false,"cpu_program_snapshot_reads":true,"program_snapshot_byte_cap":6144,"program_snapshot_count_cap":32,"positive_reference_pair_cap":2,"zero_mesh_root_window_offset":1904,"zero_mesh_root_window_bytes":64,"zero_mesh_root_snapshot_byte_cap":2048}})",
                        present, MaxFrames, MaxRecords) << '\n';
                }
            }
        }
        return active;
    }

    void Write(u64 present, std::string_view fields) {
        if (!active) return;
        if (present < first_present || present - first_present >= MaxFrames) {
            Finish(present, "frame_limit");
            return;
        }
        file << fmt::format(
            R"({{"kind":"draw_attempt","guest_present":{},"attempt":{},)",
            present, ++attempt_serial) << fields << "}\n";
        if (++records >= MaxRecords) Finish(present, "record_limit");
    }

private:
    void Finish(u64 present, const char* reason) {
        file << fmt::format(
            R"({{"kind":"draw_attempt_trace_end","guest_present":{},"records":{},"reason":"{}"}})",
            present, records, reason) << '\n';
        file.close();
        active = false;
    }

    std::string prefix;
    std::ofstream file;
    bool active{};
    u64 first_present{}, last_poll{~u64{}}, attempt_serial{};
    u32 records{}, capture_index{};
    u32 program_snapshots{}, positive_reference_pairs{};
    std::array<GPUVAddr, MaxProgramSnapshots> snapshot_addresses{};
    std::array<std::array<GPUVAddr, 2>, 2> reference_pairs{};
};

// CPU-only, no-flush snapshot for unique mesh/zero-count programs in an armed trace.
// Header interpretation is reported as raw fields, not used to change shader classification.
std::string OotMeshProgramSnapshot(Maxwell3D& maxwell3d, GPUVAddr address, bool zero_count,
                                   u32 slot = 1) {
    auto& memory = maxwell3d.GetMemoryManager();
    const auto& regs = maxwell3d.regs;
    const auto mapped = [&](GPUVAddr start, size_t bytes) {
        return bytes != 0 && start <= (std::numeric_limits<GPUVAddr>::max)() - (bytes - 1) &&
            memory.IsWithinGPUAddressRange(start) &&
            memory.IsWithinGPUAddressRange(start + bytes - 1) &&
            memory.IsFullyMappedRange(start, bytes);
    };
    const auto words_json = [](const auto& words) {
        std::string result;
        for (const u32 word : words) {
            if (!result.empty()) result += ',';
            result += std::to_string(word);
        }
        return result;
    };
    std::array<u32, 32> header{};
    std::array<u32, 16> prefix{};
    const bool header_valid = mapped(address, sizeof(header));
    const bool prefix_address_valid = address <= (std::numeric_limits<GPUVAddr>::max)() - 128;
    const GPUVAddr prefix_address = prefix_address_valid ? address + 128 : 0;
    const bool prefix_valid = prefix_address_valid && mapped(prefix_address, sizeof(prefix));
    const auto header_device = header_valid ? memory.GpuToCpuAddress(address) : std::nullopt;
    const auto prefix_device = prefix_valid ? memory.GpuToCpuAddress(prefix_address) : std::nullopt;
    bool header_dirty{}, prefix_dirty{};
    if (header_valid) {
        header_dirty = memory.IsMemoryDirty(address, sizeof(header),
                                            VideoCommon::CacheType::BufferCache);
        memory.ReadBlockUnsafe(address, header.data(), sizeof(header));
    }
    if (prefix_valid) {
        prefix_dirty = memory.IsMemoryDirty(prefix_address, sizeof(prefix),
                                            VideoCommon::CacheType::BufferCache);
        memory.ReadBlockUnsafe(prefix_address, prefix.data(), sizeof(prefix));
    }
    std::string fields = fmt::format(
        R"(,"{}":{{"representative":"{}","source":"guest_cpu_no_flush","slot":{},"address":"{:#x}","header_bytes":128,"native_prefix_bytes":64,"header_valid":{},"header_translated":{},"header_gpu_modified":{},"header_words":[{}],"prefix_valid":{},"prefix_translated":{},"prefix_gpu_modified":{},"native_prefix_words":[{}],"mesh_control":{},"mesh_registers":[{},{},{}],"raw_draw_begin_end_a":{},"raw_draw_begin_end_b":{})",
        slot == 1 ? "program_snapshot" : "paired_mesh_program_snapshot",
        zero_count ? "zero_count" : "nonzero_mesh_reference", slot, address,
        header_valid, header_device.has_value(), header_dirty,
        header_valid ? words_json(header) : "", prefix_valid, prefix_device.has_value(),
        prefix_dirty, prefix_valid ? words_json(prefix) : "",
        regs.reg_array[Maxwell3D::Regs::MeshControl],
        regs.reg_array[Maxwell3D::Regs::MeshShaderA],
        regs.reg_array[Maxwell3D::Regs::MeshShaderB],
        regs.reg_array[Maxwell3D::Regs::MeshInitShader],
        regs.reg_array[0x9c], regs.reg_array[0x9d]);
    if (header_valid) {
        fields += fmt::format(
            R"(,"legacy_common0_fields":{{"sph_type":{},"version":{},"shader_type":{},"kills_pixels":{},"does_global_store":{},"sass_version":{}}})",
            header[0] & 31, (header[0] >> 5) & 31, (header[0] >> 10) & 15,
            (header[0] >> 15) & 1, (header[0] >> 16) & 1, (header[0] >> 17) & 15);
    }
    fields += '}';
    return fields;
}

// A mesh-only program in slot 1 consumes stage 0 constant buffers. Retain the
// four observed root records once with its existing bounded program snapshot.
// ReadBlockUnsafe deliberately preserves CPU/no-flush provenance even if dirty.
std::string OotZeroMeshRootSnapshot(Maxwell3D& maxwell3d) {
    constexpr u32 Offset = 0x770;
    constexpr u32 Bytes = 64;
    const auto& cbuf = maxwell3d.state.shader_stages[0].const_buffers[0];
    auto& memory = maxwell3d.GetMemoryManager();
    const bool range_valid = cbuf.enabled && cbuf.size >= Offset + Bytes &&
        cbuf.address <= (std::numeric_limits<GPUVAddr>::max)() - (Offset + Bytes - 1);
    const GPUVAddr address = range_valid ? cbuf.address + Offset : 0;
    const bool mapped = range_valid && memory.IsWithinGPUAddressRange(address) &&
        memory.IsWithinGPUAddressRange(address + Bytes - 1) &&
        memory.IsFullyMappedRange(address, Bytes);
    const auto device = mapped ? memory.GpuToCpuAddress(address) : std::nullopt;
    const bool readable = mapped && device.has_value();
    bool dirty{};
    std::array<u32, Bytes / sizeof(u32)> words{};
    std::string values;
    if (readable) {
        dirty = memory.IsMemoryDirty(address, Bytes, VideoCommon::CacheType::BufferCache);
        memory.ReadBlockUnsafe(address, words.data(), Bytes);
        for (const u32 word : words) {
            if (!values.empty()) values += ',';
            values += std::to_string(word);
        }
    }
    return fmt::format(
        R"(,"zero_mesh_root_snapshot":{{"source":"guest_cpu_no_flush","stage":0,"bank":0,"cbuf_address":"{:#x}","cbuf_enabled":{},"cbuf_size":{},"byte_offset":{},"bytes":{},"address":"{:#x}","range_valid":{},"mapped":{},"translated":{},"readable":{},"gpu_modified_checked":{},"gpu_modified":{},"words":[{}]}})",
        cbuf.address, cbuf.enabled, cbuf.size, Offset, Bytes, address, range_valid,
        mapped, device.has_value(), readable, readable, dirty, values);
}

void TraceOotDrawAttempt(Maxwell3D& maxwell3d, bool is_indirect, bool is_indexed,
                         u32 requested_instances, bool host_execute) {
    static OotDrawAttemptTrace trace{std::getenv("EDEN_NX2_OOT_DRAW_TRACE")};
    if (!trace.Enabled() || Settings::GetCurrentProgramID() != 0x0400486024DEC000ULL) {
        return;
    }
    const u64 present = Core::g_nx2_present_count.load(std::memory_order_relaxed);
    if (!trace.Begin(present)) return;

    const auto& regs = maxwell3d.regs;
    const bool native_turing = maxwell3d.GetNativeMacroEngine().native_turing;
    const auto& draw = maxwell3d.draw_manager.draw_state;
    const auto& indirect = maxwell3d.draw_manager.indirect_state;
    const auto shader_address = [&](size_t index) {
        if (!native_turing) return regs.program_region.Address() + regs.pipelines[index].offset;
        return (u64{regs.pipelines[index].reserved[0] & 0xff} << 32) |
               regs.pipelines[index].reserved[1];
    };
    std::string fields = fmt::format(
        R"("indirect":{},"indexed":{},"native_turing":{},"host_execute":{},"host_gate_result":"{}","gpu_predicate":"not_read","guest_vs":"{:#x}","guest_ps":"{:#x}","mesh_enabled":{},"topology":{},"draw_mode":{},"element_count":{},"element_count_source":"{}","requested_instances":{},"instance_count_source":"{}","pending_instances":{},"drawn_instances":{},"vertex_first":{},"vertex_count":{},"index_first":{},"index_count":{},"base_index_bits":{},"base_instance":{})",
        is_indirect, is_indexed, native_turing, host_execute,
        host_execute ? "passes_host_gate" : "skipped_by_host_gate",
        shader_address(1), shader_address(5), regs.IsMeshEnabled(),
        static_cast<u32>(draw.topology), static_cast<u32>(draw.draw_mode),
        draw.draw_element_count, is_indirect ? "gpu_indirect_not_read" : "draw_state",
        is_indirect ? "null" : std::to_string(requested_instances),
        is_indirect ? "gpu_indirect_not_read" : "ProcessDraw_argument",
        draw.instance_count, draw.drawn_instances, draw.vertex_buffer.first,
        draw.vertex_buffer.count, draw.index_buffer.first, draw.index_buffer.count,
        draw.base_index, draw.base_instance);
    fields += fmt::format(
        R"(,"guest_rt0":"{:#x}","guest_rt_count":{},"guest_rt0_format":{},"guest_rt0_extent":[{},{}],"guest_depth":"{:#x}","guest_depth_format":{},"zeta_enable":{},"color_mask_common":{},"color_mask0":{},"blend0":{},"depth_test":{},"depth_write":{},"depth_func":{},"rasterize_enable":{},"cull_enable":{},"cull_face":{},"front_face":{},"render_enable_mode":{},"render_enable_override":{},"predicate_address":"{:#x}","alpha_test_enabled":{},"alpha_test_func":{},"alpha_test_ref_bits":{},"alpha_to_coverage":{},"alpha_to_one":{})",
        regs.rt[0].Address(), regs.rt_control.count.Value(), static_cast<u32>(regs.rt[0].format),
        regs.rt[0].width, regs.rt[0].height, regs.zeta.Address(),
        static_cast<u32>(regs.zeta.format), regs.zeta_enable, regs.color_mask_common,
        regs.color_mask[0].raw, regs.blend.enable[0], regs.depth_test_enable,
        regs.depth_write_enabled, static_cast<u32>(regs.depth_test_func), regs.rasterize_enable,
        regs.gl_cull_test_enabled, static_cast<u32>(regs.gl_cull_face),
        static_cast<u32>(regs.gl_front_face), static_cast<u32>(regs.render_enable.mode),
        static_cast<u32>(regs.render_enable_override), regs.render_enable.Address(),
        regs.alpha_test_enabled, static_cast<u32>(regs.alpha_test_func),
        std::bit_cast<u32>(regs.alpha_test_ref),
        regs.anti_alias_alpha_control.alpha_to_coverage.Value(),
        regs.anti_alias_alpha_control.alpha_to_one.Value());
    fields += R"(,"shader_slots":[)";
    for (size_t index = 0; index < 6; ++index) {
        if (index) fields += ',';
        fields += fmt::format(R"({{"slot":{},"enabled":{},"address":"{:#x}"}})",
            index, regs.IsShaderConfigEnabled(index), shader_address(index));
    }
    fields += R"(],"color_masks":[)";
    for (size_t index = 0; index < regs.color_mask.size(); ++index) {
        if (index) fields += ',';
        fields += std::to_string(regs.color_mask[index].raw);
    }
    fields += ']';
    if (is_indirect) {
        fields += fmt::format(
            R"(,"indirect_params":{{"is_byte_count":{},"include_count":{},"count_address":"{:#x}","argument_address":"{:#x}","buffer_size":{},"max_draw_count":{},"stride":{},"argument_words":"not_read"}})",
            indirect.is_byte_count, indirect.include_count, indirect.count_start_address,
            indirect.indirect_start_address, indirect.buffer_size, indirect.max_draw_counts,
            indirect.stride);
    }
    if (trace.CapturingProgramSnapshots() && native_turing && !is_indirect &&
        regs.IsShaderConfigEnabled(1) &&
        (draw.draw_element_count == 0 || regs.IsMeshEnabled())) {
        const bool zero_count = draw.draw_element_count == 0;
        const bool paired_mesh = regs.IsMeshEnabled() && regs.IsShaderConfigEnabled(3);
        const bool wanted = trace.WantProgramSnapshots(shader_address(1),
            paired_mesh ? shader_address(3) : 0, zero_count);
        const char* vertex_status = "reference_cap";
        const char* mesh_status = paired_mesh ? "reference_cap" : "not_bound";
        if (wanted) {
            const auto snapshot = [&](u32 slot) {
                const GPUVAddr address = shader_address(slot);
                if (!trace.ClaimProgramSnapshot(address)) {
                    return trace.ProgramSnapshotSkipReason(address);
                }
                fields += OotMeshProgramSnapshot(maxwell3d, address, zero_count, slot);
                if (slot == 1 && zero_count && regs.IsMeshEnabled() && !paired_mesh) {
                    fields += OotZeroMeshRootSnapshot(maxwell3d);
                }
                return "captured";
            };
            vertex_status = snapshot(1);
            if (paired_mesh) mesh_status = snapshot(3);
        }
        fields += fmt::format(
            R"(,"program_snapshot_status":{{"slot1":"{}","slot3":"{}"}})",
            vertex_status, mesh_status);
    }
    trace.Write(present, fields);
}

void TraceOotMeshCount(Maxwell3D& maxwell3d, bool indexed, u32 instances, bool host_execute) {
    auto& trace = MeshCountProvenance::Get();
    if (!trace.Enabled()) return;
    const auto& regs = maxwell3d.regs;
    const u64 present = Core::g_nx2_present_count.load(std::memory_order_relaxed);
    if (!trace.BeginDraw(present, Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL,
                         maxwell3d.GetNativeMacroEngine().native_turing, regs.IsMeshEnabled())) return;
    static_assert(MAXWELL3D_REG_INDEX(vertex_buffer.count) == 0x35e);
    static_assert(MAXWELL3D_REG_INDEX(vertex_buffer.first) == 0x35d);
    static_assert(MAXWELL3D_REG_INDEX(draw.begin) == 0x586);
    static_assert(MAXWELL3D_REG_INDEX(draw.end) == 0x585);
    static_assert(MAXWELL3D_REG_INDEX(index_buffer.first) == 0x5f7);
    static_assert(MAXWELL3D_REG_INDEX(index_buffer.count) == 0x5f8);
    const auto& draw = maxwell3d.draw_manager.draw_state;
    const auto shader = [&](u32 slot) {
        return (u64{regs.pipelines[slot].reserved[0] & 0xff} << 32) |
               regs.pipelines[slot].reserved[1];
    };
    trace.Draw(&maxwell3d, present, draw.vertex_buffer.count, fmt::format(
        R"("indexed":{},"host_execute":{},"instances":{},"element_count":{},"vertex_count":{},"index_count":{},"vertex_first":{},"base_instance":{},"guest_vs":"{:#x}","guest_ps":"{:#x}","mesh_control":{},"mesh_a":{},"mesh_b":{},"mesh_init":{},"modern_start_register":{},"modern_count_register":{},"legacy_count_register":{})",
        indexed, host_execute, instances, draw.draw_element_count, draw.vertex_buffer.count,
        draw.index_buffer.count, draw.vertex_buffer.first, draw.base_instance, shader(1), shader(5),
        regs.reg_array[Maxwell3D::Regs::MeshControl], regs.reg_array[Maxwell3D::Regs::MeshShaderA],
        regs.reg_array[Maxwell3D::Regs::MeshShaderB], regs.reg_array[Maxwell3D::Regs::MeshInitShader],
        regs.reg_array[0x9c], regs.reg_array[0x9d], regs.vertex_buffer.count));
}

} // Anonymous namespace


void Maxwell3D::DrawManager::ProcessMethodCall(Maxwell3D& maxwell3d, u32 method, u32 argument) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(clear_surface):
        return Clear(maxwell3d, 1);
    case MAXWELL3D_REG_INDEX(draw.begin):
        return DrawBegin(maxwell3d);
    case MAXWELL3D_REG_INDEX(draw.end):
        return DrawEnd(maxwell3d);
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
        break;
    case MAXWELL3D_REG_INDEX(index_buffer.count):
        draw_state.draw_indexed = true;
        break;
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
        draw_state.instance_count++;
        [[fallthrough]];
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
        return DrawIndexSmall(maxwell3d, argument);
    case MAXWELL3D_REG_INDEX(draw_inline_index):
        SetInlineIndexBuffer(maxwell3d, argument);
        break;
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_2x16.even);
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_2x16.odd);
        break;
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_4x8.index0);
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_4x8.index1);
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_4x8.index2);
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_4x8.index3);
        break;
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
        DrawArrayInstanced(maxwell3d, maxwell3d.regs.vertex_array_instance_first.topology.Value(), maxwell3d.regs.vertex_array_instance_first.start.Value(), maxwell3d.regs.vertex_array_instance_first.count.Value(), false);
        break;
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent): {
        DrawArrayInstanced(maxwell3d, maxwell3d.regs.vertex_array_instance_subsequent.topology.Value(), maxwell3d.regs.vertex_array_instance_subsequent.start.Value(), maxwell3d.regs.vertex_array_instance_subsequent.count.Value(), true);
        break;
    }
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0): {
        DrawTexture(maxwell3d);
        break;
    }
    default:
        break;
    }
}

void Maxwell3D::DrawManager::Clear(Maxwell3D& maxwell3d, u32 layer_count) {
    if (maxwell3d.ShouldExecute()) {
        maxwell3d.rasterizer->Clear(layer_count);
    }
}

void Maxwell3D::DrawManager::DrawDeferred(Maxwell3D& maxwell3d, bool loop_continues) {
    if (draw_state.draw_mode != DrawMode::Instance || draw_state.instance_count == 0) {
        if (!loop_continues) {
            draw_state.drawn_instances = 0;
        }
        return;
    }
    // The first draw of a batch also covers the loop's first instance; a batch that continues a
    // loop split by a register write draws only the instances after the ones already drawn.
    const bool continued = draw_state.drawn_instances != 0;
    DrawEnd(maxwell3d, draw_state.instance_count + (continued ? 0 : 1), true);
    draw_state.drawn_instances =
        loop_continues ? draw_state.drawn_instances + draw_state.instance_count : 0;
    draw_state.instance_count = 0;
}

bool Maxwell3D::DrawManager::IsInstanceLoopMethod(u32 method) noexcept {
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.count):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.count):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::DrawManager::DrawArray(Maxwell3D& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology, u32 vertex_first, u32 vertex_count, u32 base_instance, u32 num_instances) {
    draw_state.topology = topology;
    draw_state.vertex_buffer.first = vertex_first;
    draw_state.vertex_buffer.count = vertex_count;
    draw_state.base_instance = base_instance;
    ProcessDraw(maxwell3d, false, num_instances);
}

void Maxwell3D::DrawManager::DrawArrayInstanced(Maxwell3D& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology, u32 vertex_first, u32 vertex_count, bool subsequent) {
    draw_state.topology = topology;
    draw_state.vertex_buffer.first = vertex_first;
    draw_state.vertex_buffer.count = vertex_count;
    if (!subsequent) {
        draw_state.instance_count = 1;
        draw_state.drawn_instances = 0;
    }
    draw_state.base_instance = draw_state.instance_count - 1;
    draw_state.draw_mode = DrawMode::Instance;
    draw_state.instance_count++;
    ProcessDraw(maxwell3d, false, 1);
}

void Maxwell3D::DrawManager::DrawIndex(Maxwell3D& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology, u32 index_first, u32 index_count, u32 base_index, u32 base_instance, u32 num_instances) {
    draw_state.topology = topology;
    draw_state.index_buffer = maxwell3d.regs.index_buffer;
    draw_state.index_buffer.first = index_first;
    draw_state.index_buffer.count = index_count;
    draw_state.base_index = base_index;
    draw_state.base_instance = base_instance;
    ProcessDraw(maxwell3d, true, num_instances);
}

void Maxwell3D::DrawManager::DrawArrayIndirect(Maxwell3D& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology) {
    draw_state.topology = topology;
    ProcessDrawIndirect(maxwell3d);
}

void Maxwell3D::DrawManager::DrawIndexedIndirect(Maxwell3D& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology, u32 index_first, u32 index_count) {
    draw_state.topology = topology;
    draw_state.index_buffer = maxwell3d.regs.index_buffer;
    draw_state.index_buffer.first = index_first;
    draw_state.index_buffer.count = index_count;
    ProcessDrawIndirect(maxwell3d);
}

void Maxwell3D::DrawManager::SetInlineIndexBuffer(Maxwell3D& maxwell3d, u32 index) {
    draw_state.inline_index_draw_indexes.push_back(u8(index & 0x000000ff));
    draw_state.inline_index_draw_indexes.push_back(u8((index & 0x0000ff00) >> 8));
    draw_state.inline_index_draw_indexes.push_back(u8((index & 0x00ff0000) >> 16));
    draw_state.inline_index_draw_indexes.push_back(u8((index & 0xff000000) >> 24));
    draw_state.draw_mode = DrawMode::InlineIndex;
}

void Maxwell3D::DrawManager::SetInlineIndexBuffer(Maxwell3D& maxwell3d, u32 method,
                                                  const u32* base_start, u32 amount) {
    auto& index_buffer = draw_state.inline_index_draw_indexes;
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw_inline_index): {
        const auto* const bytes = reinterpret_cast<const u8*>(base_start);
        index_buffer.insert(index_buffer.end(), bytes, bytes + size_t(amount) * sizeof(u32));
        break;
    }
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even): {
        const size_t offset = index_buffer.size();
        index_buffer.resize(offset + size_t(amount) * 2 * sizeof(u32));
        u8* dst = index_buffer.data() + offset;
        for (u32 i = 0; i < amount; ++i) {
            const u32 word = base_start[i];
            const u32 indexes[2]{word & 0xFFFF, word >> 16};
            std::memcpy(dst, indexes, sizeof(indexes));
            dst += sizeof(indexes);
        }
        break;
    }
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0): {
        const size_t offset = index_buffer.size();
        index_buffer.resize(offset + size_t(amount) * 4 * sizeof(u32));
        u8* dst = index_buffer.data() + offset;
        for (u32 i = 0; i < amount; ++i) {
            const u32 word = base_start[i];
            const u32 indexes[4]{word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF,
                                 word >> 24};
            std::memcpy(dst, indexes, sizeof(indexes));
            dst += sizeof(indexes);
        }
        break;
    }
    }
    draw_state.draw_mode = DrawMode::InlineIndex;
}

void Maxwell3D::DrawManager::DrawBegin(Maxwell3D& maxwell3d) {
    auto reset_instance_count = maxwell3d.regs.draw.instance_id == Maxwell3D::Regs::Draw::InstanceId::First;
    auto increment_instance_count = maxwell3d.regs.draw.instance_id == Maxwell3D::Regs::Draw::InstanceId::Subsequent;
    if (reset_instance_count) {
        DrawDeferred(maxwell3d);
        draw_state.instance_count = 0;
        draw_state.draw_mode = DrawMode::General;
    } else if (increment_instance_count) {
        draw_state.instance_count++;
        draw_state.draw_mode = DrawMode::Instance;
    }
    draw_state.topology = maxwell3d.regs.draw.topology;
}

void Maxwell3D::DrawManager::DrawEnd(Maxwell3D& maxwell3d, u32 instance_count, bool force_draw) {
    switch (draw_state.draw_mode) {
    case DrawMode::Instance:
        if (!force_draw) {
            break;
        }
        [[fallthrough]];
    case DrawMode::General:
        draw_state.base_instance = maxwell3d.regs.global_base_instance_index;
        if (draw_state.draw_mode == DrawMode::Instance && draw_state.drawn_instances != 0) {
            draw_state.base_instance += draw_state.drawn_instances + 1;
        }
        draw_state.base_index = maxwell3d.regs.global_base_vertex_index;
        if (draw_state.draw_indexed) {
            draw_state.index_buffer = maxwell3d.regs.index_buffer;
            ProcessDraw(maxwell3d, true, instance_count);
        } else {
            draw_state.vertex_buffer = maxwell3d.regs.vertex_buffer;
            ProcessDraw(maxwell3d, false, instance_count);
        }
        draw_state.draw_indexed = false;
        break;
    case DrawMode::InlineIndex:
        draw_state.base_instance = maxwell3d.regs.global_base_instance_index;
        draw_state.base_index = maxwell3d.regs.global_base_vertex_index;
        draw_state.index_buffer = maxwell3d.regs.index_buffer;
        draw_state.index_buffer.count = u32(draw_state.inline_index_draw_indexes.size() / 4);
        draw_state.index_buffer.format = Maxwell3D::Regs::IndexFormat::UnsignedInt;
        maxwell3d.dirty.flags[VideoCommon::Dirty::IndexBuffer] = true;
        ProcessDraw(maxwell3d, true, instance_count);
        draw_state.inline_index_draw_indexes.clear();
        break;
    }
}

void Maxwell3D::DrawManager::DrawIndexSmall(Maxwell3D& maxwell3d, u32 argument) {
    Maxwell3D::Regs::IndexBufferSmall index_small_params{argument};
    draw_state.base_instance = maxwell3d.regs.global_base_instance_index;
    draw_state.base_index = maxwell3d.regs.global_base_vertex_index;
    draw_state.index_buffer = maxwell3d.regs.index_buffer;
    draw_state.index_buffer.first = index_small_params.first;
    draw_state.index_buffer.count = index_small_params.count;
    draw_state.topology = index_small_params.topology;
    maxwell3d.dirty.flags[VideoCommon::Dirty::IndexBuffer] = true;
    ProcessDraw(maxwell3d, true, 1);
}

void Maxwell3D::DrawManager::DrawTexture(Maxwell3D& maxwell3d) {
    draw_texture_state.dst_x0 = f32(maxwell3d.regs.draw_texture.dst_x0) / 4096.f;
    draw_texture_state.dst_y0 = f32(maxwell3d.regs.draw_texture.dst_y0) / 4096.f;
    const auto dst_width = f32(maxwell3d.regs.draw_texture.dst_width) / 4096.f;
    const auto dst_height = f32(maxwell3d.regs.draw_texture.dst_height) / 4096.f;
    const bool lower_left{maxwell3d.regs.window_origin.mode != Maxwell3D::Regs::WindowOrigin::Mode::UpperLeft};
    if (lower_left) {
        draw_texture_state.dst_y0 = f32(maxwell3d.regs.surface_clip.height) - draw_texture_state.dst_y0;
    }
    draw_texture_state.dst_x1 = draw_texture_state.dst_x0 + dst_width;
    draw_texture_state.dst_y1 = draw_texture_state.dst_y0 + dst_height;
    draw_texture_state.src_x0 = f32(maxwell3d.regs.draw_texture.src_x0) / 4096.f;
    draw_texture_state.src_y0 = f32(maxwell3d.regs.draw_texture.src_y0) / 4096.f;
    draw_texture_state.src_x1 = (f32(maxwell3d.regs.draw_texture.dx_du) / 4294967296.f) * dst_width + draw_texture_state.src_x0;
    draw_texture_state.src_y1 = (f32(maxwell3d.regs.draw_texture.dy_dv) / 4294967296.f) * dst_height + draw_texture_state.src_y0;
    draw_texture_state.src_sampler = maxwell3d.regs.draw_texture.src_sampler;
    draw_texture_state.src_texture = maxwell3d.regs.draw_texture.src_texture;
    maxwell3d.rasterizer->DrawTexture();
}

void Maxwell3D::DrawManager::UpdateTopology(Maxwell3D& maxwell3d) {
    switch (maxwell3d.regs.primitive_topology_control) {
    case Maxwell3D::Regs::PrimitiveTopologyControl::UseInBeginMethods:
        break;
    case Maxwell3D::Regs::PrimitiveTopologyControl::UseSeparateState:
        switch (maxwell3d.regs.topology_override) {
        case Maxwell3D::Regs::PrimitiveTopologyOverride::None:
            break;
        case Maxwell3D::Regs::PrimitiveTopologyOverride::Points:
            draw_state.topology = Maxwell3D::Regs::PrimitiveTopology::Points;
            break;
        case Maxwell3D::Regs::PrimitiveTopologyOverride::Lines:
            draw_state.topology = Maxwell3D::Regs::PrimitiveTopology::Lines;
            break;
        case Maxwell3D::Regs::PrimitiveTopologyOverride::LineStrip:
            draw_state.topology = Maxwell3D::Regs::PrimitiveTopology::LineStrip;
            break;
        default:
            draw_state.topology = Maxwell3D::Regs::PrimitiveTopology(maxwell3d.regs.topology_override);
            break;
        }
        break;
    }
}

void Maxwell3D::DrawManager::ProcessDraw(Maxwell3D& maxwell3d, bool draw_indexed, u32 instance_count) {
    draw_state.draw_element_count =
        draw_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count;
    LOG_TRACE(HW_GPU, "called, topology={}, count={}", draw_state.topology, draw_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count);
    UpdateTopology(maxwell3d);
    const bool host_execute = maxwell3d.ShouldExecute();
    TraceOotDrawAttempt(maxwell3d, false, draw_indexed, instance_count, host_execute);
    TraceOotMeshCount(maxwell3d, draw_indexed, instance_count, host_execute);
    if (host_execute) {
        maxwell3d.rasterizer->Draw(draw_indexed, instance_count);
    }
}

void Maxwell3D::DrawManager::ProcessDrawIndirect(Maxwell3D& maxwell3d) {
    draw_state.draw_element_count = 0;
    LOG_TRACE(HW_GPU, "called, topology={}, is_indexed={}, includes_count={}, buffer_size={}, max_draw_count={}", draw_state.topology, indirect_state.is_indexed, indirect_state.include_count, indirect_state.buffer_size, indirect_state.max_draw_counts);
    UpdateTopology(maxwell3d);
    const bool host_execute = maxwell3d.ShouldExecute();
    TraceOotDrawAttempt(maxwell3d, true, indirect_state.is_indexed, 0, host_execute);
    if (host_execute) {
        maxwell3d.rasterizer->DrawIndirect();
    }
}
} // namespace Tegra::Engines
