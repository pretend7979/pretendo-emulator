// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class ImageView;
class MemoryAllocator;
class Scheduler;

/// EDEN_NX2_FRAME_DUMP=<dir> reads back one whole frame: the color targets of every pass, the
/// presented image. Create <dir>/capture.now after reaching the scene to capture once.
/// RT/present RAW readbacks are bounded to 128 MiB; optional sampled inputs have a separate
/// 32 MiB cap. EDEN_NX2_FRAME_DUMP_SAMPLED=1 captures color/D32 inputs before each draw only in
/// the shader-anchored selected pass. Conversion and image previews happen offline.
class FrameDump {
public:
    explicit FrameDump(const Device& device, MemoryAllocator& memory_allocator, Scheduler& scheduler);
    ~FrameDump();

    [[nodiscard]] bool Capturing() const noexcept {
        return capturing;
    }

    struct Sampled {
        u32 stage;
        u32 tic;
        u32 resolved_tic;
        u32 tsc;
        ImageView* view;
        std::array<u32, 8> tic_words{};
        std::array<u32, 8> tsc_words{};
        bool tic_valid{};
        bool tsc_valid{};
    };

    [[nodiscard]] bool CapturingSampled() const noexcept;
    /// Copies inputs before the current draw. True means the render pass must be resumed.
    bool NoteSampled(std::span<const Sampled> textures);

    /// Starts a pass when the framebuffer changes and captures the previous one.
    /// Returns true when copies were recorded, which ends the current render pass.
    bool BeginPass(const void* framebuffer, std::span<ImageView* const> colors,
                   ImageView* depth = nullptr);
    /// Explicit-address, D32-only depth observation for bounded fog-anchored bursts.
    [[nodiscard]] bool CapturingFogDepth() const noexcept;
    /// Captures the preceding draw in the selected pass before another draw is configured.
    /// True means copies ended the render pass, which the caller must resume.
    bool BeforeDraw();
    void NoteDraw(std::span<const u64> shader_hashes, u64 guest_vs = 0, u64 guest_ps = 0);
    /// Bounded CPU-side metadata for selected draws; gathering it must not flush GPU work.
    [[nodiscard]] bool CapturingMetadata() const noexcept;
    void NoteDrawMetadata(std::string_view fields);
    /// Opt-in CPU-only command/state records, capped separately from heavy draw metadata.
    [[nodiscard]] bool CapturingDrawSubmissions() const noexcept;
    void NoteDrawSubmission(std::string_view fields, bool draw_noted);
    struct ComputeInput {
        u32 texture_index;
        u32 tic;
        u32 cbuf_index;
        u32 cbuf_offset;
        ImageView* view;
    };
    struct ComputeBufferInput {
        u32 binding_index{};
        u32 cbuf_index{};
        u32 cbuf_offset{};
        VkBuffer buffer{};
        u64 source_capacity{};
        u64 offset{};
        u32 size{};
        GPUVAddr gpu_addr{};
        GPUVAddr pointer{};
        bool pointer_valid{};
        bool is_written{};
        u32 pointer_table_offset{~0U};
        u32 total_bindings{};
        bool binding_selection_truncated{};
        bool capture_requested{true};
        u64 device_addr{};
        u32 segment_count{};
        const char* skip_reason{};
        bool uniform_buffer{};
        bool descriptor_checked{};
        bool descriptor_matches{};
        bool descriptor_address_mode{};
        u32 descriptor_entry{};
        u32 shader_binding{};
        VkBuffer descriptor_buffer{};
        u64 descriptor_offset{};
        u64 descriptor_range{};
        u64 descriptor_address{};
    };
    struct TaskBufferInput {
        ComputeBufferInput binding;
        u32 stage{};
        u64 shader_hash{};
        u32 pointer_offset{~0U};
    };
    [[nodiscard]] bool CapturingTaskInputs() const noexcept;
    [[nodiscard]] bool CapturingMaterialInputs() const noexcept;
    [[nodiscard]] bool CapturingCanopyInputs() const noexcept;
    [[nodiscard]] static std::array<u64, 2> CanopyInputPair(u32 variant) noexcept;
    [[nodiscard]] std::array<u64, 2> CanopyInputShaders() const noexcept;
    [[nodiscard]] bool CanopyInputIsMesh() const noexcept;
    [[nodiscard]] bool TaskInputsRecorded() const noexcept { return task_inputs_recorded; }
    /// First supported task/mesh pair after the selected anchor; true means copies ended the pass.
    bool NoteTaskInputs(u64 task_hash, u64 mesh_hash, std::string_view fields,
                        std::span<const TaskBufferInput> buffers);
    /// Private bounded sky readbacks: one immediate dispatch or two after an explicit trigger.
    [[nodiscard]] static bool ComputeCaptureStopEnabled() noexcept;
    [[nodiscard]] static bool ComputeCaptureSkyEnabled() noexcept;
    [[nodiscard]] static bool DelayedSkyCaptureEnabled() noexcept;
    [[nodiscard]] bool CompleteDelayedSkyCapture();
    [[nodiscard]] static u64 ComputeCaptureStopNativeKey() noexcept;
    [[nodiscard]] bool ReadyForSkyCaptureDispatch() const noexcept;
    bool BeginComputeCaptureStop(u64 native_key);
    [[nodiscard]] bool CapturingComputeCaptureStop() const noexcept { return compute_capture_stop; }
    void NoteCaptureStopMetadata(std::string_view fields);
    void FinishComputeCaptureStop(bool kernel_dispatched = false);
    /// Capture at most four validated UBOs (64 KiB each), in stop or selected live producer mode.
    void NoteCaptureStopUniformBuffer(ComputeBufferInput input);
    [[nodiscard]] static u64 ProducerNativeKey() noexcept;
    [[nodiscard]] bool ProducerInputsRecorded() const noexcept { return producer_inputs_recorded; }
    [[nodiscard]] bool CapturingProducerInputs() const noexcept;
    [[nodiscard]] bool CapturingParticleInputs() const noexcept;
    [[nodiscard]] static bool IsParticleProducerPair(u64 native_key, u64 canonical_key) noexcept;
    bool SelectProducerInputDispatch(u64 native_key);
    /// First selected producer group only, base mip/all bounded layers; true begins this group.
    bool NoteProducerInputs(u64 native_key, std::span<const ComputeInput> inputs,
                            std::span<const ComputeBufferInput> buffers = {});
    /// One repeat-triggered OOT frame; only existing bounded fog producer/input hooks.
    [[nodiscard]] bool CapturingFogInputs() const noexcept { return capturing && fog_inputs_capture; }
    [[nodiscard]] bool CapturingLightingInputs() const noexcept {
        return CapturingFogInputs() && ProducerNativeKey() == 0xd9b1657f3a0b4bc2ULL;
    }
    [[nodiscard]] bool CapturingShInputs() const noexcept {
        return CapturingFogInputs() && ProducerNativeKey() == 0x203053e20b8c989aULL;
    }
    /// Explicit, persistent-until-off A/B control for the captured OOT D9 scene only.
    [[nodiscard]] u32 LightingRangeMinimum(u64 native_key, u64 canonical_key, u32 cbuf,
        u32 cbuf_offset, u32 pointer_offset, bool is_written) const noexcept;
    [[nodiscard]] std::optional<bool> NativeDomainRangeOverride() const noexcept {
        return native_domain_range;
    }
    [[nodiscard]] bool NativeDomainCorrectionEnabled() const noexcept;
    [[nodiscard]] bool NativeDomainRangeEnabled(u64 native_key, u64 canonical_key, u32 cbuf,
        u32 cbuf_offset, u32 pointer_offset, bool is_written) const noexcept;
    /// Independent, persistent graphics experiments; disabled for other titles and stop probes.
    [[nodiscard]] bool GraphicsIndirectRangeTrial() const noexcept;
    [[nodiscard]] bool GraphicsNativePointerTrial() const noexcept;
    void NoteLightingOutput(const ComputeInput& output, bool descriptor_matches);
    void NoteLightingOutputsAfter(u64 native_key);
    /// Images a compute shader writes (EDEN_NX2_FRAME_DUMP_WRITTEN_SHADER, by default OoT's fog
    /// volume 780c0536b89fc89e), noted before its dispatch and read back after it.
    void NoteWrittenImage(u64 shader_hash, ImageView& view, u32 slot);
    void NoteWrittenImagesAfter();
    [[nodiscard]] bool CapturingComputeInputs() const noexcept;
    /// Copies at most three ec68 sampled volumes before dispatch, within existing sampled budget.
    bool NoteComputeInputs(u64 native_key, std::span<const ComputeInput> inputs);
    [[nodiscard]] bool CapturingComputeMetadata() const noexcept;
    void NoteComputeMetadata(std::string_view fields);
    void NoteOperation(const char* operation);
    /// Captures the current pass; returns true when copies were recorded.
    bool EndPass();
    void CapturePresent(ImageView& view);
    /// Waits for pending readbacks and writes them once they pass the staging budget.
    void DrainIfFull();
    void TickFrame();

private:
    struct Target {
        u32 slot;
        VkImage image;
        VideoCore::Surface::PixelFormat format;
        u32 width;
        u32 height;
        u32 depth;
        s32 level;
        s32 layer;
        GPUVAddr gpu_addr;
        bool rescaled;
        bool msaa;
        VkImageAspectFlags aspect;
        u32 block_width{1};
        u32 block_height{1};
    };

    struct Readback {
        vk::Buffer buffer;
        Target target;
        size_t size_bytes;
        std::string name;
        std::string tags;
        bool raw_buffer{};
    };

    std::optional<Target> MakeTarget(ImageView& view, u32 slot, bool sampled = false,
                                     bool producer = false) const;
    bool Record(const Target& target, std::string name, std::string tags, bool sampled = false,
                bool producer = false);
    bool RecordProducerBuffer(u64 native_key, const ComputeBufferInput& input);
    bool RecordTaskBuffer(const TaskBufferInput& input);
    bool CaptureLastDraw();
    bool CaptureFogDepth();
    void Drain();
    void PollDelayedSkyCapture();
    bool ConsumeLightingRangeTrigger();
    bool ConsumeGraphicsBufferTrigger();
    [[nodiscard]] size_t ProducerSizeCap() const noexcept;
    void Write(const Readback& readback, std::span<const u8> data);
    void WriteFile(const std::string& name, std::span<const u8> data);

    const Device& device;
    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;

    u32 delayed_sky_poll_frames{};
    u32 delayed_sky_completed{};
    bool compute_capture_stop{};
    bool compute_capture_stop_consumed{};
    u32 capture_stop_uniform_count{};
    u32 capture_stop_uniform_mask{};
    u32 capture_stop_sampled_count{};
    bool capture_stop_post_dispatch{};
    std::vector<ComputeBufferInput> capture_stop_outputs;
    bool enabled{};
    bool capturing{};
    bool done{};
    bool truncated{};
    std::filesystem::path directory;
    std::filesystem::path trigger_file;
    std::chrono::steady_clock::time_point start;
    u64 frame{};
    u32 burst_remaining{};
    u32 burst_index{};

    std::ofstream manifest;
    std::vector<Readback> readbacks;
    size_t pending_bytes{};
    size_t written_bytes{};
    size_t sampled_bytes{};
    size_t producer_bytes{};
    size_t producer_buffer_bytes{};
    size_t task_buffer_bytes{};
    u32 task_input_records{};
    bool task_inputs_recorded{};
    bool material_inputs_capture{};
    bool canopy_inputs_capture{};
    u32 canopy_input_variant{};
    u32 material_input_skip{};
    u32 material_anchor_matches{};
    u32 sequence{};
    u32 images{};
    u32 metadata_records{};
    u32 compute_metadata_records{};
    std::vector<Target> lighting_output_targets;
    bool lighting_outputs_finished{};
    std::vector<Target> written_targets;
    u64 written_shader{};
    u32 written_dispatches{};
    bool fog_inputs_capture{};
    bool graphics_indirect_range_trial{};
    bool graphics_native_pointer_trial{};
    bool lighting_range_diagnostic{};
    bool sh_range_diagnostic{};
    std::optional<bool> native_domain_range;
    u32 lighting_range_minimum24{35191040};
    u32 lighting_range_minimum32{121642624};
    bool compute_inputs_recorded{};
    bool producer_inputs_recorded{};
    u32 producer_input_skip{};
    u32 producer_matching_dispatches{};
    u32 producer_selected_ordinal{~0U};
    u32 draw_submission_records{};

    const void* pass_framebuffer{};
    std::vector<Target> pass_targets;
    std::optional<Target> fog_depth_target;
    bool fog_depth_recorded{};
    std::vector<u64> pass_hashes;
    u32 pass_index{};
    u32 pass_first_draw{};
    u32 draws{};
};

} // namespace Vulkan
