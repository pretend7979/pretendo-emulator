// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// AMD FidelityFX FSR 3 frame generation (SDK v1.1.4): the SDK's optical flow and frame
/// interpolation passes, run on the presented frames. Without game motion vectors the frame
/// interpolation falls back on its optical flow wherever the game vectors do not match.
class AmdFsr3FrameGen {
public:
    explicit AmdFsr3FrameGen(const Device& device, Scheduler& scheduler,
                             MemoryAllocator& memory_allocator, VkExtent2D extent, VkFormat format);
    ~AmdFsr3FrameGen();

    static bool IsSupported(const Device& device, VkFormat format);

    /// Takes the newly presented frame as the current interpolation source.
    void Process(VkImage current_frame);

    /// Writes the frame halfway between the previous and the current one.
    void Generate(VkImage destination);

    [[nodiscard]] bool IsReady() const {
        return frames_processed >= 2;
    }

private:
    class Pass;
    struct Texture {
        vk::Image image;
        vk::ImageView view;
        std::vector<vk::ImageView> levels;
    };
    struct Resource {
        VkImageView view{};
        VkBuffer buffer{};
        VkDeviceSize offset{};
        VkDeviceSize size{};
    };

    Texture CreateTexture(VkFormat format, u32 width, u32 height, u32 levels = 1,
                          VkImageUsageFlags extra_usage = 0);
    Resource Uniform(const void* data, size_t size);
    void RunOpticalFlow(bool reset);

    const Device& device;
    Scheduler& scheduler;
    MemoryAllocator& allocator;
    VkExtent2D display;
    VkExtent2D render;
    VkFormat format;

    DescriptorPool descriptor_pool;
    ComputePassDescriptorQueue descriptor_queue;
    vk::Sampler linear_sampler;
    VkSampler linear_sampler_handle{};
    std::vector<std::unique_ptr<Pass>> passes;

    vk::Buffer uniform_buffer;
    VkDeviceSize uniform_cursor{};
    VkDeviceSize uniform_alignment{};

    // Interpolation sources and output (display size, presented format).
    Texture previous_source;
    Texture current_source;
    Texture output;

    // Frame interpolation (render size unless noted).
    Texture input_motion_vectors;
    Texture input_depth;
    Texture dilated_motion_vectors;
    Texture dilated_depth;
    Texture reconstructed_previous_depth;
    Texture reconstructed_interpolated_depth;
    std::array<Texture, 2> game_vector_field;
    std::array<Texture, 2> optical_flow_vector_field;
    Texture disocclusion_mask;
    Texture inpainting_pyramid; // display / 2, full mip chain
    Texture default_distortion;
    vk::Buffer counters;

    // Optical flow (display size).
    std::array<std::array<Texture, 7>, 2> of_input;
    std::array<std::array<Texture, 7>, 2> of_flow;
    Texture of_scd_histogram;
    Texture of_scd_previous_histogram;
    Texture of_scd_temp;
    Texture of_vector;
    Texture of_confidence; // never written: the SDK binds nothing here
    Texture of_scd_output;

    u32 of_frame_index{};
    u32 of_resource_index{};
    u64 frames_processed{};
    bool initialized{};
    bool pending_reset{};
};

} // namespace Vulkan
