// SPDX-License-Identifier: GPL-3.0-or-later

// Host side of AMD's FSR 3 frame generation, after the SDK's ffx_opticalflow.cpp and
// ffx_frameinterpolation.cpp (v1.1.4): the same resources, constants and pass order, recorded
// on this renderer's scheduler instead of through an FFX backend.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <numbers>
#include <span>

#include "common/alignment.h"
#include "common/logging.h"
#include "video_core/host_shaders/ffx_fg_fi_disocclusion_mask_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_fi_game_motion_vector_field_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_fi_game_vector_field_inpainting_pyramid_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_fi_game_vector_field_inpainting_pyramid_nowave_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_fi_inpainting_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_fi_inpainting_pyramid_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_fi_inpainting_pyramid_nowave_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_fi_interpolation_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_fi_optical_flow_vector_field_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_fi_reconstruct_and_dilate_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_fi_reconstruct_previous_depth_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_fi_setup_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_of_filter_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_of_luminance_pyramid_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_of_luminance_pyramid_nowave_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_of_prepare_luma_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_of_scale_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_of_scd_divergence_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_of_scd_histogram_comp_spv.h"
#include "video_core/host_shaders/ffx_fg_of_search_comp_spv.h"
// vk_compute_pass.h (through gpu.h) before any Vulkan header: on Linux vulkan.h pulls in
// Xlib.h, whose Success macro breaks NvResult::Success.
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/renderer_vulkan/present/amd_fsr3_frame_gen.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {

// cbFI_t in ffx_frameinterpolation_callbacks_glsl.h (std140).
struct FiConstants {
    std::array<s32, 2> render_size;
    std::array<s32, 2> display_size;
    std::array<f32, 2> display_size_rcp;
    f32 camera_near;
    f32 camera_far;
    std::array<s32, 2> upscaler_target_size;
    s32 mode;
    s32 reset;
    std::array<f32, 4> device_to_view_depth;
    f32 delta_time;
    s32 hudless_attached;
    std::array<s32, 2> distortion_field_size;
    std::array<f32, 2> optical_flow_scale;
    s32 optical_flow_block_size;
    u32 dispatch_flags;
    std::array<s32, 2> max_render_size;
    s32 optical_flow_half_res_mode;
    s32 num_instances;
    std::array<s32, 2> interpolation_rect_base;
    std::array<s32, 2> interpolation_rect_size;
    std::array<f32, 3> debug_bar_color;
    u32 back_buffer_transfer_function;
    std::array<f32, 2> min_max_luminance;
    f32 tan_half_fov;
    s32 pad;
    std::array<f32, 2> jitter;
    std::array<f32, 2> motion_vector_scale;
};
static_assert(sizeof(FiConstants) == 176);

// cbInpaintingPyramid_t and cbOF_SPD_t.
struct SpdConstants {
    u32 mips;
    u32 num_work_groups;
    std::array<u32, 2> work_group_offset;
    u32 num_work_groups_of_input_pyramid;
    std::array<u32, 3> pad;
};

// cbOF_t.
struct OfConstants {
    std::array<s32, 2> input_luma_resolution;
    u32 pyramid_level;
    u32 pyramid_level_count;
    u32 frame_index;
    u32 back_buffer_transfer_function;
    std::array<f32, 2> min_max_luminance;
};

enum class Kind : u8 { Texture, Image, Uniform, Storage };
constexpr Kind T = Kind::Texture;
constexpr Kind I = Kind::Image;
constexpr Kind U = Kind::Uniform;
constexpr Kind S = Kind::Storage;

struct Binding {
    u32 binding;
    Kind kind;
};

// Descriptor bindings of each pass file (its FFX_*_BIND_* numbers).
constexpr std::array<Binding, 6> FI_RECONSTRUCT_AND_DILATE{{{0, T}, {1, T}, {2, I}, {3, I}, {4, I}, {5, U}}};
constexpr std::array<Binding, 8> FI_SETUP{{{0, T}, {1, I}, {2, I}, {3, I}, {4, I}, {5, I}, {6, S}, {7, U}}};
constexpr std::array<Binding, 6> FI_RECONSTRUCT_PREVIOUS_DEPTH{{{0, T}, {1, T}, {2, T}, {3, T}, {4, I}, {5, U}}};
constexpr std::array<Binding, 8> FI_GAME_MOTION_VECTOR_FIELD{{{0, T}, {1, T}, {2, T}, {3, T}, {4, T}, {5, I}, {6, I}, {7, U}}};
constexpr std::array<Binding, 18> FI_GAME_VECTOR_FIELD_PYRAMID{{{0, T}, {1, T}, {2, S}, {3, I}, {4, I}, {5, I}, {6, I}, {7, I}, {8, I}, {9, I}, {10, I}, {11, I}, {12, I}, {13, I}, {14, I}, {15, I}, {16, U}, {17, U}}};
constexpr std::array<Binding, 8> FI_OPTICAL_FLOW_VECTOR_FIELD{{{0, T}, {1, T}, {2, T}, {3, T}, {4, T}, {6, I}, {7, I}, {8, U}}};
constexpr std::array<Binding, 9> FI_DISOCCLUSION_MASK{{{0, T}, {1, T}, {2, T}, {3, T}, {4, T}, {5, T}, {6, T}, {7, I}, {8, U}}};
constexpr std::array<Binding, 11> FI_INTERPOLATION{{{0, T}, {1, T}, {2, T}, {3, T}, {4, T}, {5, T}, {6, T}, {7, T}, {8, S}, {9, I}, {10, U}}};
constexpr std::array<Binding, 17> FI_INPAINTING_PYRAMID{{{0, T}, {1, S}, {2, I}, {3, I}, {4, I}, {5, I}, {6, I}, {7, I}, {8, I}, {9, I}, {10, I}, {11, I}, {12, I}, {13, I}, {14, I}, {15, U}, {16, U}}};
constexpr std::array<Binding, 6> FI_INPAINTING{{{0, T}, {1, T}, {2, T}, {3, T}, {7, I}, {8, U}}};
constexpr std::array<Binding, 3> OF_PREPARE_LUMA{{{0, T}, {1, I}, {2, U}}};
constexpr std::array<Binding, 9> OF_LUMINANCE_PYRAMID{{{0, I}, {1, I}, {2, I}, {3, I}, {4, I}, {5, I}, {6, I}, {7, U}, {8, U}}};
constexpr std::array<Binding, 3> OF_SCD_HISTOGRAM{{{0, T}, {1, I}, {2, U}}};
constexpr std::array<Binding, 5> OF_SCD_DIVERGENCE{{{0, I}, {1, I}, {2, I}, {3, I}, {4, U}}};
constexpr std::array<Binding, 5> OF_SEARCH{{{0, T}, {1, T}, {2, I}, {3, I}, {4, U}}};
constexpr std::array<Binding, 3> OF_FILTER{{{0, T}, {1, I}, {2, U}}};
constexpr std::array<Binding, 6> OF_SCALE{{{0, T}, {1, T}, {2, T}, {3, I}, {4, I}, {5, U}}};

enum PassId : size_t {
    PassReconstructAndDilate,
    PassSetup,
    PassReconstructPreviousDepth,
    PassGameMotionVectorField,
    PassGameVectorFieldPyramid,
    PassOpticalFlowVectorField,
    PassDisocclusionMask,
    PassInterpolation,
    PassInpaintingPyramid,
    PassInpainting,
    PassPrepareLuma,
    PassLuminancePyramid,
    PassScdHistogram,
    PassScdDivergence,
    PassSearch,
    PassFilter,
    PassScale,
    PassCount,
};

constexpr u32 PYRAMID_BINDINGS = 13;
constexpr u32 OF_LEVELS = 7;
constexpr u32 OF_BLOCK_SIZE = 8;
constexpr u32 HISTOGRAM_BINS = 256;
constexpr u32 HISTOGRAMS_PER_DIM = 3;
constexpr u32 HISTOGRAM_SHIFTS = 3;
constexpr u32 OF_MAX_QUEUED_FRAMES = 16;

// Without game camera data the depth stays at the far plane; these only scale thresholds.
constexpr f32 CAMERA_NEAR = 0.1f;
constexpr f32 CAMERA_FAR = 1000.0f;
constexpr f32 CAMERA_FOV = 60.0f * std::numbers::pi_v<f32> / 180.0f;

constexpr VkDeviceSize UNIFORM_RING_SIZE = 256 * 1024;

constexpr VkMemoryBarrier MemoryBarrier(VkAccessFlags src, VkAccessFlags dst) {
    return VkMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src,
        .dstAccessMask = dst,
    };
}

constexpr VkImageSubresourceRange ALL_LEVELS{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = VK_REMAINING_MIP_LEVELS,
    .baseArrayLayer = 0,
    .layerCount = 1,
};

u32 DivCeil(u32 value, u32 divisor) {
    return (value + divisor - 1) / divisor;
}

/// ffxSpdSetup over a rect at the origin.
SpdConstants SpdSetup(u32 width, u32 height, s32 mips, std::array<u32, 2>& groups) {
    groups = {(width - 1) / 64 + 1, (height - 1) / 64 + 1};
    SpdConstants result{};
    result.num_work_groups = groups[0] * groups[1];
    result.mips = mips >= 0 ? static_cast<u32>(mips)
                            : static_cast<u32>((std::min)(
                                  std::floor(std::log2(static_cast<f32>((std::max)(width, height)))),
                                  12.0f));
    result.num_work_groups_of_input_pyramid = result.num_work_groups;
    return result;
}

} // Anonymous namespace

class AmdFsr3FrameGen::Pass final : public ComputePass {
public:
    explicit Pass(const Device& device_, Scheduler& scheduler, DescriptorPool& descriptor_pool,
                  std::span<const Binding> bindings, std::span<const u32> code,
                  const VkSampler* sampler)
        : Pass(device_, scheduler, descriptor_pool, MakeLayout(bindings, sampler), code) {}

    void Dispatch(Scheduler& scheduler_, ComputePassDescriptorQueue& queue,
                  std::span<const Resource> resources, u32 groups_x, u32 groups_y,
                  u32 groups_z = 1) {
        queue.Acquire(scheduler_, resources.size());
        for (const Resource& resource : resources) {
            if (resource.buffer) {
                queue.AddBuffer(resource.buffer, resource.offset, resource.size);
            } else {
                queue.AddImage(resource.view);
            }
        }
        const void* const descriptor_data{queue.UpdateData()};
        scheduler_.Record([this, descriptor_data, groups_x, groups_y,
                           groups_z](vk::CommandBuffer cmdbuf) {
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptor_data);
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmdbuf.Dispatch(groups_x, groups_y, groups_z);
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                   MemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT,
                                                 VK_ACCESS_SHADER_READ_BIT |
                                                     VK_ACCESS_SHADER_WRITE_BIT));
        });
    }

private:
    struct Layout {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        std::vector<VkDescriptorUpdateTemplateEntry> templates;
        DescriptorBankInfo bank;
    };

    explicit Pass(const Device& device_, Scheduler& scheduler, DescriptorPool& descriptor_pool,
                  const Layout& pass_layout, std::span<const u32> code)
        : ComputePass(device_, scheduler, descriptor_pool, pass_layout.bindings,
                      pass_layout.templates, pass_layout.bank, {}, code) {}

    static Layout MakeLayout(std::span<const Binding> bindings, const VkSampler* sampler) {
        Layout result;
        for (size_t index = 0; index < bindings.size(); ++index) {
            VkDescriptorType type{};
            switch (bindings[index].kind) {
            case Kind::Texture:
                type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                ++result.bank.sampled_images;
                break;
            case Kind::Image:
                type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                ++result.bank.images;
                break;
            case Kind::Uniform:
                type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                ++result.bank.uniform_buffers;
                break;
            case Kind::Storage:
                type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                ++result.bank.storage_buffers;
                break;
            }
            result.bindings.push_back({
                .binding = bindings[index].binding,
                .descriptorType = type,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            });
            result.templates.push_back({
                .dstBinding = bindings[index].binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = type,
                .offset = index * sizeof(DescriptorUpdateEntry),
                .stride = sizeof(DescriptorUpdateEntry),
            });
        }
        // s_LinearClamp (the optical flow passes declare none; an unused binding is harmless).
        result.bindings.push_back({
            .binding = 1000,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = sampler,
        });
        ++result.bank.samplers;
        result.bank.score = static_cast<s32>(result.bindings.size());
        return result;
    }
};


namespace {

constexpr VkImageCopy FullCopy(VkExtent2D extent) {
    return VkImageCopy{
        .srcSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset{},
        .dstSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset{},
        .extent{extent.width, extent.height, 1},
    };
}

/// Everything recorded before is visible to everything recorded after.
void FullBarrier(vk::CommandBuffer cmdbuf) {
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           0,
                           MemoryBarrier(VK_ACCESS_MEMORY_WRITE_BIT,
                                         VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT));
}

FiConstants MakeConstants(VkExtent2D render, VkExtent2D display, bool reset) {
    FiConstants constants{};
    constants.render_size = {static_cast<s32>(render.width), static_cast<s32>(render.height)};
    constants.max_render_size = constants.render_size;
    constants.display_size = {static_cast<s32>(display.width), static_cast<s32>(display.height)};
    constants.display_size_rcp = {1.0f / static_cast<f32>(display.width),
                                  1.0f / static_cast<f32>(display.height)};
    constants.upscaler_target_size = constants.display_size;
    constants.interpolation_rect_size = constants.display_size;
    constants.reset = reset ? 1 : 0;
    constants.camera_near = CAMERA_NEAR;
    constants.camera_far = CAMERA_FAR;
    constants.delta_time = 1.0f / 60.0f;
    constants.distortion_field_size = {1, 1};
    constants.optical_flow_scale = {1.0f / static_cast<f32>(display.width),
                                    1.0f / static_cast<f32>(display.height)};
    constants.optical_flow_block_size = static_cast<s32>(OF_BLOCK_SIZE);
    constants.min_max_luminance = {0.0f, 1.0f};
    constants.motion_vector_scale = {1.0f, 1.0f};
    // setupDeviceDepthToViewSpaceDepthParams: standard, finite depth.
    const f32 q = CAMERA_FAR / (CAMERA_NEAR - CAMERA_FAR);
    const f32 aspect = static_cast<f32>(render.width) / static_cast<f32>(render.height);
    const f32 cot_half_fov = std::cos(0.5f * CAMERA_FOV) / std::sin(0.5f * CAMERA_FOV);
    constants.device_to_view_depth = {-q, q * CAMERA_NEAR, aspect / cot_half_fov,
                                      1.0f / cot_half_fov};
    const f32 horizontal = std::atan(std::tan(CAMERA_FOV / 2.0f) * aspect) * 2.0f;
    constants.tan_half_fov = std::tan(horizontal * 0.5f);
    return constants;
}

} // Anonymous namespace

AmdFsr3FrameGen::AmdFsr3FrameGen(const Device& device_, Scheduler& scheduler_,
                                 MemoryAllocator& memory_allocator, VkExtent2D extent,
                                 VkFormat format_)
    : device{device_}, scheduler{scheduler_}, allocator{memory_allocator}, display{extent},
      render{(std::max)(1U, extent.width / 2), (std::max)(1U, extent.height / 2)},
      format{format_}, descriptor_pool{device_, scheduler_},
      descriptor_queue{device_, ComputePassDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE} {
    linear_sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
    linear_sampler_handle = *linear_sampler;

    const bool quads = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_QUAD_BIT) &&
                       (device.GetSubgroupSupportedStages() & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    const auto pass = [&](std::span<const Binding> bindings, std::span<const u32> code) {
        return std::make_unique<Pass>(device, scheduler, descriptor_pool, bindings, code,
                                      &linear_sampler_handle);
    };
    const auto spd = [&](std::span<const u32> wave, std::span<const u32> nowave) {
        return quads ? wave : nowave;
    };
    passes.resize(PassCount);
    passes[PassReconstructAndDilate] =
        pass(FI_RECONSTRUCT_AND_DILATE, FFX_FG_FI_RECONSTRUCT_AND_DILATE_COMP_SPV);
    passes[PassSetup] = pass(FI_SETUP, FFX_FG_FI_SETUP_COMP_SPV);
    passes[PassReconstructPreviousDepth] =
        pass(FI_RECONSTRUCT_PREVIOUS_DEPTH, FFX_FG_FI_RECONSTRUCT_PREVIOUS_DEPTH_COMP_SPV);
    passes[PassGameMotionVectorField] =
        pass(FI_GAME_MOTION_VECTOR_FIELD, FFX_FG_FI_GAME_MOTION_VECTOR_FIELD_COMP_SPV);
    passes[PassGameVectorFieldPyramid] =
        pass(FI_GAME_VECTOR_FIELD_PYRAMID,
             spd(FFX_FG_FI_GAME_VECTOR_FIELD_INPAINTING_PYRAMID_COMP_SPV,
                 FFX_FG_FI_GAME_VECTOR_FIELD_INPAINTING_PYRAMID_NOWAVE_COMP_SPV));
    passes[PassOpticalFlowVectorField] =
        pass(FI_OPTICAL_FLOW_VECTOR_FIELD, FFX_FG_FI_OPTICAL_FLOW_VECTOR_FIELD_COMP_SPV);
    passes[PassDisocclusionMask] = pass(FI_DISOCCLUSION_MASK, FFX_FG_FI_DISOCCLUSION_MASK_COMP_SPV);
    passes[PassInterpolation] = pass(FI_INTERPOLATION, FFX_FG_FI_INTERPOLATION_COMP_SPV);
    passes[PassInpaintingPyramid] =
        pass(FI_INPAINTING_PYRAMID, spd(FFX_FG_FI_INPAINTING_PYRAMID_COMP_SPV,
                                        FFX_FG_FI_INPAINTING_PYRAMID_NOWAVE_COMP_SPV));
    passes[PassInpainting] = pass(FI_INPAINTING, FFX_FG_FI_INPAINTING_COMP_SPV);
    passes[PassPrepareLuma] = pass(OF_PREPARE_LUMA, FFX_FG_OF_PREPARE_LUMA_COMP_SPV);
    passes[PassLuminancePyramid] =
        pass(OF_LUMINANCE_PYRAMID, spd(FFX_FG_OF_LUMINANCE_PYRAMID_COMP_SPV,
                                       FFX_FG_OF_LUMINANCE_PYRAMID_NOWAVE_COMP_SPV));
    passes[PassScdHistogram] = pass(OF_SCD_HISTOGRAM, FFX_FG_OF_SCD_HISTOGRAM_COMP_SPV);
    passes[PassScdDivergence] = pass(OF_SCD_DIVERGENCE, FFX_FG_OF_SCD_DIVERGENCE_COMP_SPV);
    passes[PassSearch] = pass(OF_SEARCH, FFX_FG_OF_SEARCH_COMP_SPV);
    passes[PassFilter] = pass(OF_FILTER, FFX_FG_OF_FILTER_COMP_SPV);
    passes[PassScale] = pass(OF_SCALE, FFX_FG_OF_SCALE_COMP_SPV);

    uniform_alignment = (std::max<VkDeviceSize>)(device.GetUniformBufferAlignment(), 256);
    uniform_buffer = allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = UNIFORM_RING_SIZE,
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        MemoryUsage::Upload);
    counters = allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = 64,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        MemoryUsage::DeviceLocal);

    const u32 dw = display.width;
    const u32 dh = display.height;
    const u32 rw = render.width;
    const u32 rh = render.height;
    constexpr VkImageUsageFlags transfer =
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    previous_source = CreateTexture(format, dw, dh, 1, transfer);
    current_source = CreateTexture(format, dw, dh, 1, transfer);
    output = CreateTexture(format, dw, dh, 1, transfer);

    input_motion_vectors = CreateTexture(VK_FORMAT_R16G16_SFLOAT, rw, rh);
    input_depth = CreateTexture(VK_FORMAT_R32_SFLOAT, rw, rh);
    dilated_motion_vectors = CreateTexture(VK_FORMAT_R16G16_SFLOAT, rw, rh);
    dilated_depth = CreateTexture(VK_FORMAT_R32_SFLOAT, rw, rh);
    reconstructed_previous_depth = CreateTexture(VK_FORMAT_R32_UINT, rw, rh);
    reconstructed_interpolated_depth = CreateTexture(VK_FORMAT_R32_UINT, rw, rh);
    for (size_t i = 0; i < 2; ++i) {
        game_vector_field[i] = CreateTexture(VK_FORMAT_R32_UINT, rw, rh);
        optical_flow_vector_field[i] = CreateTexture(VK_FORMAT_R32_UINT, rw, rh);
    }
    disocclusion_mask = CreateTexture(VK_FORMAT_R8G8_UNORM, rw, rh);
    const u32 pyramid_w = (std::max)(1U, dw / 2);
    const u32 pyramid_h = (std::max)(1U, dh / 2);
    const u32 pyramid_levels = static_cast<u32>(std::floor(
                                   std::log2(static_cast<f32>((std::max)(pyramid_w, pyramid_h))))) +
                               1;
    inpainting_pyramid = CreateTexture(VK_FORMAT_R16G16B16A16_SFLOAT, pyramid_w, pyramid_h,
                                       (std::min)(pyramid_levels, PYRAMID_BINDINGS));
    default_distortion = CreateTexture(VK_FORMAT_R8G8_UNORM, 1, 1);

    for (size_t set = 0; set < 2; ++set) {
        for (u32 level = 0; level < OF_LEVELS; ++level) {
            of_input[set][level] = CreateTexture(VK_FORMAT_R8_UINT, (std::max)(1U, dw >> level),
                                                 (std::max)(1U, dh >> level));
        }
        u32 fw = DivCeil(dw, OF_BLOCK_SIZE);
        u32 fh = DivCeil(dh, OF_BLOCK_SIZE);
        for (u32 level = 0; level < OF_LEVELS; ++level) {
            of_flow[set][level] = CreateTexture(VK_FORMAT_R16G16_SINT, fw, fh);
            fw = Common::AlignUp(fw, 2U) / 2;
            fh = Common::AlignUp(fh, 2U) / 2;
        }
    }
    const u32 histogram_width = HISTOGRAM_BINS * HISTOGRAMS_PER_DIM * HISTOGRAMS_PER_DIM;
    of_scd_histogram = CreateTexture(VK_FORMAT_R32_UINT, histogram_width, 1);
    of_scd_previous_histogram = CreateTexture(VK_FORMAT_R32_SFLOAT, histogram_width, 1);
    of_scd_temp = CreateTexture(VK_FORMAT_R32_UINT, 3, 1);
    of_vector = CreateTexture(VK_FORMAT_R16G16_SINT, DivCeil(dw, OF_BLOCK_SIZE),
                              DivCeil(dh, OF_BLOCK_SIZE));
    of_confidence = CreateTexture(VK_FORMAT_R32_UINT, DivCeil(dw, OF_BLOCK_SIZE),
                                  DivCeil(dh, OF_BLOCK_SIZE));
    of_scd_output = CreateTexture(VK_FORMAT_R32_UINT, 3, 1);
    LOG_INFO(Render_Vulkan, "AMD FSR 3 frame generation: {}x{}", dw, dh);
}

AmdFsr3FrameGen::~AmdFsr3FrameGen() = default;

bool AmdFsr3FrameGen::IsSupported(const Device& device, VkFormat format) {
    if (!device.IsStorageImageExtendedFormatsSupported() ||
        !device.IsFormatlessImageLoadSupported()) {
        return false;
    }
    constexpr VkFormatFeatureFlags storage =
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    for (const VkFormat needed :
         {VK_FORMAT_R8_UINT, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_SINT,
          VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, format}) {
        if (!device.IsFormatSupported(needed, storage, FormatType::Optimal)) {
            return false;
        }
    }
    return device.IsFormatSupported(VK_FORMAT_R32_UINT,
                                    storage | VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT,
                                    FormatType::Optimal);
}

AmdFsr3FrameGen::Texture AmdFsr3FrameGen::CreateTexture(VkFormat texture_format, u32 width,
                                                        u32 height, u32 levels,
                                                        VkImageUsageFlags extra_usage) {
    Texture texture;
    texture.image = allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = texture_format,
        .extent = {(std::max)(width, 1U), (std::max)(height, 1U), 1},
        .mipLevels = levels,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | extra_usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    const auto view = [&](u32 base_level, u32 level_count) {
        return device.GetLogical().CreateImageView(VkImageViewCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *texture.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = texture_format,
            .components{},
            .subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, base_level, level_count, 0, 1},
        });
    };
    texture.view = view(0, levels);
    for (u32 level = 0; level < levels; ++level) {
        texture.levels.push_back(view(level, 1));
    }
    return texture;
}

AmdFsr3FrameGen::Resource AmdFsr3FrameGen::Uniform(const void* data, size_t size) {
    const VkDeviceSize slot = Common::AlignUp(static_cast<VkDeviceSize>(size), uniform_alignment);
    if (uniform_cursor + slot > UNIFORM_RING_SIZE) {
        uniform_cursor = 0;
    }
    std::memcpy(uniform_buffer.Mapped().data() + uniform_cursor, data, size);
    const Resource result{.buffer = *uniform_buffer, .offset = uniform_cursor, .size = slot};
    uniform_cursor += slot;
    return result;
}

void AmdFsr3FrameGen::Process(VkImage current_frame) {
    descriptor_queue.TickFrame();
    const bool first = !initialized;
    initialized = true;
    std::vector<VkImage> images;
    if (first) {
        for (const Texture* texture :
             {&previous_source, &current_source, &output, &input_motion_vectors, &input_depth,
              &dilated_motion_vectors, &dilated_depth, &reconstructed_previous_depth,
              &reconstructed_interpolated_depth, &game_vector_field[0], &game_vector_field[1],
              &optical_flow_vector_field[0], &optical_flow_vector_field[1], &disocclusion_mask,
              &inpainting_pyramid, &default_distortion, &of_scd_histogram,
              &of_scd_previous_histogram, &of_scd_temp, &of_vector, &of_confidence,
              &of_scd_output}) {
            images.push_back(*texture->image);
        }
        for (const auto* sets : {&of_input, &of_flow}) {
            for (const auto& set : *sets) {
                for (const Texture& texture : set) {
                    images.push_back(*texture.image);
                }
            }
        }
    }
    const VkImage depth = *input_depth.image;
    const VkImage previous = *previous_source.image;
    const VkImage current = *current_source.image;
    const VkImage reconstructed = *reconstructed_previous_depth.image;
    const VkBuffer counter_buffer = *counters;
    const VkExtent2D extent = display;
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([images = std::move(images), first, depth, previous, current, current_frame,
                      reconstructed, counter_buffer, extent](vk::CommandBuffer cmdbuf) {
        if (first) {
            std::vector<VkImageMemoryBarrier> barriers;
            for (const VkImage image : images) {
                barriers.push_back(VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = image,
                    .subresourceRange = ALL_LEVELS,
                });
            }
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, barriers);
            for (const VkImage image : images) {
                cmdbuf.ClearColorImage(image, VK_IMAGE_LAYOUT_GENERAL, VkClearColorValue{},
                                       ALL_LEVELS);
            }
            cmdbuf.FillBuffer(counter_buffer, 0, VK_WHOLE_SIZE, 0);
            // Standard depth: the far plane everywhere until the game supplies depth.
            VkClearColorValue far_plane{};
            far_plane.float32[0] = 1.0f;
            cmdbuf.ClearColorImage(depth, VK_IMAGE_LAYOUT_GENERAL, far_plane, ALL_LEVELS);
        }
        FullBarrier(cmdbuf);
        // The SDK copies the current source to the previous one after each dispatch.
        cmdbuf.CopyImage(current, VK_IMAGE_LAYOUT_GENERAL, previous, VK_IMAGE_LAYOUT_GENERAL,
                         FullCopy(extent));
        FullBarrier(cmdbuf);
        cmdbuf.CopyImage(current_frame, VK_IMAGE_LAYOUT_GENERAL, current, VK_IMAGE_LAYOUT_GENERAL,
                         FullCopy(extent));
        // ffxFrameInterpolationPrepare clears the previous nearest depth (to 1, standard depth).
        VkClearColorValue far_bits{};
        far_bits.uint32[0] = 0x3f800000U;
        cmdbuf.ClearColorImage(reconstructed, VK_IMAGE_LAYOUT_GENERAL, far_bits, ALL_LEVELS);
        FullBarrier(cmdbuf);
    });

    RunOpticalFlow(first);

    // ffxFrameInterpolationPrepare: dilate the render size motion vectors and depth.
    const FiConstants constants = MakeConstants(render, display, first);
    const Resource cb = Uniform(&constants, sizeof(constants));
    const std::array prepare{
        Resource{.view = *input_motion_vectors.view},
        Resource{.view = *input_depth.view},
        Resource{.view = *reconstructed_previous_depth.view},
        Resource{.view = *dilated_motion_vectors.view},
        Resource{.view = *dilated_depth.view},
        cb,
    };
    passes[PassReconstructAndDilate]->Dispatch(scheduler, descriptor_queue, prepare,
                                               DivCeil(render.width, 8),
                                               DivCeil(render.height, 8));
    ++frames_processed;
    pending_reset = first;
}

void AmdFsr3FrameGen::RunOpticalFlow(bool reset) {
    const u32 dw = display.width;
    const u32 dh = display.height;
    if (reset) {
        of_frame_index = 0;
    } else {
        ++of_frame_index;
    }
    OfConstants constants{
        .input_luma_resolution = {static_cast<s32>(dw), static_cast<s32>(dh)},
        .pyramid_level = 0,
        .pyramid_level_count = OF_LEVELS,
        .frame_index = of_frame_index,
        .back_buffer_transfer_function = 0,
        .min_max_luminance = {0.0f, 1.0f},
    };
    std::array<u32, 2> pyramid_groups{};
    const SpdConstants spd = SpdSetup(dw, dh, 4, pyramid_groups);
    const Resource cb = Uniform(&constants, sizeof(constants));
    const Resource cb_spd = Uniform(&spd, sizeof(spd));

    const bool odd = (of_resource_index & 1) != 0;
    const auto& input = of_input[odd ? 1 : 0];
    const auto& previous_input = of_input[odd ? 0 : 1];

    const std::array prepare{Resource{.view = *current_source.view},
                             Resource{.view = *input[0].view}, cb};
    passes[PassPrepareLuma]->Dispatch(scheduler, descriptor_queue, prepare,
                                      DivCeil(DivCeil(dw, 2), 16), DivCeil(DivCeil(dh, 2), 16));
    const std::array pyramid{
        Resource{.view = *input[0].view}, Resource{.view = *input[1].view},
        Resource{.view = *input[2].view}, Resource{.view = *input[3].view},
        Resource{.view = *input[4].view}, Resource{.view = *input[5].view},
        Resource{.view = *input[6].view}, cb,
        cb_spd,
    };
    passes[PassLuminancePyramid]->Dispatch(scheduler, descriptor_queue, pyramid,
                                           pyramid_groups[0], pyramid_groups[1]);
    const std::array histogram{Resource{.view = *input[0].view},
                               Resource{.view = *of_scd_histogram.view}, cb};
    const u32 strata_width = (dw / 4) / HISTOGRAMS_PER_DIM;
    passes[PassScdHistogram]->Dispatch(scheduler, descriptor_queue, histogram,
                                       DivCeil(strata_width, 32), 16,
                                       HISTOGRAMS_PER_DIM * HISTOGRAMS_PER_DIM);
    const std::array divergence{
        Resource{.view = *of_scd_histogram.view}, Resource{.view = *of_scd_previous_histogram.view},
        Resource{.view = *of_scd_temp.view}, Resource{.view = *of_scd_output.view}, cb};
    passes[PassScdDivergence]->Dispatch(scheduler, descriptor_queue, divergence,
                                        HISTOGRAMS_PER_DIM * HISTOGRAMS_PER_DIM, HISTOGRAM_SHIFTS);

    std::array<VkExtent2D, OF_LEVELS> sizes{};
    sizes[0] = {DivCeil(dw, OF_BLOCK_SIZE), DivCeil(dh, OF_BLOCK_SIZE)};
    for (u32 level = 1; level < OF_LEVELS; ++level) {
        sizes[level] = {(sizes[level - 1].width + 1) / 2, (sizes[level - 1].height + 1) / 2};
    }
    for (s32 level = OF_LEVELS - 1; level >= 0; --level) {
        const bool odd_level = (level & 1) != 0;
        const auto& flow_a = of_flow[odd != odd_level ? 1 : 0];
        const auto& flow_b = of_flow[odd != odd_level ? 0 : 1];
        constants.pyramid_level = static_cast<u32>(level);
        const Resource level_cb = Uniform(&constants, sizeof(constants));

        const u32 luma_w = (std::max)(dw >> level, 1U);
        const u32 luma_h = (std::max)(dh >> level, 1U);
        const std::array search{
            Resource{.view = *input[level].view}, Resource{.view = *previous_input[level].view},
            Resource{.view = *flow_a[level].view}, Resource{.view = *of_scd_output.view},
            level_cb};
        passes[PassSearch]->Dispatch(scheduler, descriptor_queue, search,
                                     DivCeil(DivCeil(luma_w, 4) * 16, 64), DivCeil(luma_h, 16));

        // Filter: the search result in, the other flow (or the shared output at level 0) out.
        const VkImageView filtered = level == 0 ? *of_vector.view : *flow_b[level].view;
        const std::array filter{Resource{.view = *flow_a[level].view}, Resource{.view = filtered},
                                level_cb};
        passes[PassFilter]->Dispatch(scheduler, descriptor_queue, filter,
                                     DivCeil(sizes[level].width, 16),
                                     DivCeil(sizes[level].height, 4));
        if (level > 0) {
            const std::array scale{
                Resource{.view = *input[level].view},
                Resource{.view = *previous_input[level].view},
                Resource{.view = *flow_b[level].view},
                Resource{.view = *flow_b[level - 1].view},
                Resource{.view = *of_scd_output.view},
                level_cb,
            };
            passes[PassScale]->Dispatch(scheduler, descriptor_queue, scale,
                                        DivCeil(sizes[level - 1].width, 4),
                                        DivCeil(sizes[level - 1].height, 4));
        }
    }
    of_resource_index = (of_resource_index + 1) % OF_MAX_QUEUED_FRAMES;
}

void AmdFsr3FrameGen::Generate(VkImage destination) {
    const FiConstants constants = MakeConstants(render, display, pending_reset);
    pending_reset = false;
    const Resource cb = Uniform(&constants, sizeof(constants));
    const Resource counter{.buffer = *counters, .offset = 0, .size = 64};
    const u32 rx = DivCeil(render.width, 8);
    const u32 ry = DivCeil(render.height, 8);
    const u32 dx = DivCeil(display.width, 8);
    const u32 dy = DivCeil(display.height, 8);
    const u32 ofx = DivCeil(display.width / OF_BLOCK_SIZE, 8);
    const u32 ofy = DivCeil(display.height / OF_BLOCK_SIZE, 8);

    const auto pyramid_level = [&](u32 index) {
        const u32 last = static_cast<u32>(inpainting_pyramid.levels.size()) - 1;
        return Resource{.view = *inpainting_pyramid.levels[(std::min)(index, last)]};
    };
    const Resource game_x{.view = *game_vector_field[0].view};
    const Resource game_y{.view = *game_vector_field[1].view};
    const Resource of_x{.view = *optical_flow_vector_field[0].view};
    const Resource of_y{.view = *optical_flow_vector_field[1].view};
    const Resource previous{.view = *previous_source.view};
    const Resource current{.view = *current_source.view};
    const Resource distortion{.view = *default_distortion.view};
    const Resource disocclusion{.view = *disocclusion_mask.view};
    const Resource pyramid{.view = *inpainting_pyramid.view};
    const Resource scd{.view = *of_scd_output.view};
    const Resource out{.view = *output.view};

    const std::array setup{scd,
                           game_x,
                           game_y,
                           of_x,
                           of_y,
                           disocclusion,
                           counter,
                           cb};
    passes[PassSetup]->Dispatch(scheduler, descriptor_queue, setup, rx, ry);

    if (!constants.reset) {
        const VkImage interpolated_depth = *reconstructed_interpolated_depth.image;
        scheduler.Record([interpolated_depth](vk::CommandBuffer cmdbuf) {
            VkClearColorValue far_bits{};
            far_bits.uint32[0] = 0x3f800000U;
            cmdbuf.ClearColorImage(interpolated_depth, VK_IMAGE_LAYOUT_GENERAL, far_bits,
                                   ALL_LEVELS);
            FullBarrier(cmdbuf);
        });
        const std::array previous_depth{Resource{.view = *dilated_motion_vectors.view},
                                        Resource{.view = *dilated_depth.view}, current, distortion,
                                        Resource{.view = *reconstructed_interpolated_depth.view},
                                        cb};
        passes[PassReconstructPreviousDepth]->Dispatch(scheduler, descriptor_queue,
                                                       previous_depth, rx, ry);
        const std::array game_field{Resource{.view = *dilated_motion_vectors.view},
                                    Resource{.view = *dilated_depth.view},
                                    previous,
                                    current,
                                    distortion,
                                    game_x,
                                    game_y,
                                    cb};
        passes[PassGameMotionVectorField]->Dispatch(scheduler, descriptor_queue, game_field, rx,
                                                    ry);
        std::array<u32, 2> groups{};
        const SpdConstants spd = SpdSetup(render.width, render.height, -1, groups);
        const Resource cb_spd = Uniform(&spd, sizeof(spd));
        const std::array game_pyramid{
            game_x,           game_y,           counter,          pyramid_level(0),
            pyramid_level(1), pyramid_level(2), pyramid_level(3), pyramid_level(4),
            pyramid_level(5), pyramid_level(6), pyramid_level(7), pyramid_level(8),
            pyramid_level(9), pyramid_level(10), pyramid_level(11), pyramid_level(12),
            cb,               cb_spd,
        };
        passes[PassGameVectorFieldPyramid]->Dispatch(scheduler, descriptor_queue, game_pyramid,
                                                     groups[0], groups[1]);
        const std::array of_field{Resource{.view = *of_vector.view},
                                  Resource{.view = *of_confidence.view},
                                  Resource{.view = *dilated_depth.view},
                                  previous,
                                  current,
                                  of_x,
                                  of_y,
                                  cb};
        passes[PassOpticalFlowVectorField]->Dispatch(scheduler, descriptor_queue, of_field, ofx,
                                                     ofy);
        const std::array mask{game_x,
                              game_y,
                              Resource{.view = *reconstructed_previous_depth.view},
                              Resource{.view = *dilated_depth.view},
                              Resource{.view = *reconstructed_interpolated_depth.view},
                              pyramid,
                              distortion,
                              disocclusion,
                              cb};
        passes[PassDisocclusionMask]->Dispatch(scheduler, descriptor_queue, mask, rx, ry);
    }

    const std::array interpolate{game_x,       game_y,  of_x,    of_y, previous, current,
                                 disocclusion, pyramid, counter, out,  cb};
    passes[PassInterpolation]->Dispatch(scheduler, descriptor_queue, interpolate, dx, dy);

    std::array<u32, 2> groups{};
    const SpdConstants spd = SpdSetup(display.width, display.height, -1, groups);
    const Resource cb_spd = Uniform(&spd, sizeof(spd));
    const std::array inpainting_pyramid_pass{
        out,              counter,          pyramid_level(0),  pyramid_level(1),
        pyramid_level(2), pyramid_level(3), pyramid_level(4),  pyramid_level(5),
        pyramid_level(6), pyramid_level(7), pyramid_level(8),  pyramid_level(9),
        pyramid_level(10), pyramid_level(11), pyramid_level(12), cb,
        cb_spd,
    };
    passes[PassInpaintingPyramid]->Dispatch(scheduler, descriptor_queue, inpainting_pyramid_pass,
                                            groups[0], groups[1]);
    const std::array inpainting{scd, pyramid, current, current, out, cb};
    passes[PassInpainting]->Dispatch(scheduler, descriptor_queue, inpainting, dx, dy);

    const VkImage source = *output.image;
    const VkExtent2D extent = display;
    scheduler.Record([source, destination, extent](vk::CommandBuffer cmdbuf) {
        FullBarrier(cmdbuf);
        cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_GENERAL, destination, VK_IMAGE_LAYOUT_GENERAL,
                         FullCopy(extent));
        FullBarrier(cmdbuf);
    });
}

} // namespace Vulkan
