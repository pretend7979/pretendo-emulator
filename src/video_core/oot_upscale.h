// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include "shader_recompiler/frontend/ampere/program_override.h"
#include "video_core/texture_cache/image_info.h"

namespace VideoCommon::OotUpscale {
constexpr u64 ProgramId = 0x0400486024DEC000ULL;
constexpr u64 ProgramKey = 0x6f9367bd41dc0b58ULL;
constexpr u32 InstructionCount = 1281;
struct ProgramVariant {
    u64 key;
    u32 instruction_count;
};
constexpr std::array ProgramVariants{
    ProgramVariant{ProgramKey, InstructionCount},
    ProgramVariant{0xd8efd9f0c95abd80ULL, 1372},
};
constexpr u32 MaximumInstructionCount = 1372;
constexpr u32 InputOffset = 0xc0;
constexpr u32 OutputOffset = 0xd0;

inline bool MatchesProgram(u64 program_id, std::span<const Shader::Ampere::Instruction> code) {
    if (program_id != ProgramId) {
        return false;
    }
    for (const auto& variant : ProgramVariants) {
        if (code.size() == variant.instruction_count) {
            return Shader::Ampere::NativeProgramKey(code) == variant.key;
        }
    }
    return false;
}

// TIC dimensions are positive and the replacement is bounded by the normal
// 16K 2D image limit. Dynamic resolution selects the input size independently of the output.
inline constexpr bool ValidDimensions(u32 input_width, u32 input_height,
                                      u32 output_width, u32 output_height) {
    constexpr u32 MaximumDimension = 16384;
    return input_width > 0 && input_height > 0 && output_width > 0 && output_height > 0 &&
           input_width <= MaximumDimension && input_height <= MaximumDimension &&
           output_width <= MaximumDimension && output_height <= MaximumDimension &&
           output_width >= input_width && output_height >= input_height;
}

// This captured output pass reads RGB9E5 at c0 and writes R11G11B10 at d0. c8 is
// an integer auxiliary image; the scalar at b8 can accidentally name another TIC.
inline bool ValidPair(Tegra::Texture::TICEntry input, Tegra::Texture::TICEntry output) {
    using namespace Tegra::Texture;
    input = TextureViewHeader(input, true);
    output = TextureViewHeader(output, true);
    const auto color = [](const TICEntry& image, TextureFormat format) {
        return image.format == format && image.texture_type == TextureType::Texture2D && image.IsBlockLinear() &&
               image.Depth() == 1 && image.Address() != 0 && image.srgb_conversion == 0 &&
               image.r_type == ComponentType::FLOAT && image.g_type == ComponentType::FLOAT &&
               image.b_type == ComponentType::FLOAT && image.a_type == ComponentType::FLOAT &&
               image.x_source == SwizzleSource::R && image.y_source == SwizzleSource::G &&
               image.z_source == SwizzleSource::B && image.w_source == SwizzleSource::OneFloat &&
               image.block_width == 0 && image.block_height == 4 && image.block_depth == 0 &&
               image.tile_width_spacing == 0 && image.max_mip_level == 0 &&
               image.res_min_mip_level == 0 && image.res_max_mip_level == 0 &&
               image.msaa_mode == MsaaMode::Msaa1x1 && image.is_sparse == 0;
    };
    return color(input, TextureFormat::E5B9G9R9) &&
           color(output, TextureFormat::B10G11R11) &&
           ValidDimensions(input.Width(), input.Height(), output.Width(), output.Height()) &&
           input.Address() != output.Address();
}
} // namespace VideoCommon::OotUpscale
