// SPDX-License-Identifier: GPL-3.0-or-later
// Guarded translation regression adapted from Lichii commit 97b32079c05ccd3d90245b0a376da694968296da.
#include <array>
#include <span>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"

namespace {
namespace IR = Shader::IR;
using Shader::Ampere::Instruction;
constexpr Instruction Exit{{0x794d, 0, 0x03800000, 0}};

class GatherEnvironment final : public Shader::Environment {
public:
    u32 binding{};
    std::span<const u64> code;

    GatherEnvironment() { stage = Shader::Stage::Compute; }
    u64 ReadInstruction(u32 address) override {
        for (u32 index = 0; index < code.size(); ++index)
            if (address == 0x50 + index * 8 + ((index + 1) / 3) * 8) return code[index];
        FAIL("Read beyond gather test program");
        return 0;
    }
    u32 ReadCbufValue(u32 bank, u32 offset) override { return (bank << 16) | offset; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return binding; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 4096; }
    std::array<u32, 3> WorkgroupSize() const override { return {32, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return {}; }
    void Dump(u64, u64) override {}
};

std::vector<Instruction> GuardedGather(bool half, u32 guard) {
    return {
        Instruction{{0x000e7919, 0, 0x2100, 0}},
        Instruction{{0x0e00720c, 0xff, 0x03f25070, 0}},
        Instruction{{(0xff107424U & ~0xf000U) | (guard << 12), 0xfffeffff, 0x078e00ff, 0}},
        Instruction{{(0xff117424U & ~0xf000U) | (guard << 12), 0xfffcfffd, 0x078e00ff, 0}},
        Instruction{{(0x06187b63U & ~0xf000U) | (guard << 12), 0x30007c10,
                     0x001e2f0eU | (half ? 1U << 15 : 0U), 0}},
        Exit,
    };
}
}

TEST_CASE("Guarded native gather constants survive full translation and SPIRV emission",
          "[video_core][ampere][gather]") {
    for (const bool half : {false, true}) {
        for (const u32 guard : {1U, 7U, 9U}) {
            const auto lowered = Shader::Ampere::LowerToMaxwell(GuardedGather(half, guard));
            INFO(lowered.error);
            REQUIRE(lowered);
            auto code = lowered.instructions;
            // Make both result registers observable through shared-memory stores.
            for (const u32 reg : {14U, 24U})
                code.insert(code.end() - 1, 0xef58000000070000ULL | reg | (255ULL << 8) |
                    (u64{reg * 4} << 20) | (4ULL << 48));
            GatherEnvironment env;
            env.code = code;
            env.binding = lowered.required_texture_buffer;
            Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
            Shader::Maxwell::Flow::CFG cfg{env, flow_pool, Shader::Maxwell::Location{0x50}};
            Shader::ObjectPool<IR::Inst> inst_pool;
            Shader::ObjectPool<IR::Block> block_pool;
            Shader::HostTranslateInfo host;
            host.ApplyDescriptorLimitPolicy();
            auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
            u32 gathers = 0;
            constexpr std::array<std::array<u32, 4>, 2> expected{{
                {{0xffffffff, 0xffffffff, 0xfffffffe, 0xffffffff}},
                {{0xfffffffd, 0xffffffff, 0xfffffffc, 0xffffffff}},
            }};
            for (const auto* block : program.blocks) {
                for (const auto& inst : *block) {
                    if (inst.GetOpcode() != IR::Opcode::ImageGather) continue;
                    ++gathers;
                    for (u32 argument = 2; argument < 4; ++argument) {
                        const auto* offsets = inst.Arg(argument).InstRecursive();
                        REQUIRE(offsets->GetOpcode() == IR::Opcode::CompositeConstructU32x4);
                        CHECK(offsets->AreAllArgsImmediates());
                        if (!offsets->AreAllArgsImmediates()) continue;
                        for (u32 lane = 0; lane < 4; ++lane)
                            CHECK(offsets->Arg(lane).U32() == expected[argument - 2][lane]);
                    }
                }
            }
            REQUIRE(gathers == 1);
            Shader::Profile profile{};
            profile.supported_spirv = 0x00010400;
            const auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, program);
            REQUIRE(spirv.size() > 5);
            CHECK(spirv.front() == 0x07230203);
            u32 gather_offsets = 0;
            for (size_t i = 5; i < spirv.size();) {
                const u32 count = spirv[i] >> 16, op = spirv[i] & 0xffff;
                REQUIRE(count != 0);
                REQUIRE(count <= spirv.size() - i);
                // OpImageGather: result type, result id, image, coordinate, component, operands.
                if (op == 96 && count >= 8 && (spirv[i + 6] & 0x20) != 0) ++gather_offsets;
                i += count;
            }
            CHECK(gather_offsets == 1);
        }
    }
}

TEST_CASE("Unproven native gather offsets keep their guarded register reads",
          "[video_core][ampere][gather]") {
    for (u32 variant = 0; variant < 6; ++variant) {
        auto code = GuardedGather(false, 9);
        u32 gather_index = 4;
        if (variant == 0) {
            // A different guard can leave the old offset live when the gather executes.
            code[2].words[0] = (code[2].words[0] & ~0xf000U) | 0x1000U;
        } else if (variant == 1) {
            code.insert(code.begin() + 4, code[1]); // P1 is written after both definitions.
            ++gather_index;
        } else if (variant == 2) {
            code.insert(code.begin() + 4, Instruction{{0x00107a02, 0, 0xf00, 0}});
            ++gather_index; // R16 now comes from a constant buffer, not a literal.
        } else if (variant == 3) {
            code.insert(code.begin() + 2, Instruction{{0x00007947, 16, 0x03800000, 0}});
            ++gather_index; // Incoming edge skips the first literal definition.
        } else if (variant == 4) {
            code.erase(code.begin() + 2, code.begin() + 4);
            gather_index -= 2; // Runtime register offsets remain accepted.
        } else {
            code.insert(code.begin() + 4, Instruction{{0x54107381, 0, 0x001e8d00, 0}});
            ++gather_index; // A128-bit load overwrites the offset register pair.
        }
        const auto lowered = Shader::Ampere::LowerToMaxwell(code);
        INFO(lowered.error);
        REQUIRE(lowered);
        const u32 start = lowered.source_instruction_offsets[gather_index];
        for (u32 lane = 0; lane < 2; ++lane) {
            const u64 copy = lowered.instructions[start + 2 + lane];
            CHECK((copy >> 48) == 0x5c98);
            CHECK(((copy >> 16) & 15) == 9);
            CHECK(((copy >> 20) & 255) == 16 + lane);
        }
    }
}

TEST_CASE("Literal native gather offsets survive unrelated loads and shared stores",
          "[video_core][ampere][gather]") {
    auto code = GuardedGather(true, 7);
    code.insert(code.begin() + 4, Instruction{{0x54147381, 0, 0x001e8d00, 0}});
    code.insert(code.begin() + 5, Instruction{{0x08007388, 12, 0x800, 0}});
    const auto lowered = Shader::Ampere::LowerToMaxwell(code);
    INFO(lowered.error);
    REQUIRE(lowered);
    const u32 start = lowered.source_instruction_offsets[6];
    for (u32 lane = 0; lane < 2; ++lane) {
        const u64 move = lowered.instructions[start + 2 + lane];
        CHECK((move >> 56) == 0x01);
        CHECK(((move >> 16) & 15) == 7);
        CHECK(static_cast<u32>(move >> 20) == (lane ? 0xfffcfffdU : 0xfffeffffU));
    }
}
