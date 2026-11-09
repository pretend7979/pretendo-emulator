// SPDX-License-Identifier: GPL-3.0-or-later
#include <array>
#include <bit>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr u32 Untouched = 0xdeadbeef;
constexpr std::array<u32, 4> Texel{0x3f800000, 0xc0000000, 0x3f000000, 0x40800000};
constexpr std::array<u32, 4> HalfTexel{0x3c00, 0xc000, 0x3800, 0x4400};

u32 ConvertExactHalf(u32 value) {
    if (value == 0) return 0;
    for (size_t i = 0; i < Texel.size(); ++i) {
        if (value == Texel[i]) return HalfTexel[i];
    }
    FAIL("Unexpected value passed to the texture packing conversion");
    return 0;
}

// Evaluate the emitted conversion/packing tail using Maxwell F2F and ISCADD semantics.
std::array<u32, 256> ExecutePacking(const Shader::Ampere::LoweredProgram& program,
                                  u32 mask, bool predicate_zero) {
    std::array<u32, 256> registers;
    registers.fill(Untouched);
    registers[255] = 0;
    bool sampled = false;
    for (const u64 word : program.instructions) {
        const u32 guard = (word >> 16) & 15;
        const bool condition = (guard & 7) == 7 ? true : predicate_zero;
        const bool enabled = condition != ((guard & 8) != 0);
        const bool texture = (word >> 60) == 0xc || (word >> 56) == 0xde;
        if (texture) {
            sampled = true;
            if (enabled) {
                u32 output = 250;
                for (u32 component = 0; component < 4; ++component) {
                    if ((mask >> component) & 1) registers[output++] = Texel[component];
                }
            }
            continue;
        }
        if (!sampled || (word >> 56) == 0xe3) continue;
        const u32 destination = word & 255, a = (word >> 8) & 255, b = (word >> 20) & 255;
        if ((word & 0xfff8000000000000ULL) == 0x5ca8000000000000ULL) {
            REQUIRE(((word >> 8) & 3) == 1); // F16 destination, F32 source.
            REQUIRE(((word >> 10) & 3) == 2);
            if (enabled && destination != 255) registers[destination] = ConvertExactHalf(registers[b]);
        } else if ((word & 0xfff8000000000000ULL) == 0x5c18000000000000ULL) {
            REQUIRE(((word >> 47) & 7) == 0); // No carry or operand negation.
            if (enabled && destination != 255)
                registers[destination] = (registers[a] << ((word >> 39) & 31)) + registers[b];
        } else {
            FAIL("Unexpected instruction after the texture sample");
        }
    }
    REQUIRE(sampled);
    return registers;
}
} // namespace

TEST_CASE("Ampere TEX F16 packs sampled components into their native halves",
          "[video_core][ampere][texture_packing]") {
    for (const bool bindless : {false, true}) {
        for (u32 mask = 1; mask < 16; ++mask) {
            for (const bool split : {false, true}) {
                for (const u32 guard : {7U, 0U, 8U}) {
                    for (const bool predicate_zero : {false, true}) {
                        Instruction texture{{0x04080b60U | (guard << 12), 0x30000106,
                                             0x001e8000U | (mask << 8) | (split ? 24U : 255U), 0}};
                        if (bindless) {
                            texture.words[0] = (texture.words[0] & ~0xfffU) | 0x361;
                            texture.words[1] = 0x38000006;
                        }
                        const auto lowered = Shader::Ampere::LowerToMaxwell(std::array{texture, Exit});
                        CAPTURE(bindless, mask, split, guard, predicate_zero, lowered.error);
                        REQUIRE(static_cast<bool>(lowered));
                        const auto registers = ExecutePacking(lowered, mask, predicate_zero);
                        const bool enabled = guard == 7 || (guard == 0 ? predicate_zero : !predicate_zero);
                        std::vector<u32> components;
                        for (u32 component = 0; component < 4; ++component)
                            if ((mask >> component) & 1) components.push_back(HalfTexel[component]);
                        const u32 first = components[0] | (components.size() > 1 ? components[1] << 16 : 0);
                        CHECK(registers[8] == (enabled ? first : Untouched));
                        const u32 second_destination = split ? 24 : 9;
                        const u32 second = components.size() > 2
                            ? components[2] | (components.size() > 3 ? components[3] << 16 : 0) : Untouched;
                        CHECK(registers[second_destination] == (enabled ? second : Untouched));
                    }
                }
            }
        }
    }
}
