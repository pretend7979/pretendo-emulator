// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

TEST_CASE("Ampere F2I round up preserves conversion mode", "[video_core][ampere]") {
    using Shader::Ampere::Instruction;
    using Shader::Ampere::LowerToMaxwell;
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};

    for (const bool is_signed : {false, true}) {
        const Instruction convert{{0x00067305, 0x00000009,
                                   is_signed ? 0x0021b100U : 0x0020b000U, 0}};
        const auto lowered = LowerToMaxwell(std::array{convert, exit});
        REQUIRE(static_cast<bool>(lowered));
        const auto instruction = std::ranges::find_if(lowered.instructions, [](u64 word) {
            return (word & 0xfff0000000000000ULL) == 0x5cb0000000000000ULL;
        });
        REQUIRE(instruction != lowered.instructions.end());
        REQUIRE((*instruction & 0xff) == 6);
        REQUIRE(((*instruction >> 8) & 3) == 2);
        REQUIRE(((*instruction >> 10) & 3) == 2);
        REQUIRE(((*instruction >> 12) & 1) == static_cast<u64>(is_signed));
        REQUIRE(((*instruction >> 39) & 3) == 2);
        REQUIRE(((*instruction >> 44) & 1) == static_cast<u64>(is_signed));
    }

    const Instruction unsupported{{0x00067305, 0x00000009, 0x0020b001, 0}};
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{unsupported, exit})));
}

TEST_CASE("Ampere F2I signed byte clamps after conversion", "[video_core][ampere]") {
    using Shader::Ampere::Instruction;
    using Shader::Ampere::LowerToMaxwell;
    const Instruction convert{{0x00097305, 0x00000005, 0x0021e100, 0}};
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    const auto lowered = LowerToMaxwell(std::array{convert, exit});
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(lowered.instructions.size() == 3);

    const u64 f2i = lowered.instructions[0];
    REQUIRE((f2i & 0xff) == 9);
    REQUIRE(((f2i >> 8) & 3) == 1);
    REQUIRE(((f2i >> 10) & 3) == 2);
    REQUIRE(((f2i >> 12) & 1) == 1);
    REQUIRE(((f2i >> 39) & 3) == 3);
    REQUIRE(((f2i >> 44) & 1) == 1);

    const u64 i2i = lowered.instructions[1];
    REQUIRE((i2i & 0xff) == 9);
    REQUIRE(((i2i >> 8) & 3) == 0);
    REQUIRE(((i2i >> 10) & 3) == 1);
    REQUIRE(((i2i >> 12) & 3) == 3);
    REQUIRE(((i2i >> 20) & 0xff) == 9);
    REQUIRE(((i2i >> 50) & 1) == 1);
}

TEST_CASE("Ampere F2IP packs two nearest-even unsigned bytes", "[video_core][ampere][f2ip]") {
    using Shader::Ampere::Instruction;
    using Shader::Ampere::LowerToMaxwell;
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    const std::array inputs{
        Instruction{{0x28297243, 0x27, 0x000004ff, 0x000fc400}},
        Instruction{{0x2c2d7243, 0x15, 0x000004ff, 0x000fc400}},
    };
    for (const auto input : inputs) {
        const auto lowered = LowerToMaxwell(std::array{input, exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 16);
        size_t conversions = 0;
        for (const u64 word : lowered.instructions) {
            if ((word >> 48) != 0x5cb0) continue;
            ++conversions;
            CHECK(((word >> 8) & 3) == 2);
            CHECK(((word >> 10) & 3) == 2);
            CHECK(((word >> 12) & 1) == 0);
            CHECK(((word >> 39) & 3) == 0);
        }
        CHECK(conversions == 2);
        const u64 pack = lowered.instructions[14];
        CHECK((pack & 255) == input.Bits(16, 8));
        CHECK(((pack >> 8) & 255) == 251);
        CHECK(((pack >> 20) & 255) == 250);
        CHECK(((pack >> 39) & 31) == 8);
    }
    for (const u32 mode : {0x000044ffU, 0x000084ffU, 0x0000c4ffU, 0x000014ffU,
                           0x00000406U}) {
        auto invalid = inputs[0];
        invalid.words[2] = mode;
        CHECK_FALSE(LowerToMaxwell(std::array{invalid, exit}));
    }
    auto invalid = inputs[0];
    invalid.words[1] |= 0x40000000;
    CHECK_FALSE(LowerToMaxwell(std::array{invalid, exit}));
}

TEST_CASE("Ampere F2IP preserves its legacy truncating sequence", "[video_core][ampere][f2ip]") {
    using Shader::Ampere::Instruction;
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    const Instruction convert{{0xff0a7243, 9, 0x0000c4ff, 0x000fe400}};
    const auto lowered = Shader::Ampere::LowerToMaxwell(std::array{convert, exit});
    REQUIRE(lowered);
    const std::array expected{
        0x5ba0000000000000ULL | 0x70000 | 250 | (u64{9} << 8) | (u64{255} << 20) |
            (u64{9} << 39) | (u64{7} << 48),
        0x5cb0000000000000ULL | 0x70000 | 10 | (u64{1} << 8) | (u64{2} << 10) |
            (u64{250} << 20) | (u64{3} << 39) | (u64{1} << 44),
        0x5ce0000000000000ULL | 0x70000 | 10 | (u64{1} << 10) | (u64{10} << 20) |
            (u64{1} << 50),
        0xe30000000007000fULL};
    REQUIRE(std::ranges::equal(lowered.instructions, expected));
}

TEST_CASE("Ampere F2IP protects both source registers from scratch writes",
          "[video_core][ampere][f2ip]") {
    using Shader::Ampere::Instruction;
    for (u32 source = 244; source < 255; ++source) {
        for (const u32 destination : {9U, source, 254U}) {
            const Instruction input{{0x7243 | (destination << 16) | (source << 24),
                                      250, 0x000004ff, 0x000fc400}};
            const auto pressure = Shader::Ampere::PrepareRegisterPressure(input, 244);
            REQUIRE(pressure.valid);
            REQUIRE_FALSE(pressure.prefix.empty());
            CHECK(pressure.instruction.Bits(24, 8) < 244);
            CHECK(pressure.instruction.Bits(32, 8) < 244);
            CHECK(pressure.instruction.Bits(16, 8) < 244);
            if (destination == source)
                CHECK(pressure.instruction.Bits(16, 8) == pressure.instruction.Bits(24, 8));
            if (source == 250)
                CHECK(pressure.instruction.Bits(24, 8) == pressure.instruction.Bits(32, 8));
            CHECK(pressure.instruction.words[2] == input.words[2]);
        }
    }
}
