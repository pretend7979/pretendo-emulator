// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction CapturedRed{{0x0c00098e, 0xfffff007, 0x0010e100, 0x0011e400}};

Instruction Offset(s32 value, u32 guard = 7, u32 operation = 0) {
    auto red = CapturedRed;
    red.words[0] = (red.words[0] & ~0xf000U) | (guard << 12);
    red.words[1] = (static_cast<u32>(value) << 8) | 7;
    red.words[2] = (red.words[2] & ~0x03800000U) | (operation << 23);
    return red;
}

TEST_CASE("Register-address global reductions preserve signed offsets and guards",
          "[video_core][ampere]") {
    for (const s32 offset : {-0x80000, -16, -1, 0, 1, 0x7ffff}) {
        for (const u32 guard : {0U, 7U, 8U, 14U}) {
            for (u32 operation = 0; operation < 8; ++operation) {
                const std::array code{Offset(offset, guard, operation), Exit};
                const auto lowered = LowerToMaxwell(code);
                INFO(lowered.error);
                REQUIRE(static_cast<bool>(lowered));
                const auto it = std::ranges::find_if(lowered.instructions, [](u64 word) {
                    return (word & 0xfff8000000000000ULL) == 0xebf8000000000000ULL;
                });
                REQUIRE(it != lowered.instructions.end());
                REQUIRE((*it & 255) == 7);
                REQUIRE(((*it >> 8) & 255) == 12);
                REQUIRE(((*it >> 16) & 15) == guard);
                REQUIRE(((*it >> 23) & 7) == operation);
                const s32 emitted_offset = static_cast<s32>((*it >> 28) << 12) >> 12;
                REQUIRE(emitted_offset == offset);
                REQUIRE(((*it >> 48) & 1) == 1);
            }
        }
    }
    const std::array code{CapturedRed, Exit};
    REQUIRE(static_cast<bool>(LowerToMaxwell(code)));
}

TEST_CASE("Register reductions reject unrepresentable offsets and scratch operands",
          "[video_core][ampere]") {
    for (const s32 offset : {-0x80001, 0x80000, -0x800000, 0x7fffff}) {
        const std::array code{Offset(offset), Exit};
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(code)));
    }
    for (const u32 address : {1U, 243U, 244U, 254U}) {
        auto red = Offset(-16);
        red.words[0] = (red.words[0] & 0x00ffffffU) | (address << 24);
        const std::array code{red, Exit};
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(code)));
    }
    for (const u32 data : {244U, 250U}) {
        auto red = Offset(-16);
        red.words[1] = (red.words[1] & ~255U) | data;
        const std::array code{red, Exit};
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(code)));
    }
    for (const u32 flag : {1U << 9, 1U << 10, 1U << 27, 1U << 28}) {
        auto red = Offset(-16);
        red.words[2] ^= flag;
        const std::array code{red, Exit};
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(code)));
    }
}
} // namespace
