// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <random>
#include <stdexcept>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction FloatRed{{0xff00798e, 0, 0x0c10e704, 0x000fe200}};

u32 ExecuteReduction(const Shader::Ampere::LoweredProgram& lowered,
                     const std::array<u32, 32>& lanes, u32 active) {
    std::array<u32, 256> r{};
    for (const u64 w : lowered.instructions) {
        const u32 d = w & 255, a = (w >> 8) & 255, b = (w >> 20) & 255;
        if ((w >> 56) == 0xe3)
            continue;
        if ((w >> 56) == 0x01)
            r[d] = (w >> 20) & 0xffffffff;
        else if ((w & 0xfffe000000000000ULL) == 0x50d8000000000000ULL)
            r[d] = active;
        else if ((w >> 52) == 0xef1)
            r[d] = lanes[b & 31];
        else if ((w >> 48) == 0x3828)
            r[d] = r[a] >> (b & 31);
        else if ((w >> 56) == 0x04)
            r[d] = r[a] & ((w >> 20) & 0xffffffff);
        else if ((w >> 52) == 0x5c1)
            r[d] = r[a] + (((w >> 48) & 1) ? 0U - r[b] : r[b]);
        else if ((w >> 52) == 0x5c4) {
            const u32 av = ((w >> 39) & 1) ? ~r[a] : r[a];
            const u32 bv = ((w >> 40) & 1) ? ~r[b] : r[b];
            switch ((w >> 41) & 3) {
            case 0:
                r[d] = av & bv;
                break;
            case 1:
                r[d] = av | bv;
                break;
            case 2:
                r[d] = av ^ bv;
                break;
            default:
                throw std::runtime_error("Unsupported LOP in reduction test");
            }
        } else if ((w >> 52) == 0x5c2) {
            r[d] = ((w >> 42) & 1) ? (std::max)(r[a], r[b]) : (std::min)(r[a], r[b]);
        } else
            throw std::runtime_error(
                fmt::format("Unexpected emitted reduction instruction {:016x}", w));
        r[255] = 0;
    }
    return r[221]; // UR5's Maxwell mirror.
}

u32 ReferenceReduction(const std::array<u32, 32>& lanes, u32 active, u32 mode) {
    if (mode == 0x10200 || mode == 0x14200) {
        s32 result =
            mode == 0x10200 ? (std::numeric_limits<s32>::max)() : (std::numeric_limits<s32>::min)();
        for (u32 i = 0; i < 32; ++i)
            if (active & (1U << i)) {
                const s32 v = std::bit_cast<s32>(lanes[i]);
                result = mode == 0x10200 ? (std::min)(result, v) : (std::max)(result, v);
            }
        return std::bit_cast<u32>(result);
    }
    u32 result = mode == 0x10000 ? ~0U : 0U;
    for (u32 i = 0; i < 32; ++i)
        if (active & (1U << i)) {
            if (mode == 0xc000)
                result += lanes[i];
            else if (mode == 0x4000)
                result |= lanes[i];
            else if (mode == 0x8000)
                result ^= lanes[i];
            else if (mode == 0x10000)
                result = (std::min)(result, lanes[i]);
            else if (mode == 0x14000)
                result = (std::max)(result, lanes[i]);
            else
                throw std::runtime_error("Unknown reference reduction");
        }
    return result;
}
} // namespace

TEST_CASE("OOT uniform-address floating reduction emits F32 atomic add", "[video_core][ampere]") {
    for (const u32 data : {0U, 1U, 3U, 7U, 9U, 255U}) {
        auto red = FloatRed;
        red.words[1] = data;
        const auto result = LowerToMaxwell(std::array{red, Exit});
        REQUIRE(static_cast<bool>(result));
        REQUIRE(result.instructions.size() == 2);
        const u64 atomic = result.instructions[0];
        REQUIRE((atomic >> 52) == 0xebf);
        REQUIRE((atomic & 255) == data);
        REQUIRE(((atomic >> 8) & 255) == 220); // UR4:UR5 address pair.
        REQUIRE(((atomic >> 20) & 7) == 3);    // Maxwell F32.
        REQUIRE(((atomic >> 23) & 7) == 0);    // ADD.
        REQUIRE(((atomic >> 48) & 1) == 1);    // 64-bit address.
    }
    auto offset = FloatRed;
    offset.words[1] = 0xfffffc09; // ADD F32 [UR4 - 4], R9.
    const auto result = LowerToMaxwell(std::array{offset, Exit});
    REQUIRE(static_cast<bool>(result));
    REQUIRE(((result.instructions[0] >> 28) & 0xfffff) == 0xffffc);
}

TEST_CASE("Uniform-address reductions retain U32 operations and strict type guards",
          "[video_core][ampere]") {
    for (u32 op = 0; op < 8; ++op) {
        auto red = FloatRed;
        red.words[2] = 0x0c10e104 | (op << 23);
        const auto result = LowerToMaxwell(std::array{red, Exit});
        REQUIRE(static_cast<bool>(result));
        REQUIRE(((result.instructions[0] >> 20) & 7) == 0);
        REQUIRE(((result.instructions[0] >> 23) & 7) == op);
    }
    for (const u32 type : {2U, 4U, 5U, 6U, 7U}) {
        auto bad = FloatRed;
        bad.words[2] = (bad.words[2] & ~0xe00U) | (type << 9);
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    }
    for (const u32 op : {1U, 2U, 5U, 7U}) {
        auto bad = FloatRed;
        bad.words[2] |= op << 23;
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    }
    auto bad = FloatRed;
    bad.words[2] = (bad.words[2] & ~63U) | 5U; // Odd address pair.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    bad = FloatRed;
    bad.words[1] = 216; // Native data may not alias a uniform mirror.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    bad = FloatRed;
    bad.words[1] = 0x08000000; // Signed offset exceeds Maxwell's 20-bit field.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    bad = FloatRed;
    bad.words[2] ^= 0x2000; // Different ordering mode remains unsupported.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
}

TEST_CASE("OOT signed global min and max map the integer signedness", "[video_core][ampere]") {
    for (const u32 op : {1U, 2U}) {
        Instruction red{{0xff00098e, 1, 0x0c10e304U | (op << 23), 0x000fe200}};
        const auto result = LowerToMaxwell(std::array{red, Exit});
        REQUIRE(static_cast<bool>(result));
        REQUIRE(result.instructions.size() == 2);
        const u64 atomic = result.instructions[0];
        REQUIRE(((atomic >> 20) & 7) == 1); // Maxwell S32 affects MIN/MAX ordering.
        REQUIRE(((atomic >> 23) & 7) == op);
        REQUIRE(((atomic >> 16) & 15) == 0); // Original P0 guard.
        REQUIRE((atomic & 255) == 1);
    }
    for (const u32 op : {0U, 3U, 4U, 5U, 6U, 7U}) {
        const Instruction red{{0xff00098e, 1, 0x0c10e304U | (op << 23), 0x000fe200}};
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{red, Exit})));
    }
}

TEST_CASE("Signed warp min and max preserve inactive lanes and extreme values",
          "[video_core][ampere]") {
    std::mt19937 random(0x51a7);
    for (const u32 mode : {0xc000U, 0x4000U, 0x8000U, 0x10000U, 0x14000U, 0x10200U, 0x14200U}) {
        const Instruction redux{{0x000573c4, 0, mode, 0x001e2400}};
        const auto result = LowerToMaxwell(std::array{redux, Exit});
        REQUIRE(static_cast<bool>(result));
        for (u32 trial = 0; trial < 40; ++trial) {
            std::array<u32, 32> values;
            for (auto& v : values)
                v = random();
            if (trial == 0)
                values.fill(0xffffffffU);
            if (trial == 1)
                values.fill(0x80000000U);
            if (trial == 2)
                values.fill(0x7fffffffU);
            values[3] = 0x80000000U;
            values[8] = 0x7fffffffU;
            const u32 active = trial < 4 ? 1U << trial : trial == 4 ? ~0U : random() | 1U;
            CAPTURE(mode, trial, active);
            REQUIRE(ExecuteReduction(result, values, active) ==
                    ReferenceReduction(values, active, mode));
        }
    }
}

TEST_CASE("Warp reductions reject unsupported signed operations and flags",
          "[video_core][ampere]") {
    for (const u32 mode : {0x200U, 0x4200U, 0x8200U, 0xc200U, 0x30200U, 0x110200U}) {
        const Instruction bad{{0x000573c4, 0, mode, 0x001e2400}};
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    }
    REQUIRE_FALSE(static_cast<bool>(
        LowerToMaxwell(std::array{Instruction{{0x000503c4, 0, 0x10200, 0x001e2400}}, Exit})));
}

TEST_CASE("Uniform reduction address pairs cannot substitute an undefined URZ mirror",
          "[video_core][ampere]") {
    for (const u32 mode : {0x0c10e100U, 0x0c90e300U, 0x0c10e700U}) {
        const Instruction valid{{0xff00798e, 0, mode | 4U, 0x000fe200}};
        const auto lowered = LowerToMaxwell(std::array{valid, Exit});
        REQUIRE(static_cast<bool>(lowered));
        REQUIRE(((lowered.instructions[0] >> 8) & 255) == 220); // UR4:UR5 stays adjacent.
        const Instruction invalid{{0xff00798e, 0, mode | 62U, 0x000fe200}};
        // UR62 maps to R214 after retry, but native UR63 is RZ, not R215.
        const auto rejected = LowerToMaxwell(std::array{invalid, Exit});
        REQUIRE_FALSE(static_cast<bool>(rejected));
        REQUIRE(rejected.instructions.empty());
    }
}
