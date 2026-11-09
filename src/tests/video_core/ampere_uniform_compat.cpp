// SPDX-License-Identifier: GPL-3.0-or-later
#include <array>
#include <bit>
#include <span>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
u32 Bits(u64 word, u32 bit, u32 width) {
    return (word >> bit) & ((u64{1} << width) - 1);
}
void Set(Instruction& instruction, u32 bit, u32 width, u32 value) {
    const u32 mask = ((1U << width) - 1) << (bit % 32);
    auto& word = instruction.words[bit / 32];
    word = (word & ~mask) | ((value << (bit % 32)) & mask);
}
bool Compare(u32 a, u32 b, u32 op, bool signed_order) {
    const bool less = signed_order ? static_cast<s32>(a) < static_cast<s32>(b) : a < b;
    switch (op) {
    case 1:
        return less;
    case 2:
        return a == b;
    case 3:
        return less || a == b;
    case 4:
        return !less && a != b;
    case 5:
        return a != b;
    case 6:
        return !less;
    default:
        FAIL("Unsupported comparison");
        return false;
    }
}
bool Combine(bool a, bool b, u32 op) {
    return op == 0 ? a && b : op == 1 ? a || b : a != b;
}
struct Machine {
    std::array<u32, 256> r{};
    std::array<bool, 8> p{};
    u32 Read(u32 index) const {
        return index == 255 ? 0 : r[index];
    }
    bool Predicate(u32 index) const {
        return ((index & 7) == 7 || p[index & 7]) != bool(index & 8);
    }
};
void Execute(Machine& m, std::span<const u64> code) {
    for (const u64 w : code) {
        if ((w >> 56) == 0xe3 || !m.Predicate(Bits(w, 16, 4)))
            continue;
        const u32 dst = w & 255, a = m.Read(Bits(w, 8, 8)), b = m.Read(Bits(w, 20, 8));
        if ((w >> 56) == 1)
            m.r[dst] = Bits(w, 20, 32);
        else if ((w & 0xfff8000000000000ULL) == 0x38e8000000000000ULL) {
            const u32 mask = Bits(w, 20, 19);
            u32 result = a & ~mask;
            for (u32 i = 0; i < 7; ++i)
                if ((mask >> i) & 1)
                    result |= u32{m.p[i]} << i;
            m.r[dst] = result;
        } else if ((w & 0xfff8000000000000ULL) == 0x38f0000000000000ULL) {
            const u32 mask = Bits(w, 20, 19);
            for (u32 i = 0; i < 7; ++i)
                if ((mask >> i) & 1)
                    m.p[i] = (a >> i) & 1;
        } else if ((w >> 52) == 0x5b6) {
            const bool compare = Compare(a, b, Bits(w, 49, 3), Bits(w, 48, 1));
            const bool result = Combine(compare, m.Predicate(Bits(w, 39, 4)), Bits(w, 45, 2));
            if (Bits(w, 3, 3) != 7)
                m.p[Bits(w, 3, 3)] = result;
            if (Bits(w, 0, 3) != 7)
                m.p[Bits(w, 0, 3)] = !result;
        } else if ((w >> 52) == 0x5ca) {
            m.r[dst] = m.Predicate(Bits(w, 39, 4)) ? a : b;
        } else if ((w & 0xfff8000000000000ULL) == 0x5c48000000000000ULL) {
            const u32 shift = Bits(w, 39, 1) ? b & 31 : b;
            m.r[dst] = shift < 32 ? a << shift : 0;
        } else if ((w >> 52) == 0x5c9)
            m.r[dst] = b;
        else
            FAIL("Unexpected uniform regression instruction " << std::hex << w);
    }
}
Machine Initial(u32 uniform_bits) {
    Machine m;
    m.r[239] = 0xa5a50000 | uniform_bits;
    for (u32 i = 0; i < 7; ++i)
        m.p[i] = (0x53 >> i) & 1;
    return m;
}
} // namespace

TEST_CASE("Guarded uniform compares preserve false outputs and all ordinary predicates",
          "[ampere][uniform]") {
    for (u32 output = 0; output < 7; ++output) {
        for (const u32 guard : {0U, 4U, 6U, 7U, 8U, 12U, 14U}) {
            for (u32 accum = 0; accum < 8; ++accum) {
                for (u32 combine = 0; combine < 3; ++combine) {
                    auto instruction = Instruction{{0x1500488c, 0x1f, 0x0bf04070, 0x000fe200}};
                    Set(instruction, 12, 4, guard);
                    Set(instruction, 81, 3, output);
                    Set(instruction, 87, 3, accum);
                    Set(instruction, 74, 2, combine);
                    Set(instruction, 90, 1, output & 1);
                    const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                    INFO(lowered.error);
                    REQUIRE(static_cast<bool>(lowered));
                    for (const u32 uniform : {0U, 0x7fU, 0x55U, 0x2aU}) {
                        for (const u32 value : {0U, 32U}) {
                            auto m = Initial(uniform);
                            m.r[237] = value;
                            const auto original = m;
                            const bool enabled =
                                guard == 7 ||
                                (((uniform >> (6 - (guard & 7))) & 1) != bool(guard & 8));
                            const bool accumulator =
                                (accum == 7 || ((uniform >> (6 - accum)) & 1)) != bool(output & 1);
                            const bool result = Combine(value > 31, accumulator, combine);
                            const u32 mask = 1U << (6 - output);
                            const u32 expected =
                                enabled ? (original.r[239] & ~mask) | (result ? mask : 0U)
                                        : original.r[239];
                            Execute(m, lowered.instructions);
                            REQUIRE(m.r[239] == expected);
                            REQUIRE(m.p == original.p);
                            REQUIRE(m.r[237] == value);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Guarded uniform select preserves destination and predicate aliases",
          "[ampere][uniform]") {
    for (u32 condition = 0; condition < 7; ++condition) {
        for (const u32 guard : {0U, 4U, 6U, 7U, 8U, 12U, 14U}) {
            for (const bool invert : {false, true}) {
                auto instruction = Instruction{{0x04040887, 0x20, 0x08800000, 0x000fe200}};
                Set(instruction, 12, 4, guard);
                Set(instruction, 87, 3, condition);
                Set(instruction, 90, 1, invert);
                const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                INFO(lowered.error);
                REQUIRE(static_cast<bool>(lowered));
                for (const u32 uniform : {0U, 0x7fU, 0x55U, 0x2aU}) {
                    auto m = Initial(uniform);
                    m.r[220] = 0xdeadbeef;
                    const auto original = m;
                    const bool enabled =
                        guard == 7 || (((uniform >> (6 - (guard & 7))) & 1) != bool(guard & 8));
                    const bool selected = (((uniform >> (6 - condition)) & 1) != invert);
                    Execute(m, lowered.instructions);
                    REQUIRE(m.r[220] == (enabled && !selected ? 32U : original.r[220]));
                    REQUIRE(m.r[239] == original.r[239]);
                    REQUIRE(m.p == original.p);
                }
            }
        }
    }
}

TEST_CASE("Uniform immediate shifts use UP guards and native clamping", "[ampere][uniform]") {
    for (const u32 guard : {0U, 4U, 7U, 8U, 12U}) {
        for (const bool wrap : {false, true}) {
            auto instruction = Instruction{{0x3f147499, 0xffffffff, 0x08010615, 0x000fe200}};
            Set(instruction, 12, 4, guard);
            Set(instruction, 75, 1, wrap);
            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
            INFO(lowered.error);
            REQUIRE(static_cast<bool>(lowered));
            for (const u32 uniform : {0U, 0x7fU}) {
                for (const u32 shift : {0U, 1U, 31U, 32U, 63U, 64U, 0xffffffffU}) {
                    auto m = Initial(uniform);
                    m.r[236] = 0x12345678;
                    m.r[237] = shift;
                    const auto original = m;
                    const bool enabled =
                        guard == 7 || (((uniform >> (6 - (guard & 7))) & 1) != bool(guard & 8));
                    const u32 count = wrap ? shift & 31 : shift;
                    const u32 expected = count < 32 ? 0xffffffffU << count : 0U;
                    Execute(m, lowered.instructions);
                    REQUIRE(m.r[236] == (enabled ? expected : original.r[236]));
                    REQUIRE(m.r[239] == original.r[239]);
                    REQUIRE(m.p == original.p);
                }
            }
        }
    }
}

TEST_CASE("Uniform bit operations and scalar selects address high uniform registers",
          "[ampere][uniform]") {
    for (const u32 opcode : {0x2bdU, 0x2beU, 0x2bfU}) {
        auto instruction =
            Instruction{{0x00117200 | opcode, 34, opcode == 0x2bd ? 0x080e0000U : 0x08000000U, 0}};
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        INFO(lowered.error);
        REQUIRE(static_cast<bool>(lowered));
        const u64 word = lowered.instructions.front();
        REQUIRE((word & 255) == 233);
        REQUIRE(Bits(word, opcode == 0x2be ? 8 : 20, 8) == 214);
        Set(instruction, 16, 8, 63);
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{instruction, Exit})));
    }
    const Instruction select{{0x04037c07, 7, 0x08000000, 0x000fe400}};
    const auto lowered = LowerToMaxwell(std::array{select, Exit});
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(lowered.instructions.front() == 0x5ca000000df70403ULL);
    for (const bool predicate : {false, true}) {
        Machine m;
        m.r[4] = 0x12345678;
        m.r[223] = 0x87654321;
        m.p[0] = predicate;
        Execute(m, lowered.instructions);
        REQUIRE(m.r[3] == (predicate ? m.r[4] : m.r[223]));
    }
}
