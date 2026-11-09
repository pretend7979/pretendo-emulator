// SPDX-License-Identifier: GPL-3.0-or-later
#include <array>
#include <span>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction Lea{{0x01017211, 0x80000000, 0x078ff0ff, 0x000fca00}};
constexpr Instruction Add{{0x0f067290, 0x00000006, 0x0ff5e03f, 0x000fe200}};
constexpr Instruction F2i{{0x00037305, 0x80000013, 0x0021f100, 0x000fe200}};
u32 Bits(u64 value, u32 bit, u32 width) {
    return (value >> bit) & ((1ULL << width) - 1);
}
void Set(Instruction& instruction, u32 bit, u32 width, u32 value) {
    const u32 mask = ((1U << width) - 1) << (bit % 32);
    auto& word = instruction.words[bit / 32];
    word = (word & ~mask) | ((value << (bit % 32)) & mask);
}
struct Machine {
    std::array<u32, 256> registers{};
    std::array<bool, 8> predicates{};
    u32 Read(u32 index) const {
        return index == 255 ? 0 : registers[index];
    }
    bool Predicate(u32 encoded) const {
        return ((encoded & 7) == 7 || predicates[encoded & 7]) != bool(encoded & 8);
    }
};
void Execute(Machine& state, std::span<const u64> words, bool stop_at_f2i = false) {
    for (const u64 word : words) {
        if ((word >> 56) == 0xe3)
            continue;
        if (!state.Predicate(Bits(word, 16, 4)))
            continue;
        const u32 op = word >> 52;
        const u32 destination = word & 255;
        const u32 a = state.Read(Bits(word, 8, 8)), b = state.Read(Bits(word, 20, 8));
        if (op == 0x5cb && stop_at_f2i)
            return;
        if (op == 0x382) {
            const u32 count = Bits(word, 20, 19);
            REQUIRE(count < 32);
            state.registers[destination] =
                Bits(word, 48, 1) ? static_cast<s32>(a) >> count : a >> count;
        } else if ((word & 0xfff8000000000000ULL) == 0x5c18000000000000ULL) {
            REQUIRE(Bits(word, 47, 3) == 0);
            state.registers[destination] = (a << Bits(word, 39, 5)) + b;
        } else if (op == 0x5c1) {
            REQUIRE(Bits(word, 43, 1) == 0);
            REQUIRE(Bits(word, 47, 1) == 0);
            state.registers[destination] = a + (Bits(word, 48, 1) ? 0U - b : b);
        } else if (op == 0x5c9) {
            state.registers[destination] = b;
        } else if ((word >> 56) == 1) {
            state.registers[destination] = Bits(word, 20, 32);
        } else if (op == 0x5ca) {
            state.registers[destination] = state.Predicate(Bits(word, 39, 4)) ? a : b;
        } else if ((word & 0xfff8000000000000ULL) == 0x38e8000000000000ULL) {
            const u32 mask = Bits(word, 20, 19);
            u32 value = a & ~mask;
            for (u32 i = 0; i < 7; ++i)
                if ((mask >> i) & 1)
                    value |= u32{state.predicates[i]} << i;
            state.registers[destination] = value;
        } else if ((word & 0xfff8000000000000ULL) == 0x38f0000000000000ULL) {
            const u32 mask = Bits(word, 20, 19);
            for (u32 i = 0; i < 7; ++i)
                if ((mask >> i) & 1)
                    state.predicates[i] = (a >> i) & 1;
        } else if (op == 0x5b6) {
            REQUIRE(Bits(word, 49, 3) == 1);
            REQUIRE(Bits(word, 48, 1) == 0);
            REQUIRE(Bits(word, 39, 4) == 7);
            const bool less = a < b;
            if (Bits(word, 3, 3) != 7)
                state.predicates[Bits(word, 3, 3)] = less;
            if (Bits(word, 0, 3) != 7)
                state.predicates[Bits(word, 0, 3)] = !less;
        } else if ((word >> 56) == 4) {
            const u32 immediate = Bits(word, 20, 32);
            const u32 mode = Bits(word, 53, 2);
            REQUIRE((mode == 0 || mode == 2));
            state.registers[destination] = mode == 0 ? a & immediate : a ^ immediate;
        } else {
            FAIL("Unexpected instruction in compute regression interpreter");
        }
    }
}
} // namespace

TEST_CASE("OOT high LEA applies negative base after shifted high word", "[ampere][compute]") {
    for (const bool negative : {false, true}) {
        auto instruction = Lea;
        Set(instruction, 63, 1, negative);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        REQUIRE(static_cast<bool>(lowered));
        REQUIRE(lowered.instructions.size() == 3);
        REQUIRE(Bits(lowered.instructions[1], 48, 1) == negative);
        for (const u32 a : {0U, 1U, 4U, 0x80000000U, 0xffffffffU}) {
            for (const u32 b : {0U, 1U, 0x7fffffffU, 0xffffffffU}) {
                Machine state;
                state.registers[1] = a;
                state.registers[0] = b;
                Execute(state, lowered.instructions);
                const u32 high = (u64{a} << 30) >> 32;
                REQUIRE(state.registers[1] == (negative ? high - b : high + b));
                REQUIRE(state.registers[0] == b);
            }
        }
    }
    auto invalid = Lea;
    Set(invalid, 62, 1, 1);
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{invalid, Exit})));
    invalid = Lea;
    Set(invalid, 72, 1, 1);
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{invalid, Exit})));
}

TEST_CASE("OOT uniform adds write UP2 through UP6 without clobbering predicates",
          "[ampere][compute]") {
    for (u32 output = 0; output < 7; ++output) {
        auto instruction = Add;
        Set(instruction, 81, 3, output);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        REQUIRE(static_cast<bool>(lowered));
        for (const u32 a : {0U, 1U, 0x80000000U, 0xffffffffU}) {
            for (const u32 b : {0U, 1U, 0x80000000U, 0xffffffffU}) {
                for (u32 predicates = 0; predicates < 128; ++predicates) {
                    Machine state;
                    state.registers[231] = a; // UR15
                    state.registers[222] = b; // UR6, also the destination
                    state.registers[239] = 0xa55a0055;
                    for (u32 i = 0; i < 7; ++i)
                        state.predicates[i] = (predicates >> i) & 1;
                    const auto before = state.predicates;
                    Execute(state, lowered.instructions);
                    const u64 sum = u64{a} + b;
                    const u32 mask = 1U << (6 - output);
                    REQUIRE(state.registers[222] == static_cast<u32>(sum));
                    REQUIRE(state.registers[239] ==
                            ((0xa55a0055U & ~mask) | (sum >> 32 ? mask : 0U)));
                    REQUIRE(state.predicates == before);
                }
            }
        }
    }
}

TEST_CASE("OOT F2I applies source negation before integer conversion", "[ampere][compute]") {
    for (const u32 modifiers : {0U, 0x40000000U, 0x80000000U, 0xc0000000U}) {
        auto instruction = F2i;
        instruction.words[1] = 19 | modifiers;
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        REQUIRE(static_cast<bool>(lowered));
        for (const u32 input : {0U, 0x80000000U, 0x40600000U, 0xc0600000U, 0x7fc00001U}) {
            Machine state;
            state.registers[19] = input;
            Execute(state, lowered.instructions, true);
            u32 modified = input;
            if (modifiers & 0x40000000U)
                modified &= 0x7fffffff;
            if (modifiers & 0x80000000U)
                modified ^= 0x80000000;
            const u64 conversion = lowered.instructions[lowered.instructions.size() - 2];
            REQUIRE((conversion >> 52) == 0x5cb);
            REQUIRE(state.Read(Bits(conversion, 20, 8)) == modified);
            REQUIRE(Bits(conversion, 8, 2) == 2);
            REQUIRE(Bits(conversion, 12, 1) == 1);
            REQUIRE(Bits(conversion, 39, 2) == 3);
            REQUIRE(Bits(conversion, 44, 1) == 1);
            REQUIRE(state.registers[19] == input);
        }
    }
    auto invalid = F2i;
    invalid.words[1] |= 0x20000000;
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{invalid, Exit})));
}

TEST_CASE("Uniform extended adds read any UP carry and preserve its state", "[ampere][compute]") {
    for (u32 condition = 0; condition < 7; ++condition) {
        for (const bool negate : {false, true}) {
            auto instruction = Add;
            Set(instruction, 74, 1, 1);
            Set(instruction, 81, 3, 7);
            Set(instruction, 87, 3, condition);
            Set(instruction, 90, 1, negate);
            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
            REQUIRE(static_cast<bool>(lowered));
            for (const u32 a : {0U, 1U, 0xffffffffU}) {
                for (const u32 b : {0U, 1U, 0xffffffffU}) {
                    for (u32 predicates = 0; predicates < 128; ++predicates) {
                        Machine state;
                        state.registers[231] = a;
                        state.registers[222] = b;
                        state.registers[239] = 0xa55a0000 | predicates;
                        for (u32 i = 0; i < 7; ++i)
                            state.predicates[i] = (predicates >> ((i + 2) % 7)) & 1;
                        const auto before = state.predicates;
                        Execute(state, lowered.instructions);
                        const bool carry = bool((predicates >> (6 - condition)) & 1) != negate;
                        REQUIRE(state.registers[222] == a + b + u32{carry});
                        REQUIRE(state.registers[239] == (0xa55a0000U | predicates));
                        REQUIRE(state.predicates == before);
                    }
                }
            }
        }
    }
    auto invalid = Add;
    Set(invalid, 74, 1, 1);
    Set(invalid, 81, 3, 7);
    Set(invalid, 87, 3, 2);
    Set(invalid, 12, 4, 0); // Guard and carry both need the R253 predicate save slot.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{invalid, Exit})));
}

TEST_CASE("Uniform LEA writes every UP carry without clobbering predicates", "[ampere][compute]") {
    constexpr Instruction lea{{0x060a7291, 0x0000000a, 0x0f84103f, 0x000fe200}};
    for (u32 output = 0; output < 7; ++output) {
        for (const u32 shift : {0U, 2U, 31U}) {
            auto instruction = lea;
            Set(instruction, 81, 3, output);
            Set(instruction, 75, 5, shift);
            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
            REQUIRE(static_cast<bool>(lowered));
            for (const u32 a : {0U, 1U, 0x7fffffffU, 0x80000000U, 0xffffffffU}) {
                for (const u32 b : {0U, 1U, 0x7fffffffU, 0xffffffffU}) {
                    for (u32 predicates = 0; predicates < 128; ++predicates) {
                        Machine state;
                        state.registers[222] = a; // UR6
                        state.registers[226] = b; // UR10, also the destination
                        const u32 uniform_predicates = 0xa55a0000U | predicates;
                        state.registers[239] = uniform_predicates;
                        for (u32 i = 0; i < 7; ++i)
                            state.predicates[i] = (predicates >> ((i + 2) % 7)) & 1;
                        const auto before = state.predicates;
                        Execute(state, lowered.instructions);
                        const u64 sum = u64{static_cast<u32>(a << shift)} + b;
                        const u32 mask = 1U << (6 - output);
                        REQUIRE(state.registers[226] == static_cast<u32>(sum));
                        REQUIRE(state.registers[239] ==
                                ((uniform_predicates & ~mask) | (sum >> 32 ? mask : 0U)));
                        REQUIRE(state.predicates == before);
                        REQUIRE(state.registers[222] == a);
                    }
                }
            }
        }
    }
}

TEST_CASE("Captured uniform LEA carry feeds the following high-word add", "[ampere][compute]") {
    constexpr std::array pairs{
        std::array{Instruction{{0x060a7291, 0x0000000a, 0x0f84103f, 0x000fe200}},
                   Instruction{{0x3f0b7290, 0x0000000b, 0x097fe43f, 0x000fe200}}},
        std::array{Instruction{{0x0b047291, 0x00000004, 0x0f84103f, 0x000fc600}},
                   Instruction{{0x3f057290, 0x00000005, 0x097fe43f, 0x000fe400}}},
        std::array{Instruction{{0x070a7291, 0x0000000a, 0x0f86103f, 0x000fe200}},
                   Instruction{{0x3f0b7290, 0x0000000b, 0x09ffe43f, 0x000fe200}}}};
    for (const auto& pair : pairs) {
        const auto lowered = LowerToMaxwell(std::array{pair[0], pair[1], Exit});
        REQUIRE(static_cast<bool>(lowered));
        const u32 a_register = 216 + pair[0].Bits(24, 8);
        const u32 low_register = 216 + pair[0].Bits(16, 8);
        const u32 high_register = 216 + pair[1].Bits(16, 8);
        for (const u32 a : {0U, 1U, 0x3fffffffU}) {
            for (const u32 low : {0U, 4U, 0xfffffff8U, 0xffffffffU}) {
                for (const u32 high : {0U, 0x12345678U, 0xffffffffU}) {
                    Machine state;
                    state.registers[a_register] = a;
                    state.registers[low_register] = low;
                    state.registers[high_register] = high;
                    state.registers[239] = 0x5aa50055;
                    for (u32 i = 0; i < 7; ++i)
                        state.predicates[i] = i & 1;
                    const auto before = state.predicates;
                    Execute(state, lowered.instructions);
                    const u64 sum = u64{a << 2} + low;
                    const u32 mask = 1U << (6 - pair[0].Bits(81, 3));
                    REQUIRE(state.registers[low_register] == static_cast<u32>(sum));
                    REQUIRE(state.registers[high_register] == high + static_cast<u32>(sum >> 32));
                    REQUIRE(state.registers[239] ==
                            ((0x5aa50055U & ~mask) | (sum >> 32 ? mask : 0U)));
                    REQUIRE(state.predicates == before);
                }
            }
        }
    }
}

TEST_CASE("Uniform LEA carry retains unsupported mode and guard rejection", "[ampere][compute]") {
    constexpr Instruction lea{{0x060a7291, 0x0000000a, 0x0f84103f, 0x000fe200}};
    for (u32 output = 0; output < 7; ++output) {
        auto instruction = lea;
        Set(instruction, 81, 3, output);
        for (const u32 bit : {62U, 63U, 72U, 74U, 80U}) {
            auto invalid = instruction;
            Set(invalid, bit, 1, 1);
            REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{invalid, Exit})));
        }
        for (u32 guard = 0; guard < 16; ++guard) {
            auto guarded = instruction;
            Set(guarded, 12, 4, guard);
            const auto lowered = LowerToMaxwell(std::array{guarded, Exit});
            REQUIRE(static_cast<bool>(lowered) == (guard == 7 || guard == 15));
            if (guard == 15) {
                // @!PT never executes, so it must not write the result or either predicate bank.
                Machine state;
                state.registers.fill(0xa55a0055);
                state.predicates.fill(true);
                const auto before = state;
                Execute(state, lowered.instructions);
                REQUIRE(state.registers == before.registers);
                REQUIRE(state.predicates == before.predicates);
            }
        }
        auto immediate = instruction;
        Set(immediate, 0, 12, 0x891);
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{immediate, Exit})));
    }
}
