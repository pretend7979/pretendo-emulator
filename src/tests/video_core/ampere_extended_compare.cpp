// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdint>
#include <span>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};

u32 Bits(u64 value, u32 offset, u32 count) {
    return (value >> offset) & ((1ULL << count) - 1);
}

void SetBits(Instruction& instruction, u32 offset, u32 count, u32 value) {
    const u32 shift = offset % 32;
    const u32 mask = ((1U << count) - 1) << shift;
    auto& word = instruction.words[offset / 32];
    word = (word & ~mask) | ((value << shift) & mask);
}

struct Machine {
    std::array<u32, 256> registers{};
    std::array<bool, 8> predicates{};

    bool Predicate(u32 encoded) const {
        return ((encoded & 7) == 7 || predicates[encoded & 7]) != bool(encoded & 8);
    }

    u32 Register(u32 reg) const {
        return reg == 255 ? 0U : registers[reg];
    }
};

bool Combine(bool a, bool b, u32 op) {
    switch (op) {
    case 0:
        return a && b;
    case 1:
        return a || b;
    case 2:
        return a != b;
    default:
        FAIL("Reserved predicate operation");
        return false;
    }
}

bool Compare(u32 a, u32 b, u32 op, bool is_signed) {
    const bool less = is_signed ? static_cast<s32>(a) < static_cast<s32>(b) : a < b;
    const bool equal = a == b;
    switch (op) {
    case 0:
        return false;
    case 1:
        return less;
    case 2:
        return equal;
    case 3:
        return less || equal;
    case 4:
        return !less && !equal;
    case 5:
        return !equal;
    case 6:
        return !less;
    case 7:
        return true;
    default:
        FAIL("Invalid comparison");
        return false;
    }
}

void Execute(Machine& machine, std::span<const u64> instructions) {
    for (const u64 word : instructions) {
        if ((word >> 56) == 0xe3)
            continue; // EXIT
        if (!machine.Predicate(Bits(word, 16, 4)))
            continue;
        const u32 opcode = word >> 52;
        if (opcode == 0x39a || opcode == 0x38a) { // SEL, signed 20-bit immediate.
            u32 immediate = Bits(word, 20, 19);
            if (Bits(word, 56, 1))
                immediate |= 0xfff80000U;
            const u32 result = machine.Predicate(Bits(word, 39, 4))
                                   ? machine.Register(Bits(word, 8, 8))
                                   : immediate;
            const u32 destination = Bits(word, 0, 8);
            if (destination != 255)
                machine.registers[destination] = result;
        } else if (opcode == 0x5b6) { // ISETP, register operands, no CC extension.
            REQUIRE(Bits(word, 43, 1) == 0);
            const bool comparison =
                Compare(machine.Register(Bits(word, 8, 8)), machine.Register(Bits(word, 20, 8)),
                        Bits(word, 49, 3), Bits(word, 48, 1));
            const bool input = machine.Predicate(Bits(word, 39, 4));
            const bool first = Combine(comparison, input, Bits(word, 45, 2));
            const bool second = Combine(!comparison, input, Bits(word, 45, 2));
            const u32 first_destination = Bits(word, 3, 3);
            const u32 second_destination = Bits(word, 0, 3);
            // All inputs are read before either predicate output is written.
            if (first_destination != 7)
                machine.predicates[first_destination] = first;
            if (second_destination != 7)
                machine.predicates[second_destination] = second;
        } else {
            FAIL("Interpreter encountered an unexpected instruction");
        }
    }
}

std::array<bool, 8> Expected(const Instruction& instruction, const Machine& before) {
    auto predicates = before.predicates;
    if (!before.Predicate(instruction.Bits(12, 4)))
        return predicates;
    const u32 a = before.Register(instruction.Bits(24, 8));
    // These tests use c0c's uniform source, UR0-UR21 mirrored at R216-R237.
    REQUIRE(instruction.Bits(32, 8) < 22);
    const u32 b = before.Register(216 + instruction.Bits(32, 8));
    const u32 comparison = instruction.Bits(76, 3);
    const bool signed_high = instruction.Bits(73, 1);
    const bool low = before.Predicate(instruction.Bits(68, 4));
    bool extended{};
    if (comparison == 2) {
        extended = a == b && low;
    } else if (comparison == 5) {
        extended = a != b || low;
    } else {
        extended = Compare(a, b, comparison <= 3 ? 1U : 4U, signed_high) || (a == b && low);
    }
    const bool result =
        Combine(extended, before.Predicate(instruction.Bits(87, 4)), instruction.Bits(74, 2));
    const u32 destination = instruction.Bits(81, 3);
    if (destination != 7)
        predicates[destination] = result;
    return predicates;
}

Machine Inputs(u32 a, u32 b, u32 predicate_bits, u32 uniform_source = 9) {
    Machine machine;
    machine.registers[8] = a;
    machine.registers[216 + uniform_source] = b;
    for (u32 i = 0; i < 7; ++i)
        machine.predicates[i] = (predicate_bits >> i) & 1;
    machine.predicates[7] = true;
    return machine;
}

constexpr std::array Captured{
    Instruction{{0xff007c0c, 0x00000007, 0x08f05100, 0x000fe200}},
    Instruction{{0xff007c0c, 0x00000009, 0x08f25120, 0x000fda00}},
    Instruction{{0xff007c0c, 0x00000009, 0x08f25120, 0x000fe400}},
    Instruction{{0xff007c0c, 0x00000009, 0x08f25120, 0x000fe200}},
    Instruction{{0xff007c0c, 0x00000009, 0x08f25120, 0x000fc600}},
    Instruction{{0xff007c0c, 0x00000007, 0x08f05100, 0x000fe400}},
    Instruction{{0xff007c0c, 0x00000009, 0x08f05100, 0x000fda00}},
    Instruction{{0xff007c0c, 0x00000009, 0x09725110, 0x000fe200}},
    Instruction{{0xff007c0c, 0x00000009, 0x09f45120, 0x000fe400}},
    Instruction{{0xff007c0c, 0x00000009, 0x08705130, 0x000fda00}},
    Instruction{{0xff007c0c, 0x00000009, 0x08705130, 0x000fe200}},
    Instruction{{0xff007c0c, 0x00000009, 0x08f05100, 0x000fe400}},
    Instruction{{0xff007c0c, 0x00000009, 0x09f05100, 0x000fda00}},
};
} // namespace

TEST_CASE("Ampere ISETP.EX preserves captured accumulator chains", "[ampere][isetp][captured]") {
    for (const auto& instruction : Captured) {
        CAPTURE(instruction.words);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        REQUIRE(static_cast<bool>(lowered));
        for (const u32 b : {0U, 1U, 0x80000000U, 0xffffffffU}) {
            for (u32 bits = 0; bits < 128; ++bits) {
                CAPTURE(b, bits);
                auto machine = Inputs(0, b, bits, instruction.Bits(32, 8));
                const auto expected = Expected(instruction, machine);
                Execute(machine, lowered.instructions);
                CHECK(machine.predicates == expected);
            }
        }
    }
}

TEST_CASE("Ampere ISETP.EX keeps comparisons accumulators inversions and guards",
          "[ampere][isetp][behavior]") {
    struct PredicateInputs {
        u32 low, accumulator;
    };
    constexpr std::array inputs{
        PredicateInputs{2, 1}, PredicateInputs{1, 2},  PredicateInputs{1, 1},
        PredicateInputs{9, 9}, PredicateInputs{10, 9}, PredicateInputs{7, 15},
        PredicateInputs{8, 7}, PredicateInputs{2, 3},
    };
    constexpr std::array values{
        std::array<u32, 2>{0, 0},
        std::array<u32, 2>{0, 1},
        std::array<u32, 2>{1, 0},
        std::array<u32, 2>{1, 1},
        std::array<u32, 2>{0x80000000U, 0x7fffffffU},
        std::array<u32, 2>{0x7fffffffU, 0x80000000U},
        std::array<u32, 2>{0xffffffffU, 0},
        std::array<u32, 2>{0, 0xffffffffU},
        std::array<u32, 2>{0xffffffffU, 0xffffffffU},
    };
    for (u32 compare = 1; compare <= 6; ++compare) {
        for (u32 is_signed = 0; is_signed < 2; ++is_signed) {
            for (u32 combine = 0; combine < 3; ++combine) {
                for (const auto input : inputs) {
                    for (const u32 guard : {7U, 4U, 12U}) {
                        auto instruction = Captured[1];
                        SetBits(instruction, 24, 8, 8);
                        SetBits(instruction, 12, 4, guard);
                        SetBits(instruction, 68, 4, input.low);
                        SetBits(instruction, 73, 1, is_signed);
                        SetBits(instruction, 74, 2, combine);
                        SetBits(instruction, 76, 3, compare);
                        SetBits(instruction, 87, 4, input.accumulator);
                        CAPTURE(compare, is_signed, combine, input.low, input.accumulator, guard);
                        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                        REQUIRE(static_cast<bool>(lowered));
                        for (const auto value : values) {
                            for (u32 bits = 0; bits < 32; bits += 2) {
                                // Exhaust P1-P4; retain unrelated P0/P5/P6 values too.
                                auto machine = Inputs(value[0], value[1], bits | 0x61U);
                                const auto expected = Expected(instruction, machine);
                                Execute(machine, lowered.instructions);
                                CHECK(machine.predicates == expected);
                                CHECK(machine.registers[8] == value[0]);
                                CHECK(machine.registers[225] == value[1]);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Ampere ISETP.EX retains rejected modes", "[ampere][isetp][rejected]") {
    struct InvalidField {
        u32 offset, width, value;
    };
    constexpr std::array invalid{
        InvalidField{84, 3, 0}, InvalidField{74, 2, 3}, InvalidField{76, 3, 0},
        InvalidField{76, 3, 7}, InvalidField{12, 4, 1}, InvalidField{12, 4, 9},
    };
    for (const auto field : invalid) {
        auto instruction = Captured[1];
        SetBits(instruction, field.offset, field.width, field.value);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        CHECK_FALSE(static_cast<bool>(lowered));
        CHECK(lowered.error == "native extended integer comparison mode");
        CHECK(lowered.instructions.empty());
    }
}

TEST_CASE("Ampere ISETP.EX plain AND PT fast path stays unchanged", "[ampere][isetp][fastpath]") {
    for (u32 compare = 1; compare <= 6; ++compare) {
        for (u32 is_signed = 0; is_signed < 2; ++is_signed) {
            for (const u32 low : {1U, 9U, 7U, 8U}) {
                for (const u32 guard : {7U, 4U, 12U}) {
                    auto instruction = Captured[1];
                    SetBits(instruction, 24, 8, 8);
                    SetBits(instruction, 12, 4, guard);
                    SetBits(instruction, 68, 4, low);
                    SetBits(instruction, 73, 1, is_signed);
                    SetBits(instruction, 76, 3, compare);
                    SetBits(instruction, 87, 4, 7);
                    const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                    CAPTURE(compare, is_signed, low, guard, lowered.error);
                    REQUIRE(static_cast<bool>(lowered));
                    // Exact encoding of the unmodified lowerer's established fast path.
                    const auto old_isetp = [&](u32 op, u32 input, u32 combine) {
                        return 0x5b60000000000000ULL | (u64{guard} << 16) | 7 | (1ULL << 3) |
                               (8ULL << 8) | (225ULL << 20) | (u64{input & 7} << 39) |
                               (u64{input >> 3} << 42) | (u64{combine} << 45) |
                               (u64{is_signed} << 48) | (u64{op} << 49);
                    };
                    std::vector<u64> expected;
                    if (compare == 2)
                        expected.push_back(old_isetp(2, low, 0));
                    else if (compare == 5)
                        expected.push_back(old_isetp(5, low, 1));
                    else {
                        expected.push_back(old_isetp(2, low, 0));
                        expected.push_back(old_isetp(compare <= 3 ? 1U : 4U, 1, 1));
                    }
                    const auto exit_only = LowerToMaxwell(std::array{Exit});
                    expected.push_back(exit_only.instructions.at(0));
                    CHECK(lowered.instructions == expected);
                    CHECK(lowered.source_instruction_offsets ==
                          std::vector<u32>{0, static_cast<u32>(expected.size() - 1)});
                }
            }
        }
    }
}
