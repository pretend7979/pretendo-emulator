// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/maxwell/decode.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwellWith;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction UniformMarker{{0x00007c02, 0, 0x08000f00, 0}};
constexpr Instruction Logic{{0xe1e17212, 0x000000e4, 0x078efe0f, 0x000fe400}};
constexpr Instruction Packed{{0xffdf7243, 0x00000016, 0x0000c4ff, 0x000fe200}};
constexpr Instruction Extend{{0xe6e6781a, 1, 0, 0x000fe200}};

u32 Bits(u64 word, u32 offset, u32 count) {
    return (word >> offset) & ((1ULL << count) - 1);
}
void Field(Instruction& instruction, u32 bit, u32 count, u32 value) {
    const u32 mask = ((1U << count) - 1) << (bit % 32);
    auto& word = instruction.words[bit / 32];
    word = (word & ~mask) | ((value << (bit % 32)) & mask);
}

struct Machine {
    std::array<u32, 768> registers{};
    std::array<bool, 8> predicates{};
    u32 Read(u32 reg) const {
        return reg == 255 ? 0U : registers[reg];
    }
    void Write(u32 reg, u32 value) {
        if (reg != 255)
            registers[reg] = value;
    }
    bool Predicate(u32 encoded) const {
        return ((encoded & 7) == 7 || predicates[encoded & 7]) != bool(encoded & 8);
    }
    u32 Native(u32 reg, u32 reserved) const {
        return reg == 255 ? 0U : Read(reg + (reg >= reserved ? 256 : 0));
    }
    void Put(u32 reg, u32 reserved, u32 value) {
        if (reg != 255)
            registers[reg + (reg >= reserved ? 256 : 0)] = value;
    }
};

// Interpret the actual Maxwell fields used by these small expansions, including logical
// shadow-bank copies. Expected native results are computed separately below.
void Execute(Machine& machine, const std::vector<u64>& code) {
    for (const u64 word : code) {
        if (!machine.Predicate(Bits(word, 16, 4)))
            continue;
        const u32 destination = Bits(word, 0, 8);
        const u32 a = Bits(word, 8, 8), b = Bits(word, 20, 8), c = Bits(word, 39, 8);
        const u32 opcode = word >> 52;
        if (opcode == 0x5c9 && Bits(word, 39, 4) == 0 && Bits(word, 44, 4) == 10) {
            machine.Write(destination + Bits(word, 8, 2) * 256,
                          machine.Read(b + Bits(word, 10, 2) * 256));
        } else if (opcode == 0x5be) {
            REQUIRE(Bits(word, 38, 1) == 0);
            REQUIRE(Bits(word, 47, 1) == 0);
            u32 result = 0;
            const u32 lut = Bits(word, 28, 8);
            for (u32 row = 0; row < 8; ++row) {
                if (!((lut >> row) & 1))
                    continue;
                result |= ((row & 4) ? machine.Read(a) : ~machine.Read(a)) &
                          ((row & 2) ? machine.Read(b) : ~machine.Read(b)) &
                          ((row & 1) ? machine.Read(c) : ~machine.Read(c));
            }
            machine.Write(destination, result);
            const u32 pred = Bits(word, 48, 3), mode = Bits(word, 36, 2);
            REQUIRE((mode == 0 || mode == 3));
            if (pred != 7)
                machine.predicates[pred] = mode == 3 && result != 0;
        } else if (Shader::Maxwell::Decode(word) == Shader::Maxwell::Opcode::FCMP_reg) {
            REQUIRE(Bits(word, 48, 4) == 7); // FCMP.NUM selects A for a numerical comparison C:0.
            const bool numeric = !std::isnan(std::bit_cast<float>(machine.Read(c)));
            machine.Write(destination, numeric ? machine.Read(a) : machine.Read(b));
        } else if ((word & 0xfff8000000000000ULL) == 0x5cb0000000000000ULL) {
            REQUIRE(Bits(word, 8, 2) == 1);  // U16 destination
            REQUIRE(Bits(word, 10, 2) == 2); // F32 source
            REQUIRE(Bits(word, 12, 1) == 0);
            REQUIRE(Bits(word, 39, 2) == 3); // truncation
            REQUIRE(Bits(word, 44, 1) == 1); // FTZ
            REQUIRE(Bits(word, 45, 1) == 0);
            REQUIRE(Bits(word, 47, 1) == 0);
            REQUIRE(Bits(word, 49, 1) == 0);
            const float value = std::bit_cast<float>(machine.Read(b));
            REQUIRE(!std::isnan(value));
            machine.Write(destination,
                          static_cast<u32>(std::clamp(std::trunc(value), 0.0F, 65535.0F)));
        } else if (opcode == 0x5ce) {
            REQUIRE(Bits(word, 8, 2) == 0);  // U8 destination
            REQUIRE(Bits(word, 10, 2) == 1); // U16 source
            REQUIRE(Bits(word, 12, 2) == 0);
            REQUIRE(Bits(word, 41, 3) == 0);
            REQUIRE(Bits(word, 45, 1) == 0);
            REQUIRE(Bits(word, 47, 1) == 0);
            REQUIRE(Bits(word, 49, 1) == 0);
            REQUIRE(Bits(word, 50, 1) == 1);
            machine.Write(destination, std::min(machine.Read(b) & 0xffffU, 255U));
        } else if ((word & 0xfff0000000000000ULL) == 0x3800000000000000ULL) {
            const u32 immediate = Bits(word, 20, 19);
            REQUIRE((immediate & 255) == 0); // offset zero
            const u32 count = (immediate >> 8) & 255;
            REQUIRE(count <= 32);
            u32 result = count == 32 ? machine.Read(a) : machine.Read(a) & ((1U << count) - 1);
            if (Bits(word, 48, 1) && count && count < 32 && ((result >> (count - 1)) & 1))
                result |= ~((1U << count) - 1);
            machine.Write(destination, result);
        } else {
            CAPTURE(word);
            FAIL("Unexpected lowered instruction");
        }
    }
}

Machine Initial(u32 predicate_bits) {
    Machine machine;
    for (u32 i = 0; i < machine.registers.size(); ++i)
        machine.registers[i] = i * 0x1020304U;
    machine.registers[255] = 0;
    for (u32 i = 0; i < 7; ++i)
        machine.predicates[i] = (predicate_bits >> i) & 1;
    machine.predicates[7] = true;
    return machine;
}

std::vector<u64> Lower(const Instruction& instruction, u32 reserved) {
    const auto lowered =
        reserved == 244
            ? LowerToMaxwellWith(std::array{instruction, Exit}, 0)
            : LowerToMaxwellWith(std::array{instruction, UniformMarker, Exit}, 216 - reserved);
    INFO(lowered.error);
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(lowered.source_instruction_offsets.size() >= 2);
    return {lowered.instructions.begin(),
            lowered.instructions.begin() + lowered.source_instruction_offsets[1]};
}

void Preserved(const Machine& before, const Machine& after, u32 destination, u32 reserved) {
    for (u32 reg = 0; reg < 255; ++reg)
        if (reg != destination)
            CHECK(after.Native(reg, reserved) == before.Native(reg, reserved));
}

u32 NativeLut(u32 lut, u32 a, u32 b, u32 c) {
    u32 result = 0;
    for (u32 bit = 0; bit < 32; ++bit) {
        const u32 index = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1);
        result |= ((lut >> index) & 1) << bit;
    }
    return result;
}

u32 NativeByte(float value) {
    if (std::isnan(value) || value < 0)
        return 0;
    if (value > 255)
        return 255;
    return static_cast<u32>(value);
}
} // namespace

TEST_CASE("High-register LOP3 preserves all sources, aliases and predicate results",
          "[ampere][pressure]") {
    for (u32 reserved : {244U, 216U, 212U}) {
        for (u32 alias = 0; alias < 5; ++alias) {
            for (u32 guard : {7U, 0U, 8U, 15U}) {
                for (u32 pred_target : {7U, 2U}) {
                    auto instruction = Logic;
                    const u32 a = reserved, b = reserved + 4, c = reserved + 2;
                    const u32 destination = std::array{reserved + 1, a, b, c, 255U}[alias];
                    Field(instruction, 12, 4, guard);
                    Field(instruction, 16, 8, destination);
                    Field(instruction, 24, 8, a);
                    Field(instruction, 32, 8, b);
                    Field(instruction, 64, 8, c);
                    Field(instruction, 81, 3, pred_target);
                    for (u32 lut = 0; lut < 256; ++lut) {
                        Field(instruction, 72, 8, lut);
                        const auto code = Lower(instruction, reserved);
                        for (u32 predicates : {0U, 0x7fU}) {
                            auto machine = Initial(predicates);
                            machine.Put(a, reserved, 0x0f0ff0f0);
                            machine.Put(b, reserved, 0x3333cccc);
                            machine.Put(c, reserved, 0x5555aaaa);
                            const Machine before = machine;
                            const bool active = before.Predicate(guard);
                            const u32 result =
                                NativeLut(lut, before.Native(a, reserved),
                                          before.Native(b, reserved), before.Native(c, reserved));
                            Execute(machine, code);
                            CHECK(machine.Native(destination, reserved) ==
                                  (destination == 255 ? 0U
                                   : active           ? result
                                                      : before.Native(destination, reserved)));
                            auto expected_predicates = before.predicates;
                            if (active && pred_target != 7)
                                expected_predicates[pred_target] = result != 0;
                            CHECK(machine.predicates == expected_predicates);
                            Preserved(before, machine, destination, reserved);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Packed F2IP exact U8 mode saturates and clears the upper bytes",
          "[ampere][conversion]") {
    std::vector<u32> inputs;
    for (float value : {-std::numeric_limits<float>::infinity(),
                        -65536.0F,
                        -255.0F,
                        -1.0F,
                        -0.5F,
                        -0.0F,
                        0.0F,
                        0.5F,
                        0.999F,
                        1.0F,
                        1.5F,
                        127.75F,
                        254.5F,
                        255.0F,
                        255.75F,
                        256.0F,
                        65535.0F,
                        65536.0F,
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()})
        inputs.push_back(std::bit_cast<u32>(value));
    for (u32 value : {1U, 0x80000001U, 0x007fffffU, 0x807fffffU, 0x7f800001U, 0xff800001U})
        inputs.push_back(value);
    u32 random = 0x9137a1d2;
    for (u32 i = 0; i < 300; ++i) {
        random = random * 1664525U + 1013904223U;
        inputs.push_back(random);
    }
    for (u32 reserved : {244U, 216U, 212U}) {
        for (u32 source : {22U, reserved, 255U}) {
            for (u32 destination : {17U, reserved + 1, source, 255U}) {
                for (u32 guard : {7U, 0U, 8U, 15U}) {
                    auto instruction = Packed;
                    Field(instruction, 12, 4, guard);
                    Field(instruction, 16, 8, destination);
                    Field(instruction, 32, 8, source);
                    const auto code = Lower(instruction, reserved);
                    for (u32 bits : inputs)
                        for (u32 predicates : {0U, 0x7fU}) {
                            auto machine = Initial(predicates);
                            machine.Put(source, reserved, bits);
                            const Machine before = machine;
                            const u32 result =
                                NativeByte(std::bit_cast<float>(before.Native(source, reserved)));
                            Execute(machine, code);
                            CHECK(machine.Native(destination, reserved) ==
                                  (destination == 255 ? 0U
                                   : before.Predicate(guard)
                                       ? result
                                       : before.Native(destination, reserved)));
                            CHECK(machine.predicates == before.predicates);
                            Preserved(before, machine, destination, reserved);
                        }
                }
            }
        }
    }
}

TEST_CASE("SGXT high-register metadata preserves immediate and both banks", "[ampere][pressure]") {
    for (u32 reserved : {244U, 216U, 212U}) {
        for (bool is_signed : {false, true}) {
            for (u32 count : {0U, 1U, 7U, 16U, 31U, 32U}) {
                for (u32 destination : {reserved + 2, reserved + 3}) {
                    for (u32 guard : {7U, 0U, 8U, 15U}) {
                        auto instruction = Extend;
                        instruction.words[1] = count;
                        Field(instruction, 12, 4, guard);
                        Field(instruction, 16, 8, destination);
                        Field(instruction, 24, 8, reserved + 2);
                        Field(instruction, 73, 1, is_signed);
                        const auto prepared =
                            Shader::Ampere::PrepareRegisterPressure(instruction, reserved);
                        REQUIRE(prepared.valid);
                        CHECK(prepared.instruction.words[1] == count);
                        const auto code = Lower(instruction, reserved);
                        for (u32 value : {0U, 1U, 0x12345678U, 0x87654321U, 0xffffffffU}) {
                            for (u32 predicates : {0U, 0x7fU}) {
                                auto machine = Initial(predicates);
                                machine.Put(reserved + 2, reserved, value);
                                const Machine before = machine;
                                u32 result = count == 32 ? value : value & ((1U << count) - 1);
                                if (is_signed && count && count < 32 &&
                                    ((result >> (count - 1)) & 1))
                                    result |= ~((1U << count) - 1);
                                Execute(machine, code);
                                CHECK(machine.Native(destination, reserved) ==
                                      (before.Predicate(guard)
                                           ? result
                                           : before.Native(destination, reserved)));
                                CHECK(machine.predicates == before.predicates);
                                Preserved(before, machine, destination, reserved);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Packed F2IP rejects unimplemented packing and source modes", "[ampere][conversion]") {
    for (u32 bit : {24U, 40U, 63U, 64U, 72U, 74U, 78U, 80U, 91U}) {
        auto instruction = Packed;
        instruction.words[bit / 32] ^= 1U << (bit % 32);
        const auto result = LowerToMaxwellWith(std::array{instruction, Exit}, 0);
        CHECK(!result);
        CHECK(result.error == "native packed float-to-byte mode");
    }
}
