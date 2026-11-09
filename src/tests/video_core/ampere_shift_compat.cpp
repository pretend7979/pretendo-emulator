// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction RegisterShift{{0x032b7219, 0x0000002c, 0x0000122d, 0x000fe200}};
constexpr Instruction UniformImmediate{{0x06087899, 0x00000018, 0x08001207, 0x004fe400}};
constexpr std::array<u32, 20> Counts{0,   1,   7,   16,         21,         24,        31,
                                     32,  33,  47,  63,         64,         65,        127,
                                     128, 255, 256, 0x7fffffff, 0x80000000, 0xffffffff};

u32 Bits(u64 word, u32 offset, u32 count) {
    return (word >> offset) & ((1ULL << count) - 1);
}
void Field(Instruction& instruction, u32 bit, u32 count, u32 value) {
    const u32 mask = ((1U << count) - 1) << (bit % 32);
    auto& word = instruction.words[bit / 32];
    word = (word & ~mask) | ((value << (bit % 32)) & mask);
}
u32 Uniform(u32 reg) {
    REQUIRE((reg < 28 || reg == 63));
    return reg == 63 ? 255 : reg < 22 ? 216 + reg : 218 + reg;
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
};

void Execute(Machine& machine, const std::vector<u64>& code) {
    for (const u64 word : code) {
        if ((word >> 56) == 0xe3)
            continue;
        if (!machine.Predicate(Bits(word, 16, 4)))
            continue;
        const u32 destination = Bits(word, 0, 8);
        const u32 a = Bits(word, 8, 8), b = Bits(word, 20, 8), c = Bits(word, 39, 8);
        const u32 opcode = word >> 52;
        if (opcode == 0x5c9 && Bits(word, 39, 4) == 0 && Bits(word, 44, 4) == 10) {
            machine.Write(destination + Bits(word, 8, 2) * 256,
                          machine.Read(Bits(word, 20, 8) + Bits(word, 10, 2) * 256));
        } else if ((word >> 56) == 1) {
            machine.Write(destination, Bits(word, 20, 32));
        } else if ((word & 0xfff8000000000000ULL) == 0x5cf8000000000000ULL) {
            REQUIRE(Bits(word, 37, 2) == 2); // MaxShift::U64
            REQUIRE(Bits(word, 48, 2) == 0);
            REQUIRE(Bits(word, 50, 1) == 0); // clamp, not wrap
            const u64 packed = machine.Read(a) | (u64{machine.Read(c)} << 32);
            const u32 safe_count = (std::min)(machine.Read(b), 63U);
            machine.Write(destination, static_cast<u32>(packed >> safe_count));
        } else if ((word & 0xfff8000000000000ULL) == 0x3828000000000000ULL) {
            // GetImm20 includes bits20..38 and the sign flag at56. The new helper always uses6,
            // so the existing SHR frontend's out-of-range branch is never reached.
            const u32 count = Bits(word, 20, 19);
            REQUIRE(count == 6);
            REQUIRE(Bits(word, 48, 1) == 0);
            machine.Write(destination, machine.Read(a) >> count);
        } else if (opcode == 0x5b4) {
            REQUIRE(Bits(word, 49, 3) == 2); // ICMP.EQ against zero
            REQUIRE(Bits(word, 48, 1) == 0);
            machine.Write(destination, machine.Read(c) == 0 ? machine.Read(a) : machine.Read(b));
        } else if ((word & 0xfff8000000000000ULL) == 0x38e8000000000000ULL) {
            REQUIRE(Bits(word, 40, 3) == 0); // P2R.PR, byte0
            const u32 mask = Bits(word, 20, 19);
            u32 value = machine.Read(a) & ~mask;
            for (u32 i = 0; i < 7; ++i)
                value |= ((mask >> i) & 1) && machine.predicates[i] ? 1U << i : 0;
            machine.Write(destination, value);
        } else if ((word & 0xfff0000000000000ULL) == 0x38f0000000000000ULL) {
            REQUIRE(Bits(word, 40, 3) == 0); // R2P.PR, byte0
            const u32 mask = Bits(word, 20, 19), value = machine.Read(a);
            for (u32 i = 0; i < 7; ++i)
                if ((mask >> i) & 1)
                    machine.predicates[i] = (value >> i) & 1;
        } else if ((word & 0xfff8000000000000ULL) == 0xf0c8000000000000ULL) {
            REQUIRE(Bits(word, 20, 8) == 0); // S2R lane ID, fixture lane19
            machine.Write(destination, 19);
        } else {
            CAPTURE(word);
            FAIL("Unexpected lowered instruction");
        }
    }
}

bool Active(const Instruction& instruction, const Machine& machine, bool uniform) {
    const u32 guard = instruction.Bits(12, 4);
    if (!uniform)
        return machine.Predicate(guard);
    const bool value = (guard & 7) == 7 || ((machine.Read(239) >> (6 - (guard & 7))) & 1);
    return value != bool(guard & 8);
}

u32 Shift(u32 low, u32 high, u32 count) {
    return count >= 64 ? 0U : static_cast<u32>((low | (u64{high} << 32)) >> count);
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
} // namespace

TEST_CASE("Register U64 funnels clamp arbitrary counts and preserve aliases", "[ampere][shift]") {
    for (const bool high_registers : {false, true}) {
        for (u32 alias = 0; alias < 5; ++alias) {
            for (const u32 guard : {7U, 0U, 8U, 1U, 9U, 15U}) {
                auto instruction = RegisterShift;
                const u32 low = high_registers ? 246 : 3;
                const u32 count_register = high_registers ? 247 : 44;
                const u32 high = high_registers ? 248 : 45;
                const u32 destination =
                    std::array{high_registers ? 249U : 43U, low, count_register, high, 255U}[alias];
                Field(instruction, 12, 4, guard);
                Field(instruction, 16, 8, destination);
                Field(instruction, 24, 8, low);
                Field(instruction, 32, 8, count_register);
                Field(instruction, 64, 8, high);
                const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                CAPTURE(high_registers, alias, guard, lowered.error);
                REQUIRE(static_cast<bool>(lowered));
                for (const u32 count : Counts) {
                    for (const u32 low_word : {0U, 1U, 0x12345678U, 0xffffffffU}) {
                        for (const u32 high_word : {0U, 1U, 0x87654321U, 0xffffffffU}) {
                            for (const u32 predicates : {0U, 1U, 2U, 0x7fU}) {
                                auto machine = Initial(predicates);
                                const auto put = [&](u32 reg, u32 value) {
                                    machine.Write(reg + (reg >= 244 ? 256 : 0), value);
                                };
                                put(low, low_word);
                                put(high, high_word);
                                put(count_register, count);
                                const Machine before = machine;
                                const u32 expected =
                                    Active(instruction, before, false)
                                        ? Shift(before.Native(low, 244), before.Native(high, 244),
                                                before.Native(count_register, 244))
                                        : before.Native(destination, 244);
                                Execute(machine, lowered.instructions);
                                CHECK(machine.Native(destination, 244) ==
                                      (destination == 255 ? 0U : expected));
                                CHECK(machine.predicates == before.predicates);
                                for (u32 reg = 0; reg < 255; ++reg)
                                    if (reg != destination)
                                        CHECK(machine.Native(reg, 244) == before.Native(reg, 244));
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Uniform U64 funnels use immediate counts and preserve UP guards", "[ampere][shift]") {
    for (u32 alias = 0; alias < 3; ++alias) {
        for (const u32 guard : {7U, 0U, 8U, 1U, 9U, 15U}) {
            for (const bool zero_low : {false, true}) {
                for (const bool zero_high : {false, true}) {
                    for (const u32 count : Counts) {
                        auto instruction = UniformImmediate;
                        const u32 low = zero_low ? 63 : 6, high = zero_high ? 63 : 7;
                        const u32 destination = std::array{8U, 6U, 7U}[alias];
                        Field(instruction, 12, 4, guard);
                        Field(instruction, 16, 8, destination);
                        Field(instruction, 24, 8, low);
                        Field(instruction, 64, 8, high);
                        instruction.words[1] = count;
                        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                        REQUIRE(static_cast<bool>(lowered));
                        for (const u32 low_word : {0U, 0x12345678U, 0xffffffffU}) {
                            for (const u32 high_word : {0U, 0x87654321U, 0xffffffffU}) {
                                for (const u32 up : {0U, 0x20U, 0x40U, 0x7fU}) {
                                    auto machine = Initial(up ^ 0x35U);
                                    machine.Write(Uniform(low), low_word);
                                    machine.Write(Uniform(high), high_word);
                                    machine.Write(239, up);
                                    const auto before = machine;
                                    const u32 expected =
                                        Active(instruction, before, true)
                                            ? Shift(before.Read(Uniform(low)),
                                                    before.Read(Uniform(high)), count)
                                            : before.Read(Uniform(destination));
                                    Execute(machine, lowered.instructions);
                                    CHECK(machine.Read(Uniform(destination)) == expected);
                                    CHECK(machine.predicates == before.predicates);
                                    CHECK(machine.Read(239) == before.Read(239));
                                    for (u32 reg = 0; reg < 255; ++reg)
                                        CHECK(machine.Native(reg, 216) == before.Native(reg, 216));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Existing U32 shifts remain byte-for-byte unchanged", "[ampere][shift]") {
    const Instruction reg{{0x03047219, 5, 0x000016ff, 0}};
    const auto r = LowerToMaxwell(std::array{reg, Exit});
    REQUIRE(static_cast<bool>(r));
    REQUIRE(r.instructions.size() == 2);
    CHECK(r.instructions[0] ==
          (0x5c28000000000000ULL | (7ULL << 16) | 4 | (3ULL << 8) | (5ULL << 20)));
    const Instruction uni{{0x06087899, 16, 0x0800163f, 0}};
    const auto u = LowerToMaxwell(std::array{uni, Exit});
    REQUIRE(static_cast<bool>(u));
    REQUIRE(u.instructions.size() == 2);
    CHECK(u.instructions[0] ==
          (0x3828000000000000ULL | (7ULL << 16) | 224 | (222ULL << 8) | (16ULL << 20)));
}

TEST_CASE("Unimplemented U64 shift layouts remain rejected", "[ampere][shift]") {
    for (const u32 extra : {0x800U, 0x10000U, 0x100U, 0x20000U}) {
        auto instruction = RegisterShift;
        instruction.words[2] |= extra;
        CHECK_FALSE(static_cast<bool>(LowerToMaxwell(std::array{instruction, Exit})));
        auto uniform = UniformImmediate;
        uniform.words[2] |= extra;
        CHECK_FALSE(static_cast<bool>(LowerToMaxwell(std::array{uniform, Exit})));
    }
    auto invalid = RegisterShift;
    invalid.words[1] = 256; // Not a valid register field.
    CHECK_FALSE(static_cast<bool>(LowerToMaxwell(std::array{invalid, Exit})));
    auto uniform = UniformImmediate;
    Field(uniform, 16, 8, 63);
    CHECK_FALSE(static_cast<bool>(LowerToMaxwell(std::array{uniform, Exit})));
}

TEST_CASE("Lane ID S2R shadows only its high destination", "[ampere][shift][pressure]") {
    for (const u32 guard : {7U, 0U, 8U}) {
        Instruction instruction{{0x00f77919, 0, 0, 0x000e6200}};
        Field(instruction, 12, 4, guard);
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        REQUIRE(static_cast<bool>(lowered));
        for (const u32 predicates : {0U, 1U}) {
            auto machine = Initial(predicates);
            const auto before = machine;
            const u32 expected = before.Predicate(guard) ? 19 : before.Native(247, 244);
            Execute(machine, lowered.instructions);
            CHECK(machine.Native(247, 244) == expected);
            for (u32 reg = 0; reg < 255; ++reg)
                if (reg != 247)
                    CHECK(machine.Native(reg, 244) == before.Native(reg, 244));
        }
    }
}
