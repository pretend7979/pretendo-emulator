// SPDX-License-Identifier: GPL-3.0-or-later
#include <array>
#include <bit>
#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
using Shader::Ampere::PrepareRegisterPressure;
using Banks = std::array<std::array<u32, 256>, 3>;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};

void Field(Instruction& instruction, u32 bit, u32 count, u32 value) {
    const u32 mask = ((1U << count) - 1) << (bit % 32);
    auto& word = instruction.words[bit / 32];
    word = (word & ~mask) | ((value << (bit % 32)) & mask);
}

Banks Initial() {
    Banks banks;
    for (u32 bank = 0; bank < 3; ++bank) {
        for (u32 reg = 0; reg < 255; ++reg) {
            banks[bank][reg] = std::bit_cast<u32>((reg + bank * 257 + 1) / 16.0F);
        }
        banks[bank][255] = 0;
    }
    return banks;
}

void Copies(Banks& banks, const std::vector<u64>& words) {
    for (u64 word : words) {
        REQUIRE((word >> 48) == 0x5c98);
        REQUIRE(((word >> 44) & 15) == 10);
        const u32 source_bank = (word >> 10) & 3;
        const u32 source = (word >> 20) & 255;
        const u32 destination_bank = (word >> 8) & 3;
        const u32 destination = word & 255;
        REQUIRE(source_bank < banks.size());
        REQUIRE(destination_bank < banks.size());
        banks[destination_bank][destination] = banks[source_bank][source];
    }
}

// Model native operations over logical banks, then compare their execution through the
// prepared base registers and the real emitted bank-copy prologue/epilogue.
void Native(Banks& banks, const Instruction& instruction, u32 reserved, bool prepared) {
    const auto read = [&](u32 reg) {
        return reg == 255 ? 0U : banks[!prepared && reg >= reserved ? 1 : 0][reg];
    };
    const auto write = [&](u32 reg, u32 value) {
        if (reg != 255)
            banks[!prepared && reg >= reserved ? 1 : 0][reg] = value;
    };
    const auto as_float = [&](u32 reg) { return std::bit_cast<float>(read(reg)); };
    const u32 destination = instruction.Bits(16, 8);
    const u32 a = instruction.Bits(24, 8), c = instruction.Bits(64, 8);
    switch (instruction.Bits(0, 12)) {
    case 0xa20:
        write(destination, std::bit_cast<u32>(as_float(a) * 0.5F));
        break;
    case 0xa23:
        write(destination, std::bit_cast<u32>(std::fma(as_float(a), 0.5F, as_float(c))));
        break;
    case 0x621:
        write(destination, std::bit_cast<u32>(as_float(a) + 0.5F));
        break;
    case 0x202:
        write(destination, read(instruction.Bits(32, 8)));
        break;
    case 0x245:
        write(destination, std::bit_cast<u32>(static_cast<float>(read(instruction.Bits(32, 8)))));
        break;
    case 0x209:
        write(destination,
              std::bit_cast<u32>(std::fmax(as_float(a), as_float(instruction.Bits(32, 8)))));
        break;
    case 0x923:
        write(destination, 0x51000000U + read(a));
        break;
    case 0x321: {
        const u32 vertex = read(instruction.Bits(32, 8));
        for (u32 i = 0; i <= instruction.Bits(74, 2); ++i) {
            write(destination + i, 0xa0000000U + vertex + instruction.Bits(40, 10) + i);
        }
        break;
    }
    case 0x381: {
        const u64 address = read(a) | (u64{read(a + 1)} << 32);
        REQUIRE(address == 0x1234567800010000ULL);
        const u32 count = instruction.Bits(73, 3) == 6 ? 4 : 2;
        for (u32 i = 0; i < count; ++i)
            write(destination + i, 0xb0000000U + i);
        break;
    }
    default:
        FAIL("Missing native model");
    }
}

constexpr std::array Captured{
    Instruction{{0xfff47321, 0x000100ff, 0x00000400, 0x000e6800}},
    Instruction{{0x07f57a23, 0x00c01e00, 0x00010018, 0x040fe200}},
    Instruction{{0x1de97a20, 0x00c00900, 0x00410000, 0x000fe200}},
    Instruction{{0x14f37621, 0x01001000, 0x00010000, 0x000fe200}},
    Instruction{{0xffda7923, 0x00000000, 0x00007800, 0x000e2800}},
    Instruction{{0x00e18202, 0x000000ff, 0x00000f00, 0x000fe200}},
    Instruction{{0xe0e07381, 0x00000000, 0x001ead00, 0x000f2200}},
    Instruction{{0x00dc7245, 0x000000dc, 0x00201000, 0x000fc600}},
    Instruction{{0xffdb7209, 0x000000db, 0x07810000, 0x000fc600}},
};
} // namespace

TEST_CASE("Ampere pressure copies preserve both logical banks and their neighbors",
          "[ampere][pressure]") {
    for (const u32 reserved : {216U, 244U}) {
        for (const auto& instruction : Captured) {
            CAPTURE(reserved, instruction.words);
            const auto pressure = PrepareRegisterPressure(instruction, reserved);
            REQUIRE(pressure.valid);
            auto expected = Initial();
            if (instruction.Bits(0, 12) == 0x209) {
                const u32 reg = instruction.Bits(32, 8);
                expected[reg >= reserved ? 1 : 0][reg] = std::bit_cast<u32>(-2.0F);
            }
            if (instruction.Bits(0, 12) == 0x381) {
                const u32 bank = instruction.Bits(24, 8) >= reserved ? 1 : 0;
                expected[bank][instruction.Bits(24, 8)] = 0x00010000;
                expected[bank][instruction.Bits(24, 8) + 1] = 0x12345678;
            }
            auto actual = expected;
            Native(expected, instruction, reserved, false);
            Copies(actual, pressure.prefix);
            Native(actual, pressure.instruction, reserved, true);
            Copies(actual, pressure.suffix);
            CHECK(actual[0] == expected[0]);
            CHECK(actual[1] == expected[1]);
            if (!pressure.prefix.empty()) {
                // A false execution predicate runs only the unconditional SSA bank renames.
                actual = Initial();
                expected = actual;
                Copies(actual, pressure.prefix);
                Copies(actual, pressure.suffix);
                CHECK(actual[0] == expected[0]);
                CHECK(actual[1] == expected[1]);
            }
        }
    }
}

TEST_CASE("Ampere pressure leaves non-register operand encodings unchanged", "[ampere][pressure]") {
    for (const u32 opcode :
         {0x423U, 0x623U, 0x823U, 0xa23U, 0xc23U, 0xe23U, 0x820U, 0xa20U, 0xc20U, 0x421U,
          0x621U, 0x821U, 0xa21U, 0xe21U, 0x810U, 0xa10U, 0xc10U, 0x817U, 0xa17U, 0xc17U,
          0x808U, 0xa08U, 0x80bU, 0xa0bU, 0xc0bU, 0x80cU, 0xa0cU, 0xc0cU}) {
        Instruction instruction{{0x08007000U | opcode, 0xf4f4f4f4, 0x08010009, 0}};
        // Predicate setters have no GPR output. Other instructions write R244.
        if ((opcode & 0x1ff) != 0xb && (opcode & 0x1ff) != 0xc) {
            Field(instruction, 16, 8, 244);
        } else {
            Field(instruction, 24, 8, 244);
        }
        const auto pressure = PrepareRegisterPressure(instruction, 216);
        CAPTURE(opcode);
        REQUIRE(pressure.valid);
        CHECK(pressure.instruction.words[1] == instruction.words[1]);
    }
    Instruction ald{{0xfff47321, 0x0001a0f8, 0x00000400, 0}};
    const auto pressure = PrepareRegisterPressure(ald, 216);
    REQUIRE(pressure.valid);
    CHECK((pressure.instruction.words[1] & ~255U) == (ald.words[1] & ~255U));
    CHECK(pressure.instruction.Bits(32, 8) != ald.Bits(32, 8)); // real vertex operand
    CHECK(pressure.instruction.Bits(24, 8) == 255);             // RZ stays RZ
}

TEST_CASE("Ampere attribute pressure keeps implicit spans inside one quartet",
          "[ampere][pressure]") {
    for (const u32 reserved : {216U, 244U}) {
        for (const u32 components : {1U, 2U, 3U, 4U}) {
            auto instruction = Captured[0];
            Field(instruction, 16, 8, reserved);
            Field(instruction, 74, 2, components - 1);
            const auto pressure = PrepareRegisterPressure(instruction, reserved);
            REQUIRE(pressure.valid);
            CHECK(pressure.instruction.Bits(16, 8) + components <= reserved);
            CHECK(pressure.prefix.size() == 8);
            CHECK(pressure.suffix.size() == 8);
        }
        auto crossing = Captured[0];
        Field(crossing, 16, 8, reserved - 1);
        Field(crossing, 74, 2, 1); // pair crosses the native/shadow boundary
        CHECK_FALSE(PrepareRegisterPressure(crossing, reserved).valid);
        if (reserved == 244) {
            const auto lowered = LowerToMaxwell(std::array{crossing, Exit});
            CHECK_FALSE(static_cast<bool>(lowered));
            CHECK(lowered.error == "native program uses lowering scratch registers");
        }
    }
    for (const u32 source : {253U, 254U}) {
        auto over_rz = Captured[0];
        Field(over_rz, 16, 8, source);
        Field(over_rz, 74, 2, 2);
        CHECK_FALSE(PrepareRegisterPressure(over_rz, 244).valid);
    }
}

TEST_CASE("Ampere U64 right funnels preserve the high input and low result",
          "[ampere][pressure][shift]") {
    for (const u32 shift : {0U, 1U, 21U, 31U, 32U}) {
        Instruction instruction{{0x0e0e7819, shift, 0x00001211, 0x020fe200}};
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        REQUIRE(static_cast<bool>(lowered));
        REQUIRE(lowered.instructions.size() == 2);
        const u64 word = lowered.instructions[0];
        CHECK((word & 0xfff8000000000000ULL) == 0x38f8000000000000ULL);
        CHECK(((word >> 37) & 3) == 0); // bounded immediate uses the U32 clamping range
        CHECK(((word >> 8) & 255) == 14);
        CHECK(((word >> 39) & 255) == 17);
        CHECK(((word >> 20) & 0x7ffff) == shift);
        for (const u32 low : {0U, 1U, 0x12345678U, 0xffffffffU}) {
            for (const u32 high : {0U, 1U, 0x87654321U, 0xffffffffU}) {
                const u64 packed = low | (u64{high} << 32);
                const u32 native = shift == 0    ? low
                                   : shift == 32 ? high
                                                 : (low >> shift) | (high << (32 - shift));
                const u32 maxwell = static_cast<u32>(packed >> shift);
                CHECK(maxwell == native);
            }
        }
    }
    for (const u32 bad_mode : {0x00001a11U, 0x00011211U, 0x00001311U}) {
        // wrapping, high-result and reserved extra fields remain outside this extension.
        const Instruction instruction{{0x0e0e7819, 21, bad_mode, 0x020fe200}};
        CHECK_FALSE(static_cast<bool>(LowerToMaxwell(std::array{instruction, Exit})));
    }
}

TEST_CASE("Ampere captured high-register instructions lower through bank copies",
          "[ampere][pressure]") {
    for (const auto& instruction : Captured) {
        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
        CAPTURE(instruction.words, lowered.error);
        REQUIRE(static_cast<bool>(lowered));
    }
}
