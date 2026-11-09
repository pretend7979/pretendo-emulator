// SPDX-License-Identifier: GPL-3.0-or-later
#include <array>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction Ptp{{0x06189b63, 0x30007c10, 0x001e2f0e, 0x000f4600}};
constexpr Instruction HalfPtp{{0x0a0c7b63, 0x30003000, 0x001eaf0d, 0x000fc200}};
constexpr Instruction Depth{{0x180c7b63, 0x30800e1c, 0x001e4f0e, 0x000f4200}};

u32 Source(u64 word) {
    return (word >> 20) & 255;
}
u32 Destination(u64 word) {
    return word & 255;
}
u32 Guard(u64 word) {
    return (word >> 16) & 15;
}

void CheckCopy(u64 word, u32 destination, u32 source, u32 predicate) {
    REQUIRE((word & 0xfff0000000000000ULL) == 0x5c90000000000000ULL);
    REQUIRE(Destination(word) == destination);
    REQUIRE(Source(word) == source);
    REQUIRE(Guard(word) == predicate);
}
} // namespace

TEST_CASE("OOT signed SGXT lowers to signed Maxwell extraction", "[video_core][ampere]") {
    for (const u32 first :
         {0x0835781aU, 0x131a781aU, 0x1416781aU, 0x1822781aU, 0x1c1c781aU, 0x2020781aU}) {
        const Instruction sgxt{{first, 1, 0x200, 0x000fe400}};
        const auto result = LowerToMaxwell(std::array{sgxt, Exit});
        REQUIRE(static_cast<bool>(result));
        REQUIRE(result.instructions.size() == 2);
        const u64 bfe = result.instructions[0];
        REQUIRE((bfe >> 52) == 0x380);
        REQUIRE(((bfe >> 48) & 1) == 1);
        REQUIRE(((bfe >> 8) & 255) == (first >> 24));
        REQUIRE(Destination(bfe) == ((first >> 16) & 255));
        REQUIRE(((bfe >> 20) & 0xffff) == 0x100);
    }
}

TEST_CASE("SGXT signed and unsigned forms retain valid widths and reject modifiers",
          "[video_core][ampere]") {
    for (const u32 width : {0U, 1U, 8U, 16U, 32U}) {
        for (const u32 signed_bit : {0U, 0x200U}) {
            const Instruction sgxt{{0x0403781a, width, signed_bit, 0x000fe400}};
            const auto result = LowerToMaxwell(std::array{sgxt, Exit});
            REQUIRE(static_cast<bool>(result));
            REQUIRE(((result.instructions[0] >> 48) & 1) == (signed_bit != 0));
            REQUIRE(((result.instructions[0] >> 20) & 0xffff) == (width << 8));
        }
    }
    for (const u32 unsupported : {1U, 0x400U, 0x800U, 0x10000U}) {
        const Instruction sgxt{{0x0403781a, 1, 0x200U | unsupported, 0x000fe400}};
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{sgxt, Exit})));
    }
    REQUIRE_FALSE(static_cast<bool>(
        LowerToMaxwell(std::array{Instruction{{0x0403781a, 33, 0x200, 0x000fe400}}, Exit})));
    REQUIRE_FALSE(static_cast<bool>(
        LowerToMaxwell(std::array{Instruction{{0x04ff781a, 1, 0x200, 0x000fe400}}, Exit})));
}

TEST_CASE("OOT per-pixel gathers preserve coordinate and packed offset registers",
          "[video_core][ampere]") {
    const auto result = LowerToMaxwell(std::array{Ptp, Exit});
    REQUIRE(static_cast<bool>(result));
    REQUIRE(result.required_texture_buffer == 0);
    REQUIRE(result.instructions.size() == 10);
    CheckCopy(result.instructions[0], 250, 6, 9);
    CheckCopy(result.instructions[1], 251, 7, 9);
    CheckCopy(result.instructions[2], 246, 16, 9);
    CheckCopy(result.instructions[3], 247, 17, 9);
    const u64 gather = result.instructions[4];
    REQUIRE((gather >> 58) == 0x32);
    REQUIRE(Guard(gather) == 9);
    REQUIRE(((gather >> 54) & 3) == 2);
    REQUIRE(((gather >> 50) & 1) == 0);
    REQUIRE(((gather >> 28) & 7) == 2);
    REQUIRE(((gather >> 20) & 255) == 246);
    REQUIRE(((gather >> 36) & 0x1fff) == 0x7c);
    for (u32 i = 0; i < 4; ++i) {
        CheckCopy(result.instructions[5 + i], i < 2 ? 24 + i : 14 + i - 2, 250 + i, 9);
    }
}

TEST_CASE("OOT half precision per-pixel gathers retain split packed results",
          "[video_core][ampere]") {
    const auto result = LowerToMaxwell(std::array{HalfPtp, Exit});
    REQUIRE(static_cast<bool>(result));
    REQUIRE(result.instructions.size() == 16);
    CheckCopy(result.instructions[0], 250, 10, 7);
    CheckCopy(result.instructions[1], 251, 11, 7);
    CheckCopy(result.instructions[2], 246, 0, 7);
    CheckCopy(result.instructions[3], 247, 1, 7);
    REQUIRE(((result.instructions[4] >> 54) & 3) == 2);
    CheckCopy(result.instructions[5], 248, 250, 7);
    CheckCopy(result.instructions[6], 249, 251, 7);
    CheckCopy(result.instructions[7], 246, 252, 7);
    CheckCopy(result.instructions[8], 247, 253, 7);
    REQUIRE(Destination(result.instructions[11]) == 12);
    REQUIRE(Destination(result.instructions[14]) == 13);
}

TEST_CASE("OOT depth gathers pass the reference as metadata", "[video_core][ampere]") {
    const auto result = LowerToMaxwell(std::array{Depth, Exit});
    REQUIRE(static_cast<bool>(result));
    REQUIRE(result.required_texture_buffer == 2);
    REQUIRE(result.instructions.size() == 9);
    CheckCopy(result.instructions[0], 250, 24, 7);
    CheckCopy(result.instructions[1], 251, 25, 7);
    CheckCopy(result.instructions[2], 246, 28, 7);
    const u64 gather = result.instructions[3];
    REQUIRE(((gather >> 50) & 1) == 1);
    REQUIRE(((gather >> 54) & 3) == 0);
    REQUIRE(((gather >> 20) & 255) == 246);
    REQUIRE(((gather >> 36) & 0x1fff) == 14);
}

TEST_CASE("Per-pixel gather input pairs respect dynamic reserved boundaries",
          "[video_core][ampere]") {
    const Instruction uniform_array{{0x201c7f63, 0xb000002b, 0x0c1e0f16, 0x000f4200}};
    for (const bool uniforms : {false, true}) {
        const u32 boundary = uniforms ? 216U : 244U;
        for (const bool second_pair : {false, true}) {
            for (const u32 source : {boundary - 2, boundary - 1, 255U}) {
                CAPTURE(uniforms, second_pair, source);
                auto gather = Ptp;
                if (second_pair)
                    gather.words[1] = (gather.words[1] & ~255U) | source;
                else
                    gather.words[0] = (gather.words[0] & 0x00ffffffU) | (source << 24);
                std::vector<Instruction> program{gather};
                if (uniforms)
                    program.push_back(uniform_array);
                program.push_back(Exit);
                const auto result = LowerToMaxwell(program);
                if (source == boundary - 1) {
                    REQUIRE_FALSE(static_cast<bool>(result));
                    REQUIRE(result.instructions.empty());
                } else {
                    REQUIRE(static_cast<bool>(result));
                    const u32 first = second_pair ? 2 : 0;
                    REQUIRE(Source(result.instructions[first]) == source);
                    REQUIRE(Source(result.instructions[first + 1]) ==
                            (source == 255 ? 255U : source + 1));
                }
            }
        }
    }
}

TEST_CASE("Extended gathers keep unsupported modes rejected", "[video_core][ampere]") {
    auto bad = Ptp;
    bad.words[2] |= 1U << 12; // Reserved offset mode 3.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    bad = Ptp;
    bad.words[1] = (bad.words[1] & ~(7U << 29)) | (5U << 29); // Unimplemented array PTP.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    bad = Ptp;
    bad.words[1] &= ~(1U << 28); // Unpacked extended input list.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    bad = Ptp;
    bad.words[0] = (bad.words[0] & ~0xfffU) | 0xf63;
    bad.words[1] &= ~0x07ffff00U;
    bad.words[2] |= 1U << 27; // Valid UR0 handle, but unimplemented bindless PTP.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    bad = Ptp;
    bad.words[2] &= ~(7U << 17); // Sparse predicate output.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    bad = Ptp;
    bad.words[2] |= 1U << 16; // Unknown precision mode.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    bad = Depth;
    bad.words[2] |= 1U << 12; // Combined depth/offset input layout not implemented.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    bad = Depth;
    bad.words[2] |= 1U << 23; // Unknown depth component selection.
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{bad, Exit})));
    REQUIRE_FALSE(static_cast<bool>(
        LowerToMaxwell(std::array{Ptp, Depth, Exit}))); // Different cbuf bindings.
}

TEST_CASE("Gather output spans respect shadow boundaries without rejecting packed outputs",
          "[video_core][ampere]") {
    const Instruction uniform_array{{0x201c7f63, 0xb000002b, 0x0c1e0f16, 0x000f4200}};
    for (const bool uniforms : {false, true}) {
        const u32 boundary = uniforms ? 216U : 244U;
        const auto lower = [&](const Instruction& instruction) {
            std::vector<Instruction> program{instruction};
            if (uniforms)
                program.push_back(uniform_array);
            program.push_back(Exit);
            return LowerToMaxwell(program);
        };
        const auto set_output = [](Instruction& instruction, bool second, u32 reg) {
            if (second)
                instruction.words[2] = (instruction.words[2] & ~255U) | reg;
            else
                instruction.words[0] = (instruction.words[0] & ~0x00ff0000U) | (reg << 16);
        };
        for (const bool second : {false, true}) {
            for (const u32 reg : {boundary - 2, boundary - 1}) {
                auto instruction = Ptp;
                set_output(instruction, second, reg);
                const auto result = lower(instruction);
                CAPTURE(uniforms, second, reg);
                if (reg == boundary - 1) {
                    REQUIRE_FALSE(static_cast<bool>(result));
                    REQUIRE(result.error == "native texture gather mode");
                    REQUIRE(result.instructions.empty());
                } else {
                    REQUIRE(static_cast<bool>(result));
                }
            }
        }
        // RZ as the primary output discards the first pair, keeping the second pair valid.
        auto discard_first = Ptp;
        set_output(discard_first, false, 255);
        set_output(discard_first, true, boundary - 2);
        REQUIRE(static_cast<bool>(lower(discard_first)));
        set_output(discard_first, true, boundary - 1);
        REQUIRE_FALSE(static_cast<bool>(lower(discard_first)));

        // RZ as the second output means all four components occupy the first output span.
        auto unsplit = Ptp;
        set_output(unsplit, true, 255);
        set_output(unsplit, false, boundary - 4);
        REQUIRE(static_cast<bool>(lower(unsplit)));
        set_output(unsplit, false, boundary - 3);
        REQUIRE_FALSE(static_cast<bool>(lower(unsplit)));
        set_output(unsplit, false, 255);
        REQUIRE(static_cast<bool>(lower(unsplit))); // both outputs discarded

        // Half results pack each pair into one native GPR. The rewritten scratch registers
        // R248/R246 are temporary destinations and must not be checked as native output spans.
        auto packed = HalfPtp;
        set_output(packed, false, boundary - 1);
        set_output(packed, true, boundary - 2);
        REQUIRE(static_cast<bool>(lower(packed)));
    }
}
