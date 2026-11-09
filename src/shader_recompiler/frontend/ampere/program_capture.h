// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace Shader::Ampere {
class ProgramCaptureState {
public:
    enum class Result { Continue, Complete, InvalidTarget };
    static constexpr u32 MaximumInstructions = 65536;

    Result Observe(const Instruction& instruction, u32 index) {
        if (index != next_index) history_count = 0;
        const bool linked_call = IsManuallyLinkedCall(instruction, index);
        history[index % history.size()] = instruction;
        history_count = std::min(history_count + 1, static_cast<u32>(history.size()));
        next_index = index + 1;
        const u32 opcode = instruction.Bits(0, 12);
        if (opcode == 0x944 || opcode == 0x947 || opcode == 0x945) {
            // BRA keeps its .DIV/.CONV mode in the offset's two low bits.
            const u32 low = instruction.words[1] & (opcode == 0x947 ? ~3U : ~0U);
            const u64 raw = low | (u64{instruction.words[2] & 0x3ffff} << 32);
            const s64 relative = opcode == 0x945 ? static_cast<s32>(instruction.words[1])
                                                : static_cast<s64>(raw << 14) >> 14;
            const s64 target = s64{index} * 16 + 16 + relative;
            if (target < 0 || target % 16 || target >= MaximumInstructions * 16)
                return Result::InvalidTarget;
            // NOINC transfers are calls only when the explicit return link matches the lowerer.
            const bool noinc_call = opcode == 0x944 && (instruction.words[2] & 0x400000) != 0;
            if (opcode == 0x944 && (!noinc_call || linked_call)) {
                last_call_target = std::max(last_call_target, static_cast<u32>(target / 16));
            } else {
                last_forward_target = std::max(last_forward_target, static_cast<u32>(target / 16));
            }
            // The compiler ends every program with an unconditional branch to itself; with a jump
            // table (whose targets live in a constant buffer) that is the only reliable end.
            if (opcode == 0x947 && computed_branch && instruction.Bits(12, 4) == 7 &&
                target == s64{index} * 16)
                return Result::Complete;
        }
        if (opcode == 0x949 || opcode == 0x958) computed_branch = true;
        if (computed_branch) return Result::Continue;
        if (opcode == 0x94d && instruction.Bits(12, 4) == 7 && index >= last_forward_target) {
            if (!last_call_target) return Result::Complete;
            main_exited = true;
        }
        if (opcode == 0x950 && instruction.Bits(12, 4) == 7 && main_exited &&
            index >= last_call_target && index >= last_forward_target)
            return Result::Complete;
        return Result::Continue;
    }

private:
    bool IsManuallyLinkedCall(const Instruction& instruction, u32 index) const {
        if (main_exited || instruction.words[0] != 0x7944 ||
            (instruction.words[2] & ~0x3ffffU) != 0x03c00000)
            return false;
        std::array<bool, 256> written{};
        for (u32 distance = 1; distance <= history_count; ++distance) {
            const Instruction& prior = history[(index - distance) % history.size()];
            const u32 opcode = prior.Bits(0, 12);
            if ((prior.words[0] & ~0x00ff0000U) == 0x00007802 &&
                prior.words[1] == (index + 1) * 16 && prior.words[2] == 0xf00) {
                const u32 link = prior.Bits(16, 8);
                return link < 250 && !written[link];
            }
            if (opcode == 0x947 || opcode == 0x949 || opcode == 0x958 ||
                opcode == 0x944 || opcode == 0x950 || opcode == 0x94d ||
                opcode == 0x945 || opcode == 0x941 || opcode == 0x942)
                break;
            const u32 destination = prior.Bits(16, 8);
            written[destination] = true;
            if (opcode == 0x225 || opcode == 0x625 || opcode == 0x825)
                written[(destination + 1) & 0xff] = true;
        }
        return false;
    }

    std::array<Instruction, 64> history{};
    u32 history_count{};
    u32 next_index{};
    u32 last_call_target{};
    u32 last_forward_target{};
    bool main_exited{};
    bool computed_branch{};
};
} // namespace Shader::Ampere
