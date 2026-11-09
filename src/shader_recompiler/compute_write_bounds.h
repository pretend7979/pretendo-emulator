// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <bit>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <vector>

#include "common/common_types.h"

namespace Shader {

struct ComputeWriteBound {
    u32 index{};
    u32 cbuf_index{};
    u32 cbuf_offset{};
    // Group X/Y/Z followed by local invocation X/Y/Z. Coefficients are nonnegative.
    std::array<u32, 6> coefficients{};
    u32 constant_end{};
    u32 alignment{};
};

struct ComputeWriteBounds {
    std::array<u32, 3> workgroup_size{};
    std::vector<ComputeWriteBound> storage;
};

inline bool ComputeWriteBoundsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_COMPUTE_WRITE_BOUNDS");
        return value && std::string_view{value} == "1";
    }();
    return enabled;
}

// Evaluate a proof only for the same compiled local size and an actual direct launch.
// A failed proof keeps the original full dirty range; zero is never a narrowed mark.
inline std::optional<u32> ComputeWriteBoundEnd(
    const ComputeWriteBound& proof, const std::array<u32, 3>& compiled_local,
    const std::array<u32, 3>& actual_local, const std::array<u32, 3>& groups,
    bool indirect, u32 binding_alignment) {
    if (indirect || compiled_local != actual_local || !proof.constant_end ||
        !std::has_single_bit(proof.alignment) || proof.alignment != binding_alignment) {
        return std::nullopt;
    }
    u64 end = proof.constant_end;
    for (size_t axis = 0; axis < 3; ++axis) {
        if (!actual_local[axis] || !groups[axis]) return std::nullopt;
        for (size_t kind = 0; kind < 2; ++kind) {
            const u64 extent = (kind == 0 ? groups[axis] : actual_local[axis]) - 1;
            const u64 term = extent * proof.coefficients[kind * 3 + axis];
            if (term > 0xffffffffULL - end) return std::nullopt;
            end += term;
        }
    }
    return static_cast<u32>(end);
}

} // namespace Shader
