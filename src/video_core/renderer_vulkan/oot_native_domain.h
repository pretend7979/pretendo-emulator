// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <optional>
#include <string_view>

#include "common/common_types.h"

namespace Vulkan {

// A launch-time baseline overrides both automatic and fixed diagnostic ranges.
inline bool OotNativeDomainBaselineRequested() noexcept {
    static const bool baseline = [] {
        const char* value = std::getenv("EDEN_NX2_OOT_NATIVE_DOMAIN_BASELINE");
        return value && std::string_view{value} == "1";
    }();
    return baseline;
}

constexpr bool OotNativeDomainPolicyEnabled(std::optional<bool> override, bool baseline,
                                          bool capture_stop) noexcept {
    return !baseline && !capture_stop && override.value_or(true);
}

constexpr bool IsOotNativeDomainTarget(u64 title, u64 native_key, u64 canonical_key, u32 cbuf,
                                     u32 cbuf_offset, u32 pointer_offset, bool written) noexcept {
    if (title != 0x0400486024DEC000ULL || written || cbuf != 0 ||
        (pointer_offset != 24 && pointer_offset != 32)) {
        return false;
    }
    return (cbuf_offset == 96 &&
            ((native_key == 0xd9b1657f3a0b4bc2ULL && canonical_key == 0x75d4369f3b1f5868ULL) ||
             (native_key == 0x203053e20b8c989aULL && canonical_key == 0xc73574a9766531cbULL))) ||
           (cbuf_offset == 0x100 &&
            ((native_key == 0x8dc54d1adc12150fULL && canonical_key == 0xe78efceda31f97a8ULL) ||
             (native_key == 0xf35f519f99b98203ULL && canonical_key == 0xf98012dc9a4465ffULL) ||
             (native_key == 0x4d9a7a9cfd503656ULL && canonical_key == 0x7f0d176391897d0eULL)));
}

} // namespace Vulkan
