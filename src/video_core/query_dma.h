// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>

#include "common/common_types.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon::QueryDma {

/// Writes the query reports in [source, source + size) to guest memory, waiting for pending host
/// results. Runs where the rasterizer's caches are owned. (From Lichii #215.)
inline void FlushPending(const Tegra::MemoryManager& memory,
                         VideoCore::RasterizerInterface& rasterizer, GPUVAddr source, size_t size) {
    // EDEN_NX2_QUERY_DMA_FLUSH=0 skips it (measurement only).
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_QUERY_DMA_FLUSH");
        return value == nullptr || value[0] != '0';
    }();
    if (!enabled) {
        return;
    }
    constexpr GPUVAddr page_size = 0x1000;
    const GPUVAddr end = source + size;
    for (GPUVAddr begin = source; begin < end;) {
        const GPUVAddr page_end = (std::min)((begin & ~(page_size - 1)) + page_size, end);
        if (const auto device = memory.GpuToCpuAddress(begin)) {
            rasterizer.FlushQueryResults(*device, page_end - begin);
        }
        begin = page_end;
    }
}

} // namespace VideoCommon::QueryDma
