// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <functional>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/gpu_dirty_memory_manager.h"

namespace {

using Range = std::pair<PAddr, size_t>;
constexpr PAddr Base = 0x100000;
constexpr PAddr CacheLineSize = 64;

void SortUnique(std::vector<PAddr>& lines) {
    std::ranges::sort(lines);
    lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
}

std::vector<PAddr> ReferenceLines(std::span<const Range> writes) {
    std::vector<PAddr> lines;
    for (const auto [address, size] : writes) {
        if (size == 0) {
            continue;
        }
        for (PAddr line = address - address % CacheLineSize; line < address + size;
             line += CacheLineSize) {
            lines.push_back(line);
        }
    }
    SortUnique(lines);
    return lines;
}

void CheckWrites(std::span<const Range> writes) {
    Core::GPUDirtyMemoryManager manager;
    for (const auto [address, size] : writes) {
        manager.Collect(address, size);
    }

    std::vector<PAddr> lines;
    std::function<void(PAddr, size_t)> gather = [&](PAddr address, size_t size) {
        REQUIRE(address % CacheLineSize == 0);
        REQUIRE(size != 0);
        REQUIRE(size % CacheLineSize == 0);
        for (PAddr line = address; line < address + size; line += CacheLineSize) {
            lines.push_back(line);
        }
    };
    manager.Gather(gather);
    SortUnique(lines);
    CHECK(lines == ReferenceLines(writes));

    lines.clear();
    manager.Gather(gather);
    CHECK(lines.empty());
}

} // Anonymous namespace

TEST_CASE("GPUDirtyMemoryManager preserves writes across tracking blocks", "[core]") {
    constexpr std::array writes{
        Range{Base, 4096},         Range{Base + 0x7f0, 32}, Range{Base, 2048},
        Range{Base + 17, 1},       Range{Base + 0x7f0, 16}, Range{Base + 0x800, 64},
        Range{Base + 0x7f0, 4096}, Range{Base, 0},          Range{Base + 17, 0},
    };
    for (const auto& write : writes) {
        CAPTURE(write.first, write.second);
        CheckWrites(std::span{&write, 1});
    }

    constexpr std::array overlapping{
        Range{Base + 0x1000, 4096},
        Range{Base + 13, 4096},
        Range{Base + 33, 8},
        Range{Base + 0x2000, 0},
    };
    CheckWrites(overlapping);
}

TEST_CASE("GPUDirtyMemoryManager matches cache-line coverage", "[core]") {
    constexpr std::array<PAddr, 8> offsets{0, 1, 63, 64, 0x7bf, 0x7f0, 0x7ff, 0xfff};
    constexpr std::array<size_t, 7> sizes{0, 1, 16, 63, 64, 2048, 4096};
    for (const PAddr offset : offsets) {
        for (const size_t size : sizes) {
            CAPTURE(offset, size);
            const std::array writes{Range{Base + offset, size}};
            CheckWrites(writes);
        }
    }
}
