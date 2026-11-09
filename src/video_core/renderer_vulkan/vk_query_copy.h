// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

namespace Vulkan {

struct QueryCopyRangePlan {
    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> ranges;
    std::vector<size_t> groups;
    VkDeviceSize total_size{};
};

template <typename Query, size_t Extent>
QueryCopyRangePlan PlanQueryCopyRanges(std::span<Query, Extent> queries, VkDeviceSize page_size) {
    if (page_size == 0 || (page_size & (page_size - 1)) != 0) {
        throw std::invalid_argument("Query page size must be a power of two");
    }
    constexpr auto Max = std::numeric_limits<VkDeviceSize>::max();
    const auto mask = page_size - 1;
    QueryCopyRangePlan result;
    for (const auto& query : queries) {
        if (query.size == 0) {
            continue;
        }
        if (query.size > Max - query.address || query.address + query.size > Max - mask ||
            query.size > Max - result.total_size) {
            throw std::overflow_error("Query destination range wraps");
        }
        result.total_size += query.size;
        result.ranges.emplace_back(query.address & ~mask,
                                   (query.address + query.size + mask) & ~mask);
    }
    std::sort(result.ranges.begin(), result.ranges.end());
    size_t count = 0;
    for (const auto range : result.ranges) {
        if (count != 0 && range.first <= result.ranges[count - 1].second) {
            result.ranges[count - 1].second = std::max(result.ranges[count - 1].second, range.second);
        } else {
            result.ranges[count++] = range;
        }
    }
    result.ranges.resize(count);
    for (const auto& range : result.ranges) {
        if (range.second - range.first > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("Query destination range exceeds buffer-cache size");
        }
    }
    result.groups.reserve(queries.size());
    for (const auto& query : queries) {
        if (query.size == 0) {
            result.groups.push_back(std::numeric_limits<size_t>::max());
            continue;
        }
        const auto after = std::upper_bound(result.ranges.begin(), result.ranges.end(), query.address,
                                           [](VkDeviceSize address, const auto& range) {
                                               return address < range.first;
                                           });
        result.groups.push_back(static_cast<size_t>(std::distance(result.ranges.begin(), after) - 1));
    }
    return result;
}

// A query batch may write the same guest address repeatedly. vkCmdCopyBuffer forbids
// overlapping destinations in one command; retain the bytes from the last query write.
inline std::vector<VkBufferCopy> NonOverlappingQueryCopies(std::span<const VkBufferCopy> copies) {
    std::map<VkDeviceSize, VkBufferCopy> writes;
    for (const auto& copy : copies) {
        if (copy.size == 0) {
            continue;
        }
        if (copy.size > std::numeric_limits<VkDeviceSize>::max() - copy.dstOffset ||
            copy.size > std::numeric_limits<VkDeviceSize>::max() - copy.srcOffset) {
            throw std::overflow_error("Query copy range wraps VkDeviceSize");
        }
        const VkDeviceSize end = copy.dstOffset + copy.size;
        auto it = writes.lower_bound(copy.dstOffset);
        if (it != writes.begin()) {
            const auto previous = std::prev(it);
            if (previous->second.dstOffset + previous->second.size > copy.dstOffset) {
                it = previous;
            }
        }
        while (it != writes.end() && it->first < end) {
            const VkBufferCopy old = it->second;
            it = writes.erase(it);
            const VkDeviceSize old_end = old.dstOffset + old.size;
            if (old.dstOffset < copy.dstOffset) {
                writes.emplace(old.dstOffset, VkBufferCopy{old.srcOffset, old.dstOffset,
                                                           copy.dstOffset - old.dstOffset});
            }
            if (old_end > end) {
                writes.emplace(
                    end, VkBufferCopy{old.srcOffset + end - old.dstOffset, end, old_end - end});
                break;
            }
        }
        writes.emplace(copy.dstOffset, copy);
    }
    std::vector<VkBufferCopy> result;
    result.reserve(writes.size());
    for (const auto& [offset, copy] : writes) {
        result.push_back(copy);
    }
    return result;
}

} // namespace Vulkan
