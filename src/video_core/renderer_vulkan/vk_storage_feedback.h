// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <optional>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// Storage buffers bound through bare guest pointers carry no size, so they are bound with a
/// guessed one (EDEN_NX2_NATIVE_SSBO_MB, 8 MiB by default). Shaders record the end of any read
/// past it here, one u32 per storage binding in a region per shader stage, and later bindings of
/// that slot grow to cover it. The GPU writes a device-local buffer, copied for the CPU once a
/// frame. EDEN_NX2_SSBO_FEEDBACK=0 turns it off.
class StorageFeedback {
public:
    static constexpr u32 REGION_SIZE = 256; // u32[Shader::Info::MAX_SSBOS]

    static void Initialize(const Device& device, MemoryAllocator& allocator, Scheduler& scheduler);
    static void Shutdown();
    /// Null when off or not initialized.
    [[nodiscard]] static StorageFeedback* Get();

    /// The default size of a bare-pointer storage binding; shaders report reads past it.
    [[nodiscard]] static u32 DefaultNativeSize();

    /// A region for one shader stage, or nothing when the buffer is full.
    [[nodiscard]] std::optional<u32> AllocateRegion();

    /// The byte end, from the binding's start, of the furthest read past the default size so far
    /// (as of the last copy), or 0.
    [[nodiscard]] u32 Read(u32 region_offset, size_t binding);

    /// Copies the regions in use for the CPU. Once a frame.
    void Tick(Scheduler& scheduler);

    [[nodiscard]] VkBuffer Handle() const {
        return *device_buffer;
    }

private:
    explicit StorageFeedback(MemoryAllocator& allocator, Scheduler& scheduler);

    vk::Buffer device_buffer;
    vk::Buffer download_buffer;
    const u32* mapped{};
    std::atomic<u32> next_region{REGION_SIZE}; // region 0 stays unused
    std::atomic<u32> reads_since_invalidate{};
};

} // namespace Vulkan
