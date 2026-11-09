// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "common/literals.h"
#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_storage_feedback.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {
using namespace Common::Literals;

constexpr VkDeviceSize BUFFER_SIZE = 16_MiB; // 65536 shader stages
constexpr u32 INVALIDATE_INTERVAL = 4096;

std::unique_ptr<StorageFeedback> instance;

VkBufferCreateInfo BufferInfo(VkBufferUsageFlags usage) {
    return VkBufferCreateInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = BUFFER_SIZE,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
}

VkMemoryBarrier Barrier(VkAccessFlags src, VkAccessFlags dst) {
    return VkMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src,
        .dstAccessMask = dst,
    };
}
} // Anonymous namespace

void StorageFeedback::Initialize(const Device& device, MemoryAllocator& allocator,
                                 Scheduler& scheduler) {
    const char* value = std::getenv("EDEN_NX2_SSBO_FEEDBACK");
    if ((value && std::strcmp(value, "0") == 0) ||
        device.GetStorageBufferAlignment() > REGION_SIZE) {
        return;
    }
    instance.reset(new StorageFeedback(allocator, scheduler));
    LOG_INFO(Render_Vulkan, "Storage buffer size feedback on (reads past {} MiB)",
             DefaultNativeSize() / 1_MiB);
}

void StorageFeedback::Shutdown() {
    instance.reset();
}

StorageFeedback* StorageFeedback::Get() {
    return instance.get();
}

u32 StorageFeedback::DefaultNativeSize() {
    static const u32 size = [] {
        const char* value = std::getenv("EDEN_NX2_NATIVE_SSBO_MB");
        const long megabytes = value ? std::strtol(value, nullptr, 10) : 0;
        return megabytes > 0 && megabytes <= 2048 ? static_cast<u32>(megabytes) * 1_MiB
                                                  : static_cast<u32>(8_MiB);
    }();
    return size;
}

StorageFeedback::StorageFeedback(MemoryAllocator& allocator, Scheduler& scheduler) {
    device_buffer = allocator.CreateBuffer(
        BufferInfo(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        MemoryUsage::DeviceLocal);
    download_buffer =
        allocator.CreateBuffer(BufferInfo(VK_BUFFER_USAGE_TRANSFER_DST_BIT), MemoryUsage::Download);
    std::memset(download_buffer.Mapped().data(), 0, BUFFER_SIZE);
    download_buffer.Flush();
    mapped = reinterpret_cast<const u32*>(download_buffer.Mapped().data());

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([buffer = *device_buffer](vk::CommandBuffer cmdbuf) {
        cmdbuf.FillBuffer(buffer, 0, VK_WHOLE_SIZE, 0);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               0,
                               Barrier(VK_ACCESS_TRANSFER_WRITE_BIT,
                                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
    });
}

std::optional<u32> StorageFeedback::AllocateRegion() {
    const u32 offset = next_region.fetch_add(REGION_SIZE, std::memory_order_relaxed);
    if (offset + REGION_SIZE > BUFFER_SIZE) {
        return std::nullopt;
    }
    return offset;
}

u32 StorageFeedback::Read(u32 region_offset, size_t binding) {
    if (reads_since_invalidate.fetch_add(1, std::memory_order_relaxed) + 1 >= INVALIDATE_INTERVAL) {
        reads_since_invalidate.store(0, std::memory_order_relaxed);
        download_buffer.Invalidate();
    }
    return mapped[region_offset / sizeof(u32) + binding];
}

void StorageFeedback::Tick(Scheduler& scheduler) {
    const VkDeviceSize used = (std::min<VkDeviceSize>)(
        next_region.load(std::memory_order_relaxed), BUFFER_SIZE);
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([source = *device_buffer, destination = *download_buffer,
                      used](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, Barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT));
        cmdbuf.CopyBuffer(source, destination,
                          VkBufferCopy{.srcOffset = 0, .dstOffset = 0, .size = used});
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                               Barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT));
    });
}

} // namespace Vulkan
