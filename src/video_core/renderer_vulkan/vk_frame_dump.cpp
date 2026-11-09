// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "video_core/renderer_vulkan/vk_frame_dump.h"
#include "video_core/renderer_vulkan/oot_native_domain.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {
namespace {

using VideoCore::Surface::PixelFormat;

constexpr size_t SIZE_CAP = size_t{1024} << 20;
constexpr size_t SAMPLED_SIZE_CAP = size_t{512} << 20;
constexpr size_t PRODUCER_SIZE_CAP = size_t{128} << 20;
constexpr size_t FOG_COMPUTE_RESERVE = size_t{128} * 72 * 96 * 10;
constexpr size_t LIGHTING_OUTPUT_RESERVE = size_t{2} * 64 * 36 * 96 * 4;
constexpr size_t PRESENT_RESERVE = size_t{32} << 20;
constexpr size_t STAGING_BUDGET = size_t{64} << 20;

std::atomic_bool& DelayedSkyCaptureArmed() {
    static std::atomic_bool armed{false};
    return armed;
}

std::optional<size_t> CaptureByteSize(u32 width, u32 height, u32 depth, u32 texel_bytes) {
    size_t bytes = texel_bytes;
    if (!bytes) return std::nullopt;
    for (const u32 dimension : {width, height, depth}) {
        if (!dimension || bytes > std::numeric_limits<size_t>::max() / dimension)
            return std::nullopt;
        bytes *= dimension;
    }
    return bytes;
}

std::optional<size_t> CaptureStorageByteSize(u32 width, u32 height, u32 depth,
                                              u32 bytes_per_block, u32 block_width,
                                              u32 block_height) {
    if ((block_width != 1 && block_width != 4) || block_width != block_height)
        return std::nullopt;
    const u32 columns = static_cast<u32>((u64{width} + block_width - 1) / block_width);
    const u32 rows = static_cast<u32>((u64{height} + block_height - 1) / block_height);
    return CaptureByteSize(columns, rows, depth, bytes_per_block);
}

u32 ReadPassSetting(const char* name, u32 fallback) {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    const std::string_view text{value};
    u32 parsed{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    return error == std::errc{} && end == text.data() + text.size() ? parsed : fallback;
}

std::optional<u64> InitialDrawShaderAnchor() {
    static const std::optional<u64> anchor = []() -> std::optional<u64> {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_DRAW_SHADER");
        if (!value) return std::nullopt;
        std::string_view text{value};
        if (text.starts_with("0x") || text.starts_with("0X")) text.remove_prefix(2);
        u64 parsed{};
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed, 16);
        if (error != std::errc{} || end != text.data() + text.size() || parsed == 0)
            return std::nullopt;
        return parsed;
    }();
    return anchor;
}

std::optional<u64>& DrawShaderAnchor() {
    static std::optional<u64> anchor = InitialDrawShaderAnchor();
    return anchor;
}

bool RepeatCaptures() {
    static const bool repeat = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_REPEAT");
        return value && std::strcmp(value, "1") == 0;
    }();
    return repeat;
}

u32& DrawPassSelection() {
    // Each capture follows one frame. Its pass number can vary with the visible scene.
    static u32 value = DrawShaderAnchor() ? ~0U :
        ReadPassSetting("EDEN_NX2_FRAME_DUMP_DRAW_PASS", ~0U);
    return value;
}

u32 DrawCapturePass() {
    return DrawPassSelection();
}

bool& PhaseCaptures() {
    static bool phases = false;
    return phases;
}

u32& BurstFrames() {
    // Bounded, explicitly triggered OOT lighting diagnostics; ordinary captures stay unchanged.
    static u32 count = 1;
    return count;
}

std::optional<GPUVAddr> FogDepthAddress() {
    static const std::optional<GPUVAddr> address = []() -> std::optional<GPUVAddr> {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_FOG_DEPTH_ADDRESS");
        if (!value) return std::nullopt;
        std::string_view text{value};
        if (text.starts_with("0x") || text.starts_with("0X")) text.remove_prefix(2);
        GPUVAddr parsed{};
        const auto [last, error] = std::from_chars(text.data(), text.data() + text.size(), parsed, 16);
        if (error != std::errc{} || last != text.data() + text.size() || parsed == 0)
            return std::nullopt;
        return parsed;
    }();
    return address;
}

u32& AnchorDraw() {
    static u32 draw = ~0U;
    return draw;
}

std::optional<u32>& MaterialInputsSelection() {
    static std::optional<u32> selected;
    return selected;
}

std::optional<std::array<u32, 2>>& CanopyInputsSelection() {
    static std::optional<std::array<u32, 2>> selected;
    return selected;
}

u32 ParticleProducerSkip() {
    static const u32 skip = [] {
        if (!std::getenv("EDEN_NX2_FRAME_DUMP_PRODUCER_SKIP")) return 0U;
        return ReadPassSetting("EDEN_NX2_FRAME_DUMP_PRODUCER_SKIP", ~0U);
    }();
    return skip;
}

u64& FogInputsSelection() {
    static u64 selected{};
    return selected;
}

std::atomic<u64>& LiveCaptureProducerKey() {
    static std::atomic<u64> key{};
    return key;
}

bool ConsumeRepeatTrigger(const std::filesystem::path& path) {
    // The caller moves completed output aside before creating the next trigger directory.
    std::error_code error;
    if (std::filesystem::exists(path.parent_path() / "manifest.jsonl", error) || error) return false;
    std::ifstream input{path, std::ios::binary};
    if (!input) return false;
    std::array<char, 64> buffer{};
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (input.bad() || count == static_cast<std::streamsize>(buffer.size())) return false;
    std::string_view text{buffer.data(), static_cast<size_t>(count)};
    constexpr std::string_view whitespace{" \t\r\n"};
    const size_t begin = text.find_first_not_of(whitespace);
    text = begin == std::string_view::npos ? std::string_view{} : text.substr(begin);
    const size_t end = text.find_last_not_of(whitespace);
    if (end != std::string_view::npos) text = text.substr(0, end + 1);
    std::optional<std::array<u32, 2>> canopy_inputs;
    std::string canopy_anchor;
    if (text == "canopy-inputs" || text.starts_with("canopy-inputs ")) {
        if (Settings::GetCurrentProgramID() != 0x0400486024DEC000ULL ||
            FrameDump::ComputeCaptureStopEnabled()) return false;
        std::array<u32, 2> selection{};
        if (text != "canopy-inputs") {
            auto values = text.substr(14);
            for (u32 index = 0; index < selection.size(); ++index) {
                const size_t space = values.find(' ');
                const auto value = values.substr(0, space);
                const auto [last, parse_error] =
                    std::from_chars(value.data(), value.data() + value.size(), selection[index]);
                if (parse_error != std::errc{} || last != value.data() + value.size() ||
                    selection[index] > (index == 0 ? 4U : 63U)) return false;
                if (space == std::string_view::npos) break;
                if (index == 1) return false;
                values.remove_prefix(space + 1);
            }
        }
        canopy_inputs = selection;
        canopy_anchor = fmt::format("{:016x}", FrameDump::CanopyInputPair(selection[0])[1]);
        text = canopy_anchor;
    }
    std::optional<u32> material_skip;
    if (text == "material-inputs" || text.starts_with("material-inputs ")) {
        if (Settings::GetCurrentProgramID() != 0x0400486024DEC000ULL ||
            FrameDump::ComputeCaptureStopEnabled()) return false;
        u32 skip = 9;
        if (text != "material-inputs") {
            const auto value = text.substr(16);
            const auto [last, parse_error] =
                std::from_chars(value.data(), value.data() + value.size(), skip);
            if (parse_error != std::errc{} || last != value.data() + value.size() || skip > 63)
                return false;
        }
        material_skip = skip;
        text = "6cb075ad84a5c605";
    }
    const bool fog_inputs = text == "fog-inputs";
    const bool lighting_inputs = text == "lighting-inputs";
    const bool sh_inputs = text == "sh-inputs";
    if (fog_inputs || lighting_inputs || sh_inputs) {
        if (Settings::GetCurrentProgramID() != 0x0400486024DEC000ULL ||
            FrameDump::ProducerNativeKey() != 0xe883414e8c9be33bULL ||
            FrameDump::ComputeCaptureStopEnabled()) return false;
        text = sh_inputs ? "phase 6cb075ad84a5c605" :
               lighting_inputs ? "phase d35ea867501a0d2d" : "phase 3ddfb1d7a854c849";
    }
    u32 burst_frames = 1;
    const bool burst = text.starts_with("burst ");
    if (burst) {
        text.remove_prefix(6);
        const auto space = text.find(' ');
        if (space == std::string_view::npos) return false;
        const auto [last, parse_error] =
            std::from_chars(text.data(), text.data() + space, burst_frames);
        if (parse_error != std::errc{} || last != text.data() + space ||
            burst_frames < 2 || burst_frames > 64) return false;
        text.remove_prefix(space + 1);
    }
    const bool phases = burst || text.starts_with("phase ");
    if (phases && !burst) text.remove_prefix(6);
    if (phases && text.empty()) return false;
    std::optional<u64> anchor = InitialDrawShaderAnchor();
    if (!text.empty()) {
        if (text.size() != 16) return false;
        u64 parsed{};
        const auto [last, parse_error] = std::from_chars(text.data(), text.data() + text.size(), parsed, 16);
        if (parse_error != std::errc{} || last != text.data() + text.size() || parsed == 0) return false;
        anchor = parsed;
    }
    input.close();
    if (!std::filesystem::remove(path, error) || error) return false;
    // These are touched only on the GPU thread, together with per-capture counters below.
    MaterialInputsSelection() = material_skip;
    CanopyInputsSelection() = canopy_inputs;
    FogInputsSelection() = sh_inputs ? 0x203053e20b8c989aULL :
                           lighting_inputs ? 0xd9b1657f3a0b4bc2ULL :
                           fog_inputs ? 0xe883414e8c9be33bULL : 0;
    DrawShaderAnchor() = anchor;
    PhaseCaptures() = phases;
    BurstFrames() = burst_frames;
    AnchorDraw() = ~0U;
    DrawPassSelection() = anchor ? ~0U : ReadPassSetting("EDEN_NX2_FRAME_DUMP_DRAW_PASS", ~0U);
    return true;
}

u32 FirstCapturePass() {
    // Before the anchor appears, retain metadata without spending the readback budget.
    if (DrawShaderAnchor()) return DrawCapturePass();
    static const u32 value = ReadPassSetting("EDEN_NX2_FRAME_DUMP_FIRST_PASS", 0);
    return value;
}


} // Anonymous namespace

u32 FrameDump::LightingRangeMinimum(u64 native_key, u64 canonical_key, u32 cbuf,
    u32 cbuf_offset, u32 pointer_offset, bool is_written) const noexcept {
    if (NativeDomainCorrectionEnabled() || OotNativeDomainBaselineRequested()) return 0;
    const bool d9 = lighting_range_diagnostic && native_key == 0xd9b1657f3a0b4bc2ULL &&
                    canonical_key == 0x75d4369f3b1f5868ULL;
    const bool sh = sh_range_diagnostic && native_key == 0x203053e20b8c989aULL &&
                    canonical_key == 0xc73574a9766531cbULL;
    if ((!d9 && !sh) || is_written || cbuf != 0 || cbuf_offset != 96 ||
        Settings::GetCurrentProgramID() != 0x0400486024DEC000ULL || ComputeCaptureStopEnabled()) return 0;
    // Fixed bounds derived from the captured scene's 48 tile records. Diagnostic only;
    // these are not generic allocation sizes or a final native-pointer policy.
    if (sh) return pointer_offset == 24 ? 35191040U : pointer_offset == 32 ? 121642624U : 0U;
    return pointer_offset == 24 ? lighting_range_minimum24 :
           pointer_offset == 32 ? lighting_range_minimum32 : 0U;
}

bool FrameDump::NativeDomainCorrectionEnabled() const noexcept {
    return Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL &&
           OotNativeDomainPolicyEnabled(native_domain_range, OotNativeDomainBaselineRequested(),
                                       ComputeCaptureStopEnabled());
}

bool FrameDump::NativeDomainRangeEnabled(u64 native_key, u64 canonical_key, u32 cbuf,
    u32 cbuf_offset, u32 pointer_offset, bool is_written) const noexcept {
    return NativeDomainCorrectionEnabled() && IsOotNativeDomainTarget(
        Settings::GetCurrentProgramID(), native_key, canonical_key, cbuf, cbuf_offset,
        pointer_offset, is_written);
}

bool FrameDump::GraphicsIndirectRangeTrial() const noexcept {
    return graphics_indirect_range_trial &&
        Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL && !ComputeCaptureStopEnabled();
}

bool FrameDump::GraphicsNativePointerTrial() const noexcept {
    return graphics_native_pointer_trial &&
        Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL && !ComputeCaptureStopEnabled();
}

bool FrameDump::ConsumeGraphicsBufferTrigger() {
    if (!RepeatCaptures() || Capturing() || ComputeCaptureStopEnabled() ||
        Settings::GetCurrentProgramID() != 0x0400486024DEC000ULL ||
        ProducerNativeKey() != 0xe883414e8c9be33bULL) return false;
    std::ifstream input{trigger_file, std::ios::binary};
    if (!input) return false;
    std::array<char, 64> buffer{};
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (input.bad() || count == static_cast<std::streamsize>(buffer.size())) return false;
    std::string_view text{buffer.data(), static_cast<size_t>(count)};
    constexpr std::string_view whitespace{" \t\r\n"};
    const size_t begin = text.find_first_not_of(whitespace);
    text = begin == std::string_view::npos ? std::string_view{} : text.substr(begin);
    const size_t end = text.find_last_not_of(whitespace);
    if (end != std::string_view::npos) text = text.substr(0, end + 1);
    bool* setting{};
    bool value{};
    if (text == "vegetation-range-on" || text == "vegetation-range-off") {
        setting = &graphics_indirect_range_trial;
        value = text == "vegetation-range-on";
    } else if (text == "graphics-native-on" || text == "graphics-native-off") {
        setting = &graphics_native_pointer_trial;
        value = text == "graphics-native-on";
    } else {
        return false;
    }
    input.close();
    std::error_code error;
    if (!std::filesystem::remove(trigger_file, error) || error) return false;
    *setting = value;
    LOG_INFO(Render_Vulkan, "OOT graphics buffer trial {} at frame {} (indirect64={}, native={})",
             text, frame, graphics_indirect_range_trial, graphics_native_pointer_trial);
    return true;
}

size_t FrameDump::ProducerSizeCap() const noexcept {
    if (NativeDomainCorrectionEnabled() && (CapturingLightingInputs() || CapturingShInputs()))
        return size_t{336} << 20;
    return (CapturingLightingInputs() && LightingRangeMinimum(0xd9b1657f3a0b4bc2ULL,
                0x75d4369f3b1f5868ULL, 0, 96, 24, false)) ||
           (CapturingShInputs() && LightingRangeMinimum(0x203053e20b8c989aULL,
                0xc73574a9766531cbULL, 0, 96, 24, false))
        ? size_t{272} << 20 : PRODUCER_SIZE_CAP;
}

bool FrameDump::ConsumeLightingRangeTrigger() {
    if (!RepeatCaptures() || Capturing() || ComputeCaptureStopEnabled() ||
        Settings::GetCurrentProgramID() != 0x0400486024DEC000ULL ||
        ProducerNativeKey() != 0xe883414e8c9be33bULL) return false;
    std::ifstream input{trigger_file, std::ios::binary};
    if (!input) return false;
    std::array<char, 64> buffer{};
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (input.bad() || count == static_cast<std::streamsize>(buffer.size())) return false;
    std::string_view text{buffer.data(), static_cast<size_t>(count)};
    constexpr std::string_view whitespace{" \t\r\n"};
    const size_t begin = text.find_first_not_of(whitespace);
    text = begin == std::string_view::npos ? std::string_view{} : text.substr(begin);
    const size_t end = text.find_last_not_of(whitespace);
    if (end != std::string_view::npos) text = text.substr(0, end + 1);
    if (text == "native-domain-on" || text == "native-domain-off" ||
        text == "native-domain-auto") {
        input.close();
        std::error_code error;
        if (!std::filesystem::remove(trigger_file, error) || error) return false;
        native_domain_range = text == "native-domain-auto" ? std::nullopt
            : std::optional<bool>{text == "native-domain-on"};
        // An explicit baseline must not retain a previous fixed scene-range experiment.
        if (text == "native-domain-off") {
            lighting_range_diagnostic = false;
            sh_range_diagnostic = false;
        }
        LOG_INFO(Render_Vulkan,
                 "OOT native readonly domain {} at frame {} (effective={}, baseline_env={})",
                 text, frame, NativeDomainCorrectionEnabled(), OotNativeDomainBaselineRequested());
        return true;
    }
    const bool sh = text == "sh-range-on" || text == "sh-range-off";
    const bool turn_off = text == "range-off" || text == "sh-range-off";
    u32 minimum24 = 35191040;
    u32 minimum32 = 121642624;
    if (!sh && !turn_off && text != "range-on") {
        if (!text.starts_with("range-on ")) return false;
        text.remove_prefix(9);
        const auto read_size = [&](u32& value) {
            const size_t first = text.find_first_not_of(whitespace);
            if (first == std::string_view::npos) return false;
            text.remove_prefix(first);
            const auto [last, parse_error] =
                std::from_chars(text.data(), text.data() + text.size(), value);
            if (parse_error != std::errc{} || last == text.data()) return false;
            text.remove_prefix(static_cast<size_t>(last - text.data()));
            if (!text.empty() && whitespace.find(text.front()) == std::string_view::npos) return false;
            return true;
        };
        if (!read_size(minimum24) || !read_size(minimum32) ||
            text.find_first_not_of(whitespace) != std::string_view::npos) return false;
    }
    if (!turn_off && (minimum24 <= (u32{8} << 20) || minimum24 > 35191040 ||
        minimum32 <= (u32{8} << 20) || minimum32 > 130027776 ||
        (minimum24 & 3) != 0 || (minimum32 & 3) != 0)) return false;
    input.close();
    std::error_code error;
    if (!std::filesystem::remove(trigger_file, error) || error) return false;
    if (sh) {
        sh_range_diagnostic = !turn_off;
    } else {
        if (!turn_off) {
            lighting_range_minimum24 = minimum24;
            lighting_range_minimum32 = minimum32;
        }
        lighting_range_diagnostic = !turn_off;
    }
    LOG_INFO(Render_Vulkan,
             "OOT {} scene range diagnostic {} at frame {} (child24={}, child32={})",
             sh ? "SH" : "D9", turn_off ? "disabled" : "enabled", frame,
             sh ? 35191040U : lighting_range_minimum24,
             sh ? 121642624U : lighting_range_minimum32);
    return true;
}

FrameDump::FrameDump(const Device& device_, MemoryAllocator& memory_allocator_, Scheduler& scheduler_)
    : device{device_}, memory_allocator{memory_allocator_}, scheduler{scheduler_} {
    const char* const dir = std::getenv("EDEN_NX2_FRAME_DUMP");
    if (DelayedSkyCaptureEnabled()) {
        const char* immediate = std::getenv("EDEN_NX2_COMPUTE_CAPTURE_STOP");
        if ((immediate && *immediate) || (dir && *dir))
            throw std::runtime_error("DIAGNOSTIC delayed sky requires other capture modes to be unset");
    }
    if (!dir || *dir == '\0') {
        return;
    }
    enabled = true;
    directory = dir;
    start = std::chrono::steady_clock::now();
    trigger_file = directory / "capture.now";
    LOG_INFO(Render_Vulkan, "Frame dump armed: create {} to capture one frame", trigger_file.string());
}

FrameDump::~FrameDump() {
    LiveCaptureProducerKey().store(0, std::memory_order_release);
}

bool FrameDump::CapturingSampled() const noexcept {
    static const bool sampled = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_SAMPLED");
        return value && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    return capturing && sampled && DrawShaderAnchor().has_value() &&
           DrawCapturePass() != ~0U && pass_index == DrawCapturePass() &&
           (!PhaseCaptures() || draws == AnchorDraw() + 1);
}

bool FrameDump::CapturingFogDepth() const noexcept {
    return capturing && BurstFrames() >= 2 && BurstFrames() <= 64 && FogDepthAddress() &&
           DrawShaderAnchor() == std::optional<u64>{0x3ddfb1d7a854c849ULL};
}

bool FrameDump::CaptureFogDepth() {
    if (!CapturingFogDepth() || fog_depth_recorded || draws == 0 ||
        pass_index != DrawCapturePass() || draws != AnchorDraw() + 1) return false;
    fog_depth_recorded = true;
    const auto skip = [&](std::string_view reason) {
        manifest << fmt::format(
            R"({{"kind":"depth_before_fog","pass":{},"before_draw":{},"requested_gpu_addr":"{:#x}","skipped":"{}"}})",
            pass_index, draws - 1, *FogDepthAddress(), reason) << '\n';
        return false;
    };
    if (!fog_depth_target) return skip("requested D32 depth attachment not observed this frame");
    const Target& target = *fog_depth_target;
    constexpr size_t DepthByteCap = size_t{1440} * 810 * sizeof(float);
    const auto bytes = CaptureByteSize(target.width, target.height, target.depth, sizeof(float));
    if (target.gpu_addr != *FogDepthAddress() || target.format != PixelFormat::D32_FLOAT ||
        target.aspect != VK_IMAGE_ASPECT_DEPTH_BIT || target.depth != 1 || target.msaa ||
        !bytes || *bytes > DepthByteCap) return skip("unsupported or oversized depth target");
    return Record(
        target, fmt::format("{:04}_pass{:03}_before_draw{:04}_depth", sequence++, pass_index, draws - 1),
        fmt::format(R"("kind":"depth_before_fog","lighting_stage":"before_fog_depth","pass":{},"before_draw":{},"requested_gpu_addr":"{:#x}")",
                    pass_index, draws - 1, *FogDepthAddress()));
}

bool FrameDump::NoteSampled(std::span<const Sampled> textures) {
    if (!CapturingSampled() || draws == 0) return false;
    struct Seen {
        Target target;
        u32 binding;
    };
    std::vector<Seen> seen;
    const auto words_json = [](const std::array<u32, 8>& words) {
        std::string text;
        for (const u32 word : words) text += fmt::format("{}{}", text.empty() ? "" : ",", word);
        return text;
    };
    bool recorded = false;
    if (draws == pass_first_draw + 1 || draws == AnchorDraw() + 1) {
        // Preserve the incoming target before the first shader-anchored draw modifies it.
        for (const Target& target : pass_targets) {
            recorded |= Record(
                target,
                fmt::format("{:04}_pass{:03}_start_before_draw{:04}_rt{}", sequence++, pass_index,
                            draws - 1, target.slot),
                fmt::format(R"("kind":"pass_start","pass":{},"before_draw":{},"slot":{})",
                            pass_index, draws - 1, target.slot));
        }
    }
    recorded |= CaptureFogDepth();
    // Temporal runs retain the incoming target above, without repeatedly downloading textures.
    if (BurstFrames() > 1) return recorded;
    for (u32 index = 0; index < textures.size(); ++index) {
        const Sampled& texture = textures[index];
        std::string tags = fmt::format(
            R"("pass":{},"draw":{},"binding":{},"stage":{},"tic":{},"resolved_tic":{},"tsc":{},"tic_valid":{},"tsc_valid":{},"tic_words":[{}],"tsc_words":[{}])",
            pass_index, draws - 1, index, texture.stage, texture.tic, texture.resolved_tic, texture.tsc,
            texture.tic_valid, texture.tsc_valid, words_json(texture.tic_words),
            words_json(texture.tsc_words));
        const ImageView* const view = texture.view;
        const Image* const image = view ? view->BackingImage() : nullptr;
        const char* skipped = !image ? "no backing image" : nullptr;
        if (view) {
            const auto& swizzle = view->Swizzle();
            tags += fmt::format(
                R"(,"view_format":"{}","view_type":{},"view_width":{},"view_height":{},"view_depth":{},"base_level":{},"levels":{},"base_layer":{},"layers":{},"swizzle":[{},{},{},{}])",
                view->format, static_cast<u32>(view->type), view->size.width, view->size.height,
                view->size.depth, view->range.base.level, view->range.extent.levels,
                view->range.base.layer, view->range.extent.layers, static_cast<u32>(swizzle.r),
                static_cast<u32>(swizzle.g), static_cast<u32>(swizzle.b), static_cast<u32>(swizzle.a));
        }
        if (image) {
            using namespace VideoCore::Surface;
            const auto& info = image->info;
            tags += fmt::format(
                R"(,"image_format":"{}","image_type":{},"image_width":{},"image_height":{},"image_depth":{},"samples":{},"image_gpu_addr":"{:#x}","host_level_offset":{},"image_flags":{})",
                info.format, static_cast<u32>(info.type), info.size.width, info.size.height,
                info.size.depth, info.num_samples, image->gpu_addr, image->host_level_offset,
                static_cast<u32>(image->flags));
            if (info.num_samples > 1) skipped = "msaa";
            else if ((GetFormatType(info.format) != SurfaceType::ColorTexture ||
                      GetFormatType(view->format) != SurfaceType::ColorTexture) &&
                     !(info.format == PixelFormat::D32_FLOAT && view->format == PixelFormat::D32_FLOAT))
                skipped = "unsupported depth/stencil";
            else if (DefaultBlockWidth(info.format) != 1 || DefaultBlockHeight(info.format) != 1)
                skipped = "compressed";
            else if (info.type != VideoCommon::ImageType::e2D &&
                     info.type != VideoCommon::ImageType::e3D) skipped = "not 2D/3D";
            else if (image->host_level_offset != 0) skipped = "trimmed host mip levels";
            else if (True(image->flags & (VideoCommon::ImageFlagBits::Unbacked |
                                         VideoCommon::ImageFlagBits::SparseLayers |
                                         VideoCommon::ImageFlagBits::SparseTiles))) skipped = "unbacked/sparse host storage";
            else if (info.is_sparse) skipped = "sparse";
            else if (!(image->UsageFlags() & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) skipped = "not transferable";
        }
        const auto target = !skipped ? MakeTarget(*texture.view, index, true) : std::nullopt;
        if (!target && !skipped) skipped = "unsupported view";
        if (skipped) {
            manifest << fmt::format(R"({{"kind":"sampled_binding",{},"skipped":"{}"}})", tags, skipped) << '\n';
            continue;
        }
        // Small floating cubemaps can contain a bad face/mip that the base view hides.
        const bool small_cube = view->type == VideoCommon::ImageViewType::Cube &&
                                view->format == PixelFormat::B10G11R11_FLOAT &&
                                image->info.format == PixelFormat::B10G11R11_FLOAT &&
                                target->width <= 128 && target->height <= 128;
        s32 level_count = 1, layer_count = 1;
        if (small_cube) {
            const auto& range = view->range;
            level_count = range.extent.levels;
            layer_count = range.extent.layers;
            if (range.base.level < 0 || range.base.layer < 0 ||
                level_count <= 0 || level_count > 8 || layer_count != 6 ||
                s64{range.base.level} + level_count > image->info.resources.levels ||
                s64{range.base.layer} + layer_count > image->info.resources.layers) {
                manifest << fmt::format(R"({{"kind":"sampled_binding",{},"skipped":"invalid small cube range"}})",
                                        tags) << '\n';
                continue;
            }
        }
        for (s32 mip = 0; mip < level_count; ++mip) {
            for (s32 face = 0; face < layer_count; ++face) {
                Target subresource = *target;
                subresource.level += mip;
                subresource.layer += face;
                subresource.width = std::max(1U, target->width >> mip);
                subresource.height = std::max(1U, target->height >> mip);
                subresource.depth = std::max(1U, target->depth >> mip);
                const std::string subresource_tags = tags + fmt::format(
                    R"(,"sample_level":{},"sample_layer":{})", subresource.level, subresource.layer);
                const auto duplicate = std::ranges::find_if(seen, [&](const Seen& previous) {
                    const Target& a = previous.target;
                    return a.image == subresource.image && a.format == subresource.format &&
                           a.level == subresource.level && a.layer == subresource.layer &&
                           a.width == subresource.width && a.height == subresource.height &&
                           a.depth == subresource.depth &&
                           a.aspect == subresource.aspect;
                });
                if (duplicate != seen.end()) {
                    manifest << fmt::format(R"({{"kind":"sampled_binding",{},"copy_binding":{}}})",
                                            subresource_tags, duplicate->binding) << '\n';
                    continue;
                }
                seen.push_back({subresource, index});
                manifest << fmt::format(R"({{"kind":"sampled_binding",{},"copy_binding":{},"captured_levels":1,"captured_layers":1,"captured_depth":{}}})",
                                        subresource_tags, index, subresource.depth) << '\n';
                const std::string suffix = small_cube ? fmt::format("_mip{}_face{}", subresource.level,
                                                                   subresource.layer) : "";
                recorded |= Record(subresource,
                    fmt::format("{:04}_pass{:03}_before_draw{:04}_sample{}{}", sequence++, pass_index,
                                draws - 1, index, suffix),
                    fmt::format(R"("kind":"sampled",{})", subresource_tags), true);
            }
        }
    }
    return recorded;
}

std::optional<FrameDump::Target> FrameDump::MakeTarget(ImageView& view, u32 slot, bool sampled,
                                                        bool producer) const {
    using namespace VideoCore::Surface;
    const Image* const image = view.BackingImage();
    if (!image) {
        return std::nullopt;
    }
    const auto& info = image->info;
    const bool inspect_host_format = sampled && view.format == info.format &&
        (info.format == PixelFormat::D32_FLOAT || (producer &&
         (info.format == PixelFormat::D16_UNORM || info.format == PixelFormat::BC4_UNORM ||
          info.format == PixelFormat::BC4_SNORM)));
    const auto host_format = inspect_host_format
        ? MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, false, info.format).format
        : VK_FORMAT_UNDEFINED;
    const bool depth = sampled && view.format == info.format &&
        ((info.format == PixelFormat::D32_FLOAT && host_format == VK_FORMAT_D32_SFLOAT) ||
         (producer && info.format == PixelFormat::D16_UNORM && host_format == VK_FORMAT_D16_UNORM));
    const bool producer_bc4 = producer && sampled && view.format == info.format &&
        !True(image->flags & VideoCommon::ImageFlagBits::Converted) &&
        ((info.format == PixelFormat::BC4_UNORM && host_format == VK_FORMAT_BC4_UNORM_BLOCK) ||
         (info.format == PixelFormat::BC4_SNORM && host_format == VK_FORMAT_BC4_SNORM_BLOCK));
    const bool producer_noise = producer && view.format == info.format &&
        (info.format == PixelFormat::R8_UNORM || info.format == PixelFormat::R8G8_UNORM);
    const bool volume = info.type == VideoCommon::ImageType::e3D;
    if (volume && (!sampled || view.type != VideoCommon::ImageViewType::e3D ||
                   (info.format != PixelFormat::B10G11R11_FLOAT && info.format != PixelFormat::R16_FLOAT && !producer_noise) ||
                   view.format != info.format || info.num_samples != 1 || image->IsRescaled() ||
                   image->host_level_offset != 0 || view.range.base.layer != 0 ||
                   view.range.extent.layers != 1 || info.resources.layers != 1 ||
                   view.range.base.level < 0 || view.range.base.level >= 32 ||
                   view.range.base.level >= info.resources.levels || view.range.extent.levels <= 0 ||
                   s64{view.range.base.level} + view.range.extent.levels > info.resources.levels ||
                   !info.size.width || !info.size.height || !info.size.depth)) {
        return std::nullopt;
    }
    if (info.type == VideoCommon::ImageType::Buffer ||
        (GetFormatType(info.format) != SurfaceType::ColorTexture && !depth) ||
        (!producer_bc4 && (DefaultBlockWidth(info.format) != 1 || DefaultBlockHeight(info.format) != 1))) {
        return std::nullopt;
    }
    const bool rescaled = image->IsRescaled();
    const auto& resolution = Settings::values.resolution_info;
    const u32 width = rescaled ? resolution.ScaleUp(info.size.width) : info.size.width;
    const u32 height = rescaled ? resolution.ScaleUp(info.size.height) : info.size.height;
    const u32 level = static_cast<u32>(view.range.base.level);
    return Target{
        .slot = slot,
        .image = image->Handle(),
        .format = BytesPerBlock(view.format) == BytesPerBlock(info.format) ? view.format
                                                                           : info.format,
        .width = std::max(1U, width >> level),
        .height = std::max(1U, height >> level),
        .depth = volume ? std::max(1U, info.size.depth >> level) : 1U,
        .level = view.range.base.level,
        .layer = view.range.base.layer,
        .gpu_addr = image->gpu_addr,
        .rescaled = rescaled,
        .msaa = info.num_samples > 1,
        .aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
        .block_width = producer_bc4 ? 4U : 1U,
        .block_height = producer_bc4 ? 4U : 1U,
    };
}

bool FrameDump::BeginPass(const void* framebuffer, std::span<ImageView* const> colors,
                          ImageView* depth) {
    if (framebuffer == pass_framebuffer) {
        return false;
    }
    const bool recorded = EndPass();
    pass_framebuffer = framebuffer;
    pass_targets.clear();
    // Metadata only: retain the requested earlier depth attachment without reading it now.
    if (CapturingFogDepth() && depth) {
        if (const auto target = MakeTarget(*depth, 0, true);
            target && target->gpu_addr == *FogDepthAddress() &&
            target->format == PixelFormat::D32_FLOAT && target->aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
            fog_depth_target = target;
        }
    }
    for (u32 slot = 0; slot < colors.size(); ++slot) {
        if (!colors[slot]) {
            continue;
        }
        if (const auto target = MakeTarget(*colors[slot], slot)) {
            pass_targets.push_back(*target);
        }
    }
    pass_hashes.clear();
    pass_first_draw = draws;
    return recorded;
}

bool FrameDump::BeforeDraw() {
    if (PhaseCaptures() || !pass_framebuffer || pass_index != DrawCapturePass() ||
        pass_index < FirstCapturePass() || pass_hashes.empty() || draws <= pass_first_draw) {
        return false;
    }
    const bool recorded = CaptureLastDraw();
    pass_hashes.clear();
    return recorded;
}

void FrameDump::NoteDraw(std::span<const u64> shader_hashes, u64 guest_vs, u64 guest_ps) {
    bool select_anchor = true;
    if ((material_inputs_capture || canopy_inputs_capture) && DrawCapturePass() == ~0U) {
        const auto canopy_pair = CanopyInputShaders();
        const u64 first = canopy_inputs_capture ? canopy_pair[0] : 0x00e4eb4772d5076bULL;
        const u64 fragment = canopy_inputs_capture ? canopy_pair[1] : 0x6cb075ad84a5c605ULL;
        const bool exact_pipeline = shader_hashes.size() == 6 && shader_hashes[0] == 0 &&
            shader_hashes[1] == first && shader_hashes[2] == 0 &&
            shader_hashes[3] == 0 && shader_hashes[4] == 0 && shader_hashes[5] == fragment;
        select_anchor = Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL &&
            exact_pipeline && material_anchor_matches++ == material_input_skip;
    }
    if (const auto anchor = DrawShaderAnchor(); select_anchor && anchor &&
        DrawCapturePass() == ~0U &&
        std::ranges::find(shader_hashes, *anchor) != shader_hashes.end()) {
        DrawPassSelection() = pass_index;
        AnchorDraw() = draws;
        LOG_INFO(Render_Vulkan, "Frame dump shader anchor {:016x} selected pass {} before draw {}",
                 *anchor, pass_index, draws);
    }
    const bool capture_draw = pass_index == DrawCapturePass() && pass_index >= FirstCapturePass();
    if (capture_draw && !PhaseCaptures()) pass_hashes.clear();
    for (const u64 hash : shader_hashes) {
        if (hash != 0 && pass_hashes.size() < 32 &&
            std::ranges::find(pass_hashes, hash) == pass_hashes.end()) {
            pass_hashes.push_back(hash);
        }
    }
    if (capture_draw && (!PhaseCaptures() || draws - pass_first_draw < 4096)) {
        std::string shaders;
        const std::span<const u64> draw_hashes =
            PhaseCaptures() ? shader_hashes : std::span<const u64>{pass_hashes};
        for (const u64 hash : draw_hashes) {
            if (hash == 0) continue;
            shaders += fmt::format("{}\"{:016x}\"", shaders.empty() ? "" : ",", hash);
        }
        manifest << fmt::format(
                        R"({{"kind":"draw_state","pass":{},"draw":{},"guest_vs":"{:#x}","guest_ps":"{:#x}","shaders":[{}]}})",
                        pass_index, draws, guest_vs, guest_ps, shaders)
                 << '\n';
    }
    ++draws;
}

bool FrameDump::CapturingMetadata() const noexcept {
    return capturing && metadata_records < 8 && draws != 0 &&
           pass_index == DrawCapturePass() && pass_index >= FirstCapturePass();
}

void FrameDump::NoteDrawMetadata(std::string_view fields) {
    if (!CapturingMetadata()) return;
    ++metadata_records;
    manifest << fmt::format(R"({{"kind":"draw_metadata","pass":{},"draw":{},{} }})",
                            pass_index, draws - 1, fields) << '\n';
}

bool FrameDump::CapturingDrawSubmissions() const noexcept {
    if (!capturing) return false;
    static const bool metadata_enabled = ReadPassSetting("EDEN_NX2_FRAME_DUMP_DRAW_METADATA", 0) == 1;
    return metadata_enabled && draw_submission_records < 4096 &&
           pass_index == DrawCapturePass() && pass_index >= FirstCapturePass();
}

void FrameDump::NoteDrawSubmission(std::string_view fields, bool draw_noted) {
    if (!CapturingDrawSubmissions()) return;
    ++draw_submission_records;
    if (draw_noted && draws != 0) {
        manifest << fmt::format(
            R"({{"kind":"draw_submission","pass":{},"draw":{},{}}})",
            pass_index, draws - 1, fields) << '\n';
    } else {
        // A failure can occur before framebuffer lookup; this is the current pass cursor.
        manifest << fmt::format(
            R"({{"kind":"draw_submission","pass_cursor":{},"before_draw":{},{}}})",
            pass_index, draws, fields) << '\n';
    }
    if (draw_submission_records == 4096) {
        manifest << R"({"kind":"draw_submission_limit","limit":4096})" << '\n';
    }
}

bool FrameDump::CapturingMaterialInputs() const noexcept {
    return material_inputs_capture && capturing &&
        Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL && !ComputeCaptureStopEnabled();
}

std::array<u64, 2> FrameDump::CanopyInputPair(u32 variant) noexcept {
    // Exact shader pairs observed producing canopy or bright foliage in the captured OOT scene.
    constexpr std::array<std::array<u64, 2>, 5> pairs{{
        {0x4bcce0ad3a25fab8ULL, 0xf16638b51d8e93acULL},
        {0xca6282eca40f1327ULL, 0x19ad527101fabe38ULL},
        {0xa4f1f7c5b7efb230ULL, 0x9686fb4564c02e70ULL},
        {0x02bb192580e8205fULL, 0x69cec056ae29e908ULL},
        {0xeb82ba5dfc9276f5ULL, 0xf16638b51d8e93acULL},
    }};
    return variant < pairs.size() ? pairs[variant] : std::array<u64, 2>{};
}

std::array<u64, 2> FrameDump::CanopyInputShaders() const noexcept {
    return CanopyInputPair(canopy_input_variant);
}

bool FrameDump::CanopyInputIsMesh() const noexcept {
    return canopy_input_variant == 0 || canopy_input_variant == 4;
}

bool FrameDump::CapturingCanopyInputs() const noexcept {
    return canopy_inputs_capture && capturing &&
        Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL && !ComputeCaptureStopEnabled();
}

bool FrameDump::CapturingTaskInputs() const noexcept {
    static const bool task_enabled = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_TASK_INPUTS");
        return value && std::strcmp(value, "1") == 0;
    }();
    return (task_enabled || CapturingMaterialInputs() || CapturingCanopyInputs()) && capturing && BurstFrames() == 1 &&
           task_input_records < 16 &&
           draws != 0 && pass_index == DrawCapturePass() && pass_index >= FirstCapturePass();
}

bool FrameDump::NoteTaskInputs(u64 task_hash, u64 mesh_hash, std::string_view fields,
                               std::span<const TaskBufferInput> buffers) {
    const bool colored_foliage = task_hash == 0xf378ff9ca77352f0ULL &&
                                 mesh_hash == 0x87183e2c33d87f2aULL;
    const bool foliage = colored_foliage ||
        (task_hash == 0xf378ff9ca77352f0ULL && mesh_hash == 0x277d8ef0f4003ac4ULL);
    const u64 foliage_fragment = colored_foliage ? 0xc4a9495a97b2ae03ULL : 0x4b0ca12cb55286d7ULL;
    const bool material = CapturingMaterialInputs() && task_hash == 0x00e4eb4772d5076bULL &&
                          mesh_hash == 0x6cb075ad84a5c605ULL;
    const auto canopy_pair = CanopyInputShaders();
    const bool canopy = CapturingCanopyInputs() && task_hash == canopy_pair[0] &&
                        mesh_hash == canopy_pair[1];
    const bool selected_pair = material || canopy;
    if (!CapturingTaskInputs() || (CapturingMaterialInputs() && !material) ||
        (CapturingCanopyInputs() && !canopy) ||
        (!selected_pair && !foliage && (task_hash != 0x46cc6aaba42f700fULL || mesh_hash == 0)))
        return false;
    const size_t buffer_cap = foliage || selected_pair ? 32 : 16;
    ++task_input_records;
    const auto tags = canopy ? fmt::format(
        R"("pass":{},"draw":{},"{}":"{:016x}","fragment_shader_hash":"{:016x}","capture_scope":"{}","canopy_variant":{},"skipped_matches":{},"matching_draw_ordinal":{},"early_readback_may_change_timing":true)",
        pass_index, draws - 1, CanopyInputIsMesh() ? "mesh_hash" : "vertex_hash", task_hash,
        mesh_hash, CanopyInputIsMesh() ? "selected_mesh_fragment_pair" : "selected_vertex_fragment_pair",
        canopy_input_variant, material_input_skip, material_anchor_matches - 1) : material ? fmt::format(
        R"("pass":{},"draw":{},"vertex_hash":"{:016x}","fragment_shader_hash":"{:016x}","capture_scope":"selected_vertex_fragment_pair","skipped_matches":{},"early_readback_may_change_timing":true)",
        pass_index, draws - 1, task_hash, mesh_hash, material_input_skip) : fmt::format(
        R"("pass":{},"draw":{},"task_hash":"{:016x}","mesh_hash":"{:016x}","capture_scope":"first_matching_task_mesh_pair","early_readback_may_change_timing":true)",
        pass_index, draws - 1, task_hash, mesh_hash);
    if (task_inputs_recorded) {
        manifest << fmt::format(R"({{"kind":"task_input_group",{},"skipped":"first matching pair already captured"}})", tags) << '\n';
        return false;
    }
    task_inputs_recorded = true;
    manifest << fmt::format(R"({{"kind":"task_input_metadata",{}{}{}}})", tags,
                            fields.empty() ? "" : ",", fields) << '\n';
    const size_t before = task_buffer_bytes;
    std::array<std::pair<u32, u32>, 32> seen{};
    u32 seen_count{}, copied{};
    for (const auto& input : buffers.first(std::min(buffers.size(), buffer_cap))) {
        const auto& binding = input.binding;
        const bool task = input.stage == 0 && input.shader_hash == task_hash &&
                          (selected_pair || foliage ||
                           (binding.cbuf_index == 0 && binding.cbuf_offset == 0x670));
        const bool mesh = !selected_pair && input.stage == 2 && input.shader_hash == mesh_hash;
        const bool fragment = input.stage == 4 &&
            ((selected_pair && input.shader_hash == mesh_hash) ||
             (foliage && input.shader_hash == foliage_fragment &&
              (colored_foliage || binding.uniform_buffer)));
        if ((!task && !mesh && !fragment) ||
            ((foliage || selected_pair) && !binding.descriptor_checked)) {
            manifest << fmt::format(R"({{"kind":"task_buffer",{},"stage":{},"binding_index":{},"skipped":"unexpected descriptor"}})",
                                    tags, input.stage, binding.binding_index) << '\n';
            continue;
        }
        const auto key = std::pair{input.stage, binding.binding_index |
            (binding.uniform_buffer ? 0x80000000U : 0U)};
        if (std::find(seen.begin(), seen.begin() + seen_count, key) != seen.begin() + seen_count) {
            manifest << fmt::format(R"({{"kind":"task_buffer",{},"stage":{},"binding_index":{},"skipped":"duplicate descriptor"}})",
                                    tags, input.stage, binding.binding_index) << '\n';
            continue;
        }
        seen[seen_count++] = key;
        if (RecordTaskBuffer(input)) ++copied;
    }
    manifest << fmt::format(R"({{"kind":"task_input_group",{},"supplied_buffers":{},"considered_buffers":{},"unique_buffers":{},"copied_buffers":{},"supplied_buffers_complete":{},"captured_bytes":{},"task_byte_cap":{}}})",
                            tags, buffers.size(), std::min(buffers.size(), buffer_cap), seen_count, copied,
                            !buffers.empty() && copied == buffers.size() && buffers.size() <= buffer_cap, task_buffer_bytes - before,
                            128 * 1024 * 1024) << '\n';
    return copied != 0;
}

bool FrameDump::RecordTaskBuffer(const TaskBufferInput& task_input) {
    const auto& input = task_input.binding;
    const std::string name = fmt::format("{:04}_draw{}_stage{}_{}{}_cb{}_off{:x}",
        sequence++, draws - 1, task_input.stage, input.uniform_buffer ? "ubo" : "ssbo",
        input.binding_index, input.cbuf_index, input.cbuf_offset);
    const bool pointer_in_binding = input.pointer_valid && input.pointer >= input.gpu_addr &&
        input.pointer - input.gpu_addr < input.size;
    std::string tags = fmt::format(
        R"("kind":"task_buffer","pass":{},"draw":{},"stage":{},"shader_hash":"{:016x}","binding_index":{},"cbuf_index":{},"cbuf_offset":{},"pointer_table_offset":{},"aligned_gpu_addr":"{:#x}","host_buffer_offset":{},"host_buffer_capacity":{},"bound_size":{},"segment_count":{},"capture_source":"current_bound_gpu_buffer","early_readback_may_change_timing":true,"pointer_source":"guest_cpu_snapshot","pointer_valid":{},"pointer":"{:#x}","pointer_in_binding":{},"pointer_delta":{})",
        pass_index, draws - 1, task_input.stage, task_input.shader_hash, input.binding_index, input.cbuf_index, input.cbuf_offset,
        task_input.pointer_offset == ~0U ? "null" : std::to_string(task_input.pointer_offset),
        input.gpu_addr, input.offset, input.source_capacity, input.size, input.segment_count,
        input.pointer_valid, input.pointer, pointer_in_binding,
        pointer_in_binding ? std::to_string(input.pointer - input.gpu_addr) : "null");
    tags += fmt::format(
        R"(,"uniform_buffer":{},"descriptor_checked":{},"descriptor_matches":{},"descriptor_entry":{},"shader_binding":{},"descriptor_address_mode":{},"descriptor_buffer":"{}","descriptor_offset":{},"descriptor_range":{},"descriptor_address":"{:#x}")",
        input.uniform_buffer, input.descriptor_checked, input.descriptor_matches,
        input.descriptor_entry, input.shader_binding, input.descriptor_address_mode,
        fmt::ptr(input.descriptor_buffer), input.descriptor_offset, input.descriptor_range,
        input.descriptor_address);
    const auto skip = [&](const char* reason) {
        manifest << fmt::format(R"({{"name":"{}",{},"skipped":"{}"}})", name, tags, reason) << '\n';
        return false;
    };
    if (input.skip_reason) return skip(input.skip_reason);
    if ((input.descriptor_checked && !input.descriptor_matches) ||
        (input.uniform_buffer && !input.descriptor_checked))
        return skip("descriptor not validated");
    if (input.is_written) return skip("writable task input is unsupported");
    if (input.buffer == VK_NULL_HANDLE || input.size == 0 || input.segment_count >= 2)
        return skip("null, empty or multi-range binding");
    if (input.offset > input.source_capacity || input.size > input.source_capacity - input.offset)
        return skip("binding outside host buffer");
    if ((input.offset & 3) != 0 || (input.size & 3) != 0)
        return skip("unaligned buffer copy");
    constexpr size_t BufferByteCap = 128 * 1024 * 1024;
    if (input.size > BufferByteCap || task_buffer_bytes > BufferByteCap - input.size)
        return skip("128 MiB task buffer cap");
    vk::Buffer buffer = memory_allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = input.size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        }, MemoryUsage::Download);
    const VkBufferCopy copy{input.offset, 0, input.size};
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([source = input.buffer, destination = *buffer, copy](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, before);
        cmdbuf.CopyBuffer(source, destination, copy);
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT |
                             VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                               0, after);
    });
    pending_bytes += input.size;
    task_buffer_bytes += input.size;
    readbacks.push_back({std::move(buffer), {}, input.size, name, tags, true});
    return true;
}

bool FrameDump::DelayedSkyCaptureEnabled() noexcept {
    static const bool enabled = [] {
        const char* dir = std::getenv("EDEN_NX2_COMPUTE_CAPTURE_SKY_DELAYED");
        return dir && *dir;
    }();
    return enabled;
}

void FrameDump::PollDelayedSkyCapture() {
    if (!DelayedSkyCaptureEnabled() || DelayedSkyCaptureArmed().load(std::memory_order_acquire) ||
        delayed_sky_completed >= 2 || ++delayed_sky_poll_frames % 30 != 0) return;
    const char* output = std::getenv("EDEN_NX2_COMPUTE_CAPTURE_SKY_DELAYED");
    if (!output || std::strlen(output) > 4096)
        throw std::runtime_error("DIAGNOSTIC delayed sky output path is invalid");
    const std::filesystem::path root{output};
    if (!root.is_absolute())
        throw std::runtime_error("DIAGNOSTIC delayed sky requires an absolute output root");
    std::error_code error;
    const auto trigger = root / "capture.now";
    const bool exists = std::filesystem::exists(trigger, error);
    if (error) throw std::runtime_error("DIAGNOSTIC delayed sky trigger cannot be inspected");
    if (!exists) return;
    const char* immediate = std::getenv("EDEN_NX2_COMPUTE_CAPTURE_STOP");
    if ((immediate && *immediate) || enabled || capturing || !readbacks.empty())
        throw std::runtime_error("DIAGNOSTIC delayed sky cannot overlap another capture mode");
    if (!std::filesystem::is_regular_file(trigger, error) || error)
        throw std::runtime_error("DIAGNOSTIC delayed sky trigger is not a regular file");
    for (const char* leaf : {"sky-01", "sky-02"}) {
        if (std::filesystem::exists(root / leaf, error) || error)
            throw std::runtime_error("DIAGNOSTIC delayed sky output already exists");
    }
    if (!std::filesystem::remove(trigger, error) || error)
        throw std::runtime_error("DIAGNOSTIC delayed sky trigger could not be consumed");
    DelayedSkyCaptureArmed().store(true, std::memory_order_release);
    LOG_INFO(Render_Vulkan, "Delayed sky capture armed for two dispatches in {}", root.string());
}

bool FrameDump::CompleteDelayedSkyCapture() {
    if (!DelayedSkyCaptureEnabled() || !DelayedSkyCaptureArmed().load(std::memory_order_acquire) ||
        compute_capture_stop || capturing || !readbacks.empty() || !capture_stop_post_dispatch ||
        !compute_capture_stop_consumed || delayed_sky_completed >= 2)
        throw std::runtime_error("DIAGNOSTIC delayed sky completion state invalid");
    ++delayed_sky_completed;
    if (delayed_sky_completed == 2) return true;
    compute_capture_stop_consumed = false;
    done = false;
    return false;
}

bool FrameDump::ComputeCaptureStopEnabled() noexcept {
    if (DelayedSkyCaptureEnabled())
        return DelayedSkyCaptureArmed().load(std::memory_order_acquire);
    static const bool enabled = [] {
        const char* dir = std::getenv("EDEN_NX2_COMPUTE_CAPTURE_STOP");
        return dir && *dir;
    }();
    return enabled;
}

bool FrameDump::ComputeCaptureSkyEnabled() noexcept {
    if (DelayedSkyCaptureEnabled()) return ComputeCaptureStopEnabled();
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_COMPUTE_CAPTURE_SKY");
        return value && std::strcmp(value, "1") == 0;
    }();
    return ComputeCaptureStopEnabled() && enabled;
}

u64 FrameDump::ComputeCaptureStopNativeKey() noexcept {
    return ComputeCaptureSkyEnabled() ? 0x799e4ced3a1e4961ULL : 0x9694a462ae7e186cULL;
}

bool FrameDump::ReadyForSkyCaptureDispatch() const noexcept {
    if (!compute_capture_stop || !ComputeCaptureSkyEnabled() || capture_stop_post_dispatch ||
        capture_stop_uniform_mask != 0xb || capture_stop_sampled_count != 1 ||
        capture_stop_outputs.size() != 1) return false;
    const auto& output = capture_stop_outputs.front();
    return output.binding_index == 0 && output.cbuf_index == 0 && output.cbuf_offset == 32 &&
           output.size >= 112 && output.size <= 4096 && output.is_written &&
           output.descriptor_checked && output.descriptor_matches;
}

bool FrameDump::BeginComputeCaptureStop(u64 native_key) {
    if (!ComputeCaptureStopEnabled() || native_key != ComputeCaptureStopNativeKey()) return false;
    if (compute_capture_stop_consumed || capturing || !readbacks.empty()) {
        throw std::runtime_error("DIAGNOSTIC capture-stop cannot reuse an active capture; kernel not dispatched");
    }
    compute_capture_stop_consumed = true;
    const char* output = std::getenv(DelayedSkyCaptureEnabled()
        ? "EDEN_NX2_COMPUTE_CAPTURE_SKY_DELAYED" : "EDEN_NX2_COMPUTE_CAPTURE_STOP");
    if (!output || std::strlen(output) > 4096) {
        throw std::runtime_error("DIAGNOSTIC capture-stop output path is invalid; kernel not dispatched");
    }
    directory = output;
    if (DelayedSkyCaptureEnabled()) {
        if (delayed_sky_completed >= 2)
            throw std::runtime_error("DIAGNOSTIC delayed sky capture limit reached");
        directory /= fmt::format("sky-{:02}", delayed_sky_completed + 1);
    }
    if (!directory.is_absolute()) {
        throw std::runtime_error("DIAGNOSTIC capture-stop requires an absolute output path; kernel not dispatched");
    }
    std::error_code error;
    const bool created = std::filesystem::create_directories(directory, error);
    if (!created || error || !std::filesystem::is_empty(directory, error) || error) {
        throw std::runtime_error("DIAGNOSTIC capture-stop requires a writable unused output directory; kernel not dispatched");
    }
    manifest.clear();
    manifest.open(directory / "manifest.jsonl", std::ios::trunc);
    if (!manifest) {
        throw std::runtime_error("DIAGNOSTIC capture-stop cannot open manifest; kernel not dispatched");
    }
    compute_capture_stop = capturing = true;
    capture_stop_uniform_count = capture_stop_uniform_mask = capture_stop_sampled_count = 0;
    capture_stop_post_dispatch = false;
    capture_stop_outputs.clear();
    pending_bytes = written_bytes = sampled_bytes = producer_bytes = producer_buffer_bytes = 0;
    sequence = images = metadata_records = compute_metadata_records = 0;
    producer_inputs_recorded = false;
    manifest << fmt::format(
        R"({{"kind":"compute_capture_stop_begin","native_key":"{:016x}","dispatch_will_be_skipped":{},"robust_buffer_access":{},"ssbo_copy_limit":8,"ssbo_byte_limit":67108864,"cbuf_source":"guest_cpu_snapshot_not_bound_ubo_bytes"}})",
        native_key, !ComputeCaptureSkyEnabled(), device.IsRobustBufferAccessEnabled()) << '\n';
    if (DelayedSkyCaptureEnabled()) {
        manifest << fmt::format(
            R"({{"kind":"delayed_sky_capture","index":{},"dispatch_limit":2,"trigger_consumed":true,"guest_present":{}}})",
            delayed_sky_completed + 1, Core::g_nx2_present_count.load(std::memory_order_relaxed)) << '\n';
    }
    manifest.flush();
    return true;
}

void FrameDump::NoteCaptureStopMetadata(std::string_view fields) {
    if (!compute_capture_stop && !CapturingLightingInputs() && !CapturingShInputs() &&
        !CapturingParticleInputs()) return;
    const char* kind = compute_capture_stop ? "compute_capture_stop_metadata" :
                       CapturingShInputs() ? "sh_metadata" :
                       CapturingParticleInputs() ? "particle_metadata" : "lighting_metadata";
    constexpr size_t MetadataByteCap = size_t{2} * 1024 * 1024;
    if (fields.size() > MetadataByteCap) {
        manifest << fmt::format(R"({{"kind":"{}","skipped":"metadata cap"}})", kind) << '\n';
    } else {
        manifest << "{\"kind\":\"" << kind << "\"," << fields << "}\n";
    }
    manifest.flush();
}

void FrameDump::NoteCaptureStopUniformBuffer(ComputeBufferInput input) {
    const bool particle = CapturingParticleInputs();
    if (!compute_capture_stop && !CapturingLightingInputs() && !CapturingShInputs() && !particle) return;
    input.uniform_buffer = true;
    if (capture_stop_uniform_count >= (particle ? 8U : 4U)) {
        input.skip_reason = particle ? "eight uniform buffer copy limit" : "four uniform buffer copy limit";
    } else {
        ++capture_stop_uniform_count;
    }
    if (input.descriptor_range > 65536 || input.size > 65536)
        input.skip_reason = "64 KiB uniform buffer byte cap";
    if (RecordProducerBuffer(ProducerNativeKey(), input) && input.cbuf_index < 32)
        capture_stop_uniform_mask |= u32{1} << input.cbuf_index;
}

void FrameDump::FinishComputeCaptureStop(bool kernel_dispatched) {
    if (!compute_capture_stop) {
        throw std::runtime_error("DIAGNOSTIC capture-stop is not active; kernel not dispatched");
    }
    if (kernel_dispatched) {
        if (!ReadyForSkyCaptureDispatch())
            throw std::runtime_error("DIAGNOSTIC sky post-dispatch capture state invalid");
        capture_stop_post_dispatch = true;
        for (const auto& output : capture_stop_outputs) {
            if (!RecordProducerBuffer(ComputeCaptureStopNativeKey(), output))
                throw std::runtime_error("DIAGNOSTIC sky output readback could not be recorded");
        }
    }
    manifest << fmt::format(
        R"({{"kind":"compute_capture_stop_drain_begin","kernel_dispatched":{}}})",
        kernel_dispatched) << '\n';
    manifest.flush();
    // Configuration recorded uploads/bindings even if every readback was rejected.
    if (readbacks.empty()) scheduler.Finish();
    else Drain();
    manifest << fmt::format(
        R"({{"kind":"compute_capture_stop_complete","kernel_dispatched":{},"producer_inputs_recorded":{},"bytes":{},"producer_buffer_bytes":{}}})",
        kernel_dispatched, producer_inputs_recorded, written_bytes, producer_buffer_bytes) << '\n';
    manifest.flush();
    manifest.close();
    const bool saved = static_cast<bool>(manifest);
    compute_capture_stop = capturing = false;
    done = true;
    if (!saved) throw std::runtime_error("DIAGNOSTIC capture-stop manifest write failed");
}

u64 FrameDump::ProducerNativeKey() noexcept {
    if (ComputeCaptureStopEnabled()) return ComputeCaptureStopNativeKey();
    if (const u64 live_key = LiveCaptureProducerKey().load(std::memory_order_acquire))
        return live_key;
    static const u64 key = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_PRODUCER_KEY");
        if (!value) return u64{0xe883414e8c9be33bULL};
        std::string_view text{value};
        if (text.starts_with("0x") || text.starts_with("0X")) text.remove_prefix(2);
        if (text.size() != 16) return u64{};
        u64 parsed{};
        const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed, 16);
        return result.ec == std::errc{} && result.ptr == text.data() + text.size() ? parsed : u64{};
    }();
    return key;
}

bool FrameDump::CapturingProducerInputs() const noexcept {
    if (compute_capture_stop) return true;
    static const bool producer_enabled = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_PRODUCER");
        return value && std::strcmp(value, "1") == 0;
    }();
    return (producer_enabled || CapturingFogInputs()) && ProducerNativeKey() != 0 &&
           capturing && BurstFrames() == 1 && compute_metadata_records < 256;
}

bool FrameDump::IsParticleProducerPair(u64 native_key, u64 canonical_key) noexcept {
    return (native_key == 0x8dc54d1adc12150fULL && canonical_key == 0xe78efceda31f97a8ULL) ||
           (native_key == 0xf35f519f99b98203ULL && canonical_key == 0xf98012dc9a4465ffULL) ||
           (native_key == 0x4d9a7a9cfd503656ULL && canonical_key == 0x7f0d176391897d0eULL);
}

bool FrameDump::CapturingParticleInputs() const noexcept {
    const u64 key = ProducerNativeKey();
    return Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL &&
        !ComputeCaptureStopEnabled() && CapturingProducerInputs() &&
        (key == 0x8dc54d1adc12150fULL || key == 0xf35f519f99b98203ULL ||
         key == 0x4d9a7a9cfd503656ULL);
}

bool FrameDump::SelectProducerInputDispatch(u64 native_key) {
    if (!CapturingProducerInputs() || native_key != ProducerNativeKey() || producer_inputs_recorded)
        return false;
    if (!CapturingParticleInputs()) return true;
    const u32 ordinal = producer_matching_dispatches++;
    if (producer_input_skip > 63 || ordinal != producer_input_skip) return false;
    producer_selected_ordinal = ordinal;
    return true;
}

bool FrameDump::NoteProducerInputs(u64 native_key, std::span<const ComputeInput> inputs,
                                   std::span<const ComputeBufferInput> buffers) {
    if (!CapturingProducerInputs() || native_key != ProducerNativeKey()) return false;
    const auto tags = fmt::format(
        R"("dispatch":{},"native_key":"{:016x}","capture_scope":"selected_producer_dispatch_this_frame","requested_skip":{},"matching_dispatch_ordinal":{},"budget":"producer")",
        compute_metadata_records, native_key, CapturingParticleInputs() ? producer_input_skip : 0,
        CapturingParticleInputs() ? producer_selected_ordinal : 0);
    if (producer_inputs_recorded) {
        manifest << fmt::format(R"({{"kind":"producer_input_group",{},"skipped":"first matching dispatch already captured"}})", tags) << '\n';
        return false;
    }
    producer_inputs_recorded = true;
    const size_t bytes_before = producer_bytes;
    u32 copied_inputs{}, copied_subresources{};
    for (const auto& input : inputs.first(std::min(inputs.size(), size_t{16}))) {
        const auto input_tags = tags + fmt::format(
            R"(,"descriptor_class":"sampled_texture","texture_index":{},"tic":{},"cbuf_index":{},"cbuf_offset":{})",
            input.texture_index, input.tic, input.cbuf_index, input.cbuf_offset);
        const auto skip = [&](std::string_view reason) {
            manifest << fmt::format(R"({{"kind":"producer_input",{},"skipped":"{}"}})", input_tags, reason) << '\n';
        };
        const Image* const image = input.view ? input.view->BackingImage() : nullptr;
        if (input.texture_index >= 16 || !image ||
            (input.view->type != VideoCommon::ImageViewType::e2D &&
             input.view->type != VideoCommon::ImageViewType::e2DArray &&
             input.view->type != VideoCommon::ImageViewType::e3D) ||
            (image->info.type != VideoCommon::ImageType::e2D &&
             image->info.type != VideoCommon::ImageType::e3D) ||
            ((input.view->type == VideoCommon::ImageViewType::e3D) !=
             (image->info.type == VideoCommon::ImageType::e3D)) ||
            image->info.is_sparse || image->info.num_samples != 1 || image->IsRescaled() ||
            image->host_level_offset != 0 ||
            True(image->flags & (VideoCommon::ImageFlagBits::Unbacked |
                                 VideoCommon::ImageFlagBits::SparseLayers |
                                 VideoCommon::ImageFlagBits::SparseTiles |
                                 VideoCommon::ImageFlagBits::Converted)) ||
            !(image->UsageFlags() & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            skip("unsupported, converted, sparse or non-transferable image");
            continue;
        }
        const auto& range = input.view->range;
        if (range.base.level < 0 || range.base.level >= 32 || range.extent.levels <= 0 ||
            s64{range.base.level} + range.extent.levels > image->info.resources.levels ||
            range.base.layer < 0 || range.extent.layers <= 0 || range.extent.layers > 16 ||
            s64{range.base.layer} + range.extent.layers > image->info.resources.layers) {
            skip("invalid mip/layer range or more than16 layers");
            continue;
        }
        const auto target = MakeTarget(*input.view, input.texture_index, true, true);
        if (!target || target->width > 4096 || target->height > 4096 || target->depth > 128) {
            skip("unsupported format or oversized view");
            continue;
        }
        u32 copied_layers{};
        // This bounded diagnostic captures only the base mip; other mips are not included.
        // Preserve all array layers at that base level;3D depth is copied in one subresource.
        for (s32 layer = 0; layer < range.extent.layers; ++layer) {
            Target subresource = *target;
            subresource.layer += layer;
            const auto subresource_tags = input_tags + fmt::format(
                R"(,"view_base_level":{},"view_levels":{},"view_base_layer":{},"view_layers":{},"captured_levels":1,"captured_layer":{},"early_readback_may_change_timing":true)",
                range.base.level, range.extent.levels, range.base.layer,
                range.extent.layers, subresource.layer);
            if (Record(subresource,
                       fmt::format("{:04}_compute{:03}_e883_input{}_layer{}", sequence++,
                                   compute_metadata_records, input.texture_index, subresource.layer),
                       fmt::format(R"("kind":"producer_input",{})", subresource_tags), false, true)) {
                ++copied_layers;
                ++copied_subresources;
            }
        }
        if (copied_layers) ++copied_inputs;
    }
    if (compute_capture_stop && ComputeCaptureSkyEnabled())
        capture_stop_sampled_count = copied_inputs == 1 && copied_subresources == 1 ? 1 : 0;
    u32 copied_buffers{};
    const size_t buffer_limit = native_key == 0xe883414e8c9be33bULL ? 2 : 64;
    const size_t copy_limit = native_key == 0xe883414e8c9be33bULL ? 2 :
        (CapturingShInputs() && native_key == 0x203053e20b8c989aULL) ||
        CapturingParticleInputs() ? 10 : 8;
    u32 requested_copies{};
    for (const auto& buffer : buffers.first(std::min(buffers.size(), buffer_limit))) {
        auto selected = buffer;
        selected.capture_requested = buffer.capture_requested && requested_copies < copy_limit;
        if (selected.capture_requested) ++requested_copies;
        if (RecordProducerBuffer(native_key, selected)) {
            ++copied_buffers;
            if (compute_capture_stop && ComputeCaptureSkyEnabled() && selected.is_written)
                capture_stop_outputs.push_back(selected);
        }
    }
    manifest << fmt::format(R"({{"kind":"producer_input_group",{},"considered_inputs":{},"captured_inputs":{},"captured_subresources":{},"considered_buffers":{},"buffer_metadata_limit":{},"buffer_copy_limit":{},"requested_buffer_copies":{},"captured_buffers":{},"captured_bytes":{},"producer_byte_cap":{}}})",
                            tags, std::min(inputs.size(), size_t{16}), copied_inputs,
                            copied_subresources, std::min(buffers.size(), buffer_limit), buffer_limit, copy_limit, requested_copies, copied_buffers,
                            producer_bytes - bytes_before,
                            ProducerSizeCap()) << '\n';
    return true;
}

bool FrameDump::RecordProducerBuffer(u64 native_key, const ComputeBufferInput& input) {
    if (!CapturingProducerInputs() || native_key != ProducerNativeKey()) return false;
    const std::string name = fmt::format("{:04}_compute{}_{}{}_cb{}_off{:x}",
        sequence++, compute_metadata_records, input.uniform_buffer ? "uniform" : "producer_ssbo",
        input.binding_index, input.cbuf_index, input.cbuf_offset);
    const bool pointer_in_binding = input.pointer_valid && input.pointer >= input.gpu_addr &&
        input.pointer - input.gpu_addr < input.size;
    const std::string tags = fmt::format(
        R"("kind":"{}","dispatch":{},"native_key":"{:016x}","binding_index":{},"cbuf_index":{},"cbuf_offset":{},"aligned_gpu_addr":"{:#x}","aligned_device_addr":"{:#x}","host_buffer_offset":{},"host_buffer_capacity":{},"bound_size":{},"segment_count":{},"capture_source":"current_bound_gpu_buffer","capture_phase":"{}","capture_requested":{},"binding_available":{},"pointer_table_offset":{},"pointer_is_table_root":{},"total_bindings":{},"binding_selection_truncated":{},"is_written":{},"access":"{}","early_readback_may_change_timing":true,"pointer_source":"guest_cpu_snapshot","pointer_valid":{},"pointer":"{:#x}","pointer_in_binding":{},"pointer_delta":{})",
        input.uniform_buffer ? "uniform_buffer_input" : "producer_buffer",
        compute_metadata_records, native_key, input.binding_index, input.cbuf_index, input.cbuf_offset,
        input.gpu_addr, input.device_addr, input.offset, input.source_capacity, input.size, input.segment_count,
        capture_stop_post_dispatch ? "post_dispatch" : "pre_dispatch",
        input.capture_requested, input.buffer != VK_NULL_HANDLE && input.size != 0 && input.skip_reason == nullptr,
        input.pointer_table_offset == ~0U ? "null" : std::to_string(input.pointer_table_offset),
        input.pointer_table_offset != ~0U, input.total_bindings, input.binding_selection_truncated,
        input.is_written, input.is_written ? "read_write" : "read_only", input.pointer_valid, input.pointer, pointer_in_binding,
        pointer_in_binding ? std::to_string(input.pointer - input.gpu_addr) : "null");
    const auto skip = [&](const char* reason) {
        manifest << fmt::format(R"({{"name":"{}",{},"skipped":"{}"}})", name, tags, reason) << '\n';
        return false;
    };
    if (input.descriptor_checked) {
        manifest << fmt::format(
            R"({{"kind":"{}","binding_index":{},"descriptor_entry":{},"shader_binding":{},"snapshot_buffer":"{}","descriptor_address_mode":{},"descriptor_buffer":"{}","descriptor_offset":{},"descriptor_range":{},"descriptor_address":"{:#x}","matches_snapshot":{}}})",
            input.uniform_buffer ? "uniform_descriptor_validation" : "producer_descriptor",
            input.binding_index, input.descriptor_entry, input.shader_binding, fmt::ptr(input.buffer),
            input.descriptor_address_mode, fmt::ptr(input.descriptor_buffer), input.descriptor_offset,
            input.descriptor_range, input.descriptor_address, input.descriptor_matches) << '\n';
    }
    if (!input.capture_requested) return skip("metadata only");
    if (input.descriptor_checked && !input.descriptor_matches)
        return skip("descriptor does not match validated binding snapshot");
    if (input.skip_reason) return skip(input.skip_reason);
    if (input.buffer == VK_NULL_HANDLE || input.size == 0 || input.segment_count >= 2)
        return skip("null, empty or multi-range binding");
    if (input.offset > input.source_capacity || input.size > input.source_capacity - input.offset)
        return skip("binding outside host buffer");
    if ((input.offset & 3) != 0 || (input.size & 3) != 0)
        return skip("unaligned buffer copy");
    const bool dynamic_range_capture = NativeDomainCorrectionEnabled() &&
        ((CapturingLightingInputs() && native_key == 0xd9b1657f3a0b4bc2ULL) ||
         (CapturingShInputs() && native_key == 0x203053e20b8c989aULL));
    const bool expanded_range_capture =
        (CapturingLightingInputs() && LightingRangeMinimum(native_key,
            0x75d4369f3b1f5868ULL, 0, 96, 24, false)) ||
        (CapturingShInputs() && LightingRangeMinimum(native_key,
            0xc73574a9766531cbULL, 0, 96, 24, false));
    const size_t BufferByteCap = dynamic_range_capture ? size_t{320} << 20 :
        expanded_range_capture ? size_t{256} << 20 :
        (native_key == 0xe883414e8c9be33bULL ? size_t{16} : size_t{64}) << 20;
    if (!input.uniform_buffer &&
        (input.size > BufferByteCap || producer_buffer_bytes > BufferByteCap - input.size))
        return skip("producer buffer byte cap");
    const size_t producer_limit = ProducerSizeCap() -
        (CapturingFogInputs() && !compute_inputs_recorded ? FOG_COMPUTE_RESERVE : 0) -
        (CapturingLightingInputs() && !lighting_outputs_finished ? LIGHTING_OUTPUT_RESERVE : 0);
    if (input.size > producer_limit || producer_bytes > producer_limit - input.size)
        return skip("producer size cap, including reserved fog compute inputs");
    vk::Buffer buffer = memory_allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = input.size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        }, MemoryUsage::Download);
    const VkBufferCopy copy{input.offset, 0, input.size};
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([source = input.buffer, destination = *buffer, copy](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, before);
        cmdbuf.CopyBuffer(source, destination, copy);
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT |
                             VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                               0, after);
    });
    pending_bytes += input.size;
    producer_bytes += input.size;
    if (!input.uniform_buffer) producer_buffer_bytes += input.size;
    readbacks.push_back({std::move(buffer), {}, input.size, name, tags, true});
    return true;
}




void FrameDump::NoteLightingOutput(const ComputeInput& output, bool descriptor_matches) {
    if (!CapturingLightingInputs() || producer_inputs_recorded || lighting_output_targets.size() >= 2)
        return;
    const auto tags = fmt::format(R"("kind":"lighting_output","native_key":"d9b1657f3a0b4bc2","dispatch":{},"cbuf_index":{},"cbuf_offset":{},"descriptor_matches":{},"phase":"before_dispatch")",
                                  compute_metadata_records, output.cbuf_index, output.cbuf_offset, descriptor_matches);
    const auto skip = [&](std::string_view reason) {
        manifest << fmt::format(R"({{{},"skipped":"{}"}})", tags, reason) << '\n';
    };
    if (!descriptor_matches || output.cbuf_index != 0 ||
        (output.cbuf_offset != 0x98 && output.cbuf_offset != 0xa0) || !output.view) {
        skip("unmatched output descriptor");
        return;
    }
    const auto* image = output.view->BackingImage();
    if (!image || output.view->type != VideoCommon::ImageViewType::e3D ||
        image->info.type != VideoCommon::ImageType::e3D || image->info.is_sparse ||
        image->host_level_offset != 0 || image->info.num_samples != 1 || image->IsRescaled() ||
        True(image->flags & (VideoCommon::ImageFlagBits::Unbacked | VideoCommon::ImageFlagBits::SparseLayers |
                            VideoCommon::ImageFlagBits::SparseTiles | VideoCommon::ImageFlagBits::Converted)) ||
        !(image->UsageFlags() & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        skip("unsupported output storage");
        return;
    }
    const auto target = MakeTarget(*output.view, output.texture_index, true, true);
    if (!target || target->width != 64 || target->height != 36 || target->depth != 96 ||
        target->format != PixelFormat::B10G11R11_FLOAT || target->level != 0 || target->layer != 0 ||
        target->msaa || target->rescaled || target->aspect != VK_IMAGE_ASPECT_COLOR_BIT ||
        std::ranges::any_of(lighting_output_targets, [&](const Target& prior) { return prior.slot == target->slot; })) {
        skip("unexpected or duplicate output subresource");
        return;
    }
    if (Record(*target, fmt::format("{:04}_d9_output{}_before", sequence++, target->slot), tags, false, true))
        lighting_output_targets.push_back(*target);
}

void FrameDump::NoteLightingOutputsAfter(u64 native_key) {
    if (!CapturingLightingInputs() || native_key != 0xd9b1657f3a0b4bc2ULL || lighting_outputs_finished) return;
    lighting_outputs_finished = true;
    for (const auto& target : lighting_output_targets) {
        Record(target, fmt::format("{:04}_d9_output{}_after", sequence++, target.slot),
               fmt::format(R"("kind":"lighting_output","native_key":"d9b1657f3a0b4bc2","dispatch":{},"slot":{},"phase":"after_dispatch_record","whole_volume_rewritten":false)",
                           compute_metadata_records, target.slot), false, true);
    }
    lighting_output_targets.clear();
}

void FrameDump::NoteWrittenImage(u64 shader_hash, ImageView& view, u32 slot) {
    static const u64 wanted = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_WRITTEN_SHADER");
        return value ? std::strtoull(value, nullptr, 16) : 0x780c0536b89fc89eULL;
    }();
    if (!capturing || shader_hash != wanted) return;
    if (const auto target = MakeTarget(view, slot, true, true)) {
        written_targets.push_back(*target);
        written_shader = shader_hash;
        return;
    }
    manifest << fmt::format(R"({{"kind":"written_image","shader":"{:016x}","slot":{},"skipped":"no capture target for this view"}})",
                            shader_hash, slot) << '\n';
}

void FrameDump::NoteWrittenImagesAfter() {
    if (written_targets.empty()) return;
    for (const Target& target : written_targets) {
        Record(target,
               fmt::format("{:04}_written{}_{:016x}_img{}", sequence++, written_dispatches,
                           written_shader, target.slot),
               fmt::format(R"("kind":"written_image","shader":"{:016x}","dispatch":{},"slot":{})",
                           written_shader, written_dispatches, target.slot),
               false, true);
    }
    written_targets.clear();
    ++written_dispatches;
}

bool FrameDump::CapturingComputeInputs() const noexcept {
    static const bool inputs_enabled = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_COMPUTE_INPUTS");
        return value && std::strcmp(value, "1") == 0;
    }();
    return (inputs_enabled || CapturingFogInputs()) && capturing && BurstFrames() == 1 &&
           compute_metadata_records < 256;
}

bool FrameDump::NoteComputeInputs(u64 native_key, std::span<const ComputeInput> inputs) {
    if (!CapturingComputeInputs() || native_key != 0xec68e40eb783ff32ULL) return false;
    const auto tags = fmt::format(R"("dispatch":{},"native_key":"{:016x}","capture_scope":"first_ec68_dispatch_this_frame")",
                                  compute_metadata_records, native_key);
    if (compute_inputs_recorded) {
        manifest << fmt::format(R"({{"kind":"compute_input_group",{},"skipped":"first matching dispatch already captured"}})",
                                tags) << '\n';
        return false;
    }
    compute_inputs_recorded = true;
    constexpr size_t InputByteCap = size_t{128} * 72 * 96 * 10;
    size_t copied_bytes{};
    u32 copied_inputs{};
    bool recorded = false;
    for (const auto& input : inputs.first(std::min(inputs.size(), size_t{3}))) {
        const auto input_tags = tags + fmt::format(
            R"(,"descriptor_class":"sampled_texture","texture_index":{},"tic":{},"cbuf_index":{},"cbuf_offset":{})",
            input.texture_index, input.tic, input.cbuf_index, input.cbuf_offset);
        const auto skip = [&](std::string_view reason) {
            manifest << fmt::format(R"({{"kind":"compute_input",{},"skipped":"{}"}})",
                                    input_tags, reason) << '\n';
        };
        if (input.texture_index >= 3 || !input.view ||
            input.view->type != VideoCommon::ImageViewType::e3D ||
            !input.view->BackingImage() ||
            input.view->BackingImage()->info.type != VideoCommon::ImageType::e3D) {
            skip("not one of the first three sampled 3D textures");
            continue;
        }
        const Image* const image = input.view->BackingImage();
        if (image->info.is_sparse ||
            True(image->flags & (VideoCommon::ImageFlagBits::Unbacked |
                                 VideoCommon::ImageFlagBits::SparseLayers |
                                 VideoCommon::ImageFlagBits::SparseTiles)) ||
            !(image->UsageFlags() & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            skip("unbacked/sparse or non-transferable volume");
            continue;
        }
        const auto target = MakeTarget(*input.view, input.texture_index, true);
        if (!target || target->msaa || target->rescaled ||
            target->aspect != VK_IMAGE_ASPECT_COLOR_BIT || target->width > 128 ||
            target->height > 72 || target->depth > 96 ||
            (target->format != PixelFormat::B10G11R11_FLOAT &&
             target->format != PixelFormat::R16_FLOAT)) {
            skip("unsupported volume view");
            continue;
        }
        const auto bytes = CaptureByteSize(target->width, target->height, target->depth,
                                           VideoCore::Surface::BytesPerBlock(target->format));
        if (!bytes || *bytes > InputByteCap - copied_bytes) {
            skip("compute inputs byte cap");
            continue;
        }
        if (Record(*target,
                   fmt::format("{:04}_compute{:03}_ec68_input{}", sequence++,
                               compute_metadata_records, input.texture_index),
                   fmt::format(R"("kind":"compute_input",{},"budget":"{}")", input_tags,
                               CapturingFogInputs() ? "producer" : "sampled"),
                   !CapturingFogInputs(), CapturingFogInputs())) {
            copied_bytes += *bytes;
            ++copied_inputs;
            recorded = true;
        }
    }
    manifest << fmt::format(R"({{"kind":"compute_input_group",{},"considered_inputs":{},"captured_inputs":{},"captured_bytes":{},"early_readback_may_change_timing":true}})",
                            tags, std::min(inputs.size(), size_t{3}), copied_inputs, copied_bytes) << '\n';
    return recorded;
}

bool FrameDump::CapturingComputeMetadata() const noexcept {
    static const bool compute_trace_enabled = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_COMPUTE");
        return value && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    static const bool compute_from_start = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_DUMP_COMPUTE_FROM_START");
        return value && std::strcmp(value, "1") == 0;
    }();
    return capturing && BurstFrames() == 1 && (compute_trace_enabled || CapturingFogInputs()) &&
           compute_metadata_records < 256 &&
           (compute_from_start || CapturingFogInputs() ||
            (DrawCapturePass() != ~0U && pass_index >= DrawCapturePass()));
}

void FrameDump::NoteComputeMetadata(std::string_view fields) {
    if (!CapturingComputeMetadata()) return;
    manifest << fmt::format(
                    R"({{"kind":"native_compute","dispatch":{},"pass_cursor":{},"before_draw":{},{} }})",
                    compute_metadata_records++, pass_index, draws, fields) << '\n';
}

void FrameDump::NoteOperation(const char* operation) {
    if (pass_index == DrawCapturePass() && pass_index >= FirstCapturePass()) {
        manifest << fmt::format(
                        R"({{"kind":"operation_begin","operation":"{}","previous_pass":{},"before_draw":{}}})",
                        operation, pass_index, draws)
                 << '\n';
    }
}

bool FrameDump::CaptureLastDraw() {
    if (draws <= pass_first_draw || pass_hashes.empty()) return false;
    std::string shaders;
    for (const u64 hash : pass_hashes) {
        shaders += fmt::format("{}\"{:016x}\"", shaders.empty() ? "" : ",", hash);
    }
    bool recorded = false;
    for (const Target& target : pass_targets) {
        recorded |= Record(
            target,
            fmt::format("{:04}_pass{:03}_after_draw{:04}_rt{}", sequence++, pass_index, draws - 1,
                        target.slot),
            fmt::format(R"("kind":"draw","pass":{},"draw":{},"slot":{},"shaders":[{}])",
                        pass_index, draws - 1, target.slot, shaders));
    }
    return recorded;
}

bool FrameDump::EndPass() {
    if (!pass_framebuffer) {
        return false;
    }
    pass_framebuffer = nullptr;
    const u32 first_pass = FirstCapturePass();
    // A temporal lighting run needs the selected fog pass and the final tone pass only.
    // NoteSampled also preserves the color target immediately before the selected draw.
    constexpr u64 OotToneShader = 0xbded5ce917183214ULL;
    if (BurstFrames() > 1 && pass_index != DrawCapturePass() &&
        std::ranges::find(pass_hashes, OotToneShader) == pass_hashes.end()) {
        ++pass_index;
        return false;
    }
    if (!PhaseCaptures() && pass_index == DrawCapturePass() && pass_index >= first_pass) {
        const bool recorded = CaptureLastDraw();
        pass_hashes.clear();
        ++pass_index;
        return recorded;
    }
    std::string shaders;
    for (const u64 hash : pass_hashes) {
        shaders += fmt::format("{}\"{:016x}\"", shaders.empty() ? "" : ",", hash);
    }
    bool recorded = false;
    for (const Target& target : pass_targets) {
        if (pass_index < first_pass) {
            manifest << fmt::format(
                            R"({{"name":"{:04}_pass{:03}_rt{}","kind":"pass","pass":{},"first_draw":{},"draw_count":{},"slot":{},"shaders":[{}],"format":"{}","width":{},"height":{},"gpu_addr":"{:#x}","skipped":"before first pass"}})",
                            sequence++, pass_index, target.slot, pass_index, pass_first_draw,
                            draws - pass_first_draw, target.slot, shaders, target.format,
                            target.width, target.height, target.gpu_addr)
                     << '\n';
            continue;
        }
        recorded |= Record(
            target,
            fmt::format("{:04}_pass{:03}_d{}-{}_rt{}", sequence++, pass_index, pass_first_draw,
                        draws, target.slot),
            fmt::format(R"("kind":"pass","pass":{},"first_draw":{},"draw_count":{},"slot":{},"shaders":[{}])",
                        pass_index, pass_first_draw, draws - pass_first_draw, target.slot,
                        shaders));
    }
    ++pass_index;
    return recorded;
}

void FrameDump::CapturePresent(ImageView& view) {
    if (BurstFrames() > 1) return;
    if (const auto target = MakeTarget(view, 0)) {
        Record(*target, fmt::format("{:04}_present", sequence++),
               fmt::format(R"("kind":"present","after_draws":{})", draws));
    }
}

bool FrameDump::Record(const Target& target, std::string name, std::string tags, bool sampled,
                       bool producer) {
    if (sampled && producer) return false;
    name += target.depth > 1
                ? fmt::format("_{}_{}x{}x{}_{:x}", target.format, target.width, target.height,
                              target.depth, target.gpu_addr)
                : fmt::format("_{}_{}x{}_{:x}", target.format, target.width, target.height,
                              target.gpu_addr);
    if (target.msaa) {
        manifest << fmt::format(R"({{"name":"{}",{},"skipped":"msaa"}})", name, tags) << '\n';
        return false;
    }
    const auto target_size = CaptureStorageByteSize(target.width, target.height, target.depth,
        VideoCore::Surface::BytesPerBlock(target.format), target.block_width, target.block_height);
    if (!target_size) {
        manifest << fmt::format(R"({{"name":"{}",{},"skipped":"invalid capture dimensions"}})",
                                name, tags) << '\n';
        return false;
    }
    const size_t size = *target_size;
    const size_t budget = tags.find("\"kind\":\"present\"") != std::string::npos
                              ? SIZE_CAP : SIZE_CAP - PRESENT_RESERVE;
    const size_t used = producer ? producer_bytes : sampled ? sampled_bytes
        : written_bytes + pending_bytes - sampled_bytes - producer_bytes - task_buffer_bytes;
    const size_t producer_limit = ProducerSizeCap() -
        (CapturingFogInputs() && !compute_inputs_recorded ? FOG_COMPUTE_RESERVE : 0) -
        (CapturingLightingInputs() && !lighting_outputs_finished ? LIGHTING_OUTPUT_RESERVE : 0);
    const size_t limit = producer ? producer_limit : sampled ? SAMPLED_SIZE_CAP : budget;
    if (size > limit || used > limit - size) {
        truncated = true;
        manifest << fmt::format(R"({{"name":"{}",{},"skipped":"{}"}})", name, tags,
                                producer ? "producer size cap, including reserved fog compute inputs" :
                                    sampled ? "32 MiB sampled size cap" : "capture size cap") << '\n';
        return false;
    }
    vk::Buffer buffer = memory_allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        },
        MemoryUsage::Download);
    const VkBufferImageCopy copy{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource{
            .aspectMask = target.aspect,
            .mipLevel = static_cast<u32>(target.level),
            .baseArrayLayer = static_cast<u32>(target.layer),
            .layerCount = 1,
        },
        .imageOffset{0, 0, 0},
        .imageExtent{target.width, target.height, target.depth},
    };
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = target.image, destination = *buffer, copy](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, before);
        cmdbuf.CopyImageToBuffer(image, VK_IMAGE_LAYOUT_GENERAL, destination, copy);
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT |
                             VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0,
                               after);
    });
    pending_bytes += size;
    if (sampled) sampled_bytes += size;
    if (producer) producer_bytes += size;
    readbacks.push_back({std::move(buffer), target, size, std::move(name), std::move(tags)});
    return true;
}

void FrameDump::DrainIfFull() {
    if (pending_bytes >= STAGING_BUDGET) {
        Drain();
    }
}

void FrameDump::Drain() {
    if (readbacks.empty()) {
        return;
    }
    scheduler.Finish();
    for (Readback& readback : readbacks) {
        readback.buffer.Invalidate();
        Write(readback, readback.buffer.Mapped());
    }
    readbacks.clear();
    pending_bytes = 0;
}

void FrameDump::Write(const Readback& readback, std::span<const u8> mapped) {
    if (readback.raw_buffer) {
        WriteFile(readback.name + ".raw", mapped.first(readback.size_bytes));
        manifest << fmt::format(
            R"({{"name":"{}",{},"layout":"linear_buffer_bytes","raw":"{}.raw","bytes":{}}})",
            readback.name, readback.tags, readback.name, readback.size_bytes) << '\n';
        return;
    }
    const Target& target = readback.target;
    const size_t size = readback.size_bytes;
    const size_t block_rows = (u64{target.height} + target.block_height - 1) / target.block_height;
    const bool compressed = target.block_width > 1;
    const std::string storage_tags = compressed ? fmt::format(
        R"(,"block_width":{},"block_height":{},"bytes_per_block":{})",
        target.block_width, target.block_height,
        VideoCore::Surface::BytesPerBlock(target.format)) : "";
    WriteFile(readback.name + ".raw", mapped.first(size));
    ++images;
    manifest << fmt::format(
                    R"({{"name":"{}",{},"format":"{}","width":{},"height":{},"depth":{},"row_bytes":{},"slice_bytes":{},"layout":"{}","level":{},"layer":{},"gpu_addr":"{:#x}","rescaled":{},"aspect":{},"raw":"{}.raw","bytes":{}{}}})",
                    readback.name, readback.tags, target.format, target.width, target.height,
                    target.depth, size / target.depth / block_rows, size / target.depth,
                    compressed ? "tightly_packed_xyz_blocks" : "tightly_packed_xyz",
                    target.level, target.layer, target.gpu_addr, target.rescaled, target.aspect, readback.name,
                    size, storage_tags)
             << '\n';
}

void FrameDump::WriteFile(const std::string& name, std::span<const u8> data) {
    std::ofstream file{directory / name, std::ios::binary};
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (compute_capture_stop) {
        file.close();
        if (!file) {
            throw std::runtime_error("DIAGNOSTIC capture-stop payload write failed; kernel not dispatched");
        }
    }
    written_bytes += data.size();
}

void FrameDump::TickFrame() {
    if (DelayedSkyCaptureEnabled()) {
        PollDelayedSkyCapture();
        return;
    }
    if (!enabled || done) {
        return;
    }
    ++frame;
    bool continue_burst = false;
    if (capturing) {
        EndPass();
        Drain();
        manifest << fmt::format(
                        R"({{"kind":"summary","images":{},"passes":{},"draws":{},"bytes":{},"sampled_bytes":{},"truncated":{}}})",
                        images, pass_index, draws, written_bytes, sampled_bytes, truncated)
                 << '\n';
        capturing = false;
        material_inputs_capture = false;
        canopy_inputs_capture = false;
        fog_inputs_capture = false;
        LiveCaptureProducerKey().store(0, std::memory_order_release);
        continue_burst = burst_remaining > 1;
        if (continue_burst) {
            --burst_remaining;
            ++burst_index;
        } else {
            if (BurstFrames() > 1) {
                manifest << fmt::format(R"({{"kind":"burst_summary","frames":{}}})",
                                        burst_index + 1) << '\n';
            }
            manifest.close();
            done = !RepeatCaptures();
            LOG_INFO(Render_Vulkan, "Frame dump completed {} frame(s) in {}",
                     burst_index + 1, directory.string());
            return;
        }
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::error_code error;
    if (!continue_burst) {
        if (frame % 30 != 0 || !std::filesystem::exists(trigger_file, error)) return;
        if (ConsumeGraphicsBufferTrigger() || ConsumeLightingRangeTrigger()) return;
        if (RepeatCaptures() && !ConsumeRepeatTrigger(trigger_file)) {
            LOG_WARNING(Render_Vulkan, "Frame dump: invalid trigger or existing capture output");
            return;
        }
        burst_remaining = BurstFrames();
        burst_index = 0;
        sequence = 0;
    }
    // All previous readbacks have completed. Keep only unique file numbering across a burst.
    readbacks.clear();
    pending_bytes = written_bytes = sampled_bytes = producer_bytes = producer_buffer_bytes = 0;
    task_buffer_bytes = 0;
    task_input_records = 0;
    task_inputs_recorded = false;
    material_inputs_capture = MaterialInputsSelection().has_value() && BurstFrames() == 1 &&
        Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL && !ComputeCaptureStopEnabled();
    canopy_inputs_capture = CanopyInputsSelection() && BurstFrames() == 1 &&
        Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL && !ComputeCaptureStopEnabled();
    canopy_input_variant = canopy_inputs_capture ? (*CanopyInputsSelection())[0] : 0;
    material_input_skip = canopy_inputs_capture ? (*CanopyInputsSelection())[1] :
                                                MaterialInputsSelection().value_or(0);
    material_anchor_matches = 0;
    images = pass_index = pass_first_draw = draws = 0;
    metadata_records = compute_metadata_records = draw_submission_records = 0;
    fog_inputs_capture = FogInputsSelection() && BurstFrames() == 1 &&
                         Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL;
    capture_stop_uniform_count = capture_stop_uniform_mask = 0;
    lighting_output_targets.clear();
    lighting_outputs_finished = false;
    written_targets.clear();
    written_dispatches = 0;
    compute_inputs_recorded = false;
    producer_inputs_recorded = false;
    producer_matching_dispatches = 0;
    producer_selected_ordinal = ~0U;
    producer_input_skip = ParticleProducerSkip();
    truncated = false;
    pass_framebuffer = nullptr;
    pass_targets.clear();
    fog_depth_target.reset();
    fog_depth_recorded = false;
    pass_hashes.clear();
    AnchorDraw() = ~0U;
    DrawPassSelection() = DrawShaderAnchor() ? ~0U :
        ReadPassSetting("EDEN_NX2_FRAME_DUMP_DRAW_PASS", ~0U);
    if (!continue_burst) {
        manifest.clear();
        std::filesystem::create_directories(directory, error);
        manifest.open(directory / "manifest.jsonl", std::ios::trunc);
        if (!manifest) {
            LOG_ERROR(Render_Vulkan, "Frame dump: cannot write {}", directory.string());
            done = true;
            material_inputs_capture = false;
            canopy_inputs_capture = false;
            fog_inputs_capture = false;
            return;
        }
    }
    LiveCaptureProducerKey().store(fog_inputs_capture ? FogInputsSelection() : 0,
                                   std::memory_order_release);
    manifest << fmt::format(R"({{"kind":"frame","frame":{},"guest_present":{},"burst_index":{},"seconds":{:.3f},"shader_anchor":"{:016x}","mode":"{}"}})",
                            frame, Core::g_nx2_present_count.load(std::memory_order_relaxed),
                            burst_index, elapsed, DrawShaderAnchor().value_or(0),
                            canopy_inputs_capture ? "canopy-inputs" :
                            material_inputs_capture ? "material-inputs" : fog_inputs_capture ?
                            (FogInputsSelection() == 0x203053e20b8c989aULL ? "sh-inputs" :
                             FogInputsSelection() == 0xd9b1657f3a0b4bc2ULL ? "lighting-inputs" : "fog-inputs") :
                            BurstFrames() > 1 ? "burst" : PhaseCaptures() ? "phase" : "draw") << '\n';
    if (canopy_inputs_capture) {
        const auto pair = CanopyInputShaders();
        manifest << fmt::format(
            R"({{"kind":"canopy_input_selection","variant":{},"{}":"{:016x}","fragment_hash":"{:016x}","scope":"{}","skip_matches":{},"buffer_descriptor_cap":32,"buffer_byte_cap":134217728,"uniforms_first":true,"compressed_atlases":"not_captured","mesh_outputs":"not_captured"}})",
            canopy_input_variant, CanopyInputIsMesh() ? "mesh_hash" : "vertex_hash", pair[0], pair[1],
            CanopyInputIsMesh() ? "selected_mesh_fragment_draw" : "selected_vertex_fragment_draw",
            material_input_skip) << '\n';
    }
    if (material_inputs_capture) {
        manifest << fmt::format(
            R"({{"kind":"material_input_selection","vertex_hash":"00e4eb4772d5076b","fragment_hash":"6cb075ad84a5c605","skip_matches":{}}})",
            material_input_skip) << '\n';
    }
    manifest << fmt::format(R"({{"kind":"graphics_buffer_trial_state","indirect64":{},"native":{}}})",
                            GraphicsIndirectRangeTrial(), GraphicsNativePointerTrial()) << '\n';
    const bool effective_domain = NativeDomainCorrectionEnabled();
    const bool effective_sh = LightingRangeMinimum(0x203053e20b8c989aULL,
        0xc73574a9766531cbULL, 0, 96, 24, false) != 0;
    const bool effective_d9 = LightingRangeMinimum(0xd9b1657f3a0b4bc2ULL,
        0x75d4369f3b1f5868ULL, 0, 96, 24, false) != 0;
    if (fog_inputs_capture && (FogInputsSelection() == 0x203053e20b8c989aULL ||
                              FogInputsSelection() == 0xd9b1657f3a0b4bc2ULL)) {
        manifest << fmt::format(
            R"({{"kind":"native_domain_range_state","enabled":{},"default_automatic":true,"override":"{}","baseline_env":{},"existing_owner_only":true,"overrides_scene_minima":true,"failure_fallback":"ordinary_binding","producer_cap":{},"buffer_cap":{}}})",
            effective_domain, !native_domain_range ? "auto" : *native_domain_range ? "on" : "off",
            OotNativeDomainBaselineRequested(), effective_domain ? size_t{336} << 20 :
                ((FogInputsSelection() == 0x203053e20b8c989aULL ? effective_sh : effective_d9) ? size_t{272} << 20 : PRODUCER_SIZE_CAP),
            effective_domain ? size_t{320} << 20 :
                ((FogInputsSelection() == 0x203053e20b8c989aULL ? effective_sh : effective_d9) ? size_t{256} << 20 : size_t{64} << 20)) << '\n';
    }
    if (fog_inputs_capture && FogInputsSelection() == 0x203053e20b8c989aULL) {
        manifest << R"({"kind":"sh_input_selection","native_key":"203053e20b8c989a","expected_canonical_key":"c73574a9766531cb","capture_phase":"pre_dispatch","read_write_inputs_included":true,"post_dispatch_output_captured":false,"uniform_copy_limit":4,"buffer_copy_limit":10})" << '\n';
        manifest << fmt::format(
            R"({{"kind":"sh_range_state","enabled":{},"child24_minimum":35191040,"child32_minimum":121642624,"producer_cap":{},"buffer_cap":{}}})",
            effective_sh, effective_domain ? size_t{336} << 20 :
                effective_sh ? size_t{272} << 20 : PRODUCER_SIZE_CAP,
            effective_domain ? size_t{320} << 20 :
                effective_sh ? size_t{256} << 20 : size_t{64} << 20) << '\n';
    }
    if (fog_inputs_capture && FogInputsSelection() == 0xd9b1657f3a0b4bc2ULL) {
        manifest << fmt::format(R"({{"kind":"lighting_range_state","enabled":{},"child24_minimum":{},"child32_minimum":{},"producer_cap":{},"buffer_cap":{}}})",
            effective_d9, lighting_range_minimum24, lighting_range_minimum32,
            effective_domain ? size_t{336} << 20 :
                effective_d9 ? size_t{272} << 20 : PRODUCER_SIZE_CAP,
            effective_domain ? size_t{320} << 20 :
                effective_d9 ? size_t{256} << 20 : size_t{64} << 20) << '\n';
    }
    capturing = true;
    LOG_INFO(Render_Vulkan, "Frame dump: capturing frame {} ({:.1f} s after boot) into {}", frame,
             elapsed, directory.string());
}

} // namespace Vulkan
