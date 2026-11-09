// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <source_location>
#include <string_view>
#include <thread>

namespace VideoCommon {

/// Opt-in, bounded host breadcrumbs. No Vulkan calls, GPU waits, or dynamic formatting.
/// EDEN_NX2_COMMAND_TRACE is a file prefix; .0.tsv/.1.tsv retain at most 8 MiB each.
class CommandTrace {
public:
    static constexpr std::size_t FileLimit = 8 * 1024 * 1024;

    static CommandTrace& Get() noexcept {
        static CommandTrace trace;
        return trace;
    }

    bool Enabled() const noexcept {
        return enabled.load(std::memory_order_relaxed);
    }

    std::uint64_t NextId() noexcept {
        return next_id.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    /// Columns: sequence, microseconds, thread, event, id, tick, known_gpu_tick,
    /// a, b, c, source, line, command_type. Tabs/newlines in strings become spaces.
    void Record(std::string_view event, std::uint64_t id, std::uint64_t tick,
                std::uint64_t known = 0, std::uint64_t a = 0, std::uint64_t b = 0,
                std::uint64_t c = 0,
                std::source_location source = std::source_location::current(),
                std::string_view type = {}) noexcept {
        if (!Enabled()) return;
        try {
            std::lock_guard lock{mutex};
            if (!Enabled()) return;
            char event_text[96], source_text[768], type_text[2048];
            Clean(event_text, event);
            Clean(source_text, source.file_name());
            Clean(type_text, type);
            const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            const auto thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
            char line[4096];
            const int size = std::snprintf(line, sizeof(line),
                "%llu\t%lld\t%llu\t%s\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%s\t%u\t%s\n",
                static_cast<unsigned long long>(++sequence), static_cast<long long>(micros),
                static_cast<unsigned long long>(thread), event_text,
                static_cast<unsigned long long>(id), static_cast<unsigned long long>(tick),
                static_cast<unsigned long long>(known), static_cast<unsigned long long>(a),
                static_cast<unsigned long long>(b), static_cast<unsigned long long>(c),
                source_text, source.line(), type_text);
            if (size <= 0 || static_cast<std::size_t>(size) >= sizeof(line)) return;
            if (!file || bytes + static_cast<std::size_t>(size) > FileLimit) {
                if (file) {
                    std::fclose(file);
                    file = nullptr;
                    slot ^= 1;
                }
                char path[4096];
                const int length = std::snprintf(path, sizeof(path), "%s.%u.tsv", prefix, slot);
                if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(path)) {
                    enabled.store(false, std::memory_order_relaxed);
                    return;
                }
                file = std::fopen(path, "wb");
                bytes = 0;
                if (!file) {
                    enabled.store(false, std::memory_order_relaxed);
                    return;
                }
            }
            const auto written = std::fwrite(line, 1, static_cast<std::size_t>(size), file);
            bytes += written;
            if (written != static_cast<std::size_t>(size) || std::fflush(file) != 0) {
                enabled.store(false, std::memory_order_relaxed);
            }
        } catch (...) {
            // The diagnostics must not turn a recoverable application error into a crash.
            enabled.store(false, std::memory_order_relaxed);
        }
    }

    /// Small explicit host-phase marker; all numeric arguments are caller-defined.
    void Mark(std::string_view event, std::uint64_t tick, std::uint64_t known = 0,
              std::uint64_t a = 0, std::uint64_t b = 0, std::uint64_t c = 0,
              std::source_location source = std::source_location::current()) noexcept {
        if (Enabled()) Record(event, 0, tick, known, a, b, c, source);
    }

private:
    CommandTrace() noexcept {
        const char* value = std::getenv("EDEN_NX2_COMMAND_TRACE");
        if (!value || value[0] == '\0') return;
        const std::size_t length = std::strlen(value);
        if (length + 16 >= sizeof(prefix)) return;
        std::memcpy(prefix, value, length + 1);
        enabled.store(true, std::memory_order_relaxed);
    }

    ~CommandTrace() noexcept {
        if (file) std::fclose(file);
    }

    template <std::size_t N>
    static void Clean(char (&out)[N], std::string_view value) noexcept {
        const auto count = value.size() < N - 1 ? value.size() : N - 1;
        for (std::size_t i = 0; i < count; ++i) {
            const char ch = value[i];
            out[i] = ch == '\t' || ch == '\r' || ch == '\n' || ch == '\0' ? ' ' : ch;
        }
        out[count] = '\0';
    }

    std::atomic<bool> enabled{};
    std::atomic<std::uint64_t> next_id{};
    std::mutex mutex;
    char prefix[4096]{};
    std::FILE* file{};
    std::uint64_t sequence{};
    std::size_t bytes{};
    unsigned slot{};
};

template <class T>
constexpr const char* CommandTraceType() noexcept {
#ifdef _MSC_VER
    return __FUNCSIG__;
#else
    return __PRETTY_FUNCTION__;
#endif
}

/// Paired markers also report exception unwinding; the original exception propagates unchanged.
class CommandTraceScope {
public:
    CommandTraceScope(const char* phase_, std::uint64_t id_, std::uint64_t tick_,
                      std::uint64_t a_ = 0, std::uint64_t b_ = 0, std::uint64_t c_ = 0,
                      std::source_location source_ = std::source_location::current(),
                      const char* type_ = "") noexcept
        : phase{phase_}, id{id_}, tick{tick_}, a{a_}, b{b_}, c{c_}, source{source_},
          type{type_}, active{CommandTrace::Get().Enabled()},
          exceptions{active ? std::uncaught_exceptions() : 0} {
        Note("begin");
    }
    ~CommandTraceScope() noexcept {
        if (active) Note(std::uncaught_exceptions() > exceptions ? "exception" : "end");
    }
    CommandTraceScope(const CommandTraceScope&) = delete;
    CommandTraceScope& operator=(const CommandTraceScope&) = delete;

private:
    void Note(const char* suffix) noexcept {
        if (!active) return;
        auto& trace = CommandTrace::Get();
        if (!trace.Enabled()) return;
        char event[96];
        std::snprintf(event, sizeof(event), "%s.%s", phase, suffix);
        trace.Record(event, id, tick, 0, a, b, c, source, type);
    }
    const char* phase;
    std::uint64_t id, tick, a, b, c;
    std::source_location source;
    const char* type;
    bool active;
    int exceptions;
};
} // namespace VideoCommon
