// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "common/common_types.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace Tegra {

// Private, marker-armed CPU provenance. It observes existing inputs and
// performs no GPU read.
class MeshCountProvenance {
public:
  static constexpr u32 MaxRecords = 4096, MaxCaptures = 8, MaxOwners = 8;
  static constexpr u64 MaxFrames = 16, MaxBytes = 8 * 1024 * 1024;
  static constexpr u32 MaxParams = 32, MaxDmas = 4, MaxWords = 32,
                       MaxHistory = 8;
  enum class MethodKind {
    VertexCount,
    ModernCount,
    VertexFirst,
    Begin,
    End,
    IndexCount,
    IndexFirst
  };
  static constexpr std::optional<MethodKind> Kind(u32 method) {
    switch (method) {
    case 0x35e:
      return MethodKind::VertexCount; // byte 0xd78
    case 0x9d:
      return MethodKind::ModernCount; // DRAW_VERTEX_ARRAY_BEGIN_END_B, byte
                                      // 0x274
    case 0x35d:
    case 0x9c:
      return MethodKind::VertexFirst;
    case 0x586:
      return MethodKind::Begin;
    case 0x585:
      return MethodKind::End;
    case 0x5f7:
      return MethodKind::IndexFirst;
    case 0x5f8:
      return MethodKind::IndexCount;
    default:
      return std::nullopt;
    }
  }

private:
  struct Dma {
    u64 address;
    u32 requested, words, ram;
    bool fifo, dirty;
    std::array<u32, MaxWords> data;
  };
  struct Context {
    const void *owner;
    u64 epoch, call, generation;
    u32 start;
    std::span<const u32> parameters, code;
    u32 dma_total;
    std::array<Dma, MaxDmas> dmas;
  };
  struct Origin {
    bool known{};
    u32 path{};
    u64 present{}, serial{}, call{}, generation{};
    u32 start{}, method{}, raw{}, effective{}, dma_total{};
    size_t parameter_count{};
    std::array<u32, MaxParams> parameters{};
    std::array<Dma, MaxDmas> dmas{};
    std::string code_file;
  };
  struct Method {
    u64 serial{}, present{}, call{};
    u32 method{}, raw{}, effective{}, start{};
    bool macro{};
    u32 path{};
  };
  struct Owner {
    const void *identity{};
    std::array<bool, 2> count_seen{};
    std::array<Origin, 2> count;
    std::array<Method, MaxHistory> history{};
    u32 methods{};
  };
  inline static thread_local Context *current{};

public:
  class Scope {
  public:
    Scope(const void *owner, u32 start, u64 generation,
          std::span<const u32> parameters, std::span<const u32> code) {
      auto &trace = Get();
      if (!trace.Enabled() ||
          !(context.epoch = trace.active_epoch.load(std::memory_order_acquire)))
        return;
      context.owner = owner;
      context.start = start;
      context.generation = generation;
      context.parameters = parameters;
      context.code = code;
      context.dma_total = 0;
      context.call = trace.calls.fetch_add(1, std::memory_order_relaxed) + 1;
      previous = current;
      current = &context;
      installed = true;
    }
    ~Scope() {
      if (installed)
        current = previous;
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

  private:
    Context context;
    Context *previous{};
    bool installed{};
  };

  static MeshCountProvenance &Get() {
    static MeshCountProvenance trace{std::getenv("EDEN_NX2_MESH_COUNT_TRACE")};
    return trace;
  }
  bool Enabled() const noexcept { return !prefix.empty(); }
  bool Active() const noexcept {
    return active_epoch.load(std::memory_order_relaxed) != 0;
  }

  void NoteDma(const void *owner, u64 address, u32 requested, u32 ram,
               bool fifo, bool dirty, std::span<const u32> words) {
    if (!current || current->owner != owner ||
        current->epoch != active_epoch.load(std::memory_order_acquire))
      return;
    Dma &dma = current->dmas[current->dma_total++ % MaxDmas];
    dma = {};
    dma.address = address;
    dma.requested = requested;
    dma.words = static_cast<u32>(words.size());
    dma.ram = ram;
    dma.fifo = fifo;
    dma.dirty = dirty;
    std::copy_n(words.begin(), std::min<size_t>(words.size(), MaxWords),
                dma.data.begin());
  }

  void NoteMethod(const void *owner, u64 present, MethodKind kind, u32 method,
                  u32 raw, u32 effective, u32 path) {
    if (!Active())
      return;
    std::scoped_lock lock{mutex};
    if (!InWindow(present))
      return;
    Owner *state = FindOwner(owner);
    if (!state)
      return;
    const bool known =
        path == 0 && current && current->owner == owner &&
        current->epoch == active_epoch.load(std::memory_order_relaxed);
    const u64 serial = ++method_serial;
    state->history[state->methods++ % MaxHistory] = {
        serial, present,   known ? current->call : 0,  method,
        raw,    effective, known ? current->start : 0, known,
        path};
    if (kind != MethodKind::VertexCount && kind != MethodKind::ModernCount)
      return;
    const u32 count_slot = kind == MethodKind::ModernCount ? 1 : 0;
    state->count_seen[count_slot] = true;
    Origin &origin = state->count[count_slot];
    origin = {};
    origin.present = present;
    origin.serial = serial;
    origin.method = method;
    origin.raw = raw;
    origin.effective = effective;
    origin.path = path;
    origin.known = known;
    if (known) {
      origin.call = current->call;
      origin.start = current->start;
      origin.generation = current->generation;
      origin.parameter_count = current->parameters.size();
      std::copy_n(current->parameters.begin(),
                  std::min<size_t>(current->parameters.size(), MaxParams),
                  origin.parameters.begin());
      origin.dma_total = current->dma_total;
      const u32 count = std::min(current->dma_total, MaxDmas);
      for (u32 i = 0; i < count; ++i)
        origin.dmas[i] =
            current->dmas[(current->dma_total - count + i) % MaxDmas];
      if (effective == 0 || count_slot == 1)
        origin.code_file = SaveCode(*current, state - owners.data());
    }
    if (count_slot == 1) {
      WriteRow(
          present,
          fmt::format(
              R"({{"kind":"mesh_modern_count_write","guest_present":{},"record":{},"owner":{},"origin":{}}})",
              present, records, state - owners.data(),
              OriginJson(origin, count_slot, std::nullopt)) +
              '\n');
    }
  }

  // Only final native mesh attempts poll the marker. The first attempt may
  // precede a captured writer.
  bool BeginDraw(u64 present, bool valid_title, bool native, bool mesh) {
    if (!Enabled() || !valid_title || !native || !mesh)
      return false;
    std::scoped_lock lock{mutex};
    if (Active() && !InWindow(present))
      return false;
    if (!Active() && present != last_poll && captures < MaxCaptures) {
      last_poll = present;
      std::error_code ec;
      const std::filesystem::path marker{prefix + ".now"};
      const auto path = fmt::format("{}-{:02}.jsonl", prefix, captures);
      if (std::filesystem::exists(marker, ec) &&
          !std::filesystem::exists(path, ec)) {
        file.clear();
        file.open(path, std::ios::binary | std::ios::out);
        if (file) {
          std::filesystem::remove(marker, ec);
          first_present = present;
          records = 0;
          bytes = 0;
          code_count = 0;
          for (auto &owner : owners)
            owner = {};
          ++captures;
          active_epoch.store(++epoch, std::memory_order_release);
          file
              << fmt::format(
                     R"({{"kind":"mesh_count_trace","guest_present":{},"max_frames":{},"max_records":{},"max_data_row_bytes":{},"origin_scope":"ordinary_back_native_macro_only","writer_paths":{{"0":"CallMethod","1":"split_front_no_macro_origin","2":"state_sink_effective_value_only"}},"extra_gpu_reads":false,"extra_gpu_flushes":false}})",
                     present, MaxFrames, MaxRecords, MaxBytes)
              << '\n';
        }
      }
    }
    return Active();
  }

  void Draw(const void *owner, u64 present, u32 vertex_count,
            std::string_view fields) {
    std::scoped_lock lock{mutex};
    if (!InWindow(present))
      return;
    Owner *state = FindOwner(owner);
    std::string out =
        fmt::format(
            R"({{"kind":"mesh_count_draw","guest_present":{},"record":{},"owner":{},)",
            present, records, state ? state - owners.data() : -1) +
        std::string{fields};
    out += ",\"count_writers\":[";
    for (u32 count_slot = 0; count_slot < 2; ++count_slot) {
      if (count_slot)
        out += ',';
      if (!state || !state->count_seen[count_slot])
        out += "null";
      else {
        out += OriginJson(state->count[count_slot], count_slot, vertex_count);
      }
    }
    out += "],\"recent_methods\":[";
    if (state)
      for (u32 i = 0, n = std::min(state->methods, MaxHistory); i < n; ++i) {
        const auto &m = state->history[(state->methods - n + i) % MaxHistory];
        if (i)
          out += ',';
        out += fmt::format(
            R"({{"serial":{},"guest_present":{},"method":{},"raw":{},"effective":{},"macro_known":{},"macro_call":{},"macro_start":{},"writer_path":{}}})",
            m.serial, m.present, m.method, m.raw, m.effective, m.macro, m.call,
            m.start, m.path);
      }
    out += "]}\n";
    WriteRow(present, out);
  }

private:
  explicit MeshCountProvenance(const char *path) : prefix{path ? path : ""} {}
  template <size_t N>
  static std::string Words(const std::array<u32, N> &data, u32 count) {
    std::string out;
    for (u32 i = 0; i < count; ++i) {
      if (i)
        out += ',';
      out += std::to_string(data[i]);
    }
    return out;
  }
  static std::string OriginJson(const Origin &c, u32 count_slot,
                                std::optional<u32> vertex_count) {
    std::string out = fmt::format(
        R"({{"source":"{}","method":{},"raw":{},"effective":{},"guest_present":{},"method_serial":{},"matches_vertex_count":{},"macro_origin_known":{},"writer_path":{},"macro_call":{},"macro_start":{},"code_generation":{},"code_file":"{}","parameter_count":{},"parameters_truncated":{},"parameters":[{}],"dma_total_before_count_write":{},"dma_records":[)",
        count_slot ? "modern_begin_end_b" : "legacy_vertex_count", c.method,
        c.raw, c.effective, c.present, c.serial,
        vertex_count ? (c.effective == *vertex_count ? "true" : "false")
                     : "null",
        c.known, c.path, c.call, c.start, c.generation, c.code_file,
        c.parameter_count, c.parameter_count > MaxParams,
        Words(c.parameters,
              static_cast<u32>(std::min<size_t>(c.parameter_count, MaxParams))),
        c.dma_total);
    for (u32 i = 0; i < std::min(c.dma_total, MaxDmas); ++i) {
      const auto &d = c.dmas[i];
      if (i)
        out += ',';
      out += fmt::format(
          R"({{"address":"{:#x}","requested_words":{},"read_words":{},"ram":{},"fifo":{},"dirty_before_existing_flush":{},"captured_words":{},"truncated":{},"words":[{}]}})",
          d.address, d.requested, d.words, d.ram, d.fifo, d.dirty,
          std::min(d.words, MaxWords), d.words > MaxWords,
          Words(d.data, std::min(d.words, MaxWords)));
    }
    out += "]}";
    return out;
  }
  void WriteRow(u64 present, std::string_view out) {
    if (out.size() > MaxBytes - bytes) {
      Finish(present, "byte_limit");
      return;
    }
    file << out;
    bytes += out.size();
    if (++records == MaxRecords)
      Finish(present, "record_limit");
  }
  Owner *FindOwner(const void *owner) {
    if (!owner)
      return nullptr;
    for (auto &state : owners)
      if (state.identity == owner)
        return &state;
    for (auto &state : owners)
      if (!state.identity) {
        state.identity = owner;
        return &state;
      }
    return nullptr;
  }
  bool InWindow(u64 present) {
    if (!Active())
      return false;
    if (present < first_present || present - first_present >= MaxFrames) {
      Finish(present, "frame_limit");
      return false;
    }
    return true;
  }
  std::string SaveCode(const Context &context, size_t owner) {
    for (u32 i = 0; i < code_count; ++i)
      if (code_generations[i] == context.generation && code_owners[i] == owner)
        return code_files[i];
    if (code_count == 2 || context.code.empty() || context.code.size() > 0x3000)
      return {};
    // A basename keeps JSON independent of backslash escaping in the configured
    // prefix.
    const std::filesystem::path path{fmt::format("{}-{:02}-owner{}-code{}.bin",
                                                 prefix, captures - 1, owner,
                                                 context.generation)};
    std::error_code ec;
    if (std::filesystem::exists(path, ec))
      return {};
    std::ofstream code{path, std::ios::binary};
    if (!code)
      return {};
    code.write(reinterpret_cast<const char *>(context.code.data()),
               context.code.size_bytes());
    if (!code)
      return {};
    code_generations[code_count] = context.generation;
    code_owners[code_count] = owner;
    code_files[code_count] = path.filename().string();
    return code_files[code_count++];
  }
  void Finish(u64 present, const char *reason) {
    file
        << fmt::format(
               R"({{"kind":"mesh_count_trace_end","guest_present":{},"records":{},"bytes":{},"reason":"{}"}})",
               present, records, bytes, reason)
        << '\n';
    file.close();
    active_epoch.store(0, std::memory_order_release);
  }
  const std::string prefix;
  std::ofstream file;
  std::mutex mutex;
  std::atomic<u64> active_epoch{}, calls{};
  std::array<Owner, MaxOwners> owners{};
  u64 epoch{}, first_present{}, last_poll{~u64{}}, method_serial{}, bytes{};
  u32 captures{}, records{}, code_count{};
  std::array<u64, 2> code_generations{};
  std::array<size_t, 2> code_owners{};
  std::array<std::string, 2> code_files;
};
} // namespace Tegra
