// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string_view>
#include <thread>

#include "common/common_types.h"

namespace Common::CopybackWatch {

constexpr size_t SpanBytes = 100;
constexpr size_t MaxWatches = 8;
constexpr size_t EventCapacity = 128;

enum class Kind : u32 { Unknown, BufferAsync, BufferReactive, BufferQuery, Query, Texture };
enum class Seam : u32 { DeviceMemory, GmmuNormal, GmmuBig, QueryDirect, TextureSwizzle };
enum class ArmReason : u32 { Disabled, InvalidIdentity, RegistryBusy, Capacity, Armed, Existing, AllocationFailure, SessionInactive };

struct Identity {
    u64 session{}, gmmu{}, device{}, physical{}, host{};
};

struct Event {
    u64 ticket{}, start_ns{}, completion_ns{}, thread{}, destination{}, source{}, copy_size{};
    u64 writer_address{}, writer_session{};
    Kind kind{};
    Seam seam{};
    u32 overlap_offset{}, overlap_size{};
    std::array<u8, SpanBytes> bytes{};
    bool complete{};
};

struct Slot {
    std::atomic_flag busy{};
    Event event{};
};

struct Watch {
    explicit Watch(Identity identity_) : identity{identity_} {}
    const Identity identity;
    std::atomic<u64> tickets{}, dropped_busy{}, dropped_overflow{}, dropped_ticket_race{};
    std::array<Slot, EventCapacity> slots{};
};

struct Attachment {
    std::shared_ptr<Watch> watch{};
    u64 ticket{};
    ArmReason reason{ArmReason::Disabled};
};

struct Registry {
    std::atomic_flag busy{};
    std::atomic<u64> active_session{}, session_counter{};
    std::atomic<u64> dropped_busy{}, rejected_capacity{}, rejected_identity{}, rejected_allocation{};
    std::array<std::shared_ptr<Watch>, MaxWatches> watches{};
};

inline Registry& State() {
    static Registry state;
    return state;
}

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PB_COPYBACK_WATCH");
        return value && std::string_view{value} == "1";
    }();
    return enabled;
}

inline u64 Time() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

// This diagnostic deliberately supports one GPU session per process. After teardown,
// a fresh launch is required; old queued work cannot attach to a later game's watch.
inline u64 BeginSession() {
    if (!Enabled()) return 0;
    auto& state = State();
    u64 expected = 0;
    if (!state.session_counter.compare_exchange_strong(expected, 1, std::memory_order_acq_rel,
                                                       std::memory_order_relaxed)) return 0;
    state.active_session.store(1, std::memory_order_release);
    return 1;
}
inline void EndSession(u64 session) {
    if (Enabled() && session) State().active_session.compare_exchange_strong(
        session, 0, std::memory_order_acq_rel, std::memory_order_relaxed);
}

inline u64 Session() {
    return Enabled() ? State().active_session.load(std::memory_order_acquire) : 0;
}

inline bool ValidRange(u64 begin, u64 size) {
    return begin && size && size - 1 <= UINT64_MAX - begin;
}

inline bool Overlap(u64 a, u64 a_size, u64 b, u64 b_size) {
    if (!ValidRange(a, a_size) || !ValidRange(b, b_size)) return false;
    return a <= b ? b - a < a_size : a - b < b_size;
}

// Registration happens only at the already validated submission snapshot. It never
// dereferences the remembered host interval and never waits for another observer.
inline Attachment Arm(Identity identity) {
    if (!Enabled()) return {};
    auto& state = State();
    const u64 session = state.active_session.load(std::memory_order_acquire);
    if (!session) return {.reason = ArmReason::SessionInactive};
    if (identity.session != session ||
        !ValidRange(identity.host, SpanBytes) || !ValidRange(identity.device, SpanBytes) ||
        !ValidRange(identity.physical, SpanBytes)) {
        state.rejected_identity.fetch_add(1, std::memory_order_relaxed);
        return {.reason = ArmReason::InvalidIdentity};
    }
    if (state.busy.test_and_set(std::memory_order_acquire)) {
        state.dropped_busy.fetch_add(1, std::memory_order_relaxed);
        return {.reason = ArmReason::RegistryBusy};
    }
    if (state.active_session.load(std::memory_order_acquire) != session) {
        state.busy.clear(std::memory_order_release);
        return {.reason = ArmReason::SessionInactive};
    }
    Attachment result{.reason = ArmReason::Capacity};
    size_t free_slot = MaxWatches;
    for (size_t i = 0; i < MaxWatches; ++i) {
        const auto& watch = state.watches[i];
        if (!watch || watch->identity.session != session) {
            if (free_slot == MaxWatches) free_slot = i;
        } else if (watch->identity.host == identity.host &&
                   watch->identity.physical == identity.physical) {
            result = {watch, watch->tickets.load(std::memory_order_acquire), ArmReason::Existing};
            break;
        }
    }
    // Allocation is confined to registration, not the writer or dump callbacks.
    // Release the registry before allocation and revalidate through a second try-lock.
    state.busy.clear(std::memory_order_release);
    if (result.watch) return result;
    if (free_slot == MaxWatches) {
        state.rejected_capacity.fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    std::shared_ptr<Watch> candidate;
    try {
        candidate = std::make_shared<Watch>(identity);
    } catch (...) {
        state.rejected_allocation.fetch_add(1, std::memory_order_relaxed);
        return {.reason = ArmReason::AllocationFailure};
    }
    if (state.busy.test_and_set(std::memory_order_acquire)) {
        state.dropped_busy.fetch_add(1, std::memory_order_relaxed);
        return {.reason = ArmReason::RegistryBusy};
    }
    std::shared_ptr<Watch> retired;
    if (state.active_session.load(std::memory_order_acquire) != session) {
        result.reason = ArmReason::InvalidIdentity;
    } else {
        free_slot = MaxWatches;
        for (size_t i = 0; i < MaxWatches; ++i) {
            const auto& watch = state.watches[i];
            if (!watch || watch->identity.session != session) {
                if (free_slot == MaxWatches) free_slot = i;
            } else if (watch->identity.host == identity.host &&
                       watch->identity.physical == identity.physical) {
                result = {watch, watch->tickets.load(std::memory_order_acquire), ArmReason::Existing};
                break;
            }
        }
        if (!result.watch && free_slot != MaxWatches) {
            retired = std::move(state.watches[free_slot]);
            state.watches[free_slot] = std::move(candidate);
            result = {state.watches[free_slot], 0, ArmReason::Armed};
        }
    }
    state.busy.clear(std::memory_order_release);
    if (result.reason == ArmReason::Capacity)
        state.rejected_capacity.fetch_add(1, std::memory_order_relaxed);
    return result;
}

inline std::array<std::shared_ptr<Watch>, MaxWatches> ActiveWatches(u64 session) {
    std::array<std::shared_ptr<Watch>, MaxWatches> result{};
    if (!session || Session() != session) return result;
    auto& state = State();
    if (state.busy.test_and_set(std::memory_order_acquire)) {
        state.dropped_busy.fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    for (size_t i = 0; i < MaxWatches && state.active_session.load(std::memory_order_acquire) == session; ++i)
        if (state.watches[i] && state.watches[i]->identity.session == session)
            result[i] = state.watches[i];
    state.busy.clear(std::memory_order_release);
    return result;
}

inline bool OverlapsHostRange(const void* pointer, size_t size) {
    const u64 session = Session();
    if (!session) return false;
    for (const auto& watch : ActiveWatches(session))
        if (watch && Overlap(reinterpret_cast<uintptr_t>(pointer), size,
                             watch->identity.host, SpanBytes)) return true;
    return false;
}

struct WriterContext { Kind kind{}; u64 address{}, session{}; };
inline thread_local WriterContext context{};

inline u64 WriterSession() {
    return context.kind == Kind::Unknown && !context.session ? Session() : context.session;
}

class ScopedWriter {
public:
    explicit ScopedWriter(Kind kind, u64 address = 0, u64 session = WriterSession()) : active{Enabled()} {
        if (active) {
            previous = context;
            context = {kind, address, session};
        }
    }
    ~ScopedWriter() { if (active) context = previous; }
    ScopedWriter(const ScopedWriter&) = delete;
    ScopedWriter& operator=(const ScopedWriter&) = delete;
private:
    bool active{};
    WriterContext previous{};
};
// Source-only observation around an existing copy. No guest destination is read,
// no mapping/cache operation is called, and a busy slot only drops the observation.
class Observation {
public:
    Observation(void* destination, const void* source, size_t size, Seam seam) {
        const u64 session = context.kind == Kind::Unknown && !context.session ? Session() : context.session;
        if (!session || !source) return;
        const u64 dst = reinterpret_cast<uintptr_t>(destination);
        const u64 src = reinterpret_cast<uintptr_t>(source);
        if (!ValidRange(src, size) || !ValidRange(dst, size)) return;
        auto watches = ActiveWatches(session);
        for (auto& watch : watches) {
            if (!watch || !Overlap(dst, size, watch->identity.host, SpanBytes)) continue;
            // Stop on counter overflow rather than reusing an ambiguous generation.
            u64 ticket = watch->tickets.load(std::memory_order_relaxed);
            if (ticket == UINT64_MAX) {
                watch->dropped_overflow.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (!watch->tickets.compare_exchange_strong(
                    ticket, ticket + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                watch->dropped_ticket_race.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            Slot& slot = watch->slots[ticket % EventCapacity];
            if (slot.busy.test_and_set(std::memory_order_acquire)) {
                watch->dropped_busy.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            const u64 begin = (std::max)(dst, watch->identity.host);
            const u64 source_offset = begin - dst;
            const u32 watch_offset = static_cast<u32>(begin - watch->identity.host);
            const u32 count = static_cast<u32>((std::min)(u64{SpanBytes - watch_offset},
                                                         u64{size} - source_offset));
            slot.event = Event{.ticket = ticket, .start_ns = Time(),
                .thread = static_cast<u64>(std::hash<std::thread::id>{}(std::this_thread::get_id())),
                .destination = dst, .source = src, .copy_size = size,
                .writer_address = context.address, .writer_session = context.session,
                .kind = context.kind, .seam = seam,
                .overlap_offset = watch_offset, .overlap_size = count};
            std::memcpy(slot.event.bytes.data(), static_cast<const u8*>(source) + source_offset, count);
            pending[used++] = Pending{std::move(watch), &slot};
        }
    }
    ~Observation() {
        for (size_t i = 0; i < used; ++i) {
            pending[i].slot->event.completion_ns = Time();
            pending[i].slot->event.complete = true;
            pending[i].slot->busy.clear(std::memory_order_release);
        }
    }
    Observation(const Observation&) = delete;
    Observation& operator=(const Observation&) = delete;
private:
    struct Pending { std::shared_ptr<Watch> watch{}; Slot* slot{}; };
    std::array<Pending, MaxWatches> pending{};
    size_t used{};
};

inline void* Copy(void* destination, const void* source, size_t size, Seam seam) {
    if (!Session()) return std::memcpy(destination, source, size);
    Observation observation{destination, source, size, seam};
    return std::memcpy(destination, source, size);
}

struct Snapshot {
    std::array<Event, EventCapacity> events{};
    size_t count{};
    u64 tickets{}, dropped_busy{}, dropped_overflow{}, dropped_ticket_race{}, busy_slots{};
};

inline Snapshot ReadSnapshot(Watch& watch) {
    Snapshot result;
    result.tickets = watch.tickets.load(std::memory_order_acquire);
    result.dropped_busy = watch.dropped_busy.load(std::memory_order_relaxed);
    result.dropped_overflow = watch.dropped_overflow.load(std::memory_order_relaxed);
    result.dropped_ticket_race = watch.dropped_ticket_race.load(std::memory_order_relaxed);
    for (auto& slot : watch.slots) {
        if (slot.busy.test_and_set(std::memory_order_acquire)) {
            ++result.busy_slots;
            continue;
        }
        if (slot.event.complete) result.events[result.count++] = slot.event;
        slot.busy.clear(std::memory_order_release);
    }
    std::sort(result.events.begin(), result.events.begin() + result.count,
              [](const Event& a, const Event& b) { return a.ticket < b.ticket; });
    return result;
}

} // namespace Common::CopybackWatch
