// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/thread_sampler.h"

#ifdef _WIN32

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <windows.h>

#include "common/common_types.h"

namespace Common::ThreadSampler {
namespace {

struct Target {
    std::string name;
    HANDLE handle;
    u64 samples{};
    std::unordered_map<u64, u64> self;
    std::unordered_map<u64, u64> inclusive;
    std::unordered_map<u64, u64> attributed;
    std::map<std::array<u64, 3>, u64> chains;
};

class Sampler {
public:
    static Sampler& Get() {
        static Sampler sampler;
        return sampler;
    }

    bool Enabled() const {
        return !path.empty();
    }

    void Register(const char* name) {
        HANDLE handle{};
        if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
                             THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                             FALSE, 0)) {
            return;
        }
        std::scoped_lock lock{mutex};
        targets.push_back(Target{.name = name, .handle = handle});
        if (!worker.joinable()) {
            worker = std::thread([this] { Run(); });
            worker.detach();
        }
    }

private:
    Sampler() {
        if (const char* value = std::getenv("EDEN_NX2_SAMPLE"); value && *value) {
            path = value;
        }
        if (const char* value = std::getenv("EDEN_NX2_SAMPLE_START")) {
            start_delay = std::chrono::seconds{std::atoi(value)};
        }
        const auto base = reinterpret_cast<const u8*>(GetModuleHandleA(nullptr));
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
        image_begin = reinterpret_cast<u64>(base);
        image_end = image_begin + nt->OptionalHeader.SizeOfImage;
    }

    // Unwinds with the unwind tables while the thread is suspended. The walk stops when the stack
    // pointer leaves the thread's stack, or at code without an unwind entry that isn't a leaf
    // function of the executable (JIT code: its stack layout is unknown).
    size_t Capture(HANDLE thread, std::array<u64, 32>& frames) const {
        CONTEXT context{};
        context.ContextFlags = CONTEXT_FULL;
        if (!GetThreadContext(thread, &context)) {
            return 0;
        }
        MEMORY_BASIC_INFORMATION stack{};
        if (VirtualQuery(reinterpret_cast<LPCVOID>(context.Rsp), &stack, sizeof(stack)) == 0) {
            return 0;
        }
        const u64 stack_low = reinterpret_cast<u64>(stack.AllocationBase);
        const u64 stack_high = reinterpret_cast<u64>(stack.BaseAddress) + stack.RegionSize;
        const auto on_stack = [&](u64 address) {
            return address >= stack_low && address + sizeof(u64) <= stack_high;
        };
        size_t count = 0;
        while (count < frames.size() && context.Rip != 0 && on_stack(context.Rsp)) {
            frames[count++] = context.Rip;
            DWORD64 image_base{};
            PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
            if (function == nullptr) {
                if (context.Rip < image_begin || context.Rip >= image_end) {
                    break;
                }
                context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
                context.Rsp += sizeof(DWORD64);
                continue;
            }
            PVOID handler_data{};
            DWORD64 establisher{};
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context,
                             &handler_data, &establisher, nullptr);
        }
        return count;
    }

    void Run() {
        std::this_thread::sleep_for(start_delay);
        auto last_write = std::chrono::steady_clock::now();
        std::array<u64, 32> frames{};
        std::unordered_set<u64> seen;
        for (;;) {
            {
                std::scoped_lock lock{mutex};
                for (Target& target : targets) {
                    if (SuspendThread(target.handle) == static_cast<DWORD>(-1)) {
                        continue;
                    }
                    const size_t count = Capture(target.handle, frames);
                    ResumeThread(target.handle);
                    if (count == 0) {
                        continue;
                    }
                    ++target.samples;
                    ++target.self[frames[0]];
                    // Time in DLLs (waits, memcpy, the driver) charged to the executable's
                    // function that called into them.
                    // The first three of the executable's frames, as call chains.
                    std::array<u64, 3> chain{};
                    size_t chain_length = 0;
                    for (size_t i = 0; i < count && chain_length < chain.size(); ++i) {
                        if (frames[i] >= image_begin && frames[i] < image_end) {
                            chain[chain_length++] = i == 0 ? frames[i] : frames[i] - 1;
                        }
                    }
                    if (chain_length != 0) {
                        ++target.attributed[chain[0]];
                        ++target.chains[chain];
                    }
                    seen.clear();
                    for (size_t i = 0; i < count; ++i) {
                        // A return address can be the first byte of the next function: count
                        // the call instruction instead.
                        const u64 address = i == 0 ? frames[i] : frames[i] - 1;
                        if (seen.insert(address).second) {
                            ++target.inclusive[address];
                        }
                    }
                }
            }
            const auto now = std::chrono::steady_clock::now();
            if (now - last_write >= std::chrono::seconds{10}) {
                Write();
                last_write = now;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }

    // The DLL an address outside the executable belongs to (the driver, ntdll, ...).
    std::string ModuleName(u64 address) {
        HMODULE module{};
        if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCSTR>(address), &module)) {
            return "unknown";
        }
        const auto [it, inserted] = module_names.try_emplace(reinterpret_cast<u64>(module));
        if (inserted) {
            char name[MAX_PATH]{};
            const DWORD length = GetModuleFileNameA(module, name, MAX_PATH);
            std::string path_name(name, length);
            const size_t slash = path_name.find_last_of("\\/");
            it->second = slash == std::string::npos ? path_name : path_name.substr(slash + 1);
        }
        return it->second;
    }

    void Write() {
        std::scoped_lock lock{mutex};
        const std::string temp = path + ".tmp";
        std::FILE* file = std::fopen(temp.c_str(), "w");
        if (file == nullptr) {
            return;
        }
        for (const Target& target : targets) {
            std::fprintf(file, "thread\t%s\t%llu\n", target.name.c_str(),
                         static_cast<unsigned long long>(target.samples));
            const auto dump = [&](const char* kind, const std::unordered_map<u64, u64>& counts) {
                for (const auto& [address, count] : counts) {
                    if (address >= image_begin && address < image_end) {
                        std::fprintf(file, "%s\t%s\t0x%llx\t%llu\n", target.name.c_str(), kind,
                                     static_cast<unsigned long long>(address - image_begin),
                                     static_cast<unsigned long long>(count));
                    } else {
                        std::fprintf(file, "%s\t%s\text:%s\t%llu\n", target.name.c_str(), kind,
                                     ModuleName(address).c_str(),
                                     static_cast<unsigned long long>(count));
                    }
                }
            };
            dump("self", target.self);
            dump("incl", target.inclusive);
            dump("attr", target.attributed);
            for (const auto& [chain, count] : target.chains) {
                std::fprintf(file, "%s\tchain\t0x%llx>0x%llx>0x%llx\t%llu\n", target.name.c_str(),
                             static_cast<unsigned long long>(chain[0] ? chain[0] - image_begin : 0),
                             static_cast<unsigned long long>(chain[1] ? chain[1] - image_begin : 0),
                             static_cast<unsigned long long>(chain[2] ? chain[2] - image_begin : 0),
                             static_cast<unsigned long long>(count));
            }
        }
        std::fclose(file);
        MoveFileExA(temp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING);
    }

    std::string path;
    std::chrono::seconds start_delay{};
    u64 image_begin{};
    u64 image_end{};
    std::mutex mutex;
    std::unordered_map<u64, std::string> module_names;
    std::vector<Target> targets;
    std::thread worker;
};

} // namespace

void RegisterCurrentThread(const char* name) {
    Sampler& sampler = Sampler::Get();
    if (sampler.Enabled()) {
        sampler.Register(name);
    }
}

} // namespace Common::ThreadSampler

#else

namespace Common::ThreadSampler {
void RegisterCurrentThread(const char*) {}
} // namespace Common::ThreadSampler

#endif
