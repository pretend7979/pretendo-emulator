// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>

#include "common/alignment.h"
#include "common/scope_exit.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/breadth_first_search.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/native_pointer_ranges.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {
/// Address in constant buffers to the storage buffer descriptor
struct StorageBufferAddr {
    auto operator<=>(const StorageBufferAddr&) const noexcept = default;

    u32 index;
    u32 offset;
    u32 pointer_offset{~0U};
};

/// Block iterator to a global memory instruction and the storage buffer it uses
struct StorageInst {
    StorageBufferAddr storage_buffer;
    IR::Inst* inst;
    IR::Block* block;
    std::optional<IR::U32> pointer_base;
};

/// Bias towards a certain range of constant buffers when looking for storage buffers
struct Bias {
    u32 index;
    u32 offset_begin;
    u32 offset_end;
    u32 alignment;
};

using boost::container::flat_set;
using boost::container::small_vector;
using StorageBufferSet =
    flat_set<StorageBufferAddr, std::less<StorageBufferAddr>, small_vector<StorageBufferAddr, 16>>;
using StorageInstVector = small_vector<StorageInst, 24>;
using StorageWritesSet =
    flat_set<StorageBufferAddr, std::less<StorageBufferAddr>, small_vector<StorageBufferAddr, 16>>;

struct StorageInfo {
    StorageBufferSet set;
    StorageInstVector to_replace;
    StorageWritesSet writes;
    /// Track results by the instruction the search starts from, with and without the NVN bias:
    /// accesses through the same address share them, and a search can walk thousands of
    /// instructions (the IR does not change while storage buffers are collected).
    std::unordered_map<const IR::Inst*, std::optional<StorageBufferAddr>> biased_tracks;
    std::unordered_map<const IR::Inst*, std::optional<StorageBufferAddr>> unbiased_tracks;
    std::unordered_map<const IR::Inst*, small_vector<StorageBufferAddr, 4>> native_candidates;
    /// A native access with several candidate pointers was left global.
    bool native_runtime_bounds{};
};

/// Returns true when the instruction is a global memory instruction
bool IsGlobalMemory(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadGlobalS8:
    case IR::Opcode::LoadGlobalU8:
    case IR::Opcode::LoadGlobalS16:
    case IR::Opcode::LoadGlobalU16:
    case IR::Opcode::LoadGlobal32:
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::LoadGlobal128:
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return true;
    default:
        return false;
    }
}

/// Returns true when the instruction is a global memory instruction
bool IsGlobalMemoryWrite(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return true;
    default:
        return false;
    }
}

/// Converts a global memory opcode to its storage buffer equivalent
IR::Opcode GlobalToStorage(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::LoadGlobalS8:
        return IR::Opcode::LoadStorageS8;
    case IR::Opcode::LoadGlobalU8:
        return IR::Opcode::LoadStorageU8;
    case IR::Opcode::LoadGlobalS16:
        return IR::Opcode::LoadStorageS16;
    case IR::Opcode::LoadGlobalU16:
        return IR::Opcode::LoadStorageU16;
    case IR::Opcode::LoadGlobal32:
        return IR::Opcode::LoadStorage32;
    case IR::Opcode::LoadGlobal64:
        return IR::Opcode::LoadStorage64;
    case IR::Opcode::LoadGlobal128:
        return IR::Opcode::LoadStorage128;
    case IR::Opcode::WriteGlobalS8:
        return IR::Opcode::WriteStorageS8;
    case IR::Opcode::WriteGlobalU8:
        return IR::Opcode::WriteStorageU8;
    case IR::Opcode::WriteGlobalS16:
        return IR::Opcode::WriteStorageS16;
    case IR::Opcode::WriteGlobalU16:
        return IR::Opcode::WriteStorageU16;
    case IR::Opcode::WriteGlobal32:
        return IR::Opcode::WriteStorage32;
    case IR::Opcode::WriteGlobal64:
        return IR::Opcode::WriteStorage64;
    case IR::Opcode::WriteGlobal128:
        return IR::Opcode::WriteStorage128;
    case IR::Opcode::GlobalAtomicIAdd32:
        return IR::Opcode::StorageAtomicIAdd32;
    case IR::Opcode::GlobalAtomicSMin32:
        return IR::Opcode::StorageAtomicSMin32;
    case IR::Opcode::GlobalAtomicUMin32:
        return IR::Opcode::StorageAtomicUMin32;
    case IR::Opcode::GlobalAtomicSMax32:
        return IR::Opcode::StorageAtomicSMax32;
    case IR::Opcode::GlobalAtomicUMax32:
        return IR::Opcode::StorageAtomicUMax32;
    case IR::Opcode::GlobalAtomicInc32:
        return IR::Opcode::StorageAtomicInc32;
    case IR::Opcode::GlobalAtomicDec32:
        return IR::Opcode::StorageAtomicDec32;
    case IR::Opcode::GlobalAtomicAnd32:
        return IR::Opcode::StorageAtomicAnd32;
    case IR::Opcode::GlobalAtomicOr32:
        return IR::Opcode::StorageAtomicOr32;
    case IR::Opcode::GlobalAtomicXor32:
        return IR::Opcode::StorageAtomicXor32;
    case IR::Opcode::GlobalAtomicExchange32:
        return IR::Opcode::StorageAtomicExchange32;
    case IR::Opcode::GlobalAtomicIAdd64:
        return IR::Opcode::StorageAtomicIAdd64;
    case IR::Opcode::GlobalAtomicSMin64:
        return IR::Opcode::StorageAtomicSMin64;
    case IR::Opcode::GlobalAtomicUMin64:
        return IR::Opcode::StorageAtomicUMin64;
    case IR::Opcode::GlobalAtomicSMax64:
        return IR::Opcode::StorageAtomicSMax64;
    case IR::Opcode::GlobalAtomicUMax64:
        return IR::Opcode::StorageAtomicUMax64;
    case IR::Opcode::GlobalAtomicAnd64:
        return IR::Opcode::StorageAtomicAnd64;
    case IR::Opcode::GlobalAtomicOr64:
        return IR::Opcode::StorageAtomicOr64;
    case IR::Opcode::GlobalAtomicXor64:
        return IR::Opcode::StorageAtomicXor64;
    case IR::Opcode::GlobalAtomicExchange64:
        return IR::Opcode::StorageAtomicExchange64;
    case IR::Opcode::GlobalAtomicIAdd32x2:
        return IR::Opcode::StorageAtomicIAdd32x2;
    case IR::Opcode::GlobalAtomicSMin32x2:
        return IR::Opcode::StorageAtomicSMin32x2;
    case IR::Opcode::GlobalAtomicUMin32x2:
        return IR::Opcode::StorageAtomicUMin32x2;
    case IR::Opcode::GlobalAtomicSMax32x2:
        return IR::Opcode::StorageAtomicSMax32x2;
    case IR::Opcode::GlobalAtomicUMax32x2:
        return IR::Opcode::StorageAtomicUMax32x2;
    case IR::Opcode::GlobalAtomicAnd32x2:
        return IR::Opcode::StorageAtomicAnd32x2;
    case IR::Opcode::GlobalAtomicOr32x2:
        return IR::Opcode::StorageAtomicOr32x2;
    case IR::Opcode::GlobalAtomicXor32x2:
        return IR::Opcode::StorageAtomicXor32x2;
    case IR::Opcode::GlobalAtomicExchange32x2:
        return IR::Opcode::StorageAtomicExchange32x2;
    case IR::Opcode::GlobalAtomicAddF32:
        return IR::Opcode::StorageAtomicAddF32;
    case IR::Opcode::GlobalAtomicAddF16x2:
        return IR::Opcode::StorageAtomicAddF16x2;
    case IR::Opcode::GlobalAtomicMinF16x2:
        return IR::Opcode::StorageAtomicMinF16x2;
    case IR::Opcode::GlobalAtomicMaxF16x2:
        return IR::Opcode::StorageAtomicMaxF16x2;
    case IR::Opcode::GlobalAtomicAddF32x2:
        return IR::Opcode::StorageAtomicAddF32x2;
    case IR::Opcode::GlobalAtomicMinF32x2:
        return IR::Opcode::StorageAtomicMinF32x2;
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return IR::Opcode::StorageAtomicMaxF32x2;
    default:
        throw InvalidArgument("Invalid global memory opcode {}", opcode);
    }
}

/// Returns true when a storage buffer address satisfies a bias
bool MeetsBias(const StorageBufferAddr& storage_buffer, const Bias& bias) noexcept {
    return storage_buffer.index == bias.index && storage_buffer.offset >= bias.offset_begin &&
           storage_buffer.offset < bias.offset_end;
}

struct LowAddrInfo {
    IR::U32 value;
    s32 imm_offset;
};

/// Tries to track the first 32-bits of a global memory instruction
std::optional<LowAddrInfo> TrackLowAddress(IR::Inst* inst) {
    // The first argument is the low level GPU pointer to the global memory instruction
    const IR::Value addr{inst->Arg(0)};
    if (addr.IsImmediate()) {
        // Not much we can do if it's an immediate
        return std::nullopt;
    }
    // This address is expected to either be a PackUint2x32, a IAdd64, or a CompositeConstructU32x2
    IR::Inst* addr_inst{addr.InstRecursive()};
    s32 imm_offset{0};
    if (addr_inst->GetOpcode() == IR::Opcode::IAdd64) {
        // If it's an IAdd64, get the immediate offset it is applying and grab the address
        // instruction. This expects for the instruction to be canonicalized having the address on
        // the first argument and the immediate offset on the second one.
        const IR::U64 imm_offset_value{addr_inst->Arg(1)};
        if (!imm_offset_value.IsImmediate()) {
            return std::nullopt;
        }
        imm_offset = static_cast<s32>(static_cast<s64>(imm_offset_value.U64()));
        const IR::U64 iadd_addr{addr_inst->Arg(0)};
        if (iadd_addr.IsImmediate()) {
            return std::nullopt;
        }
        addr_inst = iadd_addr.InstRecursive();
    }
    // With IAdd64 handled, now PackUint2x32 is expected
    if (addr_inst->GetOpcode() == IR::Opcode::PackUint2x32) {
        // PackUint2x32 is expected to be generated from a vector
        const IR::Value vector{addr_inst->Arg(0)};
        if (vector.IsImmediate()) {
            return std::nullopt;
        }
        addr_inst = vector.InstRecursive();
    }
    // The vector is expected to be a CompositeConstructU32x2
    if (addr_inst->GetOpcode() != IR::Opcode::CompositeConstructU32x2) {
        return std::nullopt;
    }
    // Grab the first argument from the CompositeConstructU32x2, this is the low address.
    return LowAddrInfo{
        .value{IR::U32{addr_inst->Arg(0)}},
        .imm_offset = imm_offset,
    };
}

struct PointerWords {
    IR::Value low;
    IR::Value high;
    u64 displacement{};
};

// Only exact 64-bit pointer construction is accepted. In particular a select,
// phi, truncated pointer, or arithmetic on one half is not a static table root.
std::optional<PointerWords> SplitPointerWords(IR::Value address) {
    u64 displacement{};
    if (address.IsImmediate()) return std::nullopt;
    const IR::Inst* inst = address.InstRecursive();
    if (inst->GetOpcode() == IR::Opcode::IAdd64) {
        if (!inst->Arg(1).IsImmediate()) return std::nullopt;
        displacement = inst->Arg(1).U64();
        if (inst->Arg(0).IsImmediate()) return std::nullopt;
        inst = inst->Arg(0).InstRecursive();
    }
    if (inst->GetOpcode() != IR::Opcode::PackUint2x32 || inst->Arg(0).IsImmediate())
        return std::nullopt;
    inst = inst->Arg(0).InstRecursive();
    if (inst->GetOpcode() != IR::Opcode::CompositeConstructU32x2) return std::nullopt;
    return PointerWords{inst->Arg(0), inst->Arg(1), displacement};
}

std::optional<StorageBufferAddr> StaticCbufWord(IR::Value value) {
    if (value.IsImmediate()) return std::nullopt;
    const IR::Inst* inst = value.InstRecursive();
    u32 component{};
    if (inst->GetOpcode() == IR::Opcode::CompositeExtractU32x2) {
        if (!inst->Arg(1).IsImmediate() || inst->Arg(1).U32() > 1 ||
            inst->Arg(0).IsImmediate()) return std::nullopt;
        component = inst->Arg(1).U32();
        inst = inst->Arg(0).InstRecursive();
        if (inst->GetOpcode() != IR::Opcode::GetCbufU32x2) return std::nullopt;
    } else if (inst->GetOpcode() != IR::Opcode::GetCbufU32) {
        return std::nullopt;
    }
    if (!inst->Arg(0).IsImmediate() || !inst->Arg(1).IsImmediate()) return std::nullopt;
    const u32 offset = inst->Arg(1).U32();
    if (offset > 0xfffffffbu) return std::nullopt;
    return StorageBufferAddr{inst->Arg(0).U32(), offset + component * 4};
}

std::optional<StorageBufferAddr> StaticPointerTableAddress(IR::Value address, Environment& env) {
    const auto words = SplitPointerWords(address);
    if (!words || words->displacement > 0xfffffff8u) return std::nullopt;
    const auto low = StaticCbufWord(words->low);
    const auto high = StaticCbufWord(words->high);
    if (!low || !high || low->index != high->index || high->offset != low->offset + 4 ||
        !Common::IsAligned(low->offset, 8) ||
        !env.IsCbufStorageBufferPointer(low->index, low->offset)) return std::nullopt;
    return StorageBufferAddr{low->index, low->offset, static_cast<u32>(words->displacement)};
}

// A fixed byte offset proves the generated direct storage write stays inside a known span.
std::optional<u32> FixedOffsetFromRoot(IR::Value value, const StorageBufferAddr& root,
                                     unsigned depth = 0) {
    if (depth > 16 || value.IsImmediate()) return std::nullopt;
    if (const auto word = StaticCbufWord(value)) {
        return word->index == root.index && word->offset == root.offset
            ? std::optional<u32>{0} : std::nullopt;
    }
    const IR::Inst* inst = value.InstRecursive();
    if (inst->GetOpcode() != IR::Opcode::IAdd32) return std::nullopt;
    for (size_t a = 0; a < 2; ++a) {
        if (!inst->Arg(a).IsImmediate()) continue;
        const auto base = FixedOffsetFromRoot(inst->Arg(1 - a), root, depth + 1);
        const u64 offset = base ? u64{*base} + inst->Arg(a).U32() : ~u64{0};
        if (offset <= 0xffff) return static_cast<u32>(offset);
    }
    return std::nullopt;
}

std::optional<u16> FixedStorageWriteEnd(const StorageInst& replacement, u64 alignment) {
    if (replacement.pointer_base || replacement.storage_buffer.pointer_offset != ~0U ||
        alignment == 0 || !std::has_single_bit(alignment)) return std::nullopt;
    u32 width{};
    switch (replacement.inst->GetOpcode()) {
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::GlobalAtomicIAdd32: width = 4; break;
    case IR::Opcode::WriteGlobal64: width = 8; break;
    case IR::Opcode::WriteGlobal128: width = 16; break;
    default: return std::nullopt;
    }
    const auto address = TrackLowAddress(replacement.inst);
    if (!address) return std::nullopt;
    const auto offset = FixedOffsetFromRoot(address->value, replacement.storage_buffer);
    if (!offset) return std::nullopt;
    const s64 byte_offset = s64{*offset} + address->imm_offset;
    if (byte_offset < 0 || alignment > 0xffff) return std::nullopt;
    const u64 end = static_cast<u64>(byte_offset) + alignment - 1 + width;
    return end <= 0xffff ? std::optional<u16>{static_cast<u16>(end)} : std::nullopt;
}

// Conservative unsigned bounds recognize the zero carry terms left by native lowering.
u64 UniformWriteU32Maximum(IR::Value value, unsigned depth = 0) {
    value = value.Resolve();
    if (depth > 16 || value.Type() != IR::Type::U32) return 0xffffffffULL;
    if (value.IsImmediate()) return value.U32();
    const IR::Inst* inst = value.Inst();
    if (inst->GetOpcode() == IR::Opcode::BitFieldUExtract &&
        inst->Arg(1).IsImmediate() && inst->Arg(2).IsImmediate()) {
        const u32 shift = inst->Arg(1).U32();
        const u32 bits = inst->Arg(2).U32();
        if (shift >= 32 || bits > 32 - shift) return 0xffffffffULL;
        const u64 mask = (u64{1} << bits) - 1;
        return (std::min)(UniformWriteU32Maximum(inst->Arg(0), depth + 1) >> shift, mask);
    }
    if (inst->GetOpcode() == IR::Opcode::ShiftRightLogical32 &&
        inst->Arg(1).IsImmediate() && inst->Arg(1).U32() < 32) {
        return UniformWriteU32Maximum(inst->Arg(0), depth + 1) >> inst->Arg(1).U32();
    }
    if (inst->GetOpcode() == IR::Opcode::IAdd32) {
        const u64 sum = UniformWriteU32Maximum(inst->Arg(0), depth + 1) +
                        UniformWriteU32Maximum(inst->Arg(1), depth + 1);
        return (std::min)(sum, u64{0xffffffff});
    }
    return 0xffffffffULL;
}

bool UniformWriteHighWord(IR::Value value, const StorageBufferAddr& root,
                          const StorageBufferAddr& offset, const IR::Inst* low_add,
                          u32& terms, unsigned depth = 0) {
    value = value.Resolve();
    if (depth > 16) return false;
    if (UniformWriteU32Maximum(value) == 0) return true;
    u32 term{};
    if (const auto word = StaticCbufWord(value)) {
        if (word->index == root.index && word->offset == root.offset + 4) term = 1;
        else if (word->index == offset.index && word->offset == offset.offset + 4) term = 2;
        else return false;
    } else {
        if (value.IsImmediate()) return false;
        const IR::Inst* inst = value.Inst();
        if (inst->GetOpcode() == IR::Opcode::IAdd32) {
            return UniformWriteHighWord(inst->Arg(0), root, offset, low_add, terms, depth + 1) &&
                   UniformWriteHighWord(inst->Arg(1), root, offset, low_add, terms, depth + 1);
        }
        if (inst->GetOpcode() != IR::Opcode::SelectU32 ||
            !inst->Arg(1).IsImmediate() || inst->Arg(1).U32() != 1 ||
            !inst->Arg(2).IsImmediate() || inst->Arg(2).U32() != 0 ||
            inst->Arg(0).IsImmediate()) return false;
        const IR::Inst* carry = inst->Arg(0).InstRecursive();
        if (carry->GetOpcode() != IR::Opcode::GetCarryFromOp ||
            carry->Arg(0).IsImmediate() || carry->Arg(0).InstRecursive() != low_add) return false;
        term = 4;
    }
    if (terms & term) return false;
    terms |= term;
    return true;
}

// Match root + paired U64 uniform offset, preserving the actual low-word carry and both highs.
std::optional<StorageBufferUniformWrite> UniformStorageWrite(const StorageInst& replacement,
                                                            u64 alignment) {
    if (replacement.pointer_base || replacement.storage_buffer.pointer_offset != ~0U ||
        alignment == 0 || alignment > 0xffff || !std::has_single_bit(alignment))
        return std::nullopt;
    const auto opcode = replacement.inst->GetOpcode();
    if (opcode != IR::Opcode::WriteGlobal32 && opcode != IR::Opcode::GlobalAtomicIAdd32)
        return std::nullopt;
    const auto words = SplitPointerWords(replacement.inst->Arg(0));
    if (!words || (words->displacement != 0 && words->displacement != ~u64{15}) ||
        words->low.IsImmediate()) return std::nullopt;
    const IR::Inst* low_add = words->low.InstRecursive();
    if (low_add->GetOpcode() != IR::Opcode::IAdd32) return std::nullopt;
    const auto first = StaticCbufWord(low_add->Arg(0));
    const auto second = StaticCbufWord(low_add->Arg(1));
    const auto& root = replacement.storage_buffer;
    if (!first || !second || first == second || root.index >= Info::MAX_CBUFS ||
        root.offset > 0xfff8 || !Common::IsAligned(root.offset, 8)) return std::nullopt;
    const auto is_root = [&](const auto& word) {
        return word.index == root.index && word.offset == root.offset;
    };
    const auto offset = is_root(*first) ? second : is_root(*second) ? first : std::nullopt;
    if (!offset || offset->index >= Info::MAX_CBUFS || offset->offset > 0xfff8 ||
        !Common::IsAligned(offset->offset, 8)) return std::nullopt;
    u32 terms{};
    if (!UniformWriteHighWord(words->high, root, *offset, low_add, terms) || terms != 7)
        return std::nullopt;
    return StorageBufferUniformWrite{
        .root_cbuf_index = root.index,
        .root_cbuf_offset = root.offset,
        .offset_cbuf_index = offset->index,
        .offset_cbuf_offset = offset->offset,
        .displacement = words->displacement == 0 ? 0 : -16,
        .width = 4,
    };
}

struct NestedPointer {
    StorageBufferAddr descriptor;
    IR::U32 low_base;
};

std::optional<NestedPointer> TrackNestedPointer(IR::Value address, Environment& env) {
    const auto words = SplitPointerWords(address);
    if (!words || words->displacement > 0x7fffffffu || words->low.IsImmediate() ||
        words->high.IsImmediate()) return std::nullopt;
    const IR::Inst* low = words->low.InstRecursive();
    const IR::Inst* high = words->high.InstRecursive();
    std::optional<StorageBufferAddr> table;
    if (low->GetOpcode() == IR::Opcode::CompositeExtractU32x2 &&
        high->GetOpcode() == IR::Opcode::CompositeExtractU32x2 &&
        low->Arg(1).IsImmediate() && high->Arg(1).IsImmediate() &&
        low->Arg(1).U32() == 0 && high->Arg(1).U32() == 1 &&
        !low->Arg(0).IsImmediate() && !high->Arg(0).IsImmediate() &&
        low->Arg(0).InstRecursive() == high->Arg(0).InstRecursive()) {
        const IR::Inst* load = low->Arg(0).InstRecursive();
        if (load->GetOpcode() == IR::Opcode::LoadGlobal64)
            table = StaticPointerTableAddress(load->Arg(0), env);
    } else if (low->GetOpcode() == IR::Opcode::LoadGlobal32 &&
               high->GetOpcode() == IR::Opcode::LoadGlobal32) {
        const auto low_table = StaticPointerTableAddress(low->Arg(0), env);
        const auto high_table = StaticPointerTableAddress(high->Arg(0), env);
        if (low_table && high_table && low_table->index == high_table->index &&
            low_table->offset == high_table->offset &&
            high_table->pointer_offset == low_table->pointer_offset + 4) table = low_table;
    }
    if (!table || !Common::IsAligned(table->pointer_offset, 8)) return std::nullopt;
    return NestedPointer{*table, IR::U32{words->low}};
}

// Keep dynamic offsets and merged addresses global, but supply their proven
// one-level table pointers to the full-address fallback.
small_vector<StorageBufferAddr, 4> NativeNestedCandidates(const IR::U32& low, Environment& env) {
    small_vector<StorageBufferAddr, 4> result;
    small_vector<const IR::Inst*, 16> pending{low.InstRecursive()};
    std::unordered_set<const IR::Inst*> visited{pending.front()};
    const auto enqueue = [&](IR::Value value) {
        if (!value.IsImmediate() && visited.insert(value.InstRecursive()).second)
            pending.push_back(value.InstRecursive());
    };
    for (size_t next = 0; next < pending.size(); ++next) {
        const IR::Inst* inst = pending[next];
        switch (inst->GetOpcode()) {
        case IR::Opcode::IAdd32:
        case IR::Opcode::Phi:
            for (size_t arg = 0; arg < inst->NumArgs(); ++arg) enqueue(inst->Arg(arg));
            break;
        case IR::Opcode::SelectU32:
            enqueue(inst->Arg(1)); enqueue(inst->Arg(2));
            break;
        case IR::Opcode::CompositeExtractU32x2:
            if (inst->Arg(1).IsImmediate() && inst->Arg(1).U32() == 0 &&
                !inst->Arg(0).IsImmediate()) {
                const IR::Inst* load = inst->Arg(0).InstRecursive();
                if (load->GetOpcode() == IR::Opcode::LoadGlobal64) {
                    const auto table = StaticPointerTableAddress(load->Arg(0), env);
                    if (table && Common::IsAligned(table->pointer_offset, 8) &&
                        std::ranges::find(result, *table) == result.end()) result.push_back(*table);
                }
            }
            break;
        default: break;
        }
    }
    return result;
}

/// Tries to track the storage buffer address used by a global memory instruction
std::optional<StorageBufferAddr> Track(const IR::Value& value, const Bias* bias) {
    const auto pred{[bias](const IR::Inst* inst) -> std::optional<StorageBufferAddr> {
        if (inst->GetOpcode() != IR::Opcode::GetCbufU32 &&
            inst->GetOpcode() != IR::Opcode::GetCbufU32x2) {
            return std::nullopt;
        }
        const IR::Value index{inst->Arg(0)};
        const IR::Value offset{inst->Arg(1)};
        if (!index.IsImmediate()) {
            // Definitely not a storage buffer if it's read from a
            // non-immediate index
            return std::nullopt;
        }
        if (!offset.IsImmediate()) {
            // TODO: Support SSBO arrays
            return std::nullopt;
        }
        const StorageBufferAddr storage_buffer{
            .index = index.U32(),
            .offset = offset.U32(),
        };
        const u32 alignment{bias ? bias->alignment : 8U};
        if (!Common::IsAligned(storage_buffer.offset, alignment)) {
            // The SSBO pointer has to be aligned
            return std::nullopt;
        }
        if (bias && !MeetsBias(storage_buffer, *bias)) {
            // We have to blacklist some addresses in case we wrongly
            // point to them
            return std::nullopt;
        }
        return storage_buffer;
    }};
    // A pointer loaded from memory is a new address, not an offset into the buffer that supplied
    // it. Do not follow that load back to its own constant-buffer descriptor.
    const auto stop_at_memory = [](IR::Value operand) {
        if (!operand.IsImmediate() && IsGlobalMemory(*operand.InstRecursive())) {
            return IR::Value{u32{0}};
        }
        return operand;
    };
    return BreadthFirstSearch(value, pred, stop_at_memory);
}

/// Native programs: every constant buffer pointer the low address can come from, following only
/// additions, selects and phis. A predicated address assignment becomes a phi whose other input
/// can point into a different buffer, so the search does not stop at the first pointer.
small_vector<StorageBufferAddr, 4> NativePointerCandidates(const IR::U32& low_addr,
                                                           Environment& env) {
    small_vector<StorageBufferAddr, 4> candidates;
    small_vector<const IR::Inst*, 16> pending{low_addr.InstRecursive()};
    std::unordered_set<const IR::Inst*> visited{pending.front()};
    const auto enqueue = [&](const IR::Value& value) {
        if (!value.IsImmediate() && visited.insert(value.InstRecursive()).second) {
            pending.push_back(value.InstRecursive());
        }
    };
    for (size_t next = 0; next < pending.size(); ++next) {
        const IR::Inst* const inst{pending[next]};
        switch (inst->GetOpcode()) {
        case IR::Opcode::GetCbufU32:
        case IR::Opcode::GetCbufU32x2: {
            const IR::Value index{inst->Arg(0)};
            const IR::Value offset{inst->Arg(1)};
            if (!index.IsImmediate() || !offset.IsImmediate()) {
                break;
            }
            const StorageBufferAddr pointer{.index = index.U32(), .offset = offset.U32()};
            if (Common::IsAligned(pointer.offset, 8) &&
                std::ranges::find(candidates, pointer) == candidates.end() &&
                env.IsCbufStorageBufferPointer(pointer.index, pointer.offset)) {
                candidates.push_back(pointer);
            }
            break;
        }
        case IR::Opcode::IAdd32:
        case IR::Opcode::Phi:
            for (size_t arg = inst->NumArgs(); arg--;) {
                enqueue(inst->Arg(arg));
            }
            break;
        case IR::Opcode::SelectU32:
            enqueue(inst->Arg(1));
            enqueue(inst->Arg(2));
            break;
        case IR::Opcode::CompositeExtractU32x2:
            if (inst->Arg(1).IsImmediate() && inst->Arg(1).U32() == 0) {
                enqueue(inst->Arg(0));
            }
            break;
        default:
            break;
        }
    }
    return candidates;
}

// Native compute pointer families remain global: the backend selects a binding
// using the complete runtime address, rather than rebasing on one guessed root.
class NativeComputePointers {
public:
    explicit NativeComputePointers(IR::Program& program_) : program{program_} {
        for (const IR::Block* block : program.blocks)
            for (const IR::Inst& inst : block->Instructions())
                owners.emplace(&inst, block);
    }

    bool Collect(IR::Block& block, IR::Inst& inst, StorageInfo& info) {
        const auto words = SplitPointerWords(inst.Arg(0));
        if (!words)
            return false;
        Result result;
        const bool complete = Pair(words->low, words->high, &block, result);
        if (!result.extended && !result.saw_extended_word)
            return false;
        // A mismatched table-pointer pair must not fall back to tracking only its low word.
        if (!complete)
            throw NotImplementedException("Native compute pointer has an unproven reachable pair");
        if (!result.extended)
            return false;
        for (const auto& pointer : result.accesses) {
            if (pointer.pointer_offset != ~0U) {
                // Updating an indirectly selected target requires a separately
                // proved writable-table lifetime contract.
                if (IsGlobalMemoryWrite(inst))
                    throw NotImplementedException("Native compute indirect pointer writes");
                info.set.insert(StorageBufferAddr{pointer.index, pointer.offset});
            }
            info.set.insert(pointer);
            if (IsGlobalMemoryWrite(inst))
                info.writes.insert(pointer);
        }
        info.native_runtime_bounds = true;
        return true;
    }

private:
    using PathCondition = BoundedNativeOffsets::PathCondition;
    static constexpr auto UnrestrictedPath = BoundedNativeOffsets::UnrestrictedPath;

    struct Result {
        StorageBufferSet accesses;
        bool extended{};
        bool saw_extended_word{};
    };

    static bool IsExtendedWord(IR::Value value) {
        if (value.IsImmediate())
            return false;
        const auto* word = value.InstRecursive();
        if (word->GetOpcode() != IR::Opcode::CompositeExtractU32x2 ||
            !word->Arg(1).IsImmediate() || word->Arg(1).U32() > 1 ||
            word->Arg(0).IsImmediate())
            return false;
        const auto* source = word->Arg(0).InstRecursive();
        return source->GetOpcode() == IR::Opcode::LoadGlobal64 ||
               (source->GetOpcode() == IR::Opcode::GetCbufU32x2 &&
                !source->Arg(1).IsImmediate());
    }

    IR::Value Canonical(IR::Value value) {
        value = value.Resolve();
        // Direct static pointers and numeric offset arithmetic need no CFG/BDD
        // analysis. Copy-Phi cycles require the bounded canonicalization.
        if (!value.IsImmediate() && value.InstRecursive()->GetOpcode() == IR::Opcode::Phi)
            return Ranges().Canonical(value);
        return value;
    }

    BoundedNativeOffsets& Ranges() {
        if (!ranges) {
            ranges = std::make_unique<BoundedNativeOffsets>(program);
            // The pass collects instructions in reverse postorder. Prime the
            // immutable offset slice in source block order to bound recursive
            // walks through the long, predicated SSA chains.
            for (const IR::Block* block : program.blocks)
                static_cast<void>(ranges->Offsets(IR::Value{0U}, block));
            for (const IR::Block* block : program.blocks)
                for (const IR::Inst& inst : block->Instructions())
                    if (inst.GetOpcode() == IR::Opcode::GetCbufU32x2 && !inst.Arg(1).IsImmediate())
                        static_cast<void>(ranges->Offsets(inst.Arg(1), block));
        }
        return *ranges;
    }

    void AddCbuf(const IR::Inst& load, const IR::Block* use, StorageBufferSet& out) {
        if (!load.Arg(0).IsImmediate() || load.Arg(0).U32() >= 8)
            throw NotImplementedException("Native compute pointer bank is not bounded");
        std::optional<std::vector<u32>> offsets;
        if (load.Arg(1).IsImmediate())
            offsets = std::vector<u32>{load.Arg(1).U32()};
        else
            offsets = Ranges().Offsets(load.Arg(1), use ? use : owners.at(&load));
        if (!offsets)
            throw NotImplementedException("Native compute pointer offsets are not bounded");
        for (const u32 offset : *offsets) {
            if ((offset & 7) != 0 || offset > 0xfff8)
                throw NotImplementedException(
                    "Native compute pointer slot is outside a constant buffer");
            const StorageBufferAddr pointer{load.Arg(0).U32(), offset};
            // A structural dynamic/table pointer is retained even when the slot
            // happens to contain null at pipeline creation.
            out.insert(pointer);
        }
    }

    bool TableRoots(IR::Value low, IR::Value high, const IR::Block* use, StorageBufferSet& out,
                    unsigned depth = 0, PathCondition path = UnrestrictedPath) {
        if (depth > 128)
            throw NotImplementedException("Native compute pointer pair analysis budget");
        low = Canonical(low);
        high = Canonical(high);
        if (low.IsImmediate() || high.IsImmediate()) {
            return low.IsImmediate() && high.IsImmediate() && low.U32() == 0 && high.U32() == 0;
        }
        const IR::Inst* lo = low.InstRecursive();
        const IR::Inst* hi = high.InstRecursive();
        if (lo->GetOpcode() == IR::Opcode::Phi || hi->GetOpcode() == IR::Opcode::Phi) {
            const IR::Inst* select = lo->GetOpcode() == IR::Opcode::Phi ? lo : hi;
            const bool paired = lo->GetOpcode() == IR::Opcode::Phi &&
                                hi->GetOpcode() == IR::Opcode::Phi &&
                                owners.at(lo) == owners.at(hi);
            for (size_t a = 0; a < select->NumArgs(); ++a) {
                const IR::Block* pred = select->PhiBlock(a);
                const auto incoming =
                    Ranges().ConstrainPathToEdge(path, pred, owners.at(select), use);
                if (!incoming)
                    continue;
                IR::Value l = lo == select ? select->Arg(a) : low;
                IR::Value h = hi == select ? select->Arg(a) : high;
                if (paired) {
                    const IR::Inst* other = lo == select ? hi : lo;
                    bool found{};
                    for (size_t b = 0; b < other->NumArgs(); ++b) {
                        if (other->PhiBlock(b) != pred)
                            continue;
                        if (other == lo)
                            l = other->Arg(b);
                        else
                            h = other->Arg(b);
                        found = true;
                        break;
                    }
                    if (!found)
                        throw LogicError("Mismatched native pointer Phi predecessors");
                }
                if (!TableRoots(l, h, use, out, depth + 1, incoming))
                    return false;
            }
            return true;
        }
        if (lo->GetOpcode() == IR::Opcode::SelectU32 || hi->GetOpcode() == IR::Opcode::SelectU32) {
            const IR::Inst* select = lo->GetOpcode() == IR::Opcode::SelectU32 ? lo : hi;
            const bool paired = lo->GetOpcode() == IR::Opcode::SelectU32 &&
                                hi->GetOpcode() == IR::Opcode::SelectU32 &&
                                lo->Arg(0).Resolve() == hi->Arg(0).Resolve();
            for (size_t a = 1; a != 3; ++a)
                if (!TableRoots(lo == select || paired ? lo->Arg(a) : low,
                                hi == select || paired ? hi->Arg(a) : high, use, out, depth + 1, path))
                    return false;
            return true;
        }
        if (lo->GetOpcode() == IR::Opcode::ShuffleIndex &&
            hi->GetOpcode() == IR::Opcode::ShuffleIndex) {
            for (size_t a = 1; a < lo->NumArgs(); ++a)
                if (lo->Arg(a).Resolve() != hi->Arg(a).Resolve())
                    return false;
            // Source-lane predicates cannot be constrained by the receiving lane.
            return TableRoots(lo->Arg(0), hi->Arg(0), nullptr, out, depth + 1);
        }
        if (lo->GetOpcode() == IR::Opcode::CompositeExtractU32x2 &&
            hi->GetOpcode() == IR::Opcode::CompositeExtractU32x2 && lo->Arg(1).IsImmediate() &&
            hi->Arg(1).IsImmediate() && lo->Arg(1).U32() == 0 && hi->Arg(1).U32() == 1 &&
            !lo->Arg(0).IsImmediate() && !hi->Arg(0).IsImmediate() &&
            lo->Arg(0).InstRecursive() == hi->Arg(0).InstRecursive()) {
            const IR::Inst* source = lo->Arg(0).InstRecursive();
            if (source->GetOpcode() == IR::Opcode::GetCbufU32x2) {
                AddCbuf(*source, use, out);
                return true;
            }
        }
        const auto low_word = StaticCbufWord(low);
        const auto high_word = StaticCbufWord(high);
        if (low_word && high_word && low_word->index == high_word->index && low_word->index < 8 &&
            (low_word->offset & 7) == 0 && low_word->offset <= 0xfff8 &&
            high_word->offset == low_word->offset + 4) {
            out.insert(*low_word);
            return true;
        }
        return false;
    }

    static void Merge(Result& into, const Result& from) {
        into.accesses.insert(from.accesses.begin(), from.accesses.end());
        into.extended |= from.extended;
        into.saw_extended_word |= from.saw_extended_word;
    }

    // Follow both address words together. A numeric low word is not a pointer,
    // and halves selected from different incoming values cannot be combined.
    bool Pair(IR::Value low, IR::Value high, const IR::Block* use, Result& result,
              unsigned depth = 0, PathCondition path = UnrestrictedPath) {
        if (depth > 512)
            throw NotImplementedException("Native compute pointer provenance recursion budget");
        low = Canonical(low);
        high = Canonical(high);
        result.saw_extended_word |= IsExtendedWord(low) || IsExtendedWord(high);
        if (low.IsImmediate() || high.IsImmediate())
            return low.IsImmediate() && high.IsImmediate() && low.U32() == 0 && high.U32() == 0;
        const IR::Inst* lo = low.InstRecursive();
        const IR::Inst* hi = high.InstRecursive();
        const auto key = std::tuple{lo, hi, use, path};
        if (!pair_pending.insert(key).second)
            return false;
        SCOPE_EXIT { pair_pending.erase(key); };
        const auto lop = lo->GetOpcode(), hop = hi->GetOpcode();
        // Added offsets and carry bits are numeric terms. A matching pair must
        // still be found; partial pointer branches cannot be ignored as offsets.
        if (lop == IR::Opcode::IAdd32 || hop == IR::Opcode::IAdd32) {
            bool complete{}, invalid{};
            const size_t nl = lop == IR::Opcode::IAdd32 ? 2 : 1;
            const size_t nh = hop == IR::Opcode::IAdd32 ? 2 : 1;
            for (size_t a = 0; a < nl; ++a)
                for (size_t b = 0; b < nh; ++b) {
                    Result candidate;
                    const bool found = Pair(nl == 2 ? lo->Arg(a) : low, nh == 2 ? hi->Arg(b) : high,
                                            use, candidate, depth + 1, path);
                    invalid |= candidate.extended && !found;
                    complete |= found;
                    Merge(result, candidate);
                }
            return complete && !invalid;
        }
        if (lop == IR::Opcode::Phi || hop == IR::Opcode::Phi) {
            const IR::Inst* select = lop == IR::Opcode::Phi ? lo : hi;
            const bool paired =
                lop == IR::Opcode::Phi && hop == IR::Opcode::Phi && owners.at(lo) == owners.at(hi);
            bool complete = true;
            for (size_t a = 0; a < select->NumArgs(); ++a) {
                const IR::Block* pred = select->PhiBlock(a);
                const auto incoming =
                    Ranges().ConstrainPathToEdge(path, pred, owners.at(select), use);
                if (!incoming)
                    continue;
                IR::Value l = lo == select ? select->Arg(a) : low;
                IR::Value h = hi == select ? select->Arg(a) : high;
                if (paired) {
                    const IR::Inst* other = lo == select ? hi : lo;
                    bool found{};
                    for (size_t b = 0; b < other->NumArgs(); ++b) {
                        if (other->PhiBlock(b) != pred)
                            continue;
                        if (other == lo)
                            l = other->Arg(b);
                        else
                            h = other->Arg(b);
                        found = true;
                        break;
                    }
                    if (!found)
                        throw LogicError("Mismatched native pointer Phi predecessors");
                }
                complete &= Pair(l, h, use, result, depth + 1, incoming);
            }
            return complete;
        }
        if (lop == IR::Opcode::SelectU32 || hop == IR::Opcode::SelectU32) {
            const IR::Inst* select = lop == IR::Opcode::SelectU32 ? lo : hi;
            const bool paired = lop == IR::Opcode::SelectU32 && hop == IR::Opcode::SelectU32 &&
                                lo->Arg(0).Resolve() == hi->Arg(0).Resolve();
            bool complete = true;
            for (size_t a = 1; a != 3; ++a)
                complete &=
                    Pair(lo == select || paired ? lo->Arg(a) : low,
                         hi == select || paired ? hi->Arg(a) : high, use, result, depth + 1, path);
            return complete;
        }
        if (lop == IR::Opcode::ShuffleIndex || hop == IR::Opcode::ShuffleIndex) {
            if (lop != hop)
                return false;
            for (size_t a = 1; a < lo->NumArgs(); ++a)
                if (lo->Arg(a).Resolve() != hi->Arg(a).Resolve())
                    return false;
            Result source;
            const bool complete = Pair(lo->Arg(0), hi->Arg(0), nullptr, source, depth + 1);
            source.extended |= complete && !source.accesses.empty();
            Merge(result, source);
            return complete;
        }
        if (lop == IR::Opcode::CompositeExtractU32x2 && hop == IR::Opcode::CompositeExtractU32x2 &&
            lo->Arg(1).IsImmediate() && hi->Arg(1).IsImmediate() && lo->Arg(1).U32() == 0 &&
            hi->Arg(1).U32() == 1 && !lo->Arg(0).IsImmediate() && !hi->Arg(0).IsImmediate() &&
            lo->Arg(0).InstRecursive() == hi->Arg(0).InstRecursive()) {
            const IR::Inst* source = lo->Arg(0).InstRecursive();
            if (source->GetOpcode() == IR::Opcode::GetCbufU32x2) {
                const bool dynamic = !source->Arg(1).IsImmediate();
                AddCbuf(*source, use, result.accesses);
                result.extended |= dynamic;
                return true;
            }
            if (source->GetOpcode() == IR::Opcode::LoadGlobal64) {
                const auto words = SplitPointerWords(source->Arg(0));
                if (!words || words->displacement > 0xfffffff8 || (words->displacement & 7))
                    throw NotImplementedException(
                        "Native compute pointer table offset is not static");
                StorageBufferSet parents;
                if (!TableRoots(words->low, words->high, owners.at(source), parents, 0, path))
                    throw NotImplementedException(
                        "Native compute pointer table has an unproven reachable root");
                for (auto pointer : parents) {
                    pointer.pointer_offset = static_cast<u32>(words->displacement);
                    result.accesses.insert(pointer);
                }
                result.extended |= !parents.empty();
                return true;
            }
        }
        const auto l = StaticCbufWord(low), h = StaticCbufWord(high);
        if (l && h && l->index == h->index && l->offset + 4 == h->offset && (l->offset & 7) == 0 &&
            l->index < 8 && l->offset <= 0xfff8) {
            result.accesses.insert(*l);
            return true;
        }
        return false;
    }

    IR::Program& program;
    std::unique_ptr<BoundedNativeOffsets> ranges;
    std::unordered_map<const IR::Inst*, const IR::Block*> owners;
    std::set<std::tuple<const IR::Inst*, const IR::Inst*, const IR::Block*, PathCondition>>
        pair_pending;
};


/// Collects the storage buffer used by a global memory instruction and the instruction itself
void CollectStorageBuffers(IR::Block& block, IR::Inst& inst, StorageInfo& info, Environment& env,
                           bool allow_nested) {
    if (allow_nested) {
        if (const auto nested = TrackNestedPointer(inst.Arg(0), env)) {
            const auto& buffer = nested->descriptor;
            // Keep the direct parent even if later DCE removes the pointer-table load.
            info.set.insert(StorageBufferAddr{buffer.index, buffer.offset});
            info.set.insert(buffer);
            info.native_runtime_bounds = true;
            info.to_replace.push_back(StorageInst{buffer, &inst, &block, nested->low_base});
            return;
        }
    }
    // NVN puts storage buffers in a specific range, we have to bias towards these addresses to
    // avoid getting false positives
    static constexpr Bias nvn_bias{
        .index = 0,
        .offset_begin = 0x100,
        .offset_end = 0x700,
        .alignment = 16,
    };
    // Track the low address of the instruction
    const std::optional<LowAddrInfo> low_addr_info{TrackLowAddress(&inst)};
    if (!low_addr_info) {
        // Failed to track the low address, use NVN fallbacks
        return;
    }
    // First try to find storage buffers in the NVN address
    const IR::U32 low_addr{low_addr_info->value};
    if (allow_nested && !low_addr.IsImmediate()) {
        const auto nested = NativeNestedCandidates(low_addr, env);
        if (!nested.empty()) {
            for (const auto& buffer : nested) {
                info.set.insert(StorageBufferAddr{buffer.index, buffer.offset});
                info.set.insert(buffer);
            }
            for (const auto& direct : NativePointerCandidates(low_addr, env)) info.set.insert(direct);
            info.native_runtime_bounds = true;
            return;
        }
    }

    if (env.IsNativeProgram() && !low_addr.IsImmediate()) {
        const IR::Inst* const start{low_addr.InstRecursive()};
        auto it{info.native_candidates.find(start)};
        if (it == info.native_candidates.end()) {
            it = info.native_candidates.emplace(start, NativePointerCandidates(low_addr, env)).first;
        }
        const auto& candidates{it->second};
        if (candidates.size() > 1) {
            // Bind every candidate and keep the access global; the backend picks the buffer at
            // run time. Rebasing on one of them drops the other path's accesses.
            for (const StorageBufferAddr& candidate : candidates) {
                info.set.insert(candidate);
                if (IsGlobalMemoryWrite(inst)) {
                    info.writes.insert(candidate);
                }
            }
            info.native_runtime_bounds = true;
            return;
        }
        if (candidates.size() == 1) {
            // The one pointer the address is built from. The NVN search below only takes 16-byte
            // aligned slots, so for a pointer at an 8-byte slot it can return a dimension or stride
            // the address arithmetic reads instead, and the access then lands in the wrong buffer.
            const StorageBufferAddr storage_buffer{candidates.front()};
            if (IsGlobalMemoryWrite(inst)) {
                info.writes.insert(storage_buffer);
            }
            info.set.insert(storage_buffer);
            info.to_replace.push_back(StorageInst{
                .storage_buffer{storage_buffer},
                .inst = &inst,
                .block = &block,
            });
            return;
        }
    }
    const auto cached_track = [&](const Bias* bias, auto& cache) {
        if (low_addr.IsImmediate()) {
            return Track(low_addr, bias);
        }
        const IR::Inst* const start{low_addr.InstRecursive()};
        if (const auto it = cache.find(start); it != cache.end()) {
            return it->second;
        }
        const std::optional<StorageBufferAddr> result{Track(low_addr, bias)};
        cache.emplace(start, result);
        return result;
    };
    std::optional<StorageBufferAddr> storage_buffer{cached_track(&nvn_bias, info.biased_tracks)};
    if (!storage_buffer) {
        // If it fails, track without a bias
        storage_buffer = cached_track(nullptr, info.unbiased_tracks);
        if (!storage_buffer) {
            // If that also fails, use NVN fallbacks
            LOG_WARNING(Shader, "Storage buffer failed to track, using global memory fallbacks");
            return;
        }
        LOG_DEBUG(Shader, "Storage buffer tracked without bias, index {} offset {}",
                    storage_buffer->index, storage_buffer->offset);
    }
    // Collect storage buffer and the instruction
    if (IsGlobalMemoryWrite(inst)) {
        info.writes.insert(*storage_buffer);
    }
    info.set.insert(*storage_buffer);
    info.to_replace.push_back(StorageInst{
        .storage_buffer{*storage_buffer},
        .inst = &inst,
        .block = &block,
    });
}

/// Returns the offset in indices (not bytes) for an equivalent storage instruction
IR::U32 StorageOffset(IR::Block& block, IR::Inst& inst, StorageBufferAddr buffer, u32 alignment,
                      const std::optional<IR::U32>& pointer_base) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    IR::U32 offset;
    if (const std::optional<LowAddrInfo> low_addr{TrackLowAddress(&inst)}) {
        offset = low_addr->value;
        if (low_addr->imm_offset != 0) {
            offset = ir.IAdd(offset, ir.Imm32(low_addr->imm_offset));
        }
    } else {
        offset = ir.UConvert(32, IR::U64{inst.Arg(0)});
    }
    // Subtract the least significant 32 bits from the guest offset. The result is the storage
    // buffer offset in bytes.
    IR::U32 low_cbuf = pointer_base ? *pointer_base :
        ir.GetCbuf(ir.Imm32(buffer.index), ir.Imm32(buffer.offset));

    // Align the offset base to match the host alignment requirements
    low_cbuf = ir.BitwiseAnd(low_cbuf, ir.Imm32(~(alignment - 1U)));
    return ir.ISub(offset, low_cbuf);
}

/// Replace a global memory load instruction with its storage buffer equivalent
void ReplaceLoad(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                 const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::Value value{&*block.PrependNewInst(it, new_opcode, {storage_index, offset})};
    inst.ReplaceUsesWith(value);
}

/// Replace a global memory write instruction with its storage buffer equivalent
void ReplaceWrite(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                  const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    block.PrependNewInst(it, new_opcode, {storage_index, offset, inst.Arg(1)});
    inst.Invalidate();
}

/// Replace an atomic operation on global memory instruction with its storage buffer equivalent
void ReplaceAtomic(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                   const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::Value value{
        &*block.PrependNewInst(it, new_opcode, {storage_index, offset, inst.Arg(1)})};
    inst.ReplaceUsesWith(value);
}

/// Replace a global memory instruction with its storage buffer equivalent
void Replace(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
             const IR::U32& offset) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadGlobalS8:
    case IR::Opcode::LoadGlobalU8:
    case IR::Opcode::LoadGlobalS16:
    case IR::Opcode::LoadGlobalU16:
    case IR::Opcode::LoadGlobal32:
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::LoadGlobal128:
        return ReplaceLoad(block, inst, storage_index, offset);
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
        return ReplaceWrite(block, inst, storage_index, offset);
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return ReplaceAtomic(block, inst, storage_index, offset);
    default:
        throw InvalidArgument("Invalid global memory opcode {}", inst.GetOpcode());
    }
}
} // Anonymous namespace

bool GlobalMemoryToStorageBufferPass(Environment& env, IR::Program& program,
                                     const HostTranslateInfo& host_info) {
    StorageInfo info;
    const bool extended_compute = env.IsNativeProgram() && program.stage == Stage::Compute &&
        host_info.support_native_compute_pointer_tracking;
    std::optional<NativeComputePointers> compute_pointers;
    if (extended_compute) compute_pointers.emplace(program);
    bool has_extended_compute_pointers{};
    const bool readonly = std::ranges::none_of(program.post_order_blocks, [](const IR::Block* block) {
        return std::ranges::any_of(block->Instructions(), IsGlobalMemoryWrite);
    });
    const bool allow_nested = readonly && env.IsNativeProgram() &&
        (program.stage == Stage::Task || program.stage == Stage::Mesh ||
         program.stage == Stage::Fragment);
    for (IR::Block* const block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (!IsGlobalMemory(inst)) {
                continue;
            }
            if (compute_pointers && compute_pointers->Collect(*block, inst, info)) {
                has_extended_compute_pointers = true;
                continue;
            }
            CollectStorageBuffers(*block, inst, info, env, allow_nested);
        }
    }
    std::map<StorageBufferAddr, u16> fixed_write_ends;
    boost::container::static_vector<StorageBufferUniformWrite, 32> uniform_write_proofs;
    bool writes_bounded = true;
    if (!readonly && env.IsNativeProgram() &&
        host_info.support_graphics_readonly_tables_with_writes) {
        for (const IR::Block* block : program.post_order_blocks) {
            for (const IR::Inst& inst : block->Instructions()) {
                if (!IsGlobalMemoryWrite(inst)) continue;
                const auto replacement = std::ranges::find_if(info.to_replace,
                    [&](const StorageInst& item) { return item.inst == &inst; });
                const auto end = replacement == info.to_replace.end() ? std::nullopt
                    : FixedStorageWriteEnd(*replacement, host_info.min_ssbo_alignment);
                if (end) {
                    auto& maximum = fixed_write_ends[replacement->storage_buffer];
                    maximum = (std::max)(maximum, *end);
                    continue;
                }
                const auto uniform = replacement != info.to_replace.end() &&
                        (program.stage == Stage::Task || program.stage == Stage::Mesh)
                    ? UniformStorageWrite(*replacement, host_info.min_ssbo_alignment)
                    : std::nullopt;
                if (!uniform) {
                    writes_bounded = false;
                } else if (std::ranges::find(uniform_write_proofs, *uniform) == uniform_write_proofs.end()) {
                    if (uniform_write_proofs.size() == uniform_write_proofs.capacity())
                        writes_bounded = false;
                    else
                        uniform_write_proofs.push_back(*uniform);
                }
            }
        }
        if (!writes_bounded) {
            fixed_write_ends.clear();
            uniform_write_proofs.clear();
        }
    }
    if (!readonly && writes_bounded && env.IsNativeProgram() &&
        (program.stage == Stage::Task || program.stage == Stage::Mesh) &&
        host_info.support_graphics_readonly_tables_with_writes) {
        if (writes_bounded) {
            for (IR::Block* block : program.post_order_blocks) {
                for (IR::Inst& inst : block->Instructions()) {
                    if (!IsGlobalMemory(inst) || IsGlobalMemoryWrite(inst)) continue;
                    small_vector<StorageBufferAddr, 4> children;
                    if (const auto nested = TrackNestedPointer(inst.Arg(0), env)) {
                        children.push_back(nested->descriptor);
                    } else if (const auto low = TrackLowAddress(&inst);
                               low && !low->value.IsImmediate()) {
                        children = NativeNestedCandidates(low->value, env);
                    }
                    if (children.empty() || std::ranges::any_of(children, [&](const auto& child) {
                            return info.writes.contains(StorageBufferAddr{child.index, child.offset});
                        })) continue;
                    info.to_replace.erase(std::remove_if(info.to_replace.begin(),
                        info.to_replace.end(),
                        [&](const StorageInst& item) { return item.inst == &inst; }),
                        info.to_replace.end());
                    CollectStorageBuffers(*block, inst, info, env, true);
                }
            }
        }
    }
    // A new indirect family must not overflow the fixed descriptor capacity.
    // Leave those accesses on the existing global fallback when it cannot fit.
    const u32 descriptor_limit = has_extended_compute_pointers
        ? std::min<u32>(Info::MAX_SSBOS, host_info.max_compute_storage_buffers)
        : Info::MAX_GRAPHICS_SSBOS;
    if (has_extended_compute_pointers && info.set.size() > descriptor_limit)
        throw NotImplementedException("Native compute pointer families require {} storage buffers, host limit {}",
                                      info.set.size(), descriptor_limit);
    if (!has_extended_compute_pointers && info.set.size() > descriptor_limit) {
        for (auto it = info.set.begin(); it != info.set.end();) {
            if (it->pointer_offset != ~0U) it = info.set.erase(it);
            else ++it;
        }
        info.to_replace.erase(std::remove_if(info.to_replace.begin(), info.to_replace.end(),
            [](const StorageInst& inst) { return inst.pointer_base.has_value(); }),
            info.to_replace.end());
    }
    if (info.set.size() > descriptor_limit)
        throw NotImplementedException("Storage buffer descriptor capacity exceeded");
    if (extended_compute) {
        for (const auto& pointer : info.set) {
            if (pointer.pointer_offset != ~0U &&
                info.writes.contains(StorageBufferAddr{pointer.index, pointer.offset}))
                throw NotImplementedException("Native compute writes pointer table cbuf{} offset{:#x}", pointer.index, pointer.offset);
        }
    }
    program.info.native_global_bounds |= info.native_runtime_bounds;
    for (const StorageBufferAddr& storage_buffer : info.set) {
        program.info.storage_buffers_descriptors.push_back({
            .cbuf_index = storage_buffer.index,
            .cbuf_offset = storage_buffer.offset,
            .count = 1,
            .is_written = info.writes.contains(storage_buffer),
            .has_uniform_write_proofs = std::ranges::any_of(uniform_write_proofs,
                [&](const auto& proof) {
                    return proof.root_cbuf_index == storage_buffer.index &&
                           proof.root_cbuf_offset == storage_buffer.offset &&
                           storage_buffer.pointer_offset == ~0U;
                }),
            .fixed_write_end = fixed_write_ends.contains(storage_buffer)
                ? fixed_write_ends.at(storage_buffer) : u16{0},
            .pointer_offset = storage_buffer.pointer_offset,
        });
    }
    program.info.uniform_storage_write_proofs = uniform_write_proofs;
    for (const StorageInst& storage_inst : info.to_replace) {
        const StorageBufferAddr storage_buffer{storage_inst.storage_buffer};
        const auto it{info.set.find(storage_inst.storage_buffer)};
        const IR::U32 index{IR::Value{static_cast<u32>(info.set.index_of(it))}};
        IR::Block* const block{storage_inst.block};
        IR::Inst* const inst{storage_inst.inst};
        const IR::U32 offset{
            StorageOffset(*block, *inst, storage_buffer, u32(host_info.min_ssbo_alignment),
                          storage_inst.pointer_base)};
        Replace(*block, *inst, index, offset);
    }
    return has_extended_compute_pointers;
}

void JoinStorageInfo(Info& base, Info& source) {
    base.native_global_bounds |= source.native_global_bounds;
    auto& descriptors = base.storage_buffers_descriptors;
    for (auto& desc : source.storage_buffers_descriptors) {
        auto it{std::ranges::find_if(descriptors, [&desc](const auto& existing) {
            return desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset && desc.count == existing.count &&
                   desc.pointer_offset == existing.pointer_offset;
        })};
        if (it != descriptors.end()) {
            if (desc.is_written) {
                const bool covered = !it->is_written ||
                    it->fixed_write_end || it->has_uniform_write_proofs;
                if (covered && (desc.fixed_write_end || desc.has_uniform_write_proofs)) {
                    it->fixed_write_end = (std::max)(it->fixed_write_end, desc.fixed_write_end);
                    it->has_uniform_write_proofs |= desc.has_uniform_write_proofs;
                } else {
                    it->fixed_write_end = 0;
                    it->has_uniform_write_proofs = false;
                }
            }
            it->is_written |= desc.is_written;
            continue;
        }
        descriptors.push_back(desc);
    }
    auto& proofs = base.uniform_storage_write_proofs;
    bool capacity_exceeded{};
    for (const auto& proof : source.uniform_storage_write_proofs) {
        if (std::ranges::find(proofs, proof) != proofs.end()) continue;
        if (proofs.size() == proofs.capacity()) {
            capacity_exceeded = true;
            break;
        }
        proofs.push_back(proof);
    }
    if (capacity_exceeded) {
        proofs.clear();
        for (auto& desc : descriptors) {
            if (!desc.has_uniform_write_proofs) continue;
            desc.has_uniform_write_proofs = false;
            desc.fixed_write_end = 0;
        }
    } else {
        proofs.erase(std::remove_if(proofs.begin(), proofs.end(), [&](const auto& proof) {
            return std::ranges::none_of(descriptors, [&](const auto& desc) {
                return desc.is_written && desc.has_uniform_write_proofs &&
                       desc.pointer_offset == ~0U && desc.cbuf_index == proof.root_cbuf_index &&
                       desc.cbuf_offset == proof.root_cbuf_offset;
            });
        }), proofs.end());
    }
}

} // namespace Shader::Optimization
