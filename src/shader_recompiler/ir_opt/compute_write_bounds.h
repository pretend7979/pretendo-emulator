// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <optional>

#include "shader_recompiler/compute_write_bounds.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"

namespace Shader::Optimization {
namespace ComputeWriteBoundsDetail {

struct Affine {
    std::array<u32, 6> coefficients{};
    u32 constant{};
    u32 roots{};
};

inline std::optional<IR::Value> Resolve(IR::Value value, u32& budget) {
    for (;;) {
        if (!budget) return std::nullopt;
        --budget;
        if (!value.IsIdentity()) return value;
        value = value.Inst()->Arg(0);
    }
}

inline std::optional<u32> Immediate(IR::Value value, u32& budget) {
    const auto resolved = Resolve(value, budget);
    if (!resolved || !resolved->IsImmediate() || resolved->Type() != IR::Type::U32)
        return std::nullopt;
    return resolved->U32();
}

inline bool Root(IR::Value value, const StorageBufferDescriptor& descriptor, u32& budget) {
    auto resolved = Resolve(value, budget);
    if (!resolved || resolved->IsImmediate()) return false;
    const IR::Inst* inst = resolved->Inst();
    if (inst->GetOpcode() == IR::Opcode::CompositeExtractU32x2) {
        const auto component = Immediate(inst->Arg(1), budget);
        resolved = Resolve(inst->Arg(0), budget);
        if (!component || *component != 0 || !resolved || resolved->IsImmediate()) return false;
        inst = resolved->Inst();
        if (inst->GetOpcode() != IR::Opcode::GetCbufU32x2) return false;
    } else if (inst->GetOpcode() != IR::Opcode::GetCbufU32) {
        return false;
    }
    const auto index = Immediate(inst->Arg(0), budget);
    const auto offset = Immediate(inst->Arg(1), budget);
    return index && offset && *index == descriptor.cbuf_index && *offset == descriptor.cbuf_offset;
}

inline std::optional<Affine> Add(Affine a, const Affine& b) {
    if (a.roots + b.roots > 1 || u64{a.constant} + b.constant > 0xffffffffULL)
        return std::nullopt;
    a.roots += b.roots;
    a.constant += b.constant;
    for (size_t i = 0; i < a.coefficients.size(); ++i) {
        if (u64{a.coefficients[i]} + b.coefficients[i] > 0xffffffffULL) return std::nullopt;
        a.coefficients[i] += b.coefficients[i];
    }
    return a;
}

inline std::optional<Affine> Scale(Affine a, u32 factor) {
    // Do not erase an intermediate range with multiplication by zero.
    if (!factor || (a.roots && factor != 1) || u64{a.constant} * factor > 0xffffffffULL)
        return std::nullopt;
    a.constant *= factor;
    for (auto& coefficient : a.coefficients) {
        if (u64{coefficient} * factor > 0xffffffffULL) return std::nullopt;
        coefficient *= factor;
    }
    return a;
}


struct HalfProduct {
    IR::Value source;
    u32 factor;
};

// Match a single unsigned half multiplied by an immediate. Merely finding the
// same Phi or Select below both halves does not prove an affine source.
inline std::optional<HalfProduct> MatchHalfProduct(IR::Value value, u32 bit_offset,
                                                 u32& budget) {
    const auto product = Resolve(value, budget);
    if (!product || product->IsImmediate() ||
        product->Inst()->GetOpcode() != IR::Opcode::IMul32) return std::nullopt;
    for (size_t side = 0; side < 2; ++side) {
        const auto factor = Immediate(product->Inst()->Arg(side), budget);
        if (!factor || !*factor) continue;
        const auto half = Resolve(product->Inst()->Arg(1 - side), budget);
        if (!half || half->IsImmediate() ||
            half->Inst()->GetOpcode() != IR::Opcode::BitFieldUExtract) continue;
        const IR::Inst* extract = half->Inst();
        const auto offset = Immediate(extract->Arg(1), budget);
        const auto count = Immediate(extract->Arg(2), budget);
        const auto input = Resolve(extract->Arg(0), budget);
        if (offset && *offset == bit_offset && count && *count == 16 && input &&
            input->Type() == IR::Type::U32) return HalfProduct{*input, *factor};
    }
    return std::nullopt;
}

struct HalvesRecombine {
    IR::Value source;
    u32 factor;
    u32 addend;
};

// ((hi16(x) * k) << 16) + (lo16(x) * k + c) equals x*k+c modulo 2^32.
// The caller must still prove x affine, reject duplicated roots, and check the
// complete dispatch maximum for overflow. This does not rewrite shader IR.
inline std::optional<HalvesRecombine> MatchHalvesRecombine(const IR::Inst& add,
                                                         u32& budget) {
    for (size_t side = 0; side < 2; ++side) {
        const auto shifted = Resolve(add.Arg(side), budget);
        if (!shifted || shifted->IsImmediate() ||
            shifted->Inst()->GetOpcode() != IR::Opcode::ShiftLeftLogical32) continue;
        const auto shift = Immediate(shifted->Inst()->Arg(1), budget);
        if (!shift || *shift != 16) continue;
        const auto high = MatchHalfProduct(shifted->Inst()->Arg(0), 16, budget);
        if (!high) continue;
        auto low_value = Resolve(add.Arg(1 - side), budget);
        if (!low_value) continue;
        u32 addend{};
        if (!low_value->IsImmediate() && low_value->Inst()->GetOpcode() == IR::Opcode::IAdd32) {
            const IR::Inst* low_add = low_value->Inst();
            bool found = false;
            for (size_t term = 0; term < 2; ++term) {
                const auto constant = Immediate(low_add->Arg(term), budget);
                if (!constant) continue;
                low_value = Resolve(low_add->Arg(1 - term), budget);
                if (!low_value) return std::nullopt;
                addend = *constant;
                found = true;
                break;
            }
            if (!found) continue;
        }
        const auto low = MatchHalfProduct(*low_value, 0, budget);
        if (low && low->source == high->source && low->factor == high->factor)
            return HalvesRecombine{low->source, low->factor, addend};
    }
    return std::nullopt;
}

inline std::optional<Affine> Parse(IR::Value value, const StorageBufferDescriptor& descriptor,
                                    u32& budget, u32 depth = 0) {
    if (depth > 64) return std::nullopt;
    const auto resolved = Resolve(value, budget);
    if (!resolved || resolved->Type() != IR::Type::U32) return std::nullopt;
    if (resolved->IsImmediate()) return Affine{.constant = resolved->U32()};
    if (Root(*resolved, descriptor, budget)) return Affine{.roots = 1};
    const IR::Inst* inst = resolved->Inst();
    switch (inst->GetOpcode()) {
    case IR::Opcode::CompositeExtractU32x3: {
        const auto component = Immediate(inst->Arg(1), budget);
        const auto vector = Resolve(inst->Arg(0), budget);
        if (!component || *component >= 3 || !vector || vector->IsImmediate()) return std::nullopt;
        const auto opcode = vector->Inst()->GetOpcode();
        if (opcode != IR::Opcode::WorkgroupId && opcode != IR::Opcode::LocalInvocationId)
            return std::nullopt;
        Affine result;
        result.coefficients[*component + (opcode == IR::Opcode::LocalInvocationId ? 3 : 0)] = 1;
        return result;
    }
    case IR::Opcode::IAdd32: {
        if (const auto recombined = MatchHalvesRecombine(*inst, budget)) {
            const auto input = Parse(recombined->source, descriptor, budget, depth + 1);
            const auto scaled = input ? Scale(*input, recombined->factor) : std::nullopt;
            return scaled ? Add(*scaled, Affine{.constant = recombined->addend}) : std::nullopt;
        }
        const auto a = Parse(inst->Arg(0), descriptor, budget, depth + 1);
        const auto b = Parse(inst->Arg(1), descriptor, budget, depth + 1);
        return a && b ? Add(*a, *b) : std::nullopt;
    }
    case IR::Opcode::ShiftLeftLogical32: {
        const auto shift = Immediate(inst->Arg(1), budget);
        if (!shift || *shift >= 32) return std::nullopt;
        const auto a = Parse(inst->Arg(0), descriptor, budget, depth + 1);
        return a ? Scale(*a, u32{1} << *shift) : std::nullopt;
    }
    case IR::Opcode::IMul32:
        for (size_t side = 0; side < 2; ++side) {
            const auto factor = Immediate(inst->Arg(side), budget);
            if (!factor) continue;
            const auto a = Parse(inst->Arg(1 - side), descriptor, budget, depth + 1);
            return a ? Scale(*a, *factor) : std::nullopt;
        }
        return std::nullopt;
    default:
        return std::nullopt;
    }
}

inline std::optional<Affine> Offset(IR::Value value, const StorageBufferDescriptor& descriptor,
                                   u32 alignment, u32 width) {
    u32 budget = 2048;
    const auto offset = Resolve(value, budget);
    if (!offset || offset->IsImmediate() || offset->Inst()->GetOpcode() != IR::Opcode::ISub32)
        return std::nullopt;
    const IR::Inst* sub = offset->Inst();
    auto base = Resolve(sub->Arg(1), budget);
    if (!base || base->IsImmediate()) return std::nullopt;
    if (base->Inst()->GetOpcode() == IR::Opcode::BitwiseAnd32) {
        const IR::Inst* mask = base->Inst();
        bool matched = false;
        for (size_t side = 0; side < 2; ++side) {
            const auto bits = Immediate(mask->Arg(side), budget);
            if (bits && *bits == ~(alignment - 1) && Root(mask->Arg(1 - side), descriptor, budget)) {
                matched = true;
                break;
            }
        }
        if (!matched) return std::nullopt;
    } else if (alignment != 1 || !Root(*base, descriptor, budget)) {
        return std::nullopt;
    }
    auto result = Parse(sub->Arg(0), descriptor, budget);
    if (!result || result->roots != 1) return std::nullopt;
    const u64 end = u64{result->constant} + alignment - 1 + width;
    if (end > 0xffffffffULL) return std::nullopt;
    result->constant = static_cast<u32>(end);
    return result;
}

} // namespace ComputeWriteBoundsDetail

// Analyze actual translated storage offsets, not an intended native instruction pattern.
// Unknown writes invalidate the whole result; no read-binding extent is changed.
inline ComputeWriteBounds AnalyzeComputeWriteBounds(const IR::Program& program, u32 alignment) {
    ComputeWriteBounds result;
    if (program.stage != Stage::Compute || program.info.stores_global_memory ||
        !std::has_single_bit(alignment)) return result;
    // native_global_bounds can describe only indirect reads. The complete emitted
    // write scan below still rejects every unresolved/global/atomic write; read
    // routing and full descriptor binding extents remain unchanged.
    result.workgroup_size = program.workgroup_size;
    // Match the emitter, including blocks retained in the syntax list after CFG pruning.
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        if (node.type != IR::AbstractSyntaxNode::Type::Block) continue;
        const IR::Block* block = node.data.block;
        if (!block) return {};
        for (const IR::Inst& inst : block->Instructions()) {
            const auto name = IR::NameOf(inst.GetOpcode());
            if (name.starts_with("WriteGlobal") || name.starts_with("GlobalAtomic") ||
                name.starts_with("StorageAtomic") || name.find("ImageWrite") != std::string_view::npos ||
                name.find("ImageAtomic") != std::string_view::npos) return {};
            if (!name.starts_with("WriteStorage")) continue;
            u32 width{};
            switch (inst.GetOpcode()) {
            case IR::Opcode::WriteStorage32: width = 4; break;
            case IR::Opcode::WriteStorage64: width = 8; break;
            case IR::Opcode::WriteStorage128: width = 16; break;
            default: return {};
            }
            u32 budget = 64;
            const auto index = ComputeWriteBoundsDetail::Immediate(inst.Arg(0), budget);
            if (!index || *index >= program.info.storage_buffers_descriptors.size()) return {};
            const auto& descriptor = program.info.storage_buffers_descriptors[*index];
            if (!descriptor.is_written || descriptor.count != 1 || descriptor.pointer_offset != ~0U ||
                descriptor.cbuf_index >= Info::MAX_CBUFS || descriptor.cbuf_offset > 0xfff8 ||
                descriptor.cbuf_offset % 8 != 0) return {};
            const auto proof = ComputeWriteBoundsDetail::Offset(inst.Arg(1), descriptor, alignment, width);
            if (!proof) return {};
            auto entry = std::ranges::find(result.storage, *index, &ComputeWriteBound::index);
            if (entry == result.storage.end()) {
                result.storage.push_back(ComputeWriteBound{.index = *index,
                    .cbuf_index = descriptor.cbuf_index, .cbuf_offset = descriptor.cbuf_offset,
                    .coefficients = proof->coefficients, .constant_end = proof->constant,
                    .alignment = alignment});
            } else {
                entry->constant_end = (std::max)(entry->constant_end, proof->constant);
                for (size_t i = 0; i < entry->coefficients.size(); ++i)
                    entry->coefficients[i] = (std::max)(entry->coefficients[i], proof->coefficients[i]);
            }
        }
    }
    return result;
}

} // namespace Shader::Optimization
