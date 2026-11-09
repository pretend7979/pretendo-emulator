// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
class FoldEnvironment final : public Shader::Environment {
public:
    bool IsNativeProgram() const override { return true; }
    bool IsCbufStorageBufferPointer(u32 bank, u32 offset) override {
        return bank == 0 && offset == 0x20;
    }
    u64 ReadInstruction(u32) override { return 0; }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    Shader::TextureType ReadTextureType(u32) override { return {}; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override { return {}; }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 0; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return {}; }
    void Dump(u64, u64) override {}
};

struct FoldFixture {
    Shader::ObjectPool<Shader::IR::Inst> pool;
    Shader::IR::Block block{pool};
    Shader::IR::IREmitter ir{block};
    Shader::IR::Program program;
    FoldEnvironment env;

    FoldFixture() {
        program.stage = Shader::Stage::Compute;
        program.post_order_blocks.push_back(&block);
    }
    void Fold() { Shader::Optimization::ConstantPropagationPass(env, program); }
};

u32 Evaluate(Shader::IR::Value value, u32 first, u32 second) {
    using namespace Shader::IR;
    value = value.Resolve();
    if (value.IsImmediate()) return value.U32();
    const auto* inst = value.Inst();
    const auto arg = [&](size_t index) { return Evaluate(inst->Arg(index), first, second); };
    switch (inst->GetOpcode()) {
    case Opcode::GetCbufU32: return inst->Arg(0).U32() == 0 ? first : second;
    case Opcode::IMul32: return arg(0) * arg(1);
    case Opcode::IAdd32: return arg(0) + arg(1);
    case Opcode::ShiftLeftLogical32: return arg(0) << arg(1);
    case Opcode::BitFieldUExtract: return (arg(0) >> arg(1)) & ((u32{1} << arg(2)) - 1);
    default: throw std::runtime_error("Unexpected opcode in integer-fold regression");
    }
}
} // namespace

TEST_CASE("Integer multiply identities preserve unsigned arithmetic and associated flags",
          "[shader_recompiler][constant_propagation]") {
    using namespace Shader;
    for (const bool reverse : {false, true}) {
        for (const u32 factor : {0U, 1U, 2U, 0xffffffffU}) {
            CAPTURE(reverse, factor);
            FoldFixture f;
            const auto input = f.ir.GetCbuf(f.ir.Imm32(0), f.ir.Imm32(0));
            const auto result = reverse ? f.ir.IMul(f.ir.Imm32(factor), input)
                                        : f.ir.IMul(input, f.ir.Imm32(factor));
            f.Fold();
            if (factor == 0) {
                REQUIRE(result.Resolve().IsImmediate());
                CHECK(result.Resolve().U32() == 0);
            } else if (factor == 1) {
                CHECK(result.Resolve() == input.Resolve());
            } else {
                CHECK(result.Resolve().Inst()->GetOpcode() == IR::Opcode::IMul32);
            }
            for (const u32 word : {0U, 1U, 0xffffU, 0x10000U, 0x7fffffffU, 0x80000000U,
                                   0xabcdef01U, 0xffffffffU}) {
                CHECK(Evaluate(result, word, 0) == word * factor);
            }
        }
    }
    for (const bool sign : {false, true}) {
        for (const u32 factor : {0U, 1U}) {
            FoldFixture f;
            const auto input = f.ir.GetCbuf(f.ir.Imm32(0), f.ir.Imm32(0));
            const auto result = f.ir.IMul(input, f.ir.Imm32(factor));
            const auto flag = sign ? f.ir.GetSignFromOp(result) : f.ir.GetZeroFromOp(result);
            f.Fold();
            CHECK(result.Resolve().Inst()->GetOpcode() == IR::Opcode::IMul32);
            CHECK(flag.Inst()->Arg(0).Resolve() == result.Resolve());
        }
    }
    FoldFixture f;
    const auto wrapped = f.ir.IMul(f.ir.Imm32(0x80000001U), f.ir.Imm32(2));
    f.Fold();
    CHECK(wrapped.Resolve().U32() == 2);
}

TEST_CASE("Unsigned half recombination folds only identical disjoint halves",
          "[shader_recompiler][constant_propagation]") {
    using namespace Shader;
    for (const bool reverse : {false, true}) {
        for (u32 variant = 0; variant < 7; ++variant) {
            CAPTURE(reverse, variant);
            FoldFixture f;
            const auto first = f.ir.GetCbuf(f.ir.Imm32(0), f.ir.Imm32(0));
            const auto second = f.ir.GetCbuf(f.ir.Imm32(1), f.ir.Imm32(0));
            const auto low = f.ir.BitFieldExtract(first, f.ir.Imm32(0),
                                                 f.ir.Imm32(variant == 2 ? 15 : 16), false);
            const auto high = f.ir.BitFieldExtract(variant == 1 ? second : first,
                f.ir.Imm32(variant == 3 ? 15 : 16), f.ir.Imm32(variant == 4 ? 15 : 16),
                variant == 5);
            const auto shifted = f.ir.ShiftLeftLogical(high, f.ir.Imm32(variant == 6 ? 15 : 16));
            const auto result = reverse ? f.ir.IAdd(low, shifted) : f.ir.IAdd(shifted, low);
            f.Fold();
            if (variant == 0) {
                CHECK(result.Resolve() == first.Resolve());
                // Exercise every halfword both as the high and low half, plus deterministic
                // mixed words. Evaluation follows the actual optimized IR.
                u32 random = 0x55aa1122U;
                for (u32 word = 0; word < 65536; ++word) {
                    random = random * 1664525U + 1013904223U;
                    REQUIRE(Evaluate(result, word, 0) == word);
                    REQUIRE(Evaluate(result, word << 16, 0) == word << 16);
                    REQUIRE(Evaluate(result, random, 0) == random);
                }
            } else {
                CHECK(result.Resolve() != first.Resolve());
                CHECK(result.Resolve().Inst()->GetOpcode() == IR::Opcode::IAdd32);
            }
        }
    }
    FoldFixture f;
    const auto input = f.ir.GetCbuf(f.ir.Imm32(0), f.ir.Imm32(0));
    const auto low = f.ir.BitFieldExtract(input, f.ir.Imm32(0), f.ir.Imm32(16), false);
    const auto high = f.ir.BitFieldExtract(input, f.ir.Imm32(16), f.ir.Imm32(16), false);
    const auto sum = f.ir.IAdd(f.ir.ShiftLeftLogical(high, f.ir.Imm32(16)), low);
    const auto carry = f.ir.GetCarryFromOp(sum);
    f.Fold();
    CHECK(sum.Resolve().Inst()->GetOpcode() == IR::Opcode::IAdd32);
    CHECK(carry.Inst()->Arg(0).Resolve() == sum.Resolve());
}

TEST_CASE("XMAD multiplication by one exposes the native atomic destination pointer",
          "[shader_recompiler][constant_propagation][global_memory]") {
    using namespace Shader;
    for (const bool reverse : {false, true}) {
        FoldFixture f;
        auto& ir = f.ir;
        const auto pointer = ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x20));
        const auto high_pointer = ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x24));
        const auto low = ir.BitFieldExtract(pointer, ir.Imm32(0), ir.Imm32(16), false);
        const auto high = ir.BitFieldExtract(pointer, ir.Imm32(16), ir.Imm32(16), false);
        const auto low_product = ir.IMul(low, ir.Imm32(1));
        const auto cross = ir.IAdd(ir.IMul(low, ir.Imm32(0)), ir.IMul(high, ir.Imm32(1)));
        const auto shifted = ir.ShiftLeftLogical(cross, ir.Imm32(16));
        const IR::U32 product{reverse ? ir.IAdd(low_product, shifted) : ir.IAdd(shifted, low_product)};
        const auto offset = ir.GetCbuf(ir.Imm32(1), ir.Imm32(0));
        const auto address = ir.PackUint2x32(ir.CompositeConstruct(ir.IAdd(product, offset), high_pointer));
        const auto atomic = ir.GlobalAtomicIAdd(address, ir.Imm32(1));
        f.Fold();
        HostTranslateInfo host{};
        host.min_ssbo_alignment = 16;
        Optimization::GlobalMemoryToStorageBufferPass(f.env, f.program, host);
        REQUIRE(f.program.info.storage_buffers_descriptors.size() == 1);
        const auto& descriptor = f.program.info.storage_buffers_descriptors.front();
        CHECK(descriptor.cbuf_index == 0);
        CHECK(descriptor.cbuf_offset == 0x20);
        CHECK(descriptor.is_written);
        CHECK(atomic.Resolve().Inst()->GetOpcode() == IR::Opcode::StorageAtomicIAdd32);
        const auto* byte_offset = atomic.Resolve().Inst()->Arg(1).InstRecursive();
        REQUIRE(byte_offset->GetOpcode() == IR::Opcode::ISub32);
        const auto* base = byte_offset->Arg(1).InstRecursive();
        REQUIRE(base->GetOpcode() == IR::Opcode::BitwiseAnd32);
        const auto* cbuf = base->Arg(0).InstRecursive();
        CHECK(cbuf->Arg(0).U32() == 0);
        CHECK(cbuf->Arg(1).U32() == 0x20);
    }
}
