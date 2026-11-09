// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
class StubEnvironment final : public Shader::Environment {
public:
    bool native{};
    bool IsNativeProgram() const override { return native; }
    bool IsCbufStorageBufferPointer(u32 bank, u32 offset) override {
        return native && bank == 0 && (offset == 0x780 || offset == 0x790);
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
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};
} // namespace

TEST_CASE("Global address tracking stops at a memory load", "[shader_recompiler][global_memory]") {
    using namespace Shader;
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::IREmitter ir{block};
    const auto low = ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x110));
    const auto high = ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x114));
    const auto source = ir.PackUint2x32(ir.CompositeConstruct(low, high));
    const auto loaded = ir.LoadGlobal64(source);
    const auto destination = ir.PackUint2x32(ir.CompositeConstruct(
        ir.CompositeExtract(loaded, 0), ir.CompositeExtract(loaded, 1)));
    ir.WriteGlobal32(destination, ir.Imm32(1));

    IR::Program program;
    program.post_order_blocks.push_back(&block);
    HostTranslateInfo host{};
    host.min_ssbo_alignment = 16;
    StubEnvironment env;
    Optimization::GlobalMemoryToStorageBufferPass(env, program, host);

    const auto has = [&](IR::Opcode opcode) {
        return std::ranges::any_of(block.Instructions(), [=](const IR::Inst& inst) {
            return inst.GetOpcode() == opcode;
        });
    };
    REQUIRE(has(IR::Opcode::LoadStorage64));
    REQUIRE(has(IR::Opcode::WriteGlobal32));
    REQUIRE_FALSE(has(IR::Opcode::WriteStorage32));
}

TEST_CASE("Native mesh pointers loaded from fixed table entries use distinct storage buffers",
          "[shader_recompiler][global_memory]") {
    using namespace Shader;
    for (const Stage stage : {Stage::Task, Stage::Mesh}) {
        for (const bool split_load : {false, true}) {
            for (const u32 table_offset : {0U, 8U, 24U, 32U}) {
                CAPTURE(stage, split_load, table_offset);
                ObjectPool<IR::Inst> pool;
                IR::Block block{pool};
                IR::IREmitter ir{block};
                const auto root = ir.PackUint2x32(ir.CompositeConstruct(
                    ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x780)),
                    ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x784))));
                const IR::U64 table{ir.IAdd(root, ir.Imm64(u64{table_offset}))};
                IR::U32 low;
                IR::U32 high;
                if (split_load) {
                    low = ir.LoadGlobal32(table);
                    high = ir.LoadGlobal32(IR::U64{ir.IAdd(root, ir.Imm64(u64{table_offset} + 4))});
                } else {
                    const auto pointer = ir.LoadGlobal64(table);
                    low = IR::U32{ir.CompositeExtract(pointer, 0)};
                    high = IR::U32{ir.CompositeExtract(pointer, 1)};
                }
                const auto child = ir.PackUint2x32(ir.CompositeConstruct(low, high));
                const auto access = ir.LoadGlobal32(IR::U64{ir.IAdd(child, ir.Imm64(u64{32}))});
                IR::Program program;
                program.stage = stage;
                program.post_order_blocks.push_back(&block);
                HostTranslateInfo host{};
                host.min_ssbo_alignment = 16;
                StubEnvironment env;
                env.native = true;
                Optimization::GlobalMemoryToStorageBufferPass(env, program, host);

                const auto* load = access.Resolve().Inst();
                REQUIRE(load->GetOpcode() == IR::Opcode::LoadStorage32);
                REQUIRE(program.info.storage_buffers_descriptors.size() == 2);
                const auto& descriptor = program.info.storage_buffers_descriptors.at(load->Arg(0).U32());
                REQUIRE(descriptor.cbuf_index == 0);
                REQUIRE(descriptor.cbuf_offset == 0x780);
                REQUIRE(descriptor.pointer_offset == table_offset);
                REQUIRE_FALSE(descriptor.is_written);
                REQUIRE(std::ranges::any_of(program.info.storage_buffers_descriptors,
                    [](const auto& desc) { return desc.pointer_offset == ~0U && desc.cbuf_offset == 0x780; }));

                // The byte offset must subtract the loaded child pointer, not
                // the root table's address. Both have the same cbuf identity.
                const auto* subtract = load->Arg(1).InstRecursive();
                REQUIRE(subtract->GetOpcode() == IR::Opcode::ISub32);
                const auto* aligned = subtract->Arg(1).InstRecursive();
                REQUIRE(aligned->GetOpcode() == IR::Opcode::BitwiseAnd32);
                REQUIRE(aligned->Arg(0).Resolve() == low.Resolve());
                REQUIRE(aligned->Arg(1).U32() == ~15U);
            }
        }
    }
}


namespace {
using namespace Shader;

struct UniformWriteProgram {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::IREmitter ir{block};
    IR::Program program;
    StubEnvironment env;
    HostTranslateInfo host{};

    explicit UniformWriteProgram(Stage stage = Stage::Mesh) {
        program.stage = stage;
        program.post_order_blocks.push_back(&block);
        env.native = true;
        host.min_ssbo_alignment = 16;
        host.support_graphics_readonly_tables_with_writes = true;
    }

    IR::U64 Root(u32 offset) {
        return ir.PackUint2x32(ir.CompositeConstruct(
            ir.GetCbuf(ir.Imm32(0), ir.Imm32(offset)),
            ir.GetCbuf(ir.Imm32(0), ir.Imm32(offset + 4))));
    }

    IR::U32 ReadChild() {
        const auto child = ir.LoadGlobal64(Root(0x780));
        return ir.LoadGlobal32(ir.PackUint2x32(ir.CompositeConstruct(
            ir.CompositeExtract(child, 0), ir.CompositeExtract(child, 1))));
    }

    IR::U64 UniformAddress(u32 offset, s32 displacement, bool correct_carry = true) {
        const auto root_low = ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x790));
        const auto root_high = ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x794));
        const auto uniform_low = ir.GetCbuf(ir.Imm32(1), ir.Imm32(offset));
        const auto uniform_high = ir.GetCbuf(ir.Imm32(1), ir.Imm32(offset + 4));
        const IR::U32 low{ir.IAdd(root_low, uniform_low)};
        const auto carry = ir.GetCarryFromOp(correct_carry ? low
            : IR::U32{ir.IAdd(root_low, ir.Imm32(1))});
        const IR::U32 high{ir.IAdd(ir.IAdd(root_high, uniform_high),
                                  IR::U32{ir.Select(carry, ir.Imm32(1), ir.Imm32(0))})};
        const auto zero = ir.ShiftRightLogical(
            ir.BitFieldExtract(root_low, ir.Imm32(0), ir.Imm32(16), false), ir.Imm32(16));
        const auto address = ir.PackUint2x32(ir.CompositeConstruct(low, ir.IAdd(high, zero)));
        return displacement == 0 ? address
            : IR::U64{ir.IAdd(address, ir.Imm64(static_cast<u64>(s64{displacement})))};
    }

    void Run() {
        Optimization::GlobalMemoryToStorageBufferPass(env, program, host);
    }
};
} // namespace

TEST_CASE("Native readonly children retain runtime uniform write proofs",
          "[shader_recompiler][global_memory][uniform_writes]") {
    using namespace Shader;
    for (const auto stage : {Stage::Task, Stage::Mesh}) {
        CAPTURE(stage);
        UniformWriteProgram test{stage};
        const auto read = test.ReadChild();
        std::array<IR::U32, 4> atomics;
        size_t index{};
        for (const u32 offset : {0U, 8U}) {
            for (const s32 displacement : {0, -16}) {
                atomics[index++] = IR::U32{test.ir.GlobalAtomicIAdd(
                    test.UniformAddress(offset, displacement), test.ir.Imm32(1))};
            }
        }
        test.ir.WriteGlobal32(IR::U64{test.ir.IAdd(test.Root(0x790), test.ir.Imm64(u64{32}))},
                              test.ir.Imm32(1));
        test.Run();
        REQUIRE(read.Resolve().Inst()->GetOpcode() == IR::Opcode::LoadStorage32);
        for (const auto& atomic : atomics) {
            REQUIRE(atomic.Resolve().Inst()->GetOpcode() == IR::Opcode::StorageAtomicIAdd32);
        }
        const auto& info = test.program.info;
        const auto written = std::ranges::find_if(info.storage_buffers_descriptors,
            [](const auto& descriptor) { return descriptor.is_written; });
        REQUIRE(written != info.storage_buffers_descriptors.end());
        REQUIRE(written->has_uniform_write_proofs);
        REQUIRE(written->fixed_write_end == 32 + 15 + 4);
        REQUIRE(info.uniform_storage_write_proofs.size() == 4);
        for (const u32 offset : {0U, 8U}) {
            for (const s32 displacement : {0, -16}) {
                const StorageBufferUniformWrite expected{0, 0x790, 1, offset, displacement, 4};
                REQUIRE(std::ranges::find(info.uniform_storage_write_proofs, expected) !=
                        info.uniform_storage_write_proofs.end());
            }
        }
    }
}

TEST_CASE("Native uniform write proofs reject an unrelated carry",
          "[shader_recompiler][global_memory][uniform_writes]") {
    using namespace Shader;
    UniformWriteProgram test;
    const auto read = test.ReadChild();
    (void)test.ir.GlobalAtomicIAdd(test.UniformAddress(0, 0, false), test.ir.Imm32(1));
    test.Run();
    REQUIRE(read.Resolve().Inst()->GetOpcode() == IR::Opcode::LoadGlobal32);
    REQUIRE(test.program.info.uniform_storage_write_proofs.empty());
    REQUIRE(std::ranges::none_of(test.program.info.storage_buffers_descriptors,
        [](const auto& descriptor) { return descriptor.has_uniform_write_proofs; }));
}

TEST_CASE("An unknown global writer disables uniform readonly child admission",
          "[shader_recompiler][global_memory][uniform_writes]") {
    using namespace Shader;
    UniformWriteProgram test;
    const auto read = test.ReadChild();
    (void)test.ir.GlobalAtomicIAdd(test.UniformAddress(8, -16), test.ir.Imm32(1));
    const IR::U64 unknown{test.ir.IAdd(test.Root(0x790),
                                      test.ir.UConvert(64, test.ir.LaneId()))};
    test.ir.WriteGlobal32(unknown, test.ir.Imm32(1));
    test.Run();
    REQUIRE(read.Resolve().Inst()->GetOpcode() == IR::Opcode::LoadGlobal32);
    REQUIRE(test.program.info.uniform_storage_write_proofs.empty());
    REQUIRE(std::ranges::none_of(test.program.info.storage_buffers_descriptors,
        [](const auto& descriptor) {
            return descriptor.has_uniform_write_proofs || descriptor.fixed_write_end != 0;
        }));
}

TEST_CASE("Joining runtime uniform write proofs preserves unknown writer fallback",
          "[shader_recompiler][global_memory][uniform_writes]") {
    using namespace Shader;
    Info base;
    base.storage_buffers_descriptors.push_back({.cbuf_index = 0, .cbuf_offset = 0x790,
        .count = 1, .is_written = true, .has_uniform_write_proofs = true});
    base.uniform_storage_write_proofs.push_back({0, 0x790, 1, 0, 0, 4});
    Info source = base;
    source.uniform_storage_write_proofs.front().offset_cbuf_offset = 8;
    Optimization::JoinStorageInfo(base, source);
    REQUIRE(base.uniform_storage_write_proofs.size() == 2);
    REQUIRE(base.storage_buffers_descriptors.front().has_uniform_write_proofs);
    Optimization::JoinStorageInfo(base, source);
    REQUIRE(base.uniform_storage_write_proofs.size() == 2);

    source.uniform_storage_write_proofs.clear();
    source.storage_buffers_descriptors.front().has_uniform_write_proofs = false;
    Optimization::JoinStorageInfo(base, source);
    REQUIRE(base.uniform_storage_write_proofs.empty());
    REQUIRE_FALSE(base.storage_buffers_descriptors.front().has_uniform_write_proofs);
    REQUIRE(base.storage_buffers_descriptors.front().fixed_write_end == 0);
}
