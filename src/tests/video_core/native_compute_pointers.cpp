// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <map>
#include <memory>
#include <vector>
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/backend/nvn_storage_descriptor.h"
namespace {
using namespace Shader;
struct Env final : Environment {
    bool native = true, accept = false, accept_all_static = false;
    u32 queries{};
    bool IsNativeProgram() const override { return native; }
    bool IsCbufStorageBufferPointer(u32 b, u32 o) override {
        ++queries;
        return (accept_all_static && b == 0 && (o & 7) == 0) ||
           (accept && b == 0 && (o == 0x780 || o == 0x790));
    }
    u64 ReadInstruction(u32) override { return 0; }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    TextureType ReadTextureType(u32) override { return {}; }
    TexturePixelFormat ReadTexturePixelFormat(u32) override { return {}; }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 0; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return {}; }
    void Dump(u64, u64) override {}
};

void Check(bool good, const char* label) {
    INFO(label);
    CHECK(good);
}
struct Test {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::IREmitter ir{block};
    IR::Program program;
    Env env;
    HostTranslateInfo host{};
    Test() {
        program.stage = Stage::Compute;
        program.blocks.push_back(&block);
        program.post_order_blocks.push_back(&block);
        host.min_ssbo_alignment = 16;
        host.support_native_compute_pointer_tracking = true;
        host.max_compute_storage_buffers = 64;
    }
    IR::U32 Offset(u32 a = 0, u32 b = 16) {
        return IR::U32{
            ir.Select(ir.ILessThan(ir.LaneId(), ir.Imm32(16), false), ir.Imm32(a), ir.Imm32(b))};
    }
    IR::U64 Root(IR::U32 offset) {
        const auto words = ir.GetCbuf(ir.Imm32(4), offset, 64, false);
        return Pair(words);
    }
    IR::U64 Pair(IR::Value words) {
        return ir.PackUint2x32(
            ir.CompositeConstruct(ir.CompositeExtract(words, 0), ir.CompositeExtract(words, 1)));
    }
    IR::U64 Child(IR::U64 root) { return Pair(ir.LoadGlobal64(root)); }
    void Run() { Optimization::GlobalMemoryToStorageBufferPass(env, program, host); }
};
} // namespace

TEST_CASE("Native compute pointer families preserve paired addresses and complete layouts",
          "[shader_recompiler][global_memory]") {
    for (unsigned mode = 0; mode < 5; ++mode) {
        Test t;
        auto root = t.Root(t.Offset());
        IR::Inst* access{};
        if (mode == 0)
            access = t.ir.LoadGlobal32(root).Inst();
        if (mode == 1) {
            t.ir.WriteGlobal32(root, t.ir.Imm32(1));
            access = &t.block.back();
        }
        if (mode == 2)
            access = t.ir.GlobalAtomicIAdd(root, t.ir.Imm32(1)).Inst();
        if (mode == 3)
            access = t.ir.GlobalAtomicUMax(root, t.ir.Imm32(7)).Inst();
        if (mode == 4)
            access = t.ir.LoadGlobal32(t.Child(root)).Inst();
        t.Run();
        const auto& ds = t.program.info.storage_buffers_descriptors;
        Check(ds.size() == (mode == 4 ? 4u : 2u), "all finite roots and child descriptors");
        Check(t.env.queries == 0, "dynamic pointer discovery never queries current null/mapping");
        Check(t.program.info.native_global_bounds, "full64 global bounds enabled");
        for (const auto& d : ds) {
            Check(d.cbuf_index == 4 && (d.cbuf_offset == 0 || d.cbuf_offset == 16),
                  "finite descriptor identity");
            Check(d.is_written == (mode >= 1 && mode <= 3),
                  "all and only direct writable families marked");
        }
        Check(access->GetOpcode() == (mode == 0 || mode == 4 ? IR::Opcode::LoadGlobal32
                                      : mode == 1            ? IR::Opcode::WriteGlobal32
                                      : mode == 2            ? IR::Opcode::GlobalAtomicIAdd32
                                                             : IR::Opcode::GlobalAtomicUMax32),
              "access remains full64 global opcode");
    }
    for (unsigned mode = 0; mode < 4; ++mode) {
        Test t;
        auto ptr = t.Root(t.Offset());
        static_cast<void>(t.ir.LoadGlobal32(t.Child(ptr)));
        if (mode == 0)
            t.host.support_native_compute_pointer_tracking = false;
        if (mode == 1)
            t.env.native = false;
        if (mode == 2)
            t.program.stage = Stage::Fragment;
        if (mode == 3)
            t.host.max_compute_storage_buffers = 3;
        bool threw = false;
        try {
            t.Run();
        } catch (const Exception&) {
            threw = true;
        }
        Check(threw == (mode == 3), "backend/stage/hardware gate");
        Check(t.program.info.storage_buffers_descriptors.empty(),
              "disabled or overflow path publishes no partial descriptors");
    }
    for (u32 count : {32u, 39u, 64u, 65u}) {
        Test t;
        for (u32 i = 0; i < count; ++i) {
            auto off = t.ir.IAdd(t.ir.Imm32(i * 16), t.ir.Imm32(0));
            static_cast<void>(t.ir.LoadGlobal32(t.Root(IR::U32{off})));
        }
        bool threw = false;
        try {
            t.Run();
        } catch (const Exception&) {
            threw = true;
        }
        Check(threw == (count == 65), "complete capacity64 limit");
        Check(t.program.info.storage_buffers_descriptors.size() == (count == 65 ? 0u : count),
              "no partial over-capacity layout");
    }
    for (unsigned mode = 0; mode < 4; ++mode) {
        Test t;
        IR::U32 off =
            mode == 0 ? t.ir.LaneId() : t.Offset(mode == 1 ? 1 : 0, mode == 2 ? 0x10000 : 16);
        auto root = t.Root(off);
        if (mode == 3)
            t.ir.WriteGlobal32(t.Child(root), t.ir.Imm32(1));
        else
            static_cast<void>(t.ir.LoadGlobal32(root));
        bool threw = false;
        try {
            t.Run();
        } catch (const Exception&) {
            threw = true;
        }
        Check(threw, "unknown, misaligned, out-of-cbuf, or nested-write family rejected");
        Check(t.program.info.storage_buffers_descriptors.empty(),
              "invalid family leaves no partial published layout");
    }
    {
        Test t;
        auto root = t.Root(t.Offset());
        t.ir.WriteGlobal32(root, t.ir.Imm32(1));
        static_cast<void>(t.ir.LoadGlobal32(t.Child(root)));
        bool threw = false;
        try {
            t.Run();
        } catch (const Exception&) {
            threw = true;
        }
        Check(threw, "same root table writer rejected");
        Check(t.program.info.storage_buffers_descriptors.empty(), "mutable table failure atomic");
    }

    for (unsigned mode = 0; mode < 3; ++mode) {
        Test t;
        auto a = t.ir.LoadGlobal64(t.Root(t.Offset(0, 16)));
        auto b = t.ir.LoadGlobal64(t.Root(t.Offset(32, 48)));
        auto p = t.ir.IEqual(t.ir.LaneId(), t.ir.Imm32(3));
        auto al = t.ir.CompositeExtract(a, 0), ah = t.ir.CompositeExtract(a, 1),
             bl = t.ir.CompositeExtract(b, 0), bh = t.ir.CompositeExtract(b, 1);
        auto low = mode == 2 ? t.ir.Select(p, al, bl) : al;
        auto high = mode == 0 ? ah : t.ir.Select(p, ah, bh);
        static_cast<void>(t.ir.LoadGlobal32(t.ir.PackUint2x32(t.ir.CompositeConstruct(low, high))));
        bool threw = false;
        try {
            t.Run();
        } catch (const Exception&) {
            threw = true;
        }
        Check(threw == (mode == 1),
              "mixed high halves reject; matching selectors preserve complete pairs");
        Check(t.program.info.storage_buffers_descriptors.size() == (mode == 1   ? 0
                                                                    : mode == 2 ? 8
                                                                                : 6),
              "paired descriptor publication");
    }
    {
        Test t;
        t.env.accept = true;
        auto base = t.ir.GetCbuf(t.ir.Imm32(0), t.ir.Imm32(0x780), 64, false);
        auto numeric = t.ir.GetCbuf(t.ir.Imm32(4), t.ir.LaneId(), 64, false);
        auto lo = t.ir.IAdd(IR::U32{t.ir.CompositeExtract(base, 0)},
                            IR::U32{t.ir.CompositeExtract(numeric, 0)});
        static_cast<void>(t.ir.LoadGlobal32(
            t.ir.PackUint2x32(t.ir.CompositeConstruct(lo, t.ir.CompositeExtract(base, 1)))));
        bool threw = false;
        try {
            t.Run();
        } catch (const Exception&) {
            threw = true;
        }
        Check(!threw, "numeric dynamic64 low offset is not an unbounded pointer");
        Check(t.program.info.storage_buffers_descriptors.size() == 1,
              "numeric offset creates no pointer descriptors");
        if (!t.program.info.storage_buffers_descriptors.empty())
            Check(t.program.info.storage_buffers_descriptors[0].cbuf_offset == 0x780,
                  "numeric offset keeps proven static base");
    }
    for (bool shuffled : {false, true}) {
        Test t;
        auto words = t.ir.GetCbuf(t.ir.Imm32(4), t.Offset(), 64, false);
        IR::U32 lo{t.ir.CompositeExtract(words, 0)}, hi{t.ir.CompositeExtract(words, 1)};
        if (shuffled) {
            lo = t.ir.ShuffleIndex(lo, t.ir.Imm32(0), t.ir.Imm32(31), t.ir.Imm32(0));
            hi = t.ir.ShuffleIndex(hi, t.ir.Imm32(0), t.ir.Imm32(31), t.ir.Imm32(0));
        }
        auto low = t.ir.IAdd(lo, t.ir.Imm32(16));
        auto high = t.ir.IAdd(
            hi, IR::U32{t.ir.Select(t.ir.ILessThan(low, lo, false), t.ir.Imm32(1), t.ir.Imm32(0))});
        static_cast<void>(t.ir.LoadGlobal32(t.ir.PackUint2x32(t.ir.CompositeConstruct(low, high))));
        t.Run();
        Check(t.program.info.storage_buffers_descriptors.size() == 2,
              "paired64 addition and carry preserve every root");
    }

    for (unsigned mode = 0; mode < 3; ++mode) {
        Test t;
        IR::Block yes{t.pool}, no{t.pool}, merge{t.pool};
        t.program.blocks.insert(t.program.blocks.end(), {&yes, &no, &merge});
        t.program.post_order_blocks.insert(t.program.post_order_blocks.begin(),
                                           {&merge, &no, &yes});
        auto a = t.ir.LoadGlobal64(t.Root(t.Offset(0, 16))),
             b = t.ir.LoadGlobal64(t.Root(t.Offset(32, 48)));
        auto al = t.ir.CompositeExtract(a, 0), ah = t.ir.CompositeExtract(a, 1),
             bl = t.ir.CompositeExtract(b, 0), bh = t.ir.CompositeExtract(b, 1);
        auto cond = t.ir.IEqual(t.ir.LaneId(), t.ir.Imm32(3));
        t.block.AddBranch(&yes);
        t.block.AddBranch(&no);
        yes.AddBranch(&merge);
        no.AddBranch(&merge);
        IR::AbstractSyntaxNode node;
        node.type = IR::AbstractSyntaxNode::Type::Block;
        node.data.block = &t.block;
        t.program.syntax_list.push_back(node);
        node.type = IR::AbstractSyntaxNode::Type::If;
        node.data.if_node = {cond, &yes, &no};
        t.program.syntax_list.push_back(node);
        auto phi = [&](IR::Value l, IR::Value r) {
            auto* i = t.pool.Create(IR::Opcode::Phi, 0);
            i->SetFlags(IR::Type::U32);
            i->AddPhiOperand(&yes, l);
            i->AddPhiOperand(&no, r);
            merge.Instructions().push_back(*i);
            return IR::U32{IR::Value{i}};
        };
        auto low = phi(al, bl);
        auto high = phi(ah, mode == 0 ? bh : mode == 1 ? ah : IR::Value{u32{123}});
        IR::IREmitter ir{merge};
        static_cast<void>(ir.LoadGlobal32(ir.PackUint2x32(ir.CompositeConstruct(low, high))));
        bool threw = false;
        try {
            t.Run();
        } catch (const Exception&) {
            threw = true;
        }
        Check(threw == (mode != 0),
              "paired Phi uses same predecessors; mixed or unknown reachable arm rejects");
        Check(t.program.info.storage_buffers_descriptors.size() == (mode == 0 ? 8 : 0),
              "Phi rejection publishes no partial family");
    }
}

TEST_CASE("Static native compute pointers only analyze ranges when needed", "[shader_recompiler][global_memory]") {
    for (const bool shuffle : {false, true}) {
        Test t;
        t.env.accept = true;
        auto words = t.ir.GetCbuf(t.ir.Imm32(0), t.ir.Imm32(0x780), 64, false);
        IR::U32 low{t.ir.CompositeExtract(words, 0)}, high{t.ir.CompositeExtract(words, 1)};
        if (shuffle) {
            low = t.ir.ShuffleIndex(low, t.ir.Imm32(0), t.ir.Imm32(31), t.ir.Imm32(0));
            high = t.ir.ShuffleIndex(high, t.ir.Imm32(0), t.ir.Imm32(31), t.ir.Imm32(0));
        } else {
            // An unrelated deep dynamic index would exhaust range recursion if
            // the direct static access eagerly constructed and primed the helper.
            IR::U32 offset = t.ir.Imm32(0);
            for (u32 i = 0; i < 600; ++i) offset = IR::U32{t.ir.IAdd(offset, t.ir.Imm32(1))};
            static_cast<void>(t.ir.GetCbuf(t.ir.Imm32(4), offset, 64, false));
        }
        const auto access = t.ir.LoadGlobal32(t.ir.PackUint2x32(t.ir.CompositeConstruct(low, high)));
        REQUIRE_NOTHROW(t.Run());
        REQUIRE(t.program.info.storage_buffers_descriptors.size() == 1);
        REQUIRE(t.program.info.storage_buffers_descriptors[0].cbuf_offset == 0x780);
        if (shuffle) REQUIRE(access.Inst()->GetOpcode() == IR::Opcode::LoadGlobal32);
    }
}

TEST_CASE("Ordinary compute and graphics retain the 32 storage buffer limit", "[shader_recompiler][global_memory]") {
    for (const Stage stage : {Stage::Compute, Stage::Fragment}) {
        Test t;
        t.program.stage = stage;
        t.env.accept_all_static = true;
        for (u32 i = 0; i < 33; ++i) {
            const auto words = t.ir.GetCbuf(t.ir.Imm32(0), t.ir.Imm32(i * 16), 64, false);
            static_cast<void>(t.ir.LoadGlobal32(t.Pair(words)));
        }
        REQUIRE_THROWS_AS(t.Run(), NotImplementedException);
        REQUIRE(t.program.info.storage_buffers_descriptors.empty());
    }
}

TEST_CASE("Native pointer words retain joint paths across separate Phi blocks",
          "[shader_recompiler][global_memory]") {
    for (unsigned mode = 0; mode < 16; ++mode) {
        CAPTURE(mode);
        Test t;
        std::vector<std::unique_ptr<IR::Block>> extra_blocks;
        IR::Block* current = &t.block;
        const auto block = [&]() {
            auto* result = extra_blocks.emplace_back(std::make_unique<IR::Block>(t.pool)).get();
            t.program.blocks.push_back(result);
            t.program.post_order_blocks.insert(t.program.post_order_blocks.begin(), result);
            return result;
        };
        const auto syntax_block = [&](IR::Block* value) {
            IR::AbstractSyntaxNode node;
            node.type = IR::AbstractSyntaxNode::Type::Block;
            node.data.block = value;
            t.program.syntax_list.push_back(node);
        };
        const auto branch = [&](IR::Block* from, IR::Block* yes, IR::Block* no, IR::U1 cond) {
            from->AddBranch(yes);
            from->AddBranch(no);
            syntax_block(from);
            IR::AbstractSyntaxNode node;
            node.type = IR::AbstractSyntaxNode::Type::If;
            node.data.if_node = {cond, yes, no};
            t.program.syntax_list.push_back(node);
        };
        const auto choice = [&](IR::U1 cond, IR::Value yes_value, IR::Value no_value) {
            auto* yes = block();
            auto* no = block();
            auto* merge = block();
            branch(current, yes, no, cond);
            yes->AddBranch(merge);
            no->AddBranch(merge);
            auto* phi = t.pool.Create(IR::Opcode::Phi, 0);
            phi->SetFlags(IR::Type::U32);
            phi->AddPhiOperand(yes, yes_value);
            phi->AddPhiOperand(no, no_value);
            merge->Instructions().push_back(*phi);
            current = merge;
            return IR::U32{IR::Value{phi}};
        };
        const auto a = t.ir.LoadGlobal64(t.Root(t.Offset(0, 16)));
        const auto b = t.ir.LoadGlobal64(t.Root(t.Offset(32, 48)));
        auto al = t.ir.CompositeExtract(a, 0), ah = t.ir.CompositeExtract(a, 1);
        auto bl = t.ir.CompositeExtract(b, 0), bh = t.ir.CompositeExtract(b, 1);
        if (mode == 8 || mode == 9) {
            const auto pa = t.ir.GetCbuf(t.ir.Imm32(4), t.ir.Imm32(64), 64, false);
            const auto pb = t.ir.GetCbuf(t.ir.Imm32(4), t.ir.Imm32(80), 64, false);
            al = t.ir.CompositeExtract(pa, 0);
            ah = t.ir.CompositeExtract(pa, 1);
            bl = t.ir.CompositeExtract(pb, 0);
            bh = t.ir.CompositeExtract(pb, 1);
        }
        auto predicate = t.ir.IEqual(t.ir.LaneId(), t.ir.Imm32(3));
        if (mode == 12) {
            auto* undef = t.pool.Create(IR::Opcode::UndefU1, 0);
            t.block.Instructions().push_back(*undef);
            predicate = IR::U1{IR::Value{undef}};
        }
        const auto different = t.ir.IEqual(t.ir.LaneId(), t.ir.Imm32(7));
        auto high_predicate = mode == 1 ? t.ir.LogicalNot(predicate)
                            : mode == 2 ? different : predicate;
        if (mode == 5) {
            auto* identity = t.pool.Create(IR::Opcode::Identity, 0);
            identity->SetArg(0, predicate);
            t.block.Instructions().push_back(*identity);
            high_predicate = IR::U1{IR::Value{identity}};
        }
        IR::Value high_yes = mode == 3 || mode == 9 ? bh : ah;
        IR::Value high_no = bh;
        if (mode == 4) {
            auto* undef = t.pool.Create(IR::Opcode::UndefU32, 0);
            t.block.Instructions().push_back(*undef);
            high_no = IR::Value{undef};
        }
        if (mode == 10) {
            bl = t.ir.Imm32(0);
            high_no = t.ir.Imm32(0);
        }
        if (mode == 11) {
            bl = t.ir.Imm32(0);
            high_no = t.ir.Imm32(1);
        }
        auto low = choice(predicate, al, bl);
        auto high = mode == 15 ? IR::U32{bh} : choice(high_predicate, high_yes, high_no);
        if (mode == 6) {
            low = choice(predicate, low, bl);
            high = choice(predicate, high, bh);
        }
        if (mode == 13) {
            // A predicate defined in a CFG loop must not correlate separate iterations.
            current->AddBranch(&t.block);
        }
        if (mode == 15) {
            auto* read = block();
            auto* exit = block();
            branch(current, read, exit, t.ir.LogicalNot(predicate));
            current = read;
        }
        IR::IREmitter ir{*current};
        if (mode == 7) {
            const auto sum = ir.IAdd(low, ir.Imm32(16));
            high = ir.IAdd(high, IR::U32{ir.Select(ir.ILessThan(sum, low, false),
                                                 ir.Imm32(1), ir.Imm32(0))});
            low = sum;
        }
        if (mode == 14 || mode == 15) {
            low = ir.ShuffleIndex(low, ir.Imm32(0), ir.Imm32(31), ir.Imm32(0));
            high = ir.ShuffleIndex(high, ir.Imm32(0), ir.Imm32(31), ir.Imm32(0));
        }
        auto address = ir.PackUint2x32(ir.CompositeConstruct(low, high));
        if (mode == 8 || mode == 9) {
            const auto child = ir.LoadGlobal64(address);
            address = ir.PackUint2x32(ir.CompositeConstruct(ir.CompositeExtract(child, 0),
                                                           ir.CompositeExtract(child, 1)));
        }
        static_cast<void>(ir.LoadGlobal32(address));
        // Immediate/Phi null halves remain conservatively unsupported.
        const bool expected_reject = mode == 1 || mode == 2 || mode == 3 || mode == 4 ||
                                     mode == 9 || mode == 10 || mode == 11 || mode == 12 ||
                                     mode == 13 || mode == 15;
        bool rejected = false;
        try {
            t.Run();
        } catch (const Exception&) {
            rejected = true;
        }
        CHECK(rejected == expected_reject);
        if (expected_reject) {
            CHECK(t.program.info.storage_buffers_descriptors.empty());
        } else {
            CHECK(t.program.info.native_global_bounds);
            CHECK_FALSE(t.program.info.storage_buffers_descriptors.empty());
            for (const auto& descriptor : t.program.info.storage_buffers_descriptors) {
                CHECK_FALSE(descriptor.is_written);
                CHECK(descriptor.cbuf_index == 4);
                CHECK(descriptor.cbuf_offset <= 80);
            }
        }
    }
}
