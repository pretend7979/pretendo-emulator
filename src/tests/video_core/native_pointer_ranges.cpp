// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/ir_opt/native_pointer_ranges.h"

namespace {
using namespace Shader;
using Optimization::BoundedNativeOffsets;

struct RangeFixture {
    IR::Block* Block() {
        blocks.push_back(std::make_unique<IR::Block>(pool));
        IR::Block* result = blocks.back().get();
        program.blocks.push_back(result);
        return result;
    }

    IR::Inst* Inst(IR::Block* block, IR::Opcode opcode,
                   std::initializer_list<IR::Value> arguments = {}) {
        IR::Inst* result = pool.Create(opcode, 0);
        block->Instructions().push_back(*result);
        size_t index{};
        for (const auto argument : arguments) {
            result->SetArg(index++, argument);
        }
        return result;
    }

    IR::Inst* Phi(IR::Block* block,
                  std::initializer_list<std::pair<IR::Block*, IR::Value>> arguments = {}) {
        IR::Inst* result = Inst(block, IR::Opcode::Phi);
        result->SetFlags(IR::Type::U32);
        for (const auto& [predecessor, value] : arguments) {
            result->AddPhiOperand(predecessor, value);
        }
        return result;
    }

    void Branch(IR::Block* from, IR::Block* yes, IR::Block* no, IR::U1 condition) {
        from->AddBranch(yes);
        from->AddBranch(no);
        IR::AbstractSyntaxNode node;
        node.type = IR::AbstractSyntaxNode::Type::Block;
        node.data.block = from;
        program.syntax_list.push_back(node);
        node.type = IR::AbstractSyntaxNode::Type::If;
        node.data.if_node = {condition, yes, no};
        program.syntax_list.push_back(node);
    }

    ObjectPool<IR::Inst> pool;
    std::vector<std::unique_ptr<IR::Block>> blocks;
    IR::Program program;
};
} // namespace

TEST_CASE("Native pointer ranges preserve u32 arithmetic and unknown values", "[video_core]") {
    RangeFixture fixture;
    IR::Block* block = fixture.Block();
    IR::IREmitter ir{*block};
    const auto wrapped = ir.IAdd(ir.Imm32(~0U), ir.Imm32(17));
    const auto unknown = ir.LaneId();
    const auto invalid_shift = fixture.Inst(
        block, IR::Opcode::ShiftLeftLogical32, {IR::Value{1U}, IR::Value{32U}});
    BoundedNativeOffsets ranges{fixture.program};
    REQUIRE(ranges.Offsets(wrapped, block) == std::optional{std::vector<u32>{16}});
    REQUIRE_FALSE(ranges.Offsets(unknown, block));
    REQUIRE_FALSE(ranges.Offsets(IR::Value{invalid_shift}, block));
}

TEST_CASE("Native pointer copy cycles require one external seed", "[video_core]") {
    RangeFixture fixture;
    IR::Block* block = fixture.Block();
    block->AddBranch(block);
    IR::Inst* copy = fixture.Phi(block);
    copy->AddPhiOperand(block, IR::Value{48U});
    copy->AddPhiOperand(block, IR::Value{copy});
    IR::Inst* recurrence = fixture.Phi(block);
    IR::Inst* increment = fixture.Inst(block, IR::Opcode::IAdd32,
                                       {IR::Value{recurrence}, IR::Value{16U}});
    recurrence->AddPhiOperand(block, IR::Value{0U});
    recurrence->AddPhiOperand(block, IR::Value{increment});
    BoundedNativeOffsets ranges{fixture.program};
    REQUIRE(ranges.Offsets(IR::Value{copy}, block) == std::optional{std::vector<u32>{48}});
    REQUIRE_FALSE(ranges.Offsets(IR::Value{recurrence}, block));
}

TEST_CASE("Native pointer paths exclude only unreachable unknown inputs", "[video_core]") {
    RangeFixture fixture;
    auto* entry = fixture.Block();
    auto* yes = fixture.Block();
    auto* no = fixture.Block();
    auto* merge = fixture.Block();
    auto* known_use = fixture.Block();
    auto* unknown_use = fixture.Block();
    IR::IREmitter ir{*entry};
    const auto predicate = ir.IEqual(ir.LaneId(), ir.Imm32(1));
    fixture.Branch(entry, yes, no, predicate);
    yes->AddBranch(merge);
    no->AddBranch(merge);
    auto* value = fixture.Phi(merge, {{yes, IR::Value{16U}}, {no, ir.LaneId()}});
    fixture.Branch(merge, known_use, unknown_use, predicate);
    BoundedNativeOffsets ranges{fixture.program};
    REQUIRE(ranges.Offsets(IR::Value{value}, known_use) ==
            std::optional{std::vector<u32>{16}});
    REQUIRE_FALSE(ranges.Offsets(IR::Value{value}, unknown_use));
    REQUIRE_FALSE(ranges.Offsets(IR::Value{value}, nullptr));
    REQUIRE_FALSE(ranges.MayReachEdge(entry, yes, no));
}

TEST_CASE("Native pointer shuffle values are not filtered by the receiving lane", "[video_core]") {
    RangeFixture fixture;
    auto* entry = fixture.Block();
    auto* yes = fixture.Block();
    auto* no = fixture.Block();
    auto* merge = fixture.Block();
    IR::IREmitter ir{*entry};
    const auto predicate = ir.ILessThan(ir.LaneId(), ir.Imm32(16), false);
    fixture.Branch(entry, yes, no, predicate);
    yes->AddBranch(merge);
    no->AddBranch(merge);
    auto* value = fixture.Phi(merge, {{yes, IR::Value{16U}}, {no, IR::Value{32U}}});
    auto* shuffle = fixture.Inst(merge, IR::Opcode::ShuffleIndex,
                                 {IR::Value{value}, IR::Value{0U}, IR::Value{0U}, IR::Value{31U}});
    auto* result = fixture.Inst(merge, IR::Opcode::SelectU32,
                                {predicate, IR::Value{0U}, IR::Value{shuffle}});
    BoundedNativeOffsets ranges{fixture.program};
    REQUIRE((ranges.Offsets(IR::Value{result}, merge) ==
             std::optional{std::vector<u32>{0, 16, 32}}));
}

TEST_CASE("Native pointer predicates retain undefined alternatives and masked bits", "[video_core]") {
    SECTION("Undefined Boolean uses are not one stable atom") {
        RangeFixture fixture;
        auto* block = fixture.Block();
        auto* undef = fixture.Inst(block, IR::Opcode::UndefU1);
        auto* inner = fixture.Inst(block, IR::Opcode::SelectU32,
                                   {IR::Value{undef}, IR::Value{16U}, IR::Value{32U}});
        auto* inverse = fixture.Inst(block, IR::Opcode::LogicalNot, {IR::Value{undef}});
        auto* result = fixture.Inst(block, IR::Opcode::SelectU32,
                                    {IR::Value{inverse}, IR::Value{inner}, IR::Value{48U}});
        BoundedNativeOffsets ranges{fixture.program};
        REQUIRE((ranges.Offsets(IR::Value{result}, block) ==
                 std::optional{std::vector<u32>{16, 32, 48}}));
    }
    SECTION("Predicate masks preserve every unselected bit") {
        for (const u32 mask : {0U, 3U, 0x7fU, 0x80U, 255U}) {
            for (u32 bit = 0; bit < 8; ++bit) {
                for (const bool previous : {false, true}) {
                    for (const bool incoming : {false, true}) {
                        RangeFixture fixture;
                        auto* block = fixture.Block();
                        auto* selected = fixture.Inst(block, IR::Opcode::BitFieldUExtract,
                                                      {IR::Value{mask}, IR::Value{bit}, IR::Value{1U}});
                        auto* zero = fixture.Inst(block, IR::Opcode::GetZeroFromOp,
                                                 {IR::Value{selected}});
                        auto* predicate = fixture.Inst(block, IR::Opcode::SelectU1,
                            {IR::Value{zero}, IR::Value{previous}, IR::Value{incoming}});
                        auto* result = fixture.Inst(block, IR::Opcode::SelectU32,
                            {IR::Value{predicate}, IR::Value{16U}, IR::Value{32U}});
                        BoundedNativeOffsets ranges{fixture.program};
                        const bool expected = (mask & (1U << bit)) ? incoming : previous;
                        REQUIRE(ranges.Offsets(IR::Value{result}, block) ==
                                std::optional{std::vector<u32>{expected ? 16U : 32U}});
                    }
                }
            }
        }
    }
}

TEST_CASE("Native pointer analysis refuses an exhausted proof", "[video_core]") {
    RangeFixture fixture;
    auto* block = fixture.Block();
    IR::Value value{0U};
    for (size_t index = 0; index < 600; ++index) {
        value = IR::Value{fixture.Inst(block, IR::Opcode::IAdd32, {value, IR::Value{1U}})};
    }
    BoundedNativeOffsets ranges{fixture.program};
    REQUIRE_THROWS_AS(ranges.Offsets(value, block), NotImplementedException);
}
