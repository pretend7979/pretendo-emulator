// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/structured_control_flow.h"
#include "shader_recompiler/host_translate_info.h"

namespace {
namespace IR = Shader::IR;
using NodeType = IR::AbstractSyntaxNode::Type;

class Environment final : public Shader::Environment {
public:
    Environment() {
        stage = Shader::Stage::Fragment;
        start_address = 8;
    }
    std::map<u32, u64> code;
    u64 ReadInstruction(u32 address) override {
        return code.at(address);
    }
    u32 ReadCbufValue(u32, u32) override {
        return 0;
    }
    Shader::TextureType ReadTextureType(u32) override {
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 1;
    }
    u32 TextureBoundBuffer() const override {
        return 0;
    }
    u32 LocalMemorySize() const override {
        return 0;
    }
    u32 SharedMemorySize() const override {
        return 0;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {1, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return {};
    }
    void Dump(u64, u64) override {}
};

struct ShaderBuilder {
    Environment env;
    u32 next{8};
    u32 Emit(u64 instruction) {
        const u32 address = next;
        env.code[address] = instruction;
        next += 8;
        if (next % 32 == 0)
            next += 8; // Skip Maxwell scheduling words.
        return address;
    }
    void Inputs() {
        // Runtime cbuf values independently define P0, P1 and P2; none is constant
        // folded.
        for (u32 predicate = 0; predicate < 3; ++predicate) {
            const u32 reg = 10 + predicate;
            Emit(0x4c98000000070000ULL | reg | (15ULL << 39) | (u64{predicate} << 20));
            Emit(0x5b60000000070000ULL | 7 | (u64{predicate} << 3) | (u64{reg} << 8) |
                 (255ULL << 20) | (7ULL << 39) | (5ULL << 49));
        }
    }
    u32 BranchUnlessP0() {
        return Emit(0xe24000000008000fULL);
    }
    void Target(u32 branch) {
        env.code.at(branch) |= u64{(next - branch - 8) & 0xffffffU} << 20;
    }
    void Kill() {
        Emit(0xe33000000001000fULL);
    } // @P1 KIL
    void Exit(u32 predicate = 7) {
        Emit(0xe30000000000000fULL | (u64{predicate} << 16));
    }
    void Marker() {
        Emit(0x010000000077f000ULL);
    } // MOV32I R0,7
};

enum class Program { EarlyReturn, SingleReturn, NestedKill };
struct Shape {
    size_t demote{std::numeric_limits<size_t>::max()};
    size_t first_return{std::numeric_limits<size_t>::max()};
    size_t marker{std::numeric_limits<size_t>::max()};
    u32 returns{}, demote_depth{};
};
Shape Compile(Program program, bool reorder) {
    ShaderBuilder shader;
    shader.Inputs();
    if (program == Program::NestedKill) {
        const u32 skip = shader.BranchUnlessP0();
        shader.Kill();
        shader.Marker();
        shader.Target(skip);
    } else {
        shader.Kill();
        if (program == Program::EarlyReturn) {
            // The two instructions implement EXIT iff P0 && P2, nesting its return in
            // ASL.
            const u32 skip = shader.BranchUnlessP0();
            shader.Exit(2);
            shader.Target(skip);
        }
    }
    shader.Marker();
    shader.Exit();
    Shader::ObjectPool<IR::Inst> instructions{512};
    Shader::ObjectPool<IR::Block> blocks{64};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks{64};
    Shader::Maxwell::Flow::CFG cfg{shader.env, flow_blocks, shader.env.StartAddress()};
    Shader::HostTranslateInfo host{};
    host.needs_demote_reorder = reorder;
    const auto asl = Shader::Maxwell::BuildASL(instructions, blocks, shader.env, cfg, host);
    Shape shape;
    u32 depth{};
    for (size_t index = 0; index < asl.size(); ++index) {
        const auto& node = asl[index];
        if (node.type == NodeType::If)
            ++depth;
        if (node.type == NodeType::EndIf)
            --depth;
        if (node.type == NodeType::Return) {
            ++shape.returns;
            shape.first_return = std::min(shape.first_return, index);
        }
        if (node.type != NodeType::Block)
            continue;
        for (const auto& inst : node.data.block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::DemoteToHelperInvocation) {
                shape.demote = index;
                shape.demote_depth = depth;
            }
            if (inst.GetOpcode() == IR::Opcode::SetRegister && inst.Arg(0).Reg() == IR::Reg::R0)
                shape.marker = index;
        }
    }
    return shape;
}
} // namespace

TEST_CASE("Fragment demotion precedes nested early returns", "[shader_recompiler][demote]") {
    for (bool reorder : {false, true}) {
        const auto shape = Compile(Program::EarlyReturn, reorder);
        REQUIRE(shape.returns == 2);
        REQUIRE(shape.demote < shape.first_return);
    }
}
TEST_CASE("Single-return fragment demotion remains eligible for reordering",
          "[shader_recompiler][demote]") {
    const auto original = Compile(Program::SingleReturn, false);
    const auto reordered = Compile(Program::SingleReturn, true);
    REQUIRE(original.returns == 1);
    REQUIRE(reordered.returns == 1);
    REQUIRE(original.demote < original.marker);
    REQUIRE(reordered.marker < reordered.demote);
    REQUIRE(reordered.demote < reordered.first_return);
}
TEST_CASE("Nested fragment demotion retains its conditional guards",
          "[shader_recompiler][demote]") {
    for (bool reorder : {false, true}) {
        const auto shape = Compile(Program::NestedKill, reorder);
        REQUIRE(shape.returns == 1);
        REQUIRE(shape.demote_depth == 2);
        REQUIRE(shape.demote < shape.first_return);
    }
}
