// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <map>
#include <stdexcept>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"

namespace {
namespace IR = Shader::IR;
using Shader::Ampere::Instruction;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
struct Env final : Shader::Environment {
    std::map<u32, u64> code;
    Env() {
        stage = Shader::Stage::Mesh;
        start_address = 8;
        mesh_registers = {0x2014, 32U << 20, 32U | (1U << 12) | (1U << 22)};
    }
    bool IsNativeProgram() const override {
        return true;
    }
    u64 ReadInstruction(u32 address) override {
        return code.at(address);
    }
    u32 ReadCbufValue(u32, u32) override {
        return 0;
    }
    Shader::TextureType ReadTextureType(u32) override {
        throw std::logic_error("Unexpected texture");
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        throw std::logic_error("Unexpected texture");
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
struct Fixture {
    Shader::ObjectPool<IR::Inst> instructions{1024};
    Shader::ObjectPool<IR::Block> blocks{128};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks{128};
    Env env;
    IR::Program Translate(const std::vector<Instruction>& native) {
        auto lowered = Shader::Ampere::LowerToMaxwell(native);
        INFO(lowered.error);
        REQUIRE(static_cast<bool>(lowered));
        u32 address = 8;
        for (u64 word : lowered.instructions) {
            env.code[address] = word;
            address += 8;
            if (address % 32 == 0)
                address += 8;
        }
        Shader::Maxwell::Flow::CFG cfg{env, flow_blocks, env.StartAddress()};
        Shader::HostTranslateInfo host{};
        host.support_float64 = true;
        host.support_float16 = true;
        host.support_int64 = true;
        return Shader::Maxwell::TranslateProgram(instructions, blocks, env, cfg, host);
    }
};
Instruction Mov(u32 target, u32 value) {
    return Instruction{{(target << 16) | 0x7802, value, 0xf00, 0}};
}
Instruction Store(u32 data, u32 address = 255, u32 predicate = 7) {
    return Instruction{
        {(address << 24) | (predicate << 12) | 0x927, (0x200U << 8) | data, 0xb800, 0}};
}
void ExpectStore(u32 data, bool uniform = false, bool guarded = false, bool condition = true,
                 u32 address = 255) {
    constexpr u32 Bits = 0x3f234567;
    std::vector<Instruction> native;
    if (uniform)
        native.push_back(Instruction{{0x7882, 123, 0, 0}}); // UMOV UR0,123 reserves UR mirrors.
    if (guarded) {
        native.push_back(Mov(5, condition ? 1 : 0));
        native.push_back(Instruction{{0x0500720c, 255, 0x03f05000, 0}}); // ISETP.NE P0,PT,R5,RZ,PT.
    }
    native.push_back(Mov(data, address == data ? 0x20 : Bits));
    if (address != 255 && address != data)
        native.push_back(Mov(address, 0x40));
    native.push_back(Store(data, address, guarded ? 0 : 7));
    native.push_back(Exit);
    Fixture fixture;
    const auto program = fixture.Translate(native);
    unsigned stores = 0;
    for (size_t node_index = 0; node_index < program.syntax_list.size(); ++node_index) {
        const auto& node = program.syntax_list[node_index];
        if (node.type == IR::AbstractSyntaxNode::Type::If) {
            auto condition_value = node.data.if_node.cond.Resolve();
            while (!condition_value.IsImmediate() &&
                   condition_value.Inst()->GetOpcode() == IR::Opcode::ConditionRef)
                condition_value = condition_value.Inst()->Arg(0).Resolve();
            REQUIRE(condition_value.IsImmediate());
            if (!condition_value.U1()) {
                unsigned depth = 1;
                while (depth) {
                    REQUIRE(++node_index < program.syntax_list.size());
                    const auto type = program.syntax_list[node_index].type;
                    if (type == IR::AbstractSyntaxNode::Type::If)
                        ++depth;
                    if (type == IR::AbstractSyntaxNode::Type::EndIf)
                        --depth;
                }
            }
            continue;
        }
        if (node.type == IR::AbstractSyntaxNode::Type::Return)
            break;
        if (node.type != IR::AbstractSyntaxNode::Type::Block)
            continue;
        for (const auto& inst : node.data.block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::IsbeWriteAttribute)
                continue;
            ++stores;
            const auto value = inst.Arg(1).Resolve();
            INFO("data R" << data << ", uniform=" << uniform << ", guarded=" << guarded);
            REQUIRE(value.IsImmediate());
            REQUIRE(value.U32() == (address == data ? 0x20 : Bits));
            const auto actual_address = inst.Arg(0).Resolve();
            REQUIRE(actual_address.IsImmediate());
            REQUIRE(actual_address.U32() == 0x200U + (address == 255    ? 0
                                                      : address == data ? 0x20
                                                                        : 0x40));
        }
    }
    REQUIRE(stores == ((!guarded || condition) ? 1U : 0U));
}
TEST_CASE("Native mesh stores read shadowed high data registers", "[video_core][ampere][mesh]") {
    for (u32 reg : {244U, 245U, 248U, 250U, 252U})
        ExpectStore(reg);
}
TEST_CASE("Native mesh store pressure boundary follows uniform mirrors",
          "[video_core][ampere][mesh]") {
    ExpectStore(216, true);
    ExpectStore(220, true);
    ExpectStore(244, true);
    ExpectStore(215, true);
}
TEST_CASE("Native mesh low stores and aliased high addresses remain correct",
          "[video_core][ampere][mesh]") {
    ExpectStore(4);
    ExpectStore(243);
    ExpectStore(244, false, false, true, 244);
    ExpectStore(248, false, false, true, 244);
}
TEST_CASE("Guarded native mesh stores preserve high data", "[video_core][ampere][mesh]") {
    ExpectStore(244, false, true, true);
    ExpectStore(244, false, true, false);
}
} // namespace
