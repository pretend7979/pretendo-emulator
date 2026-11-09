// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/program.h"

namespace Shader::Optimization {

/// Proves finite u32 offsets without treating unknown branches or loop recurrences as constants.
/// Construct lazily for a native compute program that actually uses indirect constant-buffer reads.
class BoundedNativeOffsets {
  public:
    using PathCondition = u32;
    static constexpr PathCondition UnrestrictedPath = 1;

    explicit BoundedNativeOffsets(const IR::Program &program) {
        if (program.blocks.empty()) {
            return;
        }
        if (program.blocks.size() > MaxGraphNodes) {
            throw NotImplementedException("Native pointer offset CFG budget exceeded");
        }
        entry = program.blocks.front();
        std::unordered_map<const IR::Block *, size_t> block_indices;
        std::vector<const IR::Inst *> copies;
        std::unordered_map<const IR::Inst *, size_t> copy_indices;
        for (const auto *block : program.blocks) {
            block_indices.emplace(block, block_indices.size());
            for (const auto &inst : block->Instructions()) {
                Spend();
                if (owners.size() >= MaxGraphNodes) {
                    throw NotImplementedException("Native pointer offset IR budget exceeded");
                }
                owners.emplace(&inst, block);
                if (IsCopy(inst)) {
                    copy_indices.emplace(&inst, copies.size());
                    copies.push_back(&inst);
                }
            }
        }
        std::vector<std::vector<size_t>> copy_edges(copies.size());
        for (size_t i = 0; i < copies.size(); ++i) {
            const auto *inst = copies[i];
            for (size_t a = 0; a < inst->NumArgs(); ++a) {
                if (const auto found = copy_indices.find(RawInst(inst->Arg(a)));
                    found != copy_indices.end()) {
                    copy_edges[i].push_back(found->second);
                }
            }
        }
        // SCCs are returned after their dependencies. Collapse only copies with one external value.
        for (const auto &group : Components(copy_edges)) {
            std::unordered_set<const IR::Inst *> inside;
            for (const size_t i : group) {
                inside.insert(copies[i]);
            }
            std::optional<IR::Value> external;
            bool agrees = true;
            for (const size_t i : group) {
                const auto *inst = copies[i];
                for (size_t a = 0; a < inst->NumArgs(); ++a) {
                    const IR::Value value = inst->Arg(a);
                    if (inside.contains(RawInst(value))) {
                        continue;
                    }
                    const auto candidate = Canonical(value);
                    if (external && *external != candidate) {
                        agrees = false;
                    }
                    external = candidate;
                }
            }
            if (agrees && external) {
                for (const size_t i : group) {
                    aliases.emplace(copies[i], *external);
                }
            }
        }
        std::vector<std::vector<size_t>> cfg(program.blocks.size());
        for (size_t i = 0; i < program.blocks.size(); ++i) {
            for (const auto *next : program.blocks[i]->ImmSuccessors()) {
                if (const auto found = block_indices.find(next); found != block_indices.end()) {
                    cfg[i].push_back(found->second);
                }
            }
        }
        for (const auto &group : Components(cfg)) {
            if (group.size() > 1 ||
                std::ranges::find(cfg[group[0]], group[0]) != cfg[group[0]].end()) {
                for (const size_t i : group) {
                    loop_blocks.insert(program.blocks[i]);
                }
            }
        }
        const IR::Block *current{};
        for (const auto &syntax : program.syntax_list) {
            const auto branch = [&](const IR::Block *yes, const IR::Block *no,
                                    IR::Value condition) {
                if (!current) {
                    return;
                }
                if (yes == no) {
                    unconditional_edges.insert({current, yes});
                    edge_conditions.erase({current, yes});
                    return;
                }
                AddEdgeCondition(current, yes, condition, true);
                AddEdgeCondition(current, no, condition, false);
            };
            using Type = IR::AbstractSyntaxNode::Type;
            switch (syntax.type) {
            case Type::Block:
                current = syntax.data.block;
                break;
            case Type::If:
                branch(syntax.data.if_node.body, syntax.data.if_node.merge,
                       syntax.data.if_node.cond);
                break;
            case Type::Repeat:
                branch(syntax.data.repeat.loop_header, syntax.data.repeat.merge,
                       syntax.data.repeat.cond);
                break;
            case Type::Break:
                branch(syntax.data.break_node.merge, syntax.data.break_node.skip,
                       syntax.data.break_node.cond);
                break;
            default:
                break;
            }
        }
    }

    /// Nullopt means some reachable value remains unknown; an empty vector means unreachable.
    [[nodiscard]] std::optional<std::vector<u32>> Offsets(IR::Value value, const IR::Block *use) {
        auto values = Evaluate(value);
        if (use) {
            values = Scale(values, Reach(use));
        }
        if (values.contains(Unknown)) {
            return std::nullopt;
        }
        std::vector<u32> result;
        for (const auto &[number, condition] : values) {
            if (condition) {
                result.push_back(static_cast<u32>(number));
            }
        }
        return result;
    }

    /// Resolves only proven Phi/Identity copies, including cycles with a unique external seed.
    [[nodiscard]] IR::Value Canonical(IR::Value value) {
        while (const auto *inst = RawInst(value)) {
            Spend();
            const auto found = aliases.find(inst);
            if (found == aliases.end()) {
                break;
            }
            value = found->second;
        }
        return value;
    }

    /// False is a proof that this incoming edge cannot occur at the use, not a reachability guess.
    [[nodiscard]] bool MayReachEdge(const IR::Block *from, const IR::Block *phi_block,
                                    const IR::Block *use) {
        return ConstrainPathToEdge(UnrestrictedPath, from, phi_block, use) != False;
    }

    /// Retains the joint path while following separately predicated address words.
    [[nodiscard]] PathCondition ConstrainPathToEdge(PathCondition path, const IR::Block *from,
                                                   const IR::Block *phi_block,
                                                   const IR::Block *use) {
        return And(path, And(Edge(from, phi_block), Reach(use)));
    }

    [[nodiscard]] size_t NodeCount() const noexcept { return nodes.size(); }
    [[nodiscard]] size_t OperationCount() const noexcept { return operations; }

  private:
    using Condition = u32;
    using Values = std::map<u64, Condition>;
    using Boolean = std::array<Condition, 2>; // possible false, possible true; may overlap
    static constexpr Condition False = 0;
    static constexpr Condition True = 1;
    static constexpr u64 Unknown = u64{1} << 32;
    static constexpr size_t MaxNodes = 1'000'000;
    static constexpr size_t MaxGraphNodes = 262'144;
    static constexpr size_t MaxOperations = 20'000'000;
    static constexpr size_t MaxRecursion = 512;
    static constexpr size_t MaxValues = 128;
    struct Node {
        u32 variable{};
        Condition low{};
        Condition high{};
        auto operator<=>(const Node &) const = default;
    };
    struct DepthGuard {
        explicit DepthGuard(BoundedNativeOffsets &owner_) : owner{owner_} {
            owner.Spend();
            if (++owner.depth > MaxRecursion) {
                --owner.depth;
                throw NotImplementedException("Native pointer offset recursion budget exceeded");
            }
        }
        ~DepthGuard() { --owner.depth; }
        BoundedNativeOffsets &owner;
    };

    void Spend() {
        if (++operations > MaxOperations) {
            throw NotImplementedException("Native pointer offset operation budget exceeded");
        }
    }
    void AddEdgeCondition(const IR::Block *from, const IR::Block *to, IR::Value condition,
                          bool outcome) {
        const auto key = std::pair{from, to};
        if (unconditional_edges.contains(key)) {
            return;
        }
        const auto [it, inserted] = edge_conditions.emplace(key, std::pair{condition, outcome});
        if (!inserted && (it->second.first != condition || it->second.second != outcome)) {
            // Multiple structural exits can name one CFG edge. Never retain just the last guard.
            edge_conditions.erase(it);
            unconditional_edges.insert(key);
        }
    }
    static const IR::Inst *RawInst(IR::Value value) {
        // Avoid IsImmediate/Resolve following a cyclic Identity chain.
        return value.IsIdentity() || value.IsPhi() || !value.IsImmediate() ? value.Inst() : nullptr;
    }
    static bool IsCopy(const IR::Inst &inst) {
        return inst.GetOpcode() == IR::Opcode::Phi || inst.GetOpcode() == IR::Opcode::Identity;
    }
    std::vector<std::vector<size_t>> Components(const std::vector<std::vector<size_t>> &graph) {
        // Iterative Tarjan: shader size cannot consume the host call stack during SCC discovery.
        constexpr size_t Unvisited = std::numeric_limits<size_t>::max();
        std::vector<size_t> number(graph.size(), Unvisited), low(graph.size()), stack;
        std::vector<bool> active(graph.size());
        std::vector<std::vector<size_t>> groups;
        struct Frame {
            size_t vertex;
            size_t next;
        };
        std::vector<Frame> dfs;
        size_t serial{};
        const auto enter = [&](size_t v) {
            number[v] = low[v] = serial++;
            stack.push_back(v);
            active[v] = true;
            dfs.push_back({v, 0});
        };
        for (size_t root = 0; root < graph.size(); ++root) {
            if (number[root] != Unvisited) {
                continue;
            }
            enter(root);
            while (!dfs.empty()) {
                Spend();
                auto &frame = dfs.back();
                const size_t v = frame.vertex;
                if (frame.next < graph[v].size()) {
                    const size_t child = graph[v][frame.next++];
                    if (number[child] == Unvisited) {
                        enter(child);
                    } else if (active[child]) {
                        low[v] = std::min(low[v], number[child]);
                    }
                    continue;
                }
                if (low[v] == number[v]) {
                    std::vector<size_t> group;
                    for (;;) {
                        const size_t child = stack.back();
                        stack.pop_back();
                        active[child] = false;
                        group.push_back(child);
                        if (child == v) {
                            break;
                        }
                    }
                    groups.push_back(std::move(group));
                }
                dfs.pop_back();
                if (!dfs.empty()) {
                    low[dfs.back().vertex] = std::min(low[dfs.back().vertex], low[v]);
                }
            }
        }
        return groups;
    }
    Condition MakeNode(u32 variable, Condition low, Condition high) {
        if (low == high) {
            return low;
        }
        const Node key{variable, low, high};
        if (const auto found = unique.find(key); found != unique.end()) {
            return found->second;
        }
        if (nodes.size() >= MaxNodes) {
            throw NotImplementedException("Native pointer offset BDD budget exceeded");
        }
        const Condition result = static_cast<Condition>(nodes.size());
        unique.emplace(key, result);
        nodes.push_back(key);
        return result;
    }
    Condition Not(Condition condition) {
        DepthGuard guard{*this};
        if (condition < 2) {
            return 1 - condition;
        }
        if (const auto found = negations.find(condition); found != negations.end()) {
            return found->second;
        }
        const auto node = nodes[condition];
        const auto result = MakeNode(node.variable, Not(node.low), Not(node.high));
        negations.emplace(condition, result);
        return result;
    }
    Condition And(Condition a, Condition b) {
        DepthGuard guard{*this};
        if (!a || !b) {
            return False;
        }
        if (a == True) {
            return b;
        }
        if (b == True || a == b) {
            return a;
        }
        if (a > b) {
            std::swap(a, b);
        }
        const u64 key = (u64{a} << 32) | b;
        if (const auto found = conjunctions.find(key); found != conjunctions.end()) {
            return found->second;
        }
        const Node x = nodes[a], y = nodes[b];
        const u32 variable = std::min(x.variable, y.variable);
        const auto low =
            And(x.variable == variable ? x.low : a, y.variable == variable ? y.low : b);
        const auto high =
            And(x.variable == variable ? x.high : a, y.variable == variable ? y.high : b);
        const auto result = MakeNode(variable, low, high);
        conjunctions.emplace(key, result);
        return result;
    }
    Condition Or(Condition a, Condition b) { return Not(And(Not(a), Not(b))); }
    Condition Widen(Condition condition) {
        if (condition < 2) {
            return condition;
        }
        if (const auto found = small_conditions.find(condition); found != small_conditions.end()) {
            return found->second ? condition : True;
        }
        std::vector<Condition> pending{condition};
        std::unordered_set<Condition> visited;
        while (!pending.empty()) {
            Spend();
            const auto next = pending.back();
            pending.pop_back();
            if (next < 2 || !visited.insert(next).second) {
                continue;
            }
            if (visited.size() > 256) {
                small_conditions.emplace(condition, false);
                return True;
            }
            pending.push_back(nodes[next].low);
            pending.push_back(nodes[next].high);
        }
        small_conditions.emplace(condition, true);
        return condition;
    }
    Values Merge(Values a, const Values &b) {
        for (const auto &[value, condition] : b) {
            if (condition) {
                a[value] = Or(a[value], condition);
            }
        }
        return a;
    }
    Values Scale(const Values &values, Condition condition) {
        Values result;
        for (const auto &[value, possible] : values) {
            if (const auto both = And(possible, condition)) {
                result.emplace(value, both);
            }
        }
        return result;
    }
    Condition Reach(const IR::Block *block) {
        DepthGuard guard{*this};
        if (!block || block == entry) {
            return True;
        }
        if (const auto found = reach_cache.find(block); found != reach_cache.end()) {
            return found->second;
        }
        if (!reach_active.insert(block).second) {
            return True;
        }
        Condition result = False;
        for (const auto *predecessor : block->ImmPredecessors()) {
            result = Widen(Or(result, Edge(predecessor, block)));
        }
        reach_active.erase(block);
        reach_cache.emplace(block, result);
        return result;
    }
    Condition Edge(const IR::Block *from, const IR::Block *to) {
        Condition result = Reach(from);
        if (const auto found = edge_conditions.find({from, to}); found != edge_conditions.end()) {
            result = And(result, PossibleBoolean(found->second.first)[found->second.second]);
        }
        return Widen(result);
    }
    Boolean PossibleBoolean(IR::Value value) {
        DepthGuard guard{*this};
        value = Canonical(value);
        const auto *inst = RawInst(value);
        if (!inst) {
            if (!value.IsEmpty() && value.Type() == IR::Type::U1) {
                return value.U1() ? Boolean{False, True} : Boolean{True, False};
            }
            if (!value.IsEmpty() && value.Type() == IR::Type::U32 && value.U32() <= 1) {
                return value.U32() ? Boolean{False, True} : Boolean{True, False};
            }
            return {True, True};
        }
        if (const auto found = boolean_cache.find(inst); found != boolean_cache.end()) {
            return found->second;
        }
        if (!boolean_active.insert(inst).second) {
            return {True, True};
        }
        Boolean result;
        if (inst->GetOpcode() == IR::Opcode::Identity ||
            inst->GetOpcode() == IR::Opcode::ConditionRef) {
            result = PossibleBoolean(inst->Arg(0));
        } else if (inst->GetOpcode() == IR::Opcode::LogicalNot) {
            const auto arg = PossibleBoolean(inst->Arg(0));
            result = {arg[1], arg[0]};
        } else {
            auto values = Evaluate(value);
            const Condition unknown = values[Unknown];
            Boolean uncertain{False, False};
            if (unknown) {
                const auto owner = owners.find(inst);
                if (inst->GetOpcode() == IR::Opcode::UndefU1 || owner == owners.end() ||
                    loop_blocks.contains(owner->second)) {
                    uncertain = {unknown, unknown};
                } else {
                    const auto [it, inserted] = atoms.emplace(inst, static_cast<u32>(atoms.size()));
                    const auto atom = MakeNode(it->second, False, True);
                    uncertain = {And(unknown, Not(atom)), And(unknown, atom)};
                }
            }
            result = {Or(values[0], uncertain[0]), Or(values[1], uncertain[1])};
        }
        boolean_active.erase(inst);
        boolean_cache.emplace(inst, result);
        return result;
    }
    static std::optional<u32> Binary(IR::Opcode op, u32 a, u32 b) {
        using Op = IR::Opcode;
        switch (op) {
        case Op::IAdd32:
            return a + b;
        case Op::ISub32:
            return a - b;
        case Op::IMul32:
            return a * b;
        case Op::ShiftLeftLogical32:
            return b < 32 ? std::optional<u32>{a << b} : std::nullopt;
        case Op::ShiftRightLogical32:
            return b < 32 ? std::optional<u32>{a >> b} : std::nullopt;
        case Op::BitwiseAnd32:
            return a & b;
        case Op::BitwiseOr32:
            return a | b;
        case Op::BitwiseXor32:
            return a ^ b;
        case Op::UGreaterThan:
            return a > b;
        case Op::UGreaterThanEqual:
            return a >= b;
        case Op::ULessThan:
            return a < b;
        case Op::ULessThanEqual:
            return a <= b;
        case Op::IEqual:
            return a == b;
        case Op::INotEqual:
            return a != b;
        default:
            return std::nullopt;
        }
    }
    Values Evaluate(IR::Value value) {
        DepthGuard guard{*this};
        value = Canonical(value);
        const auto *inst = RawInst(value);
        if (!inst) {
            if (!value.IsEmpty() && value.Type() == IR::Type::U32) {
                return {{value.U32(), True}};
            }
            if (!value.IsEmpty() && value.Type() == IR::Type::U1) {
                return {{value.U1(), True}};
            }
            return {{Unknown, True}};
        }
        if (const auto found = value_cache.find(inst); found != value_cache.end()) {
            return found->second;
        }
        if (!value_active.insert(inst).second) {
            return {{Unknown, True}};
        }
        const auto arg = [&](size_t index) { return Evaluate(inst->Arg(index)); };
        using Op = IR::Opcode;
        Values result;
        switch (inst->GetOpcode()) {
        case Op::Identity:
        case Op::ConditionRef:
            result = arg(0);
            break;
        case Op::Phi:
            if (const auto owner = owners.find(inst); owner != owners.end()) {
                for (size_t i = 0; i < inst->NumArgs(); ++i) {
                    const auto incoming = arg(i);
                    const auto incoming_edge = Edge(inst->PhiBlock(i), owner->second);
                    result = Merge(std::move(result), Scale(incoming, incoming_edge));
                }
            } else {
                result = {{Unknown, True}};
            }
            break;
        case Op::SelectU1:
        case Op::SelectU32: {
            const auto condition = PossibleBoolean(inst->Arg(0));
            const auto yes = Scale(arg(1), condition[1]);
            const auto no = Scale(arg(2), condition[0]);
            result = Merge(yes, no);
            break;
        }
        case Op::GetZeroFromOp:
            for (const auto &[number, condition] : arg(0)) {
                const u64 output = number == Unknown ? Unknown : number == 0;
                result[output] = Or(result[output], condition);
            }
            break;
        case Op::INeg32:
            for (const auto &[number, condition] : arg(0)) {
                const u64 output = number == Unknown ? Unknown : u32{0} - static_cast<u32>(number);
                result[output] = Or(result[output], condition);
            }
            break;
        case Op::BitFieldUExtract: {
            const auto offset = Canonical(inst->Arg(1)), width = Canonical(inst->Arg(2));
            if (!RawInst(offset) && !RawInst(width) && offset.Type() == IR::Type::U32 &&
                width.Type() == IR::Type::U32 && offset.U32() < 32 &&
                width.U32() <= 32 - offset.U32()) {
                const u32 mask = width.U32() == 32 ? ~u32{0} : (u32{1} << width.U32()) - 1;
                for (const auto &[number, condition] : arg(0)) {
                    const u64 output =
                        number == Unknown ? Unknown : (number >> offset.U32()) & mask;
                    result[output] = Or(result[output], condition);
                }
            } else {
                result = {{Unknown, True}};
            }
            break;
        }
        case Op::ShuffleIndex:
            // The source lane need not satisfy this lane's predicates. Retain its values, not
            // guards.
            for (const auto &[number, condition] : arg(0)) {
                if (condition) {
                    result[number] = True;
                }
            }
            break;
        case Op::LogicalNot: {
            const auto a = PossibleBoolean(inst->Arg(0));
            result = {{0, a[1]}, {1, a[0]}};
            break;
        }
        case Op::LogicalAnd:
        case Op::LogicalOr:
        case Op::LogicalXor: {
            const auto a = PossibleBoolean(inst->Arg(0)), b = PossibleBoolean(inst->Arg(1));
            if (inst->GetOpcode() == Op::LogicalAnd) {
                result = {{0, Or(a[0], b[0])}, {1, And(a[1], b[1])}};
            } else if (inst->GetOpcode() == Op::LogicalOr) {
                result = {{0, And(a[0], b[0])}, {1, Or(a[1], b[1])}};
            } else {
                result = {{0, Or(And(a[0], b[0]), And(a[1], b[1]))},
                          {1, Or(And(a[0], b[1]), And(a[1], b[0]))}};
            }
            break;
        }
        case Op::IAdd32:
        case Op::ISub32:
        case Op::IMul32:
        case Op::ShiftLeftLogical32:
        case Op::ShiftRightLogical32:
        case Op::BitwiseAnd32:
        case Op::BitwiseOr32:
        case Op::BitwiseXor32:
        case Op::UGreaterThan:
        case Op::UGreaterThanEqual:
        case Op::ULessThan:
        case Op::ULessThanEqual:
        case Op::IEqual:
        case Op::INotEqual: {
            const auto a = arg(0), b = arg(1);
            for (const auto &[x, cx] : a) {
                for (const auto &[y, cy] : b) {
                    Spend();
                    if (const auto condition = And(cx, cy)) {
                        const auto output = x == Unknown || y == Unknown
                                                ? std::nullopt
                                                : Binary(inst->GetOpcode(), static_cast<u32>(x),
                                                         static_cast<u32>(y));
                        const u64 key = output ? u64{*output} : Unknown;
                        result[key] = Or(result[key], condition);
                    }
                }
            }
            break;
        }
        default:
            result = {{Unknown, True}};
            break;
        }
        if (result.size() > MaxValues) {
            result = {{Unknown, True}};
        }
        for (auto it = result.begin(); it != result.end();) {
            if (!it->second) {
                it = result.erase(it);
            } else {
                it->second = Widen(it->second);
                ++it;
            }
        }
        value_active.erase(inst);
        value_cache.emplace(inst, result);
        return result;
    }

    const IR::Block *entry{};
    size_t operations{};
    size_t depth{};
    std::unordered_map<const IR::Inst *, const IR::Block *> owners;
    std::unordered_map<const IR::Inst *, IR::Value> aliases;
    std::unordered_set<const IR::Block *> loop_blocks;
    std::map<std::pair<const IR::Block *, const IR::Block *>, std::pair<IR::Value, bool>>
        edge_conditions;
    std::set<std::pair<const IR::Block *, const IR::Block *>> unconditional_edges;
    std::vector<Node> nodes{{}, {}};
    std::map<Node, Condition> unique;
    std::unordered_map<Condition, Condition> negations;
    std::unordered_map<u64, Condition> conjunctions;
    std::unordered_map<Condition, bool> small_conditions;
    std::unordered_map<const IR::Inst *, u32> atoms;
    std::unordered_map<const IR::Block *, Condition> reach_cache;
    std::unordered_map<const IR::Inst *, Values> value_cache;
    std::unordered_map<const IR::Inst *, Boolean> boolean_cache;
    std::unordered_set<const IR::Block *> reach_active;
    std::unordered_set<const IR::Inst *> value_active;
    std::unordered_set<const IR::Inst *> boolean_active;
};

} // namespace Shader::Optimization
