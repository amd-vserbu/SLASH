/**
 * The MIT License (MIT)
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <vrt/graph/device/fpga/control_lowering.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vrt::graph::fpga {

namespace {

unsigned scalarBitWidth(ScalarType type) {
    switch (type) {
        case ScalarType::U8:
        case ScalarType::I8:
            return 8;
        case ScalarType::U16:
        case ScalarType::I16:
            return 16;
        case ScalarType::U32:
        case ScalarType::I32:
        case ScalarType::F32:
            return 32;
        case ScalarType::U64:
        case ScalarType::I64:
        case ScalarType::F64:
            return 64;
    }
    return 64;
}

bool isOrderingCompare(CompareOp op) {
    switch (op) {
        case CompareOp::LT:
        case CompareOp::LE:
        case CompareOp::GT:
        case CompareOp::GE:
            return true;
        default:
            return false;
    }
}

/// Flip a comparison operator when the operands are swapped
/// (`a OP b`  <=>  `b FLIP(OP) a`).
CompareOp flipOperands(CompareOp op) {
    switch (op) {
        case CompareOp::LT: return CompareOp::GT;
        case CompareOp::LE: return CompareOp::GE;
        case CompareOp::GT: return CompareOp::LT;
        case CompareOp::GE: return CompareOp::LE;
        default:            return op;  // EQ/NE are symmetric
    }
}

/// Identify the (scalar, constant) operand split.  Returns false unless exactly
/// one operand is a scalar and the other is a constant.
bool splitScalarConstant(const Condition& cond, const ConditionOperand** scalar,
                         const ConditionOperand** constant, bool* scalarIsLhs) {
    if (!cond.lhs() || !cond.rhs()) return false;
    const ConditionOperand& l = *cond.lhs();
    const ConditionOperand& r = *cond.rhs();
    if (l.isScalar() && r.isConstant()) {
        *scalar = &l;
        *constant = &r;
        *scalarIsLhs = true;
        return true;
    }
    if (l.isConstant() && r.isScalar()) {
        *scalar = &r;
        *constant = &l;
        *scalarIsLhs = false;
        return true;
    }
    return false;
}

}  // namespace

bool isRp1EvaluableCondition(const Condition& cond) {
    if (cond.isAlways() || cond.isEpsilonCompare()) return false;

    const ConditionOperand* scalar = nullptr;
    const ConditionOperand* constant = nullptr;
    bool scalarIsLhs = true;
    if (!splitScalarConstant(cond, &scalar, &constant, &scalarIsLhs)) return false;

    const ScalarType type = scalar->type();
    if (!isIntegerScalarType(type)) return false;
    if (scalarBitWidth(type) > 32) return false;  // signal slot is 32-bit

    // Normalise to `scalar OP constant`.
    CompareOp op = cond.op();
    if (!scalarIsLhs) op = flipOperands(op);

    // Ordering comparisons are unsigned on RP1; reject signed scalars.
    if (isOrderingCompare(op) && isSignedIntegerScalarType(type)) return false;

    // Constant must fit in the 32-bit slot.
    const std::uint64_t bits = constant->constantBits();
    if (bits > std::numeric_limits<std::uint32_t>::max()) return false;
    const std::uint32_t value = static_cast<std::uint32_t>(bits);

    // LE/GT are rewritten to LT/GE with value +/- 1; reject if that overflows.
    if (op == CompareOp::LE && value == std::numeric_limits<std::uint32_t>::max()) {
        // `s <= UINT32_MAX` is always true; not a meaningful loop/branch gate.
        return false;
    }
    if (op == CompareOp::GT && value == std::numeric_limits<std::uint32_t>::max()) {
        // `s > UINT32_MAX` is always false.
        return false;
    }
    return true;
}

Rp1Compare mapRp1Condition(const Condition& cond) {
    if (!isRp1EvaluableCondition(cond)) {
        throw std::logic_error("mapRp1Condition: condition is not RP1-evaluable");
    }

    const ConditionOperand* scalar = nullptr;
    const ConditionOperand* constant = nullptr;
    bool scalarIsLhs = true;
    splitScalarConstant(cond, &scalar, &constant, &scalarIsLhs);

    CompareOp op = cond.op();
    if (!scalarIsLhs) op = flipOperands(op);

    Rp1Compare out;
    out.scalarName    = scalar->name();
    out.scalarScopeId = scalar->scopeId();
    out.scalarType    = scalar->type();
    out.value         = static_cast<std::uint32_t>(constant->constantBits());

    switch (op) {
        case CompareOp::EQ: out.op = RP1_COP_EQ; break;
        case CompareOp::NE: out.op = RP1_COP_NE; break;
        case CompareOp::LT: out.op = RP1_COP_LT; break;
        case CompareOp::GE: out.op = RP1_COP_GE; break;
        case CompareOp::LE:  // s <= v  <=>  s < v + 1
            out.op = RP1_COP_LT;
            out.value += 1u;
            break;
        case CompareOp::GT:  // s > v   <=>  s >= v + 1
            out.op = RP1_COP_GE;
            out.value += 1u;
            break;
        default:
            throw std::logic_error("mapRp1Condition: unexpected operator after normalisation");
    }
    return out;
}

rp1_condop_t invertRp1Op(rp1_condop_t op) {
    switch (op) {
        case RP1_COP_EQ:     return RP1_COP_NE;
        case RP1_COP_NE:     return RP1_COP_EQ;
        case RP1_COP_LT:     return RP1_COP_GE;
        case RP1_COP_GE:     return RP1_COP_LT;
        case RP1_COP_AND_NZ: return RP1_COP_AND_Z;
        case RP1_COP_AND_Z:  return RP1_COP_AND_NZ;
    }
    throw std::logic_error("invertRp1Op: unknown rp1_condop_t");
}

std::uint32_t SignalSlotAllocator::alloc() {
    for (std::uint32_t s = 0; s < used_.size(); ++s) {
        if (!used_[s]) {
            used_[s] = true;
            return s;
        }
    }
    throw std::runtime_error("SignalSlotAllocator: RP1 signal array exhausted");
}

std::uint8_t LoopIdAllocator::alloc() {
    if (next_ >= RP1_MAX_LOOPS) {
        throw std::runtime_error("LoopIdAllocator: exceeded RP1_MAX_LOOPS loop ids");
    }
    return static_cast<std::uint8_t>(next_++);
}

}  // namespace vrt::graph::fpga
