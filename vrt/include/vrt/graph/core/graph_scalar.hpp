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

/**
 * @file graph_scalar.hpp
 * @brief GraphScalar — a typed scalar value used in IOMap bindings.
 *
 * A GraphScalar is either:
 *  - A compile-time constant (created via GraphScalar::constant()), or
 *  - A reference to a named global variable (created via GraphScalar::globalVar()).
 *
 * Values are stored as raw uint64_t bits; ScalarType governs their interpretation.
 * No templates are used to avoid metaprogramming complexity.
 */

#ifndef VRT_GRAPH_CORE_GRAPH_SCALAR_HPP
#define VRT_GRAPH_CORE_GRAPH_SCALAR_HPP

#include <stdexcept>
#include <string>

#include <vrt/graph/core/types.hpp>

namespace vrt::graph {

class GraphScalar {
   public:
    /**
     * @brief Creates a compile-time constant scalar.
     *
     * @param type   Element type governing bit interpretation.
     * @param bits   Raw bit pattern of the value (e.g. reinterpret_cast a float to uint32_t).
     */
    static GraphScalar constant(ScalarType type, uint64_t bits) {
        return GraphScalar(type, bits, "");
    }

    /**
     * @brief Creates a reference to a named global variable.
     *
     * Used primarily for output scalars whose value is written by a kernel and
     * consumed by another node or read back by the host.
     *
     * @param type      Element type.
     * @param varName   Name of the global variable (must be unique within the Graph).
     */
    static GraphScalar globalVar(ScalarType type, std::string varName) {
        if (varName.empty()) {
            throw std::invalid_argument("GraphScalar::globalVar: varName must not be empty");
        }
        return GraphScalar(type, 0, std::move(varName));
    }

    /**
     * @brief Returns the element type.
     */
    ScalarType type() const { return type_; }

    /**
     * @brief Returns true if this scalar is a compile-time constant.
     */
    bool isConstant() const { return varName_.empty(); }

    /**
     * @brief Returns the raw bit pattern.  Only valid when isConstant() == true.
     */
    uint64_t constantBits() const {
        if (!isConstant()) {
            throw std::logic_error("GraphScalar::constantBits() called on a globalVar scalar");
        }
        return bits_;
    }

    /**
     * @brief Returns the global variable name.  Only valid when isConstant() == false.
     */
    const std::string& varName() const {
        if (isConstant()) {
            throw std::logic_error("GraphScalar::varName() called on a constant scalar");
        }
        return varName_;
    }

   private:
    GraphScalar(ScalarType type, uint64_t bits, std::string varName)
        : type_(type), bits_(bits), varName_(std::move(varName)) {}

    ScalarType  type_;
    uint64_t    bits_;
    std::string varName_;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_CORE_GRAPH_SCALAR_HPP
