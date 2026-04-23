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
 * @file graph_buffer.hpp
 * @brief GraphBuffer — typed, opaque buffer token used in graph construction.
 *
 * A GraphBuffer is a first-class value in the graph: a (name, element-type)
 * pair that the compiler resolves to a concrete, device-allocated buffer at
 * compile time.
 *
 * Tokens are minted via the public factory:
 *   GraphBuffer::make(BufferType, std::string)
 *
 * In normal usage the factory is invoked indirectly through:
 *  - Graph::inputBuffer()         — graph-level inputs (no producer node)
 *  - IOMap::bindOutputBuffer()    — output of a kernel node
 *  - IOMap::bindRWBuffer()        — output side of an in-place RW operation
 *
 * A default-constructed GraphBuffer is invalid (valid() == false).
 */

#ifndef VRT_GRAPH_CORE_GRAPH_BUFFER_HPP
#define VRT_GRAPH_CORE_GRAPH_BUFFER_HPP

#include <stdexcept>
#include <string>
#include <utility>

#include <vrt/graph/core/types.hpp>

namespace vrt::graph {

class GraphBuffer {
   public:
    GraphBuffer() = default;

    /**
     * @brief Mint a new, valid buffer token.
     *
     * @param type  Element type of the buffer.
     * @param name  Logical name (unique within its Graph).  Must be non-empty.
     * @throws std::invalid_argument if @p name is empty.
     */
    static GraphBuffer make(BufferType type, std::string name) {
        if (name.empty()) {
            throw std::invalid_argument(
                "GraphBuffer::make: name must not be empty");
        }
        return GraphBuffer(type, std::move(name));
    }

    /**
     * @brief Returns the logical name of this buffer (unique within its Graph).
     */
    const std::string& name() const { return name_; }

    /**
     * @brief Returns the element type of this buffer.
     *
     * For a default-constructed (invalid) token the returned value is
     * unspecified and should not be relied upon.
     */
    BufferType type() const { return type_; }

    /**
     * @brief Returns false for default-constructed (unbound) tokens.
     */
    bool valid() const { return !name_.empty(); }

   private:
    GraphBuffer(BufferType type, std::string name)
        : type_(type), name_(std::move(name)) {}

    BufferType  type_ = BufferType::U8;  // placeholder for default-constructed tokens
    std::string name_;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_CORE_GRAPH_BUFFER_HPP
