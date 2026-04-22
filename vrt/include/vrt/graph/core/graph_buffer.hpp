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
 * @brief GraphBuffer — opaque buffer token used in graph construction.
 *
 * A GraphBuffer is a first-class value in the graph.  The compiler resolves
 * each token to a concrete, device-allocated buffer at compile time.
 *
 * Tokens are produced in two ways:
 *  - Graph::inputBuffer()         — graph-level inputs (no producer node)
 *  - IOMap::bindOutputBuffer()    — output of a kernel node
 *  - IOMap::bindRWBuffer()        — output side of an in-place RW operation
 *
 * A default-constructed GraphBuffer is invalid (valid() == false).
 */

#ifndef VRT_GRAPH_CORE_GRAPH_BUFFER_HPP
#define VRT_GRAPH_CORE_GRAPH_BUFFER_HPP

#include <string>

namespace vrt::graph {

class Graph;  // friend; creates tokens
class IOMap;  // friend; creates output tokens

class GraphBuffer {
   public:
    GraphBuffer() = default;

    /**
     * @brief Returns the logical name of this buffer (unique within its Graph).
     */
    const std::string& name() const { return name_; }

    /**
     * @brief Returns false for default-constructed (unbound) tokens.
     */
    bool valid() const { return !name_.empty(); }

   private:
    friend class Graph;
    friend class IOMap;
    friend class BridgeRouter;

    explicit GraphBuffer(std::string name) : name_(std::move(name)) {}

    std::string name_;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_CORE_GRAPH_BUFFER_HPP
