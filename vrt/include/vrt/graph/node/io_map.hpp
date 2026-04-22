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
 * @file io_map.hpp
 * @brief IOMap — binds concrete graph tokens to a kernel node's typed ports.
 *
 * IOMap is constructed by the user before calling Graph::addNode().  It maps
 * each port name declared in the kernel's IOTypeMap to a concrete value:
 *  - Input scalars   → GraphScalar (constant or global variable)
 *  - Output scalars  → GraphScalar (global variable name, written by the kernel)
 *  - Input buffers   → an existing GraphBuffer token (produced earlier in the graph)
 *  - Output buffers  → a new GraphBuffer token (captured by the caller)
 *  - RW buffers      → an existing input token + a new output token
 *
 * All bind*() methods return *this for method chaining.
 */

#ifndef VRT_GRAPH_NODE_IO_MAP_HPP
#define VRT_GRAPH_NODE_IO_MAP_HPP

#include <atomic>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <vrt/graph/core/graph_buffer.hpp>
#include <vrt/graph/core/graph_scalar.hpp>

namespace vrt::graph {

class Graph;  // friend; reads bindings during compilation

class IOMap {
   public:
    /**
     * @brief Bind a scalar port to a GraphScalar (constant or global variable).
     *
     * Works for both input and output scalar ports.
     */
    IOMap& bindScalar(std::string portName, GraphScalar scalar) {
        scalars_.emplace(std::move(portName), std::move(scalar));
        return *this;
    }

    /**
     * @brief Bind an input buffer port to an existing GraphBuffer token.
     */
    IOMap& bindInputBuffer(std::string portName, GraphBuffer buf) {
        if (!buf.valid()) {
            throw std::invalid_argument("bindInputBuffer: invalid (default-constructed) GraphBuffer");
        }
        inputBuffers_.emplace(std::move(portName), std::move(buf));
        return *this;
    }

    /**
     * @brief Bind an output buffer port; creates and returns a new GraphBuffer token.
     *
     * The caller must capture @p out before using it as an input to subsequent nodes.
     * The token's name is auto-generated; use the returned IOMap& for chaining.
     *
     * @param portName  Port name matching an outputBuffers entry in the IOTypeMap.
     * @param out       Receives the newly created GraphBuffer token.
     */
    IOMap& bindOutputBuffer(std::string portName, GraphBuffer& out) {
        std::string tokenName = nextTokenName(portName);
        out = GraphBuffer(tokenName);
        outputBuffers_.emplace(std::move(portName), out);
        return *this;
    }

    /**
     * @brief Bind an RW buffer port pair: consume @p in, produce a new token into @p out.
     *
     * @p inPortName and @p outPortName must match the in/out sides of the same
     * RWBufferPort entry in the kernel's IOTypeMap.
     *
     * @param inPortName   Port name of the consumed (input) side.
     * @param outPortName  Port name of the produced (output) side.
     * @param in           Existing token to consume.
     * @param out          Receives the newly created output token.
     */
    IOMap& bindRWBuffer(std::string inPortName, std::string outPortName,
                        GraphBuffer in, GraphBuffer& out) {
        if (!in.valid()) {
            throw std::invalid_argument("bindRWBuffer: invalid (default-constructed) input GraphBuffer");
        }
        std::string tokenName = nextTokenName(outPortName);
        out = GraphBuffer(tokenName);
        rwBuffers_.emplace_back(RWBinding{std::move(inPortName), std::move(outPortName),
                                          std::move(in), out});
        return *this;
    }

    // --- Accessors used by GraphCompiler (not part of the public user API) ---

    const std::map<std::string, GraphScalar>&  scalars()       const { return scalars_; }
    const std::map<std::string, GraphBuffer>&  inputBuffers()  const { return inputBuffers_; }
    const std::map<std::string, GraphBuffer>&  outputBuffers() const { return outputBuffers_; }

    struct RWBinding {
        std::string inPort;
        std::string outPort;
        GraphBuffer in;
        GraphBuffer out;
    };
    const std::vector<RWBinding>& rwBuffers() const { return rwBuffers_; }

   private:
    std::string nextTokenName(const std::string& portName) {
        static std::atomic<uint32_t> globalCounter{0};
        return portName + "_buf_" + std::to_string(globalCounter++);
    }

    std::map<std::string, GraphScalar> scalars_;
    std::map<std::string, GraphBuffer> inputBuffers_;
    std::map<std::string, GraphBuffer> outputBuffers_;
    std::vector<RWBinding>             rwBuffers_;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_NODE_IO_MAP_HPP
