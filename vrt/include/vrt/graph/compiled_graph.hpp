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
 * @file compiled_graph.hpp
 * @brief CompiledGraph — executable graph snapshot produced from an authored Graph.
 *
 * Graph owns authoring state. CompiledGraph owns the lowered top-level DGraphs,
 * the scalar state visible to those DGraphs, bridge instances needed by compiled
 * bridge operations, and the device plans compiled from the DGraphs.
 */

#ifndef VRT_GRAPH_COMPILED_GRAPH_HPP
#define VRT_GRAPH_COMPILED_GRAPH_HPP

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <vrt/graph/core/graph_scalar.hpp>
#include <vrt/graph/crossdevice/bridge.hpp>
#include <vrt/graph/device/device.hpp>
#include <vrt/graph/device/dgraph.hpp>

namespace vrt::graph {

/**
 * @brief Executable compiled graph snapshot.
 *
 * This class is move-only because device plans are move-only. It owns the
 * top-level per-device plans; nested child plans remain owned by the device
 * plan implementations that compile those child DGraphs.
 */
class CompiledGraph {
   public:
    CompiledGraph(std::vector<DGraph> dgraphs,
                  std::shared_ptr<std::map<std::string, uint64_t>> scalarValues,
                  std::map<std::string, ScalarType> scalarTypes,
                  std::vector<std::shared_ptr<IBridge>> bridgePins)
        : dgraphs_(std::move(dgraphs)),
          scalarValues_(std::move(scalarValues)),
          scalarTypes_(std::move(scalarTypes)),
          bridgePins_(std::move(bridgePins)) {
        if (!scalarValues_) {
            scalarValues_ = std::make_shared<std::map<std::string, uint64_t>>();
        }
        plans_.reserve(dgraphs_.size());
        for (const auto& dg : dgraphs_) {
            plans_.push_back(dg.device->compilePlan(dg));
        }
    }

    CompiledGraph(const CompiledGraph&) = delete;
    CompiledGraph& operator=(const CompiledGraph&) = delete;
    CompiledGraph(CompiledGraph&&) noexcept = default;
    CompiledGraph& operator=(CompiledGraph&&) noexcept = default;
    ~CompiledGraph() = default;

    /**
     * @brief Returns the lowered per-device DGraphs.
     */
    const std::vector<DGraph>& dgraphs() const { return dgraphs_; }

    /**
     * @brief Set a root-scope scalar in this compiled snapshot from raw bits.
     */
    void setScalarBits(const std::string& name, uint64_t bits) {
        requireScalar(name, "CompiledGraph::setScalarBits");
        (*scalarValues_)[scopedScalarKey(rootScopeId_, name)] = bits;
    }

    /**
     * @brief Read a root-scope scalar in this compiled snapshot as raw bits.
     */
    uint64_t scalarBits(const std::string& name) const {
        requireScalar(name, "CompiledGraph::scalarBits");
        const std::string key = scopedScalarKey(rootScopeId_, name);
        auto valueIt = scalarValues_->find(key);
        if (valueIt == scalarValues_->end()) {
            throw std::out_of_range(
                "CompiledGraph::scalarBits: scalar '" + name + "' has no value");
        }
        return valueIt->second;
    }

    template <class T>
    void setScalar(const std::string& name, T value) {
        static_assert(std::is_arithmetic_v<T>,
                      "CompiledGraph::setScalar only supports arithmetic types");
        requireScalarType<T>(name, "CompiledGraph::setScalar");
        setScalarBits(name, detail::valueToBits(value));
    }

    template <class T>
    T getScalar() const = delete;

    template <class T>
    T getScalar(const std::string& name) const {
        static_assert(std::is_arithmetic_v<T>,
                      "CompiledGraph::getScalar only supports arithmetic types");
        requireScalarType<T>(name, "CompiledGraph::getScalar");
        uint64_t bits = scalarBits(name);
        T value{};
        std::memcpy(&value, &bits, sizeof(T));
        return value;
    }

    /**
     * @brief Start asynchronous execution on every top-level device plan.
     */
    void launch() {
        for (auto& plan : plans_) {
            plan->launch();
        }
    }

    /**
     * @brief Wait for every top-level device plan to complete.
     */
    void wait() {
        for (auto& plan : plans_) {
            plan->wait();
        }
    }

    /**
     * @brief Execute synchronously. Equivalent to launch() followed by wait().
     */
    void run() {
        launch();
        wait();
    }

   private:
    void requireScalar(const std::string& name, const char* method) const {
        if (scalarTypes_.find(name) == scalarTypes_.end()) {
            throw std::out_of_range(
                std::string(method) + ": unknown scalar '" + name + "'");
        }
    }

    template <class T>
    void requireScalarType(const std::string& name, const char* method) const {
        auto declaredType = scalarTypes_.find(name);
        if (declaredType == scalarTypes_.end()) {
            throw std::out_of_range(
                std::string(method) + ": unknown scalar '" + name + "'");
        }
        if (declaredType->second != typeToScalarType<T>()) {
            throw std::invalid_argument(
                std::string(method) + ": type mismatch for scalar '" + name + "'");
        }
    }

    static constexpr uint64_t rootScopeId_ = 0;

    std::vector<DGraph>                       dgraphs_;
    std::shared_ptr<std::map<std::string, uint64_t>> scalarValues_;
    std::map<std::string, ScalarType>         scalarTypes_;
    std::vector<std::shared_ptr<IBridge>>     bridgePins_;
    std::vector<std::unique_ptr<IDevicePlan>> plans_;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_COMPILED_GRAPH_HPP
