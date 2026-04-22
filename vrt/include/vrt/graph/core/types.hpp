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
 * @file types.hpp
 * @brief Shared enums for the VRT graph API.
 */

#ifndef VRT_GRAPH_CORE_TYPES_HPP
#define VRT_GRAPH_CORE_TYPES_HPP

namespace vrt::graph {

/**
 * @brief Element type of a scalar kernel argument.
 */
enum class ScalarType {
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    F32, F64,
};

/**
 * @brief Element type of a buffer kernel argument.
 */
enum class BufferType {
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    F32, F64,
};

/**
 * @brief Class of device that a kernel runs on.
 */
enum class DeviceType {
    CPU,
    GPU,
    FPGA,
    MOCK_CPU,
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_CORE_TYPES_HPP
