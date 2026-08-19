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

#define RP1_LATENCY_TESTING
#include "rp1_latency.cpp"

/**
 * @brief Verify result validation, flush accounting, and PMU wrap.
 */
int main() {
    vrt::graph::fpga::Rp1GraphResult result;
    result.flags = RP1_RESULT_TRACE_ENABLED;
    result.activeImageId = 1u;
    result.imageState = vrt::graph::fpga::Rp1ImageState::Known;
    result.completedOperations = 3u;
    result.graphElapsedTicks = 161u;
    result.publishElapsedTicks = 170u;
    result.traceWriteIndex = 10u;
    try {
        validateGraphResult(result, 3u, /*traceExpected=*/true);
    } catch (const std::runtime_error&) {
        return 1;
    }

    result.flags |= RP1_RESULT_RECOVERY_REQUIRED;
    try {
        validateGraphResult(result, 3u, /*traceExpected=*/true);
        return 1;
    } catch (const std::runtime_error&) {
        result.flags &= ~RP1_RESULT_RECOVERY_REQUIRED;
    }

    result.flags |= RP1_RESULT_INFINITE_WORK_REMAINS;
    try {
        validateGraphResult(result, 3u, /*traceExpected=*/true);
        return 1;
    } catch (const std::runtime_error&) {
        result.flags &= ~RP1_RESULT_INFINITE_WORK_REMAINS;
    }

    vrt::graph::fpga::Rp1TraceCapture capture;
    const auto add = [&](std::uint16_t event, std::uint16_t node,
                         std::uint32_t timestamp) {
        capture.entries.emplace_back();
        rp1_trace_entry_t& entry = capture.entries.back();
        entry.event = event;
        entry.node_index = node;
        entry.timestamp = timestamp;
    };

    const std::uint32_t max = std::numeric_limits<std::uint32_t>::max();
    add(RP1_TRACE_GRAPH_START, 0xFFFFu, max - 100u);
    add(RP1_TRACE_KERNEL_LAUNCH, 0u, max - 90u);
    add(RP1_TRACE_KERNEL_DONE, 0u, max - 80u);
    add(RP1_TRACE_FLUSH_START, 0xFFFFu, max - 70u);
    add(RP1_TRACE_FLUSH_END, 0xFFFFu, 10u);
    add(RP1_TRACE_KERNEL_LAUNCH, 1u, 20u);
    add(RP1_TRACE_KERNEL_DONE, 1u, 30u);
    add(RP1_TRACE_KERNEL_LAUNCH, 2u, 40u);
    add(RP1_TRACE_KERNEL_DONE, 2u, 50u);
    add(RP1_TRACE_GRAPH_DONE, 0xFFFFu, 60u);

    const TraceIntervals intervals = extractTrace(capture, 3);
    if (intervals.dispatch != 10u ||
        intervals.kernelSpan != 141u ||
        intervals.graph != 161u ||
        intervals.launchGaps !=
            std::vector<std::uint32_t>({111u, 20u}) ||
        intervals.launchGapsExcludingFlush !=
            std::vector<std::uint32_t>({30u, 20u}) ||
        intervals.handoffGaps !=
            std::vector<std::uint32_t>({101u, 10u}) ||
        intervals.handoffGapsExcludingFlush !=
            std::vector<std::uint32_t>({20u, 10u}) ||
        intervals.flushDurations !=
            std::vector<std::uint32_t>({81u})) {
        return 1;
    }

    capture.entries.erase(
        std::remove_if(
            capture.entries.begin(), capture.entries.end(),
            [](const rp1_trace_entry_t& entry) {
                return entry.event == RP1_TRACE_KERNEL_DONE &&
                       entry.node_index == 2u;
            }),
        capture.entries.end());
    try {
        (void)extractTrace(capture, 3);
    } catch (const std::runtime_error&) {
        return 0;
    }
    return 1;
}
