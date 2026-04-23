#include <vrt/graph/graph.hpp>

#include <memory>

#include <vrt/graph/crossdevice/cpu_fpga_bridge.hpp>
#include <vrt/graph/device/cpu_device.hpp>

#if defined(VRT_HAS_GPU) && (VRT_HAS_GPU == 1)
#include <vrt/graph/crossdevice/cpu_gpu_bridge.hpp>
#endif

namespace vrt::graph {

Graph Graph::withDefaults() {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    graph.registerBridgeFactory(DeviceType::CPU, DeviceType::FPGA,
                                CpuFpgaBridgeFactory());
    graph.registerBridgeFactory(DeviceType::FPGA, DeviceType::CPU,
                                CpuFpgaBridgeFactory());

#if defined(VRT_HAS_GPU) && (VRT_HAS_GPU == 1)
    graph.registerBridgeFactory(DeviceType::CPU, DeviceType::GPU,
                                CpuGpuBridgeFactory());
    graph.registerBridgeFactory(DeviceType::GPU, DeviceType::CPU,
                                CpuGpuBridgeFactory());
#endif

    return graph;
}

std::shared_ptr<CpuDevice> Graph::cpuDevice() const {
    auto it = devices_.find("cpu");
    if (it != devices_.end()) {
        return std::dynamic_pointer_cast<CpuDevice>(it->second);
    }

    for (const auto& [id, device] : devices_) {
        (void)id;
        if (device->type() == DeviceType::CPU) {
            return std::dynamic_pointer_cast<CpuDevice>(device);
        }
    }

    return nullptr;
}

void Graph::validateDeclaredScalars() const {
    for (const auto& node : nodes_) {
        for (const auto& [portName, scalar] : node.ioMap.scalars()) {
            (void)portName;
            if (scalar.isConstant()) continue;

            auto it = scalarTypes_.find(scalar.varName());
            if (it == scalarTypes_.end()) {
                throw std::runtime_error(
                    "Graph: global scalar '" + scalar.varName() +
                    "' is not declared; call Graph::globalScalar() first");
            }
            if (it->second != scalar.type()) {
                throw std::runtime_error(
                    "Graph: global scalar '" + scalar.varName() +
                    "' type mismatch between declaration and binding");
            }
        }
    }
}

}  // namespace vrt::graph