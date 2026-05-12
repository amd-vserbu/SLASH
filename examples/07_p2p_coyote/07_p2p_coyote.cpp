/**
 * The MIT License (MIT)
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <coyote/cThread.hpp>

#include <slash/ctldev.h>

#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>

#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

enum CoyoteCtrlReg : uint32_t {
    CTRL = 0,
    STATUS = 1,
    BAR_BASE = 2,
    OPERAND_A = 3,
    OPERAND_B = 4,
    CTID = 5,
    RESULT = 6,
    POLL_COUNT = 7,
    LAST_CTRL = 8,
    ERROR = 9,
};

constexpr uint64_t P2P_CTRL_START = 0x1;
constexpr uint64_t P2P_STATUS_DONE = 1ull << 0;
constexpr uint64_t P2P_STATUS_ERROR = 1ull << 2;

uint32_t parse_u32(const char *arg, const char *name) {
    char *end = nullptr;
    unsigned long value = strtoul(arg, &end, 0);
    if (end == arg || *end != '\0' || value > UINT32_MAX) {
        throw std::invalid_argument(std::string("Invalid ") + name + ": " + arg);
    }
    return static_cast<uint32_t>(value);
}

void print_usage(const char *argv0) {
    std::cerr
        << "Usage: " << argv0
        << " <slash_ctl_path> <coyote_device> <coyote_vfid> <a> <b> [slash_bar]\n\n"
        << "Example:\n  " << argv0 << " /dev/slash_ctl0 0 0 7 9\n";
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 6 || argc > 7) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string slash_ctl_path = argv[1];
    uint32_t coyote_device = 0;
    uint32_t coyote_vfid = 0;
    uint32_t operand_a = 0;
    uint32_t operand_b = 0;
    uint32_t slash_bar = 0;

    struct slash_ctldev *ctldev = nullptr;
    struct slash_p2p_bar *p2p_bar = nullptr;
    std::unique_ptr<coyote::cThread> coyote_thread;
    void *token = nullptr;
    bool token_mapped = false;

    int exit_code = 1;

    try {
        coyote_device = parse_u32(argv[2], "coyote_device");
        coyote_vfid = parse_u32(argv[3], "coyote_vfid");
        operand_a = parse_u32(argv[4], "a");
        operand_b = parse_u32(argv[5], "b");
        slash_bar = (argc == 7) ? parse_u32(argv[6], "slash_bar") : 0u;

        ctldev = slash_ctldev_open(slash_ctl_path.c_str());
        if (!ctldev) {
            throw std::runtime_error("slash_ctldev_open failed");
        }

        p2p_bar = slash_p2p_bar_open(ctldev, static_cast<int>(slash_bar), O_CLOEXEC);
        if (!p2p_bar) {
            throw std::runtime_error("slash_p2p_bar_open failed");
        }

        if (!p2p_bar->p2p_capable) {
            throw std::runtime_error("SLASH BAR is not P2P-capable on this system");
        }

        std::cout << "SLASH exporter: BDF=" << p2p_bar->device_info.bdf
                  << " BAR=" << p2p_bar->bar_number
                  << " len=" << p2p_bar->len
                  << " fd=" << p2p_bar->fd << std::endl;

        coyote_thread = std::make_unique<coyote::cThread>(
            static_cast<int32_t>(coyote_vfid),
            getpid(),
            coyote_device
        );

        token = coyote_thread->mapExternalDmabuf(
            p2p_bar->fd,
            static_cast<uint64_t>(p2p_bar->len),
            -1,
            false
        );
        token_mapped = true;

        const uint64_t token_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(token));
        const uint64_t ctid = static_cast<uint64_t>(coyote_thread->getCtid());

        coyote_thread->setCSR(token_addr, BAR_BASE);
        coyote_thread->setCSR(static_cast<uint64_t>(operand_a), OPERAND_A);
        coyote_thread->setCSR(static_cast<uint64_t>(operand_b), OPERAND_B);
        coyote_thread->setCSR(ctid, CTID);
        coyote_thread->setCSR(P2P_CTRL_START, CTRL);

        constexpr uint32_t kHostPollLimit = 200000;
        uint64_t status = 0;
        for (uint32_t i = 0; i < kHostPollLimit; ++i) {
            status = coyote_thread->getCSR(STATUS);
            if ((status & P2P_STATUS_DONE) != 0u) {
                break;
            }
            if ((status & P2P_STATUS_ERROR) != 0u) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        if ((status & P2P_STATUS_DONE) == 0u) {
            throw std::runtime_error("Timed out waiting for Coyote kernel completion");
        }

        if ((status & P2P_STATUS_ERROR) != 0u) {
            const uint64_t err = coyote_thread->getCSR(ERROR);
            throw std::runtime_error("Coyote kernel reported an error, code=" + std::to_string(err));
        }

        const uint32_t hw_result = static_cast<uint32_t>(coyote_thread->getCSR(RESULT) & 0xffffffffu);
        const uint32_t expected = operand_a + operand_b;

        std::cout << "Inputs: a=" << operand_a << " b=" << operand_b << std::endl;
        std::cout << "Result: hw=" << hw_result << " expected=" << expected << std::endl;
        std::cout << "Status=0x" << std::hex << status
                  << " poll_count=" << std::dec << coyote_thread->getCSR(POLL_COUNT)
                  << " last_ctrl=0x" << std::hex << coyote_thread->getCSR(LAST_CTRL)
                  << std::dec << std::endl;

        if (hw_result != expected) {
            throw std::runtime_error("Result mismatch");
        }

        exit_code = 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
    }

    if (token_mapped && coyote_thread) {
        try {
            coyote_thread->unmapExternalDmabuf(token);
        } catch (const std::exception &e) {
            std::cerr << "WARN: unmapExternalDmabuf failed: " << e.what() << std::endl;
            exit_code = 1;
        }
    }

    if (p2p_bar != nullptr) {
        if (slash_p2p_bar_close(p2p_bar) != 0) {
            perror("slash_p2p_bar_close");
            exit_code = 1;
        }
    }

    if (ctldev != nullptr) {
        if (slash_ctldev_close(ctldev) != 0) {
            perror("slash_ctldev_close");
            exit_code = 1;
        }
    }

    return exit_code;
}
