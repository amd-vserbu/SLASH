/**
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software Foundation; version 2.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if
 * not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */

/**
 * @file slash_pcie.c
 *
 * PCI driver for the SLASH control function (PF2).
 *
 * This driver binds to PCI device 10EE:50B6 (the V80 SLASH control
 * function).  On probe it creates a control device (slash_ctldev) that
 * exposes BAR information and dma-buf-backed BAR mappings to userspace,
 * and registers the device with the hotplug subsystem so that it can
 * be removed/reset/rescanned during FPGA reconfiguration.
 *
 * PF1 (the QDMA function, device 10EE:50B5) is handled by a separate
 * PCI driver registered in slash_qdma.c.
 */

#include "slash_pcie.h"

#include <linux/aer.h>
#include <linux/delay.h>
#include <linux/io.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/printk.h>

#include "slash.h"
#include "slash_ctldev.h"
#include "slash_hotplug_driver.h"

static int slash_pcie_probe(struct pci_dev *pdev, const struct pci_device_id *id);
static void slash_pcie_remove(struct pci_dev *pdev);
static bool slash_pcie_is_root_port(const struct pci_dev *bridge);
static int slash_pcie_setup_aer_masks(struct pci_dev *pdev, struct slash_pcie_aer_ctx *ctx);
static int slash_pcie_restore_aer_masks(struct pci_dev *pdev, const struct slash_pcie_aer_ctx *ctx);
static int slash_pcie_read_bar0_ap_ctrl(struct pci_dev *pdev, u32 *ap_ctrl);
static int slash_pcie_check_bar0_ready(struct pci_dev *pdev);

#define PF2_AXI_PROBE_MASK \
    (PCI_ERR_UNC_COMP_TIME | PCI_ERR_UNC_UNSUP | PCI_ERR_UNC_COMP_ABORT)

struct slash_pcie_aer_ctx {
    struct pci_dev *bridge;
    int ep_aer;
    int br_aer;
    u32 saved_ep_mask;
    u32 saved_br_mask;
    bool ep_masked;
    bool br_masked;
};

/* Match only the SLASH control function (PF2, device 0x50B6). */
static const struct pci_device_id slash_pcie_ids[] = {
    {PCI_DEVICE(SLASH_PCIE_VENDOR_ID, SLASH_PCIE_DEVICE_ID)},
    {0,}
};
MODULE_DEVICE_TABLE(pci, slash_pcie_ids);

static struct pci_driver slash_pcie_driver = {
    .name = SLASH_PCIE_DRV_NAME,
    .id_table = slash_pcie_ids,
    .probe = slash_pcie_probe,
    .remove = slash_pcie_remove,
};

/**
 * slash_pcie_is_root_port() - Determine whether the immediate upstream bridge is a root port.
 * @bridge: Upstream bridge for PF2 (may be NULL).
 *
 * Return: true if @bridge is a root-port-like parent, false otherwise.
 */
static bool slash_pcie_is_root_port(const struct pci_dev *bridge)
{
    if (!bridge)
        return true;

    return (bridge->bus && bridge->bus->parent == NULL)
        || (pci_pcie_type(bridge) == PCI_EXP_TYPE_ROOT_PORT);
}

/**
 * slash_pcie_restore_aer_masks() - Clear induced AER errors and restore saved masks.
 * @pdev: PF2 endpoint.
 * @ctx:  Saved AER context from slash_pcie_setup_aer_masks().
 *
 * Return: 0 on success, -EIO on config-space write failure.
 */
static int slash_pcie_restore_aer_masks(struct pci_dev *pdev, const struct slash_pcie_aer_ctx *ctx)
{
    int ret;
    int err = 0;

    if (ctx->ep_masked) {
        ret = pci_write_config_dword(pdev, ctx->ep_aer + PCI_ERR_UNCOR_STATUS,
                                     PF2_AXI_PROBE_MASK);
        if (ret != PCIBIOS_SUCCESSFUL) {
            dev_err(&pdev->dev,
                    "slash: failed to clear endpoint AER status: %d\n", ret);
            err = -EIO;
        }

        ret = pci_write_config_dword(pdev, ctx->ep_aer + PCI_ERR_UNCOR_MASK,
                                     ctx->saved_ep_mask);
        if (ret != PCIBIOS_SUCCESSFUL) {
            dev_err(&pdev->dev,
                    "slash: failed to restore endpoint AER mask: %d\n", ret);
            err = -EIO;
        } else {
            dev_dbg(&pdev->dev, "slash: AER mask restored on endpoint\n");
        }
    }

    if (ctx->br_masked && ctx->bridge) {
        ret = pci_write_config_dword(ctx->bridge, ctx->br_aer + PCI_ERR_UNCOR_STATUS,
                                     PF2_AXI_PROBE_MASK);
        if (ret != PCIBIOS_SUCCESSFUL) {
            dev_err(&pdev->dev,
                    "slash: failed to clear bridge AER status on %s: %d\n",
                    pci_name(ctx->bridge), ret);
            err = -EIO;
        }

        ret = pci_write_config_dword(ctx->bridge, ctx->br_aer + PCI_ERR_UNCOR_MASK,
                                     ctx->saved_br_mask);
        if (ret != PCIBIOS_SUCCESSFUL) {
            dev_err(&pdev->dev,
                    "slash: failed to restore bridge AER mask on %s: %d\n",
                    pci_name(ctx->bridge), ret);
            err = -EIO;
        } else {
            dev_dbg(&pdev->dev, "slash: AER mask restored on bridge %s\n",
                    pci_name(ctx->bridge));
        }
    }

    return err;
}

/**
 * slash_pcie_setup_aer_masks() - Save and apply temporary AER masks for PF2 BAR0 probe.
 * @pdev: PF2 endpoint.
 * @ctx:  Output context used later by slash_pcie_restore_aer_masks().
 *
 * On root-port topologies, fail closed if AER capability is missing on either
 * PF2 endpoint or immediate upstream bridge.
 *
 * Return: 0 on success, negative errno on failure.
 */
static int slash_pcie_setup_aer_masks(struct pci_dev *pdev, struct slash_pcie_aer_ctx *ctx)
{
    bool root_port;
    int ret;

    ctx->bridge = pdev->bus ? pdev->bus->self : NULL;
    ctx->ep_aer = pci_find_ext_capability(pdev, PCI_EXT_CAP_ID_ERR);
    ctx->br_aer = ctx->bridge ?
        pci_find_ext_capability(ctx->bridge, PCI_EXT_CAP_ID_ERR) : 0;

    root_port = slash_pcie_is_root_port(ctx->bridge);

    if (root_port && (!ctx->ep_aer || !ctx->br_aer)) {
        dev_err(&pdev->dev,
                "slash: refusing PF2 BAR0 probe on root-port without required AER capabilities "
                "(endpoint=%s bridge=%s)\n",
                ctx->ep_aer ? "present" : "missing",
                ctx->br_aer ? "present" : "missing");
        return -EOPNOTSUPP;
    }

    if (ctx->ep_aer) {
        ret = pci_read_config_dword(pdev, ctx->ep_aer + PCI_ERR_UNCOR_MASK,
                                    &ctx->saved_ep_mask);
        if (ret != PCIBIOS_SUCCESSFUL) {
            dev_err(&pdev->dev,
                    "slash: failed to read endpoint AER mask: %d\n", ret);
            return -EIO;
        }

        ret = pci_write_config_dword(pdev, ctx->ep_aer + PCI_ERR_UNCOR_MASK,
                                     ctx->saved_ep_mask | PF2_AXI_PROBE_MASK);
        if (ret != PCIBIOS_SUCCESSFUL) {
            dev_err(&pdev->dev,
                    "slash: failed to set endpoint AER mask: %d\n", ret);
            return -EIO;
        }

        ctx->ep_masked = true;
        dev_dbg(&pdev->dev,
                "slash: AER masked on endpoint (saved=0x%08x, set=0x%08x)\n",
                ctx->saved_ep_mask, ctx->saved_ep_mask | PF2_AXI_PROBE_MASK);
    } else {
        dev_warn(&pdev->dev,
                 "slash: no AER capability on PF2 endpoint — "
                 "BAR0 probe without endpoint error masking\n");
    }

    if (ctx->br_aer && ctx->bridge) {
        ret = pci_read_config_dword(ctx->bridge, ctx->br_aer + PCI_ERR_UNCOR_MASK,
                                    &ctx->saved_br_mask);
        if (ret != PCIBIOS_SUCCESSFUL) {
            dev_err(&pdev->dev,
                    "slash: failed to read bridge AER mask on %s: %d\n",
                    pci_name(ctx->bridge), ret);
            goto err_restore;
        }

        ret = pci_write_config_dword(ctx->bridge, ctx->br_aer + PCI_ERR_UNCOR_MASK,
                                     ctx->saved_br_mask | PF2_AXI_PROBE_MASK);
        if (ret != PCIBIOS_SUCCESSFUL) {
            dev_err(&pdev->dev,
                    "slash: failed to set bridge AER mask on %s: %d\n",
                    pci_name(ctx->bridge), ret);
            goto err_restore;
        }

        ctx->br_masked = true;
        dev_dbg(&pdev->dev,
                "slash: AER masked on bridge %s (saved=0x%08x, set=0x%08x)\n",
                pci_name(ctx->bridge), ctx->saved_br_mask,
                ctx->saved_br_mask | PF2_AXI_PROBE_MASK);
    } else {
        dev_warn(&pdev->dev,
                 "slash: no AER capability on upstream bridge — "
                 "BAR0 probe without bridge error masking\n");
    }

    return 0;

err_restore:
    if (slash_pcie_restore_aer_masks(pdev, ctx)) {
        dev_err(&pdev->dev,
                "slash: failed to restore AER state after setup failure\n");
    }
    return -EIO;
}

/**
 * slash_pcie_read_bar0_ap_ctrl() - Read PF2 BAR0 ap_ctrl register once.
 * @pdev: PF2 endpoint.
 * @ap_ctrl: Output register value.
 *
 * Return: 0 on success, negative errno on failure.
 */
static int slash_pcie_read_bar0_ap_ctrl(struct pci_dev *pdev, u32 *ap_ctrl)
{
    void __iomem *bar0;

    bar0 = pci_iomap(pdev, 0, sizeof(u32));
    if (!bar0) {
        dev_err(&pdev->dev,
                "slash: pci_iomap(BAR0) failed during PF2 readiness check\n");
        return -ENOMEM;
    }

    *ap_ctrl = ioread32(bar0);
    pci_iounmap(pdev, bar0);
    return 0;
}

/**
 * slash_pcie_check_bar0_ready() - Verify that PF2 BAR0 fabric is ready.
 * @pdev: PF2 endpoint.
 *
 * Return: 0 if ready, -EPROBE_DEFER if fabric is not ready yet, other
 *         negative errno values on hard failure.
 */
static int slash_pcie_check_bar0_ready(struct pci_dev *pdev)
{
    struct slash_pcie_aer_ctx aer_ctx = {0};
    u32 ap_ctrl = 0xFFFFFFFF;
    int err;
    int restore_err;

    err = slash_pcie_setup_aer_masks(pdev, &aer_ctx);
    if (err)
        return err;

    err = slash_pcie_read_bar0_ap_ctrl(pdev, &ap_ctrl);

    restore_err = slash_pcie_restore_aer_masks(pdev, &aer_ctx);
    if (!err && restore_err)
        err = restore_err;

    if (err)
        return err;

    if (ap_ctrl == 0xFFFFFFFF) {
        dev_info(&pdev->dev,
                 "slash: PF2 BAR0 AXI fabric not ready (ap_ctrl=0x%08x) — "
                 "deferring probe\n",
                 ap_ctrl);
        return -EPROBE_DEFER;
    }

    dev_info(&pdev->dev,
             "slash: PF2 BAR0 AXI fabric ready (ap_ctrl=0x%08x)\n", ap_ctrl);

    return 0;
}

/**
 * slash_pcie_probe() - Bind to a SLASH control function.
 * @pdev: PCI device being probed.
 * @id:   Matching device ID entry (unused).
 *
 * Verifies that this is the expected physical function (PF2), enables
 * the PCI device, creates the control character device, and registers
 * with the hotplug subsystem.
 *
 * Takes an extra reference on @pdev (pci_dev_get) because the control
 * device and hotplug subsystem hold pointers to it beyond this function.
 * The reference is released in slash_pcie_remove().
 *
 * Return: 0 on success, negative errno on failure.
 */
static int slash_pcie_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    int err;

    (void) id; /* Unused */

    dev_info(&pdev->dev, "slash: probe start for %s\n", pci_name(pdev));
    dev_dbg(&pdev->dev, "slash: vendor=0x%04x device=0x%04x fn=%u\n", pdev->vendor, pdev->device, PCI_FUNC(pdev->devfn));

    /*
     * The SLASH design places the control interface on PF2.  Reject
     * any other function — this guards against unexpected device ID
     * collisions or misconfigured FPGA designs.
     */
    if (PCI_FUNC(pdev->devfn) != SLASH_PCIE_PF) {
        dev_err(&pdev->dev, "slash: expected PF %u, got %u\n", SLASH_PCIE_PF, PCI_FUNC(pdev->devfn));
        return -EINVAL;
    }

    /* Hold a reference for the lifetime of the driver binding. */
    pci_dev_get(pdev);

    err = pci_enable_device(pdev);
    if (err) {
        dev_err(&pdev->dev, "slash: pci_enable_device() failed: %d\n", err);
        goto err_put_device;
    }

    /* Bus mastering is required for the device to perform DMA. */
    pci_set_master(pdev);
    dev_dbg(&pdev->dev, "slash: bus mastering enabled\n");

    /* Ensure PF2 AXI fabric behind BAR0 is ready before exposing userspace access. */
    err = slash_pcie_check_bar0_ready(pdev);
    if (err) {
        if (err != -EPROBE_DEFER) {
            dev_err(&pdev->dev,
                    "slash: PF2 BAR0 readiness check failed: %d\n", err);
        }
        goto err_disable_device;
    }

    err = slash_ctldev_create(pdev);
    if (err) {
        dev_err(&pdev->dev, "slash: control device create failed: %d\n", err);
        goto err_disable_device;
    }

    dev_info(&pdev->dev, "slash: probe successful\n");
    return 0;

err_disable_device:
    pci_clear_master(pdev);
    pci_disable_device(pdev);

err_put_device:
    pci_dev_put(pdev);

    return err;
}

/**
 * slash_pcie_remove() - Unbind from a SLASH control function.
 * @pdev: PCI device being removed.
 *
 * Tears down resources in reverse probe order: hotplug unregister,
 * control device destroy, then PCI cleanup.
 */
static void slash_pcie_remove(struct pci_dev *pdev)
{
    dev_info(&pdev->dev, "slash: remove start for %s\n", pci_name(pdev));

    slash_ctldev_destroy(pdev);
    pci_clear_master(pdev);
    pci_disable_device(pdev);
    pci_dev_put(pdev);

    dev_info(&pdev->dev, "slash: remove complete\n");
}

int __init slash_pcie_init(void)
{
    int err;

    pr_info("slash: registering PCIe driver '%s'\n", SLASH_NAME);
    err = pci_register_driver(&slash_pcie_driver);

    if (err) {
        pr_err("slash: pci_register_driver failed: %d\n", err);
        return err;
    }

    pr_info("slash: driver '%s' registered\n", SLASH_NAME);
    return 0;
}

void __exit slash_pcie_exit(void)
{
    pr_info("slash: unregistering PCIe driver '%s'\n", SLASH_NAME);
    pci_unregister_driver(&slash_pcie_driver);
    pr_info("slash: driver '%s' unregistered\n", SLASH_NAME);
}
