/**
 * This file is part of the Coyote <https://github.com/fpgasystems/Coyote>
 *
 * MIT Licence
 */

import lynxTypes::*;

module p2p_coyote_axi_ctrl_parser (
  input  logic                        aclk,
  input  logic                        aresetn,

  AXI4L.s                             axi_ctrl,

  output logic                        start_pulse,
  output logic [VADDR_BITS-1:0]       bar_base_vaddr,
  output logic [31:0]                 operand_a,
  output logic [31:0]                 operand_b,
  output logic [PID_BITS-1:0]         ctid,

  input  logic [3:0]                  status_bits,
  input  logic [31:0]                 result_value,
  input  logic [31:0]                 poll_count,
  input  logic [31:0]                 last_ctrl,
  input  logic [31:0]                 error_code
);

/////////////////////////////////////
//          CONSTANTS             //
///////////////////////////////////
localparam integer N_REGS = 10;
localparam integer ADDR_MSB = $clog2(N_REGS);
localparam integer ADDR_LSB = $clog2(AXIL_DATA_BITS/8);
localparam integer AXI_ADDR_BITS = ADDR_LSB + ADDR_MSB;

/////////////////////////////////////
//          REGISTERS             //
///////////////////////////////////
logic [AXI_ADDR_BITS-1:0] axi_awaddr;
logic axi_awready;
logic [AXI_ADDR_BITS-1:0] axi_araddr;
logic axi_arready;
logic [1:0] axi_bresp;
logic axi_bvalid;
logic axi_wready;
logic [AXIL_DATA_BITS-1:0] axi_rdata;
logic [1:0] axi_rresp;
logic axi_rvalid;
logic aw_en;

logic [N_REGS-1:0][AXIL_DATA_BITS-1:0] ctrl_reg;
logic ctrl_reg_wren;
logic ctrl_reg_rden;

/////////////////////////////////////
//         REGISTER MAP           //
///////////////////////////////////
localparam integer COMMAND_REG = 0;   // W1S: bit0=start
localparam integer STATUS_REG = 1;    // RO: [0]=done [1]=busy [2]=error [3]=timeout
localparam integer BAR_BASE_REG = 2;  // WR: token/base virtual address for mapped dmabuf
localparam integer OPERAND_A_REG = 3; // WR: addend A
localparam integer OPERAND_B_REG = 4; // WR: addend B
localparam integer CTID_REG = 5;      // WR: cThread ID used in SQ requests
localparam integer RESULT_REG = 6;    // RO: addition result
localparam integer POLL_COUNT_REG = 7;// RO: number of CTRL polls performed
localparam integer LAST_CTRL_REG = 8; // RO: last observed SLASH ctrl register value
localparam integer ERROR_REG = 9;     // RO: implementation-specific error code

/////////////////////////////////////
//         WRITE PROCESS          //
///////////////////////////////////
assign ctrl_reg_wren = axi_wready && axi_ctrl.wvalid && axi_awready && axi_ctrl.awvalid;

always_ff @(posedge aclk) begin
  if (aresetn == 1'b0) begin
    ctrl_reg <= 0;
  end
  else begin
    ctrl_reg[COMMAND_REG] <= 0;

    if (ctrl_reg_wren) begin
      case (axi_awaddr[ADDR_LSB+:ADDR_MSB])
        COMMAND_REG,
        BAR_BASE_REG,
        OPERAND_A_REG,
        OPERAND_B_REG,
        CTID_REG: begin
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if (axi_ctrl.wstrb[i]) begin
              ctrl_reg[axi_awaddr[ADDR_LSB+:ADDR_MSB]][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        end
        default: ;
      endcase
    end
  end
end

/////////////////////////////////////
//         READ PROCESS           //
///////////////////////////////////
assign ctrl_reg_rden = axi_arready & axi_ctrl.arvalid & ~axi_rvalid;

always_ff @(posedge aclk) begin
  if (aresetn == 1'b0) begin
    axi_rdata <= 0;
  end
  else begin
    axi_rdata <= 0;
    if (ctrl_reg_rden) begin
      case (axi_araddr[ADDR_LSB+:ADDR_MSB])
        COMMAND_REG,
        BAR_BASE_REG,
        OPERAND_A_REG,
        OPERAND_B_REG,
        CTID_REG: axi_rdata <= ctrl_reg[axi_araddr[ADDR_LSB+:ADDR_MSB]];
        STATUS_REG: axi_rdata[3:0] <= status_bits;
        RESULT_REG: axi_rdata[31:0] <= result_value;
        POLL_COUNT_REG: axi_rdata[31:0] <= poll_count;
        LAST_CTRL_REG: axi_rdata[31:0] <= last_ctrl;
        ERROR_REG: axi_rdata[31:0] <= error_code;
        default: ;
      endcase
    end
  end
end

/////////////////////////////////////
//       OUTPUT ASSIGNMENT        //
///////////////////////////////////
always_comb begin
  start_pulse = ctrl_reg[COMMAND_REG][0];
  bar_base_vaddr = ctrl_reg[BAR_BASE_REG][VADDR_BITS-1:0];
  operand_a = ctrl_reg[OPERAND_A_REG][31:0];
  operand_b = ctrl_reg[OPERAND_B_REG][31:0];
  ctid = ctrl_reg[CTID_REG][PID_BITS-1:0];
end

/////////////////////////////////////
//     STANDARD AXI CONTROL       //
///////////////////////////////////
// I/O
assign axi_ctrl.awready = axi_awready;
assign axi_ctrl.arready = axi_arready;
assign axi_ctrl.bresp = axi_bresp;
assign axi_ctrl.bvalid = axi_bvalid;
assign axi_ctrl.wready = axi_wready;
assign axi_ctrl.rdata = axi_rdata;
assign axi_ctrl.rresp = axi_rresp;
assign axi_ctrl.rvalid = axi_rvalid;

// awready and awaddr
always_ff @(posedge aclk) begin
  if (aresetn == 1'b0) begin
    axi_awready <= 1'b0;
    axi_awaddr <= 0;
    aw_en <= 1'b1;
  end
  else begin
    if (~axi_awready && axi_ctrl.awvalid && axi_ctrl.wvalid && aw_en) begin
      axi_awready <= 1'b1;
      aw_en <= 1'b0;
      axi_awaddr <= axi_ctrl.awaddr;
    end
    else if (axi_ctrl.bready && axi_bvalid) begin
      aw_en <= 1'b1;
      axi_awready <= 1'b0;
    end
    else begin
      axi_awready <= 1'b0;
    end
  end
end

// arready and araddr
always_ff @(posedge aclk) begin
  if (aresetn == 1'b0) begin
    axi_arready <= 1'b0;
    axi_araddr <= 0;
  end
  else begin
    if (~axi_arready && axi_ctrl.arvalid) begin
      axi_arready <= 1'b1;
      axi_araddr <= axi_ctrl.araddr;
    end
    else begin
      axi_arready <= 1'b0;
    end
  end
end

// bvalid and bresp
always_ff @(posedge aclk) begin
  if (aresetn == 1'b0) begin
    axi_bvalid <= 0;
    axi_bresp <= 2'b0;
  end
  else begin
    if (axi_awready && axi_ctrl.awvalid && ~axi_bvalid && axi_wready && axi_ctrl.wvalid) begin
      axi_bvalid <= 1'b1;
      axi_bresp <= 2'b0;
    end
    else if (axi_ctrl.bready && axi_bvalid) begin
      axi_bvalid <= 1'b0;
    end
  end
end

// wready
always_ff @(posedge aclk) begin
  if (aresetn == 1'b0) begin
    axi_wready <= 1'b0;
  end
  else begin
    if (~axi_wready && axi_ctrl.wvalid && axi_ctrl.awvalid && aw_en) begin
      axi_wready <= 1'b1;
    end
    else begin
      axi_wready <= 1'b0;
    end
  end
end

// rvalid and rresp
always_ff @(posedge aclk) begin
  if (aresetn == 1'b0) begin
    axi_rvalid <= 0;
    axi_rresp <= 0;
  end
  else begin
    if (axi_arready && axi_ctrl.arvalid && ~axi_rvalid) begin
      axi_rvalid <= 1'b1;
      axi_rresp <= 2'b0;
    end
    else if (axi_rvalid && axi_ctrl.rready) begin
      axi_rvalid <= 1'b0;
    end
  end
end

endmodule
