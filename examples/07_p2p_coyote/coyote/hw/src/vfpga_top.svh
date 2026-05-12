/**
 * This file is part of the Coyote <https://github.com/fpgasystems/Coyote>
 *
 * MIT Licence
 */

import lynxTypes::*;

///////////////////////////////////////
//       CONTROL/PARAM SIGNALS      //
/////////////////////////////////////
logic                  start_pulse;
logic [VADDR_BITS-1:0] bar_base_vaddr;
logic [31:0]           operand_a;
logic [31:0]           operand_b;
logic [PID_BITS-1:0]   ctrl_pid;

logic [3:0]            status_bits;
logic [31:0]           result_value;
logic [31:0]           poll_count;
logic [31:0]           last_ctrl;
logic [31:0]           error_code;

p2p_coyote_axi_ctrl_parser inst_axi_ctrl_parser (
    .aclk(aclk),
    .aresetn(aresetn),
    .axi_ctrl(axi_ctrl),
    .start_pulse(start_pulse),
    .bar_base_vaddr(bar_base_vaddr),
    .operand_a(operand_a),
    .operand_b(operand_b),
    .ctid(ctrl_pid),
    .status_bits(status_bits),
    .result_value(result_value),
    .poll_count(poll_count),
    .last_ctrl(last_ctrl),
    .error_code(error_code)
);

///////////////////////////////////////
//        SLASH BAR OFFSETS         //
/////////////////////////////////////
localparam logic [VADDR_BITS-1:0] SLASH_CTRL_OFFS = 16'h0000;
localparam logic [VADDR_BITS-1:0] SLASH_A_OFFS = 16'h0010;
localparam logic [VADDR_BITS-1:0] SLASH_B_OFFS = 16'h0018;
localparam logic [VADDR_BITS-1:0] SLASH_RESULT_OFFS = 16'h0020;

localparam logic [31:0] SLASH_CTRL_START = 32'h0000_0001;
localparam logic [31:0] SLASH_CTRL_DONE_MASK = 32'h0000_0002;
localparam logic [31:0] MAX_CTRL_POLLS = 32'd100000;

///////////////////////////////////////
//         STATE MACHINE            //
/////////////////////////////////////
typedef enum logic [3:0] {
    ST_IDLE,
    ST_WR_A,
    ST_WR_B,
    ST_WR_START,
    ST_RD_CTRL_REQ,
    ST_RD_CTRL_WAIT,
    ST_RD_RES_REQ,
    ST_RD_RES_WAIT
} state_t;

state_t state_C;

logic wr_req_done;
logic wr_data_done;
logic done_latched;
logic error_latched;
logic timeout_latched;

always_ff @(posedge aclk) begin
    if (aresetn == 1'b0) begin
        state_C <= ST_IDLE;
        wr_req_done <= 1'b0;
        wr_data_done <= 1'b0;
        done_latched <= 1'b0;
        error_latched <= 1'b0;
        timeout_latched <= 1'b0;

        result_value <= 32'd0;
        poll_count <= 32'd0;
        last_ctrl <= 32'd0;
        error_code <= 32'd0;
    end
    else begin
        case (state_C)
            ST_IDLE: begin
                wr_req_done <= 1'b0;
                wr_data_done <= 1'b0;

                if (start_pulse) begin
                    done_latched <= 1'b0;
                    error_latched <= 1'b0;
                    timeout_latched <= 1'b0;
                    result_value <= 32'd0;
                    poll_count <= 32'd0;
                    last_ctrl <= 32'd0;
                    error_code <= 32'd0;
                    state_C <= ST_WR_A;
                end
            end

            ST_WR_A,
            ST_WR_B,
            ST_WR_START: begin
                if (sq_wr.valid && sq_wr.ready) begin
                    wr_req_done <= 1'b1;
                end

                if (axis_host_send[0].tvalid && axis_host_send[0].tready) begin
                    wr_data_done <= 1'b1;
                end

                if (wr_req_done && wr_data_done) begin
                    wr_req_done <= 1'b0;
                    wr_data_done <= 1'b0;

                    case (state_C)
                        ST_WR_A: state_C <= ST_WR_B;
                        ST_WR_B: state_C <= ST_WR_START;
                        default: state_C <= ST_RD_CTRL_REQ;
                    endcase
                end
            end

            ST_RD_CTRL_REQ: begin
                if (sq_rd.valid && sq_rd.ready) begin
                    state_C <= ST_RD_CTRL_WAIT;
                end
            end

            ST_RD_CTRL_WAIT: begin
                if (axis_host_recv[0].tvalid && axis_host_recv[0].tready) begin
                    last_ctrl <= axis_host_recv[0].tdata[31:0];

                    if ((axis_host_recv[0].tdata[31:0] & SLASH_CTRL_DONE_MASK) != 0) begin
                        state_C <= ST_RD_RES_REQ;
                    end
                    else if (poll_count >= MAX_CTRL_POLLS) begin
                        done_latched <= 1'b1;
                        error_latched <= 1'b1;
                        timeout_latched <= 1'b1;
                        error_code <= 32'h0000_0001;
                        state_C <= ST_IDLE;
                    end
                    else begin
                        poll_count <= poll_count + 1;
                        state_C <= ST_RD_CTRL_REQ;
                    end
                end
            end

            ST_RD_RES_REQ: begin
                if (sq_rd.valid && sq_rd.ready) begin
                    state_C <= ST_RD_RES_WAIT;
                end
            end

            ST_RD_RES_WAIT: begin
                if (axis_host_recv[0].tvalid && axis_host_recv[0].tready) begin
                    result_value <= axis_host_recv[0].tdata[31:0];
                    done_latched <= 1'b1;
                    state_C <= ST_IDLE;
                end
            end

            default: begin
                state_C <= ST_IDLE;
            end
        endcase
    end
end

always_comb begin
    status_bits = 4'b0;
    status_bits[0] = done_latched;
    status_bits[1] = (state_C != ST_IDLE);
    status_bits[2] = error_latched;
    status_bits[3] = timeout_latched;
end

///////////////////////////////////////
//      REQUEST/DATA GENERATION      //
/////////////////////////////////////
logic [VADDR_BITS-1:0] wr_addr;
logic [31:0] wr_data;
logic [VADDR_BITS-1:0] rd_addr;

always_comb begin
    wr_addr = SLASH_A_OFFS;
    wr_data = operand_a;

    case (state_C)
        ST_WR_A: begin
            wr_addr = SLASH_A_OFFS;
            wr_data = operand_a;
        end
        ST_WR_B: begin
            wr_addr = SLASH_B_OFFS;
            wr_data = operand_b;
        end
        ST_WR_START: begin
            wr_addr = SLASH_CTRL_OFFS;
            wr_data = SLASH_CTRL_START;
        end
        default: begin
            wr_addr = SLASH_A_OFFS;
            wr_data = operand_a;
        end
    endcase
end

always_comb begin
    rd_addr = SLASH_CTRL_OFFS;

    case (state_C)
        ST_RD_CTRL_REQ,
        ST_RD_CTRL_WAIT: rd_addr = SLASH_CTRL_OFFS;
        ST_RD_RES_REQ,
        ST_RD_RES_WAIT: rd_addr = SLASH_RESULT_OFFS;
        default: rd_addr = SLASH_CTRL_OFFS;
    endcase
end

always_comb begin
    // Defaults
    sq_rd.data = 0;
    sq_rd.valid = 1'b0;
    sq_wr.data = 0;
    sq_wr.valid = 1'b0;
    cq_rd.ready = 1'b1;
    cq_wr.ready = 1'b1;

    axis_host_recv[0].tready = 1'b0;

    axis_host_send[0].tdata = '0;
    axis_host_send[0].tkeep = '0;
    axis_host_send[0].tid = '0;
    axis_host_send[0].tlast = 1'b1;
    axis_host_send[0].tvalid = 1'b0;

    // Write requests + payload beats
    if (state_C == ST_WR_A || state_C == ST_WR_B || state_C == ST_WR_START) begin
        sq_wr.data.last = 1'b1;
        sq_wr.data.pid = ctrl_pid;
        sq_wr.data.len = 32'd4;
        sq_wr.data.vaddr = bar_base_vaddr + wr_addr;
        sq_wr.data.strm = STRM_HOST;
        sq_wr.data.opcode = LOCAL_WRITE;
        sq_wr.valid = ~wr_req_done;

        axis_host_send[0].tdata[31:0] = wr_data;
        axis_host_send[0].tkeep[3:0] = 4'hF;
        axis_host_send[0].tvalid = ~wr_data_done;
    end

    // Read requests
    if (state_C == ST_RD_CTRL_REQ || state_C == ST_RD_RES_REQ) begin
        sq_rd.data.last = 1'b1;
        sq_rd.data.pid = ctrl_pid;
        sq_rd.data.len = 32'd4;
        sq_rd.data.vaddr = bar_base_vaddr + rd_addr;
        sq_rd.data.strm = STRM_HOST;
        sq_rd.data.opcode = LOCAL_READ;
        sq_rd.valid = 1'b1;
    end

    if (state_C == ST_RD_CTRL_WAIT || state_C == ST_RD_RES_WAIT) begin
        axis_host_recv[0].tready = 1'b1;
    end
end

// Tie off unused interfaces.
always_comb notify.tie_off_m();
