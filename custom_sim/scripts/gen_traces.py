#!/usr/bin/env python3
"""
Trace generator for DMA-generator + double-buffered SRAM + int128 MxV simulation.

=== KEY DESIGN DECISION ===
The DMA generator is an EXTERNAL hardware unit with USER-CONTROLLED fill latency.
It does NOT go through the GPU memory hierarchy (no L1, L2, DRAM involvement).

Therefore:
  - The trace contains ONLY the compute phase (LDS + LDG + IMAD)
  - DMA fill time is a parameter you specify, NOT simulated
  - Double-buffer overlap is calculated analytically in the post-processing step
  - The simulator gives us accurate compute-per-tile timing
  - Total time = analytical formula combining fill time + compute time

=== ARCHITECTURE ===
  Per SM, per tile iteration:
    Buffer A: [matrix tile N]    <- compute reads this (LDS, fast)
    Buffer B: [matrix tile N+1]  <- DMA fills this (external, user-controlled cycles)
    DRAM: [vector]               <- loaded via LDG (goes through real memory hierarchy)
    INT units: 4x IMAD per int128 element

=== GLOBAL SYNCHRONIZATION MODEL ===
  1. DMA fills Buffer X on ALL SMs simultaneously. Fill starts only after
     ALL SMs have finished reading Buffer X (global barrier).
  2. ALL SMs start reading Buffer X only after DMA fill is done.
  3. Buffers alternate A→B→A→B each tile within a loop.

=== DOUBLE BUFFER TIMELINE (N tiles per loop) ===
  t=0:           DMA fills Buffer A (tile 0)         [FILL_CYCLES]
  t=FILL:        ALL SMs read A  +  DMA fills B      [max(FILL, COMPUTE)]
  t=FILL+max:    ALL SMs read B  +  DMA fills A      [max(FILL, COMPUTE)]
  ...            (N-1 overlapped phases)
  t=FILL+(N-1)*max:  ALL SMs read last buffer        [COMPUTE]

  loop_time = FILL + (N-1)*max(FILL,COMPUTE) + COMPUTE
            = FILL + N*COMPUTE  if compute-bound
            = N*FILL + COMPUTE  if fill-bound

=== TRACE STRUCTURE (per thread block) ===
  For each tile iteration:
    BAR.SYNC  ← models the global barrier: "all SMs done reading previous
                buffer AND DMA fill of this buffer is complete".
                Since all thread blocks execute identical instructions,
                they all hit this barrier simultaneously — consistent with
                global-sync semantics. The actual fill wait time is added
                analytically in analyze_results.py.
    For each row this warp owns:
      For each column:
        LDS.128    (read int128 matrix element from SRAM buffer)
        LDG.E.128  (read int128 vector element from DRAM)
        4x IMAD    (int128 MAC: 4 int32 multiply-accumulate)
  Tiles alternate between buffer A and buffer B addresses.

Usage:
  python3 gen_traces.py --tile-rows 8 --tile-cols 8 --num-tiles 4 --num-loops 3
"""

import argparse
import os


def gen_kernel_trace(args):
    """Generate kernel trace with multiple loops of double-buffered compute."""

    elem_bytes = 16  # int128 = 16 bytes

    tile_elems = args.tile_rows * args.tile_cols

    # Shared memory addresses (per SM, private)
    shmem_base = 0xFF000000
    buf_a_base = shmem_base
    buf_b_base = shmem_base + tile_elems * elem_bytes
    local_base = 0xFF800000

    # Global memory: vector in DRAM
    vector_global_base = 0x00007F0010000000

    threads_per_block = args.threads_per_block
    warps_per_block = threads_per_block // 32
    num_blocks = args.num_blocks

    shmem_bytes = 2 * tile_elems * elem_bytes

    # Work distribution: each warp handles some rows
    rows_per_warp = max(1, args.tile_rows // warps_per_block)

    lines = []

    # === Kernel header ===
    lines.append(f"-kernel name = custom_dma_dbuf_matvec")
    lines.append(f"-kernel id = 1")
    lines.append(f"-grid dim = ({num_blocks},1,1)")
    lines.append(f"-block dim = ({threads_per_block},1,1)")
    lines.append(f"-shmem = {shmem_bytes}")
    lines.append(f"-nregs = 32")
    lines.append(f"-cuda stream id = 0")
    lines.append(f"-binary version = 80")
    lines.append(f"-enable lineinfo = 0")
    lines.append(f"-accelsim tracer version = 3")
    lines.append(f"-shmem base_addr = {hex(shmem_base)}")
    lines.append(f"-local mem base_addr = {hex(local_base)}")
    lines.append("")
    # parse_kernel_info (trace_parser.cc) reads header '-' lines until it hits
    # the first '#' line and breaks, consuming that line.  get_next_threadblock_traces
    # then reads from the next line expecting '#BEGIN_TB'.  So we need exactly one
    # '#' separator here between the header and the first '#BEGIN_TB'.
    lines.append("#")

    pc_counter = [0x0010]

    def next_pc():
        p = pc_counter[0]
        pc_counter[0] += 0x10
        return p

    def fmt_pc(p):
        return f"{p:04x}"

    MASK_ALL = "ffffffff"

    # === Generate per-thread-block traces ===
    for block_id in range(num_blocks):
        pc_counter[0] = 0x0010

        lines.append("#BEGIN_TB")
        lines.append(f"thread block = {block_id},0,0")

        for warp_id in range(warps_per_block):
            warp_insts = []

            warp_row_start = warp_id * rows_per_warp

            # ==============================================================
            # Loop structure:
            #   for loop in range(num_loops):
            #     for tile in range(num_tiles):
            #       BAR.SYNC  (models sync point where we'd wait for DMA)
            #       Compute: LDS(matrix) + LDG(vector) + 4xIMAD
            #     STG (store loop result)
            # ==============================================================

            for loop_iter in range(args.num_loops):
                for tile in range(args.num_tiles):
                    # Pick buffer A or B (alternating)
                    if tile % 2 == 0:
                        compute_buf = buf_a_base
                    else:
                        compute_buf = buf_b_base

                    # --- BAR.SYNC: synchronization point ---
                    # In real execution, this is where we'd wait for DMA fill.
                    # In simulation, this is ~free (all warps arrive together).
                    # The actual DMA wait time is added analytically.
                    p = next_pc()
                    warp_insts.append(
                        f"{fmt_pc(p)} {MASK_ALL} 0 BAR.SYNC 0 0"
                    )

                    # --- Compute: matrix tile (SRAM) x vector (DRAM) ---
                    for row in range(rows_per_warp):
                        global_row = warp_row_start + row
                        if global_row >= args.tile_rows:
                            break

                        for col in range(args.tile_cols):
                            elem_idx = global_row * args.tile_cols + col
                            shmem_addr = compute_buf + elem_idx * elem_bytes

                            # Use different vector addresses per loop
                            vec_addr = (vector_global_base
                                        + loop_iter * args.tile_cols * elem_bytes
                                        + col * elem_bytes)

                            # LDS: Load matrix element from SRAM (fast, 2 cyc)
                            p = next_pc()
                            warp_insts.append(
                                f"{fmt_pc(p)} {MASK_ALL} 1 R2 LDS.128 1 R3 "
                                f"16 1 {hex(shmem_addr)} 0"
                            )

                            # LDG: Load vector element from DRAM
                            p = next_pc()
                            warp_insts.append(
                                f"{fmt_pc(p)} {MASK_ALL} 1 R4 LDG.E.128 1 R5 "
                                f"16 1 {hex(vec_addr)} 0"
                            )

                            # int128 MAC = 4 x int32 IMAD
                            for part in range(4):
                                p = next_pc()
                                warp_insts.append(
                                    f"{fmt_pc(p)} {MASK_ALL} 1 R{6 + part} "
                                    f"IMAD 3 R2 R4 R{6 + part} 0"
                                )

                # --- End of one loop: store result to global memory ---
                for row in range(rows_per_warp):
                    global_row = warp_row_start + row
                    if global_row >= args.tile_rows:
                        break
                    result_addr = (0x00007F0020000000
                                   + loop_iter * args.tile_rows * elem_bytes
                                   + global_row * elem_bytes)
                    p = next_pc()
                    warp_insts.append(
                        f"{fmt_pc(p)} {MASK_ALL} 0 STG.E.128 2 R6 R7 "
                        f"16 1 {hex(result_addr)} 0"
                    )

            # Write warp instructions
            lines.append(f"warp = {warp_id}")
            lines.append(f"insts = {len(warp_insts)}")
            for inst in warp_insts:
                lines.append(inst)

        lines.append("#END_TB")

    return lines


def gen_kernelslist(args):
    """Generate the top-level kernelslist.g file."""
    elem_bytes = 16

    vector_size = args.num_loops * args.tile_cols * elem_bytes
    output_size = args.num_loops * args.tile_rows * elem_bytes

    lines = []
    lines.append(f"MemcpyHtoD,0x00007f0010000000,{vector_size}")
    lines.append(f"MemcpyHtoD,0x00007f0020000000,{output_size}")
    lines.append("kernel-1.traceg")
    return lines


def print_summary(args):
    """Print configuration summary and analytical formulas."""
    elem_bytes = 16
    tile_elems = args.tile_rows * args.tile_cols
    total_tiles = args.num_loops * args.num_tiles
    shmem_per_buf = tile_elems * elem_bytes
    shmem_total = 2 * shmem_per_buf

    warps = args.threads_per_block // 32
    rows_per_warp = max(1, args.tile_rows // warps)
    insts_per_elem = 6  # 1 LDS + 1 LDG + 4 IMAD
    insts_per_tile_per_warp = rows_per_warp * args.tile_cols * insts_per_elem + 1  # +1 for BAR

    print(f"\n{'='*60}")
    print(f"  SIMULATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Matrix tile:     {args.tile_rows} x {args.tile_cols} int128 elements")
    print(f"  Element size:    {elem_bytes} bytes (int128)")
    print(f"  Tile data:       {shmem_per_buf} bytes per buffer")
    print(f"  Shared mem:      {shmem_total} bytes (2 buffers)")
    print(f"  Threads/block:   {args.threads_per_block}")
    print(f"  Warps/block:     {warps}")
    print(f"  Rows/warp:       {rows_per_warp}")
    print(f"  Num blocks:      {args.num_blocks}")
    print(f"  Tiles/loop:      {args.num_tiles}")
    print(f"  Num loops:       {args.num_loops}")
    print(f"  Total tiles:     {total_tiles}")
    print(f"  Insts/tile/warp: {insts_per_tile_per_warp}")
    print(f"")
    print(f"  DOUBLE-BUFFER TIMING FORMULA:")
    print(f"  {'─'*40}")
    print(f"  Per loop:")
    print(f"    loop_time = fill_cycles")
    print(f"              + num_tiles * max(fill_cycles, compute_per_tile)")
    print(f"  Total:")
    print(f"    total = num_loops * loop_time")
    print(f"")
    print(f"  After simulation, run:")
    print(f"    python3 analyze_results.py --sim-log <log> --fill-cycles <N>")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate traces for DMA + double-buffer + int128 MxV"
    )
    parser.add_argument("--tile-rows", type=int, default=8,
                        help="Matrix tile rows (default: 8)")
    parser.add_argument("--tile-cols", type=int, default=8,
                        help="Matrix tile cols = vector length (default: 8)")
    parser.add_argument("--threads-per-block", type=int, default=256,
                        help="Threads per block (default: 256)")
    parser.add_argument("--num-blocks", type=int, default=2,
                        help="Thread blocks (default: 2, full A100=108)")
    parser.add_argument("--num-tiles", type=int, default=4,
                        help="Tiles per loop for double buffering (default: 4)")
    parser.add_argument("--num-loops", type=int, default=3,
                        help="Number of outer loops (default: 3)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "traces"),
                        help="Output directory")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    kl_lines = gen_kernelslist(args)
    kl_path = os.path.join(args.outdir, "kernelslist.g")
    with open(kl_path, "w") as f:
        f.write("\n".join(kl_lines) + "\n")
    print(f"Written: {kl_path}")

    kt_lines = gen_kernel_trace(args)
    kt_path = os.path.join(args.outdir, "kernel-1.traceg")
    with open(kt_path, "w") as f:
        f.write("\n".join(kt_lines) + "\n")
    print(f"Written: {kt_path}")

    print_summary(args)


if __name__ == "__main__":
    main()
