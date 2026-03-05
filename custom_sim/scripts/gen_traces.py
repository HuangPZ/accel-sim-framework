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

=== MODES ===

  MODE 1: num_sm_groups=1  (double buffer, single group)
    1 group of NUM_BLOCKS SMs, each with 2 SRAM buffers.
    DMA fills buffer B while all SMs compute buffer A, then swaps.
    Per-tile time = max(FILL, COMPUTE)

    Timeline:
      t=0:        DMA fills Buffer A (all SMs)          [FILL]
      t=FILL:     ALL SMs compute A  +  DMA fills B     [max(FILL,COMPUTE)]
      t=FILL+max: ALL SMs compute B  +  DMA fills A     [max(FILL,COMPUTE)]
      ...

  MODE 2: num_sm_groups=K  (K-group pipeline, single buffer each)
    K groups of NUM_BLOCKS SMs, each group with 1 SRAM buffer.
    DMA serves groups round-robin: fills G0, then G1, ..., then back to G0.
    While DMA fills group k, all other groups compute.

    Per-tile time per group:
      fill-bound  (COMPUTE ≤ (K-1)×FILL): K×FILL  (DMA paces everything)
      compute-bound (COMPUTE > (K-1)×FILL): FILL+COMPUTE

    Total throughput (all groups):
      fill-bound:    K  / (K×FILL)         = 1/FILL    (same as double-buffer)
      compute-bound: K  / (FILL+COMPUTE)   > 1/COMPUTE if COMPUTE > FILL

    KEY TRADEOFF vs double-buffer:
      Double buffer:  1 group, 2× SRAM/SM, per-tile = max(FILL, COMPUTE)
      K=2 groups:     2× SMs,  1× SRAM/SM, per-tile per-group = FILL+max(FILL,COMPUTE)
      K=2 total throughput = 2/(FILL+max(FILL,COMPUTE))
        compute-bound: > 1/COMPUTE when COMPUTE > FILL  ← more useful work done
        fill-bound:    = 1/FILL (DMA always the bottleneck)

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
    """Generate kernel trace supporting single-group double-buffer or K-group pipeline."""

    elem_bytes = 16  # int128 = 16 bytes
    K = args.num_sm_groups

    tile_elems = args.tile_rows * args.tile_cols

    # Shared memory addresses (per SM, private).
    # K=1 (double buffer): 2 buffers in SRAM.
    # K>1 (multi-group):   1 buffer per SM; groups are the pipeline stages.
    shmem_base = 0xFF000000
    buf_a_base = shmem_base
    buf_b_base = shmem_base + tile_elems * elem_bytes  # only used when K=1
    local_base = 0xFF800000

    # Global memory
    vector_global_base = 0x00007F0010000000
    output_global_base = 0x00007F0020000000

    threads_per_block = args.threads_per_block
    warps_per_block = threads_per_block // 32
    # Total blocks: K groups × NUM_BLOCKS SMs each
    total_blocks = args.num_blocks * K

    # SRAM: 2 buffers for double-buffer (K=1), 1 buffer per SM for K>1
    bufs_per_sm = 2 if K == 1 else 1
    matrix_shmem_bytes = bufs_per_sm * tile_elems * elem_bytes
    # Optional: vector slice also in SRAM (eliminates LDG scoreboard stalls)
    vec_shmem_bytes = args.tile_cols * elem_bytes if args.vector_in_sram else 0
    vec_shmem_base = shmem_base + matrix_shmem_bytes  # start of vec slot in shmem
    shmem_bytes = matrix_shmem_bytes + vec_shmem_bytes

    # Register allocation for DRAM vector prefetch:
    #   R0-R1: temps/address regs
    #   R2-R5: current matrix element (LDS dest, int128 = 4×32-bit)
    #   R6-R9: accumulator (int128 = 4×32-bit)
    #   R10+:  prefetched vector elements (tile_cols × 4 regs each)
    VEC_REG_BASE = 10
    if not args.vector_in_sram:
        nregs = VEC_REG_BASE + args.tile_cols * 4 + 4  # +4 safety
        assert nregs <= 255, (
            f"Vector prefetch needs {nregs} regs for tile_cols={args.tile_cols} "
            f"(max 255). Reduce tile_cols to ≤{(255 - VEC_REG_BASE - 4) // 4}."
        )
    else:
        nregs = 32

    # Work distribution: each warp handles some rows within a tile
    rows_per_warp = max(1, args.tile_rows // warps_per_block)

    # GPU binary version mapping
    gpu_binary_versions = {"a100": 80, "v100": 70}
    binary_version = gpu_binary_versions.get(args.gpu, 80)

    lines = []

    # === Kernel header ===
    mode = "double-buffer" if K == 1 else f"{K}-group-pipeline"
    lines.append(f"-kernel name = custom_dma_{mode}_matvec")
    lines.append(f"-kernel id = 1")
    lines.append(f"-grid dim = ({total_blocks},1,1)")
    lines.append(f"-block dim = ({threads_per_block},1,1)")
    lines.append(f"-shmem = {shmem_bytes}")
    lines.append(f"-nregs = {nregs}")
    lines.append(f"-cuda stream id = 0")
    lines.append(f"-binary version = {binary_version}")
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
    for block_id in range(total_blocks):
        pc_counter[0] = 0x0010

        # Which SM group this block belongs to, and its index within the group
        group_id = block_id // args.num_blocks
        block_in_group = block_id % args.num_blocks

        lines.append("#BEGIN_TB")
        lines.append(f"thread block = {block_id},0,0")

        for warp_id in range(warps_per_block):
            warp_insts = []

            warp_row_start = warp_id * rows_per_warp

            # Each SM group works on a different slice of output rows,
            # so all K groups produce independent useful output.
            group_row_offset = group_id * args.tile_rows

            for loop_iter in range(args.num_loops):
                for tile in range(args.num_tiles):
                    # K=1: alternate A/B (double buffer)
                    # K>1: single buffer per SM (groups are the pipeline stages)
                    if K == 1:
                        compute_buf = buf_a_base if tile % 2 == 0 else buf_b_base
                    else:
                        compute_buf = buf_a_base  # single buffer

                    # --- BAR.SYNC: synchronization point ---
                    p = next_pc()
                    warp_insts.append(
                        f"{fmt_pc(p)} {MASK_ALL} 0 BAR.SYNC 0 0"
                    )

                    if not args.vector_in_sram:
                        # === DRAM VECTOR: PREFETCH all elements into regs ===
                        # Issue all tile_cols LDGs back-to-back into R10+.
                        # By the time compute starts (~tile_cols cycles later),
                        # L1D hits (~30cy) are covered; cold DRAM miss (~200cy)
                        # stalls once then all subsequent elements arrive.
                        for col in range(args.tile_cols):
                            vec_addr = (vector_global_base
                                        + loop_iter * args.tile_cols * elem_bytes
                                        + col * elem_bytes)
                            vec_reg = VEC_REG_BASE + col * 4
                            p = next_pc()
                            warp_insts.append(
                                f"{fmt_pc(p)} {MASK_ALL} 1 R{vec_reg} "
                                f"LDG.E.128 1 R0 16 1 {hex(vec_addr)} 0"
                            )

                        # Compute: LDS matrix from SRAM + IMAD with prefetched vec regs
                        for row in range(rows_per_warp):
                            global_row = warp_row_start + row
                            if global_row >= args.tile_rows:
                                break
                            for col in range(args.tile_cols):
                                elem_idx = global_row * args.tile_cols + col
                                shmem_addr = compute_buf + elem_idx * elem_bytes
                                vec_reg = VEC_REG_BASE + col * 4

                                # LDS: matrix element from SRAM (2 cyc)
                                p = next_pc()
                                warp_insts.append(
                                    f"{fmt_pc(p)} {MASK_ALL} 1 R2 LDS.128 1 R3 "
                                    f"16 1 {hex(shmem_addr)} 0"
                                )
                                # int128 MAC = 4× IMAD using prefetched vec reg
                                for part in range(4):
                                    p = next_pc()
                                    warp_insts.append(
                                        f"{fmt_pc(p)} {MASK_ALL} 1 R{6 + part} "
                                        f"IMAD 3 R2 R{vec_reg + part} R{6 + part} 0"
                                    )
                    else:
                        # === SRAM VECTOR: inline LDS for both operands (no prefetch) ===
                        for row in range(rows_per_warp):
                            global_row = warp_row_start + row
                            if global_row >= args.tile_rows:
                                break
                            for col in range(args.tile_cols):
                                elem_idx = global_row * args.tile_cols + col
                                shmem_addr = compute_buf + elem_idx * elem_bytes
                                vec_shmem_addr = vec_shmem_base + col * elem_bytes

                                # LDS: matrix element
                                p = next_pc()
                                warp_insts.append(
                                    f"{fmt_pc(p)} {MASK_ALL} 1 R2 LDS.128 1 R3 "
                                    f"16 1 {hex(shmem_addr)} 0"
                                )
                                # LDS: vector element
                                p = next_pc()
                                warp_insts.append(
                                    f"{fmt_pc(p)} {MASK_ALL} 1 R4 LDS.128 1 R5 "
                                    f"16 1 {hex(vec_shmem_addr)} 0"
                                )
                                # int128 MAC = 4× IMAD
                                for part in range(4):
                                    p = next_pc()
                                    warp_insts.append(
                                        f"{fmt_pc(p)} {MASK_ALL} 1 R{6 + part} "
                                        f"IMAD 3 R2 R4 R{6 + part} 0"
                                    )

                # --- End of one loop: store result to global memory ---
                # Each group writes to a different output region
                for row in range(rows_per_warp):
                    global_row = warp_row_start + row
                    if global_row >= args.tile_rows:
                        break
                    result_addr = (output_global_base
                                   + group_id * args.num_loops * args.tile_rows * elem_bytes
                                   + loop_iter * args.tile_rows * elem_bytes
                                   + (group_row_offset + global_row) * elem_bytes)
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
    K = args.num_sm_groups

    vector_size = args.num_loops * args.tile_cols * elem_bytes
    # K groups each produce num_loops * tile_rows output elements
    output_size = K * args.num_loops * args.tile_rows * elem_bytes

    lines = []
    lines.append(f"MemcpyHtoD,0x00007f0010000000,{vector_size}")
    lines.append(f"MemcpyHtoD,0x00007f0020000000,{output_size}")
    lines.append("kernel-1.traceg")
    return lines


def print_summary(args):
    """Print configuration summary."""
    elem_bytes = 16
    K = args.num_sm_groups
    tile_elems = args.tile_rows * args.tile_cols
    total_tiles = args.num_loops * args.num_tiles
    bufs_per_sm = 2 if K == 1 else 1
    matrix_shmem = bufs_per_sm * tile_elems * elem_bytes
    vec_shmem = args.tile_cols * elem_bytes if args.vector_in_sram else 0
    shmem_per_sm = matrix_shmem + vec_shmem
    total_blocks = args.num_blocks * K
    warps = args.threads_per_block // 32
    rows_per_warp = max(1, args.tile_rows // warps)
    if args.vector_in_sram:
        vec_inst = "LDS.128 (~2cy, inline)"
        insts_per_tile_per_warp = rows_per_warp * args.tile_cols * 6 + 1  # BAR + R×C×(LDS_mat+LDS_vec+4×IMAD)
    else:
        nregs_needed = 10 + args.tile_cols * 4 + 4
        vec_inst = f"LDG.E.128 → R10..R{10+args.tile_cols*4-1} (prefetch {args.tile_cols} elems, {nregs_needed} regs)"
        insts_per_tile_per_warp = args.tile_cols + rows_per_warp * args.tile_cols * 5 + 1  # BAR + C×LDG_prefetch + R×C×(LDS_mat+4×IMAD)

    mode = "double-buffer (K=1)" if K == 1 else f"{K}-group pipeline"
    print(f"\n{'='*62}")
    print(f"  TRACE CONFIG  [{mode}]")
    print(f"{'='*62}")
    print(f"  SM groups:       {K}  ×  {args.num_blocks} SMs  =  {total_blocks} total blocks")
    print(f"  SRAM / SM:       {shmem_per_sm} bytes  ({bufs_per_sm} matrix buf{'s' if bufs_per_sm>1 else ''}" +
          (f" + {vec_shmem}B vec slot" if args.vector_in_sram else "") + ")")
    print(f"  Tile:            {args.tile_rows}×{args.tile_cols} int128  = {tile_elems*elem_bytes} bytes")
    print(f"  Vector source:   {vec_inst}")
    print(f"  Tiles/loop:      {args.num_tiles}    Loops: {args.num_loops}    Total tiles/group: {total_tiles}")
    print(f"  Insts/tile/warp: {insts_per_tile_per_warp}")
    if K == 1:
        print(f"  Formula:  per-tile = max(FILL, COMPUTE)")
    else:
        print(f"  Formula:  per-tile/group = K×FILL if COMPUTE≤(K-1)×FILL (fill-bound)")
        print(f"                           = FILL+COMPUTE otherwise (compute-bound)")
        print(f"  Total throughput: {K}×  tiles/cycle across all groups")
    print(f"\n  After simulation, run:")
    print(f"    python3 analyze_results.py --sim-log <log> --fill-cycles <N> --num-sm-groups {K}")
    print(f"{'='*62}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate traces for DMA + double-buffer + int128 MxV"
    )
    parser.add_argument("--tile-rows", type=int, default=8,
                        help="Matrix tile rows (default: 8)")
    parser.add_argument("--tile-cols", type=int, default=8,
                        help="Matrix tile cols = vector length (default: 8)")
    parser.add_argument("--threads-per-block", type=int, default=32,
                        help="Threads per block (default: 32, =1 warp for max prefetch benefit)")
    parser.add_argument("--num-blocks", type=int, default=2,
                        help="Thread blocks (default: 2, full A100=108)")
    parser.add_argument("--num-tiles", type=int, default=4,
                        help="Tiles per loop for double buffering (default: 4)")
    parser.add_argument("--num-loops", type=int, default=3,
                        help="Number of outer loops (default: 3)")
    parser.add_argument("--num-sm-groups", type=int, default=1,
                        help="Number of SM groups for pipeline (1=double-buffer, 2/3=K-group; default: 1)")
    parser.add_argument("--vector-in-sram", action="store_true",
                        help="Place vector slice in SRAM (LDS ~2cy) instead of DRAM (LDG ~200cy). "
                             "Eliminates W0_Scoreboard stalls and W32 BAR divergence.")
    parser.add_argument("--gpu", type=str, default="a100", choices=["a100", "v100"],
                        help="Target GPU for binary_version in trace header (default: a100)")
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
