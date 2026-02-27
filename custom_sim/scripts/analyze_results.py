#!/usr/bin/env python3
"""
Post-processing script: combine Accel-Sim results with analytical DMA timing.

=== GLOBAL SYNCHRONIZATION MODEL ===
  The DMA and all SMs are globally synchronized at each tile boundary:

    1. DMA fills Buffer X on ALL SMs in parallel (all start and end together).
       Fill does NOT start until ALL SMs have finished reading Buffer X.

    2. ALL SMs start reading Buffer X only after the DMA fill of Buffer X is
       confirmed done. No SM reads ahead.

  This means:
    - The critical read-time is determined by the SLOWEST SM (global barrier).
    - gpu_sim_cycle (from GPGPU-Sim) is already the "last SM done" time,
      so it correctly represents the global sync barrier time.
    - compute_per_tile = gpu_sim_cycle / total_tiles  (worst-case SM, correct)

=== DOUBLE-BUFFER TIMELINE (per loop, N tiles) ===

    t=0:                  DMA fills Buffer A (tile 0)       [FILL_CYCLES]
    t=FILL:               ALL SMs read A  AND  DMA fills B  [max(FILL, COMPUTE)]
    t=FILL+max:           ALL SMs read B  AND  DMA fills A  [max(FILL, COMPUTE)]
    ...                   (N-1 overlapped phases)
    t=FILL+(N-1)*max:     ALL SMs read last buffer          [COMPUTE]

    loop_time = FILL + (N-1)*max(FILL, COMPUTE) + COMPUTE
              = FILL + COMPUTE + (N-1)*max(FILL, COMPUTE)

  Which simplifies to:
    fill <= compute:  loop = fill + N * compute   (fill hidden, compute-bound)
    fill >  compute:  loop = N * fill + compute   (SMs idle waiting, fill-bound)

  Total:  total_cycles = num_loops * loop_time

Usage:
  python3 analyze_results.py \\
    --sim-log sim_output.log \\
    --fill-cycles 1024 \\
    --num-tiles 8 \\
    --num-loops 4 \\
    --clock-mhz 1410
"""

import argparse
import re
import sys


def extract_sim_stats(log_path):
    """Parse key stats from Accel-Sim output log."""
    stats = {}
    with open(log_path, "r") as f:
        for line in f:
            # gpu_sim_cycle = 12345
            m = re.search(r"gpu_sim_cycle\s*=\s*(\d+)", line)
            if m:
                stats["gpu_sim_cycle"] = int(m.group(1))

            m = re.search(r"gpu_tot_sim_cycle\s*=\s*(\d+)", line)
            if m:
                stats["gpu_tot_sim_cycle"] = int(m.group(1))

            m = re.search(r"gpu_sim_insn\s*=\s*(\d+)", line)
            if m:
                stats["gpu_sim_insn"] = int(m.group(1))

            m = re.search(r"gpu_ipc\s*=\s*([\d.]+)", line)
            if m:
                stats["gpu_ipc"] = float(m.group(1))

            m = re.search(r"gpgpu_simulation_time\s*=.*?(\d+)\s*sec", line)
            if m:
                stats["wall_time_sec"] = int(m.group(1))

            # Shared memory stats
            m = re.search(r"gpgpu_n_shmem_insn\s*=\s*(\d+)", line)
            if m:
                stats["shmem_insn"] = int(m.group(1))

            # L2 stats
            m = re.search(r"L2_total_cache_accesses\s*=\s*(\d+)", line)
            if m:
                stats["l2_accesses"] = int(m.group(1))

            m = re.search(r"L2_total_cache_misses\s*=\s*(\d+)", line)
            if m:
                stats["l2_misses"] = int(m.group(1))

            # DRAM
            m = re.search(r"total dram reads\s*=\s*(\d+)", line)
            if m:
                stats["dram_reads"] = int(m.group(1))

    return stats


def analyze(args, stats):
    """Apply globally-synchronized double-buffer formula."""
    sim_cycles = stats.get("gpu_sim_cycle") or stats.get("gpu_tot_sim_cycle", 0)
    total_tiles = args.num_tiles * args.num_loops

    if total_tiles == 0 or args.num_tiles == 0:
        print("ERROR: num_tiles and num_loops must be > 0")
        sys.exit(1)

    # gpu_sim_cycle is the time until the LAST SM finishes all work.
    # Dividing by total_tiles gives the per-tile time at the global barrier:
    # no SM can start the next tile until the slowest SM finishes this tile.
    # This is exactly the right metric for global-sync double buffering.
    compute_per_tile = sim_cycles / total_tiles

    fill = args.fill_cycles

    # Global-sync double-buffer formula per loop (N = num_tiles):
    #
    #   loop_time = FILL + (N-1)*max(FILL, COMPUTE) + COMPUTE
    #
    #   fill <= compute: loop = fill + N*compute   (DMA always ready, compute-bound)
    #   fill >  compute: loop = N*fill + compute   (SMs idle at each barrier, fill-bound)
    #
    # The DMA cannot start filling buffer X until ALL SMs finish reading buffer X
    # (global barrier). All SMs start reading buffer X only after fill is done.
    if fill <= compute_per_tile:
        loop_time = fill + args.num_tiles * compute_per_tile
        regime = "COMPUTE-BOUND (fill <= compute; fill completes before SMs finish reading)"
    else:
        loop_time = args.num_tiles * fill + compute_per_tile
        regime = "FILL-BOUND (fill > compute; all SMs idle at each global barrier)"

    total_cycles = args.num_loops * loop_time

    # Utilization: how much of total time is spent doing useful compute
    total_compute = total_tiles * compute_per_tile
    utilization = total_compute / total_cycles if total_cycles > 0 else 0

    # Wall-clock time for compute phase only (what the simulator measured)
    freq_hz = args.clock_mhz * 1e6
    compute_time_us = sim_cycles / freq_hz * 1e6
    total_time_us = total_cycles / freq_hz * 1e6

    # ── Throughput of consuming the matrix (SRAM reads) ──────────────────────
    # All num_blocks SMs run in parallel; sim_cycles is the wall-clock time.
    # Each SM reads tile_rows × tile_cols × elem_bytes per tile.
    elem_bytes = 16  # int128
    tile_bytes = args.tile_rows * args.tile_cols * elem_bytes

    # Total matrix bytes consumed (all SMs, all tiles, all loops)
    matrix_bytes_total = args.num_blocks * total_tiles * tile_bytes
    matrix_bytes_per_sm = total_tiles * tile_bytes

    # DRAM vector bytes: each loop uses a different vector slice (tile_cols × elem_bytes)
    vector_bytes_per_sm_per_loop = args.tile_cols * elem_bytes
    vector_bytes_total = args.num_blocks * args.num_loops * vector_bytes_per_sm_per_loop

    # Compute throughput using the actual sim_cycles (compute wall-clock time)
    matrix_bw_total_GBs  = (matrix_bytes_total / 1e9) / (sim_cycles / freq_hz) if sim_cycles else 0
    matrix_bw_per_sm_GBs = (matrix_bytes_per_sm / 1e9) / (sim_cycles / freq_hz) if sim_cycles else 0
    vector_bw_GBs        = (vector_bytes_total / 1e9) / (sim_cycles / freq_hz) if sim_cycles else 0

    # int128 MACs: 1 per (row, col) element per tile per loop per SM
    total_int128_macs = args.num_blocks * total_tiles * args.tile_rows * args.tile_cols
    # Each int128 MAC = 4 int32 MAD instructions = 8 int32 ops (mul + add)
    total_int32_ops = total_int128_macs * 8
    gops = (total_int32_ops / 1e9) / (sim_cycles / freq_hz) if sim_cycles else 0
    tops = gops / 1e3

    return {
        "sim_compute_cycles": sim_cycles,
        "compute_per_tile": compute_per_tile,
        "fill_per_tile": fill,
        "regime": regime,
        "loop_time": loop_time,
        "total_cycles": total_cycles,
        "utilization_pct": utilization * 100,
        "compute_time_us": compute_time_us,
        "total_time_us": total_time_us,
        # throughput
        "matrix_bytes_total": matrix_bytes_total,
        "matrix_bytes_per_sm": matrix_bytes_per_sm,
        "vector_bytes_total": vector_bytes_total,
        "matrix_bw_total_GBs": matrix_bw_total_GBs,
        "matrix_bw_per_sm_GBs": matrix_bw_per_sm_GBs,
        "vector_bw_GBs": vector_bw_GBs,
        "total_int128_macs": total_int128_macs,
        "total_int32_ops": total_int32_ops,
        "gops": gops,
        "tops": tops,
    }


def print_report(args, stats, results):
    """Print formatted analysis report."""
    print()
    print("=" * 64)
    print("  DOUBLE-BUFFER DMA + MxV ANALYSIS")
    print("=" * 64)

    print(f"\n  --- Simulation Results (compute only) ---")
    print(f"  Simulated compute cycles:   {stats.get('gpu_sim_cycle', 'N/A')}")
    print(f"  Simulated instructions:     {stats.get('gpu_sim_insn', 'N/A')}")
    print(f"  Simulated IPC:              {stats.get('gpu_ipc', 'N/A')}")
    print(f"  Shared memory instructions: {stats.get('shmem_insn', 'N/A')}")
    print(f"  L2 accesses:                {stats.get('l2_accesses', 'N/A')}")
    print(f"  L2 misses:                  {stats.get('l2_misses', 'N/A')}")
    print(f"  DRAM reads:                 {stats.get('dram_reads', 'N/A')}")

    print(f"\n  --- Configuration ---")
    print(f"  Tiles per loop:  {args.num_tiles}")
    print(f"  Number of loops: {args.num_loops}")
    print(f"  Total tiles:     {args.num_tiles * args.num_loops}")
    print(f"  DMA fill cycles: {args.fill_cycles} (user-specified)")
    print(f"  Clock frequency: {args.clock_mhz} MHz")

    fill = args.fill_cycles
    cpt = results["compute_per_tile"]
    N = args.num_tiles

    print(f"\n  --- Global-Sync Double-Buffer Analysis ---")
    print(f"  [NOTE] gpu_sim_cycle = time until LAST SM done = correct global barrier time")
    print(f"  Compute per tile (slowest SM):  {cpt:.1f} cycles")
    print(f"  DMA fill per tile (all SMs):    {fill} cycles  [your parameter]")
    print(f"  Regime:  {results['regime']}")
    print()

    # Show the explicit timeline for one loop
    t = 0
    print(f"  --- Timeline (1 loop, {N} tiles, global sync) ---")
    print(f"  t={t:>8.0f}:  DMA fills Buffer A (tile 0) on ALL SMs   [{fill} cyc]")
    t += fill
    for i in range(1, N):
        buf_read = "A" if (i - 1) % 2 == 0 else "B"
        buf_fill = "B" if i % 2 == 1 else "A"
        phase = max(fill, cpt)
        print(f"  t={t:>8.0f}:  ALL SMs read {buf_read} (tile {i-1})  +  "
              f"DMA fills {buf_fill} (tile {i})  [max({fill},{cpt:.0f})={phase:.0f} cyc]")
        t += phase
    buf_last = "A" if (N - 1) % 2 == 0 else "B"
    print(f"  t={t:>8.0f}:  ALL SMs read {buf_last} (tile {N-1}) — last tile, no fill  [{cpt:.0f} cyc]")
    t += cpt
    print(f"  t={t:>8.0f}:  Loop done")
    print()

    print(f"  Time per loop:        {results['loop_time']:.1f} cycles")
    print(f"  TOTAL CYCLES:         {results['total_cycles']:.0f}  ({args.num_loops} loops)")
    print(f"  Compute time only:    {results['compute_time_us']:.2f} us  (sim measured)")
    print(f"  Total w/ DMA:         {results['total_time_us']:.2f} us")
    print(f"  SM utilization:       {results['utilization_pct']:.1f}%")

    print(f"\n  --- Matrix Consumption Throughput (compute phase) ---")
    print(f"  Tile size:            {args.tile_rows} × {args.tile_cols} int128 = {args.tile_rows*args.tile_cols*16} bytes")
    print(f"  SMs (blocks):         {args.num_blocks}  (all in parallel)")
    print(f"  Matrix data/SM:       {results['matrix_bytes_per_sm']/1024:.1f} KB  ({args.num_tiles}tiles×{args.num_loops}loops)")
    print(f"  Matrix data total:    {results['matrix_bytes_total']/1024/1024:.2f} MB  (all {args.num_blocks} SMs)")
    print(f"  SRAM BW total:        {results['matrix_bw_total_GBs']:.1f} GB/s  (all SMs combined)")
    print(f"  SRAM BW per SM:       {results['matrix_bw_per_sm_GBs']:.2f} GB/s")
    print(f"  DRAM vector data:     {results['vector_bytes_total']/1024:.1f} KB total  ({results['vector_bw_GBs']:.2f} GB/s)")
    print(f"  int128 MACs:          {results['total_int128_macs']:,}  →  {results['total_int32_ops']:,} int32 ops")
    print(f"  Compute throughput:   {results['gops']:.1f} GOPS  ({results['tops']:.4f} TOPS)")

    # Sensitivity: show what happens at different fill_cycles
    print(f"\n  --- Sensitivity: fill_cycles sweep (compute_per_tile = {cpt:.0f}) ---")
    print(f"  {'fill_cyc':>10}  {'total_cyc':>12}  {'util%':>8}  {'regime'}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*20}")
    sweep_vals = sorted(set([10, 50, 100, 200, 500, 1000, 2000, 5000, args.fill_cycles]))
    for fc in sweep_vals:
        if fc <= cpt:
            lt = fc + N * cpt
        else:
            lt = N * fc + cpt
        tc = args.num_loops * lt
        u = (N * args.num_loops * cpt) / tc * 100 if tc > 0 else 0
        r = "compute" if fc <= cpt else "fill"
        marker = "  <-- current" if fc == args.fill_cycles else ""
        print(f"  {fc:>10}  {tc:>12.0f}  {u:>7.1f}%  {r}{marker}")

    print()
    print("=" * 64)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Combine Accel-Sim results with analytical DMA timing"
    )
    parser.add_argument("--sim-log", type=str, required=True,
                        help="Path to Accel-Sim output log")
    parser.add_argument("--fill-cycles", type=int, required=True,
                        help="DMA fill time per tile in GPU clock cycles")
    parser.add_argument("--num-tiles", type=int, required=True,
                        help="Tiles per loop (must match trace generation)")
    parser.add_argument("--num-loops", type=int, required=True,
                        help="Number of outer loops (must match trace generation)")
    parser.add_argument("--clock-mhz", type=int, default=1410,
                        help="GPU clock freq in MHz for wall-time estimate (default: 1410)")
    parser.add_argument("--tile-rows", type=int, default=32,
                        help="Tile rows (must match trace generation, default: 32)")
    parser.add_argument("--tile-cols", type=int, default=32,
                        help="Tile cols (must match trace generation, default: 32)")
    parser.add_argument("--num-blocks", type=int, default=32,
                        help="Number of thread blocks/SMs (must match trace generation, default: 32)")

    args = parser.parse_args()

    stats = extract_sim_stats(args.sim_log)
    if "gpu_sim_cycle" not in stats and "gpu_tot_sim_cycle" not in stats:
        print(f"ERROR: Could not find gpu_sim_cycle in {args.sim_log}")
        print("Make sure the simulation completed successfully.")
        sys.exit(1)

    results = analyze(args, stats)
    print_report(args, stats, results)


if __name__ == "__main__":
    main()
