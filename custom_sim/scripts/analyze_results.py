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
    """Apply timing formula for double-buffer (K=1) or K-group pipeline (K>1)."""
    sim_cycles = stats.get("gpu_sim_cycle") or stats.get("gpu_tot_sim_cycle", 0)
    total_tiles = args.num_tiles * args.num_loops
    K = args.num_sm_groups

    if total_tiles == 0:
        print("ERROR: num_tiles and num_loops must be > 0")
        sys.exit(1)

    # sim_cycles = time until LAST SM finishes.
    # With K groups all running simultaneously on different SMs, each group
    # processes total_tiles tiles. sim_cycles covers all groups in parallel.
    # compute_per_tile is per-group (all groups do identical work).
    compute_per_tile = sim_cycles / total_tiles
    fill = args.fill_cycles
    freq_hz = args.clock_mhz * 1e6

    if K == 1:
        # ── Double buffer (single group, 2 SRAM buffers per SM) ──────────────
        # Per loop: fill + N*compute (compute-bound) or N*fill + compute (fill-bound)
        if fill <= compute_per_tile:
            time_per_tile = compute_per_tile          # fill hidden
            regime = "COMPUTE-BOUND  (fill ≤ compute; DMA always finishes in time)"
        else:
            time_per_tile = fill                      # SMs wait for DMA
            regime = "FILL-BOUND  (fill > compute; all SMs idle at each barrier)"

        total_group_time = args.num_loops * (fill + (args.num_tiles - 1) * max(fill, compute_per_tile) + compute_per_tile)
        total_cycles = total_group_time
        throughput_tiles_per_cycle = 1.0 / max(fill, compute_per_tile)
        num_groups_str = "1 group (double-buffer)"

    else:
        # ── K-group pipeline (K groups, each with 1 SRAM buffer per SM) ───────
        # DMA serves groups round-robin. While DMA fills group k,
        # all other K-1 groups compute independently.
        #
        # Per-tile time per group:
        #   fill-bound    (compute ≤ (K-1)×fill): K×fill  per tile
        #   compute-bound (compute >  (K-1)×fill): fill+compute per tile
        if compute_per_tile <= (K - 1) * fill:
            time_per_tile = K * fill
            regime = f"FILL-BOUND  (compute ≤ (K-1)×fill = {(K-1)*fill:.0f}; DMA paces all {K} groups)"
        else:
            time_per_tile = fill + compute_per_tile
            regime = f"COMPUTE-BOUND  (compute > (K-1)×fill = {(K-1)*fill:.0f}; DMA always ready)"

        total_cycles = total_tiles * time_per_tile    # per-group time
        # Total throughput = K groups completing tiles simultaneously
        throughput_tiles_per_cycle = K / time_per_tile
        num_groups_str = f"{K} groups (single-buffer each, {K}× SMs)"

    utilization = compute_per_tile / time_per_tile if time_per_tile > 0 else 0
    compute_time_us = sim_cycles / freq_hz * 1e6
    total_time_us = total_cycles / freq_hz * 1e6

    # ── Throughput metrics ────────────────────────────────────────────────────
    elem_bytes = 16  # int128
    tile_bytes = args.tile_rows * args.tile_cols * elem_bytes

    # Total matrix bytes consumed: K groups × total_tiles × tile_bytes (all SMs in parallel)
    matrix_bytes_total = args.num_blocks * K * total_tiles * tile_bytes
    matrix_bytes_per_sm = total_tiles * tile_bytes

    vector_bytes_total = args.num_blocks * K * args.num_loops * args.tile_cols * elem_bytes

    # Throughput based on total wall-clock time (sim_cycles = all groups in parallel)
    wall_s = sim_cycles / freq_hz
    matrix_bw_total_GBs  = (matrix_bytes_total / 1e9) / wall_s if wall_s else 0
    matrix_bw_per_sm_GBs = (matrix_bytes_per_sm / 1e9) / wall_s if wall_s else 0
    vector_bw_GBs        = (vector_bytes_total / 1e9) / wall_s if wall_s else 0

    total_int128_macs = args.num_blocks * K * total_tiles * args.tile_rows * args.tile_cols
    total_int32_ops = total_int128_macs * 8
    gops = (total_int32_ops / 1e9) / wall_s if wall_s else 0

    return {
        "K": K,
        "num_groups_str": num_groups_str,
        "sim_compute_cycles": sim_cycles,
        "compute_per_tile": compute_per_tile,
        "fill_per_tile": fill,
        "time_per_tile": time_per_tile,
        "regime": regime,
        "total_cycles": total_cycles,
        "throughput_tiles_per_cycle": throughput_tiles_per_cycle,
        "utilization_pct": utilization * 100,
        "compute_time_us": compute_time_us,
        "total_time_us": total_time_us,
        "matrix_bytes_total": matrix_bytes_total,
        "matrix_bytes_per_sm": matrix_bytes_per_sm,
        "vector_bytes_total": vector_bytes_total,
        "matrix_bw_total_GBs": matrix_bw_total_GBs,
        "matrix_bw_per_sm_GBs": matrix_bw_per_sm_GBs,
        "vector_bw_GBs": vector_bw_GBs,
        "total_int128_macs": total_int128_macs,
        "total_int32_ops": total_int32_ops,
        "gops": gops,
    }


def print_report(args, stats, results):
    K = results["K"]
    cpt = results["compute_per_tile"]
    fill = args.fill_cycles
    N = args.num_tiles

    print()
    print("=" * 66)
    print(f"  DMA + MxV ANALYSIS  [{results['num_groups_str']}]")
    print("=" * 66)

    print(f"\n  --- Simulation Results (compute only, all {K}×{args.num_blocks} SMs) ---")
    print(f"  gpu_sim_cycle:              {stats.get('gpu_sim_cycle', 'N/A')}")
    print(f"  gpu_sim_insn:               {stats.get('gpu_sim_insn', 'N/A')}")
    print(f"  IPC:                        {stats.get('gpu_ipc', 'N/A')}")
    print(f"  Shared mem instructions:    {stats.get('shmem_insn', 'N/A')}")
    print(f"  DRAM reads:                 {stats.get('dram_reads', 'N/A')}")

    print(f"\n  --- Configuration ---")
    print(f"  SM groups (K):    {K}   ({results['num_groups_str']})")
    print(f"  SMs per group:    {args.num_blocks}   (total SMs = {K*args.num_blocks})")
    print(f"  Tile:             {args.tile_rows}×{args.tile_cols} int128  = {args.tile_rows*args.tile_cols*16} bytes")
    bufs = 2 if K == 1 else 1
    print(f"  SRAM / SM:        {bufs} buffer{'s' if bufs>1 else ''}  = {bufs*args.tile_rows*args.tile_cols*16} bytes")
    print(f"  Tiles/loop:       {N}    Loops: {args.num_loops}    Total tiles/group: {N*args.num_loops}")
    print(f"  DMA fill cycles:  {fill}  [your parameter]")
    print(f"  Clock:            {args.clock_mhz} MHz")

    print(f"\n  --- Timing Analysis ---")
    print(f"  Compute per tile:  {cpt:.1f} cycles  (from simulation)")
    print(f"  DMA fill per tile: {fill} cycles")
    print(f"  Regime:            {results['regime']}")
    print(f"  Time per tile:     {results['time_per_tile']:.1f} cycles/tile/group")
    print(f"  SM utilization:    {results['utilization_pct']:.1f}%")
    print(f"  Total cycles:      {results['total_cycles']:.0f}  (per group, {args.num_loops} loops)")
    print(f"  Compute time:      {results['compute_time_us']:.2f} µs  (sim measured)")
    print(f"  Total w/ DMA:      {results['total_time_us']:.2f} µs  (per group)")

    # Timeline for one loop
    print(f"\n  --- Pipeline Timeline (1 loop, K={K}, {N} tiles/group) ---")
    if K == 1:
        t = 0
        print(f"  t={t:>7.0f}:  DMA → all SMs fill Buf A  [{fill} cyc]")
        t += fill
        for i in range(1, N):
            ba = "A" if (i-1)%2==0 else "B"
            bf = "B" if i%2==1 else "A"
            ph = max(fill, cpt)
            print(f"  t={t:>7.0f}:  all SMs compute {ba}  +  DMA fill {bf}  [max({fill},{cpt:.0f})={ph:.0f}]")
            t += ph
        bl = "A" if (N-1)%2==0 else "B"
        print(f"  t={t:>7.0f}:  all SMs compute {bl} (last)  [{cpt:.0f} cyc]")
        t += cpt
        print(f"  t={t:>7.0f}:  loop done")
    else:
        t = 0
        print(f"  t={t:>7.0f}:  DMA → G0 fill  [{fill} cyc]  G1..G{K-1} idle (startup)")
        t += fill
        for g in range(1, K):
            others = [f"G{x} compute" for x in range(g)]
            print(f"  t={t:>7.0f}:  DMA → G{g} fill  [{fill} cyc]  {', '.join(others)}")
            t += fill
        print(f"  ... steady state ...")
        tpt = results["time_per_tile"]
        print(f"  Each group: {tpt:.0f} cyc/tile  (DMA serves K groups round-robin)")

    print(f"\n  --- Throughput (compute phase, all groups in parallel) ---")
    print(f"  Matrix data total:     {results['matrix_bytes_total']/1024/1024:.2f} MB  ({K}groups×{args.num_blocks}SMs×{N*args.num_loops}tiles)")
    print(f"  Matrix data / SM:      {results['matrix_bytes_per_sm']/1024:.1f} KB")
    print(f"  SRAM BW (all SMs):     {results['matrix_bw_total_GBs']:.1f} GB/s")
    print(f"  SRAM BW / SM:          {results['matrix_bw_per_sm_GBs']:.2f} GB/s")
    print(f"  DRAM vector BW:        {results['vector_bw_GBs']:.2f} GB/s")
    print(f"  int128 MACs:           {results['total_int128_macs']:,}  →  {results['total_int32_ops']:,} int32 ops")
    print(f"  Compute throughput:    {results['gops']:.1f} GOPS")
    print(f"  Tile throughput:       {results['throughput_tiles_per_cycle']*args.clock_mhz*1e6/1e9:.2f} Gtiles/s")

    # Sensitivity sweep: compare K=1 double-buffer vs K=2,3 pipeline
    print(f"\n  --- Sensitivity: fill_cycles sweep (compute={cpt:.0f}, K={K}) ---")
    print(f"  {'fill_cyc':>10}  {'time/tile':>10}  {'util%':>8}  {'regime'}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*30}")
    sweep_vals = sorted(set([10, 50, 100, 200, 500, 1000, 2000, 5000, fill]))
    for fc in sweep_vals:
        if K == 1:
            tpt_s = max(fc, cpt)
            r = "compute" if fc <= cpt else "fill"
        else:
            if cpt <= (K-1)*fc:
                tpt_s = K*fc
                r = f"fill (K×fill={K*fc:.0f})"
            else:
                tpt_s = fc + cpt
                r = f"compute (fill+cpt={fc+cpt:.0f})"
        u = cpt / tpt_s * 100 if tpt_s else 0
        marker = "  ←" if fc == fill else ""
        print(f"  {fc:>10}  {tpt_s:>10.0f}  {u:>7.1f}%  {r}{marker}")

    print()
    print("=" * 66)
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
    parser.add_argument("--num-sm-groups", type=int, default=1,
                        help="Number of SM groups (1=double-buffer, 2/3=K-group; default: 1)")
    parser.add_argument("--clock-mhz", type=int, default=1410,
                        help="GPU clock freq in MHz (default: 1410)")
    parser.add_argument("--tile-rows", type=int, default=32,
                        help="Tile rows (must match trace generation, default: 32)")
    parser.add_argument("--tile-cols", type=int, default=32,
                        help="Tile cols (must match trace generation, default: 32)")
    parser.add_argument("--num-blocks", type=int, default=32,
                        help="SMs per group (must match trace generation, default: 32)")

    args = parser.parse_args()

    stats = extract_sim_stats(args.sim_log)
    if "gpu_sim_cycle" not in stats and "gpu_tot_sim_cycle" not in stats:
        print(f"ERROR: Could not find gpu_sim_cycle in {args.sim_log}")
        sys.exit(1)

    results = analyze(args, stats)
    print_report(args, stats, results)


if __name__ == "__main__":
    main()
