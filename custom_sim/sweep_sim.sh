#!/bin/bash
# Sweep FILL_CYCLES across multiple values in parallel SLURM jobs.
# Usage: bash custom_sim/sweep_sim.sh
# Each value gets its own job — all run simultaneously.

ACCELSIM_DIR=/share/suh-scrap2/ph448/work/second/accel-sim-framework
CUSTOM_DIR=$ACCELSIM_DIR/custom_sim

# Fill values to sweep (cycles)
FILL_VALUES=(100 200 500 1024 2000 5000)

# Shared config (use reduced blocks/loops for speed)
TILE_ROWS=32; TILE_COLS=32; THREADS_PER_BLOCK=256
NUM_BLOCKS=2          # reduced from 32 — identical work, same cycles/tile
NUM_TILES=8; NUM_LOOPS=1   # reduced loops — same cycles/tile
NUM_SM_GROUPS=1
CLOCK_MHZ=1410

for FC in "${FILL_VALUES[@]}"; do
    SWEEP_DIR=$CUSTOM_DIR/sweep_fill${FC}
    mkdir -p $SWEEP_DIR

    # Pre-generate traces for this config (same for all fill values)
    TRACE_DIR=$CUSTOM_DIR/traces_sweep
    mkdir -p $TRACE_DIR
    python3 $CUSTOM_DIR/scripts/gen_traces.py \
        --tile-rows $TILE_ROWS --tile-cols $TILE_COLS \
        --threads-per-block $THREADS_PER_BLOCK --num-blocks $NUM_BLOCKS \
        --num-tiles $NUM_TILES --num-loops $NUM_LOOPS \
        --num-sm-groups $NUM_SM_GROUPS --outdir $TRACE_DIR

    # Submit one job per fill value
    sbatch --job-name="sweep_f${FC}" \
           --output="$SWEEP_DIR/slurm-%j.out" \
           --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=20G \
           --time=4:00:00 --partition=suh \
           --wrap="
set -e
eval \"\$(conda shell.bash hook)\" && conda activate accelsim
export PATH=$ACCELSIM_DIR/.local/bin:\$PATH
cd $ACCELSIM_DIR/gpu-simulator
source setup_environment.sh
mkdir -p $SWEEP_DIR
cp $CUSTOM_DIR/configs/gpgpusim.config $SWEEP_DIR/
cp $CUSTOM_DIR/configs/trace.config $SWEEP_DIR/
cd $SWEEP_DIR
$ACCELSIM_DIR/gpu-simulator/bin/release/accel-sim.out \
    -trace $TRACE_DIR/kernelslist.g \
    -config gpgpusim.config -config trace.config \
    -gpgpu_max_completed_cta $NUM_BLOCKS \
    2>&1 | tee sim_output.log
python3 $CUSTOM_DIR/scripts/analyze_results.py \
    --sim-log sim_output.log \
    --fill-cycles $FC --num-tiles $NUM_TILES --num-loops $NUM_LOOPS \
    --num-sm-groups $NUM_SM_GROUPS --clock-mhz $CLOCK_MHZ \
    --tile-rows $TILE_ROWS --tile-cols $TILE_COLS --num-blocks 32
"
    echo "Submitted job for FILL_CYCLES=$FC → $SWEEP_DIR"
done
