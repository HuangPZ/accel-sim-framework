#!/bin/bash
#SBATCH -J accelsim_custom
#SBATCH -o ./custom_sim/zslurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --get-user-env
#SBATCH --mem 50G
#SBATCH -t 48:00:00
#SBATCH --requeue
#SBATCH --partition=suh

set -e

###############################################################################
# Paths
###############################################################################
export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH

ACCELSIM_DIR=/share/suh-scrap2/ph448/work/second/accel-sim-framework
CUSTOM_DIR=$ACCELSIM_DIR/custom_sim
ACCEL_SIM_BIN=$ACCELSIM_DIR/gpu-simulator/bin/release/accel-sim.out

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate accelsim

###############################################################################
# Configuration — adjust these parameters
###############################################################################
TILE_ROWS=32
TILE_COLS=32
THREADS_PER_BLOCK=256
NUM_BLOCKS=32          # thread blocks (full A100 = 108)
NUM_TILES=8           # tiles per loop for double buffering
NUM_LOOPS=4           # outer loop iterations
NUM_SM_GROUPS=2       # 1=double-buffer (2 SRAM bufs/SM), 2/3=K-group pipeline (1 SRAM buf/SM, K× SMs)
FILL_CYCLES=1024       # DMA fill cycles per tile (your controllable parameter)
CLOCK_MHZ=1410        # GPU clock for wall-time estimate
VECTOR_IN_SRAM=0      # 1=vector slice in SRAM via LDS (~2cy, NO scoreboard/BAR stalls)
                      # 0=vector from DRAM via LDG (~200cy, causes 43% W0_Scoreboard + W32)

# Quick-test override: uncomment to use a tiny config (~5K lines, fast sim)
# TILE_ROWS=8; TILE_COLS=8; NUM_BLOCKS=2; NUM_TILES=2; NUM_LOOPS=2; NUM_SM_GROUPS=1

# FAST-SIM override: all blocks do identical work, so reduce blocks+loops for speed.
# Cycles-per-tile result is unchanged; analyze_results.py extrapolates throughput.
# Typical speedup: 16x-32x vs full config.
# UNCOMMENT to enable:
# NUM_BLOCKS=2; NUM_LOOPS=1

###############################################################################
# Step 1: Generate traces (always regenerate fresh — never use stale traces)
###############################################################################
echo "=== Step 1: Generating traces ==="
rm -f $CUSTOM_DIR/traces/kernelslist.g $CUSTOM_DIR/traces/kernel-*.traceg

python3 $CUSTOM_DIR/scripts/gen_traces.py \
    --tile-rows $TILE_ROWS \
    --tile-cols $TILE_COLS \
    --threads-per-block $THREADS_PER_BLOCK \
    --num-blocks $NUM_BLOCKS \
    --num-tiles $NUM_TILES \
    --num-loops $NUM_LOOPS \
    --num-sm-groups $NUM_SM_GROUPS \
    $([ "$VECTOR_IN_SRAM" = "1" ] && echo --vector-in-sram) \
    --outdir $CUSTOM_DIR/traces

echo ""
echo "Generated files:"
ls -la $CUSTOM_DIR/traces/

echo ""
echo "=== kernelslist.g ==="
cat $CUSTOM_DIR/traces/kernelslist.g

echo ""
echo "=== First 50 lines of kernel-1.traceg ==="
head -50 $CUSTOM_DIR/traces/kernel-1.traceg

###############################################################################
# Step 2: Setup simulator environment
###############################################################################
echo ""
echo "=== Step 2: Setting up environment ==="
cd $ACCELSIM_DIR/gpu-simulator

# Add makedepend shim to PATH
export PATH=$ACCELSIM_DIR/.local/bin:$PATH

export GPGPUSIM_REPO=https://github.com/accel-sim/gpgpu-sim_distribution.git
export GPGPUSIM_BRANCH=dev
source setup_environment.sh

###############################################################################
# Step 3: Run simulation
###############################################################################
echo ""
echo "=== Step 3: Running Accel-Sim ==="

# Create a run directory
RUN_DIR=$CUSTOM_DIR/run_$(date +%Y%m%d_%H%M%S)
mkdir -p $RUN_DIR

echo "Run directory: $RUN_DIR"

# The simulator expects configs in the working directory
cp $CUSTOM_DIR/configs/gpgpusim.config $RUN_DIR/
cp $CUSTOM_DIR/configs/trace.config $RUN_DIR/

cd $RUN_DIR

echo ""
echo "Running: $ACCEL_SIM_BIN -trace $CUSTOM_DIR/traces/kernelslist.g -config gpgpusim.config -config trace.config"
echo ""

$ACCEL_SIM_BIN \
    -trace $CUSTOM_DIR/traces/kernelslist.g \
    -config gpgpusim.config \
    -config trace.config \
    -gpgpu_max_completed_cta $((NUM_BLOCKS * NUM_SM_GROUPS)) \
    2>&1 | tee sim_output.log

###############################################################################
# Step 4: Extract key stats
###############################################################################
echo ""
echo "=============================================="
echo "  Simulation complete!"
echo "  Full output: $RUN_DIR/sim_output.log"
echo "=============================================="
echo ""
echo "=== Key Performance Stats ==="
grep -E "gpu_sim_cycle|gpu_sim_insn|gpu_ipc|gpu_tot_sim_cycle|gpgpu_simulation_time|L2_cache_stats|total dram reads|total dram writes|gpu_stall|gpgpu_n_shmem" $RUN_DIR/sim_output.log || true
echo ""
echo "=== Memory Stats ==="
grep -E "L1D|L2_total|dram_total|shmem" $RUN_DIR/sim_output.log | head -20 || true

###############################################################################
# Step 5: Analytical post-processing (combine sim + DMA timing)
###############################################################################
echo ""
echo "=== Step 5: Double-Buffer Analysis ==="
python3 $CUSTOM_DIR/scripts/analyze_results.py \
    --sim-log $RUN_DIR/sim_output.log \
    --fill-cycles $FILL_CYCLES \
    --num-tiles $NUM_TILES \
    --num-loops $NUM_LOOPS \
    --clock-mhz $CLOCK_MHZ \
    --tile-rows $TILE_ROWS \
    --tile-cols $TILE_COLS \
    --num-blocks $NUM_BLOCKS \
    --num-sm-groups $NUM_SM_GROUPS

echo "Done. All outputs in: $RUN_DIR"
