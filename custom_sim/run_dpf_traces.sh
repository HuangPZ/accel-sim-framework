#!/bin/bash
#SBATCH -J accelsim_dpf
#SBATCH --mail-type=END
#SBATCH --mail-user=ph448@cornell.edu
#SBATCH -o ./custom_sim/zslurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --get-user-env
#SBATCH --mem 70G
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
ACCEL_SIM_BIN=$ACCELSIM_DIR/gpu-simulator/bin/release/accel-sim.out

# NVBit trace directory (pre-generated — skip trace generation)
TRACE_DIR=/share/suh-scrap2/ph448/work/GPU-DPF/nvbit_traces/DPF_HYBRID_KK16384_NN32_MM1_SALSA20_12_CIPHER_matmul0_fuse0
# Use the post-processed kernelslist.g which references .traceg.xz files
# (the format accel-sim expects with #BEGIN_TB/#END_TB block markers).
KERNELSLIST=$TRACE_DIR/kernelslist.g

# GPU config: SM80 A100
GPGPUSIM_CFG=$ACCELSIM_DIR/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config
TRACE_CFG=$ACCELSIM_DIR/gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate accelsim

###############################################################################
# Step 1: Setup simulator environment
###############################################################################
echo "=== Step 1: Setting up environment ==="
cd $ACCELSIM_DIR/gpu-simulator

export PATH=$ACCELSIM_DIR/.local/bin:$PATH
export GPGPUSIM_REPO=https://github.com/accel-sim/gpgpu-sim_distribution.git
export GPGPUSIM_BRANCH=dev
source setup_environment.sh

###############################################################################
# Step 2: Create run directory
###############################################################################
RUN_DIR=$ACCELSIM_DIR/custom_sim/run_dpf_$(date +%Y%m%d_%H%M%S)
mkdir -p $RUN_DIR
echo "Run directory: $RUN_DIR"

# Copy config files
cp $GPGPUSIM_CFG $RUN_DIR/gpgpusim.config
cp $TRACE_CFG    $RUN_DIR/trace.config

###############################################################################
# Step 3: Show what will be simulated
###############################################################################
echo ""
echo "=== Step 3: Kernels to simulate ==="
echo "kernelslist: $KERNELSLIST"
cat $KERNELSLIST

###############################################################################
# Step 4: Run simulation
###############################################################################
echo ""
echo "=== Step 4: Running Accel-Sim ==="
cd $RUN_DIR 

echo "Running: $ACCEL_SIM_BIN -trace $KERNELSLIST -config gpgpusim.config -config trace.config"
echo ""

# Note: kernelslist must live next to the .trace.xz files so that trace_parser
# can resolve the bare filenames by prepending its own directory.
$ACCEL_SIM_BIN \
    -trace $KERNELSLIST \
    -config gpgpusim.config \
    -config trace.config \
    2>&1 | tee sim_output.log

###############################################################################
# Step 5: Extract key stats
###############################################################################
echo ""
echo "=============================================="
echo "  Simulation complete!"
echo "  Full output: $RUN_DIR/sim_output.log"
echo "=============================================="
echo ""
echo "=== Key Performance Stats ==="
grep -E "gpu_sim_cycle|gpu_sim_insn|gpu_ipc|gpu_tot_sim_cycle|gpgpu_simulation_time|L2_cache_stats|total dram reads|total dram writes|gpu_stall|gpgpu_n_shmem" \
    $RUN_DIR/sim_output.log || true
echo ""
echo "=== Memory Stats ==="
grep -E "L1D|L2_total|dram_total|shmem" $RUN_DIR/sim_output.log | head -20 || true

echo "Done. All outputs in: $RUN_DIR"
