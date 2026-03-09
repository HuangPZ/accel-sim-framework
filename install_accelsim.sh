#!/bin/bash
#SBATCH -J accelsim_install
#SBATCH -o ./zslurm-%j.out
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
# 0. Paths — adjust these if needed
###############################################################################
export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH

ACCELSIM_DIR=/share/suh-scrap2/ph448/work/second/accel-sim-framework
CONDA_ENV_NAME=accelsim

###############################################################################
# 1. Verify CUDA is accessible on the compute node
###############################################################################
echo "=== Checking CUDA ==="
nvcc --version || { echo "ERROR: nvcc not found at $CUDA_INSTALL_PATH"; exit 1; }

# ###############################################################################
# # 2. Create conda environment with all build dependencies
# #    (no sudo needed — everything comes from conda-forge)
# ###############################################################################
# echo "=== Setting up conda environment ==="

# # Initialize conda for this shell
# eval "$(conda shell.bash hook)"

# # Create env if it doesn't exist
# if ! conda info --envs | grep -q "^${CONDA_ENV_NAME} "; then
#     conda create -n $CONDA_ENV_NAME python=3.10 -y
# fi
# conda activate $CONDA_ENV_NAME

# # Install C/C++ build dependencies from conda-forge
# # These provide the -dev/-devel headers that accel-sim needs
# conda install -c conda-forge -y \
#     zlib \
#     boost-cpp \
#     libxml2 \
#     bison \
#     flex \
#     mesalib \
#     xorg-libx11 \
#     openssl

# # Install Python dependencies
# pip install -r $ACCELSIM_DIR/requirements.txt

###############################################################################
# 3. Export include/lib paths so the build can find conda-installed libs
###############################################################################
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

###############################################################################
# 3b. Create a makedepend shim (xutils-dev is not installed and we can't sudo)
#     makedepend is only used for header dependency generation — not critical
#     for a clean build. This no-op shim lets make proceed.
###############################################################################
SHIM_DIR=$ACCELSIM_DIR/.local/bin
mkdir -p $SHIM_DIR
cat > $SHIM_DIR/makedepend << 'SHIM'
#!/bin/bash
# no-op shim for makedepend (xutils-dev not available)
# Touch the output file if -f is specified so make doesn't fail
for arg in "$@"; do
    case "$prev" in
        -f) touch "$arg" 2>/dev/null ;;
    esac
    prev="$arg"
done
exit 0
SHIM
chmod +x $SHIM_DIR/makedepend
export PATH=$SHIM_DIR:$PATH
echo "=== Created makedepend shim at $SHIM_DIR/makedepend ==="

###############################################################################
# 4. Build Accel-Sim
###############################################################################
cd $ACCELSIM_DIR/gpu-simulator

echo "=== Sourcing setup_environment.sh ==="
# Non-interactive: pre-set the repo/branch so it won't prompt
export GPGPUSIM_REPO=https://github.com/accel-sim/gpgpu-sim_distribution.git
export GPGPUSIM_BRANCH=dev
source setup_environment.sh

echo "=== Building Accel-Sim ==="
make -j$(nproc)

echo ""
echo "=============================================="
echo "  Accel-Sim build complete!"
echo "  Binary: $ACCELSIM_DIR/gpu-simulator/bin/release/accel-sim.out"
echo "=============================================="
echo ""

# Quick sanity check
ls -lh $ACCELSIM_DIR/gpu-simulator/bin/release/accel-sim.out
