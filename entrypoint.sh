#!/bin/bash
# FlashVSR-Pro container entrypoint script

# Source Conda
source /opt/conda/etc/profile.d/conda.sh
conda activate flashvsr

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/opt/conda/envs/flashvsr/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Change to project directory
cd /workspace/FlashVSR-Pro

echo "[INFO] Conda environment 'flashvsr' activated."
echo "[INFO] Python path: $(which python)"

# If command given, run it; otherwise start interactive bash
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec bash
fi