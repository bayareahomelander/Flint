#!/usr/bin/env bash
# Create or update the flint micromamba environment on the Linux GPU server,
# without root. Run from anywhere: bash scripts/setup_server.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
micromamba="${MICROMAMBA:-$HOME/.local/bin/micromamba}"

# The server's /tmp is a small RAM disk; keep package downloads on disk.
export TMPDIR="$HOME/.cache/flint-tmp"
mkdir -p "$TMPDIR"

if "$micromamba" env list | awk '{print $1}' | grep -qx flint; then
    "$micromamba" install -y -n flint --strict-channel-priority -f "$root/environment.yml"
else
    "$micromamba" create -y -n flint --strict-channel-priority -f "$root/environment.yml"
fi
"$micromamba" run -n flint python -m pip install --no-input -r "$root/requirements.txt"

"$micromamba" run -n flint bash -c '
    nvcc --version | tail -1
    g++ --version | head -1
    cmake --version | head -1
    echo "ninja $(ninja --version)"
    nsys --version
    python -c "import torch; print(\"torch\", torch.__version__, \"CUDA\", torch.version.cuda, torch.cuda.get_device_name())"
'
