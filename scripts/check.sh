#!/usr/bin/env bash
# Configure, build, and test Flint on the Linux server, inside the flint
# environment. Extra arguments go to pytest. Run from anywhere:
#   ~/.local/bin/micromamba run -n flint bash scripts/check.sh [pytest args]
set -euo pipefail

cd "$(dirname "$0")/.."

# The server's /tmp is a small RAM disk; keep nvcc's intermediate files on disk.
export TMPDIR="$HOME/.cache/flint-tmp"
mkdir -p "$TMPDIR"

cmake --preset release
cmake --build --preset release
python -m pytest "$@"
