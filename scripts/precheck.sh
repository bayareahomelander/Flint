#!/usr/bin/env bash
# Builds and tests the uncommitted working tree on the Linux server before a
# commit, without touching the server's checkout. Run from Git Bash on the
# laptop once the laptop's tests pass:  bash scripts/precheck.sh
#
# It snapshots the working tree (tracked and untracked files, minus ignored
# ones) as a temporary commit, pushes it to refs/precheck/latest on the
# `server` remote, and runs check.sh and sanitize.py on a copy of it in
# ~/.cache/flint-tmp. The host, path, and SSH key come from this repository's
# config, so nothing here names the server.
set -euo pipefail

cd "$(dirname "$0")/.."

# Snapshot through a separate index, so the real index and branch stay as they are.
index="$(git rev-parse --absolute-git-dir)/precheck-index"
rm -f "$index"
trap 'rm -f "$index"' EXIT
GIT_INDEX_FILE="$index" git add -A
tree="$(GIT_INDEX_FILE="$index" git write-tree)"
snapshot="$(git commit-tree "$tree" -p HEAD -m "precheck snapshot")"

# The remote is user@host:path; its ref outside refs/heads leaves the
# server's checkout alone, and + lets each snapshot replace the last.
url="$(git remote get-url server)"
host="${url%%:*}"
repo="${url#*:}"
git push --quiet server "+$snapshot:refs/precheck/latest"

# Run SSH the way Git does, so core.sshCommand's quoting works the same.
ssh_command="$(git config core.sshCommand || echo ssh)"
echo "precheck: testing snapshot ${snapshot:0:7} of $(git rev-parse --short HEAD) plus local changes"
sh -c "$ssh_command -o BatchMode=yes \"\$@\"" ssh "$host" bash -s -- "$repo" <<'REMOTE'
set -uo pipefail
repo="$1"
micromamba="${MICROMAMBA:-$HOME/.local/bin/micromamba}"
dir="$HOME/.cache/flint-tmp/precheck"
cleanup_ref() { git -C "$repo" update-ref -d refs/precheck/latest; }

# The GPU is shared: refuse if anyone else is using it, by process or by memory,
# or if nvidia-smi can't tell.
if ! apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ||
    ! used_mib="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" ||
    ! [[ "$used_mib" =~ ^[0-9]+$ ]]; then
    echo "precheck: nvidia-smi can't report the GPU's processes and memory; try later" >&2
    cleanup_ref
    exit 2
fi
processes="$(grep -c . <<<"$apps")"
if [ "$processes" -ne 0 ] || [ "$used_mib" -gt 512 ]; then
    echo "precheck: the GPU is busy ($processes processes, $used_mib MiB used); try later" >&2
    cleanup_ref
    exit 2
fi

rm -rf "$dir"
mkdir -p "$dir"
git -C "$repo" archive refs/precheck/latest | tar -x -C "$dir"
cleanup_ref

status=0
"$micromamba" run -n flint bash "$dir/scripts/check.sh" -q || status=1
if [ "$status" -eq 0 ]; then
    "$micromamba" run -n flint python "$dir/scripts/sanitize.py" || status=1
fi
if [ "$status" -eq 0 ]; then
    rm -rf "$dir"
    echo "precheck: passed"
else
    echo "precheck: FAILED; the server copy is kept in $dir" >&2
fi
exit "$status"
REMOTE
