#!/usr/bin/env bash
# Fetches the ProteoWizard commit pinned in pwiz.lock into ./pwiz: shallow (that commit only),
# blobless and sparse (scripts/pwiz-sparse-paths.txt), so only the library sources Centrix
# builds from are downloaded. Rerun after changing pwiz.lock; an existing ./pwiz is moved to it.
#
# Usage: scripts/get-pwiz.sh [target-dir]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-$root/pwiz}"
repository="$(sed -n 's/^repository=//p' "$root/pwiz.lock")"
commit="$(sed -n 's/^commit=//p' "$root/pwiz.lock")"
if [[ -z "$repository" || -z "$commit" ]]; then
    echo "pwiz.lock must name a repository and a commit" >&2
    exit 1
fi

if [[ ! -d "$target/.git" ]]; then
    mkdir -p "$target"
    git -C "$target" init --quiet
    git -C "$target" remote add origin "$repository"
fi
git -C "$target" config core.longpaths true
git -C "$target" sparse-checkout init --no-cone
grep -v '^#' "$root/scripts/pwiz-sparse-paths.txt" | grep -v '^[[:space:]]*$' > "$target/.git/info/sparse-checkout"
if [[ "$(git -C "$target" rev-parse --verify --quiet HEAD || true)" == "$commit" ]]; then
    # The paths may have changed.
    git -C "$target" sparse-checkout reapply
else
    echo "Fetching pwiz $commit"
    git -C "$target" fetch --quiet --depth 1 --filter=blob:none origin "$commit"
    git -C "$target" -c advice.detachedHead=false checkout --quiet --force "$commit"
fi
echo "pwiz is at $commit"
