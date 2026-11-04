#!/usr/bin/env bash
# Installs Centrix from this unpacked package: the package's folder is copied to <prefix>/lib/centrix-<version> and
# <prefix>/bin/centrix links to it, replacing an earlier version's link. Needs no .NET (the package carries its own).
#
# Usage: ./install.sh [--prefix <dir>]   (default ~/.local; /usr/local or /opt need sudo)
#        ./install.sh --uninstall [--prefix <dir>]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
prefix="$HOME/.local"
uninstall=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) prefix="$2"; shift 2 ;;
        --uninstall) uninstall=1; shift ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        *) echo "Unknown option $1" >&2; exit 2 ;;
    esac
done

name="$(basename "$here")"          # centrix-<version>-linux-x64
target="$prefix/lib/${name%-linux-x64}"
link="$prefix/bin/centrix"

if [[ $uninstall == 1 ]]; then
    [[ -L "$link" ]] && rm "$link"
    rm -rf "$target"
    echo "Removed $target and $link"
    exit 0
fi

[[ -x "$here/centrix" ]] || { echo "No centrix executable beside $0; run this from the unpacked package" >&2; exit 1; }
mkdir -p "$prefix/lib" "$prefix/bin"
rm -rf "$target"
cp -R "$here" "$target"
ln -sfn "$target/centrix" "$link"
echo "Installed Centrix in $target"
echo "  $link -> $target/centrix"
case ":$PATH:" in
    *":$prefix/bin:"*) ;;
    *) echo "  Add $prefix/bin to your PATH to run centrix from any directory." ;;
esac
