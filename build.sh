#!/usr/bin/env bash
# Builds Centrix, and with --test tests it: fetches the pinned ProteoWizard tree (pwiz.lock) unless
# PWIZ_ROOT names one, then builds Centrix.sln.
#
# Usage: ./build.sh [--debug] [--test] [--i-agree-to-the-vendor-licenses]
#   --i-agree-to-the-vendor-licenses  build the vendor readers (Thermo .raw and the rest), agreeing to
#                                     the vendors' licenses (pwiz/pwiz/data/vendor_readers/*/EULA.*.txt);
#                                     without it centrix reads open formats only
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
configuration=Release
test=0
args=()
for arg in "$@"; do
    case "$arg" in
        --debug) configuration=Debug ;;
        --test) test=1 ;;
        --i-agree-to-the-vendor-licenses) args+=(-p:IAgreeToVendorLicenses=true) ;;
        *) echo "Unknown option $arg" >&2; exit 2 ;;
    esac
done
if [[ -z "${PWIZ_ROOT:-}" ]]; then
    "$root/scripts/get-pwiz.sh"
fi
dotnet build "$root/Centrix.sln" -c "$configuration" -p:Platform=x64 --nologo "${args[@]}"
if [[ $test == 1 ]]; then
    dotnet test "$root/Centrix.sln" -c "$configuration" -p:Platform=x64 --nologo --no-build "${args[@]}"
fi
echo "Built $root/src/Centrix/bin/x64/$configuration/net10.0/centrix"
