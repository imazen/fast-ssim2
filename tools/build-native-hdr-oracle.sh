#!/usr/bin/env bash
# Development-only C++ oracle; run this through the usual build resource wrapper.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo 'usage: build-native-hdr-oracle.sh SOURCE BUILD OUTPUT' >&2
    exit 2
fi
oracle_source=$1
oracle_build=$2
oracle_output=$3
oracle_revision=81feacf6a5b0bbd7ec11b4a881a4615033ab0b10
if [[ $(git -C "$oracle_source" rev-parse HEAD) != "$oracle_revision" ]]; then
    echo "oracle source must be Cloudinary SSIMULACRA2 $oracle_revision" >&2
    exit 2
fi
# Build configuration may differ; metric and bundled-library sources must not.
git -C "$oracle_source" diff --exit-code "$oracle_revision" -- ssimulacra2.cc ssimulacra2.h lib
mkdir -p "$(dirname "$oracle_output")"
oracle_new=$(mktemp -p "$(dirname "$oracle_output")" .native-hdr-oracle.XXXXXX)
read -r -a oracle_flags <<< "$(pkg-config --cflags --libs libhwy lcms2)"
"${CXX:-c++}" -std=c++11 -O2 -ffp-contract=off \
    -I"$oracle_source" -I"$oracle_source/lib/include" -I"$oracle_build/lib/include" \
    tools/native-hdr-oracle.cc "$oracle_source/ssimulacra2.cc" \
    "$oracle_build/lib/libjxl.a" "$oracle_build/lib/libjxl_threads.a" \
    "${oracle_flags[@]}" -lpthread -o "$oracle_new"
chmod +x "$oracle_new"
if [[ -e "$oracle_output" ]]; then
    oracle_backup="$oracle_output.bak.$(date -u +%Y%m%dT%H%M%SZ).$$"
    [[ ! -e "$oracle_backup" ]]
    mv -- "$oracle_output" "$oracle_backup"
fi
mv -- "$oracle_new" "$oracle_output"
sha256sum "$oracle_output"
