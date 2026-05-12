#!/usr/bin/env bash
set -euo pipefail
 
CC=${CC:-gcc}
CFLAGS="-O2 -Wall -Wextra -std=c99"
LIBS="-lm"
OUTDIR="results"
 
mkdir -p "$OUTDIR"
 
targets=(
    "euler"
    "heun"
)
 
for name in "${targets[@]}"; do
    echo "=== $name ==="
    $CC $CFLAGS "${name}.c" -o "${OUTDIR}/${name}" $LIBS
    (cd "$OUTDIR" && "./${name}")
    echo ""
done
 
echo "All done. Output files in ./${OUTDIR}/"

