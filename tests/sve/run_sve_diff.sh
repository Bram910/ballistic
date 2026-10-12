#!/usr/bin/env bash
# Compares Ballistic's SVE against qemu-aarch64 with 128-bit vectors.
#
# Usage, from the repository root after a debug build with tests on Linux x86-64:
#
#     tests/sve/run_sve_diff.sh [SAMPLES] [MNEMONICS] [REPORTS] > sve.csv
#
# Set $BAL_SVE_SEED to vary the random samples.
#
# Needs clang and ld.lld with the AArch64 target (to build the reference) and qemu-aarch64-static
# (or $BAL_SVE_QEMU) to run it. The arguments and output are those of bal_sve_diff.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CC="${CC_AARCH64:-$(command -v clang || command -v clang-15)}"
DIFF="${BAL_SVE_DIFF:-$ROOT/build/debug/bin/linux/bal_sve_diff}"
REFERENCE="$(mktemp /tmp/bal_sve_reference.XXXXXX)"
trap 'rm -f "$REFERENCE"' EXIT

# Freestanding and static, linked away from the data area the tests map at 0x200000.
"$CC" -target aarch64-linux-gnu -march=armv8.6-a+sve2 \
    -O1 -ffreestanding -fno-builtin -fno-stack-protector -nostdlib -static -fno-pie \
    -fuse-ld=lld -Wl,--image-base=0x10000000 -I"$ROOT/tests/sve" \
    -o "$REFERENCE" "$ROOT/tests/sve/sve_reference.c"

# The JIT maps its code buffers near the program; disable randomisation like the other tests.
ASAN_OPTIONS=detect_leaks=0 setarch "$(uname -m)" -R "$DIFF" "$REFERENCE" "$@"
