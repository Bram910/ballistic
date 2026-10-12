This differential tester checks Ballistic's SVE and SVE2 instructions. The fuzzer cannot: Unicorn's
QEMU 5.0 aborts with SVE enabled. Instead, random encodings of every SVE decoder table entry run
through Ballistic and through a small AArch64 reference program under `qemu-aarch64` with 128-bit
vectors, the vector length Ballistic implements. The two final states are compared:
- the general purpose registers, SP and NZCV;
- the FPSR flags;
- the Z and P registers and FFR;
- a 16 KiB data area;
- whether the instruction faulted.

# Usage

Linux x86-64 only. Building Ballistic in debug mode also builds `bal_sve_diff`. The script builds
the reference, which needs clang and lld with the AArch64 target, and runs it with
`qemu-aarch64-static` (from `qemu-user-static`; set `$BAL_SVE_QEMU` to use another):

```
tests/sve/run_sve_diff.sh [SAMPLES] [MNEMONICS] [REPORTS] > sve.csv
```

- `SAMPLES`: random encodings per decoder entry (default 24).
- `MNEMONICS`: limits the run to a comma separated list such as `LD1B,FADD`.
- `REPORTS`: the number of mismatches to describe, on lines starting with `#`.

The samples are the same on every run; set `$BAL_SVE_SEED` to draw others.

Each entry gets one CSV line, `index,mnemonic,mask,expected,status,translated,matched,samples`, the
format of the status tracker's coverage tool. The status is one of:
- `verified`: every compared sample matched;
- `wrong`;
- `todo`: Ballistic rejects encodings qemu runs;
- `undefined`: both reject them; these are mostly SVE2.1 and SME2 instructions newer than the
  reference's qemu.

# Known qemu differences

The reference is qemu 6.2, as packaged by Ubuntu 22.04. A few of its behaviours differ from
Ballistic's, where the architecture allows either or qemu is wrong. `is_known_qemu_difference()` in
`sve_diff.c` leaves those samples out:
- non-fault and first-fault loads that start in the upper half of the vector (a qemu predicate
  indexing bug);
- first-fault loads that cross a page after their first element;
- WHILERW with operands less than one element apart;
- gathers from addresses qemu-user maps for itself.
