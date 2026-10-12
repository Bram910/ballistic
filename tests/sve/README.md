This differential tester checks Ballistic's SVE, SVE2 and SME instructions. The fuzzer cannot:
Unicorn's QEMU 5.0 aborts with SVE enabled and predates SME. Instead, random encodings of every SVE decoder table entry run
through Ballistic and through a small AArch64 reference program under `qemu-aarch64` with 128-bit
vectors, the vector length Ballistic implements. The two final states are compared:
- the general purpose registers, SP and NZCV;
- the FPSR flags;
- the Z and P registers and FFR;
- a 16 KiB data area;
- whether the instruction faulted;
- for SME, SVCR, TPIDR2_EL0, and ZA and ZT0 while ZA is on.

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

Set `$BAL_SVE_SPACE` to `sme` to test the SME encoding space instead. That also covers SMSTART,
SMSTOP and the SVCR and TPIDR2_EL0 system registers, which are reported on a `#` line. The tests then
start in random streaming mode and ZA states, with random ZA and ZT0 contents. That needs a qemu
with SME2, 10.1 or later, run with streaming vector length 128 and FEAT_SME_FA64, which
Ballistic implements. Debian's `qemu-user` package has a static `qemu-aarch64` that can be
extracted without installing it:

```
BAL_SVE_SPACE=sme BAL_SVE_QEMU=/path/to/qemu-aarch64 tests/sve/run_sve_diff.sh > sme.csv
```

Each entry gets one CSV line, `index,mnemonic,mask,expected,status,translated,matched,samples`, the
format of the status tracker's coverage tool. The status is one of:
- `verified`: every compared sample matched;
- `wrong`;
- `todo`: Ballistic rejects encodings qemu runs;
- `undefined`: both reject them; these are mostly SVE2.1 and SME2 instructions newer than the
  reference's qemu;
- `unchecked`: qemu rejects every sample Ballistic runs, an extension qemu lacks (FEAT_LSE128
  and FEAT_THE in qemu 11.1).

With `$BAL_SVE_SPACE` set to `all`, it tests the decoder entries listed in `MNEMONICS` from any
encoding space. This checks the loads and stores newer than Unicorn against qemu (10.1 or later):
- The FEAT_MOPS CPY* and SET* instructions run as whole prologue, main and epilogue sequences,
  with distinct registers, small sizes and addresses inside the data area. NZCV is not compared:
  the prologue reports in it which of the algorithms the architecture allows is used, and
  Ballistic's and qemu's differ.
- qemu lacks FEAT_LRCPC3, so for LDAPUR and STLUR (SIMD&FP), LDAP1, STL1, the writeback LDAPR and
  STLR, LDIAPP and STILP, the reference runs the equivalent LDUR, STUR, LD1, ST1, LDR, STR, LDP
  or STP. Those have no alignment checks, so faults only Ballistic raises are not compared.
- qemu also lacks FEAT_LSE128 and FEAT_THE, so LDCLRP, LDSETP, SWPP and the RCW* instructions
  cannot be checked this way.

# Known qemu differences

The reference is qemu 6.2, as packaged by Ubuntu 22.04. A few of its behaviours differ from
Ballistic's, where the architecture allows either or qemu is wrong. `is_known_qemu_difference()` in
`sve_diff.c` leaves those samples out:
- non-fault and first-fault loads that start in the upper half of the vector (a qemu predicate
  indexing bug);
- first-fault loads that cross a page after their first element;
- WHILERW with operands less than one element apart;
- gathers from addresses qemu-user maps for itself.
