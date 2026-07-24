#!/usr/bin/env bash
# Functional equivalence check for vpc.sv: re-packs an N-element vector from PI
# to PO lanes/beat and verifies the flat element sequence is preserved (identity),
# across multiple/non-multiple/coprime/1:1 ratios, up/downscale, padded lengths,
# with and without AXIS backpressure. Uses Vivado xsim.
set -e
source /mnt/labstore/Xilinx/Vivado/2023.1/settings64.sh 2>/dev/null || true
HERE="$(cd "$(dirname "$0")" && pwd)"
VPC="${VPC_SRC:-$HERE/vpc.sv}"   # self-contained local copy by default
TB="$HERE/vpc_tb.sv"
mkdir -p "$HERE/simwork" && cd "$HERE/simwork"
xvlog -sv "$VPC" "$TB" > compile.log 2>&1

run_cfg() { # name W N PI PO STALL
  local name="$1" W="$2" N="$3" PI="$4" PO="$5" STALL="$6"
  xelab work.vpc_tb -timescale 1ns/1ps -generic_top "W=$W" -generic_top "N=$N" \
    -generic_top "PI=$PI" -generic_top "PO=$PO" -generic_top "STALL=$STALL" \
    -s "snap_$name" --O0 > "elab_$name.log" 2>&1
  xsim "snap_$name" -R > "sim_$name.log" 2>&1
  printf "%-28s %s\n" "$name (N$N ${PI}->${PO} st$STALL)" \
    "$(grep -E 'VPC_TB_PASS|VPC_TB_FAIL' sim_$name.log | head -1)"
  grep MISMATCH "sim_$name.log" | head -1 | sed 's/^/    /' || true
}

for ST in 0 1; do
  echo "===== backpressure=$ST ====="
  run_cfg mult_down   8  16   4  2 $ST
  run_cfg mult_up     8  16   2  4 $ST
  run_cfg nonmult_dn  8  20   5  2 $ST
  run_cfg nonmult_up  8  20   2  5 $ST
  run_cfg coprime     4  42   3  7 $ST
  run_cfg one_to_one  8   9   3  3 $ST
  run_cfg pad_eq      8  10   4  4 $ST
  run_cfg pad_nonmult 8   7   3  2 $ST
  run_cfg wide_nm     4 100  25 10 $ST
  run_cfg hi_ratio    8 480 480 15 $ST
done
