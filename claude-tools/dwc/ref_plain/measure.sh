#!/bin/bash
# Synthesize the plain multiple-only DWC at several widths; print LUT/FF.
cd "$(dirname "$0")"
printf "%-12s %6s %6s\n" "in->out" "LUT" "FF"
for pair in "32 16 4" "64 32 4" "128 64 4" "256 128 4" "512 256 4" "1024 512 4"; do
  set -- $pair
  export INW=$1 OUTW=$2 NIW=$3
  rm -rf proj
  vitis_hls -f run.tcl >/dev/null 2>&1
  xml="proj/sol1/syn/report/top_csynth.xml"
  if [ -f "$xml" ]; then
    lut=$(grep -oP '(?<=<LUT>)[0-9]+' "$xml" | head -1)
    ff=$(grep -oP '(?<=<FF>)[0-9]+' "$xml" | head -1)
    printf "%5s->%-5s %6s %6s\n" "$INW" "$OUTW" "$lut" "$ff"
  else
    printf "%5s->%-5s   FAIL\n" "$INW" "$OUTW"
  fi
done
