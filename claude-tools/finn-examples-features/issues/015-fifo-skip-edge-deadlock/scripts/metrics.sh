#!/bin/bash
# Issue 015: one line per feature build of a build dir: est fps, rtlsim fps, ratio, verified, FIFO kbit
# usage: metrics.sh <FINN_BUILD_DIR>
for b in $(ls -dtr $1/*/ | grep -v "est_\|code_gen\|tav_\|logs\|vivado_\|rtlsim_\|lock"); do
  f=$b/report/feature_metrics.json; [ -f $f ] || { echo "$(basename $b) (no metrics) $(ls $b/verification_output 2>/dev/null)"; continue; }
  python3 -c "import json;d=json.load(open('$f'));print('$(basename $b)', round(d['est_fps'],1), round(d['rtlsim_fps'] or 0,1), round(d['rtlsim_vs_est'] or 0,3), d['verified'], round(d['fifo_kbit'],1))"
done
