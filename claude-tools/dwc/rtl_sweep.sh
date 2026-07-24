#!/bin/bash
export PATH=/mnt/labstore/Xilinx/Vivado/2023.1/bin:$PATH
O=/tmp/$1; rm -rf $O; mkdir -p $O; shift
while read -r ib ob ni no; do
  [ -z "$ib" ] && continue
  ( D=$(mktemp -d); cd $D && vivado -mode batch -source /home/lstasytis/backup/finn/claude-tools/dwc/rtlgen_timing.tcl -tclargs $ib $ob $ni $no >v.log 2>&1; grep "RTLGEN" v.log|grep -v "puts " > $O/${ib}_${ob}.txt; rm -rf $D ) &
  while [ $(jobs -r|wc -l) -ge 8 ]; do wait -n; done
done
wait; cat $O/*.txt; echo RTLSWEEP_DONE
