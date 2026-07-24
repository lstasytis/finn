#!/bin/bash
# synth_one.sh INW OUTW  -> "INW->OUTW LUT=.. FF=.. clk=..ns II=.. <OK|SLOW|II!>"
# Synthesizes StreamingDataWidthConverterGeneralized_Batch via vitis_hls csynth
# at 300MHz (3.333ns). Flags timing (>3.333ns) and II!=1.
set -e
INW=$1; OUTW=$2
export FINN_ROOT=/home/lstasytis/backup/finn
export INW OUTW
D=$(mktemp -d /tmp/dwc_synth.XXXXXX)
cp ref_gen/top.cpp ref_gen/run.tcl "$D"/
( cd "$D" && vitis_hls -f run.tcl >synth.log 2>&1 ) || { echo "$INW->$OUTW SYNTH_FAIL"; tail -5 "$D/synth.log"; rm -rf "$D"; exit 1; }
RPT="$D/proj/sol1/syn/report/top_csynth.rpt"
if [ ! -f "$RPT" ]; then echo "$INW->$OUTW NO_REPORT"; rm -rf "$D"; exit 1; fi
II=$(grep -oE "Final II = [0-9]+" "$D/synth.log" | tail -1 | grep -oE "[0-9]+$")
python3 - "$RPT" "$INW" "$OUTW" "${II:-?}" <<'PY'
import re,sys
rpt,inw,outw,ii=sys.argv[1:5]
txt=open(rpt).read()
lut=ff="?"; clk="?"
lines=txt.splitlines()
for i,l in enumerate(lines):
    if re.search(r'\|\s*Name\s*\|.*LUT',l):
        hdr=[c.strip() for c in l.strip().strip('|').split('|')]
        for j in range(i,min(i+40,len(lines))):
            if re.match(r'\s*\|\s*Total\s*\|',lines[j]):
                d=dict(zip(hdr,[c.strip() for c in lines[j].strip().strip('|').split('|')]))
                lut=d.get('LUT','?'); ff=d.get('FF','?'); break
        break
# estimated clock (ns)
m=re.search(r'\|ap_clk\s*\|\s*[\d.]+ ns\|\s*([\d.]+) ns',txt)
if m: clk=m.group(1)
flag="OK"
try:
    if float(clk)>3.333: flag="SLOW"
except: pass
if ii not in ("1","?"): flag=(flag+"+II!" if flag!="OK" else "II!")
print(f"{inw}->{outw} LUT={lut} FF={ff} clk={clk}ns II={ii} {flag}")
PY
rm -rf "$D"
