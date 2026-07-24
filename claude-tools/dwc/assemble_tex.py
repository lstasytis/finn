#!/usr/bin/env python3
import json, os, glob
BASE="/home/lstasytis/backup/finn/claude-tools/dwc"
# which points belong to which plot
PLOT1={(25,10),(105,10),(305,10),(505,10),(755,10),(1005,10)}
PLOT2={(10,25),(10,105),(10,305),(10,505),(10,755),(10,1005)}
PLOT3={(44,48),(88,96),(176,192),(352,384),(420,448),(480,512),(720,768)}
def plot_of(i,o):
    if (i,o) in PLOT1: return "plot1"
    if (i,o) in PLOT2: return "plot2"
    if (i,o) in PLOT3: return "plot3"
    return None
lut=json.load(open(f"{BASE}/plotdata.json"))
ff=json.load(open(f"{BASE}/plotdata_ff.json"))
# clear any prior newrtl/newhls
for d in (lut,ff):
    for pk in d:
        d[pk].pop("newrtl",None); d[pk].pop("newhls",None)
        d[pk].setdefault("newrtl",[]); d[pk].setdefault("newhls",[])
n=0
for f in glob.glob("/tmp/texmeas/*_*.txt"):
    txt=open(f).read().split()
    if len(txt)<9 or txt[0]!="PT": continue
    i,o=int(txt[1]),int(txt[2])
    # PT i o newrtl RLUT RFF newhls HLUT HFF
    rl,rf,hl,hf=txt[4],txt[5],txt[7],txt[8]
    pk=plot_of(i,o)
    if not pk: continue
    def num(x):
        try: return int(x)
        except: return None
    if num(rl) is not None: lut[pk]["newrtl"].append([i,o,num(rl)])
    if num(rf) is not None: ff[pk]["newrtl"].append([i,o,num(rf)])
    if num(hl) is not None: lut[pk]["newhls"].append([i,o,num(hl)])
    if num(hf) is not None: ff[pk]["newhls"].append([i,o,num(hf)])
    n+=1
json.dump(lut,open(f"{BASE}/plotdata.json","w"),indent=1)
json.dump(ff,open(f"{BASE}/plotdata_ff.json","w"),indent=1)
print(f"merged {n} points")
for pk in ("plot1","plot2","plot3"):
    print(pk,"newrtl",sorted(lut[pk]["newrtl"]),"\n   newhls",sorted(lut[pk]["newhls"]))
