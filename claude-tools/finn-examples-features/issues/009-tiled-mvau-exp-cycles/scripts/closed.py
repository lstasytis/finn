import math
def in_stall(SF, TH):
    fm = SF*TH; buf = 1 << math.ceil(math.log2(fm+2))
    return max(0, (TH-1)*(SF-1)+3-(buf-fm))
def out_stall(SF, NF, TH):
    if NF < 2: return 0
    fm = SF*TH; buf = 1 << math.ceil(math.log2((NF-1)*TH+3)); free0 = buf-NF*TH
    R = NF*TH-(NF-1); wb = max(0, free0+TH-4); t = fm*(wb//TH+1)+wb%TH
    return max(0, R-t, -free0)
def closed(SF, NF, TH):
    return max(in_stall(SF, TH), out_stall(SF, NF, TH))
