"""Cycle model of mvu_tiled_axi's activation replay (input_gen DIMS {NF,SF,TH}), the
weight buffer (assumed always ready), the compute pipeline, the A/B output slice and the
output reorder (input_gen DIMS {TH,NF}). Gives steady-state cycles per group of TH vectors.
tiledsim.py SF NF TH [groups]"""
import math
import sys


class InputGen:
    """finn-rtllib/mvu_tiled/input_gen.sv, cycle-level."""

    def __init__(self, FM, DIMS, COEFS):
        D = len(DIMS)
        self.D, self.DIMS, self.COEFS = D, DIMS, COEFS
        W = [FM] + list(COEFS)
        R = [True] + [False] * D
        for i in range(1, D + 1):
            R[i] = R[i - 1] and COEFS[i - 1] > 0 and COEFS[i - 1] * DIMS[i - 1] <= W[i - 1]
        rp = [0] * (D + 1)
        rw = 0
        for i in range(D, -1, -1):
            if i < D:
                rw = (DIMS[i] - 1) * COEFS[i] + rw
            rp[i] = W[i] - rw
        fp = [0] * (D + 1)
        fw = 0
        for i in range(D, -1, -1):
            if i < D:
                fw = (DIMS[i] - 1) * COEFS[i] + fw if R[i + 1] else 0
            fp[i] = fw - W[i] if R[i] else 0
        m = max([-x for x in rp] + [0])
        rw = fw = 0
        for i in range(D - 1, -1, -1):
            rw = (DIMS[i] - 1) * COEFS[i] + rw
            fw = (DIMS[i] - 1) * COEFS[i] + fw if R[i + 1] else 0
            m = max(m, rw - fw)
        self.R, self.rp, self.fp = R, rp, fp
        self.BUF = 1 << math.ceil(math.log2(m + 1 + 2))
        self.Wp = self.WpZ = self.Rp = 0
        self.Cap = -self.BUF + 1
        self.cnt = [d - 2 for d in DIMS]
        self.OVld = 0

    def comb(self, ivld, ordy):
        D = self.D
        done = [c < 0 for c in self.cnt] + [True]
        term = [False] * (D + 1)
        term[D] = True
        for i in range(D - 1, -1, -1):
            term[i] = term[i + 1] and done[i]
        self.irdy = 1 if self.Cap < 0 else 0
        self.has_data = self.Rp < self.WpZ
        self.advance = self.has_data and ((not self.OVld) or ordy)
        rp_inc = fp_inc = 0
        for i in range(D, -1, -1):
            if term[i]:
                rp_inc = self.rp[i]
                if self.R[i]:
                    fp_inc = self.fp[i]
        self._n = (ivld, ordy, done, term, rp_inc, fp_inc)
        self.ostep = (not self.OVld) or ordy  # output register loads
        return self.irdy, self.OVld

    def seq(self):
        ivld, ordy, done, term, rp_inc, fp_inc = self._n
        istep = self.irdy and ivld
        self.WpZ = self.Wp
        if istep:
            self.Wp += 1
        self.Cap += (fp_inc if self.advance else 0) + (1 if istep else 0)
        if self.advance:
            self.Rp += rp_inc
            for i in range(self.D):
                if term[i + 1] and self.DIMS[i] > 1:
                    self.cnt[i] += (self.DIMS[i] - 1) if done[i] else -1
        emitted = 0
        if (not self.OVld) or ordy:
            emitted = 1 if self.has_data else 0
            self.OVld = emitted
        return istep, emitted


def sim(SF, NF, TH, groups=8):
    fm = SF * TH
    act = InputGen(fm, [NF, SF, TH], [0, 1, SF])
    out = InputGen(NF * TH, [TH, NF], [1, TH])
    A_rdy, B_vld = 1, 0
    LAT = 6  # compute pipeline depth (valid/last shift), value irrelevant for throughput
    pipe = [0] * LAT  # 1 where a beat carrying an output travels
    beat = 0
    words_out = 0
    cyc = 0
    ends = []
    total = groups * NF * TH
    while words_out < total and cyc < 10**7:
        cyc += 1
        en = A_rdy
        # activation replay output -> compute (weights always valid)
        a_irdy, avld = act.comb(1, en)  # ardy = en && wvld
        istb = avld and en
        ovld = pipe[-1] if en else 0
        o_irdy, o_ovld = out.comb(B_vld, 1)
        m_int_rdy = o_irdy
        b_load = (not B_vld) or m_int_rdy
        # sequential
        if en:
            last = (beat % fm) // TH == SF - 1  # sf == SF-1 beats carry outputs
            pipe = [1 if (istb and last) else 0] + pipe[:-1]
            if istb:
                beat += 1
        nA_rdy = (A_rdy and not ovld) or b_load
        if b_load:
            B_vld = ovld or (not A_rdy)
        A_rdy = nA_rdy
        act.seq()
        _, em = out.seq()
        words_out += em
        if em and words_out % (NF * TH) == 0:
            ends.append(cyc)
    per_group = [ends[i + 1] - ends[i] for i in range(len(ends) - 1)]
    return act.BUF, out.BUF, per_group


if __name__ == "__main__":
    SF, NF, TH = map(int, sys.argv[1:4])
    g = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    bi, bo, pg = sim(SF, NF, TH, g)
    print("BUF_in", bi, "BUF_out", bo, "ideal/group", NF * SF * TH, "per group", pg)
