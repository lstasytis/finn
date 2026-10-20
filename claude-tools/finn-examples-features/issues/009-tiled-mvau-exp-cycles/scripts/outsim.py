"""Cycle model of the tiled MVU output path (A/B register slice + reorder input_gen),
to understand the per-group output stall. outsim.py SF NF TH [groups]"""
import math
import sys


def sim(SF, NF, TH, groups=8, ordy_always=True):
    fm = SF * TH
    # reorder input_gen parameters: DIMS {TH, NF}, COEFS {1, TH}, FM_SIZE NF*TH
    max_occ = (NF - 1) * TH
    addr_bits = math.ceil(math.log2(max_occ + 1 + 2))
    BUF = 1 << addr_bits
    # pipeline: beat counter per group; a beat produces an output word when sf == SF-1
    beat = 0  # within frame of `groups`
    total_beats = groups * NF * fm
    # A/B slice state
    A_rdy, A_dat = 1, None
    B_vld, B_dat = 0, None
    # input_gen state
    Wp, WpZ, Rp = 0, 0, 0
    Cap = -BUF + 1
    cnt = [TH - 2, NF - 2]  # Cnt per level, done when negative
    dims = [TH, NF]
    OVld = 0
    out_words = 0
    cyc = 0
    group_end_cycles = []
    beats_done = 0
    last_word_cycle = None
    while out_words < groups * NF * TH and cyc < 10**7:
        cyc += 1
        # --- combinational ---
        en = A_rdy
        # pipeline output this cycle (word leaving the compute core into A/B)
        ovld = 0
        if en and beats_done < total_beats:
            b = beats_done % fm
            ovld = 1 if (b // TH) == SF - 1 else 0
        irdy = 1 if Cap < 0 else 0
        m_int_rdy = irdy
        b_load = (not B_vld) or m_int_rdy
        # input_gen
        done = [c < 0 for c in cnt] + [True]
        term = [False, False, True]
        term[1] = term[2] and done[1]
        term[0] = term[1] and done[0]
        has_data = Rp < WpZ
        ordy = 1 if ordy_always else 1
        advance = has_data and ((not OVld) or ordy)
        # rp/fp increments
        # TERMINAL_RP_INC: [1, 1-(NF-1)*TH, TH]; TERMINAL_FP_INC: [(TH-1)-NF*TH, -1, 0]
        rp_inc, fp_inc = TH, 0
        if term[1]:
            rp_inc, fp_inc = 1 - (NF - 1) * TH, -1
        if term[0]:
            rp_inc, fp_inc = 1, (TH - 1) - NF * TH
        istep = irdy and B_vld
        # --- sequential ---
        if en and beats_done < total_beats:
            beats_done += 1
        # A/B
        nA_rdy = (A_rdy and not ovld) or b_load
        if b_load:
            B_vld = ovld or (not A_rdy)
        A_rdy = nA_rdy
        # input_gen
        WpZ = Wp
        if istep:
            Wp += 1
        Cap = Cap + (fp_inc if advance else 0) + (1 if istep else 0)
        if advance:
            Rp += rp_inc
            # counters
            step1 = True  # level 1 (inner) steps every advance (term[2] = 1)
            if step1:
                if dims[1] > 1:
                    cnt[1] = cnt[1] + (dims[1] - 1 if done[1] else -1)
            if term[1]:
                if dims[0] > 1:
                    cnt[0] = cnt[0] + (dims[0] - 1 if done[0] else -1)
        # output register
        if (not OVld) or ordy:
            OVld = 1 if has_data else 0
            if has_data:
                out_words += 1
                if out_words % (NF * TH) == 0:
                    group_end_cycles.append(cyc)
    ends = group_end_cycles
    per_group = [ends[i + 1] - ends[i] for i in range(len(ends) - 1)]
    return BUF, per_group


if __name__ == "__main__":
    SF, NF, TH = map(int, sys.argv[1:4])
    g = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    BUF, pg = sim(SF, NF, TH, g)
    print("BUF_o", BUF, "ideal/group", NF * SF * TH, "per group", pg)
