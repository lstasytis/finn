// Fast standalone testbench for StreamingDataWidthConverterGeneralized_Batch.
// Compiles with g++ against the Vitis HLS headers (no synthesis) so kernel
// logic can be validated in seconds across a large width/padding matrix.
// ap_uint-native and bit-vector based => correct for arbitrary widths (>64b).
//
// build: g++ -std=c++14 -I$XILINX_HLS/include -I<finn-hlslib> tb_dwc.cpp -o tb_dwc
#define AP_INT_MAX_W 8191
#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>
#include <vector>
#include "streamtools.h"

using namespace std;

// deterministic pseudo-random bit
static inline int prbit(long i) {
  unsigned long x = (unsigned long)(i + 1) * 2654435761UL;
  x ^= x >> 13; x *= 0x9E3779B1UL; x ^= x >> 15;
  return x & 1;
}

template <int InWidth, int OutWidth, int NumIn, int NumOut>
bool run_case(int numReps) {
  hls::stream<ap_uint<InWidth>> in("in");
  hls::stream<ap_uint<OutWidth>> out("out");
  // build inputs and expected outputs as bit vectors
  vector<vector<ap_uint<OutWidth>>> allref;
  long gcnt = 0;
  for (int r = 0; r < numReps; r++) {
    vector<int> bits;
    for (int w = 0; w < NumIn; w++) {
      ap_uint<InWidth> a = 0;
      for (int b = 0; b < InWidth; b++) {
        int v = prbit(gcnt++);
        a[b] = v;
        bits.push_back(v);
      }
      in.write(a);
    }
    vector<ap_uint<OutWidth>> ref(NumOut, 0);
    for (int w = 0; w < NumOut; w++)
      for (int b = 0; b < OutWidth; b++) {
        long idx = (long)w * OutWidth + b;
        ref[w][b] = (idx < (long)bits.size()) ? bits[idx] : 0;  // pad with 0
      }
    allref.push_back(ref);
  }
  StreamingDataWidthConverterGeneralized_Batch<InWidth, OutWidth, NumIn, NumOut>(
      in, out, numReps);
  bool ok = true; int errs = 0;
  for (int r = 0; r < numReps; r++)
    for (int w = 0; w < NumOut; w++) {
      if (out.empty()) { cerr << "  underflow r=" << r << " w=" << w << "\n"; return false; }
      ap_uint<OutWidth> got = out.read();
      if (got != allref[r][w]) {
        if (errs < 3)
          cerr << "  MISMATCH r=" << r << " w=" << w << " got=" << got.to_string(16)
               << " exp=" << allref[r][w].to_string(16) << "\n";
        ok = false; errs++;
      }
    }
  if (!out.empty()) { cerr << "  overflow: extra output words\n"; ok = false; }
  return ok;
}

#define CASE(IW, OW, NI, NO, REPS)                                              \
  do {                                                                         \
    bool r = run_case<IW, OW, NI, NO>(REPS);                                   \
    cout << (r ? "PASS " : "FAIL ") << "IW=" << IW << " OW=" << OW             \
         << " NI=" << NI << " NO=" << NO << " reps=" << REPS << "\n";          \
    if (!r) fails++;                                                           \
  } while (0)

int main() {
  int fails = 0;
  // multiple-case, no padding
  CASE(8, 4, 2, 4, 3);   CASE(4, 8, 4, 2, 3);
  CASE(16, 8, 3, 6, 2);  CASE(8, 16, 6, 3, 2);   CASE(32, 8, 2, 8, 1);
  // non-multiple, no padding (equal total bits)
  CASE(6, 4, 2, 3, 2);   CASE(4, 6, 3, 2, 2);
  CASE(10, 6, 3, 5, 2);  CASE(6, 10, 5, 3, 2);
  // padding (out bits > in bits)
  CASE(4, 13, 6, 2, 2);  CASE(8, 8, 2, 4, 2);    CASE(10, 6, 1, 2, 2);
  // cropping (out bits < in bits)
  CASE(10, 4, 2, 2, 2);  CASE(16, 8, 3, 4, 1);   CASE(8, 4, 3, 4, 1);
  // coprime widths
  CASE(7, 5, 5, 7, 2);   CASE(5, 7, 7, 5, 2);
  CASE(9, 7, 7, 9, 1);   CASE(13, 11, 11, 13, 1);
  // single word edge cases
  CASE(8, 4, 1, 2, 1);   CASE(4, 8, 2, 1, 1);
  CASE(8, 4, 1, 1, 1);   CASE(4, 8, 1, 1, 1);    CASE(6, 10, 1, 1, 3);
  // heavy crop / heavy pad
  CASE(32, 4, 2, 2, 1);  CASE(4, 32, 2, 2, 1);
  CASE(16, 6, 4, 3, 2);  CASE(6, 16, 3, 4, 2);
  // the two crop cases that exposed the drain bug
  CASE(16, 6, 4, 3, 3);  CASE(40, 96, 5, 2, 2);
  // larger widths, non-multiple + pad/crop
  CASE(96, 40, 2, 5, 2); CASE(48, 32, 4, 5, 2);  CASE(48, 32, 4, 7, 2);
  // wide words (>64 bit) - multiple and non-multiple
  CASE(128, 64, 2, 4, 2);  CASE(256, 128, 3, 6, 1);
  CASE(512, 256, 2, 4, 1); CASE(1024, 512, 1, 2, 1);
  CASE(96, 128, 4, 3, 2);  CASE(128, 96, 3, 4, 2);
  CASE(1024, 768, 3, 4, 1);
  // real mobilenet-v1 DWC width pairs (all multiple-ratio, high fan-out ratios)
  CASE(480, 15, 1, 32, 1);  CASE(576, 18, 1, 32, 1);  CASE(448, 14, 1, 32, 1);
  CASE(44, 11, 1, 4, 2);    CASE(88, 11, 1, 8, 1);    CASE(4, 32, 8, 1, 2);
  CASE(4, 64, 16, 1, 1);    CASE(160, 10, 1, 16, 1);  CASE(272, 17, 1, 16, 1);
  CASE(8, 64, 8, 1, 2);
  // folding-optimizer style: pad a real channel width to a rounder value
  CASE(44, 16, 1, 3, 2);    // 44 -> 48 pad, non-multiple
  CASE(88, 32, 1, 3, 1);    // 88 -> 96 pad
  CASE(480, 128, 1, 4, 1);  // 480 -> 512 pad, non-multiple
  // multiple fast-path: high-ratio downscale with pad / crop
  CASE(32, 4, 1, 10, 2);    // 32->40 pad  (outPerIn=8)
  CASE(32, 4, 2, 10, 2);    // 64->40 crop
  CASE(128, 8, 1, 20, 1);   // 128->160 pad (outPerIn=16)
  CASE(128, 8, 3, 20, 1);   // 384->160 crop
  CASE(480, 15, 1, 40, 1);  // 480->600 pad (outPerIn=32)
  CASE(480, 15, 2, 40, 1);  // 960->600 crop
  // multiple fast-path: high-ratio upscale with pad / crop
  CASE(4, 32, 6, 1, 2);     // 24->32 pad  (inPerOut=8)
  CASE(4, 32, 10, 1, 2);    // 40->32 crop
  CASE(8, 64, 6, 1, 1);     // 48->64 pad  (inPerOut=8)
  CASE(8, 64, 10, 1, 1);    // 80->64 crop
  cout << (fails ? "\nTOTAL FAILURES: " : "\nALL PASS ") << fails << "\n";
  return fails ? 1 : 0;
}
