// Original FINN HLS non-multiple DWC: two StreamingDataWidthConverter_Batch
// cascaded through an LCM(inWidth,outWidth) intermediate stream.
#define AP_INT_MAX_W 8191
#include <ap_int.h>
#include <hls_stream.h>
#include "streamtools.h"
#ifndef INW
#define INW 22
#endif
#ifndef OUTW
#define OUTW 10
#endif
static constexpr unsigned gcd_(unsigned a, unsigned b){ return b ? gcd_(b, a%b) : a; }
constexpr unsigned LCMW = (INW / gcd_(INW,OUTW)) * OUTW;
constexpr unsigned NIW  = LCMW / INW;   // input words per LCM word
void top(hls::stream<ap_uint<INW>> &in, hls::stream<ap_uint<OUTW>> &out) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE ap_ctrl_none port=return
  hls::stream<ap_uint<LCMW>> mid("mid");
  StreamingDataWidthConverter_Batch<INW, LCMW, NIW>(in, mid, 1);
  StreamingDataWidthConverter_Batch<LCMW, OUTW, 1>(mid, out, 1);
}
