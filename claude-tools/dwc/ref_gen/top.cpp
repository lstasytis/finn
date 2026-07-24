#define AP_INT_MAX_W 8191
#include <ap_int.h>
#include <hls_stream.h>
#include "streamtools.h"
#ifndef INW
#define INW 96
#endif
#ifndef OUTW
#define OUTW 40
#endif
static constexpr unsigned gcd_(unsigned a, unsigned b){ return b ? gcd_(b, a%b) : a; }
constexpr unsigned LCM_ = (INW / gcd_(INW,OUTW)) * OUTW;
constexpr unsigned NIW_ = LCM_ / INW;   // exact one-LCM-frame
constexpr unsigned NOW_ = LCM_ / OUTW;
void top(hls::stream<ap_uint<INW>> &in, hls::stream<ap_uint<OUTW>> &out) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE ap_ctrl_none port=return
  StreamingDataWidthConverterGeneralized_Batch<INW, OUTW, NIW_, NOW_>(in, out, 1);
}
