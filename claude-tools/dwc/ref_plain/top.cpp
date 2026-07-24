#define AP_INT_MAX_W 8191
#include <ap_int.h>
#include <hls_stream.h>
#include "streamtools.h"
#ifndef INW
#define INW 128
#endif
#ifndef OUTW
#define OUTW 64
#endif
// enough input words to exercise the full datapath (>=1 output word)
constexpr unsigned NIW_ = (INW >= OUTW) ? 4u : (4u * (OUTW / INW));
void top(hls::stream<ap_uint<INW>> &in, hls::stream<ap_uint<OUTW>> &out) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE ap_ctrl_none port=return
  StreamingDataWidthConverter_Batch<INW, OUTW, NIW_>(in, out, 1);
}
