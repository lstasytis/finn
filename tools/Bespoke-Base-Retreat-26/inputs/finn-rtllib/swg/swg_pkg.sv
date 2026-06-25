package swg;
	typedef enum logic [2:0] {
		STATE_START,
		STATE_LOOP_SIMD,
		STATE_LOOP_KW,
		STATE_LOOP_KH,
		STATE_LOOP_W,
		STATE_LOOP_H
	} state_e;
endpackage : swg
