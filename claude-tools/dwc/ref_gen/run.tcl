open_project -reset proj
set_top top
add_files top.cpp -cflags "-I$::env(FINN_ROOT)/deps/finn-hlslib -DINW=$::env(INW) -DOUTW=$::env(OUTW)"
open_solution -reset sol1
set_part {xc7z020clg400-1}
create_clock -period 3.333 -name default
csynth_design
exit
