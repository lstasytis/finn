# FINN environment with Vivado/Vitis 2024.2 from the labstore (sourced in place)
_strip() { echo "$1" | tr ':' '\n' | grep -v '/Xilinx/' | paste -sd: -; }
export PATH=$(_strip "$PATH")
export LD_LIBRARY_PATH=$(_strip "$LD_LIBRARY_PATH")
export PYTHONUSERBASE=/tmp/home_dir/.local
export VIVADO_PATH=/mnt/labstore/Xilinx/Vivado/2024.2
export VITIS_PATH=/mnt/labstore/Xilinx/Vitis/2024.2
export HLS_PATH=/mnt/labstore/Xilinx/Vitis_HLS/2024.2
source $VITIS_PATH/settings64.sh >/dev/null 2>&1 || true
source $VIVADO_PATH/settings64.sh >/dev/null 2>&1 || true
export XILINX_VIVADO=$VIVADO_PATH
export XILINX_VITIS=$VITIS_PATH
export XILINX_HLS=$HLS_PATH
export PATH=/tmp/home_dir/.local/bin:$HLS_PATH/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu/:${XILINX_VIVADO}/lib/lnx64.o
