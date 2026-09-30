#!/bin/bash
# Core-scaling experiment: FINN's Vivado flow vs DynaRapid (cold library) at N cores.
#
# Run inside the FINN container (Vivado 2024.2, deps fetched by fetch-repos.sh, DynaRapid
# compiled). Strictly serial; do not run anything else on the machine meanwhile.
#
#   experiments/dynarapid/run_scaling.sh            # all defaults
#   MODELS="cnv" CORES="64 32 16" experiments/dynarapid/run_scaling.sh
#
# Environment (all optional):
#   MODELS     models to run (default "cnv tfc"; prepared by prepare_model.py if missing)
#   MODES      flows to run at each core count (default "vivado dynarapid")
#   CORES      core counts, largest first (default: nproc, then halving down to 4)
#   PART       FPGA part (default xczu7ev-ffvc1156-2-e, ZCU104)
#   D          data directory (default $FINN_BUILD_DIR/dr_scaling)
#   O          output directory of the timed runs (default $D/scaling)
#   MAX_SLOTS  memory bound on concurrent Vivado runs (default from free memory, as
#              finn.util.dynarapid.tools.vivado_slots: 0.85 * 0.6 * free GB / 4.5 GB)
#
# Per run: taskset -c 0-(N-1), NUM_DEFAULT_WORKERS / --workers N,
# DYNARAPID_VIVADO_SLOTS=min(N, MAX_SLOTS), Vivado general.maxThreads=min(N, 8) via
# ~/.Xilinx/Vivado/Vivado_init.tcl (an existing file is restored at the end), a fresh
# DynaRapid library per run, the shell built once beforehand (untimed, its time is in
# $D/prepare_shell.log). Memory is sampled with free every 5 s.
# Summary: python experiments/dynarapid/summarize_scaling.py $D/scaling
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
EXP=$REPO/experiments/dynarapid
D=${D:-$FINN_BUILD_DIR/dr_scaling}
PART=${PART:-xczu7ev-ffvc1156-2-e}
MODELS=${MODELS:-"cnv tfc"}
MODES=${MODES:-"vivado dynarapid"}
if [ -z "${CORES:-}" ]; then
    CORES=""; n=$(nproc); while [ $n -ge 4 ]; do CORES="$CORES $n"; n=$((n / 2)); done
fi
if [ -z "${MAX_SLOTS:-}" ]; then
    free_gb=$(awk '/MemAvailable/{print int($2 / 1048576)}' /proc/meminfo)
    MAX_SLOTS=$(python3 -c "print(max(1, int(0.85 * 0.6 * $free_gb / 4.5)))")
fi
O=${O:-$D/scaling}
S=$D/lib/shells
INIT=$HOME/.Xilinx/Vivado/Vivado_init.tcl
mkdir -p $O/bit $(dirname $INIT)
[ -f $INIT ] && [ ! -f $INIT.scaling_backup ] && cp $INIT $INIT.scaling_backup
restore() { if [ -f $INIT.scaling_backup ]; then mv $INIT.scaling_backup $INIT; else rm -f $INIT; fi; }
trap restore EXIT
cd $EXP
echo "MODELS=$MODELS MODES=$MODES CORES=$CORES MAX_SLOTS=$MAX_SLOTS PART=$PART D=$D $(date -Is)"

# models (front end + HLS IP generation; not timed)
for m in $MODELS; do
    if [ ! -f $D/$m/dataflow_ipgen.onnx ]; then
        echo "PREPARE $m $(date -Is)"
        python prepare_model.py --topology $m --out $D/$m --part $PART > $D/prepare_$m.log 2>&1 \
            || { echo "prepare $m failed, see $D/prepare_$m.log"; exit 1; }
    fi
done
# shell (built by one untimed DynaRapid run; shared by the models with the same IODMA widths)
for m in $MODELS; do
    echo "SHELL $m $(date -Is)"
    python run_bitfile_experiment.py --model $D/$m/dataflow_ipgen.onnx --out $D/prepare_shell_$m \
        --mode dynarapid --library $D/lib/lib --shell-lib $S --workers $(nproc) \
        >> $D/prepare_shell.log 2>&1 || { echo "shell run for $m failed, see $D/prepare_shell.log"; exit 1; }
done

memsample() { # $1 = output file; used memory (MB) every 5 s until killed
    while true; do echo "$(date +%s) $(free -m | awk '/^Mem:/{print $3}')"; sleep 5; done > "$1"
}

run() { # model mode N
    local m=$1 mode=$2 n=$3 name=$1_$2_n$3
    local slots=$(( n < MAX_SLOTS ? n : MAX_SLOTS )) thr=$(( n < 8 ? n : 8 ))
    echo "set_param general.maxThreads $thr" > $INIT
    rm -rf $O/bit/$name $O/lib_$name
    local extra=""
    if [ $mode = dynarapid ]; then
        mkdir -p $O/lib_$name; ln -sfn $S $O/lib_$name/shells
        extra="--library $O/lib_$name/lib --shell-lib $S"
    fi
    memsample $O/mem_$name.txt & local mp=$!
    echo "START $name slots=$slots threads=$thr $(date -Is)"
    DYNARAPID_VIVADO_SLOTS=$slots NUM_DEFAULT_WORKERS=$n taskset -c 0-$((n-1)) \
        /usr/bin/time -v python run_bitfile_experiment.py --model $D/$m/dataflow_ipgen.onnx \
        --out $O/bit/$name --mode $mode --workers $n $extra > $O/$name.log 2>&1
    local rc=$?
    kill $mp
    echo "END $name rc=$rc $(date -Is)"
}

for m in $MODELS; do
    for n in $CORES; do
        for mode in $MODES; do run $m $mode $n; done
    done
done
python $EXP/summarize_scaling.py $O
echo ALLDONE
