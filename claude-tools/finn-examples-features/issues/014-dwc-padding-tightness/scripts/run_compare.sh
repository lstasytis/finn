#!/bin/bash
# run_compare.sh <worktree> <tag> [models...]: fold_compare.py on the test/dwc reference estimates,
# one process per model in parallel (max 12), JSON+log to $FINN_BUILD_DIR/cmp/<model>_<tag>.*
WT=$1; TAG=$2; shift 2
cd $WT && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export OMP_NUM_THREADS=1
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/014-dwc-padding-tightness/scripts
R=/home/lstasytis/finn/build/finn_build/fx/dwc
mkdir -p $FINN_BUILD_DIR/cmp
declare -A REF=(
 [tfc-w1a1]=est_bnn-pynq_tfc-w1a1_AUP-ZU3_8GB_wd9i3_2s [lfc-w1a1]=est_bnn-pynq_lfc-w1a1_AUP-ZU3_8GB_sslnyvq4
 [lfc-w1a2]=est_bnn-pynq_lfc-w1a2_AUP-ZU3_8GB_dxj9xzv9 [cnv-w1a1]=est_bnn-pynq_cnv-w1a1_AUP-ZU3_8GB_0h9hhv5r
 [cnv-w1a2]=est_bnn-pynq_cnv-w1a2_AUP-ZU3_8GB_774gwwp5 [cnv-w2a2]=est_bnn-pynq_cnv-w2a2_AUP-ZU3_8GB_vapl4vt7
 [gtsrb]=est_gtsrb_AUP-ZU3_8GB_bs_bamye [kws]=est_kws_AUP-ZU3_8GB_of6igztd
 [cybersecurity-mlp]=est_cybersecurity-mlp_AUP-ZU3_8GB_bp7z5u4p [vgg10-radioml]=est_vgg10-radioml_ZCU104_i6l7nvym
 [mobilenet_v1]=est_mobilenet_v1_U55C_yu8t2txc [tfc-w1a2]=est_bnn-pynq_tfc-w1a2_AUP-ZU3_8GB_6hs2izbn
 [resnet18_4w4a]=est_resnet18_resnet18_4w4a_U55C_jv8qa6p2 [espcn]=est_espcn_KV260_SOM_81na9e4h
)
for m in "$@"; do
  d=${REF[$m]}; board=$(echo $d | sed -E 's/.*_(AUP-ZU3_8GB|ZCU104|U55C|KV260_SOM|VCK190|V80)_.*/\1/')
  timeout 7200 python $S/fold_compare.py $R/$d $board $FINN_BUILD_DIR/cmp/${m}_$TAG.json ${PADS:-0,4} ${SEED:-1} \
     > $FINN_BUILD_DIR/cmp/${m}_$TAG.log 2>&1 &
  echo "$m $!"
done
wait
