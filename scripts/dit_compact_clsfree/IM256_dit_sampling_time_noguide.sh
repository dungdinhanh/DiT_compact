#!/bin/bash



MODEL_FLAGS=""

SAMPLE_FLAGS="--num-fid-samples 50000 --per-proc-batch-size 32 "
#SAMPLE_FLAGS="--num-fid-samples 6 --per-proc-batch-size 2 "
SAMPLE_FLAGS="--num-fid-samples 60 --per-proc-batch-size 60 "
#SAMPLE_FLAGS="--batch_size 2 --num_samples 30000 --timestep_respacing 250"


cmd="cd ../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

#scales=( "0.5" "1.0" "2.0" )
scales=( "0" )
basefolder="."



for scale in "${scales[@]}"
do
cmd="python sample_compact_ddp_server_time.py $MODEL_FLAGS --cfg-scale ${scale}  $SAMPLE_FLAGS  \
 --sample-dir runs/DiTCompactT/IMN256/scale${scale}2/ --ckpt DiTmodels/DiT-XL-2-256x256.pt --base_folder ${basefolder} --skip 1"
echo ${cmd}
eval ${cmd}
done







#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}