#!/bin/bash

#PBS -q gpuvolta
#PBS -P pg44
#PBS -l walltime=48:00:00
#PBS -l mem=80GB
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l jobfs=128GB
#PBS -l wd
#PBS -l storage=scratch/zg12
#PBS -M adin6536@uni.sydney.edu.au
#PBS -o output_256/quadcompt_dit_sc9p0_sk11.txt
#PBS -e errors/quadcompt_cond_quad_error1.txt


module load use.own
module load python3/3.9.2
module load gdiff
#module load ASDiffusion



MODEL_FLAGS=""

SAMPLE_FLAGS="--num-fid-samples 50000 --per-proc-batch-size 40 "
#SAMPLE_FLAGS="--num-fid-samples 6 --per-proc-batch-size 2 "
#SAMPLE_FLAGS="--num-fid-samples 6 --per-proc-batch-size 2 "
#SAMPLE_FLAGS="--batch_size 2 --num_samples 30000 --timestep_respacing 250"


cmd="cd ../"
#echo ${cmd}
#eval ${cmd}

cmd="ls"
echo ${cmd}
eval ${cmd}

#scales=( "0.5" "1.0" "2.0" )
scales=( "9.0" )
basefolder="/scratch/zg12/dd9648/"
skips=("11")



for scale in "${scales[@]}"
do
  for skip in "${skips[@]}"
  do
cmd="WORLD_SIZE=1 RANK=0 MASTER_IP=127.0.0.1 MASTER_PORT=29510 MARSV2_WHOLE_LIFE_STATE=0 python3  sample_compact_ddp_server.py $MODEL_FLAGS --cfg-scale ${scale}  $SAMPLE_FLAGS  \
 --sample-dir runsDiT/DiTCompactQuad/IMN256/scale${scale}_sk${skip}/ --ckpt DiTmodels/DiT-XL-2-256x256.pt --base_folder ${basefolder} --skip ${skip} --skip_type quadratic"
echo ${cmd}
eval ${cmd}
done
done


for scale in "${scales[@]}"
do
cmd="python3 evaluations/evaluator_tolog.py ${basefolder}/reference/VIRTUAL_imagenet256_labeled.npz \
 ${basefolder}/runsDiT/DiTCompactQuad/IMN256/scale${scale}_sk${skip}/reference/samples_50000x256x256x3.npz"
echo ${cmd}
eval ${cmd}
done





#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}