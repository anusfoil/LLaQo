#!/bin/bash
#$ -N caption_audio
#$ -o /data/home/eey340/WORKPLACE/LAM/engine/data/_preprocess/logs/audio-0000x.log
#$ -wd /data/home/eey340/WORKPLACE/LAM/engine/data/_preprocess
#$ -l h_vmem=11G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y
#$ -pe smp 1
export lab_path=/data/EECS-MachineListeningLab

export WORK_DIR=/data/home/eey340/WORKPLACE/LAM/engine/data/_preprocess
export SCRATCH_DIR=/data/scratch/eey340/acav/acav10m
export OUTPUT_DIR=${lab_path}/datasets/ACAV/acav10m

module load ffmpeg/4.1.6
source /data/home/eey340/venvs/lam/bin/activate

export OMP_NUM_THREADS=1

cd ${WORK_DIR}

# 7. Caption audio
partial_prefix=custom_ncentroids-500-subset_size-10M_part00000

python3 ${WORK_DIR}/captioning.py \
    audio_captioning \
    --json_path=${OUTPUT_DIR}/meta/${partial_prefix}.json \
    --output_dir=${OUTPUT_DIR}/annotations/${partial_prefix} \
    --min_interval=0 \
    --batch_size=10 \
    # --overwrite \
    # --mini_data \

# 7*. Caption a list of audios
# start_idx=3
# batch_size=10
# end_idx=${start_idx}+${batch_size}-1

# log_dir=/data/home/eey340/WORKPLACE/LAM/engine/data/_preprocess/logs
# prefix=custom_ncentroids-500-subset_size-10M_part
# log_path=${log_dir}/logs${idx}

# for ((i=${start_idx};i<=${end_idx};i++)); do
#     idx=$(printf "%05d" $i)
#     partial_prefix=${prefix}${idx}
#     log_path=${log_dir}/audio-${idx}.log

#     python3 ${WORK_DIR}/captioning.py \
#         audio_captioning \
#         --json_path=${OUTPUT_DIR}/meta/${partial_prefix}.json \
#         --output_dir=${OUTPUT_DIR}/annotations/${partial_prefix} \
#         --min_interval=33 \
#         --batch_size=10 \
#         > ${log_path}
#         # --overwrite \
#         # --mini_data \
# done
