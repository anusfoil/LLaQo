#!/bin/bash
#$ -l gpu=1
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -l h_rt=240:0:0
#$ -wd /data/home/acw630/WORKPLACE/LAM/engine/data/_preprocess
#$ -j y
#$ -N frames_43x
#$ -o /data/home/acw630/WORKPLACE/LAM/engine/data/_preprocess/caption_frames43x.log
#$ -m beas
#$ -l gpu_type='ampere'
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------
# Variables
partial_prefix=custom_ncentroids-500-subset_size-10M_part00435

# Environment variables
export PYTHONPATH="/data/home/acw630/WORKPLACE/python.lib:${PYTHONPATH}"
export HF_HOME=/data/EECS-MachineListeningLab/huan/hf_cache
export HF_DATASETS_CACHE="/data/EECS-MachineListeningLab/huan/lm"

export lab_path=/data/EECS-MachineListeningLab
export WORK_DIR=/data/home/acw630/WORKPLACE/LAM/engine/data/_preprocess
export SCRATCH_DIR=/data/scratch/acw630/acav/acav10m
export OUTPUT_DIR=${lab_path}/datasets/ACAV/acav10m

module load ffmpeg/4.1.6
source /data/home/acw630/venvs/lam/bin/activate

nvidia-smi

cd ${WORK_DIR}

python3 ${WORK_DIR}/captioning.py \
    frame_captioning \
    --json_path=${OUTPUT_DIR}/meta/${partial_prefix}.json \
    --output_dir=${OUTPUT_DIR}/annotations/${partial_prefix} \
    --batch_size=8 \
    # --overwrite \
    # --mini_data \
