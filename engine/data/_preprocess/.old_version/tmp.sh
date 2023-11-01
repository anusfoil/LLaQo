#!/bin/bash
#$ -N part39
#$ -o ~/acav200k/logs39
#$ -wd /data/scratch/eey340/acav/acav200k
#$ -pe smp 1
#$ -l h_vmem=10G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y
source /data/home/eey340/venvs/lam/bin/activate
nvidia-smi
# Variables
lab_path=/data/EECS-MachineListeningLab
work_dir=/data/home/eey340/WORKPLACE/LAM/engine/data/_preprocess
scratch_dir=/data/scratch/eey340/acav/acav200k
output_dir=${lab_path}/datasets/ACAV/acav200k
module load ffmpeg/4.1.6

source /data/home/eey340/venvs/lam/bin/activate
csv_name=ncentroids-500-subset_size-200K_part00039  # ncentroids-500-subset_size-200K_part00039
wd=/data/scratch/eey340/acav/acav200k

# Replace the following line with a program or command
python3 /data/home/eey340/WORKPLACE/preprocessing/dataset.py \
download_wavs \
--csv_path ${wd}/meta/${csv_name}.csv \
--clips_dir ${wd}/videos/${csv_name}

# Tease meta
python3 /data/home/eey340/WORKPLACE/LAM/engine/data/_preprocess/tease_meta.py \
--origin_csv_dir=${scratch_dir}/meta \
--video_storage_dir=${scratch_dir}/videos \
--target_csv_dir=${lab_path}/datasets/ACAV/acav200k/meta

# Extract frames
python3 ${work_dir}/extract_video.py \
--csv_dir=${output_dir}/meta \
--frame_output_dir=${output_dir}/frames

# Extract audios
python3 ${work_dir}/extract_audio.py \
--csv_dir=${output_dir}/meta \
--audio_output_dir=${output_dir}/audios

# Curate json file
python3 ${work_dir}/create_json.py \
--output_path=$lab_path/datasets/ACAV/acav200k/meta/acav200k.json \
--local_dir=$lab_path/datasets/ACAV/acav200k

# Extract audio features
python3 ${work_dir}/extract_audio_feature.py \
--json_path=$lab_path/datasets/ACAV/acav200k/meta/acav200k.json \
--ckpt_path=$lab_path/jinhua/pretrained_models/finetuned.pth



# Generated by Job Script Builder on 2023-03-24
# For assistance, please email its-research-support@qmul.ac.uk
