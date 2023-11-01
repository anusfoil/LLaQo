#!/bin/bash
#$ -l gpu=1
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -l h_rt=240:0:0
#$ -wd /data/home/eey340/WORKPLACE/LAM/engine/data/_preprocess
#$ -j y
#$ -N extract_feature
#$ -o /data/home/eey340/WORKPLACE/LAM/engine/data/_preprocess/caption_frames.log
#$ -m beas
#$ -l gpu_type='ampere'
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------
# Environment variables
source /data/home/eey340/venvs/lam/bin/activate
nvidia-smi
# Variables
lab_path=/data/EECS-MachineListeningLab
# Start to do experiment
module load ffmpeg/4.1.6
python3 process_video.py \
caption_frame \
--json_path=$lab_path/datasets/ACAV/acav200k/meta/acav200k.json \
--output_dir=$lab_path/datasets/ACAV/acav200k/audio_annotations \
--batch_size=6 \
--mini_data \

# python3 extract_audio_feature.py \
# --dataset_dir=$lab_path/datasets/ACAV/acav200k/tmp  \
# --json_path=$lab_path/datasets/ACAV/acav200k/meta/acav200k.json \
# --ckpt_path=$lab_path/jinhua/pretrained_models/finetuned.pth \
# --batch_size=128 \
# # --mini_data \

# find . -type f -print | sort | tar -cf ./foo.tar -T -
# python3 make_wds.py \
# --output_dir=$lab_path/datasets/ACAV/acav200k/acav200k  \
# --dataset_dir=$lab_path/datasets/ACAV/acav200k/tmp  \
# --json_path=$lab_path/datasets/ACAV/acav200k/meta/acav200k.json \
# # --mini_data \

# python3 rank_captions.py \
# --weights_path=$lab_path/jinhua/pretrained_models/laion_clap/music_speech_audioset_epoch_15_esc_89.98.pt \
# --json_path=$lab_path/datasets/ACAV/acav200k/meta/acav200k.json \
# --meta_dir=$lab_path/datasets/ACAV/acav200k/tmp \
# # --mini_data \


# CUDA_VISIBLE_DEVICES=-1 python3 \~check_threshold.py \
# --weights_path=$lab_path/jinhua/pretrained_models/laion_clap/music_speech_audioset_epoch_15_esc_89.98.pt \
# --json_path=$lab_path/datasets/ACAV/acav200k/meta/acav200k.json \
# --meta_dir=$lab_path/datasets/ACAV/acav200k/tmp\

# python3 inference.py \
# --json_path=$lab_path/datasets/ACAV/acav200k/meta/acav200k.json \
# --batch_size=8 \
# # --mini_data \
