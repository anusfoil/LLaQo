#!/bin/bash
#$ -N curate_data
#$ -o /data/home/acw630/WORKPLACE/LAM/engine/data/_preprocess/logs/curate_data.log
#$ -wd /data/home/acw630/WORKPLACE/LAM/engine/data/_preprocess
#$ -l h_vmem=11G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y
##### Array jobs ######
#$ -t 751-770
#$ -tc 1000
#$ -pe smp 1
##### Array jobs ######
###$ -l gpu=1
###$ -l gpu_type='ampere'
###$ -pe smp 8

export lab_path=/data/EECS-MachineListeningLab

export WORK_DIR=/data/home/acw630/WORKPLACE/LAM/engine/data/_preprocess
export SCRATCH_DIR=/data/scratch/acw630/acav/acav10m
export OUTPUT_DIR=${lab_path}/datasets/ACAV/acav10m

module load ffmpeg/4.1.6
source /data/home/acw630/venvs/lam/bin/activate

export OMP_NUM_THREADS=1

# 1. Split source csv file to split csv, each containing 5000 pieces
# python3 ${WORK_DIR}/prepare_acav.py \
# split_source_csv_to_partial_csvs \
# --source_csv=${SCRATCH_DIR}/ncentroids-500-subset_size-10M.csv \
# --audios_num_per_file=10000 \
# --partial_csvs_output_dir=${SCRATCH_DIR}/partial_csvs \

# 2. Download wav using split csv, ones can use subprocess for this part
# log_dir=/data/home/acw630/acav10m
# prefix=ncentroids-500-subset_size-10M_part

# idx=$((${SGE_TASK_ID}-1))
# idx=$(printf "%05d" ${idx})
# csv_name=${prefix}${idx}
# log_path=${log_dir}/logs${idx}
# python3 ${WORK_DIR}/prepare_acav.py \
#     download_videos \
#     --csv_path ${SCRATCH_DIR}/partial_csvs/${csv_name}.csv \
#     --clips_dir ${SCRATCH_DIR}/downloaded_videos/${csv_name} \
#     # --mini_data \

# start_idx=100
# batch_size=100
# end_idx=${start_idx}+${batch_size}-1
# idx=$(printf "%05d" ${SGE_TASK_ID})
# csv_name=${prefix}${idx}
# log_path=${log_dir}/logs${idx}
# for ((i=${start_idx};i<=${end_idx};i++)); do
#     idx=$(printf "%05d" $i)
#     csv_name=${prefix}${idx}
#     log_path=${log_dir}/logs${idx}

#     nohup python3 ${WORK_DIR}/prepare_acav.py \
#         download_videos \
#         --csv_path ${SCRATCH_DIR}/partial_csvs/${csv_name}.csv \
#         --clips_dir ${SCRATCH_DIR}/downloaded_videos/${csv_name} \
#         --mini_data \
#         > ${log_path} 2>&1 &
# done

# 3. Downsample video, ones can use subprocess for this part
prefix=custom_ncentroids-500-subset_size-10M_part

idx=$((${SGE_TASK_ID}-1))
idx=$(printf "%05d" ${idx})
partial_prefix=${prefix}${idx}
python3 ${WORK_DIR}/preprocessing.py \
    process_video \
    --csv_pth=${SCRATCH_DIR}/partial_csvs/${partial_prefix}.csv \
    --output_dir=${OUTPUT_DIR}/frames/${partial_prefix} \
    # --mini_data \

# 4. Resample audio, ones can use subprocess for this part
prefix=custom_ncentroids-500-subset_size-10M_part

idx=$((${SGE_TASK_ID}-1))
idx=$(printf "%05d" ${idx})
partial_prefix=${prefix}${idx}
python3 ${WORK_DIR}/preprocessing.py \
    process_audio \
    --csv_pth=${SCRATCH_DIR}/partial_csvs/${partial_prefix}.csv \
    --output_dir=${OUTPUT_DIR}/audios/${partial_prefix} \

# 5. Create json file for sequtial dataloader, ones can use subprocess for this part
prefix=custom_ncentroids-500-subset_size-10M_part

idx=$((${SGE_TASK_ID}-1))
idx=$(printf "%05d" ${idx})
partial_prefix=${prefix}${idx}
python3 ${WORK_DIR}/preprocessing.py \
    process_meta \
    --prefix=${partial_prefix} \
    --local_dir=${OUTPUT_DIR} \
    --output_dir=${OUTPUT_DIR}/meta \

# 6. Frame captioning
# partial_prefix=custom_ncentroids-500-subset_size-10M_part00000

# python3 ${WORK_DIR}/captioning.py \
#     frame_captioning \
#     --json_path=${OUTPUT_DIR}/meta/${partial_prefix}.json \
#     --output_dir=${OUTPUT_DIR}/annotations/${partial_prefix} \
#     --batch_size=8 \
#     # --mini_data \

# 7. Caption audio
# cd ${WORK_DIR}
# partial_prefix=custom_ncentroids-500-subset_size-10M_part00002

# python3 ${WORK_DIR}/captioning.py \
#     audio_captioning \
#     --json_path=${OUTPUT_DIR}/meta/${partial_prefix}.json \
#     --output_dir=${OUTPUT_DIR}/annotations/${partial_prefix} \
#     --min_interval=33 \
#     --batch_size=10 \
#     # --overwrite \
#     # --mini_data \

# 8. refine caption using CLAP
# partial_prefix=custom_ncentroids-500-subset_size-10M_part00002

# python3 ${WORK_DIR}/rank_captions.py \
# --weights_path=$lab_path/huan/pretrained_models/laion_clap/music_speech_audioset_epoch_15_esc_89.98.pt \
# --json_path=${OUTPUT_DIR}/meta/${partial_prefix}.json \
# --meta_dir=${OUTPUT_DIR}/annotations/${partial_prefix} \
# --enable_gpt \
# --mini_data \
# # --use_cuda \


# 9. (Optional) Generate audio q & a
# partial_prefix=custom_ncentroids-500-subset_size-10M_part00000

# python3 ${WORK_DIR}/captioning.py \
#     qa_generations \
#     --json_path=${OUTPUT_DIR}/meta/${partial_prefix}.json \
#     --output_dir=${OUTPUT_DIR}/annotations/${partial_prefix} \
#     --min_interval=40 \
#     # --mini_data \

# 10. Create webdataset
# python3 ${WORK_DIR}/make_wds.py \
#     --json_path=${OUTPUT_DIR}/meta/${partial_prefix}.json \
#     --output_dir=${OUTPUT_DIR}/foo \
#     --dataset_dir=${OUTPUT_DIR}
#     # --mini_data \
