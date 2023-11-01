#!/bin/bash
#$ -N download_data
#$ -o /data/home/eey340/WORKPLACE/LAM/engine/data/_preprocess/logs/download_data.log
#$ -wd /data/home/eey340/WORKPLACE/LAM/engine/data/_preprocess
#$ -l h_vmem=11G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y
##### Array jobs ######
#$ -t 751-770
#$ -tc 1000
#$ -pe smp 1
# 831-850

export lab_path=/data/EECS-MachineListeningLab

export WORK_DIR=/data/home/eey340/WORKPLACE/LAM/engine/data/_preprocess
export SCRATCH_DIR=/data/scratch/eey340/acav/acav10m
export OUTPUT_DIR=${lab_path}/datasets/ACAV/acav10m

module load ffmpeg/4.1.6
source /data/home/eey340/venvs/lam/bin/activate

export OMP_NUM_THREADS=1

# 1. Split source csv file to split csv, each containing 5000 pieces
# python3 ${WORK_DIR}/prepare_acav.py \
# split_source_csv_to_partial_csvs \
# --source_csv=${SCRATCH_DIR}/ncentroids-500-subset_size-10M.csv \
# --audios_num_per_file=10000 \
# --partial_csvs_output_dir=${SCRATCH_DIR}/partial_csvs \

# 2. Download wav using split csv, ones can use subprocess for this part
log_dir=/data/home/eey340/acav10m
prefix=ncentroids-500-subset_size-10M_part

idx=$((${SGE_TASK_ID}-1))
idx=$(printf "%05d" ${idx})
csv_name=${prefix}${idx}
log_path=${log_dir}/logs${idx}
python3 ${WORK_DIR}/prepare_acav.py \
    download_videos \
    --csv_path ${SCRATCH_DIR}/partial_csvs/${csv_name}.csv \
    --clips_dir ${SCRATCH_DIR}/downloaded_videos/${csv_name} \
    # --mini_data \

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
