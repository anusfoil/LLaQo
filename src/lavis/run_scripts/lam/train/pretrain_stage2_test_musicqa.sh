#!/bin/bash
#$ -l gpu=2
#$ -pe smp 16
#$ -l h_vmem=11G
#$ -l h_rt=24:0:0
#$ -wd /data/home/acw630/WORKPLACE/LAM/engine/data/_preprocess
#$ -j y
#$ -N test_musicqa
#$ -o /data/home/acw630/WORKPLACE/LAM/src/lavis/run_scripts/lam/train/logs/pretraining_stage2_test_musicqa.log
#$ -m beas
#$ -l gpu_type='ampere'
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------
# Environment variables
export PYTHONPATH="/data/home/acw630/WORKPLACE/python.lib:${PYTHONPATH}"
export HF_HOME=/data/EECS-MachineListeningLab/huan/hf_cache
export HF_DATASETS_CACHE="/data/EECS-MachineListeningLab/huan/lm"
export WORK_PLACE=/data/home/acw630/WORKPLACE/LAM/src/lavis

export OMP_NUM_THREADS=1

source ~/.bashrc
source /data/home/acw630/venvs/lam/bin/activate
# nvidia-smi

# max_iters = 4891330 samples / 20 sampler_per_batch * 5 epoch
python -m torch.distributed.run \
    --nproc_per_node=2 \
    ${WORK_PLACE}/train.py \
    --cfg-path ${WORK_PLACE}/lavis/projects/lam/train/pretrain_stage2_test_musicqa.yaml \
    --options \
    run.max_iters=50000 \
    run.iters_per_inner_epoch=3000 \
    run.batch_size_train=4 \
    run.accum_grad_iters=1 \
    run.num_workers=4

# python ${WORK_PLACE}/train.py \
#     --cfg-path ${WORK_PLACE}/lavis/projects/lam/train/pretrain_stage2_test_musicqa.yaml \
#     --options \
#     run.max_iters=50000 \
#     run.iters_per_inner_epoch=3000 \
#     run.batch_size_train=6 \
#     run.accum_grad_iters=1 \
#     run.num_workers=4

# python3 ${WORK_PLACE}/train.py \
#     --cfg-path ${WORK_PLACE}/lavis/projects/lam/train/pretrain_stage2.yaml \
#     --options \
#     run.max_iters=360000 \
#     run.iters_per_inner_epoch=30000 \
#     run.batch_size_train=10 \
#     run.batch_size_eval=10 \
#     run.accum_grad_iters=20 \
#     run.num_workers=2 \
#     run.distributed=False
