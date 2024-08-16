#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8       # 12 cores (12 cores per GPU)
#$ -l h_rt=2:0:0  
#$ -l h_vmem=1G   
#$ -l gpu=1         # request 1 GPU

./run_code.sh