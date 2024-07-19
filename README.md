# LLaQo: Towards a query-based coach in expressive performance assessment


Note to self:

Training: 
qsub pretrain_stage2_test_musicqa.sh
sh  WORKPLACE/LAM/src/lavis/run_scripts/lam/train/pretrain_stage2_test_musicqa.sh 

evaluate:
sh interact_gpu_node.sh  # start the environment
source ~/venvs/lam/bin/activate or source ~/venvs/lam2/bin/activate
cd WORKPLACE/LAM/engine
python evaluate_musicqa.py

(Mullama): python /data/scratch/acw630/MU-LLaMA/MU-LLaMA/objeval.py --model /data/scratch/acw630/MU-LLaMA/MU-LLaMA/ckpts/checkpoint.pth --llama_dir /data/scratch/acw630/MU-LLaMA/MU-LLaMA/ckpts/LLaMA

start demo:
sh interact_gpu_node.sh  # start the environment
source venvs/lam/bin/activate
python chat.py
ssh -i ~/.ssh/id_rsa_apocrita -L 7860:rdg7:7860 acw630@login.hpc.qmul.ac.uk
(server_name needs to be the same as the given node, and in the port forwarding)

ltu:
source /etc/profile
module load anaconda3
conda activate venv_ltu_as
cd /data/scratch/acw630/ltu/src/ltu_as/
python inference.py

plotting
scp -r -i id_rsa  WORKPLACE/LAM/engine/results/ hz009@frank.eecs.qmul.ac.uk:/homes/hz009/Research/llaqo