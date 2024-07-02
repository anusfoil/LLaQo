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

(Mullama): python MU-LLaMA/MU-LLaMA/objeval.py --model MU-LLaMA/MU-LLaMA/ckpts/checkpoint.pth --llama_dir MU-LLaMA/MU-LLaMA/ckpts/LLaMA

start demo:
sh interact_gpu_node.sh  # start the environment
source venvs/lam/bin/activate
python chat.py
ssh -i ~/.ssh/id_rsa_apocrita -L 7860:rdg7:7860 acw630@login.hpc.qmul.ac.uk
(server_name needs to be the same as the given node, and in the port forwarding)

TODO: 
- audio length!!!
- Vicuna -> LLaMa
- design evaluation

