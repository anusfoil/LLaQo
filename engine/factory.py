import os, glob, re
import torch
from subprocess import run
from torch import Tensor
from dataclasses import dataclass


@dataclass
class QformerPretrainOutput:
    """Class for keeping track of an item in inventory."""
    meta: dict
    audio_feature: Tensor
    audio_logit_scale: Tensor
    visual_feature: Tensor
    visual_logit_scale: Tensor
    text_feature: Tensor


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def count_samples(input_shards: list):
    r"""Cound number of samples as an auxilary function for webdataset.
        Args: input_shards: list of paths of '.tar' shards."""
    num_samples = 0
    for shard in input_shards:
        cmd = ['tar', '-tf', shard]
        res = run(cmd, capture_output=True, text=True).stdout
        num_samples += len(res.split('\n')[:-1]) // 3

    return num_samples


def tensor_move_to(input, device=torch.device('cpu')):
    try:
        ori_device = input.device
    except:
        raise TypeError("Input must be a Tensor.")

    if ori_device != device:
        input.data = input.to(device)

        if input.grad is not None:
            input.grad.data = input.grad.to(device)

    return input


# def load_latest_checkpoint(llm='vicuna'):
    
#     checkpoint_dir = "/data/EECS-MachineListeningLab/huan/lam/check_point/Pretrain_stage2/test_musicqa/**/*.pth"
#     list_of_files = glob.glob(checkpoint_dir, recursive=True) # * means all if need specific format then *.csv
#     latest_file = max(list_of_files, key=os.path.getctime)

#     return latest_file


def load_latest_checkpoint(llm='vicuna', ckpt=None):
    # Directory where the checkpoints and log files are stored
    base_dir = "/data/EECS-MachineListeningLab/huan/lam/check_point/Pretrain_stage2/test_musicqa"
    
    # Get all subdirectories in the base directory
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Filter subdirectories that contain a log.txt and read the 'llm_model' from log.txt
    filtered_dirs = []
    for subdir in subdirs:
        log_path = os.path.join(base_dir, subdir, "log.txt")
        if os.path.exists(log_path):
            with open(log_path, 'r') as file:
                for line in file:
                    if '"llm_model":' in line:
                        try:
                            # This extracts the JSON object substring that contains "llm_model"
                            match = re.search(r'"llm_model":\s*"([^"]+)', line)
                            if match:
                                llm_model_path = match.group(1)
                                # Check if the model path contains the specified llm type
                                if llm in llm_model_path:
                                    filtered_dirs.append(subdir)
                                    break
                        except Exception as e:
                            print(f"Error parsing JSON or extracting llm_model in {log_path}: {str(e)}")
     
    if ckpt:
        # If a specific checkpoint is specified, return it if it exists
        ckpt_path = os.path.join(base_dir, ckpt)
        if os.path.exists(ckpt_path):
            return ckpt_path
        else:
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
     
    # Now find the latest checkpoint in the filtered directories
    latest_time = -1
    latest_file = None
    
    for subdir in filtered_dirs:
        # Check all .pth files in this directory
        files = glob.glob(os.path.join(base_dir, subdir, "*.pth"))
        for file in files:
            file_time = os.path.getctime(file)
            if file_time > latest_time:
                latest_time = file_time
                latest_file = file

    return latest_file
