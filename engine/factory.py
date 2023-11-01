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
