r"""Extract audio feature using AudioMAE."""
# -*- coding: utf-8 -*-
# Author: Jinhua Liang

import os
import torch
import torchaudio
import argparse
import webdataset as wds
from timm.models.layers import to_2tuple
from torch import nn
from tqdm import tqdm
import sys

sys.path.append("..")
from dataloader import AVDataset

sys.path.append("../../../src")
from utils import read_json


@torch.no_grad()
def main(args):
    annotation_dir = args.annotation_dir if args.annotation_dir is not None else args.dataset_dir

    audio_conf = {
        'num_mel_bins': 128,
        'target_length': 1024,
        'freqm': 0,
        'timem': 0,
        'mixup': 0.0,
        'dataset': "audioset",
        'mode': 'train',
        'mean': -5.081,
        'std': 4.4849,
        'noise': True,
        'label_smooth': 0,
        'im_res': 224
    }
    dataset = AVDataset(dataset_json_file=args.json_path,
                        audio_conf=audio_conf,
                        label_csv=None)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'acav10m-shard-%06d.tar')

    with wds.ShardWriter(output_path, maxcount=10000) as sink:
        for idx, (datum, fbank, _) in tqdm(enumerate(dataset)):
            if args.mini_data and idx == 9:
                break

            # vid = datum['video_id']
            # video_feature_path = os.path.join(args.dataset_dir, f"{vid}.pt")
            # output_json_path = os.path.join(args.dataset_dir, f"{vid}.json")

            # video_feature = torch.load(video_feature_path,
            #                            map_location=torch.device('cpu'))
            # assert video_feature['audio_embedding'].size(dim=0) == 1

            fname = datum['filename']
            ann_path = os.path.join(annotation_dir, f"{fname}.json")
            datum = read_json(ann_path)

            # waveform, _ = torchaudio.load(os.path.join(args.dataset_dir, f"{fname}.wav"))

            sink.write({
                '__key__': f"{idx:09d}",
                'fbank.pyd': fbank,
                # 'pyd': {
                #     "query_output": video_feature['query_output'],
                # },
                'json': {
                    'filename': fname,
                    'caption': datum['audio_caption'],
                    'qa': datum['audio_qa'],
                },
            })

            print(f"{fname} is processed already.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--annotation_dir", type=str)
    parser.add_argument("--json_path", type=str)

    parser.add_argument("--mini_data", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
