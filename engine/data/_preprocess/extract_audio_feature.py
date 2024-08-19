r"""Extract audio feature using AudioMAE."""
# -*- coding: utf-8 -*-
# Author: Jinhua Liang

import os
import logging
import torch
import argparse
from timm.models.layers import to_2tuple
from torch import nn, tensor_split
from tqdm import tqdm
import sys

sys.path.append("..")
from dataloader import AVDataset

sys.path.append("../../../src")
from audiomae import models_vit
from utils import write_json, read_json


class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=stride)  # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size)  # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


@torch.no_grad()
def main(args):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AVDataset(dataset_json_file=args.json_path,
                        audio_conf=audio_conf,
                        label_csv=None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
    )
    model = models_vit.__dict__["vit_base_patch16"](
        num_classes=527,
        drop_path_rate=0.1,
        global_pool=True,
        mask_2d=True,
        use_custom_patch=False,
    )
    ln_audio = torch.nn.LayerNorm(model.hidden_size, eps=1e-6).to(device)

    target_length = {
        'audioset': 1024,
        'k400': 1024,
        'esc50': 512,
        'speechcommands': 128
    }
    img_size = (target_length["audioset"], 128)  # 1024, 128
    in_chans = 1
    emb_dim = 768

    model.patch_embed = PatchEmbed_new(
        img_size=img_size,
        patch_size=(16, 16),
        in_chans=in_chans,
        embed_dim=emb_dim,
        stride=16)  # no overlap. stride=img_size=16
    num_patches = model.patch_embed.num_patches
    model.pos_embed = nn.Parameter(
        torch.zeros(1, num_patches + 1,
                    emb_dim), requires_grad=False)  # fixed sin-cos embedding

    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.ckpt_path)
    checkpoint_model = checkpoint['model']
    # state_dict = model.state_dict()

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    model = model.to(device)

    for batch_idx, batch in tqdm(enumerate(dataloader)):
        if args.mini_data and batch_idx == 1:
            break
        data, audio, _ = batch
        outputs = model(audio.to(device))
        audio_embeddings = ln_audio(outputs["audio_embedding"]).cpu()
        audio_embeddings = tensor_split(audio_embeddings,
                                        args.batch_size,
                                        dim=0)

        for video_id, audio_emb in zip(data['video_id'], audio_embeddings):
            assert audio_emb.size(dim=0) == 1, audio_emb.size()

            video_feature_path = os.path.join(args.dataset_dir,
                                              f"{video_id}.pt")
            video_feature = torch.load(video_feature_path,
                                       map_location=torch.device('cpu'))

            torch.save(
                {
                    "audio_embedding": audio_emb.detach().clone(),
                    "query_output": video_feature["query_output"],
                    "input_t5": video_feature["input_t5"],
                }, video_feature_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--json_path", type=str)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--mini_data", action="store_true", default=False)
    parser.add_argument("--ckpt_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
