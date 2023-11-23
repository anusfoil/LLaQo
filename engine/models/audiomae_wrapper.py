import os
import math
import logging
import torch
import torchaudio
from torch import Tensor
from timm.models.layers import to_2tuple
from torch import nn

import sys

# FIXME: RELETIVE PATH
# sys.path.append("..")
sys.path.append("/data/home/eey340/WORKPLACE/LAM/engine")
from factory import tensor_move_to

# sys.path.append("../../src")
sys.path.append("/data/home/eey340/WORKPLACE/LAM/src")
from audiomae import models_vit
from utils import read_json


class PatchEmbed(nn.Module):
    r"""Flexible Image to Patch Embedding."""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )  # with overlapped patches
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        # self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size)  # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class AudioMAE(nn.Module):
    r"""A wrapper for AudioMAE."""
    audio_config = {
        "target_length": 1024,
        "n_mels": 128,
        "mean": -5.081,
        "std": 4.4849,
    }

    def __init__(
        self,
        model_name: str = "vit_base_patch16",
        hidden_size: int = 768,
        num_classes: int = 527,
        drop_path_rate: float = 0.1,
        global_pool: bool = True,
        mask_2d: bool = True,
        # padding
        target_length: int = 1024,  # audioset
        use_custom_patch: bool = False,
        # other
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.model = models_vit.__dict__[model_name](
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
            mask_2d=mask_2d,
            use_custom_patch=use_custom_patch,
        )

        img_size = (target_length, 128)  # 1024, 128
        self.target_length = target_length
        self.hidden_size = emb_dim = hidden_size

        self.model.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=(16, 16),
            in_chans=1,
            embed_dim=emb_dim,
            stride=16,
        )  # no overlap. stride=img_size=16
        num_patches = self.model.patch_embed.num_patches
        self.model.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.device = device
        self.model = self.model.to(device)

    def from_pretrained(self, ckpt_path: str) -> str:
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        logging.info(f"Load pre-trained checkpoint from: {ckpt_path}")
        checkpoint_model = checkpoint["model"]

        msg = self.model.load_state_dict(checkpoint_model, strict=False)

        return msg

    def forward(self, batch: Tensor) -> Tensor:
        return self.model(batch.to(self.device))

    def get_audio_embedding(self, batch):
        r"""Get audio embeddings with various lengths."""
        B, C, T, F = batch.size()
        n_seg = math.ceil(T / self.target_length)

        input_vit = torch.zeros(
            B * n_seg, C, self.target_length, F, device=batch.device
        )
        for i in range(n_seg):
            T_stt = self.target_length * i
            T_end = min(T_stt + self.target_length, T)
            input_vit[B * i : B * (i + 1), :, :, :] += batch[:, :, T_stt:T_end, :]

        output = self.model(input_vit.to(self.device))
        output["batch_size"] = B

        return output

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0 : waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0 : waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        # try:
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.audio_config["n_mels"],
            dither=0.0,
            frame_shift=10,
        )
        # except:
        #     fbank = torch.zeros([512, 128]) + 0.01
        #     print('there is a loading error')

        target_length = self.audio_config["target_length"]
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def preprocess_audio(self, audio_path, skip_norm=True):
        try:
            fbank = self._wav2fbank(audio_path, None, 0)
        except:
            fbank = torch.zeros(self.audio_config["target_length"], 128) + 0.01
            print("there is an error in loading audio")

        # normalize the input for both training and test
        if not skip_norm:
            fbank = (fbank - self.audio_config["norm_mean"]) / (
                self.audio_config["norm_std"]
            )
        # skip normalization the input ONLY when you are trying to get the normalization stats.

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank.unsqueeze(dim=0)

    @classmethod
    def create_audiomae(
        cls,
        target_length=1024,
        drop_path_rate=0.1,
        ckpt_path=None,
        precision="fp16",
        device=torch.device("cpu"),
    ):
        audiomae = AudioMAE(
            model_name="vit_base_patch16",
            hidden_size=768,
            num_classes=527,
            drop_path_rate=drop_path_rate,
            global_pool=True,
            mask_2d=True,
            target_length=target_length,
            use_custom_patch=False,
            device=device,
        )

        if ckpt_path is not None:
            msg = audiomae.from_pretrained(ckpt_path)
            print(f"Load from checkpoint: {msg}.")

        if precision == "fp16":
            convert_weights_to_fp16(audiomae.model)

        return audiomae


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

    model.apply(_convert_weights_to_fp16)
