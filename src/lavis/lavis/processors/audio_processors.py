"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor


@registry.register_processor("audio_processor")
class AudioProcessor(BaseProcessor):

    def __init__(
        self,
        num_mel_bins: int,
        target_length: int,
        freqm: int,
        timem: int,
        mean: float,
        std: float,
        noise: bool,
    ):
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.freqm = freqm
        self.timem = timem
        self.mean = mean
        self.std = std
        self.noise = noise

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        num_mel_bins = cfg.get("num_mel_bins", 128)
        target_length = cfg.get("target_length", 1024)
        freqm = cfg.get("freqm", 0)
        timem = cfg.get("timem", 0)
        mean = cfg.get("mean", -5.081)
        std = cfg.get("std", 4.4849)
        noise = cfg.get("noise", False)

        return cls(
            num_mel_bins=num_mel_bins,
            target_length=target_length,
            freqm=freqm,
            timem=timem,
            mean=mean,
            std=std,
            noise=noise,
        )

    def __call__(self, audio_path):
        try:
            fbank = self._get_fbank(audio_path)
        except:
            fbank = torch.zeros([self.target_length, self.num_mel_bins]) + 0.01
            print('There is an error in loading audio.')

        return fbank

    def _get_fbank(self, audio_path):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=sr,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=self.melbins,
                dither=0.0,
                frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error.')

        # Cut and pad
        target_length = self.target_length
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # Normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)

        # Add (optional) noise during training
        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0],
                                       fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(
                fbank,
                np.random.randint(-self.target_length, self.target_length), 0)

        return fbank
