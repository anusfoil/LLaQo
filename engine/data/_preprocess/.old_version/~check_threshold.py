import os
import sys
import torch
from torch import Tensor
from tqdm import tqdm
import time

sys.path.append("..")
from dataloader import AVDataset

sys.path.append("../../../src")
from laion_clap import CLAPWrapper
from utils import write_json, read_json

import os
import yaml
import logging
import torch
import torchaudio
import pandas as pd
from glob import glob
from random import randint
from typing import Tuple, List, Callable, Union, Literal
from torch import Tensor
from torch.nn.functional import one_hot
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from abc import ABC, abstractmethod
import sys


def set_logger(name, level='debug', log_dir=None):
    """ Create logger to record result and output to console and file at the same time."""
    if level == 'debug':
        security = logging.DEBUG
    elif level == 'info':
        security = logging.INFO
    elif level == 'warning':
        security = logging.WARNING
    elif level == 'error':
        security = logging.ERROR
    # Initialise a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # set logger to the lowest level
    formatter = logging.Formatter(
        fmt=
        '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%b.%d,%Y-%H:%M:%S')
    # Set console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(security)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Set file handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir, f"{time.strftime('%b-%d_%H-%M', time.localtime())}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(security)
        logger.addHandler(file_handler)
        logger.info(f"The log is writen to the file: {log_dir}.")
    return logger


log = set_logger(__name__, log_dir="./log")


def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        res = yaml.safe_load(f)
    return res


class _DataSet(ABC):
    r"""Template for dataset class"""

    def __len__(self):
        return len(self.indices)

    def _load_audio(self,
                    audio_dir: str,
                    file_names: list,
                    resample: bool = True) -> list:
        r"""Return a list of torch.Tensor, shape = (n_file, n_temporal_step).\n
        when pass a single audio, e.g., 'audio = _load_audio(`audio_dir`, [`file_names`])[0]'.
        """
        audios = list()
        for fname in file_names:
            wav, sr = torchaudio.load(os.path.join(audio_dir, fname))
            if resample and (sr != self.audio_sample_rate):
                wav = torchaudio.functional.resample(
                    wav, orig_freq=sr, new_freq=self.audio_sample_rate)
            audios.append(wav.squeeze())
        return audios

    def _transform(self, x: List[Tensor],
                   process_fn: List[callable]) -> List[Tensor]:
        r"""Transform `x` with a series of `process_fn`. 
        Return a list of torch.Tensor."""
        for fn in process_fn:
            x = fn(x)
        return x

    def _process_audio(self, audios: list, process_fn: List[callable]) -> list:
        r"""Return a list of torch.Tensor, shape = (n_samples, n_channels=1, n_mels, n_frames).\n
            This is similar to `_transform()` but faster.
        """
        audios = torch.stack(audios, dim=0)
        for fn in process_fn:
            audios = fn(audios)
        # Split tensors and get a copy for each audio
        return [wav for wav in audios.split(1, dim=0)]


class ESC50(_DataSet):
    r"""ESC-50 dataset."""

    def __init__(
        self,
        audio_dir: str,
        csv_path: str,
        *,
        output_fmt: list = [
            'file_name', 'waveform', 'log_mel', 'class_id', 'onehot',
            'class_category'
        ],  # overall = ('file_name', 'waveform', 'log_mel', 'class_id', 'onehot', 'class_category')
        fold: list = [1, 2, 3, 4, 5],
        cfg_path: str = '../config/esc50_config.yaml',
        wav_transform: List[Callable] = [],
        spec_transform: List[Callable] = [],
    ):
        self.audio_dir = audio_dir
        self._detokeniser = self._create_detokeniser(csv_path)
        self.num_class = len(self.labelset)

        self.meta = self._load_meta(csv_path, fold)  # meta: filename -> onehot
        self.indices = list(self.meta.keys())  # indices: -> filename

        # Audio setting
        fmt = [
            'file_name', 'waveform', 'log_mel', 'class_id', 'onehot',
            'class_category'
        ]
        fmt_index = dict(zip(fmt,
                             list(range(len(fmt)))))  # dict: `format` -> `idx`
        self.pos = [fmt_index[f] for f in output_fmt]

        cfgs = read_yaml(cfg_path)
        self.label_type = cfgs["mode"]
        mel_scale = MelSpectrogram(cfgs['sample_rate'],
                                   cfgs['n_fft'],
                                   cfgs['win_length'],
                                   cfgs['hop_length'],
                                   cfgs['f_min'],
                                   cfgs['f_max'],
                                   n_mels=cfgs['n_mel'],
                                   window_fn=torch.hann_window)
        power_to_db = AmplitudeToDB(stype='power')
        self.log_mel_fn = [mel_scale, power_to_db]
        self.audio_sample_rate = cfgs['sample_rate']
        # Data augment
        self.wav_transform, self.mixup = wav_transform, None
        if self.wav_transform:
            fn_names = list()
            for id, item in enumerate(self.wav_transform):
                fn_name = item.__class__.__name__
                if fn_name != "PartialMixup":
                    fn_names.append(fn_name)
                else:
                    assert set(output_fmt) <= {
                        'waveform', 'log_mel', 'onehot'
                    }, "Mixup cannot support outputs except 'waveform', 'log_mel', 'onehot'."
                    self.mixup = self.wav_transform.pop(id)
                    self.label_type = "multi_label"
                    log.info(f"Use mixup.")
            log.info(f"Augment waveforms using {fn_names}.")

        self.spec_transform = spec_transform
        if self.spec_transform:
            fn_names = [
                item.__class__.__name__ for item in self.spec_transform
            ]
            log.info(f"Augment spectrograms using {fn_names}.")

    def __getitem__(
        self, item: Union[int,
                          Tuple[str,
                                int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns a tuple = (file_name, waveform, log_mel, class_id, onehot, class_category)."""
        try:
            file_name, onehot = item
        except TypeError:
            file_name = self.indices[item]
            onehot = self.meta[file_name]
        # Ground truth curation
        class_id = self.detokenise(onehot, tgt_type="class_id")
        class_category = self.detokenise(onehot, tgt_type="category")
        # Waveform curation
        waveform = self._load_audio(self.audio_dir, [file_name],
                                    resample=False)
        if self.wav_transform:
            waveform = self._transform(waveform, self.wav_transform)

        if self.mixup:
            sec_fname = self.indices[randint(0, len(self.indices) - 1)]
            sec_wav = self._load_audio(self.audio_dir, [sec_fname],
                                       resample=False)
            sec_onehot = self.meta[sec_fname]
            waveform, onehot = self.mixup(
                [*waveform, *sec_wav],
                [onehot.float(), sec_onehot.float()])
            onehot = onehot[0]
        # Log mel curation
        log_mel = self._process_audio(waveform, self.log_mel_fn)
        if self.spec_transform:
            log_mel = self._transform(log_mel, self.spec_transform)

        output = (file_name, waveform[0], log_mel[0], class_id, onehot.float(),
                  class_category)
        return [output[id] for id in self.pos]

    def _load_meta(self, csv_path: str, fold: list) -> dict:
        r"""Load meta information.
        Args:
            csv_path: str, path to `esc50.csv`.
            fold: list, fold id(s) needed for train/val dataset.
        Returns:
            dict: filename -> onehot.
        Note that `onehot` is a LongTensor.
        """
        meta = dict()
        df = pd.read_csv(
            csv_path
        )  # `esc50.csv` format: ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
        for _, r in df.iterrows():
            if int(r['fold']) in fold:
                meta[r['filename']] = one_hot(
                    torch.tensor(int(r['target'])), num_classes=self.num_class
                )  # convert to int idx from str type from csv file
        return meta

    def _create_detokeniser(self, csv_path: str) -> dict:
        r"""Returns a dict: class_id -> class_category."""
        detokeniser = dict()
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            t = int(r['target'])
            if t not in detokeniser.keys():
                detokeniser[t] = r['category']
            else:
                assert detokeniser[t] == r['category']
        return detokeniser

    def tokenise(
        self,
        input: Union[str, list],
        tgt_type: Literal["class_id",
                          "onehot"] = "onehot") -> Union[Tensor, List]:
        r"""Tokenise ONE class_category to either `class_id` or `onehot`."""

        def _tokenise(cat, tokeniser, tgt_type):
            class_id = tokeniser[cat]
            if tgt_type == "class_id":
                return class_id
            else:
                return one_hot(torch.tensor(class_id),
                               num_classes=self.num_class)

        tokeniser = {
            class_category: class_id
            for class_id, class_category in self._detokeniser.items()
        }

        if isinstance(input, list):
            res = []
            for item in input:
                res.append(_tokenise(item, tokeniser, tgt_type))
            return res
        else:
            return _tokenise(input, tokeniser, tgt_type)

    def detokenise(self,
                   input: Union[Tensor, list],
                   tgt_type: str = Literal["class_id", "category"]) -> dict:
        r"""Detokenise ONE `onehot` used in 'meta' to either `class_id` or `category`."""

        def _detokenise(onehot, detokeniser, tgt_type):
            class_id = onehot.nonzero()[0].item()
            if tgt_type == "class_id":
                return class_id
            else:
                return detokeniser[class_id]

        if isinstance(input, list):
            res = []
            for item in input:
                res.append(_detokenise(item, self._detokeniser, tgt_type))
            return res
        else:
            return _detokenise(input, self._detokeniser, tgt_type)

    @property
    def labelset(self):
        return list(set(list(self._detokeniser.values())))


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


def sort_by_weights(input: list, weights: Tensor) -> list:
    scores, sorted_idx = torch.sort(weights, descending=True)

    res = []
    for idx in sorted_idx.tolist():
        res.append(input[idx])

    return scores.tolist(), res


def main(json_path, weights_path, meta_dir, mini_data, use_cuda):
    clap = CLAPWrapper(weights_path,
                       enable_fusion=False,
                       use_cuda=use_cuda,
                       sampling_rate=48000)
    audio_dir = "/data/EECS-MachineListeningLab/datasets/ESC-50/audio"
    dataset = ESC50(
        audio_dir=audio_dir,
        csv_path=
        "/data/EECS-MachineListeningLab/datasets/ESC-50/meta/esc50.csv",
        fold=[1, 2, 3, 4, 5],
        cfg_path=
        '/data/home/eey340/WORKPLACE/class_dropout/config/esc50_config.yaml',
        output_fmt=['file_name', 'class_category'],
    )
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             num_workers=4,
                                             pin_memory=True)

    total_sim = 0
    cnt = 0
    for ids, (fname, cls) in tqdm(enumerate(dataloader)):
        if mini_data and ids > 0:
            break
        wav_path = [os.path.join(audio_dir, fname[0])]
        # datum = read_json(json_path)
        caps = [
            "The image contains " + cls[0]
        ]  # This is a sound of : 5.461989738583565, The image contains: 3.300266095210332
        sim = clap.extract_feature_and_calculate_similarity(
            wav_path,
            caps,
            enable_softmax=False,
            resample=True,
        ).detach().squeeze().item()  #shape = (1, num_caps)
        # score, sorted_caps = sort_by_weights(caps, sim)

        # caps_and_scores = [(cap, score[id])
        #                    for id, cap in enumerate(sorted_caps)]
        total_sim += sim
        cnt += 1
        print(f"{caps[0]}: {sim}")

    log.info(f"Average similarity: {total_sim/cnt}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str)
    parser.add_argument('--meta_dir', type=str)
    parser.add_argument('--json_path', type=str)
    parser.add_argument("--mini_data", action="store_true", default=False)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    args = parser.parse_args()

    main(
        json_path=args.json_path,
        weights_path=args.weights_path,
        meta_dir=args.meta_dir,
        use_cuda=args.use_cuda,
        mini_data=args.mini_data,
    )
