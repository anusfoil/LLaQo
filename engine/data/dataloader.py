# -*- coding: utf-8 -*-
# modified from: Yuan Gong

import os
import csv
import json
import math
import logging
import os.path
import torchaudio
import torch
import PIL
import torch.nn.functional
import numpy as np
import webdataset as wds
from glob import glob
from torch.utils.data import Dataset
from functools import partial
import random
import torchvision.transforms as T
from PIL import Image
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(FILE_DIR, ".."))
from factory import count_samples


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup


def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup


def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list


def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


class AVDataset(Dataset):

    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.data = self.pro_data(self.data)
        self.prefix = data_json['prefix']
        n_frames = data_json['n_frames']
        self.is_supervised = True if label_csv else False
        print('Dataset has {:d} samples'.format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        if self.is_supervised:
            self.label_smooth = self.audio_conf.get('label_smooth', 0.0)
            print('Using Label Smoothing: ' + str(self.label_smooth))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(
            self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = 0  # self.audio_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get(
            'skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print(
                'now skip normalization (use it ONLY when you are computing the normalization stats).'
            )
        else:
            print(
                'use dataset mean {:.3f} and std {:.3f} to normalize the input.'
                .format(self.norm_mean, self.norm_std))

        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')

        if self.is_supervised:
            self.index_dict = make_index_dict(label_csv)
            self.label_num = len(self.index_dict)
            print('number of classes is {:d}'.format(self.label_num))

        self.target_length = self.audio_conf.get('target_length')

        # train or eval
        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get('frame_use', -1)
        # by default, 10 frames are used
        self.total_frame = n_frames  #self.audio_conf.get('total_frame', 10)
        # print('now use frame {:d} from total {:d} frames'.format(
        #     self.frame_use, self.total_frame))
        print(f"Now use {self.total_frame} frames for each video")

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(
            self.im_res, self.im_res))
        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(mean=[0.4850, 0.4560, 0.4060],
                        std=[0.2290, 0.2240, 0.2250])
        ])

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [
                data_json[i]['filename'], data_json[i]['audio_path'],
                data_json[i]['frame_path']
            ]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum['filename'] = np_data[0]
        datum['audio_path'] = np_data[1]
        datum['frame_path'] = np_data[2]
        return datum

    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            image_tensor = self.preprocess(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (
                1 - mix_lambda) * image_tensor2
            return image_tensor

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
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 -
                                                     mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

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
            print('there is a loading error')

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def read_frames(self, frame_path):
        return [
            self.get_image(os.path.join(frame_path.format(fid=fid)))
            for fid in range(self.total_frame)
        ]

    def randselect_img(self, filename, video_path):
        if self.mode == 'eval':
            # if not specified, use the middle frame
            if self.frame_use == -1:
                frame_idx = int((self.total_frame) / 2)
            else:
                frame_idx = self.frame_use
        else:
            frame_idx = random.randint(0, 9)

        while os.path.exists(video_path + '/frame_' + str(frame_idx) + '/' +
                             filename + '.jpg') == False and frame_idx >= 1:
            print('frame {:s} {:d} does not exist'.format(filename, frame_idx))
            frame_idx -= 1
        out_path = video_path + '/frame_' + str(
            frame_idx) + '/' + filename + '.jpg'
        #print(out_path)
        return out_path

    def __getitem__(self, index):
        if random.random() < self.mixup:
            datum = self.data[index]
            datum = self.decode_data(datum)
            mix_sample_idx = random.randint(0, self.num_samples - 1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)
            try:
                fbank = self._wav2fbank(datum['wav'], mix_datum['wav'],
                                        mix_lambda)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio')
            try:
                image = self.get_image(
                    self.randselect_img(datum['filename'],
                                        datum['video_path']),
                    self.randselect_img(mix_datum['filename'],
                                        datum['video_path']), mix_lambda)
            except:
                image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
                print('there is an error in loading image')

            if self.is_supervised:
                label_indices = np.zeros(
                    self.label_num) + (self.label_smooth / self.label_num)
                for label_str in datum['labels'].split(','):
                    label_indices[int(
                        self.index_dict[label_str])] += mix_lambda * (
                            1.0 - self.label_smooth)
                for label_str in mix_datum['labels'].split(','):
                    label_indices[int(self.index_dict[label_str])] += (
                        1.0 - mix_lambda) * (1.0 - self.label_smooth)
                label_indices = torch.FloatTensor(label_indices)

        else:
            datum = self.data[index]
            datum = self.decode_data(datum)
            try:
                fbank = self._wav2fbank(datum['audio_path'], None, 0)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio')

            try:
                frames = self.read_frames(datum["frame_path"])
                frames = torch.stack(frames)
            except:
                frames = torch.zeros(
                    [self.total_frame, 3, self.im_res, self.im_res]) + 0.01
                print('there is an error in loading frames')

            if self.is_supervised:
                # label smooth for negative samples, epsilon/label_num
                label_indices = np.zeros(
                    self.label_num) + (self.label_smooth / self.label_num)
                for label_str in datum['labels'].split(','):
                    label_indices[int(
                        self.index_dict[label_str])] = 1.0 - self.label_smooth
                label_indices = torch.FloatTensor(label_indices)

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

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0],
                                       fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(
                fbank,
                np.random.randint(-self.target_length, self.target_length), 0)

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return datum, fbank.unsqueeze(dim=0), frames

    def __len__(self):
        return self.num_samples


def get_wds_dataset(
    dataset_dir,
    is_train,
    args,
    audio_ext="audio.feature.pyd",
    visual_ext="video.feature.pyd",
    meta_ext="meta.json",
    _SHARD_SHUFFLE_SIZE=2000,
    _SHARD_SHUFFLE_INITIAL=500,
    _SAMPLE_SHUFFLE_SIZE=5000,
    _SAMPLE_SHUFFLE_INITIAL=1000,
):
    input_shards = glob(os.path.join(dataset_dir, "*.tar"))

    num_samples = count_samples(input_shards)

    pipeline = [wds.SimpleShardList(input_shards)]
    # at this point we have an iterator over all the shards
    if is_train or args.parallel_eval:
        pipeline.extend([
            wds.detshuffle(
                bufsize=_SHARD_SHUFFLE_SIZE,
                initial=_SHARD_SHUFFLE_INITIAL,
                seed=args.seed,
            ),
            wds.split_by_node,
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker at each node
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
                rng=random.Random(args.seed),
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    pipeline.extend([
        wds.decode(),
        wds.to_tuple(meta_ext, audio_ext, visual_ext),
        wds.batched(args.batch_size)
    ])

    # pipeline.append(
    #     wds.batched(
    #         args.batch_size,
    #         partial=not (is_train or args.parallel_eval),
    #         collation_fn=partial(collate_fn_with_preprocess,
    #                              audio_ext=audio_ext,
    #                              text_ext=text_ext,
    #                              max_len=max_len,
    #                              audio_cfg=model_cfg['audio_cfg'],
    #                              args=args,
    #                              ),

    #     )
    # )

    dataset = wds.DataPipeline(*pipeline)
    if is_train or args.parallel_eval:
        # roll over and repeat a few samples to get same number of full batches on each node
        global_batch_size = args.batch_size * args.world_size
        num_batches = math.ceil(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = math.ceil(num_batches /
                                       num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(
            num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    kwargs = {}
    if args.horovod:  # multi-node training on summit
        kwargs["multiprocessing_context"] = "forkserver"

    if is_train:
        if args.prefetch_factor:
            prefetch_factor = args.prefetch_factor
        else:
            prefetch_factor = max(2, args.batch_size // args.workers)
    else:
        prefetch_factor = 2

    dataloader = wds.WebLoader(dataset,
                               batch_size=None,
                               shuffle=False,
                               num_workers=args.workers,
                               pin_memory=True,
                               prefetch_factor=prefetch_factor,
                               **kwargs)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return dataloader


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path")
    parser.add_argument("--tar_path")
    args = parser.parse_args()

    # im_res = 224
    # audio_conf = {
    #     'num_mel_bins': 128,
    #     'target_length': 1024,
    #     'freqm': 0,
    #     'timem': 0,
    #     'mixup': 0.0,
    #     'dataset': "audioset",
    #     'mode': 'train',
    #     'mean': -5.081,
    #     'std': 4.4849,
    #     'noise': True,
    #     'label_smooth': 0,
    #     'im_res': im_res
    # }
    # val_audio_conf = {
    #     'num_mel_bins': 128,
    #     'target_length': 1024,
    #     'freqm': 0,
    #     'timem': 0,
    #     'mixup': 0,
    #     'dataset': "audioset",
    #     'mode': 'eval',
    #     'mean': -5.081,
    #     'std': 4.4849,
    #     'noise': False,
    #     'im_res': im_res
    # }

    # dataset = AVDataset(dataset_json_file=args.json_path,
    #                     audio_conf=audio_conf,
    #                     label_csv=None)

    # print(dataset[100][1][0])
    """ Test webdataset."""
    import webdataset as wds
    from torch.utils.data import DataLoader
    from itertools import islice

    dataset = (wds.WebDataset(args.tar_path).decode().to_tuple(
        "audio.feature.pyd", "video.feature.pyd", "meta.json"))
    # dataloader = DataLoader(dataset.batched(16),
    #                         num_workers=4,
    #                         batch_size=None)

    for batch_idx, batch in enumerate(dataset):
        print(batch[0].size())
        # print(batch[1][0]["query_output"])
        # print(batch[2])

        if batch_idx == 1000:
            break
