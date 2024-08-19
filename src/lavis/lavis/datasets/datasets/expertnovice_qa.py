"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os, sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from collections import OrderedDict
from random import randint
sys.path.append(os.path.dirname(__file__))
from audio_processor import fbankProcessor
import hook

try:
    from lavis.datasets.datasets.base_dataset import BaseDataset

    # from lavis.datasets.datasets.prompt_template import _QUESTION4CLASSIFICATION_ as _QUESTION_
except:
    from base_dataset import BaseDataset

    # from prompt_template import _QUESTION4CLASSIFICATION_ as _QUESTION_


ANSWERS_CSV = '/data/EECS-MachineListeningLab/datasets/LLaQo/expert_novice/evaluation_qa.csv'
AUDIO_DIR = '/data/EECS-MachineListeningLab/datasets/LLaQo/expert_novice/recordings_and_alignments'

class ExpertNoviceDataset(Dataset):
    """Expert Novice dataset."""

    def __init__(self, answers_csv=ANSWERS_CSV, audio_dir=AUDIO_DIR, transform=None,
                 audio_processor=fbankProcessor.build_processor(),
                 split='train'):
        """
        Arguments:
            answers_csv (string): Path to the csv file with con espressione game answer.
            audio_dir (string): Directory with all the audios.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.audio_qa = pd.read_csv(answers_csv)
        self.audio_qa = self.audio_qa[self.audio_qa['split'] == split]
        self.audio_dir = audio_dir
        self.transform = transform

        self.audio_qa['audio_path'] = self.audio_qa['piece_name'] + "-" + self.audio_qa['recording_number'].apply(lambda x: str(x).zfill(2))
        self.audio_qa['audio_path'] = self.audio_qa['piece_name'].apply(lambda x: "".join(x.split())) + "/" + self.audio_qa['audio_path']

        self.audio_processor = audio_processor

    def __len__(self):
        return len(self.audio_qa)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = os.path.join(self.audio_dir,
                                self.audio_qa['audio_path'].iloc[idx]) + ".wav"
        # audio_path = os.path.join(self._audio_dir,
        #                         self.audio_answers['audio_path'].iloc[idx]) + ".wav"

        sample = {
                'audio_path': audio_path, 
                'question': self.audio_qa['Q'].iloc[idx],
                'answer': self.audio_qa['A'].iloc[idx]}
                # 'answer': self.audio_answers['answer'].iloc[idx]}
        
        sample["waveform"], sample["fbank"] = self.audio_processor(audio_path)[:-1]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ExpertNoviceDatasetQA(BaseDataset):
    def __init__(self, vis_processor, audio_root, seg_name, **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=audio_root)

        self.inner_dataset = ExpertNoviceDataset(ANSWERS_CSV, AUDIO_DIR)

        self._add_instance_ids()

    def __len__(self):
        # return 100
        return len(self.inner_dataset)

    def __getitem__(self, index):
        datum = self.inner_dataset[index]

        return {
            'audio_path': datum['audio_path'], 
            "audio": datum["fbank"],
            "text_input": datum["question"],
            "text_output": datum["answer"],
        }

    def displ_item(self, index):
        datum = self.inner_dataset[index]

        return {
            'audio_path': datum['audio_path'], 
            "audio": datum["fbank"],
            "text_input": datum["question"],
            "text_output": datum["answer"],
        }


if __name__ == "__main__":
    import torch

    dataset = ExpertNoviceDatasetQA(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="all_train",
    )
    print(next(iter(dataset)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    for datum in loader:
        print(datum)
        hook()
