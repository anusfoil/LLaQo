"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os, sys, math
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from collections import OrderedDict
from random import randint
import copy
sys.path.append(os.path.dirname(__file__))
from audio_processor import fbankProcessor
import hook

try:
    from lavis.datasets.datasets.base_dataset import BaseDataset

    # from lavis.datasets.datasets.prompt_template import _QUESTION4CLASSIFICATION_ as _QUESTION_
except:
    from base_dataset import BaseDataset

    # from prompt_template import _QUESTION4CLASSIFICATION_ as _QUESTION_


def transform_YCUPPE_dataset():
    """YCUPPE is a dataset with student performances of entry levels, with different teacher's rating in the scale of 100. 
    
    """
    qa_csv = []
    test_idx = [randint(0, 150) for _ in range(15)]
    for i in range(1, 14):
        piece_dir = f"/data/EECS-MachineListeningLab/datasets/LLaQo/YCU-PPE-III/raw/{i}"
        score_csv = pd.read_csv(f"{piece_dir}/score.csv", names=['name', 's1', 's2', 's3'])
        for idx, row in score_csv.iterrows():

            row['split'] = 'test' if idx in test_idx else 'train'
            row['audio_path'] = f"{piece_dir}/wav/{row['name']}.wav"
            row['rating'] = math.ceil((row['s1'] + row['s2'] + row['s3']) / 30) / 2 # from 0 - 5, round to 0.5
            
            row['Q'] = "How would you rate the performance, in a scale of 5?"
            row['A'] = str(row['rating'])
            qa_csv.append(copy.deepcopy(row))
            row['Q'] = "What kind of performance might this be?"
            row['A'] = "This is a student's performance."
            qa_csv.append(copy.deepcopy(row))
            row['Q'] = "Which difficulty level is the piece, in a scale of 9?"
            row['A'] = "3"
            qa_csv.append(copy.deepcopy(row))

    qa_csv = pd.DataFrame(qa_csv)
    qa_csv.to_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/YCU-PPE-III/audio_qa.csv")


ANSWERS_CSV = '/data/EECS-MachineListeningLab/datasets/LLaQo/YCU-PPE-III/audio_qa.csv'


class YCUPPEDataset(Dataset):
    """PISA dataset."""

    def __init__(self, answers_csv=ANSWERS_CSV, transform=None,
                 audio_processor=fbankProcessor.build_processor(),
                 split='train'):
        """
        Arguments:
            answers_csv (string): Path to the csv file with con espressione game answer.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.audio_qa = pd.read_csv(answers_csv)
        self.audio_qa = self.audio_qa[self.audio_qa['split'] == split]
        self.transform = transform

        self.audio_processor = audio_processor

    def __len__(self):
        return len(self.audio_qa)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = self.audio_qa['audio_path'].iloc[idx]

        sample = {
                'audio_path': audio_path, 
                'question': self.audio_qa['Q'].iloc[idx],
                'answer': self.audio_qa['A'].iloc[idx]}
        
        sample["waveform"], sample["fbank"] = self.audio_processor(audio_path)[:-1]

        if self.transform:
            sample = self.transform(sample)

        return sample


class YCUPPEDatasetQA(BaseDataset):
    def __init__(self, vis_processor, audio_root, seg_name, **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=audio_root)

        self.inner_dataset = YCUPPEDataset(ANSWERS_CSV)

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
    transform_YCUPPE_dataset()
    hook()

    dataset = YCUPPEDatasetQA(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="all_train",
    )
    print(next(iter(dataset)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    for datum in loader:
        print(datum)
        # hook()
