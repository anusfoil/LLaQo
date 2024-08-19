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


def transform_Burgmuller_dataset():
    """Burgmuller is a dataset with student performances of intermediate Burgmuller etudes.
      The performer did not learn the pieces thouroughly, so the performance has a lot of errors.
    
    """
    qa_csv = []
    test_idx = [randint(0, 25) for _ in range(5)]
    error_notes = pd.read_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/Burgmuller/error_notes.csv")
    for idx, name in enumerate(error_notes.name.unique()):

        notes = error_notes[error_notes['name'] == name]

        row = {'audio_path': f"/data/EECS-MachineListeningLab/datasets/LLaQo/Burgmuller/{name}.wav",
               'split': 'test' if idx in test_idx else 'train'}
        
        row['Q'] = "What kind of performance might this be?"
        row['A'] = "This is a student's performance."
        qa_csv.append(copy.deepcopy(row))
        row['Q'] = "Which difficulty level is the piece, in a scale of 9?"
        row['A'] = "5"
        qa_csv.append(copy.deepcopy(row))
        row['Q'] = "What might this piece be and who is the composer?"
        row['A'] = "This is a practice piece for student, and it's from Burgmuller etude set."
        qa_csv.append(copy.deepcopy(row))

        for j in range(len(notes)):
            row['Q'] = notes.iloc[j, 1]
            row['A'] = notes.iloc[j, 2]
            qa_csv.append(copy.deepcopy(row))

    qa_csv = pd.DataFrame(qa_csv)
    qa_csv.to_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/Burgmuller/audio_qa.csv")


ANSWERS_CSV = '/data/EECS-MachineListeningLab/datasets/LLaQo/Burgmuller/audio_qa.csv'


class BurgmullerDataset(Dataset):
    """Burgmuller dataset."""

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


class BurgmullerDatasetQA(BaseDataset):
    def __init__(self, vis_processor, audio_root, seg_name, **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=audio_root)

        self.inner_dataset = BurgmullerDataset(ANSWERS_CSV)

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
    # transform_Burgmuller_dataset()
    # hook()

    dataset = BurgmullerDatasetQA(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="all_train",
    )
    print(next(iter(dataset)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    for datum in loader:
        print(datum)
        hook()
