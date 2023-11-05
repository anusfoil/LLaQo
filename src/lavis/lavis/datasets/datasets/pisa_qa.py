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


def transform_PISA_dataset():
    """PISA is a dataset with student performance of various difficulty levels. 
    The 
    
    """
    meta_csv = pd.read_csv('/data/EECS-MachineListeningLab/datasets/LLaQo/PISA/Annotations_v1.csv')
    for i in range(12, 29):
        meta_csv.drop(columns=f'Unnamed: {i}', inplace=True)

    # divide into train and test split
    meta_csv['split'] = 'train'
    test_idx = [randint(0, 59) for _ in range(10)]
    meta_csv.loc[test_idx, 'split'] = 'test'

    # populate QA pairs
    qa_csv = []
    for idx, row in meta_csv.iterrows():
        row['Q'] = "Which difficulty level is the piece, in a scale of 9?"
        row['A'] = str(row['level_song'])
        qa_csv.append(copy.deepcopy(row))
        row['Q'] = "Which skill level is the performer in, in a scale of 9?"
        row['A'] = str(row['level_player'])
        qa_csv.append(copy.deepcopy(row))
        row['Q'] = "How would you rate the performance, in a scale of 5?"
        row['A'] = "4" # PISA performances are all clean and accurate
        qa_csv.append(copy.deepcopy(row))
        row['Q'] = "What kind of performance might this be?"
        row['A'] = "This is a examplary student's performance, and everything is executed correctly."
        qa_csv.append(copy.deepcopy(row))
        if not pd.isna(row['piece_description']):
            row['Q'] = "What is the stylistic period of the piece? Who is the potential composer?"
            row['A'] = row['piece_description']
            qa_csv.append(copy.deepcopy(row))           

    qa_csv = pd.DataFrame(qa_csv)
    qa_csv.to_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/PISA/Annotations_v2.csv")


ANSWERS_CSV = '/data/EECS-MachineListeningLab/datasets/LLaQo/PISA/Annotations_v2.csv'
AUDIO_DIR = '/data/EECS-MachineListeningLab/datasets/LLaQo/PISA/processed_samples_audio'

class PISADataset(Dataset):
    """PISA dataset."""

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

        self.audio_processor = audio_processor

    def __len__(self):
        return len(self.audio_qa)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = os.path.join(self.audio_dir,
                                str(int(self.audio_qa['filename'].iloc[idx]))) + ".wav"

        sample = {
                'audio_path': audio_path, 
                'question': self.audio_qa['Q'].iloc[idx],
                'answer': self.audio_qa['A'].iloc[idx]}
        
        sample["waveform"], sample["fbank"] = self.audio_processor(audio_path)[:-1]

        if self.transform:
            sample = self.transform(sample)

        return sample


class PISADatasetQA(BaseDataset):
    def __init__(self, vis_processor, audio_root, seg_name, **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=audio_root)

        self.inner_dataset = PISADataset(ANSWERS_CSV, AUDIO_DIR)

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
    transform_PISA_dataset()

    dataset = PISADatasetQA(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="all_train",
    )
    print(next(iter(dataset)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    for datum in loader:
        print(datum)
        hook()
