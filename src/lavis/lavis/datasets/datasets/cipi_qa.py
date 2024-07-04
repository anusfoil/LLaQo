"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os, sys, math, glob
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from collections import OrderedDict
from random import randint, choice
import copy
sys.path.append(os.path.dirname(__file__))
from audio_processor import fbankProcessor
import hook

try:
    from lavis.datasets.datasets.base_dataset import BaseDataset

except:
    from base_dataset import BaseDataset



def transform_CIPI_dataset():
    """ CIPI (Can I play it?) dataset for difficulty annotation
    """
    qa_csv = []
    
    metadata = pd.read_csv("/data/scratch/acw630/difficulty_cipi/CIPI_youtube_links.csv")
    test_idx = [randint(0, 730) for _ in range(110)]
    for idx, row in metadata.iterrows():

        row['audio_path'] = "/data/scratch/acw630/difficulty_cipi/" + row['audio_path']
        row['split'] = 'test' if idx in test_idx else 'train'
        
        row['Q'] = choice([
            "Who might be the composer of this piece?",
            "Who is likely the composer of the music being played?",
            "Could you guess the composer of this piece?",
            "Can you identify the composer of this piece?"
        ])
        row['A'] = row['Composer']
        qa_csv.append(copy.deepcopy(row))
        
        row['Q'] = choice([
            "how would you rate the difficulty of the piece played in this track, on a scale of 1 to 9?"
            "On a scale from 1 to 9, how challenging is the piece featured in this track?",
            "What difficulty level would you assign to the composition in this track, on a scale of 1 to 9?",
            "Rate the complexity of the piece in this track on a scale of 1 to 9."
        ])
        row['A'] = str(row['henle'])
        qa_csv.append(copy.deepcopy(row))        

        row['Q'] = choice([
            "Does this sound like a performance by a student or a master?",
            "Would you say this performance sounds more like it's by a student or a master?",
            "From the sound of it, is this a student or master level performance?",
        ])
        row['A'] = choice([
            "This is a master performance. It's mistake-free and has a high level of musicality.",
            "This performance is of a master level, free from errors and rich in musical expression.",
            "This appeared to be performed by really advanced player. Possibly a pianist."
        ])
        qa_csv.append(copy.deepcopy(row))    


    qa_csv = pd.DataFrame(qa_csv)
    qa_csv.to_csv("/data/scratch/acw630/difficulty_cipi/audio_qa.csv")


ANSWERS_CSV = '/data/scratch/acw630/difficulty_cipi/audio_qa.csv'


class CIPIDataset(Dataset):
    """ dataset."""

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
                'answer': str(self.audio_qa['A'].iloc[idx])}
        
        sample["waveform"], sample["fbank"] = self.audio_processor(audio_path)[:-1]

        if self.transform:
            sample = self.transform(sample)

        return sample


class CIPIDatasetQA(BaseDataset):
    def __init__(self, vis_processor, audio_root, seg_name, **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=audio_root)

        self.inner_dataset = CIPIDataset(ANSWERS_CSV)

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
    transform_CIPI_dataset()
    hook()

    dataset = CIPIDatasetQA(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="all_train",
    )
    print(next(iter(dataset)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    for datum in loader:
        print(datum)
        # hook()
