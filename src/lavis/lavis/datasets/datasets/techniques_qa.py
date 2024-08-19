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

QS = [
    "What is the most prominent piano technique employed in this track?",
    "Which piano technique stands out the most in this composition?",
    "What is the primary piano technique used in this track?",
    "Can you identify the dominant piano technique in this track?",
    "What piano technique is most noticeable?"
]

def transform_Techniques_dataset():
    """ Technique dataset with different piano techniques.
    """
    qa_csv = []
    
    metadata = pd.read_csv("/data/scratch/acw630/PianoJudge/techniques/metadata.csv")
    metadata = metadata.sample(frac=1, random_state=42)
    
    split_index = int(len(metadata) * 0.8)
    metadata['split'] = ''
    metadata['split'][:split_index] = 'train'
    metadata['split'][split_index:] = 'val'
    
    for idx, row in metadata.iterrows():

        row['audio_path'] = "/data/scratch/acw630/PianoJudge/techniques/" + row['id'] + ".wav"
        
        row['Q'] = choice(QS)
        if "|" in row['technique']:
            row['A'] = row['technique'].split("|")[0]
        else:
            row['A'] = row['technique']
        
        # remove description and comments
        row.drop(['description', 'comments'], inplace=True)
            
        qa_csv.append(copy.deepcopy(row))

    qa_csv = pd.DataFrame(qa_csv)
    qa_csv.to_csv("/data/scratch/acw630/PianoJudge/techniques/audio_qa.csv")


ANSWERS_CSV = '/data/scratch/acw630/PianoJudge/techniques/audio_qa.csv'


class TechniquesDataset(Dataset):
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
                'answer': self.audio_qa['A'].iloc[idx]}
        
        sample["waveform"], sample["fbank"] = self.audio_processor(audio_path)[:-1]

        if self.transform:
            sample = self.transform(sample)

        return sample


class TechniquesDatasetQA(BaseDataset):
    def __init__(self, vis_processor, audio_root, seg_name, **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=audio_root)

        self.inner_dataset = TechniquesDataset(ANSWERS_CSV)

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
    transform_Techniques_dataset()
    hook()

    dataset = TechniquesDatasetQA(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="all_train",
    )
    print(next(iter(dataset)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    for datum in loader:
        print(datum)
        # hook()
