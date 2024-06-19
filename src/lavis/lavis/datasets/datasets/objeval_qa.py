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
from random import randint
import copy
sys.path.append(os.path.dirname(__file__))
from audio_processor import fbankProcessor
import hook

try:
    from lavis.datasets.datasets.base_dataset import BaseDataset

except:
    from base_dataset import BaseDataset


QMAP = {
    "Is the legato even?": "How would you rate if the legato is even, on a scale of 0 to 6?",
    "Are the note values uniform?": "How would you rate the uniformity of the note values, on a scale of 0 to 6?",
    "How solid is the sound?": "How would you rate the solidity of the sound, on a scale of 0 to 6?",
    "How clean is the attack?": "How would you rate the cleanliness of the attack, on a scale of 0 to 6?",
    "Are the left and right hands balanced?": "How would you rate the balance between the left and right hands, on a scale of 0 to 6?",
    "Are the timings aligned on the left and right hands?": "How would you rate the alignment of timings between the left and right hands, on a scale of 0 to 6?",
    "Is it played with the correct rhythm?": "How would you rate the correctness of the rhythm, on a scale of 0 to 6?",
    "Is the tempo kept constant?": "How would you rate the consistency of the tempo, on a scale of 0 to 6?",
    "Are the lines connected?": "How would you rate the connectivity of the lines, on a scale of 0 to 6?",
    "Is it played with a sense of tonality?": "How would you rate the sense of tonality, on a scale of 0 to 6?",
    "Is the dynamics change natural?": "How would you rate the naturalness of the dynamics change, on a scale of 0 to 6?"
}

QCA = {
    "Is the legato even?": "coordination",
    "Are the timings aligned on the left and right hands?": "coordination",
    "Are the note values uniform?": "coordination",
    "Are the lines connected?": "articulation",
    "How clean is the attack?": "articulation",
    "Is it played with the correct rhythm?": "rhythm and tempo",
    "Is the tempo kept constant?": "rhythm and tempo",
    "Is it played with a sense of tonality?": "tone production",
    "How solid is the sound?": "tone production",
    "Is the dynamics change natural?": "dynamics",
    "Are the left and right hands balanced?": "dynamics",
}


def transform_Objeval_dataset():
    """Objective eval dataset, with standardized audio and teacher's rating on each dimension.
    """
    qa_csv = []
    ratings = pd.read_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/objeval/qa/sub_001.csv")
    wav_paths = glob.glob("/data/EECS-MachineListeningLab/datasets/LLaQo/objeval/songs/**/*.wav", recursive=True)

    for idx, row in ratings.iterrows():
        audio_path = row['fname']
        row['audio_path'] = [wp for wp in wav_paths if audio_path in wp][0]
        
        row['question_id'] = row['question_source_id']
        if row['question_source_id'] in [1, 2]:
            row['Q'] = "How would you rate the overall performance, on a scale of 0 to 6?"
            row['A'] = str(row['score'])
            row['question_category'] = 'summary'
            qa_csv.append(copy.deepcopy(row))
        else:
            row['Q'] = QMAP[row['quesition']]
            row['A'] = str(row['score'])
            row['question_category'] = QCA[row['quesition']]
            qa_csv.append(copy.deepcopy(row))

    qa_csv = pd.DataFrame(qa_csv)
    qa_csv.to_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/objeval/audio_qa.csv")


ANSWERS_CSV = '/data/EECS-MachineListeningLab/datasets/LLaQo/objeval/audio_qa.csv'


class ObjevalDataset(Dataset):
    """ dataset."""

    def __init__(self, answers_csv=ANSWERS_CSV, transform=None,
                 audio_processor=fbankProcessor.build_processor()):
        """
        Arguments:
            answers_csv (string): Path to the csv file with con espressione game answer.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.audio_qa = pd.read_csv(answers_csv)
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
                'answer': self.audio_qa['A'].iloc[idx],
                'qcategory': self.audio_qa['question_category'].iloc[idx],
                'qidx': self.audio_qa['question_id'].iloc[idx]}
        
        sample["waveform"], sample["fbank"] = self.audio_processor(audio_path)[:-1]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ObjevalDatasetQA(BaseDataset):
    def __init__(self, vis_processor, audio_root, seg_name, **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=audio_root)

        self.inner_dataset = ObjevalDataset(ANSWERS_CSV)

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
    transform_Objeval_dataset()
    hook()

    dataset = ObjevalDatasetQA(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="all_train",
    )
    print(next(iter(dataset)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    for datum in loader:
        print(datum)
        # hook()
