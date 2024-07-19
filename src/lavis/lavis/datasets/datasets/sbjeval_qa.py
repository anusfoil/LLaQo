"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os, sys, math, glob
import random as rand
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from collections import OrderedDict
import copy
sys.path.append(os.path.dirname(__file__))
from audio_processor import fbankProcessor
import hook

try:
    from lavis.datasets.datasets.base_dataset import BaseDataset

except:
    from base_dataset import BaseDataset


Qs = ["Is the legato even?",
      "Are the note values uniform?",
      "How solid is the sound?",
      "How clean is the attack?",
      "Are the left and right hands balanced?",
      "Are the timings aligned on the left and right hands?",
      "Is it played with the correct rhythm?",
      "Is the tempo kept constant?",
      "Are the lines connected?",
      "Is it played with a sense of tonality?",
      "Is the dynamics change natural?"]


def transform_Sbjeval_dataset():
    """Objective eval dataset, with standardized audio and teacher's rating on each dimension.
    """
    
    objeval = pd.read_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/objeval/audio_qa.csv")
    audio_paths = objeval['audio_path'].unique()
    qa_csv = pd.DataFrame()
    
    for q in ["What's your overall feedback on this performance? ",
              "What does the student need to work on? ",
              ]:
        qa_csv = pd.concat([qa_csv, pd.DataFrame({
            'audio_path': audio_paths,
            'Q': [q] * len(audio_paths)
        })])

    for ap in audio_paths:
        qa_csv = pd.concat([qa_csv, pd.DataFrame({
            'audio_path': [ap] * 3,
            'Q': rand.sample(Qs, 3)
        })])

    epnv = pd.read_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/expert_novice/evaluation_qa.csv")
    epnv['audio_path'] = epnv['piece_name'] + "-" + epnv['recording_number'].apply(lambda x: str(x).zfill(2))
    epnv['audio_path'] = epnv['piece_name'].apply(lambda x: "".join(x.split())) + "/" + epnv['audio_path']
    AUDIO_DIR = '/data/EECS-MachineListeningLab/datasets/LLaQo/expert_novice/recordings_and_alignments'
    epnv['audio_path'] = AUDIO_DIR + "/" + epnv['audio_path'] + ".wav"
    audio_paths = epnv[epnv['split'] == 'test']['audio_path'].unique().tolist()
    for q in ["What's your overall feedback on this performance? ",
              "What does the student need to work on? ",
              "How would you describe the emotional intent of the performance? "]:
        qa_csv = pd.concat([qa_csv, pd.DataFrame({
            'audio_path': audio_paths,
            'Q': [q] * len(audio_paths)
        })])

    AUDIO_DIR = '/data/EECS-MachineListeningLab/datasets/LLaQo/crocus/Performance_Records'
    crocus = pd.read_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/crocus/evaluation_qa.csv")
    crocus['audio_path'] = AUDIO_DIR + "/" + crocus['audio_file']
    
    audio_paths = crocus[crocus['split'] == 'test']['audio_path'].unique().tolist()
    for q in ["What's your overall feedback on this performance? ",
              "What does the student need to work on? ",
              "How would you describe the emotional intent of the performance? "]:
        qa_csv = pd.concat([qa_csv, pd.DataFrame({
            'audio_path': audio_paths,
            'Q': [q] * len(audio_paths)
        })])    

    ycu = pd.read_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/YCU-PPE-III/audio_qa.csv")
    audio_paths = ycu[ycu['split'] == 'test']['audio_path'].unique()
    audio_paths = rand.sample(audio_paths.tolist(), 100)
    for q in ["What's your overall feedback on this performance? ",
              "What does the student need to work on? ",
              "How would you describe the emotional intent of the performance? "]:
        qa_csv = pd.concat([qa_csv, pd.DataFrame({
            'audio_path': audio_paths,
            'Q': [q] * len(audio_paths)
        })])   

    pisa = pd.read_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/PISA/Annotations_v2.csv")
    AUDIO_DIR = '/data/EECS-MachineListeningLab/datasets/LLaQo/PISA/processed_samples_audio/'
    pisa['audio_paths'] = AUDIO_DIR + pisa['filename'].astype(str) + ".wav"
    audio_paths = pisa[pisa['split'] == 'test']['audio_paths'].unique().tolist() 
    for q in ["What's your overall feedback on this performance? ",
              "What does the student need to work on? ",
              "How would you describe the emotional intent of the performance? "]:
        qa_csv = pd.concat([qa_csv, pd.DataFrame({
            'audio_path': audio_paths,
            'Q': [q] * len(audio_paths)
        })])   

    qa_csv.to_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/sbjeval/audio_qa.csv")


ANSWERS_CSV = '/data/EECS-MachineListeningLab/datasets/LLaQo/sbjeval/audio_qa.csv'


class SbjevalDataset(Dataset):
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
                }
        
        sample["waveform"], sample["fbank"] = self.audio_processor(audio_path)[:-1]

        if self.transform:
            sample = self.transform(sample)

        return sample


class SbjevalDatasetQA(BaseDataset):
    def __init__(self, vis_processor, audio_root, seg_name, **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=audio_root)

        self.inner_dataset = SbjevalDataset(ANSWERS_CSV)

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
    transform_Sbjeval_dataset()
    hook()

    dataset = SbjevalDatasetQA(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="all_train",
    )
    print(next(iter(dataset)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    for datum in loader:
        print(datum)
        # hook()
