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


TEMPO_ANSWER = {
    "FAST": "Tempo of the performance seems a bit fast.", 
    "NORMAL": "Tempo of the performance seems appropriate.",
    "SLOW":  "Tempo of the performance seems a bit slow.",
    "RUBATO":  "Tempo of the performance seems rubato, contains continuous expressive tempo alteration"
}

ARTICULATION_ANSWER = {
    "LEG": "Performed with finger legato.",
    "STA": "Performed with finger staccato.",
    "STAC": "Performed with finger staccato.",
}

INTENTION_ANSWER = {
    "NORMAL": "Sounds smooth, with appropriate amount of expression.",
    "STILL": "Performed with relatively less expression, rigid performance",
    "EXAG": "Performed with exaggerated tempo and dynamics change, very expressive.",
}

def transform_MusicGestures_dataset():
    """Music Gestrues dataset is a dataset recorded by two pianists, playing the same piece under
    different instruction such as tempo, use metronome or not, and legato / staccato. 
    
    """
    import glob
    qa_csv = []
    test_idx = [randint(0, 210) for _ in range(20)]

    piece_dir = glob.glob("/data/EECS-MachineListeningLab/datasets/LLaQo/expressive_musical_gestures/piano/**/*.wav", recursive=True)

    for idx, audio_path in enumerate(piece_dir):
        audio_info = {}

        audio_info['split'] = 'test' if idx in test_idx else 'train'
        audio_info['audio_path'] = audio_path
        audio_info['player'] = audio_path.split("/")[7]
        
        instruction = audio_path.split("/")[-2]
        metro, tempo, articulation, _ = instruction.split("_")

        audio_info['Q'] = "How is the overall tempo?"
        audio_info['A'] = TEMPO_ANSWER[tempo]
        qa_csv.append(copy.deepcopy(audio_info))
        if ("LEG" in articulation) or ("STA" in articulation):
            audio_info['Q'] = "How is the articulation in the piece?"
            audio_info['A'] = ARTICULATION_ANSWER[articulation]
            qa_csv.append(copy.deepcopy(audio_info))
        else:
            audio_info['Q'] = "How is the expressive intention in the performance?"
            audio_info['A'] = INTENTION_ANSWER[articulation]
            qa_csv.append(copy.deepcopy(audio_info))
        audio_info['Q'] = "What is the stylistic period of the piece? Who is the potential composer?"
        audio_info['A'] = "This is a piece in romantic style, and composer is Schumann."
        qa_csv.append(copy.deepcopy(audio_info))
        audio_info['Q'] = "Which difficulty level is the piece, in a scale of 9?"
        audio_info['A'] = "5"
        qa_csv.append(copy.deepcopy(audio_info))

    qa_csv = pd.DataFrame(qa_csv)
    qa_csv.to_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/expressive_musical_gestures/audio_qa.csv")
    hook()


ANSWERS_CSV = '/data/EECS-MachineListeningLab/datasets/LLaQo/expressive_musical_gestures/audio_qa.csv'


class MusicGesturesDataset(Dataset):
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


class MusicGesturesDatasetQA(BaseDataset):
    def __init__(self, vis_processor, audio_root, seg_name, **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=audio_root)

        self.inner_dataset = MusicGesturesDataset(ANSWERS_CSV)

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
    # transform_MusicGestures_dataset()

    dataset = MusicGesturesDatasetQA(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="all_train",
    )
    print(next(iter(dataset)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    for datum in loader:
        print(datum)
        hook()
