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
import random
import copy
sys.path.append(os.path.dirname(__file__))
from audio_processor import fbankProcessor
import hook

try:
    from lavis.datasets.datasets.base_dataset import BaseDataset

    # from lavis.datasets.datasets.prompt_template import _QUESTION4CLASSIFICATION_ as _QUESTION_
except:
    from base_dataset import BaseDataset

ORIGINAL_CSV = '/data/EECS-MachineListeningLab/datasets/LLaQo/con_espressione/con_espressione_game_answers_.csv'
AUDIO_DIR = '/data/EECS-MachineListeningLab/datasets/LLaQo/con_espressione/audio_all'

def transform_conespressione_dataset():
    """Con Espressione is a dataset that contains master's performance, as well as 
    """

    audio_answers = pd.read_csv(ORIGINAL_CSV)

    audio_answers['piece_name_'] = audio_answers['piece_name'].apply(lambda x: x.replace("_", '-').replace("excerpt", ""))
    audio_answers['audio_path'] = audio_answers['piece_name_'] + "_" + audio_answers['performer'].apply(lambda x: x.lower())
    audio_answers['audio_path'] = AUDIO_DIR + "/" + audio_answers['audio_path'] + ".wav"

    qa_csv = []
    test_idx = random.choices( audio_answers.music_id.unique(), k = 10)
    for _, (_, arow) in enumerate(audio_answers.iterrows()):

        row = {'audio_path': arow['audio_path'],
               'split': 'test' if arow['music_id'] in test_idx else 'train'}

        answer = arow['answer'].replace("_", " ")
        
        if random.random() < 0.3:
            row['Q'] = "How would you describe this performance?"
            if ("MIDI" in row['audio_path']) or ("midi" in row['audio_path']):
                row['A'] = f"This piece sounds like a MIDI performance without any expression. I would describe it as {answer}."
            else:
                row['A'] = answer
            qa_csv.append(copy.deepcopy(row))
        else:
            row['Q'] = "How would you assess this student performance?"
            if ("MIDI" in row['audio_path']) or ("midi" in row['audio_path']):
                row['A'] = f"This piece sounds like a MIDI performance without any expression. I would describe it as {answer}."
            else:
                row['A'] = f"Sorry, but this piece sounds like a virtuoso performance. I would describe it as {answer}."
            qa_csv.append(copy.deepcopy(row))     

        row['Q'] = "What kind of music is this piece and who is the composer?"
        row['A'] = arow['piece_style']
        qa_csv.append(copy.deepcopy(row))
        row['Q'] = "How would you rate the difficulty of this piece, in a scale of 9?"
        row['A'] = str(arow['piece_difficulty'])
        qa_csv.append(copy.deepcopy(row))

    qa_csv = pd.DataFrame(qa_csv)
    qa_csv.to_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/con_espressione/audio_qa.csv")


ANSWERS_CSV = '/data/EECS-MachineListeningLab/datasets/LLaQo/con_espressione/audio_qa.csv'


class ConEspressioneDataset(Dataset):
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




class ConEspressioneDatasetQA(BaseDataset):
    def __init__(self, vis_processor, audio_root, seg_name, **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=audio_root)

        self.inner_dataset = ConEspressioneDataset()

        self._add_instance_ids()

    def __len__(self):
        # return 100
        return len(self.inner_dataset)

    def __getitem__(self, index):
        datum = self.inner_dataset[index]

        return {
            "audio": datum["fbank"],
            "text_input": datum["question"],
            "text_output": datum["answer"],
        }

    def displ_item(self, index):
        datum = self.inner_dataset[index]

        return {
            "audio": datum["fbank"],
            "text_input": datum["question"],
            "text_output": datum["answer"],
        }



if __name__ == "__main__":
    transform_conespressione_dataset()
    hook()

    dataset = ConEspressioneDatasetQA(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="all_train",
    )
    print(next(iter(dataset)))

    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    for datum in loader:
        print(datum)
        hook()
