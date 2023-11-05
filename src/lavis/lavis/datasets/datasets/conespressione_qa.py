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

try:
    from lavis.datasets.datasets.base_dataset import BaseDataset

    # from lavis.datasets.datasets.prompt_template import _QUESTION4CLASSIFICATION_ as _QUESTION_
except:
    from base_dataset import BaseDataset

    # from prompt_template import _QUESTION4CLASSIFICATION_ as _QUESTION_


ANSWERS_CSV = '/data/EECS-MachineListeningLab/datasets/LLaQo/con_espressione/con_espressione_game_answers.csv'
AUDIO_DIR = '/data/EECS-MachineListeningLab/datasets/LLaQo/con_espressione/audio_all'

class ConEspressioneDataset(Dataset):
    """Con espressione dataset."""

    def __init__(self, answers_csv=ANSWERS_CSV, audio_dir=AUDIO_DIR, transform=None,
                 audio_processor=fbankProcessor.build_processor()):
        """
        Arguments:
            answers_csv (string): Path to the csv file with con espressione game answer.
            audio_dir (string): Directory with all the audios.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.audio_answers = pd.read_csv(answers_csv)
        self.audio_dir = audio_dir
        self.transform = transform

        self.audio_answers['piece_name_'] = self.audio_answers['piece_name'].apply(lambda x: x.replace("_", '-').replace("excerpt", ""))
        self.audio_answers['audio_path'] = self.audio_answers['piece_name_'] + "_" + self.audio_answers['performer'].apply(lambda x: x.lower())

        self.audio_processor = audio_processor

    def __len__(self):
        return len(self.audio_answers)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = os.path.join(self.audio_dir,
                                self.audio_answers['audio_path'].iloc[idx]) + ".wav"

        answer = self.audio_answers['answer'].iloc[idx]
        
        sample = {
                'audio_path': audio_path, 
                'question': "How would you describe this piece of performance?",
                'answer': answer}
        
        sample["waveform"], sample["fbank"] = self.audio_processor(audio_path)[:-1]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ConEspressioneDatasetQA(BaseDataset):
    def __init__(self, vis_processor, audio_root, seg_name, **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=audio_root)

        self.inner_dataset = ConEspressioneDataset(ANSWERS_CSV, AUDIO_DIR)

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
    import torch

    dataset = ConEspressioneDatasetQA(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="all_train",
    )
    print(next(iter(dataset)))

    # loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    # for datum in loader:
    #     print(datum)
    #     break
