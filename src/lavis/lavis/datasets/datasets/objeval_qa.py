"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os, sys, math, glob, random
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
    "Is the legato even?": "How would you rate if the legato is even? on a scale of 1 to 6, 1 is the worst and 6 is the best, use the full scale as much as possible. ",
    "Are the note values uniform?": "How would you rate the uniformity of the note values? on a scale of 1 to 6, 1 is the worst and 6 is the best, use the full scale as much as possible. ",
    "How solid is the sound?": "How would you rate the solidity of the sound? on a scale of 1 to 6, 1 is the worst and 6 is the best, use the full scale as much as possible. ",
    "How clean is the attack?": "How would you rate the cleanliness of the attack? on a scale of 1 to 6, 1 is the worst and 6 is the best, use the full scale as much as possible. ",
    "Are the left and right hands balanced?": "How would you rate the balance between the left and right hands? on a scale of 1 to 6, 1 is the worst and 6 is the best, use the full scale as much as possible. ",
    "Are the timings aligned on the left and right hands?": "How would you rate the alignment of timings between the left and right hands? on a scale of 1 to 6, 1 is the worst and 6 is the best, use the full scale as much as possible. ",
    "Is it played with the correct rhythm?": "How would you rate the correctness of the rhythm? on a scale of 1 to 6, 1 is the worst and 6 is the best, use the full scale as much as possible. ",
    "Is the tempo kept constant?": "How would you rate the consistency of the tempo? on a scale of 1 to 6, 1 is the worst and 6 is the best, use the full scale as much as possible. ",
    "Are the lines connected?": "How would you rate the connectivity of the lines? on a scale of 1 to 6, 1 is the worst and 6 is the best, use the full scale as much as possible. ",
    "Is it played with a sense of tonality?": "How would you rate the sense of tonality? on a scale of 1 to 6, 1 is the worst and 6 is the best, use the full scale as much as possible. ",
    "Is the dynamics change natural?": "How would you rate the naturalness of the dynamics change? on a scale of 1 to 6, 1 is the worst and 6 is the best, use the full scale as much as possible. "
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



def set_playdata_split():
    
    training_files_path = "/data/EECS-MachineListeningLab/datasets/LLaQo/objeval/qa/training_files.txt"
    if os.path.exists(training_files_path):
        print("training data file already exsits!")
        return
    
    # Define the path where CSV files are stored
    csv_paths = glob.glob("/data/EECS-MachineListeningLab/datasets/LLaQo/objeval/qa/*.csv")

    # Concatenate all CSV files into one DataFrame
    all_csv = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)

    # Get unique audio file names
    unique_files = all_csv['fname'].unique()

    # Randomly shuffle the array of unique files to ensure randomness in selection
    random.shuffle(unique_files)

    # Split the unique files into training set (50% of the files)
    split_index = len(unique_files) // 2
    training_files = unique_files[:split_index]

    # Write the training filenames to a text file
    with open(training_files_path, 'w') as f:
        for file in training_files:
            f.write(file + '\n')
    
    # Optionally return the DataFrame and training files list for further use in the code
    return all_csv, training_files


def transform_Objeval_dataset():
    """Objective eval dataset, with standardized audio and teacher's rating on each dimension.
    """
    
    new_audio_qa = "/data/EECS-MachineListeningLab/datasets/LLaQo/objeval/qa/new_audio_qa.csv"
    new_audio_qa = pd.read_csv(new_audio_qa)
    new_audio_qa = new_audio_qa[new_audio_qa['split'] == 'eval']
    
    qa_csv = []
    for idx, row in new_audio_qa.iterrows():
                
        if row['question_source_id'] in [1, 2]:
            row['Q'] = "How would you rate the overall performance? on a scale of 1 to 6, 1 is the worst and 6 is the best?"
            row['A'] = str(row['score'])
            row['Q2'] = row['q_eng']
            row['A2'] = row['a_eng']
            row['question_category'] = 'summary'
            qa_csv.append(copy.deepcopy(row))
        else:
            row['Q'] = QMAP[row['q_eng']]
            row['A'] = str(row['score'])
            row['Q2'] = row['q_eng']
            row['A2'] = row['a_eng']
            row['question_category'] = QCA[row['q_eng']]
            qa_csv.append(copy.deepcopy(row))

    qa_csv = pd.DataFrame(qa_csv)
    qa_csv.to_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/objeval/audio_qa.csv")



def transform_Objeval_dataset_():
    """Objective eval dataset, with standardized audio and teacher's rating on each dimension.
    """
    qa_csv = []
    
    wav_paths = glob.glob("/data/EECS-MachineListeningLab/datasets/LLaQo/objeval/songs/**/*.wav", recursive=True)
    csv_paths = glob.glob("/data/EECS-MachineListeningLab/datasets/LLaQo/objeval/qa/*.csv")
    training_files_path = "/data/EECS-MachineListeningLab/datasets/LLaQo/objeval/qa/training_files.txt"
    
    # Read the training files
    with open(training_files_path, 'r') as f:
        training_files = f.read().splitlines()
    
    for csv_path in csv_paths:

        ratings = pd.read_csv(csv_path)
        ratings = ratings[~ratings['fname'].isin(training_files)] # eval half
        
        # remove duplicate regarding the same audio, q and a
        ratings = ratings.drop_duplicates(subset=['fname', 'quesition', 'answer', 'score'])
        
        for idx, row in ratings.iterrows():

            # make sure the answer is not nan or empty
            if ((pd.isna(row['answer']) or row['answer'] == "") or (pd.isna(row['score']) or row['score'] == "")) :
                continue

            audio_path = row['fname']
            row['audio_path'] = [wp for wp in wav_paths if audio_path in wp][0]
            row['question_id'] = row['question_source_id']
                        
            if row['question_source_id'] in [1, 2]:
                row['Q'] = "How would you rate the overall performance? on a scale of 1 to 6, 1 is the worst and 6 is the best?"
                row['A'] = str(row['score'])
                row['Q2'] = row['quesition']
                row['A2'] = row['answer']
                row['question_category'] = 'summary'
                qa_csv.append(copy.deepcopy(row))
            else:
                row['Q'] = QMAP[row['quesition']]
                row['A'] = str(row['score'])
                row['Q2'] = row['quesition']
                row['A2'] = row['answer']
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
                'question2': self.audio_qa['Q2'].iloc[idx],
                'answer': self.audio_qa['A'].iloc[idx],
                'answer2': self.audio_qa['A2'].iloc[idx],
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
    # set_playdata_split()
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
