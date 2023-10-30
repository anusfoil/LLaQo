import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from processors import _fbankProcessor


# import hook

ANSWERS_CSV = "/data/EECS-MachineListeningLab/datasets/con_espressione/con_espressione_game_answers.csv"
AUDIO_DIR = "/data/EECS-MachineListeningLab/datasets/con_espressione/audio_all"


class ConEspressioneDataset(Dataset):
    """Con espressione dataset."""

    def __init__(
        self,
        answers_csv=ANSWERS_CSV,
        audio_dir=AUDIO_DIR,
        transform=None,
        *,
        audio_processor=_fbankProcessor.build_processor()
    ):
        """
        Arguments:
            answers_csv (string): Path to the csv file with con espressione game answer.
            audio_dir (string): Directory with all the audios.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.audio_processor = audio_processor

        self.audio_answers = pd.read_csv(answers_csv)
        self.audio_dir = audio_dir
        self.transform = transform

        self.audio_answers["piece_name_"] = self.audio_answers["piece_name"].apply(
            lambda x: x.replace("_", "-").replace("excerpt", "")
        )
        self.audio_answers["audio_path"] = (
            self.audio_answers["piece_name_"]
            + "_"
            + self.audio_answers["performer"].apply(lambda x: x.lower())
        )

    def __len__(self):
        return len(self.audio_answers)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = (
            os.path.join(self.audio_dir, self.audio_answers["audio_path"].iloc[idx])
            + ".wav"
        )

        answer = self.audio_answers["answer"].iloc[idx]

        sample = {
            "audio_path": audio_path,
            "question": "How would you describe this piece of performance?",
            "answer": answer,
        }

        # if self.transform:
        #     sample = self.transform(sample)
        sample["waveform"], sample["fbank"] = self.audio_processor(audio_path)[:-1]

        return sample


if __name__ == "__main__":
    ced = ConEspressioneDataset()
    print(next(iter(ced)))
