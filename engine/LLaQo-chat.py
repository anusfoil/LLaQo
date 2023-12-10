import os, sys, json
sys.path.append("../src/lavis/lavis/datasets/datasets")
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append("../src/lavis")

from lavis.models import load_model_and_preprocess
from audio_processor import fbankProcessor
import hook


def generate_answer(
        model,
        fbank, 
        input_question,
):

    output = model.generate(
        {
            "audio": fbank.unsqueeze(0).cuda(),
            "prompt": input_question, 
        },
        temperature=0.1,
    )
    result = {
        "question": input_question,
        "output": output, 
    }
    print(output)

    return result


parser = argparse.ArgumentParser()
sub_parsers = parser.add_subparsers(dest="mode")

# Basic mode
basic_parser = sub_parsers.add_parser("basic")
basic_parser.add_argument('-f', '--full', action='store_true', help='Go through the full proces')
basic_parser.add_argument('-c', '--chat', action='store_true', help='Chat with LLaQo.')
basic_parser.add_argument('--session-id', type=str, default='', help='session id, if set to empty, system will allocate an id')

args = parser.parse_args()

# set up model
checkpoint_path = "/data/EECS-MachineListeningLab/huan/lam/check_point/Pretrain_stage2/test_musicqa/20231121000/checkpoint_9000.pth"
# pth = checkpoint_path.split("/")[-1].split(".")[0]
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, _, _ = load_model_and_preprocess(
    name="lam_vicuna_instruct",
    model_type="vicuna1.5_7b-ft", 
    is_eval=True,
    device=device,
)
model.load_from_pretrained(checkpoint_path)

audio_processor=fbankProcessor.build_processor()

input_wav = []

test_audio = [
    "/data/EECS-MachineListeningLab/datasets/LLaQo/expert_novice/recordings_and_alignments/CarelessLove/Careless Love-01.wav"
]

waveform, fbank = audio_processor(test_audio[0])[:-1]

while True:
    while True:
        wav_path = input("Add audio file: ")
        if wav_path == "BREAK":
            break
        try:
            waveform, fbank = audio_processor(wav_path)[:-1]
        except Exception as e:
            print(e)
            print('audio loading failed. Please try again')
            continue
        else:
            break

    while True:
        input_text = input("Enter your instruction (input `EXIT` to exit the process, input `ANOTHER` to question another wav file): ")

        if input_text == "EXIT":
            print("LLaQo session stopped.")
            break
        if input_text == "ANOTHER":
            break
        
        result = generate_answer(model, fbank, input_text)

    if input_text == "EXIT":
        break


