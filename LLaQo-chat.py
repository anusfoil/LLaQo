import os, sys, json
sys.path.append("../src/lavis/lavis/datasets/datasets")
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gradio as gr
sys.path.append("src/lavis")
sys.path.append("../")

from lavis.models import load_model_and_preprocess
from audio_processor import fbankProcessor
from factory import *

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

    return output

if __name__ == "__main__":

    # demo.launch(show_api=False, share=True)  
    # hook() 

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='ckpts/checkpoint_2283000.pth', help='checkpoint path')
    
    sub_parsers = parser.add_subparsers(dest="mode")

    # Basic mode
    basic_parser = sub_parsers.add_parser("basic")
    basic_parser.add_argument('-f', '--full', action='store_true', help='Go through the full proces')
    basic_parser.add_argument('-c', '--chat', action='store_true', help='Chat with LLaQo.')
    basic_parser.add_argument('--session-id', type=str, default='', help='session id, if set to empty, system will allocate an id')

    args = parser.parse_args()

    # set up model
    # checkpoint_path = load_latest_checkpoint()
    checkpoint_path = args.ckpt
    print(f"Using checkpoint {checkpoint_path}")
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
    test_audio = [
        "test_audio/burgmuller_b-07-annot.wav",
        "test_audio/conespressione_beethoven_casadesus.wav",
        "test_audio/expertnovice_Careless Love-01.wav",
        "test_audio/gestures_NOMETRO_FAST_LEG_1_audio.wav",
        "test_audio/musicshape_sample0280.wav",
        "test_audio/PISA_50.wav",
        "test_audio/YCUPPE_CZ1516-26.wav"
    ]

    def process_input(wav_path, input_text):
        try:
            waveform, fbank = audio_processor(wav_path)[:-1]
        except Exception as e:
            return f"Error in audio processing: {e}"
        
        result = generate_answer(model, fbank, input_text)
        return result

    demo = gr.Interface(
        fn=process_input,
        inputs=[
            gr.Audio(type="filepath", label="Audio File (WAV)", choices=test_audio),
            gr.Textbox(label="Query for the audio")
        ],
        outputs=gr.Textbox(label="Answer"),
        title="LLaQo Demo (solo piano performance)",
        description="Upload an audio file and enter your question regards to the performance. ",
        # server_port=7861
    )

    # manual forwarding from school server: ssh -i ~/.ssh/id_rsa_apocrita -L 7860:rdg7:7860 acw630@login.hpc.qmul.ac.uk
    demo.launch(server_name="sbg19")


