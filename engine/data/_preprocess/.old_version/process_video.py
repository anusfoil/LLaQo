import os
import torch
import cv2
import argparse
import numpy as np
import torchvision.transforms as T
from glob import glob
from PIL import Image
from torchvision.utils import save_image
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
from gpt4free.deepai import Completion
import sys

sys.path.append("..")
from dataloader import AVDataset

sys.path.append("../../../src")
from utils import write_json, read_json

sys.path.append("../../../src/lavis")
from lavis.models import load_model_and_preprocess


def caption_frame(args):
    prompt = "Based on the image, describe all possible sounds in the scene. More precise, the better."

    assert args.n_captions_per_image == 1, r"fix this parameter when extracting video captions per frames."
    os.makedirs(args.output_dir, exist_ok=True)
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    enable_amp = device != torch.device("cpu")
    # load sample image
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': 1024,
        'freqm': 0,
        'timem': 0,
        'mixup': 0.0,
        'dataset': "audioset",
        'mode': 'train',
        'mean': -5.081,
        'std': 4.4849,
        'noise': True,
        'label_smooth': 0,
        'im_res': 224
    }
    dataset = AVDataset(dataset_json_file=args.json_path,
                        audio_conf=audio_conf,
                        label_csv=None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
    )

    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(
        name='blip2_vicuna_instruct',  # "blip2_t5",
        model_type='vicuna7b',  # "pretrain_flant5xl",
        is_eval=True,
        device=device)

    # do the inference
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        if args.mini_data and batch_idx == 1:
            break

        cache_outputs = []
        datum, _, videos = batch

        n_frames = videos.size(1)
        for id in range(n_frames):
            outputs = model.generate(
                {
                    "image": videos[:, id, :, :].to(
                        device
                    ),  # .to(torch.bfloat16) if enable_amp else videos,
                    "prompt":
                    prompt,  # in the video;  Based on the image, use a few words to describe what you will hear
                },
                use_nucleus_sampling=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1,
                num_captions=args.n_captions_per_image,
                temperature=1,
            )
            cache_outputs.append(
                outputs)  # (psydo) shape: (n_frames, batch_size)

        captions = []  # (psydo) shape expected to be: (batch_size, n_frames)
        for batch_idx in range(args.batch_size):
            captions.append([caps[batch_idx] for caps in cache_outputs])

        print(len(captions))
        print(len(captions[-1]))
        print(captions)

        for idx, caps in enumerate(captions):
            video_id = datum['video_id'][idx]
            audio_path = datum['audio_path'][idx]
            write_json(
                {
                    "video_id": video_id,
                    "audio_path": audio_path,
                    "caption_per_frame": caps,
                }, output_json_path)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_captioning = subparsers.add_parser('caption_frame')
    parser_captioning.add_argument("--output_dir", type=str)
    parser_captioning.add_argument("--json_path", type=str)
    parser_captioning.add_argument("--batch_size", type=int, default=1)
    parser_captioning.add_argument("--mini_data", action="store_true", default=False)
    parser_captioning.add_argument("--n_captions_per_image", type=int, default=1)

    parser_downsampling = subparsers.add_parser('downsample_video')
    parser_downsampling.add_argument(
        "--csv_dir",
        type=str,
        help=
        "Should be a csv file of a single columns, each row is the input video path."
    )
    parser_downsampling.add_argument("--frame_output_dir",
                        type=str,
                        help="The place to store the video frames.")
    parser_downsampling.add_argument("--mini_data",
                        action='store_true',
                        default=False,
                        help="Test on mini_batch.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "caption_frame":
        caption_frame(args)
    elif args.mode == "downsample_video":
        downsample_video(args)
