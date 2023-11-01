import os
import torch
from tqdm import tqdm
import argparse
from PIL import Image
import sys

sys.path.append("..")
from dataloader import AVDataset

sys.path.append("../../../src")
from utils import write_json, read_json

sys.path.append("../../../src/lavis")
from lavis.models import load_model_and_preprocess


def main(args):
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
    prompt = "Based on the image, describe all possible sounds in the scene. More precise, the better."
    print(prompt)
    cache_outputs = []
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        if args.mini_data and batch_idx == 1:
            break
        datum, _, videos = batch
        "Based on the video, provide a question with the answer: a group of men in cowboy outfits playing musical instruments on stage. Question:"
        for idx, (q_output, i_llm, cap) in enumerate(
                zip(outputs["query_output"], outputs["inputs_llm"],
                    outputs["captions"])):
            video_id = datum['video_id'][idx]
            video_feature_path = os.path.join(args.output_dir,
                                              f"{video_id}.pt")
            audio_path = datum['audio_path'][idx]
            output_json_path = os.path.join(args.output_dir,
                                            f"{video_id}.json")

            torch.save(
                {
                    "query_output": q_output.cpu(),
                    "inputs_llm": i_llm.cpu()
                }, video_feature_path)
            write_json(
                {
                    "video_id": video_id,
                    "audio_path": audio_path,
                    "video_feature_path": video_feature_path,
                    "caption": cap,
                }, output_json_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--json_path", type=str)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--mini_data", action="store_true", default=False)
    parser.add_argument("--n_captions_per_image", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
