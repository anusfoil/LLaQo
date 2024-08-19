import os
import sys
import torch
from torch import Tensor
from tqdm import tqdm

from captioning import _instruct_audio
sys.path.append("..")
from dataloader import AVDataset

sys.path.append("../../../src")
from laion_clap import CLAPWrapper
from utils import write_json, read_json

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


def sort_by_weights(input: list, weights: Tensor) -> list:
    scores, sorted_idx = torch.sort(weights, descending=True)

    res = []
    for idx in sorted_idx.tolist():
        res.append(input[idx])

    return scores.tolist(), res


def get_content(input: str) -> str:
    r"""Process the GPT-generated sentence, format='#XX: ...'"""
    return input.split(': ')[-1].strip()
    

def parse_captions(caption: str) -> list:
    r"""Parse caption produced by ChatGPT into list of captions.
        Expect input format: '#C1: ...\n#C2: ...'
    """
    caps = caption.split("\n")

    return [get_content(cap) for cap in caps]


def main(json_path, weights_path, meta_dir, mini_data, use_cuda, enable_gpt, overwrite):
    clap = CLAPWrapper(weights_path,
                       enable_fusion=False,
                       use_cuda=use_cuda,
                       sampling_rate=48000)
    dataset = AVDataset(dataset_json_file=json_path,
                        audio_conf=audio_conf,
                        label_csv=None)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             num_workers=4,
                                             pin_memory=True)

    for ids, (meta, _, _) in tqdm(enumerate(dataloader)):
        if mini_data and ids > 0:
            break

        json_path = os.path.join(meta_dir, f"{meta['filename'][0]}.json")
        datum = read_json(json_path)
        print(datum['filename'])

        # Skip if captions is ranked already, default not overwrite
        if not overwrite and "ranked_caption" in datum:
            continue

        try:
            caps = datum["audio_caption"]
        except KeyError:
            if enable_gpt:
                start_time = _instruct_audio(datum=datum, instruction_type="caption", output_json_path=json_path)
                datum = read_json(json_path)
                caps = datum["audio_caption"]
            else:
                raise KeyError(f"{meta['filename'][0]} must be captioned first! One can enable gpt to avoid this.")


        caps = parse_captions(caps)
        try:
            sim = clap.extract_feature_and_calculate_similarity(
                [meta['audio_path'][0]],
                caps,
                enable_softmax=False,
                resample=True,
            ).detach().squeeze()  #shape = (1, num_caps)
            score, sorted_caps = sort_by_weights(caps, sim)

            caps_and_scores = [(cap, score[id])
                               for id, cap in enumerate(sorted_caps)]

            datum['ranked_caption'] = caps_and_scores
            write_json(datum, json_path)

        except ValueError:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str)
    parser.add_argument('--meta_dir', type=str)
    parser.add_argument('--json_path', type=str)
    parser.add_argument("--mini_data", action="store_true", default=False)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--enable_gpt", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()

    main(
        json_path=args.json_path,
        weights_path=args.weights_path,
        meta_dir=args.meta_dir,
        use_cuda=args.use_cuda,
        mini_data=args.mini_data,
        enable_gpt=args.enable_gpt,
        overwrite=args.overwrite,
    )
