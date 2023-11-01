# -*- coding: utf-8 -*-
# modified from:
# Author: Yuan Gong
r"""This script extract audio from video, then resample and mono-chanel the audio."""
import os
import numpy as np
import argparse
from glob import glob


def process_csv(csv_pth: str, tgt_dir: str, mini_data: bool):
    os.makedirs(tgt_dir, exist_ok=True)

    input_filelist = np.loadtxt(csv_pth, delimiter=',', dtype=str)
    num_file = 10 if mini_data else input_filelist.shape[0]

    # Resample and output mono-channel audio
    for i in range(num_file):
        input_f = input_filelist[i]
        ext_len = len(input_f.split('/')[-1].split('.')[-1])
        video_id = input_f.split('/')[-1][:-ext_len - 1]
        output_f_1 = tgt_dir + '/' + video_id + '.flac' # used to be wav format
        os.system('ffmpeg -i {:s} -vn -ac 1 -ar 16000 {:s}'.format(
            input_f, output_f_1))  # save an intermediate file


def main(args):
    csv_list = glob(os.path.join(args.csv_dir, "*"))
    csv_list = csv_list[:1] if args.mini_data else csv_list
    for csv_pth in csv_list:
        print(f"Extract video from {csv_pth}.")
        tgt_dir = os.path.join(args.audio_output_dir,
                               csv_pth.split("/")[-1].split(".")[0])
        process_csv(csv_pth, tgt_dir=tgt_dir, mini_data=args.mini_data)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Easy audio feature extractor')
    parser.add_argument(
        "--csv_dir",
        type=str,
        help=
        "Should be a csv file of a single columns, each row is the input video path."
    )
    parser.add_argument("--audio_output_dir",
                        type=str,
                        help="The place to store the video frames.")
    parser.add_argument("--mini_data",
                        action='store_true',
                        default=False,
                        help="Test on mini_batch.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
