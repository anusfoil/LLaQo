# -*- coding: utf-8 -*-
# modified from:
# Author: Yuan Gong
r"""This script a fix number of imgs per video."""

import os.path
import cv2
import numpy as np
import torchvision.transforms as T
from glob import glob
from PIL import Image
from torchvision.utils import save_image
from argparse import ArgumentParser

preprocess = T.Compose([T.Resize(224),
                        T.CenterCrop(224),
                        T.ToTensor()])  # T.RandomCrop(size=(224, 224))


def extract_frame(input_video_path, target_fold, extract_frame_num):
    # TODO: you can define your own way to extract video_id
    ext_len = len(input_video_path.split('/')[-1].split('.')[-1])
    video_id = input_video_path.split('/')[-1][:-ext_len - 1]
    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # this is to avoid vggsound video's bug on not accurate frame count
    total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)),
                          int(fps * 10))
    for i in range(extract_frame_num):
        frame_idx = int(i * (total_frame_num / extract_frame_num))
        print(
            'Extract frame {:d} from original frame {:d}, total video frame {:d} at frame rate {:d}.'
            .format(i, frame_idx, total_frame_num, int(fps)))
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        _, frame = vidcap.read()
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        image_tensor = preprocess(pil_im)
        # save in 'target_path/frame_{i}/video_id.jpg'
        if os.path.exists(target_fold + '/frame_{:d}/'.format(i)) == False:
            os.makedirs(target_fold + '/frame_{:d}/'.format(i))
        save_image(image_tensor,
                   target_fold + '/frame_{:d}/'.format(i) + video_id + '.jpg')


def process_csv(csv_pth: str, tgt_dir: str, mini_data: bool):
    r"""Extract to frames all the video in a single csv."""
    input_filelist = np.loadtxt(csv_pth, dtype=str, delimiter=',')
    num_file = 10 if mini_data else input_filelist.shape[0]
    print('Total {:d} videos are input'.format(num_file))
    for file_id in range(num_file):
        try:
            print('processing video {:d}: {:s}'.format(
                file_id, input_filelist[file_id]))
            extract_frame(input_filelist[file_id],
                          tgt_dir,
                          extract_frame_num=10)
        except:
            print('error with ', print(input_filelist[file_id]))


def main(args):
    csv_list = glob(os.path.join(args.csv_dir, "*"))
    csv_list = csv_list[:1] if args.mini_data else csv_list
    for csv_pth in csv_list:
        print(f"Extract video from {csv_pth}.")
        tgt_dir = os.path.join(args.frame_output_dir,
                               csv_pth.split("/")[-1].split(".")[0])
        process_csv(csv_pth, tgt_dir=tgt_dir, mini_data=args.mini_data)


def parse_args():
    parser = ArgumentParser(
        description=
        "Python script to extract frames from a video, save as jpgs.")
    parser.add_argument(
        "--csv_dir",
        type=str,
        help=
        "Should be a csv file of a single columns, each row is the input video path."
    )
    parser.add_argument("--frame_output_dir",
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
