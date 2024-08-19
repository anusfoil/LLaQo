import os
import torch
import cv2
import argparse
import json
import numpy as np
import torchvision.transforms as T
from glob import glob
from PIL import Image
from torchvision.utils import save_image
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image


def extract_frame(input_video_path, target_fold, extract_frame_num,
                  preprocess_fn):
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
        image_tensor = preprocess_fn(pil_im)
        # save in 'target_path/frame_{i}/video_id.jpg'
        if os.path.exists(target_fold + '/frame_{:d}/'.format(i)) == False:
            os.makedirs(target_fold + '/frame_{:d}/'.format(i))
        save_image(image_tensor,
                   target_fold + '/frame_{:d}/'.format(i) + video_id + '.jpg')


def preprocess_video(csv_pth: str, tgt_dir: str, mini_data: bool):
    r"""Downsample all the video in a single csv to 1 FPS."""
    input_filelist = np.loadtxt(csv_pth, dtype=str, delimiter=',')
    num_file = 10 if mini_data else input_filelist.shape[0]
    print('Total {:d} videos are input'.format(num_file))

    preprocess = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
    ])  # T.RandomCrop(size=(224, 224))

    for file_id in range(num_file):
        try:
            print('processing video {:d}: {:s}'.format(
                file_id, input_filelist[file_id]))
            extract_frame(input_filelist[file_id],
                          tgt_dir,
                          extract_frame_num=10,
                          preprocess_fn=preprocess)
        except:
            print('error with ', print(input_filelist[file_id]))


# def downsample_video(args):
#     csv_list = glob(os.path.join(args.csv_dir, "*"))
#     csv_list = csv_list[:1] if args.mini_data else csv_list
#     for csv_pth in csv_list:
#         print(f"Extract video from {csv_pth}.")
#         tgt_dir = os.path.join(args.frame_output_dir,
#                                csv_pth.split("/")[-1].split(".")[0])
#         preprocess_video(csv_pth, tgt_dir=tgt_dir, mini_data=args.mini_data)


def preprocess_audio(csv_pth: str, tgt_dir: str, mini_data: bool):
    os.makedirs(tgt_dir, exist_ok=True)

    input_filelist = np.loadtxt(csv_pth, delimiter=',', dtype=str)
    num_file = 10 if mini_data else input_filelist.shape[0]

    # Resample and output mono-channel audio
    for i in range(num_file):
        input_f = input_filelist[i]
        ext_len = len(input_f.split('/')[-1].split('.')[-1])
        video_id = input_f.split('/')[-1][:-ext_len - 1]
        output_f_1 = tgt_dir + '/' + video_id + '.flac'  # used to be wav format
        os.system(
            'ffmpeg -threads {:s} -y -i {:s} -vn -ac 1 -ar 16000 -threads {:s} {:s}'
            .format(
                os.environ['NSLOTS'],
                input_f,
                os.environ['NSLOTS'],
                output_f_1,
            ))  # save an intermediate file


# def resample_audio(args):
#     csv_list = glob(os.path.join(args.csv_dir, "*"))
#     csv_list = csv_list[:1] if args.mini_data else csv_list
#     for csv_pth in csv_list:
#         print(f"Extract video from {csv_pth}.")
#         tgt_dir = os.path.join(args.audio_output_dir,
#                                csv_pth.split("/")[-1].split(".")[0])
#         preprocess_audio(csv_pth, tgt_dir=tgt_dir, mini_data=args.mini_data)


def write_json(data: dict, json_path: str):
    _obj = json.dumps(data, indent=1)
    with open(json_path, "w") as f:
        f.write(_obj)


def collect_local_files(local_dir: str):
    filelist = glob(os.path.join(local_dir, "*"))
    file_names, file_paths = [], {}
    for _pth in filelist:
        _name = _pth.split("/")[-1].split(".")[0]
        file_names.append(_name)
        file_paths[_name] = _pth

    return file_names, file_paths


def collect_across_frames(frames_dir: str):
    folder_names = glob(os.path.join(frames_dir, "*"))
    for fid, folder in enumerate(folder_names):
        if fid == 0:
            frame_names, _frame_paths = collect_local_files(folder)
        else:
            tmp, tmp_frame_path = collect_local_files(folder)
            if len(frame_names) == len(tmp):
                continue
            if len(frame_names) - len(tmp) > 0:
                for fname in (set(frame_names) - set(tmp)):
                    frame_names.remove(fname)
                    del _frame_paths[fname]
            if len(tmp) - len(frame_names) > 0:
                for fname in (set(tmp) - set(frame_names)):
                    frame_names.append(fname)
                    _frame_paths[fname] = tmp_frame_path[fname]

    # Format frame_path
    frame_paths = {}
    for fname, fpath in _frame_paths.items():
        frame_paths[fname] = fpath.replace(fpath.split("/")[-2], "frame_{fid}")

    return frame_names, frame_paths, len(folder_names)


def collect_partial_meta(frames_dir, audios_dir):
    frame_names, frame_paths, n_frames = collect_across_frames(frames_dir)
    audio_names, audio_paths = collect_local_files(audios_dir)

    data = []
    for filename in (set(audio_names) & set(frame_names)):
        entry = {
            "filename": filename,
            "audio_path": audio_paths[filename],
            "frame_path": frame_paths[filename],
        }
        data.append(entry)

    return data, n_frames


# def create_json(args):
#     data = []
#     folder_names = glob(os.path.join(args.local_dir, "audios", "*"))
#     folder_names = [folder_names[0]] if args.mini_data else folder_names
#     for fname in folder_names:
#         prefix = fname.split("/")[-1]
#         # print(f"Extract video from {prefix}.")
#         partial_data, n_frames = collect_partial_meta(prefix, args.local_dir)
#         data.extend(partial_data)

#     print((f"Rendering json is done: {len(data)} files ready."))
#     write_json({
#         "data": data,
#         "n_frames": n_frames,
#         "prefix": "{fid}",
#     }, args.output_path)


def create_partial_json(args):
    frames_dir = os.path.join(args.local_dir, "frames", args.prefix)
    audios_dir = os.path.join(args.local_dir, "audios", args.prefix)
    assert os.path.exists(frames_dir) and os.path.exists(
        audios_dir), r"Preprocess audio and video first!"

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{args.prefix}.json")

    data = []

    partial_data, n_frames = collect_partial_meta(frames_dir, audios_dir)
    data.extend(partial_data)

    print((f"Rendering {args.prefix} json is done: {len(data)} files ready."))
    write_json({
        "data": data,
        "n_frames": n_frames,
        "prefix": "{fid}",
    }, json_path)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_video_processing = subparsers.add_parser('process_video')
    parser_video_processing.add_argument(
        "--csv_pth",
        type=str,
        help=
        "Should be a csv file of a single columns, each row is the input video path."
    )
    parser_video_processing.add_argument(
        "--output_dir", type=str, help="The place to store the video frames.")
    parser_video_processing.add_argument("--mini_data",
                                         action='store_true',
                                         default=False,
                                         help="Test on mini_batch.")

    parser_audio_processing = subparsers.add_parser('process_audio')
    parser_audio_processing.add_argument(
        "--csv_pth",
        type=str,
        help=
        "Should be a csv file of a single columns, each row is the input video path."
    )
    parser_audio_processing.add_argument(
        "--output_dir", type=str, help="The place to store the video frames.")
    parser_audio_processing.add_argument("--mini_data",
                                         action='store_true',
                                         default=False,
                                         help="Test on mini_batch.")

    parser_meta_processing = subparsers.add_parser('process_meta')
    parser_meta_processing.add_argument(
        "--prefix", type=str, help="The prefix id of partial folder.")
    parser_meta_processing.add_argument(
        "--local_dir", type=str, help="The directory to store the dataset.")
    parser_meta_processing.add_argument(
        "--output_dir", type=str, help="The path to store the json file.")
    parser_meta_processing.add_argument("--mini_data",
                                        action='store_true',
                                        default=False,
                                        help="Test on mini_batch.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "process_video":
        preprocess_video(
            csv_pth=args.csv_pth,
            tgt_dir=args.output_dir,
            mini_data=args.mini_data,
        )

    elif args.mode == "process_audio":
        preprocess_audio(
            csv_pth=args.csv_pth,
            tgt_dir=args.output_dir,
            mini_data=args.mini_data,
        )

    elif args.mode == "process_meta":
        create_partial_json(args)
