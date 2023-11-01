# -*- coding: utf-8 -*-
# modified from Yuan Gong
import os
import json
import argparse
from glob import glob


# "video_id": "-0nqfRcnAYE",
# "wav": "/data/sls/audioset/dave_version/audio/-0nqfRcnAYE.flac",
# "image": "/data/sls/audioset/dave_version/images/-0nqfRcnAYE.png",
# "labels": "/m/04brg2",
# "afeat": "/data/sls/scratch/yuangong/avbyol/data/audioset/a_feat/audio_feat_convnext_2/-0nqfRcnAYE.npy",
# "vfeat": "/data/sls/scratch/yuangong/avbyol/data/audioset/v_feat/video_feat_convnext_2/-0nqfRcnAYE.npy",
# "video": "/data/sls/audioset/dave_version/eval/-0nqfRcnAYE.mkv"
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
            tmp, _ = collect_local_files(folder)
            assert len(frame_names) == len(tmp)
    # Format frame_path
    frame_paths = {}
    for fname, fpath in _frame_paths.items():
        frame_paths[fname] = fpath.replace(fpath.split("/")[-2], "frame_{fid}")

    return frame_names, frame_paths, len(folder_names)


def process_csv(prefix: str, local_dir: str):
    frame_names, frame_paths, n_frames = collect_across_frames(
        os.path.join(local_dir, "frames", prefix))
    audio_names, audio_paths = collect_local_files(
        os.path.join(local_dir, "audios", prefix))

    data = []
    for filename in (set(audio_names) & set(frame_names)):
        entry = {
            "video_id": filename,
            "audio_path": audio_paths[filename],
            "frame_path": frame_paths[filename],
        }
        data.append(entry)

    return data, n_frames


def main(args):
    data = []
    folder_names = glob(os.path.join(args.local_dir, "audios", "*"))
    folder_names = [folder_names[0]] if args.mini_data else folder_names
    for fname in folder_names:
        prefix = fname.split("/")[-1]
        # print(f"Extract video from {prefix}.")
        partial_data, n_frames = process_csv(prefix, args.local_dir)
        data.extend(partial_data)

    print((f"Rendering json is done: {len(data)} files ready."))
    write_json({
        "data": data,
        "n_frames": n_frames,
        "prefix": "{fid}",
    }, args.output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Easy audio feature extractor')

    parser.add_argument("--local_dir",
                        type=str,
                        help="The directory to store the dataset.")
    parser.add_argument("--output_path",
                        type=str,
                        help="The path to store the json file.")
    parser.add_argument("--mini_data",
                        action='store_true',
                        default=False,
                        help="Test on mini_batch.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
