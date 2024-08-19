import subprocess
import argparse
import os
import datetime
import time
import logging
import pandas as pd
import numpy as np
from glob import glob


def run(cmd):
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    out, _ = proc.communicate()
    return out.decode('utf-8')


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format=
        '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def split_source_csv_to_partial_csvs(args):
    r"""Split unbalanced csv to part csvs. Each part csv contains up to 5000 ids."""

    source_csv_path = args.source_csv
    partial_csvs_output_dir = args.partial_csvs_output_dir
    audios_num_per_file = args.audios_num_per_file

    create_folder(partial_csvs_output_dir)

    with open(source_csv_path, 'r') as f:
        lines = f.readlines()

    files_num = int(np.ceil(len(lines) / float(audios_num_per_file)))

    for r in range(files_num):
        lines_per_file = lines[r * audios_num_per_file:(r + 1) *
                               audios_num_per_file]

        out_csv_path = os.path.join(
            partial_csvs_output_dir,
            f"{source_csv_path.split('/')[-1].split('.')[0]}_part{r:05d}.csv")

        with open(out_csv_path, 'w') as f:
            for line in lines_per_file:
                f.write(line)

        print('Write out csv to {}'.format(out_csv_path))


def download_videos(args):
    r"""Download videos in MP4 format."""
    # Paths
    csv_path = args.csv_path
    clips_dir = args.clips_dir
    mini_data = args.mini_data
    exist_files = [] if args.overwrite else glob(os.path.join(clips_dir, "*"))
    exist_video_id = [f.split("/")[-1].split(".")[0] for f in exist_files]

    if mini_data:
        logs_dir = f'_logs/download_dataset_minidata/{get_filename(csv_path)}'
    else:
        logs_dir = f'_logs/download_dataset/{get_filename(csv_path)}'

    create_folder(clips_dir)
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Download log is saved to {}'.format(logs_dir))

    # Read csv
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    ori_len = len(lines)

    if mini_data:
        lines = lines[0:10]  # Download partial data for debug

    download_time = time.time()
    duration = 10
    resolution = 360

    # Download
    n_downloaded_files = 0
    for (n, line) in enumerate(lines):
        items = line.split(',')
        video_id = items[0]
        # If not overwrite and file exists do the next loop
        if f"YT{video_id}_{n}" in exist_video_id:
            n_downloaded_files += 1
            continue

        start_time = float(items[1])
        logging.info(f'{n} {video_id} start_time: {start_time:.1f}')

        # Download full video of whatever format
        video_name = os.path.join(clips_dir, f"_YT{video_id}.%(ext)s")
        run([
            "yt-dlp", "--quiet", "-S", "ext", f"'height>={resolution}'", "-o",
            video_name, f"https://www.youtube.com/watch?v={video_id}"
        ])

        video_paths = glob(os.path.join(clips_dir, '_YT' + video_id + '.*'))

        # If download successful
        if len(video_paths) > 0:
            n_downloaded_files += 1

            video_path = video_paths[0]  # Choose one video

            # Add 'YT' to the head because some video ids are started with '-'
            # which will cause problem.
            # Append `n` at the end of filename to avoid two video clips
            # from the same video segmentations.
            out_filepath = os.path.join(clips_dir,
                                        f'YT{video_id}' + f'_{n}.mp4')

            # Preprocess the video by trancating and re-formatting
            cmd = [
                'ffmpeg', '-threads', f"{os.environ['NSLOTS']}", '-loglevel',
                'panic', '-i', video_path, '-ac', '1', '-ar', '32000', '-ss',
                str(datetime.timedelta(seconds=start_time)), '-t',
                f'00:00:{duration}', '-avoid_negative_ts', '1',
                '-reset_timestamps', '1', '-y', '-hide_banner', '-map', '0',
                '-threads', f"{os.environ['NSLOTS']}", out_filepath
            ]  # '-vf', 'scale=640:360', '-loglevel', 'panic', '-strict', '-2'
            run(cmd)

            # Remove downloaded raw video
            os.system("rm {}".format(video_path))
            logging.info("Download and convert to {}".format(out_filepath))

    logging.info(f'Download finished ({n_downloaded_files}/{ori_len})!')
    logging.info('Time spent: {:.3f} s'.format(time.time() - download_time))
    logging.info(f'Logs can be viewed in {logs_dir}')


def glob_meta(data_storage_dir: str,
              output_csv: str,
              verbose: bool = True) -> int:
    r"""Returns single-column csv files using ',' as the separators."""
    file_list = glob(os.path.join(data_storage_dir, "*"))
    custom_df = pd.DataFrame({
        "file_path": file_list,
    })
    custom_df.to_csv(output_csv, sep=",", header=None, index=False)
    if verbose:
        print(f"The curated meta in {output_csv}:\n {custom_df}")
    return len(file_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_split = subparsers.add_parser('split_source_csv_to_partial_csvs')
    parser_split.add_argument('--source_csv',
                              type=str,
                              required=True,
                              help='Path of source_csv file to read.')
    parser_split.add_argument(
        '--partial_csvs_output_dir',
        type=str,
        required=True,
        help='Directory to save split partial csv output.')
    parser_split.add_argument('--audios_num_per_file', type=int, default=10000)

    parser_download_videos = subparsers.add_parser('download_videos')
    parser_download_videos.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path of csv file containing audio info to be downloaded.')
    parser_download_videos.add_argument(
        '--clips_dir',
        type=str,
        required=True,
        help='Directory to save out downloaded video.')
    parser_download_videos.add_argument('--overwrite',
                                        action='store_true',
                                        default=False,
                                        help='Overwrite the existing data.')
    parser_download_videos.add_argument(
        '--mini_data',
        action='store_true',
        default=False,
        help='Set true to only download 10 audios for debugging.')

    args = parser.parse_args()

    if args.mode == 'split_source_csv_to_partial_csvs':
        split_source_csv_to_partial_csvs(args)

    elif args.mode == 'download_videos':
        download_videos(args)
        chars = args.csv_path.split("/")
        target_csv_dir = os.path.join(*chars[:-1])

        # TODO: modify the path if it is not begin with '/'
        output_csv = '/' + os.path.join(target_csv_dir, f"custom_{chars[-1]}")
        glob_meta(
            data_storage_dir=args.clips_dir,
            output_csv=output_csv,
            verbose=False,
        )

    else:
        raise Exception('Incorrect arguments!')
