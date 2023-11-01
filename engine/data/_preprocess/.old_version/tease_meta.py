r"""Check the local data with the orginal meta data."""
import argparse
import os
import pandas as pd
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--origin_csv_dir',
                        type=str,
                        required=True,
                        help='Directory of csv files containing video info.')
    parser.add_argument('--video_storage_dir',
                        type=str,
                        required=True,
                        help='Root directory to save downloaded files.')
    parser.add_argument('--target_csv_dir',
                        type=str,
                        required=True,
                        help='Directory to save teased csv files.')
    parser.add_argument('--mini_data',
                        action='store_true',
                        default=False,
                        help='Set true to tease 1 csv file for debugging.')

    return parser.parse_args()


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


def main(args):
    os.makedirs(args.target_csv_dir, exist_ok=True)

    origin_csv_files = glob(os.path.join(args.origin_csv_dir, "*.csv"))

    if args.mini_data:
        origin_csv_files = origin_csv_files[:1]

    for origin_csv_pth in origin_csv_files:
        # include the first row
        ori_len = len(pd.read_csv(origin_csv_pth, header=None))
        file_name = origin_csv_pth.split("/")[-1].split(".")[0]
        teased_len = glob_meta(
            data_storage_dir=os.path.join(args.video_storage_dir, file_name),
            output_csv=os.path.join(args.target_csv_dir,
                                    f"custom_{file_name}.csv"),
            verbose=True)
        print(f"file_name is finished: {teased_len}/{ori_len}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
