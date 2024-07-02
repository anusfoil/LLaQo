import os, sys, re
import torch
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append("../src/lavis")

from lavis.datasets.datasets.objeval_qa import ObjevalDataset
# from utilities import collate_func, set_logger, write_json, collate_func

from lavis.models import load_model_and_preprocess
from factory import *

# log = set_logger(__name__)


def extract_rating_from_text(text):
    """
    Extracts the first occurring numeric rating from a text string, supporting both integers and floats.

    Args:
    text (str): A string that contains a numeric rating.

    Returns:
    float or None: The first found number in the text as a float, or None if no number is found.
    """
    # Regular expression to find numbers, including integers and floats
    match = re.search(r'\b\d+(?:\.\d+)?\b', text)
    if match:
        return float(match.group(0))  # Convert found number to float to accommodate both integers and floats
    else:
        return None  # Return None if no number is found


def generate_answer_on_musicqa(
    lam_ckpt_path="",
    results_path="/data/home/acw630/WORKPLACE/LAM/engine/results/lam_on_audioset_val_new2.csv",
    mini_data=True,
    llm="vicuna"
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # loads lam pre-trained model
    model, _, _ = load_model_and_preprocess(
        name="lam_vicuna_instruct",
        # NOTE: change to other weights
        model_type=("llama3_8b" if llm=='llama' else "vicuna1.5_7b-ft" ),
        is_eval=True,
        device=device,
    )

    model.load_from_pretrained(lam_ckpt_path)

    dataset = ObjevalDataset()

    results, mae = [], []
    with tqdm(total=10 if mini_data else len(dataset)) as pbar:
        for batch_idx, data in enumerate(dataset):
            output = model.generate(
                {
                    "audio": data["fbank"].unsqueeze(0).cuda(),
                    "prompt": data["question"], 
                },
                temperature=0.1,
            )
            output2 = model.generate(  # the semantic question
                {
                    "audio": data["fbank"].unsqueeze(0).cuda(),
                    "prompt": data["question2"], 
                },
                temperature=0.1,
            )
            
            print(output, output2)
            
            # calculate the MAE
            results.append((
                data['audio_path'],
                data['question'],
                output,
                extract_rating_from_text(output[0]),
                int(data['answer']),
                output2,
                data['answer2'],
                data["qidx"], data["qcategory"], 
                np.abs(extract_rating_from_text(output[0]) - data['answer'])))
            
            pbar.update(1)
            if mini_data and batch_idx == 10:
                break

    results = pd.DataFrame(results, 
                           columns=["audio_path", "question", "output", "output_rating", "gt", "verbal_output", "verbal_gt", "question_id", "question_category", "mae"])
    results.to_csv(results_path, index=False)
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_folder_name", type=str)
    parser.add_argument("--llm", type=str, default="vicuna")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    results_dir = "/data/home/acw630/WORKPLACE/LAM/engine/results/"
    os.makedirs(results_dir, exist_ok=True)

    mini_data = False

    checkpoint_paths = [
        load_latest_checkpoint(llm=args.llm, ckpt=args.ckpt)
    ]
    for lam_ckpt_path in checkpoint_paths:
        print(f"Using checkpoint {lam_ckpt_path}")
        pth = lam_ckpt_path.split("/")[-1].split(".")[0]
        results_path = os.path.join(results_dir, f"{pth}.csv")
        
        generate_answer_on_musicqa(
            lam_ckpt_path=lam_ckpt_path,
            results_path=results_path,
            mini_data=mini_data,
            llm=args.llm
        )
