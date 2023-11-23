import os, sys, json
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append("../src/lavis")

from lavis.datasets.datasets.conespressione_qa import ConEspressioneDataset
from lavis.datasets.datasets.expertnovice_qa import ExpertNoviceDataset
# from utilities import collate_func, set_logger, write_json, collate_func

from lavis.models import load_model_and_preprocess

# log = set_logger(__name__)


def generate_answer_on_musicqa(
    lam_ckpt_path="",
    results_json_path="/data/home/acw630/WORKPLACE/LAM/engine/results/lam_on_audioset_val_new2.json",
    mini_data=True,
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # loads lam pre-trained model
    model, _, _ = load_model_and_preprocess(
        name="lam_vicuna_instruct",
        # NOTE: change to other weights
        model_type="vicuna1.5_7b-ft", 
        is_eval=True,
        device=device,
    )

    model.load_from_pretrained(lam_ckpt_path)

    # dataset = ConEspressioneDataset()
    dataset = ExpertNoviceDataset(split='test')

    results = []
    with tqdm(total=10 if mini_data else len(dataset)) as pbar:
        for batch_idx, data in enumerate(dataset):
            # NOTE: change to gererate_new if bos is in the first position.
            # Before 9/1 use `generate` method
            # output = model.generate_new(
            #     {
            #         "audio": data["fbank"].unsqueeze(0).cuda(),
            #         "prompt": data["question"],
            #     },
            #     temperature=0.1,
            # )
            output = model.generate_new(
                {
                    "audio": data["fbank"].unsqueeze(0).cuda(),
                    "prompt": data["question"],  # "Describe the sound events",
                },
                temperature=0.1,
            )
            result = {
                "audio_path": data['audio_path'],
                "question": data['question'],
                "output": output, 
                "gt": data['answer']
            }
            print(result)
            results.append(result)

            pbar.update(1)
            if mini_data and batch_idx == 10:
                break

    with open(results_json_path, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_folder_name", type=str)
    args = parser.parse_args()

    results_json_dir = "/data/home/acw630/WORKPLACE/LAM/engine/results/tagging/ablate"
    os.makedirs(results_json_dir, exist_ok=True)

    mini_data = False
    normalize_sim = False

    # lam_ckpt_dir = (
    #     "/data/EECS-MachineListeningLab/huan/lam/check_point/Pretrain_stage2"
    # )
    # ckpt_foldername = args.ckpt_folder_name  # "20230829124"
    # checkpoint_paths = glob(os.path.join(lam_ckpt_dir, ckpt_foldername, "*.pth"))
    checkpoint_paths = [
        "/data/EECS-MachineListeningLab/huan/lam/check_point/Pretrain_stage2/test_musicqa/20231103032/checkpoint_51000.pth",
    ]
    for lam_ckpt_path in checkpoint_paths:
        pth = lam_ckpt_path.split("/")[-1].split(".")[0]
        results_json_path = os.path.join(results_json_dir, f"{pth}.json")
        generate_answer_on_musicqa(
            lam_ckpt_path=lam_ckpt_path,
            results_json_path=results_json_path,
            mini_data=mini_data,
        )