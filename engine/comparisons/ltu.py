import os, sys, re, signal
from gradio_client import Client
import pandas as pd
from tqdm import tqdm
import numpy as np
sys.path.append("../../src/lavis")

from lavis.datasets.datasets.objeval_qa import ObjevalDataset
from lavis.datasets.datasets.cipi_qa import CIPIDataset
from lavis.datasets.datasets.techniques_qa import TechniquesDataset


def extract_rating_from_text(text):
    """
    Extracts the first occurring numeric rating from a text string.

    Args:
    text (str): A string that contains a numeric rating.

    Returns:
    int or None: The first found integer in the text, or None if no integer is found.
    """
    # Regular expression to find digits
    match = re.search(r'\b\d+\b', text)
    if match:
        return int(match.group(0))  # Convert found number to integer
    else:
        return None  # Return None if no digits found


def generate_answer_on_musicqa(
    client,  # Include the client as a parameter
    results_path=None
):
    dataset = ObjevalDataset()

    results = []
    with tqdm(len(dataset)) as pbar:
        for batch_idx, data in enumerate(dataset):

            result = client.predict(
                "",
                data['audio_path'],
                "You are a piano teacher. " + data['question'],
                "7B (Default)",
                api_name="/predict"
            )
            result2 = client.predict(
                "",
                data['audio_path'],
                "You are a piano teacher. " + data['question2'],
                "7B (Default)",
                api_name="/predict"
            )
            
            rating = extract_rating_from_text(result)
            ground_truth = int(data['answer'])

            if isinstance(rating, int):
                mae_value = np.abs(rating - ground_truth)
            else:
                mae_value = None

            results.append({
                "audio_path": data['audio_path'],
                "question": data['question'],
                "response": result,
                "rating": rating,
                "gt": ground_truth,
                "verbal_output": result2,
                "verbal_gt": data['answer2'],
                "question_id": data["qidx"], 
                "question_category": data["qcategory"],
                "mae": mae_value
            })

            pbar.update(1)


    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)


def generate_answer_on_cipi(client, results_path=None):
    dataset = CIPIDataset(split="val")

    results = []
    
    # read existing
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        results = results_df.to_dict(orient="records")
    
    with tqdm(len(dataset)) as pbar:
        for batch_idx, data in enumerate(dataset):
            if batch_idx % 3 != 1:
                continue  # Only process every third batch for difficulty

            try:
                result = client.predict(
                    data['audio_path'],
                    data['question'],
                    # "7B (Default)",
                    api_name="/predict"
                )
            except Exception as e:
                print(f"Error on {data['audio_path']}: {e}")
                result = ""
            
            print(result)
            ground_truth = data['answer']

            results.append({
                "audio_path": data['audio_path'],
                "question": data['question'],
                "response": result,
                "gt": ground_truth,
            })

            pbar.update(1)

            results_df = pd.DataFrame(results)
            if results_path:
                results_df.to_csv(results_path, index=False)


def generate_answer_on_techniques(
    client,  # Include the client as a parameter
    results_path=None
):
    dataset = TechniquesDataset()

    results = []
    with tqdm(len(dataset)) as pbar:
        for batch_idx, data in enumerate(dataset):

            result = client.predict(
                "",
                data['audio_path'],
                data['question'],
                "7B (Default)",
                api_name="/predict"
            )
            
            ground_truth = data['answer']

            results.append({
                "audio_path": data['audio_path'],
                "question": data['question'],
                "response": result,
                "gt": ground_truth,
            })

            pbar.update(1)


    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)




# Example usage, assuming you have a client object set up
client = Client("https://yuangongfdu-ltu.hf.space/")
# generate_answer_on_musicqa(client, results_path="../results/ltu_results.csv")
generate_answer_on_cipi(client, results_path="../results/ltu_results_cipi_.csv")
# generate_answer_on_techniques(client, results_path="../results/ltu_results_techniques.csv")


