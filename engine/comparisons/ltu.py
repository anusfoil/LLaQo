import os, sys, re
from gradio_client import Client
import pandas as pd
from tqdm import tqdm
import numpy as np
sys.path.append("../../src/lavis")

from lavis.datasets.datasets.objeval_qa import ObjevalDataset


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
                data['audio_path'],
                "",  
                data['question'],
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
                "question_id": data["qidx"], 
                "question_category": data["qcategory"],
                "mae": mae_value
            })

            pbar.update(1)


    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)

# Example usage, assuming you have a client object set up
client = Client("https://yuangongfdu-ltu-2.hf.space/")
generate_answer_on_musicqa(client, results_path="../results/ltu_results.csv")


