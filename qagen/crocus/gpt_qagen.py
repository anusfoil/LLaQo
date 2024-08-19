import os, re, copy
import json
import openai
import pandas as pd
import tiktoken
from datetime import datetime
from tqdm import tqdm
import  hook
from start_message import *

# openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")


MODEL = 'gpt-4'

def gpt_api_stream(messages: list):
    r"""[stream responce] create response to the input message.

    Args:
        messages (list): completed conversation

    Returns:
        tuple: (results, error_desc)
    """

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,  # id
            messages=messages,
            stream=True,
        )
        completion = {'role': '', 'content': ''}
        for event in response:
            if event['choices'][0]['finish_reason'] == 'stop':
                # print(f'Received response data: {completion}')
                break
            for delta_k, delta_v in event['choices'][0]['delta'].items():
                # print(f'Stream response data: {delta_k} = {delta_v}')
                completion[delta_k] += delta_v

        # return completed chat and current completion
        return messages.append(completion), completion

    except Exception as err:
        raise Exception(f'OpenAI API exception: {err}')


def prompt_and_save():
    """prompt and save module for the crocus dataset."""

    feedback_dir = "/data/EECS-MachineListeningLab/datasets/LLaQo/crocus/Critique_Documents_Piano_Translated"
    audio_dir = "/data/EECS-MachineListeningLab/datasets/LLaQo/crocus/Performance_Records"

    info = pd.read_csv("/data/home/acw630/WORKPLACE/LAM/qagen/crocus/info.csv")

    qa_table = []
    for file in os.listdir(audio_dir):
        if not file.endswith(".wav"):
            continue
        
        info_row = info[info['track_name'] == file]
        # add the piece (recording) level questions
        qa_table.append(["N/A", file, "N/A", 
                            "What kind of performance might this be? ", 
                            "This is a student performance, but in advanced level."
                        ])   
        qa_table.append(["N/A", file, "N/A", 
                            "How would you rate the difficulty level of the piece, in a scale of 9? ", 
                            info_row['difficulty'].values[0]
                        ])
        qa_table.append(["N/A", file, "N/A", 
                            "Who might be the composer?", 
                            info_row['composer'].values[0]
                        ])  
    
    list_dir = os.listdir(feedback_dir)
    for idx in tqdm(range(0, len(list_dir), 4)):

        queries_list = []
        for j in range(0, min(4, len(list_dir)-idx)):
            if not list_dir[idx+j].endswith(".txt"):
                continue
            with open(os.path.join(feedback_dir, list_dir[idx+j]), "r") as f:
                feedback = f.read()
            queries_list.append(feedback)

            audio_file = list_dir[idx+j].split("_")[0] + ".wav"
            qa_table.append([list_dir[idx+j], audio_file, "N/A",
                            "Can you give an overall assessment of the student performance? Elaborate in detail? ", 
                            feedback])

        queries_str = "".join([f"{i}. {query} \n" for i, query in enumerate(queries_list)])

        query_message = {
            "role": "user",
            "content": "Here are the list of teacher feedbacks. Remember to use \n \
    all information given by the teacher, each teacher feedback should give 3 questions.  \n \
    Help me with all teacher feedbacks, there should be 4 of them to complete. \n" + queries_str
        }
        messages = START_MESSAGE + [query_message]
        encoding = tiktoken.encoding_for_model(MODEL)
        msg_token_len = [len(encoding.encode(msg['content'])) for msg in messages]

        response_path = f"responses/gpt_response_{idx}.txt"
        if os.path.exists(response_path):
            print("exists")
            with open(response_path, "r") as f:
                completion = f.read()
        else:
            _, response = gpt_api_stream(messages)
            completion = response['content']
            with open(response_path, "w") as f:
                f.write(completion)

        qa_pairs = parse_completion(completion)
        for j in range(0, min(4, len(list_dir)-idx)):
            
            audio_file = list_dir[idx+j].split("_")[0] + ".wav"
            qa_pairs_ = [[list_dir[idx+j], audio_file, queries_list[j]] + qa for qa in qa_pairs[j*3:j*3+3]]
            qa_table.extend(qa_pairs_)

    qa_table = pd.DataFrame(qa_table, columns=['feedback_file', 'audio_file', 'original_feedback', 'Q', 'A'])

    # divide train and test set
    qa_table['split'] = "train"
    test_list = ['n08-chp-pld007-s-p03.wav', 'n01-chp-etd003-s-p02.wav', 'n22-bac-inv015-s-p01.wav', 'n28-bet-snt008-3-s-p02.wav', 'n32-moz-snt331-1-h-p01.wav']
    qa_table.loc[qa_table['audio_file'].isin(test_list), 'split'] = 'test'

    qa_table.to_csv("evaluation_qa.csv")
    qa_table.to_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/crocus/evaluation_qa.csv")
    return 


def parse_completion(completion):
    """parse the response into a series of QA pairs"""

    rater_blocks = re.split(r'\n\n|\n\s*\n', completion)
    rater_blocks = [rb for rb in rater_blocks if "Q:" in rb]
    # assert(len(rater_blocks) == 4)
    qa_pairs = []
    for idx, block in enumerate(rater_blocks):
        potential_qs = block.split("Q:")
        qa = [q.split("A:") for q in potential_qs if "A:" in q]
        qa_pairs.extend([[q.split("\n")[0], a.split("\n")[0]] for q, a in qa])

    return qa_pairs



if __name__ == "__main__":

    prompt_and_save()
