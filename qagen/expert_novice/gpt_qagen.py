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

DIFFICULTY_DICT = {
    "Careless Love": 3,
    "Cielito Lindo": 3,
    "Lavenders Blue": 3,
    "Over the Waves": 3,
    "She Wore a Yellow Ribbon": 4,
    "The Blues": 2,
    "The Entertainer": 3,
}

CONTENT_DICT = {
    "Careless Love": "This is a traditional song, adapted in simple melody and chords accompaniment.  ",
    "Cielito Lindo": "This is a folk song, adapted in simple melody and chords accompaniment. There are some syncopated rhythm that might be challenging. ",
    "Lavenders Blue": "This is a folk song, adapted in simple melody and chords accompaniment. ",
    "Over the Waves": "This is an easy folk song, adapted in simple melody and chords accompaniment. The rhythms might be challenging.",
    "She Wore a Yellow Ribbon": "This is a traditional song. ",
    "The Blues": "This is an easy piece in the style of Blues. ",
    "The Entertainer": "This is an easy piano version of a famous tune. The swing rhythm are difficult. ",
}


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
    """prompt and save module for the expert novice dataset."""
    ANSWERS_CSV = '/data/EECS-MachineListeningLab/datasets/LLaQo/expert_novice/evaluation_data_anonymous.csv'

    answers_csv = pd.read_csv(ANSWERS_CSV)

    qa_table = []
    for id, row in tqdm(answers_csv.iterrows()):

        # add the piece (recording) level questions
        qa_table.append([row['Piece_name'], row['Recording_number'], 'N/A', -1, 
                         "What kind of performance might this be? ", 
                         "This is a student performance."
                        ])   
        qa_table.append([row['Piece_name'], row['Recording_number'], 'N/A', -1, 
                         "How would you rate the difficulty level of the piece, in a scale of 9? ", 
                         str(DIFFICULTY_DICT[row['Piece_name']])
                        ])
        qa_table.append([row['Piece_name'], row['Recording_number'], 'N/A', -1, 
                         "What kind of music is this piece and what's difficult about it?", 
                         CONTENT_DICT[row['Piece_name']]
                        ])         

        queries_list = []
        for idx in range(1, 5):
            queries_list.append(answers_csv[f'Instructor{idx}_text'][id])
            queries_list.append(answers_csv[f'Rater{idx}_text'][id])

            # add the rating & overall text question
            qa_table.append([row['Piece_name'], row['Recording_number'], row[f'Instructor{idx}_rating'], 
                            idx, "What is the overall score you would assign to the performance, in a scale of 5? ", 
                            str(row[f'Instructor{idx}_rating'])])
            qa_table.append([row['Piece_name'], row['Recording_number'], row[f'Rater{idx}_rating'], 
                            idx, "What is the overall score you would assign to the performance, in a scale of 5? ", 
                            str(row[f'Instructor{idx}_rating'])])
            qa_table.append([row['Piece_name'], row['Recording_number'], row[f'Instructor{idx}_text'], 
                            idx, "Can you give an overall assessment of the student performance? Elaborate in detail? ", 
                            row[f'Instructor{idx}_text']])
            qa_table.append([row['Piece_name'], row['Recording_number'], row[f'Rater{idx}_text'], 
                            idx, "Can you give an overall assessment of the student performance? Elaborate in detail? ", 
                            row[f'Instructor{idx}_text']])

        queries_str = "".join([f"{i}. {query} \n" for i, query in enumerate(queries_list)])

        query_message = {
            "role": "user",
            "content": "Here are the list of teacher feedbacks. Remember to use \n \
    all information given by the teacher, each teacher feedback should give 2 to 4 questions.  \n \
    Help me with all teacher feedbacks, there should be 8 of them to complete. \n" + queries_str
        }
        messages = START_MESSAGE + [query_message]
        encoding = tiktoken.encoding_for_model(MODEL)
        msg_token_len = [len(encoding.encode(msg['content'])) for msg in messages]

        response_path = f"responses/gpt_response_{id}.txt"
        if os.path.exists(response_path):
            with open(response_path, "r") as f:
                completion = f.read()
        else:
            _, response = gpt_api_stream(messages)
            completion = response['content']
            with open(response_path, "w") as f:
                f.write(completion)

        qa_pairs = parse_completion(completion)
        qa_pairs = [[row['Piece_name'], row['Recording_number'], queries_list[qa[0]]] + qa for qa in qa_pairs]
        qa_table.extend(qa_pairs)

    qa_table = pd.DataFrame(qa_table, columns=['piece_name', 'recording_number', 'original_feedback', 'assessor_idx', 'Q', 'A'])

    # divide train and test set
    qa_table['split'] = "train"
    qa_table.loc[qa_table['recording_number'] == 1, 'split'] = 'test'

    qa_table.to_csv("evaluation_qa.csv")
    qa_table.to_csv("/data/EECS-MachineListeningLab/datasets/LLaQo/expert_novice/evaluation_qa.csv")
    return 


def parse_completion(completion):
    """parse the response into a series of QA pairs"""

    rater_blocks = re.split(r'\n\n|\n\s*\n', completion)
    rater_blocks = [rb for rb in rater_blocks if "Q:" in rb]
    assert(len(rater_blocks) == 8)
    qa_pairs = []
    for idx, block in enumerate(rater_blocks):
        potential_qs = block.split("Q:")
        qa = [q.split("A:") for q in potential_qs if "A:" in q]
        qa_pairs.extend([[idx, q.split("\n")[0], a.split("\n")[0]] for q, a in qa])

    return qa_pairs



if __name__ == "__main__":

    prompt_and_save()
