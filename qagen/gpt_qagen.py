import os
import json
import openai
import pandas as pd
import tiktoken
import hook
from start_message import *

# openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")


MODEL = 'gpt-4'

def gpt_api_stream(messages: list, chatgpt_engine: str = 'gpt-3.5'):
    r"""[stream responce] create response to the input message.

    Args:
        messages (list): completed conversation

    Returns:
        tuple: (results, error_desc)
    """

    try:
        response = openai.ChatCompletion.create(
            model=chatgpt_list[chatgpt_engine][0],  # id
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


def get_list_queries():
    ANSWERS_CSV = '/data/EECS-MachineListeningLab/datasets/expert_novice/evaluation_data_anonymous.csv'

    answers_csv = pd.read_csv(ANSWERS_CSV)
    queries_list = answers_csv['Instructor2_text']
    queries_list = "".join([f"{i}. {query} \n" for i, query in enumerate(queries_list.to_list()[:20])])
    return queries_list


if __name__ == "__main__":

    queries_list = get_list_queries()
    query_message = {
        "role": "user",
        "content": "Here are the list of teacher feedbacks. Remember to use \n \
all information given by the teacher, each teacher feedback should give 3 to 4 questions.  \n \
Help me with all teacher feedbacks and don't omit information. \n" + queries_list
    }
    messages = START_MESSAGE + [query_message]
    encoding = tiktoken.encoding_for_model(MODEL)
    msg_token_len = [len(encoding.encode(msg['content'])) for msg in messages]

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
    )
    with open("gpt_response.txt", "w") as f:
        f.write(response['choices'][0]['message']['content'])
    hook()
