import os
# import sys
#
# ROOT_PATH = os.path.abspath(__file__)[:27]
# sys.path.append(os.path.join(ROOT_PATH, 'gpt4free'))
# from gpt4free.deepai import Completion

# def text_completion(prompt: str, verbose: bool = False) -> str:
#     chunks = Completion.create(prompt)
#     response = "".join([item for item in chunks])

#     if verbose:
#         print(response, end="\n", flush=True)

#     return response
import openai

# openai.log = "debug"
openai.api_base = "https://api.chatanywhere.com.cn/v1"
openai.api_key = os.getenv("OPENAI_API_KEY")

# This is free key - JL
# openai.api_key = "sk-WYjU1Vywr9KZieZjkaZOYufb469zIGRwjPrL2DMEouwIRoUw"
# This is free key - H
# openai.api_key = "sk-oIEUUDxLwF3dD9a0RSjcFH3X2yDaMFNeVP1KDj2CsyJFcgpp"
# This is free key - JW
# openai.api_key = "sk-HXyKtM4fi372BpuCQlDdGLf2E5oXHx8X3P25wdTxSC4YChuC"

# This is paid key
# openai.api_key = "sk-96zjvHwP9YfMjTHHLQwKquzBLksLZwYDC5YFsUDzhyq6y7wc"
# non-stream responce
# completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world!"}])
# print(completion.choices[0].message.content)


def gpt_api_stream(messages: list,
                   model_name: str = 'gpt-3.5-turbo-0613'
                   ):  # gpt-3.5-turbo-16k, gpt-3.5-turbo-0613
    r"""[stream responce] create response to the input message.

    Args:
        messages (list): completed conversation

    Returns:
        tuple: (results, error_desc)
    """
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
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


def main():
    cap_prompt = "Based on the following audio clip, generate 5 different sentences to describe the audio clip in the scene. The following information is provided: the time stamps and a description of the current frames. Start each caption with '#C'. The generated descriptions should cover all possible sound events and can be derived from the audio only, e.g., sound events appearing in the audio, together with its acoustic features and corresponding time stamps, and the temporal relationship between the sound events. The more detailed and diverse the descriptions, the better."
    qa_prompt = "Based on the following audio clip, generate 10 different types of complex open-ended questions that require step-by-step thinking, and corresponding answers. The following information is provided: the time stamps and  a description of the current frames.  Start each question with '#Q' and each answer with '#A'. Questions should be about the audio and the answer should be derived from the audio only, e.g., which sound event is recognized and why (e.g., based on its acoustic feature), what can be inferred based on the combination of sound events; the temporal relationship between the sound events and what can be inferred from that; the potential scenario that such an audio clip could happen, if the audio clip is special (e.g., urgent, funny, interesting, abnormal, unique, etc) and why, what mood or atmosphere this audio clip conveys, etc. The more complex and diverse the question, the better."
    input = [
        "In the image, a group of musicians is performing on stage. They are dressed in cowboy attire and playing various musical instruments, including an accordion, a guitar, and a drum. The performance is taking place in a dimly lit room with purple lighting, adding to the atmosphere of the event. The musicians are energetically playing their instruments, creating a lively and engaging performance for the audience.",
        "In the image, there is a man wearing a cowboy hat and playing an accordion on stage. He is surrounded by other musicians who are also playing various instruments, such as guitars, drums, and a saxophone. The musicians are performing in a dimly lit room, creating a lively atmosphere for the audience.",
        "In the image, there is a man wearing a cowboy hat and playing an accordion on stage. He is accompanied by two other musicians, one playing a guitar and the other holding a microphone. The three musicians are performing in a dimly lit room with purple lighting, creating a unique atmosphere for their musical performance.",
        "Three men in cowboy hats are performing on stage, playing musical instruments such as an accordion and a guitar. They are standing in front of a purple-colored backdrop, which creates a dramatic atmosphere for the performance. The musicians are dressed in cowboy attire, adding to the western theme of the event.",
        "In the image, there are three men dressed in cowboy attire and playing musical instruments on stage. Two of them are holding accordions, while the third one is playing a saxophone. The musicians are performing together on stage, creating a lively and energetic atmosphere.",
        "In the image, there are two men wearing cowboy hats and playing musical instruments on stage. One man is holding an accordion, while the other is playing a guitar. They are both performing on stage in front of a microphone, creating a lively and energetic atmosphere. Additionally, there are two microphones visible in the scene, likely used to amplify the sound of the musicians during their performance.",
        "In the scene, there are two men wearing cowboy hats and playing musical instruments on stage. One man is holding an accordion, while the other is playing a saxophone. Additionally, there is a microphone on the stage, likely used by the musicians to amplify their sound. The setting appears to be a concert or performance venue, where the musicians are showcasing their talents and entertaining the audience.",
        "In the image, there are two men wearing cowboy hats and playing musical instruments on stage. One of them is holding an accordion, while the other is playing a guitar. They are standing in front of a microphone, possibly singing or performing a song. Additionally, there are two microphones placed on the stage, likely used for amplifying their voices during the performance. The stage is illuminated by a spotlight, creating a dramatic atmosphere for the musicians to showcase their talents.",
        "In the scene, there are two men wearing cowboy hats and playing musical instruments on stage. One man is holding an accordion, while the other is playing a guitar. They are both performing together, creating a lively and energetic atmosphere. Additionally, there is a microphone in the scene, possibly used by one of the musicians to amplify their sound. The setting appears to be a concert or performance venue, where the musicians are showcasing their talents and entertaining the audience.",
        "In the image, a group of musicians is performing on stage, playing various instruments such as an accordion, a guitar, and a saxophone. The musicians are dressed in cowboy attire, adding to the western-themed atmosphere of the performance. The stage is illuminated by purple lights, creating a vibrant and energetic atmosphere for the band's performance."
    ]
    prompt_input = [
        f"[{idx}s]" + sentence for idx, sentence in enumerate(input)
    ]
    prompt_input = cap_prompt + " ".join(prompt_input)
    print(prompt_input)

    messages = [
        {
            'role': 'user',
            'content': prompt_input
        },
    ]
    gpt_api_stream(messages)
    print(messages[-1]['content'])

    # text_completion(prompt_input, verbose=True)


if __name__ == "__main__":
    main()
