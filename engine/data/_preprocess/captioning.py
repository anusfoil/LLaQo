import os
import time
import torch
import argparse
from tqdm import tqdm
import sys

from prompt_gpt import gpt_api_stream

sys.path.append("..")
from dataloader import AVDataset

sys.path.append("../../../src")
from utils import write_json, read_json

audio_conf = {
    'num_mel_bins': 128,
    'target_length': 1024,
    'freqm': 0,
    'timem': 0,
    'mixup': 0.0,
    'dataset': "audioset",
    'mode': 'train',
    'mean': -5.081,
    'std': 4.4849,
    'noise': True,
    'label_smooth': 0,
    'im_res': 224
}

cap_instruction = "Based on the following audio clip, generate 5 different sentences to describe the audio clip in the scene. The following information is provided: the time stamps and a description of the current frames. Start each caption with '#C'. The generated descriptions should cover all possible sound events and can be derived from the audio only, e.g., sound events appearing in the audio, together with its acoustic features and corresponding time stamps, and the temporal relationship between the sound events. The more detailed and diverse the descriptions, the better."
qa_instruction = "Based on the following audio clip, generate 10 different types of complex open-ended questions that require step-by-step thinking, and corresponding answers. The following information is provided: the time stamps and  a description of the current frames.  Start each question with '#Q' and each answer with '#A'. Questions should be about the audio and the answer should be derived from the audio only, e.g., which sound event is recognized and why (e.g., based on its acoustic feature), what can be inferred based on the combination of sound events; the temporal relationship between the sound events and what can be inferred from that; the potential scenario that such an audio clip could happen, if the audio clip is special (e.g., urgent, funny, interesting, abnormal, unique, etc) and why, what mood or atmosphere this audio clip conveys, etc. The more complex and diverse the question, the better."

batched_cap_instruction = {
    "prefix":
    "You are an AI hearing assistant. You are describing a list of audios. For each audio, generate 5 different sentences to describe the sounds in the scene. You will get chronological descriptions of frames in the video. The following information is provided: the time stamps and descriptions of the current frames. The generated descriptions should cover all possible sound events and be clearly audible in the recording, e.g., sound events appearing in the audio, together with its acoustic features and corresponding time stamps, and the temporal relationship between the sound events. The more detailed and diverse the descriptions, the better. Start each generated description with \"#C\".\n",
    # "You are an AI hearing assistant and you are describing a list of audios. For each audio, you have chronological descriptions of frames in the video and generate 5 different sentences to describe the sounds in the scene. The following information is provided: the time stamps and descriptions of the current frames. Start each caption with \"#C\". The generated descriptions should cover all possible sound events and be clearly audible in the recording, e.g., sound events appearing in the audio, together with its acoustic features and corresponding time stamps, and the temporal relationship between the sound events. The more detailed and diverse the descriptions, the better.\n",  # 2023.7.6
    # "You are an AI hearing assistant and you are describing a list of audios. For each audio, you have chronological descriptions of frames in the video and generate 5 different sentences to describe the sounds in the scene. The following information is provided: the time stamps and descriptions of the current frames. Start each caption with \"#C\". The generated descriptions should cover all possible sound events and the generated descriptions should be clearly audible in the recording, e.g., sound events appearing in the audio, together with its acoustic features and corresponding time stamps, and the temporal relationship between the sound events. The more detailed and diverse the descriptions, the better.\n",  # 2023.7.6
    # "You are an AI hearing assistant and you are describing a list of audios. For each audio, you have chronological descriptions of frames in the video and generate 5 different sentences to describe the sounds in the scene. The following information is provided: the time stamps and descriptions of the current frames. Start each caption with \"#C\". The generated descriptions should cover all possible sound events and one can hear the content in the audio that the descriptions depict confidently, e.g., sound events appearing in the audio, together with its acoustic features and corresponding time stamps, and the temporal relationship between the sound events. The more detailed and diverse the descriptions, the better.\n",  # by 2023.7.6
    # "You are an AI hearing assistant and you are describing a list of audios. For each audio, you get 3 frame descriptions of the corresponding video and generate 5 different sentences to describe the sounds in the scene. The following information is provided: the time stamps and a description of the current frames. Start each caption with \"#C\". The generated descriptions should cover all possible sound events and one can hear the content in the audio that the descriptions depict confidently, e.g., sound events appearing in the audio, together with its acoustic features and corresponding time stamps, and the temporal relationship between the sound events. The more detailed and diverse the descriptions, the better.\n",
    # "You are an AI hearing assistant and you are describing a list of audios. For each audio, you have frame descriptions of the corresponding video. The following information is provided: the time stamps and a description of the current frames. Generate 5 different sentences to describe the sounds in the scene. Start each description with \"#C\". The generated descriptions should cover all possible sound events and one can hear the content in the audio that the descriptions depict confidently, e.g., sound events appearing in the audio, together with its acoustic features and corresponding time stamps, and the temporal relationship between the sound events. The more detailed and diverse the descriptions, the better.\n",
    "head":
    "#Audio{}\n",
    "surfix":
    "Each audio starts with \"#Audio\". Process each audio separately and output audio descriptions only.",
}

# def text_completion(prompt: str, verbose: bool = False) -> str:
#     chunks = Completion.create(prompt)
#     response = "".join([item for item in chunks])

#     if verbose:
#         print(response, end="\n", flush=True)

#     return response


def caption_frames(args):
    sys.path.append("../../../src/lavis")
    from lavis.models import load_model_and_preprocess

    assert args.n_captions_per_image == 1, r"fix this parameter when extracting video captions per frames."
    os.makedirs(args.output_dir, exist_ok=True)
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    enable_amp = device != torch.device("cpu")

    prompt = "Based on the image, describe all possible sounds in the scene. More precise, the better."

    # load sample image
    dataset = AVDataset(dataset_json_file=args.json_path,
                        audio_conf=audio_conf,
                        label_csv=None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # fix batch size to 1, but it won't affect data parallel
        pin_memory=True,
        num_workers=4,
    )

    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(
        name='blip2_vicuna_instruct',  # "blip2_t5",
        model_type='vicuna7b',  # "pretrain_flant5xl",
        is_eval=True,
        device=device)

    # do the inference
    batch_data, batch_json_paths, batch_videos = [], [], []
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        if args.mini_data and batch_idx == 8:
            break

        cache_outputs = []
        datum, _, videos = batch
        output_json_path = os.path.join(
            args.output_dir,
            f"{datum['filename'][0]}.json")  # one file per batch

        # Skip if json file exists already, default not to overwirte
        if not args.overwrite and os.path.isfile(output_json_path):
            continue
        else:
            batch_data.append(datum)
            batch_json_paths.append(output_json_path)
            batch_videos.append(videos)

        if (len(batch_data) == args.batch_size) or (batch_idx + 1
                                                    == len(dataloader)):
            batch_videos = torch.cat(batch_videos, dim=0)
            for id in range(batch_videos.size(1)):
                outputs = model.generate(
                    {
                        "image": batch_videos[:, id, :, :, :].to(
                            device
                        ),  # .to(torch.bfloat16) if enable_amp else videos,
                        "prompt":
                        prompt,  # in the video;  Based on the image, use a few words to describe what you will hear
                    },
                    use_nucleus_sampling=False,
                    num_beams=5,
                    max_length=256,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_captions=args.n_captions_per_image,
                    temperature=1,
                )
                cache_outputs.append(
                    outputs)  # (psydo) shape: (n_frames, batch_size)

            # (psydo) shape expected to be: (batch_size, n_frames)
            captions = []
            for idx in range(len(batch_data)):
                captions.append([caps[idx] for caps in cache_outputs])

            for datum, output_json_path, caps in zip(batch_data,
                                                     batch_json_paths,
                                                     captions):
                filename = datum['filename'][0]
                audio_path = datum['audio_path'][0]

                # print(f"{filename}-{audio_path}:\n{caps}", flush=True)
                write_json(
                    {
                        "filename": filename,
                        "audio_path": audio_path,
                        "caption_per_frame": caps,
                    }, output_json_path)

            batch_data, batch_json_paths, batch_videos = [], [], []


def _instruct_audio(datum: dict, instruction_type: str, output_json_path: str):
    r"""`datum` contains meta info, including `caption_per_frame`.
        Return start time."""
    assert instruction_type in [
        "caption", "qa"
    ], "`instruction_type` should be either 'caption' or 'qa'"

    key = "audio_qa" if instruction_type == "qa" else "audio_caption"
    instruction = cap_instruction if instruction_type == "caption" else qa_instruction

    prompt = [
        f"[{i}s]" + cap for i, cap in enumerate(datum['caption_per_frame'])
    ]
    prompt = " ".join(prompt)

    messages = [
        {
            'role': 'user',
            'content': (instruction + prompt)
        },
    ]
    print(f'instruction: {messages}', flush=True)

    # audio_caption = text_completion((instruction + prompt),
    #                                 verbose=False)
    start_time = time.time()
    _, completion = gpt_api_stream(messages)
    print(f"caption {datum['filename']}: {completion['content']}", flush=True)

    write_json({
        key: completion['content'],
        **datum,
    }, output_json_path)

    return start_time


def caption_audios(args):
    # Load sample image
    dataset = AVDataset(dataset_json_file=args.json_path,
                        audio_conf=audio_conf,
                        label_csv=None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
    )

    # do the inference
    timer = 0  # process captioning audio in batch
    cache_datum = []
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        if args.mini_data and batch_idx == 5:
            break

        filename = batch[0]['filename'][0]  # one file per batch
        output_json_path = os.path.join(args.output_dir, f"{filename}.json")
        datum = read_json(output_json_path)
        # print(datum['filename'])

        # Skip if datum has audio caption already, default not to overwirte
        if not args.overwrite and 'audio_caption' in datum:
            continue

        start_time = _instruct_audio(datum=datum,
                                     instruction_type="caption",
                                     output_json_path=output_json_path)

        end_time = time.time()
        # Sleep time should be larger than minimal interval to avoid banned
        sleep_time = args.min_interval + 1 - end_time + start_time
        try:
            time.sleep(sleep_time)
        except ValueError:
            continue


def split_batch_response(response: str):
    return [m[1:].strip('\n') for m in response.split("#Audio")[1:]]


def batch_prompt_chatgpt(batch_data: dict, instruction_type: str,
                         batch_json_paths: str):
    r"""Generate a batch of audio description or Q&A by prompting ChatGPT.
        Usage:
            ```
            datum = read_json(filename)
            batch_data = [datum]
            _prompt_chatgpt(batch_data, instruction_type, batch_json_paths)
            ```
        Return start time."""
    assert instruction_type in [
        "caption"
    ], "`instruction_type` should be 'caption' for now"

    key = "audio_qa" if instruction_type == "qa" else "audio_caption"
    instruction = batched_cap_instruction if instruction_type == "caption" else null  # TODO: batched_qa_instruction

    descriptions, _vars = [], []
    for idx, datum in enumerate(batch_data):
        head = instruction["head"].format((idx + 1))

        des = [
            f"[{i}s]" + cap for i, cap in enumerate(datum['caption_per_frame'])
            if i % 4 == 1
        ]  #
        des = head + " ".join(des)
        descriptions.append(des)

    descriptions = "\n".join(descriptions)
    content = instruction['prefix'] + descriptions + "\n" + instruction[
        'surfix']

    messages = [
        {
            'role': 'user',
            'content': content
        },
    ]
    print(f'instruction: {messages}')

    start_time = time.time()
    _, completion = gpt_api_stream(messages)
    print(f"{completion['content']}", flush=True)

    batch_responses = split_batch_response(completion['content'])

    for json_path, datum, response in zip(batch_json_paths, batch_data,
                                          batch_responses):
        print(f"{datum['filename']}-{json_path}: {response}")
        write_json({
            key: response,
            **datum,
        }, json_path)

    return start_time


def caption_batched_audios(args):
    # Load sample image
    dataset = AVDataset(dataset_json_file=args.json_path,
                        audio_conf=audio_conf,
                        label_csv=None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
    )

    # do the inference
    batch_data, batch_json_paths = [], []
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        if args.mini_data and batch_idx == 10:
            break

        filename = batch[0]['filename'][0]  # one file per batch
        output_json_path = os.path.join(args.output_dir, f"{filename}.json")
        datum = read_json(output_json_path)
        # print(datum['filename'])

        # Skip if datum has audio caption already, default not to overwirte
        if not args.overwrite and 'audio_caption' in datum:
            continue
        else:
            batch_data.append(datum)
            batch_json_paths.append(output_json_path)

        if len(batch_data) == args.batch_size:
            start_time = batch_prompt_chatgpt(
                batch_data=batch_data,
                instruction_type="caption",
                batch_json_paths=batch_json_paths)
            batch_data = []
            batch_json_paths = []

            end_time = time.time()
            # Sleep time should be larger than minimal interval to avoid banned
            sleep_time = args.min_interval + 1 - end_time + start_time
            try:
                time.sleep(sleep_time)
            except ValueError:
                continue


def generate_qa(args):
    # Load sample image
    dataset = AVDataset(dataset_json_file=args.json_path,
                        audio_conf=audio_conf,
                        label_csv=None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
    )

    # do the inference
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        if args.mini_data and batch_idx == 1:
            break

        filename = batch[0]['filename'][0]
        output_json_path = os.path.join(args.output_dir, f"{filename}.json")
        datum = read_json(output_json_path)

        # Skip if datum has audio caption already, default not to overwirte
        if not args.overwrite and 'audio_qa' in datum:
            continue

        start_time = _instruct_audio(datum=datum,
                                     instruction_type="qa",
                                     output_json_path=output_json_path)

        end_time = time.time()
        # Sleep time should be larger than minimal interval to avoid banned
        sleep_time = args.min_interval + 1 - end_time + start_time
        try:
            time.sleep(sleep_time)
        except ValueError:
            continue


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_frame_captioning = subparsers.add_parser('frame_captioning')
    parser_frame_captioning.add_argument("--output_dir", type=str)
    parser_frame_captioning.add_argument("--json_path", type=str)
    parser_frame_captioning.add_argument("--batch_size", type=int, default=1)
    parser_frame_captioning.add_argument("--n_captions_per_image",
                                         type=int,
                                         default=1)
    parser_frame_captioning.add_argument("--mini_data",
                                         action="store_true",
                                         default=False)
    parser_frame_captioning.add_argument("--overwrite",
                                         action="store_true",
                                         default=False)

    parser_audio_captioning = subparsers.add_parser('audio_captioning')
    parser_audio_captioning.add_argument("--output_dir", type=str)
    parser_audio_captioning.add_argument("--json_path", type=str)
    parser_audio_captioning.add_argument("--batch_size", type=int, default=4)
    parser_audio_captioning.add_argument("--min_interval",
                                         type=int,
                                         default=31)
    parser_audio_captioning.add_argument("--mini_data",
                                         action="store_true",
                                         default=False)
    parser_audio_captioning.add_argument("--overwrite",
                                         action="store_true",
                                         default=False)

    parser_audio_qa_generation = subparsers.add_parser('qa_generations')
    parser_audio_qa_generation.add_argument("--output_dir", type=str)
    parser_audio_qa_generation.add_argument("--json_path", type=str)
    parser_audio_qa_generation.add_argument("--min_interval",
                                            type=int,
                                            default=30)
    parser_audio_qa_generation.add_argument("--mini_data",
                                            action="store_true",
                                            default=False)
    parser_audio_qa_generation.add_argument("--overwrite",
                                            action="store_true",
                                            default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == 'frame_captioning':
        caption_frames(args)

    elif args.mode == 'audio_captioning':
        # caption_audios(args)
        caption_batched_audios(args)

    else:
        generate_qa(args)
