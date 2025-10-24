import argparse
import torch
import json
import os
from models import model_loader
import re
from utils import obtain_prompt, speech_info, voice_info
from tqdm import tqdm

def main(args):

    # 1. load data

    # 1-1. set dir
    input_dir = os.path.join(args.root_dir, f"dataset/data_{args.data_type}")
    output_dir = os.path.join(args.root_dir, f"output/output_{args.data_type}", 'asr_output', args.model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1-2 load data
    with open(os.path.join(input_dir, f"script_info_{args.data_type}.json"), 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    # 1-3. load model
    model = model_loader.get_model_class(args.model_name)(args)

    # 1-4. load saved file if exist
    output_file_path = os.path.join(output_dir, f"{args.model_name}_asr.json")
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)


    # 2. inference
    for d in tqdm(dataset):
        d['model']=args.model_name

        ty = "target"  #### only asr for target voice type

        # 2-1. obtain input audio file
        input_audio_file = os.path.join(input_dir, "audio", d[f'{ty}_audio_info'][f"{ty}_input_audio_file"])

        # if d['spoken_info'] in speech_info['environment']:
        #     env_file = "all_env"
        #     input_audio_file = f'{d["case_id"]}_{d["spoken_info"]}_{d["topic"]}_{ty}.wav'
        #     input_audio_file = os.path.join(input_dir, "audio", env_file, input_audio_file)
        # else:
        #     spoken_info = speech_info['speaker-paralinguistic'][d['spoken_info']]
        #     input_audio_file = f'{d["case_id"]}_{spoken_info}_{d["topic"]}_{ty}.wav'
        #     input_audio_file = os.path.join(input_dir, "audio", spoken_info, input_audio_file)

        if not os.path.isfile(input_audio_file):
            print(input_audio_file)
            continue  # check if some input audio_file miss
        # 2-2. obtain output audio file path
        output_audio_file = f"{d['case_id']}_asr.wav"
        output_audio_file = os.path.join(output_dir, output_audio_file)

        # 2-3. check if data has been referred
        if "predicted_answer" in d.keys():
            if len(d["predicted_answer"])>0:
                continue
            else:
                print(input_audio_file)

        # 2-4. obtain system prompt & user prompt (maybe model-specific)
        if args.model_name == "qwen25omni": # because qwen has its own system prompt
            default_system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. "
        else:
            default_system_prompt = "You are a helpful assistant. "
        task_prompt = "\nYou task is to listen to the audio snippet carefully and answer the user's question."
        
        if args.model_name == "minicpm":
            # because default system prompt: {'role': 'user', 'content': ['Use the <reserved_53> voice.', 'You are a helpful assistant with the above voice style.']}
            system_prompt = task_prompt
        elif args.model_name == "desta-2.5-audio":
            system_prompt = "Focus on the audio clips and instruction. Respond directly without any other words" # From the model's github repo
        elif args.model_name == "vita-audio":
            system_prompt = "Convert the speech to text."
        elif args.model_name == "llama-omni2":
            system_prompt = "You are an ASR engine. Transcribe the audio verbatim. Output text only, no explanations.\nPlease provide your asr answer in the format: 'The audio text is <transcribed_text>'."
        else:
            system_prompt = default_system_prompt + task_prompt

        user_prompt = f"Please transcribe the speech in the input audio into text."

        if args.model_name == "vita-audio":
            user_prompt = None
        elif args.model_name == "echoX":
            system_prompt = None
            user_prompt = "What does the person say?"

        
        response = model.generate_audio(input_audio_file, output_audio_file, system_prompt, user_instruction=user_prompt)
        d['predicted_answer'] = response['response_audio_transcript']
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)



    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--root_dir", default="/share/workspace/EQ-SLM/echomind-master")
    argparser.add_argument("--data_type", default="synthesis", help="choose the data type in EchoMind: human/synthesis")
    argparser.add_argument("--api_key", default="")
    argparser.add_argument("--model_name", default="gpt4o", help="choose the evaluated SLM model")    
    args = argparser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    print(args.device)
    print(f'GPU count: {torch.cuda.device_count()}')

    
    main(args)
    