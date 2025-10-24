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
    output_dir = os.path.join(args.root_dir, f"output/output_{args.data_type}", 'mcq_output', args.model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 1-2. load model
    model = model_loader.get_model_class(args.model_name)(args)

    for mcq_file in args.MCQ_file:
        # 1-3. load data
        with open(os.path.join(input_dir, "MCQ", mcq_file), 'r', encoding='utf-8') as file:
            mcq_dataset = json.load(file)
        # 1-4. load saved file if exist
        if args.audio_output:
            output_file_path = os.path.join(output_dir, "audio_output_true", f"{args.model_name}_mcq_{mcq_file[:-5]}.json")
        else:
            output_file_path = os.path.join(output_dir, "audio_output_false", f"{args.model_name}_mcq_{mcq_file[:-5]}.json")

        if not os.path.exists(os.path.dirname(output_file_path)):
            os.makedirs(os.path.dirname(output_file_path))

        if os.path.exists(output_file_path):
            with open(output_file_path, 'r', encoding='utf-8') as f:
                mcq_dataset = json.load(f)
        
        # 2. inference
        for d in tqdm(mcq_dataset):
            # 2-1. obtain input audio file
            input_audio_file = os.path.join(input_dir, "audio", d['audio_dir'], d['audio_name'])
            if not os.path.isfile(input_audio_file):
                print(input_audio_file)
                continue  # check if some input audio_file miss

            # 2-2. obtain output audio file save path
            output_audio_file = f"{d['question_id']}.wav"
            output_audio_file = os.path.join(output_dir, output_audio_file)

            # 2-3. check if data has been referred
            if "predicted_audio_answer" in d.keys():
                if len(d["predicted_audio_answer"])>0:
                    continue
                else:
                    print(input_audio_file)

            # 2-4. obtain system prompt & user prompt (maybe model-specific)
            if args.model_name == "qwen25omni": # because qwen has its own system prompt
                base_system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. "
            else:
                base_system_prompt = "You are a helpful assistant. "
            task_prompt = "\nYour task is to determine the correct answer to a multiple-choice question based on an audio clip. Analyze the audio and select the most accurate answer (A, B, C, or D) without additional explanation."
            output_prompt = "\nPlease provide your answer in the format: 'The answer is: [A/B/C/D]'."
            
            system_prompt = base_system_prompt + task_prompt + output_prompt
            user_prompt = f"\nQuestion: {d['question']}\nOptions:\n{d['options_text']}"


            if args.audio_output:
                response = model.generate_audio(input_audio_file, output_audio_file, system_prompt, user_instruction=user_prompt, audio_output=args.audio_output)
            else:
                response = model.generate_audio(input_audio_file, output_audio_file, system_prompt, user_instruction=user_prompt)
            d['predicted_audio_answer'] = response['response_audio_transcript']
            d['predicted_text_answer'] = response['response_text']
            d['prediected_audio_file'] = output_audio_file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(mcq_dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--root_dir", default="/share/workspace/EQ-SLM/echomind-master")
    argparser.add_argument("--data_type", default="synthesis", help="choose the data type in EchoMind: human/synthesis")
    argparser.add_argument("--api_key", default="")
    argparser.add_argument("--model_name", default="gpt4o", help="choose the evaluated SLM model")
    argparser.add_argument("--MCQ_file", nargs='+', type=str, default=["understanding.json", "reasoning.json"])
    argparser.add_argument("--audio_output", action="store_true", help="whether to output audio files")
    
    args = argparser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    print(args.device)

    main(args)
    