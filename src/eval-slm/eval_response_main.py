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
    output_dir = os.path.join(args.root_dir, f"output/output_{args.data_type}", 'response_output', args.model_name, args.system_prompt)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 1-2: load original data
    with open(os.path.join(input_dir, f"script_info_{args.data_type}.json"), 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    # 1-3: load model
    model = model_loader.get_model_class(args.model_name)(args)

    # 1-4: Load saved generated responses if available
    output_json_file = os.path.join(output_dir, f"{args.model_name}_{args.system_prompt}_output_responses.json")
    if os.path.exists(output_json_file):
        with open(output_json_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        

    # 2. inference
    for d in tqdm(dataset):
        d['model'] = args.model_name
        d['ouput_prompt'] = args.system_prompt
        if 'output_content' not in d.keys():
            d['output_content'] = {
                "target": {}, 
                "neutral": {},
                "alternative": {}
            }

        for ty in args.voice_type:  
            if ty == "neutral" and d['spoken_info'] in ["male", "female", "child", "adult", "elderly"]:
                continue
            
            # 2-1. obtain input audio file
            input_audio_file = os.path.join(input_dir, "audio", d[f'{ty}_audio_info'][f"{ty}_input_audio_file"])
    
            if not os.path.isfile(input_audio_file):
                print(input_audio_file)
                continue  # check if some input audio_file miss

            # 2-2. obtain output audio file save path
            output_audio_dir = os.path.join(output_dir, d[f'{ty}_audio_info'][f"{ty}_input_audio_file"].split('/')[0])

            if not os.path.exists(output_audio_dir):
                os.makedirs(output_audio_dir)
            
            # 2-3. check if data has been referred
            if "response_audio_transcript" in d["output_content"][ty].keys(): # filter the data which has been inferred
                if len(d["output_content"][ty]["response_audio_transcript"])>0: # correct inferred
                    continue
                else:
                    print(input_audio_file) # wrong inferred
            output_audio_file = f"{d['case_id']}_{args.model_name}_response_{ty}.wav"
            d['output_content'][ty]['file'] = output_audio_file
            output_audio_file = os.path.join(output_audio_dir, output_audio_file)

            

            # 2-4. set 3 types system prompt (maybe model-specific)
            # 2-4-1: default version of system prompt
            if args.system_prompt == 'no':
                system_prompt = ""
            elif args.system_prompt == 'basic' or args.system_prompt == 'basic_plus_basic':
                system_prompt = (
                    "You are a helpful assistant.\n",
                    f"The current topic/scenario is: {d['topic']}.\n",
                    "Your task is to provide a one-turn response based on the input audio."
                )

            elif args.system_prompt == 'enhance':
                if d['spoken_info'] in voice_info['speaker']:
                    system_prompt = (
                        f"You are an assistant in a customer service setting.\n",
                        f"The current topic/scenario is: {d['topic']}.\n",
                        "Your task is to provide a one-turn response based on the input audio, considering both the spoken content and any personal information present in the user's voice. The response should:\n"
                        "- Include an appropriate form of address based on the personal information.\n"
                        "- Be tailored to the content of the user's spoken text and their personal information.\n"
                        "- Avoid any form of gender bias."
                    )

                elif d['spoken_info'] in voice_info['paralinguistic']:
                    system_prompt = (
                        f"You are a helpful assistant.\n",
                        f"The current topic/scenario is: {d['topic']}.\n",
                        "Your task is to provide a one-turn response based on the input audio, considering both the spoken content and any voice characteristics present in the user's voice."
                    )

                elif d['spoken_info'] in voice_info['environment']:
                    system_prompt = (
                        f"You are a helpful assistant.\n",
                        f"The current topic/scenario is: {d['topic']}.\n",
                        "Your task is to provide a one-turn response based on the input audio, considering both the spoken content and any background sounds present."
                    )

            elif args.system_prompt == 'qwen25omni_default':
                system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            
            
            # 2-4-2: model-specific version of system prompt
            if args.model_name == "qwen25omni" and args.system_prompt in ["basic", "enhance"]:
                system_prompt = (
                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.\n",
                ) + system_prompt[1:]

            
            if args.model_name == "echoX":
                if args.system_prompt == "no":
                    system_prompt = None
                elif args.system_prompt == "basic":
                    system_prompt = "Recognize what the voice said and respond to it."
                elif args.system_prompt == "basic_plus_basic" or args.system_prompt == "enhance":
                    default_prompt = "Recognize what the voice said and respond to it."
                    system_prompt = default_prompt + "\n" + "\n".join(system_prompt[1:])
                else:
                    raise ValueError(f"Unsupported system_prompt {args.system_prompt} for model {args.model_name}")
            
            if isinstance(system_prompt, tuple):
                system_prompt = "".join(system_prompt)
            
            response = model.generate_audio(input_audio_file, output_audio_file, system_prompt, audio_output=args.audio_output)
            
            d['output_content'][ty]['response_audio_transcript'] = response['response_audio_transcript']
            d['output_content'][ty]['response_text'] = response['response_text']
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)



    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--root_dir", default="/share/workspace/EQ-SLM/echomind-master")
    argparser.add_argument("--data_type", default="synthesis", help="choose the data type in EchoMind: human/synthesis")
    argparser.add_argument("--api_key", default="")
    argparser.add_argument("--model_name", default="gpt4o", help="choose the evaluated SLM model")
    argparser.add_argument("--system_prompt", default="no", help="choose the system prompt, options: no, basic, enhance, (qwen25omni_default is specific for Qwen-2.5 Omni)")
    argparser.add_argument("--voice_type", nargs='+', type=str, default=["target", "neutral", "alternative"])
    argparser.add_argument("--audio_output", action="store_true", help="whether to output audio files")

    
    args = argparser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    print(args.device)
    print(f'GPU count: {torch.cuda.device_count()}')


    main(args)
    
