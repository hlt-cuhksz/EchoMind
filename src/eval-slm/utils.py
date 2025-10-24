import os


speech_info = {
    "speaker-paralinguistic": {
       "male": "male", 
       "female": "female", 
       "child": "child", 
       "adult": "adult", 
       "elderly": "elderly",
        "Hoarse voice": "hoarse",
        "vocal fatigue": "tired",
        "breathless sound / gasping sound / heavy breathing sound": "breath",
        "sobbing sound": "sobbing",
        "happy-toned expressive voice": "happy",
        "sad-toned expressive voice": "sad",
        "surprise-toned expressive voice": "surprise",
        "angry-toned expressive voice": "angry",
        "fear-toned expressive voice": "fear",
        "disgust-toned expressive voice": "disgust",
        "shout" : "shout",
        "whisper": "whisper",
        "fast speaking pace": "fast",
        "slow speaking pace": "slow",
        "cough voice": "cough",
        "sigh sound": "sigh",
        "laughter sound": "laughter",
        "yawn sound": "yawn",
        "moan sound": "moan",
        },
     "environment": [
        "sound of wind blowing",
        "sound of thunderstorm and thunder",
        "sound of raining",
        "sound of driving a car",
        "sound of subway",
        "sound of sea waves",
        "sound of playing basketball",
        "sound of applause from a crowd",
        "sound of crowd cheering",
        "sound of crowd's chatter",
        "sound of children playing",
        "sound of children speaking",
        "sound of dog(s) barking",
        "sound of alarm clock ringing",
        "sound of ringtone",
        "sound of happy music",
        "sound of funny music",
        "sound of sad music",
        "sound of exciting music",
        "sound of angry music",
        "sound of vehicle honking",
     ]
}

voice_info = {
    "speaker": ["male", "female", "child", "adult", "elderly"],
    "paralinguistic": [
        "Hoarse voice",
        "vocal fatigue",
        "breathless sound / gasping sound / heavy breathing sound",
        "sobbing sound",
        "happy-toned expressive voice",
        "sad-toned expressive voice",
        "surprise-toned expressive voice",
        "angry-toned expressive voice",
        "fear-toned expressive voice",
        "disgust-toned expressive voice",
        "shout",
        "whisper",
        "fast speaking pace",
        "slow speaking pace",
        "cough voice",
        "sigh sound",
        "laughter sound",
        "yawn sound",
        "moan sound"],
     "environment": [
        "sound of wind blowing",
        "sound of thunderstorm and thunder",
        "sound of raining",
        "sound of driving a car",
        "sound of subway",
        "sound of sea waves",
        "sound of playing basketball",
        "sound of applause from a crowd",
        "sound of crowd cheering",
        "sound of crowd's chatter",
        "sound of children playing",
        "sound of children speaking",
        "sound of dog(s) barking",
        "sound of alarm clock ringing",
        "sound of ringtone",
        "sound of happy music",
        "sound of funny music",
        "sound of sad music",
        "sound of exciting music",
        "sound of angry music",
        "sound of vehicle honking",
     ]
}

def obtain_prompt(root_dir):
   prompt = {
      'basic': "",
      'enhance': {},
   }

   with open(os.path.join(root_dir, "dataset/instruction/generate_audio_response/basic_prompt.txt"), 'r', encoding="utf-8") as files:
      instruction_prompt = files.readlines()
      prompt['basic'] = "".join(instruction_prompt)
   with open(os.path.join(root_dir, "dataset/instruction/generate_audio_response/enhance_speaker_prompt.txt"), 'r', encoding="utf-8") as files:
      instruction_prompt = files.readlines()
      prompt['enhance']['speaker'] = "".join(instruction_prompt)
   with open(os.path.join(root_dir, "dataset/instruction/generate_audio_response/enhance_paralinguistic_prompt.txt"), 'r', encoding="utf-8") as files:
      instruction_prompt = files.readlines()
      prompt['enhance']['paralinguistic'] = "".join(instruction_prompt)
   with open(os.path.join(root_dir, "dataset/instruction/generate_audio_response/enhance_environment_prompt.txt"), 'r', encoding="utf-8") as files:
      instruction_prompt = files.readlines()
      prompt['enhance']['environment'] = "".join(instruction_prompt)
   
   return prompt