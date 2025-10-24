# from .gpt4o import GPT4oChat
# from .minicpmo26 import MiniCPMChat
# from .qwen25omni import Qwen25OmniChat



import importlib

class LazyModelLoader:
    def __init__(self):
        self._model_cls_mapping = {
            'gpt4o': 'models.gpt4o.GPT4oChat',
            'minicpm': 'models.minicpmo26.MiniCPMChat',
            'qwen25omni': 'models.qwen25omni.Qwen25OmniChat',
            'glm4voice': 'models.glm_4_voice.GLM4VoiceChat',
            'audio-flamingo-3': 'models.audio_flamingo_3.AudioFlamingo3',
            'audio-flamingo-3-think': 'models.audio_flamingo_3.AudioFlamingo3Think',
            'audio-flamingo-3-chat': 'models.audio_flamingo_3.AudioFlamingo3Chat',
            'desta-2.5-audio': 'models.desta25_audio.DeSTA25Audio',
            'speechgpt-2.0-preview': 'models.speechgpt_2_preview.SpeechGPT2Preview',
            'vita-audio': 'models.VitaAudio.VitaAudio',
            'baichuan':'models.baichuan.BaichuanChat',
            'baichuan_response': 'models.baichuan_response.BaichuanChat',
            'baichuan_asr':'models.baichuan_asr.ASRModel',
            'opens2s':'models.opens2s.OmniSpeechS2S',
            'echoX':'models.echoX.inference.EchoX',
            'kimi_audio':'models.kimi_audio.KimiAudioS2SModel',
            'kimi_audio_asr':'models.kimi_audio_asr.KimiAudioS2SModel',
            'kimi_audio__mcq':'models.kimi_audio_mcq.KimiAudioS2SModel',
            'llama-omni2': 'models.llama_omni2.LlamaOmni2Chat'
        }

    def get_model_class(self, model_name):
        if model_name not in self._model_cls_mapping:
            raise ValueError(f"Unknown model name: {model_name}")
        
        module_name, class_name = self._model_cls_mapping[model_name].rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)


model_loader = LazyModelLoader()
