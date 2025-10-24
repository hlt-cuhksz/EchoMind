import os  
import sys
import uuid
import torch
import torchaudio
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, WhisperFeatureExtractor

sys.path.insert(0, "echomind-master/src/eval-slm/models/GLM-4-Voice")
from speech_tokenizer.modeling_whisper import WhisperVQEncoder

sys.path.insert(0, "echomind-master/src/eval-slm/models/GLM-4-Voice/cosyvoice")
sys.path.insert(0, "echomind-master/src/eval-slm/models/GLM-4-Voice/third_party/Matcha-TTS")

from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder

class ModelWorker:
    def __init__(self, model_path, dtype="bfloat16", device='cuda'):
        self.device = device
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if dtype == "int4" else None

        self.glm_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=self.bnb_config if self.bnb_config else None,
            device_map={"": 0}
        ).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    @torch.inference_mode()
    def generate_complete(self, params):
        inputs = self.glm_tokenizer([params["prompt"]], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.glm_model.generate(
                **inputs,
                max_new_tokens=int(params.get("max_new_tokens", 256)),
                temperature=float(params.get("temperature", 1.0)),
                top_p=float(params.get("top_p", 1.0)),
                do_sample=True if float(params.get("temperature", 1.0)) > 0 else False,
                pad_token_id=self.glm_tokenizer.eos_token_id,
                eos_token_id=self.glm_tokenizer.convert_tokens_to_ids('<|user|>'),  
                return_dict_in_generate=True
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs.sequences[0][input_length:].tolist()
        
        return generated_tokens

class GLM4VoiceInference:
    def __init__(self, 
                 model_path="glm-4-voice-9b", #your local model path
                 flow_path="glm-4-voice-decoder", #your local model path
                 tokenizer_path="glm-4-voice-tokenizer",  #your local model path
                 device="cuda",
                 dtype="bfloat16"):

        self.device = device
        self.flow_path = flow_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.dtype = dtype
        
        self._initialize_models()
    
    def _initialize_models(self):
        print("Initializing all model components...")
        
        self.model_worker = ModelWorker(self.model_path, self.dtype, self.device)
        
        flow_config = os.path.join(self.flow_path, "config.yaml")
        flow_checkpoint = os.path.join(self.flow_path, 'flow.pt')
        hift_checkpoint = os.path.join(self.flow_path, 'hift.pt')
        
        print("loading audio decoder...")
        self.audio_decoder = AudioDecoder(
            config_path=flow_config, 
            flow_ckpt_path=flow_checkpoint,
            hift_ckpt_path=hift_checkpoint,
            device=self.device
        )
        
        # Speech tokenizer
        print("loading tokenizer...")
        self.whisper_model = WhisperVQEncoder.from_pretrained(
            self.tokenizer_path
        ).eval().to(self.device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.tokenizer_path
        )
        
        print("All model components initialized successfully.")
    
    def _process_audio_chunks(self, audio_tokens, uuid_str):

        chunk_size = 200 
        tts_speechs = []
        
        print(f"audio tokens={len(audio_tokens)}, chunk size={chunk_size}")
        
        for i in range(0, len(audio_tokens), chunk_size):
            chunk_tokens = audio_tokens[i:i+chunk_size]
            if not chunk_tokens:
                continue
                
            chunk_tensor = torch.tensor(chunk_tokens, device=self.device).unsqueeze(0)
            is_final = (i + chunk_size >= len(audio_tokens))
            
            if i > 0:
                torch.cuda.empty_cache()
            
            chunk_speech, _ = self.audio_decoder.token2wav(
                chunk_tensor,
                uuid=uuid_str,
                prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
                prompt_feat=torch.zeros(1, 0, 80).to(self.device),
                finalize=is_final
            )
            tts_speechs.append(chunk_speech.squeeze().cpu())
        
        if tts_speechs:
            result = torch.cat(tts_speechs, dim=-1)
            print("audio chunks synthesis completed.")
            return result
        else:
            raise RuntimeError("audio chunks Processing failed")
    
    def inference(self, 
                  audio_path=None,
                  text_input=None, 
                  output_dir="./outputs",
                  temperature=0.2,
                  top_p=0.8,
                  max_new_tokens=2000,
                  custom_system_prompt=None):

        
        os.makedirs(output_dir, exist_ok=True)
        
        if audio_path is None and text_input is None:
            raise ValueError("provide either audio_path or text_input, or both.")
        
        user_input_parts = []
        
        if audio_path is not None and text_input is not None:
            input_mode = "multimodal"
            print(f"Processing multimodal input:")
            print(f"  audio: {audio_path}")
            print(f"  text: {text_input}")
            
            audio_tokens = extract_speech_token(
                self.whisper_model, self.feature_extractor, [audio_path]
            )[0]
            if len(audio_tokens) == 0:
                raise ValueError("Failed to extract tokens from the audio.")
            audio_tokens_str = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            audio_tokens_str = "<|begin_of_audio|>" + audio_tokens_str + "<|end_of_audio|>"
            
            user_input_parts.append(audio_tokens_str)
            # user_input_parts.append(text_input)  
            user_input = "\n".join(user_input_parts)
            
            # default_system_prompt = "User will provide you with both speech and text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
            default_system_prompt = text_input  
            
        elif audio_path is not None:
            input_mode = "audio"
            print(f"processing audio input: {audio_path}")
            audio_tokens = extract_speech_token(
                self.whisper_model, self.feature_extractor, [audio_path]
            )[0]
            if len(audio_tokens) == 0:
                raise ValueError("Failed to extract tokens from the audio")
            audio_tokens_str = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            audio_tokens_str = "<|begin_of_audio|>" + audio_tokens_str + "<|end_of_audio|>"
            user_input = audio_tokens_str
            
            default_system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
        else:
            input_mode = "text"  
            user_input = text_input
            print(f"processing text input: {text_input}")
            default_system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
        
        system_prompt = custom_system_prompt if custom_system_prompt else default_system_prompt
        print(f"prompt: {custom_system_prompt}")
        
        
        inputs = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        print(f"input prompt: {inputs}")

        print(f"Starting inference, input mode:{input_mode}")
        
        params = {
            "prompt": inputs,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }
        
        with torch.no_grad():
            complete_tokens = self.model_worker.generate_complete(params)
            
            text_tokens, audio_tokens = [], []
            audio_offset = self.model_worker.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
            
            for token_id in complete_tokens:
                if token_id >= audio_offset:
                    audio_tokens.append(token_id - audio_offset)
                else:
                    text_tokens.append(token_id)
            
            
            if audio_tokens:
                
                this_uuid = str(uuid.uuid4())
                
                if len(audio_tokens) <= 800:
                    try:
                        tts_token = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)
                        tts_speech, _ = self.audio_decoder.token2wav(
                            tts_token, 
                            uuid=this_uuid,
                            prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
                            prompt_feat=torch.zeros(1, 0, 80).to(self.device),
                            finalize=True
                        )
                        tts_speech = tts_speech.squeeze().cpu()
                        
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        tts_speech = self._process_audio_chunks(audio_tokens, this_uuid)
                        
                else:
                    tts_speech = self._process_audio_chunks(audio_tokens, this_uuid)
                
            else:
                tts_speech = None
            
            complete_text = self.model_worker.glm_tokenizer.decode(complete_tokens, spaces_between_special_tokens=False)
            text_only = self.model_worker.glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)
        
        timestamp = uuid.uuid4().hex[:8]
        
        audio_output_path = os.path.join(output_dir, f"output_audio_{timestamp}.wav")
        if tts_speech.dim() == 1:
            tts_speech = tts_speech.unsqueeze(0)  
        elif tts_speech.dim() > 2:
            tts_speech = tts_speech.squeeze()  
            if tts_speech.dim() == 1:
                tts_speech = tts_speech.unsqueeze(0)
        
        torchaudio.save(audio_output_path, tts_speech, 22050, format="wav")

        
        return {
            "audio_path": audio_output_path,
            "text_path": text_output_path,
            "complete_text": complete_text,
            "text_only": text_only
        }

class GLM4VoiceChat:
    def __init__(self, args):
        self.args = args
        self.model_path = getattr(args, 'model_path', "glm-4-voice-9b") #your local model path
        self.flow_path = getattr(args, 'flow_path', "glm-4-voice-decoder")#your local model path
        self.tokenizer_path = getattr(args, 'tokenizer_path', "glm-4-voice-tokenizer")#your local model path
        self.device = getattr(args, 'device', 'cuda')
        self.dtype = getattr(args, 'dtype', 'bfloat16')
        
        
        self.inference_engine = GLM4VoiceInference(
            model_path=self.model_path,
            flow_path=self.flow_path,
            tokenizer_path=self.tokenizer_path,
            device=self.device,
            dtype=self.dtype
        )
        

    def generate_audio(self, audio, output_file, system_prompt, user_instruction, audio_output=None, max_new_tokens=2048):

        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            audio_path = audio
            text_input = user_instruction if user_instruction else None
            
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                result = self.inference_engine.inference(
                    audio_path=audio_path,
                    text_input=text_input,
                    output_dir=temp_dir,
                    temperature=0.2,
                    top_p=0.8,
                    max_new_tokens=max_new_tokens,
                    custom_system_prompt=system_prompt if system_prompt else None
                )
                
                import shutil
                if os.path.exists(result['audio_path']):
                    shutil.copy2(result['audio_path'], output_file)
                
                
                return {
                    "response_audio_transcript": result.get('text_only', ''),
                    "response_text": "",
                }
                
        except Exception as e:
            print(f"Error processing audio: {audio}")
            print(f"Error details: {str(e)}")
            
            return {
                "response_audio_transcript": "",
                "response_text": "",
                "audio_generated": False,
                "error": str(e)
            }