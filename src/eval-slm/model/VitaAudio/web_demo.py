import copy
import math
import os
import sys

import gradio as gr
import numpy as np
import torch
from numba import jit
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation import GenerationConfig

from vita_audio.data.processor.audio_processor import add_audio_input_contiguous
from vita_audio.tokenizer import get_audio_tokenizer

PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


@jit
def float_to_int16(audio: np.ndarray) -> np.ndarray:
    am = int(math.ceil(float(np.abs(audio).max())) * 32768)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)


def is_wav(file_path):
    wav_extensions = {".wav"}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in wav_extensions


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0

    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0 and count % 2 == 1:
                line = line.replace("`", r"\`")
                line = line.replace("<", "&lt;")
                line = line.replace(">", "&gt;")
                line = line.replace(" ", "&nbsp;")
                line = line.replace("*", "&ast;")
                line = line.replace("_", "&lowbar;")
                line = line.replace("-", "&#45;")
                line = line.replace(".", "&#46;")
                line = line.replace("!", "&#33;")
                line = line.replace("(", "&#40;")
                line = line.replace(")", "&#41;")
                line = line.replace("$", "&#36;")
            lines[i] = "<br>" + line

    return "".join(lines)


def _launch_demo(model, tokenizer, audio_tokenizer):
    def predict(_chatbot, task_history, task):
        chat_query = task_history[-1][0]
        print(task_history)

        messages = []

        audio_path_list = []
        if task == "Spoken QA":
            messages = [
                {
                    "role": "system",
                    # "content": "Your Name: Luke\nYour Gender: male\n\nRespond in a text-audio interleaved manner.",
                    # "content": "Your Name: Lucy\nYour Gender: female\nRespond in a text-audio interleaved manner.",
                    "content": "Your Name: Omni\nYour Gender: female\nRespond in a text-audio interleaved manner.",
                },
            ]
            for i, (q, a) in enumerate(task_history):

                if isinstance(q, (tuple, list)) and is_wav(q[0]):
                    audio_path_list.append(q[0])
                    messages = messages + [
                        {
                            "role": "user",
                            "content": f"\n<|audio|>",
                        },
                    ]
                else:
                    messages = messages + [
                        {
                            "role": "user",
                            "content": q,
                        },
                    ]
                if a != None:
                    messages = messages + [
                        {
                            "role": "assistant",
                            "content": a,
                        },
                    ]
            model.generation_config.do_sample = False

        elif task == "TTS":
            for i, (q, a) in enumerate(task_history):

                if isinstance(q, (tuple, list)) and is_wav(q[0]):
                    audio_path_list.append(q[0])
                    messages = messages + [
                        {
                            "role": "user",
                            "content": f"\n<|audio|>",
                        },
                    ]
                else:
                    messages = messages + [
                        {
                            "role": "user",
                            "content": f"Convert the text to speech.\n{q}",
                        },
                    ]
                if a != None:
                    messages = messages + [
                        {
                            "role": "assistant",
                            "content": a,
                        },
                    ]
            model.generation_config.do_sample = True
        elif task == "ASR":
            for i, (q, a) in enumerate(task_history):

                if isinstance(q, (tuple, list)) and is_wav(q[0]):
                    audio_path_list.append(q[0])
                    messages = messages + [
                        {
                            "role": "user",
                            "content": f"Convert the speech to text.\n<|audio|>",
                        },
                    ]
                else:
                    messages = messages + [
                        {
                            "role": "user",
                            "content": f"{q}",
                        },
                    ]
                if a != None:
                    messages = messages + [
                        {
                            "role": "assistant",
                            "content": a,
                        },
                    ]
                model.generation_config.do_sample = False

        add_generation_prompt = True
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            # return_tensors="pt",
        )

        input_ids, audios, audio_indices = add_audio_input_contiguous(
            input_ids, audio_path_list, tokenizer, audio_tokenizer
        )

        input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")

        # print("input", tokenizer.decode(input_ids[0], skip_special_tokens=False), flush=True)

        if audio_path_list == []:
            audios = None
            audio_indices = None

        outputs = model.generate(
            input_ids,
            audios=audios,
            audio_indices=audio_indices,
        )

        output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # print(f"{output=}", flush=True)

        audio_offset = tokenizer.convert_tokens_to_ids("<|audio_0|>")
        begin_of_audio = tokenizer.convert_tokens_to_ids("<|begin_of_audio|>")
        end_of_audio = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        response = outputs[0][len(input_ids[0]) :]

        audio_tokens = []
        text_tokens = []
        for token_id in response:
            if token_id >= audio_offset:
                audio_tokens.append(token_id - audio_offset)
            elif (
                (token_id.item() != begin_of_audio)
                and (token_id.item() != end_of_audio)
                and (token_id.item() != im_end)
            ):
                text_tokens.append(token_id)

        if len(audio_tokens) > 0:
            tts_speech = audio_tokenizer.decode(audio_tokens)
            audio_np = float_to_int16(tts_speech.cpu().numpy())
            tts_speech = (22050, audio_np)
        else:
            tts_speech = None

        # import pdb;pdb.set_trace()
        history_response = tokenizer.decode(text_tokens)
        task_history[-1] = (chat_query, history_response)

        _chatbot[-1] = (chat_query, history_response)
        # print("query",chat_query)
        # print("task_history",task_history)
        # print(_chatbot)
        # print("answer:  ",outputs)
        return _chatbot, tts_speech

    def add_text(history, task_history, text):
        task_text = text
        # import pdb;pdb.set_trace()
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_audio(history, task_history, file):
        print(file)
        if file is None:
            return history, task_history
        history = history + [((file,), None)]
        task_history = task_history + [((file,), None)]
        return history, task_history

    def reset_user_input():
        # import pdb;pdb.set_trace()
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []

    with gr.Blocks(title="VITA-Audio-Plus-Vanilla") as demo:
        gr.Markdown("""<center><font size=8>VITA-Audio-Plus-Vanilla</center>""")
        gr.Markdown(
            """<center><font size=4>The deployment of the VITA-Audio-Plus-Vanilla model employs a non-streaming deployment approach. The currently deployed model is VITA-Audio-Plus-Vanilla. For the ASR and TTS tasks, only single-turn dialogues are supported. In the Spoken QA task, generated text is used as dialogue history to reduce the context length.</center>"""
        )
        chatbot = gr.Chatbot(
            label="VITA-Audio-Plus-Vanilla", elem_classes="control-height", height=500
        )
        query = gr.Textbox(lines=2, label="Text Input")
        task_history = gr.State([])
        with gr.Row():
            add_text_button = gr.Button("Submit Text (提交文本)")
            add_audio_button = gr.Button("Submit Audio (提交音频)")
            empty_bin = gr.Button("🧹 Clear History (清除历史)")
            task = gr.Radio(choices=["ASR", "TTS", "Spoken QA"], label="TASK", value="Spoken QA")

        with gr.Row(scale=1):

            record_btn = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎤 Record or Upload Audio (录音或上传音频)",
                show_download_button=True,
                waveform_options=gr.WaveformOptions(sample_rate=16000),
            )
            audio_output = gr.Audio(
                label="Play", streaming=True, autoplay=True, show_download_button=True
            )

        add_text_button.click(
            add_text, [chatbot, task_history, query], [chatbot, task_history], show_progress=True
        ).then(reset_user_input, [], [query]).then(
            predict, [chatbot, task_history, task], [chatbot, audio_output], show_progress=True
        )

        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)

        add_audio_button.click(
            add_audio,
            [chatbot, task_history, record_btn],
            [chatbot, task_history],
            show_progress=True,
        ).then(predict, [chatbot, task_history, task], [chatbot, audio_output], show_progress=True)

    server_port = 18806
    demo.launch(
        share=False,
        debug=True,
        server_name="0.0.0.0",
        server_port=server_port,
        show_api=False,
        show_error=False,
    )


def main():

    model_name_or_path = "VITA-MLLM/VITA-Audio-Plus-Vanilla"

    device_map = "cuda:0"

    sys.path.append("third_party/GLM-4-Voice/")
    sys.path.append("third_party/GLM-4-Voice/cosyvoice/")
    sys.path.append("third_party/GLM-4-Voice/third_party/Matcha-TTS/")

    from huggingface_hub import snapshot_download

    audio_tokenizer_path = snapshot_download(repo_id="THUDM/glm-4-voice-tokenizer")
    flow_path = snapshot_download(repo_id="THUDM/glm-4-voice-decoder")

    audio_tokenizer_rank = 0
    audio_tokenizer_type = "sensevoice_glm4voice"

    torch_dtype = torch.bfloat16
    audio_tokenizer = get_audio_tokenizer(
        audio_tokenizer_path, audio_tokenizer_type, flow_path=flow_path, rank=audio_tokenizer_rank
    )
    from evaluation.get_chat_template import qwen2_chat_template as chat_template

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        chat_template=chat_template,
    )
    # print(f"{tokenizer=}")
    # print(f"{tokenizer.get_chat_template()=}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    ).eval()

    # print(f"{model.config.model_type=}")

    model.generation_config = GenerationConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    model.generation_config.max_new_tokens = 4096
    model.generation_config.chat_format = "chatml"
    model.generation_config.max_window_size = 8192
    model.generation_config.use_cache = True
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_k = 50
    model.generation_config.top_p = 1.0
    model.generation_config.num_beams = 1
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.mtp_inference_mode = [8192, 10]

    _launch_demo(model, tokenizer, audio_tokenizer)


if __name__ == "__main__":

    main()
