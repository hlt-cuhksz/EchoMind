import json
import os.path
import tempfile
import sys
import re
import uuid
import requests
from argparse import ArgumentParser
from copy import deepcopy
from io import BytesIO
import base64

import torchaudio

import gradio as gr
import torch

streaming_output = False

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="8888")
    args = parser.parse_args()


    def clear_fn():
        return [], [], [], '', '', '', None, None


    def inference_fn(
            temperature: float,
            top_p: float,
            max_new_tokens: int,
            input_mode,
            audio_path: str | None,
            input_text: str | None,
            history: list[dict],
            messages: list[dict],
            previous_input_tokens: str,
            previous_completion_tokens: str,
    ):

        if input_mode == "audio":
            assert audio_path is not None
            with open(audio_path, "rb") as audio_file:
                audio_binary = audio_file.read()
            audio_binary = base64.b64encode(audio_binary).decode("utf-8")
            history.append({"role": "user", "content": {"path": audio_path}})
            messages.append({"role": "user", "content": {"audio": audio_binary}})
            user_input = "<audio>"
        else:
            assert input_text is not None
            history.append({"role": "user", "content": input_text})
            messages.append({"role": "user", "content": input_text})
            user_input = input_text

        # Gather history
        inputs = previous_input_tokens + previous_completion_tokens
        inputs = inputs.strip()
        inputs += f"<|user|>\n{user_input}\n<|assistant|>\n"

        with torch.no_grad():
            response = requests.post(
                "http://localhost:21002/worker_generate_stream",
                headers = {"User-Agent": "Omni Speech"},
                json={
                    "model": "omnispeech",
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens
                },
                stream=True,
                timeout=20
            )

            tts_speechs = []
            complete_text = ""
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data["error_code"] == 0:
                        if history[-1]["role"] == "assistant":
                            history[-1]["content"] = data["text"]
                            messages[-1]["content"] = data["text"]
                        else:
                            history.append({"role": "assistant", "content": data["text"]})
                            messages.append({"role": "assistant", "content": data["text"]})
                        complete_text = data["text"]
                        if data["audio"]:
                            audio_binary = base64.b64decode(data["audio"])
                            wav, sr = torchaudio.load(BytesIO(audio_binary))
                            tts_speechs.append(wav)
                            if streaming_output:
                                yield history, messages, inputs, complete_text, '', audio_binary, None, deepcopy(history)
                        yield history, messages, inputs, complete_text, '', None, None, deepcopy(history)
                    else:
                        error_output = data["text"] + f" (error_code: {data['error_code']})"
                        history.append({"role": "assistant", "content": error_output})
                        messages.append({"role": "assistant", "content": error_output})
                        yield history, messages, inputs, error_output, error_output, None, None, deepcopy(history)
                        return

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_full = torch.cat(tts_speechs, dim=1)
                torchaudio.save(f, wav_full, 22050, format="wav")
                history.append({"role": "assistant", "content": {"path": f.name, "type": "audio/wav"}})
                yield history, messages, inputs, complete_text, '', None, f.name, deepcopy(history)

    def update_input_interface(input_mode):
        if input_mode == "audio":
            return [gr.update(visible=True), gr.update(visible=False)]
        else:
            return [gr.update(visible=False), gr.update(visible=True)]


    # Create the Gradio interface
    with gr.Blocks(title="OmniSpeech Demo", fill_height=True) as demo:
        with gr.Row():
            temperature = gr.Number(
                label="Temperature",
                value=0.2
            )

            top_p = gr.Number(
                label="Top p",
                value=0.8
            )

            max_new_token = gr.Number(
                label="Max new tokens",
                value=512,
            )

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            bubble_full_width=False,
            type="messages",
            scale=1,
        )

        with gr.Row():
            with gr.Column():
                input_mode = gr.Radio(["audio", "text"], label="Input Mode", value="audio")
                audio = gr.Audio(label="Input audio", type='filepath', show_download_button=True, visible=True)
                text_input = gr.Textbox(label="Input text", placeholder="Enter your text here...", lines=2, visible=False)

            with gr.Column():
                submit_btn = gr.Button("Submit")
                reset_btn = gr.Button("Clear")
                if streaming_output:
                    output_audio = gr.Audio(label="Play", streaming=True,
                        autoplay=True, show_download_button=False)
                else:
                    output_audio = gr.Audio(label="Play", streaming=False,
                        autoplay=False, show_download_button=False)
                complete_audio = gr.Audio(label="Last Output Audio (If Any)",
                    type="filepath", interactive=False, autoplay=False)



        gr.Markdown("""## Debug Info""")
        with gr.Row():
            input_tokens = gr.Textbox(
                label=f"Input Tokens",
                interactive=False,
            )

            completion_tokens = gr.Textbox(
                label=f"Completion Tokens",
                interactive=False,
            )

        detailed_error = gr.Textbox(
            label=f"Detailed Error",
            interactive=False,
        )

        history_state = gr.State([])
        messages = gr.State([])

        respond = submit_btn.click(
            inference_fn,
            inputs=[
                temperature,
                top_p,
                max_new_token,
                input_mode,
                audio,
                text_input,
                history_state,
                messages,
                input_tokens,
                completion_tokens,
            ],
            outputs=[history_state, messages, input_tokens, completion_tokens, detailed_error, output_audio, complete_audio, chatbot]
        )

        reset_btn.click(clear_fn, outputs=[chatbot, history_state, messages, input_tokens, completion_tokens, detailed_error, output_audio, complete_audio])
        input_mode.input(update_input_interface, inputs=[input_mode], outputs=[audio, text_input])

    # Launch the interface
    demo.launch(
        server_port=args.port,
        server_name=args.host
    )
