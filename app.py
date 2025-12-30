#!/usr/bin/env python3
#
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# References:
# https://gradio.app/docs/#dropdown

import logging
import os
import shutil
import tempfile
import time
import urllib.request
import uuid
from datetime import datetime

import gradio as gr

from examples import examples
from model import (
    embedding2models,
    get_file,
    get_speaker_diarization,
    read_wave,
    speaker_segmentation_models,
)

embedding_frameworks = list(embedding2models.keys())

waves = [e[-1] for e in examples]

for name in waves:
    filename = get_file(
        "csukuangfj/speaker-embedding-models",
        name,
    )

    shutil.copyfile(filename, name)


def MyPrint(s):
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    print(f"{date_time}: {s}")


def convert_to_wav(in_filename: str) -> str:
    """Convert the input audio file to a wave file"""
    out_filename = str(uuid.uuid4())
    out_filename = f"{in_filename}.wav"

    MyPrint(f"Converting '{in_filename}' to '{out_filename}'")
    _ = os.system(
        f"ffmpeg -hide_banner -loglevel error -i '{in_filename}' -ar 16000 -ac 1 '{out_filename}' -y"
    )

    return out_filename


def build_html_output(s: str, style: str = "result_item_success"):
    return f"""
    <div class='result'>
        <div class='result_item {style}'>
          {s}
        </div>
    </div>
    """


def process_uploaded_file(
    embedding_framework: str,
    embedding_model: str,
    speaker_segmentation_model: str,
    input_num_speakers: str,
    input_threshold: str,
    in_filename: str,
):
    if in_filename is None or in_filename == "":
        return "", build_html_output(
            "Please first upload a file and then click "
            'the button "submit for recognition"',
            "result_item_error",
        )

    MyPrint(f"Processing uploaded file: {in_filename}")
    try:
        return process(
            in_filename=in_filename,
            embedding_framework=embedding_framework,
            embedding_model=embedding_model,
            speaker_segmentation_model=speaker_segmentation_model,
            input_num_speakers=input_num_speakers,
            input_threshold=input_threshold,
        )
    except Exception as e:
        MyPrint(str(e))
        return "", build_html_output(str(e), "result_item_error")


def process_microphone(
    embedding_framework: str,
    embedding_model: str,
    speaker_segmentation_model: str,
    input_num_speakers: str,
    input_threshold: str,
    in_filename: str,
):
    if in_filename is None or in_filename == "":
        return "", build_html_output(
            "Please first click 'Record from microphone', speak, "
            "click 'Stop recording', and then "
            "click the button 'submit for speaker diarization'",
            "result_item_error",
        )

    MyPrint(f"Processing microphone: {in_filename}")
    try:
        return process(
            in_filename=in_filename,
            embedding_framework=embedding_framework,
            embedding_model=embedding_model,
            speaker_segmentation_model=speaker_segmentation_model,
            input_num_speakers=input_num_speakers,
            input_threshold=input_threshold,
        )
    except Exception as e:
        MyPrint(str(e))
        return "", build_html_output(str(e), "result_item_error")


def process_url(
    embedding_framework: str,
    embedding_model: str,
    speaker_segmentation_model: str,
    input_num_speakers: str,
    input_threshold: str,
    url: str,
):
    MyPrint(f"Processing URL: {url}")
    with tempfile.NamedTemporaryFile() as f:
        try:
            urllib.request.urlretrieve(url, f.name)

            return process(
                in_filename=f.name,
                embedding_framework=embedding_framework,
                embedding_model=embedding_model,
                speaker_segmentation_model=speaker_segmentation_model,
                input_num_speakers=input_num_speakers,
                input_threshold=input_threshold,
            )
        except Exception as e:
            MyPrint(str(e))
            return "", build_html_output(str(e), "result_item_error")


def process(
    embedding_framework: str,
    embedding_model: str,
    speaker_segmentation_model: str,
    input_num_speakers: str,
    input_threshold: str,
    in_filename: str,
):
    MyPrint(f"embedding_framework: {embedding_framework}")
    MyPrint(f"embedding_model: {embedding_model}")
    MyPrint(f"speaker_segmentation_model: {speaker_segmentation_model}")
    MyPrint(f"input_num_speakers: {input_num_speakers}")
    MyPrint(f"input_threshold: {input_threshold}")
    MyPrint(f"in_filename: {in_filename}")

    try:
        input_num_speakers = int(input_num_speakers)
    except ValueError:
        return "", build_html_output(
            "Please set a valid number of speakers",
            "result_item_error",
        )

    if input_num_speakers <= 0:
        try:
            input_threshold = float(input_threshold)
            if input_threshold < 0 or input_threshold > 10:
                raise ValueError("")
        except ValueError:
            return "", build_html_output(
                "Please set a valid threshold between (0, 10)",
                "result_item_error",
            )
    else:
        input_threshold = 0

    filename = convert_to_wav(in_filename)

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    MyPrint(f"Started at {date_time}")

    start = time.time()

    audio, sample_rate = read_wave(filename)

    MyPrint(f"audio, {audio.shape[0] / sample_rate}, {sample_rate}")

    sd = get_speaker_diarization(
        segmentation_model=speaker_segmentation_model,
        embedding_model=embedding_model,
        num_clusters=input_num_speakers,
        threshold=input_threshold,
    )
    MyPrint(f"{audio.shape[0] / sd.sample_rate}, {sample_rate}")

    segments = sd.process(audio).sort_by_start_time()
    s = ""
    for seg in segments:
        s += f"{seg.start:.3f} -- {seg.end:.3f} speaker_{seg.speaker:02d}\n"

    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    end = time.time()

    duration = audio.shape[0] / sd.sample_rate
    rtf = (end - start) / duration

    MyPrint(f"Finished at {date_time} s. Elapsed: {end - start: .3f} s")

    info = f"""
    Wave duration  : {duration: .3f} s <br/>
    Processing time: {end - start: .3f} s <br/>
    RTF: {end - start: .3f}/{duration: .3f} = {rtf:.3f} <br/>
    """
    if rtf > 1:
        info += (
            "<br/>We are loading the model for the first run. "
            "Please run again to measure the real RTF.<br/>"
        )

    MyPrint(info)
    MyPrint(f"\nembedding_model: {embedding_model}\nSegments: {s}")

    return s, build_html_output(info)


title = "# Speaker diarization with Next-gen Kaldi"
description = """
This space shows how to do speaker diarization with Next-gen Kaldi.

It is running on CPU within a docker container provided by Hugging Face.

See more information by visiting
<https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html>

If you want to try it on Android, please download pre-built Android
APKs for speaker diarzation by visiting
<https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/android.html>

---

Note about the two arguments:

    - number of speakers: If you know the actual number of speakers in the input file,
      please provide it. Otherwise, please set it to 0
    - clustering threshold: Used only when number of speakers is 0. A larger
      threshold results in fewer clusters, i.e., fewer speakers.
"""

# css style is copied from
# https://huggingface.co/spaces/alphacep/asr/blob/main/app.py#L113
css = """
.result {display:flex;flex-direction:column}
.result_item {padding:15px;margin-bottom:8px;border-radius:15px;width:100%}
.result_item_success {background-color:mediumaquamarine;color:white;align-self:start}
.result_item_error {background-color:#ff7070;color:white;align-self:start}
"""


def update_embedding_model_dropdown(framework: str):
    if framework in embedding2models:
        choices = embedding2models[framework]
        return gr.Dropdown(
            choices=choices,
            value=choices[0],
            interactive=True,
        )

    raise ValueError(f"Unsupported framework: {framework}")


demo = gr.Blocks(css=css)


with demo:
    gr.Markdown(title)

    embedding_framework_choices = list(embedding2models.keys())
    embedding_framework_radio = gr.Radio(
        label="Speaker embedding frameworks",
        choices=embedding_framework_choices,
        value=embedding_framework_choices[0],
    )

    embedding_model_dropdown = gr.Dropdown(
        choices=embedding2models[embedding_framework_choices[0]],
        label="Select a speaker embedding model",
        value=embedding2models[embedding_framework_choices[0]][0],
    )

    embedding_framework_radio.change(
        update_embedding_model_dropdown,
        inputs=embedding_framework_radio,
        outputs=embedding_model_dropdown,
    )

    speaker_segmentation_model_dropdown = gr.Dropdown(
        choices=speaker_segmentation_models,
        label="Select a speaker segmentation model",
        value=speaker_segmentation_models[0],
    )

    input_num_speakers = gr.Textbox(
        label="Number of speakers",
        info="Number of speakers",
        lines=1,
        max_lines=1,
        value="0",
        placeholder="Specify number of speakers in the test file",
    )

    input_threshold = gr.Textbox(
        label="Clustering threshold",
        info="Threshold for clustering",
        lines=1,
        max_lines=1,
        value="0.5",
        placeholder="Clustering for threshold",
    )

    with gr.Tabs():
        with gr.TabItem("Upload from disk"):
            uploaded_file = gr.Audio(
                sources=["upload"],  # Choose between "microphone", "upload"
                type="filepath",
                label="Upload from disk",
            )
            upload_button = gr.Button("Submit for speaker diarization")
            uploaded_output = gr.Textbox(label="Result from uploaded file")
            uploaded_html_info = gr.HTML(label="Info")

            gr.Examples(
                examples=examples,
                inputs=[
                    embedding_framework_radio,
                    embedding_model_dropdown,
                    speaker_segmentation_model_dropdown,
                    input_num_speakers,
                    input_threshold,
                    uploaded_file,
                ],
                outputs=[uploaded_output, uploaded_html_info],
                fn=process_uploaded_file,
            )
        with gr.TabItem("Record from microphone"):
            microphone = gr.Audio(
                sources=["microphone"],  # Choose between "microphone", "upload"
                type="filepath",
                label="Record from microphone",
            )

            record_button = gr.Button("Submit for speaker diarization")
            recorded_output = gr.Textbox(label="Result from recordings")
            recorded_html_info = gr.HTML(label="Info")

            gr.Examples(
                examples=examples,
                inputs=[
                    embedding_framework_radio,
                    embedding_model_dropdown,
                    speaker_segmentation_model_dropdown,
                    input_num_speakers,
                    input_threshold,
                    microphone,
                ],
                outputs=[recorded_output, recorded_html_info],
                fn=process_microphone,
            )

        with gr.TabItem("From URL"):
            url_textbox = gr.Textbox(
                max_lines=1,
                placeholder="URL to an audio file",
                label="URL",
                interactive=True,
            )

            url_button = gr.Button("Submit for speaker diarization")
            url_output = gr.Textbox(label="Result from URL")
            url_html_info = gr.HTML(label="Info")

        upload_button.click(
            process_uploaded_file,
            inputs=[
                embedding_framework_radio,
                embedding_model_dropdown,
                speaker_segmentation_model_dropdown,
                input_num_speakers,
                input_threshold,
                uploaded_file,
            ],
            outputs=[uploaded_output, uploaded_html_info],
        )

        record_button.click(
            process_microphone,
            inputs=[
                embedding_framework_radio,
                embedding_model_dropdown,
                speaker_segmentation_model_dropdown,
                input_num_speakers,
                input_threshold,
                microphone,
            ],
            outputs=[recorded_output, recorded_html_info],
        )

        url_button.click(
            process_url,
            inputs=[
                embedding_framework_radio,
                embedding_model_dropdown,
                speaker_segmentation_model_dropdown,
                input_num_speakers,
                input_threshold,
                url_textbox,
            ],
            outputs=[url_output, url_html_info],
        )

    gr.Markdown(description)

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.WARNING)

    demo.launch(share=True)
