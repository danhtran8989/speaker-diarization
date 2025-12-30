#!/usr/bin/env python3
# app_diarization_with_transcription.py
"""
Gradio app: Speaker Diarization + Whisper transcription for each speaker segment
Based on Next-gen Kaldi (sherpa-onnx) + whisper-timestamped
"""

import os
import shutil
import tempfile
import time
import urllib.request
import uuid
from datetime import datetime
import gradio as gr
import numpy as np
import torch

import sherpa_onnx
import whisper_timestamped as whisper

from model import (  # assuming your model.py is in the same folder
    embedding2models,
    get_file,
    get_speaker_diarization,
    read_wave,
    speaker_segmentation_models,
)

# ────────────────────────────────────────────────────────────────
# Configuration & Globals
# ────────────────────────────────────────────────────────────────

WHISPER_MODEL_NAME = "openai/whisper-large-v3-turbo"  # or "large-v3", "medium", etc.
WHISPER_LANGUAGE = "vi"  # Vietnamese
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Whisper will run on device: {WHISPER_DEVICE.upper()}")

# Global whisper model (loaded once)
whisper_model = None


def load_whisper_model():
    global whisper_model
    if whisper_model is None:
        print(f"Loading Whisper model '{WHISPER_MODEL_NAME}' on {WHISPER_DEVICE} ...")
        whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
    return whisper_model


def format_time(seconds: float) -> str:
    """Convert seconds → [hh:]mm:ss"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes:02d}:{secs:05.2f}"


def build_result_html(diarization_text: str, transcription_text: str) -> str:
    """Build nice HTML output with both diarization and transcription"""
    return f"""
    <div style="font-family: monospace; white-space: pre-wrap; background: #1e1e1e; color: #e0e0e0; padding: 16px; border-radius: 8px;">
        <h3 style="margin-top:0; color:#4ecdc4;">Speaker Diarization + Transcription</h3>
        <div style="margin: 12px 0; padding: 10px; background: #2d2d2d; border-left: 4px solid #4ecdc4;">
            <strong>Diarization result:</strong><br>
            {diarization_text.replace('\n', '<br>')}
        </div>
        
        <div style="margin: 12px 0; padding: 10px; background: #2d2d2d; border-left: 4px solid #ff9f1c;">
            <strong>Transcription (with speakers):</strong><br>
            {transcription_text.replace('\n', '<br>')}
        </div>
    </div>
    """


def transcribe_segment(audio_np: np.ndarray, sample_rate: int, start: float, end: float):
    """Extract and transcribe one speaker segment using whisper-timestamped"""
    duration = len(audio_np) / sample_rate
    if end > duration:
        end = duration

    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)

    segment_audio = audio_np[start_sample:end_sample]

    # whisper-timestamped expects audio in [-1,1] float32
    # (we already have it in this format from read_wave)

    try:
        result = whisper.transcribe_timestamped(
            load_whisper_model(),
            segment_audio,
            sample_rate,
            language=WHISPER_LANGUAGE,
            beam_size=1,
            temperature=0.0,
            best_of=1,
            compute_word_confidence=False,  # faster
            include_punctuation_in_confidence=False,
        )

        text = result["text"].strip()
        if not text:
            return "(không nhận diện được nội dung)"

        return text

    except Exception as e:
        return f"[Lỗi ASR: {str(e)}]"


def process_audio_with_transcription(
    embedding_framework: str,
    embedding_model: str,
    speaker_segmentation_model: str,
    input_num_speakers: str,
    input_threshold: str,
    audio_input,
):
    """
    Main processing function:
      1. Run speaker diarization
      2. For each segment → run Whisper transcription
      3. Return both results
    """
    if audio_input is None:
        return "", build_result_html("", "Vui lòng tải file âm thanh hoặc ghi âm trước.")

    # ── Prepare file ─────────────────────────────────────────────
    if isinstance(audio_input, str):  # filepath
        input_filename = audio_input
    else:  # gradio audio tuple (sr, data)
        sr, data = audio_input
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf

            sf.write(tmp.name, data, sr)
            input_filename = tmp.name

    print(f"Processing file: {input_filename}")

    # ── Diarization ──────────────────────────────────────────────
    try:
        num_speakers = int(input_num_speakers)
    except ValueError:
        return "", build_result_html("", "Số lượng người nói phải là số nguyên.")

    threshold = 0.0
    if num_speakers <= 0:
        try:
            threshold = float(input_threshold)
        except ValueError:
            return "", build_result_html("", "Ngưỡng clustering không hợp lệ.")

    # Convert to proper wav 16kHz mono if needed
    wav_filename = str(uuid.uuid4()) + ".wav"
    os.system(
        f'ffmpeg -hide_banner -loglevel error -i "{input_filename}" '
        f"-ar 16000 -ac 1 {wav_filename} -y"
    )

    try:
        audio, sample_rate = read_wave(wav_filename)
    except Exception as e:
        return "", build_result_html("", f"Lỗi đọc file wave: {str(e)}")

    start_time = time.time()

    sd = get_speaker_diarization(
        segmentation_model=speaker_segmentation_model,
        embedding_model=embedding_model,
        num_clusters=num_speakers,
        threshold=threshold,
    )

    segments = sd.process(audio).sort_by_start_time()

    # ── Build diarization text ───────────────────────────────────
    diarization_lines = []
    transcription_lines = []

    for i, seg in enumerate(segments, 1):
        start = seg.start
        end = seg.end
        speaker_id = f"SPEAKER_{seg.speaker:02d}"

        time_str = f"[{format_time(start)} → {format_time(end)}]"
        diarization_line = f"{time_str} {speaker_id}"
        diarization_lines.append(diarization_line)

        # Transcribe this segment
        text = transcribe_segment(audio, sample_rate, start, end)
        transcription_lines.append(f"{time_str} **{speaker_id}**: {text}")

    diarization_text = "\n".join(diarization_lines)
    transcription_text = "\n".join(transcription_lines)

    # ── Timing info ──────────────────────────────────────────────
    duration = len(audio) / sample_rate
    elapsed = time.time() - start_time
    rtf = elapsed / duration if duration > 0 else 0

    info = f"""
Thời lượng file: {duration:.2f} giây  
Thời gian xử lý: {elapsed:.2f} giây  
RTF: {rtf:.2f}x  
(Whisper chạy trên: {WHISPER_DEVICE.upper()})
    """

    if rtf > 1.5:
        info += "\n(Lần đầu load model sẽ chậm hơn. Chạy lần thứ 2 sẽ nhanh hơn.)"

    # ── Clean up temporary files ────────────────────────────────
    try:
        os.unlink(wav_filename)
    except:
        pass

    if not isinstance(audio_input, str):
        try:
            os.unlink(input_filename)
        except:
            pass

    # ── Final output ─────────────────────────────────────────────
    return (
        diarization_text,
        build_result_html(diarization_text, transcription_text + "\n\n" + info),
    )


# ────────────────────────────────────────────────────────────────
#        Gradio Interface
# ────────────────────────────────────────────────────────────────

css = """
.result { margin: 1em 0; }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Speaker Diarization + Transcription (Whisper)")

    gr.Markdown(
        """
    **Hướng dẫn sử dụng:**
    - Chọn mô hình embedding & segmentation mong muốn
    - Nếu biết trước số người nói → điền vào "Number of speakers"
    - Nếu không biết → để 0 và điều chỉnh ngưỡng clustering
    """
    )

    with gr.Row():
        framework = gr.Radio(
            choices=list(embedding2models.keys()),
            label="Embedding Framework",
            value=list(embedding2models.keys())[0],
        )

        emb_model = gr.Dropdown(
            choices=embedding2models[framework.value],
            label="Embedding Model",
            value=embedding2models[framework.value][0],
        )

        seg_model = gr.Dropdown(
            choices=speaker_segmentation_models,
            label="Segmentation Model",
            value=speaker_segmentation_models[0],
        )

    with gr.Row():
        num_speakers = gr.Textbox(
            label="Number of speakers", value="0", max_lines=1
        )
        threshold = gr.Textbox(
            label="Clustering threshold (when num=0)", value="0.5", max_lines=1
        )

    framework.change(
        lambda f: gr.Dropdown(
            choices=embedding2models[f], value=embedding2models[f][0]
        ),
        inputs=framework,
        outputs=emb_model,
    )

    with gr.Tabs():
        with gr.TabItem("Upload Audio"):
            audio_upload = gr.Audio(sources=["upload"], type="filepath")
            btn_upload = gr.Button("Xử lý → Diarization + Transcription")

            out_diar = gr.Textbox(label="Kết quả phân đoạn người nói")
            out_html = gr.HTML()

            btn_upload.click(
                process_audio_with_transcription,
                inputs=[
                    framework,
                    emb_model,
                    seg_model,
                    num_speakers,
                    threshold,
                    audio_upload,
                ],
                outputs=[out_diar, out_html],
            )

        with gr.TabItem("Record"):
            audio_record = gr.Audio(sources=["microphone"], type="filepath")
            btn_record = gr.Button("Xử lý → Diarization + Transcription")

            out_diar_rec = gr.Textbox(label="Kết quả phân đoạn người nói")
            out_html_rec = gr.HTML()

            btn_record.click(
                process_audio_with_transcription,
                inputs=[
                    framework,
                    emb_model,
                    seg_model,
                    num_speakers,
                    threshold,
                    audio_record,
                ],
                outputs=[out_diar_rec, out_html_rec],
            )

    gr.Markdown(
        """
    ---
    Technology stack:
    • Speaker Diarization → sherpa-onnx (Next-gen Kaldi)
    • Transcription → whisper-timestamped (large-v3-turbo)
    • Framework → Gradio
    """
    )


if __name__ == "__main__":
    demo.launch()
    # demo.launch(share=True)   # uncomment if you want public link