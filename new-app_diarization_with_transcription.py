#!/usr/bin/env python3
# app_diarization_with_transcription.py
"""
Gradio app: Speaker Diarization + Whisper transcription
Improved dark/light theme support + auto-save transcription to file
Last update: Dec 2025
"""

import os
import shutil
import tempfile
import time
import uuid
import numpy as np
import torch
import gradio as gr
import sherpa_onnx
import whisper_timestamped as whisper
import soundfile as sf

# Assuming model.py is in the same directory
from model import (
    embedding2models,
    get_file,
    get_speaker_diarization,
    read_wave,
    speaker_segmentation_models,
)

from login_and_download import hf_login
hf_login()
# ────────────────────────────────────────────────────────────────
# Configuration & Globals
# ────────────────────────────────────────────────────────────────

WHISPER_MODEL_NAME = "openai/whisper-large-v3-turbo"
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

# ────────────────────────────────────────────────────────────────
# Formatting & HTML Output
# ────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"

def build_result_html(diarization_text: str, transcription_text: str, info: str) -> str:
    return f"""
    <div class="result-container">
        <h3>Speaker Diarization + Transcription</h3>
        
        <div class="section diarization">
            <strong>Diarization result</strong>
            <div class="content">{diarization_text.replace('\n', '<br>')}</div>
        </div>
        
        <div class="section transcription">
            <strong>Transcription (with speakers)</strong>
            <div class="content">{transcription_text.replace('\n', '<br>')}</div>
        </div>
        
        <div class="info-box">
            {info.replace('\n', '<br>')}
        </div>
    </div>
    """

# ────────────────────────────────────────────────────────────────
# Transcription Logic
# ────────────────────────────────────────────────────────────────

def transcribe_segment(audio_np: np.ndarray, sample_rate: int, start: float, end: float):
    duration = len(audio_np) / sample_rate
    if end > duration:
        end = duration

    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment_audio = audio_np[start_sample:end_sample]

    if len(segment_audio) < 320:
        return "(đoạn quá ngắn)"

    try:
        result = whisper.transcribe_timestamped(
            load_whisper_model(),
            segment_audio,
            language=WHISPER_LANGUAGE,
            beam_size=1,
            temperature=0.0,
            best_of=1,
            compute_word_confidence=False,
            include_punctuation_in_confidence=False,
            vad=True,
        )

        text = result.get("text", "").strip()
        return text if text else "(không nhận diện được nội dung)"

    except Exception as e:
        print(f"Transcription error {start:.1f}–{end:.1f}: {str(e)}")
        return f"[Lỗi phiên âm: {str(e)}]"

# ────────────────────────────────────────────────────────────────
# Main Processing + Save to file
# ────────────────────────────────────────────────────────────────

def process_audio_with_transcription(
    embedding_framework: str,
    embedding_model: str,
    speaker_segmentation_model: str,
    input_num_speakers: str,
    input_threshold: str,
    audio_input,
):
    if audio_input is None:
        return "", build_result_html("", "Vui lòng tải file âm thanh hoặc ghi âm trước.", "")

    # Prepare input file
    if isinstance(audio_input, str):
        input_filename = audio_input
    else:
        sr, data = audio_input
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, data, sr)
            input_filename = tmp.name

    print(f"Processing: {input_filename}")

    # Convert to 16kHz mono
    wav_filename = f"temp_{uuid.uuid4()}.wav"
    os.system(
        f'ffmpeg -hide_banner -loglevel error -i "{input_filename}" '
        f"-ar 16000 -ac 1 {wav_filename} -y"
    )

    try:
        audio, sample_rate = read_wave(wav_filename)
    except Exception as e:
        return "", build_result_html("", f"Lỗi đọc file wave: {str(e)}", "")

    # Diarization parameters
    try:
        num_speakers = int(input_num_speakers)
    except ValueError:
        return "", build_result_html("", "Số lượng người nói phải là số nguyên.", "")

    threshold = 0.0
    if num_speakers <= 0:
        try:
            threshold = float(input_threshold)
        except ValueError:
            return "", build_result_html("", "Ngưỡng clustering không hợp lệ.", "")

    start_time = time.time()

    sd = get_speaker_diarization(
        segmentation_model=speaker_segmentation_model,
        embedding_model=embedding_model,
        num_clusters=num_speakers,
        threshold=threshold,
    )

    segments = sd.process(audio).sort_by_start_time()

    # Build results
    diarization_lines = []
    transcription_lines = []
    plain_transcription_lines = []  # for file saving

    for seg in segments:
        start = seg.start
        end = seg.end
        speaker_id = f"SPEAKER_{seg.speaker:02d}"
        time_str = f"[{format_time(start)} → {format_time(end)}]"

        diarization_lines.append(f"{time_str} {speaker_id}")
        
        text = transcribe_segment(audio, sample_rate, start, end)
        transcription_lines.append(f"{time_str} **{speaker_id}**: {text}")
        plain_transcription_lines.append(f"{time_str} {speaker_id}: {text}")

    diarization_text = "\n".join(diarization_lines)
    transcription_text = "\n".join(transcription_lines)

    # Performance info
    duration = len(audio) / sample_rate
    elapsed = time.time() - start_time
    rtf = elapsed / duration if duration > 0 else 0

    info = f"""Thời lượng file: {duration:.2f} giây  
Thời gian xử lý: {elapsed:.2f} giây  
RTF: {rtf:.2f}x  
(Whisper chạy trên: {WHISPER_DEVICE.upper()})"""
    
    if rtf > 1.5:
        info += "\n(Lần đầu load model sẽ chậm hơn. Chạy lần thứ 2 sẽ nhanh hơn.)"

    # Cleanup
    for f in [wav_filename, input_filename]:
        if f != input_filename or not isinstance(audio_input, str):
            try:
                os.unlink(f)
            except:
                pass

    # ── Save transcription to file ───────────────────────────────
    output_file = "segment_transcription.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== SEGMENT TRANSCRIPTION ===\n")
            f.write(f"Processing time: {elapsed:.2f}\n")
            f.write(f"Duration of Audio file: {duration:.2f}s | RTF - Real-time ratio): {rtf:.2f}x\n")
            f.write("-" * 50 + "\n\n")
            f.write("\n".join(plain_transcription_lines))
            f.write("\n\n" + "="*50 + "\n")
        print(f"Transcription saved to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"Failed to save transcription file: {e}")
    print("DONE")
    return diarization_text, build_result_html(diarization_text, transcription_text, info)

# ────────────────────────────────────────────────────────────────
# CSS - Modern, high contrast, theme-aware
# ────────────────────────────────────────────────────────────────

css = """
:root {
    --bg-main:      var(--background-fill-primary, #0f172a);
    --bg-panel:     var(--background-fill-secondary, #1e293b);
    --text-main:    var(--text-primary, #f1f5f9);
    --text-muted:   var(--text-secondary, #94a3b8);
    --border:       var(--border-color-primary, #334155);
    --accent:       #38bdf8;
    --warning:      #f59e0b;
    --card-bg:      rgba(30, 41, 59, 0.6);
}

@media (prefers-color-scheme: light) {
    :root {
        --bg-main:      #f8fafc;
        --bg-panel:     #ffffff;
        --text-main:    #0f172a;
        --text-muted:   #475569;
        --border:       #cbd5e1;
        --accent:       #0ea5e9;
        --warning:      #d97706;
        --card-bg:      rgba(241, 245, 249, 0.7);
    }
}

.gradio-container {
    background: var(--bg-main);
    color: var(--text-main);
    font-family: system-ui, sans-serif;
}

.result-container {
    background: var(--card-bg);
    backdrop-filter: blur(6px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.25rem 0;
    color: var(--text-main);
}

.result-container h3 {
    margin-top: 0;
    color: var(--accent);
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
}

.section {
    margin: 1.25rem 0;
    padding: 1.1rem;
    border-radius: 8px;
    background: rgba(0,0,0,0.12);
    transition: background 0.3s;
}

@media (prefers-color-scheme: light) {
    .section {
        background: rgba(255,255,255,0.7);
    }
}

.section.diarization {
    border-left: 4px solid var(--accent);
}

.section.transcription {
    border-left: 4px solid var(--warning);
}

.section strong {
    color: var(--text-main);
    display: block;
    margin-bottom: 0.6rem;
    font-size: 1.1em;
}

.section .content {
    line-height: 1.55;
    color: var(--text-main);
    white-space: pre-wrap;
}

.info-box {
    margin-top: 1.5rem;
    padding: 1rem;
    background: rgba(0,0,0,0.15);
    border-radius: 8px;
    font-size: 0.92rem;
    color: var(--text-muted);
    border: 1px dashed var(--border);
}

.gr-button-primary {
    background: var(--accent) !important;
}

.gr-button-primary:hover {
    background: color-mix(in srgb, var(--accent) 80%, black) !important;
}

.gr-textbox, .gr-dropdown, .gr-radio {
    background: var(--bg-panel) !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
}
"""

# ────────────────────────────────────────────────────────────────
# Gradio Interface
# ────────────────────────────────────────────────────────────────

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Speaker Diarization + Transcription (Whisper)")

    gr.Markdown(
        """
    **Hướng dẫn sử dụng:**
    • Chọn mô hình embedding & segmentation  
    • Nếu biết số người nói → nhập vào "Number of speakers"  
    • Nếu không biết → để 0 và điều chỉnh ngưỡng clustering
    
    Kết quả phiên âm đầy đủ sẽ được lưu vào file **segment_transcription.txt**
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
        num_speakers = gr.Textbox(label="Number of speakers", value="0", max_lines=1)
        threshold = gr.Textbox(
            label="Clustering threshold (when num=0)", value="0.5", max_lines=1
        )

    framework.change(
        lambda f: gr.Dropdown(choices=embedding2models[f], value=embedding2models[f][0]),
        inputs=framework,
        outputs=emb_model,
    )

    with gr.Tabs():
        with gr.TabItem("Upload Audio"):
            audio_upload = gr.Audio(sources=["upload"], type="filepath", label="Tải file lên")
            btn_upload = gr.Button("Xử lý → Phân đoạn + Phiên âm", variant="primary")
            out_diar = gr.Textbox(label="Kết quả phân đoạn người nói", lines=8)
            out_html = gr.HTML()
            btn_upload.click(
                process_audio_with_transcription,
                inputs=[framework, emb_model, seg_model, num_speakers, threshold, audio_upload],
                outputs=[out_diar, out_html],
            )

        with gr.TabItem("Ghi âm trực tiếp"):
            audio_record = gr.Audio(sources=["microphone"], type="filepath", label="Ghi âm")
            btn_record = gr.Button("Xử lý → Phân đoạn + Phiên âm", variant="primary")
            out_diar_rec = gr.Textbox(label="Kết quả phân đoạn người nói", lines=8)
            out_html_rec = gr.HTML()
            btn_record.click(
                process_audio_with_transcription,
                inputs=[framework, emb_model, seg_model, num_speakers, threshold, audio_record],
                outputs=[out_diar_rec, out_html_rec],
            )

    gr.Markdown(
        """
    ---
    Công nghệ sử dụng:  
    • Phân đoạn người nói → **sherpa-onnx** (Next-gen Kaldi)  
    • Phiên âm → **whisper-timestamped** (large-v3-turbo)  
    • Giao diện → **Gradio**
    """
    )

if __name__ == "__main__":
    demo.launch()
    # demo.launch(share=True)  # uncomment for public link
