#!/usr/bin/env python3
# app_diarization_with_transcription.py
"""
Gradio app: Speaker Diarization + Whisper transcription
With silence detection: silent segments are skipped for transcription
Added GUI option to adjust silence threshold (-20 dB to +10 dB)
Enhanced saved transcription file with full configuration details
Fixed sherpa-onnx FastClusteringConfig compatibility (Dec 2025)
Last update: Dec 2025
"""
import os
import tempfile
import time
import uuid
import numpy as np
import torch
import gradio as gr
import sherpa_onnx
import whisper_timestamped as whisper
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_silence

# Assuming model.py is in the same directory
from model import (
    embedding2models,
    get_file,
    read_wave,
    speaker_segmentation_models,
)
from login_and_download import hf_login
# hf_login()  # Uncomment if needed for gated models

# ────────────────────────────────────────────────────────────────
# Global variables
# ────────────────────────────────────────────────────────────────
whisper_model = None
current_whisper_model_name = None
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Whisper will run on device: {WHISPER_DEVICE.upper()}")

# ────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────
def load_whisper_model(model_name: str):
    global whisper_model, current_whisper_model_name
    if whisper_model is None or current_whisper_model_name != model_name:
        print(f"Loading Whisper model '{model_name}' on {WHISPER_DEVICE} ...")
        whisper_model = whisper.load_model(model_name, device=WHISPER_DEVICE)
        current_whisper_model_name = model_name
    return whisper_model

# ────────────────────────────────────────────────────────────────
# Formatting
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
# Silence detection function
# ────────────────────────────────────────────────────────────────
def is_segment_silent(seg_audio: AudioSegment,
                      min_silence_len: int = 500,
                      silence_thresh: int = -40,
                      silent_ratio_threshold: float = 0.9) -> bool:
    if seg_audio.dBFS >= silence_thresh - 5:  # clearly has speech
        return False

    silent_ranges = detect_silence(
        seg_audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    if not silent_ranges:
        return False

    total_silent_ms = sum(stop - start for start, stop in silent_ranges)
    segment_duration_ms = len(seg_audio)

    return total_silent_ms >= segment_duration_ms * silent_ratio_threshold

# ────────────────────────────────────────────────────────────────
# Transcription Logic
# ────────────────────────────────────────────────────────────────
def transcribe_segment(audio_np: np.ndarray, sample_rate: int, start: float, end: float, model_name: str, language: str):
    duration = len(audio_np) / sample_rate
    if end > duration:
        end = duration
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment_audio = audio_np[start_sample:end_sample]
    if len(segment_audio) < 320:
        return "(đoạn quá ngắn)"

    try:
        model = load_whisper_model(model_name)
        result = whisper.transcribe_timestamped(
            model,
            segment_audio,
            language=language if language != "auto" else None,
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
# Main Processing (WITH FIX FOR sherpa-onnx FastClusteringConfig)
# ────────────────────────────────────────────────────────────────
def process_audio_with_transcription(
    embedding_framework: str,
    embedding_model: str,
    speaker_segmentation_model: str,
    input_num_speakers: str,
    input_threshold: str,
    whisper_model_choice: str,
    whisper_language: str,
    silence_threshold_db: int,
    audio_input,
):
    global whisper_model, current_whisper_model_name

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

    try:
        clustering_threshold = float(input_threshold)
    except ValueError:
        return "", build_result_html("", "Ngưỡng clustering không hợp lệ.", "")

    start_time = time.time()

    # === FIXED PART: Correctly configure FastClusteringConfig ===
    if num_speakers > 0:
        # Fixed number of speakers → num_clusters = positive int, threshold ignored
        clustering = sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers,
            threshold=0.5  # value doesn't matter when num_clusters > 0
        )
    else:
        # Auto detect → num_clusters = -1, use user threshold
        clustering = sherpa_onnx.FastClusteringConfig(
            num_clusters=-1,
            threshold=clustering_threshold
        )

    # Load models (assuming get_file handles downloading/caching)
    segmentation_model = get_file(speaker_segmentation_model)
    embedding_model_path = get_file(embedding_model)

    # Create SpeakerDiarization object with correct config
    sd = sherpa_onnx.SpeakerDiarization(
        segmentation_model=segmentation_model,
        embedding_model=embedding_model_path,
        clustering=clustering,
        min_duration_on=0.1,   # optional: adjust if needed
        min_duration_off=0.1,
    )

    segments = sd.process(audio).sort_by_start_time()

    # Convert full audio to pydub for silence detection
    audio_pydub = AudioSegment(
        data=audio.tobytes(),
        sample_width=audio.itemsize,
        frame_rate=sample_rate,
        channels=1
    )

    # Results
    diarization_lines = []
    transcription_lines = []
    plain_transcription_lines = []

    for seg in segments:
        start = seg.start
        end = seg.end
        speaker_id = f"SPEAKER_{seg.speaker:02d}"
        time_str = f"[{format_time(start)} → {format_time(end)}]"
        diarization_lines.append(f"{time_str} {speaker_id}")

        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        seg_audio = audio_pydub[start_ms:end_ms]

        if is_segment_silent(seg_audio, silence_thresh=silence_threshold_db):
            text = "(im lặng)"
        else:
            text = transcribe_segment(audio, sample_rate, start, end, whisper_model_choice, whisper_language)
            if not text.strip():
                text = "(không nhận diện được nội dung)"

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
Whisper model: {whisper_model_choice}
Device: {WHISPER_DEVICE.upper()}
Silence detection: Enabled (threshold = {silence_threshold_db} dBFS)"""

    if rtf > 1.5:
        info += "\n(Lần đầu load model sẽ chậm hơn. Chạy lần thứ 2 sẽ nhanh hơn.)"

    # Cleanup
    for f in [wav_filename]:
        if os.path.exists(f):
            try:
                os.unlink(f)
            except:
                pass
    if not isinstance(audio_input, str):
        try:
            os.unlink(input_filename)
        except:
            pass

    # Save transcription with full config
    output_file = "segment_transcription.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== SPEAKER DIARIZATION + TRANSCRIPTION RESULT ===\n\n")
            f.write("CONFIGURATION:\n")
            f.write(f"• Embedding Framework : {embedding_framework}\n")
            f.write(f"• Embedding Model : {embedding_model}\n")
            f.write(f"• Segmentation Model : {speaker_segmentation_model}\n")
            f.write(f"• Number of Speakers : {num_speakers if num_speakers > 0 else 'Auto'}\n")
            if num_speakers <= 0:
                f.write(f"• Clustering Threshold : {clustering_threshold}\n")
            f.write(f"• Whisper Model : {whisper_model_choice}\n")
            f.write(f"• Language : {whisper_language}\n")
            f.write(f"• Silence Threshold (dBFS) : {silence_threshold_db}\n")
            f.write(f"• Processing Device : {WHISPER_DEVICE.upper()}\n\n")
            f.write("PERFORMANCE:\n")
            f.write(f"• Audio Duration : {duration:.2f} seconds\n")
            f.write(f"• Processing Time : {elapsed:.2f} seconds\n")
            f.write(f"• Real-Time Factor (RTF) : {rtf:.2f}x\n\n")
            f.write("-" * 70 + "\n\n")
            f.write("TRANSCRIPTION BY SEGMENTS:\n\n")
            f.write("\n".join(plain_transcription_lines))
            f.write("\n\n" + "="*70 + "\n")
        print(f"Full transcription saved to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"Failed to save transcription file: {e}")

    return diarization_text, build_result_html(diarization_text, transcription_text, info)

# ────────────────────────────────────────────────────────────────
# CSS
# ────────────────────────────────────────────────────────────────
css = """
:root {
    --bg-main: var(--background-fill-primary, #0f172a);
    --bg-panel: var(--background-fill-secondary, #1e293b);
    --text-main: var(--text-primary, #f1f5f9);
    --text-muted: var(--text-secondary, #94a3b8);
    --border: var(--border-color-primary, #334155);
    --accent: #38bdf8;
    --warning: #f59e0b;
    --card-bg: rgba(30, 41, 59, 0.6);
}
@media (prefers-color-scheme: light) {
    :root {
        --bg-main: #f8fafc;
        --bg-panel: #ffffff;
        --text-main: #0f172a;
        --text-muted: #475569;
        --border: #cbd5e1;
        --accent: #0ea5e9;
        --warning: #d97706;
        --card-bg: rgba(241, 245, 249, 0.7);
    }
}
.gradio-container { background: var(--bg-main); color: var(--text-main); font-family: system-ui, sans-serif; }
.result-container { background: var(--card-bg); backdrop-filter: blur(6px); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin: 1.25rem 0; }
.result-container h3 { margin-top: 0; color: var(--accent); padding-bottom: 0.5rem; border-bottom: 2px solid var(--border); }
.section { margin: 1.25rem 0; padding: 1.1rem; border-radius: 8px; background: rgba(0,0,0,0.12); transition: background 0.3s; }
@media (prefers-color-scheme: light) { .section { background: rgba(255,255,255,0.7); } }
.section.diarization { border-left: 4px solid var(--accent); }
.section.transcription { border-left: 4px solid var(--warning); }
.section strong { color: var(--text-main); display: block; margin-bottom: 0.6rem; font-size: 1.1em; }
.section .content { line-height: 1.55; color: var(--text-main); white-space: pre-wrap; }
.info-box { margin-top: 1.5rem; padding: 1rem; background: rgba(0,0,0,0.15); border-radius: 8px; font-size: 0.92rem; color: var(--text-muted); border: 1px dashed var(--border); }
.gr-button-primary { background: var(--accent) !important; }
.gr-button-primary:hover { background: color-mix(in srgb, var(--accent) 80%, black) !important; }
.gr-textbox, .gr-dropdown, .gr-radio, .gr-slider { background: var(--bg-panel) !important; border-color: var(--border) !important; color: var(--text-main) !important; }
"""

# ────────────────────────────────────────────────────────────────
# Gradio Interface
# ────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="Speaker Diarization + Transcription") as demo:
    gr.Markdown("# Speaker Diarization + Transcription (Whisper)")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown(
                """
                **Hướng dẫn sử dụng:**
                • Chọn mô hình Whisper phù hợp với ngôn ngữ và tốc độ mong muốn
                • Chọn embedding & segmentation model
                • Nếu biết số người nói → nhập vào ô tương ứng
                • Nếu không biết → để 0 và điều chỉnh ngưỡng clustering
                • Điều chỉnh **Silence threshold** để kiểm soát độ nhạy phát hiện im lặng
                Kết quả phiên âm sẽ tự động lưu vào file **segment_transcription.txt** (với đầy đủ cấu hình)
                """
            )
        with gr.Column():
            whisper_model_dropdown = gr.Dropdown(
                choices=[
                    "vinai/PhoWhisper-large",
                    "openai/whisper-large",
                    "openai/whisper-large-v2",
                    "openai/whisper-large-v3",
                    "openai/whisper-large-v3-turbo",
                    "openai/whisper-medium",
                ],
                value="openai/whisper-large-v3-turbo",
                label="Whisper Model",
                info="PhoWhisper: tốt nhất cho tiếng Việt | Turbo: nhanh & tốt | Medium: cân bằng"
            )
            whisper_lang_dropdown = gr.Dropdown(
                choices=["vi", "auto"],
                value="vi",
                label="Ngôn ngữ (Language)",
                info="Chọn 'vi' cho PhoWhisper, 'auto' cho các model OpenAI"
            )

    def update_language(model_name):
        if "PhoWhisper" in model_name:
            return gr.update(value="vi")
        else:
            return gr.update(value="auto")

    whisper_model_dropdown.change(update_language, inputs=whisper_model_dropdown, outputs=whisper_lang_dropdown)

    with gr.Row():
        framework = gr.Radio(
            choices=list(embedding2models.keys()),
            label="Embedding Framework",
            value=list(embedding2models.keys())[1],
        )
        emb_model = gr.Dropdown(
            choices=embedding2models[framework.value],
            label="Embedding Model",
            value=embedding2models[framework.value][0],
        )
        seg_model = gr.Dropdown(
            choices=speaker_segmentation_models,
            label="Segmentation Model",
            value=speaker_segmentation_models[2],
        )

    framework.change(
        lambda f: gr.Dropdown(choices=embedding2models[f], value=embedding2models[f][0]),
        inputs=framework,
        outputs=emb_model,
    )

    with gr.Row():
        num_speakers = gr.Textbox(label="Number of speakers (0 = auto)", value="0", max_lines=1)
        threshold = gr.Textbox(label="Clustering threshold (when num=0)", value="0.85", max_lines=1)

    with gr.Row():
        silence_threshold_slider = gr.Slider(
            minimum=-20,
            maximum=10,
            value=-3,
            step=1,
            label="Silence Threshold (dBFS)",
            info="Giá trị càng cao → càng dễ coi là im lặng. Thường dùng -40 đến -30."
        )

    with gr.Tabs():
        with gr.TabItem("Upload Audio"):
            audio_upload = gr.Audio(sources=["upload"], type="filepath", label="Tải file âm thanh")
            btn_upload = gr.Button("Xử lý → Phân đoạn + Phiên âm", variant="primary")
            out_diar = gr.Textbox(label="Kết quả phân đoạn người nói", lines=10)
            out_html = gr.HTML()
            btn_upload.click(
                process_audio_with_transcription,
                inputs=[
                    framework, emb_model, seg_model,
                    num_speakers, threshold,
                    whisper_model_dropdown, whisper_lang_dropdown,
                    silence_threshold_slider,
                    audio_upload
                ],
                outputs=[out_diar, out_html],
            )

        with gr.TabItem("Ghi âm trực tiếp"):
            audio_record = gr.Audio(sources=["microphone"], type="filepath", label="Ghi âm từ micro")
            btn_record = gr.Button("Xử lý → Phân đoạn + Phiên âm", variant="primary")
            out_diar_rec = gr.Textbox(label="Kết quả phân đoạn người nói", lines=10)
            out_html_rec = gr.HTML()
            btn_record.click(
                process_audio_with_transcription,
                inputs=[
                    framework, emb_model, seg_model,
                    num_speakers, threshold,
                    whisper_model_dropdown, whisper_lang_dropdown,
                    silence_threshold_slider,
                    audio_record
                ],
                outputs=[out_diar_rec, out_html_rec],
            )

    gr.Markdown(
        """
        ---
        **Công nghệ sử dụng:**
        • Phân đoạn người nói → **sherpa-onnx**
        • Phiên âm → **whisper-timestamped**
        • Silence detection → **pydub** (skip Whisper on silent segments)
        • Giao diện → **Gradio**
        """
    )

if __name__ == "__main__":
    demo.launch()
    # demo.launch(share=True)  # Uncomment for public link
