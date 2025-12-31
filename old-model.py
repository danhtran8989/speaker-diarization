# model.py
# Copyright 2024 Xiaomi Corp. (authors: Fangjun Kuang)
#
# Licensed under the Apache License, Version 2.0 (the "License");

import wave
from functools import lru_cache
from typing import Tuple

import numpy as np
import sherpa_onnx
from huggingface_hub import hf_hub_download
import onnxruntime as ort

# Detect available providers once at import time
providers = ort.get_available_providers()
use_cuda = "CUDAExecutionProvider" in providers
provider = "cuda" if use_cuda else "cpu"

print(f"ONNX Runtime providers: {providers}")
print(f"Using execution provider: {provider.upper()} {'(CUDA available)' if use_cuda else '(CPU only)'}")

def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """
    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)
        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


@lru_cache(maxsize=30)
def get_file(
    repo_id: str,
    filename: str,
    subfolder: str = ".",
) -> str:
    nn_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return nn_model_filename


def get_speaker_segmentation_model(repo_id) -> str:
    assert repo_id in (
        "pyannote/segmentation-3.0",
        # "pyannote/segmentation-3.1,
        "Revai/reverb-diarization-v1",
        "Revai/reverb-diarization-v2"
    )
    if repo_id == "pyannote/segmentation-3.0":
        return get_file(
            repo_id="csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
            filename="model.onnx",
        )
    elif repo_id == "Revai/reverb-diarization-v1":
        return get_file(
            repo_id="csukuangfj/sherpa-onnx-reverb-diarization-v1",
            filename="model.onnx",
        )
    elif repo_id == "Revai/reverb-diarization-v2":
        return get_file(
            repo_id="csukuangfj/sherpa-onnx-reverb-diarization-v2",
            filename="model.onnx",
        )


def get_speaker_embedding_model(model_name) -> str:
    assert (
        model_name
        in three_d_speaker_embedding_models
        + nemo_speaker_embedding_models
        + wespeaker_embedding_models
    )
    model_name = model_name.split("|")[0]
    return get_file(
        repo_id="csukuangfj/speaker-embedding-models",
        filename=model_name,
    )


def get_speaker_diarization(
    segmentation_model: str, embedding_model: str, num_clusters: int, threshold: float
):
    segmentation = get_speaker_segmentation_model(segmentation_model)
    embedding = get_speaker_embedding_model(embedding_model)

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation
            ),
            num_threads=4,
            debug=False,
            provider=provider,  # "cuda" or "cpu"
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding,
            num_threads=2,
            debug=False,
            provider=provider,  # "cuda" or "cpu"
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )

    print("config", config)
    if not config.validate():
        raise RuntimeError(
            "Please check your config and make sure all required files exist"
        )
    return sherpa_onnx.OfflineSpeakerDiarization(config)


speaker_segmentation_models = [
    "pyannote/segmentation-3.0",
    "Revai/reverb-diarization-v1",
    "Revai/reverb-diarization-v2"
]

nemo_speaker_embedding_models = [
    "nemo_en_speakerverification_speakernet.onnx|22MB",
    "nemo_en_titanet_large.onnx|97MB",
    "nemo_en_titanet_small.onnx|38MB",
]

three_d_speaker_embedding_models = [
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx|37.8MB",
    "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx|28.2MB",
    "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx|27MB",
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx|27MB",
    "3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx|37.8MB",
    "3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx|111MB",
    "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx|25.3MB",
    "3dspeaker_speech_eres2net_sv_zh-cn_16k-common.onnx|210MB",
    "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx|68.1MB",
]

wespeaker_embedding_models = [
    "wespeaker_en_voxceleb_CAM++.onnx|28MB",
    "wespeaker_en_voxceleb_CAM++_LM.onnx|28MB",
    "wespeaker_en_voxceleb_resnet152_LM.onnx|76MB",
    "wespeaker_en_voxceleb_resnet221_LM.onnx|91MB",
    "wespeaker_en_voxceleb_resnet293_LM.onnx|110MB",
    "wespeaker_en_voxceleb_resnet34.onnx|26MB",
    "wespeaker_en_voxceleb_resnet34_LM.onnx|26MB",
    "wespeaker_zh_cnceleb_resnet34.onnx|26MB",
    "wespeaker_zh_cnceleb_resnet34_LM.onnx|26MB",
]

embedding2models = {
    "3D-Speaker": three_d_speaker_embedding_models,
    "NeMo": nemo_speaker_embedding_models,
    "WeSpeaker": wespeaker_embedding_models,
}

