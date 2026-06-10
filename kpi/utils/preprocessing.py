import glob
import os
import shutil
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Iterable

import ffmpeg
import librosa
import numpy as np
import spacy
import torch
import torchvision.models as models
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, logging

from kpi.utils.video import Video, get_duration

logging.set_verbosity_error()

spacy.require_gpu()  # type: ignore


class LangModel:
    def __init__(self, model_name: str, init_func):
        self.model_name = model_name
        self.init_func = init_func

    @cached_property
    def model(self) -> Any:
        return self.init_func(self.model_name)

    def encode(self, text: str) -> np.ndarray:
        return np.array(self.model.encode(text))

    def pipe(self, *args, **kwargs) -> Any:
        return self.model.pipe(*args, **kwargs)


SEN_MODEL = LangModel("distiluse-base-multilingual-cased", SentenceTransformer)
WORD_MODEL = LangModel("en_core_web_lg", spacy.load)


@dataclass
class Clip:
    idx: int
    start: float
    end: float
    vfn: str
    text: str
    text_vec: np.ndarray | None = None
    visual_vec: np.ndarray | None = None
    audio_vec: np.ndarray | None = None
    vec: np.ndarray | None = None


def slice_video(video: Video, method: str, **kwargs) -> Iterable[Clip]:
    if method == "srt":
        for i, srt in enumerate(video.srt):
            end = min(srt["start"] + srt["duration"], video.duration)
            start = srt["start"]
            if end < start:
                start = end - 2
            yield Clip(
                idx=i,
                vfn=video.video_fn,
                start=start,
                text=srt["text"],
                end=end,
            )
    elif method == "fixed":
        duration = kwargs.get("duration")
        if not duration:
            raise ValueError("duration should be provided for fixed method")
        cur_id = 0
        cur_start = 0
        srt_idx = 0
        while cur_start < video.duration:
            cur_end = min(cur_start + duration, video.duration)
            cur_text = ""
            while srt_idx < len(video.srt):
                srt = video.srt[srt_idx]
                if cur_start <= srt.start < cur_end or cur_start <= srt.end < cur_end:
                    cur_text += srt.text + " "
                    srt_idx += 1
                else:
                    break
            yield Clip(
                idx=cur_id,
                vfn=video.video_fn,
                start=cur_start,
                end=cur_end,
                text=cur_text,
            )
            cur_id += 1
            cur_start += duration
    else:
        raise ValueError(f"Unknown method: {method}")


def get_clip_vec(
    clip: Clip,
    *,
    visual_func=None,
    text_func=None,
    audio_func=None,
    merge_func=None,
) -> Clip:
    if not visual_func and not text_func and not audio_func:
        raise ValueError("At least one of the functions should be provided")
    if (
        sum([bool(visual_func), bool(text_func), bool(audio_func)]) > 1
        and not merge_func
    ):
        raise ValueError(
            "When multiple functions are provided, merge_func should be provided"
        )
    cur = Clip(
        idx=clip.idx,
        start=clip.start,
        end=clip.end,
        vfn=clip.vfn,
        text=clip.text,
    )
    if visual_func:
        cur.visual_vec = visual_func(clip)
    if text_func:
        cur.text_vec = text_func(clip)
    if audio_func:
        cur.audio_vec = audio_func(clip)
    if merge_func:
        cur.vec = merge_func(cur)
    else:
        if cur.visual_vec is not None:
            cur.vec = cur.visual_vec
        elif cur.text_vec is not None:
            cur.vec = cur.text_vec
        else:
            cur.vec = cur.audio_vec
    return cur


def get_video_feature_clip(video: Video, method: str, **kwargs) -> Iterable[Clip]:
    if method == "fixed" and "duration" not in kwargs:
        return get_video_feature_clip(video, method="fixed", duration=10)

    return [
        get_clip_vec(
            clip,
            visual_func=clip_visual_resnet152_1fps,
            text_func=clip_sentence_embedding,
            audio_func=clip_audio_wav2vec,
            merge_func=clip_merge_concat,
        )
        for clip in tqdm(list(slice_video(video, method, **kwargs)))
    ]


def clip_sentence_embedding(clip: Clip) -> np.ndarray:
    return np.array(SEN_MODEL.encode(clip.text))


def clip_visual_resnet152_1fps(clip: Clip) -> np.ndarray:
    if clip.start == int(get_duration(clip.vfn)):
        image_fos = get_image_fos(clip.vfn, clip.start - 1, clip.end, fps=1)
    else:
        image_fos = get_image_fos(clip.vfn, clip.start, clip.end, fps=1)
    assert image_fos, f"No frames extracted:{clip.vfn} {clip.start} {clip.end}"
    return np.array(get_resnet152_embeddings(image_fos)).max(axis=0)


def clip_audio_wav2vec(clip: Clip) -> np.ndarray:
    if clip.start == int(get_duration(clip.vfn)):
        wav_path = get_wav_audio(clip.vfn, clip.start - 1, clip.end)
    else:
        wav_path = get_wav_audio(clip.vfn, clip.start, clip.end)
    return get_wav2vec_embedding(wav_path)


RESNET152_MODEL = None


def get_resnet152_embeddings(image_fos: list[str]) -> np.ndarray:
    global RESNET152_MODEL
    if not RESNET152_MODEL:
        RESNET152_MODEL = models.resnet152(
            weights=models.ResNet152_Weights.IMAGENET1K_V1
        )
    resnet152_torch = RESNET152_MODEL
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensors = []
    for path in image_fos:
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img)
        img_tensors.append(img_tensor)
    batch_tensor = torch.stack(img_tensors)
    resnet152_torch.eval()
    with torch.no_grad():
        batch_features = resnet152_torch(batch_tensor)
        return batch_features


def get_image_fos(vfn: str, start: float, end: float, fps: int) -> list[str]:
    # extract frames to /tmp folder and return list of file names with python-ffmpeg
    DIR = "./tmp/frames"
    if os.path.exists(DIR):
        shutil.rmtree(DIR)
    os.makedirs(DIR)
    if end - start > 1:
        cur = ffmpeg.input(vfn, ss=start, to=end)
        cur = cur.filter("fps", fps=f"{fps}")
    else:
        start = max(0, start - 1)
        end = end + 1
        cur = ffmpeg.input(vfn, ss=start, to=end)
    cur.output(f"{DIR}/%d.jpg").run(quiet=True)
    return sorted(glob.glob(f"{DIR}/*.jpg"))


def get_wav_audio(vfn: str, start: float, end: float, sr=16000) -> str:
    # extract audio to /tmp folder and return path with python-ffmpeg
    DIR = "./tmp/audio"
    if os.path.exists(DIR):
        shutil.rmtree(DIR)
    if end - start < 1:
        start = max(0, start - 1)
        end = end + 1
    os.makedirs(DIR)
    (
        ffmpeg.input(vfn, ss=start, to=end)
        .output(f"{DIR}/audio.wav", ar=sr)
        .run(quiet=True)
    )
    return f"{DIR}/audio.wav"


WAV2VEC_MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"

WAV2VEC_FEATURE_EXTRACTOR = None
WAV2VEC_MODEL = None


def get_wav2vec_embedding(
    wav_path: str,
    model_name=WAV2VEC_MODEL_NAME,
) -> np.ndarray:
    audio, sr = librosa.load(wav_path, sr=16000)
    global WAV2VEC_FEATURE_EXTRACTOR
    if not WAV2VEC_FEATURE_EXTRACTOR:
        WAV2VEC_FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    global WAV2VEC_MODEL
    if not WAV2VEC_MODEL:
        WAV2VEC_MODEL = Wav2Vec2Model.from_pretrained(model_name, use_safetensors=True)  # type: ignore
    feature_extractor = WAV2VEC_FEATURE_EXTRACTOR
    model = WAV2VEC_MODEL
    input_values = feature_extractor(
        audio, return_tensors="pt", sampling_rate=sr
    ).input_values
    with torch.no_grad():
        outputs = model(input_values)
    return outputs.extract_features.squeeze().mean(axis=0)


def clip_merge_concat(clip: Clip) -> np.ndarray:
    arr = []
    if clip.visual_vec is not None:
        arr.append(clip.visual_vec)
    if clip.text_vec is not None:
        arr.append(clip.text_vec)
    if clip.audio_vec is not None:
        arr.append(clip.audio_vec)
    return np.concatenate(arr)


def _gen_feature_func(name) -> Callable:
    def feature_func(clip: Clip) -> np.ndarray:
        feat_getter = {
            "a": lambda x: x.audio_vec,
            "t": lambda x: x.text_vec,
            "v": lambda x: x.visual_vec,
        }

        feats = [feat_getter[feat](clip) for feat in name]

        return np.concatenate(feats)

    return feature_func


FEATURE_FUNC = {
    "t": _gen_feature_func("t"),
    "tv": _gen_feature_func("tv"),
    "ta": _gen_feature_func("ta"),
    "tva": _gen_feature_func("tva"),
    "v": _gen_feature_func("v"),
    "va": _gen_feature_func("va"),
    "a": _gen_feature_func("a"),
}
