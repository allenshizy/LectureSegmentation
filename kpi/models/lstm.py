import random
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import click
import numpy as np
import torch
from torch import nn

from kpi.cli import register_model_runner
from kpi.models.base_model import Model
from kpi.utils import batched_data
from kpi.utils.preprocessing import FEATURE_FUNC, get_video_feature_clip
from kpi.utils.video import Video


class BiLSTM(Model):
    def __init__(
        self,
        visual_dim: int,
        audio_dim: int,
        text_dim: int,
        hidden_size: int = 256,
        slice_method: Literal["srt"] = "srt",
        device: str = "",
        seed: int = 0,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight: list[float] | None = None,
        feat_func: str = "tva",
        save_feature_vectors: bool = False,
        load_feature_vectors_path: str = "",
        save_model_weights: bool = False,
        load_model_weights_path: str = "",
        artifact_dir: str = "./artifacts/lstm",
        artifact_prefix: str = "bilstm",
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.slice_method: Literal["srt"] = slice_method
        self.seed = seed
        # Keep the original feature selector so preprocessing can skip unused modalities.
        self.feature_keys = feat_func
        self.feat_func = FEATURE_FUNC[feat_func]
        self.input_dim = 0
        if "t" in feat_func:
            self.input_dim += text_dim
        if "v" in feat_func:
            self.input_dim += visual_dim
        if "a" in feat_func:
            self.input_dim += audio_dim
        self.hidden_size = hidden_size

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight = weight
        self.save_feature_vectors = save_feature_vectors
        self.load_feature_vectors_path = load_feature_vectors_path
        self.save_model_weights = save_model_weights
        self.load_model_weights_path = load_model_weights_path
        self.artifact_dir = Path(artifact_dir)
        self.artifact_prefix = artifact_prefix
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._requires_training = True

        self._model = self._build_model()

        super().__init__()

    def _build_model(self):
        return _Model(self.input_dim, self.hidden_size).to(self.device)

    @property
    def requires_training(self):
        return self._requires_training

    def _get_labels_from_clips(self, clips, frags):
        def _get_label_from_clip_ends(clip_ends: list[float], frag: list[float]):
            ret = []
            last_end = -1
            for cur_end in clip_ends:
                cur_label = 0
                for f in frag:
                    if last_end <= f < cur_end:
                        cur_label = 1
                        break
                ret.append(cur_label)
                last_end = cur_end
            assert len(ret) == len(clip_ends)
            return ret

        labels = []
        for video_clips, frag in zip(clips, frags):
            clip_ends = [clip.end for clip in video_clips]
            labels.append(_get_label_from_clip_ends(clip_ends, frag))

        return labels

    def _get_all_clips(self, videos):
        return [
            list(
                get_video_feature_clip(
                    video, self.slice_method, feature_keys=self.feature_keys
                )
            )
            for video in videos
        ]

    def _get_x_from_clips(self, all_clips):
        features = [
            torch.tensor(np.array([self.feat_func(clip) for clip in video_clips]))
            for video_clips in all_clips
        ]
        return nn.utils.rnn.pad_sequence(features, batch_first=True).to(self.device)

    def _get_x_from_videos(self, videos):
        features = [
            torch.tensor(
                np.array(
                    [
                        self.feat_func(clip)
                        for clip in get_video_feature_clip(
                            video, self.slice_method, feature_keys=self.feature_keys
                        )
                    ]
                )
            )
            for video in videos
        ]
        return nn.utils.rnn.pad_sequence(features, batch_first=True).to(self.device)

    def _get_lens_from_clips(self, all_clips):
        return [len(video_clips) for video_clips in all_clips]

    def _unmask(self, x, lens):
        return torch.concat([cur[:line] for cur, line in zip(x, lens)], dim=0)

    def _artifact_stem(self) -> str:
        return (
            f"{self.artifact_prefix}_feat-{self.feature_keys}_seed-{self.seed}"
            f"_ep-{self.epochs}_{self.run_id}"
        )

    def _save_feature_vectors(self, train_x, train_y, train_lens):
        self._save_split_feature_vectors(
            split="train",
            features=[cur.detach().cpu().numpy() for cur in train_x],
            lens=train_lens,
            labels=train_y,
        )

    def _get_feature_path(self, split: str, for_save: bool) -> Path | None:
        if for_save:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            return self.artifact_dir / f"{self._artifact_stem()}_features_{split}.npz"

        if not self.load_feature_vectors_path:
            return None

        load_path = Path(self.load_feature_vectors_path)
        if load_path.is_file():
            file_name = load_path.name
            for cur_split in ["train", "val", "test"]:
                suffix = f"_features_{cur_split}.npz"
                if file_name.endswith(suffix):
                    return load_path.with_name(
                        file_name.replace(suffix, f"_features_{split}.npz")
                    )
            if split == "train":
                return load_path
            return None

        return None

    def _save_split_feature_vectors(
        self,
        split: str,
        features: list[np.ndarray],
        lens: list[int],
        labels: list[list[int]] | None = None,
        clip_ends: list[list[float]] | None = None,
    ):
        feature_path = self._get_feature_path(split=split, for_save=True)
        if feature_path is None:
            return
        payload = {
            "x": np.asarray(features, dtype=object),
            "lens": np.asarray(lens, dtype=np.int64),
        }
        if labels is not None:
            payload["y"] = np.asarray(
                [np.asarray(cur, dtype=np.int64) for cur in labels], dtype=object
            )
        if clip_ends is not None:
            payload["clip_ends"] = np.asarray(
                [np.asarray(cur, dtype=np.float32) for cur in clip_ends], dtype=object
            )
        np.savez_compressed(feature_path, **payload)
        print(f"Saved {split} feature vectors to {feature_path}")

    def _load_split_feature_vectors(self, split: str):
        feature_path = self._get_feature_path(split=split, for_save=False)
        if feature_path is None or not feature_path.exists():
            return None
        with np.load(feature_path, allow_pickle=True) as data:
            features = [np.asarray(cur, dtype=np.float32) for cur in data["x"]]
            lens = [int(cur) for cur in data["lens"]]
            labels = None
            clip_ends = None
            if "y" in data.files:
                labels = [np.asarray(cur, dtype=np.int64).tolist() for cur in data["y"]]
            if "clip_ends" in data.files:
                clip_ends = [
                    np.asarray(cur, dtype=np.float32).tolist() for cur in data["clip_ends"]
                ]
        print(f"Loaded {split} feature vectors from {feature_path}")
        return features, lens, labels, clip_ends

    def _to_padded_tensor(self, features: list[np.ndarray]) -> torch.Tensor:
        tensors = [torch.tensor(cur, dtype=torch.float32) for cur in features]
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True).to(self.device)

    def _extract_split_features(self, data):
        clips = self._get_all_clips(data.videos)
        features = [
            np.asarray([self.feat_func(clip) for clip in video_clips], dtype=np.float32)
            for video_clips in clips
        ]
        lens = [len(video_clips) for video_clips in clips]
        clip_ends = [[clip.end for clip in video_clips] for video_clips in clips]
        labels = self._get_labels_from_clips(clips, data.frags)
        return features, lens, labels, clip_ends

    def _save_model_weights(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        weights_path = self.artifact_dir / f"{self._artifact_stem()}_weights.pt"
        payload = {
            "state_dict": self._model.state_dict(),
            "meta": {
                "feature_keys": self.feature_keys,
                "input_dim": self.input_dim,
                "hidden_size": self.hidden_size,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "seed": self.seed,
            },
        }
        torch.save(payload, weights_path)
        print(f"Saved trained model weights to {weights_path}")

    def _load_model_weights(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self._model.load_state_dict(state_dict)
        self._model.eval()
        print(f"Loaded model weights from {model_path}")

    def _fit(self, X, Y, lens, loss_fn):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            self._model.train()
            _start = time.time()
            loss_tot = 0
            loss_cnt = 0
            correct = 0
            y_pos = 0
            tot = 0
            cnt_predicted = 0
            for batch in batched_data(zip(X, Y, lens), self.batch_size):
                x, y, cur_lens = zip(*batch)
                x = torch.stack(x)  # (B, T, D)
                y = torch.cat([torch.tensor(i) for i in y]).to(self.device)  # (N_CLIP)
                y_prob = self._model(x)  # (B, T, 2)
                y_prob = self._unmask(y_prob, cur_lens)  # (N_CLIP, 2)
                y_pred = y_prob.argmax(dim=-1)  # (N_CLIP)
                assert y_pred.shape == y.shape, (y_pred.shape, y.shape)
                loss = loss_fn(y_prob, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_tot += loss.item()
                loss_cnt += 1
                correct += (y_pred.round() == y).sum().item()
                cnt_predicted += y_pred.sum().item()
                tot += y.shape[0]
                y_pos += int(y.sum().item())
            loss_tot /= loss_cnt
            print(
                f"Epoch {epoch}, Loss: {loss_tot:.5f} Time: {time.time() - _start:.5f} Accuracy: {correct / tot:.5f}, tot: {tot}, total_y_pos: {y_pos}, y_neg_rate: {1 - (y_pos / tot):.5f}, ones predicted {cnt_predicted}"
            )

    def fit(
        self,
        train_data,
        val_data=None,
        test_data=None,
    ):
        if self.load_model_weights_path:
            self._load_model_weights(self.load_model_weights_path)
            self._requires_training = False
            print("Skipping BiLSTM training because pre-trained weights were loaded.")
            return

        loss_weight = torch.tensor(self.weight or [0.01, 0.99]).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight).to(self.device)

        loaded_train = self._load_split_feature_vectors("train")
        if loaded_train is not None:
            train_features, train_lens, train_y, _ = loaded_train
            if train_y is None:
                raise ValueError("Loaded train feature vectors do not contain labels.")
            train_x = self._to_padded_tensor(train_features)
        else:
            train_features, train_lens, train_y, train_clip_ends = (
                self._extract_split_features(train_data)
            )
            train_x = self._to_padded_tensor(train_features)
            if self.save_feature_vectors:
                self._save_split_feature_vectors(
                    split="train",
                    features=train_features,
                    lens=train_lens,
                    labels=train_y,
                    clip_ends=train_clip_ends,
                )

        self._fit(train_x, train_y, train_lens, loss_fn)

        if self.save_feature_vectors and val_data is not None:
            val_features, val_lens, val_y, val_clip_ends = self._extract_split_features(
                val_data
            )
            self._save_split_feature_vectors(
                split="val",
                features=val_features,
                lens=val_lens,
                labels=val_y,
                clip_ends=val_clip_ends,
            )
        if self.save_feature_vectors and test_data is not None:
            test_features, test_lens, test_y, test_clip_ends = self._extract_split_features(
                test_data
            )
            self._save_split_feature_vectors(
                split="test",
                features=test_features,
                lens=test_lens,
                labels=test_y,
                clip_ends=test_clip_ends,
            )

        if self.save_model_weights:
            self._save_model_weights()
        self._requires_training = False

    def forward(self, videos):
        features = self._get_x_from_videos(videos)
        with torch.no_grad():
            self._model.eval()
            return self._model(features)

    @property
    def _if_print_progress(self):
        return True

    def _predict_one(self, video: Video):
        # Use `predict` method with batch processing instead
        raise NotImplementedError

    def predict(self, videos):
        loaded_test = self._load_split_feature_vectors("test")
        if loaded_test is not None:
            features, _, _, clip_ends = loaded_test
            if clip_ends is None:
                raise ValueError(
                    "Loaded test feature vectors do not contain clip_ends for decoding."
                )
            if len(features) != len(videos):
                raise ValueError(
                    f"Loaded test feature vectors count mismatch: {len(features)} vs {len(videos)}"
                )
            ret = []
            for batch in batched_data(list(zip(features, clip_ends)), 32):
                batch_features, batch_clip_ends = zip(*batch)
                x = self._to_padded_tensor(list(batch_features))
                with torch.no_grad():
                    self._model.eval()
                    raw_results = self._model(x)
                preds = raw_results.argmax(dim=-1).cpu().numpy().round()
                cur = [
                    sorted([end for end, label in zip(v_ends, v_pred) if label > 0.5])
                    for v_ends, v_pred in zip(batch_clip_ends, preds)
                ]
                ret.extend(cur)
            return [self._post_process(cur, video) for cur, video in zip(ret, videos)]

        cached_features = []
        cached_lens = []
        cached_clip_ends = []
        ret = []
        for batch in batched_data(videos, 32):
            clips = [
                list(
                    get_video_feature_clip(
                        video, self.slice_method, feature_keys=self.feature_keys
                    )
                )
                for video in batch
            ]
            batch_features = [
                np.asarray([self.feat_func(clip) for clip in video_clips], dtype=np.float32)
                for video_clips in clips
            ]
            x = self._to_padded_tensor(batch_features)
            with torch.no_grad():
                self._model.eval()
                raw_results = self._model(x)
            preds = raw_results.argmax(dim=-1).cpu().numpy().round()
            cur = [
                sorted(
                    [clip.end for clip, label in zip(v_clips, v_pred) if label > 0.5]
                )
                for v_clips, v_pred in zip(clips, preds)
            ]
            ret.extend(cur)

            if self.save_feature_vectors:
                cached_features.extend(batch_features)
                cached_lens.extend([len(video_clips) for video_clips in clips])
                cached_clip_ends.extend(
                    [[clip.end for clip in video_clips] for video_clips in clips]
                )

        if self.save_feature_vectors and cached_features:
            self._save_split_feature_vectors(
                split="test",
                features=cached_features,
                lens=cached_lens,
                clip_ends=cached_clip_ends,
            )

        return [self._post_process(cur, video) for cur, video in zip(ret, videos)]


class _Model(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(_Model, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, batch_first=True, bidirectional=True
        )
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu1(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


register_model_runner(
    command_name="bilstm",
    model_cls=BiLSTM,
    model_param_build=lambda kwargs: dict(
        visual_dim=kwargs["visual_dim"],
        audio_dim=kwargs["audio_dim"],
        text_dim=kwargs["text_dim"],
        hidden_size=kwargs["hidden_size"],
        epochs=kwargs["epochs"],
        batch_size=kwargs["batch_size"],
        lr=kwargs["lr"],
        feat_func=kwargs["feat_func"],
        weight=[kwargs["neg_weight"], 1 - kwargs["neg_weight"]],
        save_feature_vectors=kwargs["save_feature_vectors"],
        load_feature_vectors_path=kwargs["load_feature_vectors_path"],
        save_model_weights=kwargs["save_model_weights"],
        load_model_weights_path=kwargs["load_model_weights_path"],
        artifact_dir=kwargs["artifact_dir"],
        artifact_prefix=kwargs["artifact_prefix"],
    ),
    additional_options=[
        click.option("--visual_dim", type=int, default=1000),
        click.option("--audio_dim", type=int, default=512),
        click.option("--text_dim", type=int, default=512),
        click.option("--hidden_size", type=int, default=256),
        click.option("--batch_size", type=int, default=32),
        click.option("--lr", type=float, default=1e-3),
        click.option("--epochs", type=int, default=100),
        click.option("--neg_weight", default=0.01),
        click.option(
            "--feat_func", type=click.Choice(list(FEATURE_FUNC.keys())), default="tva"
        ),
        click.option(
            "--save_feature_vectors/--no-save_feature_vectors",
            default=False,
            help="Save train/val/test feature vectors used by this run.",
        ),
        click.option(
            "--load_feature_vectors_path",
            type=str,
            default="",
            help="Load feature vectors from a train split .npz path and infer val/test paths.",
        ),
        click.option(
            "--save_model_weights/--no-save_model_weights",
            default=False,
            help="Save trained BiLSTM weights after fit().",
        ),
        click.option(
            "--load_model_weights_path",
            type=str,
            default="",
            help="Load BiLSTM weights from file and skip training.",
        ),
        click.option(
            "--artifact_dir",
            type=str,
            default="./artifacts/lstm",
            help="Directory for saved features and weights.",
        ),
        click.option(
            "--artifact_prefix",
            type=str,
            default="bilstm",
            help="File prefix used in artifact naming.",
        ),
    ],
)
