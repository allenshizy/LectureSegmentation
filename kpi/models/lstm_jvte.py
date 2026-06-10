import time
import click
from typing import Literal

import numpy as np
import torch
from torch import nn

from kpi.models._jvte import JVTE
from kpi.models.base_model import Model
from kpi.utils import batched_data
from kpi.utils.preprocessing import get_video_feature_clip
from kpi.utils.video import Video
from kpi.cli import register_model_runner


class BiLSTMJVTE(Model):
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        dataset: str,
        hidden_size: int = 256,
        slice_method: Literal["srt"] = "srt",
        device: str = "",
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        jvte_epochs: int = 100,
        jvte_batch_size: int = 32,
        jvte_lr: float = 1e-3,
        jvte_hidden_size: int = 1024,
        jvte_sigma: float = 0.1,
        weight: list[float] | None = None,
    ):
        self.slice_method: Literal["srt"] = slice_method
        self.input_dim = jvte_hidden_size * 2
        self.hidden_size = hidden_size
        self.jvte_epochs = jvte_epochs
        self.jvte_batch_size = jvte_batch_size
        self.jvte_lr = jvte_lr

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight = weight
        self._requires_training = True

        self._model = self._build_model()
        self._jvte = JVTE(
            dataset=dataset,
            v_dim=visual_dim,
            t_dim=text_dim,
            feat_dim=jvte_hidden_size,
            batch_size=jvte_batch_size,
            lr=jvte_lr,
            sigma=jvte_sigma,
            epochs=jvte_epochs,
            device=self.device,
        ).to(self.device)

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
            list(get_video_feature_clip(video, self.slice_method)) for video in videos
        ]

    def _get_x_from_clips(self, all_clips):
        features = [
            torch.tensor(
                np.array(
                    [self._jvte(clip).detach().cpu().numpy() for clip in video_clips]
                )
            )
            for video_clips in all_clips
        ]
        return nn.utils.rnn.pad_sequence(features, batch_first=True).to(self.device)

    def _get_x_from_videos(self, videos):
        features = [
            torch.tensor(
                np.array(
                    [
                        self._jvte(clip).detach().cpu().numpy()
                        for clip in get_video_feature_clip(video, self.slice_method)
                    ]
                )
            )
            for video in videos
        ]
        return nn.utils.rnn.pad_sequence(features, batch_first=True).to(self.device)

    def _get_lens_from_clips(self, all_clips):
        return [len(video_clips) for video_clips in all_clips]

    def _unmask(self, x, lens):
        return torch.concat([cur[:l] for cur, l in zip(x, lens)], dim=0)

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
    ):
        self._jvte.train()
        self._jvte.fit(train_data, val_data)
        self._jvte.eval()

        videos = train_data.videos
        frags = train_data.frags
        loss_weight = torch.tensor(self.weight or [0.01, 0.99]).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight).to(self.device)

        clips = self._get_all_clips(videos)  # (N_V, T)
        train_lens = self._get_lens_from_clips(clips)  # (N_V, )
        train_x = self._get_x_from_clips(clips)  # (N_V, T, D)
        train_y = self._get_labels_from_clips(clips, frags)  # (N_V, acutal_length)
        self._fit(train_x, train_y, train_lens, loss_fn)
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
        raise NotImplementedError

    def predict(self, videos):
        ret = []
        for batch in batched_data(videos, 32):
            clips = [
                list(get_video_feature_clip(video, self.slice_method))
                for video in batch
            ]
            raw_results = self.forward(batch)  # (batch_size, max_seq_len, 1)
            preds = raw_results.argmax(dim=-1).cpu().numpy().round()
            cur = [
                sorted(
                    [clip.end for clip, label in zip(v_clips, v_pred) if label > 0.5]
                )
                for v_clips, v_pred in zip(clips, preds)
            ]
            ret.extend(cur)
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu1(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


register_model_runner(
    command_name="bilstm_jvte",
    model_cls=BiLSTMJVTE,
    model_param_build=lambda kwargs: dict(
        hidden_size=kwargs["hidden_size"],
        epochs=kwargs["epochs"],
        batch_size=kwargs["batch_size"],
        lr=kwargs["lr"],
        weight=[kwargs["neg_weight"], 1 - kwargs["neg_weight"]],
        visual_dim=kwargs["visual_dim"],
        text_dim=kwargs["text_dim"],
        jvte_batch_size=kwargs["jvte_batch"],
        jvte_lr=kwargs["jvte_lr"],
        jvte_epochs=kwargs["jvte_epochs"],
        jvte_hidden_size=kwargs["jvte_hidden"],
        dataset=kwargs["dataset"],
    ),
    additional_options=[
        click.option("--hidden_size", type=int, default=256),
        click.option("--batch_size", type=int, default=32),
        click.option("--lr", type=float, default=1e-3),
        click.option("--epochs", type=int, default=100),
        click.option("--neg_weight", default=0.99),
        click.option("--visual_dim", type=int, default=1000),
        click.option("--text_dim", type=int, default=512),
        click.option("--jvte_batch", type=int, default=32),
        click.option("--jvte_lr", type=float, default=1e-3),
        click.option("--jvte_epochs", type=int, default=100),
        click.option("--jvte_hidden", type=int, default=1024),
        click.option("--jvte_sigma", type=float, default=0.1),
    ],
)
