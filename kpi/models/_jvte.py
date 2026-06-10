import torch
import time
import numpy as np
from torch import nn
from kpi.utils import batched_data
from kpi.utils.preprocessing import get_video_feature_clip


class JVTE(nn.Module):
    def __init__(
        self,
        v_dim,
        t_dim,
        feat_dim,
        sigma,
        batch_size,
        lr,
        epochs,
        device,
        dataset,
        hard_neg=1,
        soft_neg=1,
    ):
        super().__init__()
        self.v_l1 = nn.Linear(v_dim, feat_dim)
        self.v_l2 = nn.Linear(feat_dim, feat_dim)
        self.v_sigmoid = nn.Sigmoid()
        self.t_l1 = nn.Linear(t_dim, feat_dim)
        self.t_l2 = nn.Linear(feat_dim, feat_dim)
        self.t_sigmoid = nn.Sigmoid()
        self.sigma = sigma
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.hard_neg = hard_neg
        self.soft_neg = soft_neg
        self.dataset = dataset

    def forward(self, clip):
        v = torch.tensor(clip.visual_vec, device=self.device)
        t = torch.tensor(clip.text_vec, device=self.device)
        v = self.v_l1(v) * self.v_sigmoid(self.v_l2(self.v_l1(v)))
        t = self.t_l1(t) * self.t_sigmoid(self.t_l2(self.t_l1(t)))
        return torch.cat([v, t], dim=-1)

    def convert_videos(self, videos):
        v_i = []
        t_i = []
        v_j = []
        t_j = []
        all_vs = []
        all_ts = []
        for video in videos:
            clips = list(get_video_feature_clip(video, "srt"))
            for c in clips:
                all_vs.append(c.visual_vec)
                all_ts.append(c.text_vec)
        for video in videos:
            clips = list(get_video_feature_clip(video, "srt"))
            clip_vs = [c.visual_vec for c in clips]
            clip_ts = [c.text_vec for c in clips]
            for c in clips:
                for _ in range(self.hard_neg):
                    v_i.append(c.visual_vec)
                    t_i.append(c.text_vec)
                    v_j.append(clip_vs[np.random.randint(len(clip_vs))])
                    t_j.append(clip_ts[np.random.randint(len(clip_ts))])
                for _ in range(self.soft_neg):
                    v_i.append(c.visual_vec)
                    t_i.append(c.text_vec)
                    v_j.append(all_vs[np.random.randint(len(all_vs))])
                    t_j.append(all_ts[np.random.randint(len(all_ts))])

        v_i = np.array(v_i)
        t_i = np.array(t_i)
        v_j = np.array(v_j)
        t_j = np.array(t_j)
        return v_i, t_i, v_j, t_j

    def fit(self, train_data, val_data=None):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            _start = time.time()
            loss_tot = 0
            loss_cnt = 0
            for batch in batched_data(train_data.videos, self.batch_size):
                optimizer.zero_grad()
                v_i, t_i, v_j, t_j = self.convert_videos(batch)
                v_i = torch.tensor(v_i, device=self.device)
                t_i = torch.tensor(t_i, device=self.device)
                v_j = torch.tensor(v_j, device=self.device)
                t_j = torch.tensor(t_j, device=self.device)
                v_i = self.v_l1(v_i) * self.v_sigmoid(self.v_l2(self.v_l1(v_i)))
                t_i = self.t_l1(t_i) * self.t_sigmoid(self.t_l2(self.t_l1(t_i)))
                v_j = self.v_l1(v_j) * self.v_sigmoid(self.v_l2(self.v_l1(v_j)))
                t_j = self.t_l1(t_j) * self.t_sigmoid(self.t_l2(self.t_l1(t_j)))
                # for each pair
                z0 = torch.zeros(1, device=self.device)
                sii = torch.cosine_similarity(v_i, t_i)
                sij = torch.cosine_similarity(v_i, t_j)
                sji = torch.cosine_similarity(v_j, t_i)
                loss = (
                    torch.max(z0, self.sigma + sij - sii)
                    + torch.max(z0, self.sigma + sji - sii)
                ).sum()

                loss.backward()
                optimizer.step()
                loss_tot += loss.item()
                loss_cnt += v_i.shape[0]
            print(
                f"Epoch {epoch + 1}/{self.epochs} loss: {loss_tot / loss_cnt:.4f} time: {time.time() - _start:.4f}"
            )
