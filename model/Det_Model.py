# Copyright (c) 2022 Zezhi Shao
# Licensed under the Apache License 2.0
# All rights reserved.
# Modifications made by XXX on 2025-02-10:
# --------------------------------------------------------
# References:
# STID: https://github.com/GestaltCogTeam/STID
# --------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden



import torch
from torch import nn

class STID(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, cfg):
        super().__init__()
        # attributes
        self.num_nodes = cfg.data.num_vertices
        self.node_dim = cfg.model_det.node_dim
        self.input_len = cfg.history_len
        self.input_dim = cfg.model_det.input_dim
        self.embed_dim = cfg.model_det.embed_dim
        self.output_len = cfg.predict_len
        self.num_layer = cfg.model_det.num_layer
        self.temp_dim_tid = cfg.model_det.temp_dim_tid
        self.temp_dim_diw = cfg.model_det.temp_dim_diw
        self.time_of_day_size = cfg.model_det.time_of_day_size
        self.day_of_week_size = cfg.model_det.day_of_week_size
        self.if_time_in_day = cfg.model_det.if_time_in_day
        self.if_day_in_week = cfg.model_det.if_day_of_week
        self.if_spatial_prior = cfg.model_det.if_spatial

        # spatial embeddings
        if self.if_spatial_prior:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial_prior)+self.temp_dim_tid*int(self.if_time_in_day) + \
            self.temp_dim_diw*int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor,observed_tp) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]
            observed_tp (torch.Tensor): observed time points with shape [B, L, 2]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # prepare data
        input_data = history_data[..., range(self.input_dim)]
        B, L, K,_ = input_data.shape
        # B,L,2 -> B,L,K,2
        observed_tp = observed_tp.unsqueeze(-2).expand(B, L, K, 2)


        if self.if_time_in_day:
            t_i_d_data = observed_tp[..., 1] #b,l,k,2
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] ).type(torch.LongTensor)]#* self.time_of_day_size
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = observed_tp[..., 0]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :] ).type(torch.LongTensor)] #* self.day_of_week_size
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous() # B, K, L, C
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_spatial_prior:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)

        return prediction