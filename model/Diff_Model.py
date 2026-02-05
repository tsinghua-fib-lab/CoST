# Copyright (c) 2022 Zezhi Shao
# Licensed under the Apache License 2.0
# All rights reserved.
# Modifications made by XXX on 2025-02-10:
# Changes: 
# - Changed the model architecture to a diffusion denoising network for CoST.
# --------------------------------------------------------
# References:
# STID: https://github.com/GestaltCogTeam/STID
# --------------------------------------------------------


import torch
import torch.nn as nn
from torch.nn import functional as F



class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table
    




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





class STID(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self,cfg):
        super().__init__()
        # attributes
        self.num_nodes = cfg.data.num_vertices
        self.node_dim = cfg.model_diff.node_dim 
        self.input_len = cfg.history_len + cfg.predict_len
        self.input_dim = cfg.model_diff.input_dim
        self.embed_dim = cfg.model_diff.embed_dim
        self.output_len = cfg.history_len + cfg.predict_len
        self.num_layer = cfg.model_diff.num_layer
        self.temp_dim_tid = cfg.model_diff.temp_dim_tid
        self.temp_dim_diw = cfg.model_diff.temp_dim_diw
        self.time_of_day_size = cfg.model_diff.time_of_day_size
        self.day_of_week_size = cfg.model_diff.day_of_week_size
        self.if_time_in_day = cfg.model_diff.if_time_in_day
        self.if_day_in_week = cfg.model_diff.if_day_of_week
        self.if_spatial = cfg.model_diff.if_spatial

        # spatial embeddings
        if self.if_spatial:
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
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_time_in_day) + \
            self.temp_dim_diw*int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        
        
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=50,
            embedding_dim=128,
        )

        self.diffusion_projection = nn.Linear(128, self.embed_dim)
        self.his_len=cfg.history_len




    def forward(self,x, cond_info, diffusion_step,alpha_torch):
        """Feed forward of STID.
        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        history_data=x.permute(0,3,2,1) 
        input_data=x.permute(0,3,2,1)

        if self.if_time_in_day:
            t_i_d_data = cond_info[..., 0]
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] ).type(torch.LongTensor)]#* self.time_of_day_size
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = cond_info[..., 1]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :] ).type(torch.LongTensor)]#* self.day_of_week_size
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        
        time_series_emb = self.time_series_emb_layer(input_data) # (B,channel,H*W,1)

        diffusion_emb=self.diffusion_embedding(diffusion_step)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)  # (B,channel,1)

        time_series_emb = time_series_emb+diffusion_emb

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)) # B,32,H*W,1
            
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1)) # 
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1)) # 

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        hidden = self.encoder(hidden)
        prediction = self.regression_layer(hidden) 
        b, l, n, c = prediction.shape

        prediction=prediction.permute(0, 3, 2, 1).squeeze(1)  

        return prediction



class Denoising_network():
    def __init__(self) -> None:
        self.model_dict = {
            "STID": STID,
        }
    def get_model(self,cfg):
        if cfg.eps_model not in self.model_dict:
            raise ValueError(f"Model {cfg.eps_model} not found")
        if cfg.eps_model == "STID":
            return self.model_dict[cfg.eps_model](cfg)