# Copyright (c) 2021 Yusuke Tashiro
# Licensed under the MIT License
# All rights reserved.
# Modified by XXX on 2025-02-10
# --------------------------------------------------------
# References:
# CSDI: https://github.com/ermongroup/CSDI
# --------------------------------------------------------


import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Diff_Model import Denoising_network




class  Diff_base(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.history_len= cfg.history_len
        self.predict_len= cfg.predict_len
        self.model_name=cfg.eps_model
        self.device = cfg.device
        self.target_dim = cfg.data.num_vertices
        self.emb_time_dim = 128
        self.emb_feature_dim = 16
        self.is_unconditional = 0
        self.target_strategy = "test"
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        self.diffmodel = Denoising_network().get_model(cfg)
        self.num_steps = cfg.model_diff.diff_steps
        self.beta = np.linspace(
                0.0001 ** 0.5, 0.5 ** 0.5, self.num_steps
            ) ** 2
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe



    def calc_loss_valid(
        self, prior,observed_data, cond_mask,  side_info, is_train,
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
               prior,observed_data, cond_mask, side_info, is_train,set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps


    def calc_loss(
        self,latent_z,prior,observed_data, cond_mask, side_info, is_train,set_t=-1
    ):
        B, _, _ = observed_data.shape 
        
        observed_data[:,:,self.history_len:]=observed_data[:,:,self.history_len:] -prior.permute(0,2,1)

        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  
        noise = torch.randn_like(observed_data)

        latent_z=latent_z.squeeze(-1)

        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise+ (1-current_alpha ** 0.5)*torch.cat([torch.zeros_like(observed_data[...,:self.history_len]).to(latent_z.device),latent_z],dim=-1)
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask) 
        total_input[:,0,:,self.history_len:]=latent_z



        predicted = self.diffmodel(total_input, side_info, t,current_alpha) 

        target_mask = torch.ones_like(cond_mask) - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1) 
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1) 
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1) 

        return total_input

    def impute(self,scale_residual,prior, observed_data, cond_mask, side_info, n_samples):
        if self.model_name=="STID":
            B, K, L = observed_data.shape
            observed_data[:,:,self.history_len:]=observed_data[:,:,self.history_len:] -prior.permute(0,2,1)
            imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        else:
            raise ValueError(f"Model {self.model_name} not found")

        for i in range(n_samples):
            random_sign = (torch.randint(0, 2, (B, K, self.predict_len)) * 2 - 1).to(self.device)
            latent_z= scale_residual.unsqueeze(-1).expand(B,K,self.predict_len)*random_sign
            latent_z=latent_z.squeeze(-1)
            current_sample = torch.randn_like(observed_data)
            current_sample[:,:,self.history_len:]=current_sample[:,:,self.history_len:] +latent_z
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)


            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                diff_input[:,0,:,self.history_len:]=latent_z

                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device),self.alpha_torch[t])

                coeff1_yt = 1 / self.alpha_hat[t] ** 0.5
                coeff2_latent = 1-coeff1_yt
                coeff3_predict =(1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1_yt * (current_sample - coeff3_predict * predicted)+coeff2_latent*torch.cat([torch.zeros_like(observed_data[...,:self.history_len]).to(latent_z.device),latent_z],dim=-1)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise


            imputed_samples[:, i] = current_sample.detach()

        imputed_samples[:,:,:,self.history_len:]=imputed_samples[:,:,:,self.history_len:] +prior.permute(0,2,1).unsqueeze(1)
        return imputed_samples



class Diff_Forecasting( Diff_base):
    def __init__(self,cfg):
        super(Diff_Forecasting, self).__init__(cfg)
        self.target_dim_base = cfg.data.num_vertices
        self.data_name=cfg.data.name



    def process_data(self, batch):

        observed_data = batch["observed_data"].to(self.device).float() # B,L,H,W
        observed_tp = batch["timepoints"].to(self.device).float() # B,L,2
        gt_mask = batch["gt_mask"].to(self.device).float() # B,L,H,W
        scale_residual= batch["scale_residual"].to(self.device).float() # B,K,

        B,L,H,W=gt_mask.shape
        if self.model_name=="STID":
            observed_data = rearrange(observed_data,'b l h w -> b (h w) l')
            gt_mask = rearrange(gt_mask,'b l h w -> b (h w) l')
        else:
            raise ValueError(f"Model {self.model_name} not found")
        return (
            observed_data,
            observed_tp,
            gt_mask,
            scale_residual
        )        



    def get_side_info(self, observed_tp, cond_mask):
        if self.model_name=="STID":
            B, K, L = cond_mask.shape
            cond_mask=cond_mask.permute(0,2,1).unsqueeze(-1)  
            time_embed_hour = observed_tp[:,:,1].unsqueeze(2).unsqueeze(3).expand(-1, -1, self.target_dim, -1)/24
            time_embed_day = observed_tp[:,:,0].unsqueeze(2).unsqueeze(3).expand(-1, -1, self.target_dim, -1)/7   
            side_info = torch.cat([time_embed_hour,time_embed_day], dim=-1)  
            if self.is_unconditional == False:
                side_mask = cond_mask 
                side_info = torch.cat([side_info, side_mask], dim=-1)  
        else:
            raise ValueError(f"Model {self.model_name} not found")
        return side_info


    def forward(self,prior ,batch, is_train=1):
        (
            observed_data,
            observed_tp,
            gt_mask,
            scale_residual
        ) = self.process_data(batch)

        B,K,L=observed_data.shape
        random_sign = (torch.randint(0, 2, (B, K, self.predict_len)) * 2 - 1).to(self.device)
        latent_z=scale_residual.unsqueeze(-1).expand(B,K,self.predict_len)*random_sign
        cond_mask = gt_mask
        side_info = self.get_side_info(observed_tp, cond_mask)  
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(latent_z,prior,observed_data, cond_mask, side_info, is_train)


    def evaluate(self,prior ,batch, n_samples):
        (
            observed_data,
            observed_tp,
            gt_mask,
            scale_residual
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(scale_residual,prior,observed_data, cond_mask, side_info, n_samples)

        return samples
    