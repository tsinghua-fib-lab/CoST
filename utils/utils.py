# Copyright (c) 2021 Yusuke Tashiro
# Licensed under the MIT License
# All rights reserved.
# Modified by XXX on 2025-02-10
# --------------------------------------------------------
# References:
# CSDI: https://github.com/ermongroup/CSDI
# --------------------------------------------------------

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from einops import rearrange
import os
import random

def set_cpu_num(cpu_num):
    if cpu_num <= 0: return
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

def set_random_seed(seed: int):

    random.seed(seed)                        
    np.random.seed(seed)                    
    torch.manual_seed(seed)               
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        
        torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_quantile_CRPS(target, forecast, eval_points):

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points):

    eval_points = eval_points.mean(-1)
    target = target.sum(-1)

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)



def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

def smape(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()
    smape_value = torch.mean(2.0 * torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred)))    
    return smape_value.item()

def metric_our(pred,real):
    mae1 = mae(pred, real)
    rmse1 = rmse(pred, real)
    smape1= smape(pred, real)
    return mae1,smape1,rmse1


def masked_mape( labels, preds,null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



def compute_PICP( y_true, all_gen_y, ratio=95,return_CI=False):
    y_true=y_true.cpu().numpy()
    all_gen_y=all_gen_y.cpu().numpy()
    low= (100 - ratio) / 2
    high= ratio + (100 - ratio) / 2
    CI_y_pred = np.percentile(all_gen_y, q=[low, high], axis=1)
    
    # compute percentage of true y in the range of credible interval
    y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
    coverage = y_in_range.mean()
    if return_CI:
        return coverage, CI_y_pred, low, high
    else:
        return coverage
    


def compute_QICE(all_true_y, all_generated_y,n_bins=10, verbose=True):

    all_true_y=all_true_y.cpu().numpy()
    all_generated_y=all_generated_y.cpu().numpy()
    quantile_list = np.arange(n_bins + 1) * (100 / n_bins)  
    all_generated_y = all_generated_y.transpose(0, 2, 3, 1)
    all_true_y = all_true_y

    y_pred_quantiles = np.percentile(all_generated_y, q=quantile_list, axis=-1).transpose( 1, 2, 3,0)
    quantile_membership_array = ((all_true_y[..., None] - y_pred_quantiles) > 0).astype(int)
    y_true_quantile_membership = quantile_membership_array.sum(axis=-1)
    y_true_quantile_bin_count = np.array([
        (y_true_quantile_membership == v).sum()
        for v in range(n_bins + 2)
    ])
    if verbose:
        y_true_below_0, y_true_above_100 = y_true_quantile_bin_count[0], y_true_quantile_bin_count[-1]
        print(f"We have {y_true_below_0} true y smaller than min of generated y, "
              f"and {y_true_above_100} greater than max of generated y.")
    y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
    y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
    y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]
    total_samples = all_true_y.shape[0]
    y_true_ratio_by_bin = y_true_quantile_bin_count_ / (all_true_y.shape[0]*all_true_y.shape[1]*all_true_y.shape[2])
    assert np.abs(np.sum(y_true_ratio_by_bin) - 1) < 1e-10, "Sum of quantile coverage ratios shall be 1!"
    qice_coverage_ratio = np.absolute(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()

    return qice_coverage_ratio


def IS(predict,target,ratio=90):
    low=np.percentile(predict,(100-ratio)/2,axis=1)
    high=np.percentile(predict,ratio+(100-ratio)/2,axis=1)
    interal_loss=high-low
    low_loss=np.maximum(low-target,0)
    high_loss=np.maximum(target-high,0)
    loss=interal_loss+(low_loss+high_loss)*2/(1-ratio/100)

    return loss.mean()




class EarlyStopping():
    def __init__(self, patience=7, delta=0, path='model.pth', verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss





def train(
    prior_model,    
    model,
    cfg,
    train_loader,
    scaler=None,
    valid_target=None,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
):
    
    early_stopping = EarlyStopping(patience=cfg.patience, verbose=True,path=(foldername+'/model.pth'))

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if foldername != "":
        output_path = foldername + "/model.pth"
    p1 = int(0.4 * cfg.epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1], gamma=0.4
    )
    best_valid_loss = 1e10
    for epoch_no in range(cfg.epochs): #config["epochs"]
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                observed_data = train_batch["observed_data"].to(cfg.device).float()
                observed_data = rearrange(observed_data,'b l h w -> b l (h w)').unsqueeze(-1)[:,:cfg.history_len,:,:]
                observed_tp = train_batch["timepoints"].to(cfg.device).float()
                observed_tp = observed_tp[:,:cfg.history_len,:]
                prior = prior_model(observed_data,observed_tp).squeeze(-1)
                loss = model(prior,train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

        mae_val,_=evaluate(prior_model,cfg.eps_model,cfg,model, valid_loader,valid_target,scaler,nsample=3)
        early_stopping(mae_val, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break


def evaluate(prior_model,model_name,cfg,model, test_loader,target,scaler,nsample=10,test=0):
    with torch.no_grad():
        model.eval()
        mae_total = 0
        rmse_total=0
        predict=[]

        if model_name=='STID':
            b,l,k,c=target.shape
            with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    observed_data = test_batch["observed_data"].to(cfg.device).float()
                    observed_data = rearrange(observed_data,'b l h w -> b l (h w)').unsqueeze(-1)[:,:cfg.history_len,:,:]
                    observed_tp = test_batch["timepoints"].to(cfg.device).float()
                    observed_tp = observed_tp[:,:cfg.history_len,:]
                    prior = prior_model(observed_data,observed_tp).squeeze(-1)
                    output = model.evaluate(prior,test_batch, nsample) 
                    predict.append(rearrange(output[:,:,:,-cfg.predict_len:],'b n (k c) l -> b n l k c',n=nsample,k=k,c=c))

                predict=torch.cat(predict,dim=0)
                predict=scaler.inverse_transform(predict.cpu())     
            for n in range(nsample):
                pre_data=predict[:,n,:,:]
                rmse_total+=rmse(pre_data,target)
                mae_total+=mae(pre_data,target)
   
            target=target.squeeze()
            predict=predict.squeeze()
            evalpoint=torch.ones_like(target).to(target.device)
            CRPS = calc_quantile_CRPS(
                target, predict, evalpoint 
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                target, predict, evalpoint
            )      
        else:
            print('model name error')

    print('metric:')
    print(f'CRPS: {CRPS:.4f} CRPS_sum: {CRPS_sum:.4f}')
    mean=torch.mean(predict,dim=1)
    mae_=torch.mean(torch.abs(mean-target))
    rmse_=torch.sqrt(torch.mean((mean-target)**2))
    print(f'MAE RMSE: {mae_:.4f}, {rmse_:.4f}')
    if test==1:
        QICE_10=compute_QICE(target,predict,10)
        IS_90=IS(predict.cpu().numpy(),target.cpu().numpy(),90)
        print(f"CRPS: {round(CRPS,4)}")
        print(f"QICE: {round(QICE_10,4)}")
        print(f"IS: {round(IS_90,4)}")

    return mae_total/nsample,(predict,target)






