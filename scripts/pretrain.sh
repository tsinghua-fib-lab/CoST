#!/bin/bash

nohup python main_pretrain.py --batch_size 8 --data_name MobileNJ --device "cuda:3" --history_len 12 --predict_len 12 > log/STID/12_12_MobileNJ.log 2>&1 &
nohup python main_pretrain.py --batch_size 8 --data_name MobileSH --device "cuda:1" --history_len 12 --predict_len 12 > log/STID/12_12_MobileSH.log 2>&1 &
nohup python main_pretrain.py --batch_size 8 --data_name CrowdBJ --device "cuda:2" --history_len 12 --predict_len 12 > log/STID/12_12_CrowdBJ.log 2>&1 &
nohup python main_pretrain.py --batch_size 8 --data_name CrowdBM --device "cuda:0" --history_len 12 --predict_len 12 > log/STID/12_12_CrowdBM.log 2>&1 &
nohup python main_pretrain.py --batch_size 24 --data_name BikeDC --device "cuda:4" --history_len 12 --predict_len 12 > log/STID/12_12_BikeDC.log 2>&1 &
nohup python main_pretrain.py --batch_size 24 --data_name TaxiBJ --device "cuda:5" --history_len 12 --predict_len 12 > log/STID/12_12_TaxiBJ.log 2>&1 &
nohup python main_pretrain.py --batch_size 24 --data_name Los_Speed --device "cuda:6" --history_len 12 --predict_len 12 > log/STID/12_12_Los_Speed.log 2>&1 &
nohup python main_pretrain.py --batch_size 32 --data_name SST --device "cuda:7" --history_len 12 --predict_len 12 > log/STID/12_12_SST.log 2>&1 &
