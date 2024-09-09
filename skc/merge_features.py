import torch
from collections import defaultdict
import pickle
import math

device = torch.device("cuda:0")

# 读取fft特征
fft_jl = torch.load("features_data/features_fft_jl.pt").to(device)
fft_yy = torch.load("features_data/features_fft_yy.pt").to(device)
fft_zc = torch.load("features_data/features_fft_zc.pt").to(device)

length_jl = fft_jl.shape[0]
length_yy = fft_yy.shape[0]
length_zc = fft_zc.shape[0]

train_length_jl = math.ceil(length_jl * 0.8)
train_length_yy = math.ceil(length_yy * 0.8)
train_length_zc = math.ceil(length_zc * 0.8)

# 划分数据集
fft_train = torch.vstack((fft_jl[:train_length_jl], fft_yy[:train_length_jl], fft_zc[:train_length_jl])) 
fft_test = torch.vstack((fft_jl[train_length_jl:length_jl], fft_yy[train_length_yy:length_yy], fft_zc[train_length_zc:length_zc])) 

print(fft_train.shape) # 2024
print(fft_test.shape) # 504

# 读取mfcc特征
mfcc_jl = torch.load("features_data/features_mfcc_jl.pt").to(device)
mfcc_yy = torch.load("features_data/features_mfcc_yy.pt").to(device)
mfcc_zc = torch.load("features_data/features_mfcc_zc.pt").to(device)

# 划分数据集
mfcc_train = torch.vstack((mfcc_jl[:train_length_jl], mfcc_yy[:train_length_jl], mfcc_zc[:train_length_jl]))
mfcc_test = torch.vstack((mfcc_jl[train_length_jl:length_jl], mfcc_yy[train_length_yy:length_yy], mfcc_zc[train_length_zc:length_zc]))

print(mfcc_train.shape)
print(mfcc_test.shape)

# 读取wav
wav_jl = torch.load("features_data/features_wav_jl.pt").to(device)
wav_yy = torch.load("features_data/features_wav_yy.pt").to(device)
wav_zc = torch.load("features_data/features_wav_zc.pt").to(device)

# 划分数据集
wav_train = torch.vstack((wav_jl[:train_length_jl], wav_yy[:train_length_jl], wav_zc[:train_length_jl]))
wav_test = torch.vstack((wav_jl[train_length_jl:length_jl], wav_yy[train_length_yy:length_yy], wav_zc[train_length_zc:length_zc]))

print(wav_train.shape)
print(wav_test.shape)

# 建立标签

label_jl_train = torch.zeros(train_length_jl).unsqueeze(1)
label_yy_train = torch.ones(train_length_jl).unsqueeze(1)
label_zc_train = torch.full((train_length_jl,), 2).unsqueeze(1)
label_train =torch.vstack((label_jl_train, label_yy_train, label_zc_train))

print(label_train.shape)

label_jl_test = torch.zeros(length_jl - train_length_jl).unsqueeze(1)
label_yy_test = torch.ones(length_yy -train_length_yy).unsqueeze(1)
label_zc_test = torch.full((length_zc - train_length_zc,), 2).unsqueeze(1)
label_test = torch.vstack((label_jl_test, label_yy_test, label_zc_test))

print(label_test.shape)

classes = {0: 'jl', 1: 'yy', 2: 'zz'}
features = defaultdict()
features["fft_train"] = fft_train
features["fft_test"] = fft_test
features["mfcc_train"] = mfcc_train
features["mfcc_test"] = mfcc_test
features["wav_train"] = wav_train
features["wav_test"] = wav_test
features["label_train"] = label_train
features["label_test"] = label_test

# print(features)

with open('features_data/features_merge_1.pkl', 'wb') as pickle_file:
    pickle.dump(features, pickle_file)