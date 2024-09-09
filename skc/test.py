import torch
# from torch.nn import functional as F
# import torchaudio
# import torchaudio.transforms as transforms
# loaded_tensor = torch.load('E:/pythonprojects/PythonProject5/mywork/features_data/features_mfcc_zc.pt')
# print(loaded_tensor)
# print(loaded_tensor.shape)

# def extract_mel_spectrogram(audio_file, n_mels=128, sr=16000):
#     # 加载音频文件
#     waveform, sample_rate = torchaudio.load(audio_file)
    
#     # 如果采样率不同，重采样
#     if sample_rate != sr:
#         resampler = transforms.Resample(orig_freq=sample_rate, new_freq=sr)
#         waveform = resampler(waveform)

#     # 创建梅尔频谱转换器
#     mel_transform = transforms.MelSpectrogram(sample_rate=sr, n_mels=n_mels)
    
#     # 计算梅尔频谱图
#     mel_spectrogram = mel_transform(waveform)

#     # 转换为对数刻度
#     log_mel_spectrogram = transforms.AmplitudeToDB()(mel_spectrogram)
    
#     return log_mel_spectrogram

# audio_path = 'E:/pythonprojects/PythonProject5/Speech-Emotion-Recognition-yyzcjl/Speech-Emotion-Recognition-yyzcjl/yyzcjl/zc/42093-ruangong-zhaosongran.wav'  # 替换为你的音频文件路径
# log_mel_spectrogram = extract_mel_spectrogram(audio_path)
# print(log_mel_spectrogram.shape)

# import torch
# import torchaudio
# import torchaudio.transforms as transforms
# import matplotlib.pyplot as plt

# # 加载音频文件
# waveform, sample_rate = torchaudio.load("E:/pythonprojects/PythonProject5/Speech-Emotion-Recognition-yyzcjl/Speech-Emotion-Recognition-yyzcjl/yyzcjl/zc/42093-ruangong-zhaosongran.wav")
# print(waveform.shape)
# # 创建一个短时傅里叶变换（STFT）的实例
# stft_transform = transforms.Spectrogram(n_fft=800, win_length=40, hop_length=10, power=2)

# # 计算傅里叶频谱图
# spectrogram = stft_transform(waveform)

# # 将频谱图转换为dB
# db_spectrogram = 10 * torch.log10(spectrogram + 1e-10)  # 加上小常数以避免对数收敛
# db_spectrogram = db_spectrogram.squeeze(0)[:256, :]
# print(db_spectrogram.shape)
from models.ser_model import Ser_Model
from base_dataset import MyDataset
from tqdm import tqdm
import torch.nn.functional as f
import torch.nn as nn
import numpy as np
import pickle


with open("features_data/features_merge.pkl", "rb") as fin:
        features_data = pickle.load(fin)
    
dataset = MyDataset(features_data=features_data, num_classes=3)
device = torch.device("cuda:0")
model = Ser_Model().to(device)

criterion_ce = nn.CrossEntropyLoss()

def test(model, criterion_ce, test_dataset, device):
    
    total_loss = 0
    test_preds = []
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False)


    sne_labels = []
        
    model.eval()

    # for i, test_batch in enumerate(test_loader):
    with tqdm(test_loader) as td:
        for test_batch in td:
                
            # Send data to correct device
            test_data_fft_batch = test_batch["fft"].to(device)
            test_data_mfcc_batch = test_batch["mfcc"].to(device)
            test_data_wav_batch = test_batch["wav"].to(device)
            test_labels_batch = test_batch["label"].to(device,dtype=torch.long)
        
            labels = test_batch["label"].cpu().detach().numpy()
            sne_labels += list(labels)
        
            # Forward
            test_outputs = model(test_data_fft_batch, test_data_mfcc_batch, test_data_wav_batch)
            test_preds.append(f.log_softmax(test_outputs['M'], dim=1).cpu())

            #test loss
            test_loss_ce = criterion_ce(test_outputs['M'], test_labels_batch)
            # test_loss_mml = criterion_mml(test_outputs['M'], test_labels_batch)
            test_loss = test_loss_ce#  + test_loss_mml
           
            total_loss += test_loss.item()
            
    # Average loss
    test_loss = total_loss / len(test_loader)

    # Accumulate results for val data
    test_preds = np.vstack(test_preds)
    test_preds = test_dataset.get_preds(test_preds)
    
    # Make sure everything works properly
    test_wa = test_dataset.weighted_accuracy(test_preds)
    test_ua = test_dataset.unweighted_accuracy(test_preds)

    results = (test_loss, test_wa*100, test_ua*100)
    return results

with torch.no_grad():
        
        model.load_state_dict(torch.load("E:/pythonprojects/PythonProject5/mywork/results/trained_model.pth"))

        test_dataset = dataset.get_test_dataset()

        test_result = test(
            model, criterion_ce, test_dataset, device=device)

        print("*" * 40)
        print("RESULTS ON TEST SET:")
        print("Loss:{:.4f}\tWA: {:.2f}\tUA: "
              "{:.2f}".format(test_result[0], test_result[1], test_result[2]))