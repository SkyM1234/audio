# import librosa
# import numpy as np
# from transformers import Wav2Vec2FeatureExtractor
 
# processor =  Wav2Vec2FeatureExtractor.from_pretrained("E:/pythonprojects/PythonProject5/CA-MSER-main/CA-MSER-main/features_extraction/chinese_pretrain_model/")
 
 
 
# wav_path = r'E:/pythonprojects/PythonProject5/Speech-Emotion-Recognition-yyzcjl/Speech-Emotion-Recognition-yyzcjl/yyzcjl/jl/0c08101116jl0f1.wav'
# speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
# input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
# print(input_values.shape)
 
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining, Wav2Vec2Model
import torch
from torch.nn import functional as F
import os
import glob
import numpy as np
import math

# 加载 Wav2Vec 2.0 模型
model_name = "E:/pythonprojects/PythonProject5/CA-MSER-main/CA-MSER-main/features_extraction/chinese_pretrain_model/"  # 可以选择不同的模型
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.half()
model = model.to(device)
model.eval()

# 设置音频文件夹路径
audio_folder = "E:/pythonprojects/PythonProject5/Speech-Emotion-Recognition-yyzcjl/Speech-Emotion-Recognition-yyzcjl/yyzcjl/jl"  # 替换为你的音频文件夹路径

# 获取文件夹下所有支持的音频文件（例如 .wav, .mp3 等）
audio_files = glob.glob(os.path.join(audio_folder, "*.wav")) # 可以添加其他音频格式

# 提取所有音频文件的特征
features_list = []

for audio_file in audio_files:
# 打印正在处理的文件
    print(f"Processing: {audio_file}")

    # 加载音频文件
    waveform, sample_rate = torchaudio.load(audio_file)

    # 如果需要，则重新采样音频至 16000 Hz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # 分割音频
    wave_seg = []
    
    segment_size = 300
    time_wav = waveform.shape[1]
    segment_size_wav = segment_size * 160 # 48000
    start_wav, end_wav = 0, segment_size_wav
    num_segs = math.ceil(time_wav / segment_size_wav)

    for i in range(num_segs):
        if end_wav > time_wav:
            padding = (0, end_wav-time_wav)
            wave_seg.append(F.pad(waveform[:, start_wav:time_wav], pad=padding, mode='constant', value=0))
            break
        wave_seg.append(waveform[:, start_wav:end_wav])
        start_wav = start_wav + segment_size_wav
        end_wav = end_wav + segment_size_wav

    wave_seg = torch.stack(wave_seg)
    features_list.append(wave_seg)

features_tensor = torch.vstack(features_list) # [num_segs, 1, segment_size_wav]
features_tensor = features_tensor.squeeze(1) # [num_segs, segment_size_wav]


# 获取特征
with torch.no_grad():
    features = model(features_tensor.to(device)).last_hidden_state

print(features.shape)

# y, fs = librosa.load('E:/pythonprojects/PythonProject5/Speech-Emotion-Recognition-yyzcjl/Speech-Emotion-Recognition-yyzcjl/yyzcjl/jl/0c08101116jl0f1.wav', sr=16000) 
# win_length = 512 
# hop_length = 160 
# n_fft = 512 
# n_mels = 128 
# n_mfcc = 20 
# mfcc = librosa.feature.mfcc(y=y, sr=fs, win_length=win_length, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels, dct_type=1) # 特征值增加差分量 # 一阶差分 mfcc_deta = librosa.feature.delta(mfcc)
# print(mfcc.shape)