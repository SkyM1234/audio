# import librosa
# import numpy as np
# from transformers import Wav2Vec2FeatureExtractor
 
# processor =  Wav2Vec2FeatureExtractor.from_pretrained("E:/pythonprojects/PythonProject5/CA-MSER-main/CA-MSER-main/features_extraction/chinese_pretrain_model/")
 
 
 
# wav_path = r'E:/pythonprojects/PythonProject5/Speech-Emotion-Recognition-yyzcjl/Speech-Emotion-Recognition-yyzcjl/yyzcjl/jl/0c08101116jl0f1.wav'
# speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
# input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
# print(input_values.shape)
 
import torchaudio
import torchaudio.transforms as transforms
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining, Wav2Vec2Model
import torch
from torch.nn import functional as F
import os
import glob
import math

# 定义预强调滤波器
def pre_emphasis(waveform, emphasis_coeff=0.97):
    return torch.cat((waveform[:, :1], waveform[:, 1:] - emphasis_coeff * waveform[:, :-1]), dim=1)
# 加载 Wav2Vec 2.0 模型
# model_name = "E:/pythonprojects/PythonProject5/CA-MSER-main/CA-MSER-main/features_extraction/c hinese_pretrain_model/"  # 可以选择不同的模型
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
# model = Wav2Vec2Model.from_pretrained(model_name)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.half()
# model = model.to(device)
# model.eval()

# 提取所有音频文件的特征
def extract_f(audio_folder, segment_size):

    # 获取文件夹下所有支持的音频文件（例如 .wav, .mp3 等）
    audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))
    features_list_wav = []
    features_list_fft = []
    features_list_mfcc = []
    for audio_file in audio_files:
      # 打印正在处理的文件
      print(f"Processing: {audio_file}")

      # 加载音频文件
      waveform, sample_rate = torchaudio.load(audio_file)

      # 如果需要，则重新采样音频至 16000 Hz
      if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
      # 预强调滤波器
      waveform = pre_emphasis(waveform)
    
      # 分割音频
      # 1.wav
      wave_seg_wav = []
    
      time_wav = waveform.shape[1]
      segment_size_wav = segment_size * 160 # 48000
      start_wav, end_wav = 0, segment_size_wav
      num_segs = math.ceil(time_wav / segment_size_wav)

      for i in range(num_segs):
          if end_wav > time_wav:
              padding = (0, end_wav-time_wav)
              wave_seg_wav.append(F.pad(waveform[:, start_wav:time_wav], pad=padding, mode='constant', value=0))
              break
          wave_seg_wav.append(waveform[:, start_wav:end_wav])
          start_wav = start_wav + segment_size_wav
          end_wav = end_wav + segment_size_wav

      wave_seg_wav = torch.stack(wave_seg_wav)
      features_list_wav.append(wave_seg_wav)

      # fft
      wave_seg_fft = []

      n_fft = 800
      win_length = 40
      hop_length = 10
      freq = 256 
    
      stft_transform = transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=2)

      spectrogram = stft_transform(waveform) # 计算傅里叶频谱图

      db_spectrogram = 10 * torch.log10(spectrogram + 1e-10)  # 将频谱图转换为dB,加上小常数以避免对数收敛
      fft = db_spectrogram.squeeze(0)[:freq, :] # [256, ]

      time_fft = fft.shape[1]

      start_fft, end_fft = 0, segment_size
      for i in range(num_segs):
          if end_fft > time_fft:
              padding = (0, end_fft-time_fft)
              wave_seg_fft.append(F.pad(fft[:, start_fft:time_fft], pad=padding, mode='constant', value=0))
              break
          wave_seg_fft.append(fft[:, start_fft:end_fft])
          start_fft = start_fft + segment_size
          end_fft = end_fft + segment_size

      wave_seg_fft = torch.stack(wave_seg_fft)
      features_list_fft.append(wave_seg_fft)

      # 3.mfcc
      wave_seg_mfcc = []

      n_mels = 128 # 128 The value for `n_mels` (128) may be set too high
      n_mfcc = 40

      mfcc_transform = transforms.MFCC(
      sample_rate=sample_rate,
      n_mfcc=n_mfcc,  # MFCC特征的维度，通常为13或40
      melkwargs={"win_length": win_length, "n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels} 
      )
    
      mfcc = mfcc_transform(waveform).squeeze(0) # 提取 MFCC 特征

      time_mfcc = mfcc.shape[1]

      start_mfcc, end_mfcc = 0, segment_size
      for i in range(num_segs):
          if end_mfcc > time_mfcc:
              padding = (0, end_mfcc-time_mfcc)
              wave_seg_mfcc.append(F.pad(mfcc[:, start_mfcc:time_mfcc], pad=padding, mode='constant', value=0))
              break
          wave_seg_mfcc.append(mfcc[:, start_mfcc:end_mfcc])
          start_mfcc = start_mfcc + segment_size
          end_mfcc = end_mfcc + segment_size

      wave_seg_mfcc = torch.stack(wave_seg_mfcc)
      features_list_mfcc.append(wave_seg_mfcc)

    features_tensor_wav = torch.vstack(features_list_wav) # [num_segs, 1, segment_size_wav]
    features_tensor_wav = features_tensor_wav.squeeze(1) # [num_segs, segment_size_wav]

    features_tensor_fft = torch.vstack(features_list_fft).unsqueeze(1) # [num_segs, 1, freq, segment_size]
    features_tensor_mfcc = torch.vstack(features_list_mfcc).permute(0, 2, 1) # [num_segs, segment_size, n_mfcc]

    print(features_tensor_wav.shape)
    print(features_tensor_fft.shape)
    print(features_tensor_mfcc.shape)

    # torch.save(features_tensor_wav, "features_data/features_wav_jl.pt")
    # torch.save(features_tensor_fft, "features_data/features_fft_jl.pt")
    # torch.save(features_tensor_mfcc, "features_data/features_mfcc_jl.pt")

# 设置音频文件夹路径
audio_folder = "E:/pythonprojects/PythonProject5/Speech-Emotion-Recognition-yyzcjl/Speech-Emotion-Recognition-yyzcjl/yyzcjl/jl"  # 替换为你的音频文件夹路径
segment_size = 300


extract_f(audio_folder, segment_size)


# 获取特征
# with torch.no_grad():
#     features = model(features_tensor.to(device)).last_hidden_state

# print(features.shape)


