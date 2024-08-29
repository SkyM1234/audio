import argparse
from dataclasses import dataclass
import numpy as np
import soundfile as sf

import torch
import torch.nn.functional as F
import fairseq


@dataclass
class UserDirModule:
    user_dir: str

def e2v(source):
    # 假设source的形状为(batchsize, 1, 48000)
    batchsize = source.shape[0]
    
    model_dir = 'models/upstream'
    checkpoint_dir = 'models/emotion2vec_base.pt'

    model_path = UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    model = model[0]
    model.eval()

    all_feats = []

    with torch.no_grad():
        for i in range(batchsize):
            single_source = source[i]
            if task.cfg.normalize:
                single_source = F.layer_norm(single_source, single_source.shape)
            single_source = single_source.view(1, -1)
            try:
                feats = model.extract_features(single_source, padding_mask=None)
                feats = feats['x'].squeeze(0).cpu()
                all_feats.append(feats)
            except:
                raise Exception("Error in extracting features")

    # 将所有特征堆叠成一个张量
    all_feats = torch.stack(all_feats)
    return all_feats
