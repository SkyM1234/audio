import torch
from collections import defaultdict
import numpy as np

class MyDataset:
    def __init__(self, features_data, num_classes = 3):

        self.train_fft_data = features_data["fft_train"]
        self.train_mfcc_data = features_data["mfcc_train"]
        self.train_wav_data = features_data["wav_train"]
        self.train_label = features_data["label_train"]

        self.test_fft_data = features_data["fft_test"]
        self.test_mfcc_data = features_data["mfcc_test"]
        self.test_wav_data = features_data["wav_test"]
        self.test_label = features_data["label_test"]

        self.num_classes = num_classes

        self.train_data = defaultdict()
        self.train_data["fft"] = self.train_fft_data
        self.train_data["mfcc"] = self.train_mfcc_data
        self.train_data["wav"] = self.train_wav_data
        self.train_data["label"] = self.train_label

        self.test_data = defaultdict()
        self.test_data["fft"] = self.test_fft_data
        self.test_data["mfcc"] = self.test_mfcc_data
        self.test_data["wav"] = self.test_wav_data
        self.test_data["label"] = self.test_label


    
    def get_train_dataset(self):
        return TrainDataset(
            self.train_data, num_classes=self.num_classes)
    
    def get_test_dataset(self):
        return TestDataset(
            self.test_data, num_classes=self.num_classes)
    

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data, num_classes=3):
        super(TrainDataset).__init__()
        self.data_fft = data["fft"]
        self.data_mfcc = data["mfcc"]
        self.data_wav = data["wav"]
        self.label = data["label"].squeeze(1)
        self.n_samples = len(self.label)
        self.num_classes = num_classes
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        sample = {
            'fft': self.data_fft[index], 
            'mfcc': self.data_mfcc[index],
            'wav': self.data_wav[index],
            'label': self.label[index]
            } 
        return sample
    
    def get_preds(self, preds):
        preds = np.argmax(preds, axis=1)
        return preds
    
    def weighted_accuracy(self, predictions):
        acc = (self.label.numpy() == predictions).sum() / self.n_samples
        return acc


    def unweighted_accuracy(self, predictions):
        class_acc = 0
        n_classes = 0
        for c in range(self.num_classes):
            class_pred = np.multiply(( self.label.numpy() == predictions),
                                     ( self.label.numpy() == c)).sum()
            
            if (self.label == c).sum() > 0:
                 class_pred /= ( self.label == c).sum()
                 n_classes += 1

                 class_acc += class_pred
            
        return class_acc / n_classes

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, num_classes=3):
        super(TestDataset).__init__()
        self.data_fft = data["fft"]
        self.data_mfcc = data["mfcc"]
        self.data_wav = data["wav"]
        self.label = data["label"].squeeze(1) 
        self.n_samples = len(self.label)
        self.num_classes = num_classes


    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = {
            'fft': self.data_fft[index], 
            'mfcc': self.data_mfcc[index],
            'wav': self.data_wav[index],
            'label': self.label[index]
            } 
        return sample
    
    def get_preds(self, preds):             
        preds = np.argmax(preds, axis=1)
        return preds
    
    def weighted_accuracy(self, preds):
        acc = (self.label.numpy() == preds).sum() / self.n_samples
        return acc


    def unweighted_accuracy(self, preds):

        class_acc = 0
        n_classes = 0
        
        for c in range(self.num_classes):
            class_pred = np.multiply((self.label.numpy() == preds),
                                     (self.label.numpy() == c)).sum()

        
            if (self.label.numpy() == c).sum() > 0:    
                class_pred /= (self.label.numpy() == c).sum()
                n_classes += 1
                class_acc += class_pred
        
        return class_acc / n_classes