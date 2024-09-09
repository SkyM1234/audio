from base_dataset import MyDataset
import pickle
import torch
from tqdm import tqdm
from models.ser_model import Ser_Model
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from collections import defaultdict


def main():
    with open("features_data/features_merge_1.pkl", "rb") as fin:
        features_data = pickle.load(fin)
    
    dataset = MyDataset(features_data=features_data, num_classes=3)

    train_stat = train(dataset)

def train(dataset):
    train_dataset = dataset.get_train_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=16, 
                                shuffle=True)
    
    device = torch.device("cuda:0")
    
    model = Ser_Model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.00005)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mml = nn.MultiMarginLoss(margin=0.5)

    loss_format = "{:.04f}"
    acc_format2 = "{:.02f}"
    
    all_train_loss =[]
    all_train_wa =[]
    all_train_ua=[]
    train_preds = []
    
    print("Start Training!!!")
    
    for epoch in range(5):
        
        # Train one epoch
        total_loss = 0
        train_preds = []
        target=[]
        model.train()

        with tqdm(train_loader) as td:
            for train_batch in td:
       
                # Clear gradients
                optimizer.zero_grad()
            
                # Send data to correct device
                train_data_fft_batch = train_batch["fft"].to(device)
                train_data_mfcc_batch = train_batch["mfcc"].to(device)
                train_data_wav_batch = train_batch["wav"].to(device)
                train_labels_batch =  train_batch["label"].to(device,dtype=torch.long)
                print(train_labels_batch)
            
                # Forward pass
                outputs = model(train_data_fft_batch, train_data_mfcc_batch, train_data_wav_batch)
            
                
                train_preds.append(f.log_softmax(outputs['M'], dim=1).cpu().detach().numpy())

                
                # Compute the loss, gradients, and update the parameters
                train_loss_ce = criterion_ce(outputs["M"], train_labels_batch)
                train_loss_mml = criterion_mml(outputs['M'], train_labels_batch)
                train_loss = train_loss_ce + train_loss_mml
                print(train_loss)
                print("***")
 
                train_loss.backward()
                total_loss += train_loss.item()
                optimizer.step()
            
        # Evaluate training data
        train_loss = total_loss / len(train_loader)
        # Accumulate results for train data
        train_preds = np.vstack(train_preds)
        train_preds = train_dataset.get_preds(train_preds)
        print(train_preds)
        
        # Make sure everything works properly
        train_wa = train_dataset.weighted_accuracy(train_preds) * 100
        train_ua = train_dataset.unweighted_accuracy(train_preds) * 100
        
        all_train_loss.append(loss_format.format(train_loss))
        all_train_wa.append(acc_format2.format(train_wa))
        all_train_ua.append(acc_format2.format(train_ua))

    # save result
    save_path = "E:/pythonprojects/PythonProject5/mywork/results/"

    torch.save(model.state_dict(), save_path + "trained_model.pth")
    result = defaultdict()
    result["all_train_loss"] = all_train_loss
    result["all_train_wa"] = all_train_wa
    result["all_train_ua"] = all_train_ua

    with open(save_path + "progess.pkl", 'wb') as pickle_file:
        pickle.dump(result, pickle_file)
    
    

     

if __name__ == '__main__':
    main()