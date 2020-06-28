import pandas as pd
import os
import random
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
def set_seed(seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, input_dim=1024):
        super(Net, self).__init__()
        prob_dropout = 0.2
        output_size = 1
        
        self.dropout_1 = nn.Dropout(p=prob_dropout)
        self.out_proj = nn.Linear(input_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, debug=False):
        x = x.squeeze()
        
        if debug:
            print("Init ", x.shape)
            
        x = self.dropout_1(x)
        x = self.out_proj(x)
        
        if debug:
            print("out_proj ", x.shape)
            
        x = self.sig(x)
        
        if debug:
            print("sig ", x.shape)
            
        return x
    
def train(train_loader, valid_loader, input_dim=1024, learning_rate=0.001, debug=False):
    model = Net(input_dim=input_dim)
    if train_on_gpu:
        model.cuda()

    lr = learning_rate
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=5,
                                                           verbose=debug)

    n_epochs = 30000
    early_stopping_patience = 12

    valid_loss_min = np.Inf
    train_loss_min = np.Inf 
    train_acc_min = np.Inf 
    valid_acc_min = np.Inf
    last_best_epoch = 0

    len_train = len(train_loader.sampler)
    len_valid = len(valid_loader.sampler)

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0

        ###############
        # train model #
        ###############
        model.train()
        train_correct = 0
        for data, target in train_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            model.zero_grad()
            output = model(data)
            
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)


            pred = torch.round(output.squeeze()) 
            correct_tensor = pred.eq(target.float().view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            train_correct += np.sum(correct)

        ##################
        # validate model #
        ##################

        val_correct = 0
        for data, target in valid_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
                
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            valid_loss += loss.item()*data.size(0)

            pred = torch.round(output.squeeze()).int()
            correct_tensor = pred.eq(target.int().view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            val_correct += np.sum(correct)

        
        train_loss = train_loss/len_train
        valid_loss = valid_loss/len_valid
        train_acc = train_correct/len_train
        val_acc = val_correct/len_valid
        
        if debug:
            print('Epoch: {} \tT-Loss: {:.6f} \tT-Acc: {:.6f} \tV-Loss: {:.6f} \tV-Acc: {:.6f}'.format(
                epoch, train_loss, train_acc, valid_loss, val_acc))

        scheduler.step(valid_loss)

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            if debug:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                
            torch.save(model.state_dict(), 'model.pt')
            last_best_epoch = epoch
            valid_loss_min = valid_loss
            train_loss_min = train_loss
            train_acc_min = train_acc
            valid_acc_min = val_acc

        elif (epoch-last_best_epoch) > early_stopping_patience: 
            print("EarlyStopping! Epoch {}".format(epoch))
            print('Last: {} \tT-Loss: {:.6f} \tT-Acc: {:.6f} \tV-Loss: {:.6f} \tV-Acc: {:.6f}'.format(
                last_best_epoch, train_loss_min, train_acc_min, valid_loss_min, valid_acc_min))
                
            break

def test(test_loader, input_dim=1024):
    model = Net(input_dim=input_dim)
    criterion = nn.BCELoss()
    if train_on_gpu:
        model.cuda()
        
    model.load_state_dict(torch.load('model.pt'))

    test_loss = 0.0
    num_correct = 0
    y_true = np.array([])
    y_pred = np.array([])
    y_pred_proba = np.array([])

    model.eval()
    for data, target in test_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
            
        output = model(data)
        loss = criterion(output.squeeze(), target.float())
        test_loss += loss.item()*data.size(0)
        
        pred = torch.round(output.squeeze()).int()
        correct_tensor = pred.eq(target.int().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

        y_true = np.concatenate([y_true, target.int().view_as(pred).detach().numpy()])
        y_pred = np.concatenate([y_pred, pred.detach().numpy()])
        y_pred_proba = np.concatenate([y_pred_proba, output.squeeze().detach().numpy()])


    test_loss = test_loss/len(test_loader.sampler)
    print('Final test Loss: {:.6f}'.format(test_loss))
    
    return y_true, y_pred_proba

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, recall_score, precision_score
def evaluate(y_true, y_pred_proba, threshold=None, debug=False):
    max_threshold = -1
    max_f1 = 0
    max_recall = 0
    max_precision = 0
    if threshold==None:
        print("[Evaluate] No threshold argument. Finding best threshold.")
        threshold_list = [x/100 for x in range(0, 100)]
    else:
        print("[Evaluate] Threshold argument set. Using {} as threshold".format(threshold))
        threshold_list = [threshold]
        
    for threshold_it in threshold_list:
        y_pred_thr = [1 if x>=threshold_it else 0 for x in y_pred_proba]
        f1 = f1_score(y_true, y_pred_thr, average='macro')
        recall = recall_score(y_true, y_pred_thr)
        precision = precision_score(y_true, y_pred_thr)

        if debug:
            print("[Evaluate] THRESHOLD: {:.3f} \tF1: {:.8f} \tRecall: {:.8f} \tPrecision: {:.8f}".format(threshold_it, 
                                                                                                           f1, 
                                                                                                           recall, 
                                                                                                           precision))

        if f1>max_f1:
            max_f1 = f1
            max_recall = recall
            max_precision = precision
            max_threshold = threshold_it

    print("[Evaluate] ##MAX## \nTHRESHOLD: {:.3f} \tF1: {:.8f} \tRecall: {:.8f} \tPrec: {:.8f}".format(max_threshold,
                                                                                                         max_f1, 
                                                                                                         max_recall, 
                                                                                                         max_precision))
    return max_f1, max_recall, max_precision, max_threshold

if __name__ == "main":
    print('ok')